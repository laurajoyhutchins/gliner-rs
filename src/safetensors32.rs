//! A `model.safetensors` reader that works on 32-bit targets (wasm32).
//!
//! The `safetensors` crate validates each tensor by computing its size in
//! *bits* through `usize` arithmetic: `nelements * dtype.bitsize()`. On a
//! 32-bit target that overflows for any tensor with more than ~134M f32
//! elements — which every GLiNER2 checkpoint has, since mDeBERTa-v3's input
//! embedding alone is 250112 x 768 = 192M elements (6.1e9 bits, past
//! `u32::MAX`). Loading then fails with "overflow computing buffer size from
//! shape and/or element type", even though the tensor is only 768 MB and
//! fits fine.
//!
//! The format itself is simple, so this reads it directly with `u64` math:
//! an 8-byte little-endian header length, that many bytes of JSON header
//! (`{"name": {"dtype", "shape", "data_offsets": [start, end]}, ...}`), then
//! the tensor data, whose offsets are relative to the end of the header.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

#[derive(Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
}

fn dtype_of(name: &str) -> Result<DType> {
    Ok(match name {
        "U8" | "BOOL" => DType::U8,
        "U32" => DType::U32,
        "I16" => DType::I16,
        "I32" => DType::I32,
        "I64" => DType::I64,
        "BF16" => DType::BF16,
        "F16" => DType::F16,
        "F32" => DType::F32,
        "F64" => DType::F64,
        other => bail!("unsupported safetensors dtype {other:?}"),
    })
}

/// Reads every tensor out of `buffer` onto `device`. Tensors keep their
/// stored dtype; the caller converts (see [`candle_nn::VarBuilder`]).
pub fn load_buffer(buffer: &[u8], device: &Device) -> Result<HashMap<String, Tensor>> {
    if buffer.len() < 8 {
        bail!("safetensors buffer is too short to hold a header length");
    }
    // Do not truncate the serialized u64 on wasm32. A header length of
    // 2^32 + 2 previously wrapped to 2 and could admit a malformed file.
    let header_len = u64::from_le_bytes(buffer[..8].try_into().expect("8 bytes"));
    let header_len = usize::try_from(header_len).context("safetensors header length cannot fit in usize")?;
    // Match the upstream safety boundary: malformed inputs must not induce
    // arbitrarily large JSON allocations even when the caller owns the buffer.
    if header_len > 100_000_000 {
        bail!("safetensors header exceeds the 100 MB safety limit");
    }
    let data_start = 8usize.checked_add(header_len).context("safetensors header length overflows")?;
    if data_start > buffer.len() {
        bail!("safetensors header ({header_len} bytes) runs past the end of the file");
    }

    let header: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&buffer[8..data_start]).context("parsing the safetensors header")?;
    let data = &buffer[data_start..];

    // Validate every tensor and the complete data layout before allocating
    // any Candle tensor. Individual in-bounds slices are insufficient:
    // safetensors also forbids overlapping slices and uncovered bytes.
    let mut validated = Vec::with_capacity(header.len());
    let mut intervals = Vec::with_capacity(header.len());
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let info: TensorInfo =
            serde_json::from_value(value).with_context(|| format!("parsing header entry for tensor {name:?}"))?;
        let (start, end) = info.data_offsets;
        if end < start || end > data.len() as u64 {
            bail!("tensor {name:?} has data offsets {start}..{end} outside the {} byte data block", data.len());
        }
        // Keep element counts and byte sizes in u64 to support wasm32.
        let dtype = dtype_of(&info.dtype)?;
        let elements = info.shape.iter().try_fold(1u64, |acc, d| acc.checked_mul(*d)).with_context(|| {
            format!("tensor {name:?} shape {:?} overflows an element count", info.shape)
        })?;
        let expected = elements
            .checked_mul(dtype.size_in_bytes() as u64)
            .with_context(|| format!("tensor {name:?} is too large to address"))?;
        if end - start != expected {
            bail!("tensor {name:?} spans {} bytes but its shape/dtype need {expected}", end - start);
        }
        let shape: Vec<usize> = info
            .shape
            .iter()
            .map(|d| usize::try_from(*d).with_context(|| format!("tensor {name:?} dimension {d} does not fit in usize")))
            .collect::<Result<_>>()?;
        intervals.push((start, end));
        validated.push((name, dtype, shape, start as usize, end as usize));
    }

    // JSON object order is not significant. Compare ordered byte intervals,
    // allowing zero-sized tensors only at the current position.
    intervals.sort_unstable();
    let mut cursor = 0u64;
    for (start, end) in intervals {
        if start != cursor {
            bail!("safetensors data layout has an overlap or hole at byte {cursor} (next range {start}..{end})");
        }
        cursor = end;
    }
    if cursor != data.len() as u64 {
        bail!("safetensors data layout leaves {} unindexed trailing bytes", data.len() as u64 - cursor);
    }

    let mut tensors = HashMap::with_capacity(validated.len());
    for (name, dtype, shape, start, end) in validated {
        let tensor = Tensor::from_raw_buffer(&data[start..end], dtype, &shape, device)
            .with_context(|| format!("reading tensor {name:?}"))?;
        tensors.insert(name, tensor);
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer_with_tensor(header: &str, tensor_bytes: &[u8]) -> Vec<u8> {
        let mut buffer = (header.len() as u64).to_le_bytes().to_vec();
        buffer.extend_from_slice(header.as_bytes());
        buffer.extend_from_slice(tensor_bytes);
        buffer
    }

    #[test]
    fn loads_a_small_valid_tensor_and_ignores_metadata() {
        let header = r#"{"__metadata__":{"format":"pt"},"weights":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let buffer = buffer_with_tensor(header, &1.25_f32.to_le_bytes());
        let tensors = load_buffer(&buffer, &Device::Cpu).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors["weights"].to_vec1::<f32>().unwrap(), vec![1.25]);
    }

    #[test]
    fn rejects_header_lengths_that_would_wrap_on_wasm32() {
        // A u64 length is part of the file format, but wasm32 usize is 32-bit.
        // The old `as usize` cast turned this into a valid 2-byte "{}" header.
        let mut buffer = ((1u64 << 32) + 2).to_le_bytes().to_vec();
        buffer.extend_from_slice(b"{}");
        assert!(load_buffer(&buffer, &Device::Cpu).is_err());
    }

    #[test]
    fn rejects_offsets_outside_the_data_block() {
        let header = r#"{"weights":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}"#;
        let buffer = buffer_with_tensor(header, &1.25_f32.to_le_bytes());
        let error = load_buffer(&buffer, &Device::Cpu).unwrap_err().to_string();
        assert!(error.contains("outside"), "{error}");
    }

    #[test]
    fn rejects_overlapping_tensor_ranges() {
        let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let buffer = buffer_with_tensor(header, &1.25_f32.to_le_bytes());
        let err = load_buffer(&buffer, &Device::Cpu).unwrap_err().to_string();
        assert!(err.contains("overlap or hole"), "{err}");
    }

    #[test]
    fn rejects_holes_between_tensor_ranges() {
        let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}"#;
        let buffer = buffer_with_tensor(header, &[0u8; 12]);
        let err = load_buffer(&buffer, &Device::Cpu).unwrap_err().to_string();
        assert!(err.contains("overlap or hole"), "{err}");
    }

    #[test]
    fn rejects_unindexed_trailing_bytes() {
        let header = r#"{"weights":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let buffer = buffer_with_tensor(header, &[0u8; 8]);
        let err = load_buffer(&buffer, &Device::Cpu).unwrap_err().to_string();
        assert!(err.contains("unindexed trailing bytes"), "{err}");
    }

    #[test]
    fn accepts_unordered_contiguous_ranges_and_empty_tensors() {
        let header = r#"{"second":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"empty":{"dtype":"F32","shape":[0],"data_offsets":[4,4]},"first":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut data = 1.25_f32.to_le_bytes().to_vec();
        data.extend(2.5_f32.to_le_bytes());
        let tensors = load_buffer(&buffer_with_tensor(header, &data), &Device::Cpu).unwrap();
        assert_eq!(tensors["first"].to_vec1::<f32>().unwrap(), vec![1.25]);
        assert_eq!(tensors["second"].to_vec1::<f32>().unwrap(), vec![2.5]);
        assert_eq!(tensors["empty"].dims(), &[0]);
    }

    #[test]
    fn rejects_oversized_header_before_parsing_json() {
        let buffer = (100_000_001u64).to_le_bytes().to_vec();
        let err = load_buffer(&buffer, &Device::Cpu).unwrap_err().to_string();
        assert!(err.contains("100 MB safety limit"), "{err}");
    }

    #[test]
    fn rejects_shape_and_data_length_disagreement() {
        let header = r#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
        let buffer = buffer_with_tensor(header, &1.25_f32.to_le_bytes());
        let error = load_buffer(&buffer, &Device::Cpu).unwrap_err().to_string();
        assert!(error.contains("shape/dtype"), "{error}");
    }
}
