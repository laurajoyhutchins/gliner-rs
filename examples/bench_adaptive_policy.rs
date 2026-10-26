//! Real-checkpoint A/B: original single padded batch, adaptive default,
//! and explicitly bucketed batch. The *same* harness is run against the
//! previous pinned source revision and the candidate checkout.
use candle_core::{DType, Device};
use gliner_rs::{ExtractOptions, GLiNER2, Schema};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{env, time::Instant};

fn sample_documents() -> Vec<String> {
    let sentences = [
        "Panel LP-1 operates at 480 volts.",
        "Alice works for Acme in Paris.",
        "The main distribution switchboard MSB-1 is rated 480/277 volts, three-phase, four wire. It has a 400 ampere main breaker and serves the electrical panels on the first floor of the building.",
        "Schneider Electric supplies the 1200 amp switchgear in Building C. The installation uses a 2000 kVA transformer with primary 13.8 kV and secondary 480/277 V. Protective devices coordinate with the service entrance breaker, feeder breakers, and panelboard breakers under normal utility conditions.",
        "Charlie visited Denver.",
        "Bob called Emily in Chicago.",
        "Industrial motor M-2 is a 100 horsepower induction motor rated 460 volts with full-load current 124 A. The feeder conductors and overload protection are selected for the motor nameplate, operating duty, and equipment requirements.",
        "The emergency generator GEN-1 is rated 1500 kW, 480/277 volts, and supplies the life-safety switchboard through an automatic transfer switch. Its protective settings account for reduced fault current in islanded operation and available utility short-circuit current when paralleled.",
    ];
    (0..16).map(|i| {
        let base = sentences[(i * 7 + i / 5) % sentences.len()];
        let mut text = base.to_string();
        for _ in 1..((i % 3) + 1) { text.push(' '); text.push_str(base); }
        text
    }).collect()
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    (samples[1] + samples[2]) / 2.0
}

fn main() -> anyhow::Result<()> {
    let model_path = env::args().nth(1).expect("model dir argument required");
    let model = GLiNER2::load(model_path, &Device::Cpu, DType::F32)?;
    let schema = Schema::new().entities(["company", "person", "equipment", "location", "electrical rating"]);
    let opts = ExtractOptions { include_confidence: true, include_spans: true, ..Default::default() };
    let heterogeneous = sample_documents();
    let uniform = (0..16).map(|i| format!("Alice works for Acme in Denver office number {i}.")).collect::<Vec<_>>();
    for (name, docs) in [("mixed", &heterogeneous), ("uniform", &uniform)] {
        let texts = docs.iter().map(String::as_str).collect::<Vec<_>>();
        // Warm up both paths. This also checks the full output contract.
        let warm_a = model.extract_batch(&texts, &schema, &opts)?;
        let warm_b = model.extract_batch_bucketed(&texts, &schema, &opts)?;
        assert_eq!(warm_a, warm_b, "warm output mismatch for {name}");
        let mut default_ms = Vec::new();
        let mut bucket_ms = Vec::new();
        let mut final_output = warm_a;
        for pass in 0..4 {
            let run = |bucketed: bool| -> anyhow::Result<(f64, Vec<Value>)> {
                let t = Instant::now();
                let output = if bucketed {
                    model.extract_batch_bucketed(&texts, &schema, &opts)?
                } else {
                    model.extract_batch(&texts, &schema, &opts)?
                };
                Ok((t.elapsed().as_secs_f64() * 1000.0, output))
            };
            let ((a_ms, a), (b_ms, b)) = if pass % 2 == 0 {
                (run(false)?, run(true)?)
            } else {
                let b = run(true)?;
                let a = run(false)?;
                (a, b)
            };
            assert_eq!(a, b, "output mismatch in {name} pass {pass}");
            default_ms.push(a_ms);
            bucket_ms.push(b_ms);
            final_output = a;
        }
        let digest = format!("{:x}", Sha256::digest(serde_json::to_vec(&final_output)?));
        println!("{}", json!({
            "case": name,
            "documents": docs.len(),
            "default_median_ms": median(default_ms),
            "bucketed_median_ms": median(bucket_ms),
            "output_sha256": digest
        }));
    }
    Ok(())
}
