//! Real-model A/B benchmark for padding-aware extraction batching.
//! cargo run --release --example bench_bucketed -- /path/to/checkpoint
use candle_core::{DType, Device};
use gliner_rs::{ExtractOptions, GLiNER2, Schema};
use std::{env, time::Instant};

fn main() -> anyhow::Result<()> {
    let model_path = env::args().nth(1).expect("model directory required");
    let model = GLiNER2::load(model_path, &Device::Cpu, DType::F32)?;
    let schema = Schema::new().entities(["company", "person", "equipment", "location", "electrical rating"]);
    let opts = ExtractOptions { include_confidence: true, include_spans: true, ..Default::default() };
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
    let mut texts: Vec<String> = Vec::new();
    for i in 0..16 {
        let base = sentences[(i * 7 + i / 5) % sentences.len()];
        let mut text = base.to_string();
        for _ in 1..((i % 3) + 1) { text.push(' '); text.push_str(base); }
        texts.push(text);
    }
    let references: Vec<&str> = texts.iter().map(String::as_str).collect();
    let run = |bucketed: bool| -> anyhow::Result<(f64, Vec<serde_json::Value>)> {
        let t = Instant::now();
        let results = if bucketed {
            model.extract_batch_bucketed(&references, &schema, &opts)?
        } else {
            model.extract_batch(&references, &schema, &opts)?
        };
        Ok((t.elapsed().as_secs_f64() * 1000.0, results))
    };
    let mut times_base = Vec::new();
    let mut times_bucketed = Vec::new();
    for pass in 0..5 {
        if pass % 2 == 0 {
            let (a, baseline) = run(false)?;
            let (b, optimized) = run(true)?;
            assert_eq!(baseline, optimized, "batched output mismatch pass {pass}");
            times_base.push(a);
            times_bucketed.push(b);
        } else {
            let (b, optimized) = run(true)?;
            let (a, baseline) = run(false)?;
            assert_eq!(baseline, optimized, "batched output mismatch pass {pass}");
            times_base.push(a);
            times_bucketed.push(b);
        }
    }
    times_base.sort_by(f64::total_cmp);
    times_bucketed.sort_by(f64::total_cmp);
    println!("documents={} baseline_median_ms={:.3} bucketed_median_ms={:.3} speedup={:.3}x",
        references.len(), times_base[2], times_bucketed[2], times_base[2] / times_bucketed[2]);
    Ok(())
}
