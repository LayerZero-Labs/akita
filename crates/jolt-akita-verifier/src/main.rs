//! Host driver that compiles the Jolt guest program in
//! `crates/jolt-akita-verifier/guest`, feeds it the
//! [`akita_jolt_glue::AkitaJoltInputs`] blob produced by `examples/jolt_artifact`,
//! and proves that the Akita verifier returns successfully.
//!
//! Per-marker cycle counts emitted by the guest's
//! `start_cycle_tracking` / `end_cycle_tracking` calls are forwarded through
//! Jolt's `tracing` infrastructure; we initialize a tracing subscriber here
//! so they show up on stdout.

#![allow(missing_docs)]

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    about = "Prove the Akita verifier inside Jolt and report cycle counts",
    long_about = None
)]
struct Args {
    /// Path to the verifier-input blob produced by
    /// `examples/jolt_artifact`.
    #[arg(long, default_value = "target/akita_jolt_inputs.bin")]
    input: PathBuf,

    /// Directory used by Jolt for per-program build artifacts.
    #[arg(long, default_value = "/tmp/jolt-akita-targets")]
    target_dir: String,

    /// Only trace the guest (skips the ~minute-long Jolt prover step).
    /// Useful when iterating on guest panics with `JOLT_BACKTRACE=full`.
    #[arg(long, default_value_t = false)]
    trace_only: bool,
}

fn main() {
    let filter = EnvFilter::try_from_env("AKITA_JOLT_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let args = Args::parse();

    info!(input = %args.input.display(), "loading verifier-input blob");
    let blob = std::fs::read(&args.input).expect("read verifier-input blob");
    info!(bytes = blob.len(), "blob loaded");

    info!(target_dir = %args.target_dir, "compiling Akita verifier guest program");
    let mut program = guest::compile_akita_verify(&args.target_dir);

    if args.trace_only {
        info!("trace-only mode: skipping preprocessing and proof generation");
        // Native run first to surface any guest panic outside the prover.
        info!("running guest natively (sanity check)");
        let native_output = guest::akita_verify(blob.clone());
        info!(native_output, "native guest output");

        let trace_path = PathBuf::from(&args.target_dir).join("akita_verify.trace");
        info!(trace_file = %trace_path.display(), "tracing guest under emulator");
        guest::trace_akita_verify_to_file(trace_path.to_str().expect("utf-8 path"), blob);
        info!("trace done");
        return;
    }

    info!("running shared / prover / verifier preprocessing");
    let shared_preprocessing = guest::preprocess_shared_akita_verify(&mut program)
        .expect("shared preprocessing");
    let prover_preprocessing = guest::preprocess_prover_akita_verify(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_akita_verify(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
        None,
    );

    let prove_akita_verify = guest::build_prover_akita_verify(program, prover_preprocessing);
    let verify_akita_verify = guest::build_verifier_akita_verify(verifier_preprocessing);

    // Native run first to surface any guest panic outside the prover.
    info!("running guest natively (sanity check)");
    let native_output = guest::akita_verify(blob.clone());
    info!(native_output, "native guest output");
    assert_eq!(
        native_output, 0,
        "native guest run reported failure code {native_output}"
    );

    info!("invoking Jolt prover");
    let now = Instant::now();
    let (output, proof, program_io) = prove_akita_verify(blob.clone());
    let prover_secs = now.elapsed().as_secs_f64();
    info!(prover_secs, "prover finished");
    info!(
        guest_output = output,
        guest_panic = program_io.panic,
        "prover program-io"
    );

    let now = Instant::now();
    let is_valid = verify_akita_verify(blob, output, program_io.panic, proof);
    let verifier_secs = now.elapsed().as_secs_f64();
    info!(
        verifier_secs,
        is_valid, "Jolt verifier finished"
    );

    assert!(is_valid, "Jolt verifier rejected the proof");
    assert_eq!(output, 0, "guest reported Akita-verify failure: {output}");
    info!("Akita-in-Jolt proof OK");
}
