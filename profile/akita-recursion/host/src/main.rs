//! Host driver that compiles the Jolt guest program in
//! `profile/akita-recursion/guest`, feeds it the
//! [`akita_recursion_glue::AkitaJoltInputs`] blob produced by
//! `profile/akita-recursion/artifact`, and proves that the Akita verifier
//! returns successfully.
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

const TRUSTED_BENCHMARK_ARTIFACT_ENV: &str = "AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT";

#[derive(Debug, Parser)]
#[command(
    about = "Prove the Akita verifier inside Jolt and report cycle counts",
    long_about = None
)]
struct Args {
    /// Path to the verifier-input blob produced by the `artifact` binary
    /// (`profile/akita-recursion/artifact`).
    #[arg(long, default_value = "target/akita_recursion_inputs.bin")]
    input: PathBuf,

    /// Directory used by Jolt for per-program build artifacts.
    #[arg(long, default_value = "/tmp/akita-recursion-targets")]
    target_dir: String,

    /// Trace file path for `--trace-only`; defaults to
    /// `<target-dir>/akita_verify.trace`.
    #[arg(long)]
    trace_output: Option<PathBuf>,

    /// Only trace the guest (skips the ~minute-long Jolt prover step).
    /// Useful when iterating on guest panics with `JOLT_BACKTRACE=full`.
    #[arg(long, default_value_t = false)]
    trace_only: bool,
}

fn run_native_guest_or_exit(blob: &[u8]) {
    info!("running guest natively (sanity check)");
    let native_output = guest::akita_verify(blob);
    info!(native_output, "native guest output");
    if native_output != 0 {
        eprintln!("error: native guest run reported failure code {native_output}");
        std::process::exit(1);
    }
}

fn path_to_utf8_or_exit<'a>(path: &'a std::path::Path, context: &str) -> &'a str {
    match path.to_str() {
        Some(path) => path,
        None => {
            eprintln!("error: {context} must be valid UTF-8: `{}`", path.display());
            std::process::exit(2);
        }
    }
}

fn enable_trusted_benchmark_guest_build() {
    // The pinned Jolt SDK builds guest ELFs with a hard-coded `--features guest`.
    // This checked build-script cfg keeps plain `guest` strict while letting
    // this benchmark harness opt the RISC-V build into trusted setup decode.
    std::env::set_var(TRUSTED_BENCHMARK_ARTIFACT_ENV, "1");
}

fn main() {
    let filter =
        EnvFilter::try_from_env("AKITA_RECURSION_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let args = Args::parse();

    info!(input = %args.input.display(), "loading verifier-input blob");
    let blob = match std::fs::read(&args.input) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "error: verifier-input blob not found at `{}`.",
                args.input.display()
            );
            eprintln!("Generate one first with `akita-recursion-artifact`. For example:");
            eprintln!();
            eprintln!("    AKITA_NUM_VARS=20 ./target/release/akita-recursion-artifact");
            eprintln!();
            eprintln!("or, for a different blob path / arity:");
            eprintln!();
            eprintln!(
                "    AKITA_NUM_VARS=32 AKITA_RECURSION_BLOB={} \\",
                args.input.display()
            );
            eprintln!("        ./target/release/akita-recursion-artifact");
            std::process::exit(2);
        }
        Err(err) => {
            eprintln!("error: failed to read `{}`: {}", args.input.display(), err);
            std::process::exit(2);
        }
    };
    info!(bytes = blob.len(), "blob loaded");

    info!(target_dir = %args.target_dir, "compiling Akita verifier guest program");
    enable_trusted_benchmark_guest_build();
    let mut program = guest::compile_akita_verify(&args.target_dir);

    if args.trace_only {
        info!("trace-only mode: skipping preprocessing and proof generation");
        run_native_guest_or_exit(&blob);

        let trace_path = args
            .trace_output
            .unwrap_or_else(|| PathBuf::from(&args.target_dir).join("akita_verify.trace"));
        info!(trace_file = %trace_path.display(), "tracing guest under emulator");
        guest::trace_akita_verify_to_file(
            path_to_utf8_or_exit(&trace_path, "--trace-output"),
            &blob,
        );
        info!("trace done");
        return;
    }

    info!("running shared / prover / verifier preprocessing");
    let shared_preprocessing =
        guest::preprocess_shared_akita_verify(&mut program).expect("shared preprocessing");
    let prover_preprocessing = guest::preprocess_prover_akita_verify(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_akita_verify(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
        None,
    );

    let prove_akita_verify = guest::build_prover_akita_verify(program, prover_preprocessing);
    let verify_akita_verify = guest::build_verifier_akita_verify(verifier_preprocessing);

    run_native_guest_or_exit(&blob);

    info!("invoking Jolt prover");
    let now = Instant::now();
    let (output, proof, program_io) = prove_akita_verify(&blob);
    let prover_secs = now.elapsed().as_secs_f64();
    info!(prover_secs, "prover finished");
    info!(
        guest_output = output,
        guest_panic = program_io.panic,
        "prover program-io"
    );

    let now = Instant::now();
    let is_valid = verify_akita_verify(&blob, output, program_io.panic, proof);
    let verifier_secs = now.elapsed().as_secs_f64();
    info!(verifier_secs, is_valid, "Jolt verifier finished");

    assert!(is_valid, "Jolt verifier rejected the proof");
    assert_eq!(output, 0, "guest reported Akita-verify failure: {output}");
    info!("Akita-in-Jolt proof OK");
}
