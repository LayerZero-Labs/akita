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

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_recursion_glue::{AkitaJoltInputs, MAX_JOLT_BLOB_BYTES};
use akita_transcript::AkitaTranscript;
use akita_types::BasisMode;
use akita_verifier::batched_verify;
use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

const TRUSTED_BENCHMARK_ARTIFACT_ENV: &str = "AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT";
type F = fp128::Field;
const D: usize = 32;
type Cfg = fp128::D32OneHot;

const _: () = {
    assert!(D == <Cfg as CommitmentConfig>::D);
};

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

fn run_native_guest(blob: &[u8]) -> Result<(), String> {
    info!("running guest natively (sanity check)");
    let native_output = guest::akita_verify(blob);
    info!(native_output, "native guest output");
    if native_output != 0 {
        return Err(format!(
            "native guest run reported failure code {native_output}"
        ));
    }
    Ok(())
}

fn path_to_utf8<'a>(path: &'a Path, context: &str) -> Result<&'a str, String> {
    match path.to_str() {
        Some(path) => Ok(path),
        None => Err(format!(
            "{context} must be valid UTF-8: `{}`",
            path.display()
        )),
    }
}

fn enable_trusted_benchmark_guest_build() {
    // The pinned Jolt SDK builds guest ELFs with a hard-coded `--features guest`.
    // This checked build-script cfg keeps plain `guest` strict while letting
    // this benchmark harness opt the RISC-V build into trusted setup decode.
    std::env::set_var(TRUSTED_BENCHMARK_ARTIFACT_ENV, "1");
}

fn load_blob(input: &Path) -> Result<Vec<u8>, String> {
    let file = match File::open(input) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "verifier-input blob not found at `{}`.\n\
                     Generate one first with `akita-recursion-artifact`. For example:\n\n\
                         AKITA_NUM_VARS=20 ./target/release/akita-recursion-artifact\n\n\
                     or, for a different blob path / arity:\n\n\
                         AKITA_NUM_VARS=32 AKITA_RECURSION_BLOB={} \\\n\
                             ./target/release/akita-recursion-artifact",
                input.display(),
                input.display()
            ));
        }
        Err(err) => return Err(format!("failed to open `{}`: {err}", input.display())),
    };
    let metadata = file
        .metadata()
        .map_err(|err| format!("failed to stat `{}`: {err}", input.display()))?;
    if !metadata.file_type().is_file() {
        return Err(format!(
            "verifier-input blob `{}` must be a regular file",
            input.display()
        ));
    }
    if metadata.len() > MAX_JOLT_BLOB_BYTES {
        return Err(format!(
            "verifier-input blob `{}` is {} bytes, exceeding max {} bytes",
            input.display(),
            metadata.len(),
            MAX_JOLT_BLOB_BYTES
        ));
    }
    let mut reader = file.take(MAX_JOLT_BLOB_BYTES + 1);
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    reader
        .read_to_end(&mut bytes)
        .map_err(|err| format!("failed to read `{}`: {err}", input.display()))?;
    if bytes.len() as u64 > MAX_JOLT_BLOB_BYTES {
        return Err(format!(
            "verifier-input blob `{}` exceeded max {} bytes while reading",
            input.display(),
            MAX_JOLT_BLOB_BYTES
        ));
    }
    Ok(bytes)
}

fn strict_host_preflight(blob: &[u8]) -> Result<(), String> {
    info!("strictly decoding and verifying verifier-input blob before trusted benchmark replay");
    let decoded = AkitaJoltInputs::<F, D>::read_from_bytes(blob)
        .map_err(|err| format!("strict input decode failed: {err}"))?;
    let mut transcript = AkitaTranscript::<F>::unbound_verifier(&decoded.transcript_domain);
    let openings = [decoded.opening];
    batched_verify::<Cfg, _>(
        &decoded.proof,
        &decoded.verifier_setup,
        &mut transcript,
        decoded.verifier_opening_batch(&openings),
        BasisMode::Lagrange,
        decoded.setup_contribution_mode,
    )
    .map_err(|err| format!("strict host verifier rejected input blob: {err}"))?;
    info!("strict host preflight OK");
    Ok(())
}

fn run() -> Result<(), String> {
    let args = Args::parse();

    info!(input = %args.input.display(), "loading verifier-input blob");
    let blob = load_blob(&args.input)?;
    info!(bytes = blob.len(), "blob loaded");
    strict_host_preflight(&blob)?;

    info!(target_dir = %args.target_dir, "compiling Akita verifier guest program");
    enable_trusted_benchmark_guest_build();
    let mut program = guest::compile_akita_verify(&args.target_dir);

    if args.trace_only {
        info!("trace-only mode: skipping preprocessing and proof generation");
        run_native_guest(&blob)?;

        let trace_path = args
            .trace_output
            .unwrap_or_else(|| PathBuf::from(&args.target_dir).join("akita_verify.trace"));
        info!(trace_file = %trace_path.display(), "tracing guest under emulator");
        guest::trace_akita_verify_to_file(path_to_utf8(&trace_path, "--trace-output")?, &blob);
        info!("trace done");
        return Ok(());
    }

    info!("running shared / prover / verifier preprocessing");
    let shared_preprocessing = guest::preprocess_shared_akita_verify(&mut program)
        .map_err(|err| format!("shared preprocessing failed: {err}"))?;
    let prover_preprocessing = guest::preprocess_prover_akita_verify(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_akita_verify(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
        None,
    );

    let prove_akita_verify = guest::build_prover_akita_verify(program, prover_preprocessing);
    let verify_akita_verify = guest::build_verifier_akita_verify(verifier_preprocessing);

    run_native_guest(&blob)?;

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

    if !is_valid {
        return Err("Jolt verifier rejected the proof".to_string());
    }
    if output != 0 {
        return Err(format!("guest reported Akita-verify failure: {output}"));
    }
    info!("Akita-in-Jolt proof OK");
    Ok(())
}

fn main() -> ExitCode {
    let filter =
        EnvFilter::try_from_env("AKITA_RECURSION_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
