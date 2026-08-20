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

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use akita_config::proof_optimized::fp128;
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig};
use akita_recursion_glue::{AkitaJoltInputs, MAX_JOLT_BLOB_BYTES};
use akita_transcript::AkitaTranscript;
use akita_types::{prepared_verifier_ntt_cache_metadata, BasisMode};
use akita_verifier::{batched_verify, build_riscv64_terminal_ntt_cache};
use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

const TRUSTED_BENCHMARK_ARTIFACT_ENV: &str = "AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT";
const PREPARED_VERIFIER_CACHE_ENV: &str = "AKITA_RECURSION_PREPARED_VERIFIER_CACHE";
type F = fp128::Field;
type Cfg = RecursiveCommitmentConfig<fp128::OneHot>;
/// Concrete ring view used by the recursion artifact's fixed input schema.
const SOURCE_VIEW_D: usize = 512;

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

fn enable_trusted_benchmark_guest_build(prepared_cache: &Path) -> Result<(), String> {
    // The pinned Jolt SDK builds guest ELFs with a hard-coded `--features guest`.
    // This checked build-script cfg keeps plain `guest` strict while letting
    // this benchmark harness opt the RISC-V build into trusted setup decode.
    std::env::set_var(TRUSTED_BENCHMARK_ARTIFACT_ENV, "1");
    std::env::set_var(
        PREPARED_VERIFIER_CACHE_ENV,
        path_to_utf8(prepared_cache, "prepared verifier cache")?,
    );
    Ok(())
}

fn load_blob(input: &Path) -> Result<Vec<u8>, String> {
    let file = match File::open(input) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "verifier-input blob not found at `{}`.\n\
                     Generate one first with `akita-recursion-artifact`. For example:\n\n\
                         AKITA_NUM_VARS=32 ./target/release/akita-recursion-artifact\n\n\
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

fn strict_host_preflight(blob: &[u8]) -> Result<Vec<u8>, String> {
    info!("strictly decoding and verifying verifier-input blob before trusted benchmark replay");
    let decoded = AkitaJoltInputs::<F, SOURCE_VIEW_D>::read_from_bytes::<Cfg>(blob)
        .map_err(|err| format!("strict input decode failed: {err}"))?;
    let mut transcript = AkitaTranscript::<F>::unbound_verifier(&decoded.transcript_domain);
    let statement = decoded
        .verifier_statement()
        .map_err(|err| format!("strict input statement failed: {err}"))?;
    batched_verify::<Cfg, _>(
        &decoded.proof,
        &decoded.verifier_setup,
        &mut transcript,
        statement,
        BasisMode::Lagrange,
    )
    .map_err(|err| format!("strict host verifier rejected input blob: {err}"))?;
    let resolved = Cfg::resolve_schedule_selection(decoded.schedule_selection)
        .map_err(|err| format!("strict schedule resolution failed: {err}"))?;
    let cache = build_riscv64_terminal_ntt_cache(
        &decoded.verifier_setup,
        resolved.schedule(),
        decoded.schedule_selection.row_digest,
    )
    .map_err(|err| format!("prepared verifier cache build failed: {err}"))?;
    decoded
        .verifier_setup
        .install_trusted_prepared_verifier_ntt_cache(&cache, decoded.schedule_selection.row_digest)
        .map_err(|err| format!("prepared verifier cache self-check failed: {err}"))?;
    let mut cached_transcript = AkitaTranscript::<F>::unbound_verifier(&decoded.transcript_domain);
    let cached_statement = decoded
        .verifier_statement()
        .map_err(|err| format!("cached input statement failed: {err}"))?;
    batched_verify::<Cfg, _>(
        &decoded.proof,
        &decoded.verifier_setup,
        &mut cached_transcript,
        cached_statement,
        BasisMode::Lagrange,
    )
    .map_err(|err| format!("prepared verifier cache self-check rejected proof: {err}"))?;
    let metadata = prepared_verifier_ntt_cache_metadata(&cache)
        .map_err(|err| format!("prepared verifier cache metadata failed: {err}"))?;
    info!(
        cache_bytes = cache.len(),
        ring_d = metadata.ring_dimension,
        prefix_rings = metadata.base_prefix_len,
        width = metadata.width,
        "strict host preflight and prepared cache self-check OK"
    );
    Ok(cache)
}

fn digest_prefix(digest: &[u8; 32]) -> String {
    digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn publish_prepared_cache(target_dir: &Path, cache: &[u8]) -> Result<PathBuf, String> {
    let metadata = prepared_verifier_ntt_cache_metadata(cache)
        .map_err(|err| format!("prepared verifier cache metadata failed: {err}"))?;
    fs::create_dir_all(target_dir).map_err(|err| {
        format!(
            "failed to create Jolt target directory `{}`: {err}",
            target_dir.display()
        )
    })?;
    let file_name = format!(
        "akita-riscv64-q128-cache-{}-{}.bin",
        digest_prefix(&metadata.binding.setup_seed_digest),
        digest_prefix(metadata.binding.schedule_row_digest.as_bytes())
    );
    let output = target_dir.join(file_name);
    if output.exists() {
        let existing = fs::read(&output).map_err(|err| {
            format!(
                "failed to read existing prepared cache `{}`: {err}",
                output.display()
            )
        })?;
        if existing != cache {
            return Err(format!(
                "existing prepared cache `{}` disagrees with deterministic output",
                output.display()
            ));
        }
        return fs::canonicalize(&output).map_err(|err| {
            format!(
                "failed to resolve prepared cache `{}`: {err}",
                output.display()
            )
        });
    }
    let temporary = output.with_extension(format!("bin.tmp.{}", std::process::id()));
    fs::write(&temporary, cache).map_err(|err| {
        format!(
            "failed to write prepared cache `{}`: {err}",
            temporary.display()
        )
    })?;
    fs::rename(&temporary, &output).map_err(|err| {
        let _ = fs::remove_file(&temporary);
        format!(
            "failed to publish prepared cache `{}`: {err}",
            output.display()
        )
    })?;
    fs::canonicalize(&output).map_err(|err| {
        format!(
            "failed to resolve prepared cache `{}`: {err}",
            output.display()
        )
    })
}

fn run() -> Result<(), String> {
    let args = Args::parse();

    info!(input = %args.input.display(), "loading verifier-input blob");
    let blob = load_blob(&args.input)?;
    info!(bytes = blob.len(), "blob loaded");
    let prepared_cache = strict_host_preflight(&blob)?;
    let target_dir = PathBuf::from(&args.target_dir);
    let prepared_cache_path = publish_prepared_cache(&target_dir, &prepared_cache)?;

    info!(target_dir = %args.target_dir, "compiling Akita verifier guest program");
    enable_trusted_benchmark_guest_build(&prepared_cache_path)?;
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
