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

use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig};
use akita_recursion_glue::{read_blob_case, AkitaJoltCase, AkitaJoltInputs, MAX_JOLT_BLOB_BYTES};
use akita_transcript::AkitaTranscript;
use akita_types::{prepared_verifier_ntt_cache_metadata, BasisMode};
use akita_verifier::{batched_verify, build_riscv64_terminal_ntt_cache};
use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

const TRUSTED_BENCHMARK_ARTIFACT_ENV: &str = "AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT";
const PREPARED_VERIFIER_CACHE_ENV: &str = "AKITA_RECURSION_PREPARED_VERIFIER_CACHE";
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

fn run_native_guest(case: AkitaJoltCase, blob: &[u8]) -> Result<(), String> {
    info!("running guest natively (sanity check)");
    let native_output = match case {
        AkitaJoltCase::OneHotFp32 => guest::akita_verify_fp32(blob),
        AkitaJoltCase::OneHotFp64 => guest::akita_verify_fp64(blob),
        AkitaJoltCase::OneHotFp128Direct => guest::akita_verify_fp128_direct(blob),
        AkitaJoltCase::OneHotFp128Recursive => guest::akita_verify_fp128_recursive(blob),
        AkitaJoltCase::OneHotFp128MultiGroupRecursive => guest::akita_verify(blob),
    };
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

fn enable_benchmark_guest_build(prepared_cache: Option<&Path>) -> Result<(), String> {
    // The pinned Jolt SDK builds guest ELFs with a hard-coded `--features guest`.
    // This checked build-script cfg keeps plain `guest` strict while letting
    // this benchmark harness opt the RISC-V build into trusted setup decode.
    std::env::set_var(TRUSTED_BENCHMARK_ARTIFACT_ENV, "1");
    match prepared_cache {
        Some(prepared_cache) => std::env::set_var(
            PREPARED_VERIFIER_CACHE_ENV,
            path_to_utf8(prepared_cache, "prepared verifier cache")?,
        ),
        None => std::env::remove_var(PREPARED_VERIFIER_CACHE_ENV),
    }
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

macro_rules! strict_decode_and_verify {
    ($blob:expr, $field:ty, $cfg:ty, $d:expr) => {{
        type CaseExt = <$cfg as CommitmentConfig>::ExtField;
        let (schedules, inner_blob) =
            akita_recursion_glue::split_schedule_catalog::<$cfg>($blob)
                .map_err(|err| format!("trusted schedule catalog failed: {err}"))?;
        let decoded =
            AkitaJoltInputs::<$field, $d, CaseExt>::read_from_bytes::<$cfg>(inner_blob, &schedules)
                .map_err(|err| format!("strict input decode failed: {err}"))?;
        let mut transcript =
            AkitaTranscript::<$field>::unbound_verifier(&decoded.transcript_domain);
        batched_verify::<$cfg, _>(
            &decoded.proof,
            &decoded.verifier_setup,
            &schedules,
            &mut transcript,
            decoded
                .verifier_statement()
                .map_err(|err| format!("strict input statement failed: {err}"))?,
            BasisMode::Lagrange,
        )
        .map_err(|err| format!("strict host verifier rejected input blob: {err}"))?;
        (decoded, schedules)
    }};
}

macro_rules! strict_fp128_preflight {
    ($blob:expr, $cfg:ty) => {{
        let (decoded, schedules) = strict_decode_and_verify!($blob, fp128::Field, $cfg, 512);
        let resolved = schedules
            .resolve_selection(decoded.schedule_selection)
            .map_err(|err| format!("strict schedule resolution failed: {err}"))?;
        let cache = build_riscv64_terminal_ntt_cache(
            &decoded.verifier_setup,
            resolved.schedule(),
            decoded.schedule_selection.row_digest,
        )
        .map_err(|err| format!("prepared verifier cache build failed: {err}"))?;
        decoded
            .verifier_setup
            .install_trusted_prepared_verifier_ntt_cache(
                &cache,
                decoded.schedule_selection.row_digest,
            )
            .map_err(|err| format!("prepared verifier cache self-check failed: {err}"))?;
        let mut cached_transcript =
            AkitaTranscript::<fp128::Field>::unbound_verifier(&decoded.transcript_domain);
        batched_verify::<$cfg, _>(
            &decoded.proof,
            &decoded.verifier_setup,
            &schedules,
            &mut cached_transcript,
            decoded
                .verifier_statement()
                .map_err(|err| format!("cached input statement failed: {err}"))?,
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
        Ok(Some(cache))
    }};
}

fn strict_host_preflight(case: AkitaJoltCase, blob: &[u8]) -> Result<Option<Vec<u8>>, String> {
    info!(%case, "strictly decoding and verifying verifier-input blob before benchmark replay");
    match case {
        AkitaJoltCase::OneHotFp32 => {
            let (_decoded, _schedules) =
                strict_decode_and_verify!(blob, fp32::Field, fp32::OneHot, 2048);
            Ok(None)
        }
        AkitaJoltCase::OneHotFp64 => {
            let (_decoded, _schedules) =
                strict_decode_and_verify!(blob, fp64::Field, fp64::OneHot, 512);
            Ok(None)
        }
        AkitaJoltCase::OneHotFp128Direct => {
            strict_fp128_preflight!(blob, fp128::OneHot)
        }
        AkitaJoltCase::OneHotFp128Recursive => {
            strict_fp128_preflight!(blob, RecursiveCommitmentConfig<fp128::OneHot>)
        }
        AkitaJoltCase::OneHotFp128MultiGroupRecursive => {
            strict_fp128_preflight!(blob, RecursiveCommitmentConfig<fp128::OneHot>)
        }
    }
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

macro_rules! run_selected_guest {
    (
        $args:expr,
        $blob:expr,
        $case:expr,
        $compile:ident,
        $trace:ident,
        $preprocess_shared:ident,
        $preprocess_prover:ident,
        $preprocess_verifier:ident,
        $build_prover:ident,
        $build_verifier:ident
    ) => {{
        info!(case = %$case, target_dir = %$args.target_dir, "compiling Akita verifier guest program");
        let mut program = guest::$compile(&$args.target_dir);

        if $args.trace_only {
            info!(case = %$case, "trace-only mode: skipping preprocessing and proof generation");
            run_native_guest($case, $blob)?;
            let trace_path = $args.trace_output.clone().unwrap_or_else(|| {
                let case_file = $case.as_str().replace(':', "_");
                PathBuf::from(&$args.target_dir).join(format!("{case_file}.trace"))
            });
            info!(trace_file = %trace_path.display(), "tracing guest under emulator");
            guest::$trace(path_to_utf8(&trace_path, "--trace-output")?, $blob);
            info!(case = %$case, "trace done");
            return Ok(());
        }

        info!(case = %$case, "running shared / prover / verifier preprocessing");
        let shared_preprocessing = guest::$preprocess_shared(&mut program)
            .map_err(|err| format!("shared preprocessing failed: {err}"))?;
        let prover_preprocessing = guest::$preprocess_prover(shared_preprocessing.clone());
        let verifier_preprocessing = guest::$preprocess_verifier(
            shared_preprocessing,
            prover_preprocessing.generators.to_verifier_setup(),
            None,
        );
        let prove = guest::$build_prover(program, prover_preprocessing);
        let verify = guest::$build_verifier(verifier_preprocessing);

        run_native_guest($case, $blob)?;
        info!(case = %$case, "invoking Jolt prover");
        let now = Instant::now();
        let (output, proof, program_io) = prove($blob);
        let prover_secs = now.elapsed().as_secs_f64();
        info!(prover_secs, "prover finished");
        info!(
            guest_output = output,
            guest_panic = program_io.panic,
            "prover program-io"
        );

        let now = Instant::now();
        let is_valid = verify($blob, output, program_io.panic, proof);
        let verifier_secs = now.elapsed().as_secs_f64();
        info!(verifier_secs, is_valid, "Jolt verifier finished");
        if !is_valid {
            return Err("Jolt verifier rejected the proof".to_string());
        }
        if output != 0 {
            return Err(format!("guest reported Akita-verify failure: {output}"));
        }
        info!(case = %$case, "Akita-in-Jolt proof OK");
        Ok(())
    }};
}

fn run() -> Result<(), String> {
    let args = Args::parse();

    info!(input = %args.input.display(), "loading verifier-input blob");
    let blob = load_blob(&args.input)?;
    let case = read_blob_case(&blob).map_err(|err| format!("read blob case identity: {err}"))?;
    info!(%case, bytes = blob.len(), "blob loaded");
    let prepared_cache = strict_host_preflight(case, &blob)?;
    let target_dir = PathBuf::from(&args.target_dir);
    let prepared_cache_path = prepared_cache
        .as_deref()
        .map(|cache| publish_prepared_cache(&target_dir, cache))
        .transpose()?;
    enable_benchmark_guest_build(prepared_cache_path.as_deref())?;

    match case {
        AkitaJoltCase::OneHotFp32 => run_selected_guest!(
            args,
            &blob,
            case,
            compile_akita_verify_fp32,
            trace_akita_verify_fp32_to_file,
            preprocess_shared_akita_verify_fp32,
            preprocess_prover_akita_verify_fp32,
            preprocess_verifier_akita_verify_fp32,
            build_prover_akita_verify_fp32,
            build_verifier_akita_verify_fp32
        ),
        AkitaJoltCase::OneHotFp64 => run_selected_guest!(
            args,
            &blob,
            case,
            compile_akita_verify_fp64,
            trace_akita_verify_fp64_to_file,
            preprocess_shared_akita_verify_fp64,
            preprocess_prover_akita_verify_fp64,
            preprocess_verifier_akita_verify_fp64,
            build_prover_akita_verify_fp64,
            build_verifier_akita_verify_fp64
        ),
        AkitaJoltCase::OneHotFp128Direct => run_selected_guest!(
            args,
            &blob,
            case,
            compile_akita_verify_fp128_direct,
            trace_akita_verify_fp128_direct_to_file,
            preprocess_shared_akita_verify_fp128_direct,
            preprocess_prover_akita_verify_fp128_direct,
            preprocess_verifier_akita_verify_fp128_direct,
            build_prover_akita_verify_fp128_direct,
            build_verifier_akita_verify_fp128_direct
        ),
        AkitaJoltCase::OneHotFp128Recursive => run_selected_guest!(
            args,
            &blob,
            case,
            compile_akita_verify_fp128_recursive,
            trace_akita_verify_fp128_recursive_to_file,
            preprocess_shared_akita_verify_fp128_recursive,
            preprocess_prover_akita_verify_fp128_recursive,
            preprocess_verifier_akita_verify_fp128_recursive,
            build_prover_akita_verify_fp128_recursive,
            build_verifier_akita_verify_fp128_recursive
        ),
        AkitaJoltCase::OneHotFp128MultiGroupRecursive => run_selected_guest!(
            args,
            &blob,
            case,
            compile_akita_verify,
            trace_akita_verify_to_file,
            preprocess_shared_akita_verify,
            preprocess_prover_akita_verify,
            preprocess_verifier_akita_verify,
            build_prover_akita_verify,
            build_verifier_akita_verify
        ),
    }
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
