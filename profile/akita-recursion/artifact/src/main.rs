//! Generate an Akita verifier-input blob to be consumed by the Jolt guest
//! program in `profile/akita-recursion/guest`.
//!
//! Mirrors `run_profile_onehot_d32` from `crates/akita-pcs/examples/profile.rs`:
//! single-poly OneHot polynomial commitment in `D=32` mode at the canonical
//! `q=2^128-2^32+22537` prime, opened at one random point. After running the
//! prover end-to-end we re-run the host verifier as a sanity check, then
//! serialize all verifier-side state into one contiguous blob via
//! [`akita_recursion_glue::AkitaJoltInputs`].
//!
//! Output paths are controlled via `AKITA_RECURSION_BLOB` (defaults to
//! `target/akita_recursion_inputs.bin`). Set `AKITA_NUM_VARS` (default 20)
//! to regenerate at a different polynomial arity. Stick with `D=32 OneHot`
//! so the guest's hard-coded monomorphization can read the blob.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::{CanonicalField, PseudoMersenneField};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    compute::{OpeningFoldKernel, OpeningFoldPlan, RootOpeningSource},
    ComputeBackendSetup, CpuBackend, OneHotIndex, OneHotPoly,
    ProverCommitmentGroup, ProverOpeningBatch,
};
use akita_recursion_glue::AkitaJoltInputs;
use akita_transcript::AkitaTranscript;
use akita_types::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, BasisMode, BlockOrder,
    CommitmentGroup, LevelParams, OpeningBatchShape, PointVariableSelection, SetupContributionMode,
    VerifierOpeningBatch,
};
use akita_verifier::batched_verify;
use clap::{Parser, ValueEnum};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

/// Setup-contribution mode the proof is generated under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SetupModeArg {
    /// Evaluate the setup contribution directly from the expanded matrix.
    Direct,
    /// Embed the recursive setup-product sumcheck.
    Recursive,
}

impl SetupModeArg {
    fn into_mode(self) -> SetupContributionMode {
        match self {
            SetupModeArg::Direct => SetupContributionMode::Direct,
            SetupModeArg::Recursive => SetupContributionMode::Recursive,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    about = "Generate an Akita verifier-input blob for the Jolt recursion guest",
    long_about = None
)]
struct Args {
    /// Setup-contribution mode the proof is generated under. The blob records
    /// this so host preflight and guest replay verify under the same mode.
    #[arg(long, value_enum, default_value_t = SetupModeArg::Direct)]
    setup_mode: SetupModeArg,
}

type F = fp128::Field;
const D: usize = 32;
type Cfg = fp128::D32OneHot;
type Claim = <Cfg as CommitmentConfig>::ExtField;
type Challenge = <Cfg as CommitmentConfig>::ExtField;
const ONEHOT_K: usize = 256;

const TRANSCRIPT_DOMAIN: &[u8] = b"akita-recursion/onehot-d32";

fn onehot_k_for_num_vars(nv: usize) -> usize {
    let max_supported_log_k = ONEHOT_K.trailing_zeros() as usize;
    if nv >= max_supported_log_k {
        ONEHOT_K
    } else {
        1usize << nv
    }
}

fn opening_from_poly<'a, I>(
    poly: &'a OneHotPoly<F, D, I>,
    point: &[F],
    layout: &LevelParams,
    basis: BasisMode,
) -> Result<F, String>
where
    I: OneHotIndex,
    CpuBackend: OpeningFoldKernel<<OneHotPoly<F, D, I> as RootOpeningSource<F, D>>::OpeningView<'a>, F, D>,
{
    let alpha_bits = D.trailing_zeros() as usize;
    let target_num_vars = alpha_bits
        .checked_add(layout.m_vars)
        .and_then(|n| n.checked_add(layout.r_vars))
        .ok_or_else(|| "opening point target arity overflow".to_string())?;
    if point.len() > target_num_vars {
        return Err(format!(
            "opening point length {} exceeds target root arity {target_num_vars}",
            point.len()
        ));
    }
    let mut padded_point = point.to_vec();
    padded_point.resize(target_num_vars, F::zero());

    let inner_point = &padded_point[..alpha_bits];
    let reduced_point = &padded_point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        basis,
        BlockOrder::RowMajor,
    )
    .map_err(|err| format!("opening point shape should match layout: {err}"))?;

    let opening = OpeningFoldKernel::evaluate_and_fold(
        &CpuBackend,
        None,
        poly.opening_view()
            .map_err(|err| format!("opening view: {err}"))?,
        OpeningFoldPlan::Base {
            eval_outer_scalars: &ring_opening_point.b,
            fold_scalars: &ring_opening_point.a,
            block_len: layout.block_len,
        },
    )
    .map_err(|err| format!("opening fold: {err}"))?;
    let y_ring = opening.eval;
    let v = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis)
        .map_err(|err| format!("inner opening point should match ring dimension: {err}"))?;
    Ok((y_ring * v.sigma_m1()).coefficients()[0])
}

fn fp128_prime_label() -> String {
    match <F as PseudoMersenneField>::MODULUS_OFFSET {
        2355 => "q=2^128-2355".to_string(),
        0xFFFFA7F7 => "q=2^128-2^32+22537".to_string(),
        offset => format!("q=2^128-{offset:#x}"),
    }
}

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match env::var(name) {
        Ok(value) => match value.parse() {
            Ok(parsed) => Ok(parsed),
            Err(err) => Err(format!(
                "{name} must be a non-negative integer, got `{value}`: {err}"
            )),
        },
        Err(env::VarError::NotPresent) => Ok(default),
        Err(env::VarError::NotUnicode(value)) => Err(format!(
            "{name} must be valid Unicode, got `{}`",
            value.to_string_lossy()
        )),
    }
}

fn env_string(name: &str, default: &str) -> Result<String, String> {
    match env::var(name) {
        Ok(value) => Ok(value),
        Err(env::VarError::NotPresent) => Ok(default.to_string()),
        Err(env::VarError::NotUnicode(value)) => Err(format!(
            "{name} must be valid Unicode, got `{}`",
            value.to_string_lossy()
        )),
    }
}

fn publish_blob(output_path: &std::path::Path, blob: &[u8]) -> Result<(), String> {
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create output directory `{}`: {err}",
                parent.display()
            )
        })?;
    }
    let mut tmp_name = output_path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| "akita_recursion_inputs.bin".into());
    tmp_name.push(".tmp");
    let tmp_path = output_path.with_file_name(tmp_name);
    fs::write(&tmp_path, blob)
        .map_err(|err| format!("failed to write temp blob `{}`: {err}", tmp_path.display()))?;
    fs::rename(&tmp_path, output_path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        format!(
            "failed to publish blob `{}` from `{}`: {err}",
            output_path.display(),
            tmp_path.display()
        )
    })
}

fn verify_with_setup_mode(
    proof: &akita_types::AkitaBatchedProof<F, Challenge>,
    verifier_setup: &akita_types::AkitaVerifierSetup<F>,
    transcript: &mut AkitaTranscript<F>,
    claims: VerifierOpeningBatch<'_, Claim, &akita_types::Commitment<F>>,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), String> {
    batched_verify::<Cfg, _, D>(
        proof,
        verifier_setup,
        transcript,
        claims,
        BasisMode::Lagrange,
        setup_contribution_mode,
    )
    .map_err(|err| format!("{setup_contribution_mode:?}-mode verifier rejected proof: {err}"))
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let setup_contribution_mode = args.setup_mode.into_mode();

    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .ok();

    if cfg!(debug_assertions) && env::var("AKITA_ALLOW_DEBUG_PROFILE").as_deref() != Ok("1") {
        return Err(
            "akita-recursion-artifact must be run with --release for sane runtimes.\n\
             Re-run with: cargo run --release -p akita-recursion-artifact\n\
             Set AKITA_ALLOW_DEBUG_PROFILE=1 to override this guard."
                .to_string(),
        );
    }

    let log_filter =
        EnvFilter::try_new(env::var("AKITA_RECURSION_LOG").unwrap_or_else(|_| "info".to_string()))
            .unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(log_filter)
        .with_target(false)
        .try_init();

    let nv: usize = env_usize("AKITA_NUM_VARS", 20)?;
    let onehot_k = onehot_k_for_num_vars(nv);
    let output_path = PathBuf::from(env_string(
        "AKITA_RECURSION_BLOB",
        "target/akita_recursion_inputs.bin",
    )?);

    let prime = fp128_prime_label();
    tracing::info!(
        nv,
        d = D,
        onehot_k,
        prime = %prime,
        "generating Akita verifier-input artifact (single-poly OneHot, D=32)"
    );

    let layout: LevelParams = <Cfg as CommitmentConfig>::get_params_for_batched_commitment(
        &OpeningBatchShape::new(nv, 1).expect("singleton opening batch"),
    )
    .expect("layout");
    let alpha_bits = D.trailing_zeros() as usize;
    let required_vars = layout.m_vars + layout.r_vars + alpha_bits;
    // Both `main` (`required_vars <= nv`, layout fits in nv) and
    // `opening_from_poly` (`point.len() <= target_num_vars`, i.e.
    // `nv <= required_vars`) need to hold simultaneously, which means
    // they need to be equal. Catch the mismatch here with a clearer
    // message than the helper would emit.
    if required_vars != nv {
        return Err(format!(
            "OneHot D={D} layout at nv={nv} expects exactly {required_vars} variables \
             (alpha_bits={alpha_bits} + m_vars={} + r_vars={}); pick an AKITA_NUM_VARS that matches the layout",
            layout.m_vars, layout.r_vars
        ));
    }

    // The example reuses the deterministic seed from `examples/profile.rs`
    // for reproducibility.
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let total_ring = layout
        .num_blocks
        .checked_mul(layout.block_len)
        .ok_or_else(|| "total ring size overflow".to_string())?;
    let total_field = total_ring
        .checked_mul(D)
        .ok_or_else(|| "total field size overflow".to_string())?;
    let total_chunks = total_field / onehot_k;
    if total_chunks * onehot_k != total_field {
        return Err(format!(
            "OneHot K={onehot_k} must divide total field size {total_field} for nv={nv}"
        ));
    }

    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
        .collect();
    let onehot_poly = OneHotPoly::<F, D, u8>::new(onehot_k, indices)
        .map_err(|err| format!("failed to build onehot polynomial: {err}"))?;
    let opening_point: Vec<F> = (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let opening = opening_from_poly(&onehot_poly, &opening_point, &layout, BasisMode::Lagrange)?;

    let t0 = Instant::now();
    let prover_setup = match setup_contribution_mode {
        SetupContributionMode::Direct => {
            AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 1)
        }
        SetupContributionMode::Recursive => {
            AkitaCommitmentScheme::<Cfg>::setup_prover_recursion(
                nv, 1,
            )
        }
    }
    .map_err(|err| format!("prover setup failed: {err}"))?;
    let prepared = CpuBackend
        .prepare_setup(&prover_setup)
        .map_err(|err| format!("backend setup preparation failed: {err}"))?;
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend,
        &prepared,
        prover_setup.expanded.as_ref(),
    )
    .map_err(|err| format!("prover stack validation failed: {err}"))?;
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        "prover setup complete"
    );

    let t0 = Instant::now();
    let (commitment, hint) = AkitaCommitmentScheme::<Cfg>::commit(
        &prover_setup,
        std::slice::from_ref(&onehot_poly),
        &stack,
    )
    .map_err(|err| format!("commit failed: {err}"))?;
    tracing::info!(elapsed_s = t0.elapsed().as_secs_f64(), "commit complete");

    let poly_refs: [&OneHotPoly<F, D, u8>; 1] = [&onehot_poly];
    let openings = [opening];

    let t0 = Instant::now();
    let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    let prove_input = ProverOpeningBatch {
        point: opening_point[..].into(),
        groups: vec![ProverCommitmentGroup {
            point_vars: PointVariableSelection::prefix(opening_point.len(), opening_point.len())
                .map_err(|err| format!("invalid opening point shape: {err}"))?,
            polynomials: &poly_refs[..],
            commitment: (commitment.clone(), hint),
        }],
    };
    let proof = AkitaCommitmentScheme::<Cfg>::batched_prove(
        &prover_setup,
        prove_input,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        setup_contribution_mode,
    )
    .map_err(|err| format!("batched_prove failed: {err}"))?;
    tracing::info!(elapsed_s = t0.elapsed().as_secs_f64(), "prove complete");

    let verifier_setup =
        AkitaCommitmentScheme::<Cfg>::setup_verifier(&prover_setup);

    // Sanity check: the proof should verify with the same domain label.
    let t0 = Instant::now();
    let mut verifier_transcript = AkitaTranscript::<F>::unbound_verifier(TRANSCRIPT_DOMAIN);
    verify_with_setup_mode(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        VerifierOpeningBatch::from_groups(
            opening_point.clone(),
            vec![CommitmentGroup {
                claims: openings.to_vec(),
                commitment: &commitment,
            }],
        )
        .map_err(|err| format!("invalid verifier opening batch: {err}"))?,
        setup_contribution_mode,
    )
    .map_err(|err| format!("host-side sanity verify failed: {err}"))?;
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        "host-side verify OK"
    );

    let proof_shape = proof.shape();
    let inputs: AkitaJoltInputs<F, D> = AkitaJoltInputs {
        transcript_domain: TRANSCRIPT_DOMAIN.to_vec(),
        num_vars: nv as u64,
        setup_contribution_mode,
        opening_point,
        opening,
        commitment,
        verifier_setup,
        proof_shape,
        proof,
    };

    let blob = inputs
        .write_to_bytes()
        .map_err(|err| format!("encode jolt inputs blob failed: {err}"))?;
    // Round-trip before publishing so a buggy encoding fails on the host
    // instead of leaving a trusted benchmark artifact on disk.
    let decoded = AkitaJoltInputs::<F, D>::read_from_bytes(&blob)
        .map_err(|err| format!("decode jolt inputs blob (round-trip) failed: {err}"))?;
    let mut roundtrip_transcript =
        AkitaTranscript::<F>::unbound_verifier(&decoded.transcript_domain);
    let openings_rt = [decoded.opening];
    verify_with_setup_mode(
        &decoded.proof,
        &decoded.verifier_setup,
        &mut roundtrip_transcript,
        decoded.verifier_opening_batch(&openings_rt),
        decoded.setup_contribution_mode,
    )
    .map_err(|err| format!("decoded blob verify failed: {err}"))?;
    tracing::info!("decoded-blob verify OK");

    publish_blob(&output_path, &blob)?;

    let blob_kib = (blob.len() as f64) / 1024.0;
    let blob_mib = blob_kib / 1024.0;
    tracing::info!(
        nv,
        d = D,
        bytes = blob.len(),
        kib = blob_kib,
        mib = blob_mib,
        path = %output_path.display(),
        "wrote akita-recursion verifier-input blob"
    );
    eprintln!(
        "wrote {} bytes ({:.2} MiB) to {}",
        blob.len(),
        blob_mib,
        output_path.display()
    );
    Ok(())
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(2);
        }
    }
}
