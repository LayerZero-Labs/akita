//! Generate an Akita verifier-input blob to be consumed by the Jolt guest
//! program in `profile/akita-recursion/guest`.
//!
//! Mirrors the fp128 recursive multi-group one-hot profile from
//! `crates/akita-pcs/examples/profile.rs`: two 16-variable precommitted
//! one-hot groups plus two 32-variable final one-hot polynomials at the
//! canonical `q=2^128-2^32+22537` prime. After running the prover end-to-end
//! we re-run the host verifier as a sanity check, then serialize all
//! verifier-side state into one contiguous blob via
//! [`akita_recursion_glue::AkitaJoltInputs`].
//!
//! Output paths are controlled via `AKITA_RECURSION_BLOB` (defaults to
//! `target/akita_recursion_inputs.bin`). `AKITA_NUM_VARS` is pinned to 32 for
//! this grouped recursive row. The Jolt monomorphization uses the D512 root
//! envelope; the selected catalog row must use that A dimension.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig};
use akita_field::{CanonicalField, PseudoMersenneField};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    commit_setup_prefix, AkitaProverSetup, CommitOutput, ComputeBackendSetup, CpuBackend,
    GroupContext, OneHotPoly, SelectedProverOpeningData,
};
use akita_recursion_glue::AkitaJoltInputs;
use akita_transcript::AkitaTranscript;
use akita_types::{
    dispatch_for_field, lagrange_weights, AkitaScheduleLookupKey, BasisMode, CommittedGroup,
    GroupBatchStatement, OpeningClaims, PolynomialGroupClaims, PolynomialGroupLayout,
    PrecommittedGroupProfiles,
};
use akita_verifier::batched_verify;
use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    about = "Generate an Akita verifier-input blob for the Jolt recursion guest",
    long_about = None
)]
struct Args {}

type F = fp128::Field;
type BaseCfg = fp128::OneHot;
type Cfg = RecursiveCommitmentConfig<BaseCfg>;
/// Concrete root ring view used by the recursion artifact's fixed input schema.
/// The Akita schedule may select different B and D dimensions internally.
const SOURCE_VIEW_D: usize = 512;
type Claim = <Cfg as CommitmentConfig>::ExtField;
type Challenge = <Cfg as CommitmentConfig>::ExtField;
const ONEHOT_K: usize = akita_config::proof_optimized::STANDARD_ONEHOT_CHUNK_SIZE;
const PRE_GROUPS: usize = 2;
const PRE_NUM_VARS: usize = 16;
const FINAL_POLYS: usize = 2;

const TRANSCRIPT_DOMAIN: &[u8] = b"akita-recursion/onehot";

fn onehot_k_for_num_vars(nv: usize) -> usize {
    let max_supported_log_k = ONEHOT_K.trailing_zeros() as usize;
    if nv >= max_supported_log_k {
        ONEHOT_K
    } else {
        1usize << nv
    }
}

fn make_onehot_poly(num_vars: usize, seed: u64) -> Result<OneHotPoly<F, u8>, String> {
    let onehot_k = onehot_k_for_num_vars(num_vars);
    let total_field = 1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| format!("one-hot arity nv={num_vars} overflows usize"))?;
    let total_chunks = total_field / onehot_k;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
        .collect();
    OneHotPoly::<F, u8>::new(onehot_k, indices)
        .map_err(|err| format!("failed to build one-hot polynomial: {err}"))
}

fn onehot_opening(poly: &OneHotPoly<F, u8>, point: &[F]) -> Result<F, String> {
    if poly.indices().len() * poly.onehot_k() != (1usize << point.len()) {
        return Err(format!(
            "one-hot polynomial arity {} does not match opening point arity {}",
            poly.indices().len().trailing_zeros() as usize
                + poly.onehot_k().trailing_zeros() as usize,
            point.len()
        ));
    }
    let low_vars = poly.onehot_k().trailing_zeros() as usize;
    let low_weights = lagrange_weights(&point[..low_vars])
        .map_err(|err| format!("one-hot low opening weights: {err}"))?;
    let high_point = &point[low_vars..];
    let mut high_weight = high_point
        .iter()
        .copied()
        .map(|r| F::one() - r)
        .fold(F::one(), |acc, value| acc * value);
    let transitions = high_point
        .iter()
        .copied()
        .map(|r| {
            let one_minus_r = F::one() - r;
            let to_one = r * one_minus_r
                .inverse()
                .ok_or_else(|| "one-hot opening point contains a zero denominator".to_string())?;
            let to_zero = one_minus_r
                * r.inverse().ok_or_else(|| {
                    "one-hot opening point contains a zero denominator".to_string()
                })?;
            Ok((to_one, to_zero))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut opening = F::zero();
    let mut gray_index = 0usize;
    for step in 0..poly.indices().len() {
        if let Some(hot_idx) = poly.indices()[gray_index] {
            opening += high_weight * low_weights[hot_idx as usize];
        }
        let next_step = step + 1;
        if next_step == poly.indices().len() {
            break;
        }
        let next_gray = next_step ^ (next_step >> 1);
        let flipped_bit = (gray_index ^ next_gray).trailing_zeros() as usize;
        high_weight *= if next_gray & (1usize << flipped_bit) == 0 {
            transitions[flipped_bit].1
        } else {
            transitions[flipped_bit].0
        };
        gray_index = next_gray;
    }
    Ok(opening)
}

fn materialize_schedule_setup_prefix_slots(
    setup: &mut AkitaProverSetup<F>,
    backend: &CpuBackend,
    prepared: &<CpuBackend as ComputeBackendSetup<F>>::PreparedSetup,
    schedule: &akita_types::FoldSchedule,
) -> Result<(), akita_field::AkitaError> {
    for slot_id in schedule
        .recursive_folds
        .iter()
        .filter_map(|fold| fold.params.incoming_setup_prefix.as_ref())
    {
        if setup.prefix_slots.get(&slot_id.slot_id()).is_some() {
            continue;
        }
        let n_prefix = slot_id.n_prefix()?;
        let slot = dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            slot_id.d_setup(),
            |D_SETUP| {
                commit_setup_prefix::<F, D_SETUP, CpuBackend>(
                    &setup.expanded,
                    backend,
                    prepared,
                    &slot_id.commitment_params.layout,
                    n_prefix,
                    slot_id.natural_len,
                )
            }
        )?;
        setup.prefix_slots.insert(slot)?;
    }
    Ok(())
}

fn build_statement<'a>(
    selection: akita_types::OpeningScheduleSelection,
    pre_points: &'a [Vec<F>],
    pre_openings: &'a [Vec<F>],
    pre_commitments: &'a [CommittedGroup<F>],
    final_point: &'a [F],
    final_openings: Vec<F>,
    final_commitment: &'a CommittedGroup<F>,
) -> Result<GroupBatchStatement<'a, Claim, F>, String> {
    if pre_points.len() != PRE_GROUPS
        || pre_openings.len() != PRE_GROUPS
        || pre_commitments.len() != PRE_GROUPS
    {
        return Err("recursive artifact precommit group count mismatch".to_string());
    }
    let mut groups = Vec::with_capacity(PRE_GROUPS + 1);
    for group_idx in 0..PRE_GROUPS {
        groups.push(
            PolynomialGroupClaims::new(
                pre_points[group_idx].as_slice(),
                pre_openings[group_idx].clone(),
                &pre_commitments[group_idx],
            )
            .map_err(|err| format!("invalid precommit verifier group: {err}"))?,
        );
    }
    groups.push(
        PolynomialGroupClaims::new(final_point, final_openings, final_commitment)
            .map_err(|err| format!("invalid final verifier group: {err}"))?,
    );
    let claims = OpeningClaims::from_groups(groups)
        .map_err(|err| format!("invalid verifier opening claims: {err}"))?;
    GroupBatchStatement::new(selection, claims)
        .map_err(|err| format!("invalid verifier statement: {err}"))
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

fn verify_proof(
    proof: &akita_types::AkitaBatchedProof<F, Challenge>,
    verifier_setup: &akita_types::AkitaVerifierSetup<F>,
    transcript: &mut AkitaTranscript<F>,
    statement: GroupBatchStatement<'_, Claim, F>,
) -> Result<(), String> {
    batched_verify::<Cfg, _>(
        proof,
        verifier_setup,
        transcript,
        statement,
        BasisMode::Lagrange,
    )
    .map_err(|err| format!("verifier rejected proof: {err}"))
}

fn run() -> Result<(), String> {
    let _args = Args::parse();

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

    let nv: usize = env_usize("AKITA_NUM_VARS", 32)?;
    if nv != 32 {
        return Err(format!(
            "recursive OneHot benchmark is pinned to nv=32, got nv={nv}"
        ));
    }
    let onehot_k = onehot_k_for_num_vars(nv);
    let output_path = PathBuf::from(env_string(
        "AKITA_RECURSION_BLOB",
        "target/akita_recursion_inputs.bin",
    )?);

    let prime = fp128_prime_label();
    tracing::info!(
        nv,
        d = SOURCE_VIEW_D,
        onehot_k,
        prime = %prime,
        "generating Akita verifier-input artifact (recursive multi-group OneHot)"
    );

    let pre_group = PolynomialGroupLayout::new(PRE_NUM_VARS, 1);
    let pre_descriptor = BaseCfg::profile_without_precommitted_groups(pre_group)
        .map_err(|err| format!("precommit profile: {err}"))?;
    let final_group = PolynomialGroupLayout::new(nv, FINAL_POLYS);
    let key = AkitaScheduleLookupKey {
        final_group,
        precommitteds: vec![pre_descriptor; PRE_GROUPS],
    };
    let opening_layout = key
        .opening_layout()
        .map_err(|err| format!("recursive opening layout: {err}"))?;
    let schedule = Cfg::resolve_catalog_row_for_key(&key)
        .map_err(|err| format!("recursive proof schedule: {err}"))?;
    let layout = schedule
        .schedule()
        .root
        .params
        .final_group
        .commitment
        .clone();
    let alpha_bits = SOURCE_VIEW_D.trailing_zeros() as usize;
    let required_vars = layout.position_index_bits() + layout.block_index_bits() + alpha_bits;
    // Both `main` (`required_vars <= nv`, layout fits in nv) and
    // `opening_from_poly` (`point.len() <= target_num_vars`, i.e.
    // `nv <= required_vars`) need to hold simultaneously, which means
    // they need to be equal. Catch the mismatch here with a clearer
    // message than the helper would emit.
    if required_vars != nv {
        return Err(format!(
            "OneHot D={SOURCE_VIEW_D} layout at nv={nv} expects exactly {required_vars} variables \
             (alpha_bits={alpha_bits} + position_index_bits={} + block_index_bits={}); pick an AKITA_NUM_VARS that matches the layout",
            layout.position_index_bits(), layout.block_index_bits()
        ));
    }

    // The example reuses fixed deterministic seeds for reproducibility.
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let pre_points: Vec<Vec<F>> = (0..PRE_GROUPS)
        .map(|_| {
            (0..PRE_NUM_VARS)
                .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect()
        })
        .collect();
    let final_point: Vec<F> = (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();

    let t0 = Instant::now();
    let mut prover_setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, PRE_GROUPS + FINAL_POLYS)
        .map_err(|err| format!("prover setup failed: {err}"))?;
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&prover_setup)
        .map_err(|err| format!("backend setup preparation failed: {err}"))?;
    materialize_schedule_setup_prefix_slots(
        &mut prover_setup,
        &CpuBackend::DEFAULT,
        &prepared,
        schedule.schedule(),
    )
    .map_err(|err| format!("materialize recursive setup-prefix slots: {err}"))?;
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        prover_setup.expanded.as_ref(),
    )
    .map_err(|err| format!("prover stack validation failed: {err}"))?;
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        "prover setup complete"
    );

    let mut pre_polys_by_group = Vec::with_capacity(PRE_GROUPS);
    let mut pre_openings = Vec::with_capacity(PRE_GROUPS);
    let mut pre_commitments = Vec::with_capacity(PRE_GROUPS);
    let mut pre_hints = Vec::with_capacity(PRE_GROUPS);
    let t0 = Instant::now();
    for group_idx in 0..PRE_GROUPS {
        let polys = vec![make_onehot_poly(
            PRE_NUM_VARS,
            0x0bee_fcaf_2100_0000 + group_idx as u64,
        )?];
        let openings = vec![onehot_opening(&polys[0], &pre_points[group_idx])?];
        let CommitOutput {
            committed_group,
            hint,
        } = AkitaCommitmentScheme::<BaseCfg>::commit(
            &prover_setup,
            &polys,
            &stack,
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .map_err(|err| format!("precommit {group_idx} failed: {err}"))?;
        pre_polys_by_group.push(polys);
        pre_openings.push(openings);
        pre_commitments.push(committed_group);
        pre_hints.push(hint);
    }

    let final_polys = (0..FINAL_POLYS)
        .map(|poly_idx| make_onehot_poly(nv, 0x0bee_fcaf_2800_0000 + poly_idx as u64))
        .collect::<Result<Vec<_>, _>>()?;
    let final_openings = final_polys
        .iter()
        .map(|poly| onehot_opening(poly, &final_point))
        .collect::<Result<Vec<_>, _>>()?;
    let precommitteds = PrecommittedGroupProfiles::from_ordered_groups(pre_commitments.iter())
        .map_err(|err| format!("precommitted profile list: {err}"))?;
    let CommitOutput {
        committed_group: final_commitment,
        hint: final_hint,
    } = AkitaCommitmentScheme::<Cfg>::commit(
        &prover_setup,
        &final_polys,
        &stack,
        GroupContext::scheduler_with_precommitted_groups(&precommitteds),
    )
    .map_err(|err| format!("final multi-group commit failed: {err}"))?;
    tracing::info!(elapsed_s = t0.elapsed().as_secs_f64(), "commit complete");

    let pre_refs_by_group: Vec<Vec<&OneHotPoly<F, u8>>> = pre_polys_by_group
        .iter()
        .map(|polys| polys.iter().collect())
        .collect();
    let final_refs: Vec<&OneHotPoly<F, u8>> = final_polys.iter().collect();
    let mut poly_groups: Vec<&[&OneHotPoly<F, u8>]> =
        pre_refs_by_group.iter().map(Vec::as_slice).collect();
    poly_groups.push(final_refs.as_slice());
    let mut prover_groups = Vec::with_capacity(PRE_GROUPS + 1);
    for group_idx in 0..PRE_GROUPS {
        prover_groups.push(
            PolynomialGroupClaims::new(
                pre_points[group_idx].clone(),
                pre_openings[group_idx].clone(),
                pre_commitments[group_idx].clone(),
            )
            .map_err(|err| format!("invalid precommit prover group: {err}"))?,
        );
    }
    prover_groups.push(
        PolynomialGroupClaims::new(
            final_point.clone(),
            final_openings.clone(),
            final_commitment.clone(),
        )
        .map_err(|err| format!("invalid final prover group: {err}"))?,
    );
    let mut prover_hints = pre_hints;
    prover_hints.push(final_hint);
    let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    let prove_input = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        OpeningClaims::from_groups(prover_groups)
            .map_err(|err| format!("invalid prover opening claims: {err}"))?,
        prover_hints,
        poly_groups,
    )
    .map_err(|err| format!("invalid prover opening data: {err}"))?;
    let schedule_selection = prove_input.selection();
    let proof = AkitaCommitmentScheme::<Cfg>::batched_prove(
        &prover_setup,
        prove_input,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .map_err(|err| format!("batched_prove failed: {err}"))?;
    tracing::info!(elapsed_s = t0.elapsed().as_secs_f64(), "prove complete");

    let verifier_setup = AkitaCommitmentScheme::<Cfg>::setup_verifier_for_schedule(
        &prover_setup,
        schedule.schedule(),
        &opening_layout,
    )
    .map_err(|err| format!("setup_verifier_for_schedule failed: {err}"))?;

    // Sanity check: the proof should verify with the same domain label.
    let t0 = Instant::now();
    let mut verifier_transcript = AkitaTranscript::<F>::unbound_verifier(TRANSCRIPT_DOMAIN);
    verify_proof(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        build_statement(
            schedule_selection,
            &pre_points,
            &pre_openings,
            &pre_commitments,
            &final_point,
            final_openings.clone(),
            &final_commitment,
        )?,
    )
    .map_err(|err| format!("host-side sanity verify failed: {err}"))?;
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        "host-side verify OK"
    );

    let proof_shape = proof.shape();
    let inputs: AkitaJoltInputs<F, SOURCE_VIEW_D> = AkitaJoltInputs {
        transcript_domain: TRANSCRIPT_DOMAIN.to_vec(),
        num_vars: nv as u64,
        opening_point: final_point,
        openings: final_openings,
        precommitted_groups: pre_points
            .into_iter()
            .zip(pre_openings)
            .zip(pre_commitments.clone())
            .map(|((opening_point, openings), commitment)| {
                akita_recursion_glue::AkitaJoltOpeningGroup {
                    opening_point,
                    openings,
                    commitment,
                }
            })
            .collect(),
        schedule_selection,
        commitment: final_commitment,
        verifier_setup,
        proof_shape,
        proof,
    };

    let blob = inputs
        .write_to_bytes()
        .map_err(|err| format!("encode jolt inputs blob failed: {err}"))?;
    // Round-trip before publishing so a buggy encoding fails on the host
    // instead of leaving a trusted benchmark artifact on disk.
    let decoded = AkitaJoltInputs::<F, SOURCE_VIEW_D>::read_from_bytes::<Cfg>(&blob)
        .map_err(|err| format!("decode jolt inputs blob (round-trip) failed: {err}"))?;
    let mut roundtrip_transcript =
        AkitaTranscript::<F>::unbound_verifier(&decoded.transcript_domain);
    verify_proof(
        &decoded.proof,
        &decoded.verifier_setup,
        &mut roundtrip_transcript,
        decoded
            .verifier_statement()
            .map_err(|err| format!("decoded verifier statement failed: {err}"))?,
    )
    .map_err(|err| format!("decoded blob verify failed: {err}"))?;
    tracing::info!("decoded-blob verify OK");

    publish_blob(&output_path, &blob)?;

    let blob_kib = (blob.len() as f64) / 1024.0;
    let blob_mib = blob_kib / 1024.0;
    tracing::info!(
        nv,
        d = SOURCE_VIEW_D,
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
