//! Generate an Akita verifier-input blob to be consumed by the Jolt guest
//! program in `profile/akita-recursion/guest`.
//!
//! Supports the exact scalar OneHot cases in the CI profile catalog, plus the
//! older fp128 nv32 recursive multi-group example. After running the prover
//! end to end, it reruns the host verifier as a sanity check and serializes
//! the case identity and verifier-side state via
//! [`akita_recursion_glue::AkitaJoltInputs`].
//!
//! Output paths are controlled via `AKITA_RECURSION_BLOB` (defaults to
//! `target/akita_recursion_inputs.bin`). `--case` or `AKITA_RECURSION_CASE`
//! selects a catalog case. `AKITA_NUM_VARS` applies only to the legacy grouped
//! row and is pinned to 32.

#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::{derive_transcript_grinding_plan, CommitmentConfig, RecursiveCommitmentConfig};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    commit_setup_prefix, AkitaProverSetup, CommitOutput, ComputeBackendSetup, CpuBackend,
    GroupContext, OneHotPoly, SelectedProverOpeningData,
};
use akita_recursion_glue::{AkitaJoltCase, AkitaJoltInputs};
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::AkitaTranscript;
use akita_types::{
    dispatch_for_field, lagrange_weights, AkitaScheduleLookupKey, BasisMode, CommittedGroup,
    FpExtEncoding, GroupBatchStatement, OpeningClaims, OpeningClaimsLayout, PolynomialGroupClaims,
    PolynomialGroupLayout, PrecommittedGroupProfiles,
};
use akita_verifier::batched_verify;
use clap::Parser;
use jolt_field::{CanonicalEncoding, ExtField, Field, Fold, PseudoMersenne, Ring, Unreduced};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

fn load_workspace_scheme<Cfg>() -> Result<AkitaCommitmentScheme<Cfg>, akita_error::AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field + CanonicalEncoding + Unreduced + PseudoMersenne + Valid + AkitaSerialize,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: ExtField<Cfg::Field> + Ring + Unreduced + Fold + AkitaSerialize,
{
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("artifacts/schedules")
        .join(format!("{}.aks", Cfg::schedule_family_name()));
    let bytes = fs::read(&path).map_err(|error| {
        akita_error::AkitaError::InvalidSetup(format!(
            "failed to read workspace schedule artifact {}: {error}",
            path.display()
        ))
    })?;
    AkitaCommitmentScheme::from_schedule_artifact(&bytes)
}

#[derive(Debug, Parser)]
#[command(
    about = "Generate an Akita verifier-input blob for the Jolt recursion guest",
    long_about = None
)]
struct Args {
    /// Exact CI case to materialize.
    #[arg(long)]
    case: Option<String>,
}

type F = fp128::Field;
type BaseCfg = fp128::OneHot;
type Cfg = RecursiveCommitmentConfig<BaseCfg>;
/// Concrete root ring view used by the recursion artifact's fixed input schema.
/// The Akita schedule may select different B and D dimensions internally.
const SOURCE_VIEW_D: usize = 512;
type Claim = <Cfg as CommitmentConfig>::ExtField;
type Challenge = <Cfg as CommitmentConfig>::ExtField;
const PRE_GROUPS: usize = 2;
const PRE_NUM_VARS: usize = 16;
const FINAL_POLYS: usize = 2;

const TRANSCRIPT_DOMAIN: &[u8] = b"akita-recursion/onehot";

fn onehot_k_for_num_vars(nv: usize) -> usize {
    let source_chunk_size = akita_config::unit_onehot_source_chunk_size::<BaseCfg>()
        .expect("recursion artifact requires a unit-one-hot base config");
    let max_supported_log_k = source_chunk_size.trailing_zeros() as usize;
    if nv >= max_supported_log_k {
        source_chunk_size
    } else {
        1usize << nv
    }
}

fn make_onehot_poly<FF>(num_vars: usize, seed: u64) -> Result<OneHotPoly<FF, u8>, String>
where
    FF: Field + CanonicalEncoding,
{
    let onehot_k = onehot_k_for_num_vars(num_vars);
    let total_field = 1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| format!("one-hot arity nv={num_vars} overflows usize"))?;
    let total_chunks = total_field / onehot_k;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
        .collect();
    OneHotPoly::<FF, u8>::new(onehot_k, indices)
        .map_err(|err| format!("failed to build one-hot polynomial: {err}"))
}

fn onehot_opening<FF, E>(poly: &OneHotPoly<FF, u8>, point: &[E]) -> Result<E, String>
where
    FF: Field + CanonicalEncoding,
    E: ExtField<FF>,
{
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
        .map(|r| E::one() - r)
        .fold(E::one(), |acc, value| acc * value);
    let transitions = high_point
        .iter()
        .copied()
        .map(|r| {
            let one_minus_r = E::one() - r;
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
    let mut opening = E::zero();
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

fn materialize_schedule_setup_prefix_slots<FF>(
    setup: &mut AkitaProverSetup<FF>,
    backend: &CpuBackend,
    prepared: &<CpuBackend as ComputeBackendSetup<FF>>::PreparedSetup,
    schedule: &akita_types::FoldSchedule,
) -> Result<(), akita_error::AkitaError>
where
    FF: Field + CanonicalEncoding + Valid,
    CpuBackend: ComputeBackendSetup<FF>,
{
    for setup_prefix in schedule
        .recursive_folds
        .iter()
        .filter_map(|fold| fold.incoming_setup_prefix())
    {
        let slot_id = setup_prefix.slot_id().ok_or_else(|| {
            akita_error::AkitaError::InvalidSetup("group is not a setup prefix".into())
        })?;
        if setup.prefix_slots.get(&slot_id).is_some() {
            continue;
        }
        let n_prefix = slot_id.n_prefix()?;
        let slot = dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            FF,
            slot_id.d_setup(),
            |D_SETUP| {
                commit_setup_prefix::<FF, D_SETUP, CpuBackend>(
                    &setup.expanded,
                    backend,
                    prepared,
                    &slot_id.commitment_profile,
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
    for ((opening_point, openings), commitment) in
        pre_points.iter().zip(pre_openings).zip(pre_commitments)
    {
        groups.push(
            PolynomialGroupClaims::new(opening_point.as_slice(), openings.clone(), commitment)
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
    match <F as PseudoMersenne>::OFFSET {
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
    schedules: &akita_config::TrustedScheduleCatalog,
    transcript: &mut AkitaTranscript<F>,
    statement: GroupBatchStatement<'_, Claim, F>,
) -> Result<(), String> {
    batched_verify::<Cfg, _>(
        proof,
        verifier_setup,
        schedules,
        transcript,
        statement,
        BasisMode::Lagrange,
    )
    .map_err(|err| format!("verifier rejected proof: {err}"))
}

fn random_claim_point<FF, E>(num_vars: usize, seed: u64) -> Vec<E>
where
    FF: Field + CanonicalEncoding,
    E: ExtField<FF>,
{
    let mut rng = StdRng::seed_from_u64(seed);
    (0..num_vars)
        .map(|_| {
            let limbs = (0..E::DEGREE)
                .map(|_| FF::from_u128_reduced(rng.gen::<u128>()))
                .collect::<Vec<_>>();
            E::from_base_slice(&limbs)
        })
        .collect()
}

macro_rules! generate_scalar_case {
    ($case:expr, $field:ty, $cfg:ty, $d:expr, $nv:expr, $recursive:expr, $output_path:expr) => {{
        type ScalarField = $field;
        type ScalarCfg = $cfg;
        type ScalarExt = <ScalarCfg as CommitmentConfig>::ExtField;

        let case = $case;
        let num_vars = $nv;
        let scheme = load_workspace_scheme::<ScalarCfg>()
            .map_err(|err| format!("{} trusted schedule catalog: {err}", case))?;
        let opening_layout = OpeningClaimsLayout::new(num_vars, 1)
            .map_err(|err| format!("{} opening layout: {err}", case))?;
        let schedule_key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(num_vars, 1));
        let schedule = scheme
            .schedules()
            .resolve_key(&schedule_key)
            .map_err(|err| format!("{} schedule: {err}", case))?;
        let root_d = schedule.schedule().root.params.d_a();
        if root_d != $d {
            return Err(format!(
                "{} root commitment uses D={root_d}, but its Jolt input monomorphization uses D={}",
                case, $d
            ));
        }

        tracing::info!(case = %case, num_vars, d = $d, "generating scalar OneHot recursion artifact");
        let t0 = Instant::now();
        let mut prover_setup = scheme
            .setup_prover(num_vars, 1)
            .map_err(|err| format!("{} prover setup: {err}", case))?;
        let prepared = CpuBackend::DEFAULT
            .prepare_setup(&prover_setup)
            .map_err(|err| format!("{} backend setup preparation: {err}", case))?;
        if $recursive {
            materialize_schedule_setup_prefix_slots(
                &mut prover_setup,
                &CpuBackend::DEFAULT,
                &prepared,
                schedule.schedule(),
            )
            .map_err(|err| format!("{} setup prefix materialization: {err}", case))?;
        }
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            prover_setup.expanded.as_ref(),
        )
        .map_err(|err| format!("{} prover stack: {err}", case))?;
        tracing::info!(case = %case, elapsed_s = t0.elapsed().as_secs_f64(), "prover setup complete");

        let poly = make_onehot_poly::<ScalarField>(num_vars, 0x0bee_fcaf_2800_0000)?;
        let opening_point =
            random_claim_point::<ScalarField, ScalarExt>(num_vars, 0xfeed_face);
        let openings = vec![onehot_opening::<ScalarField, ScalarExt>(
            &poly,
            &opening_point,
        )?];
        let t0 = Instant::now();
        let CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit(
                &prover_setup,
                std::slice::from_ref(&poly),
                &stack,
                GroupContext::scheduler_without_precommitted_groups(),
            )
        .map_err(|err| format!("{} commit: {err}", case))?;
        tracing::info!(case = %case, elapsed_s = t0.elapsed().as_secs_f64(), "commit complete");

        let prover_group = PolynomialGroupClaims::new(
            opening_point.clone(),
            openings.clone(),
            commitment.clone(),
        )
        .map_err(|err| format!("{} prover claims: {err}", case))?;
        let poly_ref = &poly;
        let poly_group = [poly_ref];
        let prove_input = SelectedProverOpeningData::from_committed_claims::<ScalarCfg>(
            OpeningClaims::from_groups(vec![prover_group])
                .map_err(|err| format!("{} opening claims: {err}", case))?,
            vec![hint],
            vec![poly_group.as_slice()],
            scheme.schedules(),
        )
        .map_err(|err| format!("{} prover opening data: {err}", case))?;
        let schedule_selection = prove_input.selection();
        let mut prover_transcript = AkitaTranscript::<ScalarField>::new(TRANSCRIPT_DOMAIN);
        let t0 = Instant::now();
        let proof = scheme
            .batched_prove(
                &prover_setup,
                prove_input,
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
        .map_err(|err| format!("{} prove: {err}", case))?;
        tracing::info!(case = %case, elapsed_s = t0.elapsed().as_secs_f64(), "prove complete");

        let verifier_setup = scheme
            .setup_verifier_for_schedule(&prover_setup, schedule.schedule(), &opening_layout)
        .map_err(|err| format!("{} verifier setup: {err}", case))?;
        let verifier_group = PolynomialGroupClaims::new(
            opening_point.as_slice(),
            openings.clone(),
            &commitment,
        )
        .map_err(|err| format!("{} verifier claims: {err}", case))?;
        let statement = GroupBatchStatement::new(
            schedule_selection,
            OpeningClaims::from_groups(vec![verifier_group])
                .map_err(|err| format!("{} verifier opening claims: {err}", case))?,
        )
        .map_err(|err| format!("{} verifier statement: {err}", case))?;
        let mut verifier_transcript =
            AkitaTranscript::<ScalarField>::unbound_verifier(TRANSCRIPT_DOMAIN);
        batched_verify::<ScalarCfg, _>(
            &proof,
            &verifier_setup,
            scheme.schedules(),
            &mut verifier_transcript,
            statement,
            BasisMode::Lagrange,
        )
        .map_err(|err| format!("{} host-side sanity verify: {err}", case))?;

        let grinding_plan = derive_transcript_grinding_plan::<ScalarCfg>(
            schedule.schedule(),
            &opening_layout,
        )
        .map_err(|err| format!("{} derive grinding plan: {err}", case))?;
        let proof_shape = proof.shape();
        proof_shape
            .validate_grinding_plan(&grinding_plan)
            .map_err(|err| format!("{} validate proof grinding shape: {err}", case))?;
        let inputs: AkitaJoltInputs<ScalarField, $d, ScalarExt> = AkitaJoltInputs {
            case,
            transcript_domain: TRANSCRIPT_DOMAIN.to_vec(),
            num_vars: num_vars as u64,
            opening_point,
            openings,
            precommitted_groups: Vec::new(),
            schedule_selection,
            commitment,
            verifier_setup,
            proof_shape,
            proof,
        };
        let inner_blob = inputs
            .write_to_bytes()
            .map_err(|err| format!("{} encode blob: {err}", case))?;
        let decoded = AkitaJoltInputs::<ScalarField, $d, ScalarExt>::read_from_bytes::<ScalarCfg>(
            &inner_blob,
            scheme.schedules(),
        )
        .map_err(|err| format!("{} strict blob round-trip: {err}", case))?;
        let mut transcript =
            AkitaTranscript::<ScalarField>::unbound_verifier(&decoded.transcript_domain);
        batched_verify::<ScalarCfg, _>(
            &decoded.proof,
            &decoded.verifier_setup,
            scheme.schedules(),
            &mut transcript,
            decoded
                .verifier_statement()
                .map_err(|err| format!("{} decoded statement: {err}", case))?,
            BasisMode::Lagrange,
        )
        .map_err(|err| format!("{} decoded blob verify: {err}", case))?;
        let blob = akita_recursion_glue::frame_with_schedule_catalog::<ScalarCfg>(
            &inner_blob,
            scheme.schedules(),
        )
        .map_err(|err| format!("{} frame schedule catalog: {err}", case))?;
        publish_blob($output_path, &blob)?;
        eprintln!(
            "wrote {} bytes ({:.2} MiB) for {} to {}",
            blob.len(),
            blob.len() as f64 / (1024.0 * 1024.0),
            case,
            $output_path.display()
        );
        Ok(())
    }};
}

fn generate_scalar_artifact(
    case: AkitaJoltCase,
    output_path: &std::path::Path,
) -> Result<(), String> {
    match case {
        AkitaJoltCase::OneHotFp32 => generate_scalar_case!(
            case,
            fp32::Field,
            fp32::OneHot,
            2048,
            30,
            false,
            output_path
        ),
        AkitaJoltCase::OneHotFp64 => {
            generate_scalar_case!(case, fp64::Field, fp64::OneHot, 512, 30, false, output_path)
        }
        AkitaJoltCase::OneHotFp128Direct => generate_scalar_case!(
            case,
            fp128::Field,
            fp128::OneHot,
            512,
            36,
            false,
            output_path
        ),
        AkitaJoltCase::OneHotFp128Recursive => generate_scalar_case!(
            case,
            fp128::Field,
            RecursiveCommitmentConfig<fp128::OneHot>,
            512,
            36,
            true,
            output_path
        ),
        AkitaJoltCase::OneHotFp128MultiGroupRecursive => Err(
            "the grouped recursive case is generated by the legacy multi-group adapter".to_string(),
        ),
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();

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

    let output_path = PathBuf::from(env_string(
        "AKITA_RECURSION_BLOB",
        "target/akita_recursion_inputs.bin",
    )?);
    let case_name = match args.case {
        Some(case) => case,
        None => env_string(
            "AKITA_RECURSION_CASE",
            AkitaJoltCase::OneHotFp128MultiGroupRecursive.as_str(),
        )?,
    };
    let case = case_name.parse::<AkitaJoltCase>()?;
    if case != AkitaJoltCase::OneHotFp128MultiGroupRecursive {
        return generate_scalar_artifact(case, &output_path);
    }

    let nv: usize = env_usize("AKITA_NUM_VARS", 32)?;
    if nv != 32 {
        return Err(format!(
            "recursive OneHot benchmark is pinned to nv=32, got nv={nv}"
        ));
    }
    let onehot_k = onehot_k_for_num_vars(nv);
    let scheme = load_workspace_scheme::<Cfg>()
        .map_err(|err| format!("failed to load trusted recursive schedule catalog: {err}"))?;
    let base_scheme = load_workspace_scheme::<BaseCfg>()
        .map_err(|err| format!("failed to load trusted base schedule catalog: {err}"))?;

    let prime = fp128_prime_label();
    tracing::info!(
        nv,
        d = SOURCE_VIEW_D,
        onehot_k,
        prime = %prime,
        "generating Akita verifier-input artifact (recursive multi-group OneHot)"
    );

    let pre_group = PolynomialGroupLayout::new(PRE_NUM_VARS, 1);
    let pre_descriptor = base_scheme
        .schedules()
        .resolve_key(&AkitaScheduleLookupKey::single(pre_group))
        .map(|row| row.profiles().final_group)
        .map_err(|err| format!("precommit profile: {err}"))?;
    let final_group = PolynomialGroupLayout::new(nv, FINAL_POLYS);
    let key = AkitaScheduleLookupKey {
        final_group,
        precommitteds: vec![pre_descriptor; PRE_GROUPS],
    };
    let opening_layout = key
        .opening_layout()
        .map_err(|err| format!("recursive opening layout: {err}"))?;
    let schedule = scheme
        .schedules()
        .resolve_key(&key)
        .map_err(|err| format!("recursive proof schedule: {err}"))?;
    let layout = schedule.schedule().root.params.clone();
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
                .map(|_| F::from_u128_reduced(rng.gen::<u128>()))
                .collect()
        })
        .collect();
    let final_point: Vec<F> = (0..nv)
        .map(|_| F::from_u128_reduced(rng.gen::<u128>()))
        .collect();

    let t0 = Instant::now();
    let mut prover_setup = scheme
        .setup_prover(nv, PRE_GROUPS + FINAL_POLYS)
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
    for (group_idx, pre_point) in pre_points.iter().enumerate() {
        let polys = vec![make_onehot_poly(
            PRE_NUM_VARS,
            0x0bee_fcaf_2100_0000 + group_idx as u64,
        )?];
        let openings = vec![onehot_opening(&polys[0], pre_point)?];
        let CommitOutput {
            committed_group,
            hint,
        } = base_scheme
            .commit(
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
    } = scheme
        .commit(
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
    for ((opening_point, openings), commitment) in
        pre_points.iter().zip(&pre_openings).zip(&pre_commitments)
    {
        prover_groups.push(
            PolynomialGroupClaims::new(opening_point.clone(), openings.clone(), commitment.clone())
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
        scheme.schedules(),
    )
    .map_err(|err| format!("invalid prover opening data: {err}"))?;
    let schedule_selection = prove_input.selection();
    let proof = scheme
        .batched_prove(
            &prover_setup,
            prove_input,
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .map_err(|err| format!("batched_prove failed: {err}"))?;
    tracing::info!(elapsed_s = t0.elapsed().as_secs_f64(), "prove complete");

    let verifier_setup = scheme
        .setup_verifier_for_schedule(&prover_setup, schedule.schedule(), &opening_layout)
        .map_err(|err| format!("setup_verifier_for_schedule failed: {err}"))?;

    // Sanity check: the proof should verify with the same domain label.
    let t0 = Instant::now();
    let mut verifier_transcript = AkitaTranscript::<F>::unbound_verifier(TRANSCRIPT_DOMAIN);
    verify_proof(
        &proof,
        &verifier_setup,
        scheme.schedules(),
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

    let grinding_plan =
        derive_transcript_grinding_plan::<Cfg>(schedule.schedule(), &opening_layout)
            .map_err(|err| format!("derive grinding plan failed: {err}"))?;
    let proof_shape = proof.shape();
    proof_shape
        .validate_grinding_plan(&grinding_plan)
        .map_err(|err| format!("validate proof grinding shape failed: {err}"))?;
    let inputs: AkitaJoltInputs<F, SOURCE_VIEW_D> = AkitaJoltInputs {
        case: AkitaJoltCase::OneHotFp128MultiGroupRecursive,
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

    let inner_blob = inputs
        .write_to_bytes()
        .map_err(|err| format!("encode jolt inputs blob failed: {err}"))?;
    // Round-trip before publishing so a buggy encoding fails on the host
    // instead of leaving a trusted benchmark artifact on disk.
    let decoded = AkitaJoltInputs::<F, SOURCE_VIEW_D>::read_from_bytes::<Cfg>(
        &inner_blob,
        scheme.schedules(),
    )
    .map_err(|err| format!("decode jolt inputs blob (round-trip) failed: {err}"))?;
    let mut roundtrip_transcript =
        AkitaTranscript::<F>::unbound_verifier(&decoded.transcript_domain);
    verify_proof(
        &decoded.proof,
        &decoded.verifier_setup,
        scheme.schedules(),
        &mut roundtrip_transcript,
        decoded
            .verifier_statement()
            .map_err(|err| format!("decoded verifier statement failed: {err}"))?,
    )
    .map_err(|err| format!("decoded blob verify failed: {err}"))?;
    tracing::info!("decoded-blob verify OK");

    let blob =
        akita_recursion_glue::frame_with_schedule_catalog::<Cfg>(&inner_blob, scheme.schedules())
            .map_err(|err| format!("frame schedule catalog failed: {err}"))?;

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
