#![allow(missing_docs)]

use akita_prover::{ComputeBackendSetup, CpuBackend};

use akita_config::proof_optimized::fp128;
use akita_config::proof_optimized::{fp32, fp64};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::DensePoly;
use akita_prover::OneHotPoly;
use akita_prover::SelectedProverOpeningData;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_transcript::AkitaTranscript;
use akita_types::{lagrange_weights, CommittedGroupParams, FpExtEncoding};
use akita_types::{
    AkitaBatchedProof, AkitaCommitmentHint, AkitaVerifierSetup, BasisMode, CommittedGroup,
    CommittedGroupBatchProfile, GroupBatchStatement, OpeningClaims, OpeningMethod,
    OpeningScheduleSelection, PolynomialGroupClaims,
};
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};
use jolt_field::{CanonicalBytes, CanonicalEncoding, ExtField, Field, PseudoMersenne, Ring};
use jolt_field::{Fold, Unreduced};
use jolt_field::{One, Zero};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "disk-persistence")]
use std::path::PathBuf;
use std::sync::{Mutex, Once};

mod common;
use common::{load_workspace_scheme, opening_from_poly_for_layout};

type F = fp128::Field;
const DENSE_TEST_NV: usize = 14;
const ONEHOT_TEST_NV: usize = 15;
const SAME_POINT_ONEHOT_BATCH_SIZE: usize = 4;

fn singleton_layout(schedules: &TrustedScheduleCatalog, num_vars: usize) -> CommittedGroupParams {
    schedules
        .resolve_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(num_vars),
        ))
        .map(|row| row.schedule().root.params.clone())
        .expect("singleton commitment layout")
}
const SMALL_FIELD_TEST_NV: usize = 8;
const STACK_SIZE: usize = 256 * 1024 * 1024;

fn onehot_source_chunk_size<Cfg: CommitmentConfig>() -> usize {
    akita_config::unit_onehot_source_chunk_size::<Cfg>()
        .expect("one-hot fixture requires a unit-one-hot commitment config")
}

static INIT_RAYON: Once = Once::new();
static E2E_TEST_LOCK: Mutex<()> = Mutex::new(());

fn init_rayon_pool() {
    INIT_RAYON.call_once(|| {
        #[cfg(feature = "parallel")]
        rayon::ThreadPoolBuilder::new()
            .stack_size(STACK_SIZE)
            .build_global()
            .ok();
    });
}

fn random_point<FField: CanonicalEncoding>(nv: usize) -> Vec<FField> {
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| FField::from_u128_reduced(rng.gen::<u128>()))
        .collect()
}

fn random_claim_point<FField, E>(nv: usize) -> Vec<E>
where
    FField: CanonicalEncoding + Field,
    E: ExtField<FField>,
{
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| {
            let limbs = (0..E::DEGREE)
                .map(|_| FField::from_u128_reduced(rng.gen::<u128>()))
                .collect::<Vec<_>>();
            E::from_base_slice(&limbs)
        })
        .collect()
}

fn dense_lagrange_opening_from_evals<FField, E>(evals: &[FField], point: &[E]) -> E
where
    FField: Field,
    E: ExtField<FField>,
{
    let weights = lagrange_weights(point).expect("valid opening point");
    evals
        .iter()
        .zip(weights.iter())
        .fold(E::zero(), |acc, (&coeff, &weight)| {
            acc + weight * E::lift_base(coeff)
        })
}

fn run_on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(f)
        .expect("failed to spawn thread")
        .join()
        .expect("test thread panicked");
}

fn prove_input<'a, Cfg: CommitmentConfig, P: akita_prover::RootPolyMeta<Cfg::Field>>(
    selection: OpeningScheduleSelection,
    point: &'a [Cfg::ExtField],
    polynomials: &'a [&'a P],
    commitment: &'a CommittedGroup<Cfg::Field>,
    hint: AkitaCommitmentHint<Cfg::Field>,
    schedules: &TrustedScheduleCatalog,
) -> SelectedProverOpeningData<
    'a,
    Cfg::ExtField,
    akita_prover::PreparedProverGroup<'a, P>,
    Cfg::Field,
> {
    let group = PolynomialGroupClaims::new(
        point.to_vec(),
        vec![Cfg::ExtField::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("valid prover claims group");
    let opening_claims = OpeningClaims::from_groups(vec![group]).expect("valid prover claims");
    let selected = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        opening_claims,
        vec![hint],
        vec![polynomials],
        schedules,
    )
    .expect("valid prover opening data");
    assert_eq!(selected.selection(), selection);
    selected
}

fn verify_input<'a, Cfg: CommitmentConfig>(
    selection: OpeningScheduleSelection,
    point: &[Cfg::ExtField],
    openings: &[Cfg::ExtField],
    commitment: &'a CommittedGroup<Cfg::Field>,
) -> GroupBatchStatement<'a, Cfg::ExtField, Cfg::Field> {
    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.to_vec(),
        openings.to_vec(),
        commitment,
    )
    .expect("valid verifier claims group")])
    .expect("valid verifier input");
    GroupBatchStatement::new(selection, claims).expect("valid verifier statement")
}

fn selection_for<Cfg: CommitmentConfig>(
    commitment: &CommittedGroup<Cfg::Field>,
    schedules: &TrustedScheduleCatalog,
) -> OpeningScheduleSelection {
    schedules
        .resolve_profiles(&CommittedGroupBatchProfile {
            final_group: *commitment.profile(),
            precommitteds: Vec::new(),
        })
        .expect("select schedule")
        .selection()
}

type DenseFixture<FField, E, const D: usize> = (
    AkitaVerifierSetup<FField>,
    CommittedGroup<FField>,
    AkitaBatchedProof<FField, E>,
    Vec<E>,
    E,
    CommittedGroupParams,
    OpeningScheduleSelection,
);

fn make_dense_fixture<FField, const D: usize, Cfg: CommitmentConfig<Field = FField>>(
    scheme: &AkitaCommitmentScheme<Cfg>,
    nv: usize,
    transcript_label: &'static [u8],
) -> DenseFixture<FField, Cfg::ExtField, D>
where
    FField: CanonicalEncoding
        + CanonicalBytes
        + CanonicalEncoding
        + Unreduced
        + Field
        + Ring
        + 'static
        + Field
        + PseudoMersenne
        + Valid
        + AkitaDeserialize<Context = ()>
        + AkitaSerialize,
    Cfg::ExtField: ExtField<FField> + Unreduced + Fold,
    <FField as Unreduced>::Wide: From<FField>,
    Cfg::ExtField: FpExtEncoding<FField> + AkitaSerialize,
{
    let layout = singleton_layout(scheme.schedules(), nv);

    let mut rng = StdRng::seed_from_u64(0x0ddc_0ffe_e123_4567);
    let evals: Vec<FField> = (0..1usize << nv)
        .map(|_| FField::from_u128_reduced(rng.gen::<u128>()))
        .collect();

    let poly = DensePoly::<FField>::from_field_evals(nv, &evals).unwrap();
    let pt = random_claim_point::<FField, Cfg::ExtField>(nv);
    let expected_opening = dense_lagrange_opening_from_evals::<FField, Cfg::ExtField>(&evals, &pt);

    #[cfg(feature = "disk-persistence")]
    purge_setup_cache(nv);

    let setup = scheme.setup_prover(nv, 1).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint,
    } = scheme
        .commit::<_, _>(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();

    let poly_refs: [&DensePoly<FField>; 1] = [&poly];
    let commitments = [commitment];
    let selection = selection_for::<Cfg>(&commitments[0], scheme.schedules());
    let hints = vec![hint];

    let mut prover_transcript = AkitaTranscript::<FField>::new(transcript_label);
    let proof = scheme
        .batched_prove::<_, _, _>(
            &setup,
            prove_input::<Cfg, _>(
                selection,
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
                scheme.schedules(),
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

    let [commitment] = commitments;
    (
        verifier_setup,
        commitment,
        proof,
        pt,
        expected_opening,
        layout,
        selection,
    )
}

/// Remove any stale disk-persistence cache for `max_num_vars` so that a setup
/// written by a different `CommitmentConfig` doesn't get loaded by mistake.
#[cfg(feature = "disk-persistence")]
fn purge_setup_cache(max_num_vars: usize) {
    let cache_dir = std::env::var("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|_| {
            std::env::var("HOME").map(|home| {
                let mut p = PathBuf::from(&home);
                if p.join("Library/Caches").exists() {
                    p.push("Library/Caches");
                } else {
                    p.push(".cache");
                }
                p
            })
        });
    if let Ok(mut path) = cache_dir {
        path.push("akita");
        if let Ok(entries) = std::fs::read_dir(&path) {
            let needle = format!("_nv{max_num_vars}.setup");
            let batch_needle = format!("_nv{max_num_vars}_batch");
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with("akita_")
                            && (name.ends_with(&needle) || name.contains(&batch_needle))
                    })
                {
                    let _ = std::fs::remove_file(entry_path);
                }
            }
        }
    }
}

fn bump_flat_ring_vec<FField: Field>(flat: &mut akita_types::RingVec<FField>) {
    let mut coeffs = flat.coeffs().to_vec();
    let first = coeffs
        .first_mut()
        .expect("tamper target must contain at least one coefficient");
    *first += FField::one();
    *flat = akita_types::RingVec::from_coeffs(coeffs);
}

fn mutate_terminal_e_hat_digit<FField: Field>(witness: &mut akita_types::TerminalResponse<FField>) {
    bump_flat_ring_vec(&mut witness.e_fields);
}

fn terminal_witness_mut<FField: Field, E: Field>(
    proof: &mut AkitaBatchedProof<FField, E>,
) -> &mut akita_types::TerminalResponse<FField> {
    proof.terminal.terminal_response_mut()
}

fn assert_invalid_proof<T: core::fmt::Debug>(
    case: &str,
    result: Result<T, akita_error::AkitaError>,
) {
    match result {
        Err(akita_error::AkitaError::InvalidProof) => {}
        Err(akita_error::AkitaError::InvalidInput(msg)) if msg.contains("InvalidProof") => {}
        other => panic!("{case} must reject with InvalidProof, got {other:?}"),
    }
}

#[test]
fn trace_internalization_rejects_tampered_root_fold_handle() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::Dense;
        const D: usize = 256;
        let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");

        let (verifier_setup, commitment, proof, opening_point, opening, _layout, selection) =
            make_dense_fixture::<F, D, Cfg>(&scheme, DENSE_TEST_NV, b"akita_e2e/root-trace-tamper");
        let mut malformed = proof.clone();
        bump_flat_ring_vec(&mut malformed.root.opening_payload);

        let commitments = [commitment];
        let openings = [opening];
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/root-trace-tamper");
        let result = scheme.batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input::<Cfg>(
                selection,
                &opening_point[..],
                &openings[..],
                &commitments[0],
            ),
            BasisMode::Lagrange,
        );
        assert_invalid_proof("tampered root fold handle", result);
    });
}

#[test]
fn trace_internalization_rejects_tampered_recursive_fold_handle() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::OneHot;
        const NV: usize = 20;
        let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");

        let layout = scheme
            .schedules()
            .resolve_key(&AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
                NV, 2,
            )))
            .map(|row| row.schedule().root.params.clone())
            .expect("layout");
        let root_d = layout.d_a();
        let onehot_k = onehot_source_chunk_size::<Cfg>();
        let total_field = (layout.blocks().live_blocks * layout.blocks().positions_per_block)
            .checked_mul(root_d)
            .expect("total field size overflow");
        let total_chunks = total_field / onehot_k;
        assert_eq!(total_chunks * onehot_k, total_field);

        let polys: Vec<OneHotPoly<F>> = (0..2)
            .map(|poly_idx| {
                let mut rng = StdRng::seed_from_u64(0x3141_5926 + poly_idx as u64);
                let indices: Vec<Option<usize>> = (0..total_chunks)
                    .map(|_| Some(rng.gen_range(0..onehot_k)))
                    .collect();
                OneHotPoly::<F>::new(onehot_k, indices).unwrap()
            })
            .collect();
        let poly_refs: Vec<&OneHotPoly<F>> = polys.iter().collect();
        let point = random_point(NV);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| {
                opening_from_poly_for_layout(
                    poly,
                    &point,
                    &layout.final_group_scalar().expect("scalar final group"),
                    BasisMode::Lagrange,
                )
            })
            .collect();

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(NV);

        let setup = scheme.setup_prover(NV, 2).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit::<_, _>(
                &setup,
                &polys,
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        let commitments = [commitment];
        let selection = selection_for::<Cfg>(&commitments[0], scheme.schedules());

        let mut prover_transcript = AkitaTranscript::<F>::new(b"akita_e2e/recursive-trace-tamper");
        let proof = scheme
            .batched_prove::<_, _, _>(
                &setup,
                prove_input::<Cfg, _>(
                    selection,
                    &point[..],
                    &poly_refs[..],
                    &commitments[0],
                    hint,
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .unwrap();

        let mut malformed = proof.clone();
        let recursive = malformed
            .recursive_folds
            .first_mut()
            .expect("fixture should include an intermediate recursive fold");
        bump_flat_ring_vec(&mut recursive.opening_payload);

        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"akita_e2e/recursive-trace-tamper");
        let result = scheme.batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input::<Cfg>(selection, &point[..], &openings[..], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert_invalid_proof("tampered recursive fold handle", result);
    });
}

#[test]
fn trace_internalization_rejects_tampered_terminal_e_hat_digit() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::Dense;
        const D: usize = 256;
        let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");

        let (verifier_setup, commitment, proof, opening_point, opening, _layout, selection) =
            make_dense_fixture::<F, D, Cfg>(
                &scheme,
                DENSE_TEST_NV,
                b"akita_e2e/terminal-trace-tamper",
            );
        let mut malformed = proof.clone();
        mutate_terminal_e_hat_digit(terminal_witness_mut(&mut malformed));

        let commitments = [commitment];
        let openings = [opening];
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/terminal-trace-tamper");
        let result = scheme.batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input::<Cfg>(
                selection,
                &opening_point[..],
                &openings[..],
                &commitments[0],
            ),
            BasisMode::Lagrange,
        );
        assert_invalid_proof("tampered terminal e_hat digit", result);
    });
}

#[test]
fn small_field_dense_uncataloged_roots_fail_fast() {
    let fp32_catalog = akita_config::test_support::workspace_schedule_catalog::<fp32::Dense>()
        .expect("fp32 dense catalog");
    let fp64_catalog = akita_config::test_support::workspace_schedule_catalog::<fp64::Dense>()
        .expect("fp64 dense catalog");
    for result in [
        fp32_catalog.resolve_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(SMALL_FIELD_TEST_NV),
        )),
        fp64_catalog.resolve_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(SMALL_FIELD_TEST_NV + 1),
        )),
    ] {
        assert!(matches!(
            result,
            Err(akita_error::AkitaError::UnsupportedSchedule(_))
        ));
    }
}

#[test]
fn adaptive_dense_tiny_roots_and_setup_capacities_are_rejected() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::Dense;
        let nv = 4;
        let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");
        let err = scheme
            .schedules()
            .resolve_key(&AkitaScheduleLookupKey::single(
                PolynomialGroupLayout::singleton(nv),
            ))
            .expect_err("tiny roots must not produce a degenerate proof schedule");
        assert!(matches!(
            err,
            akita_error::AkitaError::UnsupportedSchedule(_)
        ));
        let setup_err = scheme
            .setup_prover(nv, 1)
            .expect_err("tiny capacity must not produce a prover setup");
        assert!(
            matches!(setup_err, akita_error::AkitaError::InvalidSetup(_)),
            "setup capacity rejection should use the setup boundary: {setup_err:?}"
        );
    });
}

#[test]
fn batched_onehot_same_point_rejects_tampered_root_stage1_range_image_evaluation() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::OneHot;
        let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");

        let nv = ONEHOT_TEST_NV;
        let layout = scheme
            .schedules()
            .resolve_key(&AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
                nv,
                SAME_POINT_ONEHOT_BATCH_SIZE,
            )))
            .map(|row| row.schedule().root.params.clone())
            .expect("layout");
        let root_d = layout.d_a();
        let onehot_k = onehot_source_chunk_size::<Cfg>();
        let total_field = (layout.blocks().live_blocks * layout.blocks().positions_per_block)
            .checked_mul(root_d)
            .expect("total field size overflow");
        let total_chunks = total_field / onehot_k;
        assert_eq!(total_chunks * onehot_k, total_field);

        let polys: Vec<OneHotPoly<F>> = (0..SAME_POINT_ONEHOT_BATCH_SIZE)
            .map(|poly_idx| {
                let mut rng = StdRng::seed_from_u64(0x8765_4321 + poly_idx as u64);
                let indices: Vec<Option<usize>> = (0..total_chunks)
                    .map(|_| Some(rng.gen_range(0..onehot_k)))
                    .collect();
                OneHotPoly::<F>::new(onehot_k, indices).unwrap()
            })
            .collect();
        let poly_group: Vec<&OneHotPoly<F>> = polys.iter().collect();
        let pt = random_point(nv);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| {
                opening_from_poly_for_layout(
                    poly,
                    &pt,
                    &layout.final_group_scalar().expect("scalar final group"),
                    BasisMode::Lagrange,
                )
            })
            .collect();

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(nv);

        let setup = scheme
            .setup_prover(nv, SAME_POINT_ONEHOT_BATCH_SIZE)
            .unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit::<_, _>(
                &setup,
                &polys,
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        let commitments = [commitment];
        let selection = selection_for::<Cfg>(&commitments[0], scheme.schedules());
        let hints = vec![hint];

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"akita_e2e/batched-onehot-s-claim-tamper");
        let proof = scheme
            .batched_prove::<_, _, _>(
                &setup,
                prove_input::<Cfg, _>(
                    selection,
                    &pt[..],
                    &poly_group[..],
                    &commitments[0],
                    hints.into_iter().next().unwrap(),
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .unwrap();

        let mut malformed = proof.clone();
        malformed.root.stage1.range_image_evaluation += F::from_u128_reduced(1);

        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"akita_e2e/batched-onehot-s-claim-tamper");
        let opening_groups = [&openings[..]];
        let result = scheme.batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input::<Cfg>(selection, &pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_err(),
            "tampered batched root stage1 range_image_evaluation must be rejected"
        );
    });
}

// ============================================================================
// Public-boundary rejection tests preserved from the pre-consolidation suite.
//
// These were previously in `src/scheme/tests/{fp32_ext4,batched}.rs` and
// `tests/akita_e2e.rs`. The correctness matrix replaced their *positive*
// round trips, but a passing round trip does not establish that the verifier
// rejects a malformed proof — that is what these cover.
// ============================================================================

const EXT4_NV: usize = 16;
const EXT4_BATCH: usize = 2;

fn ext4_onehot_poly(seed: usize) -> OneHotPoly<fp32::Field, u8> {
    let onehot_k = onehot_source_chunk_size::<fp32::OneHot>();
    assert!(
        onehot_k <= usize::from(u8::MAX) + 1,
        "test u8 one-hot fixture cannot represent chunk size {onehot_k}"
    );
    let num_chunks = (1usize << EXT4_NV) / onehot_k;
    let indices = (0..num_chunks)
        .map(|chunk| Some(((chunk * 29 + seed * 41 + 7) % onehot_k) as u8))
        .collect();
    OneHotPoly::new(onehot_k, indices).expect("fp32 one-hot polynomial")
}

fn ext4_point() -> Vec<fp32::ExtensionField> {
    (0..EXT4_NV)
        .map(|c| {
            <fp32::ExtensionField as ExtField<fp32::Field>>::from_base_slice(&[
                fp32::Field::from_u64((c * 5 + 1) as u64),
                fp32::Field::from_u64((c * 5 + 2) as u64),
                fp32::Field::from_u64((c * 5 + 3) as u64),
                fp32::Field::from_u64((c * 5 + 4) as u64),
            ])
        })
        .collect()
}

/// Coefficient packing removes EOR from every emitted early fp32 fold. The
/// terminal remains EvaluationTrace and must reject a changed or missing EOR
/// payload.
#[test]
fn fp32_ext4_rejects_wrong_opening_and_tampered_or_missing_terminal_eor() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp32::OneHot;
        type SF = fp32::Field;
        type SE = fp32::ExtensionField;
        const LABEL: &[u8] = b"soundness/fp32-ext4-eor";
        let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");

        let polys = [ext4_onehot_poly(0), ext4_onehot_poly(1)];
        let poly_refs: Vec<_> = polys.iter().collect();
        let point = ext4_point();
        let weights = lagrange_weights::<SE>(&point).expect("extension Lagrange weights");
        let openings: Vec<SE> = polys
            .iter()
            .map(|poly| {
                let k = poly.onehot_k();
                poly.indices()
                    .iter()
                    .enumerate()
                    .filter_map(|(chunk, hot)| hot.map(|i| weights[chunk * k + usize::from(i)]))
                    .fold(SE::zero(), |a, b| a + b)
            })
            .collect();

        let setup = scheme
            .setup_prover(EXT4_NV, EXT4_BATCH)
            .expect("fp32 prover setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit(
                &setup,
                &polys,
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");
        let selection = selection_for::<Cfg>(&commitment, scheme.schedules());

        #[cfg(feature = "logging-transcript")]
        let mut prover_transcript =
            akita_transcript::LoggingTranscript::wrap(AkitaTranscript::<SF>::new(LABEL));
        #[cfg(not(feature = "logging-transcript"))]
        let mut prover_transcript = AkitaTranscript::<SF>::new(LABEL);
        let proof = scheme
            .batched_prove::<_, _, _>(
                &setup,
                prove_input::<Cfg, _>(
                    selection,
                    &point[..],
                    &poly_refs[..],
                    &commitment,
                    hint,
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("fp32 extension proof");
        let resolved = scheme
            .schedules()
            .resolve_selection(selection)
            .expect("selected fp32 row");
        assert!(
            matches!(
                resolved.schedule().root.params.opening_method(),
                OpeningMethod::SubringCoefficientPacking { .. }
            ),
            "the shipped fp32 row must use coefficient packing at the root"
        );
        assert!(
            proof.root.extension_opening_reduction.is_none(),
            "coefficient packing must not emit a root EOR payload"
        );
        for (step, recursive_proof) in resolved
            .schedule()
            .recursive_folds
            .iter()
            .take(1)
            .zip(proof.recursive_folds.iter().take(1))
        {
            assert!(
                matches!(
                    step.params.opening_method(),
                    OpeningMethod::SubringCoefficientPacking { .. }
                ),
                "every emitted early fp32 fold must use coefficient packing"
            );
            assert!(
                recursive_proof.extension_opening_reduction.is_none(),
                "coefficient packing must not emit a recursive EOR payload"
            );
        }
        assert!(
            proof.terminal.extension_opening_reduction.is_some(),
            "the EvaluationTrace terminal must retain EOR"
        );

        // Baseline: the honest proof verifies.
        #[cfg(feature = "logging-transcript")]
        let mut vt = akita_transcript::LoggingTranscript::wrap(AkitaTranscript::<SF>::new(LABEL));
        #[cfg(not(feature = "logging-transcript"))]
        let mut vt = AkitaTranscript::<SF>::new(LABEL);
        let honest_result = scheme.batched_verify(
            &proof,
            &verifier_setup,
            &mut vt,
            verify_input::<Cfg>(selection, &point[..], &openings[..], &commitment),
            BasisMode::Lagrange,
        );
        #[cfg(feature = "logging-transcript")]
        {
            let opening_layout = akita_types::OpeningClaimsLayout::new(EXT4_NV, EXT4_BATCH)
                .expect("fp32 extension opening layout");
            let grinding_plan = akita_config::derive_transcript_grinding_plan::<Cfg>(
                resolved.schedule(),
                &opening_layout,
            )
            .expect("fp32 extension grinding plan");
            let prover_draw_counts = common::assert_production_grinding_audit(
                prover_transcript.events(),
                &grinding_plan,
            );
            common::assert_production_grinding_audit(vt.events(), &grinding_plan);
            let expected_pow = grinding_plan
                .runs()
                .iter()
                .filter(|run| run.kind() == akita_types::GrindingQueryKind::ProofOfWork);
            for ((site, actual_draws), run) in prover_draw_counts.iter().zip(expected_pow) {
                assert_eq!(*site, run.site());
                match site {
                    akita_types::GrindingSite::ExtensionOpeningPoint { .. }
                    | akita_types::GrindingSite::Tau0Point { .. }
                    | akita_types::GrindingSite::Tau1Point { .. } => {
                        let expected_draws = usize::try_from(run.loss_factor()).unwrap()
                            * <SE as ExtField<SF>>::DEGREE;
                        assert_eq!(
                            *actual_draws, expected_draws,
                            "extension point draw count must match the public geometry"
                        );
                    }
                    akita_types::GrindingSite::EvaluationBatch { .. }
                    | akita_types::GrindingSite::ExtensionOpeningClaimBatch { .. } => {}
                    _ => assert_eq!(
                        *actual_draws,
                        <SE as ExtField<SF>>::DEGREE,
                        "one-element challenge must consume exactly one extension-field draw"
                    ),
                }
            }
            let prover_events = common::public_transcript_events(prover_transcript.events());
            let verifier_events = common::public_transcript_events(vt.events());
            if prover_events != verifier_events {
                let first_difference = prover_events
                    .iter()
                    .zip(&verifier_events)
                    .position(|(prover, verifier)| prover != verifier)
                    .unwrap_or_else(|| prover_events.len().min(verifier_events.len()));
                panic!(
                    "fp32 extension transcript diverged at {first_difference}: prover={:?}, verifier={:?}, lengths=({}, {})",
                    prover_events.get(first_difference),
                    verifier_events.get(first_difference),
                    prover_events.len(),
                    verifier_events.len(),
                );
            }
            assert!(
                common::assert_claim_batching_follows_opening_payload(&prover_events) > 0,
                "multi-opening evaluation batching must follow the opening payload"
            );
            let event_index = |label: &[u8]| {
                prover_events
                    .iter()
                    .position(|event| {
                        common::event_label(event).is_some_and(|candidate| {
                            common::is_label_or_extension_limb(candidate, label)
                        })
                    })
                    .unwrap_or_else(|| panic!("missing transcript event for {label:?}"))
            };
            let terminal_claim = event_index(akita_transcript::labels::ABSORB_EOR_FINAL_CLAIM);
            let combined_claim = prover_events[..terminal_claim]
                .iter()
                .rposition(|event| {
                    common::event_label(event).is_some_and(|candidate| {
                        candidate == akita_transcript::labels::ABSORB_SUMCHECK_CLAIM
                    })
                })
                .expect("EOR combined claim must precede its terminal claim");
            let eta = prover_events[..combined_claim]
                .iter()
                .rposition(|event| {
                    common::event_label(event).is_some_and(|candidate| {
                        common::is_label_or_extension_limb(
                            candidate,
                            akita_transcript::labels::CHALLENGE_SUMCHECK_BATCH,
                        )
                    })
                })
                .expect("EOR eta must precede claim batching");
            assert!(
                eta < combined_claim && combined_claim < terminal_claim,
                "singleton EOR transcript must order eta before its sumcheck and terminal claims"
            );
        }
        honest_result.expect("honest fp32 extension proof must verify");

        // (1) A wrong second opening must be rejected.
        let mut wrong = openings.clone();
        wrong[1] += SE::one();
        let mut vt = AkitaTranscript::<SF>::new(LABEL);
        scheme
            .batched_verify(
                &proof,
                &verifier_setup,
                &mut vt,
                verify_input::<Cfg>(selection, &point[..], &wrong[..], &commitment),
                BasisMode::Lagrange,
            )
            .expect_err("wrong batched extension opening must reject");

        // (2) A tampered terminal EOR partial evaluation must be rejected.
        let mut tampered = proof.clone();
        *tampered
            .terminal
            .extension_opening_reduction
            .as_mut()
            .expect("terminal EOR payload")
            .partials
            .first_mut()
            .expect("terminal EOR must carry a partial evaluation") += SE::one();
        let mut vt = AkitaTranscript::<SF>::new(LABEL);
        scheme
            .batched_verify(
                &tampered,
                &verifier_setup,
                &mut vt,
                verify_input::<Cfg>(selection, &point[..], &openings[..], &commitment),
                BasisMode::Lagrange,
            )
            .expect_err("tampered terminal extension-opening reduction partial must reject");

        // (3) The individual EOR terminal handles remain bound even though the
        // round messages are compressed into one sumcheck.
        let mut tampered = proof.clone();
        *tampered
            .terminal
            .extension_opening_reduction
            .as_mut()
            .expect("terminal EOR payload")
            .final_claims
            .first_mut()
            .expect("terminal EOR must carry a terminal handle") += SE::one();
        let mut vt = AkitaTranscript::<SF>::new(LABEL);
        scheme
            .batched_verify(
                &tampered,
                &verifier_setup,
                &mut vt,
                verify_input::<Cfg>(selection, &point[..], &openings[..], &commitment),
                BasisMode::Lagrange,
            )
            .expect_err("tampered per-claim EOR terminal handle must reject");

        // (4) Omitting the required EOR entirely must be rejected.
        let mut stripped = proof.clone();
        stripped.terminal.extension_opening_reduction = None;
        let mut vt = AkitaTranscript::<SF>::new(LABEL);
        scheme
            .batched_verify(
                &stripped,
                &verifier_setup,
                &mut vt,
                verify_input::<Cfg>(selection, &point[..], &openings[..], &commitment),
                BasisMode::Lagrange,
            )
            .expect_err("omitting the required terminal extension-opening reduction must reject");
    });
}

/// A two-polynomial batched proof must reject both a wrong second opening and
/// an opening payload padded beyond the committed geometry.
#[test]
fn batched_dense_rejects_wrong_opening_and_oversized_payload() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::Dense;
        const NV: usize = 16;
        const LABEL: &[u8] = b"soundness/batched-dense-payload";
        let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");

        let len = 1usize << NV;
        let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 5) as u64)).collect();
        let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 7 + 3) as u64)).collect();
        let poly_a = DensePoly::<F>::from_field_evals(NV, &evals_a).expect("poly a");
        let poly_b = DensePoly::<F>::from_field_evals(NV, &evals_b).expect("poly b");

        let point = random_point::<F>(NV);
        let openings = [
            dense_lagrange_opening_from_evals(&evals_a, &point),
            dense_lagrange_opening_from_evals(&evals_b, &point),
        ];

        let setup = scheme.setup_prover(NV, 2).expect("setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit(
                &setup,
                &[poly_a.clone(), poly_b.clone()],
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");
        let selection = selection_for::<Cfg>(&commitment, scheme.schedules());
        let poly_group = [&poly_a, &poly_b];

        let mut prover_transcript = AkitaTranscript::<F>::new(LABEL);
        let proof = scheme
            .batched_prove::<_, _, _>(
                &setup,
                prove_input::<Cfg, _>(
                    selection,
                    &point[..],
                    &poly_group[..],
                    &commitment,
                    hint,
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("prove");

        let mut vt = AkitaTranscript::<F>::new(LABEL);
        scheme
            .batched_verify(
                &proof,
                &verifier_setup,
                &mut vt,
                verify_input::<Cfg>(selection, &point[..], &openings[..], &commitment),
                BasisMode::Lagrange,
            )
            .expect("batched verify must accept consistent openings");

        // (1) Wrong second opening.
        let mut wrong = openings;
        wrong[1] += F::one();
        let mut vt = AkitaTranscript::<F>::new(LABEL);
        assert_invalid_proof(
            "wrong second batched opening",
            scheme.batched_verify(
                &proof,
                &verifier_setup,
                &mut vt,
                verify_input::<Cfg>(selection, &point[..], &wrong[..], &commitment),
                BasisMode::Lagrange,
            ),
        );

        // (2) Opening payload padded past the committed geometry, with a
        // matching extra claim, must not be accepted.
        let mut oversized = proof.clone();
        let mut coeffs = oversized.root.opening_payload.coeffs().to_vec();
        coeffs.extend(vec![F::zero(); 256]);
        oversized.root.opening_payload = akita_types::RingVec::from_coeffs(coeffs);

        let mut oversized_openings = openings.to_vec();
        oversized_openings.push(F::zero());
        let mut vt = AkitaTranscript::<F>::new(LABEL);
        assert_invalid_proof(
            "oversized opening payload",
            scheme.batched_verify(
                &oversized,
                &verifier_setup,
                &mut vt,
                verify_input::<Cfg>(selection, &point[..], &oversized_openings[..], &commitment),
                BasisMode::Lagrange,
            ),
        );
    });
}

/// The batched one-hot proof at a schedule that crosses a partial final fold
/// row must expose a canonical terminal witness, and must be rejected when a
/// scheduled recursive fold is dropped from the suffix.
#[test]
fn batched_onehot_terminal_structure_and_truncated_recursive_suffix() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::OneHot;
        // NV=20 is large enough for the two-claim schedule to carry a
        // recursive suffix.
        const NV: usize = 20;
        const LABEL: &[u8] = b"soundness/batched-onehot-terminal";
        let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");

        let plan = scheme
            .schedules()
            .resolve_key(&AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
                NV, 2,
            )))
            .expect("runtime schedule")
            .schedule()
            .clone();
        let layout = plan.root.params.clone();
        let root_d = layout.d_a();
        let onehot_k = onehot_source_chunk_size::<Cfg>();
        let fold_params = std::iter::once(&plan.root.params)
            .chain(plan.recursive_folds.iter().map(|step| &step.params))
            .collect::<Vec<_>>();
        assert!(
            fold_params.iter().any(|params| {
                params.blocks().live_ring_elements_per_claim % params.blocks().positions_per_block
                    != 0
                    && params.blocks().live_blocks
                        == params
                            .blocks()
                            .live_ring_elements_per_claim
                            .div_ceil(params.blocks().positions_per_block)
            }),
            "fixture must cross a production fold with an exact partial final row"
        );

        let total_field = (layout.blocks().live_blocks * layout.blocks().positions_per_block)
            .checked_mul(root_d)
            .expect("total field size overflow");
        let total_chunks = total_field / onehot_k;
        assert_eq!(total_chunks * onehot_k, total_field);

        let polys: Vec<OneHotPoly<F>> = [0x1234_5678u64, 0x8765_4321u64]
            .into_iter()
            .map(|seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                let indices: Vec<Option<usize>> = (0..total_chunks)
                    .map(|_| Some(rng.gen_range(0..onehot_k)))
                    .collect();
                OneHotPoly::<F>::new(onehot_k, indices).expect("onehot poly")
            })
            .collect();
        let poly_group: Vec<&OneHotPoly<F>> = polys.iter().collect();

        let pt = random_point::<F>(NV);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| {
                opening_from_poly_for_layout(
                    poly,
                    &pt,
                    &layout.final_group_scalar().expect("scalar final group"),
                    BasisMode::Lagrange,
                )
            })
            .collect();

        let setup = scheme.setup_prover(NV, 2).expect("setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit(
                &setup,
                &polys,
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");
        let selection = selection_for::<Cfg>(&commitment, scheme.schedules());

        let mut prover_transcript = AkitaTranscript::<F>::new(LABEL);
        let proof = scheme
            .batched_prove::<_, _, _>(
                &setup,
                prove_input::<Cfg, _>(
                    selection,
                    &pt[..],
                    &poly_group[..],
                    &commitment,
                    hint,
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize");

        // Terminal witness structure survives the serialization round trip.
        let terminal = decoded.terminal_response();
        assert_eq!(
            terminal.layout.groups.len(),
            1,
            "terminal consumer must retain one canonical scalar group"
        );
        terminal
            .terminal_transcript_parts()
            .expect("terminal witness must split into canonical transcript segments");

        let mut vt = AkitaTranscript::<F>::new(LABEL);
        scheme
            .batched_verify(
                &decoded,
                &verifier_setup,
                &mut vt,
                verify_input::<Cfg>(selection, &pt[..], &openings[..], &commitment),
                BasisMode::Lagrange,
            )
            .expect("batched onehot verification must pass");

        // Dropping a scheduled recursive fold must be rejected.
        assert!(
            !decoded.recursive_folds.is_empty(),
            "fixture must carry a recursive suffix"
        );
        let mut truncated = decoded.clone();
        truncated.recursive_folds.remove(0);
        let mut vt = AkitaTranscript::<F>::new(LABEL);
        assert!(
            scheme
                .batched_verify(
                    &truncated,
                    &verifier_setup,
                    &mut vt,
                    verify_input::<Cfg>(selection, &pt[..], &openings[..], &commitment),
                    BasisMode::Lagrange,
                )
                .is_err(),
            "proof with a truncated scheduled recursive suffix must be rejected"
        );
    });
}

/// The verifier must bind the proof to the committed group's geometry: a
/// statement carrying a commitment whose profile has been altered must not
/// verify against a proof produced for the real geometry.
#[test]
fn dense_rejects_mismatched_committed_group_profile_geometry() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::Dense;
        const D: usize = 256;
        const LABEL: &[u8] = b"soundness/profile-geometry";
        let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");

        let (verifier_setup, commitment, proof, opening_point, opening, _layout, selection) =
            make_dense_fixture::<F, D, Cfg>(&scheme, DENSE_TEST_NV, LABEL);
        let openings = [opening];

        // Sanity: the honest statement verifies.
        let mut vt = AkitaTranscript::<F>::new(LABEL);
        scheme
            .batched_verify(
                &proof,
                &verifier_setup,
                &mut vt,
                verify_input::<Cfg>(selection, &opening_point[..], &openings[..], &commitment),
                BasisMode::Lagrange,
            )
            .expect("honest dense proof must verify");

        let mut mismatched = commitment.clone();
        mismatched.profile.blocks.live_blocks =
            mismatched.profile.blocks.live_blocks.saturating_add(1);
        let mut vt = AkitaTranscript::<F>::new(LABEL);
        assert_invalid_proof(
            "mismatched committed-group profile geometry",
            scheme.batched_verify(
                &proof,
                &verifier_setup,
                &mut vt,
                verify_input::<Cfg>(selection, &opening_point[..], &openings[..], &mismatched),
                BasisMode::Lagrange,
            ),
        );
    });
}
