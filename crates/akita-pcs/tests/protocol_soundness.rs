#![allow(missing_docs)]

use akita_prover::{ComputeBackendSetup, CpuBackend};

use akita_config::proof_optimized::fp128;
use akita_config::proof_optimized::{fp32, fp64};
use akita_config::test_support::akita_batched_root_layout;
use akita_config::CommitmentConfig;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps, HasWide, ReduceTo};
use akita_field::Zero;
use akita_field::{
    CanonicalBytes, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, PseudoMersenneField, RandomSampling, TranscriptChallenge,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::DensePoly;
use akita_prover::OneHotPoly;
use akita_prover::{ProverOpeningData, SelectedProverOpeningData};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_transcript::AkitaTranscript;
use akita_types::{lagrange_weights, CommittedGroupParams, FpExtEncoding};
use akita_types::{
    AkitaBatchedProof, AkitaCommitmentHint, AkitaVerifierSetup, BasisMode, CommittedGroup,
    CommittedGroupBatchProfile, GroupBatchStatement, OpeningClaims, OpeningScheduleSelection,
    PolynomialGroupClaims,
};
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "disk-persistence")]
use std::path::PathBuf;
use std::sync::{Mutex, Once};
use std::time::Instant;

mod common;
use common::opening_from_poly_for_layout;

type F = fp128::Field;
const ONEHOT_K: usize = 256;
const DENSE_TEST_NV: usize = 14;
const ONEHOT_TEST_NV: usize = 15;
const SAME_POINT_ONEHOT_BATCH_SIZE: usize = 4;

fn singleton_layout<Cfg: CommitmentConfig>(num_vars: usize) -> CommittedGroupParams {
    let opening_batch =
        akita_types::OpeningClaimsLayout::new(num_vars, 1).expect("singleton opening batch");
    Cfg::get_params_for_batched_commitment(&opening_batch).expect("singleton commitment layout")
}
const SMALL_FIELD_TEST_NV: usize = 8;
const STACK_SIZE: usize = 256 * 1024 * 1024;

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

fn random_point<FField: CanonicalField>(nv: usize) -> Vec<FField> {
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| FField::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

fn random_claim_point<FField, E>(nv: usize) -> Vec<E>
where
    FField: CanonicalField,
    E: ExtField<FField>,
{
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| {
            let limbs = (0..E::EXT_DEGREE)
                .map(|_| FField::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect::<Vec<_>>();
            E::from_base_slice(&limbs)
        })
        .collect()
}

fn dense_lagrange_opening_from_evals<FField, E>(evals: &[FField], point: &[E]) -> E
where
    FField: FieldCore,
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
    (
        selection,
        ProverOpeningData::new(opening_claims, vec![hint], vec![polynomials])
            .expect("valid prover opening data"),
    )
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
) -> OpeningScheduleSelection {
    Cfg::select_schedule_for_profiles(&CommittedGroupBatchProfile {
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

/// Count the total number of fold levels (including the batched root and the
/// terminal step) in a singleton-shaped batched proof, matching the planner's
/// `num_fold_levels` convention.
fn batched_total_fold_levels<FF: CanonicalField, E: FieldCore>(
    proof: &AkitaBatchedProof<FF, E>,
) -> usize {
    proof.num_fold_levels()
}

fn make_dense_fixture<FField, const D: usize, Cfg: CommitmentConfig<Field = FField>>(
    nv: usize,
    transcript_label: &'static [u8],
) -> DenseFixture<FField, Cfg::ExtField, D>
where
    FField: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + HasWide
        + RandomSampling
        + FromPrimitiveInt
        + 'static
        + HalvingField
        + PseudoMersenneField
        + Valid,
    Cfg::ExtField: FrobeniusExtField<FField> + HasUnreducedOps + HasOptimizedFold,
    <FField as HasWide>::Wide: From<FField> + ReduceTo<FField>,
    Cfg::ExtField: FpExtEncoding<FField> + AkitaSerialize,
{
    let layout = singleton_layout::<Cfg>(nv);

    let mut rng = StdRng::seed_from_u64(0x0ddc_0ffe_e123_4567);
    let evals: Vec<FField> = (0..1usize << nv)
        .map(|_| FField::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();

    let poly = DensePoly::<FField>::from_field_evals(nv, D, &evals).unwrap();
    let pt = random_claim_point::<FField, Cfg::ExtField>(nv);
    let expected_opening = dense_lagrange_opening_from_evals::<FField, Cfg::ExtField>(&evals, &pt);

    #[cfg(feature = "disk-persistence")]
    purge_setup_cache(nv);

    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 1).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let verifier_setup =
        AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).expect("verifier setup");
    let (commitment, hint) =
        AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack)
            .unwrap();

    let poly_refs: [&DensePoly<FField>; 1] = [&poly];
    let commitments = [commitment];
    let selection = selection_for::<Cfg>(&commitments[0]);
    let hints = vec![hint];

    let mut prover_transcript = AkitaTranscript::<FField>::new(transcript_label);
    let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
        &setup,
        prove_input::<Cfg, _>(
            selection,
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
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

fn bump_flat_ring_vec<FField: FieldCore>(flat: &mut akita_types::RingVec<FField>) {
    let mut coeffs = flat.coeffs().to_vec();
    let first = coeffs
        .first_mut()
        .expect("tamper target must contain at least one coefficient");
    *first += FField::one();
    *flat = akita_types::RingVec::from_coeffs(coeffs);
}

fn mutate_terminal_e_hat_digit<FField: FieldCore>(
    witness: &mut akita_types::TerminalResponse<FField>,
) {
    bump_flat_ring_vec(&mut witness.e_fields);
}

fn terminal_witness_mut<FField: FieldCore, E: FieldCore>(
    proof: &mut AkitaBatchedProof<FField, E>,
) -> &mut akita_types::TerminalResponse<FField> {
    proof.terminal.terminal_response_mut()
}

fn assert_invalid_proof<T: core::fmt::Debug>(
    case: &str,
    result: Result<T, akita_field::AkitaError>,
) {
    match result {
        Err(akita_field::AkitaError::InvalidProof) => {}
        Err(akita_field::AkitaError::InvalidInput(msg)) if msg.contains("InvalidProof") => {}
        other => panic!("{case} must reject with InvalidProof, got {other:?}"),
    }
}


#[test]
fn trace_internalization_rejects_tampered_root_fold_handle() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::Dense;
        const D: usize = Cfg::D;

        let (verifier_setup, commitment, proof, opening_point, opening, _layout, selection) =
            make_dense_fixture::<F, D, Cfg>(DENSE_TEST_NV, b"akita_e2e/root-trace-tamper");
        let mut malformed = proof.clone();
        bump_flat_ring_vec(&mut malformed.root.opening_payload);

        let commitments = [commitment];
        let openings = [opening];
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/root-trace-tamper");
        let result = AkitaCommitmentScheme::<Cfg>::batched_verify(
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

        let opening_batch = akita_types::OpeningClaimsLayout::new(NV, 2).expect("opening_batch");
        let layout = Cfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
        let root_d = layout.d_a();
        let total_field = (layout.num_live_blocks * layout.num_positions_per_block)
            .checked_mul(root_d)
            .expect("total field size overflow");
        let total_chunks = total_field / ONEHOT_K;
        assert_eq!(total_chunks * ONEHOT_K, total_field);

        let polys: Vec<OneHotPoly<F>> = (0..2)
            .map(|poly_idx| {
                let mut rng = StdRng::seed_from_u64(0x3141_5926 + poly_idx as u64);
                let indices: Vec<Option<usize>> = (0..total_chunks)
                    .map(|_| Some(rng.gen_range(0..ONEHOT_K)))
                    .collect();
                OneHotPoly::<F>::new(ONEHOT_K, root_d, indices).unwrap()
            })
            .collect();
        let poly_refs: Vec<&OneHotPoly<F>> = polys.iter().collect();
        let point = random_point(NV);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly_for_layout(poly, &point, &layout))
            .collect();

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(NV);

        let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(NV, 2).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup =
            AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).expect("verifier setup");
        let (commitment, hint) =
            AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, &polys, &stack).unwrap();
        let commitments = [commitment];
        let selection = selection_for::<Cfg>(&commitments[0]);

        let mut prover_transcript = AkitaTranscript::<F>::new(b"akita_e2e/recursive-trace-tamper");
        let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
            &setup,
            prove_input::<Cfg, _>(selection, &point[..], &poly_refs[..], &commitments[0], hint),
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
        let result = AkitaCommitmentScheme::<Cfg>::batched_verify(
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
        const D: usize = Cfg::D;

        let (verifier_setup, commitment, proof, opening_point, opening, _layout, selection) =
            make_dense_fixture::<F, D, Cfg>(DENSE_TEST_NV, b"akita_e2e/terminal-trace-tamper");
        let mut malformed = proof.clone();
        mutate_terminal_e_hat_digit(terminal_witness_mut(&mut malformed));

        let commitments = [commitment];
        let openings = [opening];
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/terminal-trace-tamper");
        let result = AkitaCommitmentScheme::<Cfg>::batched_verify(
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
    for result in [
        fp32::Dense::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(SMALL_FIELD_TEST_NV),
        )),
        fp64::Dense::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(SMALL_FIELD_TEST_NV + 1),
        )),
    ] {
        assert!(matches!(
            result,
            Err(akita_field::AkitaError::UnsupportedSchedule(_))
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
        let err = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(nv),
        ))
        .expect_err("tiny roots must not produce a degenerate proof schedule");
        assert!(matches!(
            err,
            akita_field::AkitaError::UnsupportedSchedule(_)
        ));
        let setup_err = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 1)
            .expect_err("tiny capacity must not produce a prover setup");
        assert!(
            matches!(setup_err, akita_field::AkitaError::InvalidSetup(_)),
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

        let nv = ONEHOT_TEST_NV;
        let layout =
            akita_batched_root_layout::<Cfg>(nv, SAME_POINT_ONEHOT_BATCH_SIZE).expect("layout");
        let root_d = layout.d_a();
        let total_field = (layout.num_live_blocks * layout.num_positions_per_block)
            .checked_mul(root_d)
            .expect("total field size overflow");
        let total_chunks = total_field / ONEHOT_K;
        assert_eq!(total_chunks * ONEHOT_K, total_field);

        let polys: Vec<OneHotPoly<F>> = (0..SAME_POINT_ONEHOT_BATCH_SIZE)
            .map(|poly_idx| {
                let mut rng = StdRng::seed_from_u64(0x8765_4321 + poly_idx as u64);
                let indices: Vec<Option<usize>> = (0..total_chunks)
                    .map(|_| Some(rng.gen_range(0..ONEHOT_K)))
                    .collect();
                OneHotPoly::<F>::new(ONEHOT_K, root_d, indices).unwrap()
            })
            .collect();
        let poly_group: Vec<&OneHotPoly<F>> = polys.iter().collect();
        let pt = random_point(nv);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly_for_layout(poly, &pt, &layout))
            .collect();

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(nv);

        let setup =
            AkitaCommitmentScheme::<Cfg>::setup_prover(nv, SAME_POINT_ONEHOT_BATCH_SIZE).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup =
            AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).expect("verifier setup");
        let (commitment, hint) =
            AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, &polys, &stack).unwrap();
        let commitments = [commitment];
        let selection = selection_for::<Cfg>(&commitments[0]);
        let hints = vec![hint];

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"akita_e2e/batched-onehot-s-claim-tamper");
        let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
            &setup,
            prove_input::<Cfg, _>(
                selection,
                &pt[..],
                &poly_group[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut malformed = proof.clone();
        malformed.root.stage1.range_image_evaluation += F::from_canonical_u128_reduced(1);

        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"akita_e2e/batched-onehot-s-claim-tamper");
        let opening_groups = [&openings[..]];
        let result = AkitaCommitmentScheme::<Cfg>::batched_verify(
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
