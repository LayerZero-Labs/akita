#![allow(missing_docs)]

use akita_prover::{ComputeBackendSetup, CpuBackend};

use akita_config::proof_optimized::fp128;
use akita_config::proof_optimized::{fp32, fp64};
use akita_config::test_support::akita_batched_root_layout;
use akita_config::CommitmentConfig;
use akita_field::{CanonicalBytes, CanonicalField, ExtField, FieldCore, TranscriptChallenge};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::AkitaProverSetup;
use akita_prover::DensePoly;
use akita_prover::OneHotPoly;
use akita_prover::{CommitmentProver, CommittedPolynomials, ProverClaims};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::AkitaScheduleLookupKey;
use akita_types::{lagrange_weights, LevelParams, RingSubfieldEncoding};
use akita_types::{
    AkitaBatchedProof, AkitaCommitmentHint, AkitaVerifierSetup, BasisMode, RingCommitment,
};
use akita_verifier::{CommitmentVerifier, CommittedOpenings, VerifierClaims};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "disk-persistence")]
use std::path::PathBuf;
use std::sync::{Mutex, Once};
use std::time::Instant;

mod common;
use common::opening_from_poly;

type F = fp128::Field;
const ONEHOT_K: usize = 256;
const FULL_TEST_NV: usize = 14;
const ONEHOT_TEST_NV: usize = 15;
const SAME_POINT_ONEHOT_BATCH_SIZE: usize = 4;
const D32_TEST_NV: usize = 12;

fn singleton_layout<Cfg: CommitmentConfig>(num_vars: usize) -> LevelParams {
    let opening_batch =
        akita_types::OpeningBatch::same_point(num_vars, 1).expect("singleton opening batch");
    Cfg::get_params_for_batched_commitment(&opening_batch).expect("singleton commitment layout")
}
const SMALL_FIELD_TEST_NV: usize = 8;
const TINY_DIRECT_TEST_NV: usize = 4;
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

fn prove_input<'a, FF: FieldCore, P, C, H>(
    point: &'a [FF],
    polynomials: &'a [P],
    commitment: &'a C,
    hint: H,
) -> ProverClaims<'a, FF, P, C, H> {
    (
        point,
        vec![CommittedPolynomials {
            polynomials,
            commitment,
            hint,
        }],
    )
}

fn verify_input<'a, FF: FieldCore, C>(
    point: &'a [FF],
    openings: &'a [FF],
    commitment: &'a C,
) -> VerifierClaims<'a, FF, C> {
    (
        point,
        vec![CommittedOpenings {
            openings,
            commitment,
        }],
    )
}

type DenseFixture<FField, E, L, const D: usize> = (
    AkitaVerifierSetup<FField>,
    RingCommitment<FField, D>,
    AkitaBatchedProof<FField, L>,
    Vec<E>,
    E,
    LevelParams,
);

/// Active log-basis of a runtime schedule's terminal direct step.
#[cfg(not(feature = "zk"))]
fn schedule_terminal_log_basis<Cfg: CommitmentConfig>(schedule: &akita_types::Schedule) -> u32 {
    let field_bits = Cfg::decomposition().field_bits();
    match schedule.steps.last() {
        Some(akita_types::Step::Direct(direct)) => direct.log_basis(field_bits),
        _ => panic!("schedule must end in a terminal direct step"),
    }
}

/// Count the total number of fold levels (including the batched root and the
/// terminal step) in a singleton-shaped batched proof, matching the planner's
/// `num_fold_levels` convention.
fn batched_total_fold_levels<FF: CanonicalField, L: FieldCore>(
    proof: &AkitaBatchedProof<FF, L>,
) -> usize {
    use akita_types::{AkitaBatchedRootProof, AkitaLevelProof};
    let root_fold = match proof.root {
        AkitaBatchedRootProof::Fold(_) | AkitaBatchedRootProof::Terminal(_) => 1,
        AkitaBatchedRootProof::ZeroFold { .. } => 0,
    };
    let suffix_fold = proof
        .steps
        .iter()
        .filter(|step| {
            matches!(
                step,
                AkitaLevelProof::Intermediate { .. } | AkitaLevelProof::Terminal { .. }
            )
        })
        .count();
    root_fold + suffix_fold
}

fn make_dense_fixture<
    FField: CanonicalField + CanonicalBytes + TranscriptChallenge + 'static,
    const D: usize,
    Cfg: CommitmentConfig<Field = FField>,
>(
    nv: usize,
    transcript_label: &'static [u8],
) -> DenseFixture<FField, Cfg::ExtField, Cfg::ExtField, D>
where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
        FField,
        D,
        ProverSetup = AkitaProverSetup<FField, D>,
        ExtField = Cfg::ExtField,
        VerifierSetup = AkitaVerifierSetup<FField>,
        Commitment = RingCommitment<FField, D>,
        CommitHint = AkitaCommitmentHint<FField, D>,
        BatchedProof = AkitaBatchedProof<FField, Cfg::ExtField>,
    >,
    Cfg::ExtField: RingSubfieldEncoding<FField> + AkitaSerialize,
    Cfg::ExtField: RingSubfieldEncoding<FField> + AkitaSerialize,
{
    let layout = singleton_layout::<Cfg>(nv);

    let mut rng = StdRng::seed_from_u64(0x0ddc_0ffe_e123_4567);
    let evals: Vec<FField> = (0..1usize << nv)
        .map(|_| FField::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();

    let poly = DensePoly::<FField, D>::from_field_evals(nv, &evals).unwrap();
    let pt = random_claim_point::<FField, Cfg::ExtField>(nv);
    let expected_opening = dense_lagrange_opening_from_evals::<FField, Cfg::ExtField>(&evals, &pt);

    #[cfg(feature = "disk-persistence")]
    purge_setup_cache(nv);

    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FField, D>>::setup_prover(nv, 1)
        .unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FField, D>>::setup_verifier(&setup);
    let (commitment, hint) =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FField, D>>::commit(
            &setup,
            &CpuBackend,
            &prepared,
            std::slice::from_ref(&poly),
        )
        .unwrap();

    let poly_refs: [&DensePoly<FField, D>; 1] = [&poly];
    let commitments = [commitment];
    let hints = vec![hint];

    let mut prover_transcript = AkitaTranscript::<FField>::new(transcript_label);
    let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FField, D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        prove_input(
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
        ),
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
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

#[cfg(not(feature = "zk"))]
fn bump_flat_ring_vec<FField: FieldCore>(flat: &mut akita_types::FlatRingVec<FField>) {
    let mut coeffs = flat.coeffs().to_vec();
    let first = coeffs
        .first_mut()
        .expect("tamper target must contain at least one coefficient");
    *first += FField::one();
    *flat = akita_types::FlatRingVec::from_coeffs(coeffs);
}

#[cfg(not(feature = "zk"))]
fn mutate_terminal_e_hat_digit<FField: FieldCore>(
    witness: &mut akita_types::CleartextWitnessProof<FField>,
    layout: akita_types::TerminalWitnessSegmentLayout,
) {
    let akita_types::CleartextWitnessProof::PackedDigits(packed) = witness else {
        panic!("trace tamper fixture should use packed terminal digits");
    };
    let mut digits = (0..packed.num_elems)
        .map(|idx| packed.digit_at(idx).expect("packed digit index"))
        .collect::<Vec<_>>();
    let digit = digits
        .get_mut(layout.e_hat_digit_offset)
        .expect("terminal e_hat offset must be in range");
    *digit = if *digit == -1 { 0 } else { -1 };
    *packed = akita_types::PackedDigits::from_i8_digits(&digits, packed.bits_per_elem);
}

#[cfg(not(feature = "zk"))]
fn terminal_witness_mut<FField: FieldCore, L: FieldCore>(
    proof: &mut AkitaBatchedProof<FField, L>,
) -> &mut akita_types::CleartextWitnessProof<FField> {
    match &mut proof.root {
        akita_types::AkitaBatchedRootProof::Terminal(terminal) => terminal
            .stage2
            .final_witness_mut()
            .expect("terminal root proof must carry terminal stage-2 proof"),
        akita_types::AkitaBatchedRootProof::Fold(_) => proof
            .steps
            .last_mut()
            .and_then(akita_types::AkitaLevelProof::as_terminal_mut)
            .and_then(|terminal| terminal.stage2_mut().final_witness_mut())
            .expect("fold-rooted proof must end in a terminal step"),
        akita_types::AkitaBatchedRootProof::ZeroFold { .. } => {
            panic!("terminal tamper test requires a folded terminal proof")
        }
    }
}

#[cfg(not(feature = "zk"))]
fn assert_invalid_proof<T: core::fmt::Debug>(
    case: &str,
    result: Result<T, akita_field::AkitaError>,
) {
    assert!(
        matches!(result, Err(akita_field::AkitaError::InvalidProof)),
        "{case} must reject with InvalidProof, got {result:?}"
    );
}

#[test]
fn full_d64_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64Full;
        const D: usize = Cfg::D;

        let layout = singleton_layout::<Cfg>(FULL_TEST_NV);

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let evals: Vec<F> = (0..1usize << FULL_TEST_NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let poly = DensePoly::<F, D>::from_field_evals(FULL_TEST_NV, &evals).unwrap();
        let pt = random_point::<F>(FULL_TEST_NV);
        let expected_opening = opening_from_poly(&poly, &pt, &layout);

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(FULL_TEST_NV);

        let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
            FULL_TEST_NV,
            1,
        )
        .unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
            &setup,
            &CpuBackend,
            &prepared,
            std::slice::from_ref(&poly),
        )
        .unwrap();

        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"akita_e2e");
        let prove_start = Instant::now();
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .unwrap();
        let prove_time = prove_start.elapsed();

        let proof_bytes = proof.size();
        assert!(proof_bytes > 0, "proof must be non-empty");
        let total_fold_levels = batched_total_fold_levels(&proof);
        assert!(total_fold_levels > 0, "proof must have at least one level");
        #[cfg(not(feature = "zk"))]
        {
            let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::singleton(FULL_TEST_NV))
                .expect("schedule plan");
            assert_eq!(total_fold_levels, plan.num_fold_levels());
        }

        let verifier_setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e");
        let verify_start = Instant::now();
        let verify_result =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input(&pt[..], opening_groups[0], &commitments[0]),
                BasisMode::Lagrange,
                akita_types::SetupContributionMode::Direct,
            );
        let verify_time = verify_start.elapsed();

        assert!(
            verify_result.is_ok(),
            "verification must pass: {:?}",
            verify_result.err()
        );

        tracing::info!(
            prove_s = prove_time.as_secs_f64(),
            verify_s = verify_time.as_secs_f64(),
            proof_bytes,
            proof_kib = proof_bytes as f64 / 1024.0,
            levels = total_fold_levels,
            "full-d64/nv{FULL_TEST_NV} e2e"
        );
    });
}

#[cfg(not(feature = "zk"))]
#[test]
fn trace_internalization_rejects_tampered_root_fold_handle() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64Full;
        const D: usize = Cfg::D;

        let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
            make_dense_fixture::<F, D, Cfg>(FULL_TEST_NV, b"akita_e2e/root-trace-tamper");
        let mut malformed = proof.clone();
        let root = malformed
            .root
            .as_fold_mut()
            .expect("fixture should use a folded root");
        bump_flat_ring_vec(&mut root.v);

        let commitments = [commitment];
        let openings = [opening];
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/root-trace-tamper");
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&opening_point[..], &openings[..], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );
        assert_invalid_proof("tampered root fold handle", result);
    });
}

#[cfg(not(feature = "zk"))]
#[test]
fn trace_internalization_rejects_tampered_recursive_fold_handle() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64OneHot;
        const D: usize = Cfg::D;
        const NV: usize = 20;

        let opening_batch = akita_types::OpeningBatch::same_point(NV, 2).expect("opening_batch");
        let layout = Cfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
        let total_field = (layout.num_blocks * layout.block_len)
            .checked_mul(D)
            .expect("total field size overflow");
        let total_chunks = total_field / ONEHOT_K;
        assert_eq!(total_chunks * ONEHOT_K, total_field);

        let polys: Vec<OneHotPoly<F, D>> = (0..2)
            .map(|poly_idx| {
                let mut rng = StdRng::seed_from_u64(0x3141_5926 + poly_idx as u64);
                let indices: Vec<Option<usize>> = (0..total_chunks)
                    .map(|_| Some(rng.gen_range(0..ONEHOT_K)))
                    .collect();
                OneHotPoly::<F, D>::new(ONEHOT_K, indices).unwrap()
            })
            .collect();
        let poly_refs: Vec<&OneHotPoly<F, D>> = polys.iter().collect();
        let point = random_point(NV);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly(poly, &point, &layout))
            .collect();

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(NV);

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 2).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let verifier_setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
            &setup,
            &CpuBackend,
            &prepared,
            &poly_refs,
        )
        .unwrap();
        let commitments = [commitment];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"akita_e2e/recursive-trace-tamper");
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(&point[..], &poly_refs[..], &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .unwrap();

        let mut malformed = proof.clone();
        let recursive = malformed
            .steps
            .iter_mut()
            .find_map(akita_types::AkitaLevelProof::as_intermediate_mut)
            .expect("fixture should include an intermediate recursive fold");
        bump_flat_ring_vec(recursive.v_mut());

        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"akita_e2e/recursive-trace-tamper");
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point[..], &openings[..], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );
        assert_invalid_proof("tampered recursive fold handle", result);
    });
}

#[cfg(not(feature = "zk"))]
#[test]
fn trace_internalization_rejects_tampered_terminal_e_hat_digit() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64Full;
        const D: usize = Cfg::D;

        let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
            make_dense_fixture::<F, D, Cfg>(FULL_TEST_NV, b"akita_e2e/terminal-trace-tamper");
        let schedule = Cfg::runtime_schedule(AkitaScheduleLookupKey::singleton(FULL_TEST_NV))
            .expect("runtime schedule");
        let terminal_params = schedule
            .fold_steps()
            .last()
            .expect("folded fixture should have a terminal fold")
            .params
            .clone();
        let terminal_layout =
            akita_types::terminal_witness_segment_layout(&terminal_params, 1, 1, F::modulus_bits())
                .expect("terminal layout");

        let mut malformed = proof.clone();
        mutate_terminal_e_hat_digit(terminal_witness_mut(&mut malformed), terminal_layout);

        let commitments = [commitment];
        let openings = [opening];
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/terminal-trace-tamper");
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&opening_point[..], &openings[..], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );
        assert_invalid_proof("tampered terminal e_hat digit", result);
    });
}

#[test]
fn full_d32_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D32Full;
        const D: usize = Cfg::D;

        let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
            make_dense_fixture::<F, D, Cfg>(D32_TEST_NV, b"akita_e2e/full-d32");

        #[cfg(not(feature = "zk"))]
        {
            let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::singleton(D32_TEST_NV))
                .expect("schedule plan");
            assert_eq!(batched_total_fold_levels(&proof), plan.num_fold_levels());
        }

        let commitments = [commitment];
        let openings = [opening];
        let opening_groups = [&openings[..]];

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/full-d32");
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&opening_point[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );

        assert!(
            result.is_ok(),
            "D32 verification must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn fp32_static_dense_round_trip() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type FSmall = fp32::Field;
        type Cfg = fp32::D64Full;
        const D: usize = Cfg::D;

        let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
            make_dense_fixture::<FSmall, D, Cfg>(SMALL_FIELD_TEST_NV, b"akita_e2e/fp32-static");

        let commitments = [commitment];
        let openings = [opening];
        let mut verifier_transcript = AkitaTranscript::<FSmall>::new(b"akita_e2e/fp32-static");
        let result =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<FSmall, D>>::batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input(&opening_point[..], &openings[..], &commitments[0]),
                BasisMode::Lagrange,
                akita_types::SetupContributionMode::Direct,
            );

        assert!(
            result.is_ok(),
            "fp32 static verification must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn fp64_static_dense_round_trip() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type FSmall = fp64::Field;
        type Cfg = fp64::D64Full;
        const D: usize = Cfg::D;

        let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
            make_dense_fixture::<FSmall, D, Cfg>(SMALL_FIELD_TEST_NV + 1, b"akita_e2e/fp64-static");

        let commitments = [commitment];
        let openings = [opening];
        let mut verifier_transcript = AkitaTranscript::<FSmall>::new(b"akita_e2e/fp64-static");
        let result =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<FSmall, D>>::batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input(&opening_point[..], &openings[..], &commitments[0]),
                BasisMode::Lagrange,
                akita_types::SetupContributionMode::Direct,
            );

        assert!(
            result.is_ok(),
            "fp64 static verification must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn full_d32_tiny_root_direct_roundtrip_and_serialization() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D32Full;
        const D: usize = Cfg::D;

        let nv = TINY_DIRECT_TEST_NV;
        #[cfg(not(feature = "zk"))]
        let plan = {
            let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::singleton(nv))
                .expect("schedule plan");
            assert_eq!(
                plan.num_fold_levels(),
                0,
                "tiny roots should use direct mode"
            );
            plan
        };

        let layout = singleton_layout::<Cfg>(nv);

        let mut rng = StdRng::seed_from_u64(0x0ddc_0ffe_e123_4567);
        let evals: Vec<F> = (0..1usize << nv)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).unwrap();
        let opening_point = random_point::<F>(nv);
        let opening = opening_from_poly(&poly, &opening_point, &layout);

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let verifier_setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
            &setup,
            &CpuBackend,
            &prepared,
            std::slice::from_ref(&poly),
        )
        .unwrap();
        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"akita_e2e/full-d32-direct-root");
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(
                &opening_point[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .unwrap();

        assert_eq!(batched_total_fold_levels(&proof), 0);
        assert!(proof.is_root_direct());
        #[cfg(not(feature = "zk"))]
        assert_eq!(proof.size(), plan.total_bytes);
        let direct_witnesses = proof
            .root
            .as_zero_fold()
            .expect("root-direct batched proof should carry per-claim field witnesses");
        assert_eq!(direct_witnesses.len(), 1);
        let direct_field = direct_witnesses[0]
            .as_field_elements()
            .expect("root-direct witness should keep raw field elements");
        assert_eq!(direct_field.coeff_len(), 1usize << nv);
        let reconstructed = DensePoly::<F, D>::from_field_evals(nv, direct_field.coeffs())
            .expect("reconstruct direct witness as dense poly");
        assert_eq!(
            opening_from_poly(&reconstructed, &opening_point, &layout),
            opening,
            "direct witness should preserve the public opening"
        );
        #[cfg(not(feature = "zk"))]
        {
            let (recomputed_commitment, _) =
                <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                    &setup,
                    &CpuBackend,
                    &prepared,
                    std::slice::from_ref(&reconstructed),
                )
                .expect("recompute commitment from direct witness");
            assert_eq!(
                recomputed_commitment, commitments[0],
                "direct witness should preserve the root commitment"
            );
        }

        let mut proof_bytes = Vec::new();
        proof
            .serialize_compressed(&mut proof_bytes)
            .expect("serialize direct-root proof");
        let mut cursor = std::io::Cursor::new(proof_bytes);
        let decoded =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&mut cursor, &proof.shape())
                .expect("deserialize direct-root proof");
        assert_eq!(decoded, proof);

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/full-d32-direct-root");
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&opening_point[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );

        assert!(
            result.is_ok(),
            "tiny D32 direct verification must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn full_d64_adaptive_mixed_basis_roundtrip_and_serialization() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64Full;
        const D: usize = Cfg::D;

        let nv = FULL_TEST_NV;
        let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
            make_dense_fixture::<F, D, Cfg>(nv, b"akita_e2e/adaptive-full-mixed");

        #[cfg(not(feature = "zk"))]
        {
            let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::singleton(nv))
                .expect("schedule plan");
            assert_eq!(batched_total_fold_levels(&proof), plan.num_fold_levels());

            assert_eq!(
                proof
                    .final_witness()
                    .as_packed_digits()
                    .expect("current terminal witness should be packed digits")
                    .bits_per_elem,
                schedule_terminal_log_basis::<Cfg>(&plan)
            );
        }

        let mut proof_bytes = Vec::new();
        proof
            .serialize_compressed(&mut proof_bytes)
            .expect("serialize adaptive proof");
        let mut cursor = std::io::Cursor::new(proof_bytes);
        let decoded =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&mut cursor, &proof.shape())
                .expect("deserialize adaptive proof");
        assert_eq!(decoded, proof);

        let commitments = [commitment];
        let openings = [opening];
        let opening_groups = [&openings[..]];

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/adaptive-full-mixed");
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&opening_point[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );
        assert!(
            result.is_ok(),
            "adaptive mixed-basis verification must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn adaptive_onehot_direct_tail_uses_terminal_schedule_basis() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64OneHot;
        const D: usize = Cfg::D;

        let nv = ONEHOT_TEST_NV;
        let layout = singleton_layout::<Cfg>(nv);
        let total_field = (layout.num_blocks * layout.block_len)
            .checked_mul(D)
            .expect("total field size overflow");
        let total_chunks = total_field / ONEHOT_K;
        assert_eq!(total_chunks * ONEHOT_K, total_field);

        let mut rng = StdRng::seed_from_u64(0x1234_abcd);
        let indices: Vec<Option<usize>> = (0..total_chunks)
            .map(|_| Some(rng.gen_range(0..ONEHOT_K)))
            .collect();
        let onehot_poly = OneHotPoly::<F, D>::new(ONEHOT_K, indices).unwrap();
        let pt = random_point::<F>(nv);
        let expected_opening = opening_from_poly(&onehot_poly, &pt, &layout);

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(nv);

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let verifier_setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
            &setup,
            &CpuBackend,
            &prepared,
            std::slice::from_ref(&onehot_poly),
        )
        .unwrap();

        let poly_refs: [&OneHotPoly<F, D>; 1] = [&onehot_poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"akita_e2e/onehot-direct-tail");
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .unwrap();

        let mut serialized = Vec::new();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize adaptive onehot proof");
        let mut cursor = std::io::Cursor::new(serialized);
        let decoded =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&mut cursor, &proof.shape())
                .expect("deserialize adaptive onehot proof");
        #[cfg(not(feature = "zk"))]
        {
            let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::singleton(nv))
                .expect("schedule plan");
            assert_eq!(batched_total_fold_levels(&proof), plan.num_fold_levels());
            // `Schedule::total_bytes` is the planner's conservative upper bound:
            // `level_proof_bytes` sizes every stage-2 sumcheck round as
            // degree-3, but the prover ships a degree-2 first round per
            // stage-2 sumcheck (one challenge-field element fewer per fold
            // level). So the runtime proof never exceeds the estimate and
            // undershoots it by at most one challenge element per fold level.
            assert!(
                proof.size() <= plan.total_bytes,
                "runtime proof {} exceeds planner upper bound {}",
                proof.size(),
                plan.total_bytes
            );
            let challenge_elem = F::zero().serialized_size(akita_serialization::Compress::No);
            let overcount = plan.total_bytes - proof.size();
            assert!(
                overcount <= plan.num_fold_levels() * challenge_elem,
                "planner estimate {} overcounts runtime proof {} by {overcount} bytes, \
                 exceeding the {} stage-2 degree-2 rounds * {challenge_elem}B tolerance",
                plan.total_bytes,
                proof.size(),
                plan.num_fold_levels()
            );
            assert_eq!(
                decoded
                    .final_witness()
                    .as_packed_digits()
                    .expect("current terminal witness should be packed digits")
                    .bits_per_elem,
                schedule_terminal_log_basis::<Cfg>(&plan)
            );
        }

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/onehot-direct-tail");
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );
        assert!(
            result.is_ok(),
            "adaptive onehot direct-tail verification must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn adaptive_onehot_schedule_stays_within_basis_envelope() {
    type Cfg = fp128::D64OneHot;

    // The planner's basis search window for `proof_optimized` configurations
    // is `[PROOF_OPTIMIZED_LOG_BASIS_MIN, PROOF_OPTIMIZED_LOG_BASIS_MAX]`,
    // which currently caps at `log_basis = 6`. Allow the DP to reach the top
    // of that window (the zk preset legitimately picks `log_basis = 6` for
    // some `nv` values once the regenerated tables stop seeding the search
    // with stale singleton plans); the assertion exists only to catch any
    // future planner change that escapes the configured envelope.
    for nv in 10..=120 {
        let schedule = match Cfg::runtime_schedule(AkitaScheduleLookupKey::singleton(nv)) {
            Ok(schedule) => schedule,
            Err(_) => continue,
        };
        let within_window = schedule.steps.iter().all(|step| match step {
            akita_types::Step::Fold(fold) => fold.params.log_basis <= 6,
            // A terminal direct ships packed digits at the terminal fold's
            // basis (window-bounded). A root direct ships raw field elements:
            // it is the zero-fold / uncommittable edge with no fold basis to
            // bound. Under honest A-role pricing, D=64 stops securing a fold
            // for very large `num_vars`, so the DP returns this edge instead
            // of a folded schedule; it carries no basis to check.
            akita_types::Step::Direct(direct) => match direct.witness_shape {
                akita_types::CleartextWitnessShape::PackedDigits((_, bits)) => bits <= 6,
                akita_types::CleartextWitnessShape::FieldElements(_) => true,
            },
        });
        assert!(
            within_window,
            "adaptive onehot schedule selected log_basis > 6 at nv={nv}: {schedule:?}"
        );
    }
}

#[test]
fn batched_onehot_same_point_round_trip() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        // NV=20 is large enough to include a recursive suffix, while the
        // two-claim opening_batch still misses singleton/4-batch generated tables
        // and routes through the planner DP fallback in `runtime_schedule`.
        type Cfg = fp128::D64OneHot;
        const D: usize = Cfg::D;
        const NV: usize = 20;

        let nv = NV;
        let opening_batch = akita_types::OpeningBatch::same_point(nv, 2).expect("opening_batch");
        let layout = Cfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
        let total_field = (layout.num_blocks * layout.block_len)
            .checked_mul(D)
            .expect("total field size overflow");
        let total_chunks = total_field / ONEHOT_K;
        assert_eq!(total_chunks * ONEHOT_K, total_field);

        let mut rng_a = StdRng::seed_from_u64(0x1234_5678);
        let mut rng_b = StdRng::seed_from_u64(0x8765_4321);
        let indices_a: Vec<Option<usize>> = (0..total_chunks)
            .map(|_| Some(rng_a.gen_range(0..ONEHOT_K)))
            .collect();
        let indices_b: Vec<Option<usize>> = (0..total_chunks)
            .map(|_| Some(rng_b.gen_range(0..ONEHOT_K)))
            .collect();
        let poly_a = OneHotPoly::<F, D>::new(ONEHOT_K, indices_a).unwrap();
        let poly_b = OneHotPoly::<F, D>::new(ONEHOT_K, indices_b).unwrap();
        let poly_group = [&poly_a, &poly_b];
        let pt = random_point(nv);
        let openings = [
            opening_from_poly(&poly_a, &pt, &layout),
            opening_from_poly(&poly_b, &pt, &layout),
        ];

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(nv);

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 2).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let verifier_setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
            &setup,
            &CpuBackend,
            &prepared,
            &poly_group,
        )
        .unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"akita_e2e/batched-onehot");
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(
                &pt[..],
                &poly_group[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .unwrap();

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize batched onehot proof");
        let mut cursor = std::io::Cursor::new(serialized);
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(&mut cursor, &proof_shape)
            .expect("deserialize batched onehot proof");

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"akita_e2e/batched-onehot");
        let opening_groups = [&openings[..]];
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );
        assert!(
            result.is_ok(),
            "batched onehot verification must pass: {:?}",
            result.err()
        );

        assert!(
            decoded.num_fold_levels() > 0,
            "test fixture must include a recursive suffix to cover truncation"
        );
        let mut truncated = decoded.clone();
        truncated.steps.remove(0);
        let mut truncated_transcript = AkitaTranscript::<F>::new(b"akita_e2e/batched-onehot");
        let truncated_result =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                &truncated,
                &verifier_setup,
                &mut truncated_transcript,
                verify_input(&pt[..], opening_groups[0], &commitments[0]),
                BasisMode::Lagrange,
                akita_types::SetupContributionMode::Direct,
            );
        assert!(
            truncated_result.is_err(),
            "proof with a truncated scheduled recursive suffix must be rejected"
        );
    });
}

#[test]
fn batched_onehot_same_point_rejects_tampered_root_stage1_s_claim() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64OneHot;
        const D: usize = Cfg::D;

        let nv = ONEHOT_TEST_NV;
        let layout =
            akita_batched_root_layout::<Cfg>(nv, SAME_POINT_ONEHOT_BATCH_SIZE).expect("layout");
        let total_field = (layout.num_blocks * layout.block_len)
            .checked_mul(D)
            .expect("total field size overflow");
        let total_chunks = total_field / ONEHOT_K;
        assert_eq!(total_chunks * ONEHOT_K, total_field);

        let polys: Vec<OneHotPoly<F, D>> = (0..SAME_POINT_ONEHOT_BATCH_SIZE)
            .map(|poly_idx| {
                let mut rng = StdRng::seed_from_u64(0x8765_4321 + poly_idx as u64);
                let indices: Vec<Option<usize>> = (0..total_chunks)
                    .map(|_| Some(rng.gen_range(0..ONEHOT_K)))
                    .collect();
                OneHotPoly::<F, D>::new(ONEHOT_K, indices).unwrap()
            })
            .collect();
        let poly_group: Vec<&OneHotPoly<F, D>> = polys.iter().collect();
        let pt = random_point(nv);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly(poly, &pt, &layout))
            .collect();

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(nv);

        let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
            nv,
            SAME_POINT_ONEHOT_BATCH_SIZE,
        )
        .unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let verifier_setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
            &setup,
            &CpuBackend,
            &prepared,
            &poly_group,
        )
        .unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"akita_e2e/batched-onehot-s-claim-tamper");
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(
                &pt[..],
                &poly_group[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .unwrap();

        let mut malformed = proof.clone();
        // After the terminal-fold soundness fix, the root may be either a
        // `Fold` (intermediate) variant with a stage-1 sumcheck or a
        // `Terminal` variant (1-fold case) with no stage-1. Tamper whichever
        // applies so the test exercises root-level tamper rejection for
        // either schedule shape.
        match malformed.root {
            akita_types::AkitaBatchedRootProof::Fold(ref mut fold) => {
                fold.stage1.s_claim += F::from_canonical_u128_reduced(1);
            }
            akita_types::AkitaBatchedRootProof::Terminal(ref mut terminal) => {
                match terminal
                    .stage2
                    .final_witness_mut()
                    .expect("terminal root proof must carry terminal stage-2 proof")
                {
                    akita_types::CleartextWitnessProof::PackedDigits(packed) => {
                        packed.data[0] ^= 1;
                    }
                    akita_types::CleartextWitnessProof::FieldElements(_) => {
                        panic!("expected packed-digits final witness for tamper test");
                    }
                }
            }
            akita_types::AkitaBatchedRootProof::ZeroFold { .. } => {
                panic!("root-direct batched proof has no folded root to tamper");
            }
        }

        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"akita_e2e/batched-onehot-s-claim-tamper");
        let opening_groups = [&openings[..]];
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );
        assert!(
            result.is_err(),
            "tampered batched root stage1 s_claim must be rejected"
        );
    });
}
