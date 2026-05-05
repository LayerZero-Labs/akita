#![allow(missing_docs)]

use akita_config::akita_batched_root_layout;
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::{CanonicalBytes, CanonicalField, FieldCore, TranscriptChallenge};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::AkitaPolyOps;
use akita_prover::DensePoly;
use akita_prover::OneHotPoly;
use akita_prover::{CommitmentProver, CommittedPolynomials, ProverClaims};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::Blake2bTranscript;
use akita_types::LevelParams;
use akita_types::{reduce_inner_opening_to_ring_element, ring_opening_point_from_field};
use akita_types::{
    AkitaBatchedProof, AkitaCommitmentHint, AkitaVerifierSetup, BasisMode, BlockOrder,
    RingCommitment,
};
use akita_types::{AkitaScheduleInputs, AkitaScheduleLookupKey, ScheduleProvider};
use akita_verifier::{CommitmentVerifier, CommittedOpenings, VerifierClaims};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "disk-persistence")]
use std::path::PathBuf;
use std::sync::{Mutex, Once};
use std::time::Instant;

type F = fp128::Field;
const ONEHOT_K: usize = 256;
const FULL_TEST_NV: usize = 14;
const ONEHOT_TEST_NV: usize = 15;
const D32_TEST_NV: usize = 12;
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
    vec![(
        point,
        vec![CommittedPolynomials {
            polynomials,
            commitment,
            hint,
        }],
    )]
}

fn verify_input<'a, FF: FieldCore, C>(
    point: &'a [FF],
    openings: &'a [FF],
    commitment: &'a C,
) -> VerifierClaims<'a, FF, C> {
    vec![(
        point,
        vec![CommittedOpenings {
            openings,
            commitment,
        }],
    )]
}

type DenseFixture<FField, const D: usize> = (
    AkitaVerifierSetup<FField>,
    RingCommitment<FField, D>,
    AkitaBatchedProof<FField>,
    Vec<FField>,
    FField,
    LevelParams,
);

/// Count the total number of fold levels (including the batched root) in a
/// singleton-shaped batched proof, matching the planner's
/// `num_fold_levels` convention.
fn batched_total_fold_levels<FF: CanonicalField>(proof: &AkitaBatchedProof<FF>) -> usize {
    let root_fold = if proof.root.as_fold().is_some() { 1 } else { 0 };
    root_fold + proof.num_fold_levels()
}

fn make_dense_fixture<
    FField: CanonicalField + CanonicalBytes + TranscriptChallenge + 'static,
    const D: usize,
    Cfg: CommitmentConfig<Field = FField>,
>(
    nv: usize,
    transcript_label: &'static [u8],
) -> DenseFixture<FField, D>
where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
        FField,
        D,
        VerifierSetup = AkitaVerifierSetup<FField>,
        Commitment = RingCommitment<FField, D>,
        CommitHint = AkitaCommitmentHint<FField, D>,
        BatchedProof = AkitaBatchedProof<FField>,
    >,
{
    let layout = Cfg::commitment_layout::<akita_types::Transparent>(nv).expect("layout");

    let mut rng = StdRng::seed_from_u64(0x0ddc_0ffe_e123_4567);
    let evals: Vec<FField> = (0..1usize << nv)
        .map(|_| FField::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();

    let poly = DensePoly::<FField, D>::from_field_evals(nv, &evals).unwrap();
    let pt = random_point::<FField>(nv);
    let expected_opening = opening_from_poly(&poly, &pt, &layout);

    #[cfg(feature = "disk-persistence")]
    purge_setup_cache(nv);

    let setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FField, D>>::setup_prover(nv, 1, 1);
    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FField, D>>::setup_verifier(&setup);
    let (commitment, hint) =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FField, D>>::commit(
            std::slice::from_ref(&poly),
            &setup,
        )
        .unwrap();

    let poly_refs: [&DensePoly<FField, D>; 1] = [&poly];
    let commitments = [commitment];
    let hints = vec![hint];

    let mut prover_transcript = Blake2bTranscript::<FField>::new(transcript_label);
    let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FField, D>>::batched_prove(
        &setup,
        prove_input(
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
        ),
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

fn opening_from_poly<FField: CanonicalField, const D: usize, P: AkitaPolyOps<FField, D>>(
    poly: &P,
    point: &[FField],
    layout: &LevelParams,
) -> FField {
    let alpha_bits = D.trailing_zeros() as usize;
    let target_num_vars = alpha_bits + layout.m_vars + layout.r_vars;
    assert!(
        point.len() <= target_num_vars,
        "opening point length {} exceeds target root arity {}",
        point.len(),
        target_num_vars
    );
    let mut padded_point = point.to_vec();
    padded_point.resize(target_num_vars, FField::zero());

    let inner_point = &padded_point[..alpha_bits];
    let reduced_point = &padded_point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
        BlockOrder::RowMajor,
    )
    .expect("opening point shape should match layout");

    let (y_ring, _) = poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        layout.block_len,
    );
    let v = reduce_inner_opening_to_ring_element::<FField, D>(inner_point, BasisMode::Lagrange)
        .expect("inner opening point should match ring dimension");
    (y_ring * v.sigma_m1()).coefficients()[0]
}

#[test]
fn full_d128_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;

        let layout =
            Cfg::commitment_layout::<akita_types::Transparent>(FULL_TEST_NV).expect("layout");

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
            1,
        );
        let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
            std::slice::from_ref(&poly),
            &setup,
        )
        .unwrap();

        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"akita_e2e");
        let prove_start = Instant::now();
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();
        let prove_time = prove_start.elapsed();

        let proof_bytes = proof.size();
        assert!(proof_bytes > 0, "proof must be non-empty");
        let total_fold_levels = batched_total_fold_levels(&proof);
        assert!(total_fold_levels > 0, "proof must have at least one level");
        let plan = Cfg::schedule_plan::<akita_types::Transparent>(
            AkitaScheduleLookupKey::singleton(FULL_TEST_NV, FULL_TEST_NV, 1),
        )
        .expect("schedule plan")
        .expect("adaptive full config should expose a schedule plan");
        assert_eq!(total_fold_levels, plan.num_fold_levels());

        let verifier_setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"akita_e2e");
        let verify_start = Instant::now();
        let verify_result =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input(&pt[..], opening_groups[0], &commitments[0]),
                BasisMode::Lagrange,
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
            "full-d128/nv{FULL_TEST_NV} e2e"
        );
    });
}

#[test]
fn full_d32_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D32Full;
        const D: usize = Cfg::D;

        let plan = Cfg::schedule_plan::<akita_types::Transparent>(
            AkitaScheduleLookupKey::singleton(D32_TEST_NV, D32_TEST_NV, 1),
        )
        .expect("schedule plan")
        .expect("adaptive D32 config should expose a schedule plan");
        let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
            make_dense_fixture::<F, D, Cfg>(D32_TEST_NV, b"akita_e2e/full-d32");

        assert_eq!(batched_total_fold_levels(&proof), plan.num_fold_levels());

        let commitments = [commitment];
        let openings = [opening];
        let opening_groups = [&openings[..]];

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"akita_e2e/full-d32");
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&opening_point[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );

        assert!(
            result.is_ok(),
            "D32 verification must pass: {:?}",
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
        let plan = Cfg::schedule_plan::<akita_types::Transparent>(
            AkitaScheduleLookupKey::singleton(nv, nv, 1),
        )
        .expect("schedule plan")
        .expect("adaptive D32 config should expose a schedule plan");
        assert_eq!(
            plan.num_fold_levels(),
            0,
            "tiny roots should use direct mode"
        );

        let layout = Cfg::commitment_layout::<akita_types::Transparent>(nv).expect("layout");

        let mut rng = StdRng::seed_from_u64(0x0ddc_0ffe_e123_4567);
        let evals: Vec<F> = (0..1usize << nv)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).unwrap();
        let opening_point = random_point::<F>(nv);
        let opening = opening_from_poly(&poly, &opening_point, &layout);

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
        let verifier_setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
            std::slice::from_ref(&poly),
            &setup,
        )
        .unwrap();
        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"akita_e2e/full-d32-direct-root");
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(
                &opening_point[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        assert_eq!(batched_total_fold_levels(&proof), 0);
        assert!(proof.is_root_direct());
        assert_eq!(proof.size(), plan.exact_proof_bytes);
        let direct_witnesses = proof
            .root
            .as_direct()
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
        let (recomputed_commitment, _) =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                std::slice::from_ref(&reconstructed),
                &setup,
            )
            .expect("recompute commitment from direct witness");
        assert_eq!(
            recomputed_commitment, commitments[0],
            "direct witness should preserve the root commitment"
        );

        let mut proof_bytes = Vec::new();
        proof
            .serialize_compressed(&mut proof_bytes)
            .expect("serialize direct-root proof");
        let mut cursor = std::io::Cursor::new(proof_bytes);
        let decoded = AkitaBatchedProof::<F>::deserialize_compressed(&mut cursor, &proof.shape())
            .expect("deserialize direct-root proof");
        assert_eq!(decoded, proof);

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"akita_e2e/full-d32-direct-root");
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&opening_point[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );

        assert!(
            result.is_ok(),
            "tiny D32 direct verification must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn full_d128_adaptive_mixed_basis_roundtrip_and_serialization() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;

        let nv = FULL_TEST_NV;
        let plan = Cfg::schedule_plan::<akita_types::Transparent>(
            AkitaScheduleLookupKey::singleton(nv, nv, 1),
        )
        .expect("schedule plan")
        .expect("adaptive full config should expose a schedule plan");
        let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
            make_dense_fixture::<F, D, Cfg>(nv, b"akita_e2e/adaptive-full-mixed");

        assert_eq!(batched_total_fold_levels(&proof), plan.num_fold_levels());

        let mut proof_bytes = Vec::new();
        proof
            .serialize_compressed(&mut proof_bytes)
            .expect("serialize adaptive proof");
        let mut cursor = std::io::Cursor::new(proof_bytes);
        let decoded = AkitaBatchedProof::<F>::deserialize_compressed(&mut cursor, &proof.shape())
            .expect("deserialize adaptive proof");
        assert_eq!(decoded, proof);

        assert_eq!(
            decoded
                .final_witness()
                .as_packed_digits()
                .expect("current terminal witness should be packed digits")
                .bits_per_elem,
            plan.terminal_state().log_basis
        );

        let commitments = [commitment];
        let openings = [opening];
        let opening_groups = [&openings[..]];

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"akita_e2e/adaptive-full-mixed");
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&opening_point[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
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
        let layout = Cfg::commitment_layout::<akita_types::Transparent>(nv).expect("layout");
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
        let plan = Cfg::schedule_plan::<akita_types::Transparent>(
            AkitaScheduleLookupKey::singleton(nv, nv, 1),
        )
        .expect("schedule plan")
        .expect("adaptive onehot config should expose a schedule plan");

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(nv);

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
        let verifier_setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
            std::slice::from_ref(&onehot_poly),
            &setup,
        )
        .unwrap();

        let poly_refs: [&OneHotPoly<F, D>; 1] = [&onehot_poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"akita_e2e/onehot-direct-tail");
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        assert_eq!(batched_total_fold_levels(&proof), plan.num_fold_levels());
        assert_eq!(
            proof.size(),
            plan.exact_proof_bytes,
            "planner should match the direct-tail proof size"
        );
        let mut serialized = Vec::new();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize adaptive onehot proof");
        let mut cursor = std::io::Cursor::new(serialized);
        let decoded = AkitaBatchedProof::<F>::deserialize_compressed(&mut cursor, &proof.shape())
            .expect("deserialize adaptive onehot proof");
        assert_eq!(
            decoded
                .final_witness()
                .as_packed_digits()
                .expect("current terminal witness should be packed digits")
                .bits_per_elem,
            plan.terminal_state().log_basis
        );

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"akita_e2e/onehot-direct-tail");
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "adaptive onehot direct-tail verification must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn adaptive_onehot_schedule_stays_below_basis6_in_current_range() {
    type Cfg = fp128::D64OneHot;

    for nv in 10..=120 {
        let plan = match Cfg::schedule_plan::<akita_types::Transparent>(
            AkitaScheduleLookupKey::singleton(nv, nv, 1),
        ) {
            Ok(Some(plan)) => plan,
            _ => continue,
        };
        assert!(
            plan.states().all(|state| state.log_basis < 6),
            "adaptive onehot schedule unexpectedly selected basis 6 at nv={nv}: {plan:?}"
        );
    }
}

#[test]
fn batched_onehot_same_point_round_trip() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64OneHot;
        const D: usize = Cfg::D;

        let nv = ONEHOT_TEST_NV;
        let layout =
            akita_batched_root_layout::<Cfg, akita_types::Transparent>(nv, 2).expect("layout");
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
        let pt = random_point(nv);
        let openings = [
            opening_from_poly(&poly_a, &pt, &layout),
            opening_from_poly(&poly_b, &pt, &layout),
        ];

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(nv);

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 2, 1);
        let verifier_setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let poly_group = [&poly_a, &poly_b];
        let (commitment, hint) =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(&poly_group, &setup)
                .unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"akita_e2e/batched-onehot");
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(
                &pt[..],
                &poly_group[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize batched onehot proof");
        let mut cursor = std::io::Cursor::new(serialized);
        let decoded = AkitaBatchedProof::<F>::deserialize_compressed(&mut cursor, &proof_shape)
            .expect("deserialize batched onehot proof");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"akita_e2e/batched-onehot");
        let opening_groups = [&openings[..]];
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "batched onehot verification must pass: {:?}",
            result.err()
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
            akita_batched_root_layout::<Cfg, akita_types::Transparent>(nv, 2).expect("layout");
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
        let pt = random_point(nv);
        let openings = [
            opening_from_poly(&poly_a, &pt, &layout),
            opening_from_poly(&poly_b, &pt, &layout),
        ];

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(nv);

        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 2, 1);
        let verifier_setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let poly_group = [&poly_a, &poly_b];
        let (commitment, hint) =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(&poly_group, &setup)
                .unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"akita_e2e/batched-onehot-s-claim-tamper");
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(
                &pt[..],
                &poly_group[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut malformed = proof.clone();
        malformed
            .root
            .as_fold_mut()
            .expect("batched s_claim tamper test expects a fold-rooted proof")
            .stage1
            .s_claim += F::from_canonical_u128_reduced(1);

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"akita_e2e/batched-onehot-s-claim-tamper");
        let opening_groups = [&openings[..]];
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_err(),
            "tampered batched root stage1 s_claim must be rejected"
        );
    });
}

#[test]
fn batched_onehot_4x30_keeps_folding_past_oversized_tail() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D64OneHot;
        const D: usize = Cfg::D;
        const NV: usize = 30;
        const BATCH_SIZE: usize = 4;

        let layout = akita_batched_root_layout::<Cfg, akita_types::Transparent>(NV, BATCH_SIZE)
            .expect("layout");
        let total_field = (layout.num_blocks * layout.block_len)
            .checked_mul(D)
            .expect("total field size overflow");
        let total_chunks = total_field / ONEHOT_K;
        assert_eq!(total_chunks * ONEHOT_K, total_field);

        let polys: Vec<OneHotPoly<F, D>> = (0..BATCH_SIZE)
            .map(|poly_idx| {
                let mut rng = StdRng::seed_from_u64(0x600d_f00d_1234_0000 + poly_idx as u64);
                let indices: Vec<Option<usize>> = (0..total_chunks)
                    .map(|_| Some(rng.gen_range(0..ONEHOT_K)))
                    .collect();
                OneHotPoly::<F, D>::new(ONEHOT_K, indices).unwrap()
            })
            .collect();
        let poly_refs: Vec<&OneHotPoly<F, D>> = polys.iter().collect();
        let pt = random_point(NV);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly(poly, &pt, &layout))
            .collect();

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(NV);

        let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
            NV, BATCH_SIZE, 1,
        );
        let verifier_setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(&poly_refs, &setup)
                .unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"akita_e2e/batched-onehot-4x30");
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize batched onehot proof");
        let mut cursor = std::io::Cursor::new(serialized);
        let decoded = AkitaBatchedProof::<F>::deserialize_compressed(&mut cursor, &proof_shape)
            .expect("deserialize batched onehot proof");

        assert!(
            decoded.final_witness().num_elems() <= 245_888,
            "expected byte-aware batched schedule to keep folding, got final_w with {} elems",
            decoded.final_witness().num_elems()
        );
        assert!(
            decoded.num_fold_levels() > 0,
            "test fixture must include a recursive suffix to cover truncation"
        );

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"akita_e2e/batched-onehot-4x30");
        let opening_groups = [&openings[..]];
        let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "batched onehot 4x30 verification must pass: {:?}",
            result.err()
        );

        let mut truncated = decoded.clone();
        truncated.steps.remove(0);
        let mut truncated_transcript =
            Blake2bTranscript::<F>::new(b"akita_e2e/batched-onehot-4x30");
        let truncated_result =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                &truncated,
                &verifier_setup,
                &mut truncated_transcript,
                verify_input(&pt[..], opening_groups[0], &commitments[0]),
                BasisMode::Lagrange,
            );
        assert!(
            truncated_result.is_err(),
            "proof with a truncated scheduled recursive suffix must be rejected"
        );
    });
}

#[test]
fn adaptive_full_setup_covers_planned_schedule_envelope() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = fp128::D128Full;
        const D: usize = Cfg::D;

        let nv = FULL_TEST_NV;
        let layout = Cfg::commitment_layout::<akita_types::Transparent>(nv).expect("layout");
        let setup =
            <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
        let plan = Cfg::schedule_plan::<akita_types::Transparent>(
            AkitaScheduleLookupKey::singleton(nv, nv, 1),
        )
        .expect("schedule plan")
        .expect("adaptive full config should expose a schedule plan");

        let mut max_inner = layout.inner_width();
        let mut max_outer = layout.outer_width();
        let mut max_d_width = layout.d_matrix_width();

        for state in plan.states().skip(1) {
            let level_inputs = AkitaScheduleInputs {
                max_num_vars: nv,
                level: state.level,
                current_w_len: state.current_w_len,
            };
            let params = Cfg::level_params_with_log_basis(
                level_inputs,
                Cfg::log_basis_at_level(level_inputs),
            );
            let recursive_layout = akita_types::recursive_level_layout_from_params(
                &params,
                state.current_w_len,
                Cfg::decomposition(),
            )
            .expect("recursive layout");
            max_inner = max_inner.max(recursive_layout.inner_width());
            max_outer = max_outer.max(recursive_layout.outer_width());
            max_d_width = max_d_width.max(recursive_layout.d_matrix_width());
        }

        let envelope = Cfg::envelope(nv);
        let total = setup.expanded.shared_matrix.total_ring_elements_at::<D>();
        assert!(total >= envelope.max_n_a * max_inner);
        assert!(total >= envelope.max_n_b * max_outer);
        assert!(total >= envelope.max_n_d * max_d_width);
    });
}

#[test]
fn adaptive_schedule_key_changes_when_schedule_changes() {
    type Cfg = fp128::D128Full;

    let mut distinct = std::collections::BTreeMap::new();
    for nv in 10..=18 {
        distinct.insert(
            Cfg::schedule_key(AkitaScheduleLookupKey::singleton(nv, nv, 1)),
            nv,
        );
    }

    assert!(
        distinct.len() >= 2,
        "adaptive schedule key should distinguish at least two nv-dependent schedules"
    );
}
