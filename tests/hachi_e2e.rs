#![allow(missing_docs)]

use hachi_pcs::algebra::{Prime128Offset275, Prime128Offset5823};
use hachi_pcs::protocol::commitment::{
    hachi_recursive_level_layout_from_params, Fp128BoundedCommitmentConfig,
    Fp128D16FullCommitmentConfig, Fp128D32FullCommitmentConfig, Fp128FullCommitmentConfig,
    Fp128OneHotCommitmentConfig, HachiCommitmentLayout, HachiScheduleInputs,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps, OneHotPoly};
use hachi_pcs::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
};
use hachi_pcs::protocol::proof::HachiProof;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{
    BasisMode, CanonicalField, CommitmentScheme, HachiDeserialize, HachiSerialize, Transcript,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "disk-persistence")]
use std::path::PathBuf;
use std::sync::{Mutex, Once};
use std::time::Instant;

type F = Prime128Offset5823;
type FSmall = Prime128Offset275;
const ONEHOT_K: usize = 256;
const FULL_TEST_NV: usize = 14;
const ONEHOT_TEST_NV: usize = 15;
const BASIS2_TEST_NV: usize = 12;
const D32_TEST_NV: usize = 12;
const D16_TEST_NV: usize = 10;
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

type DenseBasis2Fixture<FField> = (
    <HachiCommitmentScheme<128, Fp128BoundedCommitmentConfig<128, 2, 2>> as CommitmentScheme<
        FField,
        128,
    >>::VerifierSetup,
    <HachiCommitmentScheme<128, Fp128BoundedCommitmentConfig<128, 2, 2>> as CommitmentScheme<
        FField,
        128,
    >>::Commitment,
    <HachiCommitmentScheme<128, Fp128BoundedCommitmentConfig<128, 2, 2>> as CommitmentScheme<
        FField,
        128,
    >>::Proof,
    Vec<FField>,
    FField,
    HachiCommitmentLayout,
);

type DenseFixture<FField, const D: usize, Cfg> = (
    <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<FField, D>>::VerifierSetup,
    <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<FField, D>>::Commitment,
    <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<FField, D>>::Proof,
    Vec<FField>,
    FField,
    HachiCommitmentLayout,
);

fn make_dense_basis2_fixture<FField: CanonicalField + 'static>(
    nv: usize,
    transcript_label: &'static [u8],
) -> DenseBasis2Fixture<FField>
where
    HachiCommitmentScheme<128, Fp128BoundedCommitmentConfig<128, 2, 2>>:
        CommitmentScheme<FField, 128>,
{
    type Cfg = Fp128BoundedCommitmentConfig<128, 2, 2>;
    const D: usize = Cfg::D;
    let layout = Cfg::commitment_layout(nv).expect("layout");

    let mut rng = StdRng::seed_from_u64(0x1234_5678);
    let evals: Vec<FField> = (0..1usize << nv)
        .map(|_| FField::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();

    let poly = DensePoly::<FField, D>::from_field_evals(nv, &evals).unwrap();
    let pt = random_point::<FField>(nv);
    let expected_opening = opening_from_poly(&poly, &pt, &layout);

    #[cfg(feature = "disk-persistence")]
    purge_setup_cache(nv);

    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<FField, D>>::setup_prover(nv);
    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<FField, D>>::setup_verifier(&setup);
    let (commitment, hint) =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<FField, D>>::commit(
            &poly, &setup, &layout,
        )
        .unwrap();

    let mut prover_transcript = Blake2bTranscript::<FField>::new(transcript_label);
    let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<FField, D>>::prove(
        &setup,
        &poly,
        &pt,
        hint,
        &mut prover_transcript,
        &commitment,
        BasisMode::Lagrange,
        &layout,
    )
    .unwrap();

    (
        verifier_setup,
        commitment,
        proof,
        pt,
        expected_opening,
        layout,
    )
}

fn make_dense_fixture<FField: CanonicalField + 'static, const D: usize, Cfg: CommitmentConfig>(
    nv: usize,
    transcript_label: &'static [u8],
) -> DenseFixture<FField, D, Cfg>
where
    HachiCommitmentScheme<D, Cfg>: CommitmentScheme<FField, D>,
{
    let layout = Cfg::commitment_layout(nv).expect("layout");

    let mut rng = StdRng::seed_from_u64(0x0ddc_0ffe_e123_4567);
    let evals: Vec<FField> = (0..1usize << nv)
        .map(|_| FField::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();

    let poly = DensePoly::<FField, D>::from_field_evals(nv, &evals).unwrap();
    let pt = random_point::<FField>(nv);
    let expected_opening = opening_from_poly(&poly, &pt, &layout);

    #[cfg(feature = "disk-persistence")]
    purge_setup_cache(nv);

    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<FField, D>>::setup_prover(nv);
    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<FField, D>>::setup_verifier(&setup);
    let (commitment, hint) =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<FField, D>>::commit(
            &poly, &setup, &layout,
        )
        .unwrap();

    let mut prover_transcript = Blake2bTranscript::<FField>::new(transcript_label);
    let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<FField, D>>::prove(
        &setup,
        &poly,
        &pt,
        hint,
        &mut prover_transcript,
        &commitment,
        BasisMode::Lagrange,
        &layout,
    )
    .unwrap();

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
        path.push("hachi");
        if let Ok(entries) = std::fs::read_dir(&path) {
            let needle = format!("_nv{max_num_vars}.setup");
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("hachi_") && name.ends_with(&needle))
                {
                    let _ = std::fs::remove_file(entry_path);
                }
            }
        }
    }
}

fn opening_from_poly<FField: CanonicalField, const D: usize, P: HachiPolyOps<FField, D>>(
    poly: &P,
    point: &[FField],
    layout: &HachiCommitmentLayout,
) -> FField {
    let alpha_bits = D.trailing_zeros() as usize;
    assert_eq!(point.len(), alpha_bits + layout.m_vars + layout.r_vars);

    let inner_point = &point[..alpha_bits];
    let reduced_point = &point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
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
        type Cfg = Fp128FullCommitmentConfig;
        const D: usize = Cfg::D;

        let layout = Cfg::commitment_layout(FULL_TEST_NV).expect("layout");

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let evals: Vec<F> = (0..1usize << FULL_TEST_NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let poly = DensePoly::<F, D>::from_field_evals(FULL_TEST_NV, &evals).unwrap();
        let pt = random_point::<F>(FULL_TEST_NV);
        let expected_opening = opening_from_poly(&poly, &pt, &layout);

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(FULL_TEST_NV);

        let setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(FULL_TEST_NV);
        let (commitment, hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
            &poly, &setup, &layout,
        )
        .unwrap();

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"hachi_e2e");
        let prove_start = Instant::now();
        let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::prove(
            &setup,
            &poly,
            &pt,
            hint,
            &mut prover_transcript,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        )
        .unwrap();
        let prove_time = prove_start.elapsed();

        let proof_bytes = proof.size();
        assert!(proof_bytes > 0, "proof must be non-empty");
        assert!(
            !proof.levels.is_empty(),
            "proof must have at least one level"
        );
        let plan = Cfg::schedule_plan(FULL_TEST_NV)
            .expect("schedule plan")
            .expect("adaptive full config should expose a schedule plan");
        assert_eq!(proof.levels.len(), plan.levels.len());

        let verifier_setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"hachi_e2e");
        let verify_start = Instant::now();
        let verify_result = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &pt,
            &expected_opening,
            &commitment,
            BasisMode::Lagrange,
            &layout,
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
            levels = proof.levels.len(),
            "full-d128/nv{FULL_TEST_NV} e2e"
        );
    });
}

#[test]
fn full_d128_basis2_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = Fp128BoundedCommitmentConfig<128, 2, 2>;
        const D: usize = Cfg::D;

        let (verifier_setup, commitment, proof, opening_point, opening, layout) =
            make_dense_basis2_fixture::<F>(BASIS2_TEST_NV, b"hachi_e2e/basis2");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"hachi_e2e/basis2");
        let result = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        );

        assert!(
            result.is_ok(),
            "basis-2 verification must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn full_d32_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = Fp128D32FullCommitmentConfig;
        const D: usize = Cfg::D;

        let plan = Cfg::schedule_plan(D32_TEST_NV)
            .expect("schedule plan")
            .expect("adaptive D32 config should expose a schedule plan");
        let (verifier_setup, commitment, proof, opening_point, opening, layout) =
            make_dense_fixture::<FSmall, D, Cfg>(D32_TEST_NV, b"hachi_e2e/full-d32");

        assert_eq!(proof.levels.len(), plan.levels.len());

        let mut verifier_transcript = Blake2bTranscript::<FSmall>::new(b"hachi_e2e/full-d32");
        let result = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<FSmall, D>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        );

        assert!(
            result.is_ok(),
            "D32 verification must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn full_d16_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = Fp128D16FullCommitmentConfig;
        const D: usize = Cfg::D;

        let plan = Cfg::schedule_plan(D16_TEST_NV)
            .expect("schedule plan")
            .expect("adaptive D16 config should expose a schedule plan");
        let (verifier_setup, commitment, proof, opening_point, opening, layout) =
            make_dense_fixture::<FSmall, D, Cfg>(D16_TEST_NV, b"hachi_e2e/full-d16");

        assert_eq!(proof.levels.len(), plan.levels.len());

        let mut verifier_transcript = Blake2bTranscript::<FSmall>::new(b"hachi_e2e/full-d16");
        let result = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<FSmall, D>>::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        );

        assert!(
            result.is_ok(),
            "D16 verification must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn full_d128_basis2_rejects_tampered_stage1_sumcheck() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = Fp128BoundedCommitmentConfig<128, 2, 2>;
        const D: usize = Cfg::D;

        let (verifier_setup, commitment, proof, opening_point, opening, layout) =
            make_dense_basis2_fixture::<F>(BASIS2_TEST_NV, b"hachi_e2e/basis2-tamper");
        let mut malformed = proof.clone();
        let stage1_sumcheck = &mut malformed
            .levels
            .iter_mut()
            .next()
            .expect("basis-2 proof should contain at least one level")
            .stage1
            .sumcheck;
        let round0 = stage1_sumcheck
            .round_polys
            .first_mut()
            .expect("stage1 sumcheck should contain at least one round");
        let coeff0 = round0
            .coeffs_except_linear_term
            .first_mut()
            .expect("stage1 round polynomial should contain a constant term");
        *coeff0 += F::from_canonical_u128_reduced(1);

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"hachi_e2e/basis2-tamper");
        let result = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        );

        assert!(result.is_err(), "tampered stage1 sumcheck must be rejected");
    });
}

#[test]
fn full_d128_adaptive_mixed_basis_roundtrip_and_serialization() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = Fp128FullCommitmentConfig;
        const D: usize = Cfg::D;

        let nv = FULL_TEST_NV;
        let plan = Cfg::schedule_plan(nv)
            .expect("schedule plan")
            .expect("adaptive full config should expose a schedule plan");
        let (verifier_setup, commitment, proof, opening_point, opening, layout) =
            make_dense_fixture::<F, D, Cfg>(nv, b"hachi_e2e/adaptive-full-mixed");

        assert_eq!(proof.levels.len(), plan.levels.len());

        let mut proof_bytes = Vec::new();
        proof
            .serialize_compressed(&mut proof_bytes)
            .expect("serialize adaptive proof");
        let mut cursor = std::io::Cursor::new(proof_bytes);
        let decoded = HachiProof::<F>::deserialize_compressed(&mut cursor, &plan.to_proof_shape())
            .expect("deserialize adaptive proof");
        assert_eq!(decoded, proof);

        assert_eq!(
            decoded.tail.direct.bits_per_elem,
            plan.terminal_state().log_basis
        );

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"hachi_e2e/adaptive-full-mixed");
        let result = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            &opening_point,
            &opening,
            &commitment,
            BasisMode::Lagrange,
            &layout,
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
        type Cfg = Fp128OneHotCommitmentConfig;
        const D: usize = Cfg::D;

        let nv = ONEHOT_TEST_NV;
        let layout = Cfg::commitment_layout(nv).expect("layout");
        let total_field = (layout.num_blocks * layout.block_len)
            .checked_mul(D)
            .expect("total field size overflow");
        let total_chunks = total_field / ONEHOT_K;
        assert_eq!(total_chunks * ONEHOT_K, total_field);

        let mut rng = StdRng::seed_from_u64(0x1234_abcd);
        let indices: Vec<Option<usize>> = (0..total_chunks)
            .map(|_| Some(rng.gen_range(0..ONEHOT_K)))
            .collect();
        let onehot_poly =
            OneHotPoly::<F, D>::new(ONEHOT_K, indices, layout.r_vars, layout.m_vars).unwrap();
        let pt = random_point::<F>(nv);
        let expected_opening = opening_from_poly(&onehot_poly, &pt, &layout);
        let plan = Cfg::schedule_plan(nv)
            .expect("schedule plan")
            .expect("adaptive onehot config should expose a schedule plan");

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(nv);

        let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(nv);
        let verifier_setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
        let (commitment, hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
            &onehot_poly,
            &setup,
            &layout,
        )
        .unwrap();

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"hachi_e2e/onehot-direct-tail");
        let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::prove(
            &setup,
            &onehot_poly,
            &pt,
            hint,
            &mut prover_transcript,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        )
        .unwrap();

        assert_eq!(proof.levels.len(), plan.levels.len());
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
        let decoded = HachiProof::<F>::deserialize_compressed(&mut cursor, &plan.to_proof_shape())
            .expect("deserialize adaptive onehot proof");
        assert_eq!(
            decoded.tail.direct.bits_per_elem,
            plan.terminal_state().log_basis
        );

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"hachi_e2e/onehot-direct-tail");
        let result = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            &pt,
            &expected_opening,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        );
        assert!(
            result.is_ok(),
            "adaptive onehot direct-tail verification must pass: {:?}",
            result.err()
        );
    });
}

#[test]
fn adaptive_full_setup_covers_planned_schedule_envelope() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = Fp128FullCommitmentConfig;
        const D: usize = Cfg::D;

        let nv = FULL_TEST_NV;
        let layout = Cfg::commitment_layout(nv).expect("layout");
        let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(nv);
        let plan = Cfg::schedule_plan(nv)
            .expect("schedule plan")
            .expect("adaptive full config should expose a schedule plan");

        let mut max_inner = layout.inner_width;
        let mut max_outer = layout.outer_width;
        let mut max_d_width = layout.d_matrix_width;

        for state in plan.states.iter().skip(1) {
            let params = Cfg::level_params(HachiScheduleInputs {
                max_num_vars: nv,
                level: state.level,
                current_w_len: state.current_w_len,
            });
            let recursive_layout =
                hachi_recursive_level_layout_from_params::<Cfg>(&params, state.current_w_len)
                    .expect("recursive layout");
            max_inner = max_inner.max(recursive_layout.inner_width);
            max_outer = max_outer.max(recursive_layout.outer_width);
            max_d_width = max_d_width.max(recursive_layout.d_matrix_width);
        }

        assert!(setup.expanded.shared_matrix.first_row_len::<D>() >= max_inner);
        assert!(setup.expanded.shared_matrix.first_row_len::<D>() >= max_outer);
        assert!(setup.expanded.shared_matrix.first_row_len::<D>() >= max_d_width);
    });
}

#[test]
fn adaptive_schedule_key_changes_when_schedule_changes() {
    type Cfg = Fp128FullCommitmentConfig;

    let mut distinct = std::collections::BTreeMap::new();
    for nv in 10..=18 {
        distinct.insert(Cfg::schedule_key(nv), nv);
    }

    assert!(
        distinct.len() >= 2,
        "adaptive schedule key should distinguish at least two nv-dependent schedules"
    );
}
