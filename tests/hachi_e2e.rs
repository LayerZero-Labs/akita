#![allow(missing_docs)]

use hachi_pcs::algebra::Fp128;
use hachi_pcs::protocol::commitment::{Fp128FullCommitmentConfig, Fp128OneHotCommitmentConfig};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps, OneHotPoly};
use hachi_pcs::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
};
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{BasisMode, CanonicalField, CommitmentScheme, Transcript};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::{Mutex, Once};
use std::time::Instant;

type F = Fp128<0xfffffffffffffffffffffffffffffeed>;
// Keep the default e2e tests small enough for `cargo test`; the larger nv=25
// workloads remain covered by `benches/hachi_e2e.rs`, while still triggering
// the standard Labrador handoff path.
const FULL_TEST_NV: usize = 14;
// The one-hot witness grows much faster than the dense path, so use a smaller
// default size here while still exercising the standard Labrador handoff.
const ONEHOT_TEST_NV: usize = 15;
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

fn random_point(nv: usize) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
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

/// Remove any stale disk-persistence cache for `max_num_vars` so that a setup
/// written by a different `CommitmentConfig` doesn't get loaded by mistake.
#[cfg(feature = "disk-persistence")]
fn purge_setup_cache(max_num_vars: usize) {
    let cache_dir = std::env::var("LOCALAPPDATA")
        .map(std::path::PathBuf::from)
        .or_else(|_| {
            std::env::var("HOME").map(|home| {
                let mut p = std::path::PathBuf::from(&home);
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
        path.push(format!("hachi_{max_num_vars}.setup"));
        let _ = std::fs::remove_file(&path);
    }
}

fn opening_from_poly<const D: usize, P: HachiPolyOps<F, D>>(
    poly: &P,
    point: &[F],
    layout: &hachi_pcs::protocol::commitment::HachiCommitmentLayout,
) -> F {
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
    let v = reduce_inner_opening_to_ring_element::<F, D>(inner_point, BasisMode::Lagrange)
        .expect("inner opening point should match ring dimension");
    (y_ring * v.sigma_m1()).coefficients()[0]
}

// ---------------------------------------------------------------------------
// Dense ("full") prove/verify
// ---------------------------------------------------------------------------

#[test]
fn full_labrador_prove_verify() {
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
        let pt = random_point(FULL_TEST_NV);
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
        let tail_kind = if proof.has_labrador_tail() {
            "labrador"
        } else {
            "direct"
        };

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
            tail_kind,
            "full/nv{FULL_TEST_NV} e2e"
        );
    });
}

// ---------------------------------------------------------------------------
// One-hot prove/verify
// ---------------------------------------------------------------------------

#[test]
fn onehot_labrador_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = Fp128OneHotCommitmentConfig;
        const D: usize = Cfg::D;

        let layout = Cfg::commitment_layout(ONEHOT_TEST_NV).expect("layout");
        let total_ring = layout.num_blocks * layout.block_len;
        let onehot_k = D;

        let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
        let indices: Vec<Option<usize>> = (0..total_ring)
            .map(|_| Some(rng.gen_range(0..onehot_k)))
            .collect();

        let onehot_poly =
            OneHotPoly::<F, D>::new(onehot_k, indices.clone(), layout.r_vars, layout.m_vars)
                .unwrap();

        let pt = random_point(ONEHOT_TEST_NV);
        let expected_opening = opening_from_poly(&onehot_poly, &pt, &layout);

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(ONEHOT_TEST_NV);

        let setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(ONEHOT_TEST_NV);
        let (commitment, hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
            &onehot_poly,
            &setup,
            &layout,
        )
        .unwrap();

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"hachi_e2e");
        let prove_start = Instant::now();
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
        let prove_time = prove_start.elapsed();

        let proof_bytes = proof.size();
        assert!(proof_bytes > 0, "proof must be non-empty");
        assert!(
            !proof.levels.is_empty(),
            "proof must have at least one level"
        );
        let tail_kind = if proof.has_labrador_tail() {
            "labrador"
        } else {
            "direct"
        };

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
            tail_kind,
            "onehot/nv{ONEHOT_TEST_NV} e2e"
        );
    });
}
