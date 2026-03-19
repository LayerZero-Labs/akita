#![allow(missing_docs)]

use hachi_pcs::algebra::{Fp128, SparseChallengeConfig};
use hachi_pcs::error::HachiError;
use hachi_pcs::protocol::commitment::{
    CommitmentEnvelope, DecompositionParams, Fp128BoundedCommitmentConfig,
    Fp128FullCommitmentConfig, Fp128OneHotCommitmentConfig, HachiCommitmentLayout,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps, OneHotPoly};
use hachi_pcs::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
};
use hachi_pcs::protocol::proof::{HachiProof, NormCheckBody};
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::{CommitmentConfig, RingCommitment};
use hachi_pcs::{BasisMode, CanonicalField, CommitmentScheme, Transcript};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "disk-persistence")]
use std::path::PathBuf;
use std::sync::{Mutex, Once};
use std::time::Instant;

type F = Fp128<0xfffffffffffffffffffffffffffffeed>;
const ONEHOT_K: usize = 256;
// Keep the default e2e tests small enough for `cargo test`; the larger nv=25
// workloads remain covered by `benches/hachi_e2e.rs`, while still triggering
// the standard Labrador handoff path.
const FULL_TEST_NV: usize = 14;
// The one-hot witness grows much faster than the dense path, so use a smaller
// default size here while still exercising the standard Labrador handoff.
const ONEHOT_TEST_NV: usize = 15;
const BASIS2_TEST_NV: usize = 12;
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

fn make_dense_basis2_fixture(
    nv: usize,
    transcript_label: &'static [u8],
) -> (
    <HachiCommitmentScheme<128, Fp128BoundedCommitmentConfig<128, 2, 2>> as CommitmentScheme<
        F,
        128,
    >>::VerifierSetup,
    RingCommitment<F, 128>,
    HachiProof<F>,
    Vec<F>,
    F,
    HachiCommitmentLayout,
) {
    type Cfg = Fp128BoundedCommitmentConfig<128, 2, 2>;
    const D: usize = Cfg::D;
    let layout = Cfg::commitment_layout(nv).expect("layout");

    let mut rng = StdRng::seed_from_u64(0x1234_5678);
    let evals: Vec<F> = (0..1usize << nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();

    let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).unwrap();
    let pt = random_point(nv);
    let expected_opening = opening_from_poly(&poly, &pt, &layout);

    #[cfg(feature = "disk-persistence")]
    purge_setup_cache(nv);

    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(nv);
    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let (commitment, hint) =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(&poly, &setup, &layout)
            .unwrap();

    let mut prover_transcript = Blake2bTranscript::<F>::new(transcript_label);
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

fn opening_from_poly<const D: usize, P: HachiPolyOps<F, D>>(
    poly: &P,
    point: &[F],
    layout: &HachiCommitmentLayout,
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

#[derive(Clone, Copy, Debug, Default)]
struct OneHotLabradorCommitmentConfig;

impl CommitmentConfig for OneHotLabradorCommitmentConfig {
    const D: usize = Fp128OneHotCommitmentConfig::D;

    fn decomposition() -> DecompositionParams {
        Fp128OneHotCommitmentConfig::decomposition()
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        Fp128OneHotCommitmentConfig::envelope(max_num_vars)
    }

    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        Fp128OneHotCommitmentConfig::commitment_layout(max_num_vars)
    }

    fn n_b_at_level(level: usize, max_num_vars: usize, current_w_len: usize) -> usize {
        Fp128OneHotCommitmentConfig::n_b_at_level(level, max_num_vars, current_w_len)
    }

    fn n_d_at_level(level: usize, max_num_vars: usize, current_w_len: usize) -> usize {
        Fp128OneHotCommitmentConfig::n_d_at_level(level, max_num_vars, current_w_len)
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Fp128OneHotCommitmentConfig::stage1_challenge_config(d)
    }

    fn w_log_basis() -> u32 {
        Fp128OneHotCommitmentConfig::w_log_basis()
    }

    fn labrador_handoff_threshold() -> usize {
        0
    }
}

#[test]
fn full_d128_labrador_prove_verify() {
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
            "full-d128/nv{FULL_TEST_NV} e2e"
        );
    });
}

#[test]
fn onehot_d64_labrador_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = OneHotLabradorCommitmentConfig;
        const D: usize = Cfg::D;

        let layout = Cfg::commitment_layout(ONEHOT_TEST_NV).expect("layout");
        let total_field = (layout.num_blocks * layout.block_len)
            .checked_mul(D)
            .expect("total field size overflow");
        let onehot_k = ONEHOT_K;
        let total_chunks = total_field / onehot_k;
        assert_eq!(total_chunks * onehot_k, total_field);

        let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
        let indices: Vec<Option<usize>> = (0..total_chunks)
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
            "onehot-d64/nv{ONEHOT_TEST_NV} e2e"
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
            make_dense_basis2_fixture(BASIS2_TEST_NV, b"hachi_e2e/basis2");

        assert!(
            proof
                .levels
                .iter()
                .any(|level| matches!(level.body, NormCheckBody::Combined { .. })),
            "basis-2 proof should exercise the combined b=4 path"
        );

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
fn full_d128_basis2_rejects_tampered_combined_sumcheck() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        type Cfg = Fp128BoundedCommitmentConfig<128, 2, 2>;
        const D: usize = Cfg::D;

        let (verifier_setup, commitment, proof, opening_point, opening, layout) =
            make_dense_basis2_fixture(BASIS2_TEST_NV, b"hachi_e2e/basis2-tamper");
        let mut malformed = proof.clone();
        let combined_level = malformed
            .levels
            .iter_mut()
            .find(|level| matches!(level.body, NormCheckBody::Combined { .. }))
            .expect("basis-2 proof should contain a combined level");
        match &mut combined_level.body {
            NormCheckBody::Combined { sumcheck, .. } => {
                let round0 = sumcheck
                    .round_polys
                    .first_mut()
                    .expect("combined sumcheck should contain at least one round");
                let coeff0 = round0
                    .coeffs_except_linear_term
                    .first_mut()
                    .expect("combined round polynomial should contain a constant term");
                *coeff0 += F::from_canonical_u128_reduced(1);
            }
            NormCheckBody::TwoStage { .. } => unreachable!("expected combined b=4 level"),
        }

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

        assert!(
            result.is_err(),
            "tampered combined sumcheck must be rejected"
        );
    });
}
