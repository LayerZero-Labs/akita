#![allow(missing_docs)]

use hachi_pcs::algebra::poly::multilinear_eval;
use hachi_pcs::algebra::Fp128;
use hachi_pcs::protocol::commitment::{
    DecompositionParams, Fp128FullCommitmentConfig, Fp128OneHotCommitmentConfig,
    HachiCommitmentLayout,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, OneHotPoly};
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{
    BasisMode, CanonicalField, CommitmentScheme, FromSmallInt, HachiError, Transcript,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Once;
use std::time::Instant;

type F = Fp128<0xfffffffffffffffffffffffffffffeed>;
const NV: usize = 25;
const STACK_SIZE: usize = 256 * 1024 * 1024;

static INIT_RAYON: Once = Once::new();

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

// ---------------------------------------------------------------------------
// Configs that halve D down to 64 for Labrador handoff.
//
// The Labrador challenge sampler requires D <= 256, and the handoff guard
// requires D <= 64. These configs use d_at_level to halve the ring dimension
// (512 -> 256 -> 128 -> 64) while keeping all other parameters from the
// standard Fp128 configs.
// ---------------------------------------------------------------------------

fn halving_d(level: usize) -> usize {
    match level {
        0 => 512,
        1 => 256,
        2 => 128,
        _ => 64,
    }
}

fn halving_n_a(level: usize) -> usize {
    match level {
        0 => 1,
        1 => 2,
        2 => 4,
        _ => 8,
    }
}

fn halving_challenge_weight(d: usize) -> usize {
    match d {
        512 => 19,
        256 => 23,
        128 => 31,
        64 | 0 => 31,
        _ => 19,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FullLabradorConfig;

impl CommitmentConfig for FullLabradorConfig {
    const D: usize = Fp128FullCommitmentConfig::D;
    const N_A: usize = Fp128FullCommitmentConfig::N_A;
    const N_B: usize = Fp128FullCommitmentConfig::N_B;
    const N_D: usize = Fp128FullCommitmentConfig::N_D;
    const CHALLENGE_WEIGHT: usize = Fp128FullCommitmentConfig::CHALLENGE_WEIGHT;

    fn decomposition() -> DecompositionParams {
        Fp128FullCommitmentConfig::decomposition()
    }

    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        Fp128FullCommitmentConfig::commitment_layout(max_num_vars)
    }

    fn d_at_level(level: usize, _w_num_vars: usize) -> usize {
        halving_d(level)
    }

    fn n_a_at_level(level: usize) -> usize {
        halving_n_a(level)
    }

    fn challenge_weight_for_ring_dim(d: usize) -> usize {
        halving_challenge_weight(d)
    }

    fn labrador_handoff_threshold() -> usize {
        0
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct OneHotLabradorConfig;

impl CommitmentConfig for OneHotLabradorConfig {
    const D: usize = Fp128OneHotCommitmentConfig::D;
    const N_A: usize = Fp128OneHotCommitmentConfig::N_A;
    const N_B: usize = Fp128OneHotCommitmentConfig::N_B;
    const N_D: usize = Fp128OneHotCommitmentConfig::N_D;
    const CHALLENGE_WEIGHT: usize = Fp128OneHotCommitmentConfig::CHALLENGE_WEIGHT;

    fn decomposition() -> DecompositionParams {
        Fp128OneHotCommitmentConfig::decomposition()
    }

    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        Fp128OneHotCommitmentConfig::commitment_layout(max_num_vars)
    }

    fn d_at_level(level: usize, _w_num_vars: usize) -> usize {
        halving_d(level)
    }

    fn n_a_at_level(level: usize) -> usize {
        halving_n_a(level)
    }

    fn challenge_weight_for_ring_dim(d: usize) -> usize {
        halving_challenge_weight(d)
    }

    fn labrador_handoff_threshold() -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Dense ("full") prove/verify
// ---------------------------------------------------------------------------

#[test]
fn full_nv25_prove_verify() {
    init_rayon_pool();
    run_on_large_stack(|| {
        type Cfg = FullLabradorConfig;
        const D: usize = Cfg::D;

        let layout = Cfg::commitment_layout(NV).expect("layout");

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let evals: Vec<F> = (0..1usize << NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();

        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).unwrap();
        let pt = random_point(NV);
        let expected_opening = multilinear_eval(&evals, &pt).unwrap();

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(NV);

        let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(NV);
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
        assert!(
            proof.has_labrador_tail(),
            "expected Labrador tail, got direct"
        );

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

        let wrong_opening = expected_opening + F::from_u64(1);
        let mut bad_transcript = Blake2bTranscript::<F>::new(b"hachi_e2e");
        let bad_result = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
            &proof,
            &verifier_setup,
            &mut bad_transcript,
            &pt,
            &wrong_opening,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        );
        assert!(bad_result.is_err(), "must reject incorrect opening");

        eprintln!(
            "[full/nv{NV}] prove: {:.3}s | verify: {:.3}s | proof: {proof_bytes} bytes ({:.2} KiB) | levels: {}",
            prove_time.as_secs_f64(),
            verify_time.as_secs_f64(),
            proof_bytes as f64 / 1024.0,
            proof.levels.len(),
        );
    });
}

// ---------------------------------------------------------------------------
// One-hot prove/verify
// ---------------------------------------------------------------------------

#[test]
fn onehot_nv25_prove_verify() {
    init_rayon_pool();
    run_on_large_stack(|| {
        type Cfg = OneHotLabradorConfig;
        const D: usize = Cfg::D;

        let layout = Cfg::commitment_layout(NV).expect("layout");
        let total_ring = layout.num_blocks * layout.block_len;
        let onehot_k = D;

        let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
        let indices: Vec<Option<usize>> = (0..total_ring)
            .map(|_| Some(rng.gen_range(0..onehot_k)))
            .collect();

        let onehot_poly =
            OneHotPoly::<F, D>::new(onehot_k, indices.clone(), layout.r_vars, layout.m_vars)
                .unwrap();

        let dense_evals: Vec<F> = {
            let mut evals = vec![F::from_u64(0); total_ring * onehot_k];
            for (ci, opt_idx) in indices.iter().enumerate() {
                if let Some(idx) = opt_idx {
                    evals[ci * onehot_k + idx] = F::from_u64(1);
                }
            }
            evals
        };
        let dense_poly = DensePoly::<F, D>::from_field_evals(NV, &dense_evals).unwrap();
        let pt = random_point(NV);
        let expected_opening = multilinear_eval(&dense_evals, &pt).unwrap();

        #[cfg(feature = "disk-persistence")]
        purge_setup_cache(NV);

        let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(NV);
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
            &dense_poly,
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
        assert!(
            proof.has_labrador_tail(),
            "expected Labrador tail, got direct"
        );

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

        let wrong_opening = expected_opening + F::from_u64(1);
        let mut bad_transcript = Blake2bTranscript::<F>::new(b"hachi_e2e");
        let bad_result = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
            &proof,
            &verifier_setup,
            &mut bad_transcript,
            &pt,
            &wrong_opening,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        );
        assert!(bad_result.is_err(), "must reject incorrect opening");

        eprintln!(
            "[onehot/nv{NV}] prove: {:.3}s | verify: {:.3}s | proof: {proof_bytes} bytes ({:.2} KiB) | levels: {}",
            prove_time.as_secs_f64(),
            verify_time.as_secs_f64(),
            proof_bytes as f64 / 1024.0,
            proof.levels.len(),
        );
    });
}
