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
use hachi_pcs::{BasisMode, CanonicalField, CommitmentScheme, FromSmallInt, HachiError, Transcript};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

type F = Fp128<0xfffffffffffffffffffffffffffffeed>;
const NV: usize = 25;
const STACK_SIZE: usize = 64 * 1024 * 1024;

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
// Config wrappers that force a specific tail mode.
//
// We control Direct vs Labrador purely through `labrador_handoff_threshold()`
// so that tests are safe to run in parallel (no process-global env vars).
// ---------------------------------------------------------------------------

macro_rules! define_tail_config {
    ($name:ident, $base:ty, $threshold:expr, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default)]
        struct $name;

        impl CommitmentConfig for $name {
            const D: usize = <$base>::D;
            const N_A: usize = <$base>::N_A;
            const N_B: usize = <$base>::N_B;
            const N_D: usize = <$base>::N_D;
            const CHALLENGE_WEIGHT: usize = <$base>::CHALLENGE_WEIGHT;

            fn decomposition() -> DecompositionParams {
                <$base>::decomposition()
            }

            fn commitment_layout(
                max_num_vars: usize,
            ) -> Result<HachiCommitmentLayout, HachiError> {
                <$base>::commitment_layout(max_num_vars)
            }

            fn labrador_handoff_threshold() -> usize {
                $threshold
            }
        }
    };
}

define_tail_config!(
    Fp128FullDirectConfig,
    Fp128FullCommitmentConfig,
    usize::MAX,
    "Full config with threshold=MAX to force Direct tail."
);
define_tail_config!(
    Fp128FullLabradorConfig,
    Fp128FullCommitmentConfig,
    0,
    "Full config with threshold=0 to force Labrador tail."
);
define_tail_config!(
    Fp128OneHotDirectConfig,
    Fp128OneHotCommitmentConfig,
    usize::MAX,
    "OneHot config with threshold=MAX to force Direct tail."
);
define_tail_config!(
    Fp128OneHotLabradorConfig,
    Fp128OneHotCommitmentConfig,
    0,
    "OneHot config with threshold=0 to force Labrador tail."
);

// ---------------------------------------------------------------------------
// Dense ("full") prove/verify
// ---------------------------------------------------------------------------

fn full_prove_verify_inner<Cfg: CommitmentConfig>(expect_labrador: bool) {
    const D: usize = Fp128FullCommitmentConfig::D;

    let layout = Cfg::commitment_layout(NV).expect("layout");

    let mut rng = StdRng::seed_from_u64(0xdead_beef);
    let decomp = Cfg::decomposition();
    let evals: Vec<F> = if decomp.log_commit_bound >= 128 {
        (0..1usize << NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect()
    } else {
        let half_bound = 1i64 << (decomp.log_commit_bound.min(62) - 1);
        (0..1usize << NV)
            .map(|_| F::from_i64(rng.gen_range(-half_bound..half_bound)))
            .collect()
    };

    let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).unwrap();
    let pt = random_point(NV);
    let expected_opening = multilinear_eval(&evals, &pt).unwrap();

    #[cfg(feature = "disk-persistence")]
    purge_setup_cache(NV);

    // --- Setup ---
    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(NV);

    // --- Commit ---
    let (commitment, hint) =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(&poly, &setup, &layout)
            .unwrap();

    // --- Prove ---
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"proof_size_test");
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

    // --- Assertions on proof structure ---
    let proof_bytes = proof.size();
    assert!(proof_bytes > 0, "proof must be non-empty");
    assert!(!proof.levels.is_empty(), "proof must have at least one level");
    assert_eq!(
        proof.has_labrador_tail(),
        expect_labrador,
        "proof tail mismatch: expected labrador={expect_labrador}, got {}",
        proof.has_labrador_tail()
    );

    // --- Verify ---
    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"proof_size_test");
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
        "verification must pass for a correct proof: {:?}",
        verify_result.err()
    );

    // --- Verify rejects wrong opening ---
    let wrong_opening = expected_opening + F::from_u64(1);
    let mut bad_transcript = Blake2bTranscript::<F>::new(b"proof_size_test");
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
    assert!(
        bad_result.is_err(),
        "verification must reject an incorrect opening"
    );

    let tail_label = if expect_labrador { "labrador" } else { "direct" };
    eprintln!(
        "[full/nv{NV}/{tail_label}] prove: {:.3}s | verify: {:.3}s | proof: {proof_bytes} bytes ({:.2} KiB) | levels: {}",
        prove_time.as_secs_f64(),
        verify_time.as_secs_f64(),
        proof_bytes as f64 / 1024.0,
        proof.levels.len(),
    );
}

// ---------------------------------------------------------------------------
// One-hot prove/verify
// ---------------------------------------------------------------------------

fn onehot_prove_verify_inner<Cfg: CommitmentConfig>(expect_labrador: bool) {
    const D: usize = Fp128OneHotCommitmentConfig::D;

    let layout = Cfg::commitment_layout(NV).expect("layout");
    let total_ring = layout.num_blocks * layout.block_len;
    let onehot_k = D;

    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let indices: Vec<Option<usize>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..onehot_k)))
        .collect();

    let onehot_poly =
        OneHotPoly::<F, D>::new(onehot_k, indices.clone(), layout.r_vars, layout.m_vars).unwrap();

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

    // --- Setup ---
    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(NV);

    // --- Commit ---
    let (commitment, hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
        &onehot_poly,
        &setup,
        &layout,
    )
    .unwrap();

    // --- Prove ---
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"proof_size_test");
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

    // --- Assertions on proof structure ---
    let proof_bytes = proof.size();
    assert!(proof_bytes > 0, "proof must be non-empty");
    assert!(!proof.levels.is_empty(), "proof must have at least one level");
    assert_eq!(
        proof.has_labrador_tail(),
        expect_labrador,
        "proof tail mismatch: expected labrador={expect_labrador}, got {}",
        proof.has_labrador_tail()
    );

    // --- Verify ---
    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"proof_size_test");
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
        "verification must pass for a correct proof: {:?}",
        verify_result.err()
    );

    // --- Verify rejects wrong opening ---
    let wrong_opening = expected_opening + F::from_u64(1);
    let mut bad_transcript = Blake2bTranscript::<F>::new(b"proof_size_test");
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
    assert!(
        bad_result.is_err(),
        "verification must reject an incorrect opening"
    );

    let tail_label = if expect_labrador { "labrador" } else { "direct" };
    eprintln!(
        "[onehot/nv{NV}/{tail_label}] prove: {:.3}s | verify: {:.3}s | proof: {proof_bytes} bytes ({:.2} KiB) | levels: {}",
        prove_time.as_secs_f64(),
        verify_time.as_secs_f64(),
        proof_bytes as f64 / 1024.0,
        proof.levels.len(),
    );
}

// ---------------------------------------------------------------------------
// Direct tail tests (threshold=MAX configs guarantee direct witness path)
// ---------------------------------------------------------------------------

#[test]
fn full_nv25_prove_verify_direct() {
    run_on_large_stack(|| {
        full_prove_verify_inner::<Fp128FullDirectConfig>(false);
    });
}

#[test]
fn onehot_nv25_prove_verify_direct() {
    run_on_large_stack(|| {
        onehot_prove_verify_inner::<Fp128OneHotDirectConfig>(false);
    });
}

// ---------------------------------------------------------------------------
// Labrador tail tests (threshold=0 configs guarantee handoff to Labrador)
// ---------------------------------------------------------------------------

#[test]
fn full_nv25_prove_verify_labrador() {
    run_on_large_stack(|| {
        full_prove_verify_inner::<Fp128FullLabradorConfig>(true);
    });
}

#[test]
fn onehot_nv25_prove_verify_labrador() {
    run_on_large_stack(|| {
        onehot_prove_verify_inner::<Fp128OneHotLabradorConfig>(true);
    });
}
