#![allow(missing_docs)]

use hachi_pcs::algebra::poly::multilinear_eval;
use hachi_pcs::algebra::Fp128;
use hachi_pcs::protocol::commitment::{
    Fp128FullCommitmentConfig, Fp128OneHotCommitmentConfig,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, OneHotPoly};
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{BasisMode, CanonicalField, CommitmentScheme, FromSmallInt, Transcript};
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

#[test]
fn full_nv25_prove_verify_and_proof_size() {
    run_on_large_stack(full_nv25_inner);
}

fn full_nv25_inner() {
    type Cfg = Fp128FullCommitmentConfig;
    const D: usize = Cfg::D;

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
    let opening = multilinear_eval(&evals, &pt).unwrap();

    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(NV);
    let (commitment, hint) =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(&poly, &setup, &layout)
            .unwrap();

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

    let proof_bytes = proof.size();

    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"proof_size_test");
    let verify_start = Instant::now();
    <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        &pt,
        &opening,
        &commitment,
        BasisMode::Lagrange,
        &layout,
    )
    .unwrap();
    let verify_time = verify_start.elapsed();

    eprintln!(
        "[full/nv{NV}] prove: {:.3}s | verify: {:.3}s | proof size: {proof_bytes} bytes ({:.2} KiB)",
        prove_time.as_secs_f64(),
        verify_time.as_secs_f64(),
        proof_bytes as f64 / 1024.0,
    );
}

#[test]
fn onehot_nv25_prove_verify_and_proof_size() {
    run_on_large_stack(onehot_nv25_inner);
}

fn onehot_nv25_inner() {
    type Cfg = Fp128OneHotCommitmentConfig;
    const D: usize = Cfg::D;

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
    let opening = multilinear_eval(&dense_evals, &pt).unwrap();

    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(NV);
    let (commitment, hint) =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
            &onehot_poly,
            &setup,
            &layout,
        )
        .unwrap();

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

    let proof_bytes = proof.size();

    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"proof_size_test");
    let verify_start = Instant::now();
    <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        &pt,
        &opening,
        &commitment,
        BasisMode::Lagrange,
        &layout,
    )
    .unwrap();
    let verify_time = verify_start.elapsed();

    eprintln!(
        "[onehot/nv{NV}] prove: {:.3}s | verify: {:.3}s | proof size: {proof_bytes} bytes ({:.2} KiB)",
        prove_time.as_secs_f64(),
        verify_time.as_secs_f64(),
        proof_bytes as f64 / 1024.0,
    );
}
