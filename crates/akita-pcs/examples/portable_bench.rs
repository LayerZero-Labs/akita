//! Cross-branch portable benchmark for `setup_prover` + `commit` +
//! `batched_prove` + `batched_verify` on a one-hot polynomial under the
//! production `fp128::D32OneHot` config.
//!
//! Uses only public APIs that exist on both `main` and
//! `feat/tier-commit`, so this same file can be dropped into either
//! branch's `crates/akita-pcs/examples/` directory and run the
//! identical workload. On `feat/tier-commit` the same shape can be
//! compared against `tiered_bench` (which adds the tiered `f=3`
//! variant).
//!
//! Workload: `nv = 32`, `D = 32`, one-hot poly with `onehot_k = 256`,
//! single polynomial, single opening point.
//!
//! Env knobs:
//! - `AKITA_BENCH_NV` (default 32)
//! - `AKITA_BENCH_TRIALS` (default 20) — number of verify trials
//!
//! Output: per-phase wall-clock (setup, commit, prove, verify trials).

#![allow(missing_docs)]

use akita_algebra::offset_eq::eq_eval_at_index;
use akita_config::proof_optimized::fp128;
use akita_field::FromPrimitiveInt as _;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{CommitmentProver, CommittedPolynomials, OneHotPoly};
use akita_transcript::Blake2bTranscript;
use akita_types::BasisMode;
use akita_verifier::{CommitmentVerifier, CommittedOpenings};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::time::Instant;

type Cfg = fp128::D32OneHot;
type Field = fp128::Field;
const D: usize = 32;
const ONEHOT_K: usize = 256;
type Scheme = AkitaCommitmentScheme<D, Cfg>;

fn opening_from_indices(indices: &[Option<u8>], onehot_k: usize, point: &[Field]) -> Field {
    // OneHot encoding: `evals[chunk * onehot_k + idx] = 1`, all else 0.
    // `<weights, evals> = Σ_chunk lagrange_weight(point, chunk·k + idx)`.
    // Using `eq_eval_at_index` per non-empty chunk avoids materialising
    // the 2^nv weight table (which is 64 GiB at nv = 32).
    let mut acc = Field::zero();
    for (chunk, &maybe_idx) in indices.iter().enumerate() {
        if let Some(idx) = maybe_idx {
            let flat_idx = chunk * onehot_k + idx as usize;
            acc += eq_eval_at_index(point, flat_idx);
        }
    }
    acc
}

fn main() {
    let nv: usize = env::var("AKITA_BENCH_NV")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let trials: usize = env::var("AKITA_BENCH_TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    println!("=========================================================");
    println!(
        " Portable bench: fp128::D32OneHot, nv={nv}, D={D}, onehot_k={ONEHOT_K}, \
         1 poly, 1 point, {trials} verify trials"
    );
    println!("=========================================================");

    let mut rng = StdRng::seed_from_u64(0xbe0bef);

    // One-hot indices for `2^nv / onehot_k` chunks.
    let total_chunks = 1usize << (nv - ONEHOT_K.trailing_zeros() as usize);
    let t_indices = Instant::now();
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();
    println!(
        "generated {total_chunks} onehot indices ({:.2}s)",
        t_indices.elapsed().as_secs_f64()
    );
    let poly = OneHotPoly::<Field, D, u8>::new(ONEHOT_K, indices.clone()).expect("onehot poly");

    let point: Vec<Field> = (0..nv).map(|_| Field::from_u128(rng.gen::<u128>())).collect();
    let t_open = Instant::now();
    let opening = opening_from_indices(&indices, ONEHOT_K, &point);
    println!(
        "opening built ({:.2}s)",
        t_open.elapsed().as_secs_f64()
    );

    // Setup.
    let t_setup = Instant::now();
    let setup = <Scheme as CommitmentProver<Field, D>>::setup_prover(nv, 1, 1);
    let verifier_setup = <Scheme as CommitmentProver<Field, D>>::setup_verifier(&setup);
    let setup_secs = t_setup.elapsed().as_secs_f64();
    println!("setup_prover + setup_verifier: {:.4}s", setup_secs);

    // Commit.
    let t_commit = Instant::now();
    let (commitment, hint) =
        <Scheme as CommitmentProver<Field, D>>::commit(std::slice::from_ref(&poly), &setup)
            .expect("commit");
    let commit_secs = t_commit.elapsed().as_secs_f64();
    println!("commit: {:.4}s", commit_secs);

    // Prove.
    let poly_refs = [&poly];
    let commitments = [commitment];
    let t_prove = Instant::now();
    let mut prover_transcript = Blake2bTranscript::<Field>::new(b"portable_bench");
    let proof = <Scheme as CommitmentProver<Field, D>>::batched_prove(
        &setup,
        vec![(
            &point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("prove");
    let prove_secs = t_prove.elapsed().as_secs_f64();
    println!(
        "prove: {:.4}s (proof bytes: {})",
        prove_secs,
        proof.size()
    );

    // Verify N times.
    let openings = [opening];
    let mut verify_samples: Vec<f64> = Vec::with_capacity(trials);
    for trial in 0..trials {
        let mut verifier_transcript = Blake2bTranscript::<Field>::new(b"portable_bench");
        let t = Instant::now();
        let result = <Scheme as CommitmentVerifier<Field, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            vec![(
                &point[..],
                CommittedOpenings {
                    openings: &openings[..],
                    commitment: &commitments[0],
                },
            )],
            BasisMode::Lagrange,
        );
        if let Err(e) = &result {
            panic!("verify failed at trial {trial}: {e:#?}");
        }
        let elapsed = t.elapsed().as_secs_f64();
        verify_samples.push(elapsed);
        println!("  verify trial {:>2}: {:.4} ms", trial + 1, elapsed * 1000.0);
    }

    let mut sorted = verify_samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = verify_samples.iter().sum::<f64>() / verify_samples.len() as f64;
    let median = sorted[sorted.len() / 2];
    let min = sorted[0];
    let max = *sorted.last().unwrap();

    println!();
    println!("=========================================================");
    println!(" Phase wall-clock summary (mean across {trials} verify trials)");
    println!("=========================================================");
    println!(
        "  {:<10} {:>12} {:>12} {:>12} {:>12}",
        "phase", "value (s)", "value (ms)", "", ""
    );
    let print_row = |label: &str, secs: f64| {
        println!(
            "  {:<10} {:>12.4} {:>12.4}",
            label,
            secs,
            secs * 1000.0
        );
    };
    print_row("setup", setup_secs);
    print_row("commit", commit_secs);
    print_row("prove", prove_secs);
    println!();
    println!("  verify:");
    println!("    mean   = {:.4} ms", mean * 1000.0);
    println!("    median = {:.4} ms", median * 1000.0);
    println!("    min    = {:.4} ms", min * 1000.0);
    println!("    max    = {:.4} ms", max * 1000.0);
    println!();
    println!("  proof bytes: {}", proof.size());
}
