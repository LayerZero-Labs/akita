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
use akita_config::CommitmentConfig;
use akita_field::FromPrimitiveInt as _;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{CommitmentProver, CommittedPolynomials, OneHotPoly};
use akita_transcript::AkitaTranscript;
use akita_types::{AkitaVerifierSetup, BasisMode};
use akita_verifier::{CommitmentVerifier, CommittedOpenings};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::time::Instant;

type Field = fp128::Field;
const D: usize = 32;
const ONEHOT_K: usize = 256;

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

/// Per-Cfg phase wall-clock + proof-size summary.
struct PhaseStats {
    setup_secs: f64,
    commit_secs: f64,
    prove_secs: f64,
    verify_mean_ms: f64,
    verify_median_ms: f64,
    verify_min_ms: f64,
    verify_max_ms: f64,
    proof_bytes: usize,
}

fn run_one_cfg<Cfg>(label: &str, nv: usize, trials: usize) -> PhaseStats
where
    Cfg: CommitmentConfig<Field = Field, ClaimField = Field, ChallengeField = Field>,
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            Field,
            D,
            ClaimField = Field,
            VerifierSetup = AkitaVerifierSetup<Field>,
            Commitment = akita_types::RingCommitment<Field, D>,
            BatchedProof = akita_types::AkitaBatchedProof<Field, Field>,
            CommitHint = akita_types::AkitaCommitmentHint<Field, D>,
        > + CommitmentVerifier<
            Field,
            D,
            ClaimField = Field,
            VerifierSetup = AkitaVerifierSetup<Field>,
            Commitment = akita_types::RingCommitment<Field, D>,
            BatchedProof = akita_types::AkitaBatchedProof<Field, Field>,
        >,
{
    type Scheme<const DD: usize, Cfg> = AkitaCommitmentScheme<DD, Cfg>;
    println!();
    println!("---- {label} ----");
    let mut rng = StdRng::seed_from_u64(0xbe0bef);

    let total_chunks = 1usize << (nv - ONEHOT_K.trailing_zeros() as usize);
    let t_indices = Instant::now();
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();
    println!(
        "[{label}] generated {total_chunks} onehot indices ({:.2}s)",
        t_indices.elapsed().as_secs_f64()
    );
    let poly = OneHotPoly::<Field, D, u8>::new(ONEHOT_K, indices.clone()).expect("onehot poly");

    let point: Vec<Field> = (0..nv)
        .map(|_| Field::from_u128(rng.gen::<u128>()))
        .collect();
    let opening = opening_from_indices(&indices, ONEHOT_K, &point);

    let t_setup = Instant::now();
    let setup = <Scheme<D, Cfg> as CommitmentProver<Field, D>>::setup_prover(nv, 1, 1);
    let verifier_setup = <Scheme<D, Cfg> as CommitmentProver<Field, D>>::setup_verifier(&setup);
    let setup_secs = t_setup.elapsed().as_secs_f64();
    println!("[{label}] setup: {:.4}s", setup_secs);

    let t_commit = Instant::now();
    let (commitment, hint) =
        <Scheme<D, Cfg> as CommitmentProver<Field, D>>::commit(std::slice::from_ref(&poly), &setup)
            .expect("commit");
    let commit_secs = t_commit.elapsed().as_secs_f64();
    println!("[{label}] commit: {:.4}s", commit_secs);

    let poly_refs = [&poly];
    let commitments = [commitment];
    let t_prove = Instant::now();
    let mut prover_transcript = AkitaTranscript::<Field>::new(b"portable_bench");
    let proof = <Scheme<D, Cfg> as CommitmentProver<Field, D>>::batched_prove(
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
        "[{label}] prove: {:.4}s (proof bytes: {})",
        prove_secs,
        proof.size()
    );

    let openings = [opening];
    let mut verify_samples: Vec<f64> = Vec::with_capacity(trials);
    for trial in 0..trials {
        let mut verifier_transcript = AkitaTranscript::<Field>::new(b"portable_bench");
        let t = Instant::now();
        let result = <Scheme<D, Cfg> as CommitmentVerifier<Field, D>>::batched_verify(
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
            panic!("[{label}] verify failed at trial {trial}: {e:#?}");
        }
        verify_samples.push(t.elapsed().as_secs_f64());
    }

    let mut sorted = verify_samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = verify_samples.iter().sum::<f64>() / verify_samples.len() as f64;
    println!(
        "[{label}] verify: mean={:.4} ms, median={:.4} ms, min={:.4} ms, max={:.4} ms",
        mean * 1000.0,
        sorted[sorted.len() / 2] * 1000.0,
        sorted[0] * 1000.0,
        sorted.last().unwrap() * 1000.0,
    );

    PhaseStats {
        setup_secs,
        commit_secs,
        prove_secs,
        verify_mean_ms: mean * 1000.0,
        verify_median_ms: sorted[sorted.len() / 2] * 1000.0,
        verify_min_ms: sorted[0] * 1000.0,
        verify_max_ms: *sorted.last().unwrap() * 1000.0,
        proof_bytes: proof.size(),
    }
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
        " Portable bench: nv={nv}, D={D}, onehot_k={ONEHOT_K}, \
         1 poly / 1 point, {trials} verify trials"
    );
    println!("=========================================================");

    // Run BOTH the legacy production preset and the new fast-verify
    // production preset on the same workload (same point, same
    // indices) so we compare apples-to-apples across phases.
    let legacy = run_one_cfg::<fp128::D32OneHot>("fp128::D32OneHot (legacy)", nv, trials);
    let fast = run_one_cfg::<fp128::D32OneHotFastVerify>("fp128::D32OneHotFastVerify", nv, trials);

    println!();
    println!("=========================================================");
    println!(" Side-by-side summary");
    println!("=========================================================");
    println!(
        "  {:<28} {:>16} {:>16} {:>16}",
        "metric", "D32OneHot", "D32OneHotFastVerify", "Δ (fast - legacy)"
    );
    let row_secs = |name: &str, l: f64, t: f64| {
        println!(
            "  {:<28} {:>13.4}s   {:>13.4}s   {:>+13.4}s ",
            name,
            l,
            t,
            t - l
        );
    };
    let row_ms = |name: &str, l: f64, t: f64| {
        println!(
            "  {:<28} {:>14.3} ms  {:>14.3} ms  {:>+13.3} ms",
            name,
            l,
            t,
            t - l
        );
    };
    let row_bytes = |name: &str, l: usize, t: usize| {
        println!(
            "  {:<28} {:>14}  B  {:>14}  B  {:>+13}  B",
            name,
            l,
            t,
            t as i64 - l as i64
        );
    };
    row_secs("setup", legacy.setup_secs, fast.setup_secs);
    row_secs("commit", legacy.commit_secs, fast.commit_secs);
    row_secs("prove", legacy.prove_secs, fast.prove_secs);
    row_ms("verify (mean)", legacy.verify_mean_ms, fast.verify_mean_ms);
    row_ms(
        "verify (median)",
        legacy.verify_median_ms,
        fast.verify_median_ms,
    );
    row_ms("verify (min)", legacy.verify_min_ms, fast.verify_min_ms);
    row_ms("verify (max)", legacy.verify_max_ms, fast.verify_max_ms);
    row_bytes("proof size", legacy.proof_bytes, fast.proof_bytes);
}
