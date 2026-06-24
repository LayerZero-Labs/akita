//! Ignored microbenchmarks for the operator-norm predicate and `D=64`
//! exact-shell sampling, including the verifier-side rejection-sampling replay
//! cost. Not correctness tests; they print timing / distribution reports.
//!
//! ```text
//! cargo test -p akita-challenges --release op_norm::perf -- --ignored --nocapture
//! ```

use super::{Decision, OpNormTable};
use crate::sampler::exact_shell::sample_exact_shell_challenge;
use crate::sampler::xof::XofCursor;
use crate::{SparseChallenge, SparseChallengeConfig};
use akita_field::AkitaError;
use akita_field::Prime128OffsetA7F7;
use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
use akita_transcript::{AkitaTranscript, Transcript};
use std::hint::black_box;
use std::time::Instant;

type F = Prime128OffsetA7F7;

const D: usize = 64;
const Q: u32 = 48;
const T: u64 = 18;
const C1: usize = 31;
const C2: usize = 11;

fn build_table() -> OpNormTable {
    OpNormTable::new(D, Q, (2 * D) as u64, 64).unwrap()
}

fn decide_ch(
    table: &OpNormTable,
    challenge: &SparseChallenge,
    threshold: u64,
) -> Result<Decision, AkitaError> {
    table.decide_parts(&challenge.positions, &challenge.coeffs, threshold)
}

/// Float reference for the operator norm `gamma(c) = max_k |c(zeta_k)|`,
/// used ONLY to study the acceptance-probability vs threshold tradeoff (not
/// the protocol predicate, which stays integer-certified).
fn gamma_f64(ch: &SparseChallenge) -> f64 {
    use std::f64::consts::PI;
    let mut maxsq = 0.0f64;
    for k in 0..D / 2 {
        let base = (2 * k + 1) as f64 * PI / D as f64;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (&pos, &coeff) in ch.positions.iter().zip(ch.coeffs.iter()) {
            let theta = base * pos as f64;
            re += coeff as f64 * theta.cos();
            im += coeff as f64 * theta.sin();
        }
        let s = re * re + im * im;
        if s > maxsq {
            maxsq = s;
        }
    }
    maxsq.sqrt()
}

#[test]
#[ignore = "measurement: run with --release --ignored --nocapture"]
fn perf_gamma_distribution() {
    let mut cur = warm_cursor();
    let n: usize = 4_000_000;
    let mut gammas: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let ch = sample_exact_shell_challenge(&mut cur, D, C1, C2);
        gammas.push(gamma_f64(&ch));
    }
    gammas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |q: f64| gammas[((n as f64 * q) as usize).min(n - 1)];
    let mean: f64 = gammas.iter().sum::<f64>() / n as f64;
    println!("\n=== gamma(c) distribution, D={D} shell=({C1},{C2}), N={n} ===");
    println!("mean   = {mean:.3}");
    println!("p50    = {:.3}", pct(0.50));
    println!("p90    = {:.3}", pct(0.90));
    println!("p99    = {:.3}", pct(0.99));
    println!("p99.9  = {:.3}", pct(0.999));
    println!("p99.99 = {:.3}", pct(0.9999));
    println!("max    = {:.3}", gammas[n - 1]);
    println!(
        "(||c||_1 = {} is the trivial deterministic bound)",
        C1 + 2 * C2
    );
    println!("--- acceptance p(T) = Pr[gamma <= T] and avg candidates 1/p ---");
    for t in 14..=30u32 {
        let acc = gammas.partition_point(|&g| g <= t as f64);
        let p = acc as f64 / n as f64;
        let cand = if p > 0.0 { 1.0 / p } else { f64::INFINITY };
        println!("T={t:>2}: p={p:.5}  candidates/accept={cand:>7.3}");
    }
    println!();
}

fn warm_cursor() -> XofCursor {
    XofCursor::from_seed(&[0x42u8; 32])
}

fn time_ns(iters: u64, mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_nanos() as f64 / iters as f64
}

#[test]
#[ignore = "microbenchmark: run with --release --ignored --nocapture"]
fn perf_op_norm_d64() {
    let tbl = build_table();
    let mut cur = warm_cursor();

    // (E) one-time certified table construction.
    let build_ns = time_ns(2_000, || {
        black_box(OpNormTable::new(D, Q, (2 * D) as u64, 64).unwrap());
    });

    // (A) per-candidate sampling: decode one (31,11) shell from a warm XOF.
    for _ in 0..50_000 {
        black_box(sample_exact_shell_challenge(&mut cur, D, C1, C2));
    }
    let sample_ns = time_ns(1_000_000, || {
        black_box(sample_exact_shell_challenge(&mut cur, D, C1, C2));
    });

    // Pool of sampled challenges for decide-only timing (realistic mix of
    // accept / reject / indeterminate).
    let pool: Vec<SparseChallenge> = (0..4096)
        .map(|_| sample_exact_shell_challenge(&mut cur, D, C1, C2))
        .collect();
    let accepted: SparseChallenge = pool
        .iter()
        .find(|ch| {
            tbl.accept_strict_parts(&ch.positions, &ch.coeffs, T)
                .unwrap()
        })
        .cloned()
        .expect("some (31,11) shell accepts at T=18");

    // (B) op-norm check, production d/2 scan, averaged over the pool.
    for ch in &pool {
        black_box(decide_ch(&tbl, ch, T).unwrap());
    }
    let mut i = 0usize;
    let decide_ns = time_ns(1_000_000, || {
        let ch = &pool[i & (pool.len() - 1)];
        i += 1;
        black_box(decide_ch(&tbl, ch, T).unwrap());
    });

    // (B') worst case: an accepted challenge always scans all d/2 frequencies.
    let decide_worst_ns = time_ns(1_000_000, || {
        black_box(decide_ch(&tbl, &accepted, T).unwrap());
    });

    // (D) rejection sampling end-to-end (the verifier-side replay): draw and
    // check candidates until one is accepted.
    let n_accepted = 100_000u64;
    let (mut attempts, mut accepts, mut rejects, mut indet) = (0u64, 0u64, 0u64, 0u64);
    let start = Instant::now();
    while accepts < n_accepted {
        attempts += 1;
        let ch = sample_exact_shell_challenge(&mut cur, D, C1, C2);
        match decide_ch(&tbl, &ch, T).unwrap() {
            Decision::Accept => accepts += 1,
            Decision::Reject => rejects += 1,
            Decision::Indeterminate => indet += 1,
        }
    }
    let per_accepted_ns = start.elapsed().as_nanos() as f64 / n_accepted as f64;
    let p = n_accepted as f64 / attempts as f64;

    // Full public sampling path (transcript absorb + SHAKE seed + decode),
    // amortized per challenge, for n=1 and n=1024 batches.
    let cfg = SparseChallengeConfig::ExactShell {
        count_mag1: C1,
        count_mag2: C2,
        operator_norm_threshold: T as u32,
    };
    let batch_ns = |n: usize, iters: u64| -> f64 {
        time_ns(iters, || {
            let mut tr = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
            tr.append_field(b"perf-seed", &F::from_u64(0xC0FFEE));
            let chs =
                crate::sample_sparse_challenges::<F, _, D>(&mut tr, b"perf", n, &cfg, 0, false)
                    .unwrap();
            black_box(chs);
        }) / n as f64
    };
    let cold1_ns = batch_ns(1, 20_000);
    let amort1024_ns = batch_ns(1024, 2_000);

    println!("\n=== operator-norm microbench (D={D}, q={Q}, T={T}, shell=({C1},{C2})) ===");
    println!(
        "certified table build (one-time) : {build_ns:>10.0} ns  ({:.2} us)",
        build_ns / 1e3
    );
    println!("sample 1 candidate (warm XOF)    : {sample_ns:>10.2} ns");
    println!("  full path, n=1 (cold + SHAKE)  : {cold1_ns:>10.2} ns");
    println!("  full path, per chal. @ n=1024  : {amort1024_ns:>10.2} ns");
    println!("op-norm check, pool avg (d/2)    : {decide_ns:>10.2} ns  (production: usize idx, i128 accum)");
    println!("op-norm check, accepted (d/2)    : {decide_worst_ns:>10.2} ns");
    println!(
        "sample + check, one candidate    : {:>10.2} ns",
        sample_ns + decide_ns
    );
    println!("--- rejection sampling (verifier replay) ---");
    println!("empirical accept prob p          : {p:.4}  ({accepts} acc / {rejects} rej / {indet} indet, {attempts} attempts)");
    println!("avg attempts / accepted          : {:.3}", 1.0 / p);
    println!(
        "time / accepted challenge        : {per_accepted_ns:>10.2} ns  ({:.3} us)",
        per_accepted_ns / 1e3
    );
    println!();
}
