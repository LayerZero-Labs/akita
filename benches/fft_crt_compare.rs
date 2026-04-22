//! Compare FFT- vs direct-multiplication approaches to the "256 → 1024 RS
//! expand" task used by `benches/fft_smooth.rs::bench_rs_expand_256_to_1024_par`.
//!
//! The task: take a batch of polynomials (each represented as 1470 `Fp128`
//! coefficients) and produce the 1470 evaluations of every polynomial on
//! `{ω^i | i = 0..1469}`, where `ω` is the 1470-th primitive root of unity in
//! `Fp128`.  All methods below produce the same output vectors on the same
//! inputs (checked at startup).
//!
//! Three methods:
//!
//! * **FFT (baseline).**  The original approach: a mixed-radix 1470-point
//!   forward FFT in `Fp128` per polynomial, via the existing `SmoothDomain`
//!   infrastructure.  Cost ≈ `n log n` `Fp128` ops per polynomial.
//!
//! * **Horner in `Fp128`.**  Evaluate the polynomial at each of the 1470
//!   points independently, using Horner's rule with native `Fp128` mul/add.
//!   Cost ≈ `n²` `Fp128` mul-adds per polynomial.  This is the schoolbook
//!   baseline — no FFT, no CRT.
//!
//! * **Horner with CRT-based `Fp128` multiplication.**  Same Horner
//!   structure, but each inner `Fp128 × Fp128` multiplication is computed
//!   by:
//!     1. Lifting both operands to residues modulo 9 small NTT-friendly
//!        30-bit primes whose product exceeds `p² ≈ 2^256`, so the exact
//!        integer product `y·x` fits inside `∏ p_k`.
//!     2. Pointwise multiplying in each small prime (Barrett reduction,
//!        u64 arithmetic).
//!     3. Reconstructing the exact `y·x` as a 270-bit integer via Garner,
//!        then folding it back into `Fp128` via two Solinas-style folds
//!        (`p = 2^128 − 2355`).
//!   The Horner addend `+ c` is a native `Fp128` add.  This is the honest
//!   "use CRT for the field multiplication" interpretation: the integer
//!   product spans ≤ `2·128` bits, exactly the regime where lift + small
//!   muls + Garner is well-defined.
//!
//! Because Horner is `Θ(n / log n)` times slower per polynomial than FFT,
//! the bench uses a smaller default batch size (`COUNT = 256`).  Override
//! via `CRT_BENCH_COUNT`.

#![allow(missing_docs)]

use std::env;

use criterion::{black_box, criterion_group, Criterion};
use hachi_pcs::algebra::fields::fft::{primitive_root_of_unity, SmoothDomain};
use hachi_pcs::algebra::Prime128Offset2355;
use hachi_pcs::{FieldCore, FieldSampling};
use rand::{rngs::StdRng, SeedableRng};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type F = Prime128Offset2355;

const P: u128 = 0xfffffffffffffffffffffffffffff6cd;
const P_MINUS_1: u128 = P - 1;
/// `2^128 − p`.  Used in the final Solinas-style fold.
const P_COMPL: u128 = 2355;

const DOMAIN_SIZE: usize = 1470;
const K_INPUT: usize = 256;
const DEFAULT_COUNT: usize = 256;

/// Number of 30-bit primes used for CRT-based field multiplication.
/// We need `∏ p_k > p² ≈ 2^256`; nine 30-bit primes give ~270 bits, which
/// is sufficient to exactly represent `y·x` before reduction.
const NUM_PRIMES: usize = 9;

/// Number of primes used for the "lifted-Horner" and small-prime NTT
/// variants.  These keep `y` in residue form end-to-end (no per-step
/// mod-`p_128` reduction), so each small-prime arithmetic step stays in
/// `[0, p_k)` without ever needing a full-precision integer.  We only need
/// `∏ p_k > p_128 ≈ 2^128` for the final Garner reconstruction to produce
/// some `Fp128` value; five 30-bit primes give ~150 bits, plenty.
///
/// **Correctness caveat.**  The small-prime arithmetic computes `f(x_k)`
/// where `x_k = ω_k^j` is a root-of-unity *in `F_{p_k}`*, and the
/// coefficients are `c_i mod p_k`.  The *integer* value of
/// `Σ c_i · x_k^i` is not the same integer as the `Fp128` evaluation
/// `Σ c_i · ω^i mod p_128`, so the Garner reconstruction produces a
/// different `Fp128` element than the FFT baseline.  These benches are
/// labelled `_no_reduce_` and serve only as workload-cost comparisons, not
/// equivalent-output computations.
const NUM_LIFT_PRIMES: usize = 5;
const NUM_NTT_PRIMES: usize = 5;

fn generator() -> F {
    F::from_canonical_u128(2)
}

fn count_from_env() -> usize {
    env::var("CRT_BENCH_COUNT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_COUNT)
}

// ---------------------------------------------------------------------------
//  Small-prime basis: 9 distinct 30-bit primes.  Reuses the 5 Hachi Q128
//  primes (which also satisfy `2048 | p − 1`, although that's irrelevant
//  here) and adds four more found by trial division just below `2^30`.
// ---------------------------------------------------------------------------

fn small_primes() -> [u32; NUM_PRIMES] {
    // Start with the Hachi Q128 5-prime set (hardcoded in
    // `src/algebra/ntt/tables.rs`).
    let mut primes: [u32; NUM_PRIMES] = [
        1073707009, 1073698817, 1073692673, 1073682433, 1073668097, 0, 0, 0, 0,
    ];
    // Fill the remaining 4 slots with the largest primes below 2^30 that are
    // not already present.
    let mut candidate: u32 = 1073741823;
    let mut filled = 5;
    while filled < NUM_PRIMES {
        if is_prime_u32(candidate) && !primes[..filled].contains(&candidate) {
            primes[filled] = candidate;
            filled += 1;
        }
        candidate -= 2;
    }
    primes
}

fn is_prime_u32(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3u32;
    while (d as u64) * (d as u64) <= n as u64 {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

// ---------------------------------------------------------------------------
//  Shared setup: build one `coeffs_batch` used by every benchmark.
// ---------------------------------------------------------------------------

struct CrtParams {
    primes: [u32; NUM_PRIMES],
    /// `barrett_m[k] = floor(2^64 / p_k)`.  Used in `barrett_reduce`.
    barrett_m: [u64; NUM_PRIMES],
    /// `two64_mod_p[k] = 2^64 mod p_k`.  Used in the fast `u128 → u32` lift.
    two64_mod_p: [u32; NUM_PRIMES],
    /// `gamma[i][j] = p_j^{-1} mod p_i` for `j < i`; upper / diagonal zero.
    gamma: [[u32; NUM_PRIMES]; NUM_PRIMES],
    /// `pi_prefix_mod_p[i] = (p_0 · p_1 · … · p_{i-1}) mod p_128`, stored
    /// as two u64 limbs.  `pi_prefix_mod_p[0] = 1`.  Lets us compute the
    /// Garner reconstruction directly modulo `p_128` as
    /// `Σ c[i] · pi_prefix_mod_p[i] mod p_128`, skipping the detour through
    /// a 270-bit integer.
    pi_prefix_mod_p: [[u64; 2]; NUM_PRIMES],
}

impl CrtParams {
    fn new() -> Self {
        let primes = small_primes();
        let barrett_m: [u64; NUM_PRIMES] =
            std::array::from_fn(|k| ((1u128 << 64) / primes[k] as u128) as u64);
        let two64_mod_p: [u32; NUM_PRIMES] =
            std::array::from_fn(|k| ((1u128 << 64) % primes[k] as u128) as u32);
        let mut gamma = [[0u32; NUM_PRIMES]; NUM_PRIMES];
        for i in 1..NUM_PRIMES {
            let pi = primes[i] as i64;
            for j in 0..i {
                let pj = primes[j] as i64;
                gamma[i][j] = mod_inverse_i64(pj, pi) as u32;
            }
        }
        // pi_prefix_mod_p[0] = 1; pi_prefix_mod_p[i+1] = (prev * p_i) mod p_128.
        let mut pi_prefix_mod_p = [[0u64; 2]; NUM_PRIMES];
        pi_prefix_mod_p[0] = [1u64, 0u64];
        let mut prev = F::one();
        for i in 1..NUM_PRIMES {
            prev = prev * F::from_canonical_u128(primes[i - 1] as u128);
            let v = prev.to_canonical_u128();
            pi_prefix_mod_p[i] = [v as u64, (v >> 64) as u64];
        }
        Self {
            primes,
            barrett_m,
            two64_mod_p,
            gamma,
            pi_prefix_mod_p,
        }
    }
}

/// Parameters for the lifted-Horner (5-prime) path.  Shares the same shape
/// as `CrtParams` but smaller.  `primes` are the first `NUM_LIFT_PRIMES`
/// entries of the 9-prime CRT basis.
struct CrtLiftedParams {
    primes: [u32; NUM_LIFT_PRIMES],
    barrett_m: [u64; NUM_LIFT_PRIMES],
    two64_mod_p: [u32; NUM_LIFT_PRIMES],
    gamma: [[u32; NUM_LIFT_PRIMES]; NUM_LIFT_PRIMES],
    pi_prefix_mod_p: [[u64; 2]; NUM_LIFT_PRIMES],
}

impl CrtLiftedParams {
    fn new() -> Self {
        let all = small_primes();
        let primes: [u32; NUM_LIFT_PRIMES] = std::array::from_fn(|k| all[k]);
        let barrett_m: [u64; NUM_LIFT_PRIMES] =
            std::array::from_fn(|k| ((1u128 << 64) / primes[k] as u128) as u64);
        let two64_mod_p: [u32; NUM_LIFT_PRIMES] =
            std::array::from_fn(|k| ((1u128 << 64) % primes[k] as u128) as u32);
        let mut gamma = [[0u32; NUM_LIFT_PRIMES]; NUM_LIFT_PRIMES];
        for i in 1..NUM_LIFT_PRIMES {
            let pi = primes[i] as i64;
            for j in 0..i {
                let pj = primes[j] as i64;
                gamma[i][j] = mod_inverse_i64(pj, pi) as u32;
            }
        }
        let mut pi_prefix_mod_p = [[0u64; 2]; NUM_LIFT_PRIMES];
        pi_prefix_mod_p[0] = [1u64, 0u64];
        let mut prev = F::one();
        for i in 1..NUM_LIFT_PRIMES {
            prev = prev * F::from_canonical_u128(primes[i - 1] as u128);
            let v = prev.to_canonical_u128();
            pi_prefix_mod_p[i] = [v as u64, (v >> 64) as u64];
        }
        Self {
            primes,
            barrett_m,
            two64_mod_p,
            gamma,
            pi_prefix_mod_p,
        }
    }
}

/// Parameters for the small-prime NTT path.  Requires primes with
/// `1470 | (p - 1)` so each prime has a primitive 1470-th root of unity.
struct NttParams {
    primes: [u32; NUM_NTT_PRIMES],
    barrett_m: [u64; NUM_NTT_PRIMES],
    two64_mod_p: [u32; NUM_NTT_PRIMES],
    gamma: [[u32; NUM_NTT_PRIMES]; NUM_NTT_PRIMES],
    pi_prefix_mod_p: [[u64; 2]; NUM_NTT_PRIMES],
    domains: Vec<SmallFft>,
}

impl NttParams {
    fn new() -> Self {
        let primes = primes_with_1470_divisor();
        let barrett_m: [u64; NUM_NTT_PRIMES] =
            std::array::from_fn(|k| ((1u128 << 64) / primes[k] as u128) as u64);
        let two64_mod_p: [u32; NUM_NTT_PRIMES] =
            std::array::from_fn(|k| ((1u128 << 64) % primes[k] as u128) as u32);
        let mut gamma = [[0u32; NUM_NTT_PRIMES]; NUM_NTT_PRIMES];
        for i in 1..NUM_NTT_PRIMES {
            let pi = primes[i] as i64;
            for j in 0..i {
                let pj = primes[j] as i64;
                gamma[i][j] = mod_inverse_i64(pj, pi) as u32;
            }
        }
        let mut pi_prefix_mod_p = [[0u64; 2]; NUM_NTT_PRIMES];
        pi_prefix_mod_p[0] = [1u64, 0u64];
        let mut prev = F::one();
        for i in 1..NUM_NTT_PRIMES {
            prev = prev * F::from_canonical_u128(primes[i - 1] as u128);
            let v = prev.to_canonical_u128();
            pi_prefix_mod_p[i] = [v as u64, (v >> 64) as u64];
        }
        let domains: Vec<SmallFft> = (0..NUM_NTT_PRIMES)
            .map(|k| SmallFft::new(primes[k], DOMAIN_SIZE, FACTORS_1470))
            .collect();
        Self {
            primes,
            barrett_m,
            two64_mod_p,
            gamma,
            pi_prefix_mod_p,
            domains,
        }
    }
}

/// Search for 5 primes `p ≡ 1 (mod 1470)` below `2^30`, descending.
fn primes_with_1470_divisor() -> [u32; NUM_NTT_PRIMES] {
    let mut out = [0u32; NUM_NTT_PRIMES];
    let mut filled = 0;
    // Largest value `< 2^30` with `v ≡ 1 (mod 1470)`.
    let boundary = (((1u32 << 30) - 1) / 1470) * 1470 + 1;
    let mut cand = if boundary >= (1u32 << 30) {
        boundary - 1470
    } else {
        boundary
    };
    while filled < NUM_NTT_PRIMES {
        if is_prime_u32(cand) {
            out[filled] = cand;
            filled += 1;
        }
        cand = cand
            .checked_sub(1470)
            .expect("ran out of candidates below 2^30");
    }
    out
}

struct Setup {
    /// Fp128 coefficients per polynomial, length `DOMAIN_SIZE`.
    coeffs: Vec<Vec<F>>,
    /// The 1470 evaluation points `{ω^i}` as `Fp128`.
    eval_points: Vec<F>,
    /// `SmoothDomain` for the FFT path.
    domain: SmoothDomain<F>,
    /// CRT basis + Barrett + Garner precomputations for the CRT-mul path.
    crt: CrtParams,
    /// Smaller 5-prime basis for the lifted-Horner and NTT paths.
    lifted: CrtLiftedParams,
    /// NTT-friendly primes + per-prime `SmallFft` for the small-prime NTT path.
    ntt: NttParams,
    count: usize,
}

fn build_setup(count: usize) -> Setup {
    let g = generator();
    let omega = primitive_root_of_unity(g, P_MINUS_1, DOMAIN_SIZE);
    let domain = SmoothDomain::new(omega, DOMAIN_SIZE);

    // Build the Fp128 coefficient batch the same way the FFT bench does:
    // random 256 "base evals", zero-pad, IFFT to coefficients of length 1470.
    let coeffs: Vec<Vec<F>> = (0..count)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(0xff04 + i as u64);
            let base: Vec<F> = (0..K_INPUT)
                .map(|_| FieldSampling::sample(&mut rng))
                .collect();
            let mut padded = vec![F::zero(); DOMAIN_SIZE];
            padded[..K_INPUT].copy_from_slice(&base);
            domain.inverse(&padded)
        })
        .collect();

    let mut eval_points = Vec::with_capacity(DOMAIN_SIZE);
    let mut pw = F::one();
    for _ in 0..DOMAIN_SIZE {
        eval_points.push(pw);
        pw = pw * omega;
    }

    Setup {
        coeffs,
        eval_points,
        domain,
        crt: CrtParams::new(),
        lifted: CrtLiftedParams::new(),
        ntt: NttParams::new(),
        count,
    }
}

// ---------------------------------------------------------------------------
//  Small-prime arithmetic.
// ---------------------------------------------------------------------------

/// Barrett reduction of `x < 2^62` modulo `p < 2^30` using
/// `m = floor(2^64 / p)`.  Returns `x mod p`.
#[inline(always)]
fn barrett_reduce(x: u64, p: u64, m: u64) -> u64 {
    let q = ((x as u128 * m as u128) >> 64) as u64;
    let mut r = x.wrapping_sub(q.wrapping_mul(p));
    if r >= p {
        r -= p;
    }
    r
}

/// `a * b mod p` where `a, b < p < 2^30` and `m = floor(2^64 / p)`.
#[inline(always)]
fn mulmod_small(a: u32, b: u32, p: u32, m: u64) -> u32 {
    barrett_reduce((a as u64) * (b as u64), p as u64, m) as u32
}

/// Reduce a canonical `u128` value modulo a 30-bit prime using precomputed
/// Barrett / reduction constants.
///
/// Faster than `x % p` (which Rust lowers to `__umodti3` / soft-div).  We
/// decompose `x = hi·2^64 + lo` and use
/// `x mod p = ((hi mod p) · (2^64 mod p) + (lo mod p)) mod p`, where every
/// inner reduction is a single `barrett_reduce` in u64.
#[inline(always)]
fn reduce_u128_small(x: u128, p: u32, m: u64, two64_mod_p: u32) -> u32 {
    let p64 = p as u64;
    let hi = (x >> 64) as u64;
    let lo = x as u64;
    let hi_r = barrett_reduce(hi, p64, m);
    let lo_r = barrett_reduce(lo, p64, m);
    let t = hi_r.wrapping_mul(two64_mod_p as u64) + lo_r;
    barrett_reduce(t, p64, m) as u32
}

fn mod_inverse_i64(a: i64, modulus: i64) -> i64 {
    let (mut t, mut new_t) = (0i64, 1i64);
    let (mut r, mut new_r) = (modulus, a.rem_euclid(modulus));
    while new_r != 0 {
        let q = r / new_r;
        (t, new_t) = (new_t, t - q * new_t);
        (r, new_r) = (new_r, r - q * new_r);
    }
    assert_eq!(r, 1, "modular inverse does not exist");
    t.rem_euclid(modulus)
}

/// Modular exponentiation `base^exp mod p` in a 30-bit prime.
#[inline]
fn pow_mod_u32(mut base: u32, mut exp: u64, p: u64, m: u64) -> u32 {
    let mut result = 1u32;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mulmod_small(result, base, p as u32, m);
        }
        base = mulmod_small(base, base, p as u32, m);
        exp >>= 1;
    }
    result
}

/// Find a primitive `n`-th root of unity in `F_p` by trying small candidates.
/// Requires `n | (p − 1)`.
fn primitive_root_u32(p: u32, n: u32) -> u32 {
    let pm1 = (p - 1) as u64;
    assert_eq!(pm1 % n as u64, 0, "n must divide p-1");
    let k = pm1 / n as u64;
    let p64 = p as u64;
    let m = ((1u128 << 64) / p as u128) as u64;

    // Prime factors of n.  For n = 1470 = 2·3·5·7² these are {2, 3, 5, 7}.
    let mut n_prime_factors: Vec<u32> = Vec::new();
    {
        let mut nn = n;
        let mut d = 2u32;
        while (d as u64) * (d as u64) <= nn as u64 {
            if nn % d == 0 {
                n_prime_factors.push(d);
                while nn % d == 0 {
                    nn /= d;
                }
            }
            d += 1;
        }
        if nn > 1 {
            n_prime_factors.push(nn);
        }
    }

    for g in 2..p {
        let r = pow_mod_u32(g, k, p64, m);
        if r <= 1 {
            continue;
        }
        let is_primitive = n_prime_factors.iter().all(|&q| {
            let exp = (n / q) as u64;
            pow_mod_u32(r, exp, p64, m) != 1
        });
        if is_primitive {
            return r;
        }
    }
    panic!("no primitive {n}-th root in F_{p}");
}

// ---------------------------------------------------------------------------
//  Small-prime mixed-radix FFT.
//
//  A self-contained port of `algebra/fields/fft.rs::SmoothDomain` restricted
//  to u32 primes.  Uses precomputed twiddle tables + a scratch buffer of
//  size up to `max(factors)` for the per-butterfly `r`-point DFT.
// ---------------------------------------------------------------------------

/// Prime-factor decomposition of 1470 = 2·3·5·7·7.  Used as the mixed-radix
/// factorisation for the small-prime FFT.
const FACTORS_1470: &[usize] = &[2, 3, 5, 7, 7];

struct SmallStage {
    /// Radix for this stage.
    r: usize,
    /// Block size before the stage.
    block: usize,
    /// `omega_r_pow[q] = omega_r^q` for `q = 0..r`.
    omega_r_pow: [u32; 8],
    /// `twiddle_table[j] = omega_new_block^j` for `j = 0..block`.
    twiddle_table: Vec<u32>,
}

struct SmallFft {
    n: usize,
    p: u64,
    m: u64,
    digit_rev: Vec<usize>,
    stages: Vec<SmallStage>,
}

impl SmallFft {
    fn new(p: u32, n: usize, factors: &[usize]) -> Self {
        assert_eq!(
            factors.iter().product::<usize>(),
            n,
            "factor product must equal n"
        );
        let p64 = p as u64;
        let m = ((1u128 << 64) / p as u128) as u64;
        let omega = primitive_root_u32(p, n as u32);

        let digit_rev = digit_reversal_permutation_u32(n, factors);
        let stages = precompute_stages_u32(omega, p, m, n, factors);
        Self {
            n,
            p: p64,
            m,
            digit_rev,
            stages,
        }
    }

    /// Forward DFT over `F_p` at the stored primitive `n`-th root.
    ///
    /// `input.len()` must equal `self.n`.  Cost ≈ `n · Σ r_s · r_s` u64
    /// mul-adds, i.e. ≈ `n · 136 = 2·10⁵` for `n = 1470`.
    fn forward(&self, input: &[u32]) -> Vec<u32> {
        assert_eq!(input.len(), self.n);
        let n = self.n;
        let p = self.p;
        let m = self.m;
        let p32 = p as u32;

        let mut buf = vec![0u32; n];
        for (i, &rev_i) in self.digit_rev.iter().enumerate() {
            buf[rev_i] = input[i];
        }

        let mut scratch = [0u32; 8];
        for stage in &self.stages {
            let r = stage.r;
            let block = stage.block;
            let new_block = block * r;
            let twiddle_table = &stage.twiddle_table;
            let omega_r_pow = &stage.omega_r_pow;

            for group_start in (0..n).step_by(new_block) {
                for (j, &tw_base) in twiddle_table.iter().enumerate() {
                    // Twist the `r` strided inputs by powers of `tw_base`.
                    let mut tw_q = 1u32;
                    for q in 0..r {
                        let idx = group_start + q * block + j;
                        scratch[q] = mulmod_small(buf[idx], tw_q, p32, m);
                        if q + 1 < r {
                            tw_q = mulmod_small(tw_q, tw_base, p32, m);
                        }
                    }
                    // r-point DFT: `out[q_out] = Σ_q scratch[q] · ω_r^{q·q_out}`.
                    for q_out in 0..r {
                        // `r ≤ 7`, so the accumulator stays well below 2^64.
                        let mut acc: u64 = 0;
                        for q in 0..r {
                            let w = omega_r_pow[(q * q_out) % r];
                            acc += mulmod_small(scratch[q], w, p32, m) as u64;
                        }
                        buf[group_start + q_out * block + j] = if acc >= p {
                            barrett_reduce(acc, p, m) as u32
                        } else {
                            acc as u32
                        };
                    }
                }
            }
        }
        buf
    }
}

fn digit_reversal_permutation_u32(n: usize, factors: &[usize]) -> Vec<usize> {
    let s = factors.len();
    let mut perm = vec![0usize; n];
    for (k, perm_k) in perm.iter_mut().enumerate() {
        let mut digits = vec![0usize; s];
        let mut tmp = k;
        for (digit, &f) in digits.iter_mut().zip(factors.iter()) {
            *digit = tmp % f;
            tmp /= f;
        }
        let mut rev = 0usize;
        for (&f, &d) in factors.iter().zip(digits.iter()) {
            rev = rev * f + d;
        }
        *perm_k = rev;
    }
    perm
}

fn precompute_stages_u32(
    omega: u32,
    p: u32,
    m: u64,
    n: usize,
    factors: &[usize],
) -> Vec<SmallStage> {
    let p64 = p as u64;
    let mut stages = Vec::with_capacity(factors.len());
    let mut block = 1usize;
    for &r in factors.iter().rev() {
        debug_assert!(r <= 8, "radix {r} exceeds scratch capacity");
        let new_block = block * r;
        let omega_new_block = pow_mod_u32(omega, (n / new_block) as u64, p64, m);
        let omega_r = pow_mod_u32(omega_new_block, block as u64, p64, m);

        let mut omega_r_pow = [1u32; 8];
        for q in 1..r {
            omega_r_pow[q] = mulmod_small(omega_r_pow[q - 1], omega_r, p, m);
        }

        let mut twiddle_table = Vec::with_capacity(block);
        let mut tw = 1u32;
        for _ in 0..block {
            twiddle_table.push(tw);
            tw = mulmod_small(tw, omega_new_block, p, m);
        }

        stages.push(SmallStage {
            r,
            block,
            omega_r_pow,
            twiddle_table,
        });
        block = new_block;
    }
    stages
}

// ---------------------------------------------------------------------------
//  CRT-based `Fp128` multiplication: `y * x mod p`.
// ---------------------------------------------------------------------------

/// Lift `x ∈ Fp128` (passed as `u128`) to its 9 small-prime residues.
#[inline(always)]
fn lift_to_primes(x: u128, crt: &CrtParams) -> [u32; NUM_PRIMES] {
    std::array::from_fn(|k| {
        reduce_u128_small(x, crt.primes[k], crt.barrett_m[k], crt.two64_mod_p[k])
    })
}

/// Lift `x` to the 5-prime lifted-Horner basis.
#[inline(always)]
fn lift_to_primes_small(x: u128, crt: &CrtLiftedParams) -> [u32; NUM_LIFT_PRIMES] {
    std::array::from_fn(|k| {
        reduce_u128_small(x, crt.primes[k], crt.barrett_m[k], crt.two64_mod_p[k])
    })
}

/// Lift `x` to the 5-prime NTT-prime basis.
#[inline(always)]
fn lift_to_primes_ntt(x: u128, ntt: &NttParams) -> [u32; NUM_NTT_PRIMES] {
    std::array::from_fn(|k| {
        reduce_u128_small(x, ntt.primes[k], ntt.barrett_m[k], ntt.two64_mod_p[k])
    })
}

/// Full-precision `Fp128` multiplication via CRT:
///
/// 1. Lift `y` and `x` to residues modulo 9 small primes.  (The caller can
///    pass precomputed residues to avoid the lift — see `crt_mul_pre_x`.)
/// 2. Pointwise multiply residues in each small prime.
/// 3. Garner-reconstruct the exact 256-bit integer `y·x` as 9 mixed-radix
///    limbs.
/// 4. Fold that integer modulo `p = 2^128 − 2355` into a canonical u128.
///
/// Returns `(y · x) mod p` in `[0, p)`.
#[inline]
fn crt_mul(y: u128, x: u128, crt: &CrtParams) -> u128 {
    let y_res = lift_to_primes(y, crt);
    let x_res = lift_to_primes(x, crt);
    crt_mul_residues(y_res, x_res, crt)
}

/// Like `crt_mul`, but with `x`'s residues already precomputed.
#[inline]
fn crt_mul_pre_x(y: u128, x_res: &[u32; NUM_PRIMES], crt: &CrtParams) -> u128 {
    let y_res = lift_to_primes(y, crt);
    crt_mul_residues(y_res, *x_res, crt)
}

/// Multiply two elements given as their small-prime residues; return
/// `(y · x) mod p` as a canonical `u128 < p`.
#[inline]
fn crt_mul_residues(y_res: [u32; NUM_PRIMES], x_res: [u32; NUM_PRIMES], crt: &CrtParams) -> u128 {
    // Pointwise multiplication: prod_res[k] = (y_res[k] * x_res[k]) mod p_k.
    let mut prod_res = [0u32; NUM_PRIMES];
    for k in 0..NUM_PRIMES {
        prod_res[k] = mulmod_small(y_res[k], x_res[k], crt.primes[k], crt.barrett_m[k]);
    }
    // Garner-reconstruct into a 270-bit integer, stored as 5 u64 limbs
    // little-endian.  Reduce modulo p in the same pass.
    garner_reduce_mod_p(&prod_res, crt)
}

/// Garner-reconstruct residues into an `Fp128` element, using precomputed
/// `pi_prefix_mod_p` to stay modulo `p_128` throughout.
///
/// Two phases:
/// 1. **Garner mixed-radix coefficients** `c[i] ∈ [0, p_i)`.
/// 2. **Modular recomposition**: accumulate `Σ c[i] · pi_prefix_mod_p[i]`
///    in a 3-limb (192-bit) buffer, then reduce mod `p = 2^128 − 2355` via
///    one Solinas fold.  Each summand is bounded by `2^30 · 2^128 = 2^158`;
///    nine summands fit comfortably in 192 bits.
#[inline]
fn garner_reduce_mod_p(residues: &[u32; NUM_PRIMES], crt: &CrtParams) -> u128 {
    // Phase 1: Garner coefficients.  Keep `acc` in `u64` by adding `p_i`
    // before subtracting `c[j]` (both `< p_i < 2^30`); Barrett-reduce each
    // step.
    let mut c = [0u32; NUM_PRIMES];
    c[0] = residues[0];
    for i in 1..NUM_PRIMES {
        let pi = crt.primes[i] as u64;
        let mi = crt.barrett_m[i];
        let mut acc = residues[i] as u64;
        for j in 0..i {
            // (acc + pi - c[j]) < 2·p_i < 2^31; product < 2^61.
            let diff = acc + pi - c[j] as u64;
            let prod = diff * crt.gamma[i][j] as u64;
            acc = barrett_reduce(prod, pi, mi);
        }
        c[i] = acc as u32;
    }

    // Phase 2: accumulate `Σ c[i] · pi_prefix_mod_p[i]` into 3 u64 limbs.
    let mut r0: u64 = 0;
    let mut r1: u64 = 0;
    let mut r2: u64 = 0;
    for i in 0..NUM_PRIMES {
        let prefix = crt.pi_prefix_mod_p[i];
        let c_i = c[i] as u64;
        // Widening 128 × u64 → 192-bit term `[t0, t1, t2]`.
        let (t0, p0_hi) = mul64_wide(prefix[0], c_i);
        let (p1_lo, p1_hi) = mul64_wide(prefix[1], c_i);
        let mid = (p0_hi as u128) + (p1_lo as u128);
        let t1 = mid as u64;
        let t2 = p1_hi + (mid >> 64) as u64;

        // Add [t0, t1, t2] into [r0, r1, r2] with carry.
        let (s0, c0) = r0.overflowing_add(t0);
        let (s1a, c1a) = r1.overflowing_add(t1);
        let (s1, c1b) = s1a.overflowing_add(c0 as u64);
        let s2 = r2
            .wrapping_add(t2)
            .wrapping_add((c1a as u64) | (c1b as u64));
        r0 = s0;
        r1 = s1;
        r2 = s2;
    }

    // Reduce the 3-limb integer X = r0 + r1·2^64 + r2·2^128 mod p.
    //   X mod p = (r0 + r1·2^64) + r2·2355.
    // r2 is bounded by ~2^34 (9 · 2^30), so r2 · 2355 < 2^46 fits in u64.
    let x_lo: u128 = (r0 as u128) | ((r1 as u128) << 64);
    let fold = (r2 as u128) * P_COMPL;
    let (mut r, over) = x_lo.overflowing_add(fold);
    if over {
        r = r.wrapping_add(P_COMPL);
    }
    if r >= P {
        r = r.wrapping_sub(P);
    }
    if r >= P {
        r = r.wrapping_sub(P);
    }
    r
}

/// Widening 64×64 → 128 multiply, as `(lo, hi)`.
#[inline(always)]
fn mul64_wide(a: u64, b: u64) -> (u64, u64) {
    let p = (a as u128) * (b as u128);
    (p as u64, (p >> 64) as u64)
}

// ---------------------------------------------------------------------------
//  Per-polynomial evaluation routines for each method.
// ---------------------------------------------------------------------------

/// FFT baseline: one `coset_forward` per polynomial.  Same body as
/// `bench_rs_expand_256_to_1024_par` in `fft_smooth.rs`.
#[inline]
fn eval_fft(coeffs: &[F], domain: &SmoothDomain<F>) -> Vec<F> {
    domain.coset_forward(coeffs, F::one())
}

/// Direct Horner evaluation in `Fp128`.
#[inline]
fn eval_horner_fp128(coeffs: &[F], eval_points: &[F]) -> Vec<F> {
    let n = coeffs.len();
    eval_points
        .iter()
        .map(|&x| {
            let mut y = coeffs[n - 1];
            for i in (0..n - 1).rev() {
                y = y * x + coeffs[i];
            }
            y
        })
        .collect()
}

/// Horner evaluation where each inner `Fp128` multiplication is computed
/// via the 9-prime CRT path.
///
/// `coeffs_u128[i] = coeffs[i].to_canonical_u128()` is precomputed.
/// `point_residues[j] = residues of the j-th evaluation point` are
/// precomputed so every Horner loop only lifts the moving `y`.
#[inline]
fn eval_horner_crt(
    coeffs_u128: &[u128],
    point_residues: &[[u32; NUM_PRIMES]],
    crt: &CrtParams,
) -> Vec<F> {
    let n = coeffs_u128.len();
    assert_eq!(point_residues.len(), n);
    point_residues
        .iter()
        .map(|x_res| {
            let mut y: u128 = coeffs_u128[n - 1];
            for i in (0..n - 1).rev() {
                // y ← (y · x) mod p, via CRT mul.
                let prod = crt_mul_pre_x(y, x_res, crt);
                // y ← prod + c, in Fp128 (u128 add + Solinas fold).
                y = fp128_add(prod, coeffs_u128[i]);
            }
            F::from_canonical_u128(y)
        })
        .collect()
}

/// Lifted-Horner evaluation: keep the running `y` as a vector of 5
/// residues `(y mod p_k)_k` end-to-end, with pointwise mul/add in each
/// small prime.  No per-step mod-`p_128` reduction.
///
/// **Output is not equivalent to `eval_fft` / `eval_horner_fp128`.** The
/// small-prime arithmetic computes `Σ c_i · x^i mod p_k` where the integer
/// value of the sum can be far larger than `∏ p_k`, so Garner
/// reconstruction does not yield the `Fp128` value `f(x) mod p_128`.  This
/// function exists to measure the raw cost of the "ideal CRT" layout where
/// reductions mod `p_128` are elided.
#[inline]
fn eval_lifted_horner(
    coeffs_res: &[[u32; NUM_LIFT_PRIMES]],
    point_res: &[[u32; NUM_LIFT_PRIMES]],
    lifted: &CrtLiftedParams,
) -> Vec<F> {
    let n = coeffs_res.len();
    point_res
        .iter()
        .map(|x_res| {
            let mut y_res = coeffs_res[n - 1];
            for i in (0..n - 1).rev() {
                // Pointwise in each prime: `y_k ← (y_k · x_k + c_k) mod p_k`.
                for k in 0..NUM_LIFT_PRIMES {
                    let p32 = lifted.primes[k];
                    let p64 = p32 as u64;
                    let m = lifted.barrett_m[k];
                    let prod = (y_res[k] as u64) * (x_res[k] as u64);
                    let sum = prod + coeffs_res[i][k] as u64;
                    y_res[k] = barrett_reduce(sum, p64, m) as u32;
                }
            }
            F::from_canonical_u128(garner_reduce_mod_p_lifted(&y_res, lifted))
        })
        .collect()
}

/// Small-prime NTT evaluation: for each small prime, run a 1470-point
/// forward NTT (at *that* prime's 1470-th root of unity), then
/// Garner-reconstruct one `Fp128` value per output position.
///
/// **Output is not equivalent to `eval_fft`.**  The `k`-th NTT computes
/// `Σ c_i · ω_k^{ij} mod p_k`, where `ω_k` is a primitive 1470-th root of
/// unity *in `F_{p_k}`*, unrelated to the `ω` used by the `Fp128` FFT
/// baseline.  Benchmarked purely to measure the small-prime NTT workload.
#[inline]
fn eval_small_prime_ntt(coeffs_res: &[[u32; NUM_NTT_PRIMES]], ntt: &NttParams) -> Vec<F> {
    let n = coeffs_res.len();
    // Run one forward NTT per prime.
    let mut per_prime: [Vec<u32>; NUM_NTT_PRIMES] = std::array::from_fn(|_| Vec::with_capacity(n));
    for k in 0..NUM_NTT_PRIMES {
        let input: Vec<u32> = coeffs_res.iter().map(|row| row[k]).collect();
        per_prime[k] = ntt.domains[k].forward(&input);
    }
    (0..n)
        .map(|j| {
            let residues: [u32; NUM_NTT_PRIMES] = std::array::from_fn(|k| per_prime[k][j]);
            F::from_canonical_u128(garner_reduce_mod_p_ntt(&residues, ntt))
        })
        .collect()
}

/// 5-prime Garner reconstruction into a canonical `u128 < p_128`.  Same
/// structure as the 9-prime version: iterative Garner coefficients +
/// direct mod-`p_128` accumulation through precomputed `pi_prefix_mod_p`.
#[inline]
fn garner_reduce_mod_p_lifted(residues: &[u32; NUM_LIFT_PRIMES], crt: &CrtLiftedParams) -> u128 {
    let mut c = [0u32; NUM_LIFT_PRIMES];
    c[0] = residues[0];
    for i in 1..NUM_LIFT_PRIMES {
        let pi = crt.primes[i] as u64;
        let mi = crt.barrett_m[i];
        let mut acc = residues[i] as u64;
        for j in 0..i {
            let diff = acc + pi - c[j] as u64;
            let prod = diff * crt.gamma[i][j] as u64;
            acc = barrett_reduce(prod, pi, mi);
        }
        c[i] = acc as u32;
    }

    let mut r0: u64 = 0;
    let mut r1: u64 = 0;
    let mut r2: u64 = 0;
    for i in 0..NUM_LIFT_PRIMES {
        let prefix = crt.pi_prefix_mod_p[i];
        let c_i = c[i] as u64;
        let (t0, p0_hi) = mul64_wide(prefix[0], c_i);
        let (p1_lo, p1_hi) = mul64_wide(prefix[1], c_i);
        let mid = (p0_hi as u128) + (p1_lo as u128);
        let t1 = mid as u64;
        let t2 = p1_hi + (mid >> 64) as u64;

        let (s0, c0) = r0.overflowing_add(t0);
        let (s1a, c1a) = r1.overflowing_add(t1);
        let (s1, c1b) = s1a.overflowing_add(c0 as u64);
        let s2 = r2
            .wrapping_add(t2)
            .wrapping_add((c1a as u64) | (c1b as u64));
        r0 = s0;
        r1 = s1;
        r2 = s2;
    }
    finalise_3limb_mod_p(r0, r1, r2)
}

/// Same reconstruction, parameterised for the NTT basis.
#[inline]
fn garner_reduce_mod_p_ntt(residues: &[u32; NUM_NTT_PRIMES], ntt: &NttParams) -> u128 {
    let mut c = [0u32; NUM_NTT_PRIMES];
    c[0] = residues[0];
    for i in 1..NUM_NTT_PRIMES {
        let pi = ntt.primes[i] as u64;
        let mi = ntt.barrett_m[i];
        let mut acc = residues[i] as u64;
        for j in 0..i {
            let diff = acc + pi - c[j] as u64;
            let prod = diff * ntt.gamma[i][j] as u64;
            acc = barrett_reduce(prod, pi, mi);
        }
        c[i] = acc as u32;
    }

    let mut r0: u64 = 0;
    let mut r1: u64 = 0;
    let mut r2: u64 = 0;
    for i in 0..NUM_NTT_PRIMES {
        let prefix = ntt.pi_prefix_mod_p[i];
        let c_i = c[i] as u64;
        let (t0, p0_hi) = mul64_wide(prefix[0], c_i);
        let (p1_lo, p1_hi) = mul64_wide(prefix[1], c_i);
        let mid = (p0_hi as u128) + (p1_lo as u128);
        let t1 = mid as u64;
        let t2 = p1_hi + (mid >> 64) as u64;

        let (s0, c0) = r0.overflowing_add(t0);
        let (s1a, c1a) = r1.overflowing_add(t1);
        let (s1, c1b) = s1a.overflowing_add(c0 as u64);
        let s2 = r2
            .wrapping_add(t2)
            .wrapping_add((c1a as u64) | (c1b as u64));
        r0 = s0;
        r1 = s1;
        r2 = s2;
    }
    finalise_3limb_mod_p(r0, r1, r2)
}

/// Fold a 3-limb integer `r0 + r1·2^64 + r2·2^128` modulo `p = 2^128 − 2355`
/// into `[0, p)`.  `r2` is bounded by the accumulator invariants of the
/// caller (≤ `NUM_LIFT_PRIMES · 2^30` ≈ `2^33`), so `r2 · 2355` fits in u64.
#[inline(always)]
fn finalise_3limb_mod_p(r0: u64, r1: u64, r2: u64) -> u128 {
    let x_lo: u128 = (r0 as u128) | ((r1 as u128) << 64);
    let fold = (r2 as u128) * P_COMPL;
    let (mut r, over) = x_lo.overflowing_add(fold);
    if over {
        r = r.wrapping_add(P_COMPL);
    }
    if r >= P {
        r = r.wrapping_sub(P);
    }
    if r >= P {
        r = r.wrapping_sub(P);
    }
    r
}

/// Add two canonical `Fp128` values (given as `u128 < p`) and return the
/// canonical result.  Equivalent to `F::add` but stays in `u128`-land so
/// the CRT loop can keep the moving `y` as a `u128`.
#[inline(always)]
fn fp128_add(a: u128, b: u128) -> u128 {
    let (s, carry) = a.overflowing_add(b);
    // s = (a + b) mod 2^128.  Two cases need correction:
    //   carry=1  ⇒ a + b ≥ 2^128  ⇒ a + b = s + 2^128 = s + p + 2355, reduce by p.
    //   carry=0 but s ≥ p ⇒ subtract p once.
    let mut r = if carry { s.wrapping_add(P_COMPL) } else { s };
    if r >= P {
        r = r.wrapping_sub(P);
    }
    r
}

// ---------------------------------------------------------------------------
//  Benchmarks.
// ---------------------------------------------------------------------------

#[cfg(feature = "parallel")]
fn bench_fft_baseline(c: &mut Criterion, setup: &Setup) {
    let name = format!("rs_expand_compare/fft_fp128_x{}_par", setup.count);
    c.bench_function(&name, |b| {
        b.iter(|| {
            let results: Vec<Vec<F>> = setup
                .coeffs
                .par_iter()
                .map(|coeffs| eval_fft(coeffs, &setup.domain))
                .collect();
            black_box(&results);
        })
    });
}

#[cfg(feature = "parallel")]
fn bench_horner_fp128(c: &mut Criterion, setup: &Setup) {
    let name = format!("rs_expand_compare/horner_fp128_x{}_par", setup.count);
    c.bench_function(&name, |b| {
        b.iter(|| {
            let results: Vec<Vec<F>> = setup
                .coeffs
                .par_iter()
                .map(|coeffs| eval_horner_fp128(coeffs, &setup.eval_points))
                .collect();
            black_box(&results);
        })
    });
}

#[cfg(feature = "parallel")]
fn bench_horner_crt(c: &mut Criterion, setup: &Setup) {
    // Precompute `coeffs_u128` per polynomial and shared `point_residues`.
    let point_residues: Vec<[u32; NUM_PRIMES]> = setup
        .eval_points
        .iter()
        .map(|x| lift_to_primes(x.to_canonical_u128(), &setup.crt))
        .collect();
    let coeffs_u128: Vec<Vec<u128>> = setup
        .coeffs
        .iter()
        .map(|poly| poly.iter().map(|c| c.to_canonical_u128()).collect())
        .collect();

    let name = format!("rs_expand_compare/horner_crt_x{}_par", setup.count);
    c.bench_function(&name, |b| {
        b.iter(|| {
            let results: Vec<Vec<F>> = coeffs_u128
                .par_iter()
                .map(|coeffs| eval_horner_crt(coeffs, &point_residues, &setup.crt))
                .collect();
            black_box(&results);
        })
    });
}

#[cfg(feature = "parallel")]
fn bench_lifted_horner(c: &mut Criterion, setup: &Setup) {
    // Precompute per-polynomial and per-point residues in 5 small primes.
    let point_res: Vec<[u32; NUM_LIFT_PRIMES]> = setup
        .eval_points
        .iter()
        .map(|x| lift_to_primes_small(x.to_canonical_u128(), &setup.lifted))
        .collect();
    let coeffs_res: Vec<Vec<[u32; NUM_LIFT_PRIMES]>> = setup
        .coeffs
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|c| lift_to_primes_small(c.to_canonical_u128(), &setup.lifted))
                .collect()
        })
        .collect();

    let name = format!(
        "rs_expand_compare/lifted_horner_no_reduce_x{}_par",
        setup.count
    );
    c.bench_function(&name, |b| {
        b.iter(|| {
            let results: Vec<Vec<F>> = coeffs_res
                .par_iter()
                .map(|poly_res| eval_lifted_horner(poly_res, &point_res, &setup.lifted))
                .collect();
            black_box(&results);
        })
    });
}

#[cfg(feature = "parallel")]
fn bench_small_prime_ntt(c: &mut Criterion, setup: &Setup) {
    // Precompute coefficient residues in the NTT prime basis.
    let coeffs_res: Vec<Vec<[u32; NUM_NTT_PRIMES]>> = setup
        .coeffs
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|c| lift_to_primes_ntt(c.to_canonical_u128(), &setup.ntt))
                .collect()
        })
        .collect();

    let name = format!(
        "rs_expand_compare/small_prime_ntt_no_reduce_x{}_par",
        setup.count
    );
    c.bench_function(&name, |b| {
        b.iter(|| {
            let results: Vec<Vec<F>> = coeffs_res
                .par_iter()
                .map(|poly_res| eval_small_prime_ntt(poly_res, &setup.ntt))
                .collect();
            black_box(&results);
        })
    });
}

#[cfg(not(feature = "parallel"))]
fn bench_fft_baseline(_c: &mut Criterion, _setup: &Setup) {}
#[cfg(not(feature = "parallel"))]
fn bench_horner_fp128(_c: &mut Criterion, _setup: &Setup) {}
#[cfg(not(feature = "parallel"))]
fn bench_horner_crt(_c: &mut Criterion, _setup: &Setup) {}
#[cfg(not(feature = "parallel"))]
fn bench_lifted_horner(_c: &mut Criterion, _setup: &Setup) {}
#[cfg(not(feature = "parallel"))]
fn bench_small_prime_ntt(_c: &mut Criterion, _setup: &Setup) {}

fn bench_all(c: &mut Criterion) {
    let count = count_from_env();
    let setup = build_setup(count);

    bench_fft_baseline(c, &setup);
    bench_horner_fp128(c, &setup);
    bench_horner_crt(c, &setup);
    bench_lifted_horner(c, &setup);
    bench_small_prime_ntt(c, &setup);
}

criterion_group!(fft_crt_compare, bench_all);

fn main() {
    // Correctness check: all three methods must agree on a tiny input.
    // Skip via `CRT_BENCH_SKIP_CHECK=1`.
    if env::var("CRT_BENCH_SKIP_CHECK").is_err() {
        run_correctness_checks();
    }
    fft_crt_compare();
    Criterion::default().configure_from_args().final_summary();
}

fn run_correctness_checks() {
    let crt = CrtParams::new();

    // Sanity check: Garner round-trip on a handful of `u128` values.
    for v in [
        0u128,
        1u128,
        P - 1,
        0xdeadbeef_cafebabe_f00d1234_56789abcu128 % P,
        (P / 2).wrapping_add(17),
    ] {
        // Lift, multiply by 1 (identity), reconstruct → must return v mod p.
        let one_res: [u32; NUM_PRIMES] = std::array::from_fn(|_| 1);
        let v_res = lift_to_primes(v, &crt);
        let out = crt_mul_residues(v_res, one_res, &crt);
        assert_eq!(
            out, v,
            "CRT identity-mul round-trip failed for v=0x{v:032x}"
        );
    }

    // CRT-mul on random pairs must agree with native Fp128 mul.
    let mut rng = StdRng::seed_from_u64(0xdeadbeef);
    for _ in 0..256 {
        let a = F::sample(&mut rng);
        let b = F::sample(&mut rng);
        let native = (a * b).to_canonical_u128();
        let viacrt = crt_mul(a.to_canonical_u128(), b.to_canonical_u128(), &crt);
        assert_eq!(
            viacrt,
            native,
            "crt_mul disagrees with native mul for a=0x{:032x} b=0x{:032x}",
            a.to_canonical_u128(),
            b.to_canonical_u128()
        );
    }

    // Full polynomial evaluation agreement on a small batch.
    let setup = build_setup(2);
    let point_residues: Vec<[u32; NUM_PRIMES]> = setup
        .eval_points
        .iter()
        .map(|x| lift_to_primes(x.to_canonical_u128(), &setup.crt))
        .collect();
    for poly_idx in 0..setup.count {
        let coeffs_u128: Vec<u128> = setup.coeffs[poly_idx]
            .iter()
            .map(|c| c.to_canonical_u128())
            .collect();
        let fft = eval_fft(&setup.coeffs[poly_idx], &setup.domain);
        let horner = eval_horner_fp128(&setup.coeffs[poly_idx], &setup.eval_points);
        let crt_out = eval_horner_crt(&coeffs_u128, &point_residues, &setup.crt);
        assert_eq!(fft.len(), DOMAIN_SIZE);
        assert_eq!(
            horner, fft,
            "Horner/Fp128 disagrees with FFT at poly {poly_idx}"
        );
        if crt_out != fft {
            for (j, (a, b)) in crt_out.iter().zip(fft.iter()).enumerate() {
                if a != b {
                    panic!(
                        "Horner/CRT disagrees with FFT at poly {poly_idx} eval {j}: \
                         crt=0x{:032x}, fft=0x{:032x}",
                        a.to_canonical_u128(),
                        b.to_canonical_u128()
                    );
                }
            }
        }
    }
    // Internal-consistency check for the new cost-only variants:
    //   * lifted_horner must run and produce length-N output.
    //   * small_prime_ntt must run; the per-prime NTT of the constant
    //     polynomial `(c, 0, …, 0)` should be the constant vector `(c, c, …, c)`
    //     in each prime.
    let setup = build_setup(1);
    let point_res: Vec<[u32; NUM_LIFT_PRIMES]> = setup
        .eval_points
        .iter()
        .map(|x| lift_to_primes_small(x.to_canonical_u128(), &setup.lifted))
        .collect();
    let coeffs_res: Vec<[u32; NUM_LIFT_PRIMES]> = setup.coeffs[0]
        .iter()
        .map(|c| lift_to_primes_small(c.to_canonical_u128(), &setup.lifted))
        .collect();
    let lifted = eval_lifted_horner(&coeffs_res, &point_res, &setup.lifted);
    assert_eq!(lifted.len(), DOMAIN_SIZE);

    // Constant-polynomial sanity check for the NTT path.
    for k in 0..NUM_NTT_PRIMES {
        let mut constant = vec![0u32; DOMAIN_SIZE];
        constant[0] = 42;
        let out = setup.ntt.domains[k].forward(&constant);
        for (j, &v) in out.iter().enumerate() {
            assert_eq!(v, 42, "NTT of constant(42) at prime {k} position {j}");
        }
    }

    eprintln!(
        "[fft_crt_compare] correctness check passed\n  CRT primes ({}): {:?}\n  Lifted primes ({}): {:?}\n  NTT primes ({}): {:?}",
        NUM_PRIMES,
        crt.primes,
        NUM_LIFT_PRIMES,
        setup.lifted.primes,
        NUM_NTT_PRIMES,
        setup.ntt.primes,
    );
}
