//! Mixed-radix FFT over prime fields with smooth-order multiplicative subgroups.
//!
//! Iterative Cooley-Tukey DIT with pre-allocated ping-pong buffers and
//! precomputed twiddle tables. Zero heap allocations in the FFT hot path.
//!
//! Primary use case: FFT-based Reed-Solomon encoding over smooth
//! multiplicative subgroups of pseudo-Mersenne primes, e.g.
//! `p = 2^128 − 2355` with smooth order 14700 = 2² × 3 × 5² × 7².

use crate::{FieldCore, FromSmallInt, Invertible};

/// Compute `base^exp` by repeated squaring.
#[inline]
pub fn field_pow<F: FieldCore>(base: F, mut exp: u64) -> F {
    let mut result = F::one();
    let mut b = base;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * b;
        }
        b = b * b;
        exp >>= 1;
    }
    result
}

/// Compute `base^exp` for u128 exponents.
pub fn field_pow_u128<F: FieldCore>(base: F, mut exp: u128) -> F {
    let mut result = F::one();
    let mut b = base;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * b;
        }
        b = b * b;
        exp >>= 1;
    }
    result
}

fn smallest_prime_factor(n: usize) -> usize {
    if n <= 1 {
        return n;
    }
    for &p in &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        if n % p == 0 {
            return p;
        }
    }
    let mut i = 37;
    while i * i <= n {
        if n % i == 0 {
            return i;
        }
        i += 2;
    }
    n
}

fn factorize(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    while n > 1 {
        let p = smallest_prime_factor(n);
        factors.push(p);
        n /= p;
    }
    factors
}

/// Compute the mixed-radix digit-reversal permutation.
///
/// For `n = f_0 × f_1 × … × f_{s-1}`, an index `k` is written in mixed-radix
/// digits and the reversal reorders those digits.
fn digit_reversal_permutation(n: usize, factors: &[usize]) -> Vec<usize> {
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

/// Per-stage precomputed data for the FFT butterfly.
struct StageData<F> {
    /// Radix for this stage.
    r: usize,
    /// Block size before this stage.
    block: usize,
    /// `omega_r_pow[q] = omega_r^q` for `q = 0..r`.
    omega_r_pow: [F; 8],
    /// `twiddle_table[j] = omega_new_block^j` for `j = 0..block`.
    twiddle_table: Vec<F>,
    /// Winograd-style precomputed constants for this stage's radix. Layout:
    ///
    /// - `r == 3`: unused (the 2-mul DFT_3 reads `omega_r_pow[1]` and `[2]`).
    /// - `r == 5`: `[α/2, β/2, γ/2, δ/2, (α+β)/2, (γ+δ)/2]` (6 entries), where
    ///   α = ω+ω⁴, β = ω²+ω³, γ = ω−ω⁴, δ = ω²−ω³.
    /// - `r == 7`: `[α_{jk}/1]` for `(j,k)` in row-major `1..=3 × 1..=3`
    ///   (9 entries), then `[β_{jk}/1]` for the same indices (9 entries) —
    ///   18 entries total. The `/2` factor is absorbed into A_j/B_j during
    ///   the butterfly instead of into the constants.
    /// - other `r`: empty.
    ///
    /// These enable the Karatsuba-style 2/6/18-mul butterflies in place of
    /// the naive 4/16/36-mul matrix form.
    winograd: Vec<F>,
}

/// Build per-stage twiddle and root-of-unity tables.
fn precompute_stages<F: FieldCore + FromSmallInt + Invertible>(
    omega: F,
    n: usize,
    factors: &[usize],
) -> Vec<StageData<F>> {
    let mut stages = Vec::with_capacity(factors.len());
    let mut block = 1usize;

    for &r in factors.iter().rev() {
        debug_assert!(r <= 8, "radix {r} exceeds omega_r_pow capacity (max 8)");
        let new_block = block * r;
        let omega_new_block = field_pow(omega, (n / new_block) as u64);
        let omega_r = field_pow(omega_new_block, block as u64);

        let mut omega_r_pow = [F::one(); 8];
        for q in 1..r {
            omega_r_pow[q] = omega_r_pow[q - 1] * omega_r;
        }

        let mut twiddle_table = Vec::with_capacity(block);
        let mut tw = F::one();
        for _ in 0..block {
            twiddle_table.push(tw);
            tw = tw * omega_new_block;
        }

        let winograd = winograd_consts_for_radix::<F>(r, &omega_r_pow);

        stages.push(StageData {
            r,
            block,
            omega_r_pow,
            twiddle_table,
            winograd,
        });

        block = new_block;
    }
    stages
}

/// Precompute radix-specific constants used by the low-mul butterflies.
/// See the doc-comment on `StageData::winograd` for the exact layout.
fn winograd_consts_for_radix<F: FieldCore + FromSmallInt + Invertible>(
    r: usize,
    omega_r_pow: &[F; 8],
) -> Vec<F> {
    match r {
        5 => {
            let w1 = omega_r_pow[1];
            let w2 = omega_r_pow[2];
            let w3 = omega_r_pow[3];
            let w4 = omega_r_pow[4];
            let half = F::from_u64(2)
                .inv()
                .expect("2 is invertible in a non-binary field");
            let alpha_half = (w1 + w4) * half;
            let beta_half = (w2 + w3) * half;
            let gamma_half = (w1 - w4) * half;
            let delta_half = (w2 - w3) * half;
            let ab_half = alpha_half + beta_half; // = -1/2 (since α+β = -1)
            let gd_half = gamma_half + delta_half;
            vec![
                alpha_half, beta_half, gamma_half, delta_half, ab_half, gd_half,
            ]
        }
        7 => {
            // For j,k ∈ {1,2,3}:
            //   α_{jk} = (ω^{jk} + ω^{-jk}) / 2
            //   β_{jk} = (ω^{jk} − ω^{-jk}) / 2
            //
            // We fold the /2 factor into A_j / B_j at butterfly time, so
            // store α_{jk} and β_{jk} WITHOUT the /2.
            let w = omega_r_pow;
            // w[q] = ω^q for q = 0..6; we need positive and negative powers
            // mod 7.  ω^{-q} = ω^{7-q}.
            let pow = |q: isize| -> F {
                let qq = q.rem_euclid(7) as usize;
                w[qq]
            };
            let half = F::from_u64(2)
                .inv()
                .expect("2 is invertible in a non-binary field");
            let mut out = Vec::with_capacity(18);
            // alpha_{jk} = (ω^{jk} + ω^{-jk}) * half
            for j in 1..=3 {
                for k in 1..=3 {
                    let jk = (j * k) as isize;
                    out.push((pow(jk) + pow(-jk)) * half);
                }
            }
            // beta_{jk} = (ω^{jk} - ω^{-jk}) * half
            for j in 1..=3 {
                for k in 1..=3 {
                    let jk = (j * k) as isize;
                    out.push((pow(jk) - pow(-jk)) * half);
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Pre-allocated workspace for iterative mixed-radix FFT.
///
/// Stores two ping-pong buffers so the FFT hot path does zero heap
/// allocations.
struct FftWorkspace<F> {
    n: usize,
    buf_a: Vec<F>,
    buf_b: Vec<F>,
}

impl<F: FieldCore> FftWorkspace<F> {
    fn new(n: usize) -> Self {
        Self {
            n,
            buf_a: vec![F::zero(); n],
            buf_b: vec![F::zero(); n],
        }
    }

    /// Iterative mixed-radix Cooley-Tukey DIT FFT.
    ///
    /// 1. Apply digit-reversal permutation to input → buf_a.
    /// 2. Process stages from the last factor to the first (innermost radix first).
    /// 3. At each stage: in-place radix-r butterflies.
    ///
    /// Returns a reference to `buf_a` which holds the result.
    fn execute(&mut self, input: &[F], stages: &[StageData<F>], digit_rev: &[usize]) -> &[F] {
        let n = self.n;
        debug_assert_eq!(input.len(), n);

        for (i, &rev_i) in digit_rev.iter().enumerate() {
            self.buf_a[rev_i] = input[i];
        }

        self.butterfly_stages(stages);
        &self.buf_a[..n]
    }

    /// Like `execute` but reads input from `buf_b` (already filled by caller).
    fn execute_from_b(&mut self, stages: &[StageData<F>], digit_rev: &[usize]) -> &[F] {
        let n = self.n;

        for (i, &rev_i) in digit_rev.iter().enumerate() {
            self.buf_a[rev_i] = self.buf_b[i];
        }

        self.butterfly_stages(stages);
        &self.buf_a[..n]
    }

    fn butterfly_stages(&mut self, stages: &[StageData<F>]) {
        let n = self.n;
        for stage in stages {
            let r = stage.r;
            let block = stage.block;
            let new_block = block * r;
            let omega_r_pow = &stage.omega_r_pow;
            let twiddle_table = &stage.twiddle_table;

            for group_start in (0..n).step_by(new_block) {
                for (j, tw_entry) in twiddle_table.iter().enumerate() {
                    let base = group_start + j;

                    let mut x = [F::zero(); 8];
                    for (ki, xi) in x[..r].iter_mut().enumerate() {
                        *xi = self.buf_a[base + ki * block];
                    }

                    if j > 0 {
                        let tw = *tw_entry;
                        let tw2 = tw * tw;
                        match r {
                            2 => {
                                x[1] = x[1] * tw;
                            }
                            3 => {
                                x[1] = x[1] * tw;
                                x[2] = x[2] * tw2;
                            }
                            5 => {
                                let tw3 = tw2 * tw;
                                let tw4 = tw2 * tw2;
                                x[1] = x[1] * tw;
                                x[2] = x[2] * tw2;
                                x[3] = x[3] * tw3;
                                x[4] = x[4] * tw4;
                            }
                            7 => {
                                let tw3 = tw2 * tw;
                                let tw4 = tw2 * tw2;
                                let tw5 = tw4 * tw;
                                let tw6 = tw3 * tw3;
                                x[1] = x[1] * tw;
                                x[2] = x[2] * tw2;
                                x[3] = x[3] * tw3;
                                x[4] = x[4] * tw4;
                                x[5] = x[5] * tw5;
                                x[6] = x[6] * tw6;
                            }
                            _ => {
                                let mut tw_k = tw;
                                for xi in x[1..r].iter_mut() {
                                    *xi = *xi * tw_k;
                                    tw_k = tw_k * tw;
                                }
                            }
                        }
                    }

                    match r {
                        2 => {
                            self.buf_a[base] = x[0] + x[1];
                            self.buf_a[base + block] = x[0] - x[1];
                        }
                        3 => {
                            // 2-mul DFT_3 via identity 1 + ω + ω² = 0:
                            //   y₀ = x₀ + S
                            //   y₁ = x₀ + T
                            //   y₂ = x₀ − S − T
                            // where S = x₁ + x₂, T = ω·x₁ + ω²·x₂.
                            let w1 = omega_r_pow[1];
                            let w2 = omega_r_pow[2];
                            let s = x[1] + x[2];
                            let t = x[1] * w1 + x[2] * w2;
                            self.buf_a[base] = x[0] + s;
                            self.buf_a[base + block] = x[0] + t;
                            self.buf_a[base + 2 * block] = x[0] - s - t;
                        }
                        5 => {
                            // 6-mul DFT_5 (Karatsuba on symmetrized inputs).
                            // Constants layout (see winograd_consts_for_radix):
                            //   [α/2, β/2, γ/2, δ/2, (α+β)/2=−1/2, (γ+δ)/2].
                            let cc = &stage.winograd;
                            debug_assert_eq!(cc.len(), 6);
                            let a_h = cc[0];
                            let b_h = cc[1];
                            let g_h = cc[2];
                            let d_h = cc[3];
                            let ab_h = cc[4];
                            let gd_h = cc[5];

                            let a = x[1] + x[4];
                            let b = x[2] + x[3];
                            let c = x[1] - x[4];
                            let d = x[2] - x[3];

                            // P-block: P1 = A·α/2 + B·β/2, P2 = A·β/2 + B·α/2
                            //   via Karatsuba: k1=A·α/2, k2=B·β/2, k3=(A+B)·(α+β)/2
                            let k1 = a * a_h;
                            let k2 = b * b_h;
                            let k3 = (a + b) * ab_h;
                            let p1 = k1 + k2;
                            let p2 = k3 - k1 - k2;

                            // Q-block: Q1 = C·γ/2 + D·δ/2, Q2 = C·δ/2 − D·γ/2
                            //   via Karatsuba with complex-mul equivalence:
                            //     m1 = C·γ/2, m2 = D·δ/2, m3 = (C − D)·(γ+δ)/2
                            //     Q1 = m1 + m2
                            //     Q2 = m3 − m1 + m2
                            let m1 = c * g_h;
                            let m2 = d * d_h;
                            let m3 = (c - d) * gd_h;
                            let q1 = m1 + m2;
                            let q2 = m3 - m1 + m2;

                            self.buf_a[base] = x[0] + a + b;
                            self.buf_a[base + block] = x[0] + p1 + q1;
                            self.buf_a[base + 2 * block] = x[0] + p2 + q2;
                            self.buf_a[base + 3 * block] = x[0] + p2 - q2;
                            self.buf_a[base + 4 * block] = x[0] + p1 - q1;
                        }
                        7 => {
                            // 18-mul DFT_7 via conjugate-pair symmetry.
                            // Constants: 9× α_{jk} then 9× β_{jk}
                            //   (row-major j,k ∈ {1,2,3}).
                            // Note: α_{jk} / β_{jk} stored WITHOUT the /2
                            //   factor; the /2 is absorbed into (A_j, B_j)
                            //   that are used here (since A_j = x[j]+x[7-j]
                            //   is 2·avg, the formula comes out correct
                            //   only if we pre-divide by 2, but we can also
                            //   leave out the /2 on both sides and factor
                            //   it into the butterfly structure as a common
                            //   scalar — see below).
                            //
                            // Derivation:
                            //   x[j]·ω^{jk} + x[7-j]·ω^{-jk}
                            //     = (A_j + B_j)/2 · ω^{jk}
                            //     + (A_j − B_j)/2 · ω^{-jk}
                            //     = A_j·(ω^{jk}+ω^{-jk})/2
                            //     + B_j·(ω^{jk}−ω^{-jk})/2
                            //     = A_j · α_{jk} + B_j · β_{jk}   (with /2)
                            //
                            // So S_k = Σ A_j·α_{jk} already includes the
                            // /2 factor because our stored α_{jk} =
                            // (ω^{jk}+ω^{-jk})·(1/2).  Good.
                            let cc = &stage.winograd;
                            debug_assert_eq!(cc.len(), 18);

                            let a1 = x[1] + x[6];
                            let a2 = x[2] + x[5];
                            let a3 = x[3] + x[4];
                            let b1 = x[1] - x[6];
                            let b2 = x[2] - x[5];
                            let b3 = x[3] - x[4];

                            // α table row-major (j-1)*3 + (k-1).
                            let s1 = a1 * cc[0] + a2 * cc[3] + a3 * cc[6]; // k=1
                            let s2 = a1 * cc[1] + a2 * cc[4] + a3 * cc[7]; // k=2
                            let s3 = a1 * cc[2] + a2 * cc[5] + a3 * cc[8]; // k=3

                            // β table starts at offset 9.
                            let t1 = b1 * cc[9] + b2 * cc[12] + b3 * cc[15];
                            let t2 = b1 * cc[10] + b2 * cc[13] + b3 * cc[16];
                            let t3 = b1 * cc[11] + b2 * cc[14] + b3 * cc[17];

                            self.buf_a[base] = x[0] + a1 + a2 + a3;
                            self.buf_a[base + block] = x[0] + s1 + t1;
                            self.buf_a[base + 2 * block] = x[0] + s2 + t2;
                            self.buf_a[base + 3 * block] = x[0] + s3 + t3;
                            self.buf_a[base + 4 * block] = x[0] + s3 - t3;
                            self.buf_a[base + 5 * block] = x[0] + s2 - t2;
                            self.buf_a[base + 6 * block] = x[0] + s1 - t1;
                        }
                        _ => {
                            for (q, &wq) in omega_r_pow[..r].iter().enumerate() {
                                let mut val = x[0];
                                let mut w = wq;
                                for &xp in &x[1..r] {
                                    val += xp * w;
                                    w = w * wq;
                                }
                                self.buf_a[base + q * block] = val;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Mixed-radix FFT domain backed by a smooth-order multiplicative subgroup.
///
/// Stores immutable domain parameters (roots of unity, digit-reversal
/// permutation, precomputed twiddle tables). The actual FFT computation
/// uses thread-local workspaces, so the domain is `Sync` and can be shared
/// across rayon tasks.
pub struct SmoothDomain<F> {
    /// Domain size.
    pub n: usize,
    /// Primitive `n`-th root of unity.
    pub omega: F,
    /// `n^{-1}` in the field.
    n_inv: F,
    /// Digit-reversal permutation table.
    digit_rev: Vec<usize>,
    /// Precomputed per-stage data for the forward transform.
    fwd_stages: Vec<StageData<F>>,
    /// Precomputed per-stage data for the inverse transform.
    inv_stages: Vec<StageData<F>>,
}

impl<F: FieldCore + FromSmallInt + Invertible + std::fmt::Debug> SmoothDomain<F> {
    /// Create from a primitive `n`-th root of unity.
    ///
    /// Precomputes digit-reversal permutation and per-stage twiddle tables
    /// for both forward and inverse transforms. The FFT hot path allocates
    /// only two working buffers per call (via `FftWorkspace`).
    ///
    /// # Panics
    ///
    /// Panics if `omega` is zero or if `n` is not invertible in the field.
    pub fn new(omega: F, n: usize) -> Self {
        debug_assert_eq!(field_pow(omega, n as u64), F::one(), "omega^n must be 1");
        for &p in &[2usize, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
            if n % p == 0 {
                debug_assert_ne!(
                    field_pow(omega, (n / p) as u64),
                    F::one(),
                    "omega must be a primitive {n}-th root (order divides n/{p})"
                );
            }
        }
        let omega_inv = omega.inv().expect("omega must be nonzero");
        let n_inv = F::from_u64(n as u64)
            .inv()
            .expect("n must be invertible in field");
        let factors = factorize(n);
        let digit_rev = digit_reversal_permutation(n, &factors);
        let fwd_stages = precompute_stages(omega, n, &factors);
        let inv_stages = precompute_stages(omega_inv, n, &factors);
        Self {
            n,
            omega,
            n_inv,
            digit_rev,
            fwd_stages,
            inv_stages,
        }
    }

    /// Forward DFT: `Y[k] = Σ_{j=0}^{n-1} x[j] · ω^{jk}`.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != n`.
    pub fn forward(&self, input: &[F]) -> Vec<F> {
        assert_eq!(input.len(), self.n);
        let mut ws = FftWorkspace::new(self.n);
        ws.execute(input, &self.fwd_stages, &self.digit_rev)
            .to_vec()
    }

    /// Inverse DFT: `x[j] = (1/n) · Σ_{k=0}^{n-1} Y[k] · ω^{-jk}`.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != n`.
    pub fn inverse(&self, input: &[F]) -> Vec<F> {
        assert_eq!(input.len(), self.n);
        let mut ws: FftWorkspace<F> = FftWorkspace::new(self.n);
        let mut result = ws
            .execute(input, &self.inv_stages, &self.digit_rev)
            .to_vec();
        for v in &mut result {
            *v = *v * self.n_inv;
        }
        result
    }

    /// Evaluate polynomial (given as coefficients, zero-padded to `n`) at
    /// the coset `{shift · ω^i | i = 0, …, n−1}`.
    ///
    /// Twists each coefficient `c_i` by `shift^i` then applies a forward DFT.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs.len() > n`.
    pub fn coset_forward(&self, coeffs: &[F], shift: F) -> Vec<F> {
        assert!(coeffs.len() <= self.n);
        let mut ws: FftWorkspace<F> = FftWorkspace::new(self.n);
        let buf = &mut ws.buf_b[..self.n];
        let mut tw = F::one();
        for (i, &c) in coeffs.iter().enumerate() {
            buf[i] = c * tw;
            tw = tw * shift;
        }
        for v in buf[coeffs.len()..].iter_mut() {
            *v = F::zero();
        }
        ws.execute_from_b(&self.fwd_stages, &self.digit_rev)
            .to_vec()
    }

    /// Batch RS-extend: runs inverse FFT once, then `blowup−1` coset
    /// forward FFTs, reusing a single `FftWorkspace` throughout.
    ///
    /// This is the fast path for `rs_extend_fft` — avoids creating a new
    /// workspace per transform.
    ///
    /// # Panics
    ///
    /// Panics if `evals.len() != n`.
    pub fn rs_extend_batch(&self, evals: &[F], omega_n: F, blowup: usize) -> Vec<F> {
        let k = self.n;
        assert_eq!(evals.len(), k);

        let mut ws: FftWorkspace<F> = FftWorkspace::new(self.n);

        let mut coeffs = ws
            .execute(evals, &self.inv_stages, &self.digit_rev)
            .to_vec();
        for v in &mut coeffs {
            *v = *v * self.n_inv;
        }

        let mut extension = Vec::with_capacity(k * (blowup - 1));
        for j in 1..blowup {
            let shift = field_pow(omega_n, j as u64);
            let buf = &mut ws.buf_b[..k];
            let mut tw = F::one();
            for (i, &c) in coeffs.iter().enumerate() {
                buf[i] = c * tw;
                tw = tw * shift;
            }
            let result = ws.execute_from_b(&self.fwd_stages, &self.digit_rev);
            extension.extend_from_slice(result);
        }
        extension
    }
}

/// Compute a primitive `n`-th root of unity from a multiplicative generator.
///
/// Returns `g^{(p−1)/n}` which has exact multiplicative order `n`.
///
/// # Panics
///
/// Panics if `n` does not divide `p − 1`.
pub fn primitive_root_of_unity<F: FieldCore>(g: F, p_minus_1: u128, n: usize) -> F {
    assert_eq!(
        p_minus_1 % (n as u128),
        0,
        "n={n} must divide p-1={p_minus_1}"
    );
    let exp = p_minus_1 / (n as u128);
    field_pow_u128(g, exp)
}

/// RS-extend evaluations from a size-`k` subgroup to `blowup` cosets
/// using the coset FFT approach.
///
/// Given `k` evaluations on a base subgroup `K = {β^i}` where `β = ω_n^blowup`,
/// computes extension evaluations on the remaining `blowup − 1` cosets
/// `{ω_n^j · K}` for `j = 1, …, blowup−1`.
///
/// Returns `k · (blowup − 1)` extension elements (excludes the original data).
///
/// # Panics
///
/// Panics if `evals.len() != domain_k.n`.
pub fn rs_extend_fft<F: FieldCore + FromSmallInt + Invertible + std::fmt::Debug>(
    evals: &[F],
    domain_k: &SmoothDomain<F>,
    omega_n: F,
    blowup: usize,
) -> Vec<F> {
    domain_k.rs_extend_batch(evals, omega_n, blowup)
}

#[cfg(test)]
mod test_support {
    //! Prime-agnostic helpers shared by per-prime FFT parity tests.
    //!
    //! The two protocol primes (`Prime128Offset2355`, `Prime128OffsetA7F7`)
    //! have different smooth-subgroup factorizations, but the parity
    //! properties under test (FFT vs naive DFT, forward/inverse roundtrip,
    //! RS-extend consistency) are identical. Factor them out here so the
    //! per-prime modules carry only the constants and size lattices that
    //! actually differ.
    use super::*;
    use crate::FromSmallInt;
    use std::fmt::Debug;
    use std::ops::{AddAssign, MulAssign};

    /// Find a primitive `n`-th root of unity in `F` by scanning small bases.
    ///
    /// Robust against the case where a small base lands in a strict subgroup
    /// (e.g. `g = 2` is a quadratic residue mod `Prime128OffsetA7F7`, so
    /// `2^((p-1)/n)` has order `n/2` instead of `n`). Verifies primitivity
    /// against every distinct prime factor of `n`, not a fixed `{2, 3}` prefix.
    pub(super) fn find_primitive_nth_root<F>(p_minus_1: u128, n: usize) -> F
    where
        F: FieldCore + FromSmallInt,
    {
        assert_eq!(p_minus_1 % (n as u128), 0, "n={n} must divide p - 1");
        let exp = p_minus_1 / (n as u128);
        let mut prime_factors = factorize(n);
        prime_factors.sort_unstable();
        prime_factors.dedup();

        for &g in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] {
            let candidate = field_pow_u128(F::from_u64(g), exp);
            if field_pow(candidate, n as u64) != F::one() {
                continue;
            }
            let primitive = prime_factors
                .iter()
                .all(|&q| field_pow(candidate, (n / q) as u64) != F::one());
            if primitive {
                return candidate;
            }
        }
        panic!("no primitive {n}-th root of unity found in scanned bases");
    }

    /// O(n^2) naive DFT, used as oracle for the iterative FFT under test.
    fn naive_dft<F: FieldCore>(input: &[F], omega: F) -> Vec<F> {
        let n = input.len();
        let mut out = vec![F::zero(); n];
        for (k, ok) in out.iter_mut().enumerate() {
            for (j, &xj) in input.iter().enumerate() {
                *ok += xj * field_pow(omega, (j * k) as u64);
            }
        }
        out
    }

    /// For each `n` in `sizes` that divides `p - 1`, assert the iterative
    /// FFT matches the naive DFT on a deterministic input vector.
    pub(super) fn assert_fft_matches_naive_dft<F>(p_minus_1: u128, sizes: &[usize])
    where
        F: FieldCore + FromSmallInt + Invertible + Debug,
    {
        for &n in sizes {
            if p_minus_1 % (n as u128) != 0 {
                continue;
            }
            let omega = find_primitive_nth_root::<F>(p_minus_1, n);
            let input: Vec<F> = (0..n).map(|i| F::from_u64((i + 1) as u64)).collect();
            let expected = naive_dft(&input, omega);

            let factors = factorize(n);
            let digit_rev = digit_reversal_permutation(n, &factors);
            let stages = precompute_stages(omega, n, &factors);
            let mut ws: FftWorkspace<F> = FftWorkspace::new(n);
            let got = ws.execute(&input, &stages, &digit_rev).to_vec();
            assert_eq!(got, expected, "FFT mismatch for n={n}");
        }
    }

    /// `forward(inverse(x)) == x` over a smooth domain of order `n`.
    pub(super) fn assert_forward_inverse_roundtrip<F>(p_minus_1: u128, n: usize)
    where
        F: FieldCore + FromSmallInt + Invertible + Debug,
    {
        let omega = find_primitive_nth_root::<F>(p_minus_1, n);
        let domain = SmoothDomain::new(omega, n);
        let input: Vec<F> = (0..n).map(|i| F::from_u64(i as u64 + 1)).collect();
        let transformed = domain.forward(&input);
        let recovered = domain.inverse(&transformed);
        assert_eq!(input, recovered);
    }

    /// `rs_extend_fft` matches direct evaluation of the interpolating
    /// polynomial on each of the `blowup - 1` extension cosets.
    pub(super) fn assert_rs_extend_consistency<F>(p_minus_1: u128, k: usize, blowup: usize)
    where
        F: FieldCore + FromSmallInt + Invertible + Debug + AddAssign + MulAssign,
    {
        let n = k * blowup;
        let omega_n = find_primitive_nth_root::<F>(p_minus_1, n);
        let omega_k = field_pow(omega_n, blowup as u64);
        let domain_k = SmoothDomain::new(omega_k, k);

        let evals: Vec<F> = (0..k).map(|i| F::from_u64((i * 7 + 3) as u64)).collect();
        let coeffs = domain_k.inverse(&evals);
        let extension = rs_extend_fft(&evals, &domain_k, omega_n, blowup);
        assert_eq!(extension.len(), k * (blowup - 1));

        for j in 1..blowup {
            for i in 0..k {
                let point = field_pow(omega_n, j as u64) * field_pow(omega_k, i as u64);
                let mut expected = F::zero();
                let mut x_pow = F::one();
                for &c in &coeffs {
                    expected += c * x_pow;
                    x_pow *= point;
                }
                assert_eq!(
                    extension[(j - 1) * k + i],
                    expected,
                    "mismatch at coset {j}, position {i}"
                );
            }
        }
    }
}

#[cfg(test)]
mod prime_2355_tests {
    //! `Prime128Offset2355` (`p = 2^128 - 2355`) has smooth multiplicative
    //! subgroup of order `14_700 = 2^2 * 3 * 5^2 * 7^2`, drawing sizes from
    //! the `{2, 3, 5, 7}` lattice.
    use super::test_support::*;
    use crate::algebra::Prime128Offset2355;

    type F = Prime128Offset2355;
    const P_MINUS_1: u128 = 0xfffffffffffffffffffffffffffff6cd - 1;

    #[test]
    fn primitive_root_has_correct_order() {
        // `find_primitive_nth_root` already asserts both `omega^n == 1`
        // and `omega^(n/q) != 1` for every prime factor `q | n`; reaching
        // this point per `n` *is* the assertion.
        for &n in &[
            2, 3, 4, 5, 6, 7, 10, 12, 14, 15, 20, 21, 25, 28, 30, 35, 42, 49, 50, 60, 70, 75, 84,
            98, 100, 105, 140, 147, 150, 175, 196, 210, 245, 294, 300, 350, 420, 490, 525, 588,
            700, 735, 980, 1050, 1225, 1470, 2100, 2450, 2940, 3675, 4900, 7350, 14700,
        ] {
            if P_MINUS_1 % (n as u128) != 0 {
                continue;
            }
            let _ = find_primitive_nth_root::<F>(P_MINUS_1, n);
        }
    }

    #[test]
    fn small_fft_matches_naive_dft() {
        assert_fft_matches_naive_dft::<F>(
            P_MINUS_1,
            &[2, 3, 4, 5, 6, 7, 10, 12, 14, 15, 20, 21, 25, 28, 42, 49, 50],
        );
    }

    #[test]
    fn forward_inverse_roundtrip_300() {
        assert_forward_inverse_roundtrip::<F>(P_MINUS_1, 300);
    }

    #[test]
    fn forward_inverse_roundtrip_1470() {
        assert_forward_inverse_roundtrip::<F>(P_MINUS_1, 1470);
    }

    #[test]
    fn rs_extend_consistency() {
        // k = 300 = 2^2 * 3 * 5^2, blowup = 7, so n = 2_100 | 14_700.
        assert_rs_extend_consistency::<F>(P_MINUS_1, 300, 7);
    }
}

#[cfg(test)]
mod prime_a7f7_tests {
    //! `Prime128OffsetA7F7` (`p = 2^128 - 2^32 + 22537`) has smooth
    //! multiplicative subgroup of order `2^3 * 3^7 = 17_496`, with a pure
    //! radix-3 substructure of order `3^7 = 2_187`. Sizes are drawn from
    //! the `{2, 3}` lattice instead of `{2, 3, 5, 7}`.
    use super::test_support::*;
    use crate::algebra::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;
    const P_MINUS_1: u128 = 0xffffffffffffffffffffffff00005809 - 1;

    #[test]
    fn small_fft_matches_naive_dft() {
        assert_fft_matches_naive_dft::<F>(
            P_MINUS_1,
            &[2, 3, 6, 8, 9, 18, 24, 27, 54, 81, 162, 243, 486, 729],
        );
    }

    #[test]
    fn forward_inverse_roundtrip_243() {
        assert_forward_inverse_roundtrip::<F>(P_MINUS_1, 243);
    }

    #[test]
    fn forward_inverse_roundtrip_1458() {
        assert_forward_inverse_roundtrip::<F>(P_MINUS_1, 1458);
    }

    #[test]
    fn forward_inverse_roundtrip_2187() {
        assert_forward_inverse_roundtrip::<F>(P_MINUS_1, 2187);
    }

    #[test]
    fn rs_extend_consistency() {
        // k = 243 (= 3^5), blowup = 9 (= 3^2), n = 3^7 = 2_187 | 17_496.
        assert_rs_extend_consistency::<F>(P_MINUS_1, 243, 9);
    }
}
