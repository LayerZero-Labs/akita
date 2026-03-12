//! Operation-centric polynomial trait for the Hachi commitment scheme.
//!
//! [`HachiPolyOps`] exposes the four operations the Hachi commit/prove paths
//! need from a polynomial, rather than raw coefficient access.  Each
//! implementation handles every operation in its own optimal way:
//!
//! - [`DensePoly`] — standard dense algorithms (decompose + NTT matvec).
//! - [`OneHotPoly`] — sparse monomial tricks, avoids all inner ring
//!   multiplications.
//!
//! # Extensibility
//!
//! This trait is coupled to power-of-2 cyclotomic rings
//! ([`CyclotomicRing<F, D>`]).  When non-power-of-2 rings are added, the trait
//! signature will change.  Additional operation methods may be added as the
//! protocol evolves.

use crate::algebra::fields::wide::HasWide;
use crate::algebra::ring::sparse_challenge::SparseChallenge;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::onehot::{
    inner_ajtai_onehot_wide, map_onehot_to_sparse_blocks, SparseBlockEntry,
};
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
use crate::protocol::commitment::utils::linear::{
    decompose_rows_i8, mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_i8,
};
use crate::{cfg_fold_reduce, cfg_into_iter, cfg_iter, CanonicalField, FieldCore};
use std::array::from_fn;
use std::marker::PhantomData;

#[cfg(target_arch = "aarch64")]
use crate::algebra::ntt::neon;

#[cfg(target_arch = "aarch64")]
mod decompose_fold_neon;

/// Precomputed constants for balanced base-b decomposition.
struct DecomposeParams {
    half_q: u128,
    q: u128,
    mask: i128,
    half_b: i128,
    b_val: i128,
    log_basis: u32,
}

/// Decompose all D coefficients of a ring element into balanced base-b digits,
/// storing results in digit-major order for subsequent SIMD scatter.
///
/// Uses K=3 interleaved carry chains to saturate ALU throughput (3x ILP gain
/// over processing one coefficient at a time on out-of-order cores).
///
/// `digit_buf` is `[num_digits][D]` in i8, OVERWRITTEN (not accumulated).
#[inline(never)]
fn decompose_ring_interleaved<F: CanonicalField, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    digit_buf: &mut [Vec<i8>],
    num_digits: usize,
    p: &DecomposeParams,
) {
    let bulk_end = D - (D % 3);

    for base in (0..bulk_end).step_by(3) {
        let mut c0 = to_signed(ring.coeffs[base].to_canonical_u128(), p);
        let mut c1 = to_signed(ring.coeffs[base + 1].to_canonical_u128(), p);
        let mut c2 = to_signed(ring.coeffs[base + 2].to_canonical_u128(), p);

        for plane in digit_buf.iter_mut().take(num_digits) {
            let d0 = extract_balanced_digit(&mut c0, p);
            let d1 = extract_balanced_digit(&mut c1, p);
            let d2 = extract_balanced_digit(&mut c2, p);
            plane[base] = d0 as i8;
            plane[base + 1] = d1 as i8;
            plane[base + 2] = d2 as i8;
        }
    }

    for idx in bulk_end..D {
        let mut c = to_signed(ring.coeffs[idx].to_canonical_u128(), p);
        for plane in digit_buf.iter_mut().take(num_digits) {
            plane[idx] = extract_balanced_digit(&mut c, p) as i8;
        }
    }
}

#[inline(never)]
fn decompose_ring_single_digit<F: CanonicalField, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    digit_plane: &mut [i8; D],
    p: &DecomposeParams,
) {
    for (dst, coeff) in digit_plane.iter_mut().zip(ring.coeffs.iter()) {
        let centered = to_signed(coeff.to_canonical_u128(), p);
        debug_assert!(
            centered >= -(1i128 << (p.log_basis - 1)) && centered < (1i128 << (p.log_basis - 1))
        );
        *dst = centered as i8;
    }
}

#[inline(always)]
fn to_signed(canonical: u128, p: &DecomposeParams) -> i128 {
    if canonical > p.half_q {
        -((p.q - canonical) as i128)
    } else {
        canonical as i128
    }
}

#[inline(always)]
fn try_centered_i8<F: CanonicalField>(coeff: F, q: u128, half_q: u128) -> Option<i8> {
    let canonical = coeff.to_canonical_u128();
    let centered = if canonical > half_q {
        -((q - canonical) as i128)
    } else {
        canonical as i128
    };
    if (i8::MIN as i128..=i8::MAX as i128).contains(&centered) {
        Some(centered as i8)
    } else {
        None
    }
}

fn try_small_i8_cache_from_ring_coeffs<F: CanonicalField, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
) -> Option<Vec<[i8; D]>> {
    let q = (-F::one()).to_canonical_u128() + 1;
    let half_q = q / 2;
    let mut out = Vec::with_capacity(coeffs.len());

    for ring in coeffs {
        let mut digits = [0i8; D];
        for (dst, coeff) in digits.iter_mut().zip(ring.coeffs.iter()) {
            *dst = try_centered_i8(*coeff, q, half_q)?;
        }
        out.push(digits);
    }

    Some(out)
}

#[inline(always)]
fn extract_balanced_digit(c: &mut i128, p: &DecomposeParams) -> i32 {
    let d = *c & p.mask;
    let balanced = if d >= p.half_b { d - p.b_val } else { d };
    *c = (*c - balanced) >> p.log_basis;
    balanced as i32
}

/// Scalar sparse-multiply-accumulate: accumulate `challenge * digit_plane`
/// into `acc` using the rotate-and-add formulation.
///
/// `digit_plane` is `[i8; D]`, `acc` is `[i32; D]`.
/// Each challenge term rotates the digit plane and adds/subtracts contiguously.
fn sparse_mul_acc_scalar<const D: usize>(
    digit_plane: &[i8],
    challenge: &SparseChallenge,
    acc: &mut [i32; D],
) {
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let p = pos as usize;
        let split = D - p;
        if coeff > 0 {
            for i in 0..split {
                acc[i + p] += digit_plane[i] as i32;
            }
            for i in split..D {
                acc[i - split] -= digit_plane[i] as i32;
            }
        } else {
            for i in 0..split {
                acc[i + p] -= digit_plane[i] as i32;
            }
            for i in split..D {
                acc[i - split] += digit_plane[i] as i32;
            }
        }
    }
}

/// Dispatch to NEON or scalar sparse-multiply-accumulate.
#[inline(always)]
fn sparse_mul_acc<const D: usize>(
    digit_plane: &[i8],
    challenge: &SparseChallenge,
    acc: &mut [i32; D],
) {
    #[cfg(target_arch = "aarch64")]
    {
        if neon::use_neon_ntt() {
            unsafe {
                decompose_fold_neon::sparse_mul_acc_neon(
                    digit_plane.as_ptr(),
                    acc.as_mut_ptr(),
                    D,
                    &challenge.positions,
                    &challenge.coeffs,
                );
            }
            return;
        }
    }
    sparse_mul_acc_scalar::<D>(digit_plane, challenge, acc);
}

#[inline(always)]
fn accum_onehot_coeff<const D: usize>(
    acc: &mut [i32; D],
    coeff_idx: usize,
    challenge: &SparseChallenge,
) {
    debug_assert!(coeff_idx < D);
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let target = coeff_idx + pos as usize;
        if target < D {
            acc[target] += coeff as i32;
        } else {
            acc[target - D] -= coeff as i32;
        }
    }
}

#[inline(always)]
fn accum_onehot_entry<const D: usize>(
    acc: &mut [i32; D],
    entry: &SparseBlockEntry,
    challenge: &SparseChallenge,
) {
    for &coeff_idx in &entry.nonzero_coeffs {
        accum_onehot_coeff::<D>(acc, coeff_idx, challenge);
    }
}

fn signed_accum_to_ring<F: CanonicalField, const D: usize>(
    coeff_accum: [i32; D],
    modulus: u128,
) -> CyclotomicRing<F, D> {
    let coeffs = from_fn(|k| {
        let v = coeff_accum[k];
        if v >= 0 {
            F::from_canonical_u128_reduced(v as u128)
        } else {
            F::from_canonical_u128_reduced(modulus - ((-v) as u128))
        }
    });
    CyclotomicRing::from_coefficients(coeffs)
}

fn recompose_commit_inner_blocks<F: CanonicalField, const D: usize>(
    t_hat_blocks: &[Vec<[i8; D]>],
    num_digits_open: usize,
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError> {
    if num_digits_open == 0 {
        return Err(HachiError::InvalidSetup(
            "num_digits_open must be nonzero when recomposing commit witness".to_string(),
        ));
    }
    t_hat_blocks
        .iter()
        .map(|block| {
            if block.len() % num_digits_open != 0 {
                return Err(HachiError::InvalidSetup(format!(
                    "t_hat block has {} planes, expected a multiple of num_digits_open={num_digits_open}",
                    block.len()
                )));
            }
            Ok(block
                .chunks(num_digits_open)
                .map(|digits| CyclotomicRing::gadget_recompose_pow2_i8(digits, log_basis))
                .collect())
        })
        .collect()
}

/// Prover-side output of the inner Ajtai commit step.
pub struct CommitInnerWitness<F: FieldCore, const D: usize> {
    /// Undecomposed `t_i = A * s_i` rows, grouped by block.
    pub t: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Decomposed `t_hat_i = G^{-1}(t_i)` rows, grouped by block.
    pub t_hat: Vec<Vec<[i8; D]>>,
}

/// Operations the Hachi commitment scheme needs from a polynomial.
///
/// The four methods correspond to the four places in commit/prove that consume
/// polynomial data.  Implementations decide *how* to carry out each operation
/// (dense decompose + NTT, sparse monomial tricks, streaming, etc.).
pub trait HachiPolyOps<F: FieldCore, const D: usize>: Clone + Send + Sync {
    /// Per-polynomial cache type for the A-matrix commit path.
    ///
    /// `DensePoly` uses `NttSlotCache<D>` (CRT+NTT of A for dense mat-vec).
    /// `OneHotPoly` uses `()` (one-hot commit bypasses NTT entirely).
    type CommitCache: Send + Sync;

    /// Total number of ring elements in the polynomial.
    fn num_ring_elems(&self) -> usize;

    /// **Op 1 — prove: ring-space evaluation.**
    ///
    /// Computes the global weighted sum `y = Σᵢ scalars[i] · self[i]`.
    ///
    /// `scalars` has length >= `num_ring_elems`; excess entries are ignored.
    fn evaluate_ring(&self, scalars: &[F]) -> CyclotomicRing<F, D>;

    /// **Op 2 — prove: per-block fold.**
    ///
    /// For each contiguous block of `block_len` ring elements, computes
    /// `Σⱼ scalars[j] · self[i·block_len + j]`.
    ///
    /// Returns one ring element per block (total `ceil(num_ring_elems / block_len)`).
    /// `scalars` has length `block_len`.
    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>>;

    /// Fused fold + evaluation in a single pass over the polynomial.
    ///
    /// `eval_outer_scalars` is the per-block weight vector `b` (size `num_blocks`).
    /// `fold_scalars` is the per-element-in-block weight vector `a` (size `block_len`).
    ///
    /// The full evaluation scalars factor as `outer_weights[i*block_len + j] = b[i] * a[j]`,
    /// so `eval = Σ_i b[i] * fold(a)[i]` — derived from the fold result without
    /// materializing the full `2^(m_vars + r_vars)` weight vector.
    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let folded = self.fold_blocks(fold_scalars, block_len);
        let eval = folded
            .iter()
            .zip(eval_outer_scalars.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + f_i.scale(s_i)
            });
        (eval, folded)
    }

    /// **Op 3 — prove: decompose + challenge-fold.**
    ///
    /// For each block of `block_len` ring elements:
    /// 1. Decompose: `sᵢ = G⁻¹(blockᵢ)` via `balanced_decompose_pow2(num_digits, log_basis)`.
    /// 2. Accumulate: `z += cᵢ ⊗ sᵢ` (sparse challenge multiplication).
    ///
    /// Returns `z` of length `block_len · num_digits`.
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Vec<CyclotomicRing<F, D>>;

    /// **Op 4 — commit: per-block inner Ajtai.**
    ///
    /// For each block of `block_len` ring elements:
    /// 1. `sᵢ = G⁻¹(blockᵢ)` with `num_digits_commit` levels.
    /// 2. `tᵢ = A · sᵢ` (matrix-vector multiply via NTT cache or sparse path).
    /// 3. `t̂ᵢ = G⁻¹(tᵢ)` with `num_digits_open` levels (t has full-field
    ///    coefficients regardless of s's digit count).
    ///
    /// Returns one `t̂ᵢ` vector per block as `[i8; D]` digit planes.
    ///
    /// # Errors
    ///
    /// Returns an error if the cached matrix-vector multiply fails.
    fn commit_inner(
        &self,
        a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<[i8; D]>>, HachiError>;

    /// Like [`commit_inner`](Self::commit_inner), but also preserves the
    /// undecomposed `t_i` rows for prover-side consumers that would otherwise
    /// need to recompose `t_hat`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`commit_inner`](Self::commit_inner) fails or if the
    /// resulting `t_hat` blocks cannot be recomposed into full `t_i` rows.
    fn commit_inner_witness(
        &self,
        a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<CommitInnerWitness<F, D>, HachiError>
    where
        F: CanonicalField,
    {
        let t_hat = self.commit_inner(
            a_matrix,
            ntt_a,
            block_len,
            num_digits_commit,
            num_digits_open,
            log_basis,
        )?;
        let t = recompose_commit_inner_blocks::<F, D>(&t_hat, num_digits_open, log_basis)?;
        Ok(CommitInnerWitness { t, t_hat })
    }
}

/// Dense polynomial: all ring coefficients materialized in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePoly<F: FieldCore, const D: usize> {
    /// Ring coefficients in sequential block order.
    pub coeffs: Vec<CyclotomicRing<F, D>>,
    small_i8_coeffs: Option<Vec<[i8; D]>>,
}

impl<F: FieldCore + CanonicalField, const D: usize> DensePoly<F, D> {
    /// Pack field-element evaluations into ring elements.
    ///
    /// The first `α = log₂(D)` variables become coefficient slots within each
    /// ring element; the remaining variables index ring elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `D` is not a power of two, `num_vars < log₂(D)`, or
    /// `evals.len() != 2^num_vars`.
    pub fn from_field_evals(num_vars: usize, evals: &[F]) -> Result<Self, HachiError> {
        if D == 0 || !D.is_power_of_two() {
            return Err(HachiError::InvalidInput(format!(
                "ring degree D={D} is not a power of two"
            )));
        }
        let alpha = D.trailing_zeros() as usize;
        if num_vars < alpha {
            return Err(HachiError::InvalidInput(format!(
                "num_vars {num_vars} is smaller than alpha {alpha}"
            )));
        }
        let expected_len = 1usize
            .checked_shl(num_vars as u32)
            .ok_or_else(|| HachiError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if evals.len() != expected_len {
            return Err(HachiError::InvalidSize {
                expected: expected_len,
                actual: evals.len(),
            });
        }

        let outer_len = expected_len / D;
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;
        let mut coeffs = Vec::with_capacity(outer_len);
        let mut small_i8_coeffs = Vec::with_capacity(outer_len);
        let mut all_small_i8 = true;

        for i in 0..outer_len {
            let slice = &evals[i * D..(i + 1) * D];
            coeffs.push(CyclotomicRing::from_slice(slice));

            if all_small_i8 {
                let mut digits = [0i8; D];
                for (dst, coeff) in digits.iter_mut().zip(slice.iter()) {
                    if let Some(centered) = try_centered_i8(*coeff, q, half_q) {
                        *dst = centered;
                    } else {
                        all_small_i8 = false;
                        break;
                    }
                }
                if all_small_i8 {
                    small_i8_coeffs.push(digits);
                }
            }
        }

        Ok(Self {
            coeffs,
            small_i8_coeffs: all_small_i8.then_some(small_i8_coeffs),
        })
    }

    /// Wrap an existing vector of ring elements.
    pub fn from_ring_coeffs(coeffs: Vec<CyclotomicRing<F, D>>) -> Self {
        let small_i8_coeffs = try_small_i8_cache_from_ring_coeffs(&coeffs);
        Self {
            coeffs,
            small_i8_coeffs,
        }
    }
}

impl<F, const D: usize> HachiPolyOps<F, D> for DensePoly<F, D>
where
    F: FieldCore + CanonicalField,
{
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        self.coeffs.len()
    }

    fn evaluate_ring(&self, scalars: &[F]) -> CyclotomicRing<F, D> {
        #[cfg(feature = "parallel")]
        {
            self.coeffs
                .par_iter()
                .zip(scalars.par_iter())
                .fold(
                    || CyclotomicRing::<F, D>::zero(),
                    |acc, (f_i, w_i)| acc + f_i.scale(w_i),
                )
                .reduce(|| CyclotomicRing::<F, D>::zero(), |a, b| a + b)
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.coeffs
                .iter()
                .zip(scalars.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, w_i)| {
                    acc + f_i.scale(w_i)
                })
        }
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);
        cfg_into_iter!(0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                let end = (start + block_len).min(n);
                let block = &self.coeffs[start..end];
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (b_j, &a_j) in block.iter().zip(scalars.iter()) {
                    acc += b_j.scale(&a_j);
                }
                acc
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "DensePoly::decompose_fold")]
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Vec<CyclotomicRing<F, D>> {
        let n = self.coeffs.len();
        let coeffs = &self.coeffs;

        let q = (-F::one()).to_canonical_u128() + 1;
        let params = DecomposeParams {
            half_q: q / 2,
            q,
            mask: (1i128 << log_basis) - 1,
            half_b: 1i128 << (log_basis - 1),
            b_val: 1i128 << log_basis,
            log_basis,
        };

        // Single-digit dense configs (e.g. logbasis) can skip the generic
        // multi-digit decomposition buffers and accumulate one centered digit
        // plane per ring element directly.
        if num_digits == 1 {
            if let Some(small_coeffs) = &self.small_i8_coeffs {
                let coeff_accum: Vec<[i32; D]> = {
                    let _span =
                        tracing::info_span!("dense_single_digit_cached_accumulate").entered();
                    cfg_into_iter!(0..block_len)
                        .map(|elem_idx| {
                            let mut z_local = [0i32; D];

                            for (block_idx, c_i) in challenges.iter().enumerate() {
                                let global_idx = block_idx * block_len + elem_idx;
                                if global_idx >= small_coeffs.len() {
                                    continue;
                                }
                                sparse_mul_acc::<D>(&small_coeffs[global_idx], c_i, &mut z_local);
                            }

                            z_local
                        })
                        .collect()
                };

                let _span = tracing::info_span!("dense_single_digit_convert").entered();
                return cfg_into_iter!(coeff_accum)
                    .map(|arr| signed_accum_to_ring::<F, D>(arr, params.q))
                    .collect();
            }

            let coeff_accum: Vec<[i32; D]> = {
                let _span = tracing::info_span!("dense_single_digit_accumulate").entered();
                cfg_into_iter!(0..block_len)
                    .map(|elem_idx| {
                        let mut z_local = [0i32; D];
                        let mut digit_plane = [0i8; D];

                        for (block_idx, c_i) in challenges.iter().enumerate() {
                            let global_idx = block_idx * block_len + elem_idx;
                            if global_idx >= n {
                                continue;
                            }
                            let ring = &coeffs[global_idx];
                            decompose_ring_single_digit::<F, D>(ring, &mut digit_plane, &params);
                            sparse_mul_acc::<D>(&digit_plane, c_i, &mut z_local);
                        }

                        z_local
                    })
                    .collect()
            };

            let _span = tracing::info_span!("dense_single_digit_convert").entered();
            return cfg_into_iter!(coeff_accum)
                .map(|arr| signed_accum_to_ring::<F, D>(arr, params.q))
                .collect();
        }

        // Two-phase approach: decompose ring element coefficients into i8 digit
        // planes, then scatter via sparse polynomial multiply.
        let z_chunks: Vec<Vec<[i32; D]>> = {
            let _span = tracing::info_span!("dense_multi_digit_accumulate").entered();
            cfg_into_iter!(0..block_len)
                .map(|elem_idx| {
                    let mut z_local: Vec<[i32; D]> = vec![[0i32; D]; num_digits];
                    let mut digit_buf: Vec<Vec<i8>> = vec![vec![0i8; D]; num_digits];

                    for (block_idx, c_i) in challenges.iter().enumerate() {
                        let global_idx = block_idx * block_len + elem_idx;
                        if global_idx >= n {
                            continue;
                        }
                        let ring = &coeffs[global_idx];
                        decompose_ring_interleaved::<F, D>(
                            ring,
                            &mut digit_buf,
                            num_digits,
                            &params,
                        );

                        for digit in 0..num_digits {
                            sparse_mul_acc::<D>(&digit_buf[digit], c_i, &mut z_local[digit]);
                        }
                    }

                    z_local
                })
                .collect()
        };

        let _span = tracing::info_span!("dense_multi_digit_convert").entered();
        let mut z = Vec::with_capacity(block_len * num_digits);
        for chunk in z_chunks {
            for arr in chunk {
                z.push(signed_accum_to_ring::<F, D>(arr, params.q));
            }
        }
        z
    }

    #[tracing::instrument(skip_all, name = "DensePoly::commit_inner")]
    fn commit_inner(
        &self,
        _a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<[i8; D]>>, HachiError> {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);

        let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                if start >= n {
                    &[] as &[CyclotomicRing<F, D>]
                } else {
                    &self.coeffs[start..(start + block_len).min(n)]
                }
            })
            .collect();

        let t_all = mat_vec_mul_ntt_i8(ntt_a, &block_slices, num_digits_commit, log_basis);

        let results: Vec<Vec<[i8; D]>> = cfg_into_iter!(t_all)
            .map(|t_i| decompose_rows_i8(&t_i, num_digits_open, log_basis))
            .collect();

        Ok(results)
    }

    fn commit_inner_witness(
        &self,
        _a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<CommitInnerWitness<F, D>, HachiError> {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);

        let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                if start >= n {
                    &[] as &[CyclotomicRing<F, D>]
                } else {
                    &self.coeffs[start..(start + block_len).min(n)]
                }
            })
            .collect();

        let t = mat_vec_mul_ntt_i8(ntt_a, &block_slices, num_digits_commit, log_basis);
        let t_hat = cfg_iter!(t)
            .map(|t_i| decompose_rows_i8(t_i, num_digits_open, log_basis))
            .collect();
        Ok(CommitInnerWitness { t, t_hat })
    }
}

/// Ring polynomial whose coefficients are already balanced base-`2^log_basis`
/// digits.
///
/// This is the recursive `w` witness used by Hachi's later prove levels. Unlike
/// [`DensePoly`], it can skip the `i8 -> field -> dense ring` round-trip and
/// operate on the digit planes directly.
#[derive(Debug, Clone)]
pub(crate) struct BalancedDigitPoly<'a, F: FieldCore, const D: usize> {
    coeffs: &'a [[i8; D]],
    padded_ring_elems: usize,
    _marker: PhantomData<F>,
}

impl<'a, F: FieldCore, const D: usize> BalancedDigitPoly<'a, F, D> {
    /// Wrap a flat digit vector laid out as consecutive ring coefficients.
    pub(crate) fn from_i8_digits(digits: &'a [i8]) -> Result<Self, HachiError> {
        let (coeffs, remainder) = digits.as_chunks::<D>();
        if !remainder.is_empty() {
            return Err(HachiError::InvalidSize {
                expected: D,
                actual: digits.len(),
            });
        }

        Ok(Self {
            coeffs,
            padded_ring_elems: coeffs.len().next_power_of_two().max(1),
            _marker: PhantomData,
        })
    }

    #[inline]
    fn block_slice(&self, block_idx: usize, block_len: usize) -> &'a [[i8; D]] {
        let start = block_idx * block_len;
        if start >= self.coeffs.len() {
            &[]
        } else {
            &self.coeffs[start..(start + block_len).min(self.coeffs.len())]
        }
    }
}

impl<'a, F, const D: usize> HachiPolyOps<F, D> for BalancedDigitPoly<'a, F, D>
where
    F: FieldCore + CanonicalField,
{
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        self.padded_ring_elems
    }

    fn evaluate_ring(&self, scalars: &[F]) -> CyclotomicRing<F, D> {
        let total = cfg_fold_reduce!(
            0..self.coeffs.len().min(scalars.len()),
            || [F::zero(); D],
            |mut acc: [F; D], idx| {
                let scalar = scalars[idx];
                let digit = &self.coeffs[idx];
                for (coeff, &d) in acc.iter_mut().zip(digit.iter()) {
                    if d != 0 {
                        *coeff += scalar * F::from_i8(d);
                    }
                }
                acc
            },
            |mut a: [F; D], b: [F; D]| {
                for (a_coeff, b_coeff) in a.iter_mut().zip(b.iter()) {
                    *a_coeff += *b_coeff;
                }
                a
            }
        );
        CyclotomicRing::from_coefficients(total)
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        let num_blocks = self.num_ring_elems().div_ceil(block_len);
        cfg_into_iter!(0..num_blocks)
            .map(|block_idx| {
                let mut acc = [F::zero(); D];
                for (ring, &scalar) in self
                    .block_slice(block_idx, block_len)
                    .iter()
                    .zip(scalars.iter())
                {
                    for (coeff, &d) in acc.iter_mut().zip(ring.iter()) {
                        if d != 0 {
                            *coeff += scalar * F::from_i8(d);
                        }
                    }
                }
                CyclotomicRing::from_coefficients(acc)
            })
            .collect()
    }

    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> Vec<CyclotomicRing<F, D>> {
        let inner_width = block_len * num_digits;
        let num_blocks = self.num_ring_elems().div_ceil(block_len);

        let q = (-F::one()).to_canonical_u128() + 1;
        cfg_fold_reduce!(
            0..challenges.len().min(num_blocks),
            || vec![[0i32; D]; inner_width],
            |mut z_local: Vec<[i32; D]>, block_idx| {
                let challenge = &challenges[block_idx];
                for (elem_idx, digit_plane) in
                    self.block_slice(block_idx, block_len).iter().enumerate()
                {
                    sparse_mul_acc::<D>(
                        digit_plane,
                        challenge,
                        &mut z_local[elem_idx * num_digits],
                    );
                }
                z_local
            },
            |mut a: Vec<[i32; D]>, b: Vec<[i32; D]>| {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    for (a_coeff, b_coeff) in ai.iter_mut().zip(bi.iter()) {
                        *a_coeff += *b_coeff;
                    }
                }
                a
            }
        )
        .into_iter()
        .map(|arr| {
            let coeffs = from_fn(|k| {
                let v = arr[k];
                if v >= 0 {
                    F::from_canonical_u128_reduced(v as u128)
                } else {
                    F::from_canonical_u128_reduced(q - ((-v) as u128))
                }
            });
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect()
    }

    fn commit_inner(
        &self,
        _a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<[i8; D]>>, HachiError> {
        let num_blocks = self.num_ring_elems().div_ceil(block_len);
        let coeff_len = self.coeffs.len();

        let t_all = if num_digits_commit == 1 {
            let block_slices: Vec<&[[i8; D]]> = (0..num_blocks)
                .map(|block_idx| self.block_slice(block_idx, block_len))
                .collect();
            mat_vec_mul_ntt_digits_i8(ntt_a, &block_slices)
        } else {
            let ring_elems: Vec<CyclotomicRing<F, D>> = self
                .coeffs
                .iter()
                .map(|digit| {
                    let coeffs = from_fn(|k| F::from_i8(digit[k]));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect();
            let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_blocks)
                .map(|block_idx| {
                    let start = block_idx * block_len;
                    if start >= coeff_len {
                        &[] as &[CyclotomicRing<F, D>]
                    } else {
                        &ring_elems[start..(start + block_len).min(coeff_len)]
                    }
                })
                .collect();
            mat_vec_mul_ntt_i8(ntt_a, &block_slices, num_digits_commit, log_basis)
        };

        let results = cfg_into_iter!(t_all)
            .map(|t_i| decompose_rows_i8(&t_i, num_digits_open, log_basis))
            .collect();
        Ok(results)
    }

    fn commit_inner_witness(
        &self,
        _a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<CommitInnerWitness<F, D>, HachiError> {
        let num_blocks = self.num_ring_elems().div_ceil(block_len);
        let coeff_len = self.coeffs.len();

        let t = if num_digits_commit == 1 {
            let block_slices: Vec<&[[i8; D]]> = (0..num_blocks)
                .map(|block_idx| self.block_slice(block_idx, block_len))
                .collect();
            mat_vec_mul_ntt_digits_i8(ntt_a, &block_slices)
        } else {
            let ring_elems: Vec<CyclotomicRing<F, D>> = self
                .coeffs
                .iter()
                .map(|digit| {
                    let coeffs = from_fn(|k| F::from_i8(digit[k]));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect();
            let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_blocks)
                .map(|block_idx| {
                    let start = block_idx * block_len;
                    if start >= coeff_len {
                        &[] as &[CyclotomicRing<F, D>]
                    } else {
                        &ring_elems[start..(start + block_len).min(coeff_len)]
                    }
                })
                .collect();
            mat_vec_mul_ntt_i8(ntt_a, &block_slices, num_digits_commit, log_basis)
        };

        let t_hat = cfg_iter!(t)
            .map(|t_i| decompose_rows_i8(t_i, num_digits_open, log_basis))
            .collect();
        Ok(CommitInnerWitness { t, t_hat })
    }
}

/// Types usable as one-hot position indices.
///
/// Implemented for `u8`, `u16`, `u32`, and `usize`.
pub trait OneHotIndex: Copy + Send + Sync + std::fmt::Debug + 'static {
    /// Convert to `usize` for indexing.
    fn as_usize(self) -> usize;
}

impl OneHotIndex for u8 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for u16 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for u32 {
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl OneHotIndex for usize {
    #[inline]
    fn as_usize(self) -> usize {
        self
    }
}

/// One-hot polynomial: sparse witness with at most one nonzero field element
/// per chunk of size `onehot_k`.
///
/// Exploits sparsity in all four operations, avoiding inner ring
/// multiplications during commit and decomposing only nonzero monomials.
///
/// Generic over `I`: the index type stored per chunk. Use `u8` when
/// `onehot_k <= 256` to cut per-entry memory from 16 bytes to 2 bytes.
#[derive(Debug, Clone)]
pub struct OneHotPoly<F: FieldCore, const D: usize, I: OneHotIndex = usize> {
    onehot_k: usize,
    indices: Vec<Option<I>>,
    m_vars: usize,
    sparse_blocks: Vec<Vec<SparseBlockEntry>>,
    _marker: PhantomData<F>,
}

impl<F: FieldCore, const D: usize, I: OneHotIndex> OneHotPoly<F, D, I> {
    /// Build a one-hot polynomial from chunk size and hot-position indices.
    ///
    /// `indices[c]` is the hot position in chunk `c` (`None` for all-zero chunks).
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent or any index is out of range.
    pub fn new(
        onehot_k: usize,
        indices: Vec<Option<I>>,
        r_vars: usize,
        m_vars: usize,
    ) -> Result<Self, HachiError> {
        let sparse_blocks = map_onehot_to_sparse_blocks(onehot_k, &indices, r_vars, m_vars, D)?;
        Ok(Self {
            onehot_k,
            indices,
            m_vars,
            sparse_blocks,
            _marker: PhantomData,
        })
    }

    fn total_ring_elems(&self) -> usize {
        let total_field = self.indices.len() * self.onehot_k;
        total_field / D
    }

    fn decompose_fold_regular_onehot(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>>
    where
        F: CanonicalField,
    {
        let num_blocks = challenges.len().min(self.sparse_blocks.len());
        let modulus = (-F::one()).to_canonical_u128() + 1;
        let indices = &self.indices;
        debug_assert_eq!(indices.len(), self.total_ring_elems());

        let coeff_accum: Vec<[i32; D]> = {
            let _span = tracing::info_span!("onehot_regular_accumulate").entered();
            cfg_into_iter!(0..block_len)
                .map(|elem_idx| {
                    let mut coeffs = [0i32; D];
                    let mut ring_idx = elem_idx;
                    for challenge in challenges.iter().take(num_blocks) {
                        if let Some(hot_idx) = indices[ring_idx] {
                            accum_onehot_coeff::<D>(&mut coeffs, hot_idx.as_usize(), challenge);
                        }
                        ring_idx += block_len;
                    }
                    coeffs
                })
                .collect()
        };

        let _span = tracing::info_span!("onehot_regular_convert").entered();
        cfg_into_iter!(coeff_accum)
            .map(|coeffs| signed_accum_to_ring::<F, D>(coeffs, modulus))
            .collect()
    }

    fn decompose_fold_sparse_onehot(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> Vec<CyclotomicRing<F, D>>
    where
        F: CanonicalField,
    {
        let inner_width = block_len * num_digits;
        let num_blocks = challenges.len().min(self.sparse_blocks.len());
        let modulus = (-F::one()).to_canonical_u128() + 1;

        let coeff_accum = {
            let _span = tracing::info_span!("onehot_sparse_accumulate").entered();
            cfg_fold_reduce!(
                0..num_blocks,
                || vec![[0i32; D]; inner_width],
                |mut z_local: Vec<[i32; D]>, block_idx: usize| {
                    let challenge = &challenges[block_idx];
                    for entry in &self.sparse_blocks[block_idx] {
                        let z_coeffs = &mut z_local[entry.pos_in_block * num_digits];
                        accum_onehot_entry::<D>(z_coeffs, entry, challenge);
                    }
                    z_local
                },
                |mut a: Vec<[i32; D]>, b: Vec<[i32; D]>| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        for (a_coeff, b_coeff) in ai.iter_mut().zip(bi.iter()) {
                            *a_coeff += *b_coeff;
                        }
                    }
                    a
                }
            )
        };

        let _span = tracing::info_span!("onehot_sparse_convert").entered();
        cfg_into_iter!(coeff_accum)
            .map(|coeffs| signed_accum_to_ring::<F, D>(coeffs, modulus))
            .collect()
    }
}

impl<F, const D: usize, I: OneHotIndex> HachiPolyOps<F, D> for OneHotPoly<F, D, I>
where
    F: FieldCore + CanonicalField + HasWide,
{
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        self.total_ring_elems()
    }

    fn evaluate_ring(&self, scalars: &[F]) -> CyclotomicRing<F, D> {
        let block_len = 1usize << self.m_vars;
        cfg_fold_reduce!(
            0..self.sparse_blocks.len(),
            || CyclotomicRing::<F, D>::zero(),
            |mut acc: CyclotomicRing<F, D>, block_idx: usize| {
                let block_offset = block_idx * block_len;
                for entry in &self.sparse_blocks[block_idx] {
                    let ring_idx = block_offset + entry.pos_in_block;
                    if ring_idx < scalars.len() {
                        let s = scalars[ring_idx];
                        for &ci in &entry.nonzero_coeffs {
                            acc.coeffs[ci] += s;
                        }
                    }
                }
                acc
            },
            |a, b| a + b
        )
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        cfg_iter!(self.sparse_blocks)
            .map(|entries| {
                let mut coeffs_acc = [F::zero(); D];
                for entry in entries {
                    if entry.pos_in_block < scalars.len() && entry.pos_in_block < block_len {
                        let s = scalars[entry.pos_in_block];
                        for &ci in &entry.nonzero_coeffs {
                            coeffs_acc[ci] += s;
                        }
                    }
                }
                CyclotomicRing::from_coefficients(coeffs_acc)
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::decompose_fold")]
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> Vec<CyclotomicRing<F, D>> {
        // In the common regular one-hot case used by the large onehot profile,
        // each chunk is exactly one ring element with one hot coefficient.
        // Build each output ring independently instead of reducing full z
        // vectors across blocks.
        if num_digits == 1 && self.onehot_k == D {
            self.decompose_fold_regular_onehot(challenges, block_len)
        } else {
            self.decompose_fold_sparse_onehot(challenges, block_len, num_digits)
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::commit_inner")]
    fn commit_inner(
        &self,
        a_matrix: &FlatMatrix<F>,
        _ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<[i8; D]>>, HachiError> {
        let a_view = a_matrix.view::<D>();
        let n_a = a_view.num_rows();
        let zero_block_len = n_a.checked_mul(num_digits_open).unwrap();

        let t_hat_all: Vec<Vec<[i8; D]>> = cfg_iter!(self.sparse_blocks)
            .map(|block_entries| {
                if block_entries.is_empty() {
                    vec![[0i8; D]; zero_block_len]
                } else {
                    let t_i = inner_ajtai_onehot_wide(
                        &a_view,
                        block_entries,
                        block_len,
                        num_digits_commit,
                    );
                    decompose_rows_i8(&t_i, num_digits_open, log_basis)
                }
            })
            .collect();

        Ok(t_hat_all)
    }

    fn commit_inner_witness(
        &self,
        a_matrix: &FlatMatrix<F>,
        _ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<CommitInnerWitness<F, D>, HachiError> {
        let a_view = a_matrix.view::<D>();
        let n_a = a_view.num_rows();
        let zero_block_len = n_a.checked_mul(num_digits_open).unwrap();

        let per_block = cfg_iter!(self.sparse_blocks)
            .map(|block_entries| {
                if block_entries.is_empty() {
                    (
                        vec![CyclotomicRing::<F, D>::zero(); n_a],
                        vec![[0i8; D]; zero_block_len],
                    )
                } else {
                    let t_i = inner_ajtai_onehot_wide(
                        &a_view,
                        block_entries,
                        block_len,
                        num_digits_commit,
                    );
                    let t_hat_i = decompose_rows_i8(&t_i, num_digits_open, log_basis);
                    (t_i, t_hat_i)
                }
            })
            .collect::<Vec<_>>();
        let (t, t_hat): (Vec<_>, Vec<_>) = per_block.into_iter().unzip();
        Ok(CommitInnerWitness { t, t_hat })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::commitment::{
        CommitmentConfig, HachiCommitmentCore, RingCommitmentScheme,
    };
    use crate::protocol::ring_switch::w_commitment_layout;
    use crate::test_utils::{TinyConfig, D as TestD, F as TestF};
    use crate::FromSmallInt;

    #[test]
    fn dense_poly_from_field_evals_roundtrip() {
        let num_vars = 10;
        let len = 1usize << num_vars;
        let evals: Vec<TestF> = (0..len).map(|i| TestF::from_u64(i as u64)).collect();
        let poly = DensePoly::<TestF, TestD>::from_field_evals(num_vars, &evals).unwrap();
        assert_eq!(poly.num_ring_elems(), len / TestD);
    }

    #[test]
    fn dense_commit_inner_matches_ring_commit() {
        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<TestF, TestD, TinyConfig>>::setup(16)
                .unwrap();
        let layout = setup.layout();
        let num_ring = layout.num_blocks * layout.block_len;
        let evals: Vec<TestF> = (0..num_ring * TestD)
            .map(|i| TestF::from_u64(i as u64))
            .collect();

        let alpha = TestD.trailing_zeros() as usize;
        let num_vars = alpha + layout.m_vars + layout.r_vars;
        let poly = DensePoly::<TestF, TestD>::from_field_evals(num_vars, &evals).unwrap();

        let t_hat_poly = poly
            .commit_inner(
                &setup.expanded.A,
                &setup.ntt_A,
                layout.block_len,
                layout.num_digits_commit,
                layout.num_digits_open,
                layout.log_basis,
            )
            .unwrap();

        let w =
            <HachiCommitmentCore as RingCommitmentScheme<TestF, TestD, TinyConfig>>::commit_coeffs(
                &poly.coeffs,
                &setup,
            )
            .unwrap();

        assert_eq!(t_hat_poly, w.t_hat);
    }

    #[test]
    fn onehot_commit_inner_matches_ring_commit_onehot() {
        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<TestF, TestD, TinyConfig>>::setup(16)
                .unwrap();
        let layout = setup.layout();
        let total_ring = layout.num_blocks * layout.block_len;
        let onehot_k = TestD;
        let num_chunks = total_ring;
        let indices: Vec<Option<usize>> = (0..num_chunks).map(|i| Some(i % onehot_k)).collect();

        let poly = OneHotPoly::<TestF, TestD>::new(
            onehot_k,
            indices.clone(),
            layout.r_vars,
            layout.m_vars,
        )
        .unwrap();

        let t_hat_poly = poly
            .commit_inner(
                &setup.expanded.A,
                &setup.ntt_A,
                layout.block_len,
                layout.num_digits_commit,
                layout.num_digits_open,
                layout.log_basis,
            )
            .unwrap();

        let w =
            <HachiCommitmentCore as RingCommitmentScheme<TestF, TestD, TinyConfig>>::commit_onehot(
                onehot_k, &indices, &setup,
            )
            .unwrap();

        assert_eq!(t_hat_poly, w.t_hat);
    }

    #[test]
    fn onehot_decompose_fold_matches_dense_regular_onehot() {
        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<TestF, TestD, TinyConfig>>::setup(16)
                .unwrap();
        let layout = setup.layout();
        let total_ring = layout.num_blocks * layout.block_len;
        let onehot_k = TestD;
        let indices: Vec<Option<usize>> = (0..total_ring)
            .map(|i| (i % 11 != 0).then_some((i * 7 + 3) % onehot_k))
            .collect();

        let poly = OneHotPoly::<TestF, TestD>::new(
            onehot_k,
            indices.clone(),
            layout.r_vars,
            layout.m_vars,
        )
        .unwrap();

        let mut evals = vec![TestF::zero(); total_ring * onehot_k];
        for (chunk_idx, hot_idx) in indices.into_iter().enumerate() {
            if let Some(hot_idx) = hot_idx {
                evals[chunk_idx * onehot_k + hot_idx] = TestF::from_u64(1);
            }
        }

        let alpha = TestD.trailing_zeros() as usize;
        let num_vars = alpha + layout.m_vars + layout.r_vars;
        let dense = DensePoly::<TestF, TestD>::from_field_evals(num_vars, &evals).unwrap();
        let challenges: Vec<SparseChallenge> = (0..layout.num_blocks)
            .map(|i| SparseChallenge {
                positions: vec![
                    0u32,
                    ((i * 5 + 1) % TestD) as u32,
                    ((i * 9 + 2) % TestD) as u32,
                ],
                coeffs: vec![1, -1, 1],
            })
            .collect();

        assert_eq!(
            poly.decompose_fold(&challenges, layout.block_len, 1, layout.log_basis),
            dense.decompose_fold(&challenges, layout.block_len, 1, layout.log_basis)
        );
    }

    #[test]
    fn balanced_digit_poly_matches_dense_recursive_w_ops() {
        let log_basis = TinyConfig::decomposition().log_basis;
        let digits: Vec<i8> = (0..(3 * TestD)).map(|i| (i % 7) as i8 - 3).collect();
        let field_evals: Vec<TestF> = digits.iter().map(|&d| TestF::from_i64(d as i64)).collect();
        let total_coeffs = digits.len().next_power_of_two().max(TestD);
        let mut padded = field_evals.clone();
        padded.resize(total_coeffs, TestF::zero());

        let dense = DensePoly::<TestF, TestD>::from_field_evals(
            total_coeffs.trailing_zeros() as usize,
            &padded,
        )
        .unwrap();
        let digit_poly = BalancedDigitPoly::<TestF, TestD>::from_i8_digits(&digits).unwrap();

        assert_eq!(digit_poly.num_ring_elems(), dense.num_ring_elems());

        let eval_scalars: Vec<TestF> = (0..digit_poly.num_ring_elems())
            .map(|i| TestF::from_u64((i + 2) as u64))
            .collect();
        assert_eq!(
            digit_poly.evaluate_ring(&eval_scalars),
            dense.evaluate_ring(&eval_scalars)
        );

        let block_len = 2;
        let fold_scalars: Vec<TestF> = (0..block_len)
            .map(|i| TestF::from_u64((i + 5) as u64))
            .collect();
        assert_eq!(
            digit_poly.fold_blocks(&fold_scalars, block_len),
            dense.fold_blocks(&fold_scalars, block_len)
        );

        let num_blocks = digit_poly.num_ring_elems().div_ceil(block_len);
        let challenges: Vec<SparseChallenge> = (0..num_blocks)
            .map(|i| SparseChallenge {
                positions: vec![0u32, ((i + 3) % TestD) as u32],
                coeffs: vec![1, -1],
            })
            .collect();
        assert_eq!(
            digit_poly.decompose_fold(&challenges, block_len, 1, log_basis),
            dense.decompose_fold(&challenges, block_len, 1, log_basis)
        );

        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<TestF, TestD, TinyConfig>>::setup(16)
                .unwrap();
        let w_layout = w_commitment_layout::<TestF, TestD, TinyConfig>(setup.layout()).unwrap();
        let digit_commit = digit_poly
            .commit_inner(
                &setup.expanded.A,
                &setup.ntt_A,
                w_layout.block_len,
                w_layout.num_digits_commit,
                w_layout.num_digits_open,
                w_layout.log_basis,
            )
            .unwrap();
        let dense_commit = dense
            .commit_inner(
                &setup.expanded.A,
                &setup.ntt_A,
                w_layout.block_len,
                w_layout.num_digits_commit,
                w_layout.num_digits_open,
                w_layout.log_basis,
            )
            .unwrap();

        assert_eq!(digit_commit, dense_commit);
    }
}
