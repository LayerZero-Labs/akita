//! Dense polynomial: all ring coefficients materialized in memory.
//!
//! [`DensePoly`] implements [`AkitaPolyOps`](akita_prover::AkitaPolyOps) via standard
//! dense algorithms — balanced-digit decomposition, NTT-based matrix-vector
//! multiply, and parallel block folds.

use akita_algebra::ring::cyclotomic::{
    decompose_centering_threshold, BalancedDecomposePow2I8Params,
};
use akita_algebra::{CyclotomicRing, EqPolynomial};
use akita_challenges::{IntegerChallenge, SparseChallenge, TensorChallenges as TensorChallengeSet};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_sumcheck::tensor_opening_split;

use crate::backend::poly_helpers::{
    balanced_ring_decompose_fold_partitioned, build_decompose_fold_witness,
    decompose_ring_interleaved, decompose_ring_single_digit, sparse_mul_acc,
    try_small_i8_cache_from_ring_coeffs, DecomposeParams,
};
use crate::kernels::crt_ntt::NttSlotCache;
use crate::kernels::linear::{
    decompose_rows_i8_into, mat_vec_mul_ntt_dense_digits_i8, mat_vec_mul_ntt_i8_dense,
    mat_vec_mul_ntt_i8_dense_single_row, try_centered_i8,
};
use akita_types::FlatMatrix;
use akita_types::{DirectWitnessProof, FlatDigitBlocks, FlatRingVec};
use std::sync::OnceLock;

use crate::{AkitaPolyOps, CommitInnerWitness, DecomposeFoldWitness};

#[derive(Debug, Clone, PartialEq, Eq)]
struct DenseDigitCache<const D: usize> {
    num_digits: usize,
    log_basis: u32,
    planes: Vec<[i8; D]>,
}

/// Dense polynomial: all ring coefficients materialized in memory.
#[derive(Debug)]
pub struct DensePoly<F: FieldCore, const D: usize> {
    /// Actual multilinear variable count of the source witness.
    num_vars: usize,
    /// Ring coefficients in sequential block order.
    pub coeffs: Vec<CyclotomicRing<F, D>>,
    small_i8_coeffs: Option<Vec<[i8; D]>>,
    digit_cache: OnceLock<DenseDigitCache<D>>,
}

impl<F: FieldCore + Clone, const D: usize> Clone for DensePoly<F, D> {
    fn clone(&self) -> Self {
        Self {
            num_vars: self.num_vars,
            coeffs: self.coeffs.clone(),
            small_i8_coeffs: self.small_i8_coeffs.clone(),
            digit_cache: OnceLock::new(),
        }
    }
}

impl<F: FieldCore + PartialEq, const D: usize> PartialEq for DensePoly<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.num_vars == other.num_vars
            && self.coeffs == other.coeffs
            && self.small_i8_coeffs == other.small_i8_coeffs
    }
}

impl<F: FieldCore + Eq, const D: usize> Eq for DensePoly<F, D> {}

impl<F: FieldCore + CanonicalField, const D: usize> DensePoly<F, D> {
    /// Pack field-element evaluations into ring elements.
    ///
    /// The first `α = log₂(D)` variables become coefficient slots within each
    /// ring element; the remaining variables index ring elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `D` is not a power of two or if
    /// `evals.len() != 2^num_vars`.
    pub fn from_field_evals(num_vars: usize, evals: &[F]) -> Result<Self, AkitaError> {
        if D == 0 || !D.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "ring degree D={D} is not a power of two"
            )));
        }
        let expected_len = 1usize
            .checked_shl(num_vars as u32)
            .ok_or_else(|| AkitaError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if evals.len() != expected_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: evals.len(),
            });
        }

        let outer_len = expected_len.div_ceil(D);
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;
        let mut coeffs = Vec::with_capacity(outer_len);
        let mut small_i8_coeffs = Vec::with_capacity(outer_len);
        let mut all_small_i8 = true;

        for i in 0..outer_len {
            let start = i * D;
            let end = ((i + 1) * D).min(expected_len);
            let slice = &evals[start..end];
            let mut ring = CyclotomicRing::<F, D>::zero();
            for (coeff_idx, coeff) in slice.iter().enumerate() {
                ring.coeffs[coeff_idx] = *coeff;
            }
            coeffs.push(ring);

            if all_small_i8 {
                let mut digits = [0i8; D];
                for (coeff_idx, coeff) in slice.iter().enumerate() {
                    if let Some(centered) = try_centered_i8(*coeff, q, half_q) {
                        digits[coeff_idx] = centered;
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
            num_vars,
            coeffs,
            small_i8_coeffs: all_small_i8.then_some(small_i8_coeffs),
            digit_cache: OnceLock::new(),
        })
    }

    /// Wrap an existing vector of ring elements.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs.len() * D` overflows `usize`.
    pub fn from_ring_coeffs(coeffs: Vec<CyclotomicRing<F, D>>) -> Self {
        let small_i8_coeffs = try_small_i8_cache_from_ring_coeffs(&coeffs);
        let total = coeffs
            .len()
            .checked_mul(D)
            .expect("ring elems * D overflow");
        Self {
            num_vars: total.trailing_zeros() as usize,
            coeffs,
            small_i8_coeffs,
            digit_cache: OnceLock::new(),
        }
    }

    fn digit_planes_for(&self, num_digits: usize, log_basis: u32) -> Option<&[[i8; D]]> {
        if let Some(cache) = self.digit_cache.get() {
            return (cache.num_digits == num_digits && cache.log_basis == log_basis)
                .then_some(cache.planes.as_slice());
        }

        let q = (-F::one()).to_canonical_u128() + 1;
        let params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);
        let mut planes = vec![[0i8; D]; self.coeffs.len() * num_digits];
        cfg_chunks_mut!(planes, num_digits)
            .zip(cfg_iter!(self.coeffs))
            .for_each(|(dst, ring)| {
                ring.balanced_decompose_pow2_i8_into_with_params(dst, &params);
            });
        let _ = self.digit_cache.set(DenseDigitCache {
            num_digits,
            log_basis,
            planes,
        });
        let cache = self.digit_cache.get()?;
        (cache.num_digits == num_digits && cache.log_basis == log_basis)
            .then_some(cache.planes.as_slice())
    }

    fn live_coeff_len(&self) -> Result<usize, AkitaError> {
        1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })
    }

    fn tensor_shape<E>(&self, logical_point: Option<&[E]>) -> Result<(usize, usize), AkitaError>
    where
        E: ExtField<F>,
    {
        let (split_bits, width) = tensor_opening_split::<F, E>()?;
        if split_bits > self.num_vars {
            return Err(AkitaError::InvalidInput(
                "extension-opening tensor split exceeds polynomial arity".to_string(),
            ));
        }
        if width > D || !D.is_multiple_of(width) {
            return Err(AkitaError::InvalidInput(format!(
                "extension degree {width} does not evenly pack into dense ring degree {D}"
            )));
        }
        if let Some(point) = logical_point {
            if point.len() != self.num_vars {
                return Err(AkitaError::InvalidPointDimension {
                    expected: self.num_vars,
                    actual: point.len(),
                });
            }
        }
        Ok((split_bits, width))
    }

    fn tensor_extension_column_partials_with_tail_eq<E>(
        &self,
        width: usize,
        tail_eq: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        let live_len = self.live_coeff_len()?;
        let expected_tail_len = live_len / width;
        if tail_eq.len() != expected_tail_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_tail_len,
                actual: tail_eq.len(),
            });
        }

        let mut partials = vec![E::zero(); width];
        for (tail, &weight) in tail_eq.iter().enumerate() {
            let flat_idx = tail * width;
            let ring_idx = flat_idx / D;
            let coeff_idx = flat_idx % D;
            let coeffs = &self.coeffs[ring_idx].coefficients()[coeff_idx..coeff_idx + width];
            for (partial, &coeff) in partials.iter_mut().zip(coeffs.iter()) {
                *partial += weight.mul_base(coeff);
            }
        }
        Ok(partials)
    }

    fn decompose_fold_batched_tensor_dense(
        polys: &[&Self],
        tensor: &TensorChallengeSet,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError> {
        if polys.is_empty() {
            return Ok(None);
        }

        let q = (-F::one()).to_canonical_u128() + 1;
        let (tensor_challenges, blocks_per_claim) = materialize_tensor_challenges::<D>(tensor)?;
        let accum_i64 = if let Some(digit_planes) = polys
            .iter()
            .map(|poly| poly.digit_planes_for(num_digits, log_basis))
            .collect::<Option<Vec<_>>>()
        {
            let _span = tracing::info_span!("dense_tensor_cached_digit_accumulate").entered();
            accumulate_cached_digit_planes_tensor::<D>(
                &digit_planes,
                &tensor_challenges,
                blocks_per_claim,
                block_len,
                num_digits,
            )?
        } else {
            let threshold = decompose_centering_threshold(num_digits, log_basis, q);
            let params = DecomposeParams {
                threshold,
                q,
                mask: (1i128 << log_basis) - 1,
                half_b: 1i128 << (log_basis - 1),
                b_val: 1i128 << log_basis,
                log_basis,
                overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
            };
            let coeff_slices = polys
                .iter()
                .map(|poly| poly.coeffs.as_slice())
                .collect::<Vec<_>>();
            let _span = tracing::info_span!("dense_tensor_accumulate").entered();
            balanced_ring_decompose_fold_tensor_partitioned::<F, D>(
                &coeff_slices,
                &tensor_challenges,
                blocks_per_claim,
                block_len,
                num_digits,
                &params,
            )?
        };

        let _span = tracing::info_span!("dense_tensor_convert").entered();
        let centered_coeffs = narrow_tensor_accum_to_i32::<D>(accum_i64)?;
        Ok(Some(build_decompose_fold_witness::<F, D>(
            centered_coeffs,
            q,
        )))
    }
}

fn accumulate_cached_digit_planes_tensor<const D: usize>(
    digit_planes_by_poly: &[&[[i8; D]]],
    tensor_challenges: &[IntegerChallenge],
    blocks_per_claim: usize,
    block_len: usize,
    num_digits: usize,
) -> Result<Vec<[i64; D]>, AkitaError> {
    if block_len == 0 || num_digits == 0 {
        return Err(AkitaError::InvalidInput(
            "dense cached tensor decompose-fold requires non-zero block_len and num_digits"
                .to_string(),
        ));
    }
    let expected_blocks = digit_planes_by_poly
        .len()
        .checked_mul(blocks_per_claim)
        .ok_or_else(|| AkitaError::InvalidSetup("tensor challenge count overflow".to_string()))?;
    if tensor_challenges.len() != expected_blocks {
        return Err(AkitaError::InvalidSize {
            expected: expected_blocks,
            actual: tensor_challenges.len(),
        });
    }

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(block_len.max(1)).max(1);
    let elem_chunk = block_len.div_ceil(actual_threads);
    let chunks = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let elem_start = tid * elem_chunk;
            if elem_start >= block_len {
                return Ok(Vec::new());
            }
            let elem_end = (elem_start + elem_chunk).min(block_len);
            let mut acc = vec![[0i64; D]; (elem_end - elem_start) * num_digits];

            for (block_idx, challenge) in tensor_challenges.iter().enumerate() {
                let claim_idx = block_idx / blocks_per_claim;
                let local_block_idx = block_idx % blocks_per_claim;
                let digit_planes = digit_planes_by_poly[claim_idx];

                for elem_idx in elem_start..elem_end {
                    let ring_idx = local_block_idx * block_len + elem_idx;
                    let plane_base = ring_idx * num_digits;
                    if plane_base >= digit_planes.len() {
                        continue;
                    }
                    let out_base = (elem_idx - elem_start) * num_digits;
                    for digit_idx in 0..num_digits {
                        let Some(digit_plane) = digit_planes.get(plane_base + digit_idx) else {
                            continue;
                        };
                        integer_mul_acc_i64::<D>(
                            digit_plane,
                            challenge,
                            &mut acc[out_base + digit_idx],
                        );
                    }
                }
            }

            Ok(acc)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;

    Ok(chunks.into_iter().flatten().collect())
}

fn balanced_ring_decompose_fold_tensor_partitioned<F: CanonicalField, const D: usize>(
    poly_coeffs: &[&[CyclotomicRing<F, D>]],
    tensor_challenges: &[IntegerChallenge],
    blocks_per_claim: usize,
    block_len: usize,
    num_digits: usize,
    p: &DecomposeParams,
) -> Result<Vec<[i64; D]>, AkitaError> {
    if block_len == 0 || num_digits == 0 {
        return Err(AkitaError::InvalidInput(
            "dense tensor decompose-fold requires non-zero block_len and num_digits".to_string(),
        ));
    }
    let expected_blocks = poly_coeffs
        .len()
        .checked_mul(blocks_per_claim)
        .ok_or_else(|| AkitaError::InvalidSetup("tensor challenge count overflow".to_string()))?;
    if tensor_challenges.len() != expected_blocks {
        return Err(AkitaError::InvalidSize {
            expected: expected_blocks,
            actual: tensor_challenges.len(),
        });
    }

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(block_len.max(1)).max(1);
    let elem_chunk = block_len.div_ceil(actual_threads);
    let chunks = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let elem_start = tid * elem_chunk;
            if elem_start >= block_len {
                return Ok(Vec::new());
            }
            let elem_end = (elem_start + elem_chunk).min(block_len);
            let mut acc = vec![[0i64; D]; (elem_end - elem_start) * num_digits];
            let mut digit_buf = vec![[0i8; D]; num_digits];

            for (block_idx, challenge) in tensor_challenges.iter().enumerate() {
                let claim_idx = block_idx / blocks_per_claim;
                let local_block_idx = block_idx % blocks_per_claim;
                let coeff_start = local_block_idx * block_len + elem_start;
                let coeffs = poly_coeffs[claim_idx];
                if coeff_start >= coeffs.len() {
                    continue;
                }
                let coeff_end = (local_block_idx * block_len + elem_end).min(coeffs.len());
                if coeff_start >= coeff_end {
                    continue;
                }

                for (local_elem_idx, ring) in coeffs[coeff_start..coeff_end].iter().enumerate() {
                    decompose_ring_interleaved::<F, D>(ring, &mut digit_buf, num_digits, p);
                    let base = local_elem_idx * num_digits;
                    for digit_idx in 0..num_digits {
                        integer_mul_acc_i64::<D>(
                            &digit_buf[digit_idx],
                            challenge,
                            &mut acc[base + digit_idx],
                        );
                    }
                }
            }

            Ok(acc)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;

    Ok(chunks.into_iter().flatten().collect())
}

fn materialize_tensor_challenges<const D: usize>(
    tensor: &TensorChallengeSet,
) -> Result<(Vec<IntegerChallenge>, usize), AkitaError> {
    let blocks_per_claim = tensor
        .left_len
        .checked_mul(tensor.right_len)
        .ok_or_else(|| AkitaError::InvalidSetup("tensor challenge count overflow".to_string()))?;
    let expected_blocks = tensor
        .num_claims
        .checked_mul(blocks_per_claim)
        .ok_or_else(|| AkitaError::InvalidSetup("tensor challenge count overflow".to_string()))?;
    let challenges = tensor.expand_integer::<D>()?;
    if challenges.len() != expected_blocks {
        return Err(AkitaError::InvalidSize {
            expected: expected_blocks,
            actual: challenges.len(),
        });
    }
    Ok((challenges, blocks_per_claim))
}

fn integer_mul_acc_i64<const D: usize>(
    digit_plane: &[i8; D],
    challenge: &IntegerChallenge,
    acc: &mut [i64; D],
) {
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let p = pos as usize;
        let split = D - p;
        let coeff = i64::from(coeff);
        for i in 0..split {
            acc[i + p] += coeff * i64::from(digit_plane[i]);
        }
        for i in split..D {
            acc[i - split] -= coeff * i64::from(digit_plane[i]);
        }
    }
}

fn narrow_tensor_accum_to_i32<const D: usize>(
    accum_i64: Vec<[i64; D]>,
) -> Result<Vec<[i32; D]>, AkitaError> {
    let mut out = Vec::with_capacity(accum_i64.len());
    for row in accum_i64 {
        let mut narrowed = [0i32; D];
        for (dst, src) in narrowed.iter_mut().zip(row.iter()) {
            *dst = i32::try_from(*src).map_err(|_| {
                AkitaError::InvalidSetup(format!(
                    "tensor fold accumulator overflowed i32 envelope (value = {src})"
                ))
            })?;
        }
        out.push(narrowed);
    }
    Ok(out)
}

impl<F, const D: usize> AkitaPolyOps<F, D> for DensePoly<F, D>
where
    F: FieldCore + CanonicalField,
{
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        self.coeffs.len()
    }

    fn num_vars(&self) -> usize {
        self.num_vars
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

    fn fold_blocks_ring(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);
        cfg_into_iter!(0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                let end = (start + block_len).min(n);
                let block = &self.coeffs[start..end];
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (b_j, &a_j) in block.iter().zip(scalars.iter()) {
                    b_j.mul_accumulate_sparse_rhs_into(&a_j, &mut acc);
                }
                acc
            })
            .collect()
    }

    fn tensor_extension_column_partials<E>(&self, logical_point: &[E]) -> Result<Vec<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        let (split_bits, width) = self.tensor_shape::<E>(Some(logical_point))?;
        let tail_eq = EqPolynomial::evals(&logical_point[split_bits..])?;
        self.tensor_extension_column_partials_with_tail_eq(width, &tail_eq)
    }

    fn tensor_extension_column_partials_batch<E>(
        polys: &[&Self],
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        let Some(first) = polys.first() else {
            return Ok(Vec::new());
        };
        let (split_bits, width) = first.tensor_shape::<E>(Some(logical_point))?;
        let tail_eq = EqPolynomial::evals(&logical_point[split_bits..])?;
        polys
            .iter()
            .map(|poly| {
                poly.tensor_shape::<E>(Some(logical_point))?;
                poly.tensor_extension_column_partials_with_tail_eq(width, &tail_eq)
            })
            .collect()
    }

    fn tensor_packed_extension_evals<E>(&self) -> Result<Vec<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        let (_split_bits, width) = self.tensor_shape::<E>(None)?;
        let live_len = self.live_coeff_len()?;
        let mut evals = Vec::with_capacity(live_len / width);
        let mut remaining = live_len;
        for ring in &self.coeffs {
            let take = remaining.min(D);
            for coeffs in ring.coefficients()[..take].chunks_exact(width) {
                evals.push(E::from_base_slice(coeffs));
            }
            remaining -= take;
            if remaining == 0 {
                break;
            }
        }
        Ok(evals)
    }

    #[tracing::instrument(skip_all, name = "DensePoly::decompose_fold")]
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        let n = self.coeffs.len();
        let coeffs = &self.coeffs;

        if let Some(digit_planes) = self.digit_planes_for(num_digits, log_basis) {
            let coeff_accum = {
                let _span = tracing::info_span!("dense_cached_digit_accumulate").entered();
                accumulate_cached_digit_planes::<D>(digit_planes, challenges, block_len, num_digits)
            };
            let modulus = (-F::one()).to_canonical_u128() + 1;
            return build_decompose_fold_witness::<F, D>(coeff_accum, modulus);
        }

        let q = (-F::one()).to_canonical_u128() + 1;
        let threshold = decompose_centering_threshold(num_digits, log_basis, q);
        let params = DecomposeParams {
            threshold,
            q,
            mask: (1i128 << log_basis) - 1,
            half_b: 1i128 << (log_basis - 1),
            b_val: 1i128 << log_basis,
            log_basis,
            overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
        };

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
                return build_decompose_fold_witness::<F, D>(coeff_accum, params.q);
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
            return build_decompose_fold_witness::<F, D>(coeff_accum, params.q);
        }

        let centered_coeffs = {
            let _span = tracing::info_span!("dense_multi_digit_accumulate").entered();
            balanced_ring_decompose_fold_partitioned::<F, D>(
                coeffs, challenges, block_len, num_digits, &params,
            )
        };

        let _span = tracing::info_span!("dense_multi_digit_convert").entered();
        build_decompose_fold_witness::<F, D>(centered_coeffs, params.q)
    }

    #[tracing::instrument(skip_all, name = "DensePoly::decompose_fold_tensor_batched")]
    fn decompose_fold_tensor_batched(
        polys: &[&Self],
        tensor: &TensorChallengeSet,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Option<Result<DecomposeFoldWitness<F, D>, AkitaError>> {
        match Self::decompose_fold_batched_tensor_dense(
            polys, tensor, block_len, num_digits, log_basis,
        ) {
            Ok(witness_opt) => witness_opt.map(Ok),
            Err(err) => Some(Err(err)),
        }
    }

    #[tracing::instrument(skip_all, name = "DensePoly::commit_inner")]
    fn commit_inner(
        &self,
        _a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<FlatDigitBlocks<D>, AkitaError> {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);

        if let Some(digit_planes) = self.digit_planes_for(num_digits_commit, log_basis) {
            let digit_block_slices =
                digit_block_slices(digit_planes, n, block_len, num_digits_commit);
            let t_all = mat_vec_mul_ntt_dense_digits_i8::<F, D>(
                ntt_a,
                n_a,
                matrix_stride,
                &digit_block_slices,
            );
            let block_sizes: Vec<usize> = t_all
                .iter()
                .map(|t_i| t_i.len() * num_digits_open)
                .collect();
            let mut t_hat = FlatDigitBlocks::zeroed(block_sizes)?;
            let dst_blocks = t_hat.split_blocks_mut();
            #[cfg(feature = "parallel")]
            cfg_into_iter!(dst_blocks)
                .zip(cfg_iter!(t_all))
                .for_each(|(dst, t_i)| {
                    decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis)
                });
            #[cfg(not(feature = "parallel"))]
            dst_blocks
                .into_iter()
                .zip(t_all.iter())
                .for_each(|(dst, t_i)| {
                    decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis)
                });

            return Ok(t_hat);
        }

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

        if n_a == 1 {
            let t = mat_vec_mul_ntt_i8_dense_single_row(
                ntt_a,
                matrix_stride,
                &block_slices,
                num_digits_commit,
                log_basis,
            );
            let mut t_hat = FlatDigitBlocks::zeroed(vec![num_digits_open; t.len()])?;
            let dst_blocks = t_hat.split_blocks_mut();
            #[cfg(feature = "parallel")]
            cfg_into_iter!(dst_blocks)
                .zip(cfg_iter!(t))
                .for_each(|(dst, t_i)| {
                    decompose_rows_i8_into(
                        std::slice::from_ref(t_i),
                        dst,
                        num_digits_open,
                        log_basis,
                    )
                });
            #[cfg(not(feature = "parallel"))]
            dst_blocks.into_iter().zip(t.iter()).for_each(|(dst, t_i)| {
                decompose_rows_i8_into(std::slice::from_ref(t_i), dst, num_digits_open, log_basis)
            });
            return Ok(t_hat);
        }

        let t_all = mat_vec_mul_ntt_i8_dense(
            ntt_a,
            n_a,
            matrix_stride,
            &block_slices,
            num_digits_commit,
            log_basis,
        );

        let block_sizes: Vec<usize> = t_all
            .iter()
            .map(|t_i| t_i.len() * num_digits_open)
            .collect();
        let mut t_hat = FlatDigitBlocks::zeroed(block_sizes)?;
        let dst_blocks = t_hat.split_blocks_mut();
        #[cfg(feature = "parallel")]
        cfg_into_iter!(dst_blocks)
            .zip(cfg_iter!(t_all))
            .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis));
        #[cfg(not(feature = "parallel"))]
        dst_blocks
            .into_iter()
            .zip(t_all.iter())
            .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis));

        Ok(t_hat)
    }

    fn commit_inner_witness(
        &self,
        _a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError> {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);

        if let Some(digit_planes) = self.digit_planes_for(num_digits_commit, log_basis) {
            let digit_block_slices =
                digit_block_slices(digit_planes, n, block_len, num_digits_commit);
            let t = mat_vec_mul_ntt_dense_digits_i8::<F, D>(
                ntt_a,
                n_a,
                matrix_stride,
                &digit_block_slices,
            );
            let block_sizes: Vec<usize> = t.iter().map(|t_i| t_i.len() * num_digits_open).collect();
            let mut t_hat = FlatDigitBlocks::zeroed(block_sizes)?;
            let dst_blocks = t_hat.split_blocks_mut();
            #[cfg(feature = "parallel")]
            cfg_into_iter!(dst_blocks)
                .zip(cfg_iter!(t))
                .for_each(|(dst, t_i)| {
                    decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis)
                });
            #[cfg(not(feature = "parallel"))]
            dst_blocks.into_iter().zip(t.iter()).for_each(|(dst, t_i)| {
                decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis)
            });
            return Ok(CommitInnerWitness {
                recomposed_inner_rows: t,
                decomposed_inner_rows: t_hat,
            });
        }

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

        if n_a == 1 {
            let t_single = mat_vec_mul_ntt_i8_dense_single_row(
                ntt_a,
                matrix_stride,
                &block_slices,
                num_digits_commit,
                log_basis,
            );
            let mut t_hat = FlatDigitBlocks::zeroed(vec![num_digits_open; t_single.len()])?;
            let dst_blocks = t_hat.split_blocks_mut();
            #[cfg(feature = "parallel")]
            cfg_into_iter!(dst_blocks)
                .zip(cfg_iter!(t_single))
                .for_each(|(dst, t_i)| {
                    decompose_rows_i8_into(
                        std::slice::from_ref(t_i),
                        dst,
                        num_digits_open,
                        log_basis,
                    )
                });
            #[cfg(not(feature = "parallel"))]
            dst_blocks
                .into_iter()
                .zip(t_single.iter())
                .for_each(|(dst, t_i)| {
                    decompose_rows_i8_into(
                        std::slice::from_ref(t_i),
                        dst,
                        num_digits_open,
                        log_basis,
                    )
                });
            let t = t_single.into_iter().map(|ring| vec![ring]).collect();
            return Ok(CommitInnerWitness {
                recomposed_inner_rows: t,
                decomposed_inner_rows: t_hat,
            });
        }

        let t = mat_vec_mul_ntt_i8_dense(
            ntt_a,
            n_a,
            matrix_stride,
            &block_slices,
            num_digits_commit,
            log_basis,
        );
        let block_sizes: Vec<usize> = t.iter().map(|t_i| t_i.len() * num_digits_open).collect();
        let mut t_hat = FlatDigitBlocks::zeroed(block_sizes)?;
        let dst_blocks = t_hat.split_blocks_mut();
        #[cfg(feature = "parallel")]
        cfg_into_iter!(dst_blocks)
            .zip(cfg_iter!(t))
            .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis));
        #[cfg(not(feature = "parallel"))]
        dst_blocks
            .into_iter()
            .zip(t.iter())
            .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis));
        Ok(CommitInnerWitness {
            recomposed_inner_rows: t,
            decomposed_inner_rows: t_hat,
        })
    }

    fn direct_root_witness(&self) -> Result<DirectWitnessProof<F>, AkitaError> {
        let live_len = 1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })?;
        let mut coeffs = Vec::with_capacity(live_len);
        let mut remaining = live_len;
        for ring in &self.coeffs {
            let take = remaining.min(D);
            coeffs.extend_from_slice(&ring.coefficients()[..take]);
            remaining -= take;
            if remaining == 0 {
                break;
            }
        }
        Ok(DirectWitnessProof::FieldElements(FlatRingVec::from_coeffs(
            coeffs,
        )))
    }
}

fn digit_block_slices<const D: usize>(
    digit_planes: &[[i8; D]],
    num_rings: usize,
    block_len: usize,
    num_digits: usize,
) -> Vec<&[[i8; D]]> {
    let num_blocks = num_rings.div_ceil(block_len);
    (0..num_blocks)
        .map(|block_idx| {
            let ring_start = block_idx * block_len;
            let ring_end = (ring_start + block_len).min(num_rings);
            let digit_start = ring_start * num_digits;
            let digit_end = ring_end * num_digits;
            &digit_planes[digit_start..digit_end]
        })
        .collect()
}

fn accumulate_cached_digit_planes<const D: usize>(
    digit_planes: &[[i8; D]],
    challenges: &[SparseChallenge],
    block_len: usize,
    num_digits: usize,
) -> Vec<[i32; D]> {
    let inner_width = block_len * num_digits;
    cfg_into_iter!(0..inner_width)
        .map(|inner_idx| {
            let elem_idx = inner_idx / num_digits;
            let digit_idx = inner_idx % num_digits;
            let mut acc = [0i32; D];
            for (block_idx, challenge) in challenges.iter().enumerate() {
                let ring_idx = block_idx * block_len + elem_idx;
                let plane_idx = ring_idx * num_digits + digit_idx;
                let Some(digit_plane) = digit_planes.get(plane_idx) else {
                    continue;
                };
                sparse_mul_acc::<D>(digit_plane, challenge, &mut acc);
            }
            acc
        })
        .collect()
}

/// Test-only helpers for [`DensePoly`].
///
/// These live outside the production `AkitaPolyOps` trait because they are
/// only used by cross-check tests (e.g. verifying that fused prover paths
/// match a straight-line reference implementation).
#[cfg(test)]
pub(crate) mod test_helpers {
    use super::DensePoly;
    use akita_algebra::CyclotomicRing;
    use akita_field::FieldCore;
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    /// Reference ring-space evaluation for [`DensePoly`].
    ///
    /// Computes the global weighted sum `y = Σᵢ scalars[i] · self.coeffs[i]`.
    #[allow(dead_code)]
    pub(crate) fn evaluate_ring_dense<F, const D: usize>(
        poly: &DensePoly<F, D>,
        scalars: &[F],
    ) -> CyclotomicRing<F, D>
    where
        F: FieldCore,
    {
        #[cfg(feature = "parallel")]
        {
            poly.coeffs
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
            poly.coeffs
                .iter()
                .zip(scalars.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, w_i)| {
                    acc + f_i.scale(w_i)
                })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::fields::{TowerBasisFp4, TwoNr, UnitNr};
    use akita_field::Prime128OffsetA7F7 as F;
    use akita_sumcheck::{tensor_column_partials_from_base_evals, tensor_packed_witness_evals};

    fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
            F::from_u64(offset + idx as u64 + 1)
        }))
    }

    #[test]
    fn ring_fold_matches_dense_multiplication_reference() {
        const D: usize = 8;
        let coeffs = (0..4).map(|idx| ring::<D>(10 * idx)).collect::<Vec<_>>();
        let poly = DensePoly::<F, D>::from_ring_coeffs(coeffs.clone());
        let scalars = vec![ring::<D>(100), ring::<D>(200)];
        let got = poly.fold_blocks_ring(&scalars, 2);
        let expected = coeffs
            .chunks(2)
            .map(|block| {
                block
                    .iter()
                    .zip(scalars.iter())
                    .fold(CyclotomicRing::<F, D>::zero(), |acc, (coeff, scalar)| {
                        acc + (*coeff * *scalar)
                    })
            })
            .collect::<Vec<_>>();

        assert_eq!(got, expected);
    }

    #[test]
    fn dense_tensor_opening_methods_match_flat_reference() {
        const D: usize = 8;
        type E = TowerBasisFp4<F, TwoNr, UnitNr>;

        let num_vars = 5;
        let evals = (0..(1usize << num_vars))
            .map(|idx| F::from_u64(17 * idx as u64 + 9))
            .collect::<Vec<_>>();
        let point = (0..num_vars)
            .map(|idx| {
                E::from_base_slice(&[
                    F::from_u64(idx as u64 + 2),
                    F::from_u64(3 * idx as u64 + 4),
                    F::from_u64(5 * idx as u64 + 6),
                    F::from_u64(7 * idx as u64 + 8),
                ])
            })
            .collect::<Vec<_>>();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();

        let expected_partials =
            tensor_column_partials_from_base_evals::<F, E>(num_vars, &evals, &point).unwrap();
        let got_partials = poly.tensor_extension_column_partials::<E>(&point).unwrap();
        assert_eq!(got_partials, expected_partials);

        let expected_packed = tensor_packed_witness_evals::<F, E>(num_vars, &evals).unwrap();
        let got_packed = poly.tensor_packed_extension_evals::<E>().unwrap();
        assert_eq!(got_packed, expected_packed);
    }
}
