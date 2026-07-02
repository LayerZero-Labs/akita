//! Dense polynomial opening, tensor, and fold operations.
//!
//! Storage is D-free; every ring-shaped operation takes the ring dimension as
//! a method const generic and views the flat coefficients at kernel entry.

use super::poly::{DenseColumnSource, DensePoly};
use super::tensor_fold;
use crate::backend::poly_helpers::{
    balanced_ring_decompose_fold_partitioned, build_decompose_fold_witness,
    cached_digit_decompose_fold_partitioned, decompose_ring_single_digit, sparse_mul_acc,
    DecomposeParams,
};
use crate::backend::RootTensorProjectionPoly;
use crate::compute::{CommitInnerPlan, CommitmentComputeBackend};
use crate::kernels::linear::decompose_commit_blocks_into;
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::{CommitInnerWitness, DecomposeFoldWitness};
use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_algebra::{CyclotomicRing, SplitEqEvals};
use akita_challenges::{SparseChallenge, TensorChallenges as TensorChallengeSet};
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced,
};
use akita_types::{embed_ring_subfield_vector, tensor_column_partials_split_fold, FpExtEncoding};

impl<F> DensePoly<F>
where
    F: FieldCore + CanonicalField,
{
    pub(crate) fn fold_blocks<const D: usize>(
        &self,
        scalars: &[F],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let coeffs = self
            .ring_coeffs::<D>()
            .expect("DensePoly::fold_blocks: invalid ring view");
        let n = coeffs.len();
        let num_blocks = n.div_ceil(block_len);
        cfg_into_iter!(0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                let end = (start + block_len).min(n);
                let block = &coeffs[start..end];
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (b_j, &a_j) in block.iter().zip(scalars.iter()) {
                    acc += b_j.scale(&a_j);
                }
                acc
            })
            .collect()
    }

    pub(crate) fn fold_blocks_ring<const D: usize>(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let coeffs = self
            .ring_coeffs::<D>()
            .expect("DensePoly::fold_blocks_ring: invalid ring view");
        let n = coeffs.len();
        let num_blocks = n.div_ceil(block_len);
        cfg_into_iter!(0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                let end = (start + block_len).min(n);
                let block = &coeffs[start..end];
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (b_j, &a_j) in block.iter().zip(scalars.iter()) {
                    b_j.mul_accumulate_sparse_rhs_into(&a_j, &mut acc);
                }
                acc
            })
            .collect()
    }

    pub(crate) fn evaluate_and_fold<const D: usize>(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        crate::backend::poly_helpers::fused_evaluate_and_fold_base(
            self.fold_blocks::<D>(fold_scalars, block_len),
            eval_outer_scalars,
        )
    }

    pub(crate) fn evaluate_and_fold_ring<const D: usize>(
        &self,
        eval_outer_scalars: &[CyclotomicRing<F, D>],
        fold_scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        crate::backend::poly_helpers::fused_evaluate_and_fold_ring(
            self.fold_blocks_ring::<D>(fold_scalars, block_len),
            eval_outer_scalars,
        )
    }

    pub(crate) fn tensor_extension_column_partials<E, const D: usize>(
        &self,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let (split_bits, width) = self.tensor_shape::<E, D>(Some(logical_point))?;
        let split = SplitEqEvals::new(&logical_point[split_bits..])?;
        let source = DenseColumnSource {
            flat: &self.field_coeffs()[..self.live_coeff_len()?],
            width,
        };
        Ok(tensor_column_partials_split_fold::<F, E, _>(
            &split, width, &source,
        ))
    }

    pub(crate) fn tensor_extension_column_partials_batch<E, const D: usize>(
        polys: &[&Self],
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let Some(first) = polys.first() else {
            return Ok(Vec::new());
        };
        let (split_bits, width) = first.tensor_shape::<E, D>(Some(logical_point))?;
        // The Dao-Thaler / Gruen split of the tail equality table is
        // point-dependent only, so it is built once and shared across the batch.
        let split = SplitEqEvals::new(&logical_point[split_bits..])?;
        polys
            .iter()
            .map(|poly| {
                poly.tensor_shape::<E, D>(Some(logical_point))?;
                let source = DenseColumnSource {
                    flat: &poly.field_coeffs()[..poly.live_coeff_len()?],
                    width,
                };
                Ok(tensor_column_partials_split_fold::<F, E, _>(
                    &split, width, &source,
                ))
            })
            .collect()
    }

    pub(crate) fn tensor_packed_extension_evals<E, const D: usize>(
        &self,
    ) -> Result<Vec<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        let (_split_bits, width) = self.tensor_shape::<E, D>(None)?;
        let live_len = self.live_coeff_len()?;
        let flat = &self.field_coeffs()[..live_len];
        // Width-aligned runs never crossed a ring boundary in the old
        // ring-typed walk, so the flat chunking packs identical values.
        let mut evals = Vec::with_capacity(live_len / width);
        for coeffs in flat.chunks_exact(width) {
            evals.push(E::from_base_slice(coeffs));
        }
        Ok(evals)
    }

    pub(crate) fn tensor_packed_extension_sparse_evals<E>(
        &self,
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        Ok(None)
    }

    pub(crate) fn tensor_packed_extension_sparse_linear_combination<E>(
        polys: &[&Self],
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        if polys.len() != coeffs.len() {
            return Err(AkitaError::InvalidSize {
                expected: polys.len(),
                actual: coeffs.len(),
            });
        }
        let mut witnesses = Vec::with_capacity(polys.len());
        for poly in polys {
            let Some(witness) = poly.tensor_packed_extension_sparse_evals::<E>()? else {
                return Ok(None);
            };
            witnesses.push(witness);
        }
        Ok(Some(SparseExtensionOpeningWitness::linear_combination(
            coeffs.iter().copied().zip(witnesses.iter()),
        )?))
    }

    pub(crate) fn tensor_packed_extension_poly<E, const D: usize>(
        &self,
    ) -> Result<DensePoly<F>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
        E: FpExtEncoding<F>,
    {
        let evals = self.tensor_packed_extension_evals::<E, D>()?;
        let packed_len = D / E::EXT_DEGREE;
        if packed_len == 0 {
            return Err(AkitaError::InvalidInput(
                "extension degree exceeds root ring dimension".to_string(),
            ));
        }
        let mut rings = Vec::with_capacity(evals.len().div_ceil(packed_len));
        for chunk in evals.chunks(packed_len) {
            let mut values = chunk.to_vec();
            values.resize(packed_len, E::zero());
            rings.push(embed_ring_subfield_vector::<F, E, D>(
                &values,
                AkitaError::InvalidInput(
                    "root transformed witness does not encode in the ring-subfield basis"
                        .to_string(),
                ),
            )?);
        }
        Ok(DensePoly::from_ring_coeffs::<D>(rings))
    }

    pub(crate) fn tensor_packed_extension_root_poly<E, const D: usize>(
        &self,
    ) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
        E: FpExtEncoding<F>,
    {
        Ok(self.tensor_packed_extension_poly::<E, D>()?.into())
    }

    #[tracing::instrument(skip_all, name = "DensePoly::decompose_fold")]
    pub(crate) fn decompose_fold<const D: usize>(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> DecomposeFoldWitness<F> {
        let coeffs = self
            .ring_coeffs::<D>()
            .expect("DensePoly::decompose_fold: invalid ring view");
        let n = coeffs.len();

        if let Some(digit_planes) = self.digit_planes_for::<D>(num_digits, log_basis) {
            let coeff_accum = {
                let _span = tracing::info_span!("dense_cached_digit_accumulate").entered();
                cached_digit_decompose_fold_partitioned::<D>(
                    digit_planes,
                    challenges,
                    block_len,
                    num_digits,
                )
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
            if let Some(small_coeffs) = self.small_i8_ring_coeffs::<D>() {
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
    pub(crate) fn decompose_fold_tensor_batched<const D: usize>(
        polys: &[&Self],
        tensor: &TensorChallengeSet,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Option<DecomposeFoldWitness<F>>, AkitaError> {
        tensor_fold::decompose_fold_batched_tensor_dense::<F, D>(
            polys, tensor, block_len, num_digits, log_basis,
        )
    }

    #[tracing::instrument(skip_all, name = "DensePoly::commit_inner")]
    pub(crate) fn commit_inner<B, const D: usize>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        let t = self.commit_rows::<B, D>(
            backend,
            prepared,
            plan.n_a,
            plan.block_len,
            plan.num_digits_commit,
            plan.log_basis,
        )?;
        let decomposed_inner_rows =
            decompose_commit_blocks_into::<F, D>(&t, plan.num_digits_open, plan.log_basis)?;
        Ok(CommitInnerWitness::from_parts(t, decomposed_inner_rows))
    }
}
