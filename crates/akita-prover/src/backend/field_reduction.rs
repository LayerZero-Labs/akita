//! Tensor extension-opening packing helpers.

use akita_field::fields::wide::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, CanonicalField, FromPrimitiveInt};
use akita_field::{AkitaError, ExtField, FieldCore};
use akita_types::pack_tensor_base_lift_i8_digits;
use std::sync::Arc;

use crate::kernels::crt_ntt::NttSlotCache;
use crate::{AkitaPolyOps, DensePoly, RecursiveWitnessFlat, SparseRingPoly};

/// Root polynomial obtained by tensor-projecting base-field evaluations into
/// an extension-valued table.
///
/// Dense roots use the ordinary dense backend. Sparse one-hot roots use signed
/// ring coefficients so the transformed commitment path preserves sparsity.
#[derive(Debug, Clone)]
pub enum RootTensorProjectionPoly<F: FieldCore, const D: usize> {
    /// Dense transformed root polynomial.
    Dense(DensePoly<F, D>),
    /// Sparse signed-ring transformed root polynomial.
    Sparse(Arc<SparseRingPoly<F, D>>),
}

impl<F: FieldCore, const D: usize> From<DensePoly<F, D>> for RootTensorProjectionPoly<F, D> {
    fn from(poly: DensePoly<F, D>) -> Self {
        Self::Dense(poly)
    }
}

impl<F: FieldCore, const D: usize> From<SparseRingPoly<F, D>> for RootTensorProjectionPoly<F, D> {
    fn from(poly: SparseRingPoly<F, D>) -> Self {
        Self::Sparse(Arc::new(poly))
    }
}

impl<F: FieldCore, const D: usize> From<Arc<SparseRingPoly<F, D>>>
    for RootTensorProjectionPoly<F, D>
{
    fn from(poly: Arc<SparseRingPoly<F, D>>) -> Self {
        Self::Sparse(poly)
    }
}

macro_rules! dispatch_root_projection {
    ($self:expr, $poly:ident => $body:expr) => {
        match $self {
            RootTensorProjectionPoly::Dense($poly) => $body,
            RootTensorProjectionPoly::Sparse($poly) => $body,
        }
    };
}

impl<F, const D: usize> AkitaPolyOps<F, D> for RootTensorProjectionPoly<F, D>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        dispatch_root_projection!(self, poly => poly.num_ring_elems())
    }

    fn num_vars(&self) -> usize {
        dispatch_root_projection!(self, poly => poly.num_vars())
    }

    fn fold_blocks(
        &self,
        scalars: &[F],
        block_len: usize,
    ) -> Vec<akita_algebra::CyclotomicRing<F, D>> {
        dispatch_root_projection!(self, poly => poly.fold_blocks(scalars, block_len))
    }

    fn fold_blocks_ring(
        &self,
        scalars: &[akita_algebra::CyclotomicRing<F, D>],
        block_len: usize,
    ) -> Vec<akita_algebra::CyclotomicRing<F, D>> {
        dispatch_root_projection!(self, poly => poly.fold_blocks_ring(scalars, block_len))
    }

    fn evaluate_and_fold_ring(
        &self,
        eval_outer_scalars: &[akita_algebra::CyclotomicRing<F, D>],
        fold_scalars: &[akita_algebra::CyclotomicRing<F, D>],
        block_len: usize,
    ) -> (
        akita_algebra::CyclotomicRing<F, D>,
        Vec<akita_algebra::CyclotomicRing<F, D>>,
    ) {
        dispatch_root_projection!(self, poly => {
            poly.evaluate_and_fold_ring(eval_outer_scalars, fold_scalars, block_len)
        })
    }

    fn decompose_fold(
        &self,
        challenges: &[akita_challenges::SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> crate::DecomposeFoldWitness<F, D> {
        dispatch_root_projection!(self, poly => {
            poly.decompose_fold(challenges, block_len, num_digits, log_basis)
        })
    }

    fn decompose_fold_tensor_batched(
        polys: &[&Self],
        tensor: &akita_challenges::TensorChallenges,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Option<Result<crate::DecomposeFoldWitness<F, D>, AkitaError>> {
        match *polys.first()? {
            RootTensorProjectionPoly::Dense(_) => {
                let mut dense_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match *poly {
                        RootTensorProjectionPoly::Dense(inner) => dense_polys.push(inner),
                        RootTensorProjectionPoly::Sparse(_) => return None,
                    }
                }
                <DensePoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_tensor_batched(
                    &dense_polys,
                    tensor,
                    block_len,
                    num_digits,
                    log_basis,
                )
            }
            RootTensorProjectionPoly::Sparse(_) => {
                let mut sparse_polys = Vec::with_capacity(polys.len());
                for poly in polys {
                    match *poly {
                        RootTensorProjectionPoly::Sparse(inner) => {
                            sparse_polys.push(inner.as_ref())
                        }
                        RootTensorProjectionPoly::Dense(_) => return None,
                    }
                }
                <SparseRingPoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_tensor_batched(
                    &sparse_polys,
                    tensor,
                    block_len,
                    num_digits,
                    log_basis,
                )
            }
        }
    }

    fn commit_inner(
        &self,
        a_matrix: &akita_types::FlatMatrix<F>,
        ntt_a: &Self::CommitCache,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<akita_types::FlatDigitBlocks<D>, AkitaError> {
        dispatch_root_projection!(self, poly => {
            poly.commit_inner(
                a_matrix,
                ntt_a,
                n_a,
                block_len,
                num_digits_commit,
                num_digits_open,
                log_basis,
                matrix_stride,
            )
        })
    }

    fn commit_inner_witness(
        &self,
        a_matrix: &akita_types::FlatMatrix<F>,
        ntt_a: &Self::CommitCache,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<crate::CommitInnerWitness<F, D>, AkitaError>
    where
        F: CanonicalField,
    {
        dispatch_root_projection!(self, poly => {
            poly.commit_inner_witness(
                a_matrix,
                ntt_a,
                n_a,
                block_len,
                num_digits_commit,
                num_digits_open,
                log_basis,
                matrix_stride,
            )
        })
    }

    fn direct_root_witness(&self) -> Result<akita_types::DirectWitnessProof<F>, AkitaError> {
        dispatch_root_projection!(self, poly => poly.direct_root_witness())
    }
}

fn tensor_extension_split<F, E>(context: &'static str) -> Result<(usize, usize), AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let split_bits = E::EXT_DEGREE.trailing_zeros() as usize;
    let width = 1usize
        .checked_shl(split_bits as u32)
        .ok_or_else(|| AkitaError::InvalidInput("tensor extension width overflow".to_string()))?;
    if width != E::EXT_DEGREE || !E::EXT_DEGREE.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "tensor extension {context} requires power-of-two extension degree"
        )));
    }
    Ok((split_bits, width))
}

/// Pack a logical recursive digit witness into the canonical tensor extension
/// ring-subfield layout.
///
/// For degree-one fields this is the identity. For small fields this stores
/// the extension-valued tensor table in the same ring-subfield layout used by
/// folded extension openings.
///
/// # Errors
///
/// Returns an error if the logical witness length is not compatible with the
/// full tensor split or if ring-subfield packing fails.
pub fn tensor_pack_recursive_witness<F, E, const D: usize>(
    logical_w: &RecursiveWitnessFlat,
) -> Result<RecursiveWitnessFlat, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let (_split_bits, width) = tensor_extension_split::<F, E>("packing")?;
    let packed =
        pack_tensor_base_lift_i8_digits::<D>(logical_w.as_i8_digits(), E::EXT_DEGREE, width)?;
    Ok(RecursiveWitnessFlat::from_i8_digits(packed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{AkitaError, Prime32Offset99, RingSubfieldFp4};

    #[test]
    fn recursive_tensor_pack_rejects_non_divisible_digit_count() {
        type F = Prime32Offset99;
        type E = RingSubfieldFp4<F>;
        const D: usize = 32;
        let witness = RecursiveWitnessFlat::from_i8_digits(vec![1, 2, 3]);

        let err = tensor_pack_recursive_witness::<F, E, D>(&witness).unwrap_err();
        assert!(matches!(
            err,
            AkitaError::InvalidSize {
                expected: 4,
                actual: 3
            }
        ));
    }
}
