//! Polynomial backends and prover-only witness state.

mod dense;
mod field_reduction;
mod multilinear_polynomial;
pub(crate) mod onehot;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod poly_helpers;
mod recursive;
mod ring_switch;
pub(crate) mod sparse_ring;
mod tensor_fold;

#[cfg(test)]
pub(crate) mod test_support {
    use crate::DecomposeFoldWitness;
    use akita_challenges::{SparseChallenge, TensorChallenges};
    use akita_field::{AkitaError, FieldCore};

    pub(crate) fn tensor_oracle_challenges<const D: usize>() -> TensorChallenges {
        TensorChallenges {
            left: vec![
                SparseChallenge {
                    positions: vec![0],
                    coeffs: vec![1],
                },
                SparseChallenge {
                    positions: vec![(D - 1) as u32],
                    coeffs: vec![1],
                },
                SparseChallenge {
                    positions: vec![2],
                    coeffs: vec![-1],
                },
                SparseChallenge {
                    positions: vec![5],
                    coeffs: vec![1],
                },
            ],
            right: vec![
                SparseChallenge {
                    positions: vec![1],
                    coeffs: vec![1],
                },
                SparseChallenge {
                    positions: vec![3],
                    coeffs: vec![-1],
                },
                SparseChallenge {
                    positions: vec![0],
                    coeffs: vec![1],
                },
                SparseChallenge {
                    positions: vec![4],
                    coeffs: vec![1],
                },
            ],
            left_len: 2,
            right_len: 2,
            num_claims: 2,
        }
    }

    pub(crate) fn aggregate_witnesses<F: FieldCore, const D: usize>(
        witnesses: &[DecomposeFoldWitness<F, D>],
    ) -> DecomposeFoldWitness<F, D> {
        let mut acc = witnesses[0].clone();
        for witness in &witnesses[1..] {
            for (dst, src) in acc
                .z_folded_rings
                .iter_mut()
                .zip(witness.z_folded_rings.iter())
            {
                *dst += *src;
            }
            for (dst, src) in acc
                .centered_coeffs
                .iter_mut()
                .zip(witness.centered_coeffs.iter())
            {
                for k in 0..D {
                    dst[k] += src[k];
                }
            }
        }
        acc.centered_inf_norm = acc
            .centered_coeffs
            .iter()
            .flat_map(|coeffs| coeffs.iter())
            .map(|coeff| coeff.unsigned_abs())
            .max()
            .unwrap_or(0);
        acc
    }

    pub(crate) fn negacyclic_tensor_product_challenges_i8<const D: usize>(
        tensor: &TensorChallenges,
    ) -> Result<Vec<SparseChallenge>, AkitaError> {
        tensor.validate::<D>()?;
        let total_blocks = tensor.total_blocks()?;
        (0..total_blocks)
            .map(|block_idx| {
                let (_, _, left, right) = tensor.factors_for_logical_block(block_idx)?;
                sparse_tensor_product_i8::<D>(left, right)
            })
            .collect()
    }

    fn sparse_tensor_product_i8<const D: usize>(
        left: &SparseChallenge,
        right: &SparseChallenge,
    ) -> Result<SparseChallenge, AkitaError> {
        let mut coeffs = [0i16; D];
        for (&left_pos, &left_coeff) in left.positions.iter().zip(left.coeffs.iter()) {
            for (&right_pos, &right_coeff) in right.positions.iter().zip(right.coeffs.iter()) {
                let degree = left_pos as usize + right_pos as usize;
                let (pos, sign) = if degree < D {
                    (degree, 1i16)
                } else {
                    (degree - D, -1i16)
                };
                let term = i16::from(left_coeff)
                    .checked_mul(i16::from(right_coeff))
                    .and_then(|term| term.checked_mul(sign))
                    .ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "tensor reference coefficient overflow".to_string(),
                        )
                    })?;
                coeffs[pos] = coeffs[pos].checked_add(term).ok_or_else(|| {
                    AkitaError::InvalidInput("tensor reference coefficient overflow".to_string())
                })?;
            }
        }

        let mut positions = Vec::new();
        let mut sparse_coeffs = Vec::new();
        for (idx, &coeff) in coeffs.iter().enumerate() {
            if coeff == 0 {
                continue;
            }
            positions.push(idx as u32);
            sparse_coeffs.push(i8::try_from(coeff).map_err(|_| {
                AkitaError::InvalidInput(
                    "tensor reference coefficient does not fit in i8".to_string(),
                )
            })?);
        }
        Ok(SparseChallenge {
            positions,
            coeffs: sparse_coeffs,
        })
    }
}

pub use dense::{DenseBatchView, DensePoly, DenseView};
pub use field_reduction::{
    tensor_pack_recursive_witness, RootTensorProjectionBatchView, RootTensorProjectionPoly,
    RootTensorProjectionView,
};
pub use multilinear_polynomial::{
    MultilinearPolynomial, MultilinearPolynomialBatchView, MultilinearPolynomialView,
};
pub use onehot::{
    MultiChunkEntry, OneHotBatchView, OneHotIndex, OneHotPoly, OneHotView, SingleChunkEntry,
};
pub use recursive::{
    RecursiveCommitmentHintCache, RecursiveWitnessFlat, SuffixWitnessBatchView, SuffixWitnessView,
};
pub use ring_switch::{RingSwitchQuotientView, RingSwitchRelationView};
pub use sparse_ring::{SparseRingBatchView, SparseRingBlockEntry, SparseRingPoly, SparseRingView};
