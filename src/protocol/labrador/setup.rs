//! Labrador commitment key setup.

use crate::algebra::ring::CyclotomicRing;
use crate::protocol::commitment::utils::crt_ntt::{build_ntt_slot, NttSlotCache};
use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
use crate::protocol::labrador::comkey::{derive_extendable_comkey_matrix, LabradorComKeySeed};
use crate::protocol::labrador::commit::OUTER_NTT_LOG_BASIS;
use crate::protocol::labrador::types::LabradorReductionConfig;
use crate::protocol::labrador::utils::pow2_field;
use crate::{CanonicalField, FieldCore, FieldSampling};
use std::sync::Arc;

/// Matrix-only Labrador setup shared by prover and verifier recursion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorSetupMatrices<F: FieldCore, const D: usize> {
    /// Inner commitment matrix A.
    pub a_mat: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Outer commitment matrix B. Not needed for the last fold proof.
    pub b_mat: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Linear-garbage commitment matrix D. Not needed for the last fold proof.
    pub d_mat: Vec<Vec<CyclotomicRing<F, D>>>,
}

impl<F: FieldCore + CanonicalField + FieldSampling, const D: usize> LabradorSetupMatrices<F, D> {
    /// Derive the commitment-key matrices for a single Labrador level.
    #[tracing::instrument(skip_all, name = "labrador::setup_matrices")]
    pub fn new(
        config: &LabradorReductionConfig,
        num_witness_rows: usize,
        max_witness_len: usize,
        comkey_seed: &LabradorComKeySeed,
    ) -> Self {
        let a_mat = derive_extendable_comkey_matrix::<F, D>(
            config.inner_commit_rank,
            max_witness_len,
            comkey_seed,
            b"labrador/comkey/A",
        );

        let (b_mat, d_mat) = if config.outer_commit_rank > 0 && !config.tail {
            let inner_opening_digits_len =
                num_witness_rows * config.inner_commit_rank * config.aux_digit_parts;
            let linear_garbage_digits_len =
                num_witness_rows * (num_witness_rows + 1) / 2 * config.aux_digit_parts;

            let b = derive_extendable_comkey_matrix::<F, D>(
                config.outer_commit_rank,
                inner_opening_digits_len,
                comkey_seed,
                b"labrador/comkey/B",
            );
            let d = derive_extendable_comkey_matrix::<F, D>(
                config.outer_commit_rank,
                linear_garbage_digits_len,
                comkey_seed,
                b"labrador/comkey/U2",
            );
            (b, d)
        } else {
            (Vec::new(), Vec::new())
        };

        Self {
            a_mat,
            b_mat,
            d_mat,
        }
    }
}

#[inline]
fn max_linear_garbage_ntt_levels<F: CanonicalField>(config: &LabradorReductionConfig) -> usize {
    if config.aux_digit_parts == 0 || config.aux_digit_bits == 0 {
        return 0;
    }
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let field_bits = (u128::BITS - modulus.leading_zeros()) as usize;
    let aux_bits = config.aux_digit_bits;
    let carry_shift = aux_bits.saturating_mul(config.aux_digit_parts.saturating_sub(1));
    let carry_bits = field_bits.saturating_sub(carry_shift).max(1);
    let max_digit_bits = aux_bits.max(carry_bits);
    max_digit_bits.div_ceil(OUTER_NTT_LOG_BASIS as usize) + 1
}

/// Pre-derived commitment-key matrices for one Labrador level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorSetup<F: FieldCore, const D: usize> {
    /// Shared matrix payload for prover and verifier-side recursion.
    pub matrices: Arc<LabradorSetupMatrices<F, D>>,
    /// Precomputed NTT caches for D scaled by `2^{k*OUTER_NTT_LOG_BASIS}`.
    ///
    /// Index `k` corresponds to level `k` in i8-basis decomposition of
    /// linear-garbage digits.
    pub ntt_d_scaled_levels: Vec<NttSlotCache<D>>,
}

impl<F: FieldCore + CanonicalField + FieldSampling, const D: usize> LabradorSetup<F, D> {
    /// Derive all commitment-key matrices for a single Labrador level.
    #[tracing::instrument(skip_all, name = "labrador::setup")]
    pub fn new(
        config: &LabradorReductionConfig,
        num_witness_rows: usize,
        max_witness_len: usize,
        comkey_seed: &LabradorComKeySeed,
    ) -> Self {
        let matrices = Arc::new(LabradorSetupMatrices::new(
            config,
            num_witness_rows,
            max_witness_len,
            comkey_seed,
        ));
        let ntt_d_scaled_levels = if matrices.d_mat.is_empty() {
            Vec::new()
        } else {
            let max_levels = max_linear_garbage_ntt_levels::<F>(config);
            let mut slots = Vec::with_capacity(max_levels);
            let scale_step = pow2_field::<F>(OUTER_NTT_LOG_BASIS as usize);
            let mut scale = F::one();
            for _ in 0..max_levels {
                let scaled_d: Vec<Vec<CyclotomicRing<F, D>>> = matrices
                    .d_mat
                    .iter()
                    .map(|row| row.iter().map(|entry| entry.scale(&scale)).collect())
                    .collect();
                let scaled_d_flat = FlatMatrix::from_ring_matrix(&scaled_d);
                match build_ntt_slot(scaled_d_flat.view::<D>()) {
                    Ok(slot) => slots.push(slot),
                    Err(err) => {
                        tracing::debug!(
                            error = %err,
                            "failed to precompute Labrador D-matrix scaled NTT caches; using runtime fallback"
                        );
                        slots.clear();
                        break;
                    }
                }
                scale = scale * scale_step;
            }
            slots
        };
        Self {
            matrices,
            ntt_d_scaled_levels,
        }
    }

    /// Return the matrix-only setup used by verifier-side recursion.
    pub fn verifier_setup(&self) -> Arc<LabradorSetupMatrices<F, D>> {
        Arc::clone(&self.matrices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::protocol::labrador::types::LabradorReductionConfig;

    type F = Fp64<4294967197>;
    const D: usize = 64;
    const SEED: [u8; 32] = [7u8; 32];

    const NUM_ROWS: usize = 5;
    const MAX_LEN: usize = 12;

    fn standard_config() -> LabradorReductionConfig {
        LabradorReductionConfig {
            witness_digit_parts: 2,
            witness_digit_bits: 8,
            aux_digit_parts: 3,
            aux_digit_bits: 10,
            inner_commit_rank: 4,
            outer_commit_rank: 3,
            tail: false,
        }
    }

    fn tail_config() -> LabradorReductionConfig {
        LabradorReductionConfig {
            tail: true,
            outer_commit_rank: 0,
            ..standard_config()
        }
    }

    #[test]
    fn standard_setup_matrix_dimensions() {
        let cfg = standard_config();
        let setup = LabradorSetup::<F, D>::new(&cfg, NUM_ROWS, MAX_LEN, &SEED);

        assert_eq!(setup.matrices.a_mat.len(), cfg.inner_commit_rank);
        assert!(setup.matrices.a_mat.iter().all(|row| row.len() == MAX_LEN));

        let inner_opening_digits_len = NUM_ROWS * cfg.inner_commit_rank * cfg.aux_digit_parts;
        assert_eq!(setup.matrices.b_mat.len(), cfg.outer_commit_rank);
        assert!(setup
            .matrices
            .b_mat
            .iter()
            .all(|row| row.len() == inner_opening_digits_len));

        let linear_garbage_digits_len = NUM_ROWS * (NUM_ROWS + 1) / 2 * cfg.aux_digit_parts;
        assert_eq!(setup.matrices.d_mat.len(), cfg.outer_commit_rank);
        assert!(setup
            .matrices
            .d_mat
            .iter()
            .all(|row| row.len() == linear_garbage_digits_len));
        assert!(!setup.ntt_d_scaled_levels.is_empty());
    }

    #[test]
    fn tail_setup_has_empty_outer_matrices() {
        let cfg = tail_config();
        let setup = LabradorSetup::<F, D>::new(&cfg, NUM_ROWS, MAX_LEN, &SEED);

        assert_eq!(setup.matrices.a_mat.len(), cfg.inner_commit_rank);
        assert!(setup.matrices.a_mat.iter().all(|row| row.len() == MAX_LEN));

        assert!(setup.matrices.b_mat.is_empty());
        assert!(setup.matrices.d_mat.is_empty());
        assert!(setup.ntt_d_scaled_levels.is_empty());
    }
}
