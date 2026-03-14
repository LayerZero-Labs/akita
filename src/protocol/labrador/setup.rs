//! Labrador commitment key setup.

use crate::algebra::ring::CyclotomicRing;
use crate::protocol::commitment::utils::crt_ntt::{build_ntt_slot, NttSlotCache};
use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
use crate::protocol::labrador::comkey::{derive_extendable_comkey_matrix, LabradorComKeySeed};
use crate::protocol::labrador::types::LabradorReductionConfig;
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
            config.kappa,
            max_witness_len,
            comkey_seed,
            b"labrador/comkey/A",
        );

        let (b_mat, d_mat) = if config.kappa1 > 0 && !config.tail {
            let t_hat_len = num_witness_rows * config.kappa * config.fu;
            let h_hat_len = num_witness_rows * (num_witness_rows + 1) / 2 * config.fu;

            let b = derive_extendable_comkey_matrix::<F, D>(
                config.kappa1,
                t_hat_len,
                comkey_seed,
                b"labrador/comkey/B",
            );
            let d = derive_extendable_comkey_matrix::<F, D>(
                config.kappa1,
                h_hat_len,
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

/// Pre-derived commitment-key matrices for one Labrador level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorSetup<F: FieldCore, const D: usize> {
    /// Shared matrix payload for prover and verifier-side recursion.
    pub matrices: Arc<LabradorSetupMatrices<F, D>>,
    /// Cached CRT+NTT representation of A for NTT-based witness commitment.
    pub a_ntt: NttSlotCache<D>,
    /// Cached CRT+NTT representation of B for NTT-based outer commitment.
    pub b_ntt: NttSlotCache<D>,
}

impl<F: FieldCore + CanonicalField + FieldSampling, const D: usize> LabradorSetup<F, D> {
    /// Derive all commitment-key matrices for a single Labrador level.
    ///
    /// # Panics
    ///
    /// Panics if deriving the cached CRT+NTT slots for matrix `A` or `B` fails.
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
        let a_flat = FlatMatrix::from_ring_matrix(&matrices.a_mat);
        let a_ntt =
            build_ntt_slot(a_flat.view::<D>()).expect("failed to build LabradorSetup A NTT slot");

        let b_flat = FlatMatrix::from_ring_matrix(&matrices.b_mat);
        let b_ntt =
            build_ntt_slot(b_flat.view::<D>()).expect("failed to build LabradorSetup B NTT slot");

        Self {
            matrices,
            a_ntt,
            b_ntt,
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
            f: 2,
            b: 8,
            fu: 3,
            bu: 10,
            kappa: 4,
            kappa1: 3,
            tail: false,
        }
    }

    fn tail_config() -> LabradorReductionConfig {
        LabradorReductionConfig {
            tail: true,
            kappa1: 0,
            ..standard_config()
        }
    }

    #[test]
    fn standard_setup_matrix_dimensions() {
        let cfg = standard_config();
        let setup = LabradorSetup::<F, D>::new(&cfg, NUM_ROWS, MAX_LEN, &SEED);

        assert_eq!(setup.matrices.a_mat.len(), cfg.kappa);
        assert!(setup.matrices.a_mat.iter().all(|row| row.len() == MAX_LEN));

        let t_hat_len = NUM_ROWS * cfg.kappa * cfg.fu;
        assert_eq!(setup.matrices.b_mat.len(), cfg.kappa1);
        assert!(setup
            .matrices
            .b_mat
            .iter()
            .all(|row| row.len() == t_hat_len));

        let h_hat_len = NUM_ROWS * (NUM_ROWS + 1) / 2 * cfg.fu;
        assert_eq!(setup.matrices.d_mat.len(), cfg.kappa1);
        assert!(setup
            .matrices
            .d_mat
            .iter()
            .all(|row| row.len() == h_hat_len));
    }

    #[test]
    fn tail_setup_has_empty_outer_matrices() {
        let cfg = tail_config();
        let setup = LabradorSetup::<F, D>::new(&cfg, NUM_ROWS, MAX_LEN, &SEED);

        assert_eq!(setup.matrices.a_mat.len(), cfg.kappa);
        assert!(setup.matrices.a_mat.iter().all(|row| row.len() == MAX_LEN));

        assert!(setup.matrices.b_mat.is_empty());
        assert!(setup.matrices.d_mat.is_empty());
    }
}
