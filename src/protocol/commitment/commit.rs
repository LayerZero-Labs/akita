//! Ring-native §4.1 commitment core implementation.

use super::config::{
    ensure_block_layout, ensure_matrix_shape, ensure_supported_num_vars, validate_and_derive_layout,
};
use super::scheme::RingCommitmentScheme;
use super::types::{RingCommitment, RingOpening};
use super::utils::linear::{decompose_block, decompose_rows, mat_vec_mul_unchecked};
use super::utils::matrix::{derive_public_matrix, sample_public_matrix_seed, PublicMatrixSeed};
use super::CommitmentConfig;
use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::{CanonicalField, FieldCore, FieldSampling};

/// Setup for the ring-native commitment core.
#[allow(non_snake_case)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingCommitmentSetup<F: FieldCore, const D: usize> {
    /// Maximum supported variable count.
    pub max_num_vars: usize,
    /// Public seed used to derive commitment matrices.
    pub public_matrix_seed: PublicMatrixSeed,
    /// Inner matrix `A`.
    pub A: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Outer matrix `B`.
    pub B: Vec<Vec<CyclotomicRing<F, D>>>,
}

/// Concrete §4.1 commitment core.
#[derive(Clone, Copy, Default)]
pub struct HachiCommitmentCore;

impl<F, const D: usize, Cfg> RingCommitmentScheme<F, D, Cfg> for HachiCommitmentCore
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    type ProverSetup = RingCommitmentSetup<F, D>;
    type VerifierSetup = RingCommitmentSetup<F, D>;
    type Commitment = RingCommitment<F, D>;
    type Opening = RingOpening<F, D>;

    fn setup(max_num_vars: usize) -> Result<(Self::ProverSetup, Self::VerifierSetup), HachiError> {
        let layout = validate_and_derive_layout::<Cfg, D>()?;
        ensure_supported_num_vars(max_num_vars, layout.required_vars)?;
        let public_matrix_seed = sample_public_matrix_seed();
        let a_matrix =
            derive_public_matrix::<F, D>(Cfg::N_A, layout.inner_width, &public_matrix_seed, b"A");
        let b_matrix =
            derive_public_matrix::<F, D>(Cfg::N_B, layout.outer_width, &public_matrix_seed, b"B");

        let setup = RingCommitmentSetup {
            max_num_vars,
            public_matrix_seed,
            A: a_matrix,
            B: b_matrix,
        };
        ensure_matrix_shape(&setup.A, Cfg::N_A, layout.inner_width, "A")?;
        ensure_matrix_shape(&setup.B, Cfg::N_B, layout.outer_width, "B")?;
        Ok((setup.clone(), setup))
    }

    fn commit_ring_blocks(
        f_blocks: &[Vec<CyclotomicRing<F, D>>],
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::Opening), HachiError> {
        let layout = validate_and_derive_layout::<Cfg, D>()?;
        ensure_supported_num_vars(setup.max_num_vars, layout.required_vars)?;
        ensure_block_layout(f_blocks, layout)?;
        ensure_matrix_shape(&setup.A, Cfg::N_A, layout.inner_width, "A")?;
        ensure_matrix_shape(&setup.B, Cfg::N_B, layout.outer_width, "B")?;

        let mut s_all: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(layout.num_blocks);
        let mut t_hat_all: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(layout.num_blocks);
        let mut t_hat_flat: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(layout.outer_width);
        for block in f_blocks {
            let s_i = decompose_block(block, Cfg::DELTA, Cfg::LOG_BASIS);

            let t_i = mat_vec_mul_unchecked(&setup.A, &s_i);
            let t_hat_i = decompose_rows(&t_i, Cfg::DELTA, Cfg::LOG_BASIS);
            t_hat_flat.extend(t_hat_i.iter().copied());

            s_all.push(s_i);
            t_hat_all.push(t_hat_i);
        }

        let u = mat_vec_mul_unchecked(&setup.B, &t_hat_flat);
        Ok((
            RingCommitment { u },
            RingOpening {
                s: s_all,
                t_hat: t_hat_all,
            },
        ))
    }
}
