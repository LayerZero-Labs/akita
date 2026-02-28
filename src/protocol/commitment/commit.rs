//! Ring-native §4.1 commitment core implementation.

use super::config::{
    ensure_block_layout, ensure_matrix_shape, ensure_supported_num_vars, validate_and_derive_layout,
};
use super::onehot::{inner_ajtai_onehot, map_onehot_to_sparse_blocks};
use super::scheme::RingCommitmentScheme;
use super::types::RingCommitment;
use super::utils::crt_ntt::{build_ntt_cache, NttMatrixCache};
use super::utils::linear::{
    decompose_block, decompose_rows, mat_vec_mul_ntt_cached, mat_vec_mul_ntt_many_cached,
    MatrixSlot,
};
use super::utils::matrix::{derive_public_matrix, sample_public_matrix_seed, PublicMatrixSeed};
use super::CommitmentConfig;
use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::{CanonicalField, FieldCore, FieldSampling};

/// Unified setup for the ring-native commitment (§4.1) and prover (§4.2).
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
    /// Prover matrix `D ∈ R_q^{n_D × δ·2^R}` (§4.2).
    pub D: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Pre-converted CRT+NTT matrices for dense mat-vec paths.
    pub(crate) ntt_cache: NttMatrixCache<D>,
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

    fn setup(max_num_vars: usize) -> Result<(Self::ProverSetup, Self::VerifierSetup), HachiError> {
        let layout = validate_and_derive_layout::<Cfg, D>()?;
        ensure_supported_num_vars(max_num_vars, layout.required_vars)?;
        let public_matrix_seed = sample_public_matrix_seed();
        let a_matrix =
            derive_public_matrix::<F, D>(Cfg::N_A, layout.inner_width, &public_matrix_seed, b"A");
        let b_matrix =
            derive_public_matrix::<F, D>(Cfg::N_B, layout.outer_width, &public_matrix_seed, b"B");
        let d_matrix = derive_public_matrix::<F, D>(
            Cfg::N_D,
            layout.d_matrix_width,
            &public_matrix_seed,
            b"D",
        );

        let ntt_cache = build_ntt_cache::<F, D>(&a_matrix, &b_matrix, &d_matrix)?;

        let setup = RingCommitmentSetup {
            max_num_vars,
            public_matrix_seed,
            A: a_matrix,
            B: b_matrix,
            D: d_matrix,
            ntt_cache,
        };
        ensure_matrix_shape(&setup.A, Cfg::N_A, layout.inner_width, "A")?;
        ensure_matrix_shape(&setup.B, Cfg::N_B, layout.outer_width, "B")?;
        ensure_matrix_shape(&setup.D, Cfg::N_D, layout.d_matrix_width, "D")?;
        Ok((setup.clone(), setup))
    }

    #[allow(clippy::type_complexity)]
    fn commit_ring_blocks(
        f_blocks: &[Vec<CyclotomicRing<F, D>>],
        setup: &Self::ProverSetup,
    ) -> Result<
        (
            Self::Commitment,
            Vec<Vec<CyclotomicRing<F, D>>>,
            Vec<Vec<CyclotomicRing<F, D>>>,
        ),
        HachiError,
    > {
        let layout = validate_and_derive_layout::<Cfg, D>()?;
        ensure_supported_num_vars(setup.max_num_vars, layout.required_vars)?;
        ensure_block_layout(f_blocks, layout)?;
        ensure_matrix_shape(&setup.A, Cfg::N_A, layout.inner_width, "A")?;
        ensure_matrix_shape(&setup.B, Cfg::N_B, layout.outer_width, "B")?;

        let s_all: Vec<Vec<CyclotomicRing<F, D>>> = f_blocks
            .iter()
            .map(|block| decompose_block(block, Cfg::DELTA, Cfg::LOG_BASIS))
            .collect();

        let t_all = mat_vec_mul_ntt_many_cached(&setup.ntt_cache, MatrixSlot::A, &s_all)?;
        let mut t_hat_all: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(layout.num_blocks);
        let mut t_hat_flat: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(layout.outer_width);
        for t_i in t_all {
            let t_hat_i = decompose_rows(&t_i, Cfg::DELTA, Cfg::LOG_BASIS);
            t_hat_flat.extend(t_hat_i.iter().copied());
            t_hat_all.push(t_hat_i);
        }

        let u = mat_vec_mul_ntt_cached(&setup.ntt_cache, MatrixSlot::B, &t_hat_flat)?;
        Ok((RingCommitment { u }, s_all, t_hat_all))
    }

    #[allow(clippy::type_complexity)]
    fn commit_onehot(
        onehot_k: usize,
        indices: &[usize],
        setup: &Self::ProverSetup,
    ) -> Result<
        (
            Self::Commitment,
            Vec<Vec<CyclotomicRing<F, D>>>,
            Vec<Vec<CyclotomicRing<F, D>>>,
        ),
        HachiError,
    > {
        let layout = validate_and_derive_layout::<Cfg, D>()?;
        ensure_supported_num_vars(setup.max_num_vars, layout.required_vars)?;
        ensure_matrix_shape(&setup.A, Cfg::N_A, layout.inner_width, "A")?;
        ensure_matrix_shape(&setup.B, Cfg::N_B, layout.outer_width, "B")?;

        let sparse_blocks = map_onehot_to_sparse_blocks(onehot_k, indices, Cfg::R, Cfg::M, D)?;

        let mut s_all: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(layout.num_blocks);
        let mut t_hat_all: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(layout.num_blocks);
        let mut t_hat_flat: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(layout.outer_width);

        for block_entries in &sparse_blocks {
            let (t_i, s_i) =
                inner_ajtai_onehot(&setup.A, block_entries, layout.block_len, Cfg::DELTA);
            let t_hat_i = decompose_rows(&t_i, Cfg::DELTA, Cfg::LOG_BASIS);
            t_hat_flat.extend(t_hat_i.iter().copied());

            s_all.push(s_i);
            t_hat_all.push(t_hat_i);
        }

        let u = mat_vec_mul_ntt_cached(&setup.ntt_cache, MatrixSlot::B, &t_hat_flat)?;
        Ok((RingCommitment { u }, s_all, t_hat_all))
    }
}
