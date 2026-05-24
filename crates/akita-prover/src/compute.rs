//! Prover compute backend boundary.
//!
//! The first backend is the existing CPU/Rayon implementation. The boundary is
//! intentionally operation-shaped: migrated prover code asks the backend to run
//! named commit/protocol kernels, and does not reach through prepared setup for
//! raw CPU matrices or NTT slots.

use crate::backend::onehot::{
    column_sweep_ajtai_multi_chunk, column_sweep_ajtai_single_chunk, MultiChunkEntry,
    SingleChunkEntry,
};
use crate::backend::sparse_ring::{column_sweep_sparse, SparseRingBlockEntry};
use crate::kernels::crt_ntt::{build_ntt_slot, NttSlotCache};
use crate::kernels::linear::{
    fused_split_eq_quotients, mat_vec_mul_ntt_dense_digits_i8, mat_vec_mul_ntt_digits_i8_strided,
    mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row, mat_vec_mul_ntt_i8_strided,
    mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic,
};
use crate::AkitaProverSetup;
use akita_algebra::CyclotomicRing;
use akita_field::fields::wide::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_types::AkitaExpandedSetup;
use std::array::from_fn;
use std::sync::Arc;

/// Dense polynomial commit representation handed to the compute backend.
pub enum DenseCommitInput<'a, F: FieldCore, const D: usize> {
    /// Balanced digit planes are already cached by the polynomial.
    CachedDigits {
        /// Per-block digit slices.
        digit_block_slices: Vec<&'a [[i8; D]]>,
    },
    /// Ring coefficients need backend-side digit decomposition.
    CoeffBlocks {
        /// Per-block coefficient slices.
        block_slices: Vec<&'a [CyclotomicRing<F, D>]>,
        /// Number of balanced digits used for the A-side commit.
        num_digits_commit: usize,
        /// Logarithm of the gadget basis.
        log_basis: u32,
    },
}

/// Dense commit operation plan.
pub struct DenseCommitRowsPlan<'a, F: FieldCore, const D: usize> {
    /// Number of A rows to produce.
    pub n_a: usize,
    /// Dense polynomial input representation.
    pub input: DenseCommitInput<'a, F, D>,
}

/// One-hot commit input representation.
///
/// The contained entry slices are read-only plan views. They are public so
/// accelerator crates can implement [`CommitmentComputeBackend`] without
/// depending on CPU-prepared storage, while construction remains owned by the
/// polynomial representations.
pub enum OneHotCommitBlocks<'a> {
    /// One ring has at most one hot coefficient.
    SingleChunk(Vec<&'a [SingleChunkEntry]>),
    /// One ring may contain several hot coefficients.
    MultiChunk(Vec<&'a [MultiChunkEntry]>),
}

/// One-hot commit operation plan.
pub struct OneHotCommitRowsPlan<'a> {
    /// Number of A rows to produce.
    pub n_a: usize,
    /// Root block length in ring elements.
    pub block_len: usize,
    /// Number of balanced digits used for the A-side commit.
    pub num_digits_commit: usize,
    /// Per-block one-hot entries.
    pub(crate) blocks: OneHotCommitBlocks<'a>,
}

impl<'a> OneHotCommitRowsPlan<'a> {
    /// Per-block one-hot entries.
    #[inline]
    pub fn blocks(&self) -> &OneHotCommitBlocks<'a> {
        &self.blocks
    }
}

/// Sparse signed-ring commit operation plan.
pub struct SparseRingCommitRowsPlan<'a> {
    /// Number of A rows to produce.
    pub n_a: usize,
    /// Root block length in ring elements.
    pub block_len: usize,
    /// Number of balanced digits used for the A-side commit.
    pub num_digits_commit: usize,
    /// Per-block sparse signed coefficients.
    pub(crate) blocks: Vec<&'a [SparseRingBlockEntry]>,
}

impl<'a> SparseRingCommitRowsPlan<'a> {
    /// Per-block sparse signed coefficients.
    #[inline]
    pub fn blocks(&self) -> &[&'a [SparseRingBlockEntry]] {
        &self.blocks
    }
}

/// Recursive witness commit operation plan.
pub struct RecursiveWitnessCommitRowsPlan<'a, const D: usize> {
    /// Recursive witness digit rows, chunked at `D`.
    pub coeffs: &'a [[i8; D]],
    /// Number of rows to produce.
    pub n_rows: usize,
    /// Recursive block length.
    pub block_len: usize,
    /// Number of logical blocks.
    pub num_blocks: usize,
    /// Number of balanced digits used for the A-side commit.
    pub num_digits_commit: usize,
    /// Logarithm of the gadget basis.
    pub log_basis: u32,
}

/// Shared prepared-setup contract for prover compute backends.
pub trait ComputeBackendSetup<F>: Send + Sync
where
    F: FieldCore + CanonicalField,
{
    /// Backend-prepared setup for a concrete ring dimension.
    type PreparedSetup<const D: usize>: Send + Sync;

    /// Prepare backend state from a prover setup wrapper.
    fn prepare_setup<const D: usize>(
        &self,
        setup: &AkitaProverSetup<F, D>,
    ) -> Result<Self::PreparedSetup<D>, AkitaError> {
        self.prepare_expanded::<D>(setup.expanded.clone())
    }

    /// Prepare backend state from already-expanded setup data.
    fn prepare_expanded<const D: usize>(
        &self,
        expanded: Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup<D>, AkitaError>;

    /// Protocol setup backing this prepared context.
    ///
    /// This exposes setup metadata and descriptor inputs used by the prover
    /// call graph; migrated compute paths must still request named backend
    /// operations instead of inspecting backend-prepared storage.
    fn expanded<'a, const D: usize>(
        &self,
        prepared: &'a Self::PreparedSetup<D>,
    ) -> &'a Arc<AkitaExpandedSetup<F>>;

    /// Maximum shared-matrix stride supported by this prepared context.
    fn max_stride<const D: usize>(&self, prepared: &Self::PreparedSetup<D>) -> usize {
        self.expanded::<D>(prepared).seed.max_stride
    }
}

/// Linear digit mat-vec operations shared by commitment and ring-switch code.
pub trait LinearComputeBackend<F>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    /// Negacyclic single-input digit mat-vec rows.
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;

    /// Cyclic single-input digit mat-vec rows.
    fn cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;
}

/// Commitment row operations for migrated root/ring commitment work.
pub trait CommitmentComputeBackend<F>: LinearComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    /// Dense A-side commit rows.
    fn dense_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: DenseCommitRowsPlan<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>;

    /// One-hot A-side commit rows.
    fn onehot_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: OneHotCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>;

    /// Sparse signed-ring A-side commit rows.
    fn sparse_ring_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: SparseRingCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>;

    /// Recursive witness A-side commit rows.
    fn recursive_witness_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: RecursiveWitnessCommitRowsPlan<'_, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>;
}

/// Ring-switch relation operations for migrated proving work.
pub trait RingSwitchComputeBackend<F>: LinearComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    /// Fused cyclic/quotient rows used by ring-switch finalization.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn ring_switch_relation_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        n_d: usize,
        n_b: usize,
        n_a: usize,
        w_hat: &[[i8; D]],
        t_hat: &[[i8; D]],
        z_segment: &[[i32; D]],
        z_pre_centered_inf_norm: u32,
    ) -> Result<
        (
            Vec<CyclotomicRing<F, D>>,
            Vec<CyclotomicRing<F, D>>,
            Vec<CyclotomicRing<F, D>>,
        ),
        AkitaError,
    >
    where
        F: HalvingField;
}

/// Full first-PR prover compute surface.
pub trait ProverComputeBackend<F>:
    CommitmentComputeBackend<F> + RingSwitchComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
}

impl<F, B> ProverComputeBackend<F> for B
where
    F: FieldCore + CanonicalField,
    B: CommitmentComputeBackend<F> + RingSwitchComputeBackend<F>,
{
}

/// CPU backend using the existing Rust/Rayon kernels.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

/// CPU-prepared setup for one field/ring-dimension pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuPreparedSetup<F: FieldCore, const D: usize> {
    expanded: Arc<AkitaExpandedSetup<F>>,
    ntt_shared: NttSlotCache<D>,
}

impl<F> ComputeBackendSetup<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    type PreparedSetup<const D: usize> = CpuPreparedSetup<F, D>;

    fn prepare_expanded<const D: usize>(
        &self,
        expanded: Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup<D>, AkitaError> {
        let total = expanded.shared_matrix.total_ring_elements_at::<D>()?;
        let ntt_shared = build_ntt_slot(expanded.shared_matrix.ring_view::<D>(1, total)?)?;
        Ok(CpuPreparedSetup {
            expanded,
            ntt_shared,
        })
    }

    fn expanded<'a, const D: usize>(
        &self,
        prepared: &'a Self::PreparedSetup<D>,
    ) -> &'a Arc<AkitaExpandedSetup<F>> {
        &prepared.expanded
    }
}

impl<F> CommitmentComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn dense_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: DenseCommitRowsPlan<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        let stride = self.max_stride::<D>(prepared);
        Ok(match plan.input {
            DenseCommitInput::CachedDigits { digit_block_slices } => {
                mat_vec_mul_ntt_dense_digits_i8(
                    &prepared.ntt_shared,
                    plan.n_a,
                    stride,
                    &digit_block_slices,
                )
            }
            DenseCommitInput::CoeffBlocks {
                block_slices,
                num_digits_commit,
                log_basis,
            } => {
                if plan.n_a == 1 {
                    mat_vec_mul_ntt_i8_dense_single_row(
                        &prepared.ntt_shared,
                        stride,
                        &block_slices,
                        num_digits_commit,
                        log_basis,
                    )
                    .into_iter()
                    .map(|ring| vec![ring])
                    .collect()
                } else {
                    mat_vec_mul_ntt_i8_dense(
                        &prepared.ntt_shared,
                        plan.n_a,
                        stride,
                        &block_slices,
                        num_digits_commit,
                        log_basis,
                    )
                }
            }
        })
    }

    fn onehot_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: OneHotCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        let stride = self.max_stride::<D>(prepared);
        let a_view = prepared
            .expanded
            .shared_matrix
            .ring_view::<D>(plan.n_a, stride)?;
        let active_a_cols = plan
            .block_len
            .checked_mul(plan.num_digits_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))?;
        if active_a_cols > a_view.num_cols() {
            return Err(AkitaError::InvalidSetup(format!(
                "active A width {active_a_cols} exceeds setup envelope {}",
                a_view.num_cols()
            )));
        }
        Ok(match plan.blocks {
            OneHotCommitBlocks::SingleChunk(blocks) => column_sweep_ajtai_single_chunk::<F, D>(
                &a_view,
                &blocks,
                plan.n_a,
                active_a_cols,
                plan.num_digits_commit,
            ),
            OneHotCommitBlocks::MultiChunk(blocks) => column_sweep_ajtai_multi_chunk::<F, D>(
                &a_view,
                &blocks,
                plan.n_a,
                active_a_cols,
                plan.num_digits_commit,
            ),
        })
    }

    fn sparse_ring_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: SparseRingCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    {
        let stride = self.max_stride::<D>(prepared);
        let a_view = prepared
            .expanded
            .shared_matrix
            .ring_view::<D>(plan.n_a, stride)?;
        let active_a_cols = plan
            .block_len
            .checked_mul(plan.num_digits_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))?;
        if active_a_cols > a_view.num_cols() {
            return Err(AkitaError::InvalidSetup(format!(
                "active A width {active_a_cols} exceeds setup envelope {}",
                a_view.num_cols()
            )));
        }
        let a_rows = (0..plan.n_a)
            .map(|idx| a_view.row(idx))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(column_sweep_sparse(
            &a_rows,
            &plan.blocks,
            plan.n_a,
            plan.block_len,
            plan.num_digits_commit,
        ))
    }

    fn recursive_witness_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: RecursiveWitnessCommitRowsPlan<'_, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        let stride = self.max_stride::<D>(prepared);
        if plan.num_digits_commit == 1 {
            Ok(mat_vec_mul_ntt_digits_i8_strided(
                &prepared.ntt_shared,
                plan.n_rows,
                stride,
                plan.coeffs,
                plan.num_blocks,
                plan.block_len,
            ))
        } else {
            let ring_elems: Vec<CyclotomicRing<F, D>> = plan
                .coeffs
                .iter()
                .map(|digit| {
                    let coeffs = from_fn(|k| F::from_i8(digit[k]));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect();
            Ok(mat_vec_mul_ntt_i8_strided(
                &prepared.ntt_shared,
                plan.n_rows,
                stride,
                &ring_elems,
                plan.num_blocks,
                plan.block_len,
                plan.num_digits_commit,
                plan.log_basis,
            ))
        }
    }
}

impl<F> LinearComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        Ok(mat_vec_mul_ntt_single_i8(
            &prepared.ntt_shared,
            row_len,
            self.max_stride::<D>(prepared),
            digits,
        ))
    }

    fn cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        Ok(mat_vec_mul_ntt_single_i8_cyclic(
            &prepared.ntt_shared,
            row_len,
            self.max_stride::<D>(prepared),
            digits,
        ))
    }
}

impl<F> RingSwitchComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn ring_switch_relation_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        n_d: usize,
        n_b: usize,
        n_a: usize,
        w_hat: &[[i8; D]],
        t_hat: &[[i8; D]],
        z_segment: &[[i32; D]],
        z_pre_centered_inf_norm: u32,
    ) -> Result<
        (
            Vec<CyclotomicRing<F, D>>,
            Vec<CyclotomicRing<F, D>>,
            Vec<CyclotomicRing<F, D>>,
        ),
        AkitaError,
    >
    where
        F: HalvingField,
    {
        Ok(fused_split_eq_quotients(
            &prepared.ntt_shared,
            n_d,
            n_b,
            n_a,
            self.max_stride::<D>(prepared),
            w_hat,
            t_hat,
            z_segment,
            z_pre_centered_inf_norm,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp64;

    type F = Fp64<4294967197>;
    const D: usize = 32;

    fn prepared() -> CpuPreparedSetup<F, D> {
        let setup = AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, 1, 4, 8).unwrap();
        CpuBackend.prepare_setup(&setup).unwrap()
    }

    #[test]
    fn cpu_digit_rows_match_direct_kernel() {
        let prepared = prepared();
        let digits = vec![[1i8; D], [-1i8; D], [2i8; D]];
        let via_backend = CpuBackend
            .digit_rows::<D>(&prepared, 2, &digits)
            .expect("backend digit rows");
        let direct = mat_vec_mul_ntt_single_i8(&prepared.ntt_shared, 2, 8, &digits);
        assert_eq!(via_backend, direct);
    }

    #[test]
    fn cpu_cyclic_digit_rows_match_direct_kernel() {
        let prepared = prepared();
        let digits = vec![[1i8; D], [0i8; D], [-2i8; D], [3i8; D]];
        let via_backend = CpuBackend
            .cyclic_digit_rows::<D>(&prepared, 2, &digits)
            .expect("backend cyclic digit rows");
        let direct = mat_vec_mul_ntt_single_i8_cyclic(&prepared.ntt_shared, 2, 8, &digits);
        assert_eq!(via_backend, direct);
    }

    #[test]
    fn cpu_ring_switch_relation_rows_match_direct_kernel() {
        let prepared = prepared();
        let w_hat = vec![[1i8; D], [2i8; D]];
        let t_hat = vec![[-1i8; D], [3i8; D]];
        let z_segment = vec![[1i32; D], [-2i32; D], [3i32; D]];
        let via_backend = CpuBackend
            .ring_switch_relation_rows::<D>(&prepared, 1, 1, 1, &w_hat, &t_hat, &z_segment, 3)
            .expect("backend ring-switch relation rows");
        let direct = fused_split_eq_quotients(
            &prepared.ntt_shared,
            1,
            1,
            1,
            8,
            &w_hat,
            &t_hat,
            &z_segment,
            3,
        );
        assert_eq!(via_backend, direct);
    }
}
