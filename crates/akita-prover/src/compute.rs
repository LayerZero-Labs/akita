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
use crate::backend::RootTensorProjectionPoly;
use crate::kernels::crt_ntt::{build_ntt_slot, NttSlotCache};
#[cfg(test)]
use crate::kernels::linear::fused_split_eq_quotients;
use crate::kernels::linear::{
    fused_split_eq_quotients_prover_bounds, mat_vec_mul_ntt_dense_digits_i8_trusted,
    mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row, mat_vec_mul_ntt_i8_strided,
    mat_vec_mul_ntt_raw_i8_strided, mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic,
    selected_crt_i8_capacity_profile, CrtI8CapacityProfile,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
#[cfg(any(test, feature = "zk"))]
use crate::validation::MAX_I8_LOG_BASIS;
use crate::{AkitaProverSetup, CommitInnerWitness, DecomposeFoldWitness};
use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallenge, TensorChallenges};
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
    MulBaseUnreduced,
};
use akita_types::{
    AkitaExpandedSetup, CleartextWitnessProof, FlatDigitBlocks, RingSubfieldEncoding,
};
use std::array::from_fn;
use std::marker::PhantomData;
use std::sync::Arc;
#[cfg(feature = "zk")]
use std::sync::OnceLock;

/// Flat block table handed to a compute backend.
///
/// `entries[offsets[i]..offsets[i + 1]]` is the entry slice for block `i`.
/// This is the canonical compact representation for sparse per-block work:
/// CPU code may recover per-block slices, while accelerator backends can upload
/// one contiguous entry table plus one offsets table.
#[derive(Debug, Clone, Copy)]
pub struct FlatBlockTable<'a, E> {
    entries: &'a [E],
    offsets: &'a [u32],
}

impl<'a, E> FlatBlockTable<'a, E> {
    /// Build a flat block table from validated storage.
    #[inline]
    pub(crate) fn new(entries: &'a [E], offsets: &'a [u32]) -> Self {
        Self { entries, offsets }
    }

    /// Contiguous sparse entries.
    #[inline]
    pub fn entries(&self) -> &'a [E] {
        self.entries
    }

    /// Block offsets into [`Self::entries`].
    #[inline]
    pub fn offsets(&self) -> &'a [u32] {
        self.offsets
    }

    /// Number of logical blocks.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Entry slice for one block.
    pub fn block(&self, idx: usize) -> Result<&'a [E], AkitaError> {
        let lo = self.offsets.get(idx).copied().ok_or_else(|| {
            AkitaError::InvalidSetup(format!("flat block table missing offset {idx}"))
        })? as usize;
        let hi = self.offsets.get(idx + 1).copied().ok_or_else(|| {
            AkitaError::InvalidSetup(format!("flat block table missing offset {}", idx + 1))
        })? as usize;
        if lo > hi || hi > self.entries.len() {
            return Err(AkitaError::InvalidSetup(format!(
                "flat block table has malformed offsets for block {idx}: {lo}..{hi} over {} entries",
                self.entries.len()
            )));
        }
        Ok(&self.entries[lo..hi])
    }

    fn block_slices(&self) -> Result<Vec<&'a [E]>, AkitaError> {
        (0..self.num_blocks()).map(|idx| self.block(idx)).collect()
    }
}

/// Dense polynomial commit representation handed to the compute backend.
pub enum DenseCommitInput<'a, F: FieldCore, const D: usize> {
    /// Balanced digit planes are already cached by the polynomial.
    CachedDigits {
        /// Per-block digit slices.
        digit_block_slices: Vec<&'a [[i8; D]]>,
        /// Logarithm of the gadget basis used to produce the cached digits.
        log_basis: u32,
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
    SingleChunk(FlatBlockTable<'a, SingleChunkEntry>),
    /// One ring may contain several hot coefficients.
    MultiChunk(FlatBlockTable<'a, MultiChunkEntry>),
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
    pub(crate) blocks: FlatBlockTable<'a, SparseRingBlockEntry>,
}

impl<'a> SparseRingCommitRowsPlan<'a> {
    /// Per-block sparse signed coefficients.
    #[inline]
    pub fn blocks(&self) -> FlatBlockTable<'a, SparseRingBlockEntry> {
        self.blocks
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

    /// Expanded setup used to prepare this backend context.
    fn prepared_expanded_setup<'a, const D: usize>(
        &self,
        prepared: &'a Self::PreparedSetup<D>,
    ) -> &'a AkitaExpandedSetup<F>;

    /// Ensure explicit setup metadata and backend-prepared state match.
    fn validate_prepared_setup<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<(), AkitaError> {
        let prepared_expanded = self.prepared_expanded_setup::<D>(prepared);
        // Valid setup matrices are deterministic from the seed; compare the
        // compact setup identity so independently materialized equivalent
        // setups validate without re-hashing the matrix on every prover call.
        if prepared_expanded.seed() != expanded.seed() {
            return Err(AkitaError::InvalidSetup(
                "prepared compute context was built for a different setup".to_string(),
            ));
        }
        Ok(())
    }
}

/// Negacyclic digit mat-vec operations shared by commitment and protocol code.
pub trait DigitRowsComputeBackend<F>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    /// Negacyclic single-input digit mat-vec rows.
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;

    /// Negacyclic ZK B-blinding digit mat-vec rows.
    #[cfg(feature = "zk")]
    fn zk_b_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;

    /// Negacyclic ZK D-blinding digit mat-vec rows.
    #[cfg(feature = "zk")]
    fn zk_d_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;
}

/// Cyclic digit mat-vec operations needed by ring-switch relation code.
pub trait CyclicRowsComputeBackend<F>: DigitRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    /// Cyclic single-input digit mat-vec rows.
    fn cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;

    /// Cyclic ZK B-blinding digit mat-vec rows.
    #[cfg(feature = "zk")]
    fn zk_b_cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;

    /// Cyclic ZK D-blinding digit mat-vec rows.
    #[cfg(feature = "zk")]
    fn zk_d_cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;
}

/// Commitment row operations for migrated root/ring commitment work.
pub trait CommitmentComputeBackend<F>: DigitRowsComputeBackend<F>
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

/// Full ring-switch relation operation input.
pub struct RingSwitchRelationRowsPlan<'a, const D: usize> {
    /// Number of D-side cyclic rows to produce.
    pub n_d: usize,
    /// Number of B-side cyclic rows to produce.
    pub n_b: usize,
    /// Number of A-side quotient rows to produce.
    pub n_a: usize,
    /// Flat decomposed `e_hat` digits for the D-side relation rows.
    pub e_hat: &'a [[i8; D]],
    /// Flat decomposed inner-commitment digits for the B-side relation rows.
    pub t_hat: &'a [[i8; D]],
    /// One centered `z` segment contributing to A-side quotient rows.
    pub z_segment: &'a [[i32; D]],
    /// Infinity norm of the full centered `z_folded_rings` witness.
    pub z_folded_centered_inf_norm: u32,
    /// Logarithm of the gadget basis used to produce `e_hat` and `t_hat`.
    pub log_basis: u32,
}

/// Additional public-row quotient operation input.
pub struct RingSwitchQuotientRowsPlan<'a, const D: usize> {
    /// Number of A-side quotient rows to produce.
    pub n_a: usize,
    /// One centered `z` segment contributing to A-side quotient rows.
    pub z_segment: &'a [[i32; D]],
    /// Infinity norm of the full centered `z_folded_rings` witness.
    pub z_folded_centered_inf_norm: u32,
}

/// Named ring-switch relation rows returned by a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingSwitchRelationRows<F: FieldCore, const D: usize> {
    /// D-side cyclic rows.
    pub d_cyclic: Vec<CyclotomicRing<F, D>>,
    /// B-side cyclic rows.
    pub b_cyclic: Vec<CyclotomicRing<F, D>>,
    /// A-side quotient rows.
    pub a_quotients: Vec<CyclotomicRing<F, D>>,
}

/// Ring-switch relation operations for migrated proving work.
pub trait RingSwitchComputeBackend<F>: CyclicRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    /// Fused cyclic/quotient rows used by ring-switch finalization.
    fn ring_switch_relation_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: RingSwitchRelationRowsPlan<'_, D>,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField;

    /// A-side quotient rows for an additional public-row segment.
    fn ring_switch_quotient_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: RingSwitchQuotientRowsPlan<'_, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
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
    ntt_i8_capacity: CrtI8CapacityProfile,
    #[cfg(feature = "zk")]
    ntt_zk_b: OnceLock<NttSlotCache<D>>,
    #[cfg(feature = "zk")]
    ntt_zk_d: OnceLock<NttSlotCache<D>>,
}

/// CRT/NTT profile and universal i8 capacity metadata for a prepared setup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedCrtNttProfile {
    /// Stable profile identifier used by benchmark/report tooling.
    pub profile_id: &'static str,
    /// Number of CRT primes in the selected profile.
    pub num_primes: usize,
    /// Signed limb width used by the CRT NTT representation.
    pub limb_bits: u32,
    /// Largest balanced i8 log basis accepted by prover i8 kernels.
    pub max_i8_log_basis: u32,
    /// Safe accumulation width for balanced i8 digits at `max_i8_log_basis`.
    pub balanced_digit_safe_width: usize,
    /// Safe accumulation width for raw signed i8 recursive-witness inputs.
    pub raw_i8_safe_width: usize,
}

impl From<CrtI8CapacityProfile> for PreparedCrtNttProfile {
    fn from(profile: CrtI8CapacityProfile) -> Self {
        Self {
            profile_id: profile.profile_id,
            num_primes: profile.num_primes,
            limb_bits: profile.limb_bits,
            max_i8_log_basis: profile.max_i8_log_basis,
            balanced_digit_safe_width: profile.balanced_digit_safe_width,
            raw_i8_safe_width: profile.raw_i8_safe_width,
        }
    }
}

impl<F: FieldCore, const D: usize> CpuPreparedSetup<F, D> {
    /// In-memory byte footprint of the shared setup NTT cache (negacyclic plus
    /// cyclic slots). Diagnostic surface for the profiler / bench report.
    pub fn shared_ntt_cache_bytes(&self) -> usize {
        self.ntt_shared.cache_bytes()
    }

    /// CRT/NTT profile and universal i8 capacity metadata for the shared setup
    /// cache. The capacity widths are the boundary checked during backend
    /// preparation before hot i8 kernels can rely on their internal invariant.
    pub fn shared_ntt_profile(&self) -> PreparedCrtNttProfile {
        self.ntt_i8_capacity.into()
    }
}

fn validate_digit_row_request(
    row_len: usize,
    row_width: usize,
    total_ring_elements: usize,
) -> Result<(), AkitaError> {
    if row_width == 0 {
        return Err(AkitaError::InvalidSetup(
            "prepared setup row width must be nonzero".to_string(),
        ));
    }
    let required = row_len.checked_mul(row_width).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "digit row request overflows: row_len={row_len} row_width={row_width}"
        ))
    })?;
    if required > total_ring_elements {
        return Err(AkitaError::InvalidSetup(format!(
            "digit row request needs {required} setup ring elements but prepared setup has {total_ring_elements}"
        )));
    }
    Ok(())
}

#[cfg(feature = "zk")]
fn zk_digit_rows_from_slot<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    row_len: usize,
    row_width: usize,
    total_ring_elements: usize,
    digits: &[[i8; D]],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    if digits.is_empty() {
        return Ok(vec![CyclotomicRing::zero(); row_len]);
    }
    if digits.len() > row_width {
        return Err(AkitaError::InvalidSetup(
            "ZK matrix digit columns exceed row width".to_string(),
        ));
    }
    validate_digit_row_request(row_len, row_width, total_ring_elements)?;
    mat_vec_mul_ntt_single_i8(slot, row_len, row_width, digits, MAX_I8_LOG_BASIS)
}

#[cfg(feature = "zk")]
fn zk_cyclic_digit_rows_from_slot<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    row_len: usize,
    row_width: usize,
    total_ring_elements: usize,
    digits: &[[i8; D]],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    if digits.is_empty() {
        return Ok(vec![CyclotomicRing::zero(); row_len]);
    }
    if digits.len() > row_width {
        return Err(AkitaError::InvalidSetup(
            "ZK matrix digit columns exceed row width".to_string(),
        ));
    }
    validate_digit_row_request(row_len, row_width, total_ring_elements)?;
    mat_vec_mul_ntt_single_i8_cyclic(slot, row_len, row_width, digits, MAX_I8_LOG_BASIS)
}

#[cfg(feature = "zk")]
fn zk_b_slot<F: FieldCore + CanonicalField, const D: usize>(
    prepared: &CpuPreparedSetup<F, D>,
) -> Result<&NttSlotCache<D>, AkitaError> {
    if let Some(slot) = prepared.ntt_zk_b.get() {
        return Ok(slot);
    }
    let total = prepared
        .expanded
        .zk_b_matrix
        .total_ring_elements_at::<D>()?;
    let slot = build_ntt_slot(prepared.expanded.zk_b_matrix.ring_view::<D>(1, total)?)?;
    let _ = prepared.ntt_zk_b.set(slot);
    prepared.ntt_zk_b.get().ok_or_else(|| {
        AkitaError::InvalidSetup("failed to initialize ZK B prepared slot".to_string())
    })
}

#[cfg(feature = "zk")]
fn zk_d_slot<F: FieldCore + CanonicalField, const D: usize>(
    prepared: &CpuPreparedSetup<F, D>,
) -> Result<&NttSlotCache<D>, AkitaError> {
    if let Some(slot) = prepared.ntt_zk_d.get() {
        return Ok(slot);
    }
    let total = prepared
        .expanded
        .zk_d_matrix
        .total_ring_elements_at::<D>()?;
    let slot = build_ntt_slot(prepared.expanded.zk_d_matrix.ring_view::<D>(1, total)?)?;
    let _ = prepared.ntt_zk_d.set(slot);
    prepared.ntt_zk_d.get().ok_or_else(|| {
        AkitaError::InvalidSetup("failed to initialize ZK D prepared slot".to_string())
    })
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
        let ntt_i8_capacity = selected_crt_i8_capacity_profile::<F, D>()?;
        let total = expanded.shared_matrix.total_ring_elements_at::<D>()?;
        let ntt_shared = build_ntt_slot(expanded.shared_matrix.ring_view::<D>(1, total)?)?;
        Ok(CpuPreparedSetup {
            expanded,
            ntt_shared,
            ntt_i8_capacity,
            #[cfg(feature = "zk")]
            ntt_zk_b: OnceLock::new(),
            #[cfg(feature = "zk")]
            ntt_zk_d: OnceLock::new(),
        })
    }

    fn prepared_expanded_setup<'a, const D: usize>(
        &self,
        prepared: &'a Self::PreparedSetup<D>,
    ) -> &'a AkitaExpandedSetup<F> {
        prepared.expanded.as_ref()
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
        match plan.input {
            DenseCommitInput::CachedDigits {
                digit_block_slices,
                log_basis,
            } => {
                let row_width = digit_block_slices.first().map_or(0, |digits| digits.len());
                mat_vec_mul_ntt_dense_digits_i8_trusted(
                    &prepared.ntt_shared,
                    plan.n_a,
                    row_width,
                    &digit_block_slices,
                    log_basis,
                )
            }
            DenseCommitInput::CoeffBlocks {
                block_slices,
                num_digits_commit,
                log_basis,
            } => {
                let row_width = block_slices.first().map_or(Ok(0usize), |block| {
                    block.len().checked_mul(num_digits_commit).ok_or_else(|| {
                        AkitaError::InvalidSetup("dense coefficient row width overflow".to_string())
                    })
                })?;
                if plan.n_a == 1 {
                    Ok(mat_vec_mul_ntt_i8_dense_single_row(
                        &prepared.ntt_shared,
                        row_width,
                        &block_slices,
                        num_digits_commit,
                        log_basis,
                    )?
                    .into_iter()
                    .map(|ring| vec![ring])
                    .collect())
                } else {
                    mat_vec_mul_ntt_i8_dense(
                        &prepared.ntt_shared,
                        plan.n_a,
                        row_width,
                        &block_slices,
                        num_digits_commit,
                        log_basis,
                    )
                }
            }
        }
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
        let active_a_cols = plan
            .block_len
            .checked_mul(plan.num_digits_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))?;
        let a_view = prepared
            .expanded
            .shared_matrix
            .ring_view::<D>(plan.n_a, active_a_cols)?;
        Ok(match plan.blocks {
            OneHotCommitBlocks::SingleChunk(blocks) => column_sweep_ajtai_single_chunk::<F, D>(
                &a_view,
                &blocks.block_slices()?,
                plan.n_a,
                active_a_cols,
                plan.num_digits_commit,
            ),
            OneHotCommitBlocks::MultiChunk(blocks) => column_sweep_ajtai_multi_chunk::<F, D>(
                &a_view,
                &blocks.block_slices()?,
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
        let active_a_cols = plan
            .block_len
            .checked_mul(plan.num_digits_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))?;
        let a_view = prepared
            .expanded
            .shared_matrix
            .ring_view::<D>(plan.n_a, active_a_cols)?;
        let a_rows = (0..plan.n_a)
            .map(|idx| a_view.row(idx))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(column_sweep_sparse(
            &a_rows,
            &plan.blocks.block_slices()?,
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
        let row_width = plan
            .block_len
            .checked_mul(plan.num_digits_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("recursive A width overflow".to_string()))?;
        if plan.num_digits_commit == 1 {
            mat_vec_mul_ntt_raw_i8_strided(
                &prepared.ntt_shared,
                plan.n_rows,
                row_width,
                plan.coeffs,
                plan.num_blocks,
                plan.block_len,
            )
        } else {
            let ring_elems: Vec<CyclotomicRing<F, D>> = plan
                .coeffs
                .iter()
                .map(|digit| {
                    let coeffs = from_fn(|k| F::from_i8(digit[k]));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect();
            mat_vec_mul_ntt_i8_strided(
                &prepared.ntt_shared,
                plan.n_rows,
                row_width,
                &ring_elems,
                plan.num_blocks,
                plan.block_len,
                plan.num_digits_commit,
                plan.log_basis,
            )
        }
    }
}

impl<F> DigitRowsComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        validate_digit_row_request(
            row_len,
            digits.len(),
            prepared
                .expanded
                .shared_matrix
                .total_ring_elements_at::<D>()?,
        )?;
        mat_vec_mul_ntt_single_i8(
            &prepared.ntt_shared,
            row_len,
            digits.len(),
            digits,
            log_basis,
        )
    }

    #[cfg(feature = "zk")]
    fn zk_b_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        zk_digit_rows_from_slot(
            zk_b_slot(prepared)?,
            row_len,
            row_width,
            prepared
                .expanded
                .zk_b_matrix
                .total_ring_elements_at::<D>()?,
            digits,
        )
    }

    #[cfg(feature = "zk")]
    fn zk_d_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        zk_digit_rows_from_slot(
            zk_d_slot(prepared)?,
            row_len,
            row_width,
            prepared
                .expanded
                .zk_d_matrix
                .total_ring_elements_at::<D>()?,
            digits,
        )
    }
}

impl<F> CyclicRowsComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        validate_digit_row_request(
            row_len,
            digits.len(),
            prepared
                .expanded
                .shared_matrix
                .total_ring_elements_at::<D>()?,
        )?;
        mat_vec_mul_ntt_single_i8_cyclic(
            &prepared.ntt_shared,
            row_len,
            digits.len(),
            digits,
            log_basis,
        )
    }

    #[cfg(feature = "zk")]
    fn zk_b_cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        zk_cyclic_digit_rows_from_slot(
            zk_b_slot(prepared)?,
            row_len,
            row_width,
            prepared
                .expanded
                .zk_b_matrix
                .total_ring_elements_at::<D>()?,
            digits,
        )
    }

    #[cfg(feature = "zk")]
    fn zk_d_cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        zk_cyclic_digit_rows_from_slot(
            zk_d_slot(prepared)?,
            row_len,
            row_width,
            prepared
                .expanded
                .zk_d_matrix
                .total_ring_elements_at::<D>()?,
            digits,
        )
    }
}

impl<F> RingSwitchComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn ring_switch_relation_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: RingSwitchRelationRowsPlan<'_, D>,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField,
    {
        let (d_cyclic, b_cyclic, a_quotients) = fused_split_eq_quotients_prover_bounds(
            &prepared.ntt_shared,
            plan.n_d,
            plan.n_b,
            plan.n_a,
            plan.e_hat,
            plan.t_hat,
            plan.z_segment,
            plan.z_folded_centered_inf_norm,
            plan.log_basis,
        )?;
        Ok(RingSwitchRelationRows {
            d_cyclic,
            b_cyclic,
            a_quotients,
        })
    }

    fn ring_switch_quotient_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: RingSwitchQuotientRowsPlan<'_, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: HalvingField,
    {
        let (_d_cyclic, _b_cyclic, a_quotients) = fused_split_eq_quotients_prover_bounds(
            &prepared.ntt_shared,
            0,
            0,
            plan.n_a,
            &[][..],
            &[][..],
            plan.z_segment,
            plan.z_folded_centered_inf_norm,
            1,
        )?;
        Ok(a_quotients)
    }
}

// ===========================================================================
// Open, source-typed operation boundary (PO1)
//
// Everything below this banner is the *new* prover compute boundary. It sits
// ABOVE the fixed representation-named row helpers above (`dense_commit_rows`,
// `onehot_commit_rows`, `ring_switch_relation_rows`, ...), which survive only
// as lower-level standard kernels. The new layer is open by *source type* `S`
// instead of closed over Akita's built-in plan shapes:
//
// - operation kernels (`RootCommitKernel`, `OpeningFoldKernel`, ...) take the
//   borrowed representation view as a generic type parameter `S`, so a
//   downstream crate can define its own local view type and implement the
//   relevant kernel for `CpuBackend` without modifying an Akita-owned enum;
// - root polynomials expose those views through capability traits
//   (`RootCommitSource`, `RootOpeningSource`, ...) whose associated view types
//   become the `S` a kernel runs over;
// - a prover run threads operation *contexts* (`OperationCtx`) bundled into a
//   `ProverComputeStack`, each carrying a backend plus its validated prepared
//   setup, so commitment / opening / tensor / ring-switch work can run on
//   independent backends while the protocol still sees canonical Akita outputs.
//
// PO1 establishes this surface additively: the kernel traits are skeletons with
// no Akita impls yet (the six representation nodes implement them in their own
// backend files), and the monolithic `ProverComputeBackend`/`AkitaPolyOps`
// boundary is intentionally left in place for PO4 to remove.
// ===========================================================================

/// Scalar operation parameters for an inner Ajtai commit.
///
/// The polynomial data lives in the borrowed commit source view (`S`); this
/// plan carries only the shape parameters the kernel needs to size its work.
#[derive(Debug, Clone, Copy)]
pub struct CommitInnerPlan {
    /// Number of A rows to produce.
    pub n_a: usize,
    /// Root block length in ring elements.
    pub block_len: usize,
    /// Number of balanced digits used for the A-side commit.
    pub num_digits_commit: usize,
    /// Number of balanced digits used when opening (recomposition width).
    pub num_digits_open: usize,
    /// Logarithm of the gadget basis.
    pub log_basis: u32,
}

/// Fold parameters for a fused evaluate-and-fold opening.
///
/// The base/ring split preserves the current distinction between base
/// multiplier points (scalar folds) and ring multiplier points (sparse
/// ring-multiplier accumulation).
#[derive(Debug, Clone, Copy)]
pub enum OpeningFoldPlan<'a, F: FieldCore, const D: usize> {
    /// Base multiplier point: scalar fold weights.
    Base {
        /// Outer evaluation scalars applied to the folded blocks.
        eval_outer_scalars: &'a [F],
        /// Per-block fold scalars.
        fold_scalars: &'a [F],
        /// Block length in ring elements.
        block_len: usize,
    },
    /// Ring multiplier point: ring-element fold weights.
    Ring {
        /// Outer evaluation ring multipliers applied to the folded blocks.
        eval_outer_scalars: &'a [CyclotomicRing<F, D>],
        /// Per-block fold ring multipliers.
        fold_scalars: &'a [CyclotomicRing<F, D>],
        /// Block length in ring elements.
        block_len: usize,
    },
}

/// Fused evaluate-and-fold output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningFoldOutput<F: FieldCore, const D: usize> {
    /// Evaluation of the polynomial at the opening point.
    pub eval: CyclotomicRing<F, D>,
    /// Folded witness rows in ring form.
    pub folded: Vec<CyclotomicRing<F, D>>,
}

/// Decompose + challenge-fold parameters for one opening.
#[derive(Debug, Clone, Copy)]
pub struct DecomposeFoldPlan<'a> {
    /// Sparse fold challenges, outermost first.
    pub challenges: &'a [SparseChallenge],
    /// Block length in ring elements.
    pub block_len: usize,
    /// Number of balanced digits.
    pub num_digits: usize,
    /// Logarithm of the gadget basis.
    pub log_basis: u32,
}

/// Batched decompose + fold parameters at one opening point.
///
/// Both the sparse-challenge and tensor-shaped fused batched paths are exposed
/// so a representation can keep its fast batched kernel rather than folding
/// each polynomial independently and aggregating later.
#[derive(Debug, Clone, Copy)]
pub enum DecomposeFoldBatchPlan<'a> {
    /// Sparse-challenge batched fold.
    Sparse {
        /// Sparse fold challenges, outermost first.
        challenges: &'a [SparseChallenge],
        /// Block length in ring elements.
        block_len: usize,
        /// Number of balanced digits.
        num_digits: usize,
        /// Logarithm of the gadget basis.
        log_basis: u32,
    },
    /// Tensor-shaped batched fold.
    Tensor {
        /// Tensor-structured fold challenges.
        tensor: &'a TensorChallenges,
        /// Block length in ring elements.
        block_len: usize,
        /// Number of balanced digits.
        num_digits: usize,
        /// Logarithm of the gadget basis.
        log_basis: u32,
    },
}

/// Scalar operation parameters for the fused ring-switch relation rows.
///
/// The decomposed witness data (`e_hat`, `t_hat`, centered `z` segment) and the
/// centered infinity norm live in the borrowed relation source view (`S`).
#[derive(Debug, Clone, Copy)]
pub struct RingSwitchRelationPlan {
    /// Number of D-side cyclic rows to produce.
    pub n_d: usize,
    /// Number of B-side cyclic rows to produce.
    pub n_b: usize,
    /// Number of A-side quotient rows to produce.
    pub n_a: usize,
    /// Logarithm of the gadget basis used to produce `e_hat` and `t_hat`.
    pub log_basis: u32,
}

/// Scalar operation parameters for additional public-row quotient rows.
#[derive(Debug, Clone, Copy)]
pub struct RingSwitchQuotientPlan {
    /// Number of A-side quotient rows to produce.
    pub n_a: usize,
}

/// Tensor-packed root witness alternatives produced by a tensor kernel.
///
/// This is an Akita-owned *output* sum type: the set of protocol output
/// alternatives is fixed, so an enum is the right model here. It is not a
/// closed *input-source* enum, which is the pattern the open boundary forbids.
#[derive(Debug, Clone)]
pub enum TensorPackedWitness<E: FieldCore> {
    /// Dense tensor-packed evaluations (universal fallback).
    Dense(Vec<E>),
    /// Sparse tensor-packed witness preserved when the source/backend can.
    Sparse(SparseExtensionOpeningWitness<E>),
}

/// Inner Ajtai commit kernel over a borrowed commit source view `S`.
///
/// `S` is the extensibility hook: a downstream crate defines its own commit
/// view and implements `RootCommitKernel<MyCommitView<'_>, F, D>` for a backend
/// (for example `CpuBackend`) without touching an Akita-owned enum. Built-in
/// Akita views reduce to the standard `*_commit_rows` helpers above.
pub trait RootCommitKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    /// Decomposed inner commitment blocks for `source`.
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: S,
        plan: CommitInnerPlan,
    ) -> Result<FlatDigitBlocks<D>, AkitaError>;

    /// Inner commitment that also preserves the recomposed inner rows.
    fn commit_inner_witness(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: S,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError>;
}

/// Fused ring-switch relation-rows kernel over a borrowed relation view `S`.
pub trait RingSwitchRelationKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    /// Fused D/B cyclic rows plus A-side quotient rows.
    fn relation_rows(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: S,
        plan: RingSwitchRelationPlan,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField;
}

/// Additional public-row quotient kernel over a borrowed quotient view `S`.
pub trait RingSwitchQuotientKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    /// A-side quotient rows for one additional public-row segment.
    fn quotient_rows(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: S,
        plan: RingSwitchQuotientPlan,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: HalvingField;
}

/// Opening fold / decompose-fold kernel over a borrowed opening view `S`.
///
/// `prepared` is optional because some opening folds do not need setup-owned
/// state; setup-dependent work stays explicitly tied to the backend context.
pub trait OpeningFoldKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    /// Fused fold + evaluation in one pass over the source.
    fn evaluate_and_fold(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
        plan: OpeningFoldPlan<'_, F, D>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError>;

    /// Decompose + challenge-fold step.
    fn decompose_fold(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F, D>, AkitaError>;
}

/// Batched decompose-fold kernel over a borrowed opening-batch view `S`.
pub trait OpeningBatchKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    /// Fused batched decompose-fold at one opening point.
    ///
    /// Returns `Ok(None)` when the backend/source has no fused batched path,
    /// `Ok(Some(_))` for the fused witness, and `Err(_)` when a batched fold was
    /// attempted but the input was rejected.
    fn decompose_fold_batch(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError>;
}

/// Tensor projection kernel over a borrowed tensor view `S` for opening at an
/// extension-field point of type `E`.
pub trait TensorProjectionKernel<S, F, E, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    /// Tensor-column partials at one logical point.
    fn column_partials(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>;

    /// Tensor-packed root witness, dense or sparse when available.
    fn packed_witness(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
    ) -> Result<TensorPackedWitness<E>, AkitaError>;

    /// Committed tensor-projected root polynomial.
    fn root_projection(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
    ) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
    where
        F: FromPrimitiveInt,
        E: RingSubfieldEncoding<F>;
}

/// Batched tensor projection kernel over a borrowed tensor-batch view `S`.
pub trait TensorProjectionBatchKernel<S, F, E, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    /// Tensor-column partials for a same-point batch.
    fn column_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>;

    /// Sparse linear combination of tensor-packed root witnesses.
    ///
    /// Returns `Ok(None)` when a sparse combination is unavailable for the whole
    /// batch and the caller must fall back to dense materialization.
    fn sparse_linear_combination(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>;
}

/// Shape metadata every root polynomial exposes.
///
/// This is the base capability: it carries no view and no backend work, so
/// shape-only APIs can require just `RootPolyShape` without pulling in commit,
/// opening, tensor, or direct-witness capabilities.
pub trait RootPolyShape<F, const D: usize>: Clone + Send + Sync
where
    F: FieldCore,
{
    /// Total number of ring elements in the polynomial.
    fn num_ring_elems(&self) -> usize;

    /// Total number of variables (`log2(num_ring_elems() * D)`).
    ///
    /// # Panics
    ///
    /// Panics if `num_ring_elems() * D` overflows `usize`. This is a prover-only
    /// shape helper and is not reachable from verifier paths.
    fn num_vars(&self) -> usize {
        let total = self
            .num_ring_elems()
            .checked_mul(D)
            .expect("ring elems * D overflow");
        debug_assert!(
            total.is_power_of_two(),
            "total field elements must be a power of 2"
        );
        total.trailing_zeros() as usize
    }
}

/// Capability: expose a borrowed commit source view for a `RootCommitKernel`.
pub trait RootCommitSource<F, const D: usize>: RootPolyShape<F, D>
where
    F: FieldCore,
{
    /// Borrowed commit view consumed by `RootCommitKernel`.
    type CommitView<'a>
    where
        Self: 'a;

    /// Borrow a commit view of this polynomial.
    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError>;
}

/// Capability: expose borrowed opening views for the opening fold kernels.
pub trait RootOpeningSource<F, const D: usize>: RootPolyShape<F, D>
where
    F: FieldCore,
{
    /// Borrowed single-poly opening view consumed by `OpeningFoldKernel`.
    type OpeningView<'a>
    where
        Self: 'a;

    /// Borrowed same-point batch view consumed by `OpeningBatchKernel`.
    type OpeningBatchView<'a>
    where
        Self: 'a;

    /// Borrow an opening view of this polynomial.
    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError>;

    /// Borrow a same-point batch opening view over several polynomials.
    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError>;
}

/// Capability: expose borrowed tensor views for the tensor projection kernels.
pub trait RootTensorSource<F, const D: usize>: RootPolyShape<F, D>
where
    F: FieldCore,
{
    /// Borrowed single-poly tensor view consumed by `TensorProjectionKernel`.
    type TensorView<'a>
    where
        Self: 'a;

    /// Borrowed same-point batch view consumed by `TensorProjectionBatchKernel`.
    type TensorBatchView<'a>
    where
        Self: 'a;

    /// Borrow a tensor view of this polynomial.
    ///
    /// The view is extension-field independent; the opening point type `E`
    /// enters only at kernel evaluation.
    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError>;

    /// Borrow a same-point batch tensor view over several polynomials.
    fn tensor_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::TensorBatchView<'a>, AkitaError>;
}

/// Capability: materialize a direct root witness for zero-fold openings.
///
/// This is an explicit opt-in, not a hidden default on every root polynomial:
/// only proving paths that may select a root-direct schedule require it.
pub trait DirectRootWitnessSource<F, const D: usize>: RootPolyShape<F, D>
where
    F: FieldCore,
{
    /// Materialize a direct root witness payload.
    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError>;
}

/// Umbrella marker for a fully capable Akita root polynomial.
///
/// Acceptable only as a convenience bundle on top-level APIs whose behavior can
/// reach every root capability through config-selected schedules. Lower-level
/// helpers should bound the smallest capability they actually use.
pub trait AkitaRootPoly<F, const D: usize>:
    RootPolyShape<F, D>
    + RootCommitSource<F, D>
    + RootOpeningSource<F, D>
    + RootTensorSource<F, D>
    + DirectRootWitnessSource<F, D>
where
    F: FieldCore,
{
}

impl<F, const D: usize, P> AkitaRootPoly<F, D> for P
where
    F: FieldCore,
    P: RootPolyShape<F, D>
        + RootCommitSource<F, D>
        + RootOpeningSource<F, D>
        + RootTensorSource<F, D>
        + DirectRootWitnessSource<F, D>,
{
}

/// A single operation context: a backend plus its validated prepared setup.
///
/// Construction validates the prepared setup against explicit expanded-setup
/// metadata, so a kernel may assume its context was validated. The fields are
/// private to keep that invariant: an `OperationCtx` cannot exist without going
/// through a validating constructor.
pub struct OperationCtx<'a, F, B, const D: usize>
where
    F: FieldCore + CanonicalField,
    B: ComputeBackendSetup<F>,
{
    backend: &'a B,
    prepared: &'a B::PreparedSetup<D>,
    _field: PhantomData<fn() -> F>,
}

impl<'a, F, B, const D: usize> OperationCtx<'a, F, B, D>
where
    F: FieldCore + CanonicalField,
    B: ComputeBackendSetup<F>,
{
    /// Build an operation context, validating `prepared` against `expanded`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] (via
    /// [`ComputeBackendSetup::validate_prepared_setup`]) when `prepared` was not
    /// built from `expanded`.
    pub fn new(
        backend: &'a B,
        prepared: &'a B::PreparedSetup<D>,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<Self, AkitaError> {
        backend.validate_prepared_setup::<D>(prepared, expanded)?;
        Ok(Self {
            backend,
            prepared,
            _field: PhantomData,
        })
    }

    /// Borrowed backend for this operation cluster.
    pub fn backend(&self) -> &'a B {
        self.backend
    }

    /// Borrowed prepared setup for this operation cluster.
    pub fn prepared(&self) -> &'a B::PreparedSetup<D> {
        self.prepared
    }
}

/// Per-operation-cluster prover compute stack.
///
/// Each cluster (commit / opening / tensor / ring-switch) carries its own
/// backend plus prepared context, so a proof may mix backends across clusters.
/// A CPU-only prover is the degenerate case where every cluster shares one
/// backend and prepared setup ([`ProverComputeStack::uniform`]).
pub struct ProverComputeStack<'a, F, const D: usize, C, O, T, R>
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F>,
    O: ComputeBackendSetup<F>,
    T: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
{
    commit: OperationCtx<'a, F, C, D>,
    opening: OperationCtx<'a, F, O, D>,
    tensor: OperationCtx<'a, F, T, D>,
    ring_switch: OperationCtx<'a, F, R, D>,
}

impl<'a, F, const D: usize, C, O, T, R> ProverComputeStack<'a, F, D, C, O, T, R>
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F>,
    O: ComputeBackendSetup<F>,
    T: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
{
    /// Build a heterogeneous prover stack, validating every contained context
    /// against the same expanded setup before any transcript work.
    ///
    /// # Errors
    ///
    /// Returns an error if any cluster's prepared setup fails validation.
    pub fn new(
        commit: (&'a C, &'a C::PreparedSetup<D>),
        opening: (&'a O, &'a O::PreparedSetup<D>),
        tensor: (&'a T, &'a T::PreparedSetup<D>),
        ring_switch: (&'a R, &'a R::PreparedSetup<D>),
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            commit: OperationCtx::new(commit.0, commit.1, expanded)?,
            opening: OperationCtx::new(opening.0, opening.1, expanded)?,
            tensor: OperationCtx::new(tensor.0, tensor.1, expanded)?,
            ring_switch: OperationCtx::new(ring_switch.0, ring_switch.1, expanded)?,
        })
    }

    /// Commit operation context.
    pub fn commit(&self) -> &OperationCtx<'a, F, C, D> {
        &self.commit
    }

    /// Opening / decompose-fold operation context.
    pub fn opening(&self) -> &OperationCtx<'a, F, O, D> {
        &self.opening
    }

    /// Tensor projection operation context.
    pub fn tensor(&self) -> &OperationCtx<'a, F, T, D> {
        &self.tensor
    }

    /// Ring-switch operation context.
    pub fn ring_switch(&self) -> &OperationCtx<'a, F, R, D> {
        &self.ring_switch
    }
}

impl<'a, F, B, const D: usize> ProverComputeStack<'a, F, D, B, B, B, B>
where
    F: FieldCore + CanonicalField,
    B: ComputeBackendSetup<F>,
{
    /// Build a CPU-only / single-backend stack where every operation cluster
    /// shares one backend and prepared setup. Validates the prepared setup once
    /// per cluster against `expanded`.
    ///
    /// # Errors
    ///
    /// Returns an error if the prepared setup fails validation.
    pub fn uniform(
        backend: &'a B,
        prepared: &'a B::PreparedSetup<D>,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<Self, AkitaError> {
        Self::new(
            (backend, prepared),
            (backend, prepared),
            (backend, prepared),
            (backend, prepared),
            expanded,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp64;
    #[cfg(feature = "zk")]
    use akita_types::FlatMatrix;
    use akita_types::SetupMatrixEnvelope;

    type F = Fp64<4294967197>;
    const D: usize = 32;

    fn setup_envelope(max_setup_len: usize) -> SetupMatrixEnvelope {
        SetupMatrixEnvelope {
            max_setup_len,
            #[cfg(feature = "zk")]
            max_zk_b_len: 1,
            #[cfg(feature = "zk")]
            max_zk_d_len: 1,
        }
    }

    #[cfg(feature = "zk")]
    fn setup_envelope_with_zk(
        max_setup_len: usize,
        max_zk_b_len: usize,
        max_zk_d_len: usize,
    ) -> SetupMatrixEnvelope {
        SetupMatrixEnvelope {
            max_setup_len,
            max_zk_b_len,
            max_zk_d_len,
        }
    }

    fn prepared() -> CpuPreparedSetup<F, D> {
        let setup =
            AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, 1, setup_envelope(32)).unwrap();
        CpuBackend.prepare_setup(&setup).unwrap()
    }

    #[cfg(feature = "zk")]
    fn direct_negacyclic_rows(
        matrix: &FlatMatrix<F>,
        row_len: usize,
        row_width: usize,
        digits: &[[i8; D]],
    ) -> Vec<CyclotomicRing<F, D>> {
        let view = matrix.ring_view::<D>(row_len, row_width).unwrap();
        let digit_rings = digits
            .iter()
            .map(|digit| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
                    F::from_i64(digit[idx] as i64)
                }))
            })
            .collect::<Vec<_>>();
        (0..row_len)
            .map(|row_idx| {
                let row_start = row_idx * row_width;
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (entry, digit) in view.as_slice()[row_start..row_start + digit_rings.len()]
                    .iter()
                    .zip(digit_rings.iter())
                {
                    entry.mul_accumulate_into(digit, &mut acc);
                }
                acc
            })
            .collect()
    }

    #[test]
    fn cpu_prepared_setup_identity_rejects_mismatched_setup() {
        let setup_a =
            AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, 1, setup_envelope(32)).unwrap();
        let setup_b =
            AkitaProverSetup::<F, D>::generate_with_capacity(9, 1, 1, setup_envelope(32)).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup_a).unwrap();

        CpuBackend
            .validate_prepared_setup::<D>(&prepared, setup_a.expanded.as_ref())
            .expect("matching setup");
        assert!(
            CpuBackend
                .validate_prepared_setup::<D>(&prepared, setup_b.expanded.as_ref())
                .is_err(),
            "prepared context must stay bound to the setup used to create it"
        );
    }

    #[test]
    fn cpu_prepared_setup_identity_accepts_equivalent_setup() {
        let setup_a =
            AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, 1, setup_envelope(32)).unwrap();
        let setup_b =
            AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, 1, setup_envelope(32)).unwrap();
        assert!(!Arc::ptr_eq(&setup_a.expanded, &setup_b.expanded));

        let prepared = CpuBackend.prepare_setup(&setup_a).unwrap();

        CpuBackend
            .validate_prepared_setup::<D>(&prepared, setup_b.expanded.as_ref())
            .expect("equivalent deterministic setup should validate");
    }

    #[test]
    fn cpu_prepared_setup_reports_checked_crt_capacity_profile() {
        let prepared = prepared();
        let profile = prepared.shared_ntt_profile();

        assert_eq!(profile.profile_id, "Q32/2xi32");
        assert_eq!(profile.num_primes, 2);
        assert_eq!(profile.limb_bits, 32);
        assert_eq!(profile.max_i8_log_basis, MAX_I8_LOG_BASIS);
        assert!(profile.balanced_digit_safe_width > 0);
        assert!(profile.raw_i8_safe_width > 0);
    }

    #[test]
    fn cpu_digit_rows_match_direct_kernel() {
        let prepared = prepared();
        let digits = vec![[1i8; D], [-1i8; D], [2i8; D]];
        let log_basis = 3;
        let via_backend = CpuBackend
            .digit_rows::<D>(&prepared, 2, &digits, log_basis)
            .expect("backend digit rows");
        let direct =
            mat_vec_mul_ntt_single_i8(&prepared.ntt_shared, 2, digits.len(), &digits, log_basis)
                .expect("direct digit rows");
        assert_eq!(via_backend, direct);
    }

    #[test]
    fn cpu_digit_rows_accept_logical_input_longer_than_stride() {
        let prepared = prepared();
        let digits = vec![[1i8; D]; 12];
        let log_basis = 3;
        let via_backend = CpuBackend
            .digit_rows::<D>(&prepared, 2, &digits, log_basis)
            .expect("backend digit rows");
        let direct =
            mat_vec_mul_ntt_single_i8(&prepared.ntt_shared, 2, digits.len(), &digits, log_basis)
                .expect("direct digit rows");
        assert_eq!(via_backend, direct);
    }

    #[test]
    fn cpu_cyclic_digit_rows_match_direct_kernel() {
        let prepared = prepared();
        let digits = vec![[1i8; D], [0i8; D], [-2i8; D], [3i8; D]];
        let log_basis = 3;
        let via_backend = CpuBackend
            .cyclic_digit_rows::<D>(&prepared, 2, &digits, log_basis)
            .expect("backend cyclic digit rows");
        let direct = mat_vec_mul_ntt_single_i8_cyclic(
            &prepared.ntt_shared,
            2,
            digits.len(),
            &digits,
            log_basis,
        )
        .expect("direct cyclic digit rows");
        assert_eq!(via_backend, direct);
    }

    #[cfg(feature = "zk")]
    #[test]
    fn cpu_zk_digit_rows_match_direct_negacyclic_product() {
        let row_len = 2;
        let row_width = 3;
        let setup = AkitaProverSetup::<F, D>::generate_with_capacity(
            8,
            1,
            1,
            setup_envelope_with_zk(32, row_len * row_width, row_len * row_width),
        )
        .unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let digits = vec![[1i8; D], [-2i8; D]];

        let b_rows = CpuBackend
            .zk_b_digit_rows::<D>(&prepared, row_len, row_width, &digits)
            .expect("backend zkB digit rows");
        let b_direct =
            direct_negacyclic_rows(setup.expanded.zk_b_matrix(), row_len, row_width, &digits);
        assert_eq!(b_rows, b_direct);

        let d_rows = CpuBackend
            .zk_d_digit_rows::<D>(&prepared, row_len, row_width, &digits)
            .expect("backend zkD digit rows");
        let d_direct =
            direct_negacyclic_rows(setup.expanded.zk_d_matrix(), row_len, row_width, &digits);
        assert_eq!(d_rows, d_direct);
    }

    #[test]
    fn cpu_ring_switch_relation_rows_match_direct_kernel() {
        let prepared = prepared();
        let e_hat = vec![[1i8; D], [2i8; D]];
        let t_hat = vec![[-1i8; D], [3i8; D]];
        let z_segment = vec![[1i32; D], [-2i32; D], [3i32; D]];
        let via_backend = CpuBackend
            .ring_switch_relation_rows::<D>(
                &prepared,
                RingSwitchRelationRowsPlan {
                    n_d: 1,
                    n_b: 1,
                    n_a: 1,
                    e_hat: &e_hat,
                    t_hat: &t_hat,
                    z_segment: &z_segment,
                    z_folded_centered_inf_norm: 3,
                    log_basis: 3,
                },
            )
            .expect("backend ring-switch relation rows");
        let direct =
            fused_split_eq_quotients(&prepared.ntt_shared, 1, 1, 1, &e_hat, &t_hat, &z_segment, 3)
                .expect("direct fused split-eq rows");
        assert_eq!(
            (
                via_backend.d_cyclic,
                via_backend.b_cyclic,
                via_backend.a_quotients
            ),
            direct
        );
    }
}
