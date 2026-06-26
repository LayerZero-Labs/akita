use crate::compute::plans::{
    DenseCommitRowsPlan, OneHotCommitRowsPlan, RecursiveWitnessCommitRowsPlan,
    RingSwitchQuotientRowsPlan, RingSwitchRelationRows, RingSwitchRelationRowsPlan,
    SparseRingCommitRowsPlan,
};
use crate::kernels::crt_ntt::NttSlotCacheAny;
use crate::AkitaProverSetup;
use akita_algebra::CyclotomicRing;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_types::{AkitaExpandedSetup, NttCacheKey};
use std::sync::Arc;

/// Shared prepared-setup contract for prover compute backends.
///
/// ## Runtime ring cutover (phase 1)
///
/// `PreparedSetup` is keyed by [`NttCacheKey`] at runtime. Kernel traits still take
/// `const D: usize` at the API boundary; `prepare_expanded::<D>` warms the envelope
/// slot for compile-time `D`. Per-fold ring dimensions become runtime cache keys in
/// later cutover waves (`ensure_ntt_slot` at prove entry).
pub trait ComputeBackendSetup<F>: Send + Sync
where
    F: FieldCore + CanonicalField,
{
    /// Backend-prepared setup (ring dimension is a runtime cache key, not a type param).
    type PreparedSetup: Send + Sync;

    /// Prepare backend state from a prover setup wrapper.
    fn prepare_setup<const D: usize>(
        &self,
        setup: &AkitaProverSetup<F, D>,
    ) -> Result<Self::PreparedSetup, AkitaError> {
        self.prepare_expanded::<D>(setup.expanded.clone())
    }

    /// Prepare backend state from already-expanded setup data.
    fn prepare_expanded<const D: usize>(
        &self,
        expanded: Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup, AkitaError>;

    /// Build the cache for `key` if absent.
    fn ensure_ntt_slot(
        &self,
        prepared: &mut Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<(), AkitaError>;

    /// Read a previously-warmed NTT cache slot.
    fn ntt_slot<'a>(
        &self,
        prepared: &'a Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<&'a NttSlotCacheAny, AkitaError>;

    /// Expanded setup used to prepare this backend context.
    fn prepared_expanded_setup<'a>(
        &self,
        prepared: &'a Self::PreparedSetup,
    ) -> &'a AkitaExpandedSetup<F>;

    /// Ensure explicit setup metadata and backend-prepared state match.
    fn validate_prepared_setup(
        &self,
        prepared: &Self::PreparedSetup,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<(), AkitaError> {
        let prepared_expanded = self.prepared_expanded_setup(prepared);
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
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
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
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
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
        prepared: &Self::PreparedSetup,
        plan: DenseCommitRowsPlan<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>;

    /// One-hot A-side commit rows.
    fn onehot_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: OneHotCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>;

    /// Sparse signed-ring A-side commit rows.
    fn sparse_ring_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: SparseRingCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>;

    /// Recursive witness A-side commit rows.
    fn recursive_witness_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RecursiveWitnessCommitRowsPlan<'_, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>;
}

/// Ring-switch relation operations for migrated proving work.
pub trait RingSwitchComputeBackend<F>: CyclicRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    /// Fused cyclic/quotient rows used by ring-switch finalization.
    fn ring_switch_relation_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RingSwitchRelationRowsPlan<'_, D>,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField;

    /// A-side quotient rows for an additional public-row segment.
    fn ring_switch_quotient_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
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
