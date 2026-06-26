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
/// `PreparedSetup` is keyed by [`NttCacheKey`] at runtime. [`Self::prepare_setup`]
/// registers the minimum setup contract: one full-envelope slot at compile-time `D`.
/// [`Self::prepare_expanded`] leaves the cache empty. Commit/prove may call
/// [`Self::ensure_ntt_slot`] lazily; building outside the setup contract emits a
/// diagnostic warning (see warm-cache policy in `specs/runtime-ring-cutover.md`).
pub trait ComputeBackendSetup<F>: Send + Sync
where
    F: FieldCore + CanonicalField,
{
    /// Backend-prepared setup (ring dimension is a runtime cache key, not a type param).
    type PreparedSetup: Send + Sync;

    /// Prepare backend state from a prover setup wrapper.
    ///
    /// Builds the minimum NTT setup contract: the full-envelope slot at `D`.
    fn prepare_setup<const D: usize>(
        &self,
        setup: &AkitaProverSetup<F, D>,
    ) -> Result<Self::PreparedSetup, AkitaError> {
        let prepared = self.prepare_expanded::<D>(setup.expanded.clone())?;
        self.register_setup_contract_envelope_ntt::<D>(&prepared, setup.expanded.as_ref())?;
        Ok(prepared)
    }

    /// Prepare backend state from already-expanded setup data.
    ///
    /// Returns an empty NTT cache. Prefer [`Self::prepare_setup`] for commit/prove hosts.
    fn prepare_expanded<const D: usize>(
        &self,
        expanded: Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup, AkitaError>;

    /// Empty prepared state plus the full-envelope NTT slot for `D`.
    ///
    /// Ephemeral rebuild paths (suffix cross-`D`, ring-switch commit) use this instead of
    /// [`Self::prepare_setup`]. The slot is built via [`Self::ensure_ntt_slot`], so keys
    /// outside the setup prepare contract emit a sizing diagnostic.
    fn prepare_expanded_with_envelope_ntt<const D: usize>(
        &self,
        expanded: Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup, AkitaError> {
        let prepared = self.prepare_expanded::<D>(expanded.clone())?;
        let key = NttCacheKey::from_envelope(expanded.as_ref(), D)?;
        self.ensure_ntt_slot(&prepared, key)?;
        Ok(prepared)
    }

    /// Register the full-envelope NTT slot at compile-time ring degree `D` on the
    /// setup prepare contract.
    fn register_setup_contract_envelope_ntt<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<(), AkitaError> {
        let key = NttCacheKey::from_envelope(expanded, D)?;
        self.register_setup_contract_ntt_slot(prepared, key)
    }

    /// Build `key` as part of the setup prepare contract (no oversize warning).
    fn register_setup_contract_ntt_slot(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<(), AkitaError> {
        self.ensure_ntt_slot(prepared, key)
    }

    /// Build the cache for `key` if absent.
    ///
    /// Keys outside the setup prepare contract may still be built (fail-open for
    /// correctness) but should log a diagnostic warning on the CPU backend.
    fn ensure_ntt_slot(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<(), AkitaError>;

    /// Run `f` with a warmed NTT slot for `key`.
    fn with_ntt_slot<R>(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
        f: impl FnOnce(&NttSlotCacheAny) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError>;

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
