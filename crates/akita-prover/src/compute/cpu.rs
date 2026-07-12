use crate::backend::onehot::{column_sweep_ajtai_onehot, MultiChunkEntry, SingleChunkEntry};
use crate::backend::sparse_ring::column_sweep_sparse;
use crate::compute::backend::{
    CommitmentComputeBackend, ComputeBackendSetup, CyclicRowsComputeBackend,
    DigitRowsComputeBackend, RingSwitchComputeBackend,
};
use crate::compute::plans::{
    DenseCommitInput, DenseCommitRowsPlan, OneHotCommitBlocks, OneHotCommitRowsPlan,
    RecursiveWitnessCommitRowsPlan, RingSwitchQuotientRowsPlan, RingSwitchRelationRows,
    RingSwitchRelationRowsPlan, SparseRingCommitRowsPlan,
};
use crate::kernels::crt_ntt::{build_ntt_slot, NttCacheMap};
use crate::kernels::linear::{
    fused_ring_switch_relation_rows_prover_bounds, mat_vec_mul_ntt_dense_digits_i8_trusted,
    mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row, mat_vec_mul_ntt_i8_strided,
    mat_vec_mul_ntt_raw_i8_strided, mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic,
    selected_crt_i8_capacity_profile, CrtI8CapacityProfile,
};
use akita_algebra::CyclotomicRing;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_types::{dispatch_for_field, AkitaExpandedSetup, NttCacheKey, PreparedNttPlan};
use std::array::from_fn;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// CPU backend using the existing Rust/Rayon kernels.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

/// CPU-prepared setup keyed by runtime ring dimension.
///
/// NTT caches are keyed by [`NttCacheKey`]. [`ComputeBackendSetup::prepare_setup`]
/// registers every planned slot on the setup contract; additional slots may
/// be built lazily via [`ComputeBackendSetup::ensure_ntt_slot`].
#[derive(Debug)]
pub struct CpuPreparedSetup<F: FieldCore> {
    expanded: Arc<AkitaExpandedSetup<F>>,
    ntt_plan: PreparedNttPlan,
    shared_ntt: Mutex<NttCacheMap>,
    ntt_i8_capacity_by_ring_d: Mutex<HashMap<usize, CrtI8CapacityProfile>>,
    /// Keys promised at [`ComputeBackendSetup::prepare_setup`]; lazy builds outside
    /// this set emit a diagnostic warning.
    setup_contract_ntt_keys: Mutex<HashSet<NttCacheKey>>,
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

impl<F: FieldCore + CanonicalField> CpuPreparedSetup<F> {
    /// In-memory byte footprint of all shared setup NTT caches.
    pub fn shared_ntt_cache_bytes(&self) -> usize {
        self.shared_ntt
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|slot| slot.cache_bytes())
            .sum()
    }

    /// CRT/NTT profile and universal i8 capacity metadata for ring degree `D`.
    pub fn shared_ntt_profile<const D: usize>(&self) -> Result<PreparedCrtNttProfile, AkitaError> {
        self.ntt_i8_capacity_by_ring_d
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("NTT profile lock poisoned".into()))?
            .get(&D)
            .copied()
            .map(Into::into)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "prepared setup has no CRT/i8 capacity profile for ring_d={D}"
                ))
            })
    }
}

fn build_ntt_slot_for_key<F: FieldCore + CanonicalField>(
    expanded: &AkitaExpandedSetup<F>,
    key: NttCacheKey,
) -> Result<crate::kernels::crt_ntt::NttSlotCacheAny, AkitaError> {
    dispatch_for_field!(ProtocolDispatchSlot::Ntt, F, key.ring_d(), |RING_D| {
        let view = expanded
            .shared_matrix()
            .ring_view::<RING_D>(1, key.num_ring_elements())?;
        Ok(build_ntt_slot(view)?.into())
    })
}

fn record_ntt_profile_on_prepared<F: FieldCore>(
    prepared: &CpuPreparedSetup<F>,
    key: NttCacheKey,
    profile: CrtI8CapacityProfile,
) -> Result<(), AkitaError> {
    prepared
        .ntt_i8_capacity_by_ring_d
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("NTT profile lock poisoned".into()))?
        .entry(key.ring_d())
        .or_insert(profile);
    Ok(())
}

fn insert_ntt_slot_on_prepared<F: FieldCore + CanonicalField>(
    prepared: &CpuPreparedSetup<F>,
    key: NttCacheKey,
) -> Result<(), AkitaError> {
    let profile = dispatch_for_field!(ProtocolDispatchSlot::Ntt, F, key.ring_d(), |RING_D| {
        selected_crt_i8_capacity_profile::<F, RING_D>()
    })?;
    if prepared
        .shared_ntt
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?
        .contains_key(&key)
    {
        return record_ntt_profile_on_prepared(prepared, key, profile);
    }
    let slot = build_ntt_slot_for_key(prepared.expanded.as_ref(), key)?;
    let mut cache = prepared
        .shared_ntt
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?;
    if cache.contains_key(&key) {
        drop(cache);
        return record_ntt_profile_on_prepared(prepared, key, profile);
    }
    cache.insert(key, Arc::new(slot));
    drop(cache);
    record_ntt_profile_on_prepared(prepared, key, profile)
}

fn register_setup_contract_ntt_slot_on_prepared<F: FieldCore + CanonicalField>(
    prepared: &CpuPreparedSetup<F>,
    key: NttCacheKey,
) -> Result<(), AkitaError> {
    if !prepared
        .shared_ntt
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?
        .contains_key(&key)
    {
        insert_ntt_slot_on_prepared(prepared, key)?;
    }
    prepared
        .setup_contract_ntt_keys
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("NTT contract lock poisoned".into()))?
        .insert(key);
    Ok(())
}

fn ensure_ntt_slot_on_prepared<F: FieldCore + CanonicalField>(
    prepared: &CpuPreparedSetup<F>,
    key: NttCacheKey,
) -> Result<(), AkitaError> {
    if prepared
        .shared_ntt
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?
        .contains_key(&key)
    {
        return Ok(());
    }
    if !prepared
        .setup_contract_ntt_keys
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("NTT contract lock poisoned".into()))?
        .contains(&key)
    {
        tracing::warn!(
            target: "akita_prover::ntt_cache",
            ring_d = key.ring_d(),
            num_ring_elements = key.num_ring_elements(),
            setup_contract_keys = prepared
                .setup_contract_ntt_keys
                .lock()
                .map_err(|_| AkitaError::InvalidSetup("NTT contract lock poisoned".into()))?
                .len(),
            "building NTT cache slot outside setup prepare contract; \
             setup envelope or prepare path is likely undersized for this commit/prove path"
        );
    }
    insert_ntt_slot_on_prepared(prepared, key)
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
    let required = checked_ntt_prefix(row_len, row_width, "digit row request")?;
    if required > total_ring_elements {
        return Err(AkitaError::InvalidSetup(format!(
            "digit row request needs {required} setup ring elements but prepared setup has {total_ring_elements}"
        )));
    }
    Ok(())
}

fn checked_ntt_prefix(
    row_count: usize,
    row_width: usize,
    operation: &str,
) -> Result<usize, AkitaError> {
    row_count.checked_mul(row_width).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "{operation} NTT prefix overflows: row_count={row_count} row_width={row_width}"
        ))
    })
}

impl<F> ComputeBackendSetup<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    type PreparedSetup = CpuPreparedSetup<F>;

    fn prepare_expanded<const D: usize>(
        &self,
        expanded: Arc<AkitaExpandedSetup<F>>,
        plan: PreparedNttPlan,
    ) -> Result<Self::PreparedSetup, AkitaError> {
        let prepared = CpuPreparedSetup {
            expanded,
            ntt_plan: plan.clone(),
            shared_ntt: Mutex::new(NttCacheMap::new()),
            ntt_i8_capacity_by_ring_d: Mutex::new(HashMap::new()),
            setup_contract_ntt_keys: Mutex::new(HashSet::new()),
        };
        for &key in plan.slots() {
            register_setup_contract_ntt_slot_on_prepared(&prepared, key)?;
        }
        Ok(prepared)
    }

    fn ensure_ntt_slot(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<(), AkitaError> {
        ensure_ntt_slot_on_prepared(prepared, key)
    }

    fn with_ntt_slot<R>(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
        f: impl FnOnce(&crate::kernels::crt_ntt::NttSlotCacheAny) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        let slot = {
            let cache = prepared
                .shared_ntt
                .lock()
                .map_err(|_| AkitaError::InvalidSetup("NTT cache lock poisoned".into()))?;
            Arc::clone(cache.get(&key).ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "prepared setup NTT slot not warmed for ring_d={} num_ring_elements={}",
                    key.ring_d(),
                    key.num_ring_elements()
                ))
            })?)
        };
        f(&slot)
    }

    fn prepared_expanded_setup<'a>(
        &self,
        prepared: &'a Self::PreparedSetup,
    ) -> &'a AkitaExpandedSetup<F> {
        prepared.expanded.as_ref()
    }

    fn prepared_ntt_plan<'a>(&self, prepared: &'a Self::PreparedSetup) -> &'a PreparedNttPlan {
        &prepared.ntt_plan
    }
}

impl<F> CommitmentComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn dense_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: DenseCommitRowsPlan<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        match plan.input {
            DenseCommitInput::CachedDigits {
                digit_block_slices,
                log_basis,
            } => {
                let row_width = digit_block_slices.first().map_or(0, |digits| digits.len());
                let required = checked_ntt_prefix(plan.n_a, row_width, "dense commit")?;
                CpuBackend.with_planned_ntt_slot::<D, _>(prepared, required, |ntt| {
                    mat_vec_mul_ntt_dense_digits_i8_trusted(
                        ntt,
                        plan.n_a,
                        row_width,
                        &digit_block_slices,
                        log_basis,
                    )
                })
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
                    let required = checked_ntt_prefix(plan.n_a, row_width, "dense commit")?;
                    CpuBackend.with_planned_ntt_slot::<D, _>(prepared, required, |ntt| {
                        Ok(mat_vec_mul_ntt_i8_dense_single_row(
                            ntt,
                            row_width,
                            &block_slices,
                            num_digits_commit,
                            log_basis,
                        )?
                        .into_iter()
                        .map(|ring| vec![ring])
                        .collect())
                    })
                } else {
                    let required = checked_ntt_prefix(plan.n_a, row_width, "dense commit")?;
                    CpuBackend.with_planned_ntt_slot::<D, _>(prepared, required, |ntt| {
                        mat_vec_mul_ntt_i8_dense(
                            ntt,
                            plan.n_a,
                            row_width,
                            &block_slices,
                            num_digits_commit,
                            log_basis,
                        )
                    })
                }
            }
        }
    }

    fn onehot_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
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
            OneHotCommitBlocks::SingleChunk(blocks) => {
                column_sweep_ajtai_onehot::<SingleChunkEntry, F, D>(
                    &a_view,
                    &blocks.block_slices()?,
                    plan.n_a,
                    active_a_cols,
                    plan.num_digits_commit,
                )
            }
            OneHotCommitBlocks::MultiChunk(blocks) => {
                column_sweep_ajtai_onehot::<MultiChunkEntry, F, D>(
                    &a_view,
                    &blocks.block_slices()?,
                    plan.n_a,
                    active_a_cols,
                    plan.num_digits_commit,
                )
            }
        })
    }

    fn sparse_ring_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
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
        prepared: &Self::PreparedSetup,
        plan: RecursiveWitnessCommitRowsPlan<'_, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        let row_width = plan
            .block_len
            .checked_mul(plan.num_digits_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("recursive A width overflow".to_string()))?;
        if plan.num_digits_commit == 1 {
            let required = checked_ntt_prefix(plan.n_rows, row_width, "recursive witness")?;
            CpuBackend.with_planned_ntt_slot::<D, _>(prepared, required, |ntt| {
                mat_vec_mul_ntt_raw_i8_strided(
                    ntt,
                    plan.n_rows,
                    row_width,
                    plan.coeffs,
                    plan.num_blocks,
                    plan.block_len,
                )
            })
        } else {
            let ring_elems: Vec<CyclotomicRing<F, D>> = plan
                .coeffs
                .iter()
                .map(|digit| {
                    let coeffs = from_fn(|k| F::from_i8(digit[k]));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect();
            let required = checked_ntt_prefix(plan.n_rows, row_width, "recursive witness")?;
            CpuBackend.with_planned_ntt_slot::<D, _>(prepared, required, |ntt| {
                mat_vec_mul_ntt_i8_strided(
                    ntt,
                    plan.n_rows,
                    row_width,
                    &ring_elems,
                    plan.num_blocks,
                    plan.block_len,
                    plan.num_digits_commit,
                    plan.log_basis,
                )
            })
        }
    }
}

impl<F> DigitRowsComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
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
        let required = checked_ntt_prefix(row_len, digits.len(), "digit rows")?;
        CpuBackend.with_planned_ntt_slot::<D, _>(prepared, required, |ntt| {
            mat_vec_mul_ntt_single_i8(ntt, row_len, digits.len(), digits, log_basis)
        })
    }
}

impl<F> CyclicRowsComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
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
        let required = checked_ntt_prefix(row_len, digits.len(), "cyclic digit rows")?;
        CpuBackend.with_planned_ntt_slot::<D, _>(prepared, required, |ntt| {
            mat_vec_mul_ntt_single_i8_cyclic(ntt, row_len, digits.len(), digits, log_basis)
        })
    }
}

impl<F> RingSwitchComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn ring_switch_relation_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RingSwitchRelationRowsPlan<'_, D>,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField,
    {
        let required = [
            checked_ntt_prefix(plan.n_d, plan.e_hat.len(), "ring-switch D relation")?,
            checked_ntt_prefix(plan.n_b, plan.t_hat.len(), "ring-switch B relation")?,
            checked_ntt_prefix(plan.n_a, plan.z_segment.len(), "ring-switch A relation")?,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        CpuBackend.with_planned_ntt_slot::<D, _>(prepared, required, |ntt| {
            let (d_cyclic, b_cyclic, a_quotients) = fused_ring_switch_relation_rows_prover_bounds(
                ntt,
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
        })
    }

    fn ring_switch_quotient_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RingSwitchQuotientRowsPlan<'_, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: HalvingField,
    {
        let required = checked_ntt_prefix(plan.n_a, plan.z_segment.len(), "ring-switch quotient")?;
        CpuBackend.with_planned_ntt_slot::<D, _>(prepared, required, |ntt| {
            let (_d_cyclic, _b_cyclic, a_quotients) =
                fused_ring_switch_relation_rows_prover_bounds(
                    ntt,
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::backend::{
        ComputeBackendSetup, CyclicRowsComputeBackend, DigitRowsComputeBackend,
        RingSwitchComputeBackend,
    };
    use crate::compute::plans::RingSwitchRelationRowsPlan;
    use crate::kernels::linear::{
        fused_ring_switch_relation_rows, mat_vec_mul_ntt_single_i8,
        mat_vec_mul_ntt_single_i8_cyclic,
    };
    use crate::validation::MAX_I8_LOG_BASIS;
    use crate::AkitaProverSetup;
    use akita_field::Prime64Offset59;
    use akita_types::SetupMatrixEnvelope;
    use std::sync::Arc;

    type F = Prime64Offset59;
    const D: usize = 64;

    fn setup_envelope(max_setup_len: usize) -> SetupMatrixEnvelope {
        SetupMatrixEnvelope { max_setup_len }
    }

    fn prepared() -> CpuPreparedSetup<F> {
        let setup =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
        CpuBackend
            .prepare_setup(
                &setup,
                &akita_types::PreparedNttPlan::base_envelope(setup.expanded.as_ref()).unwrap(),
            )
            .unwrap()
    }

    #[test]
    fn cpu_prepared_setup_identity_rejects_mismatched_setup() {
        let setup_a =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
        let setup_b =
            AkitaProverSetup::<F>::generate_with_capacity(9, 1, D, setup_envelope(D)).unwrap();
        let prepared = CpuBackend
            .prepare_setup(
                &setup_a,
                &akita_types::PreparedNttPlan::base_envelope(setup_a.expanded.as_ref()).unwrap(),
            )
            .unwrap();

        CpuBackend
            .validate_prepared_setup(&prepared, setup_a.expanded.as_ref())
            .expect("matching setup");
        assert!(
            CpuBackend
                .validate_prepared_setup(&prepared, setup_b.expanded.as_ref())
                .is_err(),
            "prepared context must stay bound to the setup used to create it"
        );
    }

    #[test]
    fn cpu_prepared_setup_identity_accepts_equivalent_setup() {
        let setup_a =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
        let setup_b =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
        assert!(!Arc::ptr_eq(&setup_a.expanded, &setup_b.expanded));

        let prepared = CpuBackend
            .prepare_setup(
                &setup_a,
                &akita_types::PreparedNttPlan::base_envelope(setup_a.expanded.as_ref()).unwrap(),
            )
            .unwrap();

        CpuBackend
            .validate_prepared_setup(&prepared, setup_b.expanded.as_ref())
            .expect("equivalent deterministic setup should validate");
    }

    #[test]
    fn cpu_prepared_setup_reports_checked_crt_capacity_profile() {
        let prepared = prepared();
        let profile = prepared.shared_ntt_profile::<D>().expect("profile");

        assert_eq!(profile.profile_id, "Q64/3xi32");
        assert_eq!(profile.num_primes, 3);
        assert_eq!(profile.limb_bits, 32);
        assert_eq!(profile.max_i8_log_basis, MAX_I8_LOG_BASIS);
        assert!(profile.balanced_digit_safe_width > 0);
        assert!(profile.raw_i8_safe_width > 0);
    }

    #[test]
    fn prepare_setup_registers_envelope_ntt_contract() {
        let setup =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
        let prepared = CpuBackend
            .prepare_setup(
                &setup,
                &akita_types::PreparedNttPlan::base_envelope(setup.expanded.as_ref()).unwrap(),
            )
            .expect("prepared");
        assert!(prepared.shared_ntt_cache_bytes() > 0);
        let envelope_key =
            NttCacheKey::from_envelope(setup.expanded.as_ref(), D).expect("envelope key");
        CpuBackend
            .with_ntt_slot(&prepared, envelope_key, |_| Ok(()))
            .expect("envelope slot from setup contract");
    }

    #[test]
    fn prepared_plan_sorts_coalesces_and_checks_compression_requirements() {
        let setup =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
        let plan = PreparedNttPlan::with_compression_requirements(
            setup.expanded.as_ref(),
            [(32, 1), (32, 2), (32, 1)],
        )
        .expect("plan");

        assert_eq!(plan.slots().len(), 2);
        assert_eq!(plan.slots()[0].ring_d(), 32);
        assert_eq!(plan.slots()[0].num_ring_elements(), 2);
        assert_eq!(plan.slots()[1].ring_d(), D);
        assert_eq!(
            plan.resolve(32, 1).expect("containing slot"),
            plan.slots()[0]
        );

        assert!(
            PreparedNttPlan::with_compression_requirements(setup.expanded.as_ref(), [(8, 1)])
                .is_err()
        );
        let oversized_prefix = setup
            .expanded
            .shared_matrix()
            .total_ring_elements_at_dyn(32)
            .expect("D=32 setup view")
            + 1;
        assert!(PreparedNttPlan::with_compression_requirements(
            setup.expanded.as_ref(),
            [(32, oversized_prefix)]
        )
        .is_err());
    }

    #[test]
    fn prepare_expanded_with_envelope_ntt_builds_envelope_slot() {
        let setup =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
        let prepared = CpuBackend
            .prepare_expanded_with_envelope_ntt::<D>(setup.expanded.clone())
            .expect("prepared");
        assert!(prepared.shared_ntt_cache_bytes() > 0);
        let envelope_key =
            NttCacheKey::from_envelope(setup.expanded.as_ref(), D).expect("envelope key");
        CpuBackend
            .with_ntt_slot(&prepared, envelope_key, |_| Ok(()))
            .expect("envelope slot available");
    }

    #[test]
    fn prepared_plan_eagerly_warms_slots_and_lazy_fallback_remains_available() {
        let setup =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(D)).unwrap();
        let plan =
            PreparedNttPlan::with_compression_requirements(setup.expanded.as_ref(), [(32, 1)])
                .expect("plan");
        let prepared = CpuBackend.prepare_setup(&setup, &plan).expect("prepared");
        assert_eq!(prepared.shared_ntt.lock().unwrap().len(), 2);

        CpuBackend
            .with_planned_ntt_slot::<32, _>(&prepared, 1, |_| Ok(()))
            .expect("eager compression slot");
        assert_eq!(prepared.shared_ntt.lock().unwrap().len(), 2);

        CpuBackend
            .with_planned_ntt_slot::<16, _>(&prepared, 1, |_| Ok(()))
            .expect("checked lazy fallback");
        assert_eq!(prepared.shared_ntt.lock().unwrap().len(), 3);

        assert!(CpuBackend
            .with_planned_ntt_slot::<16, _>(&prepared, 99_999, |_| Ok(()))
            .is_err());
        assert_eq!(prepared.shared_ntt.lock().unwrap().len(), 3);
    }

    #[test]
    fn planned_slot_serves_zero_work_without_constructing_zero_length_cache() {
        let prepared = prepared();
        let before = prepared.shared_ntt.lock().unwrap().len();
        CpuBackend
            .with_planned_ntt_slot::<D, _>(&prepared, 0, |_| Ok(()))
            .expect("zero work uses the containing prepared slot");
        assert_eq!(prepared.shared_ntt.lock().unwrap().len(), before);
    }

    #[test]
    fn cpu_digit_rows_match_direct_kernel() {
        let prepared = prepared();
        let digits = vec![[1i8; D], [-1i8; D], [2i8; D]];
        let log_basis = 3;
        let via_backend = CpuBackend
            .digit_rows::<D>(&prepared, 2, &digits, log_basis)
            .expect("backend digit rows");
        let direct = CpuBackend
            .with_planned_ntt_slot::<D, _>(&prepared, 2 * digits.len(), |ntt| {
                mat_vec_mul_ntt_single_i8(ntt, 2, digits.len(), &digits, log_basis)
            })
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
        let direct = CpuBackend
            .with_planned_ntt_slot::<D, _>(&prepared, 2 * digits.len(), |ntt| {
                mat_vec_mul_ntt_single_i8(ntt, 2, digits.len(), &digits, log_basis)
            })
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
        let direct = CpuBackend
            .with_planned_ntt_slot::<D, _>(&prepared, 2 * digits.len(), |ntt| {
                mat_vec_mul_ntt_single_i8_cyclic(ntt, 2, digits.len(), &digits, log_basis)
            })
            .expect("direct cyclic digit rows");
        assert_eq!(via_backend, direct);
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
        let direct = CpuBackend
            .with_planned_ntt_slot::<D, _>(&prepared, z_segment.len(), |ntt| {
                fused_ring_switch_relation_rows(ntt, 1, 1, 1, &e_hat, &t_hat, &z_segment, 3)
            })
            .expect("direct fused ring-switch relation rows");
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
