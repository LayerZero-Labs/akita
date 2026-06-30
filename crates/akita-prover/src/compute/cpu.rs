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
use crate::kernels::crt_ntt::{build_ntt_slot, NttSlotCache};
use crate::kernels::linear::{
    fused_split_eq_quotients_prover_bounds, mat_vec_mul_ntt_dense_digits_i8_trusted,
    mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row, mat_vec_mul_ntt_i8_strided,
    mat_vec_mul_ntt_raw_i8_strided, mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic,
    selected_crt_i8_capacity_profile, CrtI8CapacityProfile,
};
use akita_algebra::CyclotomicRing;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_types::AkitaExpandedSetup;
use std::array::from_fn;
use std::sync::Arc;

/// CPU backend using the existing Rust/Rayon kernels.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

/// CPU-prepared setup for one field/ring-dimension pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuPreparedSetup<F: FieldCore, const D: usize> {
    expanded: Arc<AkitaExpandedSetup<F>>,
    ntt_shared: NttSlotCache<D>,
    ntt_i8_capacity: CrtI8CapacityProfile,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::backend::{
        ComputeBackendSetup, CyclicRowsComputeBackend, DigitRowsComputeBackend,
        RingSwitchComputeBackend,
    };
    use crate::compute::plans::RingSwitchRelationRowsPlan;
    use crate::kernels::linear::{
        fused_split_eq_quotients, mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic,
    };
    use crate::validation::MAX_I8_LOG_BASIS;
    use crate::AkitaProverSetup;
    use akita_field::Fp64;
    use akita_types::SetupMatrixEnvelope;
    use std::sync::Arc;

    type F = Fp64<4294967197>;
    const D: usize = 32;

    fn setup_envelope(max_setup_len: usize) -> SetupMatrixEnvelope {
        SetupMatrixEnvelope { max_setup_len }
    }

    fn prepared() -> CpuPreparedSetup<F, D> {
        let setup =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(32)).unwrap();
        CpuBackend.prepare_setup(&setup).unwrap()
    }

    #[test]
    fn cpu_prepared_setup_identity_rejects_mismatched_setup() {
        let setup_a =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(32)).unwrap();
        let setup_b =
            AkitaProverSetup::<F>::generate_with_capacity(9, 1, D, setup_envelope(32)).unwrap();
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
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(32)).unwrap();
        let setup_b =
            AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, setup_envelope(32)).unwrap();
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
