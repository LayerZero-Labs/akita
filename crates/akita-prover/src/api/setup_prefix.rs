//! Preprocessing helpers for setup-prefix commitment artifacts (slice 02B).

use crate::backend::DensePoly;
use crate::compute::{
    CommitmentExecutionPlan, CommitmentExecutor, CommitmentSource, CommitmentStatePolicy,
    IntoPortableCommitmentState,
};
#[cfg(test)]
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_types::{
    AkitaExpandedSetup, RingVec, SetupPrefixPublicCommitment, SetupPrefixSlot, SetupPrefixSlotId,
};
use jolt_field::{CanonicalEncoding, Field};

/// Commit one actual power-of-two flat prefix of the shared setup matrix.
///
/// The witness is the coefficient form of `S^flat[0..n_prefix]`. The caller
/// supplies the checked slot identity whose profile commits that exact prefix.
///
/// # Errors
///
/// Returns an error if shapes overflow, the prefix does not fit the setup matrix,
/// or backend commitment fails.
pub fn commit_setup_prefix<F, SP>(
    expanded: &AkitaExpandedSetup<F>,
    executor: &CommitmentExecutor<'_, F, SP>,
    id: &SetupPrefixSlotId,
) -> Result<SetupPrefixSlot<F>, AkitaError>
where
    F: Field + CanonicalEncoding + 'static,
    SP: CommitmentStatePolicy<F>,
    SP::State: IntoPortableCommitmentState<F>,
{
    let commitment_profile = &id.commitment_profile;
    commitment_profile.validate(
        commitment_profile
            .inner
            .matrix
            .sis_modulus_profile()
            .field_bits(),
    )?;
    commitment_profile.validate_setup_prefix_geometry(id.natural_len)?;
    let n_prefix = id.n_prefix()?;
    let ring_dimension = commitment_profile.inner.matrix.ring_dimension();
    let committed_n_prefix = 1usize
        .checked_shl(commitment_profile.group.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("setup-prefix domain overflow".into()))?;
    if committed_n_prefix != n_prefix || !n_prefix.is_multiple_of(ring_dimension) {
        return Err(AkitaError::InvalidSetup(
            "requested setup prefix does not match the full committed domain".to_string(),
        ));
    }
    let full_prefix_ring_slots = n_prefix / ring_dimension;
    let witness_ring_slots = commitment_profile
        .blocks
        .live_blocks
        .checked_mul(commitment_profile.blocks.positions_per_block)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("setup prefix witness shape overflow".to_string())
        })?;
    if witness_ring_slots != full_prefix_ring_slots {
        return Err(AkitaError::InvalidSetup(format!(
            "level params witness shape {witness_ring_slots} ring slots does not match full setup prefix {full_prefix_ring_slots}"
        )));
    }

    let available_field_len = expanded.shared_matrix().num_field_elements();
    if n_prefix > available_field_len {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length exceeds shared matrix capacity".to_string(),
        ));
    }

    let source_coefficients = expanded
        .shared_matrix()
        .as_field_slice()
        .get(..n_prefix)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("setup prefix length exceeds shared matrix capacity".into())
        })?
        .to_vec();
    let source =
        DensePoly::from_field_evals(commitment_profile.group.num_vars(), source_coefficients)?;
    let plan = CommitmentExecutionPlan::for_setup_prefix(id)?;
    let sources: [&dyn CommitmentSource<F>; 1] = [&source];
    executor.preflight_portable_export(&plan, &sources)?;
    let output = executor.execute_full(&plan, &sources)?;
    let (terminal_payload, prover_state) = output.into_parts();
    let terminal_ring_dim = plan
        .compression()
        .ok_or_else(|| AkitaError::InvalidSetup("setup-prefix plan omitted compression".into()))?
        .maps()
        .last()
        .ok_or(AkitaError::InvalidProof)?
        .ring_dimension();
    let commitment_payload =
        RingVec::from_coeffs_with_ring_dim(terminal_payload.into_coeffs(), terminal_ring_dim)?;
    let hint = prover_state.into_portable_hint()?;
    Ok(SetupPrefixSlot {
        id: id.clone(),
        commitment: SetupPrefixPublicCommitment {
            rows: vec![commitment_payload],
        },
        hint,
    })
}

#[cfg(test)]
fn extract_setup_prefix_ring_elems<F, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    full_prefix_ring_slots: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: Field,
{
    let fields = expanded.shared_matrix().as_field_slice();
    let full_prefix_field_len = full_prefix_ring_slots.checked_mul(D).ok_or_else(|| {
        AkitaError::InvalidSetup("setup prefix full field length overflow".to_string())
    })?;
    if full_prefix_field_len > fields.len() {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length exceeds shared matrix capacity".to_string(),
        ));
    }

    fields[..full_prefix_field_len]
        .chunks_exact(D)
        .map(|coeffs| {
            let mut ring = CyclotomicRing::zero();
            ring.coefficients_mut().copy_from_slice(coeffs);
            Ok(ring)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{
        BackendKindId, CommitmentExecutorBuilder, CompressionOperationCapabilities,
        ComputeBackendSetup, CpuBackend, CpuCompressionOperation, CpuInnerCommitOperation,
        CpuOuterCommitOperation, DenseType, NttExecutionRequirements, PolynomialType,
        PortableStatePolicy, ResidentStatePolicy, StageDimensionCapabilities, StageResources,
    };
    use crate::AkitaProverSetup;
    use akita_challenges::SparseChallengeConfig;
    use akita_serialization::AkitaSerialize;
    use akita_types::{
        active_setup_field_len, setup_prefix_precommitted_params, CommittedGroupParams,
        CompressionChainPlan, InnerCommitMatrixParams, OpeningClaimsLayout,
        OuterCommitMatrixParams, SetupMatrixCapacity, SisModulusProfileId, SisTableKey,
    };
    use jolt_field::Prime128OffsetA7F7 as F;
    use std::sync::Arc;

    type MissingExportExecutor<'a> = (
        CommitmentExecutor<'a, F, ResidentStatePolicy>,
        Arc<CpuInnerCommitOperation<'a, F>>,
        Arc<CpuCompressionOperation<'a, F>>,
    );

    fn prefix_level_params(ring_dimension: usize) -> CommittedGroupParams {
        let mut params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            ring_dimension,
            3,
            2,
            3,
            2,
            SparseChallengeConfig::production_for_ring_dim(ring_dimension)
                .expect("production challenge"),
        )
        .with_decomp(
            1,
            4,
            akita_types::sis::compute_num_digits_field_width(128, 3),
            2,
            2,
        )
        .expect("level params");
        let inner = params.inner().matrix;
        let coeff_linf_bound = *akita_types::sis::inner_coeff_linf_bounds(
            inner.sis_modulus_profile(),
            u32::try_from(ring_dimension).expect("ring dimension"),
        )
        .first()
        .expect("audited setup-prefix A bound");
        params.own_group_mut().profile.inner.matrix =
            InnerCommitMatrixParams::try_new_with_min_rank(
                SisTableKey {
                    policy: inner.security_policy(),
                    table_digest: inner
                        .sis_table_key()
                        .expect("L infinity test matrix")
                        .table_digest,
                    modulus_profile: inner.sis_modulus_profile(),
                    role: akita_types::sis::SisMatrixRole::Inner,
                    ring_dimension: u32::try_from(ring_dimension).expect("ring dimension"),
                    coeff_linf_bound,
                },
                inner.input_width(),
            )
            .expect("audited inner matrix");
        params = params
            .with_decomp(
                params.blocks().positions_per_block,
                params.blocks().live_ring_elements_per_claim,
                params.inner().digits.num_digits,
                params.outer().digits.num_digits,
                params.open().digits.num_digits,
            )
            .expect("layout rebuilt for audited inner rank");
        let outer = params.outer().matrix;
        params.own_group_mut().profile.outer.matrix =
            OuterCommitMatrixParams::try_new_with_min_rank(
                SisTableKey {
                    policy: outer.security_policy(),
                    table_digest: outer.sis_table_key().table_digest,
                    modulus_profile: outer.sis_modulus_profile(),
                    role: akita_types::sis::SisMatrixRole::Outer,
                    ring_dimension: u32::try_from(ring_dimension).expect("ring dimension"),
                    coeff_linf_bound: 3,
                },
                outer.input_width(),
            )
            .expect("audited outer matrix");
        params
    }

    fn setup_capacity_for(level_params: &CommittedGroupParams, n_prefix: usize) -> usize {
        let a_fields = level_params
            .inner()
            .matrix
            .output_rank()
            .checked_mul(level_params.inner().matrix.input_width())
            .and_then(|n| n.checked_mul(level_params.inner().matrix.ring_dimension()))
            .expect("A setup capacity");
        let b_fields = level_params
            .outer()
            .matrix
            .output_rank()
            .checked_mul(level_params.outer().matrix.input_width())
            .and_then(|n| n.checked_mul(level_params.outer().matrix.ring_dimension()))
            .expect("B setup capacity");
        let compression_source = level_params.outer().matrix.output_rank()
            * level_params.outer().matrix.ring_dimension();
        let compression_fields = CompressionChainPlan::for_complete_source(
            level_params.outer().matrix.sis_modulus_profile(),
            compression_source,
        )
        .expect("compression plan")
        .maps()
        .iter()
        .map(|map| map.input_width() * map.ring_dimension())
        .max()
        .expect("compression maps");
        n_prefix.max(a_fields).max(b_fields).max(compression_fields)
    }

    fn test_setup<const D: usize>(
        level_params: &CommittedGroupParams,
        n_prefix: usize,
    ) -> AkitaProverSetup<F> {
        AkitaProverSetup::<F>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity {
                num_field_elements: setup_capacity_for(level_params, n_prefix).max(1),
            },
        )
        .expect("setup")
    }

    fn portable_executor<'a>(
        setup: &'a AkitaProverSetup<F>,
        backend: &'a CpuBackend,
        prepared: &'a crate::compute::CpuPreparedSetup<F>,
    ) -> CommitmentExecutor<'a, F, PortableStatePolicy> {
        CommitmentExecutor::cpu(
            backend,
            prepared,
            &setup.expanded,
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            PortableStatePolicy,
        )
        .expect("setup-prefix executor")
    }

    fn resident_executor<'a>(
        setup: &'a AkitaProverSetup<F>,
        backend: &'a CpuBackend,
        prepared: &'a crate::compute::CpuPreparedSetup<F>,
    ) -> CommitmentExecutor<'a, F, ResidentStatePolicy> {
        CommitmentExecutor::cpu(
            backend,
            prepared,
            &setup.expanded,
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            ResidentStatePolicy,
        )
        .expect("resident setup-prefix executor")
    }

    fn executor_without_portable_export<'a>(
        setup: &'a AkitaProverSetup<F>,
        backend: &'a CpuBackend,
        prepared: &'a crate::compute::CpuPreparedSetup<F>,
    ) -> MissingExportExecutor<'a> {
        struct MissingExportRoute;

        let mut builder = CommitmentExecutorBuilder::new::<MissingExportRoute>(
            &setup.expanded,
            BackendKindId::of::<MissingExportRoute>("missing-export-route").unwrap(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            ResidentStatePolicy,
        );
        let inner = Arc::new(CpuInnerCommitOperation::new(backend, prepared));
        let outer = Arc::new(CpuOuterCommitOperation::new(
            backend,
            prepared,
            inner.as_ref(),
        ));
        let compression =
            Arc::new(CpuCompressionOperation::new(backend, prepared, &setup.expanded).unwrap());
        let instance = builder.issue_backend_instance();
        let inner_context = builder
            .operation_context(instance, "missing-export-inner", StageResources::none())
            .unwrap();
        let outer_context = builder
            .operation_context(instance, "missing-export-outer", StageResources::none())
            .unwrap();
        let compression_context = builder
            .operation_context(
                instance,
                "missing-export-compression",
                StageResources::none(),
            )
            .unwrap();
        builder
            .register_inner(
                inner.clone(),
                inner.owner().clone(),
                inner_context,
                StageDimensionCapabilities::cpu_role::<F>(akita_types::RingRole::Inner),
                None,
            )
            .unwrap();
        builder
            .register_outer(
                outer,
                inner.owner().clone(),
                outer_context,
                StageDimensionCapabilities::cpu_role::<F>(akita_types::RingRole::Outer),
            )
            .unwrap();
        builder
            .register_compression(
                compression.clone(),
                compression_context,
                CompressionOperationCapabilities::cpu::<F>(),
                None,
            )
            .unwrap();
        (builder.build().unwrap(), inner, compression)
    }

    #[test]
    fn setup_prefix_extraction_preserves_actual_tail() {
        let padded_ring_slots = 4usize;
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity {
                num_field_elements: padded_ring_slots * 64,
            },
        )
        .expect("setup");
        let fields = setup.expanded.shared_matrix().as_field_slice();
        assert_eq!(fields.len(), padded_ring_slots * 64);

        let ring_elems =
            extract_setup_prefix_ring_elems::<F, 64>(&setup.expanded, padded_ring_slots)
                .expect("extract setup prefix");

        assert_eq!(ring_elems.len(), padded_ring_slots);
        assert_eq!(ring_elems[0].coefficients(), &fields[..64]);
        assert_eq!(ring_elems[1].coefficients(), &fields[64..128]);
        assert_eq!(ring_elems[2].coefficients()[0], fields[128]);
        assert_eq!(ring_elems[2].coefficients(), &fields[128..192]);
        assert_eq!(ring_elems[3].coefficients(), &fields[192..256]);
    }

    #[test]
    fn commit_setup_prefix_requires_full_prefix_shared_setup() {
        let level_params = prefix_level_params(64);
        let witness_ring_slots = level_params
            .blocks()
            .live_blocks
            .checked_mul(level_params.blocks().positions_per_block)
            .expect("witness shape");
        let n_prefix = witness_ring_slots.checked_mul(64).expect("prefix length");
        let natural_len = n_prefix / 2 + 1;
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            8,
            1,
            SetupMatrixCapacity {
                num_field_elements: natural_len,
            },
        )
        .expect("setup");
        let available_field_len = setup.expanded.shared_matrix().as_field_slice().len();
        assert!(available_field_len >= natural_len);
        assert!(available_field_len < n_prefix);

        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).expect("prepared setup");
        let executor = portable_executor(&setup, &backend, &prepared);
        let prefix_params =
            setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
        let id = SetupPrefixSlotId {
            natural_len,
            commitment_profile: prefix_params.profile,
        };
        let error = commit_setup_prefix(&setup.expanded, &executor, &id)
            .expect_err("full-prefix source must be resident");
        assert!(
            error.to_string().contains("shared matrix capacity"),
            "unexpected error: {error}"
        );
    }

    fn assert_commit_setup_prefix_populates_singleton_slot<const D: usize>() {
        let level_params = prefix_level_params(D);
        let opening_batch = OpeningClaimsLayout::new(4, 1).expect("opening_batch");
        let witness_ring_slots = level_params
            .blocks()
            .live_blocks
            .checked_mul(level_params.blocks().positions_per_block)
            .expect("witness shape");
        let n_prefix = witness_ring_slots.checked_mul(D).expect("prefix length");
        let natural_len = active_setup_field_len(&level_params, &opening_batch)
            .expect("natural len")
            .min(n_prefix);
        let mut setup = test_setup::<D>(&level_params, n_prefix);
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).expect("prepared setup");
        let executor = portable_executor(&setup, &backend, &prepared);
        let prefix_params =
            setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
        let id = SetupPrefixSlotId {
            natural_len,
            commitment_profile: prefix_params.profile,
        };
        let slot = commit_setup_prefix(&setup.expanded, &executor, &id).expect("commit prefix");
        assert_eq!(slot.id.natural_len, natural_len);
        assert_eq!(slot.id.n_prefix().expect("full prefix len"), n_prefix);
        drop(executor);
        setup.prefix_slots.insert(slot).expect("insert");
        assert_eq!(setup.prefix_slots.len(), 1);
    }

    #[test]
    fn commit_setup_prefix_populates_d64_singleton_slot() {
        assert_commit_setup_prefix_populates_singleton_slot::<64>();
    }

    #[test]
    fn portable_and_resident_setup_prefix_slots_are_identical() {
        let level_params = prefix_level_params(64);
        let opening_batch = OpeningClaimsLayout::new(4, 1).expect("opening batch");
        let witness_ring_slots = level_params
            .blocks()
            .live_blocks
            .checked_mul(level_params.blocks().positions_per_block)
            .expect("witness shape");
        let n_prefix = witness_ring_slots.checked_mul(64).expect("prefix length");
        let natural_len = active_setup_field_len(&level_params, &opening_batch)
            .expect("natural length")
            .min(n_prefix);
        let prefix_params =
            setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
        let id = SetupPrefixSlotId {
            natural_len,
            commitment_profile: prefix_params.profile,
        };
        let setup = test_setup::<64>(&level_params, n_prefix);
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).expect("prepared setup");
        let portable = commit_setup_prefix(
            &setup.expanded,
            &portable_executor(&setup, &backend, &prepared),
            &id,
        )
        .expect("portable setup-prefix slot");
        let resident_executor = resident_executor(&setup, &backend, &prepared);
        let resident = commit_setup_prefix(&setup.expanded, &resident_executor, &id)
            .expect("resident setup-prefix slot");

        assert_eq!(resident, portable);
        assert_eq!(resident.verifier_slot(), portable.verifier_slot());
        let mut portable_hint_bytes = Vec::new();
        portable
            .hint
            .serialize_compressed(&mut portable_hint_bytes)
            .expect("serialize portable hint");
        let mut resident_hint_bytes = Vec::new();
        resident
            .hint
            .serialize_compressed(&mut resident_hint_bytes)
            .expect("serialize resident-exported hint");
        assert_eq!(resident_hint_bytes, portable_hint_bytes);
    }

    #[test]
    fn missing_portable_export_rejects_before_setup_prefix_arithmetic() {
        let level_params = prefix_level_params(64);
        let witness_ring_slots = level_params
            .blocks()
            .live_blocks
            .checked_mul(level_params.blocks().positions_per_block)
            .expect("witness shape");
        let n_prefix = witness_ring_slots.checked_mul(64).expect("prefix length");
        let prefix_params =
            setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
        let id = SetupPrefixSlotId {
            natural_len: n_prefix,
            commitment_profile: prefix_params.profile,
        };
        let setup = test_setup::<64>(&level_params, n_prefix);
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).expect("prepared setup");
        let (executor, _inner, _compression) =
            executor_without_portable_export(&setup, &backend, &prepared);

        let error = commit_setup_prefix(&setup.expanded, &executor, &id)
            .expect_err("missing export route must reject");
        assert!(error.to_string().contains("portable inner-image exporter"));
        assert!(setup.prefix_slots.is_empty());
    }

    #[test]
    fn setup_prefix_executor_matches_canonical_ntt_requirements() {
        let level_params = prefix_level_params(64);
        let witness_ring_slots = level_params
            .blocks()
            .live_blocks
            .checked_mul(level_params.blocks().positions_per_block)
            .expect("witness shape");
        let n_prefix = witness_ring_slots.checked_mul(64).expect("prefix length");
        let prefix_params =
            setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
        let id = SetupPrefixSlotId {
            natural_len: n_prefix,
            commitment_profile: prefix_params.profile,
        };
        let setup = test_setup::<64>(&level_params, n_prefix);
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).expect("prepared setup");
        let executor = resident_executor(&setup, &backend, &prepared);
        let source = DensePoly::from_field_evals(
            id.commitment_profile.group.num_vars(),
            setup.expanded.shared_matrix().as_field_slice()[..n_prefix].to_vec(),
        )
        .expect("setup-prefix source");
        let sources: [&dyn CommitmentSource<F>; 1] = [&source];
        let plan = CommitmentExecutionPlan::for_setup_prefix(&id).expect("setup-prefix plan");
        let metrics = executor
            .planned_request_ntt_cache_metrics(&plan, &sources)
            .expect("planned executor metrics");
        let metric_keys = metrics
            .iter()
            .flat_map(|metric| metric.keys.iter().copied())
            .collect::<Vec<_>>();
        let mut requirements = NttExecutionRequirements::default();
        requirements
            .add_setup_prefix_commitment(0, &id)
            .expect("canonical setup-prefix requirements");
        let mut expected_keys = Vec::new();
        for requirement in requirements.entries() {
            match expected_keys
                .iter()
                .position(|key: &akita_types::NttCacheKey| {
                    key.ring_d == requirement.key.ring_d && key.domain == requirement.key.domain
                }) {
                Some(index)
                    if requirement.key.num_ring_elements
                        > expected_keys[index].num_ring_elements =>
                {
                    expected_keys[index] = requirement.key;
                }
                Some(_) => {}
                None => expected_keys.push(requirement.key),
            }
        }
        assert_eq!(metric_keys, expected_keys);

        executor
            .prewarm_request(&plan, &sources)
            .expect("prewarm setup-prefix request");
        let planned_bytes = metrics
            .iter()
            .map(|metric| metric.cache_bytes)
            .sum::<usize>();
        assert_eq!(prepared.shared_ntt_cache_bytes(), planned_bytes);
        assert_eq!(executor.release_built_ntt_slots().unwrap(), planned_bytes);
        assert_eq!(prepared.shared_ntt_cache_bytes(), 0);
    }

    #[test]
    fn commit_setup_prefix_rejects_unaudited_outer_dimension() {
        let level_params = prefix_level_params(64);
        let witness_ring_slots = level_params
            .blocks()
            .live_blocks
            .checked_mul(level_params.blocks().positions_per_block)
            .expect("witness shape");
        let n_prefix = witness_ring_slots.checked_mul(64).expect("prefix length");
        let mut prefix_params =
            setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
        let outer = &prefix_params.profile.outer.matrix;
        prefix_params.profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
            outer.security_policy(),
            outer.sis_table_key().table_digest,
            outer.sis_modulus_profile(),
            outer.output_rank(),
            outer.input_width() * 2,
            outer.coeff_linf_bound(),
            32,
        );

        let setup = test_setup::<64>(&level_params, n_prefix);
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).expect("prepared setup");
        let executor = portable_executor(&setup, &backend, &prepared);
        let id = SetupPrefixSlotId {
            natural_len: n_prefix,
            commitment_profile: prefix_params.profile,
        };
        let error = commit_setup_prefix(&setup.expanded, &executor, &id)
            .expect_err("unaudited outer D32 must reject");
        assert!(
            error.to_string().contains("no audited SIS table key"),
            "unexpected error: {error}"
        );
    }
}
