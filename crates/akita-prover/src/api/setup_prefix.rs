//! Preprocessing helpers for setup-prefix commitment artifacts (slice 02B).

use crate::api::commitment::commit_outer_slices;
use crate::backend::{DensePoly, DenseView};
use crate::compute::compression::{execute_compression_chains, CompressionExecutionInput};
use crate::compute::{
    CommitInnerPlan, DigitRowsComputeBackend, OperationCtx, RootCommitKernel, RootCommitSource,
};
use crate::kernels::linear::decompose_commit_blocks_into;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, HalvingField, RandomSampling};
use akita_types::{
    dispatch_for_field, setup_prefix_slot_id, AkitaCommitmentHint, AkitaExpandedSetup,
    CompressionChainPlan, PrecommittedLevelParams, RingVec, SetupPrefixPublicCommitment,
    SetupPrefixSlot,
};

/// Commit one actual power-of-two flat prefix of the shared setup matrix.
///
/// The witness is the coefficient form of `S^flat[0..n_prefix]`. The caller
/// must supply `level_params` whose inner
/// witness shape satisfies `num_live_blocks * num_positions_per_block == n_prefix / D`.
///
/// # Errors
///
/// Returns an error if shapes overflow, the prefix does not fit the setup matrix,
/// or backend commitment fails.
#[allow(clippy::too_many_arguments)]
pub fn commit_setup_prefix<F, const D: usize, B>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup,
    level_params: &PrecommittedLevelParams,
    n_prefix: usize,
    natural_len: usize,
) -> Result<SetupPrefixSlot<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField,
    B: DigitRowsComputeBackend<F> + for<'a> RootCommitKernel<DenseView<'a, F, D>, F, D>,
{
    if natural_len == 0 || natural_len > n_prefix {
        return Err(AkitaError::InvalidSetup(
            "setup prefix natural length must be in 1..=n_prefix".to_string(),
        ));
    }
    if !n_prefix.is_multiple_of(D) || !n_prefix.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a power-of-two multiple of D".to_string(),
        ));
    }
    let full_prefix_ring_slots = n_prefix / D;
    let witness_ring_slots = level_params
        .layout
        .num_live_blocks
        .checked_mul(level_params.layout.num_positions_per_block)
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

    let ring_elems = extract_setup_prefix_ring_elems::<F, D>(expanded, full_prefix_ring_slots)?;
    let dense = DensePoly::from_ring_coeffs::<D>(ring_elems);
    let view = <DensePoly<F> as RootCommitSource<F, D>>::commit_view(&dense)?;
    let witnesses = backend.commit_inner_group(
        prepared,
        vec![view],
        CommitInnerPlan::from_profile(&level_params.layout),
    )?;
    let [witness] = witnesses.try_into().map_err(|witnesses: Vec<_>| {
        AkitaError::InvalidSetup(format!(
            "dense setup-prefix commit returned {} witnesses, expected one",
            witnesses.len()
        ))
    })?;
    let n_a = level_params.layout.inner_commit_matrix.output_rank();
    let recomposed_inner_rows = (0..level_params.layout.num_live_blocks)
        .map(|block| {
            witness
                .block_rows::<D>(block, n_a)
                .map(|rows| rows.to_vec())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let n_b = level_params.layout.outer_commit_matrix.output_rank();
    let d_b = level_params.layout.outer_commit_matrix.ring_dimension();
    let slice_geometry = akita_types::CommitmentSliceGeometry::try_new(
        level_params.layout.outer_slice_count,
        level_params.layout.num_live_blocks,
        1,
        n_a,
        level_params.layout.num_digits_outer,
        D,
        d_b,
    )?;
    let raw_commitment =
        dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Outer), F, d_b, |D_B| {
            let blocks = recomposed_inner_rows
                .iter()
                .map(Vec::as_slice)
                .collect::<Vec<_>>();
            let decomposed_inner_rows = decompose_commit_blocks_into::<F, D, D_B>(
                &blocks,
                level_params.layout.num_digits_outer,
                level_params.layout.log_basis_outer,
            )?;
            let u = commit_outer_slices::<F, _, D_B>(
                backend,
                prepared,
                n_b,
                std::iter::once(&decomposed_inner_rows),
                &slice_geometry,
                level_params.layout.log_basis_outer,
            )?;
            Ok::<_, AkitaError>(RingVec::from_ring_elems(&u))
        })?;
    let inner_coefficient_count = recomposed_inner_rows
        .iter()
        .map(Vec::len)
        .sum::<usize>()
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("setup-prefix inner rows overflow".into()))?;
    let mut inner_coefficients = Vec::with_capacity(inner_coefficient_count);
    for block in recomposed_inner_rows {
        for row in block {
            inner_coefficients.extend_from_slice(row.coefficients());
        }
    }
    let plan = CompressionChainPlan::for_complete_source(
        level_params
            .layout
            .outer_commit_matrix
            .sis_table_key()
            .modulus_profile,
        raw_commitment.coeff_len(),
    )?;
    let ctx = OperationCtx::new(backend, prepared, expanded)?;
    let (mut outputs, _) = execute_compression_chains(
        &ctx,
        vec![CompressionExecutionInput {
            id: (),
            plan,
            coefficients: raw_commitment.into_coeffs(),
        }],
    )?;
    let output = outputs.pop().ok_or(AkitaError::InvalidProof)?;
    let terminal_ring_dim = output
        .witness
        .plan()
        .maps()
        .last()
        .ok_or(AkitaError::InvalidProof)?
        .ring_dimension();
    let commitment_payload = RingVec::from_coeffs_with_ring_dim(
        output.terminal.coefficients().to_vec(),
        terminal_ring_dim,
    )?;
    let hint = AkitaCommitmentHint::singleton_with_outer_compression(
        RingVec::from_coeffs_with_ring_dim(inner_coefficients, D)?,
        &output.witness,
        &output.quotients,
    )?;
    let id = setup_prefix_slot_id(natural_len, level_params.clone());
    Ok(SetupPrefixSlot {
        id,
        commitment: SetupPrefixPublicCommitment {
            rows: vec![commitment_payload],
        },
        hint,
    })
}

fn extract_setup_prefix_ring_elems<F, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    full_prefix_ring_slots: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore,
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
    use crate::compute::{ComputeBackendSetup, CpuBackend};
    use crate::AkitaProverSetup;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128OffsetA7F7 as F;
    use akita_types::{
        active_setup_field_len, setup_prefix_precommitted_params, CommittedGroupParams,
        InnerCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams, SetupMatrixCapacity,
        SisModulusProfileId, SisTableKey,
    };

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
        .with_decomp(4, 3, 2, 2, 2)
        .expect("level params");
        let inner = params.inner_commit_matrix;
        params.inner_commit_matrix = InnerCommitMatrixParams::try_new_with_min_rank(
            SisTableKey {
                policy: inner.security_policy(),
                table_digest: inner
                    .sis_table_key()
                    .expect("L infinity test matrix")
                    .table_digest,
                modulus_profile: inner.sis_modulus_profile(),
                role: akita_types::sis::SisMatrixRole::Inner,
                ring_dimension: u32::try_from(ring_dimension).expect("ring dimension"),
                coeff_linf_bound: 131_071,
            },
            inner.input_width(),
        )
        .expect("audited inner matrix");
        let outer = params.outer_commit_matrix;
        params.outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
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
            .inner_commit_matrix
            .output_rank()
            .checked_mul(level_params.inner_commit_matrix.input_width())
            .and_then(|n| n.checked_mul(level_params.inner_commit_matrix.ring_dimension()))
            .expect("A setup capacity");
        let b_fields = level_params
            .outer_commit_matrix
            .output_rank()
            .checked_mul(level_params.outer_commit_matrix.input_width())
            .and_then(|n| n.checked_mul(level_params.outer_commit_matrix.ring_dimension()))
            .expect("B setup capacity");
        let compression_source = level_params.outer_commit_matrix.output_rank()
            * level_params.outer_commit_matrix.ring_dimension();
        let compression_fields = CompressionChainPlan::for_complete_source(
            level_params.outer_commit_matrix.sis_modulus_profile(),
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
        let level_params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            3,
            2,
            3,
            2,
            SparseChallengeConfig::pm1_only(3),
        )
        .with_decomp(16, 256, 2, 2, 2)
        .expect("level params");
        let witness_ring_slots = level_params
            .num_live_blocks
            .checked_mul(level_params.num_positions_per_block)
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
        let prefix_params =
            setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
        let error = commit_setup_prefix::<F, 64, _>(
            &setup.expanded,
            &backend,
            &prepared,
            &prefix_params,
            n_prefix,
            natural_len,
        )
        .expect_err("full-prefix source must be resident");
        assert!(error.to_string().contains("shared matrix capacity"));
    }

    fn assert_commit_setup_prefix_populates_singleton_slot<const D: usize>() {
        let level_params = prefix_level_params(D);
        let opening_batch = OpeningClaimsLayout::new(4, 1).expect("opening_batch");
        let witness_ring_slots = level_params
            .num_live_blocks
            .checked_mul(level_params.num_positions_per_block)
            .expect("witness shape");
        let n_prefix = witness_ring_slots.checked_mul(D).expect("prefix length");
        let natural_len = active_setup_field_len(&level_params, &opening_batch)
            .expect("natural len")
            .min(n_prefix);
        let mut setup = test_setup::<D>(&level_params, n_prefix);
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).expect("prepared setup");
        let prefix_params =
            setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
        let slot = commit_setup_prefix::<F, D, _>(
            &setup.expanded,
            &backend,
            &prepared,
            &prefix_params,
            n_prefix,
            natural_len,
        )
        .expect("commit prefix");
        assert_eq!(slot.id.natural_len, natural_len);
        assert_eq!(slot.id.n_prefix().expect("full prefix len"), n_prefix);
        setup.prefix_slots.insert(slot).expect("insert");
        assert_eq!(setup.prefix_slots.len(), 1);
    }

    #[test]
    fn commit_setup_prefix_populates_d64_singleton_slot() {
        assert_commit_setup_prefix_populates_singleton_slot::<64>();
    }

    #[test]
    fn commit_setup_prefix_rejects_unsupported_outer_dimension() {
        let level_params = prefix_level_params(64);
        let witness_ring_slots = level_params
            .num_live_blocks
            .checked_mul(level_params.num_positions_per_block)
            .expect("witness shape");
        let n_prefix = witness_ring_slots.checked_mul(64).expect("prefix length");
        let mut prefix_params =
            setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
        let outer = &prefix_params.layout.outer_commit_matrix;
        prefix_params.layout.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
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
        let error = commit_setup_prefix::<F, 64, _>(
            &setup.expanded,
            &backend,
            &prepared,
            &prefix_params,
            n_prefix,
            n_prefix,
        )
        .expect_err("ordinary outer D32 must reject");
        assert!(error.to_string().contains("unsupported ring dimension 32"));
    }
}
