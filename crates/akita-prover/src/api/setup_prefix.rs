//! Preprocessing helpers for setup-prefix commitment artifacts (slice 02B).

use crate::api::commitment::commit_outer_slices;
use crate::backend::{DensePoly, DenseView};
use crate::compute::compression::{execute_compression_chains, CompressionExecutionInput};
use crate::compute::{
    CommitInnerPlan, DigitRowsComputeBackend, OperationCtx, RootCommitKernel, RootCommitSource,
};
use crate::kernels::linear::decompose_commit_blocks_into;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_types::{
    dispatch_for_field, AkitaCommitmentHint, AkitaExpandedSetup, CompressionChainPlan,
    GroupCommitPhaseParams, RingVec, SetupPrefixPublicCommitment, SetupPrefixSlot,
    SetupPrefixSlotId,
};
use jolt_field::{CanonicalEncoding, Field};

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
    commitment_profile: &GroupCommitPhaseParams,
    n_prefix: usize,
    natural_len: usize,
) -> Result<SetupPrefixSlot<F>, AkitaError>
where
    F: Field + CanonicalEncoding,
    B: DigitRowsComputeBackend<F> + for<'a> RootCommitKernel<DenseView<'a, F, D>, F, D>,
{
    commitment_profile.validate(
        commitment_profile
            .inner
            .matrix
            .sis_modulus_profile()
            .field_bits(),
    )?;
    commitment_profile.validate_setup_prefix_geometry(natural_len)?;
    let committed_n_prefix = 1usize
        .checked_shl(commitment_profile.group.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("setup-prefix domain overflow".into()))?;
    if committed_n_prefix != n_prefix || !n_prefix.is_multiple_of(D) {
        return Err(AkitaError::InvalidSetup(
            "requested setup prefix does not match the full committed domain".to_string(),
        ));
    }
    let full_prefix_ring_slots = n_prefix / D;
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

    let ring_elems = extract_setup_prefix_ring_elems::<F, D>(expanded, full_prefix_ring_slots)?;
    let dense = DensePoly::from_ring_coeffs::<D>(ring_elems);
    let view = <DensePoly<F> as RootCommitSource<F, D>>::commit_view(&dense)?;
    let witnesses = backend.commit_inner_group(
        prepared,
        vec![view],
        CommitInnerPlan::from_profile(commitment_profile),
    )?;
    let [witness] = witnesses.try_into().map_err(|witnesses: Vec<_>| {
        AkitaError::InvalidSetup(format!(
            "dense setup-prefix commit returned {} witnesses, expected one",
            witnesses.len()
        ))
    })?;
    let n_a = commitment_profile.inner.matrix.output_rank();
    let recomposed_inner_rows = (0..commitment_profile.blocks.live_blocks)
        .map(|block| {
            witness
                .block_rows::<D>(block, n_a)
                .map(|rows| rows.to_vec())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let n_b = commitment_profile.outer.matrix.output_rank();
    let d_b = commitment_profile.outer.matrix.ring_dimension();
    let slice_geometry = akita_types::CommitmentSliceGeometry::try_new(
        commitment_profile.outer_slice_count,
        commitment_profile.blocks.live_blocks,
        1,
        n_a,
        commitment_profile.outer.digits.num_digits,
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
                commitment_profile.outer.digits.num_digits,
                commitment_profile.outer.digits.log_basis,
            )?;
            let u = commit_outer_slices::<F, _, D_B>(
                backend,
                prepared,
                n_b,
                std::iter::once(&decomposed_inner_rows),
                &slice_geometry,
                commitment_profile.outer.digits.log_basis,
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
        commitment_profile
            .outer
            .matrix
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
            relation_mode: akita_types::RingRelationMode::QuotientLift,
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
    let commitment_payload =
        RingVec::from_coeffs_with_ring_dim(output.terminal.into_coefficients(), terminal_ring_dim)?;
    let quotients = output.relation.into_quotient_lift()?;
    let hint = AkitaCommitmentHint::singleton_with_outer_compression(
        RingVec::from_coeffs_with_ring_dim(inner_coefficients, D)?,
        &output.witness,
        &quotients,
    )?;
    let id = SetupPrefixSlotId {
        natural_len,
        commitment_profile: *commitment_profile,
    };
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
    use crate::compute::{ComputeBackendSetup, CpuBackend};
    use crate::AkitaProverSetup;
    use akita_challenges::SparseChallengeConfig;
    use akita_types::{
        active_setup_field_len, setup_prefix_precommitted_params, CommittedGroupParams,
        InnerCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams, SetupMatrixCapacity,
        SisModulusProfileId, SisTableKey,
    };
    use jolt_field::Prime128OffsetA7F7 as F;

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
        let prefix_params =
            setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
        let error = commit_setup_prefix::<F, 64, _>(
            &setup.expanded,
            &backend,
            &prepared,
            &prefix_params.profile,
            n_prefix,
            natural_len,
        )
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
        let prefix_params =
            setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
        let slot = commit_setup_prefix::<F, D, _>(
            &setup.expanded,
            &backend,
            &prepared,
            &prefix_params.profile,
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
        let error = commit_setup_prefix::<F, 64, _>(
            &setup.expanded,
            &backend,
            &prepared,
            &prefix_params.profile,
            n_prefix,
            n_prefix,
        )
        .expect_err("unaudited outer D32 must reject");
        assert!(
            error.to_string().contains("no audited SIS table key"),
            "unexpected error: {error}"
        );
    }
}
