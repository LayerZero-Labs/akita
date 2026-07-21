//! Deterministic terminal checks over the revealed segment-typed witness.

use akita_algebra::CyclotomicRing;
use akita_challenges::{Challenges, SparseChallenge};
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
};
use akita_types::{
    decode_terminal_z_golomb_payload_with_cap, dispatch_for_field,
    recover_ring_subfield_inner_product, AkitaVerifierSetup, FpExtEncoding, LevelParams,
    PreparedOpeningPoint, RelationMatrixRowLayout, RingRelationInstance, SegmentTypedWitness,
};

fn sparse_challenge_ring<F, const D: usize>(
    challenge: &SparseChallenge,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
{
    challenge.validate::<D>()?;
    let mut coeffs = [F::zero(); D];
    for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
        let slot = coeffs
            .get_mut(position as usize)
            .ok_or(AkitaError::InvalidProof)?;
        *slot += F::from_i64(i64::from(coefficient));
    }
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

fn challenge_rings<F, const D: usize>(
    challenges: &Challenges,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
{
    match challenges {
        Challenges::Sparse { challenges, .. } => challenges
            .iter()
            .map(sparse_challenge_ring::<F, D>)
            .collect(),
        Challenges::Tensor { factored } => {
            factored.validate::<D>()?;
            (0..factored.total_blocks()?)
                .map(|index| {
                    let (_, _, high, low) = factored.factors_for_logical_block(index)?;
                    Ok(sparse_challenge_ring::<F, D>(high)? * sparse_challenge_ring::<F, D>(low)?)
                })
                .collect()
        }
    }
}

fn ring_dot<F, const D: usize>(
    row: &[CyclotomicRing<F, D>],
    input: &[CyclotomicRing<F, D>],
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore,
{
    if row.len() != input.len() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(row
        .iter()
        .zip(input)
        .fold(CyclotomicRing::zero(), |sum, (lhs, rhs)| {
            sum + (*lhs * *rhs)
        }))
}

#[inline]
fn centered_ring<F, const D: usize>(coeffs: &[i16; D]) -> CyclotomicRing<F, D>
where
    F: FieldCore + FromPrimitiveInt,
{
    CyclotomicRing::from_coefficients(std::array::from_fn(|index| {
        F::from_i64(i64::from(coeffs[index]))
    }))
}

fn narrow_terminal_z_i16(values: Vec<i64>) -> Result<Vec<i16>, AkitaError> {
    values
        .into_iter()
        .map(|value| i16::try_from(value).map_err(|_| AkitaError::InvalidProof))
        .collect()
}

#[tracing::instrument(skip_all, name = "terminal_direct_a_rows")]
fn check_a_rows<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    t: &[CyclotomicRing<F, D>],
    z: &[[i16; D]],
    challenges: &[CyclotomicRing<F, D>],
    n_a: usize,
    n_a_cols: usize,
    prepared_prefix_len: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
{
    if t.len()
        != challenges
            .len()
            .checked_mul(n_a)
            .ok_or(AkitaError::InvalidProof)?
        || z.len() != n_a_cols
    {
        return Err(AkitaError::InvalidProof);
    }
    let (rhs, lhs) = cfg_join!(
        || super::terminal_ntt::centered_rows(setup, n_a, z, prepared_prefix_len),
        || {
            let _span = tracing::info_span!(
                "terminal_direct_a_lhs",
                rows = n_a,
                challenges = challenges.len()
            )
            .entered();
            (0..n_a)
                .map(|row_index| {
                    challenges.iter().zip(t.chunks_exact(n_a)).try_fold(
                        CyclotomicRing::zero(),
                        |sum, (challenge, rows)| {
                            let row = rows.get(row_index).ok_or(AkitaError::InvalidProof)?;
                            Ok::<_, AkitaError>(sum + (*challenge * *row))
                        },
                    )
                })
                .collect::<Result<Vec<_>, AkitaError>>()
        }
    );
    let rhs = rhs?;
    let lhs = lhs?;
    let _span = tracing::info_span!("terminal_direct_a_compare", rows = n_a).entered();
    for (actual, expected) in lhs.iter().zip(&rhs) {
        if actual != expected {
            return Err(AkitaError::InvalidProof);
        }
    }
    Ok(())
}

/// Check reduced consistency and A rows for a quotient-free terminal witness.
#[tracing::instrument(skip_all, name = "terminal_direct_ring_relations")]
pub(super) fn verify_terminal_ring_relations<F>(
    setup: &AkitaVerifierSetup<F>,
    relation: &RingRelationInstance<F>,
    lp: &LevelParams,
    final_witness: &SegmentTypedWitness<F>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HalvingField,
{
    let witness = final_witness;
    if relation.relation_matrix_row_layout() != RelationMatrixRowLayout::WithoutCommitmentBlocks
        || witness.layout.ring_dimension != relation.role_dims().d_a()
        || witness.layout.groups.len() != relation.opening_batch().num_groups()
    {
        return Err(AkitaError::InvalidProof);
    }
    let order = relation.opening_batch().root_group_order()?;
    let mut e_offset = 0usize;
    let mut t_offset = 0usize;
    dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        F,
        relation.role_dims().d_a(),
        |D_A| {
            let e_rings = witness.e_fields.as_ring_slice::<D_A>()?;
            let t_rings = witness.t_fields.as_ring_slice::<D_A>()?;
            let mut consistency_lhs = CyclotomicRing::<F, D_A>::zero();
            let mut consistency_rhs = CyclotomicRing::<F, D_A>::zero();
            for (layout_index, &group_index) in order.iter().enumerate() {
                let params = lp.group_params(relation.opening_batch(), group_index)?;
                let group_layout = witness
                    .layout
                    .groups
                    .get(layout_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let num_polynomials = relation
                    .opening_batch()
                    .group_layout(group_index)?
                    .num_polynomials();
                let a_range = lp.a_row_range(
                    relation.opening_batch(),
                    group_index,
                    RelationMatrixRowLayout::WithoutCommitmentBlocks,
                )?;
                if a_range.len() != params.a_rows_len() {
                    return Err(AkitaError::InvalidSetup(
                        "terminal direct row ranges do not match group matrix heights".to_string(),
                    ));
                }
                let num_blocks = num_polynomials
                    .checked_mul(params.num_live_blocks())
                    .ok_or(AkitaError::InvalidProof)?;
                let e_end = e_offset
                    .checked_add(group_layout.e_field_elems)
                    .ok_or(AkitaError::InvalidProof)?;
                let t_end = t_offset
                    .checked_add(group_layout.t_field_elems)
                    .ok_or(AkitaError::InvalidProof)?;
                let (e, t, z_values) = {
                    let _span = tracing::info_span!(
                        "terminal_direct_decode",
                        group_index,
                        e_field_elems = group_layout.e_field_elems,
                        t_field_elems = group_layout.t_field_elems,
                        z_coords = group_layout.z_coords
                    )
                    .entered();
                    if !e_offset.is_multiple_of(D_A)
                        || !e_end.is_multiple_of(D_A)
                        || !t_offset.is_multiple_of(D_A)
                        || !t_end.is_multiple_of(D_A)
                    {
                        return Err(AkitaError::InvalidProof);
                    }
                    let e = e_rings
                        .get(e_offset / D_A..e_end / D_A)
                        .ok_or(AkitaError::InvalidProof)?;
                    let t = t_rings
                        .get(t_offset / D_A..t_end / D_A)
                        .ok_or(AkitaError::InvalidProof)?;
                    let cap = lp.fold_witness_linf_cap_for_params(
                        params,
                        num_polynomials,
                        F::modulus_bits(),
                    )?;
                    let z_values = decode_terminal_z_golomb_payload_with_cap(
                        witness
                            .z_payloads
                            .get(layout_index)
                            .ok_or(AkitaError::InvalidProof)?,
                        group_layout.z_coords,
                        cap,
                        Some(group_layout.z_payload_bytes),
                    )?;
                    let z_values = narrow_terminal_z_i16(z_values)?;
                    (e, t, z_values)
                };
                let z_centered = {
                    let _span = tracing::info_span!(
                        "terminal_direct_decode_z_rings",
                        group_index,
                        z_coords = z_values.len()
                    )
                    .entered();
                    if !z_values.len().is_multiple_of(D_A) {
                        return Err(AkitaError::InvalidProof);
                    }
                    let (rings, remainder) = z_values.as_chunks::<D_A>();
                    if !remainder.is_empty() {
                        return Err(AkitaError::InvalidProof);
                    }
                    rings
                };
                let challenges = {
                    let _span =
                        tracing::info_span!("terminal_direct_challenges", group_index, num_blocks)
                            .entered();
                    challenge_rings::<F, D_A>(
                        relation
                            .group_challenges()
                            .get(group_index)
                            .ok_or(AkitaError::InvalidProof)?,
                    )?
                };
                let expected_t_len = num_blocks
                    .checked_mul(params.a_rows_len())
                    .ok_or(AkitaError::InvalidProof)?;
                if e.len() != num_blocks || t.len() != expected_t_len {
                    return Err(AkitaError::InvalidProof);
                }
                let n_a = params.a_rows_len();
                let n_a_cols = params.a_col_len();
                let a_prefix_len = n_a.checked_mul(n_a_cols).ok_or(AkitaError::InvalidProof)?;
                let num_positions = params.num_positions_per_block();
                let num_digits_inner = params.num_digits_inner();
                let log_basis_inner = params.log_basis_inner();
                let multiplier = relation.group_ring_multiplier_point(group_index)?;
                let position_rings = multiplier.position_rings_trusted::<D_A>()?;
                let (consistency, a_rows) = cfg_join!(
                    || {
                        let _span = tracing::info_span!(
                            "terminal_direct_consistency",
                            group_index,
                            num_blocks,
                            num_positions
                        )
                        .entered();
                        let folded = {
                            let _span = tracing::info_span!(
                                "terminal_direct_consistency_fold_e",
                                blocks = challenges.len()
                            )
                            .entered();
                            ring_dot(&challenges, e)?
                        };
                        let reduced = {
                            let _span = tracing::info_span!(
                                "terminal_direct_consistency_reduce_z",
                                positions = num_positions,
                                digits = num_digits_inner
                            )
                            .entered();
                            let gadget = akita_types::gadget_row_scalars::<F>(
                                num_digits_inner,
                                log_basis_inner,
                            );
                            let mut reduced = CyclotomicRing::zero();
                            for position in 0..num_positions {
                                let start = position
                                    .checked_mul(num_digits_inner)
                                    .ok_or(AkitaError::InvalidProof)?;
                                let mut z_value = CyclotomicRing::zero();
                                for digit in 0..num_digits_inner {
                                    let index =
                                        start.checked_add(digit).ok_or(AkitaError::InvalidProof)?;
                                    z_value += centered_ring::<F, D_A>(
                                        z_centered.get(index).ok_or(AkitaError::InvalidProof)?,
                                    )
                                    .scale(gadget.get(digit).ok_or(AkitaError::InvalidProof)?);
                                }
                                if let Some(scale) = multiplier.position_constant_coeff(position) {
                                    reduced += z_value.scale(&scale);
                                } else {
                                    reduced += *position_rings
                                        .ok_or(AkitaError::InvalidProof)?
                                        .get(position)
                                        .ok_or(AkitaError::InvalidProof)?
                                        * z_value;
                                }
                            }
                            reduced
                        };
                        Ok::<_, AkitaError>((folded, reduced))
                    },
                    || {
                        check_a_rows::<F, D_A>(
                            setup,
                            t,
                            z_centered,
                            &challenges,
                            n_a,
                            n_a_cols,
                            a_prefix_len,
                        )
                    }
                );
                let (folded, reduced) = consistency?;
                a_rows?;
                consistency_lhs += folded;
                consistency_rhs += reduced;
                e_offset = e_end;
                t_offset = t_end;
            }
            if consistency_lhs != consistency_rhs {
                return Err(AkitaError::InvalidProof);
            }
            Ok::<(), AkitaError>(())
        }
    )?;
    if e_offset != witness.e_fields.coeff_len() || t_offset != witness.t_fields.coeff_len() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

/// Check the public opening directly against the revealed folded `e` segment.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "terminal_direct_trace")]
pub(super) fn verify_terminal_trace<F, E>(
    relation: &RingRelationInstance<F>,
    lp: &LevelParams,
    final_witness: &SegmentTypedWitness<F>,
    prepared_points: &[PreparedOpeningPoint<F, E>],
    row_coefficients: &[E],
    claim_scales: Option<&[E]>,
    global_scale: E,
    target: E,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let witness = final_witness;
    if prepared_points.len() != relation.opening_batch().num_groups()
        || row_coefficients.len() != relation.opening_batch().num_total_polynomials()
        || claim_scales.is_some_and(|scales| scales.len() != row_coefficients.len())
    {
        return Err(AkitaError::InvalidProof);
    }
    let order = relation.opening_batch().root_group_order()?;
    let mut e_offset = 0usize;
    let mut actual = E::zero();
    dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        F,
        relation.role_dims().d_a(),
        |D| {
            let e_rings = witness.e_fields.as_ring_slice::<D>()?;
            for (layout_index, &group_index) in order.iter().enumerate() {
                let params = lp.group_params(relation.opening_batch(), group_index)?;
                let group_layout = witness
                    .layout
                    .groups
                    .get(layout_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let end = e_offset
                    .checked_add(group_layout.e_field_elems)
                    .ok_or(AkitaError::InvalidProof)?;
                if !e_offset.is_multiple_of(D) || !end.is_multiple_of(D) {
                    return Err(AkitaError::InvalidProof);
                }
                let e = e_rings
                    .get(e_offset / D..end / D)
                    .ok_or(AkitaError::InvalidProof)?;
                let claim_range = relation
                    .opening_batch()
                    .root_group_claim_range(group_index)?;
                let multiplier = relation.group_ring_multiplier_point(group_index)?;
                let packed_inner = prepared_points
                    .get(group_index)
                    .ok_or(AkitaError::InvalidProof)?
                    .packed_inner_trusted::<D>()?;
                for (local_claim, claim_index) in claim_range.enumerate() {
                    let start = local_claim
                        .checked_mul(params.num_live_blocks())
                        .ok_or(AkitaError::InvalidProof)?;
                    let end = start
                        .checked_add(params.num_live_blocks())
                        .ok_or(AkitaError::InvalidProof)?;
                    let claim_e = e.get(start..end).ok_or(AkitaError::InvalidProof)?;
                    let mut outer_eval = CyclotomicRing::zero();
                    for (block, value) in claim_e.iter().enumerate() {
                        if let Some(scale) = multiplier.fold_constant_coeff(block) {
                            outer_eval += value.scale(&scale);
                        } else {
                            outer_eval += *multiplier
                                .fold_rings_trusted::<D>()?
                                .ok_or(AkitaError::InvalidProof)?
                                .get(block)
                                .ok_or(AkitaError::InvalidProof)?
                                * *value;
                        }
                    }
                    let opening =
                        recover_ring_subfield_inner_product::<F, E, D>(&outer_eval, packed_inner)?;
                    let scale = claim_scales
                        .and_then(|scales| scales.get(claim_index))
                        .copied()
                        .unwrap_or(global_scale);
                    actual += *row_coefficients
                        .get(claim_index)
                        .ok_or(AkitaError::InvalidProof)?
                        * scale
                        * opening;
                }
                e_offset = end;
            }
            Ok::<(), AkitaError>(())
        }
    )?;
    if e_offset != witness.e_fields.coeff_len() || actual != target {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::{
        build_segment_typed_witness_from_groups, AkitaExpandedSetup, AkitaSetupSeed,
        CommitmentRingDims, FlatMatrix, OpeningClaimsLayout, PolynomialGroupLayout,
        PrecommittedGroupParams, PrecommittedLevelParams, RingMultiplierOpeningPoint,
        RingOpeningPoint, RingVec, SegmentTypedWitnessGroupParts, SetupPrefixVerifierRegistry,
        SisModulusProfileId,
    };

    type F = Prime128OffsetA7F7;

    #[derive(Clone, Copy)]
    enum TerminalRowRole {
        Consistency,
        A,
    }

    #[derive(Clone, Copy)]
    struct TerminalGroupFixture<const D: usize> {
        challenge: CyclotomicRing<F, D>,
        e: CyclotomicRing<F, D>,
        t: CyclotomicRing<F, D>,
        z: CyclotomicRing<F, D>,
        z_centered: [i32; D],
    }

    fn cyclic_product<F: FieldCore, const D: usize>(
        lhs: &CyclotomicRing<F, D>,
        rhs: &CyclotomicRing<F, D>,
    ) -> CyclotomicRing<F, D> {
        let mut coefficients = [F::zero(); D];
        for (lhs_index, &lhs_coefficient) in lhs.coefficients().iter().enumerate() {
            for (rhs_index, &rhs_coefficient) in rhs.coefficients().iter().enumerate() {
                coefficients[(lhs_index + rhs_index) % D] += lhs_coefficient * rhs_coefficient;
            }
        }
        CyclotomicRing::from_coefficients(coefficients)
    }

    fn monomial<const D: usize>(index: usize, coefficient: i64) -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|slot| {
            if slot == index {
                F::from_i64(coefficient)
            } else {
                F::zero()
            }
        }))
    }

    fn group_fixture<const D: usize>(challenge_sign: i8, scale: i32) -> TerminalGroupFixture<D> {
        let challenge = monomial::<D>(1, i64::from(challenge_sign));
        let e = monomial::<D>(D - 1, i64::from(scale));
        let t = e;
        let z_constant = -i32::from(challenge_sign) * scale;
        let z_centered = std::array::from_fn(|index| if index == 0 { z_constant } else { 0 });
        let z = monomial::<D>(0, i64::from(z_constant));
        TerminalGroupFixture {
            challenge,
            e,
            t,
            z,
            z_centered,
        }
    }

    fn row_images<const D: usize>(
        role: TerminalRowRole,
        groups: &[TerminalGroupFixture<D>],
    ) -> (
        CyclotomicRing<F, D>,
        CyclotomicRing<F, D>,
        CyclotomicRing<F, D>,
        CyclotomicRing<F, D>,
    ) {
        groups.iter().fold(
            (
                CyclotomicRing::zero(),
                CyclotomicRing::zero(),
                CyclotomicRing::zero(),
                CyclotomicRing::zero(),
            ),
            |(actual_cyclic, actual_reduced, expected_cyclic, expected_reduced), group| {
                let (actual_lhs, expected_lhs) = match role {
                    TerminalRowRole::Consistency => (group.e, group.z),
                    TerminalRowRole::A => (group.t, group.z),
                };
                (
                    actual_cyclic + cyclic_product(&group.challenge, &actual_lhs),
                    actual_reduced + group.challenge * actual_lhs,
                    expected_cyclic + cyclic_product(&CyclotomicRing::one(), &expected_lhs),
                    expected_reduced + expected_lhs,
                )
            },
        )
    }

    fn legacy_residual<F, const D: usize>(
        actual_cyclic: CyclotomicRing<F, D>,
        actual_reduced: CyclotomicRing<F, D>,
        expected_cyclic: CyclotomicRing<F, D>,
        expected_reduced: CyclotomicRing<F, D>,
    ) -> CyclotomicRing<F, D>
    where
        F: FieldCore + HalvingField,
    {
        let actual_quotient = CyclotomicRing::from_coefficients(std::array::from_fn(|index| {
            (actual_cyclic.coefficients()[index] - actual_reduced.coefficients()[index]).half()
        }));
        let expected_quotient = CyclotomicRing::from_coefficients(std::array::from_fn(|index| {
            (expected_cyclic.coefficients()[index] - expected_reduced.coefficients()[index]).half()
        }));
        let quotient_delta = actual_quotient - expected_quotient;
        actual_cyclic - expected_cyclic - quotient_delta - quotient_delta
    }

    fn assert_direct_matches_legacy<const D: usize>(
        role: TerminalRowRole,
        groups: &[TerminalGroupFixture<D>],
    ) {
        let (actual_cyclic, actual_reduced, expected_cyclic, expected_reduced) =
            row_images::<D>(role, groups);

        let direct_valid = actual_reduced - expected_reduced;
        let legacy_valid = legacy_residual(
            actual_cyclic,
            actual_reduced,
            expected_cyclic,
            expected_reduced,
        );
        assert_eq!(legacy_valid, direct_valid);
        assert_eq!(direct_valid, CyclotomicRing::zero());

        let mut tampered_coefficients = *expected_reduced.coefficients();
        tampered_coefficients[D / 2] += F::one();
        let tampered_reduced = CyclotomicRing::from_coefficients(tampered_coefficients);
        let tampered_cyclic = cyclic_product(&tampered_reduced, &CyclotomicRing::one());
        let direct_tampered = actual_reduced - tampered_reduced;
        let legacy_tampered = legacy_residual(
            actual_cyclic,
            actual_reduced,
            tampered_cyclic,
            tampered_reduced,
        );
        assert_eq!(legacy_tampered, direct_tampered);
        assert_ne!(direct_tampered, CyclotomicRing::zero());
    }

    fn sparse_challenges(sign: i8) -> Challenges {
        Challenges::from_sparse(
            vec![SparseChallenge {
                positions: vec![1],
                coeffs: vec![sign],
            }],
            1,
            1,
        )
        .expect("one-claim challenge fixture")
    }

    fn grouped_terminal_fixture<const D: usize>() -> (
        AkitaVerifierSetup<F>,
        RingRelationInstance<F>,
        LevelParams,
        SegmentTypedWitness<F>,
        [TerminalGroupFixture<D>; 2],
    ) {
        let dims = CommitmentRingDims {
            inner: D,
            outer: 32,
            opening: 16,
        };
        let base_params = LevelParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            1,
            1,
            1,
            1,
            SparseChallengeConfig::production_for_ring_dim(D)
                .expect("supported A-role challenge dimension"),
        )
        .with_decomp(1, 1, 1, 1, 1)
        .expect("terminal fixture layout")
        .with_role_dims(dims)
        .expect("nested role dimensions");
        let precommitted_layout = PolynomialGroupLayout::new(0, 1);
        let mut precommitted_params =
            PrecommittedGroupParams::from_params(precommitted_layout, &base_params);
        precommitted_params.n_a = base_params.a_key.row_len();
        precommitted_params.n_b = base_params.b_key.row_len();
        precommitted_params.a_coeff_linf_bound = 1;
        precommitted_params.b_coeff_linf_bound = 1;
        let precommitted_a_width = precommitted_params
            .num_positions_per_block
            .checked_mul(base_params.num_digits_inner)
            .expect("precommitted A width");
        let precommitted_b_width = precommitted_params
            .n_a
            .checked_mul(base_params.num_digits_outer)
            .and_then(|width| width.checked_mul(precommitted_params.num_live_blocks))
            .and_then(|width| width.checked_mul(precommitted_layout.num_polynomials()))
            .expect("precommitted B width");
        let precommitted_a_key = akita_types::sis::AjtaiKeyParams::new_unchecked(
            akita_types::DEFAULT_SIS_SECURITY_POLICY,
            akita_types::SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            akita_types::SisMatrixRole::A,
            precommitted_params.n_a,
            precommitted_a_width,
            precommitted_params.a_coeff_linf_bound,
            D,
        );
        let precommitted_b_key = akita_types::sis::AjtaiKeyParams::new_unchecked(
            akita_types::DEFAULT_SIS_SECURITY_POLICY,
            akita_types::SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            akita_types::SisMatrixRole::B,
            precommitted_params.n_b,
            precommitted_b_width,
            precommitted_params.b_coeff_linf_bound,
            D,
        );
        let precommitted = PrecommittedLevelParams {
            layout: precommitted_params,
            a_key: precommitted_a_key,
            b_key: precommitted_b_key,
            log_basis_open: base_params.log_basis_open,
            num_digits_inner: base_params.num_digits_inner,
            num_digits_outer: base_params.num_digits_outer,
            num_digits_open: base_params.num_digits_open,
            num_digits_fold_one: base_params.num_digits_fold_one,
        };
        let mut params = base_params;
        params.precommitted_groups.push(precommitted);
        let opening_batch = OpeningClaimsLayout::from_root_groups(
            &[precommitted_layout],
            PolynomialGroupLayout::new(0, 1),
        )
        .expect("two-group opening layout");

        // Original group order is precommitted then final. Terminal witness
        // order is deliberately final then precommitted.
        let groups = [group_fixture::<D>(1, 1), group_fixture::<D>(-1, 2)];
        let final_e = RingVec::from_ring_elems(&[groups[1].e]);
        let final_t = [RingVec::from_ring_elems(&[groups[1].t])];
        let precommitted_e = RingVec::from_ring_elems(&[groups[0].e]);
        let precommitted_t = [RingVec::from_ring_elems(&[groups[0].t])];
        let witness = build_segment_typed_witness_from_groups(
            D,
            &[
                SegmentTypedWitnessGroupParts {
                    params: &params,
                    num_w_vectors: 1,
                    num_t_vectors: 1,
                    num_z_segments: 1,
                    e_folded: &final_e,
                    recomposed_inner_rows: &final_t,
                    z_folded_centered_flat: &groups[1].z_centered,
                },
                SegmentTypedWitnessGroupParts {
                    params: &params.precommitted_groups[0],
                    num_w_vectors: 1,
                    num_t_vectors: 1,
                    num_z_segments: 1,
                    e_folded: &precommitted_e,
                    recomposed_inner_rows: &precommitted_t,
                    z_folded_centered_flat: &groups[0].z_centered,
                },
            ],
            &params,
        )
        .expect("production terminal witness fixture");

        let one = CyclotomicRing::<F, D>::one();
        let setup = AkitaVerifierSetup::from_parts(
            Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    AkitaSetupSeed {
                        max_num_vars: 1,
                        max_num_batched_polys: 2,
                        gen_ring_dim: D,
                        max_setup_len: 1,
                        public_matrix_seed: [7; 32],
                    },
                    FlatMatrix::from_ring_slice(&[one]),
                ),
            ),
            SetupPrefixVerifierRegistry::new(),
        );
        let opening_point = RingOpeningPoint {
            position_weights: vec![F::one()],
            live_block_weights: vec![F::one()],
        };
        let multiplier = RingMultiplierOpeningPoint::from_ring(vec![one], vec![one]);
        let relation = RingRelationInstance::new(
            RelationMatrixRowLayout::WithoutCommitmentBlocks,
            vec![sparse_challenges(1), sparse_challenges(-1)],
            vec![opening_point.clone(), opening_point],
            vec![multiplier.clone(), multiplier],
            opening_batch,
            vec![F::one(); 2],
            RingVec::from_ring_elems(&[one; 2]),
            RingVec::from_ring_elems::<D>(&[CyclotomicRing::zero()]),
            RingVec::from_coeffs(Vec::new()),
            dims,
        )
        .expect("terminal relation fixture");

        (setup, relation, params, witness, groups)
    }

    fn assert_production_matches_legacy<const D: usize>() {
        let (setup, relation, params, witness, groups) = grouped_terminal_fixture::<D>();

        verify_terminal_ring_relations(&setup, &relation, &params, &witness)
            .expect("valid grouped terminal witness");
        // `akita-prover::protocol::ring_relation::relation_quotient::
        // compute_multi_group_relation_quotient` is crate-private, so importing
        // it here would require a test-only cross-crate wrapper. Instead, apply
        // its exact cyclic quotient equation to the same ring operands consumed
        // by the production direct checker above.
        assert_direct_matches_legacy::<D>(TerminalRowRole::Consistency, &groups);
        for group in &groups {
            assert_direct_matches_legacy::<D>(TerminalRowRole::A, std::slice::from_ref(group));
        }

        let mut swapped = witness.clone();
        swapped.z_payloads.swap(0, 1);
        let e_len = swapped.layout.groups[0].e_field_elems;
        let mut e = swapped.e_fields.coeffs().to_vec();
        e.rotate_left(e_len);
        swapped.e_fields = RingVec::from_coeffs(e);
        let t_len = swapped.layout.groups[0].t_field_elems;
        let mut t = swapped.t_fields.coeffs().to_vec();
        t.rotate_left(t_len);
        swapped.t_fields = RingVec::from_coeffs(t);
        assert!(verify_terminal_ring_relations(&setup, &relation, &params, &swapped).is_err());

        let mut tampered_t = witness;
        let mut t = tampered_t.t_fields.coeffs().to_vec();
        t[D / 2] += F::one();
        tampered_t.t_fields = RingVec::from_coeffs(t);
        assert!(verify_terminal_ring_relations(&setup, &relation, &params, &tampered_t).is_err());
    }

    #[test]
    fn production_terminal_checker_matches_legacy_grouped_quotient_semantics() {
        assert_production_matches_legacy::<64>();
        assert_production_matches_legacy::<128>();
    }

    #[test]
    fn decoded_terminal_witness_rejects_coefficients_outside_i16() {
        assert_eq!(
            narrow_terminal_z_i16(vec![i64::from(i16::MIN), 0, i64::from(i16::MAX)])
                .expect("i16 boundary values"),
            vec![i16::MIN, 0, i16::MAX]
        );
        assert!(matches!(
            narrow_terminal_z_i16(vec![i64::from(i16::MAX) + 1]),
            Err(AkitaError::InvalidProof)
        ));
        assert!(matches!(
            narrow_terminal_z_i16(vec![i64::from(i16::MIN) - 1]),
            Err(AkitaError::InvalidProof)
        ));
    }
}
