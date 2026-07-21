//! Deterministic terminal checks over the revealed terminal response.

use akita_algebra::CyclotomicRing;
use akita_challenges::{Challenges, SparseChallenge};
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, HalvingField,
};
use akita_types::{
    decode_terminal_z_golomb_payload, dispatch_for_field, recover_ring_subfield_inner_product,
    AkitaVerifierSetup, FpExtEncoding, PreparedOpeningPoint, RingMultiplierOpeningPoint,
    TerminalCommittedGroupParams, TerminalResponse,
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
    challenges: &Challenges,
    multiplier: &RingMultiplierOpeningPoint<F>,
    params: &TerminalCommittedGroupParams,
    sparse: &akita_challenges::SparseChallengeConfig,
    terminal_response: &TerminalResponse<F>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HalvingField,
{
    let witness = terminal_response;
    if witness.layout.ring_dimension != params.d_a() || witness.layout.groups.len() != 1 {
        return Err(AkitaError::InvalidProof);
    }
    let group_layout = witness
        .layout
        .groups
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    let (honest_cap, security_cap) = params.response_linf_bounds(sparse)?;
    dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        F,
        params.d_a(),
        |D_A| {
            let e_rings = witness.e_fields.as_ring_slice::<D_A>().map_err(|error| {
                AkitaError::InvalidInput(format!("terminal e layout failed: {error:?}"))
            })?;
            let t_rings = witness.t_fields.as_ring_slice::<D_A>().map_err(|error| {
                AkitaError::InvalidInput(format!("terminal t layout failed: {error:?}"))
            })?;
            let e = e_rings;
            let t = t_rings;
            let z_values = {
                let _span = tracing::info_span!(
                    "terminal_direct_decode",
                    e_field_elems = group_layout.e_field_elems,
                    t_field_elems = group_layout.t_field_elems,
                    z_coords = group_layout.z_coords
                )
                .entered();
                let values = decode_terminal_z_golomb_payload(
                    witness.z_payloads.first().ok_or(AkitaError::InvalidProof)?,
                    group_layout.z_coords,
                    honest_cap,
                    security_cap,
                    Some(group_layout.z_payload_bytes),
                )
                .map_err(|error| {
                    AkitaError::InvalidInput(format!("terminal z decode failed: {error:?}"))
                })?;
                narrow_terminal_z_i16(values)?
            };
            let z_centered = {
                let _span = tracing::info_span!(
                    "terminal_direct_decode_z_rings",
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
                let _span = tracing::info_span!(
                    "terminal_direct_challenges",
                    num_blocks = params.num_live_blocks
                )
                .entered();
                challenge_rings::<F, D_A>(challenges).map_err(|error| {
                    AkitaError::InvalidInput(format!(
                        "terminal challenge conversion failed: {error:?}"
                    ))
                })?
            };
            let expected_t_len = params
                .num_live_blocks
                .checked_mul(params.inner_commit_matrix.output_rank())
                .ok_or(AkitaError::InvalidProof)?;
            if e.len() != params.num_live_blocks || t.len() != expected_t_len {
                return Err(AkitaError::InvalidInput(format!(
                    "terminal raw segment ring count mismatch: e={}, expected_e={}, t={}, expected_t={expected_t_len}",
                    e.len(),
                    params.num_live_blocks,
                    t.len(),
                )));
            }
            let n_a = params.inner_commit_matrix.output_rank();
            let n_a_cols = params.inner_commit_matrix.input_width();
            let num_positions = params.num_positions_per_block;
            let num_digits_inner = params.num_digits_inner;
            let log_basis_inner = params.log_basis_inner;
            let position_rings = multiplier
                .position_rings_trusted::<D_A>()
                .map_err(|error| {
                    AkitaError::InvalidInput(format!(
                        "terminal multiplier layout failed: {error:?}"
                    ))
                })?;
            let (consistency, a_rows) = cfg_join!(
                || {
                    let _span = tracing::info_span!(
                        "terminal_direct_consistency",
                        num_blocks = params.num_live_blocks,
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
                        let gadget =
                            akita_types::gadget_row_scalars::<F>(num_digits_inner, log_basis_inner);
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
                        n_a.checked_mul(n_a_cols).ok_or(AkitaError::InvalidProof)?,
                    )
                }
            );
            let (folded, reduced) = consistency.map_err(|error| {
                AkitaError::InvalidInput(format!(
                    "terminal consistency computation failed: {error:?}"
                ))
            })?;
            a_rows.map_err(|error| {
                AkitaError::InvalidInput(format!("terminal A-row check failed: {error:?}"))
            })?;
            if folded != reduced {
                return Err(AkitaError::InvalidInput(
                    "terminal consistency equation failed".into(),
                ));
            }
            Ok::<(), AkitaError>(())
        }
    )?;
    Ok(())
}

/// Check the public opening directly against the revealed folded `e` segment.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "terminal_direct_trace")]
pub(super) fn verify_terminal_trace<F, E>(
    multiplier: &RingMultiplierOpeningPoint<F>,
    params: &TerminalCommittedGroupParams,
    terminal_response: &TerminalResponse<F>,
    prepared_point: &PreparedOpeningPoint<F, E>,
    row_coefficients: &[E],
    claim_scales: Option<&[E]>,
    global_scale: E,
    target: E,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let witness = terminal_response;
    if row_coefficients.len() != 1 || claim_scales.is_some_and(|scales| scales.len() != 1) {
        return Err(AkitaError::InvalidProof);
    }
    let mut actual = E::zero();
    dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        F,
        params.d_a(),
        |D| {
            let e_rings = witness.e_fields.as_ring_slice::<D>()?;
            let e = e_rings;
            let packed_inner = prepared_point.packed_inner_trusted::<D>()?;
            let claim_e = e;
            if claim_e.len() != params.num_live_blocks {
                return Err(AkitaError::InvalidProof);
            }
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
                .and_then(|scales| scales.first())
                .copied()
                .unwrap_or(global_scale);
            actual += row_coefficients[0] * scale * opening;
            Ok::<(), AkitaError>(())
        }
    )?;
    if actual != target {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7;

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
        let z = monomial::<D>(0, i64::from(z_constant));
        TerminalGroupFixture { challenge, e, t, z }
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

    #[test]
    fn direct_reduced_relation_matches_legacy_quotient_equation() {
        for role in [TerminalRowRole::Consistency, TerminalRowRole::A] {
            assert_direct_matches_legacy::<64>(role, &[group_fixture::<64>(1, 1)]);
            assert_direct_matches_legacy::<128>(role, &[group_fixture::<128>(-1, 2)]);
        }
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
