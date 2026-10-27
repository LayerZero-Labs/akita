//! Deterministic terminal checks over the revealed terminal response.

use akita_algebra::CyclotomicRing;
use akita_challenges::{Challenges, SparseChallenge};
use akita_error::AkitaError;
use akita_params::{dispatch_for_field, TerminalFoldParams};
use akita_types::{
    decode_terminal_z_golomb_payload, recover_ring_subfield_inner_product, FpExtEncoding,
    PreparedOpeningPoint, RingMultiplierOpeningPoint, TerminalResponse,
};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

use crate::prepared_cache::TerminalNttCache;

/// `Σ_b challenges[b] · values[b]` in `Z_q[X]/(X^D + 1)`, one signed sum of
/// ring coefficients per output coefficient.
///
/// `X^position · v` contributes `v[k − position]` at coefficient `k ≥ position`
/// and `−v[k + D − position]` below it. A challenge coefficient `c` enters as
/// `|c|` unit terms with the sign of `c`.
fn sparse_challenge_dot<'a, F, const D: usize>(
    challenges: &[SparseChallenge],
    values: impl IntoIterator<Item = &'a CyclotomicRing<F, D>>,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: Field + Ring + 'a,
{
    let mut values = values.into_iter();
    let mut terms = Vec::new();
    for challenge in challenges {
        let value = values.next().ok_or(AkitaError::InvalidProof)?;
        challenge.validate::<D>()?;
        for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
            let position = usize::try_from(position).map_err(|_| AkitaError::InvalidProof)?;
            for _ in 0..coefficient.unsigned_abs() {
                terms.push((value.coefficients(), position, coefficient < 0));
            }
        }
    }
    if values.next().is_some() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(CyclotomicRing::from_coefficients(std::array::from_fn(
        |k| {
            F::signed_sum(terms.iter().map(|&(value, position, negative)| {
                if k >= position {
                    (&value[k - position], negative)
                } else {
                    (&value[k + D - position], !negative)
                }
            }))
        },
    )))
}

#[inline]
fn centered_ring<F, const D: usize>(coeffs: &[i16; D]) -> CyclotomicRing<F, D>
where
    F: Field + Ring,
{
    CyclotomicRing::from_coefficients(std::array::from_fn(|index| {
        F::from_i64(i64::from(coeffs[index]))
    }))
}

/// `Σ_{p,d} w_p·g_d·z[p·digits + d]` for base-field position weights `w` and
/// gadget scalars `g`: the consistency fold of the decoded `z`, computed as
/// one weight-vector dot product per ring coefficient.
fn base_reduced_z<F, const D: usize>(
    position_weights: &[F],
    gadget: &[F],
    z: &[[i16; D]],
    num_positions: usize,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: Field + Ring,
{
    let terms = num_positions
        .checked_mul(gadget.len())
        .ok_or(AkitaError::InvalidProof)?;
    let position_weights = position_weights.get(..num_positions).ok_or_else(|| {
        AkitaError::InvalidInput("base multiplier product index out of range".into())
    })?;
    let z = z.get(..terms).ok_or(AkitaError::InvalidProof)?;
    let weights: Vec<F> = position_weights
        .iter()
        .flat_map(|&weight| gadget.iter().map(move |&scalar| weight * scalar))
        .collect();
    let mut column = Vec::with_capacity(terms);
    Ok(CyclotomicRing::from_coefficients(std::array::from_fn(
        |coefficient| {
            column.clear();
            column.extend(
                z.iter()
                    .map(|ring| F::from_i64(i64::from(ring[coefficient]))),
            );
            F::dot_product(&weights, &column)
        },
    )))
}

#[tracing::instrument(skip_all, name = "terminal_direct_a_rows")]
fn check_a_rows<F, const D: usize>(
    terminal_ntt: &TerminalNttCache,
    t: &[CyclotomicRing<F, D>],
    z: &[[i16; D]],
    challenges: &Challenges,
    n_a: usize,
    n_a_cols: usize,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
{
    if t.len()
        != challenges
            .as_slice()
            .len()
            .checked_mul(n_a)
            .ok_or(AkitaError::InvalidProof)?
        || z.len() != n_a_cols
    {
        return Err(AkitaError::InvalidProof);
    }
    let (rhs, lhs) = cfg_join!(|| super::ntt::centered_rows(terminal_ntt, n_a, z), || {
        let _span = tracing::info_span!(
            "terminal_direct_a_lhs",
            rows = n_a,
            challenges = challenges.as_slice().len()
        )
        .entered();
        (0..n_a)
            .map(|row_index| {
                let rows = t
                    .chunks_exact(n_a)
                    .map(|rows| rows.get(row_index).ok_or(AkitaError::InvalidProof))
                    .collect::<Result<Vec<_>, AkitaError>>()?;
                sparse_challenge_dot(challenges.as_slice(), rows)
            })
            .collect::<Result<Vec<_>, AkitaError>>()
    });
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
pub(crate) fn verify_terminal_ring_relations<F>(
    terminal_ntt: &TerminalNttCache,
    challenges: &Challenges,
    multiplier: &RingMultiplierOpeningPoint<F>,
    params: &TerminalFoldParams,
    terminal_response: &TerminalResponse<F>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
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
    if params
        .validate_terminal_linf_cap(group_layout.z_linf_cap)
        .is_err()
    {
        return Err(AkitaError::InvalidProof);
    }
    dispatch_for_field!(
        akita_params::ProtocolDispatchSlot::Role(akita_params::RingRole::Inner),
        F,
        params.d_a(),
        |D_A| {
            let e_rings = witness.e_fields.as_ring_slice::<D_A>()?;
            let t_rings = witness.t_fields.as_ring_slice::<D_A>()?;
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
                decode_terminal_z_golomb_payload(
                    witness.z_payloads.first().ok_or(AkitaError::InvalidProof)?,
                    group_layout,
                )?
            };
            if params.response_l2_sq_cap().is_some_and(|cap| {
                akita_params::sis::checked_centered_l2_sq(&z_values).is_none_or(|norm| norm > cap)
            }) {
                return Err(AkitaError::InvalidProof);
            }
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
            {
                let _span = tracing::info_span!(
                    "terminal_direct_challenges",
                    num_blocks = params.blocks.live_blocks
                )
                .entered();
                if challenges.as_slice().len() != params.blocks.live_blocks {
                    return Err(AkitaError::InvalidProof);
                }
                for challenge in challenges.as_slice() {
                    challenge.validate::<D_A>()?;
                }
            }
            let expected_t_len = params
                .blocks
                .live_blocks
                .checked_mul(params.inner.matrix.output_rank())
                .ok_or(AkitaError::InvalidProof)?;
            if e.len() != params.blocks.live_blocks || t.len() != expected_t_len {
                return Err(AkitaError::InvalidProof);
            }
            let n_a = params.inner.matrix.output_rank();
            let n_a_cols = params.inner.matrix.input_width();
            let num_positions = params.blocks.positions_per_block;
            let num_digits_inner = params.inner.digits.num_digits;
            let log_basis_inner = params.inner.digits.log_basis;
            multiplier.ensure_ring_dim::<D_A>()?;
            let (consistency, a_rows) = cfg_join!(
                || {
                    let _span = tracing::info_span!(
                        "terminal_direct_consistency",
                        num_blocks = params.blocks.live_blocks,
                        num_positions
                    )
                    .entered();
                    let folded = {
                        let _span = tracing::info_span!(
                            "terminal_direct_consistency_fold_e",
                            blocks = challenges.as_slice().len()
                        )
                        .entered();
                        sparse_challenge_dot(challenges.as_slice(), e)?
                    };
                    let reduced = {
                        let _span = tracing::info_span!(
                            "terminal_direct_consistency_reduce_z",
                            positions = num_positions,
                            digits = num_digits_inner
                        )
                        .entered();
                        let gadget = akita_params::gadget_row_scalars::<F>(
                            num_digits_inner,
                            log_basis_inner,
                        );
                        if let Some(point) = multiplier.as_base() {
                            base_reduced_z(
                                &point.position_weights,
                                &gadget,
                                z_centered,
                                num_positions,
                            )?
                        } else {
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
                                multiplier.accumulate_position_product(
                                    position,
                                    &z_value,
                                    &mut reduced,
                                )?;
                            }
                            reduced
                        }
                    };
                    Ok::<_, AkitaError>((folded, reduced))
                },
                || {
                    check_a_rows::<F, D_A>(terminal_ntt, t, z_centered, challenges, n_a, n_a_cols)
                }
            );
            let (folded, reduced) = consistency?;
            a_rows?;
            if folded != reduced {
                return Err(AkitaError::InvalidProof);
            }
            Ok::<(), AkitaError>(())
        }
    )?;
    Ok(())
}

/// Check the public opening directly against the revealed folded `e` segment.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "terminal_direct_trace")]
pub(crate) fn verify_terminal_trace<F, E>(
    multiplier: &RingMultiplierOpeningPoint<F>,
    params: &TerminalFoldParams,
    terminal_response: &TerminalResponse<F>,
    prepared_point: &PreparedOpeningPoint<F, E>,
    row_coefficients: &[E],
    claim_scales: Option<&[E]>,
    global_scale: E,
    target: E,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let witness = terminal_response;
    if row_coefficients.len() != 1 || claim_scales.is_some_and(|scales| scales.len() != 1) {
        return Err(AkitaError::InvalidProof);
    }
    let mut actual = E::zero();
    dispatch_for_field!(
        akita_params::ProtocolDispatchSlot::Role(akita_params::RingRole::Inner),
        F,
        params.d_a(),
        |D| {
            let e_rings = witness.e_fields.as_ring_slice::<D>()?;
            let e = e_rings;
            let packed_inner = prepared_point.packed_inner_trusted::<D>()?;
            let claim_e = e;
            if claim_e.len() != params.blocks.live_blocks {
                return Err(AkitaError::InvalidProof);
            }
            let claim_opening = if multiplier.as_base().is_none() {
                claim_e
                    .iter()
                    .enumerate()
                    .try_fold(E::zero(), |opening, (block, value)| {
                        let weight = multiplier
                            .fold_subfield_value::<E>(block)?
                            .ok_or(AkitaError::InvalidProof)?;
                        let value =
                            recover_ring_subfield_inner_product::<F, E, D>(value, packed_inner)?;
                        Ok::<_, AkitaError>(opening + weight * value)
                    })?
            } else {
                let mut outer_eval = CyclotomicRing::zero();
                for (block, value) in claim_e.iter().enumerate() {
                    let scale = multiplier
                        .fold_constant_coeff(block)
                        .ok_or(AkitaError::InvalidProof)?;
                    outer_eval += value.scale(&scale);
                }
                recover_ring_subfield_inner_product::<F, E, D>(&outer_eval, packed_inner)?
            };
            let scale = claim_scales
                .and_then(|scales| scales.first())
                .copied()
                .unwrap_or(global_scale);
            actual += row_coefficients[0] * scale * claim_opening;
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
    use jolt_field::{One, Prime128OffsetA7F7, Zero};

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

    fn cyclic_product<F: Field, const D: usize>(
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
        F: Field,
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

    fn assert_sparse_challenge_product<const D: usize>() {
        let challenge = SparseChallenge {
            positions: vec![0, 3, 5, (D - 1) as u32].into(),
            coeffs: vec![2, -1, 5, -2].into(),
        };
        let value = CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|index| {
            F::from_i64(index as i64 - 9)
        }));
        let dense = CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|index| {
            challenge
                .positions
                .iter()
                .position(|&position| position as usize == index)
                .map_or_else(F::zero, |position| {
                    F::from_i64(i64::from(challenge.coeffs[position]))
                })
        }));
        let actual = sparse_challenge_dot(&[challenge.clone(), challenge], [&value, &value])
            .expect("valid sparse challenge");
        assert_eq!(actual, (dense * value).scale(&F::from_i64(2)));
    }

    #[test]
    fn sparse_challenge_product_matches_schoolbook() {
        assert_sparse_challenge_product::<64>();
        assert_sparse_challenge_product::<128>();
    }

    #[test]
    fn sparse_challenge_dot_rejects_value_count_mismatch() {
        let challenge = SparseChallenge {
            positions: vec![1].into(),
            coeffs: vec![1].into(),
        };
        let value = CyclotomicRing::<F, 64>::zero();
        assert!(sparse_challenge_dot(&[challenge.clone(), challenge.clone()], [&value]).is_err());
        assert!(sparse_challenge_dot(&[challenge], [&value, &value]).is_err());
    }

    /// `Σ_p w_p · Σ_d g_d · z[p·digits + d]`, evaluated with ring scaling.
    #[test]
    fn base_reduced_z_matches_definition() {
        const D: usize = 64;
        let (positions, digits) = (3, 2);
        let weights: Vec<F> = (0..positions)
            .map(|p| F::from_i64(3 * p as i64 + 2))
            .collect();
        let gadget: Vec<F> = (0..digits).map(|d| F::from_i64(1 << (4 * d))).collect();
        let z: Vec<[i16; D]> = (0..positions * digits + 1)
            .map(|ring| std::array::from_fn(|k| ((ring * 37 + k * 11) % 41) as i16 - 20))
            .collect();
        let mut expected = CyclotomicRing::<F, D>::zero();
        for (p, weight) in weights.iter().enumerate() {
            for (d, scalar) in gadget.iter().enumerate() {
                expected += centered_ring::<F, D>(&z[p * digits + d]).scale(&(*weight * *scalar));
            }
        }
        assert_eq!(
            base_reduced_z(&weights, &gadget, &z, positions).unwrap(),
            expected
        );
        assert!(
            base_reduced_z(&weights, &gadget, &z[..positions * digits - 1], positions).is_err()
        );
        assert!(base_reduced_z(&weights[..positions - 1], &gadget, &z, positions).is_err());
    }
}
