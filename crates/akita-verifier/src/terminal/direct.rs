//! Deterministic terminal checks over the revealed terminal response.

use akita_algebra::CyclotomicRing;
use akita_challenges::{Challenges, SparseChallenge};
use akita_error::AkitaError;
use akita_params::{dispatch_for_field, TerminalFoldParams};
use akita_transcript::{field_digest, FIELD_DIGEST_BYTES};
use akita_types::{
    decode_terminal_z_golomb_payload, recover_ring_subfield_inner_product, FpExtEncoding,
    PreparedOpeningPoint, RingMultiplierOpeningPoint, RingVec, TerminalResponse,
};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

use crate::prepared_cache::TerminalNttCache;

fn sparse_challenge_mul_accumulate<F, const D: usize>(
    challenge: &SparseChallenge,
    value: &CyclotomicRing<F, D>,
    destination: &mut CyclotomicRing<F, D>,
) -> Result<(), AkitaError>
where
    F: Field + Ring,
{
    challenge.validate::<D>()?;
    for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
        let position = usize::try_from(position).map_err(|_| AkitaError::InvalidProof)?;
        match coefficient {
            1 => value.shift_accumulate_into(destination, position),
            -1 => value.shift_sub_into(destination, position),
            2 => {
                value.shift_accumulate_into(destination, position);
                value.shift_accumulate_into(destination, position);
            }
            -2 => {
                value.shift_sub_into(destination, position);
                value.shift_sub_into(destination, position);
            }
            _ => value.shift_scale_accumulate_into(
                destination,
                position,
                F::from_i64(i64::from(coefficient)),
            ),
        }
    }
    Ok(())
}

fn sparse_challenge_dot<F, const D: usize>(
    row: &[SparseChallenge],
    input: &[CyclotomicRing<F, D>],
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: Field + Ring,
{
    if row.len() != input.len() {
        return Err(AkitaError::InvalidProof);
    }
    let mut sum = CyclotomicRing::zero();
    for (challenge, value) in row.iter().zip(input) {
        sparse_challenge_mul_accumulate(challenge, value, &mut sum)?;
    }
    Ok(sum)
}

/// Digests of the last live `e` and `t` block, which the prover sends in
/// place of the blocks under `recompute-last-block`.
pub(crate) struct LastBlockDigests {
    pub(crate) e: [u8; FIELD_DIGEST_BYTES],
    pub(crate) t: [u8; FIELD_DIGEST_BYTES],
}

/// Recovers the last live block of `e` or `t`, which the prover omits, from
/// the fold relation that the block completes.
struct LastBlockRecovery<'a, F: Field, const D: usize> {
    challenge: &'a SparseChallenge,
    inverse: CyclotomicRing<F, D>,
}

impl<'a, F, const D: usize> LastBlockRecovery<'a, F, D>
where
    F: Field + Ring,
{
    /// Invert the last block's fold challenge, rejecting a challenge that is
    /// not a unit.
    fn new(challenge: &'a SparseChallenge) -> Result<Self, AkitaError> {
        let mut dense = CyclotomicRing::zero();
        sparse_challenge_mul_accumulate(challenge, &CyclotomicRing::one(), &mut dense)?;
        let inverse = dense.inverse().ok_or(AkitaError::InvalidProof)?;
        Ok(Self { challenge, inverse })
    }

    /// Solve `partial + challenge * block == target` for `block`.
    ///
    /// The quotient by the inverse is only a candidate. The sparse product
    /// checks it against the relation, so acceptance never relies on the
    /// inversion or on the dense product.
    fn solve(
        &self,
        partial: CyclotomicRing<F, D>,
        target: &CyclotomicRing<F, D>,
    ) -> Result<CyclotomicRing<F, D>, AkitaError> {
        let block = self.inverse * (*target - partial);
        let mut folded = partial;
        sparse_challenge_mul_accumulate(self.challenge, &block, &mut folded)?;
        if folded != *target {
            return Err(AkitaError::InvalidProof);
        }
        Ok(block)
    }
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

/// Both sides of the A rows `sum_b c_b * t_b == A * z`, with the left side
/// folded over the transmitted `t` blocks only.
#[tracing::instrument(skip_all, name = "terminal_direct_a_rows")]
#[allow(clippy::type_complexity)]
fn a_row_sides<F, const D: usize>(
    terminal_ntt: &TerminalNttCache,
    t: &[CyclotomicRing<F, D>],
    z: &[[i16; D]],
    challenges: &[SparseChallenge],
    n_a: usize,
    n_a_cols: usize,
) -> Result<(Vec<CyclotomicRing<F, D>>, Vec<CyclotomicRing<F, D>>), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
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
    let (rhs, lhs) = cfg_join!(|| super::ntt::centered_rows(terminal_ntt, n_a, z), || {
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
                    |mut sum, (challenge, rows)| {
                        let row = rows.get(row_index).ok_or(AkitaError::InvalidProof)?;
                        sparse_challenge_mul_accumulate(challenge, row, &mut sum)?;
                        Ok::<_, AkitaError>(sum)
                    },
                )
            })
            .collect::<Result<Vec<_>, AkitaError>>()
    });
    let rhs = rhs?;
    let lhs = lhs?;
    if lhs.len() != rhs.len() {
        return Err(AkitaError::InvalidProof);
    }
    Ok((lhs, rhs))
}

/// Check reduced consistency and A rows for a quotient-free terminal witness.
///
/// Without `last_block_digests` the response carries every live block and
/// both relations are compared as sent. With them, the response omits the
/// last live block of `e` and of `t`. Each relation is linear in its omitted
/// block with an invertible fold challenge, so the block is recovered from
/// the relation and compared with the digest the prover sent before that
/// challenge was drawn. Returns the complete `e`.
#[tracing::instrument(skip_all, name = "terminal_direct_ring_relations")]
pub(crate) fn verify_terminal_ring_relations<F>(
    terminal_ntt: &TerminalNttCache,
    challenges: &Challenges,
    multiplier: &RingMultiplierOpeningPoint<F>,
    params: &TerminalFoldParams,
    terminal_response: &TerminalResponse<F>,
    last_block_digests: Option<&LastBlockDigests>,
) -> Result<RingVec<F>, AkitaError>
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
            let (sent_challenges, recovery) = match last_block_digests {
                Some(digests) => {
                    let (last_challenge, sent_challenges) = challenges
                        .as_slice()
                        .split_last()
                        .ok_or(AkitaError::InvalidProof)?;
                    let last_block = LastBlockRecovery::<F, D_A>::new(last_challenge)?;
                    (sent_challenges, Some((last_block, digests)))
                }
                None => (challenges.as_slice(), None),
            };
            let expected_t_len = sent_challenges
                .len()
                .checked_mul(params.inner.matrix.output_rank())
                .ok_or(AkitaError::InvalidProof)?;
            if e.len() != sent_challenges.len() || t.len() != expected_t_len {
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
                            blocks = sent_challenges.len()
                        )
                        .entered();
                        sparse_challenge_dot(sent_challenges, e)?
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
                    };
                    Ok::<_, AkitaError>((folded, reduced))
                },
                || {
                    a_row_sides::<F, D_A>(
                        terminal_ntt,
                        t,
                        z_centered,
                        sent_challenges,
                        n_a,
                        n_a_cols,
                    )
                }
            );
            let (folded, reduced) = consistency?;
            let (t_folded, a_z) = a_rows?;
            let Some((last_block, digests)) = recovery else {
                let _span = tracing::info_span!("terminal_direct_a_compare", rows = n_a).entered();
                if folded != reduced || t_folded != a_z {
                    return Err(AkitaError::InvalidProof);
                }
                return Ok(witness.e_fields.clone());
            };
            let _span = tracing::info_span!("terminal_direct_a_recover", rows = n_a).entered();
            let e_last = last_block.solve(folded, &reduced)?;
            let t_last = t_folded
                .into_iter()
                .zip(&a_z)
                .map(|(partial, target)| last_block.solve(partial, target))
                .collect::<Result<Vec<_>, AkitaError>>()?;
            let t_last = RingVec::from_ring_elems(&t_last);
            if field_digest(e_last.coefficients()) != digests.e
                || field_digest(t_last.coeffs()) != digests.t
            {
                return Err(AkitaError::InvalidProof);
            }
            let mut e_fields = witness.e_fields.coeffs().to_vec();
            e_fields.extend_from_slice(e_last.coefficients());
            Ok::<_, AkitaError>(RingVec::from_coeffs(e_fields))
        }
    )
}

/// Check the public opening directly against the complete folded `e` segment.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "terminal_direct_trace")]
pub(crate) fn verify_terminal_trace<F, E>(
    multiplier: &RingMultiplierOpeningPoint<F>,
    params: &TerminalFoldParams,
    e_fields: &RingVec<F>,
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
    if row_coefficients.len() != 1 || claim_scales.is_some_and(|scales| scales.len() != 1) {
        return Err(AkitaError::InvalidProof);
    }
    let mut actual = E::zero();
    dispatch_for_field!(
        akita_params::ProtocolDispatchSlot::Role(akita_params::RingRole::Inner),
        F,
        params.d_a(),
        |D| {
            let e_rings = e_fields.as_ring_slice::<D>()?;
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
            positions: vec![0, 3, (D - 1) as u32].into(),
            coeffs: vec![2, -1, -2].into(),
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
        let mut actual = CyclotomicRing::zero();
        sparse_challenge_mul_accumulate(&challenge, &value, &mut actual)
            .expect("valid sparse challenge");
        assert_eq!(actual, dense * value);
    }

    #[test]
    fn sparse_challenge_product_matches_schoolbook() {
        assert_sparse_challenge_product::<64>();
        assert_sparse_challenge_product::<128>();
    }

    fn assert_last_block_recovery<const D: usize>() {
        let challenge = SparseChallenge {
            positions: vec![1, 5, (D - 2) as u32].into(),
            coeffs: vec![1, -2, 1].into(),
        };
        let ramp = |step: i64| {
            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|index| {
                F::from_i64(index as i64 * step - 7)
            }))
        };
        let (block, partial) = (ramp(3), ramp(-5));
        let mut target = partial;
        sparse_challenge_mul_accumulate(&challenge, &block, &mut target)
            .expect("valid sparse challenge");

        let recovery = LastBlockRecovery::<F, D>::new(&challenge).expect("challenge is a unit");
        assert_eq!(recovery.solve(partial, &target), Ok(block));

        // The relation, not the inversion, decides acceptance.
        let wrong_inverse = LastBlockRecovery {
            challenge: &challenge,
            inverse: recovery.inverse + CyclotomicRing::one(),
        };
        assert_eq!(
            wrong_inverse.solve(partial, &target),
            Err(AkitaError::InvalidProof)
        );
    }

    #[test]
    fn last_block_recovery_solves_the_fold_relation() {
        assert_last_block_recovery::<64>();
        assert_last_block_recovery::<128>();
    }

    #[test]
    fn last_block_recovery_rejects_a_non_unit_challenge() {
        let zero = SparseChallenge {
            positions: Vec::new().into(),
            coeffs: Vec::new().into(),
        };
        assert!(matches!(
            LastBlockRecovery::<F, 64>::new(&zero),
            Err(AkitaError::InvalidProof)
        ));
    }
}
