//! Deterministic terminal checks over the revealed terminal response.

use akita_algebra::row_batch::ROW_BATCH;
use akita_error::AkitaError;
use akita_params::{dispatch_for_field, TerminalFoldParams};
use akita_transcript::{field_digest, FIELD_DIGEST_BYTES};
use akita_types::{
    decode_terminal_z_golomb_payload, recover_ring_subfield_inner_product, FpExtEncoding,
    PreparedOpeningPoint, RingMultiplierOpeningPoint, RingVec, TerminalResponse,
};
use jolt_algebra::solinas::parallel::*;
use jolt_algebra::CyclotomicRing;
use jolt_algebra::{CanonicalEncoding, ExtField, Field, Ring};
use jolt_crypto::challenge::{Challenges, SparseChallenge};
use std::iter::repeat_n;

use crate::prepared_cache::TerminalNttCache;

/// `Σ_b challenges[b] · values[b]` in `Z_q[X]/(X^D + 1)`, one signed sum of
/// ring coefficients per output coefficient.
///
/// `X^position · v` contributes `v[k − position]` at coefficient `k ≥ position`
/// and `−v[k + D − position]` below it. A challenge coefficient `c` enters as
/// `|c|` unit terms with the sign of `c`. The terms are generated lazily from
/// the validated challenges, so no per-term storage is allocated.
fn sparse_challenge_dot<'a, F, const D: usize>(
    challenges: &[SparseChallenge],
    values: impl Iterator<Item = &'a CyclotomicRing<F, D>> + Clone,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: Field + Ring + 'a,
{
    if values.clone().count() != challenges.len() {
        return Err(AkitaError::InvalidProof);
    }
    for challenge in challenges {
        challenge.validate::<D>()?;
    }
    Ok(CyclotomicRing::from_coefficients(std::array::from_fn(
        |k| {
            F::signed_sum(
                challenges
                    .iter()
                    .zip(values.clone())
                    .flat_map(|(challenge, value)| {
                        let value = value.coefficients();
                        challenge.positions.iter().zip(&challenge.coeffs).flat_map(
                            move |(&position, &coefficient)| {
                                // `validate` pinned `position < D`.
                                let position = position as usize;
                                let negative = coefficient < 0;
                                let term = if k >= position {
                                    (&value[k - position], negative)
                                } else {
                                    (&value[k + D - position], !negative)
                                };
                                repeat_n(term, usize::from(coefficient.unsigned_abs()))
                            },
                        )
                    }),
            )
        },
    )))
}

/// Digests of the last live `e` and `t` block, which the prover sends in
/// place of the blocks.
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
        let dense = sparse_challenge_dot(
            std::slice::from_ref(challenge),
            [&CyclotomicRing::one()].into_iter(),
        )?;
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
        let folded = partial
            + sparse_challenge_dot(std::slice::from_ref(self.challenge), [&block].into_iter())?;
        if folded != *target {
            return Err(AkitaError::InvalidProof);
        }
        Ok(block)
    }

    /// Recover the omitted last `e` block and the omitted last `t` block, and
    /// return the `e` block once both match their digests.
    ///
    /// Each relation holds for its recovered block by construction, so the
    /// digest comparison is the only check that binds the recovered blocks to
    /// the prover's earlier messages.
    fn recover_last_blocks(
        &self,
        e_partial: CyclotomicRing<F, D>,
        e_target: &CyclotomicRing<F, D>,
        a_rows: ARowSides<F, D>,
        digests: &LastBlockDigests,
    ) -> Result<CyclotomicRing<F, D>, AkitaError>
    where
        F: CanonicalEncoding,
    {
        let e_last = self.solve(e_partial, e_target)?;
        let t_last = a_rows
            .t_partials
            .into_iter()
            .zip(&a_rows.a_z)
            .map(|(partial, target)| self.solve(partial, target))
            .collect::<Result<Vec<_>, AkitaError>>()?;
        if field_digest(e_last.coefficients()) != digests.e
            || field_digest(RingVec::from_ring_elems(&t_last).coeffs()) != digests.t
        {
            return Err(AkitaError::InvalidProof);
        }
        Ok(e_last)
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

/// `Σ_{p,d} w_p·g_d·z[p·digits + d]` for base-field position weights `w` and
/// gadget scalars `g`: the consistency fold of the decoded `z`, computed per
/// ring coefficient as dot products over [`ROW_BATCH`]-term batches.
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
    let mut products = position_weights
        .iter()
        .flat_map(|&weight| gadget.iter().map(move |&scalar| weight * scalar));
    let mut weights = [F::zero(); ROW_BATCH];
    let mut column = [F::zero(); ROW_BATCH];
    let mut sum = [F::zero(); D];
    for z_batch in z.chunks(ROW_BATCH) {
        let weights = &mut weights[..z_batch.len()];
        for (slot, product) in weights.iter_mut().zip(products.by_ref()) {
            *slot = product;
        }
        let column = &mut column[..z_batch.len()];
        for (coefficient, sum) in sum.iter_mut().enumerate() {
            for (slot, ring) in column.iter_mut().zip(z_batch) {
                *slot = F::from_i64(i64::from(ring[coefficient]));
            }
            *sum += F::dot_product(weights, column);
        }
    }
    Ok(CyclotomicRing::from_coefficients(sum))
}

/// Both sides of the A rows `sum_b c_b * t_b == A * z`, one entry per row.
struct ARowSides<F: Field, const D: usize> {
    /// The left side, folded over the transmitted `t` blocks only.
    t_partials: Vec<CyclotomicRing<F, D>>,
    /// `A * z`.
    a_z: Vec<CyclotomicRing<F, D>>,
}

#[tracing::instrument(skip_all, name = "terminal_direct_a_rows")]
fn a_row_sides<F, const D: usize>(
    terminal_ntt: &TerminalNttCache,
    t: &[CyclotomicRing<F, D>],
    z: &[[i16; D]],
    challenges: &[SparseChallenge],
    n_a: usize,
    n_a_cols: usize,
) -> Result<ARowSides<F, D>, AkitaError>
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
                sparse_challenge_dot(challenges, t.iter().skip(row_index).step_by(n_a))
            })
            .collect::<Result<Vec<_>, AkitaError>>()
    });
    let rhs = rhs?;
    let lhs = lhs?;
    if lhs.len() != rhs.len() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(ARowSides {
        t_partials: lhs,
        a_z: rhs,
    })
}

/// Check reduced consistency and A rows for a quotient-free terminal witness.
///
/// The response omits the last live block of `e` and of `t`. Each relation is
/// linear in its omitted block with an invertible fold challenge, so the block
/// is recovered from the relation and compared with the digest the prover sent
/// before that challenge was drawn. Returns the complete `e`.
#[tracing::instrument(skip_all, name = "terminal_direct_ring_relations")]
pub(crate) fn verify_terminal_ring_relations<F>(
    terminal_ntt: &TerminalNttCache,
    challenges: &Challenges,
    multiplier: &RingMultiplierOpeningPoint<F>,
    params: &TerminalFoldParams,
    terminal_response: &TerminalResponse<F>,
    last_block_digests: &LastBlockDigests,
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
            let (last_challenge, sent_challenges) = challenges
                .as_slice()
                .split_last()
                .ok_or(AkitaError::InvalidProof)?;
            let last_block = LastBlockRecovery::<F, D_A>::new(last_challenge)?;
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
                        sparse_challenge_dot(sent_challenges, e.iter())?
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
            let _span = tracing::info_span!("terminal_direct_a_recover", rows = n_a).entered();
            let e_last =
                last_block.recover_last_blocks(folded, &reduced, a_rows?, last_block_digests)?;
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
    use jolt_algebra::{One, Prime128OffsetA7F7, Zero};

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
        let actual = sparse_challenge_dot(
            &[challenge.clone(), challenge],
            [&value, &value].into_iter(),
        )
        .expect("valid sparse challenge");
        assert_eq!(actual, (dense * value).scale(&F::from_i64(2)));
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
        let target = partial
            + sparse_challenge_dot(std::slice::from_ref(&challenge), [&block].into_iter())
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

    /// Recovery succeeds on any relation-consistent input, so only the digests
    /// can reject a wrong block; each wrongly bound digest must reject.
    #[test]
    fn recover_last_blocks_binds_both_blocks_to_their_digests() {
        const D: usize = 64;
        let challenge = SparseChallenge {
            positions: vec![1, 5, (D - 2) as u32].into(),
            coeffs: vec![1, -2, 1].into(),
        };
        let ramp = |step: i64| {
            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|index| {
                F::from_i64(index as i64 * step - 7)
            }))
        };
        let complete = |partial: CyclotomicRing<F, D>, block: &CyclotomicRing<F, D>| {
            partial
                + sparse_challenge_dot(std::slice::from_ref(&challenge), [block].into_iter())
                    .expect("valid sparse challenge")
        };
        let (e_partial, e_block) = (ramp(-5), ramp(3));
        let e_target = complete(e_partial, &e_block);
        let t_partials = vec![ramp(2), ramp(-11)];
        let t_blocks = [ramp(7), ramp(13)];
        let a_z: Vec<_> = t_partials
            .iter()
            .zip(&t_blocks)
            .map(|(partial, block)| complete(*partial, block))
            .collect();

        let recovery = LastBlockRecovery::<F, D>::new(&challenge).expect("challenge is a unit");
        let recover = |digests: LastBlockDigests| {
            recovery.recover_last_blocks(
                e_partial,
                &e_target,
                ARowSides {
                    t_partials: t_partials.clone(),
                    a_z: a_z.clone(),
                },
                &digests,
            )
        };
        let e = field_digest(e_block.coefficients());
        let t = field_digest(RingVec::from_ring_elems(&t_blocks).coeffs());
        assert_eq!(recover(LastBlockDigests { e, t }), Ok(e_block));

        let flip = |mut digest: [u8; FIELD_DIGEST_BYTES]| {
            digest[0] ^= 1;
            digest
        };
        let first_t_row_only = field_digest(t_blocks[0].coefficients());
        for digests in [
            LastBlockDigests { e: flip(e), t },
            LastBlockDigests { e, t: flip(t) },
            LastBlockDigests { e: t, t: e },
            LastBlockDigests { e, t: e },
            LastBlockDigests {
                e,
                t: first_t_row_only,
            },
        ] {
            assert_eq!(recover(digests), Err(AkitaError::InvalidProof));
        }
    }

    #[test]
    fn sparse_challenge_dot_rejects_value_count_mismatch() {
        let challenge = SparseChallenge {
            positions: vec![1].into(),
            coeffs: vec![1].into(),
        };
        let value = CyclotomicRing::<F, 64>::zero();
        assert!(sparse_challenge_dot(
            &[challenge.clone(), challenge.clone()],
            [&value].into_iter()
        )
        .is_err());
        assert!(sparse_challenge_dot(&[challenge], [&value, &value].into_iter()).is_err());
    }

    /// `Σ_p w_p · Σ_d g_d · z[p·digits + d]`, evaluated with ring scaling.
    #[test]
    fn base_reduced_z_matches_definition() {
        // 6, 32, and 34 terms: below, at, and across the batch boundary.
        for positions in [3, ROW_BATCH / 2, ROW_BATCH / 2 + 1] {
            assert_base_reduced_z_matches_definition(positions, 2);
        }
    }

    fn assert_base_reduced_z_matches_definition(positions: usize, digits: usize) {
        const D: usize = 64;
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
