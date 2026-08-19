//! Extension-claim fold verifier prefix: extension-opening reduction replay.

use super::super::*;
use super::{absorb_protocol_opening_points, FoldClaimMaterial, PreparedFoldOpeningPoint};
use akita_types::{dispatch_for_field, TerminalCommittedGroupParams};

pub(in crate::protocol::core) struct PreparedProtocolPoint<F: FieldCore, E: FieldCore> {
    pub(in crate::protocol::core) prepared: PreparedOpeningPoint<F, E>,
    pub(in crate::protocol::core) protocol: Vec<E>,
}

pub(in crate::protocol::core) struct FoldEorReplay<F: FieldCore, E: FieldCore> {
    pub(in crate::protocol::core) groups: Vec<PreparedProtocolPoint<F, E>>,
    pub(in crate::protocol::core) final_relation: Option<(Vec<E>, Vec<E>)>,
}

#[derive(Clone, Copy)]
struct EorReductionShape {
    split_bits: usize,
    width: usize,
    num_rounds: usize,
}

struct EorSumcheckReplay<E: FieldCore> {
    rho: Vec<E>,
    final_claims: Vec<E>,
    final_factors: Vec<E>,
}

fn eor_reduction_shape<F, E>(
    opening_num_vars: usize,
    partials_len: usize,
    num_claims: usize,
) -> Result<EorReductionShape, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let (split_bits, width) =
        tensor_opening_split::<F, E>().map_err(|_| AkitaError::InvalidProof)?;
    let num_rounds = opening_num_vars
        .checked_sub(split_bits)
        .ok_or(AkitaError::InvalidProof)?;
    let expected_partials = width
        .checked_mul(num_claims)
        .ok_or(AkitaError::InvalidProof)?;
    if width == 1 || partials_len != expected_partials {
        return Err(AkitaError::InvalidProof);
    }
    Ok(EorReductionShape {
        split_bits,
        width,
        num_rounds,
    })
}

fn eor_input_claims_from_partials<F, E>(
    partials: &[E],
    shape: EorReductionShape,
    eta: &[E],
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    if shape.width == 0 || !partials.len().is_multiple_of(shape.width) {
        return Err(AkitaError::InvalidProof);
    }
    partials
        .chunks_exact(shape.width)
        .map(|partials| {
            let row_partials = tensor_row_partials_from_columns::<F, E>(partials)?;
            tensor_reduction_claim_from_rows::<F, E>(&row_partials, eta)
        })
        .collect()
}

fn verify_eor_sumcheck<F, E, T>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    group_points: &[&[E]],
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    requires_reduction: bool,
    transcript: &mut T,
) -> Result<Option<EorSumcheckReplay<E>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    let num_claims = opening_batch.num_total_polynomials();
    if openings.len() != num_claims || group_points.len() != opening_batch.num_groups() {
        return Err(AkitaError::InvalidProof);
    }
    // Exact presence: the per-level predicate is the sole authority, so a
    // missing required payload and an unsolicited payload both fail closed.
    if extension_opening_reduction.is_some() != requires_reduction {
        return Err(AkitaError::InvalidProof);
    }
    let Some(reduction) = extension_opening_reduction else {
        return Ok(None);
    };
    let shape = eor_reduction_shape::<F, E>(
        opening_batch.max_num_vars(),
        reduction.partials.len(),
        num_claims,
    )?;
    let mut claim_offset = 0usize;
    for (group_index, group_point) in group_points.iter().enumerate() {
        let group_layout = opening_batch.group_layout(group_index)?;
        if group_point.len() != group_layout.num_vars() || group_point.len() < shape.split_bits {
            return Err(AkitaError::InvalidProof);
        }
        for opening in openings
            .get(claim_offset..)
            .ok_or(AkitaError::InvalidProof)?
            .iter()
            .take(group_layout.num_polynomials())
        {
            let partial_start = claim_offset
                .checked_mul(shape.width)
                .ok_or(AkitaError::InvalidProof)?;
            let partial_end = partial_start
                .checked_add(shape.width)
                .ok_or(AkitaError::InvalidProof)?;
            let partials = reduction
                .partials
                .get(partial_start..partial_end)
                .ok_or(AkitaError::InvalidProof)?;
            let expected =
                derive_tensor_extension_opening_claim_from_partials::<F, E>(group_point, partials)?;
            if expected != *opening {
                return Err(AkitaError::InvalidProof);
            }
            for partial in partials {
                append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
            }
            claim_offset = claim_offset
                .checked_add(1)
                .ok_or(AkitaError::InvalidProof)?;
        }
    }
    if claim_offset != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let eta = (0..shape.split_bits)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
        .collect::<Vec<_>>();
    let input_claims = eor_input_claims_from_partials::<F, E>(&reduction.partials, shape, &eta)?;
    if input_claims.len() != num_claims || reduction.final_claims.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let claim_coefficients =
        sample_row_coefficients::<F, E, T>(opening_batch, CHALLENGE_EOR_CLAIM_BATCH, transcript)?;
    let batched_input_claim = input_claims
        .iter()
        .zip(&claim_coefficients)
        .fold(E::zero(), |acc, (&claim, &coefficient)| {
            acc + coefficient * claim
        });
    let (batched_final_claim, rho) = verify_extension_opening_reduction_sumcheck::<F, T, E, _>(
        batched_input_claim,
        shape.num_rounds,
        &reduction.sumcheck,
        transcript,
        |tr| sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND),
    )?;
    let expected_batched_final = reduction
        .final_claims
        .iter()
        .zip(&claim_coefficients)
        .fold(E::zero(), |acc, (&claim, &coefficient)| {
            acc + coefficient * claim
        });
    if batched_final_claim != expected_batched_final {
        return Err(AkitaError::InvalidProof);
    }
    for final_claim in &reduction.final_claims {
        append_ext_field::<F, E, T>(transcript, ABSORB_EOR_FINAL_CLAIM, final_claim);
    }
    let mut final_factors = Vec::with_capacity(group_points.len());
    for group_point in group_points {
        let tail_point = group_point
            .get(shape.split_bits..)
            .ok_or(AkitaError::InvalidProof)?;
        let local_rho = rho
            .get(..tail_point.len())
            .ok_or(AkitaError::InvalidProof)?;
        let mut factor = tensor_equality_factor_eval_at_point::<F, E>(tail_point, &eta, local_rho)?;
        for &extra_challenge in rho
            .get(tail_point.len()..)
            .ok_or(AkitaError::InvalidProof)?
        {
            factor *= E::one() - extra_challenge;
        }
        final_factors.push(factor);
    }
    Ok(Some(EorSumcheckReplay {
        rho,
        final_claims: reduction.final_claims.clone(),
        final_factors,
    }))
}

/// Verify the terminal fold's single-group extension-opening reduction.
///
/// Terminal proofs carry their geometry directly rather than through
/// `CommittedGroupParams`, so their replay remains an explicit terminal
/// boundary instead of being disguised as an ordinary committed-group fold.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn verify_terminal_fold_eor<F, E, T>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    challenge_point: &[E],
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    d_a: usize,
    num_positions_per_block: usize,
    num_live_blocks: usize,
    requires_reduction: bool,
    transcript: &mut T,
) -> Result<FoldEorReplay<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
        verify_terminal_fold_eor_kernel::<F, E, T, D>(
            extension_opening_reduction,
            challenge_point,
            openings,
            opening_batch,
            basis,
            num_positions_per_block,
            num_live_blocks,
            d_a.trailing_zeros() as usize,
            requires_reduction,
            transcript,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_terminal_fold_eor_kernel<F, E, T, const D: usize>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    challenge_point: &[E],
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    num_positions_per_block: usize,
    num_live_blocks: usize,
    alpha_bits: usize,
    requires_reduction: bool,
    transcript: &mut T,
) -> Result<FoldEorReplay<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    if challenge_point.len() > opening_batch.max_num_vars() || opening_batch.num_groups() != 1 {
        return Err(AkitaError::InvalidProof);
    }
    let mut eor_point = challenge_point.to_vec();
    eor_point.resize(opening_batch.max_num_vars(), E::zero());
    let replay = verify_eor_sumcheck::<F, E, T>(
        extension_opening_reduction,
        &[eor_point.as_slice()],
        openings,
        opening_batch,
        requires_reduction,
        transcript,
    )?;
    let groups = if let Some(replay) = &replay {
        let protocol_point =
            ring_subfield_packed_extension_opening_point::<F, E, D>(replay.rho.len(), &replay.rho)?;
        let prepared = prepare_opening_point::<F, E, D>(
            &protocol_point,
            basis,
            num_positions_per_block,
            num_live_blocks,
            alpha_bits,
        )?;
        vec![PreparedProtocolPoint {
            prepared,
            protocol: protocol_point,
        }]
    } else {
        Vec::new()
    };
    Ok(FoldEorReplay {
        groups,
        final_relation: replay.map(|replay| (replay.final_claims, replay.final_factors)),
    })
}

/// Verify one fold's extension-opening reduction over all opening groups.
///
/// Every group retains its native opening point and committed geometry. The
/// groups share one challenge sequence while each claim keeps its own
/// sumcheck recurrence.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn verify_fold_eor<F, E, T>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    group_points: &[&[E]],
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    lp: &CommittedGroupParams,
    requires_reduction: bool,
    transcript: &mut T,
) -> Result<FoldEorReplay<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let replay = verify_eor_sumcheck::<F, E, T>(
        extension_opening_reduction,
        group_points,
        openings,
        opening_batch,
        requires_reduction,
        transcript,
    )?;
    let mut groups = Vec::new();
    if let Some(replay) = &replay {
        groups.reserve(group_points.len());
        for (group_index, group_point) in group_points.iter().enumerate() {
            let group_lp = lp.group_params(opening_batch, group_index)?;
            let group_dims = lp.group_role_dims(opening_batch, group_index)?;
            let alpha_bits = group_dims.d_a().trailing_zeros() as usize;
            let tail_vars = group_point
                .len()
                .checked_sub(tensor_opening_split::<F, E>()?.0)
                .ok_or(AkitaError::InvalidProof)?;
            let local_rho = replay
                .rho
                .get(..tail_vars)
                .ok_or(AkitaError::InvalidProof)?;
            let (prepared, protocol_point) = dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Inner),
                F,
                group_dims.d_a(),
                |D| {
                    let protocol_point = ring_subfield_packed_extension_opening_point::<F, E, D>(
                        local_rho.len(),
                        local_rho,
                    )?;
                    let prepared = prepare_opening_point::<F, E, D>(
                        &protocol_point,
                        basis,
                        group_lp.num_positions_per_block(),
                        group_lp.num_live_blocks(),
                        alpha_bits,
                    )?;
                    Ok::<_, AkitaError>((prepared, protocol_point))
                }
            )?;
            groups.push(PreparedProtocolPoint {
                prepared,
                protocol: protocol_point,
            });
        }
    }
    Ok(FoldEorReplay {
        groups,
        final_relation: replay.map(|replay| (replay.final_claims, replay.final_factors)),
    })
}

/// Terminal-suffix extension-claim prefix: one batched EOR replay over the
/// recursive opening group, then the prepared points are absorbed.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn verify_extension_claim_terminal_suffix<F, E, T>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    protocol_point: &[E],
    opening: &E,
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    params: &TerminalCommittedGroupParams,
    transcript: &mut T,
) -> Result<FoldEorReplay<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    append_claim_values_to_transcript::<F, E, T>(std::slice::from_ref(opening), transcript);
    let FoldEorReplay {
        groups,
        final_relation,
    } = verify_terminal_fold_eor::<F, E, T>(
        extension_opening_reduction,
        protocol_point,
        std::slice::from_ref(opening),
        opening_batch,
        basis,
        params.d_a(),
        params.num_positions_per_block,
        params.num_live_blocks,
        E::EXT_DEGREE > 1,
        transcript,
    )
    .map_err(|error| {
        AkitaError::InvalidInput(format!(
            "terminal extension-opening replay failed: {error:?}"
        ))
    })?;
    let protocol_point_refs = groups
        .iter()
        .map(|group| group.protocol.as_slice())
        .collect::<Vec<_>>();
    absorb_protocol_opening_points(&protocol_point_refs, transcript);
    Ok(FoldEorReplay {
        groups,
        final_relation,
    })
}

/// Recursive-suffix extension-claim prefix: one batched EOR replay over the
/// suffix opening groups. Application claim batching remains pending
/// until the opening payload is absorbed.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn verify_extension_claim_suffix_prefix<F, E, T>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    group_points: &[&[E]],
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    lp: &CommittedGroupParams,
    transcript: &mut T,
) -> Result<FoldClaimMaterial<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let FoldEorReplay {
        groups,
        final_relation,
    } = verify_fold_eor::<F, E, T>(
        extension_opening_reduction,
        group_points,
        openings,
        opening_batch,
        basis,
        lp,
        E::EXT_DEGREE > 1,
        transcript,
    )?;
    let protocol_point_refs = groups
        .iter()
        .map(|group| group.protocol.as_slice())
        .collect::<Vec<_>>();
    absorb_protocol_opening_points(&protocol_point_refs, transcript);
    let (final_claims, factors_by_group) = final_relation.ok_or(AkitaError::InvalidProof)?;
    Ok(FoldClaimMaterial {
        prepared_points: groups
            .into_iter()
            .map(|group| PreparedFoldOpeningPoint::EvaluationTrace(group.prepared))
            .collect(),
        openings: openings.to_vec(),
        reduction_final_claims: Some(final_claims),
        reduction_factors: Some(factors_by_group),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::CompressedUniPoly;
    use akita_field::{FpExt4, Prime32Offset99};
    use akita_sumcheck::SumcheckProof;
    use akita_transcript::AkitaTranscript;
    use akita_types::{PolynomialGroupLayout, EXTENSION_OPENING_REDUCTION_DEGREE};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    fn extension_point(num_vars: usize, offset: u64) -> Vec<E> {
        (0..num_vars)
            .map(|index| {
                let value = offset + index as u64;
                E::from_base_slice(&[
                    F::from_u64(value + 1),
                    F::from_u64(value + 2),
                    F::from_u64(value + 3),
                    F::from_u64(value + 4),
                ])
            })
            .collect()
    }

    #[test]
    fn recursive_setup_prefix_groups_share_one_max_tail_eor_and_reject_tampering() {
        const SETUP_PREFIX_VARS: usize = 12;
        const WITNESS_VARS: usize = 20;
        let opening_batch = OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::singleton(SETUP_PREFIX_VARS),
            PolynomialGroupLayout::singleton(WITNESS_VARS),
        ])
        .expect("recursive setup-prefix opening layout");
        let group_points = [
            extension_point(SETUP_PREFIX_VARS, 10),
            extension_point(WITNESS_VARS, 100),
        ];
        let group_point_refs = [group_points[0].as_slice(), group_points[1].as_slice()];
        let openings = vec![E::zero(); opening_batch.num_total_polynomials()];
        let (split_bits, width) = tensor_opening_split::<F, E>().expect("tensor split");
        let max_tail_rounds = WITNESS_VARS - split_bits;
        let reduction = ExtensionOpeningReductionProof {
            partials: vec![E::zero(); width * opening_batch.num_total_polynomials()],
            sumcheck: SumcheckProof {
                round_polys: (0..max_tail_rounds)
                    .map(|_| CompressedUniPoly {
                        coeffs_except_linear_term: vec![
                            E::zero();
                            EXTENSION_OPENING_REDUCTION_DEGREE
                        ],
                    })
                    .collect(),
            },
            final_claims: vec![E::zero(); opening_batch.num_total_polynomials()],
        };

        let mut transcript = AkitaTranscript::<F>::new(b"test/recursive-setup-prefix-grouped-eor");
        let replay = verify_eor_sumcheck::<F, E, _>(
            Some(&reduction),
            &group_point_refs,
            &openings,
            &opening_batch,
            true,
            &mut transcript,
        )
        .expect("recursive setup-prefix grouped EOR")
        .expect("extension claims require EOR");
        assert_eq!(replay.rho.len(), max_tail_rounds);
        assert_eq!(replay.final_factors.len(), 2);

        let mut missing_final_claim = reduction.clone();
        missing_final_claim.final_claims.pop();
        let mut missing_claim_transcript =
            AkitaTranscript::<F>::new(b"test/recursive-setup-prefix-grouped-eor");
        let missing_claim_result = verify_eor_sumcheck::<F, E, _>(
            Some(&missing_final_claim),
            &group_point_refs,
            &openings,
            &opening_batch,
            true,
            &mut missing_claim_transcript,
        );
        assert!(
            matches!(missing_claim_result, Err(AkitaError::InvalidProof)),
            "the explicit terminal vector must contain every opening claim"
        );

        let mut tampered = reduction.clone();
        tampered.partials[0] += E::one();
        let mut tampered_transcript =
            AkitaTranscript::<F>::new(b"test/recursive-setup-prefix-grouped-eor");
        let tampered_result = verify_eor_sumcheck::<F, E, _>(
            Some(&tampered),
            &group_point_refs,
            &openings,
            &opening_batch,
            true,
            &mut tampered_transcript,
        );
        assert!(
            matches!(tampered_result, Err(AkitaError::InvalidProof)),
            "tampered setup-prefix EOR partial must reject"
        );

        let mut missing_transcript =
            AkitaTranscript::<F>::new(b"test/recursive-setup-prefix-grouped-eor");
        let missing_result = verify_eor_sumcheck::<F, E, _>(
            None,
            &group_point_refs,
            &openings,
            &opening_batch,
            true,
            &mut missing_transcript,
        );
        assert!(
            matches!(missing_result, Err(AkitaError::InvalidProof)),
            "missing setup-prefix EOR must reject for extension claims"
        );
    }

    #[test]
    fn eor_presence_must_match_predicate_exactly() {
        const NUM_VARS: usize = 12;
        let opening_batch =
            OpeningClaimsLayout::from_groups(vec![PolynomialGroupLayout::singleton(NUM_VARS)])
                .expect("single-group opening layout");
        let group_point = extension_point(NUM_VARS, 10);
        let group_point_refs = [group_point.as_slice()];
        let openings = vec![E::zero(); opening_batch.num_total_polynomials()];
        let (split_bits, width) = tensor_opening_split::<F, E>().expect("tensor split");
        let reduction = ExtensionOpeningReductionProof {
            partials: vec![E::zero(); width * opening_batch.num_total_polynomials()],
            sumcheck: SumcheckProof {
                round_polys: (0..NUM_VARS - split_bits)
                    .map(|_| CompressedUniPoly {
                        coeffs_except_linear_term: vec![
                            E::zero();
                            EXTENSION_OPENING_REDUCTION_DEGREE
                        ],
                    })
                    .collect(),
            },
            final_claims: vec![E::zero()],
        };

        // Honest gate-off level: no payload, predicate off, replay is a no-op.
        let mut idle_transcript = AkitaTranscript::<F>::new(b"test/eor-exact-presence");
        let idle_replay = verify_eor_sumcheck::<F, E, _>(
            None,
            &group_point_refs,
            &openings,
            &opening_batch,
            false,
            &mut idle_transcript,
        )
        .expect("gate-off level without payload must verify");
        assert!(idle_replay.is_none());

        // Unsolicited payload at a gate-off level must fail closed.
        let mut unsolicited_transcript = AkitaTranscript::<F>::new(b"test/eor-exact-presence");
        let unsolicited_result = verify_eor_sumcheck::<F, E, _>(
            Some(&reduction),
            &group_point_refs,
            &openings,
            &opening_batch,
            false,
            &mut unsolicited_transcript,
        );
        assert!(
            matches!(unsolicited_result, Err(AkitaError::InvalidProof)),
            "unsolicited EOR at a gate-off level must reject"
        );
    }
}
