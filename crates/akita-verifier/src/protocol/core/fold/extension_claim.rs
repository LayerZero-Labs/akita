//! Extension-claim fold verifier prefix: extension-opening reduction replay.

use super::super::*;
use super::{FoldClaimMaterial, PreparedFoldOpeningPoint};
use akita_types::{dispatch_for_field, TerminalFoldParams};

pub(in crate::protocol::core) struct PreparedProtocolPoint<F: Field, E: Field> {
    pub(in crate::protocol::core) prepared: PreparedOpeningPoint<F, E>,
    pub(in crate::protocol::core) protocol: Vec<E>,
}

pub(in crate::protocol::core) struct FoldEorReplay<F: Field, E: Field> {
    pub(in crate::protocol::core) groups: Vec<PreparedProtocolPoint<F, E>>,
    pub(in crate::protocol::core) final_relation: Option<(Vec<E>, Vec<E>)>,
}

#[derive(Clone, Copy)]
struct EorReductionShape {
    split_bits: usize,
    width: usize,
    num_rounds: usize,
}

struct EorSumcheckReplay<E: Field> {
    rho: Vec<E>,
    final_claims: Vec<E>,
    final_factors: Vec<E>,
}

struct EorPrefix<E: Field> {
    partials: Vec<E>,
    eta: Vec<E>,
    claim_coefficients: Vec<E>,
}

trait EorVerifierStream<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    fn prefix(
        &mut self,
        opening_batch: &OpeningClaimsLayout,
        openings: &[E],
        partial_count: usize,
        split_bits: usize,
    ) -> Result<EorPrefix<E>, AkitaError>;

    fn verify_sumcheck(
        &mut self,
        input_claim: E,
        num_rounds: usize,
    ) -> Result<(E, Vec<E>), AkitaError>;

    fn final_claims(&mut self, opening_batch: &OpeningClaimsLayout) -> Result<Vec<E>, AkitaError>;
}
struct NativeEorVerifierStream<'a, 'proof, 'plan> {
    grinding: &'a mut akita_types::NativeVerifierGrinding<'proof, 'plan>,
    level: u32,
}

impl<F, E> EorVerifierStream<F, E> for NativeEorVerifierStream<'_, '_, '_>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    fn prefix(
        &mut self,
        opening_batch: &OpeningClaimsLayout,
        openings: &[E],
        partial_count: usize,
        split_bits: usize,
    ) -> Result<EorPrefix<E>, AkitaError> {
        let prefix = akita_types::native_eor_verifier_prefix::<F, E>(
            self.grinding,
            opening_batch,
            openings,
            self.level,
        )?;
        if prefix.partials.len() != partial_count || prefix.eta.len() != split_bits {
            return Err(AkitaError::InvalidProof);
        }
        Ok(EorPrefix {
            partials: prefix.partials,
            eta: prefix.eta,
            claim_coefficients: prefix.claim_coefficients,
        })
    }

    fn verify_sumcheck(
        &mut self,
        input_claim: E,
        num_rounds: usize,
    ) -> Result<(E, Vec<E>), AkitaError> {
        let mut channel = akita_types::NativeGrindingSumcheckVerifier::<F, E>::new(
            self.grinding,
            akita_types::SumcheckProtocol::ExtensionOpeningReduction,
            self.level,
            0,
        );
        let replay = akita_sumcheck::verify_sumcheck_rounds_native::<F, E, _>(
            &mut channel,
            akita_types::NATIVE_EOR_SUMCHECK_INVOCATION,
            input_claim,
            num_rounds,
            akita_types::EXTENSION_OPENING_REDUCTION_DEGREE,
        )?;
        Ok((replay.output_claim, replay.challenges))
    }

    fn final_claims(&mut self, opening_batch: &OpeningClaimsLayout) -> Result<Vec<E>, AkitaError> {
        akita_types::native_eor_verifier_final_claims::<F, E>(
            self.grinding,
            opening_batch,
            self.level,
        )
    }
}

fn eor_reduction_shape<F, E>(
    opening_batch: &OpeningClaimsLayout,
) -> Result<EorReductionShape, AkitaError>
where
    F: Field,
    E: ExtField<F>,
{
    let (split_bits, width) =
        tensor_opening_split::<F, E>().map_err(|_| AkitaError::InvalidProof)?;
    let expected = canonical_extension_opening_reduction_shape(opening_batch, width)
        .map_err(|_| AkitaError::InvalidProof)?;
    if width == 1 {
        return Err(AkitaError::InvalidProof);
    }
    Ok(EorReductionShape {
        split_bits,
        width,
        num_rounds: expected.sumcheck.len(),
    })
}

fn eor_input_claims_from_partials<F, E>(
    partials: &[E],
    shape: EorReductionShape,
    eta: &[E],
) -> Result<Vec<E>, AkitaError>
where
    F: Field,
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
fn verify_eor_sumcheck_native<F, E>(
    group_points: &[&[E]],
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    requires_reduction: bool,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    level: u32,
) -> Result<Option<EorSumcheckReplay<E>>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    if !requires_reduction {
        return Ok(None);
    }
    let shape = eor_reduction_shape::<F, E>(opening_batch)?;
    let mut stream = NativeEorVerifierStream { grinding, level };
    verify_eor_sumcheck_with_stream::<F, E, _>(
        group_points,
        openings,
        opening_batch,
        shape,
        &mut stream,
    )
    .map(Some)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn verify_extension_claim_suffix_prefix_native<F, E>(
    group_points: &[&[E]],
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    lp: &CommittedGroupParams,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    level: u32,
) -> Result<FoldClaimMaterial<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize,
{
    let replay = verify_eor_sumcheck_native::<F, E>(
        group_points,
        openings,
        opening_batch,
        E::DEGREE > 1,
        grinding,
        level,
    )?
    .ok_or(AkitaError::InvalidProof)?;
    let mut prepared_points = Vec::with_capacity(group_points.len());
    let mut protocol_points = Vec::with_capacity(group_points.len());
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
        prepared_points.push(PreparedFoldOpeningPoint::EvaluationTrace(prepared));
        protocol_points.push(protocol_point);
    }
    for (group_index, protocol_point) in protocol_points.iter().enumerate() {
        akita_transcript::public_native_extensions_verifier::<F, E>(
            grinding.state_mut(),
            akita_transcript::ProtocolSiteId {
                family: akita_transcript::SITE_FAMILY_FOLD_BINDING,
                level,
                stage: 1,
                group: u32::try_from(group_index).map_err(|_| AkitaError::InvalidProof)?,
                ..akita_transcript::ProtocolSiteId::default()
            },
            protocol_point,
        )
        .map_err(|_| AkitaError::InvalidProof)?;
    }
    Ok(FoldClaimMaterial {
        prepared_points,
        openings: openings.to_vec(),
        reduction_final_claims: Some(replay.final_claims),
        reduction_factors: Some(replay.final_factors),
    })
}

pub(in crate::protocol::core) fn verify_extension_claim_terminal_suffix_native<F, E>(
    opening_point: &[E],
    opening: E,
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    params: &TerminalFoldParams,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    level: u32,
) -> Result<FoldEorReplay<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize,
{
    let replay = verify_eor_sumcheck_native::<F, E>(
        &[opening_point],
        &[opening],
        opening_batch,
        E::DEGREE > 1,
        grinding,
        level,
    )?
    .ok_or(AkitaError::InvalidProof)?;
    let protocol_point = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        params.d_a(),
        |D| {
            ring_subfield_packed_extension_opening_point::<F, E, D>(replay.rho.len(), &replay.rho)
        }
    )?;
    let prepared = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        params.d_a(),
        |D| {
            prepare_opening_point::<F, E, D>(
                &protocol_point,
                basis,
                params.blocks.positions_per_block,
                params.blocks.live_blocks,
                params.d_a().trailing_zeros() as usize,
            )
        }
    )?;
    Ok(FoldEorReplay {
        groups: vec![PreparedProtocolPoint {
            prepared,
            protocol: protocol_point,
        }],
        final_relation: Some((replay.final_claims, replay.final_factors)),
    })
}

fn verify_eor_sumcheck_with_stream<F, E, S>(
    group_points: &[&[E]],
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    shape: EorReductionShape,
    stream: &mut S,
) -> Result<EorSumcheckReplay<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    S: EorVerifierStream<F, E>,
{
    let num_claims = opening_batch.num_total_polynomials();
    if openings.len() != num_claims || group_points.len() != opening_batch.num_groups() {
        return Err(AkitaError::InvalidProof);
    }
    let partial_count = shape
        .width
        .checked_mul(num_claims)
        .ok_or(AkitaError::InvalidProof)?;
    let EorPrefix {
        partials,
        eta,
        claim_coefficients,
    } = stream.prefix(opening_batch, openings, partial_count, shape.split_bits)?;
    if partials.len() != partial_count
        || eta.len() != shape.split_bits
        || claim_coefficients.len() != num_claims
    {
        return Err(AkitaError::InvalidProof);
    }
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
            let claim_partials = partials
                .get(partial_start..partial_end)
                .ok_or(AkitaError::InvalidProof)?;
            let expected = derive_tensor_extension_opening_claim_from_partials::<F, E>(
                group_point,
                claim_partials,
            )?;
            if expected != *opening {
                return Err(AkitaError::InvalidProof);
            }
            claim_offset = claim_offset
                .checked_add(1)
                .ok_or(AkitaError::InvalidProof)?;
        }
    }
    if claim_offset != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let input_claims = eor_input_claims_from_partials::<F, E>(&partials, shape, &eta)?;
    if input_claims.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let batched_input_claim = input_claims
        .iter()
        .zip(&claim_coefficients)
        .fold(E::zero(), |acc, (&claim, &coefficient)| {
            acc + coefficient * claim
        });
    let (batched_final_claim, rho) =
        stream.verify_sumcheck(batched_input_claim, shape.num_rounds)?;
    let final_claims = stream.final_claims(opening_batch)?;
    if final_claims.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let expected_batched_final = final_claims
        .iter()
        .zip(&claim_coefficients)
        .fold(E::zero(), |acc, (&claim, &coefficient)| {
            acc + coefficient * claim
        });
    if batched_final_claim != expected_batched_final {
        return Err(AkitaError::InvalidProof);
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
    Ok(EorSumcheckReplay {
        rho,
        final_claims,
        final_factors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use akita_sumcheck::{SumcheckInstanceProver, UniPoly};
    use akita_transcript::{new_native_prover, new_native_verifier};
    use akita_types::{PolynomialGroupLayout, EXTENSION_OPENING_REDUCTION_DEGREE};
    use jolt_field::{FpExt4, Prime32Offset99, Zero};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    struct ZeroEorProver {
        rounds: usize,
    }

    impl SumcheckInstanceProver<E> for ZeroEorProver {
        fn num_rounds(&self) -> usize {
            self.rounds
        }

        fn degree_bound(&self) -> usize {
            EXTENSION_OPENING_REDUCTION_DEGREE
        }

        fn input_claim(&self) -> E {
            E::zero()
        }

        fn compute_round_univariate(&mut self, _round: usize, _claim: E) -> UniPoly<E> {
            UniPoly::from_coeffs(vec![E::zero(); EXTENSION_OPENING_REDUCTION_DEGREE + 1])
        }

        fn ingest_challenge(&mut self, _round: usize, _challenge: E) {}
    }

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
    fn native_eor_verifier_replays_stream_and_checks_eof() {
        const NUM_VARS: usize = 12;
        let level = 1;
        let opening_batch =
            OpeningClaimsLayout::from_groups(vec![PolynomialGroupLayout::singleton(NUM_VARS)])
                .unwrap();
        let group_point = extension_point(NUM_VARS, 10);
        let group_points = [group_point.as_slice()];
        let openings = vec![E::zero()];
        let (split_bits, width) = tensor_opening_split::<F, E>().unwrap();
        let rounds = NUM_VARS - split_bits;
        let partials = vec![E::zero(); width];
        let plan = {
            let mut runs = vec![akita_types::GrindingRun::proof_of_work(
                akita_types::GrindingSite::ExtensionOpeningPoint { level },
                1,
                128,
            )
            .unwrap()];
            for round in 0..rounds {
                runs.push(
                    akita_types::GrindingRun::proof_of_work(
                        akita_types::GrindingSite::SumcheckRound {
                            protocol: akita_types::SumcheckProtocol::ExtensionOpeningReduction,
                            level,
                            stage: 0,
                            round: u32::try_from(round).unwrap(),
                        },
                        1,
                        128,
                    )
                    .unwrap(),
                );
            }
            akita_types::GrindingPlan::new(runs, 128).unwrap()
        };

        let state = new_native_prover(b"native-eor-verifier", b"fixture").unwrap();
        let mut prover = akita_types::NativeProverGrinding::new(state, &plan);
        akita_types::native_eor_prover_prefix::<F, E>(
            &mut prover,
            &opening_batch,
            &openings,
            &partials,
            level,
        )
        .unwrap();
        let mut sumcheck = ZeroEorProver { rounds };
        let mut channel = akita_types::NativeGrindingSumcheckProver::<F, E>::new(
            &mut prover,
            akita_types::SumcheckProtocol::ExtensionOpeningReduction,
            level,
            0,
        );
        akita_sumcheck::prove_sumcheck_native::<F, E, _, _>(
            &mut sumcheck,
            &mut channel,
            akita_types::NATIVE_EOR_SUMCHECK_INVOCATION,
        )
        .unwrap();
        akita_types::native_eor_prover_final_claims::<F, E>(
            &mut prover,
            &opening_batch,
            &[E::zero()],
            level,
        )
        .unwrap();
        let proof = prover.finish().unwrap();

        let state = new_native_verifier(b"native-eor-verifier", b"fixture", &proof).unwrap();
        let mut verifier = akita_types::NativeVerifierGrinding::new(state, &plan);
        let replay = verify_eor_sumcheck_native::<F, E>(
            &group_points,
            &openings,
            &opening_batch,
            true,
            &mut verifier,
            level,
        )
        .unwrap()
        .unwrap();
        assert_eq!(replay.rho.len(), rounds);
        assert_eq!(replay.final_claims, vec![E::zero()]);
        verifier.finish().unwrap();
    }
}
