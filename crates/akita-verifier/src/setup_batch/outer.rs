use super::*;
use crate::stages::ring_switch::RingSwitchVerifyOutput;
use crate::SetupClaimAccumulator;
use akita_params::CommittedGroupParams;
use akita_types::{setup_claims::transcript, AkitaVerifierSetup, GrindingReplay};

/// Transcript-bound setup statement shared by the two outer stage checks.
pub(crate) struct PreparedOuterSetup<E: Field> {
    pub(crate) statement: SetupBatchStatement<E>,
    pub(crate) coefficients: Vec<E>,
    pub(crate) claim: E,
    pub(crate) relation_weight: E,
}

/// Consume the inherited batch and bind its combination with this fold's query.
pub(crate) fn prepare_outer_setup<F, E>(
    setup: &AkitaVerifierSetup<F>,
    rs: &RingSwitchVerifyOutput<E>,
    point: &[E],
    successor: &CommittedGroupParams,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    level: u32,
    accumulator: &mut SetupClaimAccumulator<'_, E>,
) -> Result<PreparedOuterSetup<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F> + Ring + MulBaseUnreduced<F>,
{
    let (target, inherited) = accumulator.batch.take().ok_or(AkitaError::InvalidProof)?;
    if target != level || inherited.polynomial() != &accumulator.polynomial {
        return Err(AkitaError::InvalidInput(
            "setup batch target or domain mismatch".into(),
        ));
    }
    inherited.polynomial().validate::<F, E>(setup.expanded())?;
    let id = successor
        .setup_prefix()
        .and_then(|prefix| prefix.slot_id())
        .ok_or(AkitaError::InvalidProof)?;
    let slot = setup
        .prefix_slots()
        .get(&id)
        .ok_or(AkitaError::InvalidProof)?;
    transcript::bind_outer_prefix(&inherited, slot, successor, grinding, level)?;
    let claim = akita_types::stage3_claim::<F, E, _>(grinding, level, E::zero())?;
    let mut claims = inherited.claims().to_vec();
    let relation_weight = rs
        .relation_matrix_evaluator
        .eval_flat_at_point_with_external_setup::<F>(point, rs.alpha, |query| {
            claims.push(akita_types::setup_claims::SetupClaim::new(
                accumulator.polynomial.clone(),
                query,
                claim,
            )?);
            Ok(claim)
        })?;
    let statement = SetupBatchStatement::new(
        inherited.polynomial().clone(),
        inherited.coefficient_len(),
        claims,
    )?;
    transcript::bind_outer_statement::<F, E, _>(grinding.state_mut(), level, &statement)?;
    let coefficients =
        batch_coefficients::<F, E, _>(grinding.state_mut(), statement.claims().len())?;
    Ok(PreparedOuterSetup {
        statement,
        coefficients,
        claim,
        relation_weight,
    })
}

/// One setup traversal for the weighted collection. Only factor tables are retained.
pub(crate) fn decide_setup_batch<F, E>(
    setup: &AkitaExpandedSetup<F>,
    statement: &SetupBatchStatement<E>,
    coefficients: &[E],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F> + Ring + MulBaseUnreduced<F>,
{
    statement.polynomial().validate::<F, E>(setup)?;
    let expected = statement.input_claim(coefficients)?;
    let mut terms = Vec::new();
    for (claim, &scale) in statement.claims().iter().zip(coefficients) {
        for term in claim
            .query()
            .compile(statement.polynomial(), statement.coefficient_len())?
        {
            terms.push((scale, term));
        }
    }
    let source = setup
        .shared_matrix()
        .as_field_slice()
        .get(..statement.polynomial().field_len())
        .ok_or(AkitaError::InvalidProof)?;
    let mut value = E::zero();
    for (row, chunk) in source.chunks_exact(statement.coefficient_len()).enumerate() {
        for (column, &base) in chunk.iter().enumerate() {
            let weight = terms.iter().try_fold(E::zero(), |sum, (scale, term)| {
                Ok::<_, AkitaError>(
                    sum + *scale
                        * term.index_weight(row)
                        * *term
                            .coefficient()
                            .get(column)
                            .ok_or(AkitaError::InvalidProof)?,
                )
            })?;
            value += E::reduce_product(weight.mul_base_unreduced(base));
        }
    }
    if value != expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::{
        setup_claims::{SetupClaim, SetupQuery},
        AkitaSetupDescriptor,
    };
    use jolt_field::{One, Prime128OffsetA7F7 as F};

    #[test]
    fn direct_batch_rejects_false_answers_that_cancel_without_weights() {
        let setup = AkitaExpandedSetup::<F>::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupDescriptor {
                max_num_vars: 6,
                max_num_batched_polys: 1,
                num_field_elements: 64,
                setup_seed: [19; 32].into(),
            },
            akita_types::derive_public_matrix_prefix(64, &[19; 32].into()),
        );
        let domain = SetupPolynomialId::new::<F, F>(&setup, 64).unwrap();
        let claims = [7, 11]
            .into_iter()
            .map(|coordinate| {
                let point = vec![F::from_u64(coordinate); 6];
                let value = akita_algebra::poly::multilinear_eval(
                    setup.shared_matrix().as_field_slice(),
                    &point,
                )
                .unwrap();
                SetupClaim::new(
                    domain.clone(),
                    SetupQuery::evaluation(point).unwrap(),
                    value,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let statement = SetupBatchStatement::new(domain.clone(), 8, claims.clone()).unwrap();
        let weights = [F::one(), F::from_u64(2)];
        decide_setup_batch(&setup, &statement, &weights).unwrap();
        let wrong = claims
            .into_iter()
            .zip([F::one(), -F::one()])
            .map(|(claim, delta)| {
                SetupClaim::new(domain.clone(), claim.query().clone(), claim.value() + delta)
                    .unwrap()
            })
            .collect();
        let wrong = SetupBatchStatement::new(domain, 8, wrong).unwrap();
        assert!(decide_setup_batch(&setup, &wrong, &weights).is_err());
    }
}
