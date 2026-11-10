use super::*;
use crate::stages::ring_switch::RingSwitchVerifyOutput;
use akita_params::CommittedGroupParams;
use akita_types::{
    setup_claims::{transcript, SetupWeightTerm},
    AkitaVerifierSetup, GrindingReplay,
};
use std::collections::BTreeMap;

/// Transcript-bound setup statement shared by the two outer stage checks.
pub(crate) struct PreparedOuterSetup<E: Field> {
    pub(crate) statement: SetupBatchStatement<E>,
    pub(crate) coefficients: Vec<E>,
    pub(crate) claim: E,
    pub(crate) relation_weight: E,
}

/// Bind the inherited batch together with this fold's query.
pub(crate) fn prepare_outer_setup<F, E>(
    setup: &AkitaVerifierSetup<F>,
    rs: &RingSwitchVerifyOutput<E>,
    point: &[E],
    successor: &CommittedGroupParams,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    level: u32,
    inherited: &SetupBatchStatement<E>,
) -> Result<PreparedOuterSetup<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F> + Ring + MulBaseUnreduced<F> + ChallengeCodec,
{
    let id = successor
        .setup_prefix()
        .and_then(|prefix| prefix.slot_id())
        .ok_or(AkitaError::InvalidProof)?;
    let slot = setup
        .prefix_slots()
        .get(&id)
        .ok_or(AkitaError::InvalidProof)?;
    transcript::bind_outer_prefix(inherited, slot, successor, grinding, level)?;
    let claim = akita_types::stage3_claim::<F, E, _>(grinding, level, E::zero())?;
    let mut claims = inherited.claims().to_vec();
    let relation_weight = rs
        .relation_matrix_evaluator
        .eval_flat_at_point_with_external_setup::<F>(point, rs.alpha, |query| {
            claims.push(akita_types::setup_claims::SetupClaim::new(
                inherited.polynomial().clone(),
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
    let coefficients = batch_coefficients::<F, E, _>(grinding, level, statement.claims().len())?;
    Ok(PreparedOuterSetup {
        statement,
        coefficients,
        claim,
        relation_weight,
    })
}

/// Maximum retained compilation storage for direct inherited-batch discharge.
/// The prover and offloaded verifier retain their existing memory policies.
const MAX_SETUP_BATCH_COMPILED_BYTES: usize = 256 * 1024 * 1024;

/// One setup traversal for the weighted collection, with lane-local terms and
/// one reusable coefficient buffer per parallel reduction worker.
pub(crate) fn decide_setup_batch<F, E>(
    setup: &AkitaExpandedSetup<F>,
    statement: &SetupBatchStatement<E>,
    coefficients: &[E],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F> + Ring + MulBaseUnreduced<F>,
{
    statement
        .polynomial()
        .validate_materialized::<F, E>(setup)?;
    let expected = statement.input_claim(coefficients)?;
    let coefficient_len = statement.coefficient_len();
    // Reject the aggregate before compiling even the first claim. The compiler
    // uses this same accounting before allocating any query's factor tables.
    let required = statement.claims().iter().try_fold(0, |bytes, claim| {
        checked::sum([
            bytes,
            claim
                .query()
                .compilation_bytes(statement.polynomial(), coefficient_len)?,
        ])
        .ok_or(AkitaError::InvalidSize {
            expected: MAX_SETUP_BATCH_COMPILED_BYTES,
            actual: usize::MAX,
        })
    })?;
    if required > MAX_SETUP_BATCH_COMPILED_BYTES {
        return Err(AkitaError::InvalidSize {
            expected: MAX_SETUP_BATCH_COMPILED_BYTES,
            actual: required,
        });
    }
    let mut remaining_bytes = MAX_SETUP_BATCH_COMPILED_BYTES;
    let mut groups = BTreeMap::new();
    let mut active_rows = 0;
    for (claim, &scale) in statement.claims().iter().zip(coefficients) {
        for mut term in claim.query().compile(
            statement.polynomial(),
            coefficient_len,
            Some(&mut remaining_bytes),
        )? {
            term.scale(scale);
            active_rows = active_rows.max(term.active_index_len());
            let lanes: &mut Vec<Vec<SetupWeightTerm<E>>> = groups
                .entry(term.lanes())
                .or_insert_with(|| (0..term.lanes()).map(|_| Vec::new()).collect());
            lanes
                .get_mut(term.lane())
                .ok_or_else(|| AkitaError::Internal("compiled setup lane is missing".into()))?
                .push(term);
        }
    }
    for lanes in groups.values_mut() {
        for terms in lanes {
            terms.sort_unstable_by_key(|term| std::cmp::Reverse(term.active_index_len()));
        }
    }
    let source = setup
        .shared_matrix()
        .as_field_slice()
        .get(..statement.polynomial().field_len())
        .ok_or(AkitaError::InvalidProof)?;
    let value = cfg_fold_reduce!(
        0..active_rows,
        || Ok((E::zero(), vec![E::zero(); coefficient_len])),
        |state: Result<(E, Vec<E>), AkitaError>, row| {
            let (mut sum, mut weight) = state?;
            let mut live = false;
            for (&lane_count, lanes) in &groups {
                let terms = lanes
                    .get(row % lane_count)
                    .ok_or_else(|| AkitaError::Internal("compiled setup lane is missing".into()))?;
                for term in terms
                    .iter()
                    .take_while(|term| row < term.active_index_len())
                {
                    let index = term.index_weight(row);
                    if index.is_zero() {
                        continue;
                    }
                    if !live {
                        weight.fill(E::zero());
                        live = true;
                    }
                    for (weight, &coefficient) in weight.iter_mut().zip(term.coefficient()) {
                        *weight += index * coefficient;
                    }
                }
            }
            if live {
                let range = checked::range(
                    checked::product([row, coefficient_len]).ok_or(AkitaError::InvalidProof)?,
                    coefficient_len,
                )
                .ok_or(AkitaError::InvalidProof)?;
                let chunk = source.get(range).ok_or(AkitaError::InvalidProof)?;
                for (&weight, &base) in weight.iter().zip(chunk) {
                    sum += E::reduce_product(weight.mul_base_unreduced(base));
                }
            }
            Ok((sum, weight))
        },
        |left: Result<(E, Vec<E>), AkitaError>, right: Result<(E, Vec<E>), AkitaError>| {
            let (sum, weight) = left?;
            Ok((sum + right?.0, weight))
        }
    )?
    .0;
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
    use jolt_field::{One, Prime128Offset6A23801F as F, Zero};

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

    #[test]
    fn direct_batch_rejects_aggregate_compilation_before_allocating_tables() {
        const LEN: usize = 4096;
        let setup = AkitaExpandedSetup::<F>::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupDescriptor {
                max_num_vars: 12,
                max_num_batched_polys: 1,
                num_field_elements: LEN,
                setup_seed: [19; 32].into(),
            },
            akita_types::derive_public_matrix_prefix(LEN, &[19; 32].into()),
        );
        let polynomial = SetupPolynomialId::new::<F, F>(&setup, LEN).unwrap();
        let query = SetupQuery::evaluation(vec![F::one(); 12]).unwrap();
        let per_claim = query.compilation_bytes(&polynomial, 2).unwrap();
        let count = MAX_SETUP_BATCH_COMPILED_BYTES / per_claim + 1;
        assert!(count <= akita_params::MAX_SETUP_BATCH_CLAIMS);
        let claim = SetupClaim::new(polynomial.clone(), query, F::zero()).unwrap();
        let batch = SetupBatchStatement::new(polynomial.clone(), 2, vec![claim; count]).unwrap();
        let batch = SetupBatchStatement::<F>::from_descriptor_bytes::<F>(
            &batch.descriptor_bytes::<F>().unwrap(),
            &polynomial,
        )
        .unwrap();
        assert!(matches!(
            decide_setup_batch(&setup, &batch, &vec![F::one(); count]),
            Err(AkitaError::InvalidSize { expected: MAX_SETUP_BATCH_COMPILED_BYTES, actual })
                if actual == per_claim * count
        ));
    }
}
