//! Verifier for the setup-product sumcheck — the verifier counterpart to the
//! prover-side `AkitaStage3Prover`.

use crate::setup_batch::PreparedOuterSetup;
use crate::stages::ring_switch::RingSwitchVerifyOutput;
#[cfg(test)]
use akita_algebra::eq_poly::{EqPolynomial, SplitEqEvals};
#[cfg(test)]
use akita_algebra::ring::eval_ring_at_pows_fast;
use akita_error::AkitaError;
use akita_params::{CommittedGroupParams, SETUP_SUMCHECK_DEGREE};
use akita_serialization::AkitaSerialize;
#[cfg(test)]
use akita_types::AkitaExpandedSetup;
use akita_types::{
    setup_claims::{SetupBatchStatement, SetupClaim, SetupPolynomialId, SetupQuery},
    setup_prefix_coverage_eval_len, AkitaVerifierSetup, FpExtEncoding,
    PreparedCoefficientFunctional, PreparedRelationAddress,
};
#[cfg(test)]
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field};

pub(crate) struct SetupSumcheckReplay<E: Field> {
    /// This fold's Stage 2 setup answer, before combining inherited claims.
    pub(crate) claim: E,
    pub(crate) setup_prefix_eval: E,
    pub(crate) challenges: Vec<E>,
}

/// Check a setup sumcheck, leaving its setup evaluation to the successor.
fn verify_setup_sumcheck<F, E>(
    statement: &SetupBatchStatement<E>,
    coefficients: &[E],
    claim: E,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    level: u32,
) -> Result<SetupSumcheckReplay<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let input = statement.input_claim(coefficients)?;
    let mut channel = akita_types::GrindingSumcheckVerifier::<F, E>::new(
        grinding,
        akita_params::SumcheckProtocol::Stage3,
        level,
        0,
    );
    let replay = akita_sumcheck::verify_sumcheck_rounds::<F, E, _>(
        &mut channel,
        0,
        input,
        akita_sumcheck::SumcheckShape::new(
            statement.polynomial().num_variables(),
            SETUP_SUMCHECK_DEGREE,
        )?,
    )?;
    let value = akita_types::stage3_prefix_eval::<F, E, _>(grinding, level, E::zero())?;
    let weight = statement.evaluate_weight(coefficients, &replay.challenges)?;
    if replay.output_claim != value * weight {
        return Err(AkitaError::InvalidProof);
    }
    Ok(SetupSumcheckReplay {
        claim,
        setup_prefix_eval: value,
        challenges: replay.challenges,
    })
}

/// Check Stage 3 in either setup mode; the successor authenticates its endpoint.
///
/// The setup contribution is evaluated at the Stage 2 point with its
/// coefficient variables removed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_stage3<F, E>(
    setup: &AkitaVerifierSetup<F>,
    rs: &RingSwitchVerifyOutput<E>,
    stage2_challenges: &[E],
    next_params: &CommittedGroupParams,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    level: u32,
    outer: Option<&PreparedOuterSetup<E>>,
) -> Result<SetupSumcheckReplay<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F>,
{
    let singleton;
    let unit_coefficient = [E::one()];
    let (statement, coefficients, claim) = if let Some(outer) = outer {
        (&outer.statement, outer.coefficients.as_slice(), outer.claim)
    } else {
        let setup_coefficient_bits = rs
            .relation_address_geometry
            .relation_coefficient_variable_count();
        let setup_x_challenges = stage2_challenges
            .get(setup_coefficient_bits..)
            .ok_or(AkitaError::InvalidProof)?;
        let fold_gadget = rs
            .relation_matrix_evaluator
            .setup_contribution_fold_gadget::<F>()?;
        let plan = rs.relation_matrix_evaluator.setup_contribution_plan::<F>(
            PreparedRelationAddress::new(setup_x_challenges)?,
            fold_gadget.as_deref(),
        )?;
        let geometry = plan.projection_geometry();
        let ring_d = geometry.base_ring_dim();
        if ring_d == 0 {
            return Err(AkitaError::InvalidSetup(
                "Stage 3 setup ring dimension must be nonzero".into(),
            ));
        }
        bind_setup_prefix_slot(
            setup,
            next_params,
            geometry.natural_field_len(),
            ring_d,
            grinding,
            level,
        )?;
        let claim = akita_types::stage3_claim::<F, E, _>(grinding, level, E::zero())?;
        let field_len =
            akita_error::checked::pow2(geometry.rounds()).ok_or(AkitaError::InvalidProof)?;
        let polynomial = SetupPolynomialId::new::<F, E>(setup.expanded(), field_len)?;
        let query = SetupQuery::contribution(
            plan,
            PreparedCoefficientFunctional::LiftedPower { alpha: rs.alpha },
        )?;
        singleton = SetupBatchStatement::new(
            polynomial.clone(),
            ring_d,
            vec![SetupClaim::new(polynomial, query, claim)?],
        )?;
        // An ordinary proof already binds its single query through the fold
        // transcript, so it needs no batch descriptor or random coefficient.
        (&singleton, unit_coefficient.as_slice(), claim)
    };
    verify_setup_sumcheck::<F, E>(statement, coefficients, claim, grinding, level)
}

/// Resolve the planned setup-prefix slot in the verifier setup, check that it
/// covers the Stage 3 setup product, and absorb its public slot id.
///
/// The coverage length returned by [`setup_prefix_coverage_eval_len`] is not
/// needed: it equals the projection geometry's `setup_index_len` because the
/// common base ring dimension is a power of two.
fn bind_setup_prefix_slot<F>(
    setup: &AkitaVerifierSetup<F>,
    next_fold_level_params: &CommittedGroupParams,
    natural_field_len: usize,
    ring_d: usize,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    level: u32,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
{
    let selected_prefix = next_fold_level_params.setup_prefix().ok_or_else(|| {
        AkitaError::InvalidSetup("Stage 3 requires a selected setup-prefix slot".to_string())
    })?;
    let selected_slot_id = selected_prefix.slot_id().ok_or_else(|| {
        AkitaError::InvalidSetup("selected setup-prefix group has no slot identity".to_string())
    })?;
    let slot = setup.prefix_slots().get(&selected_slot_id).ok_or_else(|| {
        AkitaError::InvalidSetup(
            "planned setup-prefix slot is missing from verifier setup".to_string(),
        )
    })?;
    setup_prefix_coverage_eval_len(
        None,
        &slot.id,
        next_fold_level_params,
        natural_field_len,
        ring_d,
        "verifier setup-prefix slot does not cover setup product",
    )?;
    let mut encoded_slot = Vec::new();
    slot.id
        .serialize_compressed(&mut encoded_slot)
        .map_err(|_| AkitaError::InvalidProof)?;
    akita_types::stage3_public_slot(grinding, level, &encoded_slot)
}

#[cfg(test)]
fn ring_eq_table<E: Field, const D: usize>(rho_y: &[E]) -> Result<Vec<E>, AkitaError> {
    if rho_y.len() != D.trailing_zeros() as usize {
        return Err(AkitaError::InvalidProof);
    }
    let eq_y = EqPolynomial::evals(rho_y)?;
    if eq_y.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: eq_y.len(),
        });
    }
    Ok(eq_y)
}

#[cfg(test)]
fn setup_mle_at_eq_tables<F, E, const D: usize>(
    setup: &AkitaExpandedSetup<F>,
    source_rows: usize,
    setup_eval_len: usize,
    rho_setup_idx: &[E],
    eq_y: &[E],
) -> Result<E, AkitaError>
where
    F: Field,
    E: ExtField<F> + jolt_field::MulBaseUnreduced<F>,
{
    if source_rows > setup_eval_len {
        return Err(AkitaError::InvalidSetup(
            "setup prefix is too small for selected verifier layout".into(),
        ));
    }
    let eq_setup_idx = SplitEqEvals::new(rho_setup_idx)?;
    if eq_setup_idx.len() != source_rows {
        return Err(AkitaError::InvalidSize {
            expected: source_rows,
            actual: eq_setup_idx.len(),
        });
    }
    if eq_y.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: eq_y.len(),
        });
    }
    let setup_view = setup.shared_matrix().ring_view::<D>(1, source_rows)?;
    let setup_entries = setup_view.as_slice();

    // Scan the selected setup prefix once. Each entry contracts the ring with
    // `eq_y` and the setup-index equality; the scan is `O(source_rows · D)` and
    // is the dominant recursive-mode verifier cost, so evaluate it in parallel.
    let _span = tracing::info_span!("stage3_setup_mle_scan", source_rows).entered();
    let inner_len = eq_setup_idx.in_len();
    let required_outer = source_rows.div_ceil(inner_len);
    cfg_fold_reduce!(
        0..required_outer,
        || Ok(E::zero()),
        |acc: Result<E, AkitaError>, outer_idx| {
            let start = outer_idx
                .checked_mul(inner_len)
                .ok_or(AkitaError::InvalidProof)?;
            let end = start.saturating_add(inner_len).min(source_rows);
            let entries = setup_entries
                .get(start..end)
                .ok_or(AkitaError::InvalidProof)?;
            let inner_weights = eq_setup_idx
                .e_in
                .get(..entries.len())
                .ok_or(AkitaError::InvalidProof)?;
            let mut inner = E::zero();
            for (entry, &weight) in entries.iter().zip(inner_weights) {
                inner += eval_ring_at_pows_fast(entry, eq_y) * weight;
            }
            let outer_weight = eq_setup_idx
                .e_out
                .get(outer_idx)
                .ok_or(AkitaError::InvalidProof)?;
            Ok(acc? + *outer_weight * inner)
        },
        |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use akita_types::AkitaSetupDescriptor;
    use jolt_field::{Prime128OffsetA7F7, Ring};

    type F = Prime128OffsetA7F7;
    const RING_D: usize = 64;

    #[test]
    fn setup_mle_scan_matches_dense_reference() {
        let required = 9usize;
        let source_rows = required.next_power_of_two();
        let setup_eval_len = source_rows;
        let descriptor = AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_eval_len * RING_D,
            setup_seed: [9u8; 32].into(),
        };
        let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            descriptor,
            akita_params::FlatMatrix::from_flat_data(
                (0..setup_eval_len * RING_D)
                    .map(|index| F::from_u64(11 + index as u64))
                    .collect(),
            ),
        );
        let rho_y = (0..RING_D.trailing_zeros() as usize)
            .map(|index| F::from_u64(101 + index as u64))
            .collect::<Vec<_>>();
        let eq_y = ring_eq_table::<F, RING_D>(&rho_y).expect("ring equality table");
        let rho_setup = (0..required.next_power_of_two().trailing_zeros() as usize)
            .map(|index| F::from_u64(201 + index as u64))
            .collect::<Vec<_>>();
        let eq_setup = SplitEqEvals::new(&rho_setup).expect("setup equality");
        let rings = setup
            .shared_matrix()
            .ring_view::<RING_D>(1, setup_eval_len)
            .expect("setup ring view");
        let expected = rings
            .as_slice()
            .iter()
            .take(source_rows)
            .enumerate()
            .map(|(index, ring)| {
                eq_setup.eval_at(index).expect("setup equality entry")
                    * eval_ring_at_pows_fast(ring, &eq_y)
            })
            .sum::<F>();
        assert_eq!(
            setup_mle_at_eq_tables::<F, F, RING_D>(
                &setup,
                source_rows,
                setup_eval_len,
                &rho_setup,
                &eq_y,
            )
            .expect("streamed setup scan"),
            expected
        );
    }

    #[test]
    fn stage3_checks_the_terminal_equation_even_without_rounds() {
        use akita_params::{ChallengeFieldOrder, GrindingPlan};
        use akita_sumcheck::SumcheckKernel;
        use akita_types::setup_claims::{
            SetupBatchStatement, SetupClaim, SetupPolynomialId, SetupQuery,
        };
        use akita_types::{ProverGrinding, VerifierGrinding};
        use jolt_field::One;
        use jolt_poly::UnivariatePoly;

        struct ConstantClaim(F);
        impl SumcheckKernel<F> for ConstantClaim {
            fn num_rounds(&self) -> usize {
                0
            }
            fn degree_bound(&self) -> usize {
                SETUP_SUMCHECK_DEGREE
            }
            fn input_claim(&self) -> F {
                self.0
            }
            fn round_polynomial(
                &mut self,
                _: usize,
                _: F,
            ) -> Result<UnivariatePoly<F>, AkitaError> {
                Err(AkitaError::InvalidProof)
            }
            fn bind_challenge(&mut self, _: usize, _: F) -> Result<(), AkitaError> {
                Err(AkitaError::InvalidProof)
            }
            fn finish(&mut self) -> Result<(), AkitaError> {
                Ok(())
            }
        }
        let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupDescriptor {
                max_num_vars: 0,
                max_num_batched_polys: 0,
                num_field_elements: 1,
                setup_seed: [43; 32].into(),
            },
            akita_types::derive_public_matrix_prefix::<F>(1, &[43; 32].into()),
        );
        let value = setup.shared_matrix().as_field_slice()[0];
        let domain = SetupPolynomialId::new::<F, F>(&setup, 1).unwrap();
        let statement = SetupBatchStatement::new(
            domain.clone(),
            1,
            vec![
                SetupClaim::new(domain, SetupQuery::evaluation(Vec::new()).unwrap(), value)
                    .unwrap(),
            ],
        )
        .unwrap();
        let plan = GrindingPlan::new(
            Vec::new(),
            ChallengeFieldOrder::from_full_capacity(128).unwrap(),
        )
        .unwrap();
        for coefficient in [F::one(), F::from_u64(7)] {
            for endpoint in [value, value + F::one()] {
                let state =
                    akita_transcript::new_prover_channel(b"checked-stage3", b"fixture").unwrap();
                let mut prover = ProverGrinding::new(state, &plan);
                let mut channel = akita_types::GrindingSumcheckProver::<F, F>::new(
                    &mut prover,
                    akita_params::SumcheckProtocol::Stage3,
                    0,
                    0,
                );
                akita_sumcheck::prove_sumcheck::<F, F, _, _>(
                    &mut ConstantClaim(coefficient * value),
                    &mut channel,
                    akita_sumcheck::SumcheckShape::new(0, SETUP_SUMCHECK_DEGREE).unwrap(),
                    0,
                )
                .unwrap();
                akita_types::stage3_prefix_eval::<F, F, _>(&mut prover, 0, endpoint).unwrap();
                let proof = prover.finish().unwrap();
                let state =
                    akita_transcript::new_verifier_channel(b"checked-stage3", b"fixture", &proof)
                        .unwrap();
                let mut verifier = VerifierGrinding::new(state, &plan);
                let result = verify_setup_sumcheck::<F, F>(
                    &statement,
                    &[coefficient],
                    value,
                    &mut verifier,
                    0,
                );
                if endpoint == value {
                    let replay = result.unwrap();
                    assert_eq!(replay.setup_prefix_eval, value);
                    assert_eq!(replay.claim, value);
                    verifier.finish().unwrap();
                } else {
                    assert!(matches!(result, Err(AkitaError::InvalidProof)));
                }
            }
        }
    }
}
