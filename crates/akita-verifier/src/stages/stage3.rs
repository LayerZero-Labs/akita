//! Verifier for the setup-product sumcheck — the verifier counterpart to the
//! prover-side `AkitaStage3Prover`.

use crate::protocol::ring_switch::RelationMatrixEvaluator;
#[cfg(test)]
use akita_algebra::eq_poly::{EqPolynomial, SplitEqEvals};
#[cfg(test)]
use akita_algebra::ring::eval_ring_at_pows_fast;
use akita_algebra::ring::evaluate_power_sequence_mle;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
#[cfg(test)]
use akita_types::AkitaExpandedSetup;
use akita_types::{
    setup_prefix_coverage_eval_len, AkitaVerifierSetup, CommittedGroupParams,
    PreparedRelationAddress, SetupContributionPlan, SETUP_SUMCHECK_DEGREE,
};
#[cfg(test)]
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

/// Verifier counterpart to `AkitaStage3Prover`: replays the setup product
/// sumcheck for the setup contribution at `x_challenges`.
///
/// Construct with [`SetupSumcheckVerifier::new`], which derives the setup
/// evaluation plan and sumcheck round count from the ring-switch row
/// evaluation, then call [`verify_stage3`](Self::verify_stage3)
/// with the proof and transcript.
pub(crate) struct SetupSumcheckVerifier<E: Field> {
    setup_contribution_plan: SetupContributionPlan<E>,
    alpha: E,
    ring_bits: usize,
    rounds: usize,
}
pub(crate) struct NativeSetupSumcheckReplay<E: Field> {
    pub(crate) claim: E,
    pub(crate) setup_prefix_eval: E,
    pub(crate) challenges: Vec<E>,
}

impl<E: Field> SetupSumcheckVerifier<E> {
    /// Prepare the setup-product sumcheck verifier for the setup contribution
    /// at `x_challenges`.
    ///
    /// Derives the setup evaluation plan (and thus the per-round shape) from
    /// the relation-matrix evaluation; must be called before
    /// [`verify_stage3`](Self::verify_stage3).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new<F>(
        relation_matrix_evaluator: &RelationMatrixEvaluator<E>,
        x_challenges: &[E],
        alpha: E,
    ) -> Result<Self, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let fold_gadget = relation_matrix_evaluator.setup_contribution_fold_gadget::<F>()?;
        let plan = relation_matrix_evaluator
            .take_cached_setup_contribution_plan(x_challenges)?
            .map_or_else(
                || {
                    relation_matrix_evaluator.setup_contribution_plan::<F>(
                        PreparedRelationAddress::new(x_challenges)?,
                        fold_gadget.as_deref(),
                    )
                },
                Ok,
            )?;
        let geometry = plan.projection_geometry();
        Ok(Self {
            setup_contribution_plan: plan,
            alpha,
            ring_bits: geometry.ring_bits(),
            rounds: geometry.rounds(),
        })
    }

    /// Replay stage 3 directly from the native Spongefish stream.
    pub(crate) fn verify_stage3_native<F>(
        &self,
        setup: &AkitaVerifierSetup<F>,
        next_fold_level_params: &CommittedGroupParams,
        grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
        level: u32,
    ) -> Result<NativeSetupSumcheckReplay<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + Ring + AkitaSerialize + jolt_field::MulBaseUnreduced<F>,
    {
        let ring_d = self
            .setup_contribution_plan
            .projection_geometry()
            .base_ring_dim();
        if ring_d == 0 {
            return Err(AkitaError::InvalidSetup(
                "Stage 3 setup ring dimension must be nonzero".into(),
            ));
        }
        setup_eval_len_native(
            setup,
            next_fold_level_params,
            self.setup_contribution_plan
                .projection_geometry()
                .natural_field_len(),
            ring_d,
            grinding,
            level,
        )?;
        let claim = akita_types::native_stage3_verifier_claim::<F, E>(grinding, level)?;
        let mut channel = akita_types::NativeGrindingSumcheckVerifier::<F, E>::new(
            grinding,
            akita_types::SumcheckProtocol::Stage3,
            level,
            0,
        );
        let replay = akita_sumcheck::verify_sumcheck_rounds_native::<F, E, _>(
            &mut channel,
            0,
            claim,
            self.rounds,
            SETUP_SUMCHECK_DEGREE,
        )?;
        let setup_prefix_eval =
            akita_types::native_stage3_verifier_prefix_eval::<F, E>(grinding, level)?;
        let (rho_y, rho_setup_idx) = replay.challenges.split_at(self.ring_bits);
        let setup_index_weight = self
            .setup_contribution_plan
            .evaluate_setup_index_weight_mle(rho_setup_idx, self.alpha)?;
        let alpha_val = evaluate_power_sequence_mle(self.alpha, rho_y);
        if replay.output_claim != setup_prefix_eval * setup_index_weight * alpha_val {
            return Err(AkitaError::InvalidProof);
        }
        Ok(NativeSetupSumcheckReplay {
            claim,
            setup_prefix_eval,
            challenges: replay.challenges,
        })
    }
}

fn setup_eval_len_native<F>(
    setup: &AkitaVerifierSetup<F>,
    next_fold_level_params: &CommittedGroupParams,
    natural_field_len: usize,
    ring_d: usize,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    level: u32,
) -> Result<usize, AkitaError>
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
    let setup_eval_len = setup_prefix_coverage_eval_len(
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
    akita_types::native_stage3_public_slot_verifier(grinding, level, &encoded_slot)?;
    Ok(setup_eval_len)
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
    use jolt_field::Prime128OffsetA7F7;

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
            akita_types::FlatMatrix::from_flat_data(
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
}
