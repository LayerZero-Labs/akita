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
use akita_transcript::labels::{ABSORB_SETUP_PREFIX_SLOT, ABSORB_SUMCHECK_CLAIM};
use akita_transcript::Transcript;
#[cfg(test)]
use akita_types::AkitaExpandedSetup;
use akita_types::{
    dispatch_for_field, setup_prefix_coverage_eval_len, AkitaVerifierSetup, CommittedGroupParams,
    PreparedRelationAddress, SetupContributionPlan, SetupSumcheckProof, SETUP_SUMCHECK_DEGREE,
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

    /// Verify the setup-product stage-3 sumcheck.
    ///
    /// Returns the setup opening point for the next recursive suffix level.
    pub(crate) fn verify_stage3<F, T>(
        &self,
        setup: &AkitaVerifierSetup<F>,
        next_fold_level_params: &CommittedGroupParams,
        proof: &SetupSumcheckProof<E>,
        transcript: &mut T,
        level: u32,
    ) -> Result<Vec<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + Ring + AkitaSerialize + jolt_field::MulBaseUnreduced<F>,
        T: akita_types::VerifierTranscriptGrinding<F>,
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
        let _setup_eval_len = setup_eval_len::<F, T>(
            setup,
            next_fold_level_params,
            self.setup_contribution_plan
                .projection_geometry()
                .natural_field_len(),
            ring_d,
            transcript,
        )?;
        let setup_prefix_eval = next_fold_level_params
            .setup_prefix()
            .as_ref()
            .map(|_| proof.setup_prefix_eval)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "Stage 3 requires a selected setup-prefix slot".to_string(),
                )
            })?;
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Opening),
            F,
            ring_d,
            |D| self.verify_stage3_kernel::<F, T, D>(proof, setup_prefix_eval, transcript, level,)
        )
    }

    fn verify_stage3_kernel<F, T, const D: usize>(
        &self,
        proof: &SetupSumcheckProof<E>,
        setup_prefix_eval: E,
        transcript: &mut T,
        level: u32,
    ) -> Result<Vec<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + Ring + AkitaSerialize + jolt_field::MulBaseUnreduced<F>,
        T: akita_types::VerifierTranscriptGrinding<F>,
    {
        transcript.append_serde(ABSORB_SUMCHECK_CLAIM, &proof.claim);
        let mut round = 0u32;
        let (final_claim, challenges) = proof.sumcheck.verify::<F, _, _>(
            proof.claim,
            self.rounds,
            SETUP_SUMCHECK_DEGREE,
            transcript,
            |tr| {
                let challenge = akita_types::sample_grinded_sumcheck_challenge::<F, E, T>(
                    tr,
                    akita_types::SumcheckProtocol::Stage3,
                    level,
                    0,
                    round,
                )?;
                round = round.checked_add(1).ok_or(AkitaError::InvalidProof)?;
                Ok(challenge)
            },
        )?;
        let (rho_y, rho_setup_idx) = challenges.split_at(self.ring_bits);

        let setup_val = {
            let _span = tracing::info_span!("stage3_setup_prefix", cached = true).entered();
            setup_prefix_eval
        };
        let setup_index_weight = {
            let _span = tracing::info_span!("stage3_setup_index_weight_eval").entered();
            self.setup_contribution_plan
                .evaluate_setup_index_weight_mle(rho_setup_idx, self.alpha)?
        };
        let alpha_val = evaluate_power_sequence_mle(self.alpha, rho_y);
        let setup_term = setup_val * setup_index_weight * alpha_val;
        if final_claim != setup_term {
            return Err(AkitaError::InvalidInput(
                "Stage 3 setup-product claim disagrees with the projected setup opening".into(),
            ));
        }
        Ok(challenges)
    }
}

fn setup_eval_len<F, T>(
    setup: &AkitaVerifierSetup<F>,
    next_fold_level_params: &CommittedGroupParams,
    natural_field_len: usize,
    ring_d: usize,
    transcript: &mut T,
) -> Result<usize, AkitaError>
where
    F: Field + CanonicalEncoding,
    T: Transcript<F>,
{
    let selected_prefix = next_fold_level_params.setup_prefix().ok_or_else(|| {
        AkitaError::InvalidSetup("Stage 3 requires a selected setup-prefix slot".to_string())
    })?;
    let selected_slot_id = selected_prefix.slot_id().ok_or_else(|| {
        AkitaError::InvalidSetup("selected setup-prefix group has no slot identity".to_string())
    })?;
    let slot = setup.prefix_slots.get(&selected_slot_id).ok_or_else(|| {
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
    transcript.append_serde(ABSORB_SETUP_PREFIX_SLOT, &slot.id);
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
    use akita_config::{proof_optimized::fp128::Dense, CommitmentConfig};
    use akita_transcript::AkitaTranscript;
    use akita_types::{
        derive_public_matrix_prefix, padded_setup_prefix_len, scheduled_setup_prefix,
        setup_prefix_precommitted_params, AkitaScheduleLookupKey, AkitaSetupDescriptor,
        CommittedGroupParams, CompressionChainPlan, PolynomialGroupLayout, RingVec,
        SetupPrefixPublicCommitment, SetupPrefixVerifierRegistry, SetupPrefixVerifierSlot,
    };
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::Zero;
    use std::sync::Arc;

    type F = Prime128OffsetA7F7;
    const RING_D: usize = 64;

    fn verifier_setup_with_unaligned_matrix(
        mut level_params: CommittedGroupParams,
        natural_field_len: usize,
    ) -> (AkitaVerifierSetup<F>, CommittedGroupParams) {
        let setup_seed = [7u8; 32].into();
        let descriptor = AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: natural_field_len,
            setup_seed,
        };
        let shared_matrix =
            derive_public_matrix_prefix::<F>(natural_field_len, &descriptor.setup_seed);
        let expanded = Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                descriptor,
                shared_matrix,
            ),
        );

        let full_prefix_len = padded_setup_prefix_len(natural_field_len);
        let d_setup = level_params.inner().matrix.ring_dimension();
        let d_outer = level_params.outer().matrix.ring_dimension();
        let ring_slots = full_prefix_len / d_setup;
        let setup_num_digits = akita_types::sis::compute_num_digits_field_width(
            level_params
                .inner()
                .matrix
                .sis_modulus_profile()
                .field_bits(),
            level_params.inner().digits.log_basis,
        );
        let inner_width = ring_slots
            .checked_mul(setup_num_digits)
            .expect("setup-prefix A width");
        level_params.own_group_mut().profile.inner.matrix =
            akita_types::InnerCommitMatrixParams::try_new_with_min_rank(
                level_params
                    .inner()
                    .matrix
                    .sis_table_key()
                    .expect("L-infinity setup-prefix A matrix"),
                inner_width,
            )
            .expect("full-field setup-prefix A capacity");
        let outer_width = akita_types::CommitmentSliceGeometry::try_new(
            level_params.outer_slice_count(),
            ring_slots,
            1,
            level_params.inner().matrix.output_rank(),
            level_params.outer().digits.num_digits,
            d_setup,
            d_outer,
        )
        .expect("setup-prefix slice geometry")
        .physical_input_width();
        level_params.own_group_mut().profile.outer.matrix =
            akita_types::OuterCommitMatrixParams::try_new_with_min_rank(
                level_params.outer().matrix.sis_table_key(),
                outer_width,
            )
            .expect("setup-prefix B capacity");
        let commitment_params = setup_prefix_precommitted_params(&level_params, full_prefix_len)
            .expect("setup-prefix parameters");
        let id = scheduled_setup_prefix(natural_field_len, commitment_params);
        let matrix = &id.profile.outer.matrix;
        let payload_coefficients = CompressionChainPlan::for_complete_source(
            matrix.sis_modulus_profile(),
            matrix.output_rank() * matrix.ring_dimension(),
        )
        .expect("setup-prefix compression plan")
        .terminal_coefficients();
        level_params.set_setup_prefix(Some(id)).unwrap();
        let mut prefix_slots =
            SetupPrefixVerifierRegistry::new(expanded.descriptor.setup_seed.clone());
        prefix_slots
            .insert(SetupPrefixVerifierSlot {
                id: id.slot_id().expect("setup prefix group"),
                commitment: SetupPrefixPublicCommitment {
                    rows: vec![RingVec::from_coeffs(vec![F::zero(); payload_coefficients])],
                },
            })
            .expect("insert setup-prefix slot");
        let setup = AkitaVerifierSetup::from_parts(expanded, prefix_slots).expect("verifier setup");
        (setup, level_params)
    }

    #[test]
    fn offloaded_setup_ignores_shared_matrix_divisibility() {
        let level_params = Dense::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(16),
        ))
        .expect("scalar schedule")
        .schedule()
        .root
        .params
        .clone();
        let natural_field_len = level_params
            .inner()
            .matrix
            .ring_dimension()
            .checked_mul(level_params.outer_slice_count().get())
            .expect("unaligned setup length")
            + 1;
        let expected_setup_eval_len = padded_setup_prefix_len(natural_field_len) / RING_D;
        let (setup, offloaded_params) =
            verifier_setup_with_unaligned_matrix(level_params.clone(), natural_field_len);
        assert!(!setup
            .expanded
            .shared_matrix()
            .num_field_elements()
            .is_multiple_of(RING_D));

        let mut offloaded_transcript = AkitaTranscript::<F>::new(b"test/offloaded-stage3");
        assert_eq!(
            setup_eval_len(
                &setup,
                &offloaded_params,
                natural_field_len,
                RING_D,
                &mut offloaded_transcript,
            )
            .expect("offloaded setup uses the registered prefix"),
            expected_setup_eval_len,
        );

        let mut direct_transcript = AkitaTranscript::<F>::new(b"test/direct-stage3");
        assert!(setup_eval_len(
            &setup,
            &level_params,
            natural_field_len,
            RING_D,
            &mut direct_transcript,
        )
        .is_err());
    }

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
