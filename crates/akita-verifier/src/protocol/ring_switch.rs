//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
use akita_challenges::Challenges;
use akita_error::AkitaError;
use akita_transcript::labels::{CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    build_compression_relation_weights, dispatch_for_field, shared_setup_fold_gadget,
    validate_role_dispatch, AkitaExpandedSetup, CommittedGroupParams, CompressionRelationWeights,
    FpExtEncoding, NegativeBinarySupport, OpeningClaimsLayout, PreparedRelationAddress,
    RelationAddressGeometry, RelationWitnessGeometry, RingMultiplierOpeningPoint,
    RingRelationGroupOpeningView, RingRelationInstance, RingRole, SetupContributionGroupInputs,
    SetupContributionPlan, WitnessLayout,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};
use std::sync::{Arc, Mutex};

use super::validate_log_basis;

#[cfg(feature = "benchmark-support")]
mod benchmark_support;
mod prepared_relation_point;
mod relation_evaluation;
#[cfg(test)]
mod tests;

#[cfg(feature = "benchmark-support")]
pub use benchmark_support::{
    relation_evaluator_benchmark_case, relation_evaluator_benchmark_case_with_chunks,
    RelationEvaluatorBenchmarkCase,
};

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub(crate) struct RingSwitchVerifyOutput<E: Field> {
    /// Prepared data for prepared relation-matrix MLE evaluation.
    pub relation_matrix_evaluator: RelationMatrixEvaluator<E>,
    /// Independent compact F/H contribution; ordinary A/B/D geometry is unchanged.
    pub compression_relation_weights: Option<CompressionRelationWeights<E>>,
    /// Sparse support of every F/H negative-binary digit span.
    pub negative_binary_support: Option<NegativeBinarySupport>,
    /// Canonical flat relation-witness domain and coefficient/lane split.
    pub relation_address_geometry: RelationAddressGeometry,
    /// Low-variable count used by the protocol's Stage-1 tau0 equality point.
    pub digit_range_equality_low_variable_count: usize,
    /// Challenge tau0 for the stage-1 sumcheck.
    pub tau0: Vec<E>,
    /// Challenge tau1 for the stage-2 M-row combination.
    pub tau1: Vec<E>,
    /// Basis size `b = 2^log_basis`.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: E,
}

struct RingSwitchVerifyCoreOutput<E: Field> {
    relation_matrix_evaluator: RelationMatrixEvaluator<E>,
    compression_relation_weights: Option<CompressionRelationWeights<E>>,
    negative_binary_support: Option<NegativeBinarySupport>,
    relation_address_geometry: RelationAddressGeometry,
    digit_range_equality_low_variable_count: usize,
    tau0: Option<Vec<E>>,
    tau1: Vec<E>,
    b: usize,
    alpha: E,
}

impl<E: Field> RingSwitchVerifyCoreOutput<E> {
    fn into_intermediate(self) -> Result<RingSwitchVerifyOutput<E>, AkitaError> {
        let tau0 = self.tau0.ok_or(AkitaError::InvalidProof)?;
        Ok(RingSwitchVerifyOutput {
            relation_matrix_evaluator: self.relation_matrix_evaluator,
            compression_relation_weights: self.compression_relation_weights,
            negative_binary_support: self.negative_binary_support,
            relation_address_geometry: self.relation_address_geometry,
            digit_range_equality_low_variable_count: self.digit_range_equality_low_variable_count,
            tau0,
            tau1: self.tau1,
            b: self.b,
            alpha: self.alpha,
        })
    }
}

/// Precomputed challenge-derived data for prepared relation-matrix MLE evaluation.
///
/// Stores only data that cannot be derived from context at evaluation time:
/// alpha-evaluated folding challenges and the tau1 eq-polynomial expansion.
/// Everything else is passed by reference at evaluation time to avoid
/// duplicating setup matrix views, opening points, and gadget vectors.
#[derive(Clone)]
pub struct RelationMatrixEvaluator<F: Field> {
    pub(crate) relation_address_geometry: RelationAddressGeometry,
    pub(crate) groups: Vec<RelationMatrixGroupEvaluator<F>>,
    /// Batch-wide basis used by the shared r-tail.
    pub(crate) log_basis: u32,
    pub(crate) eq_tau1: Arc<[F]>,
    pub(crate) flat_context: Option<FlatRelationContext>,
    pub(crate) setup_plan_cache: Arc<Mutex<Option<CachedSetupContributionPlan<F>>>>,
}

pub(crate) struct CachedSetupContributionPlan<F: Field> {
    x_challenges: Vec<F>,
    plan: SetupContributionPlan<F>,
}

#[derive(Clone)]
pub(crate) struct FlatRelationContext {
    pub(crate) level_params: CommittedGroupParams,
    pub(crate) opening_batch: OpeningClaimsLayout,
    pub(crate) witness_layout: Arc<WitnessLayout>,
    pub(crate) extension_degree: usize,
}

#[derive(Clone)]
pub(crate) struct RelationMatrixGroupEvaluator<F: Field> {
    pub(crate) c_alphas: Vec<F>,
    pub(crate) opening_a_evals: Vec<F>,
    pub(crate) group_id: usize,
    pub(crate) num_claims: usize,
    pub(crate) depth_fold: usize,
    pub(crate) a_row_start: usize,
    pub(crate) b_row_start: usize,
}

/// Fixed public relation inputs for verifier ring-switch replay.
pub struct RingSwitchReplay<'a, F: Field, E> {
    pub setup: &'a AkitaExpandedSetup<F>,
    pub relation: &'a RingRelationInstance<F>,
    pub row_coefficients: &'a [E],
    pub lp: &'a CommittedGroupParams,
    pub opening_source_len: usize,
    pub opening_ring_dim: usize,
}

/// Replay the verifier half of ring switching after the caller has absorbed
/// the schedule-selected outgoing witness binding.
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub(crate) fn ring_switch_verifier<F, E, T>(
    replay: &RingSwitchReplay<'_, F, E>,
    w_len: usize,
    transcript: &mut T,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_batch = relation.opening_batch();
    let num_polys = opening_batch.num_total_polynomials();
    let gamma = replay.row_coefficients;

    let alpha: E = {
        let _span = tracing::info_span!("ring_switch_transcript_challenges").entered();
        sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_RING_SWITCH)
    };

    let num_claims = relation.opening_batch().num_total_polynomials();
    let relation_geometry =
        RelationWitnessGeometry::for_level(lp, opening_batch, relation.extension_degree())?;
    // Validate each group's opening/multiplier point against that group's own
    // block geometry (final vs frozen-precommit). For a scalar batch this is the
    // single group at `lp`'s geometry, byte-identical to the historical check.
    for group_index in 0..opening_batch.num_groups() {
        let group_lp = lp.group_params(opening_batch, group_index)?;
        match relation.group_opening_view(group_index)? {
            RingRelationGroupOpeningView::EvaluationTrace {
                ring_multiplier_point,
                ..
            } => {
                if ring_multiplier_point.position_len() != group_lp.num_positions_per_block()
                    || ring_multiplier_point.fold_len() != group_lp.num_live_blocks()
                {
                    return Err(AkitaError::InvalidProof);
                }
            }
            RingRelationGroupOpeningView::SubringCoefficientPacking { geometry, .. } => {
                let expected = relation_geometry.group_opening_geometry(group_index)?;
                if geometry.extension_degree() != relation.extension_degree()
                    || geometry.a_ring_dimension()
                        != group_lp.inner_commit_matrix_params().ring_dimension()
                    || geometry.partial_base_field_width() != expected.physical_coefficient_width()
                {
                    return Err(AkitaError::InvalidProof);
                }
            }
        }
    }
    if num_polys != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    let witness_layout = relation.segment_layout(lp, None)?;
    let relation_address_geometry = lp.relation_address_geometry(
        opening_batch,
        relation.extension_degree(),
        replay.opening_ring_dim,
        witness_layout.live_coeff_len(),
    )?;
    if w_len == 0 || w_len != relation_address_geometry.digit_witness_domain().live_len() {
        return Err(AkitaError::InvalidProof);
    }
    // Bind the current roles' shared low coefficient block as the digit-range
    // check's ring phase. Outgoing witness packaging determines the checked
    // flat live length but never this point split.
    let digit_range_equality_low_variable_count =
        relation_address_geometry.relation_coefficient_variable_count();
    let num_sc_vars = relation_address_geometry.relation_point_variable_count();
    let num_i = lp.relation_row_index_num_vars(opening_batch)?;

    let (tau0, tau1) = {
        let _span = tracing::info_span!(
            "ring_switch_transcript_challenges",
            tau0_len = num_sc_vars,
            tau1_len = num_i
        )
        .entered();
        let tau0 = Some(
            (0..num_sc_vars)
                .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
                .collect(),
        );
        let tau1 = (0..num_i)
            .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
            .collect::<Vec<_>>();
        (tau0, tau1)
    };
    if gamma.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let relation_matrix_evaluator =
        prepare_relation_matrix_evaluator::<F, E>(replay, alpha, &tau1, Some(w_len))?;
    let physical_field_len = replay
        .opening_source_len
        .checked_mul(replay.opening_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("opening capacity overflow".into()))?;
    let compression_relation_weights = lp
        .payload_mode
        .is_compressed()
        .then(|| {
            build_compression_relation_weights(
                replay.setup,
                relation,
                alpha,
                lp,
                &tau1,
                &witness_layout,
                replay.opening_ring_dim,
                physical_field_len,
            )
        })
        .transpose()?;
    let negative_binary_support = lp
        .payload_mode
        .is_compressed()
        .then(|| NegativeBinarySupport::new(&witness_layout, physical_field_len))
        .transpose()?;
    RingSwitchVerifyCoreOutput {
        relation_matrix_evaluator,
        compression_relation_weights,
        negative_binary_support,
        relation_address_geometry,
        digit_range_equality_low_variable_count,
        tau0,
        tau1,
        b: 1usize
            .checked_shl(lp.open().digits.log_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("basis size overflow".to_string()))?,
        alpha,
    }
    .into_intermediate()
}

/// Prepare relation-matrix evaluator state from a fixed
/// [`RingRelationInstance`] and transcript-sampled row coefficients.
///
/// # Errors
///
/// Returns an error if gamma/challenge lengths do not match the claim shape,
/// the expanded tau1 table is too short for the level layout, or sparse
/// challenge evaluation fails.
#[tracing::instrument(skip_all, name = "prepare_relation_matrix_evaluator")]
pub fn prepare_relation_matrix_evaluator<F, E>(
    replay: &RingSwitchReplay<'_, F, E>,
    alpha: E,
    tau1: &[E],
    witness_ring_len: Option<usize>,
) -> Result<RelationMatrixEvaluator<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_batch = relation.opening_batch();
    let layout = relation.segment_layout(lp, witness_ring_len)?;
    let relation_address_geometry = lp.relation_address_geometry(
        opening_batch,
        relation.extension_degree(),
        replay.opening_ring_dim,
        layout.live_coeff_len(),
    )?;
    let opening_capacity = replay
        .opening_source_len
        .checked_mul(replay.opening_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("opening capacity overflow".into()))?;
    if layout.live_coeff_len() > opening_capacity {
        return Err(AkitaError::InvalidProof);
    }
    let rows = lp.relation_matrix_row_count(opening_batch.num_groups())?;
    if lp.has_preceding_groups() {
        return prepare_relation_matrix_evaluator_multi_group::<F, E>(
            replay,
            alpha,
            tau1,
            layout,
            rows,
            relation_address_geometry,
            relation.extension_degree(),
        );
    }
    let challenges = relation.group_ambient_a_challenges(0)?;
    let ring_multiplier_point = match relation.group_opening_view(0)? {
        RingRelationGroupOpeningView::EvaluationTrace {
            ring_multiplier_point,
            ..
        } => Some(ring_multiplier_point),
        RingRelationGroupOpeningView::SubringCoefficientPacking { .. } => None,
    };
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        lp.role_dims().d_a(),
        |D_A| {
            prepare_relation_matrix_evaluator_inner::<F, E, D_A>(
                challenges,
                ring_multiplier_point,
                alpha,
                lp,
                tau1,
                opening_batch,
                replay.row_coefficients,
                layout,
                rows,
                relation_address_geometry,
                relation.extension_degree(),
            )
        }
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_relation_matrix_evaluator_multi_group<F, E>(
    replay: &RingSwitchReplay<'_, F, E>,
    alpha: E,
    tau1: &[E],
    layout: WitnessLayout,
    rows: usize,
    relation_address_geometry: RelationAddressGeometry,
    extension_degree: usize,
) -> Result<RelationMatrixEvaluator<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_batch = relation.opening_batch();
    lp.validate_opening_batch(opening_batch)?;
    if replay.row_coefficients.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }

    let eq_tau1: std::sync::Arc<[E]> = EqPolynomial::evals_prefix(tau1, rows)?.into();

    let order = opening_batch.root_group_order()?;
    if order
        .iter()
        .any(|&group_index| layout.num_chunks_for_group(group_index) != lp.witness_chunk.num_chunks)
    {
        return Err(AkitaError::InvalidSetup(
            "multi-group witness layout does not match root group order".to_string(),
        ));
    }

    let mut groups = Vec::with_capacity(order.len());
    for &group_index in &order {
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let group_role_dims = lp.group_role_dims(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        let k_g = group_layout.num_polynomials();
        let num_live_blocks = group_lp.num_live_blocks();
        let num_positions_per_block = group_lp.num_positions_per_block();
        let depth_witness = group_lp.num_digits_inner();
        let depth_fold = group_lp.num_digits_fold();
        let log_basis_inner = group_lp.log_basis_inner();
        let log_basis_outer = group_lp.log_basis_outer();
        let log_basis_open = group_lp.log_basis_open();
        validate_log_basis(log_basis_inner)?;
        validate_log_basis(log_basis_outer)?;
        validate_log_basis(log_basis_open)?;
        let n_a = group_lp.a_rows_len();
        let n_b = group_lp.logical_b_rows_len()?;
        let inner_width = group_lp.a_col_len();
        let expected_inner_width = num_positions_per_block
            .checked_mul(depth_witness)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("multi-group inner width overflow".to_string())
            })?;
        if inner_width < expected_inner_width {
            return Err(AkitaError::InvalidSetup(
                "multi-group A-key column width is too small".to_string(),
            ));
        }

        let ring_multiplier_point = match relation.group_opening_view(group_index)? {
            RingRelationGroupOpeningView::EvaluationTrace {
                ring_multiplier_point,
                ..
            } => Some(ring_multiplier_point),
            RingRelationGroupOpeningView::SubringCoefficientPacking { .. } => None,
        };
        if let Some(ring_multiplier_point) = ring_multiplier_point {
            if ring_multiplier_point.position_len() != num_positions_per_block
                || ring_multiplier_point.fold_len() != num_live_blocks
            {
                return Err(AkitaError::InvalidProof);
            }
        }

        let total_blocks = k_g.checked_mul(num_live_blocks).ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group block count overflow".to_string())
        })?;
        let challenges = relation.group_ambient_a_challenges(group_index)?;
        if challenges.len() != total_blocks {
            return Err(AkitaError::InvalidSize {
                expected: total_blocks,
                actual: challenges.len(),
            });
        }
        let (c_alphas, opening_a_evals) = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            group_role_dims.d_a(),
            |D_GROUP| {
                let alpha_pows = scalar_powers(alpha, D_GROUP);
                let c_alphas =
                    prepare_challenge_evals::<F, E>(challenges, &alpha_pows, k_g, num_live_blocks)?;
                let opening_a_evals = match ring_multiplier_point {
                    Some(point) => (0..num_positions_per_block)
                        .map(|idx| point.eval_position_at::<E>(idx, &alpha_pows))
                        .collect::<Result<Vec<_>, _>>()?,
                    None => vec![E::zero(); num_positions_per_block],
                };
                Ok::<_, AkitaError>((c_alphas, opening_a_evals))
            }
        )?;

        let a_range = lp.a_row_range(opening_batch, group_index)?;
        let b_range = lp.commitment_row_range(opening_batch, group_index)?;
        if a_range.len() != n_a || b_range.len() != n_b {
            return Err(AkitaError::InvalidSetup(
                "multi-group row ranges do not match group matrix heights".to_string(),
            ));
        }

        groups.push(RelationMatrixGroupEvaluator {
            c_alphas,
            opening_a_evals,
            group_id: group_index,
            num_claims: k_g,
            depth_fold,
            a_row_start: a_range.start,
            b_row_start: b_range.start,
        });
    }

    let layout = Arc::new(layout);

    Ok(RelationMatrixEvaluator {
        relation_address_geometry,
        groups,
        log_basis: lp.open().digits.log_basis,
        eq_tau1,
        flat_context: Some(FlatRelationContext {
            level_params: lp.clone(),
            opening_batch: opening_batch.clone(),
            witness_layout: layout,
            extension_degree,
        }),
        setup_plan_cache: Default::default(),
    })
}

fn prepare_challenge_evals<F, E>(
    challenges: &Challenges,
    alpha_pows: &[E],
    num_claims: usize,
    num_live_blocks: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + Ring,
    E: Field + ExtField<F>,
{
    if challenges.num_claims() != num_claims {
        return Err(AkitaError::InvalidSize {
            expected: num_claims,
            actual: challenges.num_claims(),
        });
    }
    if challenges.num_live_blocks_per_claim() != num_live_blocks {
        return Err(AkitaError::InvalidSize {
            expected: num_live_blocks,
            actual: challenges.num_live_blocks_per_claim(),
        });
    }
    challenges.evals_at_pows::<F, E>(alpha_pows)
}

#[allow(clippy::too_many_arguments)]
fn prepare_relation_matrix_evaluator_inner<F, E, const D: usize>(
    challenges: &Challenges,
    ring_multiplier_point: Option<&RingMultiplierOpeningPoint<F>>,
    alpha: E,
    lp: &CommittedGroupParams,
    tau1: &[E],
    opening_batch: &OpeningClaimsLayout,
    gamma: &[E],
    layout: WitnessLayout,
    rows: usize,
    relation_address_geometry: RelationAddressGeometry,
    extension_degree: usize,
) -> Result<RelationMatrixEvaluator<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
{
    validate_role_dispatch::<D>(lp.role_dims(), RingRole::Inner)?;
    let num_polys = opening_batch.num_total_polynomials();
    let depth_fold = lp.num_digits_fold();
    let alpha_pows = scalar_powers(alpha, D);
    let num_claims = gamma.len();
    if num_polys != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    let log_basis_inner = lp.inner().digits.log_basis;
    let log_basis_outer = lp.outer().digits.log_basis;
    let log_basis_open = lp.open().digits.log_basis;
    validate_log_basis(log_basis_inner)?;
    validate_log_basis(log_basis_outer)?;
    validate_log_basis(log_basis_open)?;
    let num_live_blocks = lp.blocks().live_blocks;
    let total_blocks = num_live_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.len() != total_blocks {
        return Err(AkitaError::InvalidSize {
            expected: total_blocks,
            actual: challenges.len(),
        });
    }
    let num_positions_per_block = lp.blocks().positions_per_block;
    let n_a = lp.inner().matrix.output_rank();

    let c_alphas = prepare_challenge_evals::<F, E>(
        challenges,
        &alpha_pows,
        num_claims,
        lp.blocks().live_blocks,
    )?;
    let opening_a_evals = match ring_multiplier_point {
        Some(point) => (0..num_positions_per_block)
            .map(|idx| point.eval_position_at::<E>(idx, &alpha_pows))
            .collect::<Result<Vec<_>, _>>()?,
        None => vec![E::zero(); num_positions_per_block],
    };
    let group = RelationMatrixGroupEvaluator {
        c_alphas,
        opening_a_evals,
        group_id: 0,
        num_claims,
        depth_fold,
        a_row_start: 1,
        b_row_start: 1 + n_a,
    };

    let groups = vec![group];
    let layout = Arc::new(layout);
    let eq_tau1: std::sync::Arc<[E]> = EqPolynomial::evals_prefix(tau1, rows)?.into();

    Ok(RelationMatrixEvaluator {
        relation_address_geometry,
        groups,
        log_basis: log_basis_open,
        eq_tau1,
        flat_context: Some(FlatRelationContext {
            level_params: lp.clone(),
            opening_batch: opening_batch.clone(),
            witness_layout: layout,
            extension_degree,
        }),
        setup_plan_cache: Default::default(),
    })
}

pub(crate) fn setup_contribution_group_inputs<F: Field>(
    groups: &[RelationMatrixGroupEvaluator<F>],
) -> Vec<SetupContributionGroupInputs> {
    groups
        .iter()
        .map(|group| SetupContributionGroupInputs {
            group_id: group.group_id,
            num_claims: group.num_claims,
            depth_fold: group.depth_fold,
            a_row_start: group.a_row_start,
            b_row_start: group.b_row_start,
        })
        .collect()
}

impl<E: Field> RelationMatrixEvaluator<E> {
    /// Evaluate the canonical relation weights directly in the flattened
    /// opening domain, without materializing its padded Boolean suffix.
    pub fn eval_flat_at_point<F>(
        &self,
        point: &[E],
        setup: &AkitaExpandedSetup<F>,
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
    {
        relation_evaluation::evaluate_relation_at_point::<F, E>(
            self,
            point,
            setup,
            alpha,
            setup_claim,
        )
    }

    pub(crate) fn setup_contribution_inputs(&self) -> Vec<SetupContributionGroupInputs> {
        setup_contribution_group_inputs(&self.groups)
    }

    pub(crate) fn setup_contribution_fold_gadget<F>(&self) -> Result<Option<Vec<F>>, AkitaError>
    where
        F: Field + CanonicalEncoding,
    {
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        let setup_groups = self.setup_contribution_inputs();
        Ok(shared_setup_fold_gadget(
            &context.level_params,
            &context.opening_batch,
            &setup_groups,
        ))
    }

    pub(crate) fn setup_contribution_plan<F>(
        &self,
        relation_address: PreparedRelationAddress<E>,
        fold_gadget: Option<&[F]>,
    ) -> Result<SetupContributionPlan<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        let setup_groups = self.setup_contribution_inputs();
        SetupContributionPlan::prepare::<F>(
            &context.level_params,
            &context.opening_batch,
            context.extension_degree,
            self.eq_tau1.clone(),
            &context.witness_layout,
            &setup_groups,
            relation_address,
            fold_gadget,
            self.relation_address_geometry,
        )
    }

    pub(crate) fn take_cached_setup_contribution_plan(
        &self,
        x_challenges: &[E],
    ) -> Result<Option<SetupContributionPlan<E>>, AkitaError> {
        let mut cache = self.setup_plan_cache.lock().map_err(|_| {
            AkitaError::InvalidSetup("setup contribution plan cache is poisoned".into())
        })?;
        let Some(cached) = cache.as_ref() else {
            return Ok(None);
        };
        if cached.x_challenges.as_slice() != x_challenges {
            return Ok(None);
        }
        Ok(cache.take().map(|cached| cached.plan))
    }

    fn cache_setup_contribution_plan(
        &self,
        x_challenges: &[E],
        plan: SetupContributionPlan<E>,
    ) -> Result<(), AkitaError> {
        let mut cache = self.setup_plan_cache.lock().map_err(|_| {
            AkitaError::InvalidSetup("setup contribution plan cache is poisoned".into())
        })?;
        *cache = Some(CachedSetupContributionPlan {
            x_challenges: x_challenges.to_vec(),
            plan,
        });
        Ok(())
    }

    pub(crate) fn witness_layout(&self) -> Result<&WitnessLayout, AkitaError> {
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        Ok(&context.witness_layout)
    }
}
