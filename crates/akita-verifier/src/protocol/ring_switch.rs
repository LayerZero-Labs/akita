//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::OffsetEqWindow;
use akita_algebra::ring::scalar_powers;
use akita_challenges::Challenges;
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase, MulBaseUnreduced,
    RandomSampling,
};
use akita_transcript::labels::{CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    dispatch_for_field, gadget_row_scalars, r_decomp_levels, shared_setup_fold_gadget,
    validate_role_dispatch, AkitaExpandedSetup, CommitmentRingDims, CommittedGroupParams,
    FpExtEncoding, OpeningClaimsLayout, RelationAddressGeometry, RingMultiplierOpeningPoint,
    RingRelationInstance, RingRole, SetupContributionGroupInputs, SetupContributionPlan,
    WitnessLayout,
};
use std::sync::{Arc, Mutex};

use super::slice_mle::compute_r_contribution;
use super::validate_log_basis;
use akita_types::validate_ring_dispatch;
pub(crate) use tensor_challenges::PreparedChallengeEvals;

mod mixed_relation;
mod prepared_relation_point;
mod structured;
mod tensor_challenges;
#[cfg(test)]
mod tests;

use structured::evaluate_group_et_from_eq_slices;

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub(crate) struct RingSwitchVerifyOutput<E: FieldCore> {
    /// Prepared data for prepared relation-matrix MLE evaluation.
    pub relation_matrix_evaluator: RelationMatrixEvaluator<E>,
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

struct RingSwitchVerifyCoreOutput<E: FieldCore> {
    relation_matrix_evaluator: RelationMatrixEvaluator<E>,
    relation_address_geometry: RelationAddressGeometry,
    digit_range_equality_low_variable_count: usize,
    tau0: Option<Vec<E>>,
    tau1: Vec<E>,
    b: usize,
    alpha: E,
}

impl<E: FieldCore> RingSwitchVerifyCoreOutput<E> {
    fn into_intermediate(self) -> Result<RingSwitchVerifyOutput<E>, AkitaError> {
        let tau0 = self.tau0.ok_or(AkitaError::InvalidProof)?;
        Ok(RingSwitchVerifyOutput {
            relation_matrix_evaluator: self.relation_matrix_evaluator,
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
pub struct RelationMatrixEvaluator<F: FieldCore> {
    pub(crate) role_dims: CommitmentRingDims,
    pub(crate) relation_address_geometry: RelationAddressGeometry,
    pub(crate) groups: Vec<RelationMatrixGroupEvaluator<F>>,
    /// Batch-wide basis used by the shared r-tail.
    pub(crate) log_basis: u32,
    pub(crate) eq_tau1: Arc<[F]>,
    pub(crate) flat_context: Option<FlatRelationContext>,
    pub(crate) setup_plan_cache: Arc<Mutex<Option<CachedSetupContributionPlan<F>>>>,
}

pub(crate) struct CachedSetupContributionPlan<F: FieldCore> {
    x_challenges: Vec<F>,
    plan: SetupContributionPlan<F>,
}

#[derive(Clone)]
pub(crate) struct FlatRelationContext {
    pub(crate) level_params: CommittedGroupParams,
    pub(crate) opening_batch: OpeningClaimsLayout,
    pub(crate) witness_layout: Arc<WitnessLayout>,
    pub(crate) opening_source_len: usize,
    pub(crate) opening_ring_dim: usize,
}

#[derive(Clone)]
pub(crate) struct RelationMatrixGroupEvaluator<F: FieldCore> {
    pub(crate) c_alphas: PreparedChallengeEvals<F>,
    pub(crate) opening_a_evals: Vec<F>,
    pub(crate) group_id: usize,
    pub(crate) num_claims: usize,
    pub(crate) num_live_blocks: usize,
    pub(crate) depth_witness: usize,
    pub(crate) depth_open: usize,
    pub(crate) depth_commit: usize,
    pub(crate) depth_fold: usize,
    pub(crate) log_basis_inner: u32,
    pub(crate) log_basis_outer: u32,
    pub(crate) log_basis_open: u32,
    pub(crate) n_a: usize,
    pub(crate) a_row_start: usize,
    pub(crate) b_row_start: usize,
}

impl<E: FieldCore> RelationMatrixGroupEvaluator<E> {
    fn structured_block_challenges<F>(&self) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: MulBase<F>,
    {
        let capacity = self
            .num_claims
            .checked_mul(self.num_live_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("structured block count overflow".into()))?;
        let mut block_challenges = Vec::with_capacity(capacity);
        for claim in 0..self.num_claims {
            let factors = self
                .c_alphas
                .affine_factors::<F>(claim, self.num_live_blocks)?;
            block_challenges.extend_from_slice(
                factors
                    .low
                    .get(..self.num_live_blocks)
                    .ok_or(AkitaError::InvalidProof)?,
            );
        }
        Ok(block_challenges)
    }
}

/// Fixed public relation inputs for verifier ring-switch replay.
pub struct RingSwitchReplay<'a, F: FieldCore, E> {
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
pub(crate) fn ring_switch_verifier<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    w_len: usize,
    transcript: &mut T,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBaseUnreduced<F>,
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
    // Validate each group's opening/multiplier point against that group's own
    // block geometry (final vs frozen-precommit). For a scalar batch this is the
    // single group at `lp`'s geometry, byte-identical to the historical check.
    for group_index in 0..opening_batch.num_groups() {
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let opening_point = relation.group_opening_point(group_index)?;
        if opening_point.position_weights.len() != group_lp.num_positions_per_block()
            || opening_point.live_block_weights.len() != group_lp.num_live_blocks()
        {
            return Err(AkitaError::InvalidProof);
        }
        let multiplier_point = relation.group_ring_multiplier_point(group_index)?;
        if multiplier_point.position_len() != group_lp.num_positions_per_block()
            || multiplier_point.fold_len() != group_lp.num_live_blocks()
        {
            return Err(AkitaError::InvalidProof);
        }
    }
    if num_polys != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    let relation_address_geometry = lp.relation_address_geometry(
        opening_batch,
        replay.opening_ring_dim,
        replay.opening_source_len,
    )?;
    if w_len == 0
        || !w_len.is_multiple_of(D)
        || w_len != relation_address_geometry.digit_witness_domain().live_len()
    {
        return Err(AkitaError::InvalidProof);
    }
    let num_ring_elems = w_len / D;
    // Bind the shared low coefficient block as the digit-range check's ring
    // phase on every path. On uniform schedules this equals the outgoing
    // witness ring width (byte-identical replay); on non-uniform schedules it
    // mirrors the prover's compact relation-lane split.
    let digit_range_equality_low_variable_count =
        relation_address_geometry.common_relation_witness_variable_count();
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
        prepare_relation_matrix_evaluator::<F, E, D>(replay, alpha, &tau1, Some(num_ring_elems))?;
    RingSwitchVerifyCoreOutput {
        relation_matrix_evaluator,
        relation_address_geometry,
        digit_range_equality_low_variable_count,
        tau0,
        tau1,
        b: 1usize
            .checked_shl(lp.log_basis_open)
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
pub fn prepare_relation_matrix_evaluator<F, E, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    alpha: E,
    tau1: &[E],
    witness_ring_len: Option<usize>,
) -> Result<RelationMatrixEvaluator<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_batch = relation.opening_batch();
    let relation_address_geometry = lp.relation_address_geometry(
        opening_batch,
        replay.opening_ring_dim,
        replay.opening_source_len,
    )?;
    let layout = relation.segment_layout(lp, witness_ring_len)?;
    if layout.total_len() > replay.opening_source_len {
        return Err(AkitaError::InvalidProof);
    }
    let rows = lp.relation_matrix_row_count(opening_batch.num_groups())?;
    if lp.has_precommitted_groups() {
        return prepare_relation_matrix_evaluator_multi_group::<F, E, D>(
            replay,
            alpha,
            tau1,
            layout,
            rows,
            relation_address_geometry,
        );
    }
    let challenges = relation
        .group_challenges()
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    let ring_multiplier_point = relation.group_ring_multiplier_point(0)?;
    prepare_relation_matrix_evaluator_inner::<F, E, D>(
        challenges,
        ring_multiplier_point,
        alpha,
        lp,
        tau1,
        opening_batch,
        replay.row_coefficients,
        layout,
        replay.opening_source_len,
        replay.opening_ring_dim,
        rows,
        relation_address_geometry,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_relation_matrix_evaluator_multi_group<F, E, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    alpha: E,
    tau1: &[E],
    layout: WitnessLayout,
    rows: usize,
    relation_address_geometry: RelationAddressGeometry,
) -> Result<RelationMatrixEvaluator<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_batch = relation.opening_batch();
    lp.validate_opening_batch(opening_batch)?;
    validate_ring_dispatch::<D>()?;
    if relation_address_geometry.carrier_ring_dimension() != D {
        return Err(AkitaError::InvalidSetup(
            "multi-group relation carrier does not match verifier dispatch".into(),
        ));
    }
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

    // Reuse the carrier powers across every uniform group. Mixed-d groups
    // derive their native powers in the dispatch below.
    let carrier_alpha_pows = scalar_powers(alpha, D);
    let mut groups = Vec::with_capacity(order.len());
    for &group_index in &order {
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let group_role_dims = lp.group_role_dims(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        let k_g = group_layout.num_polynomials();
        let num_live_blocks = group_lp.num_live_blocks();
        let num_positions_per_block = group_lp.num_positions_per_block();
        let depth_witness = group_lp.num_digits_inner();
        let depth_commit = group_lp.num_digits_outer();
        let depth_open = group_lp.num_digits_open();
        let depth_fold = group_lp.num_digits_fold();
        let log_basis_inner = group_lp.log_basis_inner();
        let log_basis_outer = group_lp.log_basis_outer();
        let log_basis_open = group_lp.log_basis_open();
        validate_log_basis(log_basis_inner)?;
        validate_log_basis(log_basis_outer)?;
        validate_log_basis(log_basis_open)?;
        let n_a = group_lp.a_rows_len();
        let n_b = group_lp.b_rows_len();
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

        let opening_point = relation.group_opening_point(group_index)?;
        if opening_point.position_weights.len() != num_positions_per_block
            || opening_point.live_block_weights.len() != num_live_blocks
        {
            return Err(AkitaError::InvalidProof);
        }
        let ring_multiplier_point = relation.group_ring_multiplier_point(group_index)?;
        if ring_multiplier_point.position_len() != num_positions_per_block
            || ring_multiplier_point.fold_len() != num_live_blocks
        {
            return Err(AkitaError::InvalidProof);
        }

        let total_blocks = k_g.checked_mul(num_live_blocks).ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group block count overflow".to_string())
        })?;
        let challenges = relation
            .group_challenges()
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        if challenges.logical_len() != total_blocks {
            return Err(AkitaError::InvalidSize {
                expected: total_blocks,
                actual: challenges.logical_len(),
            });
        }
        let (c_alphas, opening_a_evals) = if group_role_dims.d_a() == D {
            // The overwhelmingly common recursive path keeps every group at
            // the carrier dimension. Preserve its monomorphized evaluator:
            // routing it through the mixed-d runtime dispatch measurably
            // increases verifier preparation at every recursive fold.
            let c_alphas = prepare_challenge_evals::<F, E, D>(
                challenges,
                &carrier_alpha_pows,
                k_g,
                num_live_blocks,
            )?;
            let opening_a_evals = (0..num_positions_per_block)
                .map(|idx| ring_multiplier_point.eval_position_at::<D, E>(idx, &carrier_alpha_pows))
                .collect::<Result<Vec<_>, _>>()?;
            (c_alphas, opening_a_evals)
        } else {
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Inner),
                F,
                group_role_dims.d_a(),
                |D_GROUP| {
                    let alpha_pows = scalar_powers(alpha, D_GROUP);
                    let c_alphas = prepare_challenge_evals::<F, E, D_GROUP>(
                        challenges,
                        &alpha_pows,
                        k_g,
                        num_live_blocks,
                    )?;
                    let opening_a_evals = (0..num_positions_per_block)
                        .map(|idx| {
                            ring_multiplier_point.eval_position_at_dyn::<E>(idx, &alpha_pows)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok::<_, AkitaError>((c_alphas, opening_a_evals))
                }
            )?
        };

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
            num_live_blocks,
            depth_witness,
            depth_open,
            depth_commit,
            depth_fold,
            log_basis_inner,
            log_basis_outer,
            log_basis_open,
            n_a,
            a_row_start: a_range.start,
            b_row_start: b_range.start,
        });
    }

    let layout = Arc::new(layout);

    Ok(RelationMatrixEvaluator {
        role_dims: relation.role_dims(),
        relation_address_geometry,
        groups,
        log_basis: lp.log_basis_open,
        eq_tau1,
        flat_context: Some(FlatRelationContext {
            level_params: lp.clone(),
            opening_batch: opening_batch.clone(),
            witness_layout: layout,
            opening_source_len: replay.opening_source_len,
            opening_ring_dim: replay.opening_ring_dim,
        }),
        setup_plan_cache: Default::default(),
    })
}

fn prepare_challenge_evals<F, E, const D: usize>(
    challenges: &Challenges,
    alpha_pows: &[E],
    num_claims: usize,
    num_live_blocks: usize,
) -> Result<PreparedChallengeEvals<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
{
    match challenges {
        Challenges::Sparse {
            challenges: sparse, ..
        } => Ok(PreparedChallengeEvals::Flat(
            sparse
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E>(alpha_pows))
                .collect::<Result<_, _>>()?,
        )),
        Challenges::Tensor { factored } => {
            if D < 2 {
                return Err(AkitaError::InvalidInput(
                    "tensor challenge factored evaluation requires D >= 2".to_string(),
                ));
            }
            factored.validate::<D>()?;
            if factored.num_claims != num_claims {
                return Err(AkitaError::InvalidSize {
                    expected: num_claims,
                    actual: factored.num_claims,
                });
            }
            let num_live_blocks_per_claim = factored.num_live_blocks_per_claim;
            if num_live_blocks_per_claim != num_live_blocks {
                return Err(AkitaError::InvalidSize {
                    expected: num_live_blocks,
                    actual: num_live_blocks_per_claim,
                });
            }
            Ok(PreparedChallengeEvals::Tensor {
                challenges: factored.clone(),
                alpha_pows: alpha_pows.to_vec(),
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_relation_matrix_evaluator_inner<F, E, const D: usize>(
    challenges: &Challenges,
    ring_multiplier_point: &RingMultiplierOpeningPoint<F>,
    alpha: E,
    lp: &CommittedGroupParams,
    tau1: &[E],
    opening_batch: &OpeningClaimsLayout,
    gamma: &[E],
    layout: WitnessLayout,
    opening_source_len: usize,
    opening_ring_dim: usize,
    rows: usize,
    relation_address_geometry: RelationAddressGeometry,
) -> Result<RelationMatrixEvaluator<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
{
    validate_role_dispatch::<D>(lp.role_dims(), RingRole::Inner)?;
    let num_polys = opening_batch.num_total_polynomials();
    let depth_fold = lp.num_digits_fold();
    let alpha_pows = scalar_powers(alpha, D);
    let num_claims = gamma.len();
    if num_polys != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    let log_basis_inner = lp.log_basis_inner;
    let log_basis_outer = lp.log_basis_outer;
    let log_basis_open = lp.log_basis_open;
    validate_log_basis(log_basis_inner)?;
    validate_log_basis(log_basis_outer)?;
    validate_log_basis(log_basis_open)?;
    let depth_witness = lp.num_digits_inner;
    let depth_commit = lp.num_digits_outer;
    let depth_open = lp.num_digits_open;
    let num_live_blocks = lp.num_live_blocks;
    let total_blocks = num_live_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.logical_len() != total_blocks {
        return Err(AkitaError::InvalidSize {
            expected: total_blocks,
            actual: challenges.logical_len(),
        });
    }
    let num_positions_per_block = lp.num_positions_per_block;
    let n_a = lp.inner_commit_matrix.output_rank();

    let c_alphas = prepare_challenge_evals::<F, E, D>(
        challenges,
        &alpha_pows,
        num_claims,
        lp.num_live_blocks,
    )?;
    let opening_a_evals = (0..num_positions_per_block)
        .map(|idx| ring_multiplier_point.eval_position_at::<D, E>(idx, &alpha_pows))
        .collect::<Result<Vec<_>, _>>()?;
    let group = RelationMatrixGroupEvaluator {
        c_alphas,
        opening_a_evals,
        group_id: 0,
        num_claims,
        num_live_blocks,
        depth_witness,
        depth_open,
        depth_commit,
        depth_fold,
        log_basis_inner,
        log_basis_outer,
        log_basis_open,
        n_a,
        a_row_start: 1,
        b_row_start: 1 + n_a,
    };

    let groups = vec![group];
    let layout = Arc::new(layout);
    let eq_tau1: std::sync::Arc<[E]> = EqPolynomial::evals_prefix(tau1, rows)?.into();

    Ok(RelationMatrixEvaluator {
        role_dims: lp.role_dims(),
        relation_address_geometry,
        groups,
        log_basis: log_basis_open,
        eq_tau1,
        flat_context: Some(FlatRelationContext {
            level_params: lp.clone(),
            opening_batch: opening_batch.clone(),
            witness_layout: layout,
            opening_source_len,
            opening_ring_dim,
        }),
        setup_plan_cache: Default::default(),
    })
}

pub(crate) fn setup_contribution_group_inputs<F: FieldCore>(
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

impl<E: FieldCore> RelationMatrixEvaluator<E> {
    /// Evaluate the canonical relation weights directly in the flattened
    /// opening domain, without materializing its padded Boolean suffix.
    pub fn eval_flat_at_point<F, const D: usize>(
        &self,
        point: &[E],
        setup: &AkitaExpandedSetup<F>,
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
    {
        // Uniform-role fast path. This fires whenever all three roles share the
        // dispatch dimension `D`, regardless of the outgoing witness ring
        // (`opening_ring_dim`). When `opening_ring_dim < D` the relation carries
        // `D / opening_ring_dim` lanes per ring element, but for uniform roles
        // the point is laid out `[coeff][lane][column]` with coeff+lane
        // occupying the low `log2(D)` bits (the address is
        // `witness_column·lanes + lane`, coeff below), so the low `log2(D)` bits
        // are exactly the `D`-ring coefficient block. Hence
        // `coefficient_eval(D) = coeff_eval · lane_eval` and the column
        // structure is identical to the `opening_ring_dim == D` case: the
        // succinct evaluator returns the same value the lane-factored mixed scan
        // would, without the explicit O(setup-columns) multiply. (Validated by
        // `mixed_d_per_level_e2e`, whose level-1 fold is a uniform-role,
        // `opening_ring_dim = D/2` step, plus tamper rejection.)
        if self.role_dims == CommitmentRingDims::uniform(D)
            && self
                .relation_address_geometry
                .common_relation_witness_coeff_count()
                == D
        {
            let coefficient_bits = D.trailing_zeros() as usize;
            if point.len() < coefficient_bits {
                return Err(AkitaError::InvalidProof);
            }
            let (coefficient_point, column_point) = point.split_at(coefficient_bits);
            let alpha_evals = scalar_powers(alpha, D);
            let coefficient_eval =
                akita_sumcheck::multilinear_eval(&alpha_evals, coefficient_point)?;
            return Ok(coefficient_eval
                * self.evaluate_uniform_columns_at_point::<F, D>(
                    column_point,
                    setup,
                    alpha,
                    setup_claim,
                )?);
        }
        mixed_relation::evaluate_lane_factored_relation_at_point::<F, E>(
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
        F: FieldCore + CanonicalField,
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
        x_challenges: &[E],
        fold_gadget: Option<&[F]>,
        alpha: E,
    ) -> Result<SetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        let setup_groups = self.setup_contribution_inputs();
        SetupContributionPlan::prepare::<F>(
            &context.level_params,
            &context.opening_batch,
            self.eq_tau1.clone(),
            &context.witness_layout,
            &setup_groups,
            x_challenges,
            fold_gadget,
            self.relation_address_geometry,
            alpha,
        )
    }

    pub(crate) fn setup_contribution_plan_deferred<F>(
        &self,
        x_challenges: &[E],
        fold_gadget: Option<&[F]>,
        alpha: E,
    ) -> Result<SetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        let setup_groups = self.setup_contribution_inputs();
        SetupContributionPlan::prepare_deferred::<F>(
            &context.level_params,
            &context.opening_batch,
            self.eq_tau1.clone(),
            &context.witness_layout,
            &setup_groups,
            x_challenges,
            fold_gadget,
            self.relation_address_geometry,
            alpha,
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn setup_index_weight_evaluator<F>(
        &self,
        plan: &SetupContributionPlan<E>,
        tau1: &[E],
        x_challenges: &[E],
        fold_gadget: &[F],
        alpha: E,
    ) -> Result<Option<akita_types::SetupIndexWeightEvaluator<E>>, AkitaError>
    where
        F: FieldCore,
        E: MulBase<F>,
    {
        let geometry = plan.projection_geometry();
        let base_ring_dim = geometry.base_ring_dim();
        if geometry.role_dims() != CommitmentRingDims::uniform(base_ring_dim)
            || self
                .relation_address_geometry
                .common_relation_witness_coeff_count()
                != base_ring_dim
        {
            return Ok(None);
        }
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        let setup_groups = self.setup_contribution_inputs();
        akita_types::SetupIndexWeightEvaluator::new::<F>(
            plan,
            &context.level_params,
            &context.opening_batch,
            &context.witness_layout,
            context.opening_source_len,
            &setup_groups,
            tau1,
            x_challenges,
            fold_gadget,
            alpha,
        )
        .map(Some)
    }

    pub(crate) fn setup_rows(&self) -> Result<usize, AkitaError> {
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        context
            .level_params
            .relation_matrix_row_count(context.opening_batch.num_groups())
    }

    pub(crate) fn opening_source_len(&self) -> Result<usize, AkitaError> {
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        Ok(context.opening_source_len)
    }

    pub(crate) fn witness_layout(&self) -> Result<&WitnessLayout, AkitaError> {
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        Ok(&context.witness_layout)
    }

    /// Evaluate uniform-role relation columns after the coefficient coordinates
    /// have been contracted by [`Self::eval_flat_at_point`].
    ///
    /// This is the optimized uniform kernel, not a second verifier entry point.
    ///
    /// # Errors
    ///
    /// Returns an error if the setup matrix cannot be viewed at `D` or an
    /// internal offset-eq evaluation receives inconsistent dimensions.
    #[inline]
    fn evaluate_uniform_columns_at_point<F, const D: usize>(
        &self,
        x_challenges: &[E],
        setup: &AkitaExpandedSetup<F>,
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
    {
        let alpha_pows_a = scalar_powers(alpha, D);
        self.eval_at_point_with_alpha_pows::<F, D>(
            x_challenges,
            setup,
            alpha,
            setup_claim,
            &alpha_pows_a,
        )
    }

    #[inline]
    fn eval_at_point_with_alpha_pows<F, const D: usize>(
        &self,
        x_challenges: &[E],
        setup: &AkitaExpandedSetup<F>,
        alpha: E,
        setup_claim: Option<E>,
        alpha_pows_a: &[E],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
    {
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        let _ring_bits = validate_ring_dispatch::<D>()?;
        validate_role_dispatch::<D>(self.role_dims, RingRole::Inner)?;
        if alpha_pows_a.len() != D {
            return Err(AkitaError::InvalidProof);
        }
        let d_b = self.role_dims.d_b();
        let d_d = self.role_dims.d_d();
        let alpha_pows_b_storage;
        let alpha_pows_b: &[E] = if d_b == D {
            alpha_pows_a
        } else {
            alpha_pows_b_storage = scalar_powers(alpha, d_b);
            &alpha_pows_b_storage
        };
        let alpha_pows_d_storage;
        let alpha_pows_d: &[E] = if d_d == D {
            alpha_pows_a
        } else if d_d == d_b {
            alpha_pows_b
        } else {
            alpha_pows_d_storage = scalar_powers(alpha, d_d);
            &alpha_pows_d_storage
        };

        let mut e_structured_contribution = E::zero();
        let mut t_structured_contribution = E::zero();
        let mut z_structured_contribution = E::zero();
        let mut span_structured_contribution = E::zero();
        let setup_groups = self.setup_contribution_inputs();
        let setup_fold_gadget = shared_setup_fold_gadget::<F>(
            &context.level_params,
            &context.opening_batch,
            &setup_groups,
        );
        let shared_log_basis = self.log_basis;
        validate_log_basis(shared_log_basis)?;
        let r_depth = r_decomp_levels::<F>(shared_log_basis);
        let max_shared_depth = self.groups.iter().try_fold(r_depth, |max_depth, group| {
            validate_log_basis(group.log_basis_inner)?;
            validate_log_basis(group.log_basis_outer)?;
            validate_log_basis(group.log_basis_open)?;
            let witness_depth = if group.log_basis_inner == shared_log_basis {
                group.depth_witness
            } else {
                0
            };
            let commit_depth = if group.log_basis_outer == shared_log_basis {
                group.depth_commit
            } else {
                0
            };
            let open_depth = if group.log_basis_open == shared_log_basis {
                group.depth_open
            } else {
                0
            };
            let fold_depth = if group.log_basis_open == shared_log_basis {
                group.depth_fold
            } else {
                0
            };
            Ok(max_depth
                .max(witness_depth)
                .max(commit_depth)
                .max(open_depth)
                .max(fold_depth))
        })?;
        let shared_gadget = gadget_row_scalars::<F>(max_shared_depth, shared_log_basis);
        let shared_gadget_ext = shared_gadget
            .iter()
            .copied()
            .map(E::lift_base)
            .collect::<Vec<_>>();

        // Build the setup-contribution plan in both direct and deferred setup
        // modes. Deferred setup still needs the challenge-derived geometry for
        // stage 3, but it can avoid materializing direct-scan slices and packed
        // scan segments because the setup matrix contribution is supplied as a
        // recursive claim.
        let fold_gadget = setup_fold_gadget.as_deref().unwrap_or(&[]);
        let deferred_setup = setup_claim.is_some();
        let uniform_deferred_setup = self.role_dims == CommitmentRingDims::uniform(D)
            && self
                .relation_address_geometry
                .common_relation_witness_coeff_count()
                == D;
        let setup_plan = {
            let _span = tracing::info_span!("setup_contribution_plan").entered();
            if deferred_setup {
                self.setup_contribution_plan_deferred::<F>(
                    x_challenges,
                    (!fold_gadget.is_empty()).then_some(fold_gadget),
                    alpha,
                )?
            } else {
                self.setup_contribution_plan::<F>(
                    x_challenges,
                    (!fold_gadget.is_empty()).then_some(fold_gadget),
                    alpha,
                )?
            }
        };

        {
            let _span = tracing::info_span!("structured_chunks").entered();
            for (group_index, group) in self.groups.iter().enumerate() {
                let g_open_ext_storage;
                let g_open_ext = if group.log_basis_open == shared_log_basis {
                    shared_gadget_ext
                        .get(..group.depth_open)
                        .ok_or(AkitaError::InvalidProof)?
                } else {
                    g_open_ext_storage =
                        gadget_row_scalars::<F>(group.depth_open, group.log_basis_open)
                            .into_iter()
                            .map(E::lift_base)
                            .collect::<Vec<_>>();
                    &g_open_ext_storage
                };
                let g_t_commit_storage;
                let g_t_commit = if group.log_basis_outer == shared_log_basis {
                    shared_gadget
                        .get(..group.depth_commit)
                        .ok_or(AkitaError::InvalidProof)?
                } else {
                    g_t_commit_storage =
                        gadget_row_scalars::<F>(group.depth_commit, group.log_basis_outer);
                    &g_t_commit_storage
                };
                let g_t_commit_ext_storage;
                let g_t_commit_ext = if group.log_basis_outer == shared_log_basis {
                    shared_gadget_ext
                        .get(..group.depth_commit)
                        .ok_or(AkitaError::InvalidProof)?
                } else {
                    g_t_commit_ext_storage = g_t_commit
                        .iter()
                        .copied()
                        .map(E::lift_base)
                        .collect::<Vec<_>>();
                    &g_t_commit_ext_storage
                };
                let g_witness_storage;
                let g_witness = if group.log_basis_inner == shared_log_basis {
                    shared_gadget
                        .get(..group.depth_witness)
                        .ok_or(AkitaError::InvalidProof)?
                } else {
                    g_witness_storage =
                        gadget_row_scalars::<F>(group.depth_witness, group.log_basis_inner);
                    &g_witness_storage
                };

                let consistency_row = context
                    .level_params
                    .consistency_row_index(&context.opening_batch, group.group_id)?;
                let consistency_weight = *self
                    .eq_tau1
                    .get(consistency_row)
                    .ok_or(AkitaError::InvalidProof)?;
                let a_row_end = group
                    .a_row_start
                    .checked_add(group.n_a)
                    .ok_or_else(|| AkitaError::InvalidSetup("A rows overflow".into()))?;
                let a_row_weights = self
                    .eq_tau1
                    .get(group.a_row_start..a_row_end)
                    .ok_or(AkitaError::InvalidProof)?;
                if deferred_setup {
                    if uniform_deferred_setup {
                        let _span =
                            tracing::info_span!("structured_group_deferred_uniform", group_index)
                                .entered();
                        let (e_contribution, t_contribution, z_contribution) =
                            evaluate_group_structured_from_eq_window::<F, E>(
                                group,
                                consistency_weight,
                                a_row_weights,
                                g_open_ext,
                                g_t_commit_ext,
                                g_witness,
                                setup_plan.eq_window(),
                                context.witness_layout.as_ref(),
                                context.opening_source_len,
                                fold_gadget,
                            )?;
                        e_structured_contribution += e_contribution;
                        t_structured_contribution += t_contribution;
                        z_structured_contribution += z_contribution;
                    } else {
                        let structured_contribution = {
                            let _span =
                                tracing::info_span!("structured_group_deferred", group_index)
                                    .entered();
                            let block_challenges = group.structured_block_challenges::<F>()?;
                            setup_plan.evaluate_structured_group::<F>(
                                group.group_id,
                                &block_challenges,
                                &group.opening_a_evals,
                                alpha,
                            )?
                        };
                        span_structured_contribution += structured_contribution;
                    }
                } else {
                    let (e_eq_slice, t_eq_slice, z_slice) = setup_plan
                        .group_column_eq_slices(group.group_id)
                        .ok_or(AkitaError::InvalidProof)?;
                    let (e_contribution, t_contribution) = {
                        let _span =
                            tracing::info_span!("structured_group_et", group_index).entered();
                        evaluate_group_et_from_eq_slices::<F, E>(
                            group,
                            consistency_weight,
                            a_row_weights,
                            g_open_ext,
                            g_t_commit_ext,
                            e_eq_slice,
                            t_eq_slice,
                        )?
                    };
                    e_structured_contribution += e_contribution;
                    t_structured_contribution += t_contribution;

                    // Reuse the prepared Z equality slice:
                    //   z_structured = Σ_pos Σ_cd z_eq_slice[pos·depth_commit + cd]
                    //                      · consistency · opening_a[pos] · commit_gadget[cd]
                    // The slice is already `-Σ_unit Σ_fold_digit eq · fold_gadget`,
                    // so this is a cheap contraction with no equality evaluation.
                    for (position, &opening_a) in group.opening_a_evals.iter().enumerate() {
                        for (commit_digit, &commit) in g_witness.iter().enumerate() {
                            let col = position
                                .checked_mul(group.depth_witness)
                                .and_then(|base| base.checked_add(commit_digit))
                                .ok_or(AkitaError::InvalidProof)?;
                            let z_eq = *z_slice.get(col).ok_or(AkitaError::InvalidProof)?;
                            z_structured_contribution +=
                                z_eq * consistency_weight * opening_a * E::lift_base(commit);
                        }
                    }
                }
            }
        }

        let setup_contribution = if let Some(claim) = setup_claim {
            claim
        } else {
            let _span = tracing::info_span!("setup_contribution", required = setup_plan.required())
                .entered();
            setup_plan.evaluate_direct::<F>(setup, alpha_pows_a, alpha_pows_b, alpha_pows_d)?
        };

        let r_contribution = {
            let _span = tracing::info_span!("relation_r_contribution").entered();
            let r_gadget = shared_gadget
                .get(..r_depth)
                .ok_or(AkitaError::InvalidProof)?;
            let alpha_pow_d = *alpha_pows_d.get(d_d - 1).ok_or(AkitaError::InvalidProof)?;
            let denom = alpha_pow_d * alpha + E::one();
            let offset_r = context.witness_layout.r_offset();
            compute_r_contribution(
                self,
                x_challenges,
                Some(setup_plan.eq_window()),
                offset_r,
                denom,
                r_gadget,
            )?
        };

        let relation_weight = e_structured_contribution
            + t_structured_contribution
            + z_structured_contribution
            + span_structured_contribution
            + setup_contribution
            + r_contribution;
        self.cache_setup_contribution_plan(x_challenges, setup_plan)?;
        Ok(relation_weight)
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_group_structured_from_eq_window<F, E>(
    group: &RelationMatrixGroupEvaluator<E>,
    consistency_weight: E,
    a_row_weights: &[E],
    g_open_ext: &[E],
    g_t_commit_ext: &[E],
    g_witness: &[F],
    eq_window: &OffsetEqWindow<E>,
    witness_layout: &WitnessLayout,
    opening_source_len: usize,
    fold_gadget: &[F],
) -> Result<(E, E, E), AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F> + MulBase<F>,
{
    if group.num_live_blocks == 0
        || group.depth_witness == 0
        || g_open_ext.len() != group.depth_open
        || g_t_commit_ext.len() != group.depth_commit
        || g_witness.len() != group.depth_witness
        || a_row_weights.len() != group.n_a
        || fold_gadget.len() < group.depth_fold
    {
        return Err(AkitaError::InvalidProof);
    }

    let challenge_factors = (0..group.num_claims)
        .map(|claim| {
            group
                .c_alphas
                .affine_factors::<F>(claim, group.num_live_blocks)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let units = witness_layout.units_for_group(group.group_id)?;
    let e_unit_stride = group.depth_open;
    let t_unit_stride = group
        .n_a
        .checked_mul(group.depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("deferred T fold stride overflow".into()))?;

    let z_cols = group
        .opening_a_evals
        .len()
        .checked_mul(group.depth_witness)
        .ok_or_else(|| AkitaError::InvalidSetup("deferred Z width overflow".into()))?;
    let unit_ranges = units
        .iter()
        .map(|unit| {
            let unit_blocks = unit.num_live_blocks();
            let e_unit_width = unit_blocks
                .checked_mul(e_unit_stride)
                .ok_or_else(|| AkitaError::InvalidSetup("deferred E unit width overflow".into()))?;
            let expected_e = group
                .num_claims
                .checked_mul(e_unit_width)
                .ok_or_else(|| AkitaError::InvalidSetup("deferred E shape overflow".into()))?;
            let e_range = unit.e_range();
            if e_range.len() != expected_e {
                return Err(AkitaError::InvalidSetup(
                    "witness E shape disagrees with resolved range".into(),
                ));
            }
            if e_range.end > opening_source_len {
                return Err(AkitaError::InvalidInput(
                    "physical E opening interval out of range".into(),
                ));
            }

            let t_unit_width = unit_blocks
                .checked_mul(t_unit_stride)
                .ok_or_else(|| AkitaError::InvalidSetup("deferred T unit width overflow".into()))?;
            let expected_t = group
                .num_claims
                .checked_mul(t_unit_width)
                .ok_or_else(|| AkitaError::InvalidSetup("deferred T shape overflow".into()))?;
            let t_range = unit.t_range();
            if t_range.len() != expected_t {
                return Err(AkitaError::InvalidSetup(
                    "witness T shape disagrees with resolved range".into(),
                ));
            }
            if t_range.end > opening_source_len {
                return Err(AkitaError::InvalidInput(
                    "physical T opening interval out of range".into(),
                ));
            }

            let expected_z = z_cols
                .checked_mul(group.depth_fold)
                .ok_or_else(|| AkitaError::InvalidSetup("deferred Z shape overflow".into()))?;
            let z_range = unit.z_range();
            if z_range.len() != expected_z {
                return Err(AkitaError::InvalidSetup(
                    "witness Z shape disagrees with resolved range".into(),
                ));
            }
            if z_range.end > opening_source_len {
                return Err(AkitaError::InvalidInput(
                    "physical Z opening interval out of range".into(),
                ));
            }

            Ok((
                unit.global_block_start(),
                unit_blocks,
                e_range,
                t_range,
                z_range,
            ))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;

    let block_claims = group
        .num_claims
        .checked_mul(group.num_live_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("deferred block count overflow".into()))?;
    let (e_acc, t_acc) = cfg_fold_reduce!(
        0..block_claims,
        || Ok((E::zero(), E::zero())),
        |acc: Result<(E, E), AkitaError>, block_claim| {
            let (mut e_acc, mut t_acc) = acc?;
            let claim = block_claim / group.num_live_blocks;
            let block = block_claim % group.num_live_blocks;
            let factors = challenge_factors
                .get(claim)
                .ok_or(AkitaError::InvalidProof)?;
            let challenge = factors
                .low
                .get(block)
                .copied()
                .ok_or(AkitaError::InvalidProof)?;
            let (unit_start, unit_blocks, e_range, t_range, _) = unit_ranges
                .iter()
                .find(|(start, len, _, _, _)| {
                    start
                        .checked_add(*len)
                        .is_some_and(|end| block >= *start && block < end)
                })
                .ok_or(AkitaError::InvalidProof)?;
            let local_block = block
                .checked_sub(*unit_start)
                .ok_or(AkitaError::InvalidProof)?;

            let e_unit_width = unit_blocks
                .checked_mul(e_unit_stride)
                .ok_or_else(|| AkitaError::InvalidSetup("deferred E unit width overflow".into()))?;
            let e_block_start = e_range
                .start
                .checked_add(claim.checked_mul(e_unit_width).ok_or_else(|| {
                    AkitaError::InvalidSetup("deferred E claim offset overflow".into())
                })?)
                .and_then(|base| {
                    local_block
                        .checked_mul(e_unit_stride)
                        .and_then(|offset| base.checked_add(offset))
                })
                .ok_or_else(|| AkitaError::InvalidSetup("deferred E block overflow".into()))?;
            let mut e_weight = E::zero();
            for (digit, &gadget) in g_open_ext.iter().enumerate() {
                let opening_index = e_block_start
                    .checked_add(digit)
                    .ok_or(AkitaError::InvalidProof)?;
                e_weight += eq_window.eval(opening_index) * gadget;
            }
            e_acc += challenge * consistency_weight * e_weight;

            let t_unit_width = unit_blocks
                .checked_mul(t_unit_stride)
                .ok_or_else(|| AkitaError::InvalidSetup("deferred T unit width overflow".into()))?;
            let t_block_start = t_range
                .start
                .checked_add(claim.checked_mul(t_unit_width).ok_or_else(|| {
                    AkitaError::InvalidSetup("deferred T claim offset overflow".into())
                })?)
                .and_then(|base| {
                    local_block
                        .checked_mul(t_unit_stride)
                        .and_then(|offset| base.checked_add(offset))
                })
                .ok_or_else(|| AkitaError::InvalidSetup("deferred T block overflow".into()))?;
            let mut t_weight = E::zero();
            for (row, &row_weight) in a_row_weights.iter().enumerate() {
                let row_start = t_block_start
                    .checked_add(row.checked_mul(group.depth_commit).ok_or_else(|| {
                        AkitaError::InvalidSetup("deferred T row overflow".into())
                    })?)
                    .ok_or_else(|| AkitaError::InvalidSetup("deferred T row overflow".into()))?;
                for (digit, &gadget) in g_t_commit_ext.iter().enumerate() {
                    let opening_index = row_start
                        .checked_add(digit)
                        .ok_or(AkitaError::InvalidProof)?;
                    t_weight += eq_window.eval(opening_index) * row_weight * gadget;
                }
            }
            t_acc += challenge * t_weight;
            Ok((e_acc, t_acc))
        },
        |lhs: Result<(E, E), AkitaError>, rhs: Result<(E, E), AkitaError>| {
            let (lhs_e, lhs_t) = lhs?;
            let (rhs_e, rhs_t) = rhs?;
            Ok((lhs_e + rhs_e, lhs_t + rhs_t))
        }
    )?;

    let z_acc = cfg_fold_reduce!(
        0..z_cols,
        || Ok(E::zero()),
        |acc: Result<E, AkitaError>, col| {
            let mut acc = acc?;
            let position = col / group.depth_witness;
            let commit_digit = col % group.depth_witness;
            let opening_a = *group
                .opening_a_evals
                .get(position)
                .ok_or(AkitaError::InvalidProof)?;
            let commit = *g_witness
                .get(commit_digit)
                .ok_or(AkitaError::InvalidProof)?;
            let mut z_eq = E::zero();
            for (_, _, _, _, z_range) in &unit_ranges {
                let col_start = col
                    .checked_mul(group.depth_fold)
                    .and_then(|local| z_range.start.checked_add(local))
                    .ok_or_else(|| AkitaError::InvalidSetup("deferred Z source overflow".into()))?;
                for (fold_digit, &fold) in fold_gadget.iter().take(group.depth_fold).enumerate() {
                    let opening_index = col_start
                        .checked_add(fold_digit)
                        .ok_or(AkitaError::InvalidProof)?;
                    z_eq -= eq_window.eval(opening_index).mul_base(fold);
                }
            }
            acc += z_eq * consistency_weight * opening_a * E::lift_base(commit);
            Ok(acc)
        },
        |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
    )?;

    Ok((e_acc, t_acc, z_acc))
}
