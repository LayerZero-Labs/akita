//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::{eval_affine_digit_interval, OffsetEqWindow};
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
    eval_relation_weight_at_point, gadget_row_scalars, r_decomp_levels, shared_setup_fold_gadget,
    validate_role_dispatch, AkitaExpandedSetup, CommitmentRingDims, FpExtEncoding, LevelParams,
    OpeningClaimsLayout, RelationMatrixRowLayout, RingMultiplierOpeningPoint, RingRelationInstance,
    RingRole, SetupContributionGroupInputs, SetupContributionPlan, WitnessLayout,
    WitnessUnitLayout,
};
use std::sync::Arc;

use super::slice_mle::compute_r_contribution;
use super::validate_log_basis;
use akita_types::validate_ring_dispatch;
pub(crate) use tensor_challenges::PreparedChallengeEvals;

mod tensor_challenges;
#[cfg(test)]
mod tests;

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub(crate) struct RingSwitchVerifyOutput<E: FieldCore> {
    /// Prepared data for prepared relation-matrix MLE evaluation.
    pub relation_matrix_evaluator: RelationMatrixEvaluator<E>,
    /// Number of upper variable bits.
    pub col_bits: usize,
    /// Number of lower variable bits.
    pub ring_bits: usize,
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
    col_bits: usize,
    ring_bits: usize,
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
            col_bits: self.col_bits,
            ring_bits: self.ring_bits,
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
    pub(crate) groups: Vec<RelationMatrixGroupEvaluator<F>>,
    /// Batch-wide basis used by the shared r-tail.
    pub(crate) log_basis_open: u32,
    pub(crate) eq_tau1: Arc<[F]>,
    pub(crate) flat_context: Option<FlatRelationContext<F>>,
}

#[derive(Clone)]
pub(crate) struct FlatRelationContext<F: FieldCore> {
    pub(crate) level_params: LevelParams,
    pub(crate) opening_batch: OpeningClaimsLayout,
    pub(crate) witness_layout: Arc<WitnessLayout>,
    pub(crate) opening_source_len: usize,
    pub(crate) row_coefficients: Vec<F>,
    pub(crate) tau1: Vec<F>,
    pub(crate) relation_matrix_row_layout: RelationMatrixRowLayout,
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
    pub(crate) depth_commit: usize,
    pub(crate) depth_open: usize,
    pub(crate) depth_fold: usize,
    pub(crate) log_basis_inner: u32,
    pub(crate) log_basis_outer: u32,
    pub(crate) log_basis_open: u32,
    pub(crate) n_a: usize,
    pub(crate) a_row_start: usize,
    pub(crate) b_row_start: usize,
}

/// Fixed public relation inputs for verifier ring-switch replay.
pub struct RingSwitchReplay<'a, F: FieldCore, E> {
    pub setup: &'a AkitaExpandedSetup<F>,
    pub relation: &'a RingRelationInstance<F>,
    pub row_coefficients: &'a [E],
    pub lp: &'a LevelParams,
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
    relation_matrix_row_layout: RelationMatrixRowLayout,
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

    let opening_capacity = replay
        .opening_source_len
        .checked_mul(replay.opening_ring_dim)
        .ok_or(AkitaError::InvalidProof)?;
    if w_len == 0
        || !w_len.is_multiple_of(D)
        || replay.opening_ring_dim == 0
        || !replay.opening_ring_dim.is_power_of_two()
        || !w_len.is_multiple_of(replay.opening_ring_dim)
        || w_len > opening_capacity
    {
        return Err(AkitaError::InvalidProof);
    }
    let num_ring_elems = w_len / D;
    let opening_ring_dim = replay.opening_ring_dim;
    let x_capacity = akita_types::opening_domain_len(replay.opening_source_len)?;
    // Mirror the prover's ring-switch geometry (see `ring_switch_finalize`):
    // uniform role dimensions expose the separable (x, y) opening domain, while
    // non-uniform roles fall back to the flattened single domain.
    let uniform = relation.role_dims() == CommitmentRingDims::uniform(opening_ring_dim);
    let (col_bits, ring_bits) = if uniform {
        (
            x_capacity.trailing_zeros() as usize,
            opening_ring_dim.trailing_zeros() as usize,
        )
    } else {
        let opening_field_len = x_capacity
            .checked_mul(opening_ring_dim)
            .ok_or(AkitaError::InvalidProof)?;
        (opening_field_len.trailing_zeros() as usize, 0usize)
    };
    let num_sc_vars = col_bits + ring_bits;
    let num_i =
        lp.relation_row_index_num_vars_for_layout(relation_matrix_row_layout, opening_batch)?;

    let (tau0, tau1) = {
        let _span = tracing::info_span!(
            "ring_switch_transcript_challenges",
            tau0_len = num_sc_vars,
            tau1_len = num_i
        )
        .entered();
        let tau0 = match relation_matrix_row_layout {
            RelationMatrixRowLayout::WithDBlock => Some(
                (0..num_sc_vars)
                    .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
                    .collect(),
            ),
            RelationMatrixRowLayout::WithoutCommitmentBlocks => None,
        };
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
        col_bits,
        ring_bits,
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
    let layout = relation.segment_layout(lp, witness_ring_len)?;
    if layout.total_len() > replay.opening_source_len {
        return Err(AkitaError::InvalidProof);
    }
    reject_mixed_d_multi_chunk::<D>(lp.role_dims(), &layout, "prepare_relation_matrix_evaluator")?;
    let opening_batch = relation.opening_batch();
    let rows = lp.relation_matrix_row_count_for(
        opening_batch.num_groups(),
        relation.relation_matrix_row_layout(),
    )?;
    if lp.has_precommitted_groups() {
        return prepare_relation_matrix_evaluator_multi_group::<F, E, D>(
            replay, alpha, tau1, layout, rows,
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
        relation.relation_matrix_row_layout(),
        layout,
        replay.opening_source_len,
        replay.opening_ring_dim,
        rows,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_relation_matrix_evaluator_multi_group<F, E, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E>,
    alpha: E,
    tau1: &[E],
    layout: WitnessLayout,
    rows: usize,
) -> Result<RelationMatrixEvaluator<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_batch = relation.opening_batch();
    lp.validate_opening_batch(opening_batch)?;
    validate_role_dispatch::<D>(relation.role_dims(), RingRole::Inner)?;
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

    let alpha_pows_a = scalar_powers(alpha, D);
    let mut groups = Vec::with_capacity(order.len());
    for &group_index in &order {
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        let k_g = group_layout.num_polynomials();
        let num_live_blocks = group_lp.num_live_blocks();
        let num_positions_per_block = group_lp.num_positions_per_block();
        let depth_witness = group_lp.num_digits_inner();
        let depth_commit = group_lp.num_digits_outer();
        let depth_open = group_lp.num_digits_open();
        let depth_fold = lp.num_digits_fold_for_params(group_lp, k_g, lp.field_bits_for_cache())?;
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
        let c_alphas =
            prepare_challenge_evals::<F, E, D>(challenges, &alpha_pows_a, k_g, num_live_blocks)?;
        let opening_a_evals = (0..num_positions_per_block)
            .map(|idx| ring_multiplier_point.eval_position_at_dyn::<E>(idx, &alpha_pows_a))
            .collect::<Result<Vec<_>, _>>()?;

        let a_range = lp.a_row_range(
            opening_batch,
            group_index,
            relation.relation_matrix_row_layout(),
        )?;
        let b_range = lp.commitment_row_range(
            opening_batch,
            group_index,
            relation.relation_matrix_row_layout(),
        )?;
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
            depth_commit,
            depth_open,
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
        groups,
        log_basis_open: lp.log_basis_open,
        eq_tau1,
        flat_context: Some(FlatRelationContext {
            level_params: lp.clone(),
            opening_batch: opening_batch.clone(),
            witness_layout: layout,
            opening_source_len: replay.opening_source_len,
            row_coefficients: replay.row_coefficients.to_vec(),
            tau1: tau1.to_vec(),
            relation_matrix_row_layout: relation.relation_matrix_row_layout(),
            opening_ring_dim: replay.opening_ring_dim,
        }),
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
    lp: &LevelParams,
    tau1: &[E],
    opening_batch: &OpeningClaimsLayout,
    gamma: &[E],
    relation_matrix_row_layout: RelationMatrixRowLayout,
    layout: WitnessLayout,
    opening_source_len: usize,
    opening_ring_dim: usize,
    rows: usize,
) -> Result<RelationMatrixEvaluator<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
{
    validate_role_dispatch::<D>(lp.role_dims, RingRole::Inner)?;
    let num_polys = opening_batch.num_total_polynomials();
    let depth_fold = lp.num_digits_fold(num_polys, lp.field_bits_for_cache())?;
    reject_mixed_d_multi_chunk::<D>(lp.role_dims, &layout, "prepare_relation_matrix_evaluator")?;
    let alpha_pows = scalar_powers(alpha, D);
    let num_claims = gamma.len();
    if num_polys != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    let log_basis = lp.log_basis_open;
    validate_log_basis(log_basis)?;
    validate_log_basis(lp.log_basis_inner)?;
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
    let n_a = lp.a_key.row_len();

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
        depth_commit,
        depth_open,
        depth_fold,
        log_basis_inner: lp.log_basis_inner,
        log_basis_outer: lp.log_basis_outer,
        log_basis_open: log_basis,
        n_a,
        a_row_start: 1,
        b_row_start: 1 + n_a,
    };

    let groups = vec![group];
    let layout = Arc::new(layout);
    let eq_tau1: std::sync::Arc<[E]> = EqPolynomial::evals_prefix(tau1, rows)?.into();

    Ok(RelationMatrixEvaluator {
        role_dims: lp.role_dims,
        groups,
        log_basis_open: log_basis,
        eq_tau1,
        flat_context: Some(FlatRelationContext {
            level_params: lp.clone(),
            opening_batch: opening_batch.clone(),
            witness_layout: layout,
            opening_source_len,
            row_coefficients: gamma.to_vec(),
            tau1: tau1.to_vec(),
            relation_matrix_row_layout,
            opening_ring_dim,
        }),
    })
}

fn reject_mixed_d_multi_chunk<const D: usize>(
    role_dims: CommitmentRingDims,
    layout: &WitnessLayout,
    context: &str,
) -> Result<(), AkitaError> {
    if layout.units().len() > layout.num_groups() && (role_dims.d_b() != D || role_dims.d_d() != D)
    {
        return Err(AkitaError::InvalidSetup(format!(
            "{context}: multi-chunk witness layout requires uniform ring dimensions across roles"
        )));
    }
    Ok(())
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
        instance: &RingRelationInstance<F>,
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
    {
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        if context.opening_ring_dim == D && self.role_dims == CommitmentRingDims::uniform(D) {
            let coefficient_bits = D.trailing_zeros() as usize;
            if point.len() < coefficient_bits {
                return Err(AkitaError::InvalidProof);
            }
            let (coefficient_point, column_point) = point.split_at(coefficient_bits);
            let alpha_evals = scalar_powers(alpha, D);
            let coefficient_eval =
                akita_sumcheck::multilinear_eval(&alpha_evals, coefficient_point)?;
            return Ok(coefficient_eval
                * self.eval_at_point_with_alpha_pows::<F, D>(
                    column_point,
                    setup,
                    alpha,
                    setup_claim,
                    &alpha_evals,
                )?);
        }
        if setup_claim.is_some() {
            return Err(AkitaError::InvalidProof);
        }
        eval_relation_weight_at_point::<F, E>(
            setup,
            instance,
            alpha,
            &scalar_powers(alpha, D),
            self.role_dims,
            &context.level_params,
            &context.tau1,
            &context.row_coefficients,
            context.relation_matrix_row_layout,
            context.opening_source_len,
            context.opening_ring_dim,
            point,
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
            context.relation_matrix_row_layout,
            self.eq_tau1.clone(),
            &context.witness_layout,
            context.opening_source_len,
            &setup_groups,
            x_challenges,
            fold_gadget,
            self.role_dims,
        )
    }

    pub(crate) fn setup_index_weight_evaluator<F>(
        &self,
        plan: &SetupContributionPlan<E>,
        tau1: &[E],
        x_challenges: &[E],
        fold_gadget: &[F],
        alpha: E,
    ) -> Result<akita_types::SetupIndexWeightEvaluator<E>, AkitaError>
    where
        F: FieldCore,
        E: MulBase<F>,
    {
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        let setup_groups = self.setup_contribution_inputs();
        akita_types::SetupIndexWeightEvaluator::new::<F>(
            plan,
            &context.level_params,
            &context.opening_batch,
            context.relation_matrix_row_layout,
            &context.witness_layout,
            context.opening_source_len,
            &setup_groups,
            tau1,
            x_challenges,
            fold_gadget,
            alpha,
        )
    }

    pub(crate) fn setup_rows(&self) -> Result<usize, AkitaError> {
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        context.level_params.relation_matrix_row_count_for(
            context.opening_batch.num_groups(),
            context.relation_matrix_row_layout,
        )
    }

    pub(crate) fn opening_source_len(&self) -> Result<usize, AkitaError> {
        let context = self.flat_context.as_ref().ok_or(AkitaError::InvalidProof)?;
        Ok(context.opening_source_len)
    }

    fn group_units<'a>(
        &self,
        context: &'a FlatRelationContext<E>,
        group: &RelationMatrixGroupEvaluator<E>,
    ) -> Result<Vec<&'a WitnessUnitLayout>, AkitaError> {
        context.witness_layout.units_for_group(group.group_id)
    }

    /// Evaluate the relation matrix at a point at the supplied point.
    ///
    /// # Errors
    ///
    /// Returns an error if the setup matrix cannot be viewed at `D` or an
    /// internal offset-eq evaluation receives inconsistent dimensions.
    #[inline]
    pub fn eval_at_point<F, const D: usize>(
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
        let shared_open_log_basis = self.log_basis_open;
        validate_log_basis(shared_open_log_basis)?;
        let r_depth = r_decomp_levels::<F>(shared_open_log_basis);
        let (max_depth_open, max_depth_fold, max_shared_depth_commit, max_shared_depth_witness) =
            self.groups.iter().try_fold(
                (0usize, 0usize, 0usize, 0usize),
                |(max_open, max_fold, max_commit, max_witness), group| {
                    validate_log_basis(group.log_basis_open)?;
                    validate_log_basis(group.log_basis_inner)?;
                    validate_log_basis(group.log_basis_outer)?;
                    if group.log_basis_open != shared_open_log_basis {
                        return Err(AkitaError::InvalidSetup(
                            "relation matrix group opening basis does not match root basis".into(),
                        ));
                    }
                    Ok((
                        max_open.max(group.depth_open),
                        max_fold.max(group.depth_fold),
                        if group.log_basis_outer == shared_open_log_basis {
                            max_commit.max(group.depth_commit)
                        } else {
                            max_commit
                        },
                        if group.log_basis_inner == shared_open_log_basis {
                            max_witness.max(group.depth_witness)
                        } else {
                            max_witness
                        },
                    ))
                },
            )?;
        let shared_gadget_depth = max_depth_open
            .max(max_depth_fold)
            .max(max_shared_depth_commit)
            .max(max_shared_depth_witness)
            .max(r_depth);
        let shared_gadget = gadget_row_scalars::<F>(shared_gadget_depth, shared_open_log_basis);
        let shared_gadget_ext = shared_gadget
            .iter()
            .copied()
            .map(E::lift_base)
            .collect::<Vec<_>>();
        let setup_fold_gadget = shared_gadget
            .get(..max_depth_fold)
            .ok_or(AkitaError::InvalidProof)?;

        // In direct setup mode, build the setup-contribution plan up front. Its
        // prepared Z equality slice (`z_eq_slice`, built in parallel and already
        // contracted over units and fold digits) is the same equality
        // evaluation the structured Z contribution needs, so reusing it removes
        // a second, serial pass over the Z addresses (Fix 6). In tensor mode the
        // setup contribution is supplied as `setup_claim`, so no plan is built
        // and the structured Z contribution falls back to a direct evaluation.
        let setup_plan = if setup_claim.is_none() {
            Some(self.setup_contribution_plan::<F>(
                x_challenges,
                (!setup_fold_gadget.is_empty()).then_some(setup_fold_gadget),
            )?)
        } else {
            None
        };

        {
            let _span = tracing::info_span!("structured_chunks").entered();
            // Bounded equality window for the serial Z fallback (tensor mode).
            let x_eq_window = setup_plan
                .is_none()
                .then(|| OffsetEqWindow::new(x_challenges))
                .transpose()?;
            for (group_index, group) in self.groups.iter().enumerate() {
                let units = self.group_units(context, group)?;

                let group_g_open_ext = shared_gadget_ext
                    .get(..group.depth_open)
                    .ok_or(AkitaError::InvalidProof)?;
                let g_t_commit_storage;
                let g_t_commit = if group.log_basis_outer == shared_open_log_basis {
                    shared_gadget
                        .get(..group.depth_commit)
                        .ok_or(AkitaError::InvalidProof)?
                } else {
                    g_t_commit_storage =
                        gadget_row_scalars::<F>(group.depth_commit, group.log_basis_outer);
                    &g_t_commit_storage
                };
                let g_t_commit_ext_storage;
                let g_t_commit_ext = if group.log_basis_outer == shared_open_log_basis {
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
                let g_witness = if group.log_basis_inner == shared_open_log_basis {
                    shared_gadget
                        .get(..group.depth_witness)
                        .ok_or(AkitaError::InvalidProof)?
                } else {
                    g_witness_storage =
                        gadget_row_scalars::<F>(group.depth_witness, group.log_basis_inner);
                    &g_witness_storage
                };

                let consistency_weight = self.eq_tau1[0];
                let a_row_end = group
                    .a_row_start
                    .checked_add(group.n_a)
                    .ok_or_else(|| AkitaError::InvalidSetup("A rows overflow".into()))?;
                let a_row_weights = self
                    .eq_tau1
                    .get(group.a_row_start..a_row_end)
                    .ok_or(AkitaError::InvalidProof)?;
                let (e_contribution, t_contribution) = if let Some(plan) = setup_plan.as_ref() {
                    let (e_eq_slice, t_eq_slice, _) = plan
                        .group_column_eq_slices(group_index)
                        .ok_or(AkitaError::InvalidProof)?;
                    evaluate_group_et_from_eq_slices::<F, E>(
                        group,
                        consistency_weight,
                        a_row_weights,
                        group_g_open_ext,
                        g_t_commit_ext,
                        e_eq_slice,
                        t_eq_slice,
                    )?
                } else {
                    evaluate_group_et_contributions::<F, E>(
                        group,
                        &units,
                        context.opening_source_len,
                        x_challenges,
                        consistency_weight,
                        a_row_weights,
                        group_g_open_ext,
                        g_t_commit_ext,
                    )?
                };
                e_structured_contribution += e_contribution;
                t_structured_contribution += t_contribution;

                if let Some(plan) = setup_plan.as_ref() {
                    // Reuse the prepared Z equality slice:
                    //   z_structured = Σ_pos Σ_cd z_eq_slice[pos·depth_commit + cd]
                    //                      · consistency · opening_a[pos] · commit_gadget[cd]
                    // The slice is already `-Σ_unit Σ_fold_digit eq · fold_gadget`,
                    // so this is a cheap contraction with no equality evaluation.
                    let (_, _, z_slice) = plan
                        .group_column_eq_slices(group_index)
                        .ok_or(AkitaError::InvalidProof)?;
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
                } else {
                    let fold_gadget = shared_gadget
                        .get(..group.depth_fold)
                        .ok_or(AkitaError::InvalidProof)?;
                    for unit in units {
                        for (position, &opening_a) in group.opening_a_evals.iter().enumerate() {
                            for (commit_digit, &commit) in g_witness.iter().enumerate() {
                                let mut z_weight = E::zero();
                                for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
                                    let z_index = unit.z_index(
                                        group.opening_a_evals.len(),
                                        group.depth_witness,
                                        group.depth_fold,
                                        position,
                                        commit_digit,
                                        fold_digit,
                                    )?;
                                    let z_opening_index =
                                        akita_types::checked_opening_source_index(
                                            context.opening_source_len,
                                            z_index,
                                        )?;
                                    let x_eq_window =
                                        x_eq_window.as_ref().ok_or(AkitaError::InvalidProof)?;
                                    z_weight -=
                                        x_eq_window.eval(z_opening_index) * E::lift_base(fold);
                                }
                                z_structured_contribution += z_weight
                                    * consistency_weight
                                    * opening_a
                                    * E::lift_base(commit);
                            }
                        }
                    }
                }
            }
        }

        let setup_contribution = if let Some(claim) = setup_claim {
            claim
        } else {
            let _span = tracing::info_span!(
                "setup_contribution",
                required = setup_plan
                    .as_ref()
                    .map_or(0, SetupContributionPlan::required)
            )
            .entered();
            let plan = setup_plan.as_ref().ok_or(AkitaError::InvalidProof)?;
            plan.evaluate_direct::<F>(setup, alpha_pows_a, alpha_pows_b, alpha_pows_d)?
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
                setup_plan.as_ref().map(SetupContributionPlan::eq_window),
                offset_r,
                denom,
                r_gadget,
            )?
        };

        Ok(e_structured_contribution
            + t_structured_contribution
            + z_structured_contribution
            + setup_contribution
            + r_contribution)
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_group_et_from_eq_slices<F, E>(
    group: &RelationMatrixGroupEvaluator<E>,
    consistency_weight: E,
    a_row_weights: &[E],
    g_open_ext: &[E],
    g_t_commit_ext: &[E],
    e_eq_slice: &[E],
    t_eq_slice: &[E],
) -> Result<(E, E), AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FieldCore + MulBase<F>,
{
    let e_stride = group.depth_open;
    let t_stride = group
        .n_a
        .checked_mul(group.depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("T fold stride overflow".into()))?;
    let block_claims = group
        .num_claims
        .checked_mul(group.num_live_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("structured block count overflow".into()))?;
    let expected_e = block_claims
        .checked_mul(e_stride)
        .ok_or_else(|| AkitaError::InvalidSetup("structured E width overflow".into()))?;
    let expected_t = block_claims
        .checked_mul(t_stride)
        .ok_or_else(|| AkitaError::InvalidSetup("structured T width overflow".into()))?;
    if e_eq_slice.len() != expected_e
        || t_eq_slice.len() != expected_t
        || g_open_ext.len() != group.depth_open
        || g_t_commit_ext.len() != group.depth_commit
        || a_row_weights.len() != group.n_a
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
    let t_weights = a_row_weights
        .iter()
        .flat_map(|&row_weight| {
            g_t_commit_ext
                .iter()
                .map(move |&gadget| row_weight * gadget)
        })
        .collect::<Vec<_>>();
    let _span = tracing::info_span!(
        "structured_et_from_setup_slices",
        block_claims,
        e_columns = expected_e,
        t_columns = expected_t
    )
    .entered();
    cfg_fold_reduce!(
        0..block_claims,
        || Ok((E::zero(), E::zero())),
        |acc: Result<(E, E), AkitaError>, block_claim| {
            let (mut e_acc, mut t_acc) = acc?;
            let claim = block_claim / group.num_live_blocks;
            let block = block_claim % group.num_live_blocks;
            let challenge = challenge_factors
                .get(claim)
                .and_then(|factors| factors.low.get(block))
                .copied()
                .ok_or(AkitaError::InvalidProof)?;
            let e_start = block_claim
                .checked_mul(e_stride)
                .ok_or(AkitaError::InvalidProof)?;
            let e_end = e_start
                .checked_add(e_stride)
                .ok_or(AkitaError::InvalidProof)?;
            let e_block = e_eq_slice
                .get(e_start..e_end)
                .ok_or(AkitaError::InvalidProof)?;
            let mut e_weight = E::zero();
            for (&eq, &gadget) in e_block.iter().zip(g_open_ext) {
                e_weight += eq * gadget;
            }
            e_acc += challenge * consistency_weight * e_weight;

            let t_start = block_claim
                .checked_mul(t_stride)
                .ok_or(AkitaError::InvalidProof)?;
            let t_end = t_start
                .checked_add(t_stride)
                .ok_or(AkitaError::InvalidProof)?;
            let t_block = t_eq_slice
                .get(t_start..t_end)
                .ok_or(AkitaError::InvalidProof)?;
            let mut t_weight = E::zero();
            for (&eq, &weight) in t_block.iter().zip(&t_weights) {
                t_weight += eq * weight;
            }
            t_acc += challenge * t_weight;
            Ok((e_acc, t_acc))
        },
        |lhs: Result<(E, E), AkitaError>, rhs: Result<(E, E), AkitaError>| {
            let (lhs_e, lhs_t) = lhs?;
            let (rhs_e, rhs_t) = rhs?;
            Ok((lhs_e + rhs_e, lhs_t + rhs_t))
        }
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_group_et_contributions<F, E>(
    group: &RelationMatrixGroupEvaluator<E>,
    units: &[&WitnessUnitLayout],
    opening_source_len: usize,
    x_challenges: &[E],
    consistency_weight: E,
    a_row_weights: &[E],
    g_open_ext: &[E],
    g_t_commit_ext: &[E],
) -> Result<(E, E), AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FieldCore + MulBase<F>,
{
    let t_fold_stride = group
        .n_a
        .checked_mul(group.depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("T fold stride overflow".into()))?;
    let claim_factors = (0..group.num_claims)
        .map(|claim| {
            group
                .c_alphas
                .affine_factors::<F>(claim, group.num_live_blocks)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let e_digit_weights = g_open_ext
        .iter()
        .map(|&gadget| consistency_weight * gadget)
        .collect::<Vec<_>>();
    // T is laid out block-major, then A-row, then opening digit. Contract the
    // contiguous `(A row, digit)` lane in one affine evaluation instead of
    // rebuilding the same low equality table once per A row.
    let t_block_weights = a_row_weights
        .iter()
        .flat_map(|&row_weight| {
            g_t_commit_ext
                .iter()
                .map(move |&gadget| row_weight * gadget)
        })
        .collect::<Vec<_>>();
    let mut e_contribution = E::zero();
    let mut t_contribution = E::zero();
    for unit in units {
        for (claim, factors) in claim_factors.iter().enumerate() {
            let e_index = unit.e_index(
                group.num_claims,
                group.depth_open,
                claim,
                unit.global_block_start(),
                0,
            )?;
            let e_opening_index =
                akita_types::checked_opening_source_index(opening_source_len, e_index)?;
            e_contribution += eval_affine_digit_interval(
                x_challenges,
                e_opening_index,
                unit.global_block_start(),
                unit.num_live_blocks(),
                group.depth_open,
                &e_digit_weights,
                &factors.high,
                &factors.low,
            )?;

            let t_index = unit.t_index(
                group.num_claims,
                group.n_a,
                group.depth_commit,
                claim,
                unit.global_block_start(),
                0,
                0,
            )?;
            let t_opening_index =
                akita_types::checked_opening_source_index(opening_source_len, t_index)?;
            t_contribution += eval_affine_digit_interval(
                x_challenges,
                t_opening_index,
                unit.global_block_start(),
                unit.num_live_blocks(),
                t_fold_stride,
                &t_block_weights,
                &factors.high,
                &factors.low,
            )?;
        }
    }
    Ok((e_contribution, t_contribution))
}
