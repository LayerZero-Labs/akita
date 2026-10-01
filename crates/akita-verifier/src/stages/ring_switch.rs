//! Ring-switch challenge replay: alpha, tau0, and tau1, plus the prepared
//! relation evaluator and Stage 2 compression state they determine.

use crate::relation::{
    prepare_relation_matrix_evaluator, RelationMatrixEvaluator, RingSwitchReplay,
};
use akita_error::AkitaError;
use akita_types::GrindingReplay;
use akita_types::{
    build_compression_relation_weights, build_reduced_compression_relation_weights,
    CompressionRelationWeights, FpExtEncoding, NegativeBinarySupport,
    ReducedCompressionRelationWeights, RelationAddressGeometry, RelationWitnessGeometry,
    RingRelationGroupOpeningView, RingRelationMode,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub(crate) struct RingSwitchVerifyOutput<E: Field> {
    /// Prepared data for prepared relation-matrix MLE evaluation.
    pub relation_matrix_evaluator: RelationMatrixEvaluator<E>,
    /// Atomic payload-mode state for Stage-2 compression and binary terms.
    pub compression: PreparedStage2Compression<E>,
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

pub(crate) enum PreparedStage2Compression<E: Field> {
    Raw,
    QuotientLift {
        weights: CompressionRelationWeights<E>,
        support: NegativeBinarySupport,
    },
    ReducedEvaluation {
        weights: ReducedCompressionRelationWeights<E>,
        support: NegativeBinarySupport,
    },
}

trait RingSwitchChallengeSource<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    fn alpha(&mut self, level: u32) -> Result<E, AkitaError>;
    fn tau0(&mut self, level: u32, count: usize) -> Result<Vec<E>, AkitaError>;
    fn tau1(&mut self, level: u32, count: usize) -> Result<Vec<E>, AkitaError>;
}

struct RingSwitchChallenges<'a, 'proof, 'plan>(
    &'a mut akita_types::VerifierGrinding<'proof, 'plan>,
);

impl<F, E> RingSwitchChallengeSource<F, E> for RingSwitchChallenges<'_, '_, '_>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    fn alpha(&mut self, level: u32) -> Result<E, AkitaError> {
        self.0
            .grinded_ext_challenge::<F, E>(akita_types::GrindingSite::RingSwitchAlpha { level })
    }

    fn tau0(&mut self, level: u32, count: usize) -> Result<Vec<E>, AkitaError> {
        self.0
            .grinded_ext_challenges::<F, E>(akita_types::GrindingSite::Tau0Point { level }, count)
    }

    fn tau1(&mut self, level: u32, count: usize) -> Result<Vec<E>, AkitaError> {
        self.0
            .grinded_ext_challenges::<F, E>(akita_types::GrindingSite::Tau1Point { level }, count)
    }
}

pub(crate) fn ring_switch_verifier<F, E>(
    replay: &RingSwitchReplay<'_, F, E>,
    w_len: usize,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    level: u32,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + MulBaseUnreduced<F>,
{
    let mut challenges = RingSwitchChallenges(grinding);
    ring_switch_verifier_with_challenges::<F, E, _>(replay, w_len, &mut challenges, level)
}

fn ring_switch_verifier_with_challenges<F, E, C>(
    replay: &RingSwitchReplay<'_, F, E>,
    w_len: usize,
    challenges: &mut C,
    level: u32,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + MulBaseUnreduced<F>,
    C: RingSwitchChallengeSource<F, E>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_batch = relation.opening_batch();
    let num_polys = opening_batch.num_total_polynomials();
    let gamma = replay.row_coefficients;

    let alpha: E = {
        let _span = tracing::info_span!("ring_switch_transcript_challenges").entered();
        challenges.alpha(level)?
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
        let tau0 = challenges.tau0(level, num_sc_vars)?;
        let tau1 = challenges.tau1(level, num_i)?;
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
    let compression = if lp.payload_mode.is_compressed() {
        let support = NegativeBinarySupport::new(&witness_layout, physical_field_len)?;
        match lp.ring_relation_mode {
            RingRelationMode::QuotientLift => PreparedStage2Compression::QuotientLift {
                weights: build_compression_relation_weights(
                    replay.setup,
                    relation,
                    alpha,
                    lp,
                    &tau1,
                    &witness_layout,
                    replay.opening_ring_dim,
                    physical_field_len,
                )?,
                support,
            },
            RingRelationMode::ReducedEvaluation => PreparedStage2Compression::ReducedEvaluation {
                weights: build_reduced_compression_relation_weights(
                    alpha,
                    lp,
                    opening_batch,
                    relation.extension_degree(),
                    &tau1,
                    &witness_layout,
                    replay.opening_ring_dim,
                    physical_field_len,
                )?,
                support,
            },
        }
    } else {
        PreparedStage2Compression::Raw
    };
    Ok(RingSwitchVerifyOutput {
        relation_matrix_evaluator,
        compression,
        relation_address_geometry,
        digit_range_equality_low_variable_count,
        tau0,
        tau1,
        b: 1usize
            .checked_shl(lp.open().digits.log_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("basis size overflow".to_string()))?,
        alpha,
    })
}
