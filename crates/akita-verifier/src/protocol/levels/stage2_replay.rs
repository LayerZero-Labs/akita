use super::Stage1Replay;
use crate::protocol::ring_switch::RingSwitchVerifyOutput;
use crate::stages::stage2::{AkitaStage2Verifier, Stage2RowEvalSource, Stage2WitnessOracle};
use crate::stages::SetupSumcheckVerifier;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
#[cfg(feature = "zk")]
use akita_r1cs::{zk_ext_mask_lc_at, ZkR1csLinearCombination, ZkRelationAccumulator};
use akita_serialization::AkitaSerialize;
#[cfg(not(feature = "zk"))]
use akita_sumcheck::SumcheckInstanceVerifierExt;
#[cfg(feature = "zk")]
use akita_sumcheck::ZkSumcheckInstanceVerifierExt;
use akita_transcript::labels::{ABSORB_STAGE2_NEXT_W_EVAL, CHALLENGE_SUMCHECK_ROUND};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    AkitaVerifierSetup, CleartextWitnessProof, LevelParams, RingMultiplierOpeningPoint,
    RingSubfieldEncoding, SetupContributionMode, SetupSumcheckProof,
};

pub(super) enum Stage2ProofReplay<'a, F: FieldCore, E: FieldCore> {
    Intermediate {
        next_w_eval: E,
        #[cfg(not(feature = "zk"))]
        sumcheck: &'a akita_sumcheck::SumcheckProof<E>,
        #[cfg(feature = "zk")]
        sumcheck_masked: &'a akita_sumcheck::SumcheckProofMasked<E>,
    },
    Terminal {
        final_witness: &'a CleartextWitnessProof<F>,
        physical_w_len: usize,
        #[cfg(not(feature = "zk"))]
        sumcheck: &'a akita_sumcheck::SumcheckProof<E>,
        #[cfg(feature = "zk")]
        sumcheck_masked: &'a akita_sumcheck::SumcheckProofMasked<E>,
    },
}

pub(super) struct Stage2ReplayInput<'a, F: FieldCore, E: FieldCore, const D: usize> {
    pub(super) setup: &'a AkitaVerifierSetup<F>,
    pub(super) stage2: Stage2ProofReplay<'a, F, E>,
    pub(super) stage1: Stage1Replay<E>,
    pub(super) rs: RingSwitchVerifyOutput<E>,
    pub(super) relation_claim: E,
    #[cfg(feature = "zk")]
    pub(super) relation_claim_mask: ZkR1csLinearCombination<E>,
    pub(super) setup_sumcheck_proof: Option<&'a SetupSumcheckProof<E>>,
    pub(super) next_fold_level_params: &'a LevelParams,
    pub(super) ring_multiplier_points: &'a [RingMultiplierOpeningPoint<F, D>],
}

pub(super) fn stage3_sumcheck_proof_for_mode<L: FieldCore>(
    mode: SetupContributionMode,
    stage3_sumcheck_proof: Option<&SetupSumcheckProof<L>>,
) -> Result<Option<&SetupSumcheckProof<L>>, AkitaError> {
    match (mode, stage3_sumcheck_proof) {
        (SetupContributionMode::Direct, None) => Ok(None),
        (SetupContributionMode::Direct, Some(_)) => Err(AkitaError::InvalidSetup(
            "direct setup-contribution mode received stage3_sumcheck_proof".to_string(),
        )),
        (SetupContributionMode::Recursive, Some(proof)) => Ok(Some(proof)),
        (SetupContributionMode::Recursive, None) => Err(AkitaError::InvalidSetup(
            "recursive setup-contribution mode is missing stage3_sumcheck_proof".to_string(),
        )),
    }
}

pub(super) fn verify_stage2_and_setup_replay<F, E, T, const D: usize>(
    transcript: &mut T,
    input: Stage2ReplayInput<'_, F, E, D>,
    #[cfg(feature = "zk")] zk_hiding_cursor: &mut usize,
    #[cfg(feature = "zk")] zk_relations: &mut ZkRelationAccumulator<E>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let Stage2ReplayInput {
        setup,
        stage2,
        stage1,
        rs,
        relation_claim,
        #[cfg(feature = "zk")]
        relation_claim_mask,
        setup_sumcheck_proof,
        next_fold_level_params,
        ring_multiplier_points,
    } = input;
    let Stage1Replay {
        batching_coeff,
        s_claim,
        stage1_point,
        #[cfg(feature = "zk")]
        s_claim_mask,
    } = stage1;
    let setup_prepared_row_eval = setup_sumcheck_proof.map(|_| rs.prepared_row_eval.clone());
    let row_eval_source = Stage2RowEvalSource::new(
        rs.prepared_row_eval,
        setup_sumcheck_proof.map(|proof| proof.claim),
    );
    #[cfg(feature = "zk")]
    let stage2_next_w_eval_mask_cursor =
        *zk_hiding_cursor + (rs.col_bits + rs.ring_bits) * 3 * <E as ExtField<F>>::EXT_DEGREE;
    let witness_oracle = match &stage2 {
        Stage2ProofReplay::Terminal {
            final_witness,
            physical_w_len,
            ..
        } => Stage2WitnessOracle::Cleartext {
            witness: final_witness,
            physical_w_len: *physical_w_len,
        },
        Stage2ProofReplay::Intermediate { next_w_eval, .. } => Stage2WitnessOracle::ClaimedEval {
            eval: *next_w_eval,
            #[cfg(feature = "zk")]
            mask: zk_ext_mask_lc_at::<F, E>(stage2_next_w_eval_mask_cursor),
        },
    };
    let stage2_verifier = AkitaStage2Verifier::new(
        batching_coeff,
        s_claim,
        #[cfg(feature = "zk")]
        s_claim_mask,
        #[cfg(feature = "zk")]
        relation_claim_mask,
        witness_oracle,
        stage1_point,
        rs.alpha_evals_y,
        row_eval_source,
        &setup.expanded,
        ring_multiplier_points,
        relation_claim,
        rs.alpha,
        rs.col_bits,
        rs.ring_bits,
    )?;

    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        #[cfg(not(feature = "zk"))]
        {
            let stage2_sumcheck = match &stage2 {
                Stage2ProofReplay::Intermediate { sumcheck, .. }
                | Stage2ProofReplay::Terminal { sumcheck, .. } => *sumcheck,
            };
            stage2_verifier.verify::<F, T, _>(stage2_sumcheck, transcript, |tr| {
                sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?
        }
        #[cfg(feature = "zk")]
        {
            let stage2_sumcheck_masked = match &stage2 {
                Stage2ProofReplay::Intermediate {
                    sumcheck_masked, ..
                }
                | Stage2ProofReplay::Terminal {
                    sumcheck_masked, ..
                } => *sumcheck_masked,
            };
            let challenges = stage2_verifier.verify_zk::<F, T, _>(
                stage2_sumcheck_masked,
                transcript,
                zk_relations,
                zk_hiding_cursor,
                |tr| sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND),
            )?;
            if matches!(stage2, Stage2ProofReplay::Intermediate { .. }) {
                *zk_hiding_cursor += <E as ExtField<F>>::EXT_DEGREE;
            }
            challenges
        }
    };
    if let Stage2ProofReplay::Intermediate { next_w_eval, .. } = stage2 {
        transcript.absorb_and_record_serde(ABSORB_STAGE2_NEXT_W_EVAL, &next_w_eval);
    }
    if let Some(stage3_sumcheck_proof) = setup_sumcheck_proof {
        let setup_prepared_row_eval = setup_prepared_row_eval
            .as_ref()
            .ok_or(AkitaError::InvalidProof)?;
        let verifier = SetupSumcheckVerifier::new::<F, D>(
            setup_prepared_row_eval,
            &sumcheck_challenges[rs.ring_bits..],
            rs.alpha,
        )?;
        verifier.verify::<F, T, D>(
            setup,
            next_fold_level_params,
            stage3_sumcheck_proof,
            transcript,
        )?;
    }
    Ok(sumcheck_challenges)
}
