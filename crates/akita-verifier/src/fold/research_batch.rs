//! Exact public replay of the explicitly admitted direct research batch.

use super::{FoldVerifyOutput, NextWitness};
use crate::stages::stage1::Stage1Verifier;
use akita_algebra::eq_poly::EqPolynomial;
use akita_challenges::VerifierFoldDraw;
use akita_error::AkitaError;
use akita_params::{
    CommittedGroupParams, DigitRangePlan, FoldParams, GrindingSite, OpeningClaimsLayout,
    SetupContributionMode, SumcheckProtocol,
};
use akita_serialization::AkitaSerialize;
use akita_transcript::ChallengeCodec;
use akita_types::binary_source::{
    binary_root_challenges, binary_root_matrix, binary_root_p64_batch_challenges,
    research_batch_relation, research_batch_statement, ResearchBatchPlan,
};
use akita_types::opening_claims::GroupClaimKind;
use akita_types::{
    AkitaVerifierSetup, CommittedGroup, DigitRangeEqualityPoint, FpExtEncoding, GrindingReplay,
    OpeningClaims, RingVec,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};
use std::sync::Arc;

pub(super) fn verify_root<F, E>(
    setup: &AkitaVerifierSetup<F>,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    claims: &OpeningClaims<'_, E, &CommittedGroup<F>>,
    layout: &OpeningClaimsLayout,
    root_params: &CommittedGroupParams,
    next: Option<&FoldParams>,
) -> Result<FoldVerifyOutput<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize + Ring,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Ring
        + AkitaSerialize
        + MulBaseUnreduced<F>
        + ChallengeCodec,
{
    let next = next.ok_or(AkitaError::InvalidProof)?;
    if next.predecessor_setup_contribution_mode() != SetupContributionMode::Direct {
        return Err(AkitaError::InvalidProof);
    }
    let profiles = (0..8)
        .map(|index| {
            root_params
                .group_params(layout, index)
                .map(|group| group.profile)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let kinds = claims
        .groups()
        .iter()
        .map(|group| group.kind().clone())
        .collect::<Vec<_>>();
    let descriptor = &setup.expanded().descriptor;
    let mut plan = ResearchBatchPlan::new(&kinds, &profiles, descriptor, next.params.d_a())?;
    if plan.physical_cells() != next.input_witness_len {
        return Err(AkitaError::InvalidProof);
    }
    research_batch_statement::<F, E, _>(grinding.state_mut(), claims, &profiles, |commitment| {
        commitment.rows().coeffs()
    })?;
    let group = claims.groups().first().ok_or(AkitaError::InvalidProof)?;
    let GroupClaimKind::Binary {
        manifest, claim, ..
    } = group.kind()
    else {
        return Err(AkitaError::InvalidProof);
    };
    let terminal = claim
        .as_ref()
        .map(|claim| {
            crate::binary_field_switch::verify_field_switch(
                manifest.descriptor(),
                claim,
                grinding.state_mut(),
            )
        })
        .transpose()?;
    plan.bind_word_terminal(terminal.as_ref())?;
    let d_count = plan.d_ranges().last().ok_or(AkitaError::InvalidProof)?.end;
    let d_payload = akita_transcript::receive_field_group::<F>(
        grinding.state_mut(),
        akita_types::FoldSite::OpeningPayload {
            level: 0,
            ring_dimension: 64,
        }
        .id()?,
        d_count,
    )?;
    // The entire D inventory was read before accepting or replaying the nonce.
    grinding.read_fold_response(GrindingSite::FoldResponse { level: 0 })?;
    let mut challenges = Vec::with_capacity(4);
    for (index, &columns) in plan.fold_coordinates().iter().enumerate() {
        let group = u32::try_from(index).map_err(|_| AkitaError::InvalidProof)?;
        challenges.push(binary_root_challenges(
            &mut VerifierFoldDraw::new(grinding.state_mut(), 0, group),
            columns,
            1,
        )?);
        grinding.record_fold_challenges(0, group, columns)?;
    }
    let next_witness = akita_transcript::receive_field_group::<F>(
        grinding.state_mut(),
        akita_types::FoldSite::NextWitnessPayload { level: 0 }.id()?,
        next.params
            .outer_payload_geometry()?
            .transmitted_coefficients(),
    )
    .map(RingVec::from_coeffs)?;
    let alpha = grinding.grinded_ext_challenge::<E>(GrindingSite::RingSwitchAlpha { level: 0 })?;
    let num_vars = usize::try_from(plan.domain_len().trailing_zeros())
        .map_err(|_| AkitaError::InvalidProof)?;
    let tau0 =
        grinding.grinded_ext_challenges::<E>(GrindingSite::Tau0Point { level: 0 }, num_vars)?;
    let batches = binary_root_p64_batch_challenges::<F, E, _>(grinding, plan.batch_count())?;
    let mut matrices = Vec::with_capacity(4);
    for kind in kinds.iter().take(4) {
        let GroupClaimKind::Binary { manifest, .. } = kind else {
            return Err(AkitaError::InvalidProof);
        };
        matrices.push(Arc::new(binary_root_matrix(
            &descriptor.setup_seed.seed,
            manifest,
        )?));
    }
    let (relation, negative) = research_batch_relation::<F, E, _>(
        &plan,
        setup.expanded(),
        claims,
        &profiles,
        &challenges,
        &d_payload,
        &matrices,
        alpha,
        &batches,
        |commitment| commitment.rows().coeffs(),
    )?;
    let equality = DigitRangeEqualityPoint::from_column_then_ring_challenges(&tau0, num_vars, 0)?;
    let stage1 =
        Stage1Verifier::new(equality, DigitRangePlan::new(8)?).verify::<F>(grinding, None, 0)?;
    let gamma = grinding.grinded_ext_challenge::<E>(GrindingSite::Stage2Batch { level: 0 })?;
    let mut channel = akita_types::GrindingSumcheckVerifier::<F, E>::new(
        grinding,
        SumcheckProtocol::Stage2,
        0,
        0,
    );
    let replay = akita_sumcheck::verify_sumcheck_rounds::<F, E, _>(
        &mut channel,
        0,
        relation.claim() + gamma * stage1.range_image_evaluation,
        akita_sumcheck::SumcheckShape::new(num_vars, 3)?,
    )?;
    let witness_eval = akita_types::stage2_w_eval::<F, E, _>(grinding, 0, E::zero())?;
    let negative_weight = negative.evaluate(&replay.challenges)?;
    let expected = relation.evaluate(&replay.challenges)? * witness_eval
        + (negative_weight + gamma * EqPolynomial::mle(&stage1.point, &replay.challenges)?)
            * witness_eval
            * (witness_eval + E::one());
    if replay.output_claim != expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(FoldVerifyOutput {
        challenges: replay.challenges,
        setup_prefix_opening: None,
        next_witness: NextWitness::OuterPayload(next_witness),
        opening: witness_eval,
    })
}
