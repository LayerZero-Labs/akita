//! One direct-setup research root, one joint response and one ordinary suffix.

use super::*;
use akita_challenges::{PreviewFoldDraw, ProverFoldDraw};
use akita_params::{DigitRangePlan, FlatBooleanDomain, GrindingSite, NextWitnessBindingPolicy};
use akita_types::binary_source::{
    binary_root_challenges, binary_root_p64_batch_challenges, research_batch_relation,
    research_batch_statement, BinaryProjectionRequest, BinaryProjectionWitness, ResearchBatchPlan,
    ResearchBatchResponse, ResearchBatchWitnessRequest,
};
use akita_types::GrindingReplay;
use jolt_field::AdditiveGroup;

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "research_root_total", fields(groups = handles.len()))]
pub(super) fn prove_root<Cfg, B>(
    expanded: &akita_types::AkitaSetupDescriptor,
    backend: &B,
    claims: &akita_types::OpeningClaims<'_, Cfg::ExtField, akita_types::Commitment<Cfg::Field>>,
    layout: &OpeningClaimsLayout,
    handles: &[B::CommitmentHandle],
    schedule: &FoldSchedule,
    proof_session: &B::ProofSessionHandle,
    grinding: &mut akita_types::ProverGrinding<'_>,
    basis: BasisMode,
) -> Result<
    ProveLevelOutput<Cfg::Field, Cfg::ExtField, B::CommitmentMaterialHandle, B::WitnessHandle>,
    AkitaError,
>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding + AkitaSerialize + Unreduced + PseudoMersenne + Ring + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<Cfg::Field>
        + AkitaSerialize
        + 'static,
    B: ProverBackend<Cfg::Field, Cfg::ExtField> + 'static,
{
    let stage = tracing::info_span!("mldsa.port.opening.root_relation").entered();
    if basis != BasisMode::Lagrange || handles.len() != 8 {
        return Err(AkitaError::InvalidInput(
            "research batch proof geometry".into(),
        ));
    }
    let next = schedule
        .recursive_folds
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    if next.predecessor_setup_contribution_mode() != SetupContributionMode::Direct {
        return Err(AkitaError::InvalidInput(
            "research root requires direct setup".into(),
        ));
    }
    let phase = tracing::info_span!(
        "research_root_plan",
        groups = handles.len(),
        physical_cells = tracing::field::Empty,
        batch_count = tracing::field::Empty
    )
    .entered();
    let profiles = (0..8)
        .map(|index| {
            schedule
                .root
                .params
                .group_params(layout, index)
                .map(|group| group.profile)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let kinds = claims
        .groups()
        .iter()
        .map(|group| group.kind().clone())
        .collect::<Vec<_>>();
    let mut plan = ResearchBatchPlan::new(&kinds, &profiles, expanded, next.params.d_a())?;
    if plan.physical_cells() != schedule.root.output_witness_len {
        return Err(AkitaError::InvalidProof);
    }
    phase.record("physical_cells", plan.physical_cells());
    phase.record("batch_count", plan.batch_count());
    drop(phase);
    let phase =
        tracing::info_span!("research_root_statement", groups = claims.num_groups()).entered();
    research_batch_statement::<Cfg::Field, Cfg::ExtField, _>(
        grinding.state_mut(),
        claims,
        &profiles,
        |commitment| commitment.rows().coeffs(),
    )?;
    drop(phase);
    let phase = tracing::info_span!(
        "research_root_word_terminal",
        carriers = tracing::field::Empty
    )
    .entered();
    let context = backend.proof_context(proof_session, 0)?;
    let group = claims.groups().first().ok_or(AkitaError::InvalidProof)?;
    let akita_types::opening_claims::GroupClaimKind::Binary {
        manifest, claim, ..
    } = group.kind()
    else {
        return Err(AkitaError::InvalidProof);
    };
    let terminal = if let Some(claim) = claim {
        let (_, source) = backend.prepare_root_group(
            proof_session,
            &context.for_group(0),
            handles.first().ok_or(AkitaError::InvalidProof)?,
            &akita_types::binary_source::BinaryFieldSwitchSourceRequest {
                manifest: manifest.as_ref().clone(),
                polynomial: *claim.polynomial(),
            },
        )?;
        let carriers = source
            .downcast_ref::<Vec<u128>>()
            .ok_or(AkitaError::InvalidProof)?;
        phase.record("carriers", carriers.len());
        Some(crate::binary_field_switch::prove_field_switch(
            manifest.descriptor(),
            claim,
            carriers,
            grinding.state_mut(),
        )?)
    } else {
        None
    };
    plan.bind_word_terminal(terminal.as_ref())?;
    drop(phase);
    let phase = tracing::info_span!("research_root_prepare_groups").entered();
    let (d_payload, preparation) = backend.prepare_root_group(
        proof_session,
        &context,
        handles.first().ok_or(AkitaError::InvalidProof)?,
        &ResearchBatchWitnessRequest {
            kinds,
            plan: plan.clone(),
            handles: handles.to_vec(),
        },
    )?;
    drop(phase);
    if plan.d_ranges().last().map(|range| range.end) != Some(d_payload.len()) {
        return Err(AkitaError::InvalidProof);
    }
    // All four D commitments precede the only response nonce.
    let phase = tracing::info_span!("research_root_send_d").entered();
    akita_transcript::send_field_group(
        grinding.state_mut(),
        akita_types::FoldSite::OpeningPayload {
            level: 0,
            ring_dimension: 64,
        }
        .id()?,
        &d_payload,
    )?;
    drop(phase);
    let phase = tracing::info_span!("research_root_response_search").entered();
    let site = GrindingSite::FoldResponse { level: 0 };
    let mut accepted = None;
    for nonce in 0..akita_params::FOLD_RESPONSE_ATTEMPTS {
        let _attempt = tracing::info_span!("research_root_response_attempt", nonce).entered();
        let mut preview = grinding.preview_fold_response(site, nonce)?;
        let work = preview.take_work()?;
        let mut challenges = Vec::with_capacity(4);
        for &columns in plan.fold_coordinates() {
            challenges.push(binary_root_challenges(
                &mut PreviewFoldDraw::new(&mut preview),
                columns,
                1,
            )?);
        }
        // Every group was drawn even if the first response will reject.
        let response = ResearchBatchResponse { challenges };
        if let Some(witness) = backend.build_root_group(
            proof_session,
            &context,
            preparation.as_ref(),
            &response,
            plan.physical_cells(),
            next.params.d_a(),
        )? {
            accepted = Some((work, response.challenges, witness));
            break;
        }
    }
    let (work, preview_challenges, witness) = accepted.ok_or_else(|| {
        AkitaError::InvalidInput("research joint response rejection exhausted".into())
    })?;
    drop(phase);
    let phase = tracing::info_span!("research_root_response_replay").entered();
    grinding.commit_fold_response(site, work)?;
    let mut challenges = Vec::with_capacity(4);
    for (index, &columns) in plan.fold_coordinates().iter().enumerate() {
        let group = u32::try_from(index).map_err(|_| AkitaError::InvalidProof)?;
        let draw = binary_root_challenges(
            &mut ProverFoldDraw::new(grinding.state_mut(), 0, group),
            columns,
            1,
        )?;
        if preview_challenges.get(index) != Some(&draw) {
            return Err(AkitaError::Internal(
                "research preview disagrees with replay".into(),
            ));
        }
        grinding.record_fold_challenges(0, group, columns)?;
        challenges.push(draw);
    }
    drop(phase);
    let phase = tracing::info_span!("research_root_commit_witness").entered();
    let committed = fold::commit_next_witness::<Cfg::Field, Cfg::ExtField, B>(
        backend,
        grinding,
        0,
        fold::FoldSuccessorParams::Recursive(next),
        plan.physical_cells(),
        NextWitnessBindingPolicy::OuterPayload,
        witness,
    )?;
    drop(phase);
    let mut witness = committed.witness_handle;
    let phase = tracing::info_span!("research_root_projection_replay").entered();
    let mut matrices = Vec::with_capacity(4);
    for (index, challenge) in challenges.iter().enumerate() {
        let _group = tracing::info_span!("research_root_projection_group", index).entered();
        let (_, residual) = backend.prepare_root_group(
            proof_session,
            &context.for_group(index),
            handles.get(index).ok_or(AkitaError::InvalidProof)?,
            &BinaryProjectionRequest {
                challenges: challenge.clone(),
            },
        )?;
        let residual = residual
            .downcast::<BinaryProjectionWitness>()
            .map_err(|_| AkitaError::InvalidProof)?;
        matrices.push(residual.matrix);
    }
    drop(phase);
    let phase = tracing::info_span!("research_root_alpha").entered();
    let alpha = grinding
        .grinded_ext_challenge::<Cfg::ExtField>(GrindingSite::RingSwitchAlpha { level: 0 })?;
    drop(phase);
    let num_vars = usize::try_from(plan.domain_len().trailing_zeros())
        .map_err(|_| AkitaError::InvalidProof)?;
    let domain = FlatBooleanDomain::new(plan.physical_cells(), num_vars)?;
    if domain.domain_len() != plan.domain_len() {
        return Err(AkitaError::InvalidProof);
    }
    let phase = tracing::info_span!("research_root_tau0").entered();
    let tau0 = grinding
        .grinded_ext_challenges::<Cfg::ExtField>(GrindingSite::Tau0Point { level: 0 }, num_vars)?;
    drop(phase);
    let phase =
        tracing::info_span!("research_root_batch_challenges", count = plan.batch_count()).entered();
    let batches = binary_root_p64_batch_challenges::<Cfg::Field, Cfg::ExtField, _>(
        grinding,
        plan.batch_count(),
    )?;
    drop(phase);
    let compile = tracing::info_span!(
        "mldsa.port.opening.root_compile",
        batch_count = plan.batch_count(),
        domain_len = domain.domain_len(),
        negative_count = tracing::field::Empty
    )
    .entered();
    let (relation, negative) = research_batch_relation::<Cfg::Field, Cfg::ExtField, _>(
        &plan,
        expanded,
        claims,
        &profiles,
        &challenges,
        &d_payload,
        &matrices,
        alpha,
        &batches,
        |commitment| commitment.rows().coeffs(),
    )?;
    compile.record("negative_count", negative.len());
    let relation_claim = relation.claim();
    drop(compile);
    drop(stage);
    let stage = tracing::info_span!("mldsa.port.opening.stage1").entered();
    let range = DigitRangePlan::new(8)?;
    let session = backend.begin_root_digit_range(&witness, domain, &tau0, range)?;
    let stage1 = fold::sumcheck::prove_stage1_session::<Cfg::Field, Cfg::ExtField, B>(
        backend, session, range, domain, tau0, None, grinding, 0,
    )?;
    drop(stage);
    let stage = tracing::info_span!(
        "mldsa.port.opening.root_prepare",
        domain_len = domain.domain_len(),
        negative_count = negative.len(),
        point_len = stage1.point.len()
    )
    .entered();
    let gamma_span = tracing::info_span!("research_root_stage2_batch").entered();
    let gamma =
        grinding.grinded_ext_challenge::<Cfg::ExtField>(GrindingSite::Stage2Batch { level: 0 })?;
    drop(gamma_span);
    let session = backend.begin_root_relation(
        &witness,
        relation,
        negative,
        relation_claim,
        &stage1.point,
        stage1.range_image_evaluation,
        gamma,
    )?;
    drop(stage);
    let stage = tracing::info_span!("mldsa.port.opening.stage2").entered();
    let stage2 = fold::sumcheck::prove_stage2_session::<Cfg::Field, Cfg::ExtField, B>(
        backend,
        session,
        grinding,
        0,
        num_vars,
        relation_claim + gamma * stage1.range_image_evaluation,
    )?;
    akita_types::stage2_w_eval::<Cfg::Field, Cfg::ExtField, _>(
        grinding,
        0,
        stage2.witness_evaluation,
    )?;
    drop(stage);
    let _advance = tracing::info_span!("research_root_advance_witness").entered();
    backend.advance_witness_level(&mut witness)?;
    Ok(ProveLevelOutput {
        next_state: SuffixProverState {
            witness_handle: witness,
            commitment_material: committed.commitment_material_handle,
            public: FoldPublicState {
                level: 1,
                binding: committed.binding,
                sumcheck_challenges: stage2.challenges,
                opening: stage2.witness_evaluation,
                setup_prefix_opening: None,
            },
        },
    })
}
