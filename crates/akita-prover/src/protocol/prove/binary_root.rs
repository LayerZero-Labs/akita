use super::*;
use akita_challenges::{PreviewFoldDraw, ProverFoldDraw};
use akita_params::{
    DigitRangePlan, FlatBooleanDomain, GrindingSite, NextWitnessBindingPolicy,
    SetupContributionMode,
};
use akita_types::binary_source::{binary_root_challenges, binary_root_statement, BinaryRootPublic};
use akita_types::opening_claims::GroupClaimKind;
use akita_types::GrindingReplay;
use jolt_field::{AdditiveGroup, Zero};

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub(super) fn prove_root<Cfg, B>(
    expanded: &akita_types::AkitaSetupDescriptor,
    _prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    backend: &B,
    claims: &akita_types::OpeningClaims<'_, Cfg::ExtField, akita_types::Commitment<Cfg::Field>>,
    layout: &OpeningClaimsLayout,
    handles: &[B::CommitmentHandle],
    schedule: &FoldSchedule,
    proof_session: &B::ProofSessionHandle,
    grinding: &mut akita_types::ProverGrinding<'_>,
    basis: BasisMode,
    _handoff: &mut super::registry::FoldHandoff<'_, '_, '_, Cfg>,
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
    if !(1..=2).contains(&claims.num_groups())
        || handles.len() != claims.num_groups()
        || Cfg::ExtField::DEGREE != 1
        || basis != BasisMode::Lagrange
    {
        return Err(AkitaError::InvalidInput(
            "unsupported binary root group geometry".into(),
        ));
    }
    let group = claims.groups().first().ok_or(AkitaError::InvalidProof)?;
    let GroupClaimKind::Binary { manifest, claim } = group.kind() else {
        return Err(AkitaError::InvalidProof);
    };
    let next = schedule
        .recursive_folds
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    if next.predecessor_setup_contribution_mode() == SetupContributionMode::Recursive {
        return Err(AkitaError::InvalidInput(
            "binary root requires direct setup evaluation".into(),
        ));
    }
    let ordinary_group = claims.groups().get(1);
    let ordinary_profile = if let Some(ordinary) = ordinary_group {
        if ordinary.kind() != &GroupClaimKind::Ordinary || ordinary.evaluations().len() != 1 {
            return Err(AkitaError::InvalidProof);
        }
        Some(schedule.root.params.group_params(layout, 1)?.profile)
    } else {
        None
    };
    let logical_len = schedule.root.output_witness_len;
    let public = BinaryRootPublic::new(
        manifest,
        claim,
        expanded,
        logical_len,
        next.params.d_a(),
        ordinary_profile,
    )?;
    binary_root_statement(grinding.state_mut(), manifest, claim)?;
    akita_transcript::public_fields_prover(
        grinding.state_mut(),
        akita_types::FoldSite::RootCommitment {
            group: 0,
            ring_dimension: 64,
        }
        .id()?,
        group.commitment().rows().coeffs(),
    )?;
    if let (Some(ordinary), Some(profile)) = (ordinary_group, ordinary_profile) {
        let compression = profile.outer_compression_plan()?;
        let ring_dimension = compression
            .maps()
            .last()
            .ok_or(AkitaError::InvalidProof)?
            .ring_dimension();
        akita_transcript::public_fields_prover(
            grinding.state_mut(),
            akita_types::FoldSite::RootCommitment {
                group: 1,
                ring_dimension,
            }
            .id()?,
            ordinary.commitment().rows().coeffs(),
        )?;
        akita_transcript::public_extensions::<Cfg::Field, Cfg::ExtField, _>(
            grinding.state_mut(),
            akita_types::FoldSite::RootPoint { group: 1 }.id()?,
            ordinary.point(),
        )?;
        akita_transcript::public_extensions::<Cfg::Field, Cfg::ExtField, _>(
            grinding.state_mut(),
            akita_types::FoldSite::Openings { level: 0 }.id()?,
            ordinary.evaluations(),
        )?;
    }
    let context = backend.proof_context(proof_session, 0)?;
    let request = akita_types::binary_source::BinaryRootWitnessRequest {
        manifest: manifest.as_ref().clone(),
        claim: claim.clone(),
        plan: public.plan().clone(),
        ordinary: ordinary_profile
            .map(|profile| {
                Ok::<_, AkitaError>((
                    profile,
                    handles.get(1).ok_or(AkitaError::InvalidProof)?.clone(),
                ))
            })
            .transpose()?,
    };
    let (d_payload, preparation) = backend.prepare_root_group(
        proof_session,
        &context.for_group(0),
        handles.first().ok_or(AkitaError::InvalidProof)?,
        &request,
    )?;
    if d_payload.len() != 128 {
        return Err(AkitaError::InvalidProof);
    }
    akita_transcript::send_field_group(
        grinding.state_mut(),
        akita_types::FoldSite::OpeningPayload {
            level: 0,
            ring_dimension: 64,
        }
        .id()?,
        &d_payload,
    )?;
    let columns = manifest.descriptor().index_map().columns();
    let site = GrindingSite::FoldResponse { level: 0 };
    let mut accepted = None;
    for nonce in 0..akita_params::FOLD_RESPONSE_ATTEMPTS {
        let mut preview = grinding.preview_fold_response(site, nonce)?;
        let challenges = binary_root_challenges(&mut PreviewFoldDraw::new(&mut preview), columns)?;
        if let Some(witness) = backend.build_root_group(
            proof_session,
            &context,
            preparation.as_ref(),
            &challenges,
            logical_len,
            next.params.d_a(),
        )? {
            accepted = Some((nonce, challenges, witness));
            break;
        }
    }
    let (nonce, preview_challenges, witness) = accepted
        .ok_or_else(|| AkitaError::InvalidInput("binary response rejection exhausted".into()))?;
    grinding.commit_fold_response(site, nonce)?;
    let challenges = binary_root_challenges(
        &mut ProverFoldDraw::new(grinding.state_mut(), 0, 0),
        columns,
    )?;
    if challenges != preview_challenges {
        return Err(AkitaError::Internal(
            "binary response preview disagrees with replay".into(),
        ));
    }
    grinding.record_fold_challenges(0, 0, columns)?;
    let committed = fold::commit_next_witness::<Cfg::Field, Cfg::ExtField, B>(
        backend,
        grinding,
        0,
        fold::FoldSuccessorParams::Recursive(next),
        logical_len,
        NextWitnessBindingPolicy::OuterPayload,
        witness,
    )?;
    let mut witness = committed.witness_handle;
    let alpha = grinding.grinded_ext_challenge::<Cfg::Field, Cfg::ExtField>(
        GrindingSite::RingSwitchAlpha { level: 0 },
    )?;
    let domain = FlatBooleanDomain::new(
        logical_len,
        public.plan().repacking().domain_len().trailing_zeros() as usize,
    )?;
    if domain.domain_len() != public.plan().repacking().domain_len() {
        return Err(AkitaError::InvalidProof);
    }
    let tau0 = grinding.grinded_ext_challenges::<Cfg::Field, Cfg::ExtField>(
        GrindingSite::Tau0Point { level: 0 },
        domain.num_vars(),
    )?;
    let batches = grinding.grinded_ext_challenges::<Cfg::Field, Cfg::ExtField>(
        GrindingSite::Tau1Point { level: 0 },
        public.batch_count(),
    )?;
    let payload = group
        .commitment()
        .rows()
        .coeffs()
        .iter()
        .copied()
        .map(Cfg::ExtField::lift_base)
        .collect::<Vec<_>>();
    let d_payload = d_payload
        .into_iter()
        .map(Cfg::ExtField::lift_base)
        .collect::<Vec<_>>();
    let ordinary_payload = ordinary_group.map(|ordinary| {
        ordinary
            .commitment()
            .rows()
            .coeffs()
            .iter()
            .copied()
            .map(Cfg::ExtField::lift_base)
            .collect::<Vec<_>>()
    });
    let ordinary_claim = ordinary_group
        .zip(ordinary_payload.as_ref())
        .map(|(group, payload)| {
            Ok::<_, AkitaError>((
                group.point(),
                *group
                    .evaluations()
                    .first()
                    .ok_or(AkitaError::InvalidProof)?,
                payload.as_slice(),
            ))
        })
        .transpose()?;
    let (relation, negative_binary_weights) = public.relation(
        &challenges,
        alpha,
        &batches,
        &payload,
        &d_payload,
        ordinary_claim,
    )?;
    let range = DigitRangePlan::new(8)?;
    let session = backend.begin_root_digit_range(&witness, domain, &tau0, range)?;
    let stage1 = fold::sumcheck::prove_stage1_session::<Cfg::Field, Cfg::ExtField, B>(
        backend, session, range, domain, tau0, None, grinding, 0,
    )?;
    let gamma =
        grinding.grinded_ext_challenge::<Cfg::Field, Cfg::ExtField>(GrindingSite::Stage2Batch {
            level: 0,
        })?;
    let mut weights = vec![Cfg::ExtField::zero(); domain.domain_len()];
    for &(address, weight) in relation.terms() {
        *weights.get_mut(address).ok_or(AkitaError::InvalidProof)? += weight;
    }
    let session = backend.begin_root_relation(
        &witness,
        weights,
        negative_binary_weights,
        relation.claim(),
        &stage1.point,
        stage1.range_image_evaluation,
        gamma,
    )?;
    let stage2 = fold::sumcheck::prove_stage2_session::<Cfg::Field, Cfg::ExtField, B>(
        backend,
        session,
        grinding,
        0,
        domain.num_vars(),
        relation.claim() + gamma * stage1.range_image_evaluation,
    )?;
    akita_types::stage2_w_eval::<Cfg::Field, Cfg::ExtField, _>(
        grinding,
        0,
        stage2.witness_evaluation,
    )?;
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
