//! Root orchestration for a staged four-group p64 session.

use super::*;
use akita_error::checked;
use akita_types::binary_source::{
    grouped_root_statement, GroupedBinaryRelation, GroupedRootPublic, GroupedRootWitnessRequest,
    OrdinaryNativeRootClaim,
};

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn prove_root<Cfg, B>(
    expanded: &akita_types::AkitaSetupDescriptor,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    backend: &B,
    claims: &akita_types::OpeningClaims<'_, Cfg::ExtField, akita_types::Commitment<Cfg::Field>>,
    handles: &[B::CommitmentHandle],
    row: &akita_config::ResolvedScheduleRow,
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
    if handles.len() != 8 || basis != BasisMode::Lagrange {
        return Err(AkitaError::InvalidProof);
    }
    let binding = akita_config::bind_grouped_root(row, expanded, claims)?;
    akita_params::binary_setup::BinaryFieldProfile::P64.check_field_pair(
        akita_params::field_modulus::<Cfg::Field>()?,
        Cfg::ExtField::DEGREE,
    )?;
    let next = row
        .schedule()
        .recursive_folds
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    if next.predecessor_setup_contribution_mode() != SetupContributionMode::Recursive
        || prefix_slots.get(binding.first()).is_none()
    {
        return Err(AkitaError::InvalidProof);
    }
    let payloads = claims
        .groups()
        .iter()
        .map(|group| group.commitment().rows().coeffs())
        .collect::<Vec<_>>();
    grouped_root_statement::<Cfg::Field, Cfg::ExtField, _, _>(
        grinding.state_mut(),
        &binding,
        claims,
        &payloads,
    )?;
    let context = backend.proof_context(proof_session, 0)?;
    let mut terminals = Vec::new();
    for (index, group) in claims.groups().iter().take(4).enumerate() {
        let GroupClaimKind::Binary {
            manifest, claim, ..
        } = group.kind()
        else {
            return Err(AkitaError::InvalidProof);
        };
        let terminal = if let Some(claim) = claim {
            let (_, source) = backend.prepare_root_group(
                proof_session,
                &context.for_group(index),
                handles.get(index).ok_or(AkitaError::InvalidProof)?,
                &akita_types::binary_source::BinaryFieldSwitchSourceRequest {
                    manifest: manifest.as_ref().clone(),
                    polynomial: *claim.polynomial(),
                },
            )?;
            let source = source
                .downcast_ref::<Vec<u128>>()
                .ok_or(AkitaError::InvalidProof)?;
            Some(crate::binary_field_switch::prove_field_switch(
                manifest.descriptor(),
                claim,
                source,
                grinding.state_mut(),
            )?)
        } else {
            None
        };
        terminals.push(terminal);
    }
    let public = GroupedRootPublic::new(binding.clone(), &terminals)?;
    let request = GroupedRootWitnessRequest {
        binding,
        terminals,
        handles: handles.to_vec(),
    };
    let (d_payload, preparation) = backend.prepare_root_group(
        proof_session,
        &context.for_group(0),
        handles.first().ok_or(AkitaError::InvalidProof)?,
        &request,
    )?;
    let plan = public.binding().plan();
    let d_ranges = plan.d_payload_ranges()?;
    let end = d_ranges.last().ok_or(AkitaError::InvalidProof)?.end;
    if end != d_payload.len() {
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
    let logical_len = row.schedule().root.output_witness_len;
    let site = GrindingSite::FoldResponse { level: 0 };
    let mut accepted = None;
    for nonce in 0..akita_params::FOLD_RESPONSE_ATTEMPTS {
        let mut preview = grinding.preview_fold_response(site, nonce)?;
        let work = preview.take_work()?;
        let mut challenges = Vec::new();
        for binary in plan.binary() {
            let descriptor = binary.source().manifest.descriptor();
            challenges.push(binary_root_challenges(
                &mut PreviewFoldDraw::new(&mut preview),
                descriptor.index_map().columns(),
                descriptor.polynomials().len(),
            )?);
        }
        if let Some(witness) = backend.build_root_group(
            proof_session,
            &context,
            preparation.as_ref(),
            &challenges,
            logical_len,
            next.params.d_a(),
        )? {
            accepted = Some((work, challenges, witness));
            break;
        }
    }
    let (work, preview, witness) = accepted
        .ok_or_else(|| AkitaError::InvalidInput("grouped response rejection exhausted".into()))?;
    grinding.commit_fold_response(site, work)?;
    let mut challenges = Vec::new();
    for (index, binary) in plan.binary().iter().enumerate() {
        let descriptor = binary.source().manifest.descriptor();
        let (columns, owners) = (
            descriptor.index_map().columns(),
            descriptor.polynomials().len(),
        );
        challenges.push(binary_root_challenges(
            &mut ProverFoldDraw::new(grinding.state_mut(), 0, index as u32),
            columns,
            owners,
        )?);
        grinding.record_fold_challenges(
            0,
            index as u32,
            checked::product([columns, owners]).ok_or(AkitaError::InvalidProof)?,
        )?;
    }
    if challenges != preview {
        return Err(AkitaError::InvalidProof);
    }
    let committed = fold::commit_next_witness::<Cfg::Field, Cfg::ExtField, B>(
        backend,
        grinding,
        0,
        fold::FoldSuccessorParams::Recursive(next),
        logical_len,
        NextWitnessBindingPolicy::OuterPayload,
        witness,
    )?;
    let mut private = Vec::new();
    for (index, challenge) in challenges.iter().enumerate() {
        let (_, residual) = backend.prepare_root_group(
            proof_session,
            &context.for_group(index),
            handles.get(index).ok_or(AkitaError::InvalidProof)?,
            &BinaryProjectionRequest {
                challenges: challenge.clone(),
            },
        )?;
        private.push(
            residual
                .downcast::<BinaryProjectionWitness>()
                .map_err(|_| AkitaError::InvalidProof)?,
        );
    }
    let alpha = grinding
        .grinded_ext_challenge::<Cfg::ExtField>(GrindingSite::RingSwitchAlpha { level: 0 })?;
    let domain = FlatBooleanDomain::new(
        logical_len,
        public.binding().outgoing().domain_len().trailing_zeros() as usize,
    )?;
    let tau0 = grinding.grinded_ext_challenges::<Cfg::ExtField>(
        GrindingSite::Tau0Point { level: 0 },
        domain.num_vars(),
    )?;
    let batches = binary_root_p64_batch_challenges::<Cfg::Field, Cfg::ExtField, _>(
        grinding,
        public.batch_count(),
    )?;
    let lift = |payload: &[Cfg::Field]| {
        payload
            .iter()
            .copied()
            .map(Cfg::ExtField::lift_base)
            .collect::<Vec<_>>()
    };
    let payloads = payloads
        .iter()
        .map(|payload| lift(payload))
        .collect::<Vec<_>>();
    let d_payload = lift(&d_payload);
    let binary = challenges
        .iter()
        .zip(&private)
        .zip(&d_ranges)
        .enumerate()
        .map(|(index, ((challenges, private), range))| {
            Ok(GroupedBinaryRelation {
                challenges,
                commitment: payloads.get(index).ok_or(AkitaError::InvalidProof)?,
                d_payload: d_payload
                    .get(range.clone())
                    .ok_or(AkitaError::InvalidProof)?,
                matrix: Some(
                    akita_types::binary_source::BinaryAOperand::new::<Cfg::Field>(&private.matrix),
                ),
            })
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let points = claims
        .groups()
        .iter()
        .skip(4)
        .map(|group| group.point_evaluations().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let signed = plan.ordinary().ok_or(AkitaError::InvalidProof)?;
    let ordinary = signed
        .plans()
        .iter()
        .zip(&points)
        .enumerate()
        .map(|(index, (source, points))| {
            let first = points.first().ok_or(AkitaError::InvalidProof)?;
            Ok(OrdinaryNativeRootClaim {
                profile: source.profile(),
                width: source.source_width().ok_or(AkitaError::InvalidProof)?,
                point: first.point,
                evaluations: first.evaluations,
                payload: payloads.get(index + 4).ok_or(AkitaError::InvalidProof)?,
            })
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let second = points
        .iter()
        .map(|points| points.get(1).copied().ok_or(AkitaError::InvalidProof))
        .collect::<Result<Vec<_>, _>>()?;
    let (relation, negative) = public
        .relation::<Cfg::Field, Cfg::ExtField>(&binary, &ordinary, &second, alpha, &batches)?;
    finish_root::<Cfg, B>(
        prefix_slots,
        backend,
        next,
        proof_session,
        grinding,
        committed,
        domain,
        tau0,
        relation,
        negative,
        |point, _| {
            Ok((
                public.stage3_weights(point, alpha, &batches)?,
                public.binding().identity().to_vec(),
            ))
        },
    )
}
