//! Replay of a staged four-group p64 session from trusted row inputs.

use super::*;
use akita_error::checked;
use akita_types::binary_source::{
    grouped_root_statement, GroupedBinaryRelation, GroupedRootPublic, OrdinaryNativeRootClaim,
};

pub(super) fn verify_root<F, E>(
    setup: &AkitaVerifierSetup<F>,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    claims: &OpeningClaims<'_, E, &CommittedGroup<F>>,
    row: &akita_config::ResolvedScheduleRow,
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
    let binding = akita_config::bind_grouped_root(row, setup.expanded().descriptor(), claims)?;
    akita_params::binary_setup::BinaryFieldProfile::P64.check_field_pair(
        akita_params::field_modulus::<F>()?, E::DEGREE,
    )?;
    binding.validate_verifier_setup(setup)?;
    let next = row.schedule().recursive_folds.first().ok_or(AkitaError::InvalidProof)?;
    if next.predecessor_setup_contribution_mode() != SetupContributionMode::Recursive {
        return Err(AkitaError::InvalidProof);
    }
    let payloads = claims.groups().iter().map(|group| group.commitment().rows().coeffs())
        .collect::<Vec<_>>();
    grouped_root_statement::<F, E, _, _>(grinding.state_mut(), &binding, claims, &payloads)?;
    let mut terminals = Vec::new();
    for group in claims.groups().iter().take(4) {
        let GroupClaimKind::Binary { manifest, claim, .. } = group.kind() else {
            return Err(AkitaError::InvalidProof);
        };
        terminals.push(claim.as_ref().map(|claim| {
            crate::binary_field_switch::verify_field_switch(
                manifest.descriptor(), claim, grinding.state_mut(),
            )
        }).transpose()?);
    }
    let public = GroupedRootPublic::new(binding, &terminals)?;
    let plan = public.binding().plan();
    let d_ranges = plan.d_payload_ranges()?;
    let end = d_ranges.last().ok_or(AkitaError::InvalidProof)?.end;
    let d_payload = akita_transcript::receive_field_group::<F>(grinding.state_mut(),
        akita_types::FoldSite::OpeningPayload { level: 0, ring_dimension: 64 }.id()?, end,
    )?;
    grinding.read_fold_response(GrindingSite::FoldResponse { level: 0 })?;
    let mut challenges = Vec::new();
    for (index, binary) in plan.binary().iter().enumerate() {
        let descriptor = binary.source().manifest.descriptor();
        let (columns, owners) = (descriptor.index_map().columns(), descriptor.polynomials().len());
        challenges.push(binary_root_challenges(
            &mut VerifierFoldDraw::new(grinding.state_mut(), 0, index as u32), columns, owners,
        )?);
        grinding.record_fold_challenges(0, index as u32,
            checked::product([columns, owners]).ok_or(AkitaError::InvalidProof)?)?;
    }
    let next_witness = RingVec::from_coeffs(akita_transcript::receive_field_group::<F>(
        grinding.state_mut(), akita_types::FoldSite::NextWitnessPayload { level: 0 }.id()?,
        next.params.outer_payload_geometry()?.transmitted_coefficients(),
    )?);
    let alpha = grinding.grinded_ext_challenge::<E>(GrindingSite::RingSwitchAlpha { level: 0 })?;
    let domain = public.binding().outgoing().domain_len();
    let tau0 = grinding.grinded_ext_challenges::<E>(GrindingSite::Tau0Point { level: 0 },
        domain.trailing_zeros() as usize)?;
    let batches = binary_root_p64_batch_challenges::<F, E, _>(grinding, public.batch_count())?;
    let lift = |payload: &[F]| payload.iter().copied().map(E::lift_base).collect::<Vec<_>>();
    let payloads = payloads.iter().map(|payload| lift(payload)).collect::<Vec<_>>();
    let d_payload = lift(&d_payload);
    let binary = challenges.iter().zip(&d_ranges).enumerate().map(|(index, (challenges, range))| {
        Ok(GroupedBinaryRelation {
            challenges, commitment: payloads.get(index).ok_or(AkitaError::InvalidProof)?,
            d_payload: d_payload.get(range.clone()).ok_or(AkitaError::InvalidProof)?, matrix: None,
        })
    }).collect::<Result<Vec<_>, AkitaError>>()?;
    let points = claims.groups().iter().skip(4)
        .map(|group| group.point_evaluations().collect::<Vec<_>>()).collect::<Vec<_>>();
    let signed = plan.ordinary().ok_or(AkitaError::InvalidProof)?;
    let ordinary = signed.plans().iter().zip(&points).enumerate().map(|(index, (source, points))| {
        let first = points.first().ok_or(AkitaError::InvalidProof)?;
        Ok(OrdinaryNativeRootClaim {
            profile: source.profile(),
            width: source.source_width().ok_or(AkitaError::InvalidProof)?,
            point: first.point, evaluations: first.evaluations,
            payload: payloads.get(index + 4).ok_or(AkitaError::InvalidProof)?,
        })
    }).collect::<Result<Vec<_>, AkitaError>>()?;
    let second = points.iter().map(|points| points.get(1).copied().ok_or(AkitaError::InvalidProof))
        .collect::<Result<Vec<_>, _>>()?;
    let (relation, negative) =
        public.relation::<F, E>(&binary, &ordinary, &second, alpha, &batches)?;
    finish_root::<F, E>(setup, grinding, tau0, relation, negative, domain,
        public.binding().first().clone(), public.binding().identity().to_vec(), next_witness,
        |root_point, point| public.stage3_weights(root_point, alpha, &batches)?.evaluate(point))
}
