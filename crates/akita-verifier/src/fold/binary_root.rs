//! Replay of the explicitly selected binary root inside the ordinary grouped proof.

use super::FoldVerifyOutput;
use crate::stages::stage1::Stage1Verifier;
use akita_algebra::eq_poly::EqPolynomial;
use akita_challenges::VerifierFoldDraw;
use akita_error::AkitaError;
use akita_params::{
    CommittedGroupParams, DigitRangePlan, FoldParams, GrindingSite, OpeningClaimsLayout,
    SetupContributionMode, SumcheckProtocol,
};
use akita_serialization::AkitaSerialize;
use akita_types::binary_source::{binary_root_challenges, binary_root_statement, BinaryRootPublic};
use akita_types::opening_claims::GroupClaimKind;
use akita_types::{
    AkitaVerifierSetup, Commitment, DigitRangeEqualityPoint, FpExtEncoding, GrindingReplay,
    OpeningClaims, RingVec,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

pub(super) fn verify_root<F, E>(
    setup: &AkitaVerifierSetup<F>,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    claims: &OpeningClaims<'_, E, &Commitment<F>>,
    opening_batch: &OpeningClaimsLayout,
    root_params: &CommittedGroupParams,
    next_fold: Option<&FoldParams>,
) -> Result<FoldVerifyOutput<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize + Ring,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    if !(1..=2).contains(&claims.num_groups())
        || E::DEGREE != 1
        || akita_params::field_modulus::<F>()? != akita_params::binary_setup::BINARY_SETUP_P
    {
        return Err(AkitaError::InvalidSetup(
            "binary root requires its admitted proof prime and at most one ordinary group".into(),
        ));
    }
    let group = claims.groups().first().ok_or(AkitaError::InvalidProof)?;
    let GroupClaimKind::Binary { manifest, claim } = group.kind() else {
        return Err(AkitaError::InvalidProof);
    };
    let ordinary = claims.groups().get(1);
    let ordinary_profile = match ordinary {
        Some(group)
            if matches!(group.kind(), GroupClaimKind::Ordinary) && group.num_evaluations() == 1 =>
        {
            Some(root_params.group_params(opening_batch, 1)?.profile)
        }
        Some(_) => return Err(AkitaError::InvalidProof),
        None => None,
    };
    let next = next_fold.ok_or_else(|| {
        AkitaError::InvalidSetup("binary root requires an ordinary successor fold".into())
    })?;
    if next.predecessor_setup_contribution_mode() == SetupContributionMode::Recursive {
        return Err(AkitaError::InvalidSetup(
            "binary root evaluates its setup directly".into(),
        ));
    }
    let logical_len = root_params.output_witness_len::<F>(opening_batch, E::DEGREE)?;
    let public = BinaryRootPublic::new(
        manifest,
        claim,
        &setup.expanded().descriptor,
        logical_len,
        next.params.d_a(),
        ordinary_profile,
    )?;
    let domain_len = public.plan().repacking().domain_len();
    if !domain_len.is_power_of_two() {
        return Err(AkitaError::InvalidProof);
    }
    let num_vars = domain_len.trailing_zeros() as usize;

    binary_root_statement(grinding.state_mut(), manifest, claim)?;
    akita_transcript::public_fields_verifier(
        grinding.state_mut(),
        akita_types::FoldSite::RootCommitment {
            group: 0,
            ring_dimension: manifest.b_ring_dimension(),
        }
        .id()?,
        group.commitment().rows().coeffs(),
    )?;
    if let (Some(group), Some(profile)) = (ordinary, ordinary_profile) {
        let compression = profile.outer_compression_plan()?;
        let dimension = compression
            .maps()
            .last()
            .ok_or(AkitaError::InvalidProof)?
            .ring_dimension();
        akita_transcript::public_fields_verifier(
            grinding.state_mut(),
            akita_types::FoldSite::RootCommitment {
                group: 1,
                ring_dimension: dimension,
            }
            .id()?,
            group.commitment().rows().coeffs(),
        )?;
        akita_transcript::public_extensions::<F, E, _>(
            grinding.state_mut(),
            akita_types::FoldSite::RootPoint { group: 1 }.id()?,
            group.point(),
        )?;
        akita_transcript::public_extensions::<F, E, _>(
            grinding.state_mut(),
            akita_types::FoldSite::Openings { level: 0 }.id()?,
            group.evaluations(),
        )?;
    }
    let payload_len =
        akita_error::checked::product([manifest.b_rows(), manifest.b_ring_dimension()])
            .ok_or(AkitaError::InvalidProof)?;
    let d_payload = akita_transcript::receive_field_group::<F>(
        grinding.state_mut(),
        akita_types::FoldSite::OpeningPayload {
            level: 0,
            ring_dimension: manifest.b_ring_dimension(),
        }
        .id()?,
        payload_len,
    )?;
    grinding.read_fold_response(GrindingSite::FoldResponse { level: 0 })?;
    let columns = manifest.descriptor().index_map().columns();
    let challenges = {
        let mut draw = VerifierFoldDraw::new(grinding.state_mut(), 0, 0);
        binary_root_challenges(&mut draw, columns)?
    };
    grinding.record_fold_challenges(0, 0, columns)?;

    let successor_payload_len = next
        .params
        .outer_payload_geometry()?
        .transmitted_coefficients();
    let next_witness = akita_transcript::receive_field_group::<F>(
        grinding.state_mut(),
        akita_types::FoldSite::NextWitnessPayload { level: 0 }.id()?,
        successor_payload_len,
    )
    .map(RingVec::from_coeffs)?;
    let alpha =
        grinding.grinded_ext_challenge::<F, E>(GrindingSite::RingSwitchAlpha { level: 0 })?;
    let tau0 =
        grinding.grinded_ext_challenges::<F, E>(GrindingSite::Tau0Point { level: 0 }, num_vars)?;
    let batches = grinding.grinded_ext_challenges::<F, E>(
        GrindingSite::Tau1Point { level: 0 },
        public.batch_count(),
    )?;
    let lift = |values: &[F]| values.iter().copied().map(E::lift_base).collect::<Vec<_>>();
    let ordinary_payload = ordinary.map(|group| lift(group.commitment().rows().coeffs()));
    let ordinary_claim = ordinary
        .zip(ordinary_payload.as_deref())
        .map(|(group, payload)| {
            Ok::<_, AkitaError>((
                group.point(),
                *group
                    .evaluations()
                    .first()
                    .ok_or(AkitaError::InvalidProof)?,
                payload,
            ))
        })
        .transpose()?;
    let (relation, negative_bits) = public.relation(
        &challenges,
        alpha,
        &batches,
        &lift(group.commitment().rows().coeffs()),
        &lift(&d_payload),
        ordinary_claim,
    )?;

    let equality_point =
        DigitRangeEqualityPoint::from_column_then_ring_challenges(&tau0, num_vars, 0)?;
    let stage1 = Stage1Verifier::new(equality_point, DigitRangePlan::new(8)?)
        .verify::<F>(grinding, None, 0)?;
    let gamma = grinding.grinded_ext_challenge::<F, E>(GrindingSite::Stage2Batch { level: 0 })?;
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
    let negative_bit_weight = akita_types::native_root_relation::NativeRootWeights::new(
        domain_len,
        negative_bits,
        E::zero(),
    )?
    .evaluate(&replay.challenges)?;
    let expected = relation.evaluate(&replay.challenges)? * witness_eval
        + (negative_bit_weight + gamma * EqPolynomial::mle(&stage1.point, &replay.challenges)?)
            * witness_eval
            * (witness_eval + E::one());
    if replay.output_claim != expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(FoldVerifyOutput {
        challenges: replay.challenges,
        setup_prefix_opening: None,
        next_witness,
        next_witness_last_block_digest: None,
        opening: witness_eval,
    })
}
