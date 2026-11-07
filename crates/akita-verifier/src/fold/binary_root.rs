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
use akita_transcript::ChallengeCodec;
use akita_types::binary_source::{
    binary_projection_values, binary_root_batch_challenges, binary_root_challenges,
    binary_root_p64_batch_challenges, binary_root_statement, BinaryProjectionRows,
    BinaryRootPublic,
};
use akita_types::opening_claims::GroupClaimKind;
use akita_types::{
    AkitaVerifierSetup, CommittedGroup, DigitRangeEqualityPoint, FpExtEncoding, GrindingReplay,
    OpeningClaims, RingVec,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

pub(super) fn verify_root<F, E>(
    setup: &AkitaVerifierSetup<F>,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    claims: &OpeningClaims<'_, E, &CommittedGroup<F>>,
    opening_batch: &OpeningClaimsLayout,
    root_params: &CommittedGroupParams,
    next_fold: Option<&FoldParams>,
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
    let _span = tracing::info_span!("binary_root_verify").entered();
    if !(1..=2).contains(&claims.num_groups()) {
        return Err(AkitaError::InvalidSetup(
            "binary root requires its admitted proof prime and at most one ordinary group".into(),
        ));
    }
    let group = claims.groups().first().ok_or(AkitaError::InvalidProof)?;
    let GroupClaimKind::Binary {
        manifest,
        claim,
        prime,
    } = group.kind()
    else {
        return Err(AkitaError::InvalidProof);
    };
    manifest
        .field_profile()
        .admit_root(akita_params::field_modulus::<F>()?, E::DEGREE)?;
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
    if next.predecessor_setup_contribution_mode() != SetupContributionMode::Recursive {
        return Err(AkitaError::InvalidSetup(
            "binary root requires a typed setup-prefix successor".into(),
        ));
    }
    let typed_prefix = next
        .params
        .setup_prefix()
        .and_then(|p| p.slot_id())
        .ok_or(AkitaError::InvalidProof)?;
    let typed = akita_types::binary_source::BinaryTypedPrefix::new(
        manifest,
        claim.is_some(),
        prime.planes().len(),
        next.params.d_a(),
    )?;
    let typed_prefix_len = typed_prefix.n_prefix()?;
    let typed_identity = typed.identity(manifest, &typed_prefix)?;
    if setup.prefix_context(&typed_prefix) != Some(typed_identity.as_slice()) {
        return Err(AkitaError::InvalidInput(
            "binary statement differs from its prepared typed prefix".into(),
        ));
    }
    // The trusted catalog validates this transition against its producer.
    // Native roots have their own coefficient geometry; recomputing an
    // ordinary root layout here would disagree with the selected prover row.
    let logical_len = next.input_witness_len;
    let ordinary_statement = ordinary
        .zip(ordinary_profile)
        .map(|(ordinary, profile)| {
            let compression = profile.outer_compression_plan()?;
            let dimension = compression
                .maps()
                .last()
                .ok_or(AkitaError::InvalidProof)?
                .ring_dimension();
            Ok::<_, AkitaError>((
                dimension,
                ordinary.commitment().rows().coeffs(),
                ordinary.point(),
                ordinary.evaluations(),
            ))
        })
        .transpose()?;
    binary_root_statement::<F, E, _>(
        grinding.state_mut(),
        manifest,
        claim.as_ref(),
        prime,
        group.commitment().rows().coeffs(),
        ordinary_statement,
    )?;
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
    let span = tracing::info_span!("binary_root_public_new").entered();
    let public = BinaryRootPublic::new(
        manifest,
        terminal.as_ref(),
        prime,
        &setup.expanded().descriptor,
        logical_len,
        next.params.d_a(),
        ordinary_profile,
    )?;
    drop(span);
    let domain_len = public.plan().repacking().domain_len();
    if !domain_len.is_power_of_two() {
        return Err(AkitaError::InvalidProof);
    }
    let num_vars = domain_len.trailing_zeros() as usize;

    let payload_len = akita_error::checked::product([
        manifest
            .d_label(claim.is_some(), prime.planes().len())?
            .rank as usize,
        manifest.b_ring_dimension(),
    ])
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
    let owners = manifest.descriptor().polynomials().len();
    let challenges = {
        let mut draw = VerifierFoldDraw::new(grinding.state_mut(), 0, 0);
        binary_root_challenges(&mut draw, columns, owners)?
    };
    grinding.record_fold_challenges(
        0,
        0,
        akita_error::checked::product([columns, owners]).ok_or(AkitaError::InvalidProof)?,
    )?;

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
    let bounds = public.plan().projection_bounds();
    let projections = if bounds.quotient().is_none() {
        Some(BinaryProjectionRows::draw(grinding.state_mut(), bounds)?)
    } else {
        None
    };
    let projected_values = if projections.is_some() {
        binary_projection_values(grinding.state_mut(), bounds, &[0; 7])?.to_vec()
    } else {
        Vec::new()
    };
    let alpha = grinding.grinded_ext_challenge::<E>(GrindingSite::RingSwitchAlpha { level: 0 })?;
    let tau0 =
        grinding.grinded_ext_challenges::<E>(GrindingSite::Tau0Point { level: 0 }, num_vars)?;
    let batches = {
        let _span = tracing::info_span!("binary_root_tau1_draw").entered();
        if bounds.quotient().is_some() {
            binary_root_p64_batch_challenges::<F, E, _>(grinding, public.batch_count())?
        } else {
            binary_root_batch_challenges::<E, _>(grinding, public.batch_count())?
        }
    };
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
    let rho = match &projections {
        Some(projections) => projections.batch(
            &batches
                .get(..7)
                .ok_or(AkitaError::InvalidProof)?
                .try_into()
                .map_err(|_| AkitaError::InvalidProof)?,
        )?,
        None => Vec::new(),
    };
    let span = tracing::info_span!("binary_root_relation_compile").entered();
    let (relation, negative_bits) = public.relation(
        &challenges,
        alpha,
        &batches,
        &rho,
        &projected_values,
        &lift(group.commitment().rows().coeffs()),
        &lift(&d_payload),
        ordinary_claim,
        None,
    )?;
    let authentication = public.authentication_batches(&batches)?.map(<[E]>::to_vec);
    let (rho, _) = public.coefficient_inputs(&batches, &rho, &projected_values)?;
    drop(batches);
    drop(span);

    let span = tracing::info_span!("binary_root_stage1_stage2_replay").entered();
    let equality_point =
        DigitRangeEqualityPoint::from_column_then_ring_challenges(&tau0, num_vars, 0)?;
    let stage1 = Stage1Verifier::new(equality_point, DigitRangePlan::new(8)?)
        .verify::<F>(grinding, None, 0)?;
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
    drop(span);
    let negative_bit_weight = akita_types::native_root_relation::NativeRootWeights::new(
        domain_len,
        negative_bits,
        E::zero(),
    )?
    .evaluate(&replay.challenges)?;
    if setup.prefix_slots().get(&typed_prefix).is_none() {
        return Err(AkitaError::InvalidProof);
    }
    let mut replay_stage3 = || {
        let _span = tracing::info_span!("binary_root_typed_stage3").entered();
        let product = akita_types::binary_source::BinaryAProduct::new(
            public.plan(),
            &rho,
            &replay.challenges,
            typed_prefix_len,
        )?;
        crate::stages::stage3::verify_stage3_session::<F, E>(
            typed_prefix_len.trailing_zeros() as usize,
            &typed_identity,
            grinding,
            0,
            |point| {
                Ok(product.evaluate(point)?
                    + typed.digit_weight(
                        point,
                        &replay.challenges,
                        alpha,
                        [&authentication[0], &authentication[1]],
                    )?)
            },
        )
    };
    let evaluate_relation = || {
        let _span = tracing::info_span!("binary_root_relation_evaluate").entered();
        relation.evaluate(&replay.challenges)
    };
    // The relation evaluation reads only the Stage 2 point, so a pool task
    // evaluates it while this thread replays Stage 3 on the transcript.
    // Errors keep their order: a Stage 3 error is returned first.
    #[cfg(feature = "parallel")]
    let (stage3, relation_evaluation) = if rayon::current_num_threads() > 1 {
        let mut relation_evaluation = None;
        let stage3 = rayon::in_place_scope(|scope| {
            scope.spawn(|_| relation_evaluation = Some(evaluate_relation()));
            replay_stage3()
        });
        (
            stage3,
            relation_evaluation.unwrap_or_else(|| {
                Err(AkitaError::Internal(
                    "binary root relation evaluation did not run".into(),
                ))
            }),
        )
    } else {
        (replay_stage3(), evaluate_relation())
    };
    #[cfg(not(feature = "parallel"))]
    let (stage3, relation_evaluation) = (replay_stage3(), evaluate_relation());
    let stage3 = stage3?;
    let relation_evaluation = relation_evaluation?;
    let expected = (relation_evaluation + stage3.claim) * witness_eval
        + (negative_bit_weight + gamma * EqPolynomial::mle(&stage1.point, &replay.challenges)?)
            * witness_eval
            * (witness_eval + E::one());
    drop(relation);
    if replay.output_claim != expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(FoldVerifyOutput {
        challenges: replay.challenges,
        setup_prefix_opening: Some((stage3.challenges, stage3.setup_prefix_eval)),
        next_witness: super::NextWitness::OuterPayload(next_witness),
        opening: witness_eval,
    })
}
