//! Shared row admission, canonical framing and statement binding.

use crate::{
    config::DigitConfig,
    session::{canonical_bytes, public_length_prefixed, NestedOpeningSession},
    ImageConfig, RootSetup, F,
};
use akita_config::{
    transcript_instance_descriptor, CommitmentConfig, ResolvedScheduleRow, SetupRequirements,
    TrustedScheduleCatalog,
};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{
    channel::ClearChannel, lowered::LoweredRootLayout, root::RootEvaluationClaims,
};
use akita_params::{
    BasisMode, CompressionChainPlan, GroupCommitPhaseParams, PolynomialGroupLayout,
};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::{
    AkitaSetupDescriptor, Commitment, CommittedGroup, GroupBatchStatement, OpeningClaims,
    PolynomialGroupClaims, RingVec,
};
use jolt_field::{CanonicalBytes, Zero};

pub(super) const OPENING_DOMAIN: &[u8] = b"akita/labinius/root-akita-opening-session/v1";

/// Every call enters the terminal failed state; only success permits the next call.
/// After an unexpected call or error, later calls reject without channel activity.
#[derive(Default, PartialEq)]
pub(super) enum Order {
    #[default]
    Image,
    Response,
    Discharge,
    Failed,
}
impl Order {
    pub(super) fn take(&mut self, expected: Self) -> Result<(), AkitaError> {
        let previous = std::mem::replace(self, Self::Failed);
        if previous != expected {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }
}

pub(super) fn resolve_rows<'a, C: DigitConfig>(
    admitted: &RootSetup,
    image: &'a TrustedScheduleCatalog<ImageConfig>,
    digits: &'a TrustedScheduleCatalog<C>,
) -> Result<(&'a ResolvedScheduleRow, &'a ResolvedScheduleRow), AkitaError> {
    let sizing = crate::RootPcsSizing::new(
        admitted.shape().profile(),
        admitted.log_num_cells(),
        admitted.log_fold_width(),
        admitted.lambda_fold(),
        C::BASE,
    )?;
    let image_row = image.resolve_key(&sizing.image_key())?;
    let grouped = digits.resolve_key(&sizing.grouped_digit_key(image)?)?;
    Ok((image_row, grouped))
}

pub(super) fn admit_catalogs<C: DigitConfig>(
    admitted: &RootSetup,
    image: &TrustedScheduleCatalog<ImageConfig>,
    digits: &TrustedScheduleCatalog<C>,
    setup: &AkitaSetupDescriptor,
) -> Result<(), AkitaError> {
    resolve_rows(admitted, image, digits)?;
    let requirements =
        SetupRequirements::from_catalog(image, setup.max_num_vars, setup.max_num_batched_polys)?
            .union(SetupRequirements::from_catalog(
                digits,
                setup.max_num_vars,
                setup.max_num_batched_polys,
            )?)?;
    if setup.num_field_elements < requirements.matrix_capacity().num_field_elements {
        return Err(AkitaError::InvalidSetup(
            "Akita setup does not cover the catalog union".into(),
        ));
    }
    Ok(())
}

/// Ask the serializer for its fixed header and the compression owner for its
/// terminal count. There is no public profile-only serialized-size function.
pub(crate) fn commitment_size(profile: GroupCommitPhaseParams) -> Result<usize, AkitaError> {
    let source = profile.outer_slice_count.complete_source_coefficients(
        profile.outer.matrix.output_rank(),
        profile.outer.matrix.ring_dimension(),
    )?;
    let terminal = CompressionChainPlan::for_complete_source(
        profile.outer.matrix.sis_table_key().modulus_profile,
        source,
    )?
    .terminal_coefficients();
    let header = CommittedGroup::new(
        profile,
        Commitment::new(RingVec::from_coeffs(Vec::<F>::new())),
    )
    .compressed_size();
    checked::sum([
        header,
        checked::product([terminal, F::zero().compressed_size()])
            .ok_or(AkitaError::InvalidProof)?,
    ])
    .ok_or(AkitaError::InvalidProof)
}

pub(super) fn validate_image(
    commitment: &CommittedGroup<F>,
    row: &ResolvedScheduleRow,
) -> Result<(), AkitaError> {
    if *commitment.profile() != row.profiles().final_group
        || !row.profiles().precommitteds.is_empty()
    {
        return Err(AkitaError::InvalidProof);
    }
    commitment.check().map_err(|_| AkitaError::InvalidProof)
}

pub(super) fn bind_image<C: DigitConfig, S: ClearChannel>(
    layout: &LoweredRootLayout,
    admitted: &RootSetup,
    setup: &AkitaSetupDescriptor,
    image_row: &ResolvedScheduleRow,
    commitment: &CommittedGroup<F>,
    channel: &mut S,
) -> Result<(), AkitaError> {
    if *layout != LoweredRootLayout::new(admitted.setup(), admitted.shape(), C::BASE)?
        || commitment.profile().group != PolynomialGroupLayout::singleton(layout.image_log_len())
    {
        return Err(AkitaError::InvalidProof);
    }
    validate_image(commitment, image_row)?;
    let setup_bytes = canonical_bytes(setup)?;
    let commitment_bytes = canonical_bytes(commitment)?;
    public_length_prefixed(channel, b"akita/labinius/root-pcs/v1")?;
    public_length_prefixed(channel, &setup_bytes)?;
    public_length_prefixed(channel, ImageConfig::schedule_family_name().as_bytes())?;
    public_length_prefixed(channel, C::schedule_family_name().as_bytes())?;
    public_length_prefixed(channel, &commitment_bytes)
}

pub(super) fn grouped_claims<'a, G>(
    claims: &'a RootEvaluationClaims<F>,
    image: G,
    response: G,
) -> Result<OpeningClaims<'a, F, G>, AkitaError> {
    let mut groups = Vec::new();
    groups
        .try_reserve_exact(2)
        .map_err(|_| AkitaError::InvalidProof)?;
    for (point, value, commitment) in [
        (&claims.image_point, claims.image_value, image),
        (&claims.response_point, claims.response_value, response),
    ] {
        let mut values = Vec::new();
        values
            .try_reserve_exact(1)
            .map_err(|_| AkitaError::InvalidProof)?;
        values.push(value);
        groups.push(PolynomialGroupClaims::new(
            point.as_slice(),
            values,
            commitment,
        )?);
    }
    OpeningClaims::from_groups(groups)
}

pub(super) fn bind_opening<'a, C: DigitConfig, S: ClearChannel>(
    setup: &AkitaSetupDescriptor,
    row: &ResolvedScheduleRow,
    claims: &'a RootEvaluationClaims<F>,
    image: &'a CommittedGroup<F>,
    response: &'a CommittedGroup<F>,
    channel: &mut S,
) -> Result<(GroupBatchStatement<'a, F, F>, NestedOpeningSession), AkitaError> {
    let groups = grouped_claims(claims, image, response)?;
    groups.validate(setup)?;
    let layout = groups.committed_layout()?;
    row.validate_opening_layout(&layout)?;
    let (_, descriptor) = transcript_instance_descriptor::<F, C>(
        setup,
        &layout,
        row.selection(),
        row.schedule(),
        BasisMode::Lagrange,
    )?;
    public_length_prefixed(channel, &descriptor)?;
    // The instance descriptor covers layout and basis, but no points or values.
    // The reduction already fixes these; explicitly bind them again in [Y,W] order.
    let mut bytes = [0u8; F::NUM_BYTES];
    for value in claims
        .image_point
        .iter()
        .chain(std::iter::once(&claims.image_value))
        .chain(claims.response_point.iter())
        .chain(std::iter::once(&claims.response_value))
    {
        value.to_bytes_le(&mut bytes);
        channel.public(&bytes)?;
    }
    let session = NestedOpeningSession::derive::<C, S>(OPENING_DOMAIN, row, channel)?;
    Ok((GroupBatchStatement::new(row.selection(), groups)?, session))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use akita_config::{policy_of, ValidatedScheduleCatalog};
    use akita_params::CommittedGroupBatchProfile;
    use akita_params::ScheduleLookupKey;
    use akita_planner::emit::{GroupedGenerationRequest, PrecommittedProducer};
    use jolt_field::One;

    #[test]
    fn discharge_rejects_swapped_w_y_group_order_after_shapes_match() {
        type C = crate::config::Digits2;
        let image_key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(10));
        let image_plan = akita_planner::find_schedule(
            &image_key,
            ImageConfig::committed_source_contract().unwrap(),
            &[],
            &policy_of::<ImageConfig>(),
            ImageConfig::ring_challenge_config,
        )
        .unwrap();
        let image_profile = GroupCommitPhaseParams::try_from_params(
            image_key.final_group,
            &image_plan.schedule.root.params,
        )
        .unwrap();
        let producer = PrecommittedProducer::try_new(
            image_profile,
            ImageConfig::committed_source_contract().unwrap(),
        )
        .unwrap();
        let request =
            GroupedGenerationRequest::new(PolynomialGroupLayout::singleton(11), vec![producer]);
        let key = request.key();
        let planned = akita_planner::find_schedule(
            &key,
            C::committed_source_contract().unwrap(),
            &request.source_contracts(),
            &policy_of::<C>(),
            C::ring_challenge_config,
        )
        .unwrap();
        let profiles = CommittedGroupBatchProfile {
            final_group: GroupCommitPhaseParams::try_from_params(
                key.final_group,
                &planned.schedule.root.params,
            )
            .unwrap(),
            precommitteds: key.precommitteds.clone(),
        };
        let catalog = TrustedScheduleCatalog::<C>::new(
            ValidatedScheduleCatalog::try_new(
                C::schedule_family_name(),
                [(profiles.clone(), planned.schedule)],
                &policy_of::<C>(),
                C::ring_challenge_config,
            )
            .unwrap(),
        )
        .unwrap();
        let row = catalog.resolve_key(&key).unwrap();
        let image = CommittedGroup::new(
            image_profile,
            Commitment::new(RingVec::from_coeffs(Vec::<F>::new())),
        );
        let response = CommittedGroup::new(
            profiles.final_group,
            Commitment::new(RingVec::from_coeffs(Vec::<F>::new())),
        );
        let claims = RootEvaluationClaims {
            image_point: vec![F::zero(); 10],
            image_value: F::zero(),
            response_point: vec![F::one(); 11],
            response_value: F::one(),
        };
        let honest = grouped_claims(&claims, &image, &response).unwrap();
        row.validate_opening_layout(&honest.committed_layout().unwrap())
            .unwrap();
        // Reverse complete groups, including their own points and values, so
        // each shape is valid and only the selected row's group order differs.
        let swapped = RootEvaluationClaims {
            image_point: claims.response_point.clone(),
            image_value: claims.response_value,
            response_point: claims.image_point.clone(),
            response_value: claims.image_value,
        };
        let reversed = grouped_claims(&swapped, &response, &image).unwrap();
        let layout = reversed.committed_layout().unwrap();
        assert!(matches!(
            row.validate_opening_layout(&layout),
            Err(AkitaError::InvalidInput(_))
        ));
        let requirements = SetupRequirements::from_catalog(&catalog, 11, 2).unwrap();
        let setup = akita_pcs::new_prover_setup(&requirements).unwrap();
        let descriptor = setup.expanded.descriptor();
        let mut honest_channel =
            akita_transcript::new_prover_channel(b"swapped-groups-test", b"").unwrap();
        bind_opening::<C, _>(
            descriptor,
            row,
            &claims,
            &image,
            &response,
            &mut honest_channel,
        )
        .unwrap();
        let mut channel =
            akita_transcript::new_prover_channel(b"swapped-groups-test", b"").unwrap();
        let mut control =
            akita_transcript::new_prover_channel(b"swapped-groups-test", b"").unwrap();
        assert!(matches!(
            bind_opening::<C, _>(descriptor, row, &swapped, &response, &image, &mut channel),
            Err(AkitaError::InvalidInput(_))
        ));
        assert_eq!(
            channel.challenge_block().unwrap(),
            control.challenge_block().unwrap()
        );
    }
}
