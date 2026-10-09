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
    ScheduleLookupKey,
};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::{
    AkitaSetupDescriptor, Commitment, CommittedGroup, GroupBatchStatement, OpeningClaims,
    PolynomialGroupClaims, RingVec,
};
use jolt_field::{CanonicalBytes, Zero};

pub(super) const OPENING_DOMAIN: &[u8] = b"akita/labinius/root-akita-opening-session/v1";

/// A failed transition also consumes its slot: an oracle is never retryable.
#[derive(Default)]
pub(super) struct Order(u8);
impl Order {
    pub(super) fn take(&mut self, expected: u8) -> Result<(), AkitaError> {
        if self.0 != expected {
            return Err(AkitaError::InvalidProof);
        }
        self.0 = expected + 1;
        Ok(())
    }
}

pub(super) fn resolve_rows<'a, C: DigitConfig>(
    admitted: &RootSetup,
    image: &'a TrustedScheduleCatalog<ImageConfig>,
    digits: &'a TrustedScheduleCatalog<C>,
) -> Result<(&'a ResolvedScheduleRow, &'a ResolvedScheduleRow), AkitaError> {
    let layout = LoweredRootLayout::new(admitted.setup(), admitted.shape(), C::BASE)?;
    let image_row = image.resolve_key(&ScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(layout.image_log_len()),
    ))?;
    let mut precommitteds = Vec::new();
    precommitteds
        .try_reserve_exact(1)
        .map_err(|_| AkitaError::InvalidSetup("root row allocation failed".into()))?;
    precommitteds.push(image_row.profiles().final_group);
    let grouped = digits.resolve_key(&ScheduleLookupKey {
        final_group: PolynomialGroupLayout::singleton(layout.witness_log_len()),
        precommitteds,
    })?;
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
pub(super) fn commitment_size(profile: GroupCommitPhaseParams) -> Result<usize, AkitaError> {
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
