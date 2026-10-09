//! One role-independent image statement binding and schedule resolution.

use akita_algebra::binary::field_switch::SwitchField;
use akita_config::{transcript_instance_descriptor, CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_labinius_verifier::{channel::ClearChannel, lowered::LoweredRootLayout};
use akita_params::{sis::labinius::LabiniusDigitBase, BasisMode, ScheduleLookupKey};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::{
    AkitaSetupDescriptor, CommittedGroup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims,
};
use jolt_field::{CanonicalBytes, CanonicalEncoding};

use super::ImageEvaluation;
use crate::{
    session::{public_length_prefixed, NestedOpeningSession},
    ImageConfig, RootSetup, F,
};

pub(super) const SESSION_DOMAIN: &[u8] = b"akita/labinius/image-akita-opening-session/v1";

pub(super) struct BoundImageOpening<'a> {
    pub statement: GroupBatchStatement<'a, F, F>,
    pub session: NestedOpeningSession,
}

pub(super) fn bind_image_statement<'a, H: SwitchField, S: ClearChannel>(
    admitted: &RootSetup,
    setup: &AkitaSetupDescriptor,
    schedules: &TrustedScheduleCatalog<ImageConfig>,
    commitment: &'a CommittedGroup<F>,
    evaluation: ImageEvaluation<'a>,
    channel: &mut S,
) -> Result<BoundImageOpening<'a>, AkitaError> {
    // Digit base chooses the existing layout owner; image addresses are base-independent.
    let layout =
        LoweredRootLayout::new(admitted.setup(), admitted.shape(), LabiniusDigitBase::Bits2)?;
    if evaluation.point.len() != layout.image_log_len() {
        return Err(AkitaError::InvalidPointDimension {
            expected: layout.image_log_len(),
            actual: evaluation.point.len(),
        });
    }
    if commitment.profile().group
        != akita_params::PolynomialGroupLayout::singleton(layout.image_log_len())
    {
        return Err(AkitaError::InvalidInput(
            "image commitment geometry mismatch".into(),
        ));
    }
    commitment.check().map_err(|_| AkitaError::InvalidProof)?;
    let key = ScheduleLookupKey::single(commitment.profile().group);
    let row = schedules.resolve_key(&key)?;
    if row.profiles().final_group != *commitment.profile()
        || !row.profiles().precommitteds.is_empty()
    {
        return Err(AkitaError::InvalidInput(
            "image profile differs from scalar row".into(),
        ));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(1)
        .map_err(|_| AkitaError::InvalidProof)?;
    values.push(evaluation.value);
    let mut groups = Vec::new();
    groups
        .try_reserve_exact(1)
        .map_err(|_| AkitaError::InvalidProof)?;
    groups.push(PolynomialGroupClaims::new(
        evaluation.point,
        values,
        commitment,
    )?);
    let claims = OpeningClaims::from_groups(groups)?;
    claims.validate(setup)?;
    let opening_layout = claims.committed_layout()?;
    row.validate_opening_layout(&opening_layout)?;
    let (_, descriptor) = transcript_instance_descriptor::<F, ImageConfig>(
        setup,
        &opening_layout,
        row.selection(),
        row.schedule(),
        BasisMode::Lagrange,
    )?;
    let identity = admitted.identity_bytes::<H>()?;
    let mut setup_bytes = Vec::new();
    setup_bytes
        .try_reserve_exact(setup.compressed_size())
        .map_err(|_| AkitaError::InvalidProof)?;
    setup
        .serialize_compressed(&mut setup_bytes)
        .map_err(|_| AkitaError::InvalidProof)?;
    let mut commitment_bytes = Vec::new();
    commitment_bytes
        .try_reserve_exact(commitment.compressed_size())
        .map_err(|_| AkitaError::InvalidProof)?;
    commitment
        .serialize_compressed(&mut commitment_bytes)
        .map_err(|_| AkitaError::InvalidProof)?;
    let image_log = u32::try_from(layout.image_log_len()).map_err(|_| AkitaError::InvalidProof)?;
    let image_len = u64::try_from(layout.image_len()).map_err(|_| AkitaError::InvalidProof)?;
    let field = akita_params::field_modulus_be_bytes::<F>()?;

    public_length_prefixed(channel, b"akita/labinius/image-pcs/v1")?;
    public_length_prefixed(channel, &identity)?;
    public_length_prefixed(channel, &setup_bytes)?;
    public_length_prefixed(channel, ImageConfig::schedule_family_name().as_bytes())?;
    channel.public(&F::MODULUS_BITS.to_le_bytes())?;
    public_length_prefixed(channel, &field)?;
    channel.public(&image_log.to_le_bytes())?;
    channel.public(&image_len.to_le_bytes())?;
    public_length_prefixed(channel, b"Y")?;
    channel.public(&image_len.to_le_bytes())?;
    public_length_prefixed(channel, &commitment_bytes)?;
    let mut field_bytes = [0u8; F::NUM_BYTES];
    for coordinate in evaluation
        .point
        .iter()
        .chain(std::iter::once(&evaluation.value))
    {
        coordinate.to_bytes_le(&mut field_bytes);
        channel.public(&field_bytes)?;
    }
    public_length_prefixed(channel, &descriptor)?;
    let session = NestedOpeningSession::derive::<ImageConfig, S>(SESSION_DOMAIN, row, channel)?;
    Ok(BoundImageOpening {
        statement: GroupBatchStatement::new(row.selection(), claims)?,
        session,
    })
}
