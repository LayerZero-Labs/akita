//! Row admission, transcript binding and framing shared by both oracles.

use crate::{
    family::{prime::prime_opening_claim, tables::DigitTables, FieldFamily},
    session::{canonical_bytes, public_length_prefixed, NestedOpeningSession},
    RootSetup,
};
use akita_config::{
    required_setup_prefix_slot_ids_for_schedule, transcript_instance_descriptor, CommitmentConfig,
    ResolvedScheduleRow, TrustedScheduleCatalog,
};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{
    channel::ClearChannel, lowered::LoweredRootLayout, root::RootEvaluationClaims,
};
use akita_params::{BasisMode, CompressionChainPlan, GroupCommitPhaseParams, SetupPrefixSlotId};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_types::{
    instance_descriptor::SetupSection, AkitaSetupDescriptor, Commitment, CommittedGroup,
    GroupBatchStatement, OpeningClaims, PolynomialGroupClaims, RingVec,
};
use jolt_field::{ExtField, Zero};

/// Domain label of the statement the image binding absorbs.
const STATEMENT_DOMAIN: &[u8] = b"akita/labinius/pcs/v1";

/// The reduction's layout of a root setup over a family's field pair.
pub(crate) fn root_layout<P: FieldFamily>(
    root: &RootSetup,
) -> Result<LoweredRootLayout, AkitaError> {
    if !matches!(P::Challenge::DEGREE, 1 | 2) {
        return Err(AkitaError::InvalidSetup(
            "prime opening supports extension degrees one and two".into(),
        ));
    }
    LoweredRootLayout::new::<P::Base, P::Challenge, _, _>(root.setup(), root.shape())
}

/// The grouped response row with the image and optional element producer.
pub(crate) fn response_row<'a, P: FieldFamily>(
    layout: &LoweredRootLayout,
    catalog: &'a TrustedScheduleCatalog<P::Digits>,
    elements: Option<&TrustedScheduleCatalog<P::Elements>>,
) -> Result<&'a ResolvedScheduleRow, AkitaError> {
    let tables = DigitTables {
        image_log_len: layout.image_log_len(),
        response_log_len: layout.witness_log_len(),
        element_log_len: checked::sum([
            layout.prime_log_len(),
            usize::from(P::Challenge::DEGREE == 2),
        ])
        .ok_or_else(|| AkitaError::InvalidSetup("prime table dimension overflow".into()))?,
    };
    let element = elements
        .map(|catalog| {
            catalog
                .resolve_key(&tables.element_key())
                .map(|row| row.profiles().final_group)
        })
        .transpose()?;
    catalog.resolve_key(&tables.response_key(catalog, element)?)
}

/// The image table's commitment profile named by the grouped response row.
fn image_profile(row: &ResolvedScheduleRow) -> Result<GroupCommitPhaseParams, AkitaError> {
    match row.profiles().precommitteds.as_slice() {
        [image] | [image, _] => Ok(*image),
        _ => Err(AkitaError::InvalidSetup(
            "the response row must name the image first with at most one prime group".into(),
        )),
    }
}

/// Require every setup-prefix slot the row's schedule plans to be in the setup.
pub(crate) fn ensure_setup_prefix_coverage(
    row: &ResolvedScheduleRow,
    contains: impl Fn(&SetupPrefixSlotId) -> bool,
) -> Result<(), AkitaError> {
    let layout = row.profiles().opening_layout()?;
    for id in required_setup_prefix_slot_ids_for_schedule(row.schedule(), &layout)? {
        if !contains(&id) {
            return Err(AkitaError::InvalidSetup(
                "planned setup-prefix slot is missing from setup".into(),
            ));
        }
    }
    Ok(())
}

/// Each oracle call enters the failed state; only success permits the next.
/// After an unexpected call or an error, later calls reject without touching
/// the channel.
#[derive(Default, PartialEq)]
pub(crate) enum Order {
    #[default]
    Image,
    Prime,
    Response,
    Discharge,
    Failed,
}

impl Order {
    pub(crate) fn take(&mut self, expected: Self) -> Result<(), AkitaError> {
        if std::mem::replace(self, Self::Failed) != expected {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }
}

/// Canonical byte length of a commitment with this profile.
///
/// The serializer supplies its fixed header and the compression owner its
/// terminal coefficient count. There is no profile-only serialized-size
/// function.
pub(crate) fn commitment_size<P: FieldFamily>(
    profile: GroupCommitPhaseParams,
) -> Result<usize, AkitaError> {
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
        Commitment::new(RingVec::from_coeffs(Vec::<P::Base>::new())),
    )
    .compressed_size();
    checked::product([terminal, P::Base::zero().compressed_size()])
        .and_then(|coefficients| checked::sum([header, coefficients]))
        .ok_or(AkitaError::InvalidProof)
}

/// Exchange one canonical commitment of a trusted, fixed-size profile.
/// The prover supplies its commitment; the verifier reads exactly the profile's size.
pub(crate) fn exchange_commitment<P: FieldFamily, S: ClearChannel>(
    profile: GroupCommitPhaseParams,
    commitment: Option<&CommittedGroup<P::Base>>,
    channel: &mut S,
) -> Result<CommittedGroup<P::Base>, AkitaError> {
    let length = commitment_size::<P>(profile)?;
    let mut bytes = match commitment {
        Some(commitment) => {
            if *commitment.profile() != profile {
                return Err(AkitaError::InvalidProof);
            }
            canonical_bytes(commitment)?
        }
        None => {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(length)
                .map_err(|_| AkitaError::InvalidProof)?;
            bytes.resize(length, 0);
            bytes
        }
    };
    if bytes.len() != length {
        return Err(AkitaError::InvalidProof);
    }
    channel
        .message(&mut bytes)
        .map_err(|_| AkitaError::InvalidProof)?;
    let received = CommittedGroup::<P::Base>::deserialize_compressed_exact(&bytes, &())
        .map_err(|_| AkitaError::InvalidProof)?;
    if *received.profile() != profile || canonical_bytes(&received)? != bytes {
        return Err(AkitaError::InvalidProof);
    }
    Ok(received)
}

/// Absorb the statement that owns the image table.
///
/// The reduction has already absorbed the root setup's identity and the field
/// pair. This adds the nested Akita setup's stable identity without provisioned
/// capacity, the schedule family and the image commitment, whose encoding
/// carries its commitment profile.
pub(crate) fn bind_image<P: FieldFamily, S: ClearChannel>(
    expected: &LoweredRootLayout,
    layout: &LoweredRootLayout,
    setup: &AkitaSetupDescriptor,
    row: &ResolvedScheduleRow,
    commitment: &CommittedGroup<P::Base>,
    channel: &mut S,
) -> Result<(), AkitaError> {
    if layout != expected || *commitment.profile() != image_profile(row)? {
        return Err(AkitaError::InvalidProof);
    }
    commitment.check().map_err(|_| AkitaError::InvalidProof)?;
    public_length_prefixed(channel, STATEMENT_DOMAIN)?;
    let setup_identity = SetupSection::from_parts(
        P::Digits::decomposition(),
        P::Digits::sis_modulus_profile(),
        &setup.setup_seed,
    )
    .map_err(|_| AkitaError::InvalidProof)?;
    public_length_prefixed(channel, &canonical_bytes(&setup_identity)?)?;
    public_length_prefixed(channel, P::Digits::schedule_family_name().as_bytes())?;
    public_length_prefixed(channel, &canonical_bytes(commitment)?)
}

/// Claims in the row's order: image, optional prime left opening, response.
pub(crate) fn grouped_claims<'a, P: FieldFamily, G>(
    layout: &LoweredRootLayout,
    claims: &'a RootEvaluationClaims<P::Challenge>,
    image: G,
    prime: Option<G>,
    response: G,
) -> Result<OpeningClaims<'a, P::Challenge, G>, AkitaError> {
    if claims.image_point.len() != layout.image_log_len()
        || claims.response_point.len() != layout.witness_log_len()
    {
        return Err(AkitaError::InvalidProof);
    }
    let prime = match (&claims.prime, prime) {
        (Some((point, value)), Some(commitment)) => {
            let (point, value) = prime_opening_claim::<P>(layout, point, *value)?;
            Some((
                akita_types::proof::scheme::OpeningPoints::from(point),
                value,
                commitment,
            ))
        }
        (None, None) => None,
        _ => return Err(AkitaError::InvalidProof),
    };
    let mut groups = Vec::new();
    groups
        .try_reserve_exact(if prime.is_some() { 3 } else { 2 })
        .map_err(|_| AkitaError::InvalidProof)?;
    let entries = std::iter::once((
        claims.image_point.as_slice().into(),
        claims.image_value,
        image,
    ))
    .chain(prime)
    .chain(std::iter::once((
        claims.response_point.as_slice().into(),
        claims.response_value,
        response,
    )));
    for (point, value, commitment) in entries {
        let mut values = Vec::new();
        values
            .try_reserve_exact(1)
            .map_err(|_| AkitaError::InvalidProof)?;
        values.push(value);
        groups.push(PolynomialGroupClaims::new(point, values, commitment)?);
    }
    OpeningClaims::from_groups(groups)
}

/// The grouped opening's statement over its public commitments.
pub(crate) fn opening_statement<'a, P: FieldFamily>(
    layout: &LoweredRootLayout,
    row: &ResolvedScheduleRow,
    claims: &'a RootEvaluationClaims<P::Challenge>,
    image: &'a CommittedGroup<P::Base>,
    prime: Option<&'a CommittedGroup<P::Base>>,
    response: &'a CommittedGroup<P::Base>,
) -> Result<GroupBatchStatement<'a, P::Challenge, P::Base>, AkitaError> {
    GroupBatchStatement::new(
        row.selection(),
        grouped_claims::<P, _>(layout, claims, image, prime, response)?,
    )
}

/// Absorb the grouped opening's instance and claims, then derive its session.
///
/// The instance descriptor covers the setup, the opening layout, the selected
/// row and the basis, but no point and no value, so every group's claim is
/// absorbed after it in the row's order, as the canonical encoding of its
/// point coordinates followed by its value.
pub(crate) fn bind_opening<P: FieldFamily, S: ClearChannel>(
    setup: &AkitaSetupDescriptor,
    row: &ResolvedScheduleRow,
    statement: &GroupBatchStatement<'_, P::Challenge, P::Base>,
    channel: &mut S,
) -> Result<NestedOpeningSession, AkitaError> {
    let groups = statement.claims();
    groups.validate(setup)?;
    let layout = groups.committed_layout()?;
    row.validate_opening_layout(&layout)?;
    let (_, descriptor) = transcript_instance_descriptor::<P::Base, P::Digits>(
        setup,
        &layout,
        row.selection(),
        row.schedule(),
        BasisMode::Lagrange,
    )?;
    public_length_prefixed(channel, &descriptor)?;
    let mut bytes = Vec::new();
    for group in groups.groups() {
        for value in group.point().iter().chain(group.evaluations()) {
            bytes
                .try_reserve(value.compressed_size())
                .map_err(|_| AkitaError::InvalidProof)?;
            value
                .serialize_compressed(&mut bytes)
                .map_err(|_| AkitaError::InvalidProof)?;
        }
    }
    channel.public(&bytes)?;
    NestedOpeningSession::derive::<P::Digits, S>(row, channel)
}
