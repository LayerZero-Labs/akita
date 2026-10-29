//! Canonical CPU ownership and commitment for explicitly selected binary sources.

use super::owned::{CommitOutput, CommitmentHandle, CommittedSource, PrivateSource, SourceHandle};
use super::owned_kind::SourceGroupKind;
use super::*;
use akita_algebra::ring::trinomial::{image::Q24Image, signed::signed_trinomial_product};
use akita_algebra::MinusTrinomial;
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::checked;
use akita_types::binary_source::{
    binary_public_digit_commitment, binary_root_matrix, BinaryCommittedGroupManifest,
    BINARY_SOURCE_Q,
};
use akita_types::{Commitment, CommittedGroup, RingVec};
use std::sync::Arc;

const SOURCE_RING_DEGREE: usize = 648;
const MAX_BINARY_SOURCE_MATERIAL_BYTES: usize = 1 << 27;

/// Immutable private source and q24 image hint. Neither table is a public payload.
#[derive(Clone)]
pub(super) struct BinarySourceStorage {
    manifest: BinaryCommittedGroupManifest,
    carrier_profile: akita_params::GroupCommitPhaseParams,
    setup_seed: [u8; 32],
    sources: Vec<[i128; SOURCE_RING_DEGREE]>,
    matrix: Vec<Vec<[i128; SOURCE_RING_DEGREE]>>,
    images: Vec<[Q24Image; SOURCE_RING_DEGREE]>,
    image_digits: Vec<i8>,
}

impl<F, E> CpuBackend<F, E>
where
    F: Field + CanonicalEncoding + 'static,
    E: Field + 'static,
{
    /// Import logical words using the descriptor's published signed packing map.
    ///
    /// Each inner vector supplies one polynomial's logical words. The explicit
    /// experimental manifest freezes the real q24 A and ordinary B setup; the
    /// carrier profile selects the ordinary suffix and makes no security claim
    /// about this binary group.
    pub fn import_binary_source(
        &self,
        manifest: BinaryCommittedGroupManifest,
        carrier_profile: akita_params::GroupCommitPhaseParams,
        words: Vec<Vec<u128>>,
    ) -> Result<SourceHandle<F, E>, AkitaError> {
        manifest.validate_carrier(&carrier_profile)?;
        if akita_params::field_modulus::<F>()? != akita_params::binary_setup::BINARY_SETUP_P
            || carrier_profile.group.num_polynomials() != manifest.descriptor().polynomials().len()
        {
            return Err(AkitaError::InvalidInput(
                "binary source requires the proof prime and its exact carrier polynomial count"
                    .into(),
            ));
        }
        let prepared = self.prepared()?;
        let setup_seed = prepared.expanded.descriptor.setup_seed.seed;
        let sources = pack_logical_words(manifest.descriptor(), &words)?;
        let matrix = binary_root_matrix(&setup_seed, &manifest)?;
        let images = image_hint(&manifest, &matrix, &sources)?;
        let image_digit_count = checked::product([images.len(), 8, SOURCE_RING_DEGREE])
            .ok_or(AkitaError::InvalidProof)?;
        let mut image_digits = Vec::new();
        image_digits
            .try_reserve_exact(image_digit_count)
            .map_err(|_| AkitaError::InvalidInput("binary image digit allocation failed".into()))?;
        for element in &images {
            for digit in 0..8 {
                image_digits.extend(
                    element
                        .iter()
                        .map(|coefficient| coefficient.digits()[digit]),
                );
            }
        }
        let metadata = SourceMetadata::try_new(
            carrier_profile.group.num_polynomials(),
            carrier_profile.group.num_vars(),
        )?;
        Ok(SourceHandle {
            owner: self.owner_id(),
            storage: Arc::new(BinarySourceStorage {
                manifest,
                carrier_profile,
                setup_seed,
                sources,
                matrix,
                images,
                image_digits,
            }),
            metadata,
        })
    }
}

pub(super) fn commit_binary_source<Cfg, F, E>(
    backend: &CpuBackend<F, E>,
    _family: &TrustedScheduleCatalog<Cfg>,
    source: &SourceHandle<F, E>,
    context: crate::commitment::GroupContext<'_>,
) -> Result<CommitOutput<F, E>, AkitaError>
where
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
    F: Field + CanonicalEncoding,
    E: Field,
{
    let binary = source.storage.binary_source().ok_or_else(|| {
        AkitaError::InvalidInput("binary source kind has no binary storage".into())
    })?;
    let profile = context.explicit_profile().ok_or_else(|| {
        AkitaError::InvalidInput("binary commitment requires an explicit suffix carrier".into())
    })?;
    if profile != &binary.carrier_profile
        || u64::try_from(context.group_index()).map_err(|_| AkitaError::InvalidProof)?
            != binary.manifest.setup_group()
    {
        return Err(AkitaError::InvalidInput(
            "binary source manifest differs from its explicit ordered commitment context".into(),
        ));
    }
    binary.manifest.validate_carrier(profile)?;
    let public = Commitment::new(RingVec::from_coeffs(binary_public_digit_commitment::<F>(
        &binary.setup_seed,
        binary.manifest.b_label(),
        &binary.image_digits,
    )?));
    let committed_group =
        CommittedGroup::new_binary(*profile, public.clone(), binary.manifest.clone())?;
    let private_handle = CommitmentHandle {
        owner: source.owner,
        committed: Arc::new(CommittedSource {
            source: source.storage.clone(),
            metadata: source.metadata,
            commitment_id: backend.owner().next_operation_id()?,
            parameters: *profile,
            producer_contract: Cfg::committed_source_contract()?,
            public,
            retained: None,
        }),
    };
    Ok(CommitOutput {
        committed_group,
        private_handle,
    })
}

fn pack_logical_words(
    descriptor: &akita_types::binary_source::BinarySourceDescriptor,
    words: &[Vec<u128>],
) -> Result<Vec<[i128; SOURCE_RING_DEGREE]>, AkitaError> {
    let index = descriptor.index_map();
    let owners = descriptor.polynomials().len();
    if words.len() != owners || words.iter().any(|words| words.len() != index.logical_len()) {
        return Err(AkitaError::InvalidInput(
            "binary logical word table differs from its descriptor".into(),
        ));
    }
    let rows = checked::product([4, index.columns()])
        .and_then(|block| checked::exact_div(index.padded_carriers(), block))
        .ok_or(AkitaError::InvalidProof)?;
    let count =
        checked::product([owners, index.columns(), rows]).ok_or(AkitaError::InvalidProof)?;
    let bytes = checked::product([count, size_of::<[i128; SOURCE_RING_DEGREE]>()])
        .ok_or(AkitaError::InvalidProof)?;
    if bytes > MAX_BINARY_SOURCE_MATERIAL_BYTES {
        return Err(AkitaError::InvalidInput(
            "binary source allocation exceeds its budget".into(),
        ));
    }
    let mut sources = Vec::new();
    sources
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidProof)?;
    sources.resize(count, [0; SOURCE_RING_DEGREE]);
    for (owner, polynomial) in words.iter().enumerate() {
        for (word, &value) in polynomial.iter().enumerate() {
            if index.word_width() < 128 && value >> index.word_width() != 0 {
                return Err(AkitaError::InvalidInput(
                    "binary logical word exceeds its width".into(),
                ));
            }
            for bit in 0..index.word_width() {
                let position = index.logical_to_ring(word, bit)?;
                let element = checked::mul_add(owner, index.columns(), position.column)
                    .and_then(|column| checked::mul_add(column, rows, position.ring_row))
                    .ok_or(AkitaError::InvalidProof)?;
                let coefficient = sources
                    .get_mut(element)
                    .and_then(|element| element.get_mut(position.coefficient))
                    .ok_or(AkitaError::InvalidProof)?;
                *coefficient = i128::from(position.sign) * ((value >> bit) & 1) as i128;
            }
        }
    }
    Ok(sources)
}

fn image_hint(
    manifest: &BinaryCommittedGroupManifest,
    matrix: &[Vec<[i128; SOURCE_RING_DEGREE]>],
    sources: &[[i128; SOURCE_RING_DEGREE]],
) -> Result<Vec<[Q24Image; SOURCE_RING_DEGREE]>, AkitaError> {
    let index = manifest.descriptor().index_map();
    let rows = checked::product([4, index.columns()])
        .and_then(|block| checked::exact_div(index.padded_carriers(), block))
        .ok_or(AkitaError::InvalidProof)?;
    let count = checked::product([
        manifest.descriptor().polynomials().len(),
        index.columns(),
        manifest.a_rank(),
    ])
    .ok_or(AkitaError::InvalidProof)?;
    let mut images = Vec::new();
    images
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidProof)?;
    for column in sources.chunks_exact(rows) {
        for matrix_row in matrix {
            let mut image = [0i128; SOURCE_RING_DEGREE];
            for (source, multiplier) in column.iter().zip(matrix_row) {
                let product = signed_trinomial_product::<SOURCE_RING_DEGREE, MinusTrinomial>(
                    source, multiplier,
                )?;
                for (destination, value) in image.iter_mut().zip(product) {
                    *destination = destination
                        .checked_add(value)
                        .ok_or(AkitaError::InvalidProof)?;
                }
            }
            let mut encoded = [Q24Image::from_residue(0)?; SOURCE_RING_DEGREE];
            for (destination, value) in encoded.iter_mut().zip(image) {
                let residue = u32::try_from(value.rem_euclid(i128::from(BINARY_SOURCE_Q)))
                    .map_err(|_| AkitaError::InvalidProof)?;
                *destination = Q24Image::from_residue(residue)?;
            }
            images.push(encoded);
        }
    }
    Ok(images)
}

impl<F: Field + CanonicalEncoding, E: Field> PrivateSource<F, E> for BinarySourceStorage {
    fn group_kind(&self) -> SourceGroupKind {
        SourceGroupKind::Binary
    }
    fn binary_source(&self) -> Option<&BinarySourceStorage> {
        Some(self)
    }
    fn commitment_sources(&self) -> Vec<&dyn crate::commitment::CommitmentSource<F>> {
        Vec::new()
    }
    fn dense_polynomials(&self) -> Result<Vec<crate::DensePoly<F>>, AkitaError> {
        Err(ordinary_operation())
    }
    #[cfg(feature = "response-model-diagnostics")]
    fn source_l2_sq(&self) -> Option<u128> {
        None
    }
    fn opening(
        &self,
        _: &CpuBackend<F, E>,
        _: &OperationBinding,
        _: &ValidatedRecursiveGroupOpeningPlan<'_, E>,
        _: crate::opaque::openings::PreparedOpeningSource<F, E>,
    ) -> Result<PreparedGroupOpening<E, CpuPreparedOpeningHandle<F, E>>, AkitaError> {
        Err(ordinary_operation())
    }
    fn probe(
        &self,
        _: &CpuBackend<F, E>,
        _: &ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<super::CpuAcceptedFoldHandle<F>>, AkitaError> {
        Err(ordinary_operation())
    }
    fn extension(
        &self,
        _: &CpuBackend<F, E>,
        _: usize,
        _: &[E],
    ) -> Result<super::source::PreparedExtensionOpeningGroup<E>, AkitaError> {
        Err(ordinary_operation())
    }
    fn begin_eor(
        &self,
        _: &CpuBackend<F, E>,
        _: usize,
        _: &[E],
        _: &[E],
        _: &[E],
        _: Vec<E>,
        _: E,
    ) -> Result<Box<dyn crate::opaque::eor::ExtensionOpeningSession<E>>, AkitaError> {
        Err(ordinary_operation())
    }
}

fn ordinary_operation() -> AkitaError {
    AkitaError::InvalidInput("binary source requires its selected native root protocol".into())
}

/// Private state fixed by the order-three U commitment before fold grinding.
struct BinaryRootPreparation {
    binding: OperationBinding,
    source: BinarySourceStorage,
    claim: akita_types::binary_source::BinaryTerminalClaim,
    plan: akita_types::binary_source::BinaryRootWitnessPlan,
    partials: Vec<i8>,
    ordinary_segments: Vec<Vec<i8>>,
}

pub(super) fn prepare_root_group<F, E>(
    backend: &CpuBackend<F, E>,
    session: &CpuProofSessionHandle,
    context: &ProofContext,
    handle: &CommitmentHandle<F, E>,
    request: &dyn std::any::Any,
) -> Result<(Vec<F>, Box<dyn std::any::Any + Send + Sync>), AkitaError>
where
    F: Field + CanonicalEncoding + 'static,
    E: Field + 'static,
{
    let binding = backend.binding(session, context)?;
    if let Some(request) =
        request.downcast_ref::<akita_types::binary_source::BinaryFieldSwitchSourceRequest>()
    {
        return field_switch_source(backend, &binding, context, handle, request);
    }
    let request = request
        .downcast_ref::<akita_types::binary_source::BinaryRootWitnessRequest<CommitmentHandle<F, E>>>()
        .ok_or_else(|| AkitaError::InvalidInput("native root request family mismatch".into()))?;
    let manifest = &request.manifest;
    let claim = &request.claim;
    let plan = &request.plan;
    let source = handle.committed.source.binary_source().ok_or_else(|| {
        AkitaError::InvalidInput("native root handle has no binary source".into())
    })?;
    let (schedule, root_layout) = binding.scope_lease().proof_plan()?;
    let destination_ring = schedule
        .recursive_folds
        .first()
        .map_or(schedule.terminal.d_a(), |fold| fold.params.d_a());
    let domain_len = akita_params::witness_commitment_domain_len(
        schedule.root.output_witness_len,
        destination_ring,
    )?;
    manifest.validate_carrier(&handle.committed.parameters)?;
    claim.validate(manifest.descriptor())?;
    if handle.owner != backend.owner_id()
        || context.fold_level() != 0
        || context
            .group_index()
            .and_then(|group| u64::try_from(group).ok())
            != Some(manifest.setup_group())
        || manifest != &source.manifest
        || handle.committed.parameters != source.carrier_profile
        || schedule.root.params.group_params(&root_layout, 0)?.profile != source.carrier_profile
        || handle.committed.retained.is_some()
        || plan.descriptor() != manifest.descriptor()
        || plan.rank() != manifest.a_rank()
        || plan.repacking().destination_ring_dimension() != destination_ring
        || plan.repacking().domain_len() != domain_len
    {
        return Err(AkitaError::InvalidInput(
            "native root request differs from its admitted commitment".into(),
        ));
    }
    let mut expected_plan = akita_types::binary_source::BinaryRootWitnessPlan::new(
        manifest.descriptor(),
        manifest.a_rank(),
        destination_ring,
    )?;
    let ordinary_segments = if let Some((profile, ordinary_handle)) = &request.ordinary {
        if root_layout.num_groups() != 2
            || schedule.root.params.group_params(&root_layout, 1)?.profile != *profile
        {
            return Err(AkitaError::InvalidInput(
                "ordinary preimage differs from its admitted ordered group".into(),
            ));
        }
        let ordinary_plan = akita_types::binary_source::OrdinaryNativeRootPlan::new(
            *profile,
            expected_plan.repacking().segments().len(),
        )?;
        expected_plan = expected_plan.with_ordinary(&ordinary_plan)?;
        let ordinary_context = context.for_group(1);
        backend.validate_context(session, &ordinary_context)?;
        let segments =
            super::binary_ordinary::prepare_ordinary_source(backend, ordinary_handle, profile)?;
        binding
            .scope_lease()
            .admit_commitment(&ordinary_context, ordinary_handle.committed.commitment_id)?;
        segments
    } else {
        if root_layout.num_groups() != 1 {
            return Err(AkitaError::InvalidInput(
                "native root request omits an admitted ordinary group".into(),
            ));
        }
        Vec::new()
    };
    expected_plan = expected_plan.with_domain_len(domain_len)?;
    if expected_plan.repacking() != plan.repacking() {
        return Err(AkitaError::InvalidInput(
            "native root request has a different coefficient map".into(),
        ));
    }
    let partials =
        crate::binary_root::binary_row_partials(manifest.descriptor(), claim, &source.sources)?;
    let payload = binary_public_digit_commitment::<F>(
        &source.setup_seed,
        akita_params::binary_setup::SetupStreamLabel {
            role: akita_params::binary_setup::SetupStreamRole::WitnessCommitment,
            ..manifest.b_label()
        },
        &partials,
    )?;
    binding
        .scope_lease()
        .admit_commitment(context, handle.committed.commitment_id)?;
    Ok((
        payload,
        Box::new(BinaryRootPreparation {
            binding,
            source: source.clone(),
            claim: claim.clone(),
            plan: plan.clone(),
            partials,
            ordinary_segments,
        }),
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_root_group<F, E>(
    backend: &CpuBackend<F, E>,
    session: &CpuProofSessionHandle,
    context: &ProofContext,
    preparation: &dyn std::any::Any,
    challenges: &akita_challenges::Challenges,
    logical_len: usize,
    destination_ring_dimension: usize,
) -> Result<Option<CpuWitnessHandle>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    let binding = backend.binding(session, context)?;
    let preparation = preparation
        .downcast_ref::<BinaryRootPreparation>()
        .ok_or_else(|| {
            AkitaError::InvalidInput("native root preparation family mismatch".into())
        })?;
    backend.validate_binding(&preparation.binding)?;
    preparation.binding.validate_lineage(&binding)?;
    let (schedule, _) = binding.scope_lease().proof_plan()?;
    if context.fold_level() != 0
        || context.group_index().is_some()
        || logical_len != schedule.root.output_witness_len
        || destination_ring_dimension != preparation.plan.repacking().destination_ring_dimension()
    {
        return Err(AkitaError::InvalidInput(
            "native root build differs from its admitted successor".into(),
        ));
    }
    let source = &preparation.source;
    let Some(response) = crate::binary_root::BinaryRootRelation::probe(
        source.manifest.descriptor(),
        &source.matrix,
        &source.sources,
        &source.images,
        challenges,
        akita_challenges::FoldChallengeFamily::BinaryRoot,
    )?
    else {
        return Ok(None);
    };
    let quotient = response.committed_quotient(
        source.manifest.descriptor(),
        &source.matrix,
        challenges,
        destination_ring_dimension,
    )?;
    let evaluation = crate::binary_root::BinaryEvaluationWitness::build(
        source.manifest.descriptor(),
        &preparation.claim,
        &preparation.partials,
        &response.response,
        challenges,
        akita_params::binary_setup::BINARY_SETUP_P,
    )?;
    let quotient_map = quotient.relation.repacking();
    let response_digits = natural_segment_digits(quotient_map, &quotient.digits, 0)?;
    let image_digits = natural_segment_digits(quotient_map, &quotient.digits, 1)?;
    let quotient_digits = natural_segment_digits(quotient_map, &quotient.digits, 2)?;
    let mut natural: Vec<&[i8]> = vec![
        &response_digits,
        &image_digits,
        &quotient_digits,
        &preparation.partials,
        &evaluation.parity_digits,
        &evaluation.value_digits,
    ];
    natural.extend(preparation.ordinary_segments.iter().map(Vec::as_slice));
    let mut digits = preparation.plan.repacking().repack(&natural)?;
    if logical_len > digits.len()
        || digits
            .get(logical_len..)
            .ok_or(AkitaError::InvalidProof)?
            .iter()
            .any(|&digit| digit != 0)
    {
        return Err(AkitaError::InvalidInput(
            "native root segments exceed the admitted live witness".into(),
        ));
    }
    digits.truncate(logical_len);
    super::binary_root_sessions::import_root_witness(
        backend,
        session,
        context,
        digits,
        destination_ring_dimension,
    )
    .map(Some)
}

fn natural_segment_digits(
    plan: &akita_types::root_repacking::RootCoefficientRepacking,
    table: &[i8],
    segment: usize,
) -> Result<Vec<i8>, AkitaError> {
    let shape = plan
        .segments()
        .get(segment)
        .ok_or(AkitaError::InvalidProof)?;
    let width = checked::product([
        shape.geometry.polynomial_modulus_dimension(),
        shape.geometry.coordinate_plane_count(),
    ])
    .ok_or(AkitaError::InvalidProof)?;
    let count =
        checked::product([shape.elements, shape.digits, width]).ok_or(AkitaError::InvalidProof)?;
    let mut natural = Vec::new();
    natural
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidProof)?;
    for element in 0..shape.elements {
        for digit in 0..shape.digits {
            for coefficient in 0..width {
                let address = plan.coefficient_address(segment, element, digit, coefficient)?;
                natural.push(*table.get(address).ok_or(AkitaError::InvalidProof)?);
            }
        }
    }
    Ok(natural)
}

/// Extract the authenticated owner's carriers in canonical low-column-bit order.
fn field_switch_source<F, E>(
    backend: &CpuBackend<F, E>,
    binding: &OperationBinding,
    context: &ProofContext,
    handle: &CommitmentHandle<F, E>,
    request: &akita_types::binary_source::BinaryFieldSwitchSourceRequest,
) -> Result<(Vec<F>, Box<dyn std::any::Any + Send + Sync>), AkitaError>
where
    F: Field + CanonicalEncoding + 'static,
    E: Field + 'static,
{
    use akita_algebra::ring::trinomial::module_map::{scalar_to_binary, unpack_signed_components};
    let source = handle.committed.source.binary_source().ok_or_else(|| {
        AkitaError::InvalidInput("field switch requires a binary source handle".into())
    })?;
    request
        .manifest
        .validate_carrier(&handle.committed.parameters)?;
    let (schedule, root_layout) = binding.scope_lease().proof_plan()?;
    if handle.owner != backend.owner_id()
        || context.fold_level() != 0
        || context
            .group_index()
            .and_then(|group| u64::try_from(group).ok())
            != Some(request.manifest.setup_group())
        || request.manifest != source.manifest
        || schedule.root.params.group_params(&root_layout, 0)?.profile != source.carrier_profile
        || handle.committed.parameters != source.carrier_profile
        || handle.committed.retained.is_some()
    {
        return Err(AkitaError::InvalidInput(
            "field-switch source owner mismatch".into(),
        ));
    }
    let descriptor = request.manifest.descriptor();
    let owner = descriptor
        .polynomials()
        .iter()
        .position(|id| id == &request.polynomial)
        .ok_or_else(|| AkitaError::InvalidInput("unknown binary polynomial".into()))?;
    let index = descriptor.index_map();
    let scalar_rows = checked::exact_div(index.padded_carriers(), index.columns())
        .ok_or(AkitaError::Internal("invalid carrier geometry".into()))?;
    let ring_rows = checked::exact_div(scalar_rows, 4)
        .ok_or(AkitaError::Internal("invalid source ring geometry".into()))?;
    let mut carriers = Vec::new();
    carriers
        .try_reserve_exact(index.padded_carriers())
        .map_err(|_| AkitaError::InvalidInput("carrier allocation failed".into()))?;
    for carrier in 0..index.padded_carriers() {
        let row = carrier / index.columns();
        let element = checked::mul_add(owner, index.columns(), carrier % index.columns())
            .and_then(|x| checked::mul_add(x, ring_rows, row / 4))
            .ok_or(AkitaError::Internal("carrier address overflow".into()))?;
        let components = unpack_signed_components(
            source
                .sources
                .get(element)
                .ok_or(AkitaError::Internal("missing source ring".into()))?,
        )?;
        let [low, high, padding] = scalar_to_binary(
            components
                .get(row % 4)
                .ok_or(AkitaError::Internal("missing source component".into()))?,
        )?
        .to_words();
        if padding != 0 {
            return Err(AkitaError::Internal(
                "nonzero source carrier padding".into(),
            ));
        }
        carriers.push(u128::from(low) | (u128::from(high) << 64));
    }
    binding
        .scope_lease()
        .admit_commitment(context, handle.committed.commitment_id)?;
    Ok((Vec::new(), Box::new(carriers)))
}
