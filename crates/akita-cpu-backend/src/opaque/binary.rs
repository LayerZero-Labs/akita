//! Canonical CPU ownership and commitment for explicitly selected binary sources.

use super::owned::{CommitOutput, CommitmentHandle, CommittedSource, PrivateSource, SourceHandle};
use super::owned_kind::SourceGroupKind;
use super::*;
use akita_algebra::ring::trinomial::image::Q24Image;
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::checked;
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::binary::{binary_root_matrix, BinaryCommittedGroupManifest, BINARY_SOURCE_Q};
use akita_types::{Commitment, CommittedGroup, RingVec};
use std::sync::Arc;

mod digit_commitment;
mod packed;
mod projection;
use digit_commitment::digit_commitment;
use packed::{fold_packed, image_hint};

#[path = "binary_import.rs"]
mod import;
use import::BinarySourceInput;

const SOURCE_RING_DEGREE: usize = 648;
const MAX_BINARY_SOURCE_MATERIAL_BYTES: usize = 1 << 27;

/// Accepted fold challenges, response and the P64 residual used to produce K.
struct AcceptedResponse {
    challenges: akita_challenges::Challenges,
    response: Vec<[i128; SOURCE_RING_DEGREE]>,
    residuals: Option<Vec<[i128; SOURCE_RING_DEGREE]>>,
}

/// Immutable private source and q24 image hint. Neither table is a public payload.
///
/// `accepted` is a private cache, not source material: the root build leaves
/// its response there and relation preparation takes it back when challenges
/// agree. P64 also caches the exact residual fixed before the outgoing
/// commitment. Both are functions of the private source and accepted fold.
#[derive(Clone)]
pub(super) struct BinarySourceStorage {
    manifest: BinaryCommittedGroupManifest,
    carrier_profile: akita_params::GroupCommitPhaseParams,
    setup_seed: [u8; 32],
    sources: Arc<Vec<u128>>,
    matrix: Arc<Vec<Vec<[i128; SOURCE_RING_DEGREE]>>>,
    images: Arc<Vec<[Q24Image; SOURCE_RING_DEGREE]>>,
    image_digits: Arc<Vec<i8>>,
    accepted: Arc<std::sync::Mutex<Option<AcceptedResponse>>>,
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
        self.import_binary_material(manifest, carrier_profile, BinarySourceInput::Logical(words))
    }

    /// Import compact low-bit-first carriers under the exact source descriptor.
    ///
    /// Each polynomial supplies `ceil(logical_len / (128 / word_width))` u128
    /// carriers, with zero unused high bits in its last carrier. Committed
    /// padding carriers and coordinates 128..161 are constructed as zeros.
    /// The logical length is checked against the manifest before allocation.
    /// Carriers are stored in the packed kernel order, never as expanded rings.
    pub fn import_packed_binary_source(
        &self,
        manifest: BinaryCommittedGroupManifest,
        carrier_profile: akita_params::GroupCommitPhaseParams,
        logical_len: usize,
        carriers: Vec<Vec<u128>>,
    ) -> Result<SourceHandle<F, E>, AkitaError> {
        self.import_binary_material(
            manifest,
            carrier_profile,
            BinarySourceInput::Packed {
                logical_len,
                carriers,
            },
        )
    }

    fn import_binary_material(
        &self,
        manifest: BinaryCommittedGroupManifest,
        carrier_profile: akita_params::GroupCommitPhaseParams,
        input: BinarySourceInput,
    ) -> Result<SourceHandle<F, E>, AkitaError> {
        manifest.validate_carrier(&carrier_profile)?;
        if akita_params::field_modulus::<F>()? != manifest.field_profile().modulus()
            || carrier_profile.group.num_polynomials() != manifest.descriptor().polynomials().len()
        {
            return Err(AkitaError::InvalidInput(
                "binary source requires the proof prime and its exact carrier polynomial count"
                    .into(),
            ));
        }
        let prepared = self.prepared()?;
        let setup_seed = prepared.expanded.descriptor.setup_seed.seed;
        let span = tracing::info_span!("binary_commit_pack_words").entered();
        let sources = input.prepare(manifest.descriptor())?;
        drop(span);
        let span = tracing::info_span!("binary_commit_a_matrix").entered();
        let matrix = binary_root_matrix(&setup_seed, &manifest)?;
        drop(span);
        let span = tracing::info_span!("binary_commit_image_hint").entered();
        let images = image_hint(manifest.descriptor(), manifest.a_rank(), &matrix, &sources)?;
        drop(span);
        let span = tracing::info_span!("binary_commit_image_digits").entered();
        let image_digits = crate::binary_root::BinaryRootRelation::image_digits(&images)?;
        drop(span);
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
                sources: Arc::new(sources),
                matrix: Arc::new(matrix),
                images: Arc::new(images),
                image_digits: Arc::new(image_digits),
                accepted: Arc::default(),
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
    F: Field + CanonicalEncoding + Valid + AkitaSerialize,
    E: Field,
{
    let binary = source.storage.binary().ok_or_else(|| {
        AkitaError::InvalidInput("binary source kind has no binary storage".into())
    })?;
    let profile = context.explicit_profile().ok_or_else(|| {
        AkitaError::InvalidInput("binary commitment requires an explicit suffix carrier".into())
    })?;
    if profile != &binary.carrier_profile {
        return Err(AkitaError::InvalidInput(
            "binary source differs from its explicit carrier profile".into(),
        ));
    }
    let group_index = u64::try_from(context.group_index()).map_err(|_| {
        AkitaError::InvalidInput("binary commitment group index does not fit u64".into())
    })?;
    if group_index != binary.manifest.setup_group() {
        return Err(AkitaError::InvalidInput(
            "binary commitment group index differs from the source setup group".into(),
        ));
    }
    binary.manifest.validate_carrier(profile)?;
    let span = tracing::info_span!("binary_commit_b_payload").entered();
    let public = Commitment::new(RingVec::from_coeffs(digit_commitment::<F>(
        &binary.setup_seed,
        binary.manifest.b_label(),
        &binary.image_digits,
    )?));
    drop(span);
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

impl<F: Field + CanonicalEncoding, E: Field> PrivateSource<F, E> for BinarySourceStorage {
    fn group_kind(&self) -> SourceGroupKind {
        SourceGroupKind::Binary
    }
    fn binary(&self) -> Option<&BinarySourceStorage> {
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
    claim: akita_types::binary::BinaryTerminalClaim,
    prime_digits: Vec<i8>,
    plan: akita_types::binary::BinaryRootWitnessPlan,
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
        request.downcast_ref::<akita_types::binary::BinaryFieldSwitchSourceRequest>()
    {
        return field_switch_source(backend, &binding, context, handle, request);
    }
    if let Some(request) = request.downcast_ref::<akita_types::binary::BinaryProjectionRequest>() {
        return projection::prepare(backend, &binding, context, handle, request);
    }
    let request = request
        .downcast_ref::<akita_types::binary::BinaryRootWitnessRequest<CommitmentHandle<F, E>>>()
        .ok_or_else(|| AkitaError::InvalidInput("native root request family mismatch".into()))?;
    let manifest = &request.manifest;
    let claim = &request.claim;
    let plan = &request.plan;
    let source = handle.committed.source.binary().ok_or_else(|| {
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
        || plan.field_profile() != manifest.field_profile()
        || plan.repacking().destination_ring_dimension() != destination_ring
        || plan.repacking().domain_len() != domain_len
    {
        return Err(AkitaError::InvalidInput(
            "native root request differs from its admitted commitment".into(),
        ));
    }
    let mut expected_plan = akita_types::binary::BinaryRootWitnessPlan::for_profile(
        manifest.descriptor(),
        manifest.a_rank(),
        destination_ring,
        manifest.field_profile(),
        request.prime.planes().len(),
    )?;
    let ordinary_segments = if let Some((profile, ordinary_handle)) = &request.ordinary {
        if root_layout.num_groups() != 2
            || schedule.root.params.group_params(&root_layout, 1)?.profile != *profile
        {
            return Err(AkitaError::InvalidInput(
                "ordinary preimage differs from its admitted ordered group".into(),
            ));
        }
        let ordinary_plan = akita_types::binary::OrdinaryNativeRootPlan::new(
            *profile,
            expected_plan.ordinary_first_segment(),
        )?;
        expected_plan = expected_plan.with_ordinary(&ordinary_plan)?;
        let ordinary_context = context.for_group(1);
        backend.validate_context(session, &ordinary_context)?;
        let segments = backend.ordinary_root_preimage(ordinary_handle, &ordinary_plan)?;
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
    let span = tracing::info_span!("binary_root_row_partials").entered();
    let partials = crate::binary_root::evaluation::packed_row_partials(
        manifest.descriptor(),
        claim,
        &source.sources,
    )?;
    drop(span);
    let span = tracing::info_span!("binary_root_prime_digits").entered();
    let prime_digits = if request.prime.planes().is_empty() {
        Vec::new()
    } else {
        crate::binary_root::prime::packed_prime_plane_digits::<F>(
            manifest.descriptor(),
            &request.prime,
            &source.sources,
        )?
    };
    drop(span);
    let span = tracing::info_span!("binary_root_d_payload").entered();
    let mut d_digits = partials.clone();
    d_digits
        .try_reserve_exact(prime_digits.len())
        .map_err(|_| AkitaError::InvalidProof)?;
    d_digits.extend_from_slice(&prime_digits);
    let payload = digit_commitment::<F>(
        &source.setup_seed,
        manifest.d_label(request.prime.planes().len())?,
        &d_digits,
    )?;
    drop(span);
    binding
        .scope_lease()
        .admit_commitment(context, handle.committed.commitment_id)?;
    Ok((
        payload,
        Box::new(BinaryRootPreparation {
            binding,
            source: source.clone(),
            claim: claim.clone(),
            prime_digits,
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
    let span = tracing::info_span!("binary_root_probe_response").entered();
    let Some(response) = fold_packed(
        source.manifest.descriptor(),
        &source.sources,
        challenges,
        akita_challenges::FoldChallengeDistribution::BinaryRoot,
    )?
    else {
        return Ok(None);
    };
    drop(span);
    let span = tracing::info_span!("binary_root_response_digits").entered();
    let response_digits = crate::binary_root::BinaryRootRelation::response_digits(&response)?;
    drop(span);
    let span = tracing::info_span!("binary_root_evaluation_witness").entered();
    let evaluation = crate::binary_root::BinaryEvaluationWitness::build(
        source.manifest.descriptor(),
        &preparation.claim,
        &preparation.partials,
        &response,
        challenges,
        source.manifest.field_profile().modulus(),
    )?;
    drop(span);
    // P64 fixes K before committing the outgoing witness. Retain exactly the
    // residual used here for the later relation preparation; P128 keeps its
    // existing response-only cache and computes residuals after J is drawn.
    let residuals = preparation
        .plan
        .projection_bounds()
        .quotient()
        .map(|_| {
            crate::binary_root::BinaryRootRelation::remainders(
                source.manifest.descriptor(),
                &source.matrix,
                &response,
                &source.images,
                challenges,
                akita_challenges::FoldChallengeDistribution::BinaryRoot,
            )
        })
        .transpose()?;
    let quotient_digits = residuals
        .as_ref()
        .zip(preparation.plan.projection_bounds().quotient())
        .map(|(residuals, bounds)| {
            crate::binary_root::BinaryRootRelation::quotient_digits(residuals, bounds)
        })
        .transpose()?;
    *source
        .accepted
        .lock()
        .map_err(|_| AkitaError::Internal("binary accepted response cache poisoned".into()))? =
        Some(AcceptedResponse {
            challenges: challenges.clone(),
            response,
            residuals,
        });
    let span = tracing::info_span!("binary_root_repack").entered();
    let mut natural: Vec<&[i8]> = vec![
        &response_digits,
        &source.image_digits,
        &preparation.partials,
        &evaluation.parity_digits,
        &evaluation.value_digits,
    ];
    if preparation.plan.prime_segment().is_some() {
        natural.push(&preparation.prime_digits);
    }
    natural.extend(preparation.ordinary_segments.iter().map(Vec::as_slice));
    natural.extend(quotient_digits.as_deref());
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
    drop(span);
    let _span = tracing::info_span!("binary_root_import_witness").entered();
    super::binary_root_sessions::import_root_witness(
        backend,
        session,
        context,
        digits,
        destination_ring_dimension,
    )
    .map(Some)
}

/// Extract the authenticated owner's carriers in canonical low-column-bit order.
fn field_switch_source<F, E>(
    backend: &CpuBackend<F, E>,
    binding: &OperationBinding,
    context: &ProofContext,
    handle: &CommitmentHandle<F, E>,
    request: &akita_types::binary::BinaryFieldSwitchSourceRequest,
) -> Result<(Vec<F>, Box<dyn std::any::Any + Send + Sync>), AkitaError>
where
    F: Field + CanonicalEncoding + 'static,
    E: Field + 'static,
{
    let source = handle.committed.source.binary().ok_or_else(|| {
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
    let _span = tracing::info_span!("binary_field_switch_source").entered();
    let descriptor = request.manifest.descriptor();
    let owner = descriptor
        .polynomials()
        .iter()
        .position(|id| id == &request.polynomial)
        .ok_or_else(|| AkitaError::InvalidInput("unknown binary polynomial".into()))?;
    let index = descriptor.index_map();
    let owner_start =
        checked::product([owner, index.padded_carriers()]).ok_or(AkitaError::InvalidProof)?;
    let mut carriers = Vec::new();
    carriers
        .try_reserve_exact(index.padded_carriers())
        .map_err(|_| AkitaError::InvalidInput("carrier allocation failed".into()))?;
    carriers.resize(index.padded_carriers(), 0u128);
    let gather = |(first, chunk): (usize, &mut [u128])| {
        for (carrier, slot) in (first * FIELD_SWITCH_GATHER_CHUNK..).zip(chunk) {
            let address = checked::sum([owner_start, index.kernel_carrier_index(carrier)?])
                .ok_or(AkitaError::InvalidProof)?;
            *slot = *source
                .sources
                .get(address)
                .ok_or(AkitaError::InvalidProof)?;
        }
        Ok::<_, AkitaError>(())
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        carriers
            .par_chunks_mut(FIELD_SWITCH_GATHER_CHUNK)
            .enumerate()
            .try_for_each(gather)?;
    }
    #[cfg(not(feature = "parallel"))]
    carriers
        .chunks_mut(FIELD_SWITCH_GATHER_CHUNK)
        .enumerate()
        .try_for_each(gather)?;
    binding
        .scope_lease()
        .admit_commitment(context, handle.committed.commitment_id)?;
    Ok((Vec::new(), Box::new(carriers)))
}

/// Carriers gathered by one parallel task of the field-switch source.
const FIELD_SWITCH_GATHER_CHUNK: usize = 1 << 14;
