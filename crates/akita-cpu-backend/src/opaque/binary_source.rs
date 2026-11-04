//! Canonical CPU ownership and commitment for explicitly selected binary sources.

use super::owned::{CommitOutput, CommitmentHandle, CommittedSource, PrivateSource, SourceHandle};
use super::owned_kind::SourceGroupKind;
use super::*;
use akita_algebra::ring::trinomial::image::Q24Image;
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::checked;
use akita_types::binary_source::{
    binary_public_digit_commitment, binary_root_matrix, BinaryCommittedGroupManifest,
    BINARY_DIGIT_COMMITMENT_DEGREE, BINARY_SOURCE_Q,
};
use akita_types::{Commitment, CommittedGroup, RingVec};
use std::sync::Arc;

mod packed;
mod projection;
use packed::{fold_packed, image_hint, pack_logical_words, probe_packed};

const SOURCE_RING_DEGREE: usize = 648;
const MAX_BINARY_SOURCE_MATERIAL_BYTES: usize = 1 << 27;

/// Immutable private source and q24 image hint. Neither table is a public payload.
#[derive(Clone)]
pub(super) struct BinarySourceStorage {
    manifest: BinaryCommittedGroupManifest,
    carrier_profile: akita_params::GroupCommitPhaseParams,
    setup_seed: [u8; 32],
    sources: Arc<Vec<u128>>,
    matrix: Arc<Vec<Vec<[i128; SOURCE_RING_DEGREE]>>>,
    images: Arc<Vec<[Q24Image; SOURCE_RING_DEGREE]>>,
    image_digits: Arc<Vec<i8>>,
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
        let span = tracing::info_span!("binary_commit_pack_words").entered();
        let sources = pack_logical_words(manifest.descriptor(), &words)?;
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
    claim: Option<akita_types::binary_source::BinaryTerminalClaim>,
    prime_digits: Vec<i8>,
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
    if let Some(request) =
        request.downcast_ref::<akita_types::binary_source::BinaryProjectionRequest>()
    {
        return projection::prepare(backend, &binding, context, handle, request);
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
    if let Some(claim) = claim {
        claim.validate(manifest.descriptor())?;
    }
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
    )?
    .with_prime_planes(claim.is_some(), request.prime.planes().len())?;
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
    let span = tracing::info_span!("binary_root_row_partials").entered();
    let partials = claim
        .as_ref()
        .map(|claim| {
            crate::binary_root::evaluation::packed_row_partials(
                manifest.descriptor(),
                claim,
                &source.sources,
            )
        })
        .transpose()?
        .unwrap_or_default();
    drop(span);
    let span = tracing::info_span!("binary_root_prime_digits").entered();
    let prime_digits = crate::binary_root::prime::packed_prime_plane_digits::<F>(
        manifest.descriptor(),
        &request.prime,
        &source.sources,
    )?;
    drop(span);
    let span = tracing::info_span!("binary_root_d_payload").entered();
    let mut d_digits = partials.clone();
    d_digits
        .try_reserve_exact(prime_digits.len())
        .map_err(|_| AkitaError::InvalidProof)?;
    d_digits.extend_from_slice(&prime_digits);
    let payload = digit_commitment::<F>(
        &source.setup_seed,
        akita_params::binary_setup::SetupStreamLabel {
            role: akita_params::binary_setup::SetupStreamRole::WitnessCommitment,
            ..manifest.b_label()
        },
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
        akita_challenges::FoldChallengeFamily::BinaryRoot,
    )?
    else {
        return Ok(None);
    };
    drop(span);
    let span = tracing::info_span!("binary_root_response_digits").entered();
    let response_digits = crate::binary_root::BinaryRootRelation::response_digits(&response)?;
    drop(span);
    let span = tracing::info_span!("binary_root_evaluation_witness").entered();
    let evaluation = preparation
        .claim
        .as_ref()
        .map(|claim| {
            crate::binary_root::BinaryEvaluationWitness::build(
                source.manifest.descriptor(),
                claim,
                &preparation.partials,
                &response,
                challenges,
                akita_params::binary_setup::BINARY_SETUP_P,
            )
        })
        .transpose()?;
    drop(span);
    let span = tracing::info_span!("binary_root_repack").entered();
    let mut natural: Vec<&[i8]> = vec![&response_digits, &source.image_digits];
    if let Some(evaluation) = &evaluation {
        natural.extend([
            preparation.partials.as_slice(),
            evaluation.parity_digits.as_slice(),
            evaluation.value_digits.as_slice(),
        ]);
    }
    if preparation.plan.prime_segment().is_some() {
        natural.push(&preparation.prime_digits);
    }
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
    request: &akita_types::binary_source::BinaryFieldSwitchSourceRequest,
) -> Result<(Vec<F>, Box<dyn std::any::Any + Send + Sync>), AkitaError>
where
    F: Field + CanonicalEncoding + 'static,
    E: Field + 'static,
{
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
    let owner_start =
        checked::product([owner, index.padded_carriers()]).ok_or(AkitaError::InvalidProof)?;
    let mut carriers = Vec::new();
    carriers
        .try_reserve_exact(index.padded_carriers())
        .map_err(|_| AkitaError::InvalidInput("carrier allocation failed".into()))?;
    for carrier in 0..index.padded_carriers() {
        let address = checked::sum([owner_start, index.kernel_carrier_index(carrier)?])
            .ok_or(AkitaError::InvalidProof)?;
        carriers.push(
            *source
                .sources
                .get(address)
                .ok_or(AkitaError::InvalidProof)?,
        );
    }
    binding
        .scope_lease()
        .admit_commitment(context, handle.committed.commitment_id)?;
    Ok((Vec::new(), Box::new(carriers)))
}

/// Setup columns in one parallel piece of a public digit commitment.
const DIGIT_COMMITMENT_PIECE_COLUMNS: usize = 256;

/// Compute [`binary_public_digit_commitment`] as the field sum of disjoint
/// column-aligned pieces, which run in parallel under the `parallel` feature.
/// The sum is exact, so the payload equals the one-piece commitment.
fn digit_commitment<F: Field>(
    seed: &[u8; 32],
    label: akita_params::binary_setup::SetupStreamLabel,
    digits: &[i8],
) -> Result<Vec<F>, AkitaError> {
    let piece = checked::product([
        DIGIT_COMMITMENT_PIECE_COLUMNS,
        BINARY_DIGIT_COMMITMENT_DEGREE,
    ])
    .ok_or(AkitaError::InvalidProof)?;
    if digits.len() <= piece {
        return binary_public_digit_commitment(seed, label, 0, digits);
    }
    let commit = |(index, chunk): (usize, &[i8])| {
        let first = checked::product([index, DIGIT_COMMITMENT_PIECE_COLUMNS])
            .ok_or(AkitaError::InvalidProof)?;
        binary_public_digit_commitment::<F>(seed, label, first, chunk)
    };
    #[cfg(feature = "parallel")]
    let pieces = {
        use rayon::prelude::*;
        digits
            .par_chunks(piece)
            .enumerate()
            .map(commit)
            .collect::<Result<Vec<_>, _>>()?
    };
    #[cfg(not(feature = "parallel"))]
    let pieces = digits
        .chunks(piece)
        .enumerate()
        .map(commit)
        .collect::<Result<Vec<_>, _>>()?;
    let mut pieces = pieces.into_iter();
    let mut payload = pieces.next().ok_or(AkitaError::InvalidProof)?;
    for other in pieces {
        if other.len() != payload.len() {
            return Err(AkitaError::InvalidProof);
        }
        for (sum, value) in payload.iter_mut().zip(other) {
            *sum += value;
        }
    }
    Ok(payload)
}
