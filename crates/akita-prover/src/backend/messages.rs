use jolt_field::Field;

type RecursiveWitnessBuildParts<F, E, BuildHandle> = (
    Vec<crate::backend::PreparedRelationGroupPublic<F, E>>,
    akita_types::RingVec<F>,
    usize,
    BuildHandle,
);

/// Aggregate fold diagnostics that cross the backend boundary without exposing
/// private response coefficients.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FoldProbeDiagnostics {
    observed_l2_sq: Option<u128>,
    #[cfg(feature = "response-model-diagnostics")]
    source_l2_sq: Option<u128>,
}

impl FoldProbeDiagnostics {
    pub const fn new(observed_l2_sq: Option<u128>) -> Self {
        Self {
            observed_l2_sq,
            #[cfg(feature = "response-model-diagnostics")]
            source_l2_sq: None,
        }
    }

    pub const fn observed_l2_sq(self) -> Option<u128> {
        self.observed_l2_sq
    }

    #[cfg(feature = "response-model-diagnostics")]
    pub const fn with_source_l2_sq(mut self, source_l2_sq: Option<u128>) -> Self {
        self.source_l2_sq = source_l2_sq;
        self
    }

    #[cfg(feature = "response-model-diagnostics")]
    pub const fn source_l2_sq(self) -> Option<u128> {
        self.source_l2_sq
    }
}

/// Result of probing a private fold response.
pub enum FoldProbeOutcome<FoldHandle> {
    /// Candidate response failed its public admission bounds.
    Rejected,
    /// Candidate response was accepted and retained by the consumer.
    Accepted {
        /// Linear handle for the accepted private response.
        fold_handle: FoldHandle,
        /// Coefficient-free aggregates used by response-model calibration.
        diagnostics: FoldProbeDiagnostics,
    },
}

/// Public scalar openings paired with a private prepared-opening handle.
pub struct PreparedGroupOpening<E: Field, OpeningHandle> {
    scalar_openings: Vec<E>,
    opening_handle: OpeningHandle,
}

impl<E: Field, OpeningHandle> PreparedGroupOpening<E, OpeningHandle> {
    pub fn new(scalar_openings: Vec<E>, opening_handle: OpeningHandle) -> Self {
        Self {
            scalar_openings,
            opening_handle,
        }
    }

    /// Public scalar opening messages.
    pub fn scalar_openings(&self) -> &[E] {
        &self.scalar_openings
    }

    /// Consume the message and return its private handle.
    pub fn into_opening_handle(self) -> OpeningHandle {
        self.opening_handle
    }

    pub fn into_parts(self) -> (Vec<E>, OpeningHandle) {
        (self.scalar_openings, self.opening_handle)
    }
}

/// Public artifacts emitted before transcript-owned fold grinding.
pub struct RecursiveWitnessBuildStart<F: Field, E: Field, BuildHandle> {
    public_groups: Vec<crate::backend::PreparedRelationGroupPublic<F, E>>,
    opening_payload: akita_types::RingVec<F>,
    opening_payload_ring_dimension: usize,
    build_handle: BuildHandle,
}

impl<F: Field, E: Field, BuildHandle> RecursiveWitnessBuildStart<F, E, BuildHandle> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        public_groups: Vec<crate::backend::PreparedRelationGroupPublic<F, E>>,
        opening_payload: akita_types::RingVec<F>,
        opening_payload_ring_dimension: usize,
        build_handle: BuildHandle,
    ) -> Self {
        Self {
            public_groups,
            opening_payload,
            opening_payload_ring_dimension,
            build_handle,
        }
    }

    /// Public relation-group descriptions.
    pub fn public_groups(&self) -> &[crate::backend::PreparedRelationGroupPublic<F, E>] {
        &self.public_groups
    }

    /// Public opening payload absorbed before challenges are drawn.
    pub fn opening_payload(&self) -> &akita_types::RingVec<F> {
        &self.opening_payload
    }

    /// Ring dimension used to encode the opening payload.
    pub const fn opening_payload_ring_dimension(&self) -> usize {
        self.opening_payload_ring_dimension
    }

    pub fn into_parts(self) -> RecursiveWitnessBuildParts<F, E, BuildHandle> {
        (
            self.public_groups,
            self.opening_payload,
            self.opening_payload_ring_dimension,
            self.build_handle,
        )
    }
}

/// One accepted fold handle and the public challenges that produced it.
pub struct RecursiveWitnessFoldInput<FoldHandle> {
    fold_handle: FoldHandle,
    challenges: akita_types::GroupFoldChallenges,
}

impl<FoldHandle> RecursiveWitnessFoldInput<FoldHandle> {
    pub fn new(fold_handle: FoldHandle, challenges: akita_types::GroupFoldChallenges) -> Self {
        Self {
            fold_handle,
            challenges,
        }
    }

    /// Consume into the private fold handle and public challenges.
    pub fn into_fold_handle_and_challenges(self) -> (FoldHandle, akita_types::GroupFoldChallenges) {
        (self.fold_handle, self.challenges)
    }

    pub const fn challenges(&self) -> &akita_types::GroupFoldChallenges {
        &self.challenges
    }

    pub const fn fold_handle(&self) -> &FoldHandle {
        &self.fold_handle
    }
}

/// Public commitment and replacement private handles after commitment.
pub enum NextWitnessBindingMessage<F: Field> {
    OuterPayload(akita_types::RingVec<F>),
    TerminalInnerState(TerminalTFieldsMessage),
}

/// Replacement handles and the message scheduled at the commitment transition.
pub struct WitnessCommitmentOutput<F: Field, WitnessHandle, MaterialHandle> {
    binding: NextWitnessBindingMessage<F>,
    witness_handle: WitnessHandle,
    commitment_material_handle: MaterialHandle,
}

impl<F: Field, WitnessHandle, MaterialHandle>
    WitnessCommitmentOutput<F, WitnessHandle, MaterialHandle>
{
    /// Construct a commitment lifecycle output.
    pub fn new(
        binding: NextWitnessBindingMessage<F>,
        witness_handle: WitnessHandle,
        commitment_material_handle: MaterialHandle,
    ) -> Self {
        Self {
            binding,
            witness_handle,
            commitment_material_handle,
        }
    }

    /// Scheduled outgoing binding; terminal state is released before challenges.
    pub fn binding(&self) -> &NextWitnessBindingMessage<F> {
        &self.binding
    }

    /// Consume into the public commitment and replacement private handles.
    pub fn into_commitment_parts(
        self,
    ) -> (NextWitnessBindingMessage<F>, WitnessHandle, MaterialHandle) {
        (
            self.binding,
            self.witness_handle,
            self.commitment_material_handle,
        )
    }
}

/// Consumer-owned relation handle paired with public geometry.
pub struct PreparedRelationHandle<RelationHandle> {
    relation_handle: RelationHandle,
    metadata: super::RelationWitnessMetadata,
}

impl<RelationHandle> PreparedRelationHandle<RelationHandle> {
    pub const fn new(
        relation_handle: RelationHandle,
        metadata: super::RelationWitnessMetadata,
    ) -> Self {
        Self {
            relation_handle,
            metadata,
        }
    }

    /// Public relation-witness geometry.
    pub const fn metadata(&self) -> super::RelationWitnessMetadata {
        self.metadata
    }

    /// Consume into the private relation handle.
    pub fn into_relation_handle(self) -> RelationHandle {
        self.relation_handle
    }

    pub fn into_parts(self) -> (RelationHandle, super::RelationWitnessMetadata) {
        (self.relation_handle, self.metadata)
    }
}

/// Canonical scheduled terminal commitment message.
pub struct TerminalTFieldsMessage {
    bytes: Vec<u8>,
}
impl TerminalTFieldsMessage {
    pub fn from_row<F>(row: &akita_types::RingVec<F>) -> Result<Self, akita_error::AkitaError>
    where
        F: Field + jolt_field::CanonicalEncoding + akita_serialization::AkitaSerialize,
    {
        Ok(Self {
            bytes: akita_types::raw_field_segment_bytes(row)?,
        })
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}
