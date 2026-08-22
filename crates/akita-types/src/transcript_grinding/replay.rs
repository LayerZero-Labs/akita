use super::*;

#[cfg(feature = "logging-transcript")]
#[derive(Default)]
struct ChallengeAudit {
    pending_query: Option<GrindingSite>,
    query_observed: bool,
    fold_root_seen: bool,
    fold_range: Option<(usize, usize)>,
    invalid: bool,
}

#[cfg(feature = "logging-transcript")]
impl ChallengeAudit {
    fn begin_query(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        self.seal_query();
        if self.invalid {
            return Err(AkitaError::InvalidProof);
        }
        self.pending_query = Some(site);
        Ok(())
    }

    fn seal_query(&mut self) {
        if self.pending_query.is_some() && !self.query_observed {
            self.invalid = true;
        }
        self.pending_query = None;
        self.query_observed = false;
    }

    fn observe_query(&mut self, label: &[u8], sparse_root: bool) -> Option<GrindingSite> {
        if let Some(site) = self.pending_query {
            let normalized = akita_transcript::ext_limb_base_label(label).unwrap_or(label);
            self.invalid |= site.proof_of_work_label() != Some(normalized);
            self.query_observed = true;
            return Some(site);
        }
        if sparse_root && !self.fold_root_seen && self.fold_range.is_none() {
            self.fold_root_seen = true;
        } else {
            self.invalid = true;
        }
        None
    }

    fn observe_fold_range(&mut self, group: usize, coordinates: usize) {
        if !self.fold_root_seen || self.fold_range.replace((group, coordinates)).is_some() {
            self.invalid = true;
        }
    }

    fn consume_fold(&mut self, group: u32, coordinates: usize) {
        let expected_group = usize::try_from(group).ok();
        if !self.fold_root_seen || self.fold_range != expected_group.map(|g| (g, coordinates)) {
            self.invalid = true;
        }
        self.fold_root_seen = false;
        self.fold_range = None;
    }

    fn is_finished(&self) -> bool {
        self.pending_query.is_none()
            && !self.query_observed
            && !self.fold_root_seen
            && self.fold_range.is_none()
            && !self.invalid
    }
}

#[derive(Clone, Copy)]
struct GrindingPlanEntry {
    site: GrindingSite,
    kind: GrindingQueryKind,
    grind_bits: u8,
    nonce_bits: u8,
}

struct GrindingPlanCursor<'a> {
    plan: &'a GrindingPlan,
    run_index: usize,
    run_offset: u64,
}

impl<'a> GrindingPlanCursor<'a> {
    const fn new(plan: &'a GrindingPlan) -> Self {
        Self {
            plan,
            run_index: 0,
            run_offset: 0,
        }
    }

    fn next(&mut self) -> Option<GrindingPlanEntry> {
        let run = *self.plan.runs.get(self.run_index)?;
        let entry = GrindingPlanEntry {
            site: run.site,
            kind: run.kind(),
            grind_bits: run.grind_bits,
            nonce_bits: run.nonce_bits,
        };
        self.run_offset += 1;
        if self.run_offset == run.multiplicity {
            self.run_index += 1;
            self.run_offset = 0;
        }
        Some(entry)
    }

    fn consume_run(
        &mut self,
        site: GrindingSite,
        kind: GrindingQueryKind,
        multiplicity: usize,
    ) -> Result<(), AkitaError> {
        let run = self
            .plan
            .runs
            .get(self.run_index)
            .ok_or(AkitaError::InvalidProof)?;
        if self.run_offset != 0
            || run.site != site
            || run.kind() != kind
            || run.grind_bits != 0
            || run.nonce_bits != 0
            || usize::try_from(run.multiplicity).ok() != Some(multiplicity)
        {
            return Err(AkitaError::InvalidProof);
        }
        self.run_index = self
            .run_index
            .checked_add(1)
            .ok_or(AkitaError::InvalidProof)?;
        Ok(())
    }

    fn is_finished(&self) -> bool {
        self.run_index == self.plan.runs.len() && self.run_offset == 0
    }
}

struct NonceStreamCursor<'a> {
    plan: GrindingPlanCursor<'a>,
    bit_offset: usize,
}

impl<'a> NonceStreamCursor<'a> {
    const fn new(plan: &'a GrindingPlan) -> Self {
        Self {
            plan: GrindingPlanCursor::new(plan),
            bit_offset: 0,
        }
    }

    fn next(
        &mut self,
        site: GrindingSite,
        expected_kind: Option<GrindingQueryKind>,
    ) -> Result<GrindingPlanEntry, AkitaError> {
        let entry = self.plan.next().ok_or(AkitaError::InvalidProof)?;
        if entry.site != site || expected_kind.is_some_and(|kind| entry.kind != kind) {
            return Err(AkitaError::InvalidInput(format!(
                "grinding replay expected {site:?}/{expected_kind:?}, plan has {:?}/{:?}",
                entry.site, entry.kind
            )));
        }
        Ok(entry)
    }

    fn consume_zero_run(
        &mut self,
        site: GrindingSite,
        kind: GrindingQueryKind,
        multiplicity: usize,
    ) -> Result<(), AkitaError> {
        self.plan.consume_run(site, kind, multiplicity)
    }

    fn advance_bits(&mut self, width: u8) -> Result<(), AkitaError> {
        self.bit_offset = self
            .bit_offset
            .checked_add(usize::from(width))
            .ok_or(AkitaError::InvalidProof)?;
        Ok(())
    }

    fn is_finished(&self, expected_bits: usize) -> bool {
        self.plan.is_finished() && self.bit_offset == expected_bits
    }
}

/// Headerless, low-bit-first packed values for the public grinding plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptNonceStream {
    bytes: Vec<u8>,
    bit_len: usize,
}

impl TranscriptNonceStream {
    /// Construct a stream from its exact raw bytes and public bit width.
    pub fn from_bytes(bytes: Vec<u8>, bit_len: usize) -> Result<Self, SerializationError> {
        let expected_bytes = nonce_byte_len(bit_len)?;
        if bytes.len() != expected_bytes {
            return Err(SerializationError::InvalidData(
                "nonce stream byte length does not match its bit width".into(),
            ));
        }
        if let (Some(&last), Some(used_bits)) = (bytes.last(), bit_len.checked_rem(8)) {
            if used_bits != 0 && last >> used_bits != 0 {
                return Err(SerializationError::InvalidData(
                    "nonce stream has nonzero high padding bits".into(),
                ));
            }
        }
        Ok(Self { bytes, bit_len })
    }

    /// Exact number of meaningful packed bits.
    #[must_use]
    pub const fn bit_len(&self) -> usize {
        self.bit_len
    }

    /// Exact raw proof bytes, without a length prefix.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Start checked sequential replay against `plan`.
    pub fn reader<'a>(
        &'a self,
        plan: &'a GrindingPlan,
    ) -> Result<TranscriptNonceReader<'a>, AkitaError> {
        if self.bit_len != plan.total_nonce_bits {
            return Err(AkitaError::InvalidProof);
        }
        Ok(TranscriptNonceReader {
            stream: self,
            cursor: NonceStreamCursor::new(plan),
        })
    }
}

/// Checked sequential writer for one public grinding plan.
pub struct TranscriptNonceWriter<'a> {
    bytes: Vec<u8>,
    cursor: NonceStreamCursor<'a>,
}

impl<'a> TranscriptNonceWriter<'a> {
    /// Allocate the exact packed byte count prescribed by `plan`.
    pub fn new(plan: &'a GrindingPlan) -> Result<Self, AkitaError> {
        let byte_len = nonce_byte_len(plan.total_nonce_bits)
            .map_err(|_| AkitaError::InvalidSetup("nonce stream byte width overflow".into()))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_len)
            .map_err(|_| AkitaError::InvalidSetup("nonce stream allocation failed".into()))?;
        bytes.resize(byte_len, 0);
        Ok(Self {
            bytes,
            cursor: NonceStreamCursor::new(plan),
        })
    }

    /// Write the next expected logical entry and reject a site or width mismatch.
    pub fn write(&mut self, site: GrindingSite, value: u32) -> Result<(), AkitaError> {
        let entry = self.cursor.next(site, None)?;
        if !value_fits(value, entry.nonce_bits) {
            return Err(AkitaError::InvalidProof);
        }
        self.write_entry(entry, value)
    }

    /// Search and commit the next scheduled proof-of-work transition.
    pub fn grind<F, T>(&mut self, transcript: &mut T, site: GrindingSite) -> Result<(), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F> + TranscriptChallengePreview,
    {
        let entry = self.next_expected(site, GrindingQueryKind::ProofOfWork)?;
        let nonce = search_grinding_nonce(transcript, entry.grind_bits, entry.nonce_bits)
            .ok_or_else(|| AkitaError::InvalidInput("transcript grinding exhausted".into()))?;
        let site_label = site.proof_of_work_label().ok_or(AkitaError::InvalidProof)?;
        let predicate =
            transcript.grinding_predicate(site_label, entry.grind_bits, entry.nonce_bits, nonce);
        match (NonZeroU8::new(entry.grind_bits), predicate) {
            (None, None) => {}
            (Some(bits), Some(predicate)) if grinding_predicate_accepts(&predicate, bits) => {}
            _ => {
                return Err(AkitaError::InvalidInput(
                    "transcript grinding preview/live mismatch".into(),
                ));
            }
        }
        self.write_entry(entry, nonce)
    }

    /// Commit the next scheduled fold-response search nonce.
    pub fn write_fold_response(
        &mut self,
        site: GrindingSite,
        nonce: u32,
    ) -> Result<(), AkitaError> {
        let entry = self.next_expected(site, GrindingQueryKind::FoldResponse)?;
        self.write_entry(entry, nonce)
    }

    /// Consume the zero-width root and indexed-coordinate records for one fold group.
    pub fn record_fold_challenges(
        &mut self,
        level: u32,
        group: u32,
        coordinate_count: usize,
    ) -> Result<(), AkitaError> {
        self.cursor.consume_zero_run(
            GrindingSite::FoldChallengeRoot { level, group },
            GrindingQueryKind::FoldChallengeRoot,
            1,
        )?;
        self.cursor.consume_zero_run(
            GrindingSite::FoldChallengeCoordinates { level, group },
            GrindingQueryKind::FoldChallengeCoordinates,
            coordinate_count,
        )
    }

    /// Finish the stream, requiring exact plan and bit-cursor exhaustion.
    pub fn finish(self) -> Result<TranscriptNonceStream, AkitaError> {
        let bit_len = self.cursor.bit_offset;
        if !self
            .cursor
            .is_finished(self.cursor.plan.plan.total_nonce_bits)
        {
            return Err(AkitaError::InvalidProof);
        }
        Ok(TranscriptNonceStream {
            bytes: self.bytes,
            bit_len,
        })
    }

    fn next_expected(
        &mut self,
        site: GrindingSite,
        kind: GrindingQueryKind,
    ) -> Result<GrindingPlanEntry, AkitaError> {
        self.cursor.next(site, Some(kind))
    }

    fn write_entry(&mut self, entry: GrindingPlanEntry, value: u32) -> Result<(), AkitaError> {
        if !value_fits(value, entry.nonce_bits) {
            return Err(AkitaError::InvalidProof);
        }
        write_bits(
            &mut self.bytes,
            self.cursor.bit_offset,
            entry.nonce_bits,
            value,
        );
        self.cursor.advance_bits(entry.nonce_bits)
    }
}

/// Checked sequential reader for one public grinding plan.
pub struct TranscriptNonceReader<'a> {
    stream: &'a TranscriptNonceStream,
    cursor: NonceStreamCursor<'a>,
}

impl TranscriptNonceReader<'_> {
    /// Read the next expected logical entry and reject a site mismatch.
    pub fn read(&mut self, site: GrindingSite) -> Result<u32, AkitaError> {
        let entry = self.cursor.next(site, None)?;
        self.read_entry(entry)
    }

    /// Read and verify the next scheduled proof-of-work transition.
    pub fn grind<F, T>(&mut self, transcript: &mut T, site: GrindingSite) -> Result<(), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
    {
        let entry = self.next_expected(site, GrindingQueryKind::ProofOfWork)?;
        let nonce = self.read_entry(entry)?;
        let site_label = site.proof_of_work_label().ok_or(AkitaError::InvalidProof)?;
        let predicate =
            transcript.grinding_predicate(site_label, entry.grind_bits, entry.nonce_bits, nonce);
        match (NonZeroU8::new(entry.grind_bits), predicate) {
            (None, None) => Ok(()),
            (Some(bits), Some(predicate)) if grinding_predicate_accepts(&predicate, bits) => Ok(()),
            _ => Err(AkitaError::InvalidProof),
        }
    }

    /// Read the next scheduled fold-response search nonce.
    pub fn read_fold_response(&mut self, site: GrindingSite) -> Result<u32, AkitaError> {
        let entry = self.next_expected(site, GrindingQueryKind::FoldResponse)?;
        self.read_entry(entry)
    }

    /// Consume the zero-width root and indexed-coordinate records for one fold group.
    pub fn record_fold_challenges(
        &mut self,
        level: u32,
        group: u32,
        coordinate_count: usize,
    ) -> Result<(), AkitaError> {
        self.cursor.consume_zero_run(
            GrindingSite::FoldChallengeRoot { level, group },
            GrindingQueryKind::FoldChallengeRoot,
            1,
        )?;
        self.cursor.consume_zero_run(
            GrindingSite::FoldChallengeCoordinates { level, group },
            GrindingQueryKind::FoldChallengeCoordinates,
            coordinate_count,
        )
    }

    /// Finish replay, requiring exact plan and bit-cursor exhaustion.
    pub fn finish(self) -> Result<(), AkitaError> {
        if !self.cursor.is_finished(self.stream.bit_len) {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }

    fn next_expected(
        &mut self,
        site: GrindingSite,
        kind: GrindingQueryKind,
    ) -> Result<GrindingPlanEntry, AkitaError> {
        self.cursor.next(site, Some(kind))
    }

    fn read_entry(&mut self, entry: GrindingPlanEntry) -> Result<u32, AkitaError> {
        let value = read_bits(
            self.stream.as_bytes(),
            self.cursor.bit_offset,
            entry.nonce_bits,
        );
        self.cursor.advance_bits(entry.nonce_bits)?;
        Ok(value)
    }
}

/// Prover-side transcript operations backed by the canonical nonce writer.
pub trait ProverTranscriptGrinding<F>: Transcript<F> + TranscriptChallengePreview
where
    F: FieldCore + CanonicalField,
{
    /// Apply the scheduled PoW transition immediately before its challenge draw.
    fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError>;

    /// Commit the accepted fold-response nonce before replaying its indexed challenges.
    fn commit_fold_response(&mut self, site: GrindingSite, nonce: u32) -> Result<(), AkitaError>;

    /// Record one group-local root draw and all indexed coordinate challenges.
    fn record_fold_challenges(
        &mut self,
        level: u32,
        group: u32,
        coordinate_count: usize,
    ) -> Result<(), AkitaError>;
}

/// Verifier-side transcript operations backed by the canonical nonce reader.
pub trait VerifierTranscriptGrinding<F>: Transcript<F>
where
    F: FieldCore + CanonicalField,
{
    /// Verify the scheduled PoW transition immediately before its challenge draw.
    fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError>;

    /// Read the next scheduled fold-response nonce before replaying its challenges.
    fn read_fold_response(&mut self, site: GrindingSite) -> Result<u32, AkitaError>;

    /// Record one group-local root draw and all indexed coordinate challenges.
    fn record_fold_challenges(
        &mut self,
        level: u32,
        group: u32,
        coordinate_count: usize,
    ) -> Result<(), AkitaError>;
}

/// Borrowed prover transcript with exclusive ownership of one nonce-stream cursor.
pub struct ProverGrindingTranscript<'transcript, 'plan, T> {
    transcript: &'transcript mut T,
    writer: TranscriptNonceWriter<'plan>,
    #[cfg(feature = "logging-transcript")]
    audit: ChallengeAudit,
}

impl<'transcript, 'plan, T> ProverGrindingTranscript<'transcript, 'plan, T> {
    /// Attach the exact public plan to an already-bound prover transcript.
    pub fn new(
        transcript: &'transcript mut T,
        plan: &'plan GrindingPlan,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            transcript,
            writer: TranscriptNonceWriter::new(plan)?,
            #[cfg(feature = "logging-transcript")]
            audit: ChallengeAudit::default(),
        })
    }

    /// Require exact cursor exhaustion and return the packed proof stream.
    pub fn finish(self) -> Result<TranscriptNonceStream, AkitaError> {
        #[cfg(feature = "logging-transcript")]
        {
            let mut audit = self.audit;
            audit.seal_query();
            if !audit.is_finished() {
                return Err(AkitaError::InvalidProof);
            }
        }
        self.writer.finish()
    }
}

/// Borrowed verifier transcript with exclusive ownership of one nonce-stream cursor.
pub struct VerifierGrindingTranscript<'transcript, 'proof, T> {
    transcript: &'transcript mut T,
    reader: TranscriptNonceReader<'proof>,
    #[cfg(feature = "logging-transcript")]
    audit: ChallengeAudit,
}

impl<'transcript, 'proof, T> VerifierGrindingTranscript<'transcript, 'proof, T> {
    /// Attach the exact public plan and proof stream to an already-bound verifier transcript.
    pub fn new(
        transcript: &'transcript mut T,
        stream: &'proof TranscriptNonceStream,
        plan: &'proof GrindingPlan,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            transcript,
            reader: stream.reader(plan)?,
            #[cfg(feature = "logging-transcript")]
            audit: ChallengeAudit::default(),
        })
    }

    /// Require exact plan and proof-bit exhaustion.
    pub fn finish(self) -> Result<(), AkitaError> {
        #[cfg(feature = "logging-transcript")]
        {
            let mut audit = self.audit;
            audit.seal_query();
            if !audit.is_finished() {
                return Err(AkitaError::InvalidProof);
            }
        }
        self.reader.finish()
    }
}

macro_rules! impl_transcript_forwarding {
    ($adapter:ident) => {
        impl<F, T> Transcript<F> for $adapter<'_, '_, T>
        where
            F: FieldCore + CanonicalField,
            T: Transcript<F>,
        {
            fn bind_instance_bytes(&mut self, instance_bytes: &[u8]) {
                #[cfg(feature = "logging-transcript")]
                self.audit.seal_query();
                self.transcript.bind_instance_bytes(instance_bytes);
            }

            fn record_wire_serde<S: AkitaSerialize>(&mut self, label: &[u8], value: &S) {
                #[cfg(feature = "logging-transcript")]
                self.audit.seal_query();
                self.transcript.record_wire_serde(label, value);
            }

            fn record_wire_bytes(&mut self, label: &[u8], bytes: &[u8]) {
                #[cfg(feature = "logging-transcript")]
                self.audit.seal_query();
                self.transcript.record_wire_bytes(label, bytes);
            }

            #[cfg(feature = "logging-transcript")]
            fn record_grinding_plan_query(&mut self, site: &[u8], multiplicity: u64) {
                self.transcript
                    .record_grinding_plan_query(site, multiplicity);
            }

            #[cfg(feature = "logging-transcript")]
            fn record_grinding_actual_query(&mut self, site: &[u8], label: &[u8]) {
                self.transcript.record_grinding_actual_query(site, label);
            }

            #[cfg(feature = "logging-transcript")]
            fn record_fold_challenge_range(&mut self, group_index: usize, coordinate_count: usize) {
                self.audit.seal_query();
                self.audit.observe_fold_range(group_index, coordinate_count);
                self.transcript
                    .record_fold_challenge_range(group_index, coordinate_count);
            }

            fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
                #[cfg(feature = "logging-transcript")]
                self.audit.seal_query();
                self.transcript.append_bytes(label, bytes);
            }

            fn append_field(&mut self, label: &[u8], value: &F) {
                #[cfg(feature = "logging-transcript")]
                self.audit.seal_query();
                self.transcript.append_field(label, value);
            }

            fn append_serde<S: AkitaSerialize>(&mut self, label: &[u8], value: &S) {
                #[cfg(feature = "logging-transcript")]
                self.audit.seal_query();
                self.transcript.append_serde(label, value);
            }

            fn challenge_scalar(&mut self, label: &[u8]) -> F {
                #[cfg(feature = "logging-transcript")]
                if let Some(site) = self.audit.observe_query(label, false) {
                    self.transcript
                        .record_grinding_actual_query(&site.canonical_bytes(), label);
                }
                self.transcript.challenge_scalar(label)
            }

            fn challenge_bytes(&mut self, label: &[u8], len: usize) -> Vec<u8> {
                #[cfg(feature = "logging-transcript")]
                if let Some(site) = self.audit.observe_query(label, false) {
                    self.transcript
                        .record_grinding_actual_query(&site.canonical_bytes(), label);
                }
                self.transcript.challenge_bytes(label, len)
            }

            fn challenge_block(
                &mut self,
                label: &[u8],
            ) -> [u8; akita_transcript::TRANSCRIPT_CHALLENGE_BLOCK_LEN] {
                #[cfg(feature = "logging-transcript")]
                if let Some(site) = self.audit.observe_query(
                    label,
                    label == akita_transcript::labels::CHALLENGE_SPARSE_CHALLENGE,
                ) {
                    self.transcript
                        .record_grinding_actual_query(&site.canonical_bytes(), label);
                }
                self.transcript.challenge_block(label)
            }

            fn grinding_predicate(
                &mut self,
                site_label: &[u8],
                grind_bits: u8,
                nonce_bits: u8,
                nonce: u32,
            ) -> Option<[u8; akita_transcript::GRINDING_PREDICATE_LEN]> {
                self.transcript
                    .grinding_predicate(site_label, grind_bits, nonce_bits, nonce)
            }
        }
    };
}

impl_transcript_forwarding!(ProverGrindingTranscript);
impl_transcript_forwarding!(VerifierGrindingTranscript);

impl<T> TranscriptChallengePreview for ProverGrindingTranscript<'_, '_, T>
where
    T: TranscriptChallengePreview,
{
    fn preview_challenge_block(
        &self,
        absorb_payloads: &[&[u8]],
    ) -> [u8; akita_transcript::TRANSCRIPT_CHALLENGE_BLOCK_LEN] {
        self.transcript.preview_challenge_block(absorb_payloads)
    }
}

impl<F, T> ProverTranscriptGrinding<F> for ProverGrindingTranscript<'_, '_, T>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F> + TranscriptChallengePreview,
{
    fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        self.writer.grind::<F, T>(self.transcript, site)?;
        #[cfg(feature = "logging-transcript")]
        {
            self.audit.begin_query(site)?;
            self.transcript
                .record_grinding_plan_query(&site.canonical_bytes(), 1);
        }
        Ok(())
    }

    fn commit_fold_response(&mut self, site: GrindingSite, nonce: u32) -> Result<(), AkitaError> {
        self.writer.write_fold_response(site, nonce)?;
        #[cfg(feature = "logging-transcript")]
        {
            self.audit.seal_query();
            if self.audit.invalid {
                return Err(AkitaError::InvalidProof);
            }
            self.transcript
                .record_grinding_plan_query(&site.canonical_bytes(), 1);
        }
        Ok(())
    }

    fn record_fold_challenges(
        &mut self,
        level: u32,
        group: u32,
        coordinate_count: usize,
    ) -> Result<(), AkitaError> {
        self.writer
            .record_fold_challenges(level, group, coordinate_count)?;
        #[cfg(feature = "logging-transcript")]
        {
            self.audit.seal_query();
            if self.audit.invalid {
                return Err(AkitaError::InvalidProof);
            }
            self.audit.consume_fold(group, coordinate_count);
            self.transcript.record_grinding_plan_query(
                &GrindingSite::FoldChallengeRoot { level, group }.canonical_bytes(),
                1,
            );
            self.transcript.record_grinding_plan_query(
                &GrindingSite::FoldChallengeCoordinates { level, group }.canonical_bytes(),
                u64::try_from(coordinate_count).map_err(|_| AkitaError::InvalidProof)?,
            );
        }
        Ok(())
    }
}

impl<F, T> VerifierTranscriptGrinding<F> for VerifierGrindingTranscript<'_, '_, T>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        self.reader.grind::<F, T>(self.transcript, site)?;
        #[cfg(feature = "logging-transcript")]
        {
            self.audit.begin_query(site)?;
            self.transcript
                .record_grinding_plan_query(&site.canonical_bytes(), 1);
        }
        Ok(())
    }

    fn read_fold_response(&mut self, site: GrindingSite) -> Result<u32, AkitaError> {
        let nonce = self.reader.read_fold_response(site)?;
        #[cfg(feature = "logging-transcript")]
        {
            self.audit.seal_query();
            if self.audit.invalid {
                return Err(AkitaError::InvalidProof);
            }
            self.transcript
                .record_grinding_plan_query(&site.canonical_bytes(), 1);
        }
        Ok(nonce)
    }

    fn record_fold_challenges(
        &mut self,
        level: u32,
        group: u32,
        coordinate_count: usize,
    ) -> Result<(), AkitaError> {
        self.reader
            .record_fold_challenges(level, group, coordinate_count)?;
        #[cfg(feature = "logging-transcript")]
        {
            self.audit.seal_query();
            if self.audit.invalid {
                return Err(AkitaError::InvalidProof);
            }
            self.audit.consume_fold(group, coordinate_count);
            self.transcript.record_grinding_plan_query(
                &GrindingSite::FoldChallengeRoot { level, group }.canonical_bytes(),
                1,
            );
            self.transcript.record_grinding_plan_query(
                &GrindingSite::FoldChallengeCoordinates { level, group }.canonical_bytes(),
                u64::try_from(coordinate_count).map_err(|_| AkitaError::InvalidProof)?,
            );
        }
        Ok(())
    }
}

fn nonce_byte_len(bit_len: usize) -> Result<usize, SerializationError> {
    akita_error::checked::div_ceil(bit_len, 8)
        .ok_or_else(|| SerializationError::InvalidData("invalid nonce stream byte width".into()))
}

fn value_fits(value: u32, width: u8) -> bool {
    width == u32::BITS as u8 || value < (1u32 << width)
}

fn write_bits(bytes: &mut [u8], offset: usize, width: u8, value: u32) {
    for bit in 0..usize::from(width) {
        if (value >> bit) & 1 == 1 {
            let stream_bit = offset + bit;
            bytes[stream_bit / 8] |= 1 << (stream_bit % 8);
        }
    }
}

fn read_bits(bytes: &[u8], offset: usize, width: u8) -> u32 {
    let mut value = u32::default();
    for bit in 0..usize::from(width) {
        let stream_bit = offset + bit;
        value |= u32::from((bytes[stream_bit / 8] >> (stream_bit % 8)) & 1) << bit;
    }
    value
}
