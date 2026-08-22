//! Public transcript-grinding policy and canonical replay plan.

use crate::instance_descriptor::digest_descriptor_bytes;
use crate::OpeningMethod;
use akita_error::AkitaError;
use akita_serialization::SerializationError;
pub use akita_transcript::{
    GRINDING_LITTLE_ENDIAN_BIT_ORDER, GRINDING_NONCE_SLACK_BITS, GRINDING_PREDICATE_BYTES,
    MAX_GRINDING_BITS,
};

/// Target work factor for every grinding-priced Fiat-Shamir query.
pub const TRANSCRIPT_SECURITY_BITS: u16 = 128;
/// Packed width of the existing fold-response search nonce.
pub const FOLD_RESPONSE_NONCE_BITS: u8 = 12;
/// Exclusive upper bound for the existing fold-response search.
pub const FOLD_RESPONSE_ATTEMPTS: u32 = 1 << FOLD_RESPONSE_NONCE_BITS;
/// Transcript-grinding binding encoding revision.
pub const GRINDING_ENCODING_VERSION: u16 = 1;
/// Query catalog and loss-policy revision.
pub const GRINDING_QUERY_POLICY_REVISION: u16 = 1;
/// Indexed fold-coordinate oracle revision.
pub const FOLD_COORDINATE_ORACLE_REVISION: u16 = 1;

const GRINDING_PLAN_DOMAIN: &[u8] = b"akita/grinding-plan/v1";
const GRINDING_POLICY_BYTES: usize = 17;

/// Fixed protocol policy shared by plan hashing and descriptor binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GrindingPolicy {
    pub encoding_version: u16,
    pub target_security_bits: u16,
    pub proof_of_work_slack_bits: u8,
    pub maximum_proof_of_work_bits: u8,
    pub predicate_bytes: u8,
    pub predicate_bit_order_tag: u8,
    pub fold_response_attempt_bits: u8,
    pub fold_response_attempts: u32,
    pub query_policy_revision: u16,
    pub fold_coordinate_oracle_revision: u16,
}

impl GrindingPolicy {
    /// The only policy accepted by this protocol revision.
    pub const ACTIVE: Self = Self {
        encoding_version: GRINDING_ENCODING_VERSION,
        target_security_bits: TRANSCRIPT_SECURITY_BITS,
        proof_of_work_slack_bits: GRINDING_NONCE_SLACK_BITS,
        maximum_proof_of_work_bits: MAX_GRINDING_BITS,
        predicate_bytes: GRINDING_PREDICATE_BYTES,
        predicate_bit_order_tag: GRINDING_LITTLE_ENDIAN_BIT_ORDER,
        fold_response_attempt_bits: FOLD_RESPONSE_NONCE_BITS,
        fold_response_attempts: FOLD_RESPONSE_ATTEMPTS,
        query_policy_revision: GRINDING_QUERY_POLICY_REVISION,
        fold_coordinate_oracle_revision: FOLD_COORDINATE_ORACLE_REVISION,
    };

    /// Fixed-width little-endian policy encoding.
    #[must_use]
    pub fn canonical_bytes(self) -> [u8; GRINDING_POLICY_BYTES] {
        let mut out = [0u8; GRINDING_POLICY_BYTES];
        out[0..2].copy_from_slice(&self.encoding_version.to_le_bytes());
        out[2..4].copy_from_slice(&self.target_security_bits.to_le_bytes());
        out[4] = self.proof_of_work_slack_bits;
        out[5] = self.maximum_proof_of_work_bits;
        out[6] = self.predicate_bytes;
        out[7] = self.predicate_bit_order_tag;
        out[8] = self.fold_response_attempt_bits;
        out[9..13].copy_from_slice(&self.fold_response_attempts.to_le_bytes());
        out[13..15].copy_from_slice(&self.query_policy_revision.to_le_bytes());
        out[15..17].copy_from_slice(&self.fold_coordinate_oracle_revision.to_le_bytes());
        out
    }

    pub(crate) fn from_canonical_bytes(bytes: [u8; GRINDING_POLICY_BYTES]) -> Self {
        Self {
            encoding_version: u16::from_le_bytes([bytes[0], bytes[1]]),
            target_security_bits: u16::from_le_bytes([bytes[2], bytes[3]]),
            proof_of_work_slack_bits: bytes[4],
            maximum_proof_of_work_bits: bytes[5],
            predicate_bytes: bytes[6],
            predicate_bit_order_tag: bytes[7],
            fold_response_attempt_bits: bytes[8],
            fold_response_attempts: u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]),
            query_policy_revision: u16::from_le_bytes([bytes[13], bytes[14]]),
            fold_coordinate_oracle_revision: u16::from_le_bytes([bytes[15], bytes[16]]),
        }
    }
}

/// Security role of one ordered plan run.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GrindingQueryKind {
    /// Public proof-of-work before a Fiat-Shamir challenge.
    ProofOfWork,
    /// Existing bounded response rejection search.
    FoldResponse,
    /// One transcript-derived sparse fold group root.
    FoldChallengeRoot,
    /// A compact run of independently indexed sparse coordinates.
    FoldChallengeCoordinates,
}

impl GrindingQueryKind {
    const fn tag(self) -> u8 {
        match self {
            Self::ProofOfWork => 0,
            Self::FoldResponse => 1,
            Self::FoldChallengeRoot => 2,
            Self::FoldChallengeCoordinates => 3,
        }
    }
}

/// Sumcheck family used in a fixed-width site payload.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SumcheckProtocol {
    ExtensionOpeningReduction,
    Stage1,
    PhysicalL2,
    Stage2,
    Stage3,
}

impl SumcheckProtocol {
    const fn tag(self) -> u32 {
        match self {
            Self::ExtensionOpeningReduction => 0,
            Self::Stage1 => 1,
            Self::PhysicalL2 => 2,
            Self::Stage2 => 3,
            Self::Stage3 => 4,
        }
    }
}

/// Fixed-width logical query identity in verifier replay order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GrindingSite {
    EvaluationBatch,
    ExtensionOpeningPoint,
    ExtensionOpeningClaimBatch,
    SumcheckRound {
        protocol: SumcheckProtocol,
        level: u32,
        stage: u32,
        round: u32,
    },
    FoldResponse {
        level: u32,
    },
    FoldChallengeRoot {
        level: u32,
        group: u32,
    },
    FoldChallengeCoordinates {
        level: u32,
        group: u32,
    },
    RingSwitchAlpha {
        level: u32,
    },
    Tau0Point {
        level: u32,
    },
    Tau1Point {
        level: u32,
    },
    Stage1InterstageBatch {
        level: u32,
        stage: u32,
    },
    L2SubclaimBatch {
        level: u32,
    },
    L2NormMerge {
        level: u32,
    },
    L2VirtualBatch {
        level: u32,
    },
    CompressionBinary {
        level: u32,
    },
    Stage2Batch {
        level: u32,
    },
}

impl GrindingSite {
    /// Security role determined by this logical site.
    #[must_use]
    pub const fn kind(self) -> GrindingQueryKind {
        match self {
            Self::FoldResponse { .. } => GrindingQueryKind::FoldResponse,
            Self::FoldChallengeRoot { .. } => GrindingQueryKind::FoldChallengeRoot,
            Self::FoldChallengeCoordinates { .. } => GrindingQueryKind::FoldChallengeCoordinates,
            _ => GrindingQueryKind::ProofOfWork,
        }
    }

    fn validate(self) -> Result<(), AkitaError> {
        let invalid = match self {
            Self::EvaluationBatch
            | Self::ExtensionOpeningPoint
            | Self::ExtensionOpeningClaimBatch => false,
            Self::SumcheckRound {
                protocol,
                level,
                stage,
                round,
            } => {
                let level_invalid =
                    level == u32::MAX && protocol != SumcheckProtocol::ExtensionOpeningReduction;
                level_invalid || stage == u32::MAX || round == u32::MAX
            }
            Self::FoldResponse { level }
            | Self::RingSwitchAlpha { level }
            | Self::Tau0Point { level }
            | Self::Tau1Point { level }
            | Self::L2SubclaimBatch { level }
            | Self::L2NormMerge { level }
            | Self::L2VirtualBatch { level }
            | Self::CompressionBinary { level }
            | Self::Stage2Batch { level } => level == u32::MAX,
            Self::FoldChallengeRoot { level, group }
            | Self::FoldChallengeCoordinates { level, group } => {
                level == u32::MAX || group == u32::MAX
            }
            Self::Stage1InterstageBatch { level, stage } => level == u32::MAX || stage == u32::MAX,
        };
        if invalid {
            return Err(AkitaError::InvalidSetup(
                "grinding site uses a reserved u32 sentinel".into(),
            ));
        }
        Ok(())
    }

    fn append_canonical_bytes(self, out: &mut Vec<u8>) {
        match self {
            Self::EvaluationBatch => out.push(0),
            Self::ExtensionOpeningPoint => out.push(1),
            Self::ExtensionOpeningClaimBatch => out.push(2),
            Self::SumcheckRound {
                protocol,
                level,
                stage,
                round,
            } => {
                out.push(3);
                push_u32(out, protocol.tag());
                push_u32(out, level);
                push_u32(out, stage);
                push_u32(out, round);
            }
            Self::FoldResponse { level } => {
                out.push(4);
                push_u32(out, level);
            }
            Self::FoldChallengeRoot { level, group } => {
                out.push(5);
                push_u32(out, level);
                push_u32(out, group);
            }
            Self::FoldChallengeCoordinates { level, group } => {
                out.push(6);
                push_u32(out, level);
                push_u32(out, group);
            }
            Self::RingSwitchAlpha { level } => {
                out.push(7);
                push_u32(out, level);
            }
            Self::Tau0Point { level } => {
                out.push(8);
                push_u32(out, level);
            }
            Self::Tau1Point { level } => {
                out.push(9);
                push_u32(out, level);
            }
            Self::Stage1InterstageBatch { level, stage } => {
                out.push(10);
                push_u32(out, level);
                push_u32(out, stage);
            }
            Self::L2SubclaimBatch { level } => {
                out.push(11);
                push_u32(out, level);
            }
            Self::L2NormMerge { level } => {
                out.push(12);
                push_u32(out, level);
            }
            Self::L2VirtualBatch { level } => {
                out.push(13);
                push_u32(out, level);
            }
            Self::CompressionBinary { level } => {
                out.push(14);
                push_u32(out, level);
            }
            Self::Stage2Batch { level } => {
                out.push(15);
                push_u32(out, level);
            }
        }
    }
}

/// One compact plan run in protocol replay order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GrindingRun {
    site: GrindingSite,
    loss_factor: u64,
    grind_bits: u8,
    nonce_bits: u8,
    multiplicity: u64,
}

impl GrindingRun {
    /// Logical replay site.
    #[must_use]
    pub const fn site(self) -> GrindingSite {
        self.site
    }

    /// Security role of this run.
    #[must_use]
    pub const fn kind(self) -> GrindingQueryKind {
        self.site.kind()
    }

    /// Conditional bad set loss factor, or zero for a non proof of work run.
    #[must_use]
    pub const fn loss_factor(self) -> u64 {
        self.loss_factor
    }

    /// Public proof of work target in low zero bits.
    #[must_use]
    pub const fn grind_bits(self) -> u8 {
        self.grind_bits
    }

    /// Packed value width consumed by each expanded entry.
    #[must_use]
    pub const fn nonce_bits(self) -> u8 {
        self.nonce_bits
    }

    /// Number of logical entries represented by this compact run.
    #[must_use]
    pub const fn multiplicity(self) -> u64 {
        self.multiplicity
    }

    /// Construct one proof-of-work site under the nominal field capacity.
    pub fn proof_of_work(
        site: GrindingSite,
        loss_factor: u64,
        nominal_capacity_bits: u32,
    ) -> Result<Self, AkitaError> {
        if !matches!(site.kind(), GrindingQueryKind::ProofOfWork) {
            return Err(AkitaError::InvalidSetup(
                "special grinding sites cannot be proof-of-work runs".into(),
            ));
        }
        let grind_bits = grind_bits_for_loss(loss_factor, nominal_capacity_bits)?;
        let nonce_bits = if grind_bits == 0 {
            0
        } else {
            grind_bits
                .checked_add(GRINDING_NONCE_SLACK_BITS)
                .ok_or_else(|| AkitaError::InvalidSetup("grinding nonce width overflow".into()))?
        };
        Ok(Self {
            site,
            loss_factor,
            grind_bits,
            nonce_bits,
            multiplicity: 1,
        })
    }

    /// Construct the existing one-per-fold response search entry.
    #[must_use]
    pub const fn fold_response(level: u32) -> Self {
        Self {
            site: GrindingSite::FoldResponse { level },
            loss_factor: 0,
            grind_bits: 0,
            nonce_bits: FOLD_RESPONSE_NONCE_BITS,
            multiplicity: 1,
        }
    }

    /// Construct one zero-width group-root audit entry.
    #[must_use]
    pub const fn fold_challenge_root(level: u32, group: u32) -> Self {
        Self {
            site: GrindingSite::FoldChallengeRoot { level, group },
            loss_factor: 0,
            grind_bits: 0,
            nonce_bits: 0,
            multiplicity: 1,
        }
    }

    /// Construct one compact zero-width indexed-coordinate run.
    #[must_use]
    pub const fn fold_challenge_coordinates(level: u32, group: u32, multiplicity: u64) -> Self {
        Self {
            site: GrindingSite::FoldChallengeCoordinates { level, group },
            loss_factor: 0,
            grind_bits: 0,
            nonce_bits: 0,
            multiplicity,
        }
    }

    fn validate(self) -> Result<(), AkitaError> {
        self.site.validate()?;
        match self.kind() {
            GrindingQueryKind::ProofOfWork => {
                if self.loss_factor == 0 || self.multiplicity != 1 {
                    return Err(AkitaError::InvalidSetup(
                        "proof-of-work run has invalid loss or multiplicity".into(),
                    ));
                }
                let expected_nonce_bits = if self.grind_bits == 0 {
                    0
                } else {
                    self.grind_bits
                        .checked_add(GRINDING_NONCE_SLACK_BITS)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("grinding nonce width overflow".into())
                        })?
                };
                if self.grind_bits > MAX_GRINDING_BITS || self.nonce_bits != expected_nonce_bits {
                    return Err(AkitaError::InvalidSetup(
                        "proof-of-work run has invalid target or nonce width".into(),
                    ));
                }
            }
            GrindingQueryKind::FoldResponse
                if self.loss_factor == 0
                    && self.grind_bits == 0
                    && self.nonce_bits == FOLD_RESPONSE_NONCE_BITS
                    && self.multiplicity == 1 => {}
            GrindingQueryKind::FoldChallengeRoot
                if self.loss_factor == 0
                    && self.grind_bits == 0
                    && self.nonce_bits == 0
                    && self.multiplicity == 1 => {}
            GrindingQueryKind::FoldChallengeCoordinates
                if self.loss_factor == 0
                    && self.grind_bits == 0
                    && self.nonce_bits == 0
                    && self.multiplicity > 0 => {}
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "grinding run kind and site do not match".into(),
                ));
            }
        }
        Ok(())
    }

    fn append_canonical_bytes(self, out: &mut Vec<u8>) {
        out.push(self.kind().tag());
        self.site.append_canonical_bytes(out);
        out.extend_from_slice(&self.loss_factor.to_le_bytes());
        out.push(self.grind_bits);
        out.push(self.nonce_bits);
        out.extend_from_slice(&self.multiplicity.to_le_bytes());
    }
}

/// Validated public transcript-grinding replay plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrindingPlan {
    runs: Vec<GrindingRun>,
    nominal_capacity_bits: u32,
    total_nonce_bits: usize,
    expanded_query_count: u64,
}

#[derive(Clone, Copy)]
struct GrindingPlanEntry {
    site: GrindingSite,
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
            nonce_bits: run.nonce_bits,
        };
        self.run_offset += 1;
        if self.run_offset == run.multiplicity {
            self.run_index += 1;
            self.run_offset = 0;
        }
        Some(entry)
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
            plan: GrindingPlanCursor::new(plan),
            bit_offset: 0,
        })
    }
}

/// Checked sequential writer for one public grinding plan.
pub struct TranscriptNonceWriter<'a> {
    bytes: Vec<u8>,
    plan: GrindingPlanCursor<'a>,
    bit_offset: usize,
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
            plan: GrindingPlanCursor::new(plan),
            bit_offset: 0,
        })
    }

    /// Write the next expected logical entry and reject a site or width mismatch.
    pub fn write(&mut self, site: GrindingSite, value: u32) -> Result<(), AkitaError> {
        let entry = self.plan.next().ok_or(AkitaError::InvalidProof)?;
        if entry.site != site || !value_fits(value, entry.nonce_bits) {
            return Err(AkitaError::InvalidProof);
        }
        write_bits(&mut self.bytes, self.bit_offset, entry.nonce_bits, value);
        self.bit_offset = self
            .bit_offset
            .checked_add(usize::from(entry.nonce_bits))
            .ok_or(AkitaError::InvalidProof)?;
        Ok(())
    }

    /// Reserve inactive entries as zero, then write the next fold-response nonce.
    pub fn write_next_fold_response(
        &mut self,
        site: GrindingSite,
        value: u32,
    ) -> Result<(), AkitaError> {
        loop {
            let entry = self.plan.next().ok_or(AkitaError::InvalidProof)?;
            if entry.site.kind() == GrindingQueryKind::FoldResponse {
                if entry.site != site || !value_fits(value, entry.nonce_bits) {
                    return Err(AkitaError::InvalidProof);
                }
                write_bits(&mut self.bytes, self.bit_offset, entry.nonce_bits, value);
                self.bit_offset += usize::from(entry.nonce_bits);
                return Ok(());
            }
            self.bit_offset += usize::from(entry.nonce_bits);
        }
    }

    /// Finish the stream, requiring that no fold-response entry was omitted.
    pub fn finish(mut self) -> Result<TranscriptNonceStream, AkitaError> {
        while let Some(entry) = self.plan.next() {
            if entry.site.kind() == GrindingQueryKind::FoldResponse {
                return Err(AkitaError::InvalidProof);
            }
            self.bit_offset += usize::from(entry.nonce_bits);
        }
        if self.bit_offset != self.plan.plan.total_nonce_bits {
            return Err(AkitaError::InvalidProof);
        }
        Ok(TranscriptNonceStream {
            bytes: self.bytes,
            bit_len: self.bit_offset,
        })
    }
}

/// Checked sequential reader for one public grinding plan.
pub struct TranscriptNonceReader<'a> {
    stream: &'a TranscriptNonceStream,
    plan: GrindingPlanCursor<'a>,
    bit_offset: usize,
}

impl TranscriptNonceReader<'_> {
    /// Read the next expected logical entry and reject a site mismatch.
    pub fn read(&mut self, site: GrindingSite) -> Result<u32, AkitaError> {
        let entry = self.plan.next().ok_or(AkitaError::InvalidProof)?;
        if entry.site != site {
            return Err(AkitaError::InvalidProof);
        }
        let value = read_bits(self.stream.as_bytes(), self.bit_offset, entry.nonce_bits);
        self.bit_offset = self
            .bit_offset
            .checked_add(usize::from(entry.nonce_bits))
            .ok_or(AkitaError::InvalidProof)?;
        Ok(value)
    }

    /// Require inactive entries to be zero, then read the next fold-response nonce.
    pub fn read_next_fold_response(&mut self, site: GrindingSite) -> Result<u32, AkitaError> {
        loop {
            let entry = self.plan.next().ok_or(AkitaError::InvalidProof)?;
            let value = read_bits(self.stream.as_bytes(), self.bit_offset, entry.nonce_bits);
            self.bit_offset += usize::from(entry.nonce_bits);
            if entry.site.kind() == GrindingQueryKind::FoldResponse {
                if entry.site != site {
                    return Err(AkitaError::InvalidProof);
                }
                return Ok(value);
            }
            if value != 0 {
                return Err(AkitaError::InvalidProof);
            }
        }
    }

    /// Finish replay, requiring zero reserved entries and no omitted fold response.
    pub fn finish(mut self) -> Result<(), AkitaError> {
        while let Some(entry) = self.plan.next() {
            let value = read_bits(self.stream.as_bytes(), self.bit_offset, entry.nonce_bits);
            self.bit_offset += usize::from(entry.nonce_bits);
            if entry.site.kind() == GrindingQueryKind::FoldResponse || value != 0 {
                return Err(AkitaError::InvalidProof);
            }
        }
        if self.bit_offset != self.stream.bit_len {
            return Err(AkitaError::InvalidProof);
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
    let mut value = 0u32;
    for bit in 0..usize::from(width) {
        let stream_bit = offset + bit;
        value |= u32::from((bytes[stream_bit / 8] >> (stream_bit % 8)) & 1) << bit;
    }
    value
}

impl GrindingPlan {
    /// Validate ordered runs and derive all aggregate counts once.
    pub fn new(runs: Vec<GrindingRun>, nominal_capacity_bits: u32) -> Result<Self, AkitaError> {
        if nominal_capacity_bits == 0 {
            return Err(AkitaError::InvalidSetup(
                "grinding nominal capacity must be nonzero".into(),
            ));
        }
        u32::try_from(runs.len())
            .map_err(|_| AkitaError::InvalidSetup("grinding plan run count exceeds u32".into()))?;
        let mut total_nonce_bits = 0usize;
        let mut expanded_query_count = 0u64;
        for run in &runs {
            run.validate()?;
            if run.kind() == GrindingQueryKind::ProofOfWork
                && run.grind_bits != grind_bits_for_loss(run.loss_factor, nominal_capacity_bits)?
            {
                return Err(AkitaError::InvalidSetup(
                    "proof-of-work run target does not match its loss and capacity".into(),
                ));
            }
            let multiplicity = usize::try_from(run.multiplicity).map_err(|_| {
                AkitaError::InvalidSetup("grinding run multiplicity exceeds usize".into())
            })?;
            let run_bits = usize::from(run.nonce_bits)
                .checked_mul(multiplicity)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("grinding run bit count overflow".into())
                })?;
            total_nonce_bits = total_nonce_bits.checked_add(run_bits).ok_or_else(|| {
                AkitaError::InvalidSetup("grinding plan bit count overflow".into())
            })?;
            expanded_query_count = expanded_query_count
                .checked_add(run.multiplicity)
                .ok_or_else(|| AkitaError::InvalidSetup("grinding query count overflow".into()))?;
        }
        if expanded_query_count >= u64::from(u32::MAX) {
            return Err(AkitaError::InvalidSetup(
                "grinding plan query count must be less than 2^32".into(),
            ));
        }
        Ok(Self {
            runs,
            nominal_capacity_bits,
            total_nonce_bits,
            expanded_query_count,
        })
    }

    #[must_use]
    pub fn runs(&self) -> &[GrindingRun] {
        &self.runs
    }

    /// Nominal extension field capacity used to price all proof of work runs.
    #[must_use]
    pub const fn nominal_capacity_bits(&self) -> u32 {
        self.nominal_capacity_bits
    }

    #[must_use]
    pub const fn total_nonce_bits(&self) -> usize {
        self.total_nonce_bits
    }

    #[must_use]
    pub const fn expanded_query_count(&self) -> u64 {
        self.expanded_query_count
    }

    /// Canonical digest input, including active policy and every run.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, AkitaError> {
        let run_count = u32::try_from(self.runs.len())
            .map_err(|_| AkitaError::InvalidSetup("grinding plan run count exceeds u32".into()))?;
        let mut out = Vec::new();
        out.extend_from_slice(GRINDING_PLAN_DOMAIN);
        out.extend_from_slice(&GrindingPolicy::ACTIVE.canonical_bytes());
        push_u32(&mut out, run_count);
        for run in &self.runs {
            run.append_canonical_bytes(&mut out);
        }
        Ok(out)
    }

    /// Blake2b-256 digest bound into the instance descriptor.
    pub fn digest(&self) -> Result<[u8; 32], AkitaError> {
        Ok(digest_descriptor_bytes(&self.canonical_bytes()?))
    }
}

/// Nominal field capacity used by the current Fiat-Shamir accounting.
pub fn nominal_challenge_capacity_bits(
    modulus_bits: u32,
    extension_degree: usize,
) -> Result<u32, AkitaError> {
    let extension_degree = u32::try_from(extension_degree)
        .map_err(|_| AkitaError::InvalidSetup("challenge extension degree exceeds u32".into()))?;
    modulus_bits
        .checked_mul(extension_degree)
        .ok_or_else(|| AkitaError::InvalidSetup("nominal challenge capacity overflow".into()))
}

/// Assign the exact public proof-of-work target for one loss factor.
pub fn grind_bits_for_loss(loss_factor: u64, nominal_capacity_bits: u32) -> Result<u8, AkitaError> {
    if loss_factor == 0 {
        return Err(AkitaError::InvalidSetup(
            "proof-of-work loss factor must be nonzero".into(),
        ));
    }
    let loss_bits = u64::BITS - loss_factor.saturating_sub(1).leading_zeros();
    let required = u32::from(TRANSCRIPT_SECURITY_BITS)
        .checked_add(loss_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("grinding target overflow".into()))?;
    let target = required.saturating_sub(nominal_capacity_bits);
    let target = u8::try_from(target)
        .map_err(|_| AkitaError::InvalidSetup("grinding target exceeds u8".into()))?;
    if target > MAX_GRINDING_BITS {
        return Err(AkitaError::InvalidSetup(format!(
            "grinding target {target} exceeds supported maximum {MAX_GRINDING_BITS}"
        )));
    }
    Ok(target)
}

/// Loss for a nonzero polynomial identity of the declared degree.
pub fn polynomial_identity_loss_factor(degree: usize) -> Result<u64, AkitaError> {
    u64::try_from(degree.max(1))
        .map_err(|_| AkitaError::InvalidSetup("polynomial degree exceeds u64".into()))
}

/// Loss for one complete multilinear point draw.
pub fn multilinear_point_loss_factor(coordinates: usize) -> Result<u64, AkitaError> {
    u64::try_from(coordinates.max(1))
        .map_err(|_| AkitaError::InvalidSetup("multilinear point width exceeds u64".into()))
}

/// Loss for batching `values` with powers of one scalar.
pub fn powers_batch_loss_factor(values: usize) -> Result<u64, AkitaError> {
    u64::try_from(values.saturating_sub(1).max(1))
        .map_err(|_| AkitaError::InvalidSetup("powers batch length exceeds u64".into()))
}

/// Canonical ring-switch polynomial loss for one opening method.
pub fn ring_switch_alpha_loss_factor(
    opening_method: OpeningMethod,
    inner_ring_dimension: usize,
) -> Result<u64, AkitaError> {
    let degree_bound = match opening_method {
        OpeningMethod::EvaluationTrace => inner_ring_dimension
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1)),
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } => challenge_subring_dimension
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1)),
    }
    .ok_or_else(|| AkitaError::InvalidSetup("ring-switch alpha degree overflow".into()))?;
    polynomial_identity_loss_factor(degree_bound)
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests;
