//! Public transcript-grinding policy and canonical replay plan.

use crate::instance_descriptor::digest_descriptor_bytes;
use crate::OpeningMethod;
use akita_error::AkitaError;
use akita_serialization::SerializationError;

/// Target work factor for every grinding-priced Fiat-Shamir query.
pub const TRANSCRIPT_SECURITY_BITS: u16 = 128;
/// Extra nonce bits that make honest proof-of-work exhaustion negligible.
pub const GRINDING_NONCE_SLACK_BITS: u8 = 7;
/// Largest proof-of-work target supported by the first implementation.
pub const MAX_GRINDING_BITS: u8 = 25;
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
/// Fixed public predicate output length.
pub const GRINDING_PREDICATE_BYTES: u8 = 32;
/// Low-bit-first predicate and packed-stream bit order.
pub const GRINDING_LITTLE_ENDIAN_BIT_ORDER: u8 = 0;

const GRINDING_PLAN_DOMAIN: &[u8] = b"akita/grinding-plan/v1";

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
    kind: GrindingQueryKind,
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
        self.kind
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
            kind: GrindingQueryKind::ProofOfWork,
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
            kind: GrindingQueryKind::FoldResponse,
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
            kind: GrindingQueryKind::FoldChallengeRoot,
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
            kind: GrindingQueryKind::FoldChallengeCoordinates,
            loss_factor: 0,
            grind_bits: 0,
            nonce_bits: 0,
            multiplicity,
        }
    }

    fn validate(self) -> Result<(), AkitaError> {
        self.site.validate()?;
        match (self.kind, self.site) {
            (GrindingQueryKind::ProofOfWork, site)
                if !matches!(
                    site,
                    GrindingSite::FoldResponse { .. }
                        | GrindingSite::FoldChallengeRoot { .. }
                        | GrindingSite::FoldChallengeCoordinates { .. }
                ) =>
            {
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
            (GrindingQueryKind::FoldResponse, GrindingSite::FoldResponse { .. })
                if self.loss_factor == 0
                    && self.grind_bits == 0
                    && self.nonce_bits == FOLD_RESPONSE_NONCE_BITS
                    && self.multiplicity == 1 => {}
            (GrindingQueryKind::FoldChallengeRoot, GrindingSite::FoldChallengeRoot { .. })
                if self.loss_factor == 0
                    && self.grind_bits == 0
                    && self.nonce_bits == 0
                    && self.multiplicity == 1 => {}
            (
                GrindingQueryKind::FoldChallengeCoordinates,
                GrindingSite::FoldChallengeCoordinates { .. },
            ) if self.loss_factor == 0
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
        out.push(self.kind.tag());
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
    kind: GrindingQueryKind,
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
            kind: run.kind,
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

    /// Write the next expected logical entry and reject a site, kind, or width mismatch.
    pub fn write(
        &mut self,
        site: GrindingSite,
        kind: GrindingQueryKind,
        value: u32,
    ) -> Result<(), AkitaError> {
        let entry = self.plan.next().ok_or(AkitaError::InvalidProof)?;
        if entry.site != site || entry.kind != kind || !value_fits(value, entry.nonce_bits) {
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
            if entry.kind == GrindingQueryKind::FoldResponse {
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
            if entry.kind == GrindingQueryKind::FoldResponse {
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
    /// Read the next expected logical entry and reject a site or kind mismatch.
    pub fn read(&mut self, site: GrindingSite, kind: GrindingQueryKind) -> Result<u32, AkitaError> {
        let entry = self.plan.next().ok_or(AkitaError::InvalidProof)?;
        if entry.site != site || entry.kind != kind {
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
            if entry.kind == GrindingQueryKind::FoldResponse {
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
            if entry.kind == GrindingQueryKind::FoldResponse || value != 0 {
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
            if run.kind == GrindingQueryKind::ProofOfWork
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
        append_grinding_policy_bytes(&mut out);
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

/// Append the fixed active policy bytes used by both plan digest and binding.
pub(crate) fn append_grinding_policy_bytes(out: &mut Vec<u8>) {
    out.extend_from_slice(&GRINDING_ENCODING_VERSION.to_le_bytes());
    out.extend_from_slice(&TRANSCRIPT_SECURITY_BITS.to_le_bytes());
    out.push(GRINDING_NONCE_SLACK_BITS);
    out.push(MAX_GRINDING_BITS);
    out.push(GRINDING_PREDICATE_BYTES);
    out.push(GRINDING_LITTLE_ENDIAN_BIT_ORDER);
    out.push(FOLD_RESPONSE_NONCE_BITS);
    out.extend_from_slice(&FOLD_RESPONSE_ATTEMPTS.to_le_bytes());
    out.extend_from_slice(&GRINDING_QUERY_POLICY_REVISION.to_le_bytes());
    out.extend_from_slice(&FOLD_COORDINATE_ORACLE_REVISION.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream_test_plan() -> GrindingPlan {
        GrindingPlan::new(
            vec![
                GrindingRun::proof_of_work(GrindingSite::RingSwitchAlpha { level: 0 }, 2, 128)
                    .unwrap(),
                GrindingRun::fold_response(0),
                GrindingRun::proof_of_work(GrindingSite::Tau0Point { level: 0 }, 4, 128).unwrap(),
                GrindingRun::fold_response(1),
            ],
            128,
        )
        .unwrap()
    }

    #[test]
    fn nonce_stream_is_little_endian_and_crosses_byte_boundaries() {
        let plan = stream_test_plan();
        let mut writer = TranscriptNonceWriter::new(&plan).unwrap();
        writer
            .write(
                GrindingSite::RingSwitchAlpha { level: 0 },
                GrindingQueryKind::ProofOfWork,
                0xa5,
            )
            .unwrap();
        writer
            .write(
                GrindingSite::FoldResponse { level: 0 },
                GrindingQueryKind::FoldResponse,
                0xabc,
            )
            .unwrap();
        writer
            .write(
                GrindingSite::Tau0Point { level: 0 },
                GrindingQueryKind::ProofOfWork,
                0x101,
            )
            .unwrap();
        writer
            .write(
                GrindingSite::FoldResponse { level: 1 },
                GrindingQueryKind::FoldResponse,
                0x123,
            )
            .unwrap();
        let stream = writer.finish().unwrap();
        assert_eq!(stream.bit_len(), 41);
        assert_eq!(stream.as_bytes(), &[0xa5, 0xbc, 0x1a, 0x70, 0x24, 0x00]);

        let mut reader = stream.reader(&plan).unwrap();
        assert_eq!(
            reader
                .read(
                    GrindingSite::RingSwitchAlpha { level: 0 },
                    GrindingQueryKind::ProofOfWork,
                )
                .unwrap(),
            0xa5
        );
        assert_eq!(
            reader
                .read(
                    GrindingSite::FoldResponse { level: 0 },
                    GrindingQueryKind::FoldResponse,
                )
                .unwrap(),
            0xabc
        );
        assert_eq!(
            reader
                .read(
                    GrindingSite::Tau0Point { level: 0 },
                    GrindingQueryKind::ProofOfWork,
                )
                .unwrap(),
            0x101
        );
        assert_eq!(
            reader
                .read(
                    GrindingSite::FoldResponse { level: 1 },
                    GrindingQueryKind::FoldResponse,
                )
                .unwrap(),
            0x123
        );
        reader.finish().unwrap();
    }

    #[test]
    fn inactive_entries_are_canonical_zero_and_fold_width_is_checked() {
        let plan = stream_test_plan();
        let mut writer = TranscriptNonceWriter::new(&plan).unwrap();
        writer
            .write_next_fold_response(GrindingSite::FoldResponse { level: 0 }, 17)
            .unwrap();
        assert!(writer
            .write_next_fold_response(
                GrindingSite::FoldResponse { level: 1 },
                FOLD_RESPONSE_ATTEMPTS,
            )
            .is_err());

        let mut writer = TranscriptNonceWriter::new(&plan).unwrap();
        writer
            .write_next_fold_response(GrindingSite::FoldResponse { level: 0 }, 17)
            .unwrap();
        writer
            .write_next_fold_response(GrindingSite::FoldResponse { level: 1 }, 23)
            .unwrap();
        let stream = writer.finish().unwrap();
        let mut reader = stream.reader(&plan).unwrap();
        assert_eq!(
            reader
                .read_next_fold_response(GrindingSite::FoldResponse { level: 0 })
                .unwrap(),
            17
        );
        assert_eq!(
            reader
                .read_next_fold_response(GrindingSite::FoldResponse { level: 1 })
                .unwrap(),
            23
        );
        reader.finish().unwrap();

        let mut malleable = stream.as_bytes().to_vec();
        malleable[0] |= 1;
        let malleable = TranscriptNonceStream::from_bytes(malleable, stream.bit_len()).unwrap();
        assert!(malleable
            .reader(&plan)
            .unwrap()
            .read_next_fold_response(GrindingSite::FoldResponse { level: 0 })
            .is_err());
    }

    #[test]
    fn nonce_stream_rejects_wrong_length_and_nonzero_padding() {
        assert!(TranscriptNonceStream::from_bytes(vec![0], 9).is_err());
        assert!(TranscriptNonceStream::from_bytes(vec![0, 0x80], 9).is_err());
        assert!(TranscriptNonceStream::from_bytes(vec![0, 1], 9).is_ok());
    }

    #[test]
    fn current_capacity_prices_exact_nominal_loss_bits() {
        for (loss, expected) in [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (u64::MAX, 64)] {
            let actual = if expected > u32::from(MAX_GRINDING_BITS) {
                grind_bits_for_loss(loss, 128).expect_err("oversized target")
            } else {
                let actual = grind_bits_for_loss(loss, 128).expect("supported target");
                assert_eq!(u32::from(actual), expected);
                continue;
            };
            assert!(matches!(actual, AkitaError::InvalidSetup(_)));
        }
    }

    #[test]
    fn nominal_security_inequality_holds_for_every_supported_target() {
        let losses = [
            1,
            2,
            3,
            4,
            5,
            (1u64 << (MAX_GRINDING_BITS - 1)) - 1,
            1u64 << (MAX_GRINDING_BITS - 1),
            (1u64 << MAX_GRINDING_BITS) - 1,
            1u64 << MAX_GRINDING_BITS,
        ];
        for loss in losses {
            let grind = grind_bits_for_loss(loss, 128).expect("supported loss");
            assert!(u128::from(loss) <= (1u128 << grind));
        }
    }

    #[test]
    fn nonce_slack_provisions_exactly_128_expected_trials() {
        for grind in 1..=MAX_GRINDING_BITS {
            let nonce_bits = grind + GRINDING_NONCE_SLACK_BITS;
            assert_eq!((1u64 << nonce_bits) / (1u64 << grind), 128);
            let failure =
                (1.0 - 2f64.powi(-i32::from(grind))).powf(2f64.powi(i32::from(nonce_bits)));
            assert!(failure <= (-128f64).exp());
        }
    }

    #[test]
    fn plan_encoding_covers_every_discriminator() {
        let capacity = 128;
        let sites = [
            GrindingSite::EvaluationBatch,
            GrindingSite::ExtensionOpeningPoint,
            GrindingSite::ExtensionOpeningClaimBatch,
            GrindingSite::SumcheckRound {
                protocol: SumcheckProtocol::ExtensionOpeningReduction,
                level: u32::MAX,
                stage: 1,
                round: 2,
            },
            GrindingSite::SumcheckRound {
                protocol: SumcheckProtocol::Stage1,
                level: 3,
                stage: 4,
                round: 5,
            },
            GrindingSite::SumcheckRound {
                protocol: SumcheckProtocol::PhysicalL2,
                level: 6,
                stage: 7,
                round: 8,
            },
            GrindingSite::SumcheckRound {
                protocol: SumcheckProtocol::Stage2,
                level: 9,
                stage: 10,
                round: 11,
            },
            GrindingSite::SumcheckRound {
                protocol: SumcheckProtocol::Stage3,
                level: 12,
                stage: 13,
                round: 14,
            },
            GrindingSite::RingSwitchAlpha { level: 1 },
            GrindingSite::Tau0Point { level: 1 },
            GrindingSite::Tau1Point { level: 1 },
            GrindingSite::Stage1InterstageBatch { level: 1, stage: 2 },
            GrindingSite::L2SubclaimBatch { level: 1 },
            GrindingSite::L2NormMerge { level: 1 },
            GrindingSite::L2VirtualBatch { level: 1 },
            GrindingSite::CompressionBinary { level: 1 },
            GrindingSite::Stage2Batch { level: 1 },
        ];
        let mut runs = sites
            .into_iter()
            .map(|site| GrindingRun::proof_of_work(site, 3, capacity).unwrap())
            .collect::<Vec<_>>();
        runs.push(GrindingRun::fold_response(2));
        runs.push(GrindingRun::fold_challenge_root(2, 3));
        runs.push(GrindingRun::fold_challenge_coordinates(2, 3, 4));
        let plan = GrindingPlan::new(runs, capacity).unwrap();
        let bytes = plan.canonical_bytes().unwrap();
        assert!(bytes.starts_with(GRINDING_PLAN_DOMAIN));
        assert_eq!(plan.expanded_query_count(), 23);
        assert_eq!(plan.total_nonce_bits(), 17 * 9 + 12);
        assert_eq!(
            plan.digest().unwrap(),
            [
                201, 71, 193, 56, 131, 65, 105, 160, 79, 152, 66, 44, 189, 232, 205, 168, 168, 208,
                84, 23, 96, 48, 174, 168, 14, 112, 165, 199, 177, 190, 157, 156,
            ]
        );
    }

    #[test]
    fn ring_switch_loss_uses_the_opening_polynomial_dimension() {
        assert_eq!(
            ring_switch_alpha_loss_factor(OpeningMethod::EvaluationTrace, 64).unwrap(),
            127
        );
        assert_eq!(
            ring_switch_alpha_loss_factor(
                OpeningMethod::SubringCoefficientPacking {
                    challenge_subring_dimension: 16,
                },
                64,
            )
            .unwrap(),
            31
        );
    }

    #[test]
    fn malformed_kind_site_and_reserved_sentinel_are_rejected() {
        let mismatched = GrindingRun {
            site: GrindingSite::FoldResponse { level: 0 },
            kind: GrindingQueryKind::ProofOfWork,
            loss_factor: 1,
            grind_bits: 0,
            nonce_bits: 0,
            multiplicity: 1,
        };
        assert!(GrindingPlan::new(vec![mismatched], 128).is_err());

        let mut underpriced =
            GrindingRun::proof_of_work(GrindingSite::RingSwitchAlpha { level: 0 }, 3, 128).unwrap();
        underpriced.grind_bits = 1;
        underpriced.nonce_bits = 8;
        assert!(GrindingPlan::new(vec![underpriced], 128).is_err());

        let reserved = GrindingRun::proof_of_work(
            GrindingSite::SumcheckRound {
                protocol: SumcheckProtocol::Stage2,
                level: u32::MAX,
                stage: 0,
                round: 0,
            },
            3,
            128,
        )
        .unwrap();
        assert!(GrindingPlan::new(vec![reserved], 128).is_err());
    }
}
