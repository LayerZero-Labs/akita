//! Public transcript-grinding policy and canonical replay plan.

use crate::descriptor_bytes::digest_descriptor_bytes;
use crate::OpeningMethod;
use akita_error::AkitaError;
use akita_transcript::native_nonce_max_bytes;
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
pub const GRINDING_ENCODING_VERSION: u16 = 3;
/// Query catalog and loss-policy revision.
pub const GRINDING_QUERY_POLICY_REVISION: u16 = 3;
/// Indexed fold-coordinate oracle revision.
pub const FOLD_COORDINATE_ORACLE_REVISION: u16 = 1;
/// Exclusive upper bound on expanded transcript queries in a complete plan.
///
/// This preserves the existing accepted set: a plan must contain fewer than
/// `u32::MAX` expanded queries.
pub const TRANSCRIPT_GRINDING_QUERY_LIMIT: u64 = u32::MAX as u64;

const GRINDING_PLAN_DOMAIN: &[u8] = b"akita/grinding-plan/v1";
const GRINDING_POLICY_BYTES: usize = 17;

fn active_grinding_policy_bytes() -> [u8; GRINDING_POLICY_BYTES] {
    let mut out = [0u8; GRINDING_POLICY_BYTES];
    out[0..2].copy_from_slice(&GRINDING_ENCODING_VERSION.to_le_bytes());
    out[2..4].copy_from_slice(&TRANSCRIPT_SECURITY_BITS.to_le_bytes());
    out[4] = GRINDING_NONCE_SLACK_BITS;
    out[5] = MAX_GRINDING_BITS;
    out[6] = GRINDING_PREDICATE_BYTES;
    out[7] = GRINDING_LITTLE_ENDIAN_BIT_ORDER;
    out[8] = FOLD_RESPONSE_NONCE_BITS;
    out[9..13].copy_from_slice(&FOLD_RESPONSE_ATTEMPTS.to_le_bytes());
    out[13..15].copy_from_slice(&GRINDING_QUERY_POLICY_REVISION.to_le_bytes());
    out[15..17].copy_from_slice(&FOLD_COORDINATE_ORACLE_REVISION.to_le_bytes());
    out
}

/// Security role of one ordered plan run.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GrindingQueryKind {
    /// Public proof-of-work before a Fiat-Shamir challenge.
    ProofOfWork,
    /// Existing bounded response rejection search.
    FoldResponse,
    /// One sparse fold group root and all independently indexed coordinates.
    FoldChallengeGroup,
}

/// Sumcheck family used in a fixed-width site payload.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SumcheckProtocol {
    ExtensionOpeningReduction,
    Stage1,
    PhysicalL2,
    Stage2,
    Stage3,
}

impl SumcheckProtocol {
    /// Canonical protocol discriminator stored in native sumcheck site identities.
    #[must_use]
    pub const fn tag(self) -> u32 {
        match self {
            Self::ExtensionOpeningReduction => 0,
            Self::Stage1 => 1,
            Self::PhysicalL2 => 2,
            Self::Stage2 => 3,
            Self::Stage3 => 4,
        }
    }

    /// Decode a canonical native sumcheck protocol discriminator.
    #[must_use]
    pub const fn from_tag(tag: u32) -> Option<Self> {
        match tag {
            0 => Some(Self::ExtensionOpeningReduction),
            1 => Some(Self::Stage1),
            2 => Some(Self::PhysicalL2),
            3 => Some(Self::Stage2),
            4 => Some(Self::Stage3),
            _ => None,
        }
    }
}

/// Fixed-width logical query identity in verifier replay order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GrindingSite {
    EvaluationBatch {
        level: u32,
    },
    ExtensionOpeningPoint {
        level: u32,
    },
    ExtensionOpeningClaimBatch {
        level: u32,
    },
    SumcheckRound {
        protocol: SumcheckProtocol,
        level: u32,
        stage: u32,
        round: u32,
    },
    FoldResponse {
        level: u32,
    },
    FoldChallengeGroup {
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
    fn native_site_id(self, detail: u32) -> akita_transcript::ProtocolSiteId {
        let mut site = akita_transcript::ProtocolSiteId {
            detail,
            ..akita_transcript::ProtocolSiteId::default()
        };
        match self {
            Self::EvaluationBatch { level } => {
                site.family = 100;
                site.level = level;
            }
            Self::ExtensionOpeningPoint { level } => {
                site.family = 101;
                site.level = level;
            }
            Self::ExtensionOpeningClaimBatch { level } => {
                site.family = 102;
                site.level = level;
            }
            Self::SumcheckRound {
                protocol,
                level,
                stage,
                round,
            } => {
                site.family = 103;
                site.invocation = protocol.tag();
                site.level = level;
                site.stage = stage;
                site.round = round;
            }
            Self::FoldResponse { level } => {
                site.family = 104;
                site.level = level;
            }
            Self::FoldChallengeGroup { level, group } => {
                site.family = 105;
                site.level = level;
                site.group = group;
            }
            Self::RingSwitchAlpha { level } => {
                site.family = 106;
                site.level = level;
            }
            Self::Tau0Point { level } => {
                site.family = 107;
                site.level = level;
            }
            Self::Tau1Point { level } => {
                site.family = 108;
                site.level = level;
            }
            Self::Stage1InterstageBatch { level, stage } => {
                site.family = 109;
                site.level = level;
                site.stage = stage;
            }
            Self::L2SubclaimBatch { level } => {
                site.family = 110;
                site.level = level;
            }
            Self::L2NormMerge { level } => {
                site.family = 111;
                site.level = level;
            }
            Self::L2VirtualBatch { level } => {
                site.family = 112;
                site.level = level;
            }
            Self::CompressionBinary { level } => {
                site.family = 113;
                site.level = level;
            }
            Self::Stage2Batch { level } => {
                site.family = 114;
                site.level = level;
            }
        }
        site
    }

    /// Security role determined by this logical site.
    #[must_use]
    pub const fn kind(self) -> GrindingQueryKind {
        match self {
            Self::FoldResponse { .. } => GrindingQueryKind::FoldResponse,
            Self::FoldChallengeGroup { .. } => GrindingQueryKind::FoldChallengeGroup,
            _ => GrindingQueryKind::ProofOfWork,
        }
    }

    /// Fold level containing this logical query.
    #[must_use]
    pub const fn level(self) -> u32 {
        match self {
            Self::EvaluationBatch { level }
            | Self::ExtensionOpeningPoint { level }
            | Self::ExtensionOpeningClaimBatch { level }
            | Self::SumcheckRound { level, .. }
            | Self::FoldResponse { level }
            | Self::FoldChallengeGroup { level, .. }
            | Self::RingSwitchAlpha { level }
            | Self::Tau0Point { level }
            | Self::Tau1Point { level }
            | Self::Stage1InterstageBatch { level, .. }
            | Self::L2SubclaimBatch { level }
            | Self::L2NormMerge { level }
            | Self::L2VirtualBatch { level }
            | Self::CompressionBinary { level }
            | Self::Stage2Batch { level } => level,
        }
    }

    fn validate(self) -> Result<(), AkitaError> {
        let invalid = match self {
            Self::SumcheckRound {
                level,
                stage,
                round,
                ..
            } => level == u32::MAX || stage == u32::MAX || round == u32::MAX,
            Self::EvaluationBatch { level }
            | Self::ExtensionOpeningPoint { level }
            | Self::ExtensionOpeningClaimBatch { level }
            | Self::FoldResponse { level }
            | Self::RingSwitchAlpha { level }
            | Self::Tau0Point { level }
            | Self::Tau1Point { level }
            | Self::L2SubclaimBatch { level }
            | Self::L2NormMerge { level }
            | Self::L2VirtualBatch { level }
            | Self::CompressionBinary { level }
            | Self::Stage2Batch { level } => level == u32::MAX,
            Self::FoldChallengeGroup { level, group } => level == u32::MAX || group == u32::MAX,
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
            Self::EvaluationBatch { level } => {
                out.push(0);
                push_u32(out, level);
            }
            Self::ExtensionOpeningPoint { level } => {
                out.push(1);
                push_u32(out, level);
            }
            Self::ExtensionOpeningClaimBatch { level } => {
                out.push(2);
                push_u32(out, level);
            }
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
            Self::FoldChallengeGroup { level, group } => {
                out.push(5);
                push_u32(out, level);
                push_u32(out, group);
            }
            Self::RingSwitchAlpha { level } => {
                out.push(6);
                push_u32(out, level);
            }
            Self::Tau0Point { level } => {
                out.push(7);
                push_u32(out, level);
            }
            Self::Tau1Point { level } => {
                out.push(8);
                push_u32(out, level);
            }
            Self::Stage1InterstageBatch { level, stage } => {
                out.push(9);
                push_u32(out, level);
                push_u32(out, stage);
            }
            Self::L2SubclaimBatch { level } => {
                out.push(10);
                push_u32(out, level);
            }
            Self::L2NormMerge { level } => {
                out.push(11);
                push_u32(out, level);
            }
            Self::L2VirtualBatch { level } => {
                out.push(12);
                push_u32(out, level);
            }
            Self::CompressionBinary { level } => {
                out.push(13);
                push_u32(out, level);
            }
            Self::Stage2Batch { level } => {
                out.push(14);
                push_u32(out, level);
            }
        }
    }
}

const CHALLENGE_ORDER_COMPARISON_CAP: GrindingUint = GrindingUint([0, 0, 0, 1]);

/// Exact cardinality of a challenge set used when pricing transcript grinding.
///
/// A field-backed order records the base modulus and extension degree, so its
/// cardinality is exactly `modulus^extension_degree`. A full-capacity order is
/// available for protocols whose challenge set has exactly a power-of-two
/// cardinality. The fixed-width internal comparison is capped at 2^192, which
/// exceeds every loss factor multiplied by Akita's 128-bit security target.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ChallengeFieldOrder {
    nominal_capacity_bits: u32,
    field_modulus: u128,
    extension_degree: u32,
    exact_cardinality: GrindingUint,
}

impl ChallengeFieldOrder {
    /// Build the exact extension-field order from a prime-field modulus.
    pub fn from_field(
        modulus_bits: u32,
        extension_degree: usize,
        field_modulus: u128,
    ) -> Result<Self, AkitaError> {
        let actual_modulus_bits = u128::BITS - field_modulus.leading_zeros();
        if field_modulus <= 1 || actual_modulus_bits != modulus_bits {
            return Err(AkitaError::InvalidSetup(
                "challenge modulus does not match its declared bit width".into(),
            ));
        }
        let extension_degree = u32::try_from(extension_degree).map_err(|_| {
            AkitaError::InvalidSetup("challenge extension degree exceeds u32".into())
        })?;
        if extension_degree == 0 || !extension_degree.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "challenge extension degree must be a nonzero power of two".into(),
            ));
        }
        let nominal_capacity_bits = modulus_bits
            .checked_mul(extension_degree)
            .ok_or_else(|| AkitaError::InvalidSetup("challenge capacity overflow".into()))?;
        let exact_cardinality = GrindingUint::pow_capped(
            GrindingUint::from_u128(field_modulus),
            extension_degree,
            CHALLENGE_ORDER_COMPARISON_CAP,
        );
        Ok(Self {
            nominal_capacity_bits,
            field_modulus,
            extension_degree,
            exact_cardinality,
        })
    }

    /// Build a challenge set whose cardinality is exactly `2^capacity_bits`.
    pub fn from_full_capacity(capacity_bits: u32) -> Result<Self, AkitaError> {
        if capacity_bits == 0 {
            return Err(AkitaError::InvalidSetup(
                "grinding nominal capacity must be nonzero".into(),
            ));
        }
        let exact_cardinality = if capacity_bits >= 192 {
            CHALLENGE_ORDER_COMPARISON_CAP
        } else {
            GrindingUint::power_of_two(capacity_bits)
        };
        Ok(Self {
            nominal_capacity_bits: capacity_bits,
            field_modulus: 0,
            extension_degree: 0,
            exact_cardinality,
        })
    }

    /// Nominal bit width used to identify the field extension.
    #[must_use]
    pub const fn nominal_capacity_bits(self) -> u32 {
        self.nominal_capacity_bits
    }

    /// Extension degree of the field-backed challenge set, or zero for an exact power of two.
    #[must_use]
    pub const fn extension_degree(self) -> u32 {
        self.extension_degree
    }

    /// Prime-field modulus of a field-backed challenge set, or zero for an exact power of two.
    #[must_use]
    pub const fn field_modulus(self) -> u128 {
        self.field_modulus
    }

    fn append_canonical_bytes(self, out: &mut Vec<u8>) {
        if let Some(base_modulus_bits) = self
            .nominal_capacity_bits
            .checked_div(self.extension_degree)
        {
            out.push(1);
            push_u32(out, base_modulus_bits);
            push_u32(out, self.extension_degree);
            out.extend_from_slice(&self.field_modulus.to_le_bytes());
        } else {
            out.push(0);
            push_u32(out, self.nominal_capacity_bits);
        }
    }

    fn satisfies_target(self, loss_factor: u64, grind_bits: u8) -> bool {
        let target =
            GrindingUint::from_u64_shifted(loss_factor, u32::from(TRANSCRIPT_SECURITY_BITS));
        self.exact_cardinality
            .cmp(&target.ceil_shift_right(u32::from(grind_bits)))
            .is_ge()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GrindingUint([u64; 4]);

impl GrindingUint {
    const ZERO: Self = Self([0; 4]);
    const ONE: Self = Self([1, 0, 0, 0]);

    fn from_u128(value: u128) -> Self {
        Self([value as u64, (value >> 64) as u64, 0, 0])
    }

    fn from_u64_shifted(value: u64, shift: u32) -> Self {
        let mut limbs = [0; 4];
        let word_shift = (shift / 64) as usize;
        let bit_shift = shift % 64;
        if word_shift >= limbs.len() {
            return Self(limbs);
        }
        limbs[word_shift] = value << bit_shift;
        if bit_shift != 0 && word_shift + 1 < limbs.len() {
            limbs[word_shift + 1] = value >> (64 - bit_shift);
        }
        Self(limbs)
    }

    fn power_of_two(bit: u32) -> Self {
        let mut limbs = [0; 4];
        let bit = bit as usize;
        limbs[bit / 64] = 1u64 << (bit % 64);
        Self(limbs)
    }

    fn cmp(self, other: &Self) -> std::cmp::Ordering {
        for index in (0..self.0.len()).rev() {
            match self.0[index].cmp(&other.0[index]) {
                std::cmp::Ordering::Equal => {}
                ordering => return ordering,
            }
        }
        std::cmp::Ordering::Equal
    }

    fn add_capped(self, other: Self, cap: Self) -> Self {
        let mut result = [0u64; 4];
        let mut carry = 0u128;
        for (index, value) in result.iter_mut().enumerate() {
            let sum = u128::from(self.0[index]) + u128::from(other.0[index]) + carry;
            *value = sum as u64;
            carry = sum >> 64;
        }
        if carry != 0 {
            cap
        } else {
            let result = Self(result);
            if result.cmp(&cap).is_ge() {
                cap
            } else {
                result
            }
        }
    }

    fn mul_capped(self, other: Self, cap: Self) -> Self {
        let mut product = Self::ZERO;
        let mut addend = self;
        for index in 0..256 {
            if other.0[index / 64] & (1u64 << (index % 64)) != 0 {
                product = product.add_capped(addend, cap);
                if product == cap {
                    return cap;
                }
            }
            if index != 255 {
                addend = addend.add_capped(addend, cap);
            }
        }
        product
    }

    fn pow_capped(mut base: Self, mut exponent: u32, cap: Self) -> Self {
        let mut result = Self::ONE;
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = result.mul_capped(base, cap);
                if result == cap {
                    return cap;
                }
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.mul_capped(base, cap);
            }
        }
        result
    }

    fn ceil_shift_right(self, shift: u32) -> Self {
        if shift == 0 {
            return self;
        }
        let word_shift = (shift / 64) as usize;
        let bit_shift = shift % 64;
        let mut result = [0u64; 4];
        if word_shift < self.0.len() {
            for (destination, output) in result
                .iter_mut()
                .enumerate()
                .take(self.0.len() - word_shift)
            {
                let source = destination + word_shift;
                *output = self.0[source] >> bit_shift;
                if bit_shift != 0 && source + 1 < self.0.len() {
                    *output |= self.0[source + 1] << (64 - bit_shift);
                }
            }
        }
        let mut rounded = Self(result);
        let has_remainder = (0..word_shift.min(self.0.len())).any(|index| self.0[index] != 0)
            || (bit_shift != 0
                && word_shift < self.0.len()
                && self.0[word_shift] & ((1u64 << bit_shift) - 1) != 0);
        if has_remainder {
            rounded = rounded.add_capped(Self::ONE, CHALLENGE_ORDER_COMPARISON_CAP);
        }
        rounded
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

    /// Construct one proof-of-work site priced against an exact challenge order.
    pub fn proof_of_work(
        site: GrindingSite,
        loss_factor: u64,
        challenge_order: ChallengeFieldOrder,
    ) -> Result<Self, AkitaError> {
        if !matches!(site.kind(), GrindingQueryKind::ProofOfWork) {
            return Err(AkitaError::InvalidSetup(
                "special grinding sites cannot be proof-of-work runs".into(),
            ));
        }
        let grind_bits = grind_bits_for_loss(loss_factor, challenge_order)?;
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

    /// Construct one zero-width group root and indexed-coordinate audit run.
    pub fn fold_challenge_group(
        level: u32,
        group: u32,
        coordinate_count: u64,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            site: GrindingSite::FoldChallengeGroup { level, group },
            loss_factor: 0,
            grind_bits: 0,
            nonce_bits: 0,
            multiplicity: coordinate_count.checked_add(1).ok_or_else(|| {
                AkitaError::InvalidSetup("fold challenge query count overflow".into())
            })?,
        })
    }

    /// Number of independently indexed coordinates in a fold challenge group.
    #[must_use]
    pub const fn fold_coordinate_count(self) -> Option<u64> {
        match self.kind() {
            GrindingQueryKind::FoldChallengeGroup => self.multiplicity.checked_sub(1),
            _ => None,
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
            GrindingQueryKind::FoldChallengeGroup
                if self.loss_factor == 0
                    && self.grind_bits == 0
                    && self.nonce_bits == 0
                    && self.multiplicity > 1 => {}
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "grinding run kind and site do not match".into(),
                ));
            }
        }
        Ok(())
    }

    fn append_canonical_bytes(self, out: &mut Vec<u8>) {
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
    challenge_order: ChallengeFieldOrder,
    total_nonce_bits: usize,
    native_nonce_max_bytes: usize,
    expanded_query_count: u64,
}

/// Aggregate transcript-grinding cost used while pricing planner candidates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TranscriptGrindingCost {
    /// Semantic nonce widths used for range checks and security accounting.
    pub total_nonce_bits: usize,
    /// Maximum canonical native bytes emitted by the independently encoded nonces.
    pub native_nonce_max_bytes: usize,
    /// Number of logical transcript queries after expanding compact runs.
    pub expanded_query_count: u64,
}

#[path = "transcript_grinding/sink.rs"]
mod sink;
pub(crate) use sink::{GrindingPlanSink, SumcheckRoundBatch};

pub(crate) struct GrindingPlanAccumulator {
    challenge_order: ChallengeFieldOrder,
    run_count: u32,
    total_nonce_bits: usize,
    native_nonce_max_bytes: usize,
    expanded_query_count: u64,
}

impl GrindingPlanAccumulator {
    pub(crate) fn new(challenge_order: ChallengeFieldOrder) -> Result<Self, AkitaError> {
        Ok(Self {
            challenge_order,
            run_count: 0,
            total_nonce_bits: 0,
            native_nonce_max_bytes: 0,
            expanded_query_count: 0,
        })
    }

    fn push_repeated(&mut self, run: GrindingRun, repetitions: u32) -> Result<(), AkitaError> {
        self.run_count = self.run_count.checked_add(repetitions).ok_or_else(|| {
            AkitaError::InvalidSetup("grinding plan run count exceeds u32".into())
        })?;
        run.validate()?;
        if run.kind() == GrindingQueryKind::ProofOfWork
            && run.grind_bits != grind_bits_for_loss(run.loss_factor, self.challenge_order)?
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
            .ok_or_else(|| AkitaError::InvalidSetup("grinding run bit count overflow".into()))?;
        let repeated_bits = run_bits
            .checked_mul(repetitions as usize)
            .ok_or_else(|| AkitaError::InvalidSetup("grinding run bit count overflow".into()))?;
        let repeated_queries = run
            .multiplicity
            .checked_mul(u64::from(repetitions))
            .ok_or_else(|| AkitaError::InvalidSetup("grinding query count overflow".into()))?;
        self.total_nonce_bits = self
            .total_nonce_bits
            .checked_add(repeated_bits)
            .ok_or_else(|| AkitaError::InvalidSetup("grinding plan bit count overflow".into()))?;
        if run.nonce_bits != 0 {
            let run_bytes = native_nonce_max_bytes(run.nonce_bits)
                .checked_mul(multiplicity)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("native grinding nonce byte count overflow".into())
                })?;
            let repeated_bytes = run_bytes.checked_mul(repetitions as usize).ok_or_else(|| {
                AkitaError::InvalidSetup("native grinding nonce byte count overflow".into())
            })?;
            self.native_nonce_max_bytes = self
                .native_nonce_max_bytes
                .checked_add(repeated_bytes)
                .ok_or_else(|| {
                AkitaError::InvalidSetup("native grinding nonce byte count overflow".into())
            })?;
        }
        self.expanded_query_count = self
            .expanded_query_count
            .checked_add(repeated_queries)
            .ok_or_else(|| AkitaError::InvalidSetup("grinding query count overflow".into()))?;
        Ok(())
    }

    pub(crate) const fn cost(&self) -> TranscriptGrindingCost {
        TranscriptGrindingCost {
            total_nonce_bits: self.total_nonce_bits,
            native_nonce_max_bytes: self.native_nonce_max_bytes,
            expanded_query_count: self.expanded_query_count,
        }
    }
}

#[path = "transcript_grinding/native_replay.rs"]
mod native_replay;
pub use native_replay::{
    NativeGrindingSumcheckProver, NativeGrindingSumcheckVerifier, NativeProofAcceptance,
    NativeProverGrinding, NativeVerifierGrinding,
};

impl GrindingPlan {
    /// Validate ordered runs and derive all aggregate counts once.
    pub fn new(
        runs: Vec<GrindingRun>,
        challenge_order: ChallengeFieldOrder,
    ) -> Result<Self, AkitaError> {
        let mut accumulator = GrindingPlanAccumulator::new(challenge_order)?;
        for &run in &runs {
            accumulator.push(run)?;
        }
        let native_nonce_max_bytes = accumulator.native_nonce_max_bytes;
        let cost = accumulator.cost();
        if cost.expanded_query_count >= TRANSCRIPT_GRINDING_QUERY_LIMIT {
            return Err(AkitaError::InvalidSetup(
                "grinding plan query count must be less than u32::MAX".into(),
            ));
        }
        Ok(Self {
            runs,
            challenge_order,
            total_nonce_bits: cost.total_nonce_bits,
            native_nonce_max_bytes,
            expanded_query_count: cost.expanded_query_count,
        })
    }

    /// Compact ordered runs in this validated plan.
    #[must_use]
    pub fn runs(&self) -> &[GrindingRun] {
        &self.runs
    }

    /// Nominal bit width of the challenge field order.
    #[must_use]
    pub const fn nominal_capacity_bits(&self) -> u32 {
        self.challenge_order.nominal_capacity_bits()
    }

    /// Exact challenge order used to price all proof-of-work runs.
    #[must_use]
    pub const fn challenge_order(&self) -> ChallengeFieldOrder {
        self.challenge_order
    }

    #[must_use]
    pub const fn total_nonce_bits(&self) -> usize {
        self.total_nonce_bits
    }

    /// Maximum bytes emitted by native inline proof-of-work and fold-response nonces.
    #[must_use]
    pub const fn native_nonce_max_bytes(&self) -> usize {
        self.native_nonce_max_bytes
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
        out.extend_from_slice(&active_grinding_policy_bytes());
        self.challenge_order.append_canonical_bytes(&mut out);
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

/// Nominal field capacity in bits used to identify an extension field.
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

/// Assign the least public proof-of-work target satisfying the exact challenge-set bound.
pub fn grind_bits_for_loss(
    loss_factor: u64,
    challenge_order: ChallengeFieldOrder,
) -> Result<u8, AkitaError> {
    if loss_factor == 0 {
        return Err(AkitaError::InvalidSetup(
            "proof-of-work loss factor must be nonzero".into(),
        ));
    }
    for grind_bits in 0..=MAX_GRINDING_BITS {
        if challenge_order.satisfies_target(loss_factor, grind_bits) {
            return Ok(grind_bits);
        }
    }
    Err(AkitaError::InvalidSetup(format!(
        "grinding target exceeds supported maximum {MAX_GRINDING_BITS}"
    )))
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
