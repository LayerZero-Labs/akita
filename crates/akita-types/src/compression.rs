//! Canonical compressed-commitment plans and compact witnesses.
//!
//! These types deliberately name no protocol role, group, schedule, or claim.
//! They are the checked arithmetic boundary shared by planning, execution, and
//! verification.

use crate::field_modulus;
use crate::sis::compression::{min_compression_secure_rank, COMPRESSION_SIS_COEFF_LINF_BOUND};
use crate::sis::{SisModulusProfileId, DEFAULT_SIS_SECURITY_POLICY};
use akita_error::{checked, AkitaError};
use jolt_field::{CanonicalEncoding, Field};

mod chain;

pub use chain::CompressionChainPlan;

/// Maximum complete B/D image accepted by the compression protocol.
pub const MAX_COMPRESSION_INPUT_BYTES: usize = 8 * 1024;

/// Exact terminal payload target for the current compression ladder.
pub const COMPRESSION_TARGET_BYTES: usize = 128;

/// Exact number of maps in every compression chain.
pub const COMPRESSION_MAP_COUNT: usize = 2;

/// Schedule-bound encoding of one fold level's public B/D images.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum CommitmentPayloadMode {
    /// Prove the two-map compression relation and transmit its 128-byte terminal payload.
    #[default]
    Compressed,
    /// Omit compression witnesses/rows and transmit the native B/D image.
    Raw,
}

impl CommitmentPayloadMode {
    /// Stable descriptor tag bound by the effective schedule digest.
    pub const fn tag(self) -> u8 {
        match self {
            Self::Compressed => 1,
            Self::Raw => 2,
        }
    }

    /// Whether this level proves and transmits compressed B/D images.
    #[must_use]
    pub const fn is_compressed(self) -> bool {
        matches!(self, Self::Compressed)
    }
}

/// Monotone planner phase for recursive commitment payloads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CommitmentPayloadPhase {
    /// The planner may keep compressing or begin the raw suffix.
    #[default]
    CompressedPrefix,
    /// Compression cannot resume after this point.
    RawSuffix,
}

impl CommitmentPayloadPhase {
    /// Payload modes admitted by the protocol at this schedule state.
    #[must_use]
    pub const fn candidate_modes(
        self,
        absolute_fold_level: usize,
        consumes_setup_prefix: bool,
    ) -> &'static [CommitmentPayloadMode] {
        if absolute_fold_level < 2 || consumes_setup_prefix {
            &[CommitmentPayloadMode::Compressed]
        } else {
            match self {
                Self::CompressedPrefix => &[
                    CommitmentPayloadMode::Compressed,
                    CommitmentPayloadMode::Raw,
                ],
                Self::RawSuffix => &[CommitmentPayloadMode::Raw],
            }
        }
    }

    /// Advance the monotone phase after selecting one payload mode.
    #[must_use]
    pub const fn after(self, mode: CommitmentPayloadMode) -> Self {
        if mode.is_compressed() {
            self
        } else {
            Self::RawSuffix
        }
    }
}

/// Checked wire and transcript geometry for one native B/D source image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentPayloadGeometry {
    source_coefficients: usize,
    transmitted_coefficients: usize,
    transcript_ring_dimension: usize,
}

impl CommitmentPayloadGeometry {
    /// Derive the canonical payload geometry selected by one schedule mode.
    pub(crate) fn for_mode(
        mode: CommitmentPayloadMode,
        profile: SisModulusProfileId,
        source_rows: usize,
        source_ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        let source_coefficients = source_rows
            .checked_mul(source_ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("commitment payload shape overflow".into()))?;
        let plan = mode
            .is_compressed()
            .then(|| CompressionChainPlan::for_complete_source(profile, source_coefficients))
            .transpose()?;
        Self::new(source_rows, source_ring_dimension, plan.as_ref())
    }

    /// Derive payload geometry from native source rows and an optional compression plan.
    pub(crate) fn new(
        source_rows: usize,
        source_ring_dimension: usize,
        compression_plan: Option<&CompressionChainPlan>,
    ) -> Result<Self, AkitaError> {
        let source_coefficients = source_rows
            .checked_mul(source_ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("commitment payload shape overflow".into()))?;
        let (transmitted_coefficients, transcript_ring_dimension) =
            if let Some(plan) = compression_plan {
                if plan.source_coefficients() != source_coefficients {
                    return Err(AkitaError::InvalidSetup(
                        "commitment payload source disagrees with compression plan".into(),
                    ));
                }
                (
                    plan.terminal_coefficients(),
                    plan.maps()
                        .last()
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "commitment compression plan has no terminal map".into(),
                            )
                        })?
                        .ring_dimension(),
                )
            } else {
                (source_coefficients, source_ring_dimension)
            };
        Ok(Self {
            source_coefficients,
            transmitted_coefficients,
            transcript_ring_dimension,
        })
    }

    /// Native source coefficient count before compression.
    #[must_use]
    pub const fn source_coefficients(self) -> usize {
        self.source_coefficients
    }

    /// Exact coefficient count carried by the proof payload.
    #[must_use]
    pub const fn transmitted_coefficients(self) -> usize {
        self.transmitted_coefficients
    }

    /// Ring dimension used to absorb and interpret the transmitted payload.
    #[must_use]
    pub const fn transcript_ring_dimension(self) -> usize {
        self.transcript_ring_dimension
    }

    /// Number of transmitted rows at the transcript ring dimension.
    pub fn transmitted_rows(self) -> Result<usize, AkitaError> {
        if self.transcript_ring_dimension == 0
            || !self
                .transmitted_coefficients
                .is_multiple_of(self.transcript_ring_dimension)
        {
            return Err(AkitaError::InvalidSetup(
                "commitment payload does not align with its transcript ring dimension".into(),
            ));
        }
        Ok(self.transmitted_coefficients / self.transcript_ring_dimension)
    }
}

/// Stable identity of the commitment-compression protocol.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompressionPolicyId {
    /// Two-map exact monotone cutover with a minimum two-fold compressed prefix.
    #[default]
    NegativeBinaryTwoMapExactMonotoneCutover8KiBV3,
}

impl CompressionPolicyId {
    /// Stable descriptor tag for this policy.
    pub const fn tag(self) -> u8 {
        match self {
            Self::NegativeBinaryTwoMapExactMonotoneCutover8KiBV3 => 3,
        }
    }

    /// Parse the stable descriptor tag.
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            3 => Some(Self::NegativeBinaryTwoMapExactMonotoneCutover8KiBV3),
            _ => None,
        }
    }

    /// Descriptive policy name used in reports and generated metadata.
    pub const fn name(self) -> &'static str {
        match self {
            Self::NegativeBinaryTwoMapExactMonotoneCutover8KiBV3 => {
                "NegativeBinaryTwoMapExactMonotoneCutover8KiBV3"
            }
        }
    }
}

/// The only compression policy supported by this protocol epoch.
pub const COMPRESSION_POLICY: CompressionPolicyId =
    CompressionPolicyId::NegativeBinaryTwoMapExactMonotoneCutover8KiBV3;

/// The two compression-only ring dimensions for one modulus profile.
///
/// These dimensions are not A/B/D commitment-matrix dimensions. In
/// particular, the q128 ladder deliberately uses D=16 and D=8 while every
/// commitment matrix is admitted only at D>=64.
#[must_use]
pub const fn compression_ring_dimensions(profile: SisModulusProfileId) -> [usize; 2] {
    match profile {
        SisModulusProfileId::Q128OffsetA7F7 => [16, 8],
        SisModulusProfileId::Q64Offset59 => [32, 16],
        SisModulusProfileId::Q32Offset99 => [64, 32],
    }
}

const fn profile_field_bits(profile: SisModulusProfileId) -> usize {
    profile.field_bits() as usize
}

const fn profile_field_bytes(profile: SisModulusProfileId) -> usize {
    profile_field_bits(profile).div_ceil(8)
}

/// One checked rank-one negative-binary compression map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressionMapPlan {
    modulus_profile: SisModulusProfileId,
    input_coefficients: usize,
    ring_dimension: usize,
    input_width: usize,
    output_rank: usize,
    output_coefficients: usize,
    real_digit_count: usize,
    padded_digit_count: usize,
}

impl CompressionMapPlan {
    /// Validate one map against exact digit geometry and the compression SIS
    /// authority.
    pub fn new(
        modulus_profile: SisModulusProfileId,
        input_coefficients: usize,
        ring_dimension: usize,
        output_rank: usize,
    ) -> Result<Self, AkitaError> {
        if input_coefficients == 0 {
            return Err(AkitaError::InvalidInput(
                "compression map input must be nonempty".into(),
            ));
        }
        if output_rank != 1 {
            return Err(AkitaError::InvalidSetup(format!(
                "compression map must be rank one, got rank {output_rank}"
            )));
        }
        if !compression_ring_dimensions(modulus_profile).contains(&ring_dimension) {
            return Err(AkitaError::InvalidSetup(format!(
                "compression ring dimension {ring_dimension} is not in the {:?} ladder for {modulus_profile:?}",
                compression_ring_dimensions(modulus_profile)
            )));
        }
        let field_bits = profile_field_bits(modulus_profile);
        let real_digit_count = input_coefficients
            .checked_mul(field_bits)
            .ok_or_else(|| AkitaError::InvalidSetup("compression digit length overflow".into()))?;
        let input_width = checked::div_ceil(real_digit_count, ring_dimension).ok_or_else(|| {
            AkitaError::InvalidSetup("compression input width divisor is zero".into())
        })?;
        let padded_digit_count = input_width.checked_mul(ring_dimension).ok_or_else(|| {
            AkitaError::InvalidSetup("compression digit capacity overflow".into())
        })?;
        let output_coefficients = output_rank
            .checked_mul(ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("compression output length overflow".into()))?;
        let secure_rank = min_compression_secure_rank(
            DEFAULT_SIS_SECURITY_POLICY,
            modulus_profile,
            u32::try_from(ring_dimension).map_err(|_| {
                AkitaError::InvalidSetup("compression ring dimension overflow".into())
            })?,
            COMPRESSION_SIS_COEFF_LINF_BOUND,
            u64::try_from(input_width)
                .map_err(|_| AkitaError::InvalidSetup("compression width overflow".into()))?,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "no compression SIS rank for profile={modulus_profile:?} d={ring_dimension} width={input_width}"
            ))
        })?;
        if secure_rank != output_rank {
            return Err(AkitaError::InvalidSetup(format!(
                "compression map rank {output_rank} disagrees with secure rank {secure_rank}"
            )));
        }
        Ok(Self {
            modulus_profile,
            input_coefficients,
            ring_dimension,
            input_width,
            output_rank,
            output_coefficients,
            real_digit_count,
            padded_digit_count,
        })
    }

    /// Exact modulus profile.
    #[must_use]
    pub fn modulus_profile(self) -> SisModulusProfileId {
        self.modulus_profile
    }

    /// Number of input image field coefficients.
    #[must_use]
    pub fn input_coefficients(self) -> usize {
        self.input_coefficients
    }

    /// Native compression ring dimension.
    #[must_use]
    pub fn ring_dimension(self) -> usize {
        self.ring_dimension
    }

    /// Number of input ring columns.
    #[must_use]
    pub fn input_width(self) -> usize {
        self.input_width
    }

    /// Output module rank.
    #[must_use]
    pub fn output_rank(self) -> usize {
        self.output_rank
    }

    /// Number of output image field coefficients.
    #[must_use]
    pub fn output_coefficients(self) -> usize {
        self.output_coefficients
    }

    /// Number of non-padding negative-binary digits.
    #[must_use]
    pub fn real_digit_count(self) -> usize {
        self.real_digit_count
    }

    /// Digit capacity after padding to a complete ring row.
    #[must_use]
    pub fn padded_digit_count(self) -> usize {
        self.padded_digit_count
    }

    /// Exact packed byte count for non-padding digits.
    #[must_use]
    pub fn packed_digit_bytes(self) -> usize {
        self.real_digit_count.div_ceil(8)
    }
}

/// Bit-packed negative-binary digits for one checked map.
///
/// Bit one encodes `-1`; bit zero encodes `0`. Padding through the final ring
/// row is implicit and always zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedNegativeBinary {
    map: CompressionMapPlan,
    bytes: Vec<u8>,
}

impl PackedNegativeBinary {
    /// Pack one complete map input in canonical bit-major order.
    pub fn from_coefficients<F: Field + CanonicalEncoding>(
        map: CompressionMapPlan,
        coefficients: &[F],
    ) -> Result<Self, AkitaError> {
        if coefficients.len() != map.input_coefficients() {
            return Err(AkitaError::InvalidSize {
                expected: map.input_coefficients(),
                actual: coefficients.len(),
            });
        }
        let modulus = field_modulus::<F>()?;
        if !map.modulus_profile().matches_modulus(modulus) {
            return Err(AkitaError::InvalidSetup(
                "compression map profile does not match the field modulus".into(),
            ));
        }
        let field_bits = usize::try_from(F::MODULUS_BITS)
            .map_err(|_| AkitaError::InvalidSetup("field bit length overflow".into()))?;
        let mut bytes = vec![0u8; map.packed_digit_bytes()];
        for (coefficient_index, coefficient) in coefficients.iter().enumerate() {
            let canonical = coefficient.to_u128_checked().ok_or_else(|| {
                AkitaError::InvalidInput("compression coefficient does not fit in u128".into())
            })?;
            let magnitude = if canonical == 0 {
                0
            } else {
                modulus.checked_sub(canonical).ok_or_else(|| {
                    AkitaError::InvalidInput("noncanonical compression coefficient".into())
                })?
            };
            for bit in 0..field_bits {
                if (magnitude >> bit) & 1 == 1 {
                    let linear = bit
                        .checked_mul(coefficients.len())
                        .and_then(|base| base.checked_add(coefficient_index))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("compression digit index overflow".into())
                        })?;
                    bytes[linear / 8] |= 1 << (linear % 8);
                }
            }
        }
        Self::from_bytes(map, bytes)
    }

    /// Validate an exact packed representation.
    pub fn from_bytes(map: CompressionMapPlan, bytes: Vec<u8>) -> Result<Self, AkitaError> {
        if bytes.len() != map.packed_digit_bytes() {
            return Err(AkitaError::InvalidSize {
                expected: map.packed_digit_bytes(),
                actual: bytes.len(),
            });
        }
        let used_in_last = map.real_digit_count() % 8;
        if used_in_last != 0 {
            let padding_mask = !((1u8 << used_in_last) - 1);
            if bytes.last().is_some_and(|last| last & padding_mask != 0) {
                return Err(AkitaError::InvalidInput(
                    "compression packed digits have nonzero padding bits".into(),
                ));
            }
        }
        Ok(Self { map, bytes })
    }

    /// Checked map carried by this digit vector.
    #[must_use]
    pub fn map(&self) -> CompressionMapPlan {
        self.map
    }

    /// Exact packed bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Expand into typed `{-1,0}` ring rows for one bounded kernel call.
    pub fn expand_rows<const D: usize>(&self) -> Result<Vec<[i8; D]>, AkitaError> {
        if self.map.ring_dimension() != D {
            return Err(AkitaError::InvalidSetup(format!(
                "compression packed digits require D={}, got D={D}",
                self.map.ring_dimension()
            )));
        }
        let mut rows = vec![[0i8; D]; self.map.input_width()];
        for linear in 0..self.map.real_digit_count() {
            if self.bytes[linear / 8] >> (linear % 8) & 1 == 1 {
                rows[linear / D][linear % D] = -1;
            }
        }
        Ok(rows)
    }

    /// Canonically recompose the packed digits into the map input image.
    pub fn recompose<F: Field + CanonicalEncoding>(&self) -> Result<Vec<F>, AkitaError> {
        if !self
            .map
            .modulus_profile()
            .matches_modulus(field_modulus::<F>()?)
        {
            return Err(AkitaError::InvalidSetup(
                "compression map profile does not match the field modulus".into(),
            ));
        }
        let field_bits = usize::try_from(F::MODULUS_BITS)
            .map_err(|_| AkitaError::InvalidSetup("field bit length overflow".into()))?;
        let mut output = vec![F::zero(); self.map.input_coefficients()];
        let output_len = output.len();
        let mut power = F::one();
        for bit in 0..field_bits {
            for (coefficient_index, coefficient) in output.iter_mut().enumerate() {
                let linear = bit
                    .checked_mul(output_len)
                    .and_then(|base| base.checked_add(coefficient_index))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("compression digit index overflow".into())
                    })?;
                if self.bytes[linear / 8] >> (linear % 8) & 1 == 1 {
                    *coefficient -= power;
                }
            }
            power += power;
        }
        Ok(output)
    }
}

/// Persistent packed stage witness for one checked chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressionChainWitness {
    plan: CompressionChainPlan,
    stages: Vec<PackedNegativeBinary>,
}

impl CompressionChainWitness {
    /// Validate exact stage count and map correspondence.
    pub fn new(
        plan: CompressionChainPlan,
        stages: Vec<PackedNegativeBinary>,
    ) -> Result<Self, AkitaError> {
        if stages.len() != plan.maps().len()
            || stages
                .iter()
                .zip(plan.maps())
                .any(|(stage, map)| stage.map() != *map)
        {
            return Err(AkitaError::InvalidInput(
                "compression witness stages do not match the chain plan".into(),
            ));
        }
        Ok(Self { plan, stages })
    }

    /// Checked chain plan.
    #[must_use]
    pub fn plan(&self) -> &CompressionChainPlan {
        &self.plan
    }

    /// Ordered packed stage digits.
    #[must_use]
    pub fn stages(&self) -> &[PackedNegativeBinary] {
        &self.stages
    }

    /// Retained packed bytes.
    pub fn retained_bytes(&self) -> Result<usize, AkitaError> {
        self.stages.iter().try_fold(0usize, |total, stage| {
            total.checked_add(stage.bytes().len()).ok_or_else(|| {
                AkitaError::InvalidSetup("compression retained witness bytes overflow".into())
            })
        })
    }
}

/// Checked flat terminal payload for exactly one source chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressionTerminalPayload<F> {
    plan: CompressionChainPlan,
    coefficients: Vec<F>,
}

impl<F: Field + CanonicalEncoding> CompressionTerminalPayload<F> {
    /// Bind one flat payload to its already validated chain plan.
    pub fn new(plan: CompressionChainPlan, coefficients: Vec<F>) -> Result<Self, AkitaError> {
        if !plan
            .modulus_profile()
            .matches_modulus(field_modulus::<F>()?)
        {
            return Err(AkitaError::InvalidSetup(
                "compression terminal plan does not match the field modulus".into(),
            ));
        }
        if coefficients.len() != plan.terminal_coefficients() {
            return Err(AkitaError::InvalidSize {
                expected: plan.terminal_coefficients(),
                actual: coefficients.len(),
            });
        }
        Ok(Self { plan, coefficients })
    }

    /// Checked chain plan.
    #[must_use]
    pub fn plan(&self) -> &CompressionChainPlan {
        &self.plan
    }

    /// Flat terminal field coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[F] {
        &self.coefficients
    }

    /// Consume the terminal payload and return its coefficient storage.
    #[must_use]
    pub fn into_coefficients(self) -> Vec<F> {
        self.coefficients
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{One, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59, Ring, Zero};

    fn round_trip<F: Field + CanonicalEncoding>(profile: SisModulusProfileId) {
        let field_bytes = usize::try_from(F::MODULUS_BITS).unwrap().div_ceil(8);
        let plan = CompressionChainPlan::for_complete_source(profile, 1024 / field_bytes).unwrap();
        let values = (0..plan.source_coefficients())
            .map(|index| {
                if index % 7 == 0 {
                    F::zero()
                } else if index % 3 == 0 {
                    -F::from_u64(index as u64 + 1)
                } else {
                    F::from_u64(index as u64 * 17 + 5)
                }
            })
            .collect::<Vec<_>>();
        let packed = PackedNegativeBinary::from_coefficients(plan.maps()[0], &values).unwrap();
        assert_eq!(packed.recompose::<F>().unwrap(), values);
        assert_eq!(packed.bytes().len(), plan.maps()[0].real_digit_count() / 8);
        assert_eq!(
            plan.unpacked_witness_bytes().unwrap(),
            plan.packed_witness_bytes().unwrap() * 8
        );
    }

    #[test]
    fn q128_q64_q32_round_trip_in_bit_major_order() {
        round_trip::<Prime128OffsetA7F7>(SisModulusProfileId::Q128OffsetA7F7);
        round_trip::<Prime64Offset59>(SisModulusProfileId::Q64Offset59);
        round_trip::<Prime32Offset99>(SisModulusProfileId::Q32Offset99);
    }

    #[test]
    fn packing_is_bit_major_and_expansion_padding_is_zero() {
        type F = Prime32Offset99;
        let map = CompressionMapPlan::new(SisModulusProfileId::Q32Offset99, 3, 32, 1).unwrap();
        let values = [F::zero(), -F::one(), -F::from_u64(2)];
        let packed = PackedNegativeBinary::from_coefficients(map, &values).unwrap();
        assert_eq!(packed.bytes()[0] & 0b0011_1111, 0b0010_0010);
        let rows = packed.expand_rows::<32>().unwrap();
        assert!(rows
            .as_flattened()
            .iter()
            .take(map.real_digit_count())
            .all(|digit| matches!(*digit, -1 | 0)));
        assert!(rows
            .as_flattened()
            .iter()
            .skip(map.real_digit_count())
            .all(|digit| *digit == 0));
    }

    #[test]
    fn packed_length_padding_and_typed_dimension_are_checked() {
        let map = CompressionMapPlan::new(SisModulusProfileId::Q32Offset99, 1, 32, 1).unwrap();
        assert!(PackedNegativeBinary::from_bytes(map, vec![]).is_err());
        assert!(PackedNegativeBinary::from_bytes(map, vec![0; 5]).is_err());
        let packed = PackedNegativeBinary::from_bytes(map, vec![0; 4]).unwrap();
        assert!(packed.expand_rows::<64>().is_err());
    }

    #[test]
    fn ladder_geometry_and_complete_image_bound_are_checked() {
        for (profile, field_bytes, dimensions) in [
            (SisModulusProfileId::Q128OffsetA7F7, 16, [16, 8]),
            (SisModulusProfileId::Q64Offset59, 8, [32, 16]),
            (SisModulusProfileId::Q32Offset99, 4, [64, 32]),
        ] {
            let plan = CompressionChainPlan::for_complete_source(
                profile,
                MAX_COMPRESSION_INPUT_BYTES / field_bytes,
            )
            .unwrap();
            assert_eq!(
                CompressionChainPlan::for_complete_source(profile, plan.source_coefficients())
                    .unwrap(),
                plan,
                "a warm canonical lookup must preserve the exact plan",
            );
            let adjacent_source = plan.source_coefficients() - 1;
            let adjacent =
                CompressionChainPlan::for_complete_source(profile, adjacent_source).unwrap();
            assert_eq!(
                CompressionChainPlan::for_complete_source(profile, adjacent_source).unwrap(),
                adjacent,
                "an adjacent warm lookup must preserve its own exact cache entry",
            );
            assert_eq!(
                plan.maps()
                    .iter()
                    .map(|map| map.ring_dimension())
                    .collect::<Vec<_>>(),
                dimensions
            );
            assert_eq!(plan.policy(), COMPRESSION_POLICY);
            assert_eq!(plan.source_bytes(), MAX_COMPRESSION_INPUT_BYTES);
            assert_eq!(plan.terminal_coefficients() * field_bytes, 128);
            assert_eq!(
                plan,
                CompressionChainPlan::new(
                    profile,
                    plan.source_coefficients(),
                    [plan.maps()[0], plan.maps()[1]],
                )
                .unwrap(),
                "canonical derivation must match checked reconstruction",
            );
            assert_eq!(
                plan.max_setup_field_elements().unwrap(),
                plan.maps()[0].padded_digit_count()
            );
            assert!(CompressionChainPlan::try_for_complete_source(
                profile,
                MAX_COMPRESSION_INPUT_BYTES / field_bytes + 1
            )
            .unwrap()
            .is_none());
            assert!(CompressionChainPlan::for_complete_source(
                profile,
                MAX_COMPRESSION_INPUT_BYTES / field_bytes + 1
            )
            .is_err());
            assert!(CompressionChainPlan::try_for_complete_source(profile, usize::MAX).is_err());
        }
    }

    #[test]
    fn payload_geometry_canonically_selects_wire_count_and_ring_dimension() {
        let profile = SisModulusProfileId::Q128OffsetA7F7;
        let compressed =
            CommitmentPayloadGeometry::for_mode(CommitmentPayloadMode::Compressed, profile, 4, 16)
                .unwrap();
        assert_eq!(compressed.source_coefficients(), 64);
        assert_eq!(compressed.transmitted_coefficients(), 8);
        assert_eq!(compressed.transcript_ring_dimension(), 8);
        assert_eq!(compressed.transmitted_rows().unwrap(), 1);

        let raw = CommitmentPayloadGeometry::for_mode(CommitmentPayloadMode::Raw, profile, 4, 16)
            .unwrap();
        assert_eq!(raw.source_coefficients(), 64);
        assert_eq!(raw.transmitted_coefficients(), 64);
        assert_eq!(raw.transcript_ring_dimension(), 16);
        assert_eq!(raw.transmitted_rows().unwrap(), 4);

        let plan = CompressionChainPlan::for_complete_source(profile, 64).unwrap();
        assert!(CommitmentPayloadGeometry::new(5, 16, Some(&plan)).is_err());
    }

    #[test]
    fn malformed_maps_chains_witnesses_and_payloads_reject() {
        let profile = SisModulusProfileId::Q128OffsetA7F7;
        assert!(CompressionMapPlan::new(profile, 0, 16, 1).is_err());
        assert!(CompressionMapPlan::new(profile, 64, 16, 2).is_err());
        assert!(CompressionMapPlan::new(profile, 10_000, 8, 1).is_err());
        assert!(CompressionMapPlan::new(profile, 64, 32, 1).is_err());

        let plan = CompressionChainPlan::for_complete_source(profile, 64).unwrap();
        let wrong_continuation =
            CompressionMapPlan::new(profile, 15, plan.maps()[1].ring_dimension(), 1).unwrap();
        assert!(
            CompressionChainPlan::new(profile, 64, [plan.maps()[0], wrong_continuation]).is_err()
        );
        assert!(CompressionChainWitness::new(plan.clone(), Vec::new()).is_err());
        assert!(CompressionTerminalPayload::<Prime128OffsetA7F7>::new(
            plan.clone(),
            vec![Prime128OffsetA7F7::zero(); plan.terminal_coefficients() - 1]
        )
        .is_err());
    }
}
