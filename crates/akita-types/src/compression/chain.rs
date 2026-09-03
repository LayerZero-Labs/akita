//! Canonical compression-chain derivation.

use std::sync::OnceLock;

use super::*;

// The protocol cap makes the complete key domain small and finite. One cell
// per admitted nonzero source length gives exact, eviction-free reuse without
// attacker-sized allocation or cache state affecting derivation semantics.
const Q128_COMPLETE_SOURCE_COUNT: usize =
    MAX_COMPRESSION_INPUT_BYTES / profile_field_bytes(SisModulusProfileId::Q128OffsetA7F7);
const Q64_COMPLETE_SOURCE_COUNT: usize =
    MAX_COMPRESSION_INPUT_BYTES / profile_field_bytes(SisModulusProfileId::Q64Offset59);
const Q32_COMPLETE_SOURCE_COUNT: usize =
    MAX_COMPRESSION_INPUT_BYTES / profile_field_bytes(SisModulusProfileId::Q32Offset99);

static Q128_COMPLETE_SOURCE_PLANS: [OnceLock<CompressionChainPlan>; Q128_COMPLETE_SOURCE_COUNT] =
    [const { OnceLock::new() }; Q128_COMPLETE_SOURCE_COUNT];
static Q64_COMPLETE_SOURCE_PLANS: [OnceLock<CompressionChainPlan>; Q64_COMPLETE_SOURCE_COUNT] =
    [const { OnceLock::new() }; Q64_COMPLETE_SOURCE_COUNT];
static Q32_COMPLETE_SOURCE_PLANS: [OnceLock<CompressionChainPlan>; Q32_COMPLETE_SOURCE_COUNT] =
    [const { OnceLock::new() }; Q32_COMPLETE_SOURCE_COUNT];

fn complete_source_plan_cache(
    modulus_profile: SisModulusProfileId,
    source_coefficients: usize,
) -> Option<&'static OnceLock<CompressionChainPlan>> {
    let index = source_coefficients.checked_sub(1)?;
    match modulus_profile {
        SisModulusProfileId::Q128OffsetA7F7 => Q128_COMPLETE_SOURCE_PLANS.get(index),
        SisModulusProfileId::Q64Offset59 => Q64_COMPLETE_SOURCE_PLANS.get(index),
        SisModulusProfileId::Q32Offset99 => Q32_COMPLETE_SOURCE_PLANS.get(index),
    }
}

/// Complete checked compression plan for one flat source image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressionChainPlan {
    policy: CompressionPolicyId,
    modulus_profile: SisModulusProfileId,
    field_bits: usize,
    field_bytes: usize,
    source_coefficients: usize,
    source_bytes: usize,
    maps: [CompressionMapPlan; COMPRESSION_MAP_COUNT],
}

impl CompressionChainPlan {
    /// Validate an already derived canonical chain.
    pub fn new(
        modulus_profile: SisModulusProfileId,
        source_coefficients: usize,
        maps: [CompressionMapPlan; COMPRESSION_MAP_COUNT],
    ) -> Result<Self, AkitaError> {
        let field_bits = profile_field_bits(modulus_profile);
        let field_bytes = profile_field_bytes(modulus_profile);
        let source_bytes = source_coefficients
            .checked_mul(field_bytes)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression source byte length overflow".into())
            })?;
        if source_coefficients == 0 {
            return Err(AkitaError::InvalidInput(
                "compression source must be nonempty".into(),
            ));
        }
        if source_bytes > MAX_COMPRESSION_INPUT_BYTES {
            return Err(AkitaError::InvalidInput(format!(
                "compression source is {source_bytes} bytes, exceeding the {MAX_COMPRESSION_INPUT_BYTES}-byte maximum"
            )));
        }
        let mut expected_input = source_coefficients;
        let mut previous_dimension: Option<usize> = None;
        for map in &maps {
            if map.modulus_profile() != modulus_profile {
                return Err(AkitaError::InvalidSetup(
                    "compression map profile disagrees with its chain".into(),
                ));
            }
            if map.input_coefficients() != expected_input {
                return Err(AkitaError::InvalidSetup(
                    "compression map input does not continue the preceding image".into(),
                ));
            }
            if let Some(previous) = previous_dimension {
                if previous.checked_div(2) != Some(map.ring_dimension()) {
                    return Err(AkitaError::InvalidSetup(
                        "compression ring dimensions must halve at each map".into(),
                    ));
                }
            }
            expected_input = map.output_coefficients();
            previous_dimension = Some(map.ring_dimension());
        }
        let terminal_bytes = expected_input.checked_mul(field_bytes).ok_or_else(|| {
            AkitaError::InvalidSetup("compression terminal byte length overflow".into())
        })?;
        if terminal_bytes != COMPRESSION_TARGET_BYTES {
            return Err(AkitaError::InvalidSetup(format!(
                "compression terminal is {terminal_bytes} bytes, expected {COMPRESSION_TARGET_BYTES}"
            )));
        }
        Ok(Self {
            policy: COMPRESSION_POLICY,
            modulus_profile,
            field_bits,
            field_bytes,
            source_coefficients,
            source_bytes,
            maps,
        })
    }

    /// Derive and validate the canonical complete-image chain.
    pub fn for_complete_source(
        modulus_profile: SisModulusProfileId,
        source_coefficients: usize,
    ) -> Result<Self, AkitaError> {
        Self::try_for_complete_source(modulus_profile, source_coefficients)?.ok_or_else(|| {
            let source_bytes =
                source_coefficients.saturating_mul(profile_field_bytes(modulus_profile));
            AkitaError::InvalidInput(format!(
                "compression source is {source_bytes} bytes, exceeding the {MAX_COMPRESSION_INPUT_BYTES}-byte maximum"
            ))
        })
    }

    /// Candidate-aware derivation. `None` means only that the source exceeds
    /// the protocol cap; malformed and uncertified shapes remain errors.
    pub fn try_for_complete_source(
        modulus_profile: SisModulusProfileId,
        source_coefficients: usize,
    ) -> Result<Option<Self>, AkitaError> {
        Self::derive_complete_source(modulus_profile, source_coefficients)
    }

    fn derive_complete_source(
        modulus_profile: SisModulusProfileId,
        source_coefficients: usize,
    ) -> Result<Option<Self>, AkitaError> {
        let field_bytes = profile_field_bytes(modulus_profile);
        let source_bytes = source_coefficients
            .checked_mul(field_bytes)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression source byte length overflow".into())
            })?;
        if source_coefficients == 0 {
            return Err(AkitaError::InvalidInput(
                "compression source must be nonempty".into(),
            ));
        }
        if source_bytes > MAX_COMPRESSION_INPUT_BYTES {
            return Ok(None);
        }
        let cache =
            complete_source_plan_cache(modulus_profile, source_coefficients).ok_or_else(|| {
                AkitaError::InvalidSetup("compression source cache domain mismatch".into())
            })?;
        if let Some(plan) = cache.get() {
            return Ok(Some(plan.clone()));
        }
        let mut maps = [None; COMPRESSION_MAP_COUNT];
        let mut input_coefficients = source_coefficients;
        for (map_count, ring_dimension) in compression_ring_dimensions(modulus_profile)
            .into_iter()
            .enumerate()
        {
            if map_count == COMPRESSION_MAP_COUNT {
                return Err(AkitaError::InvalidSetup(format!(
                    "compression chain exceeded {COMPRESSION_MAP_COUNT} maps"
                )));
            }
            let map =
                CompressionMapPlan::new(modulus_profile, input_coefficients, ring_dimension, 1)?;
            input_coefficients = map.output_coefficients();
            let output_bytes = input_coefficients.checked_mul(field_bytes).ok_or_else(|| {
                AkitaError::InvalidSetup("compression output byte length overflow".into())
            })?;
            maps[map_count] = Some(map);
            if output_bytes == COMPRESSION_TARGET_BYTES {
                let [Some(first), Some(second)] = maps else {
                    return Err(AkitaError::InvalidSetup(format!(
                        "compression chain must contain exactly {COMPRESSION_MAP_COUNT} maps"
                    )));
                };
                let plan = Self::new(modulus_profile, source_coefficients, [first, second])?;
                let _ = cache.set(plan.clone());
                return Ok(Some(plan));
            }
            if output_bytes < COMPRESSION_TARGET_BYTES {
                return Err(AkitaError::InvalidSetup(
                    "compression ladder undershot its terminal target".into(),
                ));
            }
        }
        Err(AkitaError::InvalidSetup(format!(
            "compression ladder did not reach {COMPRESSION_TARGET_BYTES} bytes"
        )))
    }

    /// Fixed protocol policy that determines this plan.
    #[must_use]
    pub fn policy(&self) -> CompressionPolicyId {
        self.policy
    }

    /// Exact modulus profile.
    #[must_use]
    pub fn modulus_profile(&self) -> SisModulusProfileId {
        self.modulus_profile
    }

    /// Canonical modulus bit width.
    #[must_use]
    pub fn field_bits(&self) -> usize {
        self.field_bits
    }

    /// Canonical field byte width.
    #[must_use]
    pub fn field_bytes(&self) -> usize {
        self.field_bytes
    }

    /// Complete source coefficient count.
    #[must_use]
    pub fn source_coefficients(&self) -> usize {
        self.source_coefficients
    }

    /// Canonical byte length of the complete source image.
    #[must_use]
    pub fn source_bytes(&self) -> usize {
        self.source_bytes
    }

    /// Ordered checked map plans.
    #[must_use]
    pub fn maps(&self) -> &[CompressionMapPlan] {
        &self.maps
    }

    /// Exact terminal coefficient count.
    #[must_use]
    pub fn terminal_coefficients(&self) -> usize {
        self.maps.last().map_or(0, |map| map.output_coefficients())
    }

    /// Total persistent packed stage-witness bytes.
    pub fn packed_witness_bytes(&self) -> Result<usize, AkitaError> {
        self.maps.iter().try_fold(0usize, |total, map| {
            total.checked_add(map.packed_digit_bytes()).ok_or_else(|| {
                AkitaError::InvalidSetup("compression packed witness bytes overflow".into())
            })
        })
    }

    /// Equivalent persistent bytes for an `i8` digit representation.
    pub fn unpacked_witness_bytes(&self) -> Result<usize, AkitaError> {
        self.maps.iter().try_fold(0usize, |total, map| {
            total.checked_add(map.real_digit_count()).ok_or_else(|| {
                AkitaError::InvalidSetup("compression unpacked witness bytes overflow".into())
            })
        })
    }

    /// Largest universal-setup matrix prefix required by any map in fields.
    pub fn max_setup_field_elements(&self) -> Result<usize, AkitaError> {
        self.maps.iter().try_fold(0usize, |maximum, map| {
            let fields = map
                .output_rank()
                .checked_mul(map.input_width())
                .and_then(|len| len.checked_mul(map.ring_dimension()))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("compression setup envelope overflow".into())
                })?;
            Ok(maximum.max(fields))
        })
    }
}
