//! Canonical compression-chain derivation.

use std::sync::OnceLock;

use super::*;
use crate::sis::compression::compression_sis_cell;

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
// A plan names its exact modulus profile, so the two 128-bit profiles cannot
// share cells even though their field byte length and key domain coincide.
static Q128_OFFSET275_COMPLETE_SOURCE_PLANS: [OnceLock<CompressionChainPlan>;
    Q128_COMPLETE_SOURCE_COUNT] = [const { OnceLock::new() }; Q128_COMPLETE_SOURCE_COUNT];
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
        SisModulusProfileId::Q128Offset275 => Q128_OFFSET275_COMPLETE_SOURCE_PLANS.get(index),
        SisModulusProfileId::Q64Offset59 => Q64_COMPLETE_SOURCE_PLANS.get(index),
        SisModulusProfileId::Q32Offset99 => Q32_COMPLETE_SOURCE_PLANS.get(index),
    }
}

fn image_coefficients(
    source_rows: usize,
    source_ring_dimension: usize,
) -> Result<usize, AkitaError> {
    source_rows
        .checked_mul(source_ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("compression source shape overflow".into()))
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
        let source_bytes = source_coefficients
            .checked_mul(profile_field_bytes(modulus_profile))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression source byte length overflow".into())
            })?;
        if source_bytes > MAX_COMPRESSION_INPUT_BYTES {
            return Err(AkitaError::InvalidInput(format!(
                "compression source is {source_bytes} bytes, exceeding the {MAX_COMPRESSION_INPUT_BYTES}-byte maximum"
            )));
        }
        Self::certified(
            COMPRESSION_POLICY,
            modulus_profile,
            source_coefficients,
            maps,
        )
    }

    /// Validate a chain of `policy` against its maps' SIS certification
    /// alone; the caller enforces the policy's source cap.
    fn certified(
        policy: CompressionPolicyId,
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
            policy,
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

    /// Derive the canonical chain for one native B or D image.
    ///
    /// Every chain derived from a B or D image goes through this constructor
    /// or [`Self::try_for_image`], so the ladder's source-dimension rule is
    /// enforced in one place.
    pub fn for_image(
        modulus_profile: SisModulusProfileId,
        source_rows: usize,
        source_ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if !compression_source_ring_dimension_supported(modulus_profile, source_ring_dimension) {
            return Err(AkitaError::InvalidSetup(format!(
                "compression source ring dimension {source_ring_dimension} is not a multiple of the first ladder dimension {}",
                compression_ring_dimensions(modulus_profile)[0]
            )));
        }
        Self::for_complete_source(
            modulus_profile,
            image_coefficients(source_rows, source_ring_dimension)?,
        )
    }

    /// Candidate-aware image derivation. `None` means the image cannot be
    /// compressed: its source exceeds the protocol cap or its ring dimension
    /// cannot feed the ladder. Malformed shapes remain errors.
    pub fn try_for_image(
        modulus_profile: SisModulusProfileId,
        source_rows: usize,
        source_ring_dimension: usize,
    ) -> Result<Option<Self>, AkitaError> {
        if !compression_source_ring_dimension_supported(modulus_profile, source_ring_dimension) {
            return Ok(None);
        }
        Self::try_for_complete_source(
            modulus_profile,
            image_coefficients(source_rows, source_ring_dimension)?,
        )
    }

    /// Candidate-aware derivation. `None` means only that the source exceeds
    /// the protocol cap; malformed and uncertified shapes remain errors.
    pub fn try_for_complete_source(
        modulus_profile: SisModulusProfileId,
        source_coefficients: usize,
    ) -> Result<Option<Self>, AkitaError> {
        Self::derive_complete_source(modulus_profile, source_coefficients)
    }

    /// The chain of `source_coefficients` under the maps' SIS certification
    /// alone, without the 8 KiB source cap of [`COMPRESSION_POLICY`]: the
    /// same two rank-one negative-binary maps, the same 128-byte terminal,
    /// under [`CompressionPolicyId::NegativeBinaryTwoMapSisBoundV1`].
    /// The cap bounds the B/D images of clear schedules; Whiteout's split
    /// images (`specs/whiteout-akita.md` §5.4) chunk at
    /// [`Self::max_certified_source_coefficients`] instead, the paper's
    /// `FCHAIN_MAX_IMAGE`.
    ///
    /// # Errors
    ///
    /// [`AkitaError`] if no certified chain compresses the source.
    pub fn for_certified_source(
        modulus_profile: SisModulusProfileId,
        source_coefficients: usize,
    ) -> Result<Self, AkitaError> {
        if source_coefficients == 0 {
            return Err(AkitaError::InvalidInput(
                "compression source must be nonempty".into(),
            ));
        }
        Self::derive_maps(
            CompressionPolicyId::NegativeBinaryTwoMapSisBoundV1,
            modulus_profile,
            source_coefficients,
        )
    }

    /// The largest source [`Self::for_certified_source`] compresses: the
    /// first map's rank-one SIS width, in source coefficients
    /// (`⌊w₁·d₁/field_bits⌋`, 15,115 at q128). The second map's input is the
    /// first map's fixed `d₁`-coefficient output, so it does not bound the
    /// source.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidSetup`] if the profile's first map has no
    /// certified cell or its chain fails to derive.
    pub fn max_certified_source_coefficients(
        modulus_profile: SisModulusProfileId,
    ) -> Result<usize, AkitaError> {
        let [first_dimension, _] = compression_ring_dimensions(modulus_profile);
        let cell = u32::try_from(first_dimension)
            .ok()
            .and_then(|dimension| {
                compression_sis_cell(modulus_profile, dimension, COMPRESSION_SIS_COEFF_LINF_BOUND)
            })
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression first map has no SIS cell".into())
            })?;
        let max = usize::try_from(cell.sis_max_width)
            .ok()
            .and_then(|width| width.checked_mul(first_dimension))
            .map(|digits| digits / profile_field_bits(modulus_profile))
            .ok_or_else(|| AkitaError::InvalidSetup("compression width overflow".into()))?;
        Self::for_certified_source(modulus_profile, max)?;
        Ok(max)
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
        let plan = Self::derive_maps(COMPRESSION_POLICY, modulus_profile, source_coefficients)?;
        let _ = cache.set(plan.clone());
        Ok(Some(plan))
    }

    /// The canonical two-map ladder of a nonempty source, certified map by
    /// map, uncached and uncapped.
    fn derive_maps(
        policy: CompressionPolicyId,
        modulus_profile: SisModulusProfileId,
        source_coefficients: usize,
    ) -> Result<Self, AkitaError> {
        let field_bytes = profile_field_bytes(modulus_profile);
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
                return Self::certified(
                    policy,
                    modulus_profile,
                    source_coefficients,
                    [first, second],
                );
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
