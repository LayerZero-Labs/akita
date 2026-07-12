//! Checked commitment-compression chain geometry.

use akita_field::{AkitaError, CanonicalField};
use akita_serialization::DEFAULT_MAX_SEQUENCE_LEN;
use std::collections::BTreeMap;

use crate::dispatch::{protocol_dispatch_tier, slot_dim_supported_for_tier, ProtocolDispatchSlot};
use crate::sis::{
    num_digits_for_bound, rounded_up_collision_inf_norm, sis_table_key_for_linf_bound,
    AjtaiKeyParams, SisModulusFamily, SisTableKey, DEFAULT_SIS_SECURITY_BITS,
};
use crate::{LevelParams, OpeningClaimsLayout, RingRole, MAX_SETUP_MATRIX_FIELD_ELEMENTS};

pub(in crate::layout) mod semantics;

#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlphabet {
    NegativeBinary,
    OpeningBase { log_basis: u32 },
}

/// Canonical digit depth used to decompose one compression-map input.
pub fn compression_digit_depth(
    alphabet: CompressionAlphabet,
    field_bits: u32,
    level_log_basis: u32,
) -> Result<usize, AkitaError> {
    match alphabet {
        CompressionAlphabet::NegativeBinary => Ok(field_bits as usize),
        CompressionAlphabet::OpeningBase { log_basis } => {
            if log_basis == 0 || log_basis >= 128 || log_basis > level_log_basis {
                return Err(AkitaError::InvalidSetup(
                    "compression opening-base log_basis must be in 1..128 and no larger than the level base".into(),
                ));
            }
            Ok(num_digits_for_bound(field_bits, field_bits, log_basis))
        }
    }
}

#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionSourceId {
    CurrentOuter,
    PrecommittedOuter { index: usize },
    Opening,
}

#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionCatalogContext<'a> {
    CoGeneratedLevel { opening: &'a OpeningClaimsLayout },
    StandaloneCommitment { max_opening_log_basis: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CompressionCatalogPurpose {
    CoGenerated {
        relation_layout: crate::layout::relation::RelationLayout,
    },
    Standalone {
        max_opening_log_basis: u32,
        source_key: AjtaiKeyParams,
        field_modulus_minus_one: u128,
    },
}

#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionMapSpec {
    key: AjtaiKeyParams,
    alphabet: CompressionAlphabet,
}

impl CompressionMapSpec {
    #[must_use]
    pub fn new(key: AjtaiKeyParams, alphabet: CompressionAlphabet) -> Self {
        Self { key, alphabet }
    }
}

#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionChainSpec {
    source: CompressionSourceId,
    maps: Vec<CompressionMapSpec>,
}

impl CompressionChainSpec {
    #[must_use]
    pub fn new(source: CompressionSourceId, maps: Vec<CompressionMapSpec>) -> Self {
        Self { source, maps }
    }
}

#[allow(dead_code)] // Compiled catalog fields are consumed by later protocol slices.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CompiledCompressionMap {
    key: AjtaiKeyParams,
    alphabet: CompressionAlphabet,
    digit_depth: usize,
    input_coeffs: usize,
    output_coeffs: usize,
}

#[allow(dead_code)] // Compiled catalog fields are consumed by later protocol slices.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CompiledCompressionChain {
    source: CompressionSourceId,
    source_output_coeffs: usize,
    maps: Vec<CompiledCompressionMap>,
    payload_coeffs: usize,
}

/// One ordered map's replay facts for prover hint sizing and preparation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // Consumed by schedule replay after compression candidate generation lands.
pub(crate) struct CompressionMapHintFacts {
    pub(crate) source: CompressionSourceId,
    pub(crate) map_index: usize,
    pub(crate) alphabet: CompressionAlphabet,
    pub(crate) native_ring_dim: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) digit_depth: usize,
    pub(crate) input_coeffs: usize,
    pub(crate) output_coeffs: usize,
    pub(crate) prefix_ring_elements: usize,
    pub(crate) is_terminal: bool,
}

/// Catalog-wide replay projection derived only from checked compiled maps.
///
/// This is not a second planner or protocol authority. It packages the
/// performance and hint-shape projections that would otherwise be re-derived
/// independently by schedule replay, prepared NTT setup, and profile reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // Consumed by schedule replay after compression candidate generation lands.
pub struct CompressionCatalogProjection {
    maps: Vec<CompressionMapHintFacts>,
    ntt_requirements: Vec<(usize, usize)>,
    payload_coeffs_by_source: Vec<(CompressionSourceId, usize)>,
    first_map_alphabet_by_source: Vec<(CompressionSourceId, CompressionAlphabet)>,
    logical_setup_coeffs: usize,
    /// Largest flat field-coefficient prefix required by any single map.
    /// Per-dimension transformed-cache maxima live in `ntt_requirements`.
    max_flat_setup_prefix_coeffs: usize,
    coalesced_cache_field_coeffs: usize,
    descriptor_bytes: Vec<u8>,
}

impl CompressionCatalogProjection {
    #[must_use]
    pub fn ntt_requirements(&self) -> &[(usize, usize)] {
        &self.ntt_requirements
    }

    #[must_use]
    pub fn payload_coeffs(&self, source: CompressionSourceId) -> Option<usize> {
        self.payload_coeffs_by_source
            .iter()
            .find_map(|&(candidate, coeffs)| (candidate == source).then_some(coeffs))
    }

    #[must_use]
    pub fn logical_setup_coeffs(&self) -> usize {
        self.logical_setup_coeffs
    }

    #[must_use]
    pub fn max_flat_setup_prefix_coeffs(&self) -> usize {
        self.max_flat_setup_prefix_coeffs
    }

    #[must_use]
    pub fn descriptor_bytes(&self) -> &[u8] {
        &self.descriptor_bytes
    }
}

fn resolve_source_key(
    lp: &LevelParams,
    source: CompressionSourceId,
) -> Result<&AjtaiKeyParams, AkitaError> {
    match source {
        CompressionSourceId::CurrentOuter => Ok(&lp.b_key),
        CompressionSourceId::PrecommittedOuter { index } => lp
            .precommitted_groups
            .get(index)
            .map(|group| &group.b_key)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "compression precommitted source index is out of range".into(),
                )
            }),
        CompressionSourceId::Opening => Ok(&lp.d_key),
    }
}

#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedCompressionCatalog {
    chains: Vec<CompiledCompressionChain>,
    purpose: CompressionCatalogPurpose,
}

impl ValidatedCompressionCatalog {
    /// Project checked map facts for schedule replay, hints, NTT preparation,
    /// profile accounting, and future schedule-identity binding.
    #[allow(dead_code)] // Consumed by schedule replay after compression candidate generation lands.
    pub fn project_for_schedule(&self) -> Result<CompressionCatalogProjection, AkitaError> {
        let total_maps = self.chains.iter().try_fold(0usize, |total, chain| {
            total.checked_add(chain.maps.len()).ok_or_else(|| {
                AkitaError::InvalidSetup("compression projected map count overflow".into())
            })
        })?;
        if total_maps > DEFAULT_MAX_SEQUENCE_LEN {
            return Err(AkitaError::InvalidSetup(
                "compression projected map count exceeds sequence cap".into(),
            ));
        }
        let mut maps = Vec::with_capacity(total_maps);
        let mut ntt_by_d = BTreeMap::<usize, usize>::new();
        let mut payload_coeffs_by_source = Vec::with_capacity(self.chains.len());
        let mut first_map_alphabet_by_source = Vec::with_capacity(self.chains.len());
        let mut logical_setup_coeffs = 0usize;
        let mut max_flat_setup_prefix_coeffs = 0usize;

        for chain in &self.chains {
            payload_coeffs_by_source.push((chain.source, chain.payload_coeffs));
            let first_alphabet = chain.maps.first().map(|map| map.alphabet).ok_or_else(|| {
                AkitaError::InvalidSetup("compiled compression chain has no maps".into())
            })?;
            first_map_alphabet_by_source.push((chain.source, first_alphabet));
            for (map_index, map) in chain.maps.iter().enumerate() {
                let native_ring_dim = map.key.sis_table_key().ring_dimension as usize;
                let prefix_ring_elements = map
                    .key
                    .row_len()
                    .checked_mul(map.key.col_len())
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("compression projected NTT prefix overflow".into())
                    })?;
                let setup_coeffs = prefix_ring_elements
                    .checked_mul(native_ring_dim)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "compression projected setup footprint overflow".into(),
                        )
                    })?;
                logical_setup_coeffs =
                    logical_setup_coeffs
                        .checked_add(setup_coeffs)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "compression total logical setup footprint overflow".into(),
                            )
                        })?;
                max_flat_setup_prefix_coeffs = max_flat_setup_prefix_coeffs.max(setup_coeffs);
                ntt_by_d
                    .entry(native_ring_dim)
                    .and_modify(|prefix| *prefix = (*prefix).max(prefix_ring_elements))
                    .or_insert(prefix_ring_elements);
                maps.push(CompressionMapHintFacts {
                    source: chain.source,
                    map_index,
                    alphabet: map.alphabet,
                    native_ring_dim,
                    rows: map.key.row_len(),
                    cols: map.key.col_len(),
                    digit_depth: map.digit_depth,
                    input_coeffs: map.input_coeffs,
                    output_coeffs: map.output_coeffs,
                    prefix_ring_elements,
                    is_terminal: map_index + 1 == chain.maps.len(),
                });
            }
        }

        let ntt_requirements = ntt_by_d.into_iter().collect::<Vec<_>>();
        let coalesced_cache_field_coeffs = ntt_requirements.iter().try_fold(
            0usize,
            |total, &(ring_d, prefix_ring_elements)| {
                let field_coeffs = ring_d.checked_mul(prefix_ring_elements).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "compression coalesced cache footprint overflow".into(),
                    )
                })?;
                total.checked_add(field_coeffs).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "compression total coalesced cache footprint overflow".into(),
                    )
                })
            },
        )?;
        let mut descriptor_bytes = Vec::new();
        self.append_projection_descriptor_bytes(&mut descriptor_bytes)?;

        Ok(CompressionCatalogProjection {
            maps,
            ntt_requirements,
            payload_coeffs_by_source,
            first_map_alphabet_by_source,
            logical_setup_coeffs,
            max_flat_setup_prefix_coeffs,
            coalesced_cache_field_coeffs,
            descriptor_bytes,
        })
    }

    #[allow(dead_code)] // Reached through the dormant schedule projection.
    fn append_projection_descriptor_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AkitaError> {
        use crate::descriptor_bytes::{push_u128, push_u32, push_usize};

        bytes.extend_from_slice(b"AKITA-COMPRESSION-CATALOG-V1");
        bytes.push(semantics::BINARY_SUPPORT_DERIVATION_VERSION);
        ensure_projection_descriptor_len(bytes)?;
        match &self.purpose {
            CompressionCatalogPurpose::CoGenerated { .. } => bytes.push(0),
            CompressionCatalogPurpose::Standalone {
                max_opening_log_basis,
                source_key,
                field_modulus_minus_one,
            } => {
                bytes.push(1);
                push_u32(bytes, *max_opening_log_basis);
                source_key.append_descriptor_bytes(bytes);
                push_u128(bytes, *field_modulus_minus_one);
                ensure_projection_descriptor_len(bytes)?;
            }
        }
        push_usize(bytes, self.chains.len());
        for chain in &self.chains {
            append_source_descriptor_bytes(bytes, chain.source);
            push_usize(bytes, chain.source_output_coeffs);
            push_usize(bytes, chain.maps.len());
            for map in &chain.maps {
                map.key.append_descriptor_bytes(bytes);
                // The global Ajtai descriptor intentionally omits native D for existing
                // schedule identities. Compression maps are heterogeneous, so bind D
                // locally as part of this catalog descriptor.
                push_u32(bytes, map.key.sis_table_key().ring_dimension);
                match map.alphabet {
                    CompressionAlphabet::NegativeBinary => bytes.push(0),
                    CompressionAlphabet::OpeningBase { log_basis } => {
                        bytes.push(1);
                        push_u32(bytes, log_basis);
                    }
                }
                push_usize(bytes, map.digit_depth);
                push_usize(bytes, map.input_coeffs);
                push_usize(bytes, map.output_coeffs);
                ensure_projection_descriptor_len(bytes)?;
            }
            push_usize(bytes, chain.payload_coeffs);
            ensure_projection_descriptor_len(bytes)?;
        }
        Ok(())
    }

    #[allow(dead_code)] // Schedule replay consumes this in the next crate cutover.
    pub(crate) fn co_generated_relation_layout(
        &self,
    ) -> Result<&crate::layout::relation::RelationLayout, AkitaError> {
        match &self.purpose {
            CompressionCatalogPurpose::CoGenerated { relation_layout } => Ok(relation_layout),
            CompressionCatalogPurpose::Standalone { .. } => Err(AkitaError::InvalidSetup(
                "standalone compression has no co-generated relation layout".into(),
            )),
        }
    }

    #[allow(dead_code)] // Terminal schedule replay consumes this in the next crate cutover.
    pub(crate) fn terminal_relation_layout<F: CanonicalField>(
        &self,
        lp: &LevelParams,
        opening: &OpeningClaimsLayout,
    ) -> Result<crate::layout::relation::RelationLayout, AkitaError> {
        let (max_opening_log_basis, source_key, field_modulus_minus_one) = match &self.purpose {
            CompressionCatalogPurpose::Standalone {
                max_opening_log_basis,
                source_key,
                field_modulus_minus_one,
            } => (*max_opening_log_basis, source_key, *field_modulus_minus_one),
            CompressionCatalogPurpose::CoGenerated { .. } => {
                return Err(AkitaError::InvalidSetup(
                    "terminal relation requires standalone incoming compression".into(),
                ));
            }
        };
        if (-F::one()).to_canonical_u128() != field_modulus_minus_one {
            return Err(AkitaError::InvalidSetup(
                "terminal compression field identity disagrees with the frozen catalog".into(),
            ));
        }
        if source_key != &lp.b_key || lp.log_basis > max_opening_log_basis {
            return Err(AkitaError::InvalidSetup(
                "terminal compression source key or opening base is incompatible".into(),
            ));
        }
        if !lp.precommitted_groups.is_empty() || self.chains.len() != 1 {
            return Err(AkitaError::InvalidSetup(
                "terminal compression must be exactly the frozen CurrentOuter chain".into(),
            ));
        }
        let chain = self.chains.first().ok_or_else(|| {
            AkitaError::InvalidSetup("terminal compression chain is missing".into())
        })?;
        if chain.source != CompressionSourceId::CurrentOuter {
            return Err(AkitaError::InvalidSetup(
                "terminal compression source must be CurrentOuter".into(),
            ));
        }
        let first_map = chain.maps.first().ok_or_else(|| {
            AkitaError::InvalidSetup("terminal compression first map is missing".into())
        })?;
        if let CompressionAlphabet::OpeningBase { log_basis } = first_map.alphabet {
            if lp.log_basis < log_basis {
                return Err(AkitaError::InvalidSetup(
                    "terminal opening base is smaller than the frozen F-chain base".into(),
                ));
            }
        }
        let local = semantics::compile::<F>(lp, &self.chains)?;
        crate::layout::relation::RelationLayout::compile_terminal_compressed(
            lp,
            opening,
            &local,
            F::modulus_bits(),
        )
    }
}

#[allow(dead_code)] // Reached through the dormant schedule projection.
fn append_source_descriptor_bytes(bytes: &mut Vec<u8>, source: CompressionSourceId) {
    use crate::descriptor_bytes::push_usize;

    match source {
        CompressionSourceId::CurrentOuter => bytes.push(0),
        CompressionSourceId::PrecommittedOuter { index } => {
            bytes.push(1);
            push_usize(bytes, index);
        }
        CompressionSourceId::Opening => bytes.push(2),
    }
}

#[allow(dead_code)] // Reached through the dormant schedule projection.
fn ensure_projection_descriptor_len(bytes: &[u8]) -> Result<(), AkitaError> {
    if bytes.len() > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSetup(
            "compression catalog descriptor exceeds sequence cap".into(),
        ));
    }
    Ok(())
}

#[allow(dead_code)] // Reached through the dormant catalog compiler below.
fn checked_coeffs(count: usize, d: usize, label: &str) -> Result<usize, AkitaError> {
    count.checked_mul(d).ok_or_else(|| {
        AkitaError::InvalidSetup(format!("compression {label} coefficient count overflow"))
    })
}

#[allow(dead_code)] // Reached through the dormant catalog compiler below.
fn audit_key(
    key: &AjtaiKeyParams,
    active_family: SisModulusFamily,
    active_security: u16,
) -> Result<SisTableKey, AkitaError> {
    let stored = key.sis_table_key();
    if key.row_len() == 0 || key.col_len() == 0 {
        return Err(AkitaError::InvalidSetup(
            "compression Ajtai key has zero rows or columns".into(),
        ));
    }
    if stored.family != active_family || stored.min_security_bits != active_security {
        return Err(AkitaError::InvalidSetup(
            "compression Ajtai key disagrees with the active SIS family or security floor".into(),
        ));
    }
    let reconstructed = AjtaiKeyParams::try_new(
        stored.min_security_bits,
        stored.family,
        key.row_len(),
        key.col_len(),
        stored.coeff_linf_bound,
        stored.ring_dimension as usize,
    )
    .map_err(|_| {
        AkitaError::InvalidSetup("compression Ajtai key is not canonically SIS certified".into())
    })?;
    if &reconstructed != key {
        return Err(AkitaError::InvalidSetup(
            "compression Ajtai key disagrees with canonical AjtaiKeyParams construction".into(),
        ));
    }
    Ok(stored)
}

#[allow(dead_code)] // Reached through the dormant catalog compiler below.
fn compile_chain<F: CanonicalField>(
    lp: &LevelParams,
    context: CompressionCatalogContext<'_>,
    range_log_basis: u32,
    gen_ring_dim: usize,
    active_family: SisModulusFamily,
    spec: &CompressionChainSpec,
) -> Result<CompiledCompressionChain, AkitaError> {
    if !(2..=3).contains(&spec.maps.len()) {
        return Err(AkitaError::InvalidSetup(
            "compression chain depth must be in 2..=3".into(),
        ));
    }
    if gen_ring_dim == 0 {
        return Err(AkitaError::InvalidSetup(
            "compression gen_ring_dim must be non-zero".into(),
        ));
    }

    let source_key = resolve_source_key(lp, spec.source)?;
    let source_table_key = source_key.sis_table_key();
    let source_d = source_table_key.ring_dimension as usize;
    let required_source_bucket = rounded_up_collision_inf_norm(
        DEFAULT_SIS_SECURITY_BITS,
        active_family,
        source_d,
        range_log_basis,
    )
    .ok_or_else(|| {
        AkitaError::InvalidSetup("compression source has no required SIS collision bucket".into())
    })?;
    let audited_source = audit_key(source_key, active_family, DEFAULT_SIS_SECURITY_BITS)?;
    if audited_source.coeff_linf_bound < required_source_bucket {
        return Err(AkitaError::InvalidSetup(
            "compression source key is below its required SIS collision bucket".into(),
        ));
    }
    let tier = protocol_dispatch_tier::<F>();
    let source_role = match spec.source {
        CompressionSourceId::CurrentOuter | CompressionSourceId::PrecommittedOuter { .. } => {
            RingRole::Outer
        }
        CompressionSourceId::Opening => RingRole::Opening,
    };
    if !source_d.is_power_of_two()
        || !slot_dim_supported_for_tier(tier, ProtocolDispatchSlot::Role(source_role), source_d)
    {
        return Err(AkitaError::InvalidSetup(
            "compression source ring dimension is outside its role dispatch policy".into(),
        ));
    }
    if !gen_ring_dim.is_multiple_of(source_d) {
        return Err(AkitaError::InvalidSetup(
            "compression source ring dimension does not divide gen_ring_dim".into(),
        ));
    }
    let source_setup_coeffs = source_key
        .row_len()
        .checked_mul(source_key.col_len())
        .and_then(|count| count.checked_mul(source_d))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("compression source setup footprint overflow".into())
        })?;
    if source_setup_coeffs > MAX_SETUP_MATRIX_FIELD_ELEMENTS {
        return Err(AkitaError::InvalidSetup(
            "compression source setup footprint exceeds cap".into(),
        ));
    }
    let source_output_coeffs = checked_coeffs(source_key.row_len(), source_d, "source output")?;
    if source_output_coeffs > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSetup(
            "compression source output exceeds sequence cap".into(),
        ));
    }

    let modulus = crate::field_modulus::<F>();
    let field_bits = 128 - modulus.saturating_sub(1).leading_zeros();
    if field_bits == 0 {
        return Err(AkitaError::InvalidSetup(
            "compression field modulus has zero bit length".into(),
        ));
    }
    let mut previous_output = source_output_coeffs;
    let mut maps = Vec::with_capacity(spec.maps.len());
    for (index, map) in spec.maps.iter().enumerate() {
        if index != 0 && matches!(map.alphabet, CompressionAlphabet::OpeningBase { .. }) {
            return Err(AkitaError::InvalidSetup(
                "compression opening-base alphabet is permitted only on the first map".into(),
            ));
        }
        if let CompressionAlphabet::OpeningBase { log_basis } = map.alphabet {
            match (context, spec.source) {
                (
                    CompressionCatalogContext::CoGeneratedLevel { .. },
                    CompressionSourceId::CurrentOuter | CompressionSourceId::Opening,
                ) if log_basis != lp.log_basis => {
                    return Err(AkitaError::InvalidSetup(
                        "co-generated current/opening first-map base must equal the level base"
                            .into(),
                    ));
                }
                (
                    CompressionCatalogContext::StandaloneCommitment { .. },
                    CompressionSourceId::CurrentOuter,
                ) if log_basis != 4 => {
                    return Err(AkitaError::InvalidSetup(
                        "standalone commitment opening-base first map must use log_basis=4".into(),
                    ));
                }
                _ => {}
            }
        }
        let digit_depth = compression_digit_depth(map.alphabet, field_bits, range_log_basis)?;
        let stored = map.key.sis_table_key();
        let required_bucket = match map.alphabet {
            CompressionAlphabet::NegativeBinary => {
                let key = sis_table_key_for_linf_bound(
                    DEFAULT_SIS_SECURITY_BITS,
                    active_family,
                    stored.ring_dimension,
                    1,
                )
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "negative-binary map has no exact SIS table row".into(),
                    )
                })?;
                key.coeff_linf_bound
            }
            CompressionAlphabet::OpeningBase { .. } => rounded_up_collision_inf_norm(
                DEFAULT_SIS_SECURITY_BITS,
                active_family,
                stored.ring_dimension as usize,
                range_log_basis,
            )
            .ok_or_else(|| {
                AkitaError::InvalidSetup("opening-base map has no required SIS table row".into())
            })?,
        };
        let table_key = audit_key(&map.key, active_family, DEFAULT_SIS_SECURITY_BITS)?;
        match map.alphabet {
            CompressionAlphabet::NegativeBinary
                if table_key.coeff_linf_bound != required_bucket =>
            {
                return Err(AkitaError::InvalidSetup(
                    "negative-binary map must use its exact SIS collision bucket".into(),
                ));
            }
            CompressionAlphabet::OpeningBase { .. }
                if table_key.coeff_linf_bound < required_bucket =>
            {
                return Err(AkitaError::InvalidSetup(
                    "opening-base map is below its required SIS collision bucket".into(),
                ));
            }
            _ => {}
        }
        let d = table_key.ring_dimension as usize;
        if !d.is_power_of_two()
            || !slot_dim_supported_for_tier(tier, ProtocolDispatchSlot::Compression, d)
            || !slot_dim_supported_for_tier(tier, ProtocolDispatchSlot::Ntt, d)
        {
            return Err(AkitaError::InvalidSetup(format!(
                "compression map {index} ring dimension {d} is outside dispatch policy"
            )));
        }
        if !gen_ring_dim.is_multiple_of(d) {
            return Err(AkitaError::InvalidSetup(format!(
                "compression map {index} ring dimension {d} does not divide gen_ring_dim={gen_ring_dim}"
            )));
        }
        let expected_input = previous_output.checked_mul(digit_depth).ok_or_else(|| {
            AkitaError::InvalidSetup("compression decomposed input length overflow".into())
        })?;
        let actual_input = checked_coeffs(map.key.col_len(), d, "map input")?;
        if actual_input > DEFAULT_MAX_SEQUENCE_LEN {
            return Err(AkitaError::InvalidSetup(format!(
                "compression map {index} input exceeds sequence cap"
            )));
        }
        if actual_input != expected_input {
            return Err(AkitaError::InvalidSetup(format!(
                "compression map {index} input has {actual_input} coefficients, expected {expected_input}"
            )));
        }
        let output_coeffs = checked_coeffs(map.key.row_len(), d, "map output")?;
        if output_coeffs > DEFAULT_MAX_SEQUENCE_LEN {
            return Err(AkitaError::InvalidSetup(format!(
                "compression map {index} output exceeds sequence cap"
            )));
        }
        let setup_coeffs = map
            .key
            .row_len()
            .checked_mul(map.key.col_len())
            .and_then(|count| count.checked_mul(d))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression setup footprint overflow".into())
            })?;
        if setup_coeffs > MAX_SETUP_MATRIX_FIELD_ELEMENTS {
            return Err(AkitaError::InvalidSetup(format!(
                "compression map {index} setup footprint {setup_coeffs} exceeds {MAX_SETUP_MATRIX_FIELD_ELEMENTS}"
            )));
        }
        maps.push(CompiledCompressionMap {
            key: map.key.clone(),
            alphabet: map.alphabet,
            digit_depth,
            input_coeffs: actual_input,
            output_coeffs,
        });
        previous_output = output_coeffs;
    }
    if previous_output > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSetup(format!(
            "compression payload coefficient count {previous_output} exceeds {DEFAULT_MAX_SEQUENCE_LEN}"
        )));
    }
    Ok(CompiledCompressionChain {
        source: spec.source,
        source_output_coeffs,
        maps,
        payload_coeffs: previous_output,
    })
}

/// Validate and compile the complete compression catalog for one level.
#[allow(dead_code)] // Wired into schedule replay in the compression cutover slice.
pub fn validate_compression_catalog<F: CanonicalField>(
    lp: &LevelParams,
    context: CompressionCatalogContext<'_>,
    gen_ring_dim: usize,
    specs: Vec<CompressionChainSpec>,
) -> Result<ValidatedCompressionCatalog, AkitaError> {
    if lp.log_basis == 0 || lp.log_basis >= 128 {
        return Err(AkitaError::InvalidSetup(
            "compression level log_basis must be in 1..128".into(),
        ));
    }
    let (range_log_basis, co_generated_opening) = match context {
        CompressionCatalogContext::CoGeneratedLevel { opening } => {
            opening.check()?;
            lp.validate_root_opening_batch(opening)?;
            (lp.log_basis, Some(opening))
        }
        CompressionCatalogContext::StandaloneCommitment {
            max_opening_log_basis,
        } => {
            if !lp.precommitted_groups.is_empty() {
                return Err(AkitaError::InvalidSetup(
                    "standalone compression cannot include precommitted groups".into(),
                ));
            }
            if !(4..128).contains(&max_opening_log_basis) || max_opening_log_basis < lp.log_basis {
                return Err(AkitaError::InvalidSetup(
                    "standalone max_opening_log_basis must be in 4..128 and cover lp.log_basis"
                        .into(),
                ));
            }
            (max_opening_log_basis, None)
        }
    };
    let precommitted_count = match context {
        CompressionCatalogContext::CoGeneratedLevel { .. } => lp.precommitted_groups.len(),
        CompressionCatalogContext::StandaloneCommitment { .. } => 0,
    };
    let opening_count = usize::from(matches!(
        context,
        CompressionCatalogContext::CoGeneratedLevel { .. }
    ));
    let expected_source_count = 1usize
        .checked_add(precommitted_count)
        .and_then(|count| count.checked_add(opening_count))
        .ok_or_else(|| AkitaError::InvalidSetup("compression source count overflow".into()))?;
    if expected_source_count > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSetup(format!(
            "compression source count {expected_source_count} exceeds cap {DEFAULT_MAX_SEQUENCE_LEN}"
        )));
    }
    let mut expected_sources = Vec::with_capacity(expected_source_count);
    expected_sources.push(CompressionSourceId::CurrentOuter);
    expected_sources.extend(
        (0..precommitted_count).map(|index| CompressionSourceId::PrecommittedOuter { index }),
    );
    if opening_count == 1 {
        expected_sources.push(CompressionSourceId::Opening);
    }
    if specs.len() != expected_source_count
        || !specs
            .iter()
            .zip(&expected_sources)
            .all(|(spec, expected)| spec.source == *expected)
    {
        return Err(AkitaError::InvalidSetup(format!(
            "compression catalog sources must be exactly {expected_sources:?} in canonical order"
        )));
    }
    let active_family = match protocol_dispatch_tier::<F>() {
        crate::ProtocolRingDispatchTierId::Fp128 => SisModulusFamily::Q128,
        crate::ProtocolRingDispatchTierId::Fp64 => SisModulusFamily::Q64,
        crate::ProtocolRingDispatchTierId::Fp32 => SisModulusFamily::Q32,
    };
    let chains = specs
        .iter()
        .map(|spec| {
            compile_chain::<F>(
                lp,
                context,
                range_log_basis,
                gen_ring_dim,
                active_family,
                spec,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let purpose = if let Some(opening) = co_generated_opening {
        let local = semantics::compile::<F>(lp, &chains)?;
        CompressionCatalogPurpose::CoGenerated {
            relation_layout: crate::layout::relation::RelationLayout::compile_compressed(
                lp,
                opening,
                &local,
                F::modulus_bits(),
            )?,
        }
    } else {
        CompressionCatalogPurpose::Standalone {
            max_opening_log_basis: range_log_basis,
            source_key: lp.b_key.clone(),
            field_modulus_minus_one: (-F::one()).to_canonical_u128(),
        }
    };
    Ok(ValidatedCompressionCatalog { chains, purpose })
}

#[cfg(test)]
mod tests;
