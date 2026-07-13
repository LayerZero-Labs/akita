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

mod choice;
pub use choice::{
    compression_digit_depth, CompressionAlphabet, CompressionCatalogContext,
    CompressionChainChoice, CompressionChainSpec, CompressionMapChoice, CompressionMapSpec,
    CompressionSourceId, FrozenCompressionChainChoice, LevelCompressionPlan,
    STANDALONE_OPENING_BASE_LOG_BASIS,
};

/// Stable prefix for compression catalog capacity misses.
///
/// These mean a candidate does not fit the setup/sequence envelope. Planner
/// search treats them as ladder misses (`Ok(None)`), not corrupted catalogs.
pub const COMPRESSION_CAPACITY_INFEASIBLE_PREFIX: &str = "compression capacity infeasible:";

/// Returns true when `err` is a capacity miss tagged with
/// [`COMPRESSION_CAPACITY_INFEASIBLE_PREFIX`].
#[must_use]
pub fn compression_capacity_infeasible(err: &AkitaError) -> bool {
    matches!(
        err,
        AkitaError::InvalidSetup(message)
            if message.starts_with(COMPRESSION_CAPACITY_INFEASIBLE_PREFIX)
    )
}

fn capacity_infeasible(detail: impl Into<String>) -> AkitaError {
    AkitaError::InvalidSetup(format!(
        "{COMPRESSION_CAPACITY_INFEASIBLE_PREFIX} {}",
        detail.into()
    ))
}
#[derive(Debug, Clone, PartialEq, Eq)]
enum CompressionCatalogPurpose {
    CoGenerated {
        relation_layout: crate::layout::relation::RelationLayout,
    },
    TerminalFold {
        relation_layout: crate::layout::relation::RelationLayout,
    },
    Standalone {
        max_opening_log_basis: u32,
        source_key: AjtaiKeyParams,
        field_modulus_minus_one: u128,
    },
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
    max_opening_log_basis: u32,
    source_output_coeffs: usize,
    maps: Vec<CompiledCompressionMap>,
    payload_coeffs: usize,
}

/// One ordered map's replay facts for prover hint sizing and preparation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionMapHintShape {
    pub source: CompressionSourceId,
    pub map_index: usize,
    pub alphabet: CompressionAlphabet,
    pub native_ring_dim: usize,
    pub rows: usize,
    pub cols: usize,
    pub digit_depth: usize,
    pub input_coeffs: usize,
    pub output_coeffs: usize,
    pub prefix_ring_elements: usize,
    pub is_terminal: bool,
}

/// Catalog-wide replay projection derived only from checked compiled maps.
///
/// This is not a second planner or protocol authority. It packages the
/// performance and hint-shape projections that would otherwise be re-derived
/// independently by schedule replay, prepared NTT setup, and profile reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionCatalogProjection {
    /// Setup generation dimension under which this catalog was validated.
    gen_ring_dim: usize,
    maps: Vec<CompressionMapHintShape>,
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

/// Catalog-wide compression setup facts aggregated across multiple projections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregatedCompressionSetup {
    gen_ring_dim: Option<usize>,
    map_hints: Vec<CompressionMapHintShape>,
    ntt_requirements: Vec<(usize, usize)>,
    max_flat_setup_prefix_coeffs: usize,
    coalesced_cache_field_coeffs: usize,
}

impl AggregatedCompressionSetup {
    #[must_use]
    pub fn gen_ring_dim(&self) -> Option<usize> {
        self.gen_ring_dim
    }

    #[must_use]
    pub fn map_hints(&self) -> &[CompressionMapHintShape] {
        &self.map_hints
    }

    #[must_use]
    pub fn ntt_requirements(&self) -> &[(usize, usize)] {
        &self.ntt_requirements
    }

    #[must_use]
    pub fn max_flat_setup_prefix_coeffs(&self) -> usize {
        self.max_flat_setup_prefix_coeffs
    }

    #[must_use]
    pub fn coalesced_cache_field_coeffs(&self) -> usize {
        self.coalesced_cache_field_coeffs
    }
}

/// Canonical accounting shape of a terminal compressed relation.
///
/// The shape is derived from the checked terminal [`RelationLayout`]. It is a
/// compact accounting view for planner recurrence and proof sizing; callers do
/// not provide or serialize any of these fields independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionTerminalRelationShape {
    relation_padded_rows: usize,
    logical_coeffs: usize,
    compression_witness_coeffs: usize,
    witness_field_coeffs: usize,
    payload_coeffs: usize,
}

impl CompressionTerminalRelationShape {
    #[must_use]
    pub fn relation_padded_rows(&self) -> usize {
        self.relation_padded_rows
    }

    #[must_use]
    pub fn logical_coeffs(&self) -> usize {
        self.logical_coeffs
    }

    #[must_use]
    pub fn compression_witness_coeffs(&self) -> usize {
        self.compression_witness_coeffs
    }

    #[must_use]
    pub fn witness_field_coeffs(&self) -> usize {
        self.witness_field_coeffs
    }

    #[must_use]
    pub fn payload_coeffs(&self) -> usize {
        self.payload_coeffs
    }
}

impl CompressionCatalogProjection {
    #[must_use]
    pub fn gen_ring_dim(&self) -> usize {
        self.gen_ring_dim
    }

    #[must_use]
    pub fn map_hints(&self) -> &[CompressionMapHintShape] {
        &self.maps
    }

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
    pub fn coalesced_cache_field_coeffs(&self) -> usize {
        self.coalesced_cache_field_coeffs
    }

    #[must_use]
    pub fn descriptor_bytes(&self) -> &[u8] {
        &self.descriptor_bytes
    }
}

/// Aggregate checked catalog projections into one setup/hint envelope.
///
/// Map hints are concatenated in caller order. Per-ring NTT prefix requirements
/// are max-coalesced by `ring_d`. Flat setup prefix and coalesced cache totals
/// follow the same rules as a single projection, but over the merged facts.
pub fn aggregate_catalog_projections(
    projections: &[&CompressionCatalogProjection],
) -> Result<AggregatedCompressionSetup, AkitaError> {
    let mut map_hints = Vec::new();
    let mut ntt_by_d = BTreeMap::<usize, usize>::new();
    let mut max_flat_setup_prefix_coeffs = 0usize;
    let mut gen_ring_dim = None;

    for projection in projections {
        match gen_ring_dim {
            None => gen_ring_dim = Some(projection.gen_ring_dim()),
            Some(expected) if expected == projection.gen_ring_dim() => {}
            Some(expected) => {
                return Err(AkitaError::InvalidSetup(format!(
                    "aggregated compression projections disagree on gen_ring_dim: {expected} vs {}",
                    projection.gen_ring_dim()
                )));
            }
        }
        map_hints.extend_from_slice(projection.map_hints());
        max_flat_setup_prefix_coeffs =
            max_flat_setup_prefix_coeffs.max(projection.max_flat_setup_prefix_coeffs());
        for &(ring_d, prefix_ring_elements) in projection.ntt_requirements() {
            ntt_by_d
                .entry(ring_d)
                .and_modify(|prefix| *prefix = (*prefix).max(prefix_ring_elements))
                .or_insert(prefix_ring_elements);
        }
    }
    if map_hints.len() > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSetup(
            "aggregated compression map count exceeds sequence cap".into(),
        ));
    }

    let ntt_requirements = ntt_by_d.into_iter().collect::<Vec<_>>();
    let coalesced_cache_field_coeffs =
        ntt_requirements
            .iter()
            .try_fold(0usize, |total, &(ring_d, prefix_ring_elements)| {
                let field_coeffs = ring_d.checked_mul(prefix_ring_elements).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "aggregated compression coalesced cache footprint overflow".into(),
                    )
                })?;
                total.checked_add(field_coeffs).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "aggregated compression total coalesced cache footprint overflow".into(),
                    )
                })
            })?;

    Ok(AggregatedCompressionSetup {
        gen_ring_dim,
        map_hints,
        ntt_requirements,
        max_flat_setup_prefix_coeffs,
        coalesced_cache_field_coeffs,
    })
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
    gen_ring_dim: usize,
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
                maps.push(CompressionMapHintShape {
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

        if let CompressionCatalogPurpose::CoGenerated { relation_layout }
        | CompressionCatalogPurpose::TerminalFold { relation_layout } = &self.purpose
        {
            let cost = relation_layout.compression_structural_cost()?;
            let payload_coeffs =
                payload_coeffs_by_source
                    .iter()
                    .try_fold(0usize, |total, &(_, coeffs)| {
                        total.checked_add(coeffs).ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "compression projected terminal payload overflow".into(),
                            )
                        })
                    })?;
            let dimension_requirements = cost
                .dimensions()
                .iter()
                .map(|dimension| {
                    (
                        dimension.native_ring_dim(),
                        dimension.max_setup_prefix_coeffs() / dimension.native_ring_dim(),
                    )
                })
                .collect::<Vec<_>>();
            let maps_are_canonical = maps.windows(2).all(|pair| {
                compression_map_order_key(&pair[0]) < compression_map_order_key(&pair[1])
            });
            let payloads_are_canonical = payload_coeffs_by_source.windows(2).all(|pair| {
                compression_source_order_key(pair[0].0) < compression_source_order_key(pair[1].0)
            });
            let payloads_match = payload_coeffs_by_source.iter().all(|&(source, coeffs)| {
                cost.maps()
                    .iter()
                    .filter(|map| {
                        map.source() == source && map.terminal_payload_coeffs() == Some(coeffs)
                    })
                    .count()
                    == 1
            });
            let maps_match = maps.iter().all(|map| {
                cost.maps()
                    .iter()
                    .filter(|relation_map| {
                        relation_map.source() == map.source
                            && relation_map.map() == map.map_index
                            && relation_map.native_ring_dim() == map.native_ring_dim
                            && relation_map.rows() == map.rows
                            && relation_map.input_coeffs() == map.input_coeffs
                            && relation_map.output_coeffs() == map.output_coeffs
                            && map.rows.checked_mul(map.input_coeffs)
                                == Some(relation_map.logical_setup_coeffs())
                            && relation_map.terminal_payload_coeffs()
                                == map.is_terminal.then_some(map.output_coeffs)
                            && relation_map.is_negative_binary()
                                == matches!(map.alphabet, CompressionAlphabet::NegativeBinary)
                    })
                    .count()
                    == 1
            });
            if cost.map_count() != maps.len()
                || !maps_are_canonical
                || !payloads_are_canonical
                || !payloads_match
                || !maps_match
                || cost.terminal_payload_coeffs() != payload_coeffs
                || cost.logical_setup_coeffs() != logical_setup_coeffs
                || cost.max_setup_prefix_coeffs() != max_flat_setup_prefix_coeffs
                || cost.coalesced_setup_cache_coeffs() != coalesced_cache_field_coeffs
                || dimension_requirements != ntt_requirements
            {
                return Err(AkitaError::InvalidSetup(
                    "compression catalog projection disagrees with its relation providers".into(),
                ));
            }
        }

        Ok(CompressionCatalogProjection {
            gen_ring_dim: self.gen_ring_dim,
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
            CompressionCatalogPurpose::TerminalFold { .. } => bytes.push(2),
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
            push_u32(bytes, chain.max_opening_log_basis);
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

    /// Return the canonical relation graph compiled for a co-generated level.
    ///
    /// Prover execution consumes this graph directly; the projected catalog is
    /// reserved for setup/cache sizing and descriptor accounting.
    pub fn co_generated_relation_layout(
        &self,
    ) -> Result<&crate::layout::relation::RelationLayout, AkitaError> {
        match &self.purpose {
            CompressionCatalogPurpose::CoGenerated { relation_layout } => Ok(relation_layout),
            CompressionCatalogPurpose::TerminalFold { .. }
            | CompressionCatalogPurpose::Standalone { .. } => Err(AkitaError::InvalidSetup(
                "compression catalog is not a co-generated WithDBlock level".into(),
            )),
        }
    }

    /// Return the canonical relation graph for a terminal `WithoutDBlock` fold.
    pub fn terminal_relation_layout(
        &self,
    ) -> Result<&crate::layout::relation::RelationLayout, AkitaError> {
        match &self.purpose {
            CompressionCatalogPurpose::TerminalFold { relation_layout } => Ok(relation_layout),
            CompressionCatalogPurpose::CoGenerated { .. }
            | CompressionCatalogPurpose::Standalone { .. } => Err(AkitaError::InvalidSetup(
                "compression catalog is not a terminal WithoutDBlock fold".into(),
            )),
        }
    }

    /// Return the canonical relation graph for either fold-level catalog purpose.
    pub fn fold_relation_layout(
        &self,
    ) -> Result<&crate::layout::relation::RelationLayout, AkitaError> {
        match &self.purpose {
            CompressionCatalogPurpose::CoGenerated { relation_layout }
            | CompressionCatalogPurpose::TerminalFold { relation_layout } => Ok(relation_layout),
            CompressionCatalogPurpose::Standalone { .. } => Err(AkitaError::InvalidSetup(
                "standalone compression catalog has no fold relation layout".into(),
            )),
        }
    }

    /// Return the canonical terminal shape derived from the checked relation.
    pub fn terminal_relation_shape(&self) -> Result<CompressionTerminalRelationShape, AkitaError> {
        let layout = self.terminal_relation_layout()?;
        let cost = layout.compression_structural_cost()?;
        Ok(CompressionTerminalRelationShape {
            relation_padded_rows: layout.row_plan().padded_row_count(),
            logical_coeffs: layout.total_coeffs(),
            compression_witness_coeffs: cost.witness_coeffs(),
            witness_field_coeffs: layout.physical_witness_field_coeff_len()?,
            payload_coeffs: cost.terminal_payload_coeffs(),
        })
    }
}

#[allow(dead_code)] // Reached through the dormant schedule projection.
fn compression_map_order_key(map: &CompressionMapHintShape) -> (u8, usize, usize) {
    let (kind, index) = compression_source_order_key(map.source);
    (kind, index, map.map_index)
}

#[allow(dead_code)] // Reached through the dormant schedule projection.
fn compression_source_order_key(source: CompressionSourceId) -> (u8, usize) {
    match source {
        CompressionSourceId::CurrentOuter => (0, 0),
        CompressionSourceId::PrecommittedOuter { index } => (1, index),
        CompressionSourceId::Opening => (2, 0),
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
    let range_log_basis = spec.max_opening_log_basis;
    if !(1..128).contains(&range_log_basis) {
        return Err(AkitaError::InvalidSetup(
            "compression source max_opening_log_basis must be in 1..128".into(),
        ));
    }
    match (context, spec.source) {
        (
            CompressionCatalogContext::CoGeneratedLevel { .. },
            CompressionSourceId::CurrentOuter | CompressionSourceId::Opening,
        ) if range_log_basis != lp.log_basis => {
            return Err(AkitaError::InvalidSetup(
                "new co-generated compression sources must freeze the level opening base".into(),
            ));
        }
        (CompressionCatalogContext::CoGeneratedLevel { .. }, _)
        | (CompressionCatalogContext::TerminalFold { .. }, _)
            if lp.log_basis > range_log_basis =>
        {
            return Err(AkitaError::InvalidSetup(
                "active opening base exceeds a frozen F-chain envelope".into(),
            ));
        }
        (CompressionCatalogContext::StandaloneCommitment, _)
            if range_log_basis < lp.log_basis
                || range_log_basis < STANDALONE_OPENING_BASE_LOG_BASIS =>
        {
            return Err(AkitaError::InvalidSetup(
                "standalone F-chain envelope must cover the commitment and minimum opening base"
                    .into(),
            ));
        }
        _ => {}
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
        return Err(capacity_infeasible(
            "compression source setup footprint exceeds setup matrix field cap",
        ));
    }
    let source_output_coeffs = checked_coeffs(source_key.row_len(), source_d, "source output")?;
    if source_output_coeffs > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(capacity_infeasible(
            "compression source output exceeds sequence cap",
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
            if index == 0
                && matches!(
                    context,
                    CompressionCatalogContext::CoGeneratedLevel { .. }
                        | CompressionCatalogContext::TerminalFold { .. }
                )
                && spec.source != CompressionSourceId::Opening
                && log_basis > lp.log_basis
            {
                return Err(AkitaError::InvalidSetup(
                    "active opening base is below a frozen F-chain first-map base".into(),
                ));
            }
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
                    CompressionCatalogContext::StandaloneCommitment,
                    CompressionSourceId::CurrentOuter,
                ) if log_basis != STANDALONE_OPENING_BASE_LOG_BASIS => {
                    return Err(AkitaError::InvalidSetup(
                        "standalone commitment opening-base first map must use frozen base 4"
                            .into(),
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
            return Err(capacity_infeasible(format!(
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
            return Err(capacity_infeasible(format!(
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
            return Err(capacity_infeasible(format!(
                "compression map {index} setup footprint {setup_coeffs} exceeds setup matrix field cap {MAX_SETUP_MATRIX_FIELD_ELEMENTS}"
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
        max_opening_log_basis: spec.max_opening_log_basis,
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
    let (relation_opening, terminal) = match context {
        CompressionCatalogContext::CoGeneratedLevel { opening } => {
            opening.check()?;
            lp.validate_root_opening_batch(opening)?;
            (Some(opening), false)
        }
        CompressionCatalogContext::TerminalFold { opening } => {
            opening.check()?;
            lp.validate_root_opening_batch(opening)?;
            (Some(opening), true)
        }
        CompressionCatalogContext::StandaloneCommitment => {
            if !lp.precommitted_groups.is_empty() {
                return Err(AkitaError::InvalidSetup(
                    "standalone compression cannot include precommitted groups".into(),
                ));
            }
            (None, false)
        }
    };
    let precommitted_count = match context {
        CompressionCatalogContext::CoGeneratedLevel { .. }
        | CompressionCatalogContext::TerminalFold { .. } => lp.precommitted_groups.len(),
        CompressionCatalogContext::StandaloneCommitment => 0,
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
    let active_family = crate::sis_family_for_field::<F>();
    let chains = specs
        .iter()
        .map(|spec| compile_chain::<F>(lp, context, gen_ring_dim, active_family, spec))
        .collect::<Result<Vec<_>, _>>()?;
    let purpose = if let Some(opening) = relation_opening {
        let local = semantics::compile::<F>(lp, &chains)?;
        if terminal {
            CompressionCatalogPurpose::TerminalFold {
                relation_layout:
                    crate::layout::relation::RelationLayout::compile_terminal_compressed(
                        lp,
                        opening,
                        &local,
                        F::modulus_bits(),
                    )?,
            }
        } else {
            CompressionCatalogPurpose::CoGenerated {
                relation_layout: crate::layout::relation::RelationLayout::compile_compressed(
                    lp,
                    opening,
                    &local,
                    F::modulus_bits(),
                )?,
            }
        }
    } else {
        CompressionCatalogPurpose::Standalone {
            max_opening_log_basis: specs
                .first()
                .ok_or_else(|| AkitaError::InvalidSetup("standalone F chain is missing".into()))?
                .max_opening_log_basis,
            source_key: lp.b_key.clone(),
            field_modulus_minus_one: (-F::one()).to_canonical_u128(),
        }
    };
    Ok(ValidatedCompressionCatalog {
        gen_ring_dim,
        chains,
        purpose,
    })
}

#[cfg(test)]
mod tests;
