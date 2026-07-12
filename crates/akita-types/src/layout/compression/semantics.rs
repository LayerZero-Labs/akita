//! Dormant compression-local semantic relation geometry.
//!
//! This module assigns no global witness or relation-row offsets. The checked
//! catalog compiler invokes it internally for co-generated levels; callers
//! cannot join an independently supplied catalog and level.

use akita_field::{AkitaError, CanonicalField};
use akita_serialization::DEFAULT_MAX_SEQUENCE_LEN;

use crate::{r_decomp_levels, LevelParams};

use super::{
    resolve_source_key, CompressionAlphabet, CompressionSourceId, ValidatedCompressionCatalog,
};

const BINARY_SUPPORT_DERIVATION_VERSION: u8 = 1;

/// Flat field-coefficient cells in the compression-local witness arena.
///
/// Ring dimension is intentionally absent: one digit segment is viewed at the
/// current map dimension and at its predecessor's output dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CoeffSpan {
    start: usize,
    len: usize,
}

impl CoeffSpan {
    fn end(self) -> Result<usize, AkitaError> {
        self.start
            .checked_add(self.len)
            .ok_or_else(|| AkitaError::InvalidSetup("compression span end overflow".into()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentId {
    Xi {
        source: CompressionSourceId,
        map: usize,
    },
    Quotient {
        source: CompressionSourceId,
        map: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompressionSegment {
    id: SegmentId,
    span: CoeffSpan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompressionRowId {
    source: CompressionSourceId,
    map: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RowSpan {
    start: usize,
    len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompressionRowRhs {
    Zero,
    TerminalPayload { coeffs: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompressionRowProvider {
    id: CompressionRowId,
    rows: RowSpan,
    input: SegmentId,
    successor: Option<SegmentId>,
    quotient: SegmentId,
    native_ring_dim: usize,
    rhs: CompressionRowRhs,
}

/// Compression support to add to one already-existing B/D row family.
/// Global row identity and the source role are resolved only during Slice 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AugmentationIntent {
    source: CompressionSourceId,
    compression_input: SegmentId,
}

/// Checked, compression-local semantics. This is stored but not executable.
#[allow(dead_code)] // Read by the relation-layout composition slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CompiledCompressionSemantics {
    segments: Vec<CompressionSegment>,
    rows: Vec<CompressionRowProvider>,
    augmentations: Vec<AugmentationIntent>,
    negative_binary_inputs: Vec<SegmentId>,
    binary_support_derivation_version: u8,
    total_coeffs: usize,
    total_rows: usize,
}

fn checked_add_capped(current: usize, len: usize, label: &str) -> Result<usize, AkitaError> {
    let next = current
        .checked_add(len)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("compression {label} overflow")))?;
    if next > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSetup(format!(
            "compression {label} {next} exceeds cap {DEFAULT_MAX_SEQUENCE_LEN}"
        )));
    }
    Ok(next)
}

fn allocate_span(cursor: &mut usize, len: usize, label: &str) -> Result<CoeffSpan, AkitaError> {
    if len == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "compression {label} must be non-zero"
        )));
    }
    let start = *cursor;
    *cursor = checked_add_capped(start, len, label)?;
    Ok(CoeffSpan { start, len })
}

fn validate_span_view(span: CoeffSpan, d: usize, label: &str) -> Result<(), AkitaError> {
    if d == 0 || !span.len.is_multiple_of(d) {
        return Err(AkitaError::InvalidSetup(format!(
            "compression {label} length {} is not divisible by native ring dimension {d}",
            span.len
        )));
    }
    let _ = span.end()?;
    Ok(())
}

fn checked_semantic_capacities(
    map_counts: impl IntoIterator<Item = usize>,
) -> Result<(usize, usize), AkitaError> {
    let total_maps = map_counts.into_iter().try_fold(0usize, |total, count| {
        total
            .checked_add(count)
            .ok_or_else(|| AkitaError::InvalidSetup("compression map count overflow".into()))
    })?;
    let total_segments = total_maps
        .checked_mul(2)
        .ok_or_else(|| AkitaError::InvalidSetup("compression segment count overflow".into()))?;
    if total_segments > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSetup(format!(
            "compression segment count {total_segments} exceeds cap {DEFAULT_MAX_SEQUENCE_LEN}"
        )));
    }
    Ok((total_maps, total_segments))
}

fn segment_span(segments: &[CompressionSegment], id: SegmentId) -> Result<CoeffSpan, AkitaError> {
    segments
        .iter()
        .find_map(|segment| (segment.id == id).then_some(segment.span))
        .ok_or_else(|| AkitaError::InvalidSetup("compression semantic segment is missing".into()))
}

/// Compile semantics already guaranteed by the parent catalog validator.
///
/// This function validates only new semantic allocation and native-view facts;
/// it does not repeat catalog geometry, security, or alphabet validation.
pub(super) fn compile<F: CanonicalField>(
    lp: &LevelParams,
    catalog: &ValidatedCompressionCatalog,
) -> Result<CompiledCompressionSemantics, AkitaError> {
    let max_depth = catalog
        .chains
        .iter()
        .map(|chain| chain.maps.len())
        .max()
        .ok_or_else(|| AkitaError::InvalidSetup("compression catalog is empty".into()))?;
    let (total_maps, total_segments) =
        checked_semantic_capacities(catalog.chains.iter().map(|chain| chain.maps.len()))?;
    let mut ordered_maps = Vec::with_capacity(total_maps);
    for map in 0..max_depth {
        for (chain, spec) in catalog.chains.iter().enumerate() {
            if spec.maps.get(map).is_some() {
                ordered_maps.push((chain, map));
            }
        }
    }

    let mut cursor = 0usize;
    let mut segments = Vec::with_capacity(total_segments);
    let mut negative_binary_inputs = Vec::with_capacity(total_maps);

    // All xi segments precede all quotients. Each suborder is layer-major and
    // retains the catalog's current/precommitted/opening source order.
    for &(chain_index, map_index) in &ordered_maps {
        let chain = &catalog.chains[chain_index];
        let map = &chain.maps[map_index];
        let d = map.key.sis_table_key().ring_dimension as usize;
        let id = SegmentId::Xi {
            source: chain.source,
            map: map_index,
        };
        let span = allocate_span(&mut cursor, map.input_coeffs, "xi coefficients")?;
        validate_span_view(span, d, "xi map view")?;
        let predecessor_d = if map_index == 0 {
            resolve_source_key(lp, chain.source)?
                .sis_table_key()
                .ring_dimension as usize
        } else {
            chain.maps[map_index - 1].key.sis_table_key().ring_dimension as usize
        };
        validate_span_view(span, predecessor_d, "xi predecessor view")?;
        segments.push(CompressionSegment { id, span });
        if map.alphabet == CompressionAlphabet::NegativeBinary {
            negative_binary_inputs.push(id);
        }
    }

    let quotient_levels = r_decomp_levels::<F>(lp.log_basis);
    for &(chain_index, map_index) in &ordered_maps {
        let chain = &catalog.chains[chain_index];
        let map = &chain.maps[map_index];
        let d = map.key.sis_table_key().ring_dimension as usize;
        let len = map
            .output_coeffs
            .checked_mul(quotient_levels)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression quotient length overflow".into())
            })?;
        let id = SegmentId::Quotient {
            source: chain.source,
            map: map_index,
        };
        let span = allocate_span(&mut cursor, len, "quotient coefficients")?;
        validate_span_view(span, d, "quotient view")?;
        segments.push(CompressionSegment { id, span });
    }

    let mut rows = Vec::with_capacity(total_maps);
    let mut total_rows = 0usize;
    for &(chain_index, map_index) in &ordered_maps {
        let chain = &catalog.chains[chain_index];
        let map = &chain.maps[map_index];
        let row_start = total_rows;
        total_rows = checked_add_capped(total_rows, map.key.row_len(), "local row count")?;
        let input = SegmentId::Xi {
            source: chain.source,
            map: map_index,
        };
        let _ = segment_span(&segments, input)?;
        let successor = if map_index + 1 < chain.maps.len() {
            let id = SegmentId::Xi {
                source: chain.source,
                map: map_index + 1,
            };
            let _ = segment_span(&segments, id)?;
            Some(id)
        } else {
            None
        };
        let quotient = SegmentId::Quotient {
            source: chain.source,
            map: map_index,
        };
        let _ = segment_span(&segments, quotient)?;
        rows.push(CompressionRowProvider {
            id: CompressionRowId {
                source: chain.source,
                map: map_index,
            },
            rows: RowSpan {
                start: row_start,
                len: map.key.row_len(),
            },
            input,
            successor,
            quotient,
            native_ring_dim: map.key.sis_table_key().ring_dimension as usize,
            rhs: if map_index + 1 < chain.maps.len() {
                CompressionRowRhs::Zero
            } else {
                CompressionRowRhs::TerminalPayload {
                    coeffs: map.output_coeffs,
                }
            },
        });
    }

    let augmentations = catalog
        .chains
        .iter()
        .map(|chain| AugmentationIntent {
            source: chain.source,
            compression_input: SegmentId::Xi {
                source: chain.source,
                map: 0,
            },
        })
        .collect();

    Ok(CompiledCompressionSemantics {
        segments,
        rows,
        augmentations,
        negative_binary_inputs,
        binary_support_derivation_version: BINARY_SUPPORT_DERIVATION_VERSION,
        total_coeffs: cursor,
        total_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128OffsetA7F7 as F;

    use crate::schedule::PrecommittedGroupParams;
    use crate::sis::{
        sis_table_key_for_linf_bound, AjtaiKeyParams, SisModulusFamily, DEFAULT_SIS_SECURITY_BITS,
    };
    use crate::{OpeningClaimsLayout, PolynomialGroupLayout, PrecommittedLevelParams};

    fn certified_key(d: usize, raw_bound: u128, cols: usize) -> AjtaiKeyParams {
        let table = sis_table_key_for_linf_bound(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q128,
            d as u32,
            raw_bound,
        )
        .expect("test SIS row");
        AjtaiKeyParams::try_new_with_min_rank(table, cols).expect("test certified key")
    }

    fn level(with_precommitted: bool) -> (LevelParams, OpeningClaimsLayout) {
        let mut lp = LevelParams::params_only(
            SisModulusFamily::Q128,
            64,
            6,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(64),
        );
        lp.b_key = certified_key(32, 63, 1);
        lp.d_key = certified_key(32, 63, 1);
        if !with_precommitted {
            return (lp, OpeningClaimsLayout::new(2, 1).unwrap());
        }
        let group = PolynomialGroupLayout::new(2, 1);
        lp.precommitted_groups.push(PrecommittedLevelParams {
            layout: PrecommittedGroupParams::from_params(group, &lp),
            a_key: lp.a_key.clone(),
            b_key: certified_key(64, 63, 1),
            num_blocks: 1,
            block_len: 1,
            num_digits_commit: 1,
            num_digits_open: 1,
            num_digits_fold_one: 1,
        });
        let opening =
            OpeningClaimsLayout::from_root_groups(&[group], PolynomialGroupLayout::new(2, 1))
                .unwrap();
        (lp, opening)
    }

    fn chain_spec(
        source: CompressionSourceId,
        source_key: &AjtaiKeyParams,
        shapes: &[(usize, CompressionAlphabet)],
    ) -> super::super::CompressionChainSpec {
        let mut previous_output =
            source_key.row_len() * source_key.sis_table_key().ring_dimension as usize;
        let maps = shapes
            .iter()
            .map(|&(d, alphabet)| {
                let depth = super::super::alphabet_facts(alphabet, 128, 6).unwrap();
                let input = previous_output * depth;
                assert!(input.is_multiple_of(d));
                let raw_bound = match alphabet {
                    CompressionAlphabet::NegativeBinary => 1,
                    CompressionAlphabet::OpeningBase { .. } => 63,
                };
                let key = certified_key(d, raw_bound, input / d);
                previous_output = key.row_len() * d;
                super::super::CompressionMapSpec { key, alphabet }
            })
            .collect();
        super::super::CompressionChainSpec { source, maps }
    }

    fn checked_mixed_catalog() -> (LevelParams, super::super::ValidatedCompressionCatalog) {
        let (lp, opening) = level(true);
        let specs = vec![
            chain_spec(
                CompressionSourceId::CurrentOuter,
                &lp.b_key,
                &[
                    (64, CompressionAlphabet::NegativeBinary),
                    (32, CompressionAlphabet::NegativeBinary),
                    (64, CompressionAlphabet::NegativeBinary),
                ],
            ),
            chain_spec(
                CompressionSourceId::PrecommittedOuter { index: 0 },
                &lp.precommitted_groups[0].b_key,
                &[
                    (32, CompressionAlphabet::OpeningBase { log_basis: 6 }),
                    (64, CompressionAlphabet::NegativeBinary),
                ],
            ),
            chain_spec(
                CompressionSourceId::Opening,
                &lp.d_key,
                &[
                    (32, CompressionAlphabet::OpeningBase { log_basis: 6 }),
                    (64, CompressionAlphabet::NegativeBinary),
                ],
            ),
        ];
        let catalog = super::super::validate_and_compile::<F>(
            &lp,
            super::super::CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
            64,
            specs,
        )
        .unwrap();
        (lp, catalog)
    }

    #[test]
    fn checked_catalog_compiles_layer_major_mixed_dimension_semantics() {
        let (_lp, catalog) = checked_mixed_catalog();
        let semantics = catalog.semantics.as_ref().unwrap();
        let xi_ids = semantics
            .segments
            .iter()
            .filter_map(|segment| matches!(segment.id, SegmentId::Xi { .. }).then_some(segment.id))
            .collect::<Vec<_>>();
        assert_eq!(
            xi_ids,
            vec![
                SegmentId::Xi {
                    source: CompressionSourceId::CurrentOuter,
                    map: 0
                },
                SegmentId::Xi {
                    source: CompressionSourceId::PrecommittedOuter { index: 0 },
                    map: 0
                },
                SegmentId::Xi {
                    source: CompressionSourceId::Opening,
                    map: 0
                },
                SegmentId::Xi {
                    source: CompressionSourceId::CurrentOuter,
                    map: 1
                },
                SegmentId::Xi {
                    source: CompressionSourceId::PrecommittedOuter { index: 0 },
                    map: 1
                },
                SegmentId::Xi {
                    source: CompressionSourceId::Opening,
                    map: 1
                },
                SegmentId::Xi {
                    source: CompressionSourceId::CurrentOuter,
                    map: 2
                },
            ]
        );
        assert_eq!(semantics.rows.len(), 7);
        assert_eq!(semantics.augmentations.len(), 3);
        assert_eq!(
            semantics.augmentations[0].source,
            CompressionSourceId::CurrentOuter
        );
        assert!(matches!(
            semantics.rows.last().unwrap().rhs,
            CompressionRowRhs::TerminalPayload { .. }
        ));
    }

    #[test]
    fn quotient_spans_use_active_base_and_native_row_dimension() {
        let (lp, catalog) = checked_mixed_catalog();
        let semantics = catalog.semantics.as_ref().unwrap();
        let levels = r_decomp_levels::<F>(lp.log_basis);
        for row in &semantics.rows {
            let quotient = segment_span(&semantics.segments, row.quotient).unwrap();
            assert_eq!(quotient.len, row.rows.len * row.native_ring_dim * levels);
            assert!(matches!(row.quotient, SegmentId::Quotient { .. }));
        }
    }

    #[test]
    fn negative_binary_inputs_are_exact_ordered_xi_ids() {
        let (_lp, catalog) = checked_mixed_catalog();
        let semantics = catalog.semantics.as_ref().unwrap();
        assert_eq!(
            semantics.negative_binary_inputs,
            vec![
                SegmentId::Xi {
                    source: CompressionSourceId::CurrentOuter,
                    map: 0
                },
                SegmentId::Xi {
                    source: CompressionSourceId::CurrentOuter,
                    map: 1
                },
                SegmentId::Xi {
                    source: CompressionSourceId::PrecommittedOuter { index: 0 },
                    map: 1
                },
                SegmentId::Xi {
                    source: CompressionSourceId::Opening,
                    map: 1
                },
                SegmentId::Xi {
                    source: CompressionSourceId::CurrentOuter,
                    map: 2
                },
            ]
        );
        assert!(semantics
            .negative_binary_inputs
            .iter()
            .all(|id| matches!(id, SegmentId::Xi { .. })));
        assert_eq!(
            semantics.binary_support_derivation_version,
            BINARY_SUPPORT_DERIVATION_VERSION
        );
    }

    #[test]
    fn all_negative_catalog_marks_every_input() {
        let (lp, opening) = level(false);
        let specs = vec![
            chain_spec(
                CompressionSourceId::CurrentOuter,
                &lp.b_key,
                &[(32, CompressionAlphabet::NegativeBinary); 2],
            ),
            chain_spec(
                CompressionSourceId::Opening,
                &lp.d_key,
                &[(32, CompressionAlphabet::NegativeBinary); 2],
            ),
        ];
        let catalog = super::super::validate_and_compile::<F>(
            &lp,
            super::super::CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
            64,
            specs,
        )
        .unwrap();
        let semantics = catalog.semantics.as_ref().unwrap();
        assert_eq!(semantics.negative_binary_inputs.len(), 4);
        assert_eq!(semantics.segments.len(), 8);
    }

    #[test]
    fn standalone_checked_catalog_has_no_proof_semantics() {
        let (lp, _) = level(false);
        let spec = chain_spec(
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[(32, CompressionAlphabet::NegativeBinary); 2],
        );
        let catalog = super::super::validate_and_compile::<F>(
            &lp,
            super::super::CompressionCatalogContext::StandaloneCommitment {
                max_opening_log_basis: 6,
            },
            64,
            vec![spec],
        )
        .unwrap();
        assert!(catalog.semantics.is_none());
    }

    #[test]
    fn allocation_and_view_helpers_reject_malformed_geometry() {
        let mut cursor = DEFAULT_MAX_SEQUENCE_LEN;
        assert!(allocate_span(&mut cursor, 1, "test").is_err());
        assert!(CoeffSpan {
            start: usize::MAX,
            len: 1
        }
        .end()
        .is_err());
        assert!(validate_span_view(CoeffSpan { start: 0, len: 33 }, 32, "test").is_err());
        assert!(validate_span_view(CoeffSpan { start: 0, len: 32 }, 0, "test").is_err());
        assert_eq!(checked_semantic_capacities([1, 2]).unwrap(), (3, 6));
        assert!(checked_semantic_capacities([usize::MAX, 1]).is_err());
        assert!(checked_semantic_capacities([DEFAULT_MAX_SEQUENCE_LEN / 2 + 1]).is_err());
    }
}
