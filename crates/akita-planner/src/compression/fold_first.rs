//! Fold-first co-generated F/H bundle selection.
//!
//! At level `L`, one byte ladder supplies the same map geometry to current F and
//! opening H (no F×H Cartesian product). The checked catalog alone derives the
//! physical successor witness. Parent wire pricing uses the child's first F
//! projection through [`CompressedFoldWirePayload`], never by appending
//! successor-F digits into the current witness.

use akita_field::{AkitaError, CanonicalField};
use akita_types::{
    compression_digit_depth, field_bytes, protocol_dispatch_tier, slot_dims_for_tier,
    CompressedFoldWirePayload, CompressionAlphabet, CompressionCatalogContext,
    CompressionCatalogProjection, CompressionSourceId, LevelParams, OpeningClaimsLayout,
    ProtocolDispatchSlot, ValidatedCompressionCatalog,
};

use super::replay::{
    derive_compression_key, replay_compression_catalog, validate_replay_policy,
    CompressionChainDescriptor, CompressionMapDescriptor,
};
use super::{
    rank_one_ring_dim_for_bytes, CompressionByteLadder, CompressionFirstMapAlphabet,
    CompressionPlannerPolicy, DEFAULT_COMPRESSION_BYTE_LADDERS,
};
use crate::PlannerPolicy;

/// Soft ceiling on guided bundle attempts per level (shipped ladder count).
pub const MAX_FOLD_FIRST_BUNDLE_ATTEMPTS: usize = DEFAULT_COMPRESSION_BYTE_LADDERS.len();

/// Instrumentation for fold-first bundle selection regressions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FoldFirstSearchStats {
    pub ladders_tried: usize,
    pub bundles_accepted: usize,
    pub exhaustive_map_shapes_tried: usize,
}

/// One checked co-generated or terminal compression bundle for a fold level.
#[derive(Debug, Clone)]
pub struct LevelCompressionBundle {
    catalog: ValidatedCompressionCatalog,
    projection: CompressionCatalogProjection,
    selected_ladder: Option<CompressionByteLadder>,
    map_ring_dimensions: Vec<usize>,
    first_alphabet: CompressionAlphabet,
    successor_witness_field_coeffs: usize,
}

impl LevelCompressionBundle {
    #[must_use]
    pub fn catalog(&self) -> &ValidatedCompressionCatalog {
        &self.catalog
    }

    #[must_use]
    pub fn projection(&self) -> &CompressionCatalogProjection {
        &self.projection
    }

    #[must_use]
    pub fn selected_ladder(&self) -> Option<CompressionByteLadder> {
        self.selected_ladder
    }

    #[must_use]
    pub fn map_ring_dimensions(&self) -> &[usize] {
        &self.map_ring_dimensions
    }

    #[must_use]
    pub fn first_alphabet(&self) -> CompressionAlphabet {
        self.first_alphabet
    }

    #[must_use]
    pub fn successor_witness_field_coeffs(&self) -> usize {
        self.successor_witness_field_coeffs
    }
}

/// Physical successor witness length in field coefficients from a checked
/// co-generated or terminal catalog. This is the fold-first authority: it does
/// not consult a successor level's F plan.
pub fn successor_witness_field_coeffs(
    catalog: &ValidatedCompressionCatalog,
) -> Result<usize, AkitaError> {
    catalog
        .fold_relation_layout()?
        .physical_witness_field_coeff_len()
}

/// Parent fold wire projection: current H payload + child F payload.
pub fn fold_first_wire_payload(
    current: &LevelCompressionBundle,
    successor: &LevelCompressionBundle,
    next_base_field_bits: u32,
) -> Result<CompressedFoldWirePayload, AkitaError> {
    CompressedFoldWirePayload::from_catalogs(
        current.projection(),
        successor.projection(),
        next_base_field_bits,
    )
}

fn first_alphabet(
    kind: CompressionFirstMapAlphabet,
    co_generated_log_basis: u32,
) -> CompressionAlphabet {
    match kind {
        CompressionFirstMapAlphabet::NegativeBinary => CompressionAlphabet::NegativeBinary,
        CompressionFirstMapAlphabet::OpeningBase => CompressionAlphabet::OpeningBase {
            log_basis: co_generated_log_basis,
        },
    }
}

fn source_output_coeffs(
    lp: &LevelParams,
    source: CompressionSourceId,
) -> Result<usize, AkitaError> {
    let key = match source {
        CompressionSourceId::CurrentOuter => &lp.b_key,
        CompressionSourceId::Opening => &lp.d_key,
        CompressionSourceId::PrecommittedOuter { .. } => {
            return Err(AkitaError::InvalidSetup(
                "fold-first bundles do not materialize precommitted F in this slice".into(),
            ));
        }
    };
    let d = key.sis_table_key().ring_dimension as usize;
    key.row_len()
        .checked_mul(d)
        .ok_or_else(|| AkitaError::InvalidSetup("compression source output overflow".into()))
}

fn ladder_maps(
    policy: &PlannerPolicy,
    field_bits: u32,
    field_element_bytes: usize,
    dimensions: &[usize],
    ladder: CompressionByteLadder,
    co_generated_log_basis: u32,
) -> Option<Vec<CompressionMapDescriptor>> {
    let images = ladder.image_bytes();
    if images.len() < 2 || images.len() > 3 {
        return None;
    }
    if images.windows(2).any(|pair| pair[0] <= pair[1]) {
        return None;
    }
    let first = first_alphabet(ladder.first_alphabet(), co_generated_log_basis);
    let mut maps = Vec::with_capacity(images.len());
    for (index, &target_bytes) in images.iter().enumerate() {
        let alphabet = if index == 0 {
            first
        } else {
            CompressionAlphabet::NegativeBinary
        };
        let d = rank_one_ring_dim_for_bytes(target_bytes, field_element_bytes)?;
        if !dimensions.contains(&d) || !policy.ring_dimension.is_multiple_of(d) {
            return None;
        }
        maps.push(CompressionMapDescriptor {
            ring_d: d,
            alphabet,
        });
    }
    // Digit-depth well-formedness for the chosen alphabets.
    for map in &maps {
        compression_digit_depth(map.alphabet, field_bits, policy.basis_range.1).ok()?;
    }
    Some(maps)
}

fn chain_is_rank_one_exact(
    policy: &PlannerPolicy,
    field_bits: u32,
    field_element_bytes: usize,
    source_output_coeffs: usize,
    maps: &[CompressionMapDescriptor],
    image_bytes: &[usize],
) -> bool {
    if maps.len() != image_bytes.len() {
        return false;
    }
    let mut previous = source_output_coeffs;
    for (map, &target_bytes) in maps.iter().zip(image_bytes.iter()) {
        let Ok(digit_depth) =
            compression_digit_depth(map.alphabet, field_bits, policy.basis_range.1)
        else {
            return false;
        };
        let Some(input_coeffs) = previous.checked_mul(digit_depth) else {
            return false;
        };
        if !input_coeffs.is_multiple_of(map.ring_d) {
            return false;
        }
        let Some(key) = derive_compression_key(
            policy,
            policy.basis_range.1,
            map.alphabet,
            map.ring_d,
            input_coeffs / map.ring_d,
        ) else {
            return false;
        };
        if key.row_len() != 1 {
            return false;
        }
        let Some(output_coeffs) = key.row_len().checked_mul(map.ring_d) else {
            return false;
        };
        let Some(output_bytes) = output_coeffs.checked_mul(field_element_bytes) else {
            return false;
        };
        if output_bytes != target_bytes {
            return false;
        }
        previous = output_coeffs;
    }
    true
}

fn finish_bundle<F: CanonicalField>(
    policy: &PlannerPolicy,
    lp: &LevelParams,
    context: CompressionCatalogContext<'_>,
    descriptors: &[CompressionChainDescriptor],
    selected_ladder: Option<CompressionByteLadder>,
    maps: &[CompressionMapDescriptor],
) -> Result<Option<LevelCompressionBundle>, AkitaError> {
    let catalog = match replay_compression_catalog::<F>(policy, lp, context, descriptors) {
        Ok(catalog) => catalog,
        Err(err) if akita_types::compression_capacity_infeasible(&err) => {
            return Ok(None);
        }
        Err(err) => return Err(err),
    };
    let projection = catalog.project_for_schedule()?;
    let successor_witness_field_coeffs = successor_witness_field_coeffs(&catalog)?;
    let first_alphabet = maps
        .first()
        .map(|map| map.alphabet)
        .ok_or_else(|| AkitaError::InvalidSetup("compression bundle has no maps".into()))?;
    Ok(Some(LevelCompressionBundle {
        catalog,
        projection,
        selected_ladder,
        map_ring_dimensions: maps.iter().map(|map| map.ring_d).collect(),
        first_alphabet,
        successor_witness_field_coeffs,
    }))
}

#[derive(Clone, Copy)]
struct SharedMapsContext<'a> {
    policy: &'a PlannerPolicy,
    lp: &'a LevelParams,
    opening: &'a OpeningClaimsLayout,
    field_bits: u32,
    field_element_bytes: usize,
}

fn try_shared_maps_co_generated<F: CanonicalField>(
    context: SharedMapsContext<'_>,
    maps: &[CompressionMapDescriptor],
    image_bytes: &[usize],
    selected_ladder: Option<CompressionByteLadder>,
) -> Result<Option<LevelCompressionBundle>, AkitaError> {
    let outer = source_output_coeffs(context.lp, CompressionSourceId::CurrentOuter)?;
    let opening_src = source_output_coeffs(context.lp, CompressionSourceId::Opening)?;
    if !chain_is_rank_one_exact(
        context.policy,
        context.field_bits,
        context.field_element_bytes,
        outer,
        maps,
        image_bytes,
    ) || !chain_is_rank_one_exact(
        context.policy,
        context.field_bits,
        context.field_element_bytes,
        opening_src,
        maps,
        image_bytes,
    ) {
        return Ok(None);
    }
    let max_opening = context.policy.basis_range.1;
    let descriptors = [
        CompressionChainDescriptor {
            source: CompressionSourceId::CurrentOuter,
            max_opening_log_basis: max_opening,
            maps: maps.to_vec(),
        },
        CompressionChainDescriptor {
            source: CompressionSourceId::Opening,
            max_opening_log_basis: max_opening,
            maps: maps.to_vec(),
        },
    ];
    finish_bundle::<F>(
        context.policy,
        context.lp,
        CompressionCatalogContext::CoGeneratedLevel {
            opening: context.opening,
        },
        &descriptors,
        selected_ladder,
        maps,
    )
}

fn try_shared_maps_terminal<F: CanonicalField>(
    context: SharedMapsContext<'_>,
    maps: &[CompressionMapDescriptor],
    image_bytes: &[usize],
    selected_ladder: Option<CompressionByteLadder>,
) -> Result<Option<LevelCompressionBundle>, AkitaError> {
    let outer = source_output_coeffs(context.lp, CompressionSourceId::CurrentOuter)?;
    if !chain_is_rank_one_exact(
        context.policy,
        context.field_bits,
        context.field_element_bytes,
        outer,
        maps,
        image_bytes,
    ) {
        return Ok(None);
    }
    let descriptors = [CompressionChainDescriptor {
        source: CompressionSourceId::CurrentOuter,
        max_opening_log_basis: context.policy.basis_range.1,
        maps: maps.to_vec(),
    }];
    finish_bundle::<F>(
        context.policy,
        context.lp,
        CompressionCatalogContext::TerminalFold {
            opening: context.opening,
        },
        &descriptors,
        selected_ladder,
        maps,
    )
}

/// Iterate guided co-generated bundles in policy order.
///
/// `Ok(None)` is an expected ladder miss from size, rank-one, or dispatch
/// constraints. Once those checks pass, replay and projection errors propagate.
pub(crate) fn iter_co_generated_bundles<'a, F: CanonicalField + 'a>(
    policy: &'a PlannerPolicy,
    compression_policy: &'a CompressionPlannerPolicy,
    lp: &'a LevelParams,
    opening: &'a OpeningClaimsLayout,
    stats: &'a mut FoldFirstSearchStats,
) -> Result<impl Iterator<Item = Result<Option<LevelCompressionBundle>, AkitaError>> + 'a, AkitaError>
{
    validate_replay_policy::<F>(policy, lp)?;
    let tier = protocol_dispatch_tier::<F>();
    let dimensions = slot_dims_for_tier(tier, ProtocolDispatchSlot::Compression);
    let field_bits = F::modulus_bits();
    let field_element_bytes = field_bytes(field_bits);
    let co_generated_log_basis = lp.log_basis;

    Ok(compression_policy
        .ladders()
        .iter()
        .copied()
        .map(move |ladder| {
            stats.ladders_tried = stats.ladders_tried.saturating_add(1);
            let Some(maps) = ladder_maps(
                policy,
                field_bits,
                field_element_bytes,
                dimensions,
                ladder,
                co_generated_log_basis,
            ) else {
                return Ok(None);
            };
            let bundle = try_shared_maps_co_generated::<F>(
                SharedMapsContext {
                    policy,
                    lp,
                    opening,
                    field_bits,
                    field_element_bytes,
                },
                &maps,
                ladder.image_bytes(),
                Some(ladder),
            )?;
            if bundle.is_some() {
                stats.bundles_accepted = stats.bundles_accepted.saturating_add(1);
            }
            Ok(bundle)
        }))
}

/// Select a co-generated F/H bundle with one shared ladder (no Cartesian product).
pub fn select_co_generated_bundle<F: CanonicalField>(
    policy: &PlannerPolicy,
    compression_policy: &CompressionPlannerPolicy,
    lp: &LevelParams,
    opening: &OpeningClaimsLayout,
    stats: &mut FoldFirstSearchStats,
) -> Result<LevelCompressionBundle, AkitaError> {
    let tier = protocol_dispatch_tier::<F>();
    let dimensions = slot_dims_for_tier(tier, ProtocolDispatchSlot::Compression);
    let field_bits = F::modulus_bits();
    let field_element_bytes = field_bytes(field_bits);
    // Co-generated first maps require b_cmp = b_range.
    let co_generated_log_basis = lp.log_basis;

    {
        for bundle in
            iter_co_generated_bundles::<F>(policy, compression_policy, lp, opening, stats)?
        {
            if let Some(bundle) = bundle? {
                return Ok(bundle);
            }
        }
    }

    if !compression_policy.allow_exhaustive_fallback() {
        return Err(AkitaError::InvalidSetup(
            "no co-generated compression byte ladder materialized and exhaustive fallback is disabled"
                .into(),
        ));
    }

    // Exhaustive over admitted shared map shapes (same maps for F and H).
    for depth in 2..=3 {
        for first_kind in [
            CompressionFirstMapAlphabet::OpeningBase,
            CompressionFirstMapAlphabet::NegativeBinary,
        ] {
            let first = first_alphabet(first_kind, co_generated_log_basis);
            let mut stack: Vec<Vec<CompressionMapDescriptor>> = vec![Vec::new()];
            while let Some(prefix) = stack.pop() {
                if prefix.len() == depth {
                    stats.exhaustive_map_shapes_tried =
                        stats.exhaustive_map_shapes_tried.saturating_add(1);
                    let image_bytes = {
                        // Reconstruct exact byte targets from rank-one dims.
                        let mut bytes = Vec::with_capacity(prefix.len());
                        let mut ok = true;
                        for map in &prefix {
                            let Some(target) = map.ring_d.checked_mul(field_element_bytes) else {
                                ok = false;
                                break;
                            };
                            bytes.push(target);
                        }
                        if !ok || bytes.windows(2).any(|pair| pair[0] <= pair[1]) {
                            continue;
                        }
                        bytes
                    };
                    if let Some(bundle) = try_shared_maps_co_generated::<F>(
                        SharedMapsContext {
                            policy,
                            lp,
                            opening,
                            field_bits,
                            field_element_bytes,
                        },
                        &prefix,
                        &image_bytes,
                        None,
                    )? {
                        stats.bundles_accepted = stats.bundles_accepted.saturating_add(1);
                        return Ok(bundle);
                    }
                    continue;
                }
                let alphabet = if prefix.is_empty() {
                    first
                } else {
                    CompressionAlphabet::NegativeBinary
                };
                for &d in dimensions {
                    if !policy.ring_dimension.is_multiple_of(d) {
                        continue;
                    }
                    let mut next = prefix.clone();
                    next.push(CompressionMapDescriptor {
                        ring_d: d,
                        alphabet,
                    });
                    stack.push(next);
                }
            }
        }
    }

    Err(AkitaError::InvalidSetup(
        "no co-generated compression bundle is covered by current dispatch and SIS tables".into(),
    ))
}

/// Iterate guided terminal (WithoutDBlock, F-only) bundles in policy order.
///
/// Semantics match [`iter_co_generated_bundles`]: `Ok(None)` is an expected
/// ladder miss; capacity-style replay misses are also `Ok(None)`; other replay
/// and projection errors propagate.
pub(crate) fn iter_terminal_bundles<'a, F: CanonicalField + 'a>(
    policy: &'a PlannerPolicy,
    compression_policy: &'a CompressionPlannerPolicy,
    lp: &'a LevelParams,
    opening: &'a OpeningClaimsLayout,
    stats: &'a mut FoldFirstSearchStats,
) -> Result<impl Iterator<Item = Result<Option<LevelCompressionBundle>, AkitaError>> + 'a, AkitaError>
{
    validate_replay_policy::<F>(policy, lp)?;
    let tier = protocol_dispatch_tier::<F>();
    let dimensions = slot_dims_for_tier(tier, ProtocolDispatchSlot::Compression);
    let field_bits = F::modulus_bits();
    let field_element_bytes = field_bytes(field_bits);
    let co_generated_log_basis = lp.log_basis;

    Ok(compression_policy
        .ladders()
        .iter()
        .copied()
        .map(move |ladder| {
            stats.ladders_tried = stats.ladders_tried.saturating_add(1);
            let Some(maps) = ladder_maps(
                policy,
                field_bits,
                field_element_bytes,
                dimensions,
                ladder,
                co_generated_log_basis,
            ) else {
                return Ok(None);
            };
            let bundle = try_shared_maps_terminal::<F>(
                SharedMapsContext {
                    policy,
                    lp,
                    opening,
                    field_bits,
                    field_element_bytes,
                },
                &maps,
                ladder.image_bytes(),
                Some(ladder),
            )?;
            if bundle.is_some() {
                stats.bundles_accepted = stats.bundles_accepted.saturating_add(1);
            }
            Ok(bundle)
        }))
}

/// Select a terminal (WithoutDBlock) F-only bundle from the same ladder list.
pub fn select_terminal_bundle<F: CanonicalField>(
    policy: &PlannerPolicy,
    compression_policy: &CompressionPlannerPolicy,
    lp: &LevelParams,
    opening: &OpeningClaimsLayout,
    stats: &mut FoldFirstSearchStats,
) -> Result<LevelCompressionBundle, AkitaError> {
    for bundle in iter_terminal_bundles::<F>(policy, compression_policy, lp, opening, stats)? {
        if let Some(bundle) = bundle? {
            return Ok(bundle);
        }
    }

    if !compression_policy.allow_exhaustive_fallback() {
        return Err(AkitaError::InvalidSetup(
            "no terminal compression byte ladder materialized and exhaustive fallback is disabled"
                .into(),
        ));
    }

    Err(AkitaError::InvalidSetup(
        "no terminal compression byte ladder materialized; terminal exhaustive fallback is requested but not implemented"
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::{BINARY_256_128, OPENING_BASE_512_256_128};
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::sis::rounded_up_collision_inf_norm;
    use akita_types::{
        AjtaiKeyParams, ChunkedWitnessCfg, DecompositionParams, SisModulusFamily,
        DEFAULT_SIS_SECURITY_BITS,
    };

    fn fixture() -> (PlannerPolicy, LevelParams, OpeningClaimsLayout) {
        let policy = PlannerPolicy {
            ring_dimension: 64,
            decomposition: DecompositionParams {
                log_basis: 4,
                log_commit_bound: 1,
                log_open_bound: Some(4),
            },
            sis_family: SisModulusFamily::Q128,
            min_sis_security_bits: DEFAULT_SIS_SECURITY_BITS,
            ring_subfield_norm_bound: 1,
            claim_ext_degree: 1,
            chal_ext_degree: 1,
            basis_range: (4, 4),
            onehot_chunk_size: 1,
            witness_chunk: ChunkedWitnessCfg::default(),
        };
        let bucket =
            rounded_up_collision_inf_norm(DEFAULT_SIS_SECURITY_BITS, SisModulusFamily::Q128, 32, 4)
                .expect("bucket");
        let key = AjtaiKeyParams::try_new_with_min_rank(
            akita_types::SisTableKey {
                min_security_bits: DEFAULT_SIS_SECURITY_BITS,
                family: SisModulusFamily::Q128,
                ring_dimension: 32,
                coeff_linf_bound: bucket,
            },
            8,
        )
        .expect("key");
        let mut lp = LevelParams::params_only(
            SisModulusFamily::Q128,
            64,
            4,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(64),
        )
        .with_decomp(1, 1, 1, 1, 0)
        .expect("relation fixture");
        lp.b_key = key.clone();
        lp.d_key = key;
        lp.stamp_role_dims_from_keys();
        let opening = OpeningClaimsLayout::new(4, 1).expect("opening");
        (policy, lp, opening)
    }

    #[test]
    fn co_generated_bundle_uses_one_shared_ladder_for_f_and_h() {
        let (policy, lp, opening) = fixture();
        let mut stats = FoldFirstSearchStats::default();
        let bundle = select_co_generated_bundle::<Prime128OffsetA7F7>(
            &policy,
            &CompressionPlannerPolicy::default(),
            &lp,
            &opening,
            &mut stats,
        )
        .expect("bundle");
        assert_eq!(bundle.selected_ladder(), Some(BINARY_256_128));
        assert_eq!(bundle.map_ring_dimensions(), &[16, 8]);
        assert_eq!(
            bundle
                .projection()
                .payload_coeffs(CompressionSourceId::CurrentOuter),
            bundle
                .projection()
                .payload_coeffs(CompressionSourceId::Opening)
        );
        assert!(stats.ladders_tried <= MAX_FOLD_FIRST_BUNDLE_ATTEMPTS);
        assert_eq!(stats.bundles_accepted, 1);
        assert_eq!(stats.exhaustive_map_shapes_tried, 0);
    }

    #[test]
    fn successor_witness_comes_from_current_catalog_not_child_f() {
        let (policy, lp, opening) = fixture();
        let mut stats = FoldFirstSearchStats::default();
        let bundle = select_co_generated_bundle::<Prime128OffsetA7F7>(
            &policy,
            &CompressionPlannerPolicy::default(),
            &lp,
            &opening,
            &mut stats,
        )
        .expect("bundle");
        let from_helper = successor_witness_field_coeffs(bundle.catalog()).expect("len");
        assert_eq!(from_helper, bundle.successor_witness_field_coeffs());
        let layout = bundle
            .catalog()
            .co_generated_relation_layout()
            .expect("co-generated");
        let cost = layout.compression_structural_cost().expect("cost");
        // Compression inputs/quotients for current F and H are present once.
        assert!(cost.witness_coeffs() > 0);
        // Physical length is the fold-first successor size.
        assert_eq!(
            layout.physical_witness_field_coeff_len().expect("physical"),
            bundle.successor_witness_field_coeffs()
        );
    }

    #[test]
    fn parent_wire_prices_child_f_not_native_b() {
        let (policy, lp, opening) = fixture();
        let mut stats = FoldFirstSearchStats::default();
        let current = select_co_generated_bundle::<Prime128OffsetA7F7>(
            &policy,
            &CompressionPlannerPolicy::default(),
            &lp,
            &opening,
            &mut stats,
        )
        .expect("current");
        let successor = select_co_generated_bundle::<Prime128OffsetA7F7>(
            &policy,
            &CompressionPlannerPolicy::default(),
            &lp,
            &opening,
            &mut stats,
        )
        .expect("successor");
        let wire = fold_first_wire_payload(&current, &successor, 128).expect("wire");
        let expected = CompressedFoldWirePayload::from_catalogs(
            current.projection(),
            successor.projection(),
            128,
        )
        .expect("expected");
        assert_eq!(wire, expected);
        let child_f = successor
            .projection()
            .payload_coeffs(CompressionSourceId::CurrentOuter)
            .expect("child F");
        let raw_b = lp.b_key.row_len() * lp.b_key.sis_table_key().ring_dimension as usize;
        assert!(child_f > 0);
        assert_ne!(
            child_f, raw_b,
            "child F payload must be the compressed terminal image, not raw B"
        );
        // Successor-F digits are priced on the wire, not added into the current
        // catalog's physical witness length.
        assert_ne!(
            current.successor_witness_field_coeffs(),
            child_f,
            "current successor witness length is not the child F payload alone"
        );
    }

    #[test]
    fn dead_end_ladder_advances_without_cartesian_product() {
        let (policy, lp, opening) = fixture();
        const DEAD_THEN_BINARY: &[CompressionByteLadder] = &[
            CompressionByteLadder::new(CompressionFirstMapAlphabet::NegativeBinary, &[64, 32]),
            BINARY_256_128,
        ];
        let mut stats = FoldFirstSearchStats::default();
        let bundle = select_co_generated_bundle::<Prime128OffsetA7F7>(
            &policy,
            &CompressionPlannerPolicy::with_ladders(128, DEAD_THEN_BINARY, false),
            &lp,
            &opening,
            &mut stats,
        )
        .expect("second ladder");
        assert_eq!(bundle.selected_ladder(), Some(BINARY_256_128));
        assert_eq!(stats.ladders_tried, 2);
        assert_eq!(stats.exhaustive_map_shapes_tried, 0);
    }

    #[test]
    fn co_generated_iterator_yields_every_materialized_ladder() {
        let (policy, lp, opening) = fixture();
        const TWO_MATERIALIZED: &[CompressionByteLadder] = &[BINARY_256_128, BINARY_256_128];
        let compression = CompressionPlannerPolicy::with_ladders(128, TWO_MATERIALIZED, false);
        let mut stats = FoldFirstSearchStats::default();
        let bundles = iter_co_generated_bundles::<Prime128OffsetA7F7>(
            &policy,
            &compression,
            &lp,
            &opening,
            &mut stats,
        )
        .expect("iterator")
        .collect::<Result<Vec<_>, _>>()
        .expect("validated ladders");
        assert_eq!(bundles.iter().filter(|bundle| bundle.is_some()).count(), 2);
        assert_eq!(stats.ladders_tried, 2);
        assert_eq!(stats.bundles_accepted, 2);
    }

    #[test]
    fn opening_base_ladder_uses_level_log_basis_not_standalone_freeze() {
        let (policy, lp, opening) = fixture();
        let mut stats = FoldFirstSearchStats::default();
        // May miss on this fixture; if it hits, first alphabet log_basis must be 4.
        let result = select_co_generated_bundle::<Prime128OffsetA7F7>(
            &policy,
            &CompressionPlannerPolicy::with_ladders(128, &[OPENING_BASE_512_256_128], true),
            &lp,
            &opening,
            &mut stats,
        );
        if let Ok(bundle) = result {
            assert!(matches!(
                bundle.first_alphabet(),
                CompressionAlphabet::OpeningBase { log_basis: 4 }
            ));
        }
    }
}
