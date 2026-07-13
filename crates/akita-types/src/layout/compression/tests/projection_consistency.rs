use super::*;

fn mixed_dimension_chain(
    source: CompressionSourceId,
    source_key: &AjtaiKeyParams,
    dimensions: &[usize],
) -> CompressionChainSpec {
    let mut output_coeffs =
        source_key.row_len() * source_key.sis_table_key().ring_dimension as usize;
    let maps = dimensions
        .iter()
        .copied()
        .map(|d| {
            let input_coeffs = output_coeffs * 128;
            let map_key = key(SisModulusFamily::Q128, d, 1, input_coeffs / d);
            output_coeffs = map_key.row_len() * d;
            CompressionMapSpec::new(map_key, CompressionAlphabet::NegativeBinary)
        })
        .collect();
    CompressionChainSpec::new(source, 6, maps)
}

#[test]
fn rejects_equal_geometry_alphabet_cross_wiring() {
    let lp = level();
    let opening = scalar_opening();
    let mut catalog = validate_compression_catalog::<Prime128OffsetA7F7>(
        &lp,
        CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
        64,
        vec![
            chain_for(
                &lp,
                CompressionSourceId::CurrentOuter,
                &lp.b_key,
                &[CompressionAlphabet::NegativeBinary; 2],
            ),
            chain_for(
                &lp,
                CompressionSourceId::Opening,
                &lp.d_key,
                &[CompressionAlphabet::NegativeBinary; 2],
            ),
        ],
    )
    .unwrap();
    // At base one, opening-base and negative-binary both use 128 digits and
    // bound one. Geometry and SIS keys therefore remain equal; only canonical
    // relation support distinguishes the semantics.
    catalog.chains[0].maps[0].alphabet = CompressionAlphabet::OpeningBase { log_basis: 1 };
    assert!(catalog.project_for_schedule().is_err());
}

#[test]
fn rejects_cross_wired_source_order() {
    let lp = level();
    let opening = scalar_opening();
    let mut catalog = validate_compression_catalog::<Prime128OffsetA7F7>(
        &lp,
        CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
        64,
        current_and_opening_specs(&lp),
    )
    .unwrap();
    catalog.chains.swap(0, 1);
    assert!(catalog.project_for_schedule().is_err());
}

#[test]
fn rejects_swapped_per_source_terminal_payloads_with_equal_total() {
    let lp = level();
    let opening = scalar_opening();
    let mut catalog = validate_compression_catalog::<Prime128OffsetA7F7>(
        &lp,
        CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
        64,
        vec![
            mixed_dimension_chain(CompressionSourceId::CurrentOuter, &lp.b_key, &[32, 32]),
            mixed_dimension_chain(CompressionSourceId::Opening, &lp.d_key, &[64, 64]),
        ],
    )
    .unwrap();
    assert_ne!(
        catalog.chains[0].payload_coeffs,
        catalog.chains[1].payload_coeffs
    );
    let original_total = catalog.chains[0].payload_coeffs + catalog.chains[1].payload_coeffs;
    let first = catalog.chains[0].payload_coeffs;
    catalog.chains[0].payload_coeffs = catalog.chains[1].payload_coeffs;
    catalog.chains[1].payload_coeffs = first;
    assert_eq!(
        catalog.chains[0].payload_coeffs + catalog.chains[1].payload_coeffs,
        original_total
    );
    assert!(catalog.project_for_schedule().is_err());
}
