use super::*;

#[test]
fn prefix_search_finds_last_true_value() {
    assert_eq!(
        max_true_in_prefix(1, 16, |value| Ok(value <= 9))
            .unwrap()
            .max_value,
        9
    );
}

#[test]
fn prefix_search_stops_when_the_first_width_is_insecure() {
    let mut probes = 0;
    let result = max_true_in_prefix(1, DEFAULT_SEARCH_CAP, |_| {
        probes += 1;
        Ok(false)
    })
    .unwrap();
    assert_eq!(result.max_value, 0);
    assert_eq!(result.next_value, Some(1));
    assert_eq!(probes, 1);
}

#[test]
fn certificate_search_brackets_distant_boundaries() {
    for (cap, hint, boundary) in [
        (1_000_000, 3, 800_000),
        (1_000_000, 900_000, 17),
        (1_000_000, 900_000, 0),
        (1_000_000, 3, 1_000_000),
    ] {
        let mut probes = 0;
        let result = certified_boundary_from_hint(cap, hint, |value| {
            probes += 1;
            Ok(value <= boundary)
        })
        .unwrap();
        assert_eq!(result.max_value, boundary);
        assert_eq!(result.hit_cap, boundary == cap);
        assert_eq!(result.next_value, (boundary < cap).then_some(boundary + 1));
        assert!(
            probes < 64,
            "hint={hint} boundary={boundary} probes={probes}"
        );
    }
}

#[test]
fn infinity_never_counts_as_secure() {
    assert!(!security_met(CostValue::Infinity, 128.0));
    assert!(secure_or_error(CostValue::Infinity, 128.0).is_err());
}

#[test]
fn csv_has_no_classical_columns() {
    assert!(!InfinityWidthRow::csv_header().contains("classical"));
}

#[test]
fn production_config_requires_the_certified_profile() {
    let mut config = InfinityWidthTableConfig::default();
    assert!(is_production_infinity_width_table_config(&config));
    config.profile = InfinityWidthProfile::LatticeEstimatorParity;
    assert!(!is_production_infinity_width_table_config(&config));
    config.profile = InfinityWidthProfile::ExhaustiveSerial;
    assert!(!is_production_infinity_width_table_config(&config));
}

#[test]
fn runtime_table_emits_direct_q128_d512_rows() {
    let rows = [10, 20, 30, 40]
        .into_iter()
        .enumerate()
        .map(|(index, max_width)| InfinityWidthRow {
            modulus_profile: AkitaModulusProfileId::Q128OffsetA7F7,
            d: 512,
            rank: u32::try_from(index + 1).unwrap(),
            coeff_linf_bound: 2,
            max_width,
            policy: SisSecurityPolicy::Quantum128BitADPS16,
            search_cap: DEFAULT_SEARCH_CAP,
            hit_cap: false,
            profile: InfinityWidthProfile::LocalMinimum,
            max_costs: None,
            next_costs: None,
        })
        .collect::<Vec<_>>();
    let runtime_rows = runtime_width_rows(&rows, 4).unwrap();
    assert_eq!(
        runtime_rows,
        [RuntimeWidthRow {
            modulus_profile: AkitaModulusProfileId::Q128OffsetA7F7,
            d: 512,
            coeff_linf_bound: 2,
            widths: vec![10, 20, 30, 40],
        }]
    );
}

#[test]
fn generation_filters_to_canonical_production_and_compression_cells() {
    assert!(scalar_origin_is_canonical(
        AkitaModulusProfileId::Q128OffsetA7F7,
        64,
        15
    ));
    assert!(!scalar_origin_is_canonical(
        AkitaModulusProfileId::Q128OffsetA7F7,
        32,
        2
    ));
    assert!(scalar_origin_is_canonical(
        AkitaModulusProfileId::Q128OffsetA7F7,
        64,
        2
    ));
    assert!(scalar_origin_is_canonical(
        AkitaModulusProfileId::Q128OffsetA7F7,
        512,
        2
    ));
    assert!(scalar_origin_is_canonical(
        AkitaModulusProfileId::Q64Offset59,
        512,
        2
    ));
    assert!(scalar_origin_is_canonical(
        AkitaModulusProfileId::Q32Offset99,
        512,
        3
    ));
    assert!(!scalar_origin_is_canonical(
        AkitaModulusProfileId::Q128OffsetA7F7,
        16,
        15
    ));
    assert!(scalar_origin_is_canonical(
        AkitaModulusProfileId::Q128OffsetA7F7,
        16,
        1
    ));
    assert!(scalar_origin_is_canonical(
        AkitaModulusProfileId::Q64Offset59,
        16,
        1
    ));
    assert!(scalar_origin_is_canonical(
        AkitaModulusProfileId::Q32Offset99,
        32,
        1
    ));
    assert!(!scalar_origin_is_canonical(
        AkitaModulusProfileId::Q128OffsetA7F7,
        32,
        1
    ));
    assert!(!scalar_origin_is_canonical(
        AkitaModulusProfileId::Q64Offset59,
        64,
        1
    ));
    assert!(!scalar_origin_is_canonical(
        AkitaModulusProfileId::Q32Offset99,
        128,
        1
    ));
}

#[test]
fn q128_d512_rows_are_estimated_directly() {
    let config = InfinityWidthTableConfig {
        profiles: vec![AkitaModulusProfileId::Q128OffsetA7F7],
        ring_dims: vec![512],
        coeff_linf_bounds: vec![67_108_863],
        max_rank: 2,
        search_cap: Some(100_000),
        profile: InfinityWidthProfile::LatticeEstimatorParity,
        ..InfinityWidthTableConfig::default()
    };
    let rows = generate_infinity_width_rows(&config).unwrap();
    validate_infinity_width_rows(&rows).unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|row| row.d == 512));
    assert_eq!(rows[0].max_width, 94_477);
    assert_eq!(rows[1].max_width, 100_000);
}
