use super::*;
#[test]
fn compiles_whole_catalog_and_derives_geometry() {
    let lp = level();
    let catalog = validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        CompressionCatalogContext::CoGeneratedLevel {
            opening: &scalar_opening(),
        },
        64,
        current_and_opening_specs(&lp),
    )
    .unwrap();
    assert_eq!(catalog.chains.len(), 2);
    assert_eq!(catalog.chains[0].source, CompressionSourceId::CurrentOuter);
    assert_eq!(catalog.chains[1].source, CompressionSourceId::Opening);
    assert_eq!(catalog.chains[0].maps.len(), 2);
    assert_eq!(catalog.chains[0].maps[0].digit_depth, 22);
    assert_eq!(catalog.chains[0].maps[1].digit_depth, 128);
    assert_eq!(
        catalog.chains[0].maps[0].key.sis_table_key().ring_dimension as usize,
        D
    );
    assert_eq!(
        catalog.chains[0].maps[0].input_coeffs,
        catalog.chains[0].source_output_coeffs * 22
    );
    assert_eq!(
        catalog.chains[0].payload_coeffs,
        catalog.chains[0].maps[1].output_coeffs
    );
}

#[test]
fn resolves_current_precommitted_and_opening_sources_in_canonical_order() {
    let mut lp = level();
    let pre_b = key(SisModulusFamily::Q128, D, 63, 2);
    let group = PolynomialGroupLayout::new(3, 1);
    lp.precommitted_groups.push(PrecommittedLevelParams {
        layout: PrecommittedGroupParams::from_params(group, &lp),
        a_key: lp.a_key.clone(),
        b_key: pre_b.clone(),
        num_blocks: 1,
        block_len: 1,
        num_digits_commit: 1,
        num_digits_open: 1,
        num_digits_fold_one: 1,
    });
    let opening =
        OpeningClaimsLayout::from_root_groups(&[group], PolynomialGroupLayout::new(4, 1)).unwrap();
    let specs = vec![
        chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        ),
        chain_for(
            &lp,
            CompressionSourceId::PrecommittedOuter { index: 0 },
            &pre_b,
            &[
                CompressionAlphabet::OpeningBase { log_basis: 4 },
                CompressionAlphabet::NegativeBinary,
            ],
        ),
        chain_for(
            &lp,
            CompressionSourceId::Opening,
            &lp.d_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        ),
    ];
    let catalog = validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
        64,
        specs,
    )
    .unwrap();
    assert_eq!(catalog.chains.len(), 3);
    assert_eq!(
        catalog.chains[1].source,
        CompressionSourceId::PrecommittedOuter { index: 0 }
    );
    let standalone_spec = chain_for(
        &lp,
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
    );
    assert!(validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        standalone(lp.log_basis),
        64,
        vec![standalone_spec],
    )
    .is_err());
}

#[test]
fn rejects_missing_duplicate_out_of_order_and_wrong_purpose_sources() {
    let lp = level();
    let opening = scalar_opening();
    let good = current_and_opening_specs(&lp);
    for specs in [
        vec![good[0].clone()],
        vec![good[0].clone(), good[0].clone()],
        vec![good[1].clone(), good[0].clone()],
        vec![
            good[0].clone(),
            CompressionChainSpec {
                source: CompressionSourceId::PrecommittedOuter { index: 0 },
                maps: good[1].maps.clone(),
            },
        ],
    ] {
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
            64,
            specs,
        )
        .is_err());
    }
    assert!(
        validate_and_compile::<Prime128OffsetA7F7>(&lp, standalone(lp.log_basis), 64, good,)
            .is_err()
    );
    let standalone_negative_binary = chain_for(
        &lp,
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
    );
    assert!(validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        standalone(lp.log_basis - 1),
        64,
        vec![standalone_negative_binary],
    )
    .is_err());
}

#[test]
fn purpose_enforces_base_and_source_semantics() {
    let lp = level();
    let mut wrong_current_base = current_and_opening_specs(&lp);
    wrong_current_base[0].maps[0].alphabet = CompressionAlphabet::OpeningBase { log_basis: 4 };
    assert!(validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        CompressionCatalogContext::CoGeneratedLevel {
            opening: &scalar_opening(),
        },
        64,
        wrong_current_base,
    )
    .is_err());

    let mut wrong_opening_base = current_and_opening_specs(&lp);
    wrong_opening_base[1].maps[0].alphabet = CompressionAlphabet::OpeningBase { log_basis: 4 };
    assert!(validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        CompressionCatalogContext::CoGeneratedLevel {
            opening: &scalar_opening(),
        },
        64,
        wrong_opening_base,
    )
    .is_err());

    let standalone_base_five = chain_for(
        &lp,
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[
            CompressionAlphabet::OpeningBase { log_basis: 5 },
            CompressionAlphabet::NegativeBinary,
        ],
    );
    assert!(validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        standalone(lp.log_basis),
        64,
        vec![standalone_base_five],
    )
    .is_err());
}

#[test]
fn rejects_chain_depths_outside_two_or_three() {
    let lp = level();
    for depth in [0, 1, 4] {
        let mut chain = chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        );
        chain.maps.resize(depth, chain.maps[0].clone());
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            64,
            vec![chain],
        )
        .is_err());
    }
}

#[test]
fn accepts_depth_three_with_negative_binary_later_maps() {
    let lp = level();
    let spec = chain_for(
        &lp,
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[
            CompressionAlphabet::OpeningBase { log_basis: 4 },
            CompressionAlphabet::NegativeBinary,
            CompressionAlphabet::NegativeBinary,
        ],
    );
    let catalog =
        validate_and_compile::<Prime128OffsetA7F7>(&lp, standalone(lp.log_basis), 64, vec![spec])
            .unwrap();
    assert_eq!(catalog.chains[0].maps.len(), 3);
}

#[test]
fn rejects_noncanonical_or_insecure_map_keys() {
    let lp = level();
    let base = chain_for(
        &lp,
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
    );
    let original = &base.maps[0].key;
    let stored = original.sis_table_key();
    let mutations = [
        AjtaiKeyParams::new_unchecked(
            138,
            SisModulusFamily::Q64,
            original.row_len(),
            original.col_len(),
            stored.coeff_linf_bound,
            D,
        ),
        AjtaiKeyParams::new_unchecked(
            137,
            SisModulusFamily::Q128,
            original.row_len(),
            original.col_len(),
            stored.coeff_linf_bound,
            D,
        ),
        AjtaiKeyParams::new_unchecked(
            138,
            SisModulusFamily::Q128,
            original.row_len(),
            original.col_len(),
            3,
            D,
        ),
        AjtaiKeyParams::new_unchecked(
            138,
            SisModulusFamily::Q128,
            0,
            original.col_len(),
            stored.coeff_linf_bound,
            D,
        ),
        AjtaiKeyParams::new_unchecked(
            138,
            SisModulusFamily::Q128,
            original.row_len(),
            usize::MAX,
            stored.coeff_linf_bound,
            D,
        ),
    ];
    for mutation in mutations {
        let mut spec = base.clone();
        spec.maps[0].key = mutation;
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            64,
            vec![spec],
        )
        .is_err());
    }
}

#[test]
fn rejects_source_key_outside_field_family_and_production_security() {
    let mut lp = level();
    lp.b_key = key(SisModulusFamily::Q64, D, 63, 1);
    let spec = chain_for(
        &lp,
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
    );
    assert!(validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        standalone(lp.log_basis),
        64,
        vec![spec],
    )
    .is_err());

    let mut lp = level();
    let source = &lp.b_key;
    lp.b_key = AjtaiKeyParams::new_unchecked(
        DEFAULT_SIS_SECURITY_BITS - 1,
        SisModulusFamily::Q128,
        source.row_len(),
        source.col_len(),
        source.coeff_linf_bound(),
        D,
    );
    let spec = chain_for(
        &level(),
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
    );
    assert!(validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        standalone(lp.log_basis),
        64,
        vec![spec],
    )
    .is_err());
}

#[test]
fn source_uses_recomputed_level_collision_bound() {
    let mut too_small = level();
    too_small.b_key = key(SisModulusFamily::Q128, D, 31, 1);
    let spec = chain_for(
        &too_small,
        CompressionSourceId::CurrentOuter,
        &too_small.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
    );
    assert!(validate_and_compile::<Prime128OffsetA7F7>(
        &too_small,
        standalone(too_small.log_basis),
        64,
        vec![spec],
    )
    .is_err());

    let mut conservative = level();
    conservative.b_key = key(SisModulusFamily::Q128, D, 127, 1);
    let spec = chain_for(
        &conservative,
        CompressionSourceId::CurrentOuter,
        &conservative.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
    );
    assert!(validate_and_compile::<Prime128OffsetA7F7>(
        &conservative,
        standalone(conservative.log_basis),
        64,
        vec![spec],
    )
    .is_ok());
}

#[test]
fn rejects_unsupported_sis_dimensions_dispatch_and_gen_divisibility() {
    let lp = level();
    let base = chain_for(
        &lp,
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
    );
    for d in [8, 16] {
        let mut spec = base.clone();
        let original = &spec.maps[0].key;
        spec.maps[0].key = AjtaiKeyParams::new_unchecked(
            138,
            SisModulusFamily::Q128,
            original.row_len(),
            original.col_len(),
            2,
            d,
        );
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            64,
            vec![spec],
        )
        .is_err());
    }
    let mut dispatch_rejected = base.clone();
    let expected_input = lp.b_key.row_len() * D * 128;
    dispatch_rejected.maps[0].key = key(SisModulusFamily::Q128, 128, 1, expected_input / 128);
    assert!(validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        standalone(lp.log_basis),
        128,
        vec![dispatch_rejected],
    )
    .is_err());
    assert!(validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        standalone(lp.log_basis),
        48,
        vec![base],
    )
    .is_err());
}

#[test]
fn rejects_invalid_bases_and_chain_discontinuity() {
    let mut lp = level();
    for log_basis in [0, 7, 128] {
        let spec = chain_for(
            &lp,
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        );
        let mut spec = spec;
        spec.maps[0].alphabet = CompressionAlphabet::OpeningBase { log_basis };
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            64,
            vec![spec],
        )
        .is_err());
    }
    let mut later_opening_base = chain_for(
        &lp,
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
    );
    later_opening_base.maps[1].alphabet = CompressionAlphabet::OpeningBase { log_basis: 4 };
    assert!(validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        standalone(lp.log_basis),
        64,
        vec![later_opening_base],
    )
    .is_err());
    for invalid_level_base in [0, 128] {
        lp.log_basis = invalid_level_base;
        let spec = chain_for(
            &level(),
            CompressionSourceId::CurrentOuter,
            &lp.b_key,
            &[CompressionAlphabet::NegativeBinary; 2],
        );
        assert!(validate_and_compile::<Prime128OffsetA7F7>(
            &lp,
            standalone(lp.log_basis),
            64,
            vec![spec],
        )
        .is_err());
    }

    let lp = level();
    let mut spec = chain_for(
        &lp,
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
    );
    let key = &spec.maps[1].key;
    spec.maps[1].key = AjtaiKeyParams::new_unchecked(
        key.min_security_bits(),
        key.sis_family(),
        key.row_len(),
        key.col_len() + 1,
        key.coeff_linf_bound(),
        D,
    );
    assert!(validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        standalone(lp.log_basis),
        64,
        vec![spec],
    )
    .is_err());
}
