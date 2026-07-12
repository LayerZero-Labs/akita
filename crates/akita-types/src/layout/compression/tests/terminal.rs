use super::*;
#[test]
fn opening_base_accepts_larger_canonical_bucket() {
    let lp = level();
    let mut spec = chain_for(
        &lp,
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[
            CompressionAlphabet::OpeningBase { log_basis: 4 },
            CompressionAlphabet::NegativeBinary,
        ],
    );
    let first_col_len = spec.maps[0].key.col_len();
    spec.maps[0].key = key(SisModulusFamily::Q128, D, 127, first_col_len);
    let second_col_len = spec.maps[0].key.row_len() * 128;
    spec.maps[1].key = key(SisModulusFamily::Q128, D, 1, second_col_len);
    let catalog =
        validate_and_compile::<Prime128OffsetA7F7>(&lp, standalone(lp.log_basis), 64, vec![spec])
            .unwrap();
    assert_eq!(catalog.chains[0].maps[0].key.coeff_linf_bound(), 127);
}

#[test]
fn frozen_standalone_terminal_join_binds_key_base_and_field() {
    use crate::layout::relation::{
        RelationGroupId, RelationRowId, RelationRowInputs, RelationRowRhs,
    };

    let lp = level();
    let spec = chain_for(
        &lp,
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[
            CompressionAlphabet::OpeningBase { log_basis: 4 },
            CompressionAlphabet::NegativeBinary,
        ],
    );
    let catalog =
        validate_and_compile::<Prime128OffsetA7F7>(&lp, standalone(6), 64, vec![spec]).unwrap();
    let opening = scalar_opening();
    let layout = catalog
        .terminal_relation_layout::<Prime128OffsetA7F7>(&lp, &opening)
        .unwrap();
    let b = layout
        .row_plan()
        .family(RelationRowId::B {
            group: RelationGroupId::Current,
        })
        .unwrap();
    assert_eq!(b.rhs(), RelationRowRhs::Zero);
    let RelationRowInputs::B {
        compression_input: Some(input),
        ..
    } = b.inputs()
    else {
        panic!("terminal B must carry frozen F input");
    };
    assert_eq!(input.log_basis(), 4);
    assert!(layout.row_plan().family(RelationRowId::D).is_err());
    assert!(layout.row_plan().families().iter().all(|family| !matches!(
        family.id(),
        RelationRowId::Compression {
            source: CompressionSourceId::Opening,
            ..
        }
    )));

    let mut too_small_base = lp.clone();
    too_small_base.log_basis = 3;
    assert!(catalog
        .terminal_relation_layout::<Prime128OffsetA7F7>(&too_small_base, &opening)
        .is_err());
    let mut wrong_key = lp.clone();
    wrong_key.b_key = key(SisModulusFamily::Q128, D, 63, 2);
    assert!(catalog
        .terminal_relation_layout::<Prime128OffsetA7F7>(&wrong_key, &opening)
        .is_err());
    assert!(catalog
        .terminal_relation_layout::<Prime64Offset59>(&lp, &opening)
        .is_err());
}

#[test]
fn binary_first_terminal_join_has_no_opening_base_lower_bound() {
    let lp = level();
    let spec = chain_for(
        &lp,
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[
            CompressionAlphabet::NegativeBinary,
            CompressionAlphabet::NegativeBinary,
        ],
    );
    let catalog =
        validate_and_compile::<Prime128OffsetA7F7>(&lp, standalone(6), 64, vec![spec]).unwrap();
    let mut terminal = lp.clone();
    terminal.log_basis = 2;
    assert!(catalog
        .terminal_relation_layout::<Prime128OffsetA7F7>(&terminal, &scalar_opening())
        .is_ok());
}

#[test]
fn standalone_range_envelope_prices_conservative_base() {
    let mut lp = level();
    lp.log_basis = 2;
    let opening_base = chain_for_profile(
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[
            CompressionAlphabet::OpeningBase { log_basis: 4 },
            CompressionAlphabet::NegativeBinary,
        ],
        SisModulusFamily::Q128,
        128,
        D,
        6,
    );
    let catalog = validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        standalone(6),
        64,
        vec![opening_base.clone()],
    )
    .unwrap();
    assert!(matches!(
        catalog.purpose,
        CompressionCatalogPurpose::Standalone {
            max_opening_log_basis: 6,
            ..
        }
    ));
    assert_eq!(catalog.chains[0].maps[0].key.coeff_linf_bound(), 63);

    let mut underpriced = opening_base;
    let col_len = underpriced.maps[0].key.col_len();
    underpriced.maps[0].key = key(SisModulusFamily::Q128, D, 3, col_len);
    assert!(
        validate_and_compile::<Prime128OffsetA7F7>(&lp, standalone(6), 64, vec![underpriced],)
            .is_err()
    );

    let negative_binary = chain_for_profile(
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
        SisModulusFamily::Q128,
        128,
        D,
        6,
    );
    assert!(validate_and_compile::<Prime128OffsetA7F7>(
        &lp,
        standalone(6),
        64,
        vec![negative_binary],
    )
    .is_ok());
}

#[test]
fn reduced_field_profiles_use_native_negative_binary_depths() {
    let mut q64 = LevelParams::params_only(
        SisModulusFamily::Q64,
        64,
        6,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(64),
    );
    q64.b_key = key(SisModulusFamily::Q64, 32, 63, 1);
    q64.d_key = key(SisModulusFamily::Q64, 32, 63, 1);
    let q64_spec = chain_for_profile(
        CompressionSourceId::CurrentOuter,
        &q64.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
        SisModulusFamily::Q64,
        64,
        32,
        6,
    );
    let q64_catalog =
        validate_and_compile::<Prime64Offset59>(&q64, standalone(6), 64, vec![q64_spec]).unwrap();
    assert_eq!(q64_catalog.chains[0].maps[0].digit_depth, 64);

    let mut q32 = LevelParams::params_only(
        SisModulusFamily::Q32,
        64,
        6,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(64),
    );
    q32.b_key = key(SisModulusFamily::Q32, 64, 63, 1);
    q32.d_key = key(SisModulusFamily::Q32, 64, 63, 1);
    let q32_spec = chain_for_profile(
        CompressionSourceId::CurrentOuter,
        &q32.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
        SisModulusFamily::Q32,
        32,
        32,
        6,
    );
    let q32_catalog =
        validate_and_compile::<Prime32Offset99>(&q32, standalone(6), 64, vec![q32_spec]).unwrap();
    assert_eq!(q32_catalog.chains[0].maps[0].digit_depth, 32);

    let q128 = level();
    let cross_family = chain_for(
        &q128,
        CompressionSourceId::CurrentOuter,
        &q128.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
    );
    assert!(
        validate_and_compile::<Prime64Offset59>(&q128, standalone(6), 64, vec![cross_family],)
            .is_err()
    );
}
