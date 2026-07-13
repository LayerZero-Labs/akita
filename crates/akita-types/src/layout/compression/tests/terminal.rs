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
    let catalog = validate_compression_catalog::<Prime128OffsetA7F7>(
        &lp,
        standalone(lp.log_basis),
        64,
        vec![spec],
    )
    .unwrap();
    assert_eq!(catalog.chains[0].maps[0].key.coeff_linf_bound(), 127);
}

fn two_choice(first: CompressionAlphabet) -> CompressionChainChoice {
    CompressionChainChoice::Two([
        CompressionMapChoice {
            ring_d: D as u32,
            alphabet: first,
        },
        CompressionMapChoice {
            ring_d: D as u32,
            alphabet: CompressionAlphabet::NegativeBinary,
        },
    ])
}

#[test]
fn terminal_fold_replay_binds_context_sources_base_and_field() {
    use crate::layout::relation::{
        RelationGroupId, RelationRowId, RelationRowInputs, RelationRowRhs,
    };

    let lp = level();
    let opening = scalar_opening();
    let choice = CompressionChoice {
        f: CompressionFChoice {
            current_outer: FrozenCompressionChainChoice::new(
                &lp.b_key,
                lp.log_basis,
                two_choice(CompressionAlphabet::OpeningBase {
                    log_basis: lp.log_basis,
                }),
            ),
            precommitted_outer: &[],
        },
        opening: None,
    };
    let catalog = choice
        .replay::<Prime128OffsetA7F7>(
            &lp,
            CompressionCatalogContext::TerminalFold { opening: &opening },
        )
        .unwrap();
    let layout = catalog.terminal_relation_layout().unwrap();
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
    assert_eq!(input.log_basis(), lp.log_basis);
    assert!(layout.row_plan().family(RelationRowId::D).is_err());
    assert!(layout.row_plan().families().iter().all(|family| !matches!(
        family.id(),
        RelationRowId::Compression {
            source: CompressionSourceId::Opening,
            ..
        }
    )));

    assert!(choice
        .replay::<Prime64Offset59>(
            &lp,
            CompressionCatalogContext::TerminalFold { opening: &opening },
        )
        .is_err());
    assert!(choice
        .replay::<Prime128OffsetA7F7>(&lp, standalone(lp.log_basis))
        .is_err());
}

#[test]
fn binary_first_terminal_join_has_no_opening_base_lower_bound() {
    let mut terminal = level();
    terminal.log_basis = 2;
    let opening = scalar_opening();
    let catalog = CompressionChoice {
        f: CompressionFChoice {
            current_outer: FrozenCompressionChainChoice::new(
                &terminal.b_key,
                4,
                two_choice(CompressionAlphabet::NegativeBinary),
            ),
            precommitted_outer: &[],
        },
        opening: None,
    }
    .replay::<Prime128OffsetA7F7>(
        &terminal,
        CompressionCatalogContext::TerminalFold { opening: &opening },
    )
    .unwrap();
    assert!(catalog.terminal_relation_layout().is_ok());
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
    let catalog = validate_compression_catalog::<Prime128OffsetA7F7>(
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
    assert!(validate_compression_catalog::<Prime128OffsetA7F7>(
        &lp,
        standalone(6),
        64,
        vec![underpriced],
    )
    .is_err());

    let negative_binary = chain_for_profile(
        CompressionSourceId::CurrentOuter,
        &lp.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
        SisModulusFamily::Q128,
        128,
        D,
        6,
    );
    assert!(validate_compression_catalog::<Prime128OffsetA7F7>(
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
        validate_compression_catalog::<Prime64Offset59>(&q64, standalone(6), 64, vec![q64_spec])
            .unwrap();
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
        validate_compression_catalog::<Prime32Offset99>(&q32, standalone(6), 64, vec![q32_spec])
            .unwrap();
    assert_eq!(q32_catalog.chains[0].maps[0].digit_depth, 32);

    let q128 = level();
    let cross_family = chain_for(
        &q128,
        CompressionSourceId::CurrentOuter,
        &q128.b_key,
        &[CompressionAlphabet::NegativeBinary; 2],
    );
    assert!(validate_compression_catalog::<Prime64Offset59>(
        &q128,
        standalone(6),
        64,
        vec![cross_family],
    )
    .is_err());
}
