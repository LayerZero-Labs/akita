use super::*;

fn chain(first: CompressionAlphabet, first_d: u32, second_d: u32) -> CompressionChainChoice {
    CompressionChainChoice::Two([
        CompressionMapChoice {
            ring_d: first_d,
            alphabet: first,
        },
        CompressionMapChoice {
            ring_d: second_d,
            alphabet: CompressionAlphabet::NegativeBinary,
        },
    ])
}

fn frozen(max: u32, first: CompressionAlphabet) -> FrozenCompressionChainChoice {
    frozen_for(&level().b_key, max, first)
}

fn frozen_for(
    source_key: &AjtaiKeyParams,
    max: u32,
    first: CompressionAlphabet,
) -> FrozenCompressionChainChoice {
    FrozenCompressionChainChoice::new(source_key, max, chain(first, 32, 32))
}

fn outer<'a>(
    current_outer: FrozenCompressionChainChoice,
    precommitted_outer: &'a [FrozenCompressionChainChoice],
) -> CompressionChoice<'a> {
    CompressionChoice {
        f: CompressionFChoice {
            current_outer,
            precommitted_outer,
        },
        opening: None,
    }
}

#[test]
fn f_descriptor_binds_every_free_choice_and_source_slot() {
    let current = frozen(6, CompressionAlphabet::OpeningBase { log_basis: 4 });
    let pre = frozen(5, CompressionAlphabet::NegativeBinary);
    let base = CompressionFChoice {
        current_outer: current,
        precommitted_outer: &[pre],
    };
    let mutations = [
        CompressionFChoice {
            current_outer: FrozenCompressionChainChoice {
                max_opening_log_basis: 7,
                ..current
            },
            precommitted_outer: &[pre],
        },
        CompressionFChoice {
            current_outer: FrozenCompressionChainChoice {
                chain: chain(CompressionAlphabet::OpeningBase { log_basis: 4 }, 64, 32),
                ..current
            },
            precommitted_outer: &[pre],
        },
        CompressionFChoice {
            current_outer: FrozenCompressionChainChoice {
                chain: chain(CompressionAlphabet::NegativeBinary, 32, 32),
                ..current
            },
            precommitted_outer: &[pre],
        },
        CompressionFChoice {
            current_outer: FrozenCompressionChainChoice {
                chain: CompressionChainChoice::Three([
                    CompressionMapChoice {
                        ring_d: 32,
                        alphabet: CompressionAlphabet::OpeningBase { log_basis: 4 },
                    },
                    CompressionMapChoice {
                        ring_d: 32,
                        alphabet: CompressionAlphabet::NegativeBinary,
                    },
                    CompressionMapChoice {
                        ring_d: 32,
                        alphabet: CompressionAlphabet::NegativeBinary,
                    },
                ]),
                ..current
            },
            precommitted_outer: &[pre],
        },
        CompressionFChoice {
            current_outer: pre,
            precommitted_outer: &[current],
        },
    ];
    for mutation in mutations {
        assert_ne!(
            base.descriptor_digest().unwrap(),
            mutation.descriptor_digest().unwrap()
        );
    }
    assert_eq!(
        base.descriptor_bytes().unwrap(),
        base.descriptor_bytes().unwrap()
    );
}

#[test]
fn same_f_choice_replays_identically_at_standalone_and_terminal() {
    let mut lp = level();
    lp.log_basis = 4;
    let current = frozen(6, CompressionAlphabet::OpeningBase { log_basis: 2 });
    let choice = outer(current, &[]);
    let standalone = choice
        .replay::<Prime128OffsetA7F7>(&lp, CompressionCatalogContext::StandaloneCommitment)
        .unwrap();
    let opening = scalar_opening();
    let terminal = choice
        .replay::<Prime128OffsetA7F7>(
            &lp,
            CompressionCatalogContext::TerminalFold { opening: &opening },
        )
        .unwrap();
    assert_eq!(standalone.chains, terminal.chains);
    assert!(terminal.terminal_relation_layout().is_ok());
}

#[test]
fn terminal_rejects_actual_base_below_any_f_first_map() {
    let mut lp = level();
    lp.log_basis = 4;
    let choice = outer(
        frozen(6, CompressionAlphabet::OpeningBase { log_basis: 5 }),
        &[],
    );
    let opening = scalar_opening();
    assert!(choice
        .replay::<Prime128OffsetA7F7>(
            &lp,
            CompressionCatalogContext::TerminalFold { opening: &opening },
        )
        .is_err());
}

#[test]
fn independent_current_and_precommitted_envelopes_match_standalone_geometry() {
    let mut lp = level();
    lp.log_basis = 4;
    let group = PolynomialGroupLayout::new(3, 1);
    let pre_b = key(SisModulusFamily::Q128, D, 63, 2);
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
    let current = frozen(6, CompressionAlphabet::OpeningBase { log_basis: 2 });
    let pre = frozen_for(&pre_b, 5, CompressionAlphabet::NegativeBinary);
    let opening =
        OpeningClaimsLayout::from_root_groups(&[group], PolynomialGroupLayout::new(4, 1)).unwrap();
    let terminal = outer(current, &[pre])
        .replay::<Prime128OffsetA7F7>(
            &lp,
            CompressionCatalogContext::TerminalFold { opening: &opening },
        )
        .unwrap();

    let mut current_lp = lp.clone();
    current_lp.precommitted_groups.clear();
    let current_catalog = outer(current, &[])
        .replay::<Prime128OffsetA7F7>(&current_lp, CompressionCatalogContext::StandaloneCommitment)
        .unwrap();
    let mut pre_lp = current_lp;
    pre_lp.b_key = pre_b;
    let pre_catalog = outer(pre, &[])
        .replay::<Prime128OffsetA7F7>(&pre_lp, CompressionCatalogContext::StandaloneCommitment)
        .unwrap();
    assert_eq!(terminal.chains[0], current_catalog.chains[0]);
    assert_eq!(
        terminal.chains[1].max_opening_log_basis,
        pre_catalog.chains[0].max_opening_log_basis
    );
    assert_eq!(
        terminal.chains[1].source_output_coeffs,
        pre_catalog.chains[0].source_output_coeffs
    );
    assert_eq!(terminal.chains[1].maps, pre_catalog.chains[0].maps);
    assert_eq!(
        terminal.chains[1].payload_coeffs,
        pre_catalog.chains[0].payload_coeffs
    );
}

#[test]
fn context_and_slot_shape_are_exact() {
    let lp = level();
    let current = frozen(6, CompressionAlphabet::NegativeBinary);
    let opening = scalar_opening();
    let with_h = CompressionChoice {
        f: CompressionFChoice {
            current_outer: current,
            precommitted_outer: &[],
        },
        opening: Some(chain(CompressionAlphabet::NegativeBinary, 32, 32)),
    };
    assert!(with_h
        .replay::<Prime128OffsetA7F7>(&lp, CompressionCatalogContext::StandaloneCommitment)
        .is_err());
    assert!(outer(current, &[])
        .replay::<Prime128OffsetA7F7>(
            &lp,
            CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
        )
        .is_err());
    for malformed in [
        FrozenCompressionChainChoice {
            chain: chain(CompressionAlphabet::NegativeBinary, 0, 32),
            ..current
        },
        FrozenCompressionChainChoice {
            chain: chain(CompressionAlphabet::NegativeBinary, 4, 32),
            ..current
        },
        FrozenCompressionChainChoice {
            chain: CompressionChainChoice::Two([
                CompressionMapChoice {
                    ring_d: 32,
                    alphabet: CompressionAlphabet::NegativeBinary,
                },
                CompressionMapChoice {
                    ring_d: 32,
                    alphabet: CompressionAlphabet::OpeningBase { log_basis: 4 },
                },
            ]),
            ..current
        },
    ] {
        assert!(outer(malformed, &[])
            .replay::<Prime128OffsetA7F7>(&lp, CompressionCatalogContext::StandaloneCommitment,)
            .is_err());
    }
}

#[test]
fn frozen_current_rejects_same_tier_key_substitution() {
    let mut lp = level();
    let choice = outer(
        frozen_for(&lp.b_key, 6, CompressionAlphabet::NegativeBinary),
        &[],
    );
    let unchanged = choice
        .replay::<Prime128OffsetA7F7>(&lp, CompressionCatalogContext::StandaloneCommitment)
        .unwrap();

    lp.b_key = key(SisModulusFamily::Q128, D, 63, 2);
    assert!(choice
        .replay::<Prime128OffsetA7F7>(&lp, CompressionCatalogContext::StandaloneCommitment)
        .is_err());

    let original_lp = level();
    let replayed = choice
        .replay::<Prime128OffsetA7F7>(
            &original_lp,
            CompressionCatalogContext::StandaloneCommitment,
        )
        .unwrap();
    assert_eq!(unchanged.chains, replayed.chains);
}

#[test]
fn frozen_precommitted_rejects_key_substitution() {
    let mut lp = level();
    let group = PolynomialGroupLayout::new(3, 1);
    let pre_b = key(SisModulusFamily::Q128, D, 63, 2);
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
    let current = frozen_for(&lp.b_key, 6, CompressionAlphabet::NegativeBinary);
    let pre = frozen_for(&pre_b, 6, CompressionAlphabet::NegativeBinary);
    let precommitted = [pre];
    let choice = outer(current, &precommitted);
    let opening =
        OpeningClaimsLayout::from_root_groups(&[group], PolynomialGroupLayout::new(4, 1)).unwrap();
    assert!(choice
        .replay::<Prime128OffsetA7F7>(
            &lp,
            CompressionCatalogContext::TerminalFold { opening: &opening },
        )
        .is_ok());

    lp.precommitted_groups[0].b_key = key(SisModulusFamily::Q128, D, 63, 3);
    assert!(choice
        .replay::<Prime128OffsetA7F7>(
            &lp,
            CompressionCatalogContext::TerminalFold { opening: &opening },
        )
        .is_err());
}

#[test]
fn frozen_descriptor_and_source_digest_bind_every_key_field() {
    let source = level().b_key;
    let frozen = FrozenCompressionChainChoice::new(
        &source,
        6,
        chain(CompressionAlphabet::NegativeBinary, 32, 32),
    );
    let mut digest_mutation = frozen;
    digest_mutation.source_key_digest[0] ^= 1;
    assert_ne!(
        outer(frozen, &[]).descriptor_digest().unwrap(),
        outer(digest_mutation, &[]).descriptor_digest().unwrap()
    );

    let different_d = AjtaiKeyParams::new_unchecked(
        source.min_security_bits(),
        source.sis_family(),
        source.row_len(),
        source.col_len(),
        source.coeff_linf_bound(),
        64,
    );
    assert_ne!(
        source.compression_source_descriptor_digest(),
        different_d.compression_source_descriptor_digest()
    );
}
