use super::super::combinatorics::BinomialCdfRow;
use super::*;

fn address() -> ColumnChallengeAddress {
    ColumnChallengeAddress {
        common_prefix: [37; 32],
        level: 0,
        group: 1,
        owner: 2,
        column: 3,
        response_nonce: 4,
        column_work_binding: [91; 32],
    }
}

#[test]
fn exact_family_sum_and_size_lex_boundaries() {
    let sampler = MultigroupColumnSampler::new().unwrap();
    let counts = sampler.profile.scalar_ring().counts();
    let specified =
        BigUint::parse_bytes(b"107268495235892251993000716928106195079952", 10).unwrap();
    let mut shell = BigUint::from(1u8);
    let mut sum = shell.clone();
    for weight in 1usize..=46 {
        shell *= 163 - weight;
        shell /= weight;
        sum += &shell;
    }
    assert_eq!(sum, specified);
    assert_eq!(sampler.profile.cardinality(), &specified);
    let zero = BigUint::from(0u8);
    assert_eq!(counts.unrank_lex(&zero, 46).unwrap(), Vec::<u16>::new());
    let final_rank = &specified - BigUint::from(1u8);
    let last = counts.unrank_lex(&final_rank, 46).unwrap();
    assert_eq!(last, (116..162).collect::<Vec<u16>>());
    assert_eq!(counts.rank_lex(&last, 46).unwrap(), final_rank);
    for weight in 1..=46 {
        let first_rank = counts.ball(weight - 1);
        let first = counts.unrank_lex(first_rank, 46).unwrap();
        assert_eq!(first, (0..weight as u16).collect::<Vec<_>>());
        assert_eq!(&counts.rank_lex(&first, 46).unwrap(), first_rank);
    }
    assert!(matches!(
        counts.unrank_lex(&specified, 46),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        counts.unrank_lex(&zero, 163),
        Err(AkitaError::InvalidInput(_))
    ));
    for invalid in [vec![162], vec![0, 0], vec![2, 1], (0..47).collect()] {
        assert!(matches!(
            counts.rank_lex(&invalid, 46),
            Err(AkitaError::InvalidInput(_))
        ));
    }
    // Exhaustive tiny supports also exercise every shell transition.
    let tiny = BinomialCdfRow::new(3);
    assert_eq!(tiny.unrank_lex(&BigUint::from(4u8), 2).unwrap(), vec![0, 1]);
}

#[test]
fn rejected_rank_advances_the_block_and_exhaustion_is_typed() {
    let sampler = MultigroupColumnSampler::new().unwrap();
    let mut seen = Vec::new();
    let upper = sampler.profile.cardinality().to_bytes_le();
    let sample = sampler
        .sample_with_blocks(|block, bytes| {
            seen.push(block);
            if block == 0 {
                bytes[..upper.len()].copy_from_slice(&upper);
            }
        })
        .unwrap();
    assert!(sample.terms().is_empty());
    assert_eq!(seen, vec![0, 1]);
    let mut blocks = 0;
    assert!(matches!(
        sampler.sample_with_blocks(|block, bytes| {
            assert_eq!(block, blocks);
            blocks += 1;
            bytes.fill(255);
        }),
        Err(AkitaError::InvalidInput(_))
    ));
    assert_eq!(blocks, 256);
}

#[test]
fn unused_candidate_bits_are_discarded_and_bitmaps_are_canonical() {
    let sampler = MultigroupColumnSampler::new().unwrap();
    let zero = sampler
        .sample_with_blocks(|_, bytes| {
            *bytes.last_mut().unwrap() = 0xfe;
        })
        .unwrap();
    assert!(zero.terms().is_empty());
    let challenge = sampler.sample(&address()).unwrap();
    challenge.validate(sampler.profile()).unwrap();
    let bitmap = challenge
        .canonical_support_encoding(sampler.profile())
        .unwrap();
    assert_eq!(bitmap.len(), 21);
    assert_eq!(bitmap[20] & !3, 0);
    assert_eq!(sampler.decode_support_bitmap(&bitmap).unwrap(), challenge);
    let mut invalid = bitmap.clone();
    invalid[20] |= 4;
    assert!(matches!(
        sampler.decode_support_bitmap(&invalid),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(sampler.decode_support_bitmap(&bitmap[..20]).is_err());
    assert!(sampler.decode_support_bitmap(&[255; 21]).is_err());
    let mut excessive = [0u8; 21];
    excessive[..6].fill(255);
    assert!(sampler.decode_support_bitmap(&excessive).is_err());
}

#[test]
fn each_address_component_binds_one_coordinate_without_a_mutable_cursor() {
    let sampler = MultigroupColumnSampler::new().unwrap();
    let base = address();
    let original = sampler.sample(&base).unwrap();
    let mut changes = [base; 6];
    changes[0].group += 1;
    changes[1].owner += 1;
    changes[2].column += 1;
    changes[3].response_nonce += 1;
    changes[4].column_work_binding[0] ^= 1;
    changes[5].common_prefix[0] ^= 1;
    let other = ColumnChallengeAddress { group: 0, ..base };
    let other_expected = sampler.sample(&other).unwrap();
    for changed in changes {
        assert_ne!(sampler.sample(&changed).unwrap(), original);
        assert_eq!(sampler.sample(&other).unwrap(), other_expected);
        assert_eq!(sampler.sample(&base).unwrap(), original);
    }
    let invalid = [
        ColumnChallengeAddress { level: 1, ..base },
        ColumnChallengeAddress { group: 64, ..base },
        ColumnChallengeAddress {
            owner: 4096,
            ..base
        },
        ColumnChallengeAddress {
            column: 256,
            ..base
        },
        ColumnChallengeAddress {
            response_nonce: 4096,
            ..base
        },
    ];
    for value in invalid {
        assert!(matches!(
            sampler.sample(&value),
            Err(AkitaError::InvalidInput(_))
        ));
    }
}

#[test]
fn indexed_xof_address_uses_little_endian_fixed_width_components() {
    let sampler = MultigroupColumnSampler::new().unwrap();
    let address = address();
    let mut canonical = Vec::new();
    canonical.extend_from_slice(MULTIGROUP_COLUMN_DOMAIN);
    canonical.extend_from_slice(&address.common_prefix);
    for value in [
        address.level,
        address.group,
        address.owner,
        address.column,
        address.response_nonce,
    ] {
        canonical.extend_from_slice(&value.to_le_bytes());
    }
    canonical.extend_from_slice(&address.column_work_binding);
    canonical.extend_from_slice(&MULTIGROUP_COLUMN_FAMILY_ID.to_le_bytes());
    let independent = sampler
        .sample_with_blocks(|block, bytes| {
            let mut xof = Shake256::default();
            xof.update(&canonical);
            xof.update(&block.to_le_bytes());
            xof.finalize_xof().read(bytes);
        })
        .unwrap();
    assert_eq!(sampler.sample(&address).unwrap(), independent);
}
