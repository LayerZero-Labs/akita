use super::*;

fn label(modulus: SetupStreamModulus) -> SetupStreamLabel {
    SetupStreamLabel {
        role: SetupStreamRole::Commitment,
        group: 7,
        level: 0,
        rank: 3,
        row: 0,
        column: 9,
        ring: SetupStreamRing::PlusTrinomial,
        degree: 648,
        modulus,
    }
}

fn fingerprint(seed: [u8; 32], label: SetupStreamLabel) -> Vec<u128> {
    let mut stream = TypedSetupStream::new(&seed, label).unwrap();
    (0..12)
        .map(|_| match label.modulus {
            SetupStreamModulus::Q(_) => stream.sample_q_centered().unwrap() as u128,
            SetupStreamModulus::Prime128Offset275 => stream.sample_p().unwrap(),
        })
        .collect()
}

#[test]
fn exact_thresholds_cover_equal_numbers_of_each_residue() {
    assert_eq!(q_rejection_threshold(14_253_409).unwrap(), 4_290_276_109);
    assert_eq!(q_rejection_threshold(33_568_993).unwrap(), 4_263_262_111);
    for q in [3, 5, 7, 14_253_409, 33_568_993, u32::MAX] {
        let threshold = q_rejection_threshold(q).unwrap();
        assert_eq!(threshold % u64::from(q), 0);
        assert!(threshold <= 1u64 << 32);
        assert!((1u64 << 32) - threshold < u64::from(q));
        // The candidate immediately below the threshold is accepted; the
        // threshold itself is rejected by the sampler's strict inequality.
        assert_eq!((threshold - 1) % u64::from(q), u64::from(q) - 1);
    }
    assert_eq!(BINARY_SETUP_P, u128::MAX - 274);
    assert!(q_rejection_threshold(0).is_err());
    assert!(q_rejection_threshold(1).is_err());
    assert!(q_rejection_threshold(4).is_err());
}

#[test]
fn every_label_component_and_seed_separates_both_domains() {
    for modulus in [
        SetupStreamModulus::Q(14_253_409),
        SetupStreamModulus::Prime128Offset275,
    ] {
        let base = label(modulus);
        let expected = fingerprint([42; 32], base);
        assert_eq!(expected, fingerprint([42; 32], base));
        let mut variants = vec![
            SetupStreamLabel {
                role: SetupStreamRole::ImageCommitment,
                ..base
            },
            SetupStreamLabel {
                group: base.group + 1,
                ..base
            },
            SetupStreamLabel {
                level: base.level + 1,
                ..base
            },
            SetupStreamLabel {
                rank: base.rank + 1,
                ..base
            },
            SetupStreamLabel {
                row: base.row + 1,
                ..base
            },
            SetupStreamLabel {
                column: base.column + 1,
                ..base
            },
            SetupStreamLabel {
                ring: SetupStreamRing::MinusTrinomial,
                ..base
            },
            SetupStreamLabel {
                degree: base.degree + 2,
                ..base
            },
        ];
        variants.push(SetupStreamLabel {
            modulus: match modulus {
                SetupStreamModulus::Q(_) => SetupStreamModulus::Q(33_568_993),
                SetupStreamModulus::Prime128Offset275 => SetupStreamModulus::Q(14_253_409),
            },
            ..base
        });
        for variant in variants {
            assert_ne!(expected, fingerprint([42; 32], variant));
        }
        assert_ne!(expected, fingerprint([43; 32], base));
    }
}

#[test]
fn centered_q24_q25_and_p_draws_are_in_their_exact_intervals() {
    for q in [14_253_409, 33_568_993] {
        let mut stream = TypedSetupStream::new(&[1; 32], label(SetupStreamModulus::Q(q))).unwrap();
        assert!(stream.sample_p().is_err());
        let mut negative = false;
        let mut positive = false;
        for _ in 0..1_000 {
            let value = stream.sample_q_centered().unwrap();
            assert!(value.abs() <= i64::from(q / 2));
            negative |= value < 0;
            positive |= value > 0;
        }
        assert!(negative && positive);
    }
    let mut stream =
        TypedSetupStream::new(&[1; 32], label(SetupStreamModulus::Prime128Offset275)).unwrap();
    assert!(stream.sample_q_centered().is_err());
    for _ in 0..1_000 {
        assert!(stream.sample_p().unwrap() < BINARY_SETUP_P);
    }
}

#[test]
fn small_q_statistical_sanity() {
    let mut stream = TypedSetupStream::new(&[0x83; 32], label(SetupStreamModulus::Q(5))).unwrap();
    let mut frequencies = [0u32; 5];
    for _ in 0..100_000 {
        let offset = usize::try_from(stream.sample_q_centered().unwrap() + 2).unwrap();
        frequencies[offset] += 1;
    }
    // More than eight standard deviations around the expectation, with a
    // deterministic seed: a sanity check, not the argument for unbiasedness.
    for frequency in frequencies {
        assert!((18_800..21_200).contains(&frequency));
    }
}

#[test]
fn matrix_entries_use_their_indexed_streams_and_preserve_width_extensions() {
    let base = SetupStreamLabel {
        column: 0,
        degree: 6,
        ..label(SetupStreamModulus::Q(14_253_409))
    };
    let small = derive_centered_q_matrix(&[3; 32], base, 2).unwrap();
    let large = derive_centered_q_matrix(&[3; 32], base, 3).unwrap();
    for row in 0..3usize {
        for column in 0..2usize {
            let mut stream = TypedSetupStream::new(
                &[3; 32],
                SetupStreamLabel {
                    row: row as u32,
                    column: column as u64,
                    ..base
                },
            )
            .unwrap();
            for coefficient in 0..6 {
                let expected = stream.sample_q_centered().unwrap();
                assert_eq!(small[(row * 2 + column) * 6 + coefficient], expected);
                assert_eq!(large[(row * 3 + column) * 6 + coefficient], expected);
            }
        }
    }
    assert!(derive_centered_q_matrix(&[3; 32], base, usize::MAX).is_err());
    assert!(derive_centered_q_matrix(&[3; 32], base, 0).is_err());
    assert!(TypedSetupStream::new(&[3; 32], SetupStreamLabel { row: 3, ..base }).is_err());
}
