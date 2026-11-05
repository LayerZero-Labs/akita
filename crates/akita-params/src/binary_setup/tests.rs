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

#[test]
fn block_p_draws_match_single_draws_and_stream_position() {
    let p_label = label(SetupStreamModulus::Prime128Offset275);
    for len in [0, 1, 63, 64, 65, 200] {
        let mut single = TypedSetupStream::new(&[5; 32], p_label).unwrap();
        let expected = (0..=len)
            .map(|_| single.sample_p().unwrap())
            .collect::<Vec<_>>();
        let mut block = TypedSetupStream::new(&[5; 32], p_label).unwrap();
        let mut actual = vec![0u128; len];
        block.fill_p(&mut actual).unwrap();
        actual.push(block.sample_p().unwrap());
        assert_eq!(actual, expected, "length {len}");
    }
    let mut q = TypedSetupStream::new(&[5; 32], label(SetupStreamModulus::Q(5))).unwrap();
    assert!(q.fill_p(&mut [0u128; 3]).is_err());
}

#[test]
fn sponge_squeezes_the_shake256_stream() {
    use shake::digest::{ExtendableOutput, Update, XofReader};
    let message = [0x5a_u8; 135];
    for length in [0, 1, 37, 120, 134, 135] {
        let mut reference = shake::Shake256::default();
        reference.update(&message[..length]);
        let mut reference = reference.finalize_xof();
        let (head, tail) = message[..length].split_at(length / 3);
        let mut squeeze = ShakeSqueeze::new(&[head, tail]).unwrap();
        // Reads straddling block boundaries at several widths.
        for width in [1, 7, 136, 16, 300, 135, 1024] {
            let mut expected = vec![0u8; width];
            reference.read(&mut expected);
            let mut actual = vec![0u8; width];
            squeeze.read(&mut actual);
            assert_eq!(actual, expected, "message {length}, read {width}");
        }
    }
    assert!(ShakeSqueeze::new(&[&message, &[0]]).is_err());
}

#[test]
fn paired_reads_match_separate_reads() {
    let first_message: &[u8] = b"first";
    let second_message: &[u8] = b"second stream";
    for (first_width, second_width) in [(0, 0), (1, 0), (0, 300), (136, 136), (500, 17), (3, 2000)]
    {
        let mut first = ShakeSqueeze::new(&[first_message]).unwrap();
        let mut second = ShakeSqueeze::new(&[second_message]).unwrap();
        let mut first_reference = first.clone();
        let mut second_reference = second.clone();
        // Offset the streams so their blocks run out at different reads.
        first.read(&mut [0; 40]);
        first_reference.read(&mut [0; 40]);
        let mut first_actual = vec![0u8; first_width];
        let mut second_actual = vec![0u8; second_width];
        ShakeSqueeze::read_pair(
            &mut first,
            &mut first_actual,
            &mut second,
            &mut second_actual,
        );
        let mut first_expected = vec![0u8; first_width];
        let mut second_expected = vec![0u8; second_width];
        first_reference.read(&mut first_expected);
        second_reference.read(&mut second_expected);
        assert_eq!(first_actual, first_expected);
        assert_eq!(second_actual, second_expected);
        let (mut next, mut next_expected) = ([0u8; 64], [0u8; 64]);
        first.read(&mut next);
        first_reference.read(&mut next_expected);
        assert_eq!(next, next_expected);
        second.read(&mut next);
        second_reference.read(&mut next_expected);
        assert_eq!(next, next_expected);
    }
}

#[test]
fn paired_p_draws_match_separate_draws_and_stream_positions() {
    let p_label = label(SetupStreamModulus::Prime128Offset275);
    let other = SetupStreamLabel { row: 1, ..p_label };
    for (first_len, second_len) in [(0, 0), (64, 64), (1, 200), (65, 3), (0, 64)] {
        let mut first = TypedSetupStream::new(&[5; 32], p_label).unwrap();
        let mut second = TypedSetupStream::new(&[5; 32], other).unwrap();
        let mut first_actual = vec![0u128; first_len];
        let mut second_actual = vec![0u128; second_len];
        TypedSetupStream::fill_p_pair(
            &mut first,
            &mut first_actual,
            &mut second,
            &mut second_actual,
        )
        .unwrap();
        first_actual.push(first.sample_p().unwrap());
        second_actual.push(second.sample_p().unwrap());
        let mut first_reference = TypedSetupStream::new(&[5; 32], p_label).unwrap();
        let mut second_reference = TypedSetupStream::new(&[5; 32], other).unwrap();
        let mut first_expected = vec![0u128; first_len + 1];
        let mut second_expected = vec![0u128; second_len + 1];
        first_reference.fill_p(&mut first_expected).unwrap();
        second_reference.fill_p(&mut second_expected).unwrap();
        assert_eq!(first_actual, first_expected);
        assert_eq!(second_actual, second_expected);
    }
    let mut p = TypedSetupStream::new(&[5; 32], p_label).unwrap();
    let mut q = TypedSetupStream::new(&[5; 32], label(SetupStreamModulus::Q(5))).unwrap();
    assert!(TypedSetupStream::fill_p_pair(&mut p, &mut [0; 1], &mut q, &mut [0; 1]).is_err());
}
