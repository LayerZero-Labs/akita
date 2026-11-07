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
            SetupStreamModulus::Prime64Offset59 => u128::from(stream.sample_p64().unwrap()),
        })
        .collect()
}

#[test]
fn exact_thresholds_cover_equal_numbers_of_each_residue() {
    for q in [3, 5, 7, 14_253_409, 33_568_993, u32::MAX] {
        let threshold = q_rejection_threshold(q).unwrap();
        assert_eq!(threshold % u64::from(q), 0);
        assert!(threshold <= 1u64 << 32);
        assert!((1u64 << 32) - threshold < u64::from(q));
        // The candidate immediately below the threshold is accepted; the
        // threshold itself is rejected by the sampler's strict inequality.
        assert_eq!((threshold - 1) % u64::from(q), u64::from(q) - 1);
    }
    assert!(q_rejection_threshold(0).is_err());
    assert!(q_rejection_threshold(1).is_err());
    assert!(q_rejection_threshold(4).is_err());
}

#[test]
fn every_label_component_and_seed_separates_both_domains() {
    for modulus in [
        SetupStreamModulus::Q(14_253_409),
        SetupStreamModulus::Prime128Offset275,
        SetupStreamModulus::Prime64Offset59,
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
                SetupStreamModulus::Prime128Offset275 | SetupStreamModulus::Prime64Offset59 => {
                    SetupStreamModulus::Q(14_253_409)
                }
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

/// Run `len` draws over `candidates` through [`PDraws`], reading blocks the
/// way [`TypedSetupStream::fill_p`] does. Returns the accepted values, or
/// `None` once the rejection budget is exhausted, and the candidates read.
fn block_p_draws(candidates: &[u128], len: usize) -> (Option<Vec<u128>>, usize) {
    let mut output = vec![0u128; len];
    let mut next = 0;
    let exhausted = {
        let mut draws = PDraws::new(&mut output);
        loop {
            if draws.done() {
                break false;
            }
            for slot in draws.block().chunks_exact_mut(16) {
                slot.copy_from_slice(&candidates[next].to_le_bytes());
                next += 1;
            }
            if draws.accept().is_err() {
                break true;
            }
        }
    };
    ((!exhausted).then_some(output), next)
}

/// `len` successive single draws, each accepting the first candidate below P
/// within its budget of 256 candidates.
fn single_p_draws(candidates: &[u128], len: usize) -> (Option<Vec<u128>>, usize) {
    let mut values = Vec::with_capacity(len);
    let mut next = 0;
    for _ in 0..len {
        let mut accepted = None;
        for _ in 0..MAX_REJECTION_ATTEMPTS {
            let candidate = candidates[next];
            next += 1;
            if candidate < BINARY_SETUP_P {
                accepted = Some(candidate);
                break;
            }
        }
        match accepted {
            Some(value) => values.push(value),
            None => return (None, next),
        }
    }
    (Some(values), next)
}

/// Synthetic candidates for the rejection paths that a real stream reaches
/// with probability about 2^-120 per candidate.
#[test]
fn block_p_draws_match_single_draws_on_rejection_paths() {
    const REJECTS: [u128; 3] = [BINARY_SETUP_P, BINARY_SETUP_P + 1, u128::MAX];
    let rejects = |count: usize| (0..count).map(|index| REJECTS[index % 3]);
    let accepts = |count: usize, from: u128| (0..count as u128).map(move |index| from + index);
    let sequence = |parts: Vec<Vec<u128>>| {
        let mut candidates: Vec<u128> = parts.into_iter().flatten().collect();
        // Enough accepted tail for any block read past the last draw.
        candidates.extend(accepts(2 * P_BLOCK, 1 << 64));
        candidates
    };
    let accepted_cases = [
        // P - 1 is accepted; P, P + 1 and 2^128 - 1 are rejected.
        (
            3,
            sequence(vec![
                vec![BINARY_SETUP_P - 1],
                rejects(3).collect(),
                vec![0, BINARY_SETUP_P - 1],
            ]),
        ),
        // A rejection run across the first block boundary of a long fill.
        (
            70,
            sequence(vec![
                accepts(62, 10).collect(),
                rejects(5).collect(),
                accepts(20, 100).collect(),
            ]),
        ),
        // 255 rejections, then an acceptance, inside one block.
        (
            3,
            sequence(vec![vec![1], rejects(255).collect(), vec![2, 3]]),
        ),
        // 255 rejections starting at the last candidate of a full block.
        (
            64,
            sequence(vec![
                accepts(63, 10).collect(),
                rejects(255).collect(),
                vec![7],
            ]),
        ),
        // The count resets at every acceptance: 400 rejections in all.
        (
            2,
            sequence(vec![
                rejects(200).collect(),
                vec![1],
                rejects(200).collect(),
                vec![2],
            ]),
        ),
    ];
    for (len, candidates) in &accepted_cases {
        let single = single_p_draws(candidates, *len);
        assert!(single.0.is_some(), "length {len}");
        assert_eq!(block_p_draws(candidates, *len), single, "length {len}");
    }
    let exhausted_cases = [
        (2, sequence(vec![vec![1], rejects(256).collect()])),
        (
            64,
            sequence(vec![accepts(10, 5).collect(), rejects(256).collect()]),
        ),
        (
            70,
            sequence(vec![accepts(63, 5).collect(), rejects(256).collect()]),
        ),
    ];
    for (len, candidates) in &exhausted_cases {
        assert_eq!(single_p_draws(candidates, *len).0, None, "length {len}");
        assert_eq!(block_p_draws(candidates, *len).0, None, "length {len}");
    }
}

/// With q = 2^31 + 1 about half of all candidates are rejected. The draws
/// must follow the reference SHAKE256 stream of the documented identity.
#[test]
fn rejection_heavy_q_draws_follow_the_reference_stream() {
    use shake::digest::{ExtendableOutput, Update, XofReader};
    let q = (1u32 << 31) + 1;
    assert_eq!(q_rejection_threshold(q).unwrap(), u64::from(q));
    let seed = [9u8; 32];
    let label = label(SetupStreamModulus::Q(q));
    let (group, level, rank) = (
        label.group.to_le_bytes(),
        label.level.to_le_bytes(),
        label.rank.to_le_bytes(),
    );
    let (row, column, degree) = (
        label.row.to_le_bytes(),
        label.column.to_le_bytes(),
        label.degree.to_le_bytes(),
    );
    let modulus = u128::from(q).to_le_bytes();
    let parts: [&[u8]; 12] = [
        SETUP_DOMAIN,
        &seed,
        &[label.role as u8],
        &group,
        &level,
        &rank,
        &row,
        &column,
        &[label.ring as u8],
        &degree,
        &[1],
        &modulus,
    ];
    let mut reference = shake::Shake256::default();
    for part in parts {
        reference.update(part);
    }
    let mut reference = reference.finalize_xof();
    let mut stream = TypedSetupStream::new(&seed, label).unwrap();
    let mut rejections = 0;
    for _ in 0..2_000 {
        let expected = loop {
            let mut bytes = [0u8; 4];
            reference.read(&mut bytes);
            let candidate = u32::from_le_bytes(bytes);
            if candidate < q {
                break if candidate > q / 2 {
                    i64::from(candidate) - i64::from(q)
                } else {
                    i64::from(candidate)
                };
            }
            rejections += 1;
        };
        assert_eq!(stream.sample_q_centered().unwrap(), expected);
    }
    assert!(rejections > 1_000, "{rejections} rejections");
}

#[test]
fn p64_draws_follow_the_versioned_eight_byte_stream() {
    use shake::digest::{ExtendableOutput, Update, XofReader};
    let seed = [42; 32];
    let p64 = label(SetupStreamModulus::Prime64Offset59);
    let mut reference = shake::Shake256::default();
    for part in [
        b"akita/binary-source/setup/shake256/v2".as_slice(),
        &seed,
        &[p64.role as u8],
        &p64.group.to_le_bytes(),
        &p64.level.to_le_bytes(),
        &p64.rank.to_le_bytes(),
        &p64.row.to_le_bytes(),
        &p64.column.to_le_bytes(),
        &[p64.ring as u8],
        &p64.degree.to_le_bytes(),
        &[3],
        &u128::from(BINARY_SETUP_P64).to_le_bytes(),
    ] {
        reference.update(part);
    }
    let mut reference = reference.finalize_xof();
    let mut stream = TypedSetupStream::new(&seed, p64).unwrap();
    for _ in 0..1_000 {
        let expected = loop {
            let mut bytes = [0; 8];
            reference.read(&mut bytes);
            let value = u64::from_le_bytes(bytes);
            if value < BINARY_SETUP_P64 {
                break value;
            }
        };
        assert_eq!(stream.sample_p64().unwrap(), expected);
    }
    assert!(stream.sample_q_centered().is_err());
    assert_ne!(
        fingerprint(seed, p64),
        fingerprint(seed, label(SetupStreamModulus::Prime128Offset275))
    );
    let mut legacy =
        TypedSetupStream::new(&seed, label(SetupStreamModulus::Prime128Offset275)).unwrap();
    assert!(legacy.sample_p64().is_err());

    // Force the rare rejection path: p - 1 accepts, p and u64::MAX reject.
    let mut stream = TypedSetupStream::new(&seed, p64).unwrap();
    for (slot, candidate) in stream.squeeze.block.chunks_exact_mut(8).zip([
        BINARY_SETUP_P64 - 1,
        BINARY_SETUP_P64,
        u64::MAX,
        17,
    ]) {
        slot.copy_from_slice(&candidate.to_le_bytes());
    }
    assert_eq!(stream.sample_p64().unwrap(), BINARY_SETUP_P64 - 1);
    assert_eq!(stream.sample_p64().unwrap(), 17);
    assert_eq!(stream.squeeze.read, 32);
}

#[test]
fn p64_block_draws_and_paired_draws_preserve_stream_positions() {
    let seed = [57; 32];
    let base = label(SetupStreamModulus::Prime64Offset59);
    let other = SetupStreamLabel { row: 1, ..base };
    let mut first = TypedSetupStream::new(&seed, base).unwrap();
    let mut second = TypedSetupStream::new(&seed, other).unwrap();
    let mut first_output = [0; 200];
    let mut second_output = [0; 65];
    TypedSetupStream::fill_p_pair(
        &mut first,
        &mut first_output,
        &mut second,
        &mut second_output,
    )
    .unwrap();
    for (label, output, mut stream) in [
        (base, first_output.as_slice(), first),
        (other, second_output.as_slice(), second),
    ] {
        let mut reference = TypedSetupStream::new(&seed, label).unwrap();
        for &value in output {
            assert_eq!(value, u128::from(reference.sample_p64().unwrap()));
        }
        assert_eq!(
            stream.sample_p().unwrap(),
            u128::from(reference.sample_p64().unwrap())
        );
    }
}

#[test]
fn paired_profiles_bind_exact_challenge_orders_and_block_unfinished_roots() {
    let p64 = BinaryFieldProfile::P64;
    assert_eq!(p64.modulus(), 18_446_744_073_709_551_557);
    assert_eq!(p64.extension_degree(), 2);
    assert_eq!(
        p64.challenge_order().unwrap(),
        crate::ChallengeFieldOrder::from_field(64, 2, p64.modulus()).unwrap()
    );
    assert_ne!(
        p64.challenge_order().unwrap(),
        BinaryFieldProfile::P128.challenge_order().unwrap()
    );
    assert!(
        matches!(p64.admit_root(p64.modulus(), 2), Err(AkitaError::UnsupportedSchedule(message)) if message.contains("not yet supported"))
    );
    for (profile, modulus, degree) in [
        (p64, p64.modulus(), 1),
        (p64, BINARY_SETUP_P, 2),
        (BinaryFieldProfile::P128, p64.modulus(), 1),
        (BinaryFieldProfile::P128, BINARY_SETUP_P, 2),
    ] {
        assert!(matches!(
            profile.admit_root(modulus, degree),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
    BinaryFieldProfile::P128
        .admit_root(BINARY_SETUP_P, 1)
        .unwrap();
}

#[test]
fn p64_digit_ranks_cover_actual_widths_and_role_boundaries() {
    use crate::sis::{
        generated_sis_max_widths, sis_role_cell, SisMatrixRole, SisModulusProfileId,
        SisSecurityPolicyId,
    };
    let field = BinaryFieldProfile::P64;
    // Read the canonical generated q64 row directly, without routing expected
    // ranks through min_secure_rank.
    let generated = generated_sis_max_widths(
        SisSecurityPolicyId::Quantum169BitADPS16,
        SisModulusProfileId::Q64Offset59,
        64,
        7,
    )
    .unwrap();
    for (role, matrix_role) in [
        (SetupStreamRole::ImageCommitment, SisMatrixRole::Outer),
        (SetupStreamRole::WitnessCommitment, SisMatrixRole::Open),
    ] {
        let cell = sis_role_cell(matrix_role, SisModulusProfileId::Q64Offset59, 64, 7).unwrap();
        let widths = &generated[..generated.len().min(cell.max_module_rank as usize)];
        let expected = |width: usize| {
            widths
                .iter()
                .position(|&cap| width as u64 <= cap)
                .map(|i| i + 1)
        };
        for (columns, owners) in [
            (2usize, 1usize),
            (2, 4096),
            (256, 1),
            (256, 10),
            (256, 4096),
        ] {
            // Natural image digits for B; U and selected coordinate planes for D.
            let planes = field.extension_degree() * owners.min(5);
            let width = match role {
                SetupStreamRole::ImageCommitment => (owners * columns * 3 * 8 * 648).div_ceil(64),
                SetupStreamRole::WitnessCommitment => {
                    ((owners + planes * field.prime_plane_digits()) * columns * 162).div_ceil(64)
                }
                _ => unreachable!(),
            };
            let rank = field.digit_commitment_rank(role, width).unwrap();
            assert_eq!(Some(rank), expected(width));
            assert!(width as u64 <= widths[rank - 1]);
            assert!(rank == 1 || width as u64 > widths[rank - 2]);
        }
        for &cap in widths {
            for width in [cap as usize, cap as usize + 1] {
                assert_eq!(
                    field.digit_commitment_rank(role, width).ok(),
                    expected(width)
                );
            }
        }
        assert!(field.digit_commitment_rank(role, 0).is_err());
        assert!(field.digit_commitment_rank(role, usize::MAX).is_err());
    }
}
