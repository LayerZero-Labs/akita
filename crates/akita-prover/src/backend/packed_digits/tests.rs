use std::{hint::black_box, time::Instant};

use rand::{rngs::StdRng, Rng, SeedableRng};

use super::*;

fn digit_range(bit_width: u8) -> std::ops::Range<i16> {
    let half = 1i16 << (bit_width - 1);
    -half..half
}

fn random_digits(rng: &mut StdRng, len: usize, bit_width: u8) -> Vec<i8> {
    let range = digit_range(bit_width);
    (0..len)
        .map(|_| rng.gen_range(range.clone()) as i8)
        .collect()
}

#[test]
fn exact_two_complement_encoding_has_no_block_headers() {
    let packed = PackedSignedDigits::from_i8_digits(vec![-2, -1, 0, 1], 2).unwrap();
    assert_eq!(packed.encoded_bytes(), &[0x4e]);
    assert_eq!(packed.decode(), [-2, -1, 0, 1]);
}

#[test]
fn round_trips_every_signed_width_and_tail_shape() {
    let mut rng = StdRng::seed_from_u64(0x5e3d_1a17);
    let lengths = [0, 1, 7, 8, 15, 16, 63, 64, 65, 127, 128, 129];
    for bit_width in 1..=8 {
        for len in lengths {
            let digits = random_digits(&mut rng, len, bit_width);
            let packed = PackedSignedDigits::from_i8_digits(digits.clone(), bit_width).unwrap();
            assert_eq!(packed.len(), len);
            assert_eq!(packed.bit_width(), bit_width);
            assert_eq!(packed.decode(), digits);
            let mut decoded = vec![0i8; len];
            packed.decode_into(&mut decoded).unwrap();
            assert_eq!(decoded, digits);
            assert_eq!(
                packed.encoded_bytes().len(),
                (len * usize::from(bit_width)).div_ceil(8)
            );
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_block_encoder_round_trips_every_width_and_a_partial_tail() {
    const LEN: usize = (1 << 16) + 13;
    let mut rng = StdRng::seed_from_u64(0x7061_636b_6564);
    for bit_width in 1..=8 {
        let digits = random_digits(&mut rng, LEN, bit_width);
        let packed = PackedSignedDigits::from_i8_digits(digits.clone(), bit_width).unwrap();
        assert_eq!(packed.decode(), digits, "bit width {bit_width}");
    }
}

#[test]
fn rejects_widths_and_digits_that_do_not_fit() {
    for bit_width in [0, 9, u8::MAX] {
        assert!(PackedSignedDigits::from_i8_digits(Vec::new(), bit_width).is_err());
    }
    assert!(PackedSignedDigits::from_i8_digits(vec![-3], 2).is_err());
    assert!(PackedSignedDigits::from_i8_digits(vec![2], 2).is_err());
    assert!(PackedSignedDigits::from_i8_digits(vec![i8::MIN, i8::MAX], 8).is_ok());

    let packed = PackedSignedDigits::from_i8_digits(vec![0, 1], 2).unwrap();
    assert!(packed.decode_into(&mut [0]).is_err());
}

#[test]
fn records_exact_bounds_during_the_pack() {
    let packed = PackedSignedDigits::from_i8_digits(vec![-7, -2, 0, 3, 6], 4).unwrap();
    assert_eq!(packed.bounds().negative_abs_max(), 7);
    assert_eq!(packed.bounds().positive_max(), 6);

    let empty = PackedSignedDigits::from_i8_digits(Vec::new(), 3).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.bounds().negative_abs_max(), 0);
    assert_eq!(empty.bounds().positive_max(), 0);
}

#[test]
fn automatic_width_is_the_smallest_exact_signed_width() {
    for (digits, expected_width) in [
        (vec![0], 1),
        (vec![-1, 0], 1),
        (vec![-2, 1], 2),
        (vec![-4, 3], 3),
        (vec![-32, 31], 6),
        (vec![i8::MIN, i8::MAX], 8),
    ] {
        let packed = PackedSignedDigits::from_i8_digits_auto(digits.clone());
        assert_eq!(packed.bit_width(), expected_width);
        assert_eq!(packed.decode(), digits);
    }
}

#[test]
fn zero_padding_is_metadata_not_encoded_storage() {
    let digits = (0..70).map(|index| (index % 4) as i8 - 2).collect();
    let packed = PackedSignedDigits::from_i8_digits(digits, 2).unwrap();
    let encoded_len = packed.encoded_bytes().len();
    let view = packed.zero_padded(256).unwrap();
    assert_eq!(view.len(), 256);
    assert_eq!(view.block_count(), 4);
    assert_eq!(packed.encoded_bytes().len(), encoded_len);

    let mut block = [99i8; DIGITS_PER_BLOCK];
    assert_eq!(view.decode_block(1, &mut block).unwrap(), 6);
    assert_eq!(&block[..6], &[-2, -1, 0, 1, -2, -1]);
    assert!(block[6..].iter().all(|&digit| digit == 0));
    assert_eq!(view.decode_block(2, &mut block).unwrap(), 0);
    assert!(block.iter().all(|&digit| digit == 0));
    assert!(view.decode_block(4, &mut block).is_err());
    assert!(packed.zero_padded(69).is_err());
}

#[test]
fn random_access_matches_the_source() {
    let mut rng = StdRng::seed_from_u64(0xb17b_0a4d);
    for bit_width in 1..=8 {
        let digits = random_digits(&mut rng, 257, bit_width);
        let packed = PackedSignedDigits::from_i8_digits(digits.clone(), bit_width).unwrap();
        for (index, &digit) in digits.iter().enumerate() {
            assert_eq!(packed.get(index), Some(digit));
        }
        assert_eq!(packed.get(digits.len()), None);
    }
}

#[test]
fn aligned_range_decode_uses_implicit_zero_suffix() {
    let digits = (0..150)
        .map(|index| (index % 8) as i8 - 4)
        .collect::<Vec<_>>();
    let packed = PackedSignedDigits::from_i8_digits(digits.clone(), 4).unwrap();
    let view = packed.zero_padded(256).unwrap();
    let mut decoded = [99i8; 128];
    assert_eq!(view.decode_range(64, &mut decoded).unwrap(), 86);
    assert_eq!(&decoded[..86], &digits[64..]);
    assert!(decoded[86..].iter().all(|&digit| digit == 0));
}

#[test]
fn unaligned_subview_iteration_matches_the_source_and_zero_suffix() {
    let digits = (0..197)
        .map(|index| (index % 16) as i8 - 8)
        .collect::<Vec<_>>();
    let packed = PackedSignedDigits::from_i8_digits(digits.clone(), 5).unwrap();
    let view = packed.zero_padded(256).unwrap().slice(13..229).unwrap();
    let expected = digits[13..]
        .iter()
        .copied()
        .chain(std::iter::repeat_n(0, 229 - digits.len()))
        .collect::<Vec<_>>();

    assert_eq!(view.iter().collect::<Vec<_>>(), expected);
    let mut decoded = vec![99; view.len()];
    assert_eq!(
        view.decode_range(0, &mut decoded).unwrap(),
        digits.len() - 13
    );
    assert_eq!(decoded, expected);
}

#[test]
fn unaligned_subview_full_block_decode_matches_the_source() {
    let digits = (0..160)
        .map(|index| (index % 16) as i8 - 8)
        .collect::<Vec<_>>();
    let packed = PackedSignedDigits::from_i8_digits(digits.clone(), 5).unwrap();
    let view = packed.view().slice(13..141).unwrap();
    let mut decoded = [99; DIGITS_PER_BLOCK];

    assert_eq!(
        view.decode_block(0, &mut decoded).unwrap(),
        DIGITS_PER_BLOCK
    );
    assert_eq!(decoded.as_slice(), &digits[13..77]);
}

#[test]
fn iterator_arrays_cross_decode_block_boundaries_exactly() {
    let digits = (0..140)
        .map(|index| (index % 8) as i8 - 4)
        .collect::<Vec<_>>();
    let packed = PackedSignedDigits::from_i8_digits(digits.clone(), 4).unwrap();
    let mut iter = packed.view().slice(62..134).unwrap().iter();
    for expected in digits[62..134].chunks_exact(4) {
        assert_eq!(iter.next_array::<4>().unwrap().as_slice(), expected);
    }
    assert!(iter.next_array::<4>().is_none());
}

#[test]
fn architecture_decoder_matches_scalar_blocks() {
    let mut rng = StdRng::seed_from_u64(0xa4c4_17ec);
    for bit_width in 1..=8 {
        let digits = random_digits(&mut rng, DIGITS_PER_BLOCK * 3, bit_width);
        let packed = PackedSignedDigits::from_i8_digits(digits, bit_width).unwrap();
        for block_index in 0..3 {
            let offset = block_index * usize::from(bit_width) * 8;
            let mut expected = [0i8; DIGITS_PER_BLOCK];
            scalar::decode_full_block(&packed.storage[offset..], bit_width, &mut expected);
            let mut actual = [0i8; DIGITS_PER_BLOCK];
            decode_full_block(&packed, block_index, &mut actual);
            assert_eq!(
                actual, expected,
                "bit width {bit_width}, block {block_index}"
            );
        }
    }
}

#[test]
#[ignore = "diagnostic microbenchmark; run explicitly with --release --ignored --nocapture"]
fn decode_microbenchmark() {
    const LEN: usize = 1 << 22;
    const SAMPLES: usize = 9;
    let mut rng = StdRng::seed_from_u64(0xdec0_de64);
    for bit_width in 2..=6 {
        let digits = random_digits(&mut rng, LEN, bit_width);
        let packed = PackedSignedDigits::from_i8_digits(digits, bit_width).unwrap();
        let mut decoded = vec![0i8; LEN];
        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let started = Instant::now();
            packed.decode_into(black_box(&mut decoded)).unwrap();
            black_box(&decoded);
            samples.push(started.elapsed());
        }
        samples.sort_unstable();
        let median = samples[SAMPLES / 2];
        eprintln!(
            "packed decode L{bit_width}: {:.3} ms, {:.2} billion digits/s",
            median.as_secs_f64() * 1_000.0,
            LEN as f64 / median.as_secs_f64() / 1e9
        );
    }
}
