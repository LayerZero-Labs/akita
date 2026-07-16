use super::coeffs::balanced_decompose_centered_i32_i8_into;
use super::evals::build_w_evals_compact;
use akita_algebra::CyclotomicRing;
use akita_field::Prime128OffsetA7F7;
use std::array::from_fn;

#[test]
fn centered_i32_decompose_matches_ring_decompose() {
    type F = Prime128OffsetA7F7;
    const D: usize = 128;

    let centered = from_fn(|i| ((37 * i as i32 + 11) % 95) - 47);
    let ring =
        CyclotomicRing::<F, D>::from_coefficients(from_fn(|i| F::from_i64(centered[i] as i64)));

    for (num_digits, log_basis) in [
        (7usize, 3u32),
        (10usize, 2u32),
        (5usize, 5u32),
        (4usize, 6u32),
    ] {
        let mut got = vec![[0i8; D]; num_digits];
        balanced_decompose_centered_i32_i8_into(&centered, &mut got, log_basis);

        let mut expected = vec![[0i8; D]; num_digits];
        ring.balanced_decompose_pow2_i8_into(&mut expected, log_basis);
        assert_eq!(
            got, expected,
            "centered i32 decomposition mismatch for num_digits={num_digits} log_basis={log_basis}"
        );
    }
}

#[test]
fn compact_witness_keeps_exact_live_prefix() {
    let witness = (0..(5 * 8)).map(|value| value as i8).collect::<Vec<_>>();
    let (compact, col_bits, ring_bits) =
        build_w_evals_compact(witness.clone().into(), 8, 1, 7).expect("valid compact witness");

    assert_eq!(compact.as_ref(), witness.as_slice());
    assert_eq!(col_bits, 3);
    assert_eq!(ring_bits, 3);
}

#[test]
fn packed_compact_witness_keeps_exact_live_prefix() {
    let witness = (0..(5 * 8)).map(|value| value as i8).collect::<Vec<_>>();
    let (compact, col_bits, ring_bits) =
        build_w_evals_compact(witness.clone().into(), 8, 2, 7).expect("valid packed witness");

    assert_eq!(compact.len(), 5 * 4);
    assert_eq!(col_bits, 3);
    assert_eq!(ring_bits, 2);
}
