//! Device balanced decomposition against the CPU's, digit for digit.

#![cfg(target_os = "macos")]

mod support;

use akita_algebra::ring::cyclotomic::{decompose_centering_threshold, BalancedDecomposePow2Params};
use akita_algebra::{CanonicalEncoding, CyclotomicRing, Field};
use akita_cpu_backend::benchmark_support::decompose_rows_i8_into;
use akita_metal::decompose::decompose;
use akita_metal::{AkitaMetal, AkitaMetalError};
use jolt_field::{Prime128Offset275, Prime128OffsetA7F7, Prime64Offset59, Ring};
use jolt_metal::runtime::DeviceBuffer;
use jolt_metal::MetalField;
use support::{gpu, SplitMix64};

const D: usize = 64;
const RINGS: usize = 24;

/// Coefficients at every centering and digit boundary, then random ones.
fn coefficients<F: Field + CanonicalEncoding>(levels: usize, log_basis: u32, seed: u64) -> Vec<F> {
    let q = (-F::one()).to_u128_checked().expect("u128 field") + 1;
    let threshold = decompose_centering_threshold(levels, log_basis, q);
    let mut edges = vec![
        0,
        1,
        2,
        q - 1,
        q - 2,
        q / 2,
        q / 2 + 1,
        threshold,
        threshold + 1,
        threshold.saturating_sub(1),
        1 << 63,
        (1 << 64) - 1,
        1 << 64,
    ];
    if q > 1 << 127 {
        edges.extend([1 << 127, (1 << 127) - 1, (1 << 127) + 1, q - (1 << 127)]);
    }
    // Each digit boundary 2^(k b) and its neighbours.
    let mut power = Some(1u128 << log_basis);
    while let Some(boundary) = power.filter(|&boundary| boundary < q / 2) {
        edges.extend([
            boundary - 1,
            boundary,
            boundary + 1,
            boundary / 2,
            q - boundary,
            q - boundary / 2,
        ]);
        power = boundary.checked_mul(1 << log_basis);
    }
    let mut rng = SplitMix64::new(seed);
    let mut values = edges
        .into_iter()
        .filter(|&value| value < q)
        .collect::<Vec<_>>();
    while values.len() < RINGS * D {
        values.push((u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64())) % q);
    }
    values.truncate(RINGS * D);
    values.into_iter().map(F::from_u128_reduced).collect()
}

fn rings<F: Field>(coefficients: &[F]) -> Vec<CyclotomicRing<F, D>> {
    coefficients
        .chunks_exact(D)
        .map(|ring| CyclotomicRing::from_coefficients(std::array::from_fn(|index| ring[index])))
        .collect()
}

fn check_i8<F: MetalField + CanonicalEncoding>(metal: &AkitaMetal, levels: usize, log_basis: u32) {
    let input = coefficients::<F>(levels, log_basis, levels as u64 * 31 + u64::from(log_basis));
    let mut expected = vec![[0i8; D]; RINGS * levels];
    decompose_rows_i8_into(&rings(&input), &mut expected, levels, log_basis);
    let buffer = DeviceBuffer::from_slice(metal.device(), &input).expect("upload");
    let (mut out, _) = decompose::<F, i8>(metal, &buffer, D, levels, log_basis).expect("decompose");
    assert_eq!(
        out.read().expect("read back"),
        expected.as_flattened(),
        "i8 levels={levels} b={log_basis}"
    );
}

fn check_i16<F: MetalField + CanonicalEncoding>(metal: &AkitaMetal, levels: usize, log_basis: u32) {
    let input = coefficients::<F>(levels, log_basis, levels as u64 * 37 + u64::from(log_basis));
    let params = BalancedDecomposePow2Params::<F>::new(levels, log_basis);
    let expected = rings(&input)
        .iter()
        .flat_map(|ring| {
            let mut planes = vec![[0i16; D]; levels];
            ring.balanced_decompose_pow2_i16_into(&mut planes, &params);
            planes
        })
        .collect::<Vec<_>>();
    let buffer = DeviceBuffer::from_slice(metal.device(), &input).expect("upload");
    let (mut out, _) =
        decompose::<F, i16>(metal, &buffer, D, levels, log_basis).expect("decompose");
    assert_eq!(
        out.read().expect("read back"),
        expected.as_flattened(),
        "i16 levels={levels} b={log_basis}"
    );
}

#[test]
fn fp128_digits_match_cpu() {
    let test = gpu();
    // The outer commitment's base-8 digits, 8-bit digits spanning exactly
    // 128 bits (the asymmetric centering threshold), and single bits.
    for (levels, log_basis) in [(43, 3), (16, 8), (32, 4), (128, 1), (1, 3)] {
        check_i8::<Prime128OffsetA7F7>(&test.metal, levels, log_basis);
    }
    // Dense inner digits: 8 x 16 bits (asymmetric) and 12 x 11 bits.
    for (levels, log_basis) in [(8, 16), (12, 11), (9, 15)] {
        check_i16::<Prime128OffsetA7F7>(&test.metal, levels, log_basis);
    }
}

#[test]
fn fp64_digits_match_cpu() {
    let test = gpu();
    for (levels, log_basis) in [(8, 8), (22, 3), (11, 6), (64, 1)] {
        check_i8::<Prime64Offset59>(&test.metal, levels, log_basis);
    }
    for (levels, log_basis) in [(6, 11), (4, 16)] {
        check_i16::<Prime64Offset59>(&test.metal, levels, log_basis);
    }
}

#[test]
fn unsupported_field_rejects_before_output_allocation() {
    let test = gpu();
    let input = [Prime128Offset275::from_u64(1); D];
    let buffer = DeviceBuffer::from_slice(test.metal.device(), &input).expect("input");
    let error = match decompose::<Prime128Offset275, i8>(&test.metal, &buffer, D, 1, 3) {
        Ok(_) => panic!("fp128 offset 275 has no Akita decomposition kernel"),
        Err(error) => error,
    };
    assert!(matches!(&error, AkitaMetalError::Shape(_)), "{error}");
}
