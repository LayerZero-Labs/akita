//! Device negacyclic NTTs against the CPU's scalar transforms, word for word.

#![cfg(target_os = "macos")]

mod support;

use akita_algebra::ntt::butterfly::{forward_ntt, inverse_ntt};
use akita_algebra::tables::{q128_primes, Q64_PRIMES};
use akita_algebra::{CrtNttParamSet, MontCoeff, NttPrime};
use akita_metal::ntt::DeviceCrtNtt;
use akita_metal::AkitaMetal;
use jolt_metal::runtime::DeviceBuffer;
use support::{gpu, SplitMix64};

/// Ring elements per test batch.
const ELEMENTS: usize = 24;

/// Words that exercise the lazy ranges: zero, units, the range ends of
/// `(-p, p)` and `(-2p, 2p)`, and the i32 extremes.
fn edge_words(p: i32) -> Vec<i32> {
    vec![
        0,
        1,
        -1,
        p - 1,
        1 - p,
        p,
        -p,
        p + 1,
        -p - 1,
        2 * p - 1,
        1 - 2 * p,
        p / 2,
        -(p / 2),
        1 << 30,
        -(1 << 30),
        i32::MAX,
        i32::MIN,
        i32::MIN + 1,
    ]
}

/// `ELEMENTS` ring elements in `[element][prime][coefficient]` order. The
/// first element cycles through the edge words of each prime; the rest draw
/// from `(-bound(p), bound(p))`, where `None` means any i32.
fn inputs<const K: usize, const D: usize>(
    primes: &[NttPrime<i32>; K],
    bound: impl Fn(i32) -> Option<i32>,
    edges: impl Fn(i32) -> bool,
    seed: u64,
) -> Vec<i32> {
    let mut rng = SplitMix64::new(seed);
    let mut words = Vec::with_capacity(ELEMENTS * K * D);
    for element in 0..ELEMENTS {
        for prime in primes {
            let edge = edge_words(prime.p)
                .into_iter()
                .filter(|&word| edges(word))
                .collect::<Vec<_>>();
            for coefficient in 0..D {
                words.push(if element == 0 {
                    edge[coefficient % edge.len()]
                } else {
                    match bound(prime.p) {
                        Some(bound) => rng.next_i32_below(bound),
                        None => rng.next_i32(),
                    }
                });
            }
        }
    }
    words
}

/// `forward_ntt` or `inverse_ntt`.
type CpuTransform<const D: usize> = fn(
    &mut [MontCoeff<i32>; D],
    NttPrime<i32>,
    &akita_algebra::ntt::NttTwiddles<i32, D>,
    akita_algebra::NttKernelPlan,
);

/// Applies `cpu` to each `(element, prime)` row of `words`.
fn cpu_rows<const K: usize, const D: usize>(
    params: &CrtNttParamSet<i32, K, D>,
    words: &[i32],
    cpu: CpuTransform<D>,
) -> Vec<i32> {
    words
        .chunks_exact(D)
        .enumerate()
        .flat_map(|(row, coefficients)| {
            let prime = row % K;
            let mut limb: [MontCoeff<i32>; D] =
                std::array::from_fn(|index| MontCoeff::from_raw(coefficients[index]));
            cpu(
                &mut limb,
                params.primes[prime],
                &params.twiddles[prime],
                params.kernel_plan(),
            );
            limb.map(MontCoeff::raw)
        })
        .collect()
}

fn check_profile<const K: usize, const D: usize>(metal: &AkitaMetal, primes: [NttPrime<i32>; K]) {
    let params = CrtNttParamSet::<i32, K, D>::new(primes);
    let device = DeviceCrtNtt::new(metal, &params).expect("upload tables");

    // Forward accepts any i32 word.
    let input = inputs::<K, D>(&primes, |_| None, |_| true, 0x5eed ^ (K * D) as u64);
    let mut buffer = DeviceBuffer::from_slice(metal.device(), &input).expect("upload");
    device.forward(metal, &mut buffer).expect("forward");
    let expected = cpu_rows(&params, &input, forward_ntt);
    assert_eq!(
        buffer.read().expect("read back"),
        expected.as_slice(),
        "forward K={K} D={D}"
    );

    // Inverse needs inputs in (-p, p), the forward transform's output range.
    let input = inputs::<K, D>(
        &primes,
        Some,
        |word| {
            primes
                .iter()
                .all(|prime| word.unsigned_abs() < prime.p.unsigned_abs())
        },
        0xfeed ^ (K * D) as u64,
    );
    let mut buffer = DeviceBuffer::from_slice(metal.device(), &input).expect("upload");
    device.inverse(metal, &mut buffer).expect("inverse");
    let expected = cpu_rows(&params, &input, inverse_ntt);
    assert_eq!(
        buffer.read().expect("read back"),
        expected.as_slice(),
        "inverse K={K} D={D}"
    );
}

fn check_all_degrees<const K: usize>(metal: &AkitaMetal, primes: [NttPrime<i32>; K]) {
    check_profile::<K, 64>(metal, primes);
    check_profile::<K, 128>(metal, primes);
    check_profile::<K, 256>(metal, primes);
    check_profile::<K, 512>(metal, primes);
    check_profile::<K, 1024>(metal, primes);
}

#[test]
fn q128_transforms_match_cpu_words() {
    let test = gpu();
    check_all_degrees(&test.metal, q128_primes());
}

#[test]
fn q64_transforms_match_cpu_words() {
    let test = gpu();
    check_all_degrees(&test.metal, Q64_PRIMES);
}

/// The 14-bit exactness tail prime runs through the same i32 kernels.
#[test]
fn tail_prime_transforms_match_cpu_words() {
    let test = gpu();
    check_all_degrees(&test.metal, [NttPrime::new(12289)]);
}

#[test]
fn rejects_partial_ring_buffers() {
    let test = gpu();
    let params = CrtNttParamSet::<i32, 3, 64>::new(Q64_PRIMES);
    let device = DeviceCrtNtt::new(&test.metal, &params).expect("upload tables");
    let mut buffer = DeviceBuffer::<i32>::zeroed(test.metal.device(), 3 * 64 + 1).expect("zeroed");
    let error = device
        .forward(&test.metal, &mut buffer)
        .expect_err("partial ring");
    assert_eq!(error.class(), akita_metal::ErrorClass::Setup);
}
