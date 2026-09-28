//! Device CRT reconstruction against `CyclotomicCrtNtt::to_ring`.
//!
//! The device reconstructs the CPU's own inverse-transform words, so any
//! difference is the reconstruction's; `transforms_then_reconstruct` then
//! checks the device inverse transform and reconstruction together.

#![cfg(target_os = "macos")]

mod support;

use akita_algebra::ntt::butterfly::{forward_ntt, inverse_ntt};
use akita_algebra::tables::{q128_primes, Q64_PRIMES};
use akita_algebra::{
    CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt, MontCoeff, NttPrime,
};
use akita_metal::ntt::DeviceCrtNtt;
use akita_metal::AkitaMetal;
use jolt_field::{Prime128OffsetA7F7, Prime64Offset59};
use jolt_metal::runtime::DeviceBuffer;
use jolt_metal::MetalField;
use support::{gpu, SplitMix64};

/// Random elements after the adversarial ones.
const RANDOM_ELEMENTS: usize = 16;

/// Canonical coefficient residues that stress Garner's digits: every
/// residue zero, one or minus one (the integers 0, 1, -1), the two halves
/// of each prime (whose digits sit at the centering boundary), and mixes of
/// these across primes (integers near the ends of the centered range).
fn adversarial_residues(p: i32, prime: usize, coefficient: usize) -> i32 {
    let patterns = [
        0,
        1,
        p - 1,
        (p - 1) / 2,
        (p + 1) / 2,
        if prime.is_multiple_of(2) {
            (p - 1) / 2
        } else {
            (p + 1) / 2
        },
        if prime == 0 { p - 1 } else { 0 },
        if prime == 0 { 0 } else { p - 1 },
    ];
    patterns[coefficient % patterns.len()]
}

/// NTT-domain elements: one whose coefficients take the adversarial
/// residues, then random words in `(-p, p)`.
fn ntt_elements<const K: usize, const D: usize>(
    params: &CrtNttParamSet<i32, K, D>,
    seed: u64,
) -> Vec<CyclotomicCrtNtt<i32, K, D>> {
    let mut rng = SplitMix64::new(seed);
    let adversarial = CyclotomicCrtNtt {
        limbs: std::array::from_fn(|prime| {
            let modulus = params.primes[prime];
            let mut limb = std::array::from_fn(|coefficient| {
                modulus.from_canonical(adversarial_residues(modulus.p, prime, coefficient))
            });
            forward_ntt(
                &mut limb,
                modulus,
                &params.twiddles[prime],
                params.kernel_plan(),
            );
            limb
        }),
    };
    std::iter::once(adversarial)
        .chain((0..RANDOM_ELEMENTS).map(|_| CyclotomicCrtNtt {
            limbs: std::array::from_fn(|prime| {
                let p = params.primes[prime].p;
                std::array::from_fn(|_| MontCoeff::from_raw(rng.next_i32_below(p)))
            }),
        }))
        .collect()
}

fn words<const K: usize, const D: usize>(elements: &[CyclotomicCrtNtt<i32, K, D>]) -> Vec<i32> {
    elements
        .iter()
        .flat_map(|element| element.limbs.iter().flatten().map(|word| word.raw()))
        .collect()
}

fn expected<F: CrtNttConvertibleField, const K: usize, const D: usize>(
    params: &CrtNttParamSet<i32, K, D>,
    elements: &[CyclotomicCrtNtt<i32, K, D>],
) -> Vec<F> {
    elements
        .iter()
        .flat_map(|element| *element.to_ring::<F>(params).coefficients())
        .collect()
}

fn check<F, const K: usize, const D: usize>(metal: &AkitaMetal, primes: [NttPrime<i32>; K])
where
    F: MetalField + CrtNttConvertibleField,
{
    let params = CrtNttParamSet::<i32, K, D>::new(primes);
    let device = DeviceCrtNtt::new(metal, &params).expect("upload tables");
    let elements = ntt_elements(&params, 0xc47 ^ (K * D) as u64);
    let expected = expected::<F, K, D>(&params, &elements);

    // Reconstruction alone, from the CPU's inverse-transform words.
    let coefficient_words = elements
        .iter()
        .flat_map(|element| {
            element.limbs.iter().enumerate().flat_map(|(prime, limb)| {
                let mut limb = *limb;
                inverse_ntt(
                    &mut limb,
                    params.primes[prime],
                    &params.twiddles[prime],
                    params.kernel_plan(),
                );
                limb.map(MontCoeff::raw)
            })
        })
        .collect::<Vec<_>>();
    let residues = DeviceBuffer::from_slice(metal.device(), &coefficient_words).expect("upload");
    let mut out = DeviceBuffer::<F>::zeroed(metal.device(), expected.len()).expect("output");
    device
        .reconstruct(metal, &residues, &mut out)
        .expect("reconstruct");
    assert_eq!(
        out.read().expect("canonical read-back"),
        expected.as_slice(),
        "reconstruct K={K} D={D}"
    );

    // Device inverse transform, then reconstruction.
    let mut residues = DeviceBuffer::from_slice(metal.device(), &words(&elements)).expect("upload");
    device.inverse(metal, &mut residues).expect("inverse");
    let mut out = DeviceBuffer::<F>::zeroed(metal.device(), expected.len()).expect("output");
    device
        .reconstruct(metal, &residues, &mut out)
        .expect("reconstruct");
    assert_eq!(
        out.read().expect("canonical read-back"),
        expected.as_slice(),
        "inverse + reconstruct K={K} D={D}"
    );
}

#[test]
fn fp128_reconstructs_from_q128_like_cpu() {
    let test = gpu();
    check::<Prime128OffsetA7F7, 6, 64>(&test.metal, q128_primes());
    check::<Prime128OffsetA7F7, 6, 256>(&test.metal, q128_primes());
    check::<Prime128OffsetA7F7, 6, 1024>(&test.metal, q128_primes());
}

#[test]
fn fp64_reconstructs_from_q64_like_cpu() {
    let test = gpu();
    check::<Prime64Offset59, 3, 64>(&test.metal, Q64_PRIMES);
    check::<Prime64Offset59, 3, 256>(&test.metal, Q64_PRIMES);
    check::<Prime64Offset59, 3, 1024>(&test.metal, Q64_PRIMES);
}

#[test]
fn rejects_mismatched_output_length() {
    let test = gpu();
    let params = CrtNttParamSet::<i32, 3, 64>::new(Q64_PRIMES);
    let device = DeviceCrtNtt::new(&test.metal, &params).expect("upload tables");
    let residues = DeviceBuffer::<i32>::zeroed(test.metal.device(), 3 * 64).expect("zeroed");
    let mut out = DeviceBuffer::<Prime64Offset59>::zeroed(test.metal.device(), 63).expect("zeroed");
    let error = device
        .reconstruct(&test.metal, &residues, &mut out)
        .expect_err("short output");
    assert_eq!(error.class(), akita_metal::ErrorClass::Setup);
}
