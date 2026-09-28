//! akita/mont.h operation by operation against `NttPrime`, on every pair of
//! edge words and on seeded random words, for every production prime.

#![cfg(target_os = "macos")]

mod support;

use akita_algebra::tables::{q128_primes, Q64_PRIMES};
use akita_algebra::{MontCoeff, NttPrime};
use akita_metal::HEADERS;
use bytemuck::{Pod, Zeroable};
use jolt_metal::runtime::{Batch, Binding, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary};
use support::{gpu, SplitMix64};

const OPS: usize = 8;
const RANDOM_PAIRS: usize = 1 << 16;

/// Layout of `akita::NttPrime`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Prime {
    p: i32,
    pinv: i32,
    mont: i32,
    montsq: i32,
}

fn edge_words(p: i32) -> Vec<i32> {
    let mut words = vec![
        0,
        1,
        -1,
        2,
        p / 2,
        -(p / 2),
        p / 2 + 1,
        -(p / 2) - 1,
        i32::MAX,
        i32::MIN,
        i32::MIN + 1,
        1 << 30,
        -(1 << 30),
    ];
    // Every word within two of 0, +-p and +-2p: the boundaries of the lazy
    // ranges and of each conditional correction.
    for center in [p, -p, 2 * p - 2, -(2 * p - 2)] {
        words.extend((-2..=2).map(|offset| center + offset));
    }
    words
}

/// The CPU result of every operation, in `MontOp` order.
fn cpu_ops(prime: NttPrime<i32>, a: i32, b: i32, canonical: i32) -> [i32; OPS] {
    let a_coefficient = MontCoeff::from_raw(a);
    [
        prime.mul(a_coefficient, MontCoeff::from_raw(b)).raw(),
        prime.csubp(a_coefficient).raw(),
        prime.caddp(a_coefficient).raw(),
        prime.reduce_range(a_coefficient).raw(),
        prime.normalize(a_coefficient).raw(),
        prime.from_canonical(canonical).raw(),
        prime.to_canonical(a_coefficient),
        prime.center(canonical),
    ]
}

fn check_prime(test: &support::GpuTest, library: &ShaderLibrary, prime: NttPrime<i32>) {
    let edges = edge_words(prime.p);
    let mut rng = SplitMix64::new(prime.p as u64);
    let (lhs, rhs): (Vec<i32>, Vec<i32>) = edges
        .iter()
        .flat_map(|&a| edges.iter().map(move |&b| (a, b)))
        .chain((0..RANDOM_PAIRS).map(|_| (rng.next_i32(), rng.next_i32())))
        .unzip();
    let canonicals = lhs
        .iter()
        .map(|&a| a.rem_euclid(prime.p))
        .collect::<Vec<_>>();
    let device = test.metal.device();
    let count = u32::try_from(lhs.len()).expect("count fits u32");
    let lhs_buffer = DeviceBuffer::from_slice(device, &lhs).expect("upload");
    let rhs_buffer = DeviceBuffer::from_slice(device, &rhs).expect("upload");
    let canonical_buffer = DeviceBuffer::from_slice(device, &canonicals).expect("upload");
    let prime_buffer = DeviceBuffer::from_slice(
        device,
        &[Prime {
            p: prime.p,
            pinv: prime.pinv,
            mont: prime.mont,
            montsq: prime.montsq,
        }],
    )
    .expect("upload");
    let mut out = DeviceBuffer::<i32>::zeroed(device, lhs.len() * OPS).expect("output");
    {
        let mut batch = Batch::new(device).expect("batch");
        batch
            .dispatch(
                library.pipeline("akita_test_mont_ops").expect("pipeline"),
                &[
                    Binding::buffer(&lhs_buffer),
                    Binding::buffer(&rhs_buffer),
                    Binding::buffer(&canonical_buffer),
                    Binding::buffer(&out),
                    Binding::buffer(&prime_buffer),
                    Binding::value(&count),
                ],
                Grid::linear(lhs.len(), 256),
            )
            .expect("dispatch");
        batch.commit_and_wait().expect("run");
    }
    let out = out.read().expect("read back");
    let names = [
        "mul",
        "csubp",
        "caddp",
        "reduce_range",
        "normalize",
        "from_canonical",
        "to_canonical",
        "center",
    ];
    for (index, ((&a, &b), &canonical)) in lhs.iter().zip(&rhs).zip(&canonicals).enumerate() {
        let expected = cpu_ops(prime, a, b, canonical);
        let got = &out[index * OPS..(index + 1) * OPS];
        for op in 0..OPS {
            assert_eq!(
                got[op], expected[op],
                "{} p={} a={a} b={b} canonical={canonical}",
                names[op], prime.p
            );
        }
    }
}

#[test]
fn montgomery_ops_match_ntt_prime() {
    let test = gpu();
    // Akita's headers follow jolt-metal's field headers, as in the library.
    let spec = jolt_metal::shaders::FIELD_HEADERS
        .iter()
        .chain(&HEADERS)
        .fold(LibrarySpec::new(), |spec, (name, text)| {
            spec.source(name, text)
        })
        .source("mont_ops.metal", include_str!("shaders/mont_ops.metal"))
        .kernel("akita_test_mont_ops");
    let library = ShaderLibrary::compile(test.metal.device(), &spec).expect("compile");
    for prime in q128_primes()
        .into_iter()
        .chain(Q64_PRIMES)
        .chain([NttPrime::new(12289)])
    {
        check_prime(&test, &library, prime);
    }
}
