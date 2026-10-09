//! Arithmetic-only binary commitment benchmarks; setup and preparation are untimed.

#![cfg(feature = "labinius")]

use std::fmt::Debug;
use std::hint::black_box;
use std::time::Duration;

use akita_algebra::{
    binary::BinaryField128, MinusTrinomial, Prime64Offset23703, SmoothFftField, TrinomialNtt,
    TrinomialRing,
};
use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
use akita_labinius_prover::{
    commit_binary_clear, commit_binary_clear_prepared, commit_kernel::pack_binary_element_i8,
    PreparedCommitMatrix,
};
use akita_labinius_verifier::profile::BinaryClearSetup;
use akita_params::sis::labinius::{LabiniusCoefficientPrime, LabiniusRingDegree};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use jolt_field::{Prime128OffsetA7F7, WithPacking};
use rand::{rngs::StdRng, RngCore, SeedableRng};

const D: usize = 648;
const K: usize = 4;
const M: usize = 4096;
const PAYLOAD_BITS: u64 = (K * M * 128) as u64;

fn random_word(rng: &mut StdRng) -> u128 {
    u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64)
}

fn benchmark_profile<F>(
    criterion: &mut Criterion,
    field: &str,
    prime: LabiniusCoefficientPrime,
    n_a: usize,
) where
    F: SmoothFftField + WithPacking + Debug,
{
    let mut rng = StdRng::seed_from_u64(0x215_197);
    let matrix = (0..n_a * M)
        .map(|_| {
            TrinomialRing::<F, D, MinusTrinomial>::from_coefficients(std::array::from_fn(|_| {
                F::from_u128(random_word(&mut rng))
            }))
            .expect("benchmark degree is valid")
        })
        .collect();
    let profile = BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 47)
        .expect("benchmark challenge profile is valid");
    let setup = BinaryClearSetup::new(
        matrix,
        n_a,
        M,
        1,
        -1024,
        1024,
        128,
        profile,
        prime,
        LabiniusRingDegree::D648,
    )
    .expect("benchmark setup is valid");
    let source = (0..setup.source_len())
        .map(|_| random_word(&mut rng))
        .collect::<Vec<_>>();
    let prepared = PreparedCommitMatrix::prepare(&setup).expect("benchmark matrix is valid");
    eprintln!(
        "{field}/minus/D={D}/n_a={n_a}/m={M}: matrix={} bytes, table={} bytes, prepared_payload={} bytes, payload={PAYLOAD_BITS} bits",
        prepared.matrix_bytes(),
        prepared.table_bytes(),
        prepared.prepared_bytes(),
    );

    let digits = source
        .chunks_exact(K)
        .map(|chunk| {
            pack_binary_element_i8::<BinaryField128, D>(chunk)
                .expect("benchmark source packs into signed bits")
        })
        .collect::<Vec<_>>();
    let domain = prepared.domain();
    let lut = prepared.i8_lut();
    let zero = domain.zero_ntt();
    let mut transformed = vec![zero.clone(); M];
    let mut forward_workspace = domain.workspace();
    for (input, output) in digits.iter().zip(&mut transformed) {
        domain
            .forward_i8_with_lut_into_workspace(input, lut, output, &mut forward_workspace)
            .expect("benchmark signed bits lie in the lookup table range");
    }
    let mut accumulators = vec![zero.clone(); n_a];
    accumulate(prepared.matrix_ntt(), &transformed, &mut accumulators);

    let parameter = format!("{field}/minus/D={D}/n_a={n_a}/m={M}");
    let mut group = criterion.benchmark_group(format!("commit_kernel/{parameter}"));
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(PAYLOAD_BITS));

    group.bench_function("prepared", |b| {
        b.iter(|| {
            black_box(
                commit_binary_clear_prepared::<BinaryField128, F, D, MinusTrinomial>(
                    black_box(&prepared),
                    black_box(&setup),
                    black_box(&source),
                )
                .expect("benchmark source has valid geometry"),
            )
        });
    });
    group.bench_function("forward_i8", |b| {
        b.iter(|| {
            for (input, output) in black_box(&digits).iter().zip(&mut transformed) {
                domain
                    .forward_i8_with_lut_into_workspace(
                        input,
                        black_box(lut),
                        output,
                        black_box(&mut forward_workspace),
                    )
                    .expect("benchmark signed bits lie in the lookup table range");
            }
            black_box(&transformed);
        });
    });
    group.bench_function("multiply_accumulate", |b| {
        b.iter(|| {
            accumulators.fill(zero.clone());
            accumulate(
                black_box(prepared.matrix_ntt()),
                black_box(&transformed),
                &mut accumulators,
            );
            black_box(&accumulators);
        });
    });
    let mut inverse_workspace = domain.workspace();
    group.bench_function("inverse", |b| {
        b.iter(|| {
            for accumulator in black_box(&accumulators) {
                black_box(
                    domain.inverse_with_workspace(accumulator, black_box(&mut inverse_workspace)),
                );
            }
        });
    });
    group.finish();

    let mut reference = criterion.benchmark_group(format!("commit_reference/{parameter}"));
    reference.sample_size(10);
    reference.warm_up_time(Duration::from_secs(1));
    reference.measurement_time(Duration::from_secs(2));
    reference.throughput(Throughput::Elements(PAYLOAD_BITS));
    reference.bench_function("commit_binary_clear", |b| {
        b.iter(|| {
            black_box(
                commit_binary_clear::<BinaryField128, F, D, MinusTrinomial>(
                    black_box(&setup),
                    black_box(&source),
                )
                .expect("benchmark source has valid geometry"),
            )
        });
    });
    reference.finish();
}

fn accumulate<F: SmoothFftField + WithPacking>(
    matrix: &[TrinomialNtt<F, D, MinusTrinomial>],
    transformed: &[TrinomialNtt<F, D, MinusTrinomial>],
    accumulators: &mut [TrinomialNtt<F, D, MinusTrinomial>],
) {
    for (element, source) in transformed.iter().enumerate() {
        for (accumulator, row) in accumulators.iter_mut().zip(matrix.chunks_exact(M)) {
            let entry = row.get(element).expect("benchmark matrix row is complete");
            accumulator.add_assign_pointwise_mul_packed(entry, source);
        }
    }
}

fn benchmarks(criterion: &mut Criterion) {
    benchmark_profile::<Prime128OffsetA7F7>(
        criterion,
        "p128_a7f7",
        LabiniusCoefficientPrime::P128OffsetA7F7,
        1,
    );
    benchmark_profile::<Prime64Offset23703>(
        criterion,
        "p64_23703",
        LabiniusCoefficientPrime::P64Offset23703,
        2,
    );
}

criterion_group!(commit_kernel, benchmarks);
criterion_main!(commit_kernel);
