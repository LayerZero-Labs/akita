#![allow(missing_docs)]

use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallenge, SparseChallengeConfig};
use akita_cpu_backend::benchmark_support::balanced_ring_decompose_fold_partitioned;
use akita_cpu_backend::standalone::{
    decompose_recursive_witness, recursive_witness_from_i8_digits,
};
use akita_types::sis::compute_num_digits_field_width;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use jolt_field::{CanonicalEncoding, Field, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};

const FIELD_COEFFICIENTS: usize = 1 << 22;
const POSITIONS_PER_BLOCK: usize = 512;

fn challenge<const D: usize>(block: usize) -> SparseChallenge {
    let config = SparseChallengeConfig::production_for_ring_dim(D).expect("production challenge");
    let weight = config.weight();
    let positions = (0..weight)
        .map(|term| ((term * 37 + block * 13) % D) as u32)
        .collect();
    let coeffs = (0..weight)
        .map(|term| {
            let magnitude = if term < config.count_pm1 { 1 } else { 2 };
            if (term + block).is_multiple_of(2) {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect();
    SparseChallenge { positions, coeffs }
}

fn dense_rings<F: Field + CanonicalEncoding, const D: usize>() -> Vec<CyclotomicRing<F, D>> {
    let num_rings = FIELD_COEFFICIENTS / D;
    (0..num_rings)
        .map(|ring| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                let index = (ring * D + coefficient) as u128;
                let mixed = index
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15_6a09_e667_f3bc_c909)
                    .rotate_left(37)
                    ^ index.wrapping_mul(0xbf58_476d_1ce4_e5b9_94d0_49bb_1331_11eb);
                F::from_u128_reduced(mixed)
            }))
        })
        .collect()
}

fn dense_case<F: Field + CanonicalEncoding, const D: usize>(
    c: &mut Criterion,
    field_label: &str,
    field_bits: u32,
    log_basis: u32,
) {
    let rings = dense_rings::<F, D>();
    let blocks = rings.len().div_ceil(POSITIONS_PER_BLOCK);
    let challenges = (0..blocks).map(challenge::<D>).collect::<Vec<_>>();
    let num_digits = compute_num_digits_field_width(field_bits, log_basis);
    let params = BalancedDecomposePow2Params::new(num_digits, log_basis);

    let mut group = c.benchmark_group(format!("decompose_fold/dense_{field_label}"));
    group.throughput(Throughput::Elements(FIELD_COEFFICIENTS as u64));
    group.bench_function(format!("d{D}_b{log_basis}_digits{num_digits}"), |b| {
        b.iter(|| {
            black_box(balanced_ring_decompose_fold_partitioned(
                black_box(&rings),
                black_box(&challenges),
                POSITIONS_PER_BLOCK,
                &params,
            ))
        });
    });
    group.finish();
}

fn suffix_case<const D: usize>(c: &mut Criterion) {
    let digits = (0..FIELD_COEFFICIENTS / D)
        .flat_map(|ring| {
            (0..D)
                .map(move |coefficient| (((ring * D + coefficient) * 11 + ring * 3) % 7) as i8 - 3)
        })
        .collect::<Vec<_>>();
    let witness = recursive_witness_from_i8_digits(digits);
    let blocks = (FIELD_COEFFICIENTS / D).div_ceil(POSITIONS_PER_BLOCK);
    let challenges = (0..blocks).map(challenge::<D>).collect::<Vec<_>>();

    let mut group = c.benchmark_group("decompose_fold/tight_suffix");
    group.throughput(Throughput::Elements(FIELD_COEFFICIENTS as u64));
    group.bench_function(format!("d{D}"), |b| {
        b.iter(|| {
            black_box(decompose_recursive_witness::<Prime128OffsetA7F7, D>(
                black_box(&witness),
                black_box(&challenges),
                POSITIONS_PER_BLOCK,
                1,
                3,
            ))
        });
    });
    group.finish();
}

fn bench_decompose_fold(c: &mut Criterion) {
    dense_case::<Prime32Offset99, 64>(c, "fp32", 32, 8);
    dense_case::<Prime32Offset99, 128>(c, "fp32", 32, 8);
    dense_case::<Prime32Offset99, 256>(c, "fp32", 32, 8);
    dense_case::<Prime32Offset99, 512>(c, "fp32", 32, 8);
    dense_case::<Prime32Offset99, 1024>(c, "fp32", 32, 8);
    dense_case::<Prime32Offset99, 2048>(c, "fp32", 32, 8);

    dense_case::<Prime64Offset59, 64>(c, "fp64", 64, 6);
    dense_case::<Prime64Offset59, 128>(c, "fp64", 64, 6);
    dense_case::<Prime64Offset59, 256>(c, "fp64", 64, 6);
    dense_case::<Prime64Offset59, 512>(c, "fp64", 64, 6);
    dense_case::<Prime64Offset59, 1024>(c, "fp64", 64, 6);
    dense_case::<Prime64Offset59, 2048>(c, "fp64", 64, 6);
    dense_case::<Prime64Offset59, 128>(c, "fp64", 64, 10);
    dense_case::<Prime64Offset59, 512>(c, "fp64", 64, 11);

    dense_case::<Prime128OffsetA7F7, 64>(c, "fp128", 128, 9);
    dense_case::<Prime128OffsetA7F7, 128>(c, "fp128", 128, 9);
    dense_case::<Prime128OffsetA7F7, 256>(c, "fp128", 128, 9);
    dense_case::<Prime128OffsetA7F7, 512>(c, "fp128", 128, 9);
    dense_case::<Prime128OffsetA7F7, 1024>(c, "fp128", 128, 9);
    dense_case::<Prime128OffsetA7F7, 2048>(c, "fp128", 128, 9);
    dense_case::<Prime128OffsetA7F7, 64>(c, "fp128", 128, 11);

    suffix_case::<64>(c);
    suffix_case::<128>(c);
}

criterion_group!(decompose_fold, bench_decompose_fold);
criterion_main!(decompose_fold);
