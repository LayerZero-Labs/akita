#![allow(missing_docs)]

use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallenge, SparseChallengeConfig};
use akita_field::{CanonicalField, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use akita_prover::backend::poly_helpers::{
    balanced_ring_decompose_fold_partitioned, balanced_tight_digit_fold_partitioned,
    DecomposeParams,
};
use akita_types::sis::compute_num_digits_field_width;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

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

fn dense_rings<F: CanonicalField, const D: usize>() -> Vec<CyclotomicRing<F, D>> {
    let num_rings = FIELD_COEFFICIENTS / D;
    (0..num_rings)
        .map(|ring| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                let index = (ring * D + coefficient) as u128;
                let mixed = index
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15_6a09_e667_f3bc_c909)
                    .rotate_left(37)
                    ^ index.wrapping_mul(0xbf58_476d_1ce4_e5b9_94d0_49bb_1331_11eb);
                F::from_canonical_u128_reduced(mixed)
            }))
        })
        .collect()
}

fn dense_case<F: CanonicalField, const D: usize>(
    c: &mut Criterion,
    field_label: &str,
    field_bits: u32,
    log_basis: u32,
) {
    let rings = dense_rings::<F, D>();
    let blocks = rings.len().div_ceil(POSITIONS_PER_BLOCK);
    let challenges = (0..blocks).map(challenge::<D>).collect::<Vec<_>>();
    let num_digits = compute_num_digits_field_width(field_bits, log_basis);
    let q = (-F::one()).to_canonical_u128() + 1;
    let threshold = decompose_centering_threshold(num_digits, log_basis, q);
    let params = DecomposeParams {
        threshold,
        q,
        mask: (1i128 << log_basis) - 1,
        half_b: 1i128 << (log_basis - 1),
        b_val: 1i128 << log_basis,
        log_basis,
        overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
    };

    let mut group = c.benchmark_group(format!("decompose_fold/dense_{field_label}"));
    group.throughput(Throughput::Elements(FIELD_COEFFICIENTS as u64));
    group.bench_function(format!("d{D}_b{log_basis}_digits{num_digits}"), |b| {
        b.iter(|| {
            black_box(balanced_ring_decompose_fold_partitioned(
                black_box(&rings),
                black_box(&challenges),
                POSITIONS_PER_BLOCK,
                num_digits,
                &params,
            ))
        });
    });
    group.finish();
}

fn suffix_case<const D: usize>(c: &mut Criterion) {
    let rings: Vec<[i8; D]> = (0..FIELD_COEFFICIENTS / D)
        .map(|ring| {
            std::array::from_fn(|coefficient| {
                (((ring * D + coefficient) * 11 + ring * 3) % 7) as i8 - 3
            })
        })
        .collect::<Vec<_>>();
    let blocks = rings.len().div_ceil(POSITIONS_PER_BLOCK);
    let challenges = (0..blocks).map(challenge::<D>).collect::<Vec<_>>();

    let mut group = c.benchmark_group("decompose_fold/tight_suffix");
    group.throughput(Throughput::Elements(FIELD_COEFFICIENTS as u64));
    group.bench_function(format!("d{D}"), |b| {
        b.iter(|| {
            black_box(balanced_tight_digit_fold_partitioned(
                black_box(&rings),
                black_box(&challenges),
                POSITIONS_PER_BLOCK,
                Some(3),
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
