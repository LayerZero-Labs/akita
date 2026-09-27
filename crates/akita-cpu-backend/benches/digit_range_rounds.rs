#![allow(missing_docs)]

use akita_algebra::eq_poly::EqPolynomial;
use akita_config::proof_optimized::{fp128, fp32, fp64};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use jolt_field::{CanonicalEncoding, Field, Fold, Ring, Unreduced};
use std::hint::black_box;
use std::time::Duration;

// The isolated kernel benchmark models the current tree-stage quartic bound.
const MAX_TREE_STAGE_Q_DEGREE: usize = 4;

// Source-included unit-test imports are unused in the harness-free bench.
#[allow(dead_code, unused_imports)]
#[path = "../src/opaque/sumcheck/digit_range/exact_prefix.rs"]
mod exact_prefix;

// Source-included unit-test imports are unused in the harness-free bench.
#[allow(dead_code, unused_imports)]
#[path = "../src/opaque/sumcheck/digit_range/round_accumulation.rs"]
mod round_accumulation;

fn quartic_affine_coefficients<E: Field + Ring>(a: E, d: E) -> [E; 5] {
    let a2 = a.square();
    let d2 = d.square();
    let a3 = a2 * a;
    let d3 = d2 * d;
    [
        a2.square(),
        E::from_u64(4) * a3 * d,
        E::from_u64(6) * a2 * d2,
        E::from_u64(4) * a * d3,
        d2.square(),
    ]
}

fn benchmark_round_accumulation<E: Field + Ring + Unreduced + Send + Sync + 'static>(
    c: &mut Criterion,
    field_name: &str,
    value: fn(usize) -> E,
) {
    for explicit_pair_count in [1usize << 10, 1usize << 16] {
        // Keep half the split-equality domain implicit to exercise the padded
        // suffix mass used by recursive rounds.
        let domain_len = 2 * explicit_pair_count;
        let total_bits = domain_len.trailing_zeros() as usize;
        let first_bits = total_bits.min(8);
        let point: Vec<E> = (0..total_bits).map(|index| value(index + 101)).collect();
        let first = EqPolynomial::evals(&point[..first_bits]).expect("first equality table");
        let second = EqPolynomial::evals(&point[first_bits..]).expect("second equality table");
        let affine_pairs: Vec<(E, E)> = (0..explicit_pair_count)
            .map(|index| (value(2 * index + 1), value(2 * index + 2)))
            .collect();
        let default_coefficients = quartic_affine_coefficients(
            value(2 * explicit_pair_count + 1),
            value(2 * explicit_pair_count + 2),
        );

        let mut group = c.benchmark_group(format!(
            "digit_range_rounds/round_accumulation/{field_name}"
        ));
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(100));
        group.measurement_time(Duration::from_millis(500));
        group.throughput(Throughput::Elements(explicit_pair_count as u64));
        group.bench_function(
            BenchmarkId::from_parameter(format!("explicit_pairs_{explicit_pair_count}")),
            |b| {
                b.iter(|| {
                    black_box(round_accumulation::accumulate_equality_weighted_round(
                        black_box(&first),
                        black_box(&second),
                        explicit_pair_count,
                        |pair_index| {
                            let (a, d) = affine_pairs[pair_index];
                            quartic_affine_coefficients(a, d)
                        },
                        black_box(default_coefficients),
                    ))
                })
            },
        );
        group.finish();
    }
}

fn fp128_value(index: usize) -> fp128::Field {
    fp128::Field::from_u128_reduced(
        (index as u128)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15_d1b5_4a32_d192_ed03)
            .wrapping_add(0x94d0_49bb_1331_11eb_2545_f491_4f6c_dd1d),
    )
}

fn benchmark_materialized_rounds<E: Field + Ring + Fold + Unreduced>(
    c: &mut Criterion,
    field_name: &str,
    value: fn(usize) -> E,
) {
    for rows in [4097usize, 65537] {
        let domain = rows.next_power_of_two();
        let initial: Vec<[E; 4]> = (0..rows)
            .map(|row| std::array::from_fn(|lane| value(4 * row + lane + 1)))
            .collect();
        let padding = std::array::from_fn(|lane| value(lane + 17));
        let weights: Vec<_> = (2..domain.trailing_zeros())
            .rev()
            .map(|bits| {
                let point: Vec<_> = (0..bits - 1).map(|i| value(i as usize + 91)).collect();
                let split = point.len().min(8);
                (
                    EqPolynomial::evals(&point[..split]).unwrap(),
                    EqPolynomial::evals(&point[split..]).unwrap(),
                )
            })
            .collect();
        let mut group = c.benchmark_group(format!("digit_range_rounds/materialized/{field_name}"));
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(100));
        group.measurement_time(Duration::from_millis(500));
        group.bench_function(BenchmarkId::from_parameter(rows), |b| {
            b.iter_batched(
                || exact_prefix::ExactPrefixTable::new(domain, initial.clone(), padding).unwrap(),
                |mut table| {
                    let mut scratch = Vec::new();
                    for (round, (first, second)) in weights.iter().enumerate() {
                        let context = E::precompute(value(round + 123));
                        black_box(
                            round_accumulation::fold_and_accumulate_equality_weighted_round(
                                &mut table,
                                &mut scratch,
                                first,
                                second,
                                |left, right| {
                                    std::array::from_fn(|lane| {
                                        E::fold_one(&context, left[lane], right[lane])
                                    })
                                },
                                |left, right| {
                                    quartic_affine_coefficients(left[0], right[0] - left[0])
                                },
                            ),
                        );
                    }
                    black_box(table);
                },
                BatchSize::LargeInput,
            )
        });
        group.finish();
    }
}

fn fp64_ext2_value(index: usize) -> fp64::ExtensionField {
    type F = fp64::Field;
    fp64::ExtensionField::new(
        F::from_u64((index as u64).wrapping_mul(31) + 7),
        F::from_u64((index as u64).wrapping_mul(47) + 19),
    )
}

fn fp32_ext4_value(index: usize) -> fp32::ExtensionField {
    type F = fp32::Field;
    fp32::ExtensionField::new(std::array::from_fn(|coordinate| {
        F::from_u64(
            (index as u64)
                .wrapping_mul(23 + coordinate as u64 * 6)
                .wrapping_add(5 + coordinate as u64),
        )
    }))
}

fn bench_digit_range_rounds(c: &mut Criterion) {
    benchmark_round_accumulation::<fp128::Field>(c, "fp128", fp128_value);
    benchmark_round_accumulation::<fp64::ExtensionField>(c, "fp64_ext2", fp64_ext2_value);
    benchmark_round_accumulation::<fp32::ExtensionField>(c, "fp32_ext4", fp32_ext4_value);
    benchmark_materialized_rounds::<fp128::Field>(c, "fp128", fp128_value);
    benchmark_materialized_rounds::<fp64::ExtensionField>(c, "fp64_ext2", fp64_ext2_value);
    benchmark_materialized_rounds::<fp32::ExtensionField>(c, "fp32_ext4", fp32_ext4_value);
}

criterion_group!(digit_range_rounds, bench_digit_range_rounds);
criterion_main!(digit_range_rounds);
