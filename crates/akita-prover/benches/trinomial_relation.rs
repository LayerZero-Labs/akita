//! Coefficient quotient construction, including both signed product sums.
//! Plan/common-operand preparation is measured separately; inputs and modular images
//! are prepared outside the timed loop. This measures a polynomial lift, not
//! multiplication of already-reduced ring elements.

use std::{fmt::Debug, hint::black_box, time::Duration};

use akita_algebra::{Field, MinusTrinomial, Prime64Offset23703, SmoothFftField, TrinomialRing};
use akita_prover::kernels::trinomial_relation::TrinomialRelationQuotientBuilder;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::Prime128OffsetA7F7;

fn input<F: Field, const D: usize>(seed: u64) -> TrinomialRing<F, D, MinusTrinomial> {
    TrinomialRing::from_coefficients(std::array::from_fn(|index| {
        F::from_u64(
            seed.wrapping_add(index as u64 * 0x9e37_79b9)
                .rotate_left((index % 61) as u32),
        )
    }))
    .unwrap()
}

fn profile<F: SmoothFftField + Debug, const D: usize>(c: &mut Criterion, name: &str) {
    let a = [input::<F, D>(17), input(31), input(47), input(71)];
    let response = [input::<F, D>(91), input(101), input(131), input(151)];
    let one = TrinomialRing::from_coefficients(std::array::from_fn(|i| {
        if i == 0 {
            F::one()
        } else {
            F::zero()
        }
    }))
    .unwrap();
    let challenges = [input::<F, D>(171), input(191), input(211), one];
    let mut images = [
        input::<F, D>(311),
        input(331),
        input(351),
        a[3].schoolbook_mul(&response[3]).unwrap(),
    ];
    for i in 0..3 {
        let residual = a[i].schoolbook_mul(&response[i]).unwrap()
            - images[i].schoolbook_mul(&challenges[i]).unwrap();
        images[3] += residual;
    }
    let mut plan =
        TrinomialRelationQuotientBuilder::<F, D, MinusTrinomial>::new(&response, &challenges)
            .unwrap();
    let expected = plan.build(&a, &images).unwrap();
    assert_eq!(expected.len(), D - 1);
    let parameter = format!("{name}/D={D}");
    let mut group = c.benchmark_group("trinomial_relation");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function(
        BenchmarkId::new("prepare_plan_and_common_operands", &parameter),
        |b| {
            b.iter(|| {
                black_box(
                    TrinomialRelationQuotientBuilder::<F, D, MinusTrinomial>::new(
                        black_box(&response),
                        black_box(&challenges),
                    )
                    .unwrap(),
                )
            })
        },
    );
    group.throughput(Throughput::Elements((8 * D) as u64));
    group.bench_function(
        BenchmarkId::new("quotient_4_plus_4_products", &parameter),
        |b| b.iter(|| black_box(plan.build(black_box(&a), black_box(&images)).unwrap())),
    );
    let rows: [[TrinomialRing<F, D, MinusTrinomial>; 4]; 8] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            TrinomialRing::from_coefficients(std::array::from_fn(|b| {
                a[j].coefficients()[b] * F::from_u64(i as u64 + 1)
            }))
            .unwrap()
        })
    });
    let row_images: [[TrinomialRing<F, D, MinusTrinomial>; 4]; 8] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            TrinomialRing::from_coefficients(std::array::from_fn(|b| {
                images[j].coefficients()[b] * F::from_u64(i as u64 + 1)
            }))
            .unwrap()
        })
    });
    group.throughput(Throughput::Elements((64 * D) as u64));
    group.bench_function(
        BenchmarkId::new("quotient_8_rows_shared_operands", &parameter),
        |b| {
            b.iter(|| {
                for (row, images) in black_box(&rows).iter().zip(black_box(&row_images)) {
                    black_box(plan.build(row, images).unwrap());
                }
            })
        },
    );
    group.finish();
}

fn benchmarks(c: &mut Criterion) {
    profile::<Prime64Offset23703, 324>(c, "p64");
    profile::<Prime64Offset23703, 648>(c, "p64");
    profile::<Prime128OffsetA7F7, 324>(c, "p128");
    profile::<Prime128OffsetA7F7, 648>(c, "p128");
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
