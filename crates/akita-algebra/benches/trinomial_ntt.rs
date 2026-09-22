use std::fmt::Debug;
use std::hint::black_box;
use std::time::Duration;

use akita_algebra::{
    Field, MinusTrinomial, PlusTrinomial, Prime64Offset23703, SmoothFftField, TrinomialModulus,
    TrinomialNttDomain, TrinomialRing,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::Prime128OffsetA7F7;

fn input<F: Field, const D: usize, M: TrinomialModulus>(seed: u64) -> TrinomialRing<F, D, M> {
    TrinomialRing::from_coefficients(std::array::from_fn(|index| {
        F::from_u64(
            seed.wrapping_add(index as u64 * 0x9e37_79b9)
                .rotate_left((index % 61) as u32),
        )
    }))
    .expect("benchmark degree is valid")
}

fn bench_profile<F, const D: usize, M>(criterion: &mut Criterion, field: &str, modulus: &str)
where
    F: SmoothFftField + Debug,
    M: TrinomialModulus,
{
    let domain = TrinomialNttDomain::<F, D, M>::new().expect("benchmark shape must fully split");
    let lhs = input::<F, D, M>(17);
    let rhs = input::<F, D, M>(93);
    let transformed = domain.forward(&lhs);
    let parameter = format!("{field}/{modulus}/D={D}");
    let mut group = criterion.benchmark_group("trinomial_ntt");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(D as u64));

    let mut forward_workspace = domain.workspace();
    group.bench_with_input(
        BenchmarkId::new("forward", &parameter),
        &parameter,
        |b, _| {
            b.iter(|| {
                black_box(
                    domain
                        .forward_with_workspace(black_box(&lhs), black_box(&mut forward_workspace)),
                )
            })
        },
    );
    let mut inverse_workspace = domain.workspace();
    group.bench_with_input(
        BenchmarkId::new("inverse", &parameter),
        &parameter,
        |b, _| {
            b.iter(|| {
                black_box(domain.inverse_with_workspace(
                    black_box(&transformed),
                    black_box(&mut inverse_workspace),
                ))
            })
        },
    );
    let mut multiply_workspace = domain.workspace();
    group.bench_with_input(
        BenchmarkId::new("multiply", &parameter),
        &parameter,
        |b, _| {
            b.iter(|| {
                black_box(domain.multiply_with_workspace(
                    black_box(&lhs),
                    black_box(&rhs),
                    black_box(&mut multiply_workspace),
                ))
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("schoolbook_oracle", &parameter),
        &parameter,
        |b, _| b.iter(|| black_box(lhs.schoolbook_mul(black_box(&rhs)).unwrap())),
    );
    group.finish();
}

fn benches(criterion: &mut Criterion) {
    bench_profile::<Prime64Offset23703, 162, PlusTrinomial>(criterion, "p64_23703", "plus");
    bench_profile::<Prime64Offset23703, 324, MinusTrinomial>(criterion, "p64_23703", "minus");
    bench_profile::<Prime64Offset23703, 648, MinusTrinomial>(criterion, "p64_23703", "minus");

    bench_profile::<Prime128OffsetA7F7, 162, PlusTrinomial>(criterion, "p128_a7f7", "plus");
    bench_profile::<Prime128OffsetA7F7, 324, MinusTrinomial>(criterion, "p128_a7f7", "minus");
    bench_profile::<Prime128OffsetA7F7, 648, MinusTrinomial>(criterion, "p128_a7f7", "minus");
    bench_profile::<Prime128OffsetA7F7, 486, PlusTrinomial>(criterion, "p128_a7f7", "plus");
}

criterion_group!(trinomial_ntt, benches);
criterion_main!(trinomial_ntt);
