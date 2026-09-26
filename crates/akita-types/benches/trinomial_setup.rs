use std::hint::black_box;
use std::time::Duration;

use akita_algebra::poly::multilinear_eval;
use akita_types::{RelationPolynomial, TrinomialASetupView, TrinomialResponseLayout};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_field::{Prime128OffsetA7F7, Ring};

type F = Prime128OffsetA7F7;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    degree: usize,
    padded_coefficient_len: usize,
    rows: usize,
    columns: usize,
    digit_depth: usize,
    response_offset: usize,
    response_domain_len: usize,
    setup_offset: usize,
    setup_domain_len: usize,
}

const CASES: [Case; 3] = [
    Case {
        name: "p128_d324_2x3_setup_2^11",
        degree: 324,
        padded_coefficient_len: 512,
        rows: 2,
        columns: 3,
        digit_depth: 2,
        response_offset: 19,
        response_domain_len: 1 << 12,
        setup_offset: 37,
        setup_domain_len: 1 << 11,
    },
    Case {
        name: "p128_d648_2x2_setup_2^12",
        degree: 648,
        padded_coefficient_len: 1024,
        rows: 2,
        columns: 2,
        digit_depth: 2,
        response_offset: 29,
        response_domain_len: 1 << 13,
        setup_offset: 53,
        setup_domain_len: 1 << 12,
    },
    Case {
        name: "p128_d648_8x16_setup_2^17",
        degree: 648,
        padded_coefficient_len: 1024,
        rows: 8,
        columns: 16,
        digit_depth: 2,
        response_offset: 29,
        response_domain_len: 1 << 16,
        setup_offset: 37,
        setup_domain_len: 1 << 17,
    },
];

fn point(domain_len: usize, seed: u64) -> Vec<F> {
    let variable_count = domain_len.ilog2() as usize;
    (0..variable_count)
        .map(|index| {
            let value = seed
                .wrapping_add(index as u64 * 0x9e37_79b9)
                .rotate_left((index % 61) as u32);
            F::from_u64(value)
        })
        .collect()
}

fn bench_case(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    case: Case,
) {
    assert_ne!(case.setup_offset % case.degree, 0);
    assert_ne!(case.response_offset % case.padded_coefficient_len, 0);

    let polynomial = RelationPolynomial::minus_trinomial(case.degree)
        .expect("benchmark degree must define a trinomial polynomial");
    let response = TrinomialResponseLayout::new(
        polynomial,
        case.padded_coefficient_len,
        case.columns,
        case.digit_depth,
        case.response_offset,
        case.response_domain_len,
    )
    .expect("benchmark response layout must fit its domain");
    let view = TrinomialASetupView::new(
        polynomial,
        case.rows,
        case.columns,
        case.setup_offset,
        case.setup_domain_len,
        response,
    )
    .expect("benchmark setup view must fit its domain");

    let response_point = point(case.response_domain_len, 0x1020_3040);
    let setup_point = point(case.setup_domain_len, 0x5060_7080);
    let alpha = F::from_u64(0x1234_5678_9abc_def0);
    let row_weights = (0..case.rows)
        .map(|row| F::from_u64(11 + row as u64 * 17))
        .collect::<Vec<_>>();
    let gadget = (0..case.digit_depth)
        .map(|digit| F::from_u64(5 + digit as u64 * 12))
        .collect::<Vec<_>>();
    let prepared = view
        .prepare(&response_point, alpha, &row_weights, &gadget)
        .expect("benchmark preparation must succeed");
    let dense = prepared
        .materialize_setup_weights()
        .expect("benchmark setup weights must materialize");
    let compact_value = prepared
        .evaluate_setup_weight_at(&setup_point)
        .expect("compact setup-weight evaluation must succeed");
    let dense_value = multilinear_eval(&dense, &setup_point)
        .expect("dense setup-weight MLE evaluation must succeed");
    assert_eq!(compact_value, dense_value);

    group.bench_with_input(BenchmarkId::new("prepare", case.name), &case, |bench, _| {
        bench.iter(|| {
            black_box(
                black_box(view)
                    .prepare(
                        black_box(&response_point),
                        black_box(alpha),
                        black_box(&row_weights),
                        black_box(&gadget),
                    )
                    .expect("benchmark preparation must succeed"),
            )
        })
    });
    group.bench_with_input(
        BenchmarkId::new("materialize_dense", case.name),
        &case,
        |bench, _| {
            bench.iter(|| {
                black_box(
                    black_box(&prepared)
                        .materialize_setup_weights()
                        .expect("benchmark setup weights must materialize"),
                )
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("evaluate_compact", case.name),
        &case,
        |bench, _| {
            bench.iter(|| {
                black_box(
                    black_box(&prepared)
                        .evaluate_setup_weight_at(black_box(&setup_point))
                        .expect("compact setup-weight evaluation must succeed"),
                )
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("evaluate_dense_mle", case.name),
        &case,
        |bench, _| {
            bench.iter(|| {
                black_box(
                    multilinear_eval(black_box(&dense), black_box(&setup_point))
                        .expect("dense setup-weight MLE evaluation must succeed"),
                )
            })
        },
    );
}

fn trinomial_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("trinomial_setup_weights");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for case in CASES {
        bench_case(&mut group, case);
    }
    group.finish();
}

criterion_group!(benches, trinomial_setup);
criterion_main!(benches);
