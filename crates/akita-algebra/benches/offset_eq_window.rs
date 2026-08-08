#![allow(missing_docs)]

use akita_algebra::offset_eq::{
    materialize_eq_tensor_left, EqPairTensorAxis, EqPairTensorFamily, OffsetEqWindow,
};
use akita_field::Prime128OffsetA7F7;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

type F = Prime128OffsetA7F7;

fn bench_fill_interval(c: &mut Criterion) {
    let challenges = (0..24)
        .map(|index| F::from_u64(index as u64 + 2))
        .collect::<Vec<_>>();
    let equality = OffsetEqWindow::new(&challenges).expect("bounded equality window");
    let mut group = c.benchmark_group("offset_eq_window_fill_interval");

    for len in [1usize << 10, 1 << 16, 1_409_024] {
        let mut output = vec![F::zero(); len];
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, _| {
            b.iter(|| {
                equality
                    .fill_interval(black_box(37), black_box(&mut output))
                    .expect("valid equality interval");
                black_box(&output);
            });
        });
    }
    group.finish();
}

fn bench_materialize_disjoint_intervals(c: &mut Criterion) {
    const OUTPUT_LEN: usize = 1_409_024;
    const INTERVALS: usize = 64;
    let challenges = (0..24)
        .map(|index| F::from_u64(index as u64 + 2))
        .collect::<Vec<_>>();
    let equality = OffsetEqWindow::new(&challenges).expect("bounded equality window");
    let interval_len = OUTPUT_LEN / INTERVALS;
    let families = (0..INTERVALS)
        .map(|index| {
            let offset = index * interval_len;
            EqPairTensorFamily::new(
                offset,
                offset + 37,
                F::one(),
                vec![EqPairTensorAxis::unit(interval_len, 1, 1)],
            )
            .expect("valid interval family")
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("offset_eq_materialize_disjoint_intervals");
    group.throughput(Throughput::Elements(OUTPUT_LEN as u64));
    group.bench_function(BenchmarkId::from_parameter(OUTPUT_LEN), |b| {
        b.iter(|| {
            black_box(
                materialize_eq_tensor_left(
                    black_box(&equality),
                    black_box(&families),
                    black_box(OUTPUT_LEN),
                )
                .expect("valid disjoint intervals"),
            );
        });
    });
    group.finish();
}

criterion_group!(
    offset_eq_window,
    bench_fill_interval,
    bench_materialize_disjoint_intervals
);
criterion_main!(offset_eq_window);
