#![allow(missing_docs)]

use akita_algebra::offset_eq::OffsetEqWindow;
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

criterion_group!(offset_eq_window, bench_fill_interval);
criterion_main!(offset_eq_window);
