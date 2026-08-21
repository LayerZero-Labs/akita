use std::time::{Duration, Instant};

use akita_field::{fp128_asm_experiment as asm, Prime128OffsetA7F7};
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput,
};

type F = Prime128OffsetA7F7;
const CHAIN_ITERS: usize = 2048;
const STREAM_ITERS: usize = 256;
const STREAMS: usize = 8;

fn per_op(elapsed: Duration, count: usize) -> Duration {
    Duration::from_secs_f64(elapsed.as_secs_f64() / count as f64)
}

fn bench_chain(
    group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    init: F,
    rhs: F,
    step: impl Fn(F, F) -> F,
) {
    group.throughput(Throughput::Elements(1));
    group.bench_function(name, |b| {
        b.iter_custom(|outer_iters| {
            let rhs = black_box(rhs);
            let mut acc = black_box(init);
            let start = Instant::now();
            for _ in 0..outer_iters {
                for _ in 0..CHAIN_ITERS {
                    acc = step(acc, rhs);
                }
            }
            black_box(acc);
            per_op(start.elapsed(), CHAIN_ITERS)
        });
    });
}

fn bench_stream(
    group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    init: [F; STREAMS],
    rhs: [F; STREAMS],
    step: impl Fn(F, F) -> F,
) {
    group.throughput(Throughput::Elements(1));
    group.bench_function(name, |b| {
        b.iter_custom(|outer_iters| {
            let rhs = black_box(rhs);
            let mut acc = black_box(init);
            let start = Instant::now();
            for _ in 0..outer_iters {
                for _ in 0..STREAM_ITERS {
                    for lane in 0..STREAMS {
                        acc[lane] = step(acc[lane], rhs[lane]);
                    }
                }
            }
            black_box(acc);
            per_op(start.elapsed(), STREAMS * STREAM_ITERS)
        });
    });
}

fn bench_operation(
    c: &mut Criterion,
    operation: &str,
    inline: impl Fn(F, F) -> F + Copy,
    fixed: impl Fn(F, F) -> F + Copy,
    linked: impl Fn(F, F) -> F + Copy,
) {
    let init = F::from_canonical_u128(0x8421_9c76_31d5_aaaa_0246_8ace_1357_9bdf);
    let rhs = F::from_canonical_u128(0x1234_5678_9abc_def0_fedc_ba98_7654_3210);
    let stream_init = std::array::from_fn(|i| {
        F::from_canonical_u128(init.to_canonical_u128().wrapping_add(i as u128 + 1))
    });
    let stream_rhs = std::array::from_fn(|i| {
        F::from_canonical_u128(rhs.to_canonical_u128().wrapping_add(17 * i as u128))
    });

    let mut latency = c.benchmark_group(format!("fp128_asm_linkage/{operation}/latency"));
    bench_chain(&mut latency, "current_inline", init, rhs, inline);
    bench_chain(&mut latency, "fixed_register_inline", init, rhs, fixed);
    bench_chain(&mut latency, "standalone_call", init, rhs, linked);
    latency.finish();

    let mut throughput = c.benchmark_group(format!("fp128_asm_linkage/{operation}/throughput"));
    bench_stream(
        &mut throughput,
        "current_inline",
        stream_init,
        stream_rhs,
        inline,
    );
    bench_stream(
        &mut throughput,
        "fixed_register_inline",
        stream_init,
        stream_rhs,
        fixed,
    );
    bench_stream(
        &mut throughput,
        "standalone_call",
        stream_init,
        stream_rhs,
        linked,
    );
    throughput.finish();
}

fn bench_fp128_asm_linkage(c: &mut Criterion) {
    bench_operation(c, "add", |a, b| a + b, asm::add_inline, asm::add_linked);
    bench_operation(c, "sub", |a, b| a - b, asm::sub_inline, asm::sub_linked);
    bench_operation(c, "mul", |a, b| a * b, asm::mul_inline, asm::mul_linked);
}

criterion_group!(benches, bench_fp128_asm_linkage);
criterion_main!(benches);
