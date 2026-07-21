#![allow(missing_docs)]

use std::time::Duration;

use akita_algebra::CyclotomicRing;
use akita_field::{CanonicalField, Prime128OffsetA7F7};
use akita_prover::kernels::linear::mat_vec_mul_ntt_digits_i8;
use akita_types::{prepare_ntt_cache, FlatMatrix, NttCacheMode, PreparedNttCache};
use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};

type F = Prime128OffsetA7F7;

const FIXED_WIDTH: usize = 128;
const RANKS: [usize; 4] = [1, 2, 4, 8];
const WIDTHS: [usize; 4] = [128, 256, 512, 1024];
const COMMON_LOG_BASES: [u32; 7] = [2, 3, 4, 5, 6, 7, 8];

fn sample_matrix<const D: usize>(rank: usize, width: usize) -> Vec<CyclotomicRing<F, D>> {
    (0..rank * width)
        .map(|entry| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                let low = (entry as u64)
                    .wrapping_mul(0x9E37_79B1_85EB_CA87)
                    .wrapping_add((coefficient as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
                let high = low.rotate_left(29) ^ 0xD6E8_FEB8_6659_FD93;
                F::from_canonical_u128_reduced(u128::from(low) | (u128::from(high) << 64))
            }))
        })
        .collect()
}

fn sample_i8_digits<const D: usize>(width: usize, log_basis: u32) -> Vec<[i8; D]> {
    debug_assert!(log_basis <= 8);
    sample_i16_digits(width, log_basis)
        .into_iter()
        .map(|ring| ring.map(|digit| digit as i8))
        .collect()
}

fn sample_i16_digits<const D: usize>(width: usize, log_basis: u32) -> Vec<[i16; D]> {
    let bound = 1i32 << (log_basis - 1);
    let span = 2 * bound;
    (0..width)
        .map(|column| {
            std::array::from_fn(|coefficient| {
                let value = column
                    .wrapping_mul(257)
                    .wrapping_add(coefficient.wrapping_mul(137))
                    .wrapping_add(bound as usize - 1)
                    % span as usize;
                (value as i32 - bound) as i16
            })
        })
        .collect()
}

fn prepare<const D: usize>(
    matrix: &[CyclotomicRing<F, D>],
    mode: NttCacheMode,
) -> PreparedNttCache<D> {
    let flat = FlatMatrix::from_ring_slice(matrix);
    let view = flat
        .ring_view::<D>(1, matrix.len())
        .expect("benchmark matrix view");
    prepare_ntt_cache(view, mode).expect("benchmark NTT cache")
}

fn i8_matvec<const D: usize>(
    cache: &PreparedNttCache<D>,
    rank: usize,
    width: usize,
    digits: &[[i8; D]],
    log_basis: u32,
) -> Vec<CyclotomicRing<F, D>> {
    let blocks = [digits];
    mat_vec_mul_ntt_digits_i8(cache, rank, width, &blocks, log_basis)
        .expect("i8 benchmark matvec")
        .into_iter()
        .next()
        .expect("one benchmark block")
}

fn bench_shape<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    rank: usize,
    width: usize,
) {
    let shape = format!("d{D}_r{rank}_w{width}");
    let matrix = sample_matrix::<D>(rank, width);
    let i8_cache = prepare(&matrix, NttCacheMode::BothTransforms);
    let i8_digits = sample_i8_digits::<D>(width, 8);
    let i8_reference = i8_matvec(&i8_cache, rank, width, &i8_digits, 8);
    group.throughput(Throughput::Elements((rank * width * D) as u64));
    group.bench_function(BenchmarkId::new("i8_l8_prover", &shape), |bench| {
        bench.iter(|| black_box(i8_matvec(&i8_cache, rank, width, black_box(&i8_digits), 8)))
    });

    for log_basis in [8, 10, 11] {
        let cache = prepare(&matrix, NttCacheMode::ExactNegacyclic { width, log_basis });
        let digits = if log_basis == 8 {
            i8_digits
                .iter()
                .map(|ring| ring.map(i16::from))
                .collect::<Vec<_>>()
        } else {
            sample_i16_digits::<D>(width, log_basis)
        };
        let layout = if cache.has_i16_tail() { "tail" } else { "base" };
        let variant = format!("i16_l{log_basis}_{layout}");
        if log_basis == 8 {
            assert_eq!(
                cache
                    .mat_vec_i16::<F>(log_basis, rank, &digits)
                    .expect("i16 L8 reference"),
                i8_reference,
                "i8 and i16 L8 paths disagree for {shape}"
            );
        }
        group.throughput(Throughput::Elements((rank * width * D) as u64));
        group.bench_function(BenchmarkId::new(variant, &shape), |bench| {
            bench.iter(|| {
                black_box(
                    cache
                        .mat_vec_i16::<F>(log_basis, rank, black_box(&digits))
                        .expect("i16 benchmark matvec"),
                )
            })
        });
    }
}

fn bench_equal_output_shape<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    rank: usize,
    width: usize,
) {
    let shape = format!("d{D}_r{rank}_w{width}");
    let matrix = sample_matrix::<D>(rank, width);
    let i8_cache = prepare(&matrix, NttCacheMode::BothTransforms);
    group.throughput(Throughput::Elements((rank * width * D) as u64));

    for log_basis in COMMON_LOG_BASES {
        let i8_digits = sample_i8_digits::<D>(width, log_basis);
        let i16_digits = i8_digits
            .iter()
            .map(|ring| ring.map(i16::from))
            .collect::<Vec<_>>();
        let i8_reference = i8_matvec(&i8_cache, rank, width, &i8_digits, log_basis);
        let cache = prepare(&matrix, NttCacheMode::ExactNegacyclic { width, log_basis });
        let layout = if cache.has_i16_tail() { "tail" } else { "base" };
        assert_eq!(
            cache
                .mat_vec_i16::<F>(log_basis, rank, &i16_digits)
                .expect("i16 common-basis reference"),
            i8_reference,
            "i8 and i16 L{log_basis} paths disagree for {shape}"
        );

        group.bench_function(
            BenchmarkId::new(format!("i8_l{log_basis}_prover"), &shape),
            |bench| {
                bench.iter(|| {
                    black_box(i8_matvec(
                        &i8_cache,
                        rank,
                        width,
                        black_box(&i8_digits),
                        log_basis,
                    ))
                })
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("i16_l{log_basis}_{layout}"), &shape),
            |bench| {
                bench.iter(|| {
                    black_box(
                        cache
                            .mat_vec_i16::<F>(log_basis, rank, black_box(&i16_digits))
                            .expect("i16 common-basis benchmark matvec"),
                    )
                })
            },
        );
    }

    for log_basis in [10, 11] {
        let digits = sample_i16_digits::<D>(width, log_basis);
        let cache = prepare(&matrix, NttCacheMode::ExactNegacyclic { width, log_basis });
        let layout = if cache.has_i16_tail() { "tail" } else { "base" };
        group.bench_function(
            BenchmarkId::new(format!("i16_l{log_basis}_{layout}"), &shape),
            |bench| {
                bench.iter(|| {
                    black_box(
                        cache
                            .mat_vec_i16::<F>(log_basis, rank, black_box(&digits))
                            .expect("i16 wide-basis benchmark matvec"),
                    )
                })
            },
        );
    }
}

fn bench_rank_ring_dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_matvec_q128/rank_ring_dim/w128");
    for rank in RANKS {
        bench_shape::<64>(&mut group, rank, FIXED_WIDTH);
        bench_shape::<128>(&mut group, rank, FIXED_WIDTH);
        bench_shape::<256>(&mut group, rank, FIXED_WIDTH);
        bench_shape::<512>(&mut group, rank, FIXED_WIDTH);
    }
    group.finish();
}

fn bench_width(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_matvec_q128/width/d64_r4");
    for width in WIDTHS {
        bench_shape::<64>(&mut group, 4, width);
    }
    group.finish();
}

fn bench_equal_output(c: &mut Criterion) {
    let mut fixed_output = c.benchmark_group("ntt_matvec_q128/equal_output/output512");
    for width in WIDTHS {
        bench_equal_output_shape::<64>(&mut fixed_output, 8, width);
        bench_equal_output_shape::<128>(&mut fixed_output, 4, width);
        bench_equal_output_shape::<256>(&mut fixed_output, 2, width);
        bench_equal_output_shape::<512>(&mut fixed_output, 1, width);
    }
    fixed_output.finish();

    let mut equal_io = c.benchmark_group("ntt_matvec_q128/equal_io/input65536_output512");
    bench_equal_output_shape::<64>(&mut equal_io, 8, 1024);
    bench_equal_output_shape::<128>(&mut equal_io, 4, 512);
    bench_equal_output_shape::<256>(&mut equal_io, 2, 256);
    bench_equal_output_shape::<512>(&mut equal_io, 1, 128);
    equal_io.finish();
}

criterion_group! {
    name = ntt_matvec;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(1));
    targets = bench_rank_ring_dim, bench_width, bench_equal_output
}
criterion_main!(ntt_matvec);
