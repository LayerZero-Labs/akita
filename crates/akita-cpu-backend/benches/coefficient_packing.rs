#![allow(missing_docs)]

use akita_algebra::CyclotomicRing;
use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_cpu_backend::standalone::{
    dense_coefficient_packing, onehot_coefficient_packing,
    recursive_witness_coefficient_packing_partials, recursive_witness_from_i8_digits,
};
use akita_cpu_backend::{DensePoly, OneHotPoly};
use akita_types::{
    BasisMode, FpExtEncoding, PreparedSubringCoefficientPackingPoint,
    SubringCoefficientPackingGeometry,
};
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};
use std::time::Duration;

fn prepared_point<F, E, const D: usize>(
    num_positions: usize,
    positions_per_block: usize,
    source_num_vars: usize,
) -> PreparedSubringCoefficientPackingPoint<E>
where
    F: Field,
    E: ExtField<F>,
{
    const S: usize = 64;
    let geometry = SubringCoefficientPackingGeometry::try_new(E::DEGREE, D, S).unwrap();
    let public_point = (0..source_num_vars)
        .map(|index| E::from_u64(index as u64 + 2))
        .collect::<Vec<_>>();
    PreparedSubringCoefficientPackingPoint::new(
        geometry,
        BasisMode::Lagrange,
        num_positions,
        positions_per_block,
        source_num_vars,
        &public_point,
    )
    .unwrap()
}

fn bench_dense_shape<F, E, const D: usize>(
    c: &mut Criterion,
    name: &str,
    num_positions: usize,
    positions_per_block: usize,
) where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    let source_len = num_positions * D;
    let source_num_vars = source_len.trailing_zeros() as usize;
    assert_eq!(source_len, 1usize << source_num_vars);
    let point = prepared_point::<F, E, D>(num_positions, positions_per_block, source_num_vars);
    let rings = (0..num_positions)
        .map(|position| {
            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coefficient| {
                F::from_u64(((position * D + coefficient) % 251 + 1) as u64)
            }))
        })
        .collect();
    let poly = DensePoly::from_ring_coeffs(rings).unwrap();

    let mut group = c.benchmark_group("coefficient_packing");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(source_len as u64));
    group.bench_function(name, |b| {
        b.iter(|| {
            dense_coefficient_packing::<F, E, D>(black_box(&poly), black_box(&point)).unwrap()
        })
    });
    group.finish();
}

fn bench_recursive_shape<const D: usize>(
    c: &mut Criterion,
    name: &str,
    num_positions: usize,
    positions_per_block: usize,
) {
    type F = fp32::Field;
    type E = fp32::ExtensionField;
    let live_len = num_positions * D;
    let source_num_vars = live_len.trailing_zeros() as usize;
    let point = prepared_point::<F, E, D>(num_positions, positions_per_block, source_num_vars);
    let digits = (0..live_len).map(|index| (index % 15) as i8 - 7).collect();
    let poly = recursive_witness_from_i8_digits(digits);
    let mut group = c.benchmark_group("coefficient_packing");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(live_len as u64));
    group.bench_function(name, |b| {
        b.iter(|| {
            recursive_witness_coefficient_packing_partials::<F, E, D>(
                black_box(&poly),
                black_box(&point),
            )
            .unwrap()
        })
    });
    group.finish();
}

fn bench_onehot_shape<F, E, const D: usize>(
    c: &mut Criterion,
    name: &str,
    num_positions: usize,
    positions_per_block: usize,
    density_percent: usize,
) where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F>,
{
    const ONEHOT_K: usize = 256;
    let source_len = num_positions * D;
    let source_num_vars = source_len.trailing_zeros() as usize;
    assert_eq!(source_len, 1usize << source_num_vars);
    assert_eq!(source_len % ONEHOT_K, 0);
    let point = prepared_point::<F, E, D>(num_positions, positions_per_block, source_num_vars);
    let indices = (0..source_len / ONEHOT_K)
        .map(|chunk| (chunk % 100 < density_percent).then_some((chunk % ONEHOT_K) as u8))
        .collect();
    let poly = OneHotPoly::<F, u8>::new(ONEHOT_K, indices).unwrap();

    let mut group = c.benchmark_group("coefficient_packing");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(source_len as u64));
    group.bench_function(name, |b| {
        b.iter(|| {
            onehot_coefficient_packing::<F, E, D>(black_box(&poly), black_box(&point)).unwrap()
        })
    });
    group.finish();
}

fn bench_coefficient_packing(c: &mut Criterion) {
    // Scaled versions of the two fp32/nv30 production calls. They retain the
    // exact ring, packing, extension, and block geometry while bounding local
    // benchmark memory and iteration time.
    bench_dense_shape::<fp32::Field, fp32::ExtensionField, 2048>(c, "fp32_level0_d2048", 4096, 512);
    bench_dense_shape::<fp32::Field, fp32::ExtensionField, 1024>(
        c,
        "fp32_stride16_d1024",
        4096,
        512,
    );
    bench_dense_shape::<fp32::Field, fp32::ExtensionField, 512>(c, "fp32_stride8_d512", 8192, 1024);
    bench_dense_shape::<fp32::Field, fp32::ExtensionField, 256>(c, "fp32_level1_d256", 16384, 2048);
    bench_recursive_shape::<256>(c, "fp32_level1_recursive_d256", 16384, 2048);
    bench_onehot_shape::<fp32::Field, fp32::ExtensionField, 256>(
        c,
        "fp32_level1_onehot_d256",
        16384,
        2048,
        80,
    );
    bench_onehot_shape::<fp32::Field, fp32::ExtensionField, 256>(
        c,
        "fp32_level1_onehot_sparse_d256",
        16384,
        2048,
        1,
    );
    bench_onehot_shape::<fp32::Field, fp32::ExtensionField, 256>(
        c,
        "fp32_level1_onehot_10pct_d256",
        16384,
        2048,
        10,
    );
    bench_onehot_shape::<fp32::Field, fp32::ExtensionField, 256>(
        c,
        "fp32_level1_onehot_25pct_d256",
        16384,
        2048,
        25,
    );
    bench_onehot_shape::<fp32::Field, fp32::ExtensionField, 256>(
        c,
        "fp32_level1_onehot_half_d256",
        16384,
        2048,
        50,
    );
    bench_onehot_shape::<fp32::Field, fp32::ExtensionField, 256>(
        c,
        "fp32_level1_onehot_65pct_d256",
        16384,
        2048,
        65,
    );
    bench_onehot_shape::<fp32::Field, fp32::ExtensionField, 256>(
        c,
        "fp32_level1_onehot_zero_d256",
        16384,
        2048,
        0,
    );
    bench_dense_shape::<fp64::Field, fp64::ExtensionField, 256>(c, "fp64_d256", 4096, 1024);
    bench_dense_shape::<fp128::Field, fp128::Field, 64>(c, "fp128_d64", 4096, 1024);
}

criterion_group!(coefficient_packing, bench_coefficient_packing);
criterion_main!(coefficient_packing);
