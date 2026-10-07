#![allow(missing_docs)]

use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_cpu_backend::DensePoly;
use akita_params::balanced_signed_digit_abs_bound;
use akita_types::{prepare_ntt_cache, NttCacheMode};
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Ring};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type F = fp128::Field;
type Cfg = fp128::Dense;
const NV: usize = 24;
// Root ring dimension of the `fp128_dense` schedule at `NV`.
const D: usize = 512;

fn make_dense_evals<Cfg: CommitmentConfig<Field = F>>(nv: usize) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(0xdead_beef);
    let len = 1usize << nv;
    let decomp = Cfg::decomposition();
    if decomp.log_commit_bound >= 128 {
        (0..len)
            .map(|_| F::from_u128_reduced(rng.gen::<u128>()))
            .collect()
    } else {
        let half_bound = 1i64 << (decomp.log_commit_bound.min(62) - 1);
        (0..len)
            .map(|_| F::from_i64(rng.gen_range(-half_bound..half_bound)))
            .collect()
    }
}

fn bench_dense_root_matvec(c: &mut Criterion) {
    let schedules = akita_config::test_support::workspace_schedule_catalog::<Cfg>()
        .expect("workspace schedule artifact");
    let evals = make_dense_evals::<Cfg>(NV);
    let poly = DensePoly::<F>::from_field_evals(NV, &evals).expect("dense poly");
    let layout = schedules
        .resolve_key(&akita_params::ScheduleLookupKey::single(
            akita_params::PolynomialGroupLayout::new(NV, 1),
        ))
        .expect("layout")
        .schedule()
        .root
        .params
        .clone();
    assert_eq!(
        layout.d_a(),
        D,
        "the nv{NV} root ring dimension changed; update `D`"
    );
    let capacity = akita_config::SetupRequirements::from_catalog::<Cfg>(&schedules, NV, 1)
        .unwrap()
        .matrix_capacity();
    let setup =
        akita_cpu_backend::AkitaProverSetup::<F>::generate_with_capacity(NV, 1, capacity).unwrap();
    let n_a = layout.inner().matrix.output_rank();
    let inner_width = layout.inner_width();
    let num_digits = layout.inner().digits.num_digits;
    let log_basis = layout.inner().digits.log_basis;
    let ntt_shared = prepare_ntt_cache(
        setup
            .expanded
            .shared_matrix()
            .ring_view::<D>(1, n_a * inner_width)
            .unwrap(),
        NttCacheMode::ExactNegacyclic {
            width: inner_width,
            rhs_abs_bound: balanced_signed_digit_abs_bound(log_basis).expect("signed digit basis"),
        },
    )
    .unwrap();
    let rings = poly.ring_coeffs::<D>().expect("dense ring view");
    let num_live_blocks = rings.len().div_ceil(layout.blocks().positions_per_block);
    let block_slices: Vec<&[akita_algebra::CyclotomicRing<F, D>]> = (0..num_live_blocks)
        .map(|i| {
            let start = i * layout.blocks().positions_per_block;
            if start >= rings.len() {
                &[] as &[akita_algebra::CyclotomicRing<F, D>]
            } else {
                &rings[start..(start + layout.blocks().positions_per_block).min(rings.len())]
            }
        })
        .collect();
    let decompose_params = BalancedDecomposePow2Params::new(num_digits, log_basis);

    let mut group = c.benchmark_group("root_kernels");
    // Per-block work of the backend's exact signed-i16 dense commit, which the
    // root takes when its digits are wider than 8 bits.
    group.bench_function(format!("dense_root_matvec_full_nv{NV}_d{D}"), |b| {
        b.iter(|| {
            cfg_iter!(black_box(&block_slices))
                .map(|block| {
                    let mut rhs = vec![[0i16; D]; inner_width];
                    for (ring, digits) in block.iter().zip(rhs.chunks_exact_mut(num_digits)) {
                        ring.balanced_decompose_pow2_i16_into(digits, &decompose_params);
                    }
                    ntt_shared.mat_vec_i16::<F>(log_basis, n_a, &rhs)
                })
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        })
    });
    group.finish();
}

criterion_group!(root_kernels, bench_dense_root_matvec);
criterion_main!(root_kernels);
