#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::CanonicalField;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::kernels::linear::{
    decompose_rows_i8_into, mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_i8_dense,
    mat_vec_mul_ntt_i8_dense_single_row,
};
use akita_prover::DensePoly;
use akita_types::{prepare_ntt_cache, NttCacheMode};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type F = fp128::Field;
type Cfg = fp128::D64Full;
const D: usize = Cfg::D;
const NV: usize = 25;

fn make_dense_evals<Cfg: CommitmentConfig<Field = F>>(nv: usize) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(0xdead_beef);
    let len = 1usize << nv;
    let decomp = Cfg::decomposition();
    if decomp.log_commit_bound >= 128 {
        (0..len)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect()
    } else {
        let half_bound = 1i64 << (decomp.log_commit_bound.min(62) - 1);
        (0..len)
            .map(|_| F::from_i64(rng.gen_range(-half_bound..half_bound)))
            .collect()
    }
}

fn bench_dense_root_matvec_full_nv25_d32(c: &mut Criterion) {
    let evals = make_dense_evals::<Cfg>(NV);
    let poly = DensePoly::<F>::from_field_evals(NV, D, &evals).expect("dense poly");
    let layout = Cfg::get_params_for_batched_commitment(
        &akita_types::OpeningClaimsLayout::new(NV, 1).expect("singleton opening batch"),
    )
    .expect("layout");
    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(NV, 1).unwrap();
    let total = setup
        .expanded
        .shared_matrix
        .total_ring_elements_at::<D>()
        .unwrap();
    let ntt_shared = prepare_ntt_cache(
        setup
            .expanded
            .shared_matrix
            .ring_view::<D>(1, total)
            .unwrap(),
        NttCacheMode::BothTransforms,
    )
    .unwrap();
    let rings = poly.ring_coeffs::<D>().expect("dense ring view");
    let num_live_blocks = rings.len().div_ceil(layout.num_positions_per_block);
    let block_slices: Vec<&[akita_algebra::CyclotomicRing<F, D>]> = (0..num_live_blocks)
        .map(|i| {
            let start = i * layout.num_positions_per_block;
            if start >= rings.len() {
                &[] as &[akita_algebra::CyclotomicRing<F, D>]
            } else {
                &rings[start..(start + layout.num_positions_per_block).min(rings.len())]
            }
        })
        .collect();

    let n_a = layout.a_key.row_len();
    let inner_width = layout.inner_width();

    let mut group = c.benchmark_group("root_kernels");
    group.bench_function("dense_root_matvec_full_nv25_d32", |b| {
        b.iter(|| {
            black_box(mat_vec_mul_ntt_i8_dense(
                &ntt_shared,
                n_a,
                inner_width,
                black_box(&block_slices),
                layout.num_digits_inner,
                layout.log_basis_inner,
            ))
            .unwrap()
        })
    });
    group.bench_function(
        "dense_root_matvec_full_nv25_d32_single_row_subkernel",
        |b| {
            b.iter(|| {
                black_box(mat_vec_mul_ntt_i8_dense_single_row(
                    &ntt_shared,
                    inner_width,
                    black_box(&block_slices),
                    layout.num_digits_inner,
                    layout.log_basis_inner,
                ))
                .unwrap()
            })
        },
    );
    let mut digit_blocks: Vec<Vec<[i8; D]>> = block_slices
        .iter()
        .map(|block| vec![[0i8; D]; block.len() * layout.num_digits_inner])
        .collect();
    group.bench_function("dense_root_predecomp_digit_matvec_full_nv25_d32", |b| {
        b.iter(|| {
            for (block, digit_block) in block_slices.iter().zip(digit_blocks.iter_mut()) {
                decompose_rows_i8_into(
                    block,
                    digit_block,
                    layout.num_digits_inner,
                    layout.log_basis_inner,
                );
            }
            let digit_block_slices: Vec<&[[i8; D]]> =
                digit_blocks.iter().map(Vec::as_slice).collect();
            black_box(mat_vec_mul_ntt_digits_i8::<F, D>(
                &ntt_shared,
                n_a,
                inner_width,
                black_box(&digit_block_slices),
                layout.log_basis_inner,
            ))
            .unwrap()
        })
    });
    group.finish();
}

criterion_group!(root_kernels, bench_dense_root_matvec_full_nv25_d32);
criterion_main!(root_kernels);
