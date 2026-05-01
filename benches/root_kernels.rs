#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hachi_pcs::protocol::commitment::utils::linear::{
    decompose_rows_i8_into, mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_i8_dense,
    mat_vec_mul_ntt_i8_dense_single_row,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::config::proof_optimized::fp128;
use hachi_pcs::protocol::hachi_poly_ops::DensePoly;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{CanonicalField, CommitmentScheme, FromSmallInt};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type F = fp128::Field;
type Cfg = fp128::D32Full;
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
    let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
    let layout = Cfg::commitment_layout(NV).expect("layout");
    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(NV, 1, 1);
    let num_blocks = poly.coeffs.len().div_ceil(layout.block_len);
    let block_slices: Vec<&[hachi_pcs::algebra::CyclotomicRing<F, D>]> = (0..num_blocks)
        .map(|i| {
            let start = i * layout.block_len;
            if start >= poly.coeffs.len() {
                &[] as &[hachi_pcs::algebra::CyclotomicRing<F, D>]
            } else {
                &poly.coeffs[start..(start + layout.block_len).min(poly.coeffs.len())]
            }
        })
        .collect();

    let envelope = Cfg::envelope(NV);
    let n_a = envelope.max_n_a;
    let inner_width = layout.inner_width();

    let mut group = c.benchmark_group("root_kernels");
    group.bench_function("dense_root_matvec_full_nv25_d32", |b| {
        b.iter(|| {
            black_box(mat_vec_mul_ntt_i8_dense(
                &setup.ntt_shared,
                n_a,
                inner_width,
                black_box(&block_slices),
                layout.num_digits_commit,
                layout.log_basis,
            ))
        })
    });
    group.bench_function(
        "dense_root_matvec_full_nv25_d32_single_row_subkernel",
        |b| {
            b.iter(|| {
                black_box(mat_vec_mul_ntt_i8_dense_single_row(
                    &setup.ntt_shared,
                    inner_width,
                    black_box(&block_slices),
                    layout.num_digits_commit,
                    layout.log_basis,
                ))
            })
        },
    );
    let mut digit_blocks: Vec<Vec<[i8; D]>> = block_slices
        .iter()
        .map(|block| vec![[0i8; D]; block.len() * layout.num_digits_commit])
        .collect();
    group.bench_function("dense_root_predecomp_digit_matvec_full_nv25_d32", |b| {
        b.iter(|| {
            for (block, digit_block) in block_slices.iter().zip(digit_blocks.iter_mut()) {
                decompose_rows_i8_into(
                    block,
                    digit_block,
                    layout.num_digits_commit,
                    layout.log_basis,
                );
            }
            let digit_block_slices: Vec<&[[i8; D]]> =
                digit_blocks.iter().map(Vec::as_slice).collect();
            black_box(mat_vec_mul_ntt_digits_i8::<F, D>(
                &setup.ntt_shared,
                n_a,
                inner_width,
                black_box(&digit_block_slices),
            ))
        })
    });
    group.finish();
}

criterion_group!(root_kernels, bench_dense_root_matvec_full_nv25_d32);
criterion_main!(root_kernels);
