#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use hachi_pcs::algebra::Fp128;
use hachi_pcs::protocol::commitment::utils::linear::{
    decompose_rows_i8, flatten_i8_blocks, mat_vec_mul_ntt_single_i8,
};
use hachi_pcs::protocol::commitment::{Fp128OneHotCommitmentConfig, HachiScheduleInputs};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{HachiPolyOps, OneHotPoly};
use hachi_pcs::protocol::{CommitmentConfig, CommitmentScheme};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

type F = Fp128<0xffffffffffffffffffffffffffffe941>;
type Cfg = Fp128OneHotCommitmentConfig;
const D: usize = Cfg::D;

const SINGLE_NUM_VARS: usize = 34;
const BATCH_NUM_VARS: usize = 29;
const BATCH_SIZE: usize = 1 << 5;
const ONEHOT_K: usize = D;
const TOTAL_FIELD_ELEMS: u64 = 1u64 << SINGLE_NUM_VARS;

fn make_onehot_poly(num_vars: usize, seed: u64) -> OneHotPoly<F, D, u8> {
    let layout = Cfg::commitment_layout(num_vars).expect("benchmark layout");
    let total_ring = layout.num_blocks * layout.block_len;
    assert_eq!(total_ring * ONEHOT_K, 1usize << num_vars);

    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();

    OneHotPoly::<F, D, u8>::new(ONEHOT_K, indices, layout.r_vars, layout.m_vars)
        .expect("benchmark onehot poly")
}

fn root_n_b(num_vars: usize, layout: hachi_pcs::protocol::HachiCommitmentLayout) -> usize {
    Cfg::level_params(HachiScheduleInputs {
        max_num_vars: num_vars,
        level: 0,
        current_w_len: layout.num_blocks * layout.block_len * D,
    })
    .n_b
}

fn bench_commit_breakdown(c: &mut Criterion) {
    let single_layout = Cfg::commitment_layout(SINGLE_NUM_VARS).expect("single layout");
    let batch_layout = Cfg::commitment_layout(BATCH_NUM_VARS).expect("batch layout");

    let single_poly = make_onehot_poly(SINGLE_NUM_VARS, 0x0bee_fcaf_e000_0030);
    let batched_polys: Vec<OneHotPoly<F, D, u8>> = (0..BATCH_SIZE)
        .map(|idx| make_onehot_poly(BATCH_NUM_VARS, 0x0bee_fcaf_e000_2500 + idx as u64))
        .collect();

    let single_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(SINGLE_NUM_VARS, 1);
    let batched_setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(
        BATCH_NUM_VARS,
        BATCH_SIZE,
    );
    let batched_poly_groups = [&batched_polys[..]];

    let single_inner = single_poly
        .commit_inner_witness(
            &single_setup.expanded.shared_matrix,
            &single_setup.ntt_shared,
            single_layout.block_len,
            single_layout.num_digits_commit,
            single_layout.num_digits_open,
            single_layout.log_basis,
        )
        .expect("single inner witness");
    let batched_inner: Vec<_> = batched_polys
        .iter()
        .map(|poly| {
            poly.commit_inner_witness(
                &batched_setup.expanded.shared_matrix,
                &batched_setup.ntt_shared,
                batch_layout.block_len,
                batch_layout.num_digits_commit,
                batch_layout.num_digits_open,
                batch_layout.log_basis,
            )
            .expect("batched inner witness")
        })
        .collect();

    let single_n_b = root_n_b(SINGLE_NUM_VARS, single_layout);
    let batch_n_b = root_n_b(BATCH_NUM_VARS, batch_layout);

    let mut group = c.benchmark_group("hachi/onehot_commit_breakdown");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(50));
    group.measurement_time(Duration::from_millis(200));
    group.throughput(Throughput::Elements(TOTAL_FIELD_ELEMS));

    group.bench_function("single_full_commit_nv34", |b| {
        b.iter(|| {
            black_box(
                <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
                    black_box(&single_poly),
                    black_box(&single_setup),
                    black_box(&single_layout),
                )
                .expect("single commit"),
            )
        })
    });

    group.bench_function("single_inner_witness_nv34", |b| {
        b.iter(|| {
            black_box(
                single_poly
                    .commit_inner_witness(
                        &single_setup.expanded.shared_matrix,
                        &single_setup.ntt_shared,
                        single_layout.block_len,
                        single_layout.num_digits_commit,
                        single_layout.num_digits_open,
                        single_layout.log_basis,
                    )
                    .expect("single inner witness"),
            )
        })
    });

    group.bench_function("single_decompose_only_nv34", |b| {
        b.iter(|| {
            black_box(
                single_inner
                    .t
                    .iter()
                    .map(|t_i| {
                        decompose_rows_i8(
                            t_i,
                            single_layout.num_digits_open,
                            single_layout.log_basis,
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        })
    });

    group.bench_function("single_outer_only_nv34", |b| {
        b.iter(|| {
            let flat = flatten_i8_blocks(&single_inner.t_hat);
            black_box(mat_vec_mul_ntt_single_i8::<F, D>(
                &single_setup.ntt_shared,
                single_n_b,
                &flat,
            ))
        })
    });

    group.bench_function("batched_full_commit_32xnv29", |b| {
        b.iter(|| {
            black_box(
                <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::batched_commit(
                    black_box(&batched_poly_groups),
                    black_box(&batched_setup),
                    black_box(&batch_layout),
                )
                .expect("batched commit"),
            )
        })
    });

    group.bench_function("batched_inner_witness_32xnv29", |b| {
        b.iter(|| {
            black_box(
                batched_polys
                    .iter()
                    .map(|poly| {
                        poly.commit_inner_witness(
                            &batched_setup.expanded.shared_matrix,
                            &batched_setup.ntt_shared,
                            batch_layout.block_len,
                            batch_layout.num_digits_commit,
                            batch_layout.num_digits_open,
                            batch_layout.log_basis,
                        )
                        .expect("batched inner witness")
                    })
                    .collect::<Vec<_>>(),
            )
        })
    });

    group.bench_function("batched_decompose_only_32xnv29", |b| {
        b.iter(|| {
            black_box(
                batched_inner
                    .iter()
                    .map(|inner| {
                        inner
                            .t
                            .iter()
                            .map(|t_i| {
                                decompose_rows_i8(
                                    t_i,
                                    batch_layout.num_digits_open,
                                    batch_layout.log_basis,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            )
        })
    });

    group.bench_function("batched_outer_only_32xnv29", |b| {
        b.iter(|| {
            let mut flat = Vec::with_capacity(BATCH_SIZE * batch_layout.outer_width);
            for inner in &batched_inner {
                flat.extend(flatten_i8_blocks(&inner.t_hat));
            }
            black_box(mat_vec_mul_ntt_single_i8::<F, D>(
                &batched_setup.ntt_shared,
                batch_n_b,
                &flat,
            ))
        })
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_commit_breakdown
}
criterion_main!(benches);
