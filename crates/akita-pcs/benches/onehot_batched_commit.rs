#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_planner::test_utils::akita_batched_root_layout;
use akita_prover::kernels::linear::{decompose_rows_i8, mat_vec_mul_ntt_single_i8};
use akita_prover::{AkitaPolyOps, CommitmentProver, OneHotPoly};
use akita_types::{AkitaScheduleInputs, LevelParams};
use criterion::{black_box, criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

type F = fp128::Field;
type Cfg = fp128::D64OneHot;
const D: usize = Cfg::D;

const SINGLE_NUM_VARS: usize = 34;
const BATCH_NUM_VARS: usize = 29;
const BATCH_SIZE: usize = 1 << 5;
const ONEHOT_K: usize = D;
const TOTAL_FIELD_ELEMS: u64 = 1u64 << SINGLE_NUM_VARS;

fn make_onehot_poly(layout: &LevelParams, seed: u64) -> OneHotPoly<F, D, u8> {
    let total_ring = layout.num_blocks * layout.block_len;
    let num_vars = layout.m_vars + layout.r_vars + D.trailing_zeros() as usize;
    assert_eq!(total_ring * ONEHOT_K, 1usize << num_vars);

    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();

    OneHotPoly::<F, D, u8>::new(ONEHOT_K, indices).expect("benchmark onehot poly")
}

fn bench_commit_breakdown(c: &mut Criterion) {
    let single_layout = Cfg::get_params_for_batched_commitment(
        &akita_types::ClaimIncidenceSummary::same_point(SINGLE_NUM_VARS, 1)
            .expect("singleton incidence"),
    )
    .expect("single layout");
    let batch_layout =
        akita_batched_root_layout::<Cfg>(BATCH_NUM_VARS, BATCH_SIZE).expect("batch layout");

    let single_poly = make_onehot_poly(&single_layout, 0x0bee_fcaf_e000_0030);
    let batched_polys: Vec<OneHotPoly<F, D, u8>> = (0..BATCH_SIZE)
        .map(|idx| make_onehot_poly(&batch_layout, 0x0bee_fcaf_e000_2500 + idx as u64))
        .collect();

    let single_setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
        SINGLE_NUM_VARS,
        1,
        1,
    );
    let batched_setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
        BATCH_NUM_VARS,
        BATCH_SIZE,
        1,
    );
    let single_inputs = AkitaScheduleInputs {
        num_vars: SINGLE_NUM_VARS,
        level: 0,
        current_w_len: single_layout.num_blocks * single_layout.block_len * D,
    };
    let single_params = akita_config::proof_optimized::level_params_with_log_basis::<Cfg>(
        single_inputs,
        Cfg::log_basis_at_level(single_inputs).expect("single log_basis_at_level"),
    )
    .expect("single level_params_with_log_basis");
    let batch_inputs = AkitaScheduleInputs {
        num_vars: BATCH_NUM_VARS,
        level: 0,
        current_w_len: batch_layout.num_blocks * batch_layout.block_len * D,
    };
    let batch_params = akita_config::proof_optimized::level_params_with_log_basis::<Cfg>(
        batch_inputs,
        Cfg::log_basis_at_level(batch_inputs).expect("batch log_basis_at_level"),
    )
    .expect("batch level_params_with_log_basis");

    let single_inner = single_poly
        .commit_inner_witness(
            &single_setup.expanded.shared_matrix,
            &single_setup.ntt_shared,
            single_params.a_key.row_len(),
            single_layout.block_len,
            single_layout.num_digits_commit,
            single_layout.num_digits_open,
            single_layout.log_basis,
            single_setup.expanded.seed.max_stride,
        )
        .expect("single inner witness");
    let batched_inner: Vec<_> = batched_polys
        .iter()
        .map(|poly| {
            poly.commit_inner_witness(
                &batched_setup.expanded.shared_matrix,
                &batched_setup.ntt_shared,
                batch_params.a_key.row_len(),
                batch_layout.block_len,
                batch_layout.num_digits_commit,
                batch_layout.num_digits_open,
                batch_layout.log_basis,
                batched_setup.expanded.seed.max_stride,
            )
            .expect("batched inner witness")
        })
        .collect();

    let single_n_b = single_params.b_key.row_len();
    let batch_n_b = batch_params.b_key.row_len();

    let mut group = c.benchmark_group("akita/onehot_commit_breakdown");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(50));
    group.measurement_time(Duration::from_millis(200));
    group.throughput(Throughput::Elements(TOTAL_FIELD_ELEMS));

    group.bench_function("single_full_commit_nv34", |b| {
        b.iter(|| {
            black_box(
                <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                    black_box(std::slice::from_ref(&single_poly)),
                    black_box(&single_setup),
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
                        single_params.a_key.row_len(),
                        single_layout.block_len,
                        single_layout.num_digits_commit,
                        single_layout.num_digits_open,
                        single_layout.log_basis,
                        single_setup.expanded.seed.max_stride,
                    )
                    .expect("single inner witness"),
            )
        })
    });

    group.bench_function("single_decompose_only_nv34", |b| {
        b.iter(|| {
            black_box(
                single_inner
                    .recomposed_inner_rows
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

    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);
    group.bench_function("single_outer_only_nv34", |b| {
        b.iter(|| {
            let flat = single_inner.decomposed_inner_rows.flat_digits().to_vec();
            black_box(mat_vec_mul_ntt_single_i8::<F, D>(
                &single_setup.ntt_shared,
                single_n_b,
                single_layout.outer_width(),
                &flat,
            ))
        })
    });
    group.measurement_time(Duration::from_millis(200));
    group.sample_size(10);

    group.bench_function("batched_full_commit_32xnv29", |b| {
        b.iter(|| {
            black_box(
                <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                    black_box(&batched_polys),
                    black_box(&batched_setup),
                )
                .expect("grouped commit"),
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
                            batch_params.a_key.row_len(),
                            batch_layout.block_len,
                            batch_layout.num_digits_commit,
                            batch_layout.num_digits_open,
                            batch_layout.log_basis,
                            batched_setup.expanded.seed.max_stride,
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
                            .recomposed_inner_rows
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
            let mut flat = Vec::with_capacity(BATCH_SIZE * batch_layout.outer_width());
            for inner in &batched_inner {
                flat.extend_from_slice(inner.decomposed_inner_rows.flat_digits());
            }
            black_box(mat_vec_mul_ntt_single_i8::<F, D>(
                &batched_setup.ntt_shared,
                batch_n_b,
                batch_layout.outer_width(),
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
