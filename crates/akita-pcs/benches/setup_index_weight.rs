#![allow(missing_docs)]

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_field::Prime128OffsetA7F7;
use akita_types::{
    gadget_row_scalars, r_decomp_levels, CommitmentRingDims, CommittedGroupParams,
    InnerCommitMatrixParams, OpenCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams,
    SetupContributionGroupInputs, SetupContributionPlan, SisModulusProfileId, WitnessLayout,
    MAX_WITNESS_CHUNKS,
};
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion,
    SamplingMode,
};
use std::time::Duration;

type F = Prime128OffsetA7F7;
const D: usize = 64;

struct SetupIndexWeightBenchCase {
    plan: SetupContributionPlan<F>,
    dense_weights: Vec<F>,
    rho: Vec<F>,
    alpha: F,
}

fn test_scalar(value: u128) -> F {
    F::from_canonical_u128(value)
}

fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(20);
    group.nresamples(1001);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
}

fn make_case(num_live_blocks: usize, blocks_per_chunk: usize) -> SetupIndexWeightBenchCase {
    assert!(num_live_blocks.is_power_of_two());
    assert!(blocks_per_chunk.is_power_of_two());
    assert!(blocks_per_chunk <= num_live_blocks);
    assert_eq!(num_live_blocks % blocks_per_chunk, 0);

    let num_claims = 2;
    let depth_open = 2;
    let depth_commit = 2;
    let num_positions_per_block = 8;
    let n_a = 2;
    let n_b = 2;
    let n_d = 2;
    let log_basis = 4;
    let mut level_params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D,
        log_basis,
        n_a,
        n_b,
        n_d,
        akita_challenges::SparseChallengeConfig::pm1_only(1),
    )
    .with_decomp(
        num_positions_per_block,
        num_live_blocks * num_positions_per_block,
        depth_commit,
        depth_open,
        depth_open,
    )
    .unwrap();
    level_params.inner_commit_matrix = InnerCommitMatrixParams::new_unchecked(
        level_params.inner_commit_matrix.security_policy(),
        level_params
            .inner_commit_matrix
            .sis_table_key()
            .table_digest,
        level_params.inner_commit_matrix.sis_modulus_profile(),
        n_a,
        num_positions_per_block * depth_commit,
        1,
        D,
    );
    level_params.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
        level_params.outer_commit_matrix.security_policy(),
        level_params
            .outer_commit_matrix
            .sis_table_key()
            .table_digest,
        level_params.outer_commit_matrix.sis_modulus_profile(),
        n_b,
        num_claims * n_a * depth_commit * num_live_blocks,
        1,
        D,
    );
    level_params.open_commit_matrix = OpenCommitMatrixParams::new_unchecked(
        level_params.open_commit_matrix.security_policy(),
        level_params.open_commit_matrix.sis_table_key().table_digest,
        level_params.open_commit_matrix.sis_modulus_profile(),
        n_d,
        num_claims * depth_open * num_live_blocks,
        1,
        D,
    );
    let depth_fold = level_params.num_digits_fold();
    let opening_batch = OpeningClaimsLayout::new(0, num_claims).unwrap();
    let layout = WitnessLayout::new(
        &level_params,
        &opening_batch,
        num_live_blocks / blocks_per_chunk,
        1 + n_a + n_b + n_d,
        r_decomp_levels::<F>(log_basis),
    )
    .unwrap();

    let tau1 = (0..3)
        .map(|idx| test_scalar(31 + idx as u128))
        .collect::<Vec<_>>();
    let eq_tau1 = EqPolynomial::evals(&tau1).unwrap().into();
    let opening_source_len = layout.total_len();
    let groups = vec![SetupContributionGroupInputs {
        group_id: 0,
        num_claims,
        depth_fold,
        a_row_start: 1,
        b_row_start: 1 + n_a,
    }];
    let relation_address_geometry = akita_types::RelationAddressGeometry::new(
        CommitmentRingDims::uniform(D),
        D,
        opening_source_len,
    )
    .unwrap();
    let full_vec_randomness = (0..relation_address_geometry.relation_lane_variable_count())
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let alpha = test_scalar(3);
    let plan = SetupContributionPlan::prepare::<F>(
        &level_params,
        &opening_batch,
        eq_tau1,
        &layout,
        &groups,
        &full_vec_randomness,
        Some(&fold_gadget),
        relation_address_geometry,
        alpha,
    )
    .unwrap();
    let rho_bits = plan.required().next_power_of_two().trailing_zeros() as usize;
    let rho = (0..rho_bits)
        .map(|idx| test_scalar(901 + idx as u128))
        .collect::<Vec<_>>();

    let dense_weights = plan.materialize_setup_index_weights(alpha).unwrap();
    let dense = dense_weights
        .iter()
        .copied()
        .enumerate()
        .fold(F::zero(), |acc, (index, weight)| {
            acc + eq_eval_at_index(&rho, index) * weight
        });
    assert_eq!(
        plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap(),
        dense
    );

    SetupIndexWeightBenchCase {
        plan,
        dense_weights,
        rho,
        alpha,
    }
}

fn bench_setup_index_weight(c: &mut Criterion) {
    let mut group = c.benchmark_group("setup_index_weight_mle");
    configure_group(&mut group);

    for num_live_blocks in [64usize, 256, 1024, 4096, 16384] {
        for (layout, blocks_per_chunk) in [
            ("single_chunk", num_live_blocks),
            (
                "up_to_64_chunks",
                64usize
                    .max(num_live_blocks / MAX_WITNESS_CHUNKS)
                    .min(num_live_blocks),
            ),
        ] {
            let case = make_case(num_live_blocks, blocks_per_chunk);
            group.bench_with_input(
                BenchmarkId::new(format!("{layout}/span_path"), num_live_blocks),
                &case,
                |b, case| {
                    b.iter(|| {
                        black_box(
                            case.plan
                                .evaluate_setup_index_weight_mle(
                                    black_box(&case.rho),
                                    black_box(case.alpha),
                                )
                                .unwrap(),
                        )
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{layout}/dense_path"), num_live_blocks),
                &case,
                |b, case| {
                    b.iter(|| {
                        black_box(case.dense_weights.iter().copied().enumerate().fold(
                            F::zero(),
                            |acc, (index, weight)| {
                                acc + eq_eval_at_index(black_box(&case.rho), index) * weight
                            },
                        ))
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(setup_index_weight, bench_setup_index_weight);
criterion_main!(setup_index_weight);
