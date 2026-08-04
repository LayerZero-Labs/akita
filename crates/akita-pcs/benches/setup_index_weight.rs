#![allow(missing_docs)]

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_field::Prime128OffsetA7F7;
use akita_types::{
    gadget_row_scalars, r_decomp_levels, CommitmentRingDims, CommittedGroupParams,
    InnerCommitMatrixParams, OpenCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams,
    PolynomialGroupLayout, PrecommittedGroupDescriptor, PrecommittedLevelParams,
    PreparedRelationAddress, SetupContributionGroupInputs, SetupContributionPlan,
    SisModulusProfileId, WitnessLayout, MAX_WITNESS_CHUNKS,
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
    make_case_with_shape(
        num_live_blocks,
        blocks_per_chunk,
        CommitmentRingDims::uniform(D),
        D,
        1,
    )
}

fn make_case_with_shape(
    num_live_blocks: usize,
    blocks_per_chunk: usize,
    role_dims: CommitmentRingDims,
    outgoing_ring_dim: usize,
    num_groups: usize,
) -> SetupIndexWeightBenchCase {
    assert!(num_live_blocks.is_power_of_two());
    assert!(blocks_per_chunk.is_power_of_two());
    assert!(blocks_per_chunk <= num_live_blocks);
    assert_eq!(num_live_blocks % blocks_per_chunk, 0);
    assert!(num_groups > 0);

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
        role_dims.d_a(),
        log_basis,
        n_a,
        n_b,
        n_d,
        if num_groups == 1 {
            akita_challenges::SparseChallengeConfig::pm1_only(1)
        } else {
            akita_challenges::SparseChallengeConfig::production_for_ring_dim(role_dims.d_a())
                .unwrap()
        },
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
        role_dims.d_a(),
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
        role_dims.d_b(),
    );
    level_params.open_commit_matrix = OpenCommitMatrixParams::new_unchecked(
        level_params.open_commit_matrix.security_policy(),
        level_params.open_commit_matrix.sis_table_key().table_digest,
        level_params.open_commit_matrix.sis_modulus_profile(),
        n_d,
        num_claims * depth_open * num_live_blocks,
        1,
        role_dims.d_d(),
    );
    if num_groups > 1 {
        let group_layout = PolynomialGroupLayout::new(0, num_claims);
        let frozen_layout = PrecommittedGroupDescriptor::from_params(group_layout, &level_params);
        let outer_projection_ratio = role_dims.d_a() / role_dims.d_b();
        let frozen_b_width = n_a
            .checked_mul(depth_commit)
            .and_then(|width| width.checked_mul(num_live_blocks))
            .and_then(|width| width.checked_mul(num_claims))
            .and_then(|width| width.checked_mul(outer_projection_ratio))
            .unwrap();
        let frozen_outer = OuterCommitMatrixParams::new_unchecked(
            level_params.outer_commit_matrix.security_policy(),
            level_params
                .outer_commit_matrix
                .sis_table_key()
                .table_digest,
            level_params.outer_commit_matrix.sis_modulus_profile(),
            n_b,
            frozen_b_width,
            level_params.outer_commit_matrix.coeff_linf_bound(),
            role_dims.d_b(),
        );
        let frozen = PrecommittedLevelParams {
            layout: frozen_layout,
            inner_commit_matrix: level_params.inner_commit_matrix.clone(),
            outer_commit_matrix: frozen_outer,
            log_basis_open: level_params.log_basis_open,
            fold_challenge_config: level_params.fold_challenge_config,
            num_digits_inner: level_params.num_digits_inner,
            num_digits_outer: level_params.num_digits_outer,
            num_digits_open: level_params.num_digits_open,
            num_digits_fold_one: level_params.num_digits_fold_one,
        };
        level_params.precommitted_groups = vec![frozen; num_groups - 1];
    }
    let depth_fold = level_params
        .num_digits_fold(num_claims, level_params.field_bits_for_cache())
        .unwrap();
    let opening_batch =
        OpeningClaimsLayout::from_group_sizes(0, &vec![num_claims; num_groups]).unwrap();
    let layout = WitnessLayout::new(
        &level_params,
        &opening_batch,
        num_live_blocks / blocks_per_chunk,
        r_decomp_levels::<F>(log_basis),
    )
    .unwrap();

    let relation_rows = level_params
        .relation_matrix_row_count(opening_batch.num_groups())
        .unwrap();
    let tau1_len = relation_rows.next_power_of_two().trailing_zeros() as usize;
    let tau1 = (0..tau1_len)
        .map(|idx| test_scalar(31 + idx as u128))
        .collect::<Vec<_>>();
    let eq_tau1: std::sync::Arc<[F]> = EqPolynomial::evals(&tau1).unwrap().into();
    let opening_source_len = layout.live_coeff_len();
    let groups = opening_batch
        .root_group_order()
        .unwrap()
        .into_iter()
        .map(|group_id| {
            let group_params = level_params.group_params(&opening_batch, group_id).unwrap();
            SetupContributionGroupInputs {
                group_id,
                num_claims,
                depth_fold: level_params
                    .num_digits_fold_for_params(
                        group_params,
                        num_claims,
                        level_params.field_bits_for_cache(),
                    )
                    .unwrap(),
                a_row_start: level_params
                    .a_row_range(&opening_batch, group_id)
                    .unwrap()
                    .start,
                b_row_start: level_params
                    .commitment_row_range(&opening_batch, group_id)
                    .unwrap()
                    .start,
            }
        })
        .collect::<Vec<_>>();
    let relation_address_geometry =
        akita_types::RelationAddressGeometry::new(role_dims, outgoing_ring_dim, opening_source_len)
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
        PreparedRelationAddress::new(&full_vec_randomness).unwrap(),
        Some(&fold_gadget),
        relation_address_geometry,
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
                BenchmarkId::new(format!("uniform/{layout}/tensor"), num_live_blocks),
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
                BenchmarkId::new(format!("uniform/{layout}/dense"), num_live_blocks),
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

    let mut shape_group = c.benchmark_group("setup_index_weight_shapes");
    configure_group(&mut shape_group);
    let num_live_blocks = 1024;
    let mixed_dims = CommitmentRingDims {
        inner: 64,
        outer: 32,
        opening: 32,
    };
    let wider_mixed_dims = CommitmentRingDims {
        inner: 64,
        outer: 16,
        opening: 32,
    };
    for (shape, role_dims, outgoing_ring_dim, num_groups, blocks_per_chunk) in [
        ("mixed_d/single_group/single_chunk", mixed_dims, 16, 1, 1024),
        ("mixed_d/single_group/64_chunks", mixed_dims, 16, 1, 16),
        (
            "mixed_d/a64_b16_d32/single_chunk",
            wider_mixed_dims,
            16,
            1,
            1024,
        ),
        ("mixed_d/a64_b16_d32/64_chunks", wider_mixed_dims, 16, 1, 16),
        (
            "uniform/two_groups/single_chunk",
            CommitmentRingDims::uniform(D),
            D,
            2,
            1024,
        ),
        (
            "uniform/two_groups/64_chunks",
            CommitmentRingDims::uniform(D),
            D,
            2,
            16,
        ),
    ] {
        let case = make_case_with_shape(
            num_live_blocks,
            blocks_per_chunk,
            role_dims,
            outgoing_ring_dim,
            num_groups,
        );
        shape_group.bench_with_input(
            BenchmarkId::new(format!("{shape}/tensor"), num_live_blocks),
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
    }
    shape_group.finish();
}

criterion_group!(setup_index_weight, bench_setup_index_weight);
criterion_main!(setup_index_weight);
