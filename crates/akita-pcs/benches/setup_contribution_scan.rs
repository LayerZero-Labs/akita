use akita_algebra::ring::scalar_powers;
use akita_field::Prime128OffsetA7F7;
use akita_types::{
    gadget_row_scalars, AkitaExpandedSetup, AkitaSetupSeed, FlatMatrix, RelationMatrixRowLayout,
    SetupContributionGroupInputs, SetupContributionPlan, SetupContributionPlanInputs,
    WitnessChunkLayout, WitnessChunkLengths, WitnessLayout,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

type F = Prime128OffsetA7F7;
const D_A: usize = 64;
const D_B_NESTED: usize = 32;
const D_D_NESTED: usize = 32;

struct ScanBenchCase {
    plan: SetupContributionPlan<F>,
    setup: AkitaExpandedSetup<F>,
    alpha_pows_a: Vec<F>,
    alpha_pows_b: Vec<F>,
    alpha_pows_d: Vec<F>,
}

fn scalar(value: u128) -> F {
    F::from_canonical_u128(value)
}

fn scan_bench_case(d_b: usize, d_d: usize) -> ScanBenchCase {
    scan_bench_case_with_layout(ScanBenchLayout {
        n_a: 32,
        n_b: 32,
        n_d: 32,
        num_claims: 2,
        num_blocks: 32,
        block_len: 8,
        depth_open: 2,
        depth_commit: 2,
        depth_fold: 2,
        log_basis: 4,
        d_b,
        d_d,
    })
}

fn wide_equal_dim_scan_bench_case() -> ScanBenchCase {
    scan_bench_case_with_layout(ScanBenchLayout {
        n_a: 6,
        n_b: 1,
        n_d: 1,
        num_claims: 1,
        num_blocks: 1024,
        block_len: 512,
        depth_open: 32,
        depth_commit: 2,
        depth_fold: 2,
        log_basis: 4,
        d_b: D_A,
        d_d: D_A,
    })
}

struct ScanBenchLayout {
    n_a: usize,
    n_b: usize,
    n_d: usize,
    num_claims: usize,
    num_blocks: usize,
    block_len: usize,
    depth_open: usize,
    depth_commit: usize,
    depth_fold: usize,
    log_basis: u32,
    d_b: usize,
    d_d: usize,
}

fn scan_bench_case_with_layout(layout: ScanBenchLayout) -> ScanBenchCase {
    let ScanBenchLayout {
        n_a,
        n_b,
        n_d,
        num_claims,
        num_blocks,
        block_len,
        depth_open,
        depth_commit,
        depth_fold,
        log_basis,
        d_b,
        d_d,
    } = layout;
    let rows = 1 + n_a + n_b + n_d;
    let z_range = block_len * depth_commit;
    let e_range = num_claims * depth_open * num_blocks;
    let t_range = n_a * depth_open * num_blocks;

    let inputs = SetupContributionPlanInputs {
        relation_matrix_row_layout: RelationMatrixRowLayout::WithDBlock,
        rows,
        n_a,
        n_b,
        n_d,
        num_groups: 1,
        num_polys_per_group: vec![num_claims],
        num_t_vectors: num_claims,
        num_claims,
        num_blocks,
        block_len,
        depth_open,
        depth_commit,
        depth_fold,
        inner_width: z_range,
        eq_tau1: (0..rows.next_power_of_two())
            .map(|idx| scalar(11 + idx as u128))
            .collect(),
    };
    let chunk_layout = WitnessLayout {
        blocks_per_chunk: num_blocks,
        chunks: vec![WitnessChunkLayout {
            offset_z: 0,
            offset_e: z_range,
            offset_t: z_range + e_range,
            offset_r: Some(z_range + e_range + t_range),
            global_block_base: 0,
        }],
        chunk_lengths: vec![WitnessChunkLengths {
            z_len: z_range,
            e_len: e_range,
            t_len: t_range,
            r_len: Some(0),
        }],
    };
    let single_group =
        SetupContributionGroupInputs::single_group_layout(&inputs, &chunk_layout, log_basis)
            .expect("valid single-group setup contribution layout");
    let groups = [single_group.group];
    let static_plan = SetupContributionPlan::prepare_static(
        &inputs,
        &groups,
        single_group.d_row_start,
        single_group.d_rows,
        single_group.d_physical_cols,
    )
    .expect("valid static setup contribution plan");
    let full_vec_randomness = (0..12)
        .map(|idx| scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let plan = SetupContributionPlan::finish_plan::<F>(
        &static_plan,
        &full_vec_randomness,
        None,
        None,
        Some(&fold_gadget),
        &groups,
    )
    .expect("valid setup contribution plan");

    let logical_required = plan.required().expect("non-empty plan");
    let base_d = D_A.min(d_b).min(d_d);
    let base_required = logical_required * (D_A / base_d);
    let setup_len = base_required.div_ceil(D_A / base_d);
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: D_A,
            max_setup_len: setup_len,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * D_A)
                .map(|idx| scalar(211 + idx as u128))
                .collect(),
            D_A,
        ),
    );

    ScanBenchCase {
        plan,
        setup,
        alpha_pows_a: scalar_powers(scalar(3), D_A),
        alpha_pows_b: scalar_powers(scalar(3), d_b),
        alpha_pows_d: scalar_powers(scalar(3), d_d),
    }
}

fn bench_setup_contribution_scan(c: &mut Criterion) {
    let equal_dims = scan_bench_case(D_A, D_A);
    let nested_dims = scan_bench_case(D_B_NESTED, D_D_NESTED);
    let wide_equal_dims = wide_equal_dim_scan_bench_case();
    let mut group = c.benchmark_group("setup_contribution_scan");
    group.throughput(Throughput::Elements(
        equal_dims.plan.required().expect("non-empty plan") as u64,
    ));
    group.bench_function("equal_role_dims", |bench| {
        bench.iter(|| {
            black_box(&equal_dims.plan)
                .evaluate_direct::<F>(
                    black_box(&equal_dims.setup),
                    black_box(&equal_dims.alpha_pows_a),
                    black_box(&equal_dims.alpha_pows_b),
                    black_box(&equal_dims.alpha_pows_d),
                )
                .expect("setup contribution direct evaluation")
        })
    });
    group.bench_function("nested_role_dims", |bench| {
        bench.iter(|| {
            black_box(&nested_dims.plan)
                .evaluate_direct::<F>(
                    black_box(&nested_dims.setup),
                    black_box(&nested_dims.alpha_pows_a),
                    black_box(&nested_dims.alpha_pows_b),
                    black_box(&nested_dims.alpha_pows_d),
                )
                .expect("setup contribution direct evaluation")
        })
    });
    group.bench_function("wide_equal_role_dims", |bench| {
        bench.iter(|| {
            black_box(&wide_equal_dims.plan)
                .evaluate_direct::<F>(
                    black_box(&wide_equal_dims.setup),
                    black_box(&wide_equal_dims.alpha_pows_a),
                    black_box(&wide_equal_dims.alpha_pows_b),
                    black_box(&wide_equal_dims.alpha_pows_d),
                )
                .expect("setup contribution direct evaluation")
        })
    });
    group.finish();
}

criterion_group!(setup_contribution_scan, bench_setup_contribution_scan);
criterion_main!(setup_contribution_scan);
