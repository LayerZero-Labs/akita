use super::*;

fn assert_retained_sweeps_match<const D: usize>(seed: u64) {
    use super::super::column_sweep::{column_sweep_ajtai_onehot_multi_forced, OneHotSweep};
    use rand::Rng;

    type F = Prime128Offset275;
    const K: usize = 256;

    let mut rng = StdRng::seed_from_u64(seed);
    let n_a = 3;
    let num_positions_per_block = 32;
    let num_vars = 18;
    let num_chunks = (1usize << num_vars) / K;
    let polys: Vec<OneHotPoly<F, u8>> = (0..3)
        .map(|_| {
            let indices = (0..num_chunks)
                .map(|_| (rng.gen::<u8>() % 4 != 0).then(|| rng.gen::<u8>()))
                .collect();
            OneHotPoly::new(K, indices).unwrap()
        })
        .collect();

    let active_a_cols = num_positions_per_block;
    let a_rows: Vec<CyclotomicRing<F, D>> = (0..n_a * active_a_cols)
        .map(|_| CyclotomicRing::random(&mut rng))
        .collect();
    let a_flat = FlatMatrix::from_ring_slice(&a_rows);
    let a_view = a_flat.ring_view::<D>(n_a, active_a_cols).unwrap();
    let sources: Vec<OneHotView<'_, F, D, u8>> =
        polys.iter().map(|poly| OneHotView { poly }).collect();

    let bucketed = column_sweep_ajtai_onehot_multi_forced(
        &a_view,
        &sources,
        n_a,
        active_a_cols,
        1,
        crate::compute::CpuBackend::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER,
        OneHotSweep::Bucketed,
    )
    .unwrap();
    let merge = column_sweep_ajtai_onehot_multi_forced(
        &a_view,
        &sources,
        n_a,
        active_a_cols,
        1,
        crate::compute::CpuBackend::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER,
        OneHotSweep::Merge,
    )
    .unwrap();

    assert_eq!(merge, bucketed);
}

#[test]
fn retained_sweeps_match_across_polys_and_dimensions() {
    assert_retained_sweeps_match::<64>(0x1a2b_3c4d);
    assert_retained_sweeps_match::<128>(0x1a2b_3c4e);
    assert_retained_sweeps_match::<256>(0x1a2b_3c4f);
}

#[test]
fn configured_scratch_budget_preserves_onehot_commit_arithmetic() {
    use crate::compute::{CommitInnerPlan, ComputeBackendSetup, CpuBackend, RootCommitKernel};
    use crate::AkitaProverSetup;
    use akita_types::SetupMatrixCapacity;

    type F = Prime128Offset275;
    const D: usize = 64;
    const K: usize = 64;

    let poly = OneHotPoly::<F, u8>::new(
        K,
        (0usize..256)
            .map(|chunk| (!chunk.is_multiple_of(5)).then_some((chunk % K) as u8))
            .collect(),
    )
    .unwrap();
    let plan = CommitInnerPlan {
        n_a: 2,
        num_positions_per_block: 16,
        num_digits_inner: 1,
        log_basis_inner: 1,
    };
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        8,
        1,
        SetupMatrixCapacity {
            num_field_elements: plan.n_a * plan.num_positions_per_block * D,
        },
    )
    .unwrap();
    let default_backend = CpuBackend::DEFAULT;
    let prepared = default_backend.prepare_setup(&setup).unwrap();
    let default = default_backend
        .commit_inner_group(
            &prepared,
            vec![OneHotView::<F, D, u8> { poly: &poly }],
            plan,
        )
        .unwrap();
    let constrained_backend = CpuBackend::with_resource_limits(usize::MAX, 1 << 20).unwrap();
    let constrained = constrained_backend
        .commit_inner_group(
            &prepared,
            vec![OneHotView::<F, D, u8> { poly: &poly }],
            plan,
        )
        .unwrap();
    assert_eq!(constrained.len(), default.len());
    for (constrained, default) in constrained.iter().zip(&default) {
        assert_eq!(constrained.inner_rows.coeffs(), default.inner_rows.coeffs());
    }

    let too_small_backend = CpuBackend::with_resource_limits(usize::MAX, 1).unwrap();
    assert!(matches!(
        too_small_backend.commit_inner_group(
            &prepared,
            vec![OneHotView::<F, D, u8> { poly: &poly }],
            plan,
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn production_selector_has_measured_regions_and_explicit_boundaries() {
    use super::super::column_sweep::{select_sweep, OneHotSweep};

    assert_eq!(select_sweep(512, 4096, 16), OneHotSweep::Bucketed);
    assert_eq!(select_sweep(512, 64, 16), OneHotSweep::Merge);

    assert_eq!(select_sweep(31, 64, 1), OneHotSweep::Bucketed);
    assert_eq!(select_sweep(32, 64, 1), OneHotSweep::Merge);
    assert_eq!(select_sweep(512, 64, 16), OneHotSweep::Merge);
    assert_eq!(select_sweep(512, 64, 17), OneHotSweep::Bucketed);
}

#[test]
fn retained_sweeps_handle_oversized_and_empty_blocks() {
    use super::super::column_sweep::{
        bucketed_sweep_tile, direct_sweep_tile, merge_sweep_tile, MERGE_COL_CHUNK,
    };

    type F = Prime128Offset275;
    const D: usize = 64;

    let mut rng = StdRng::seed_from_u64(0x05e2_517e);
    let n_a = 2;
    let num_positions_per_block = F::MAX_COMMIT_ACCUMULATIONS + 129;
    let active_a_cols = num_positions_per_block;
    let a_rows: Vec<CyclotomicRing<F, D>> = (0..n_a * active_a_cols)
        .map(|_| CyclotomicRing::random(&mut rng))
        .collect();
    let a_flat = FlatMatrix::from_ring_slice(&a_rows);
    let a_view = a_flat.ring_view::<D>(n_a, active_a_cols).unwrap();

    let big = (0..num_positions_per_block)
        .map(|pos| block_entry(pos, (pos * 7) % D))
        .collect::<Vec<_>>();
    let small = (0..97)
        .map(|pos| block_entry(pos * 11, pos % D))
        .collect::<Vec<_>>();
    let blocks = super::super::test_helpers::from_buckets(vec![big, small, Vec::new()]);
    let views = (0..blocks.num_live_blocks())
        .map(|block| blocks.block(block))
        .collect::<Vec<_>>();

    let mut chunk_buf = vec![WideCyclotomicRing::zero(); MERGE_COL_CHUNK];
    let merge = merge_sweep_tile(&a_view, &views, n_a, active_a_cols, 1, &mut chunk_buf);
    let direct = direct_sweep_tile(&a_view, &views, 1);
    let bucketed = bucketed_sweep_tile(&a_view, &views, n_a, active_a_cols, 1);
    assert_eq!(merge, direct);
    assert_eq!(bucketed, direct);
}

fn sweep_median_ms<F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    sources: &[OneHotView<'_, F, D, usize>],
    n_a: usize,
    active_a_cols: usize,
    sweep: super::super::column_sweep::OneHotSweep,
) -> f64
where
    F: Field + CanonicalEncoding + WithCommitAccumulator,
    F::Wide: AdditiveGroup + From<F>,
{
    use super::super::column_sweep::column_sweep_ajtai_onehot_multi_forced;
    use std::time::Instant;

    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        std::hint::black_box(
            column_sweep_ajtai_onehot_multi_forced(
                a_view,
                sources,
                n_a,
                active_a_cols,
                1,
                crate::compute::CpuBackend::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER,
                sweep,
            )
            .unwrap(),
        );
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    samples[2].as_secs_f64() * 1_000.0
}

fn benchmark_sweep_case<const D: usize>(
    label: &str,
    onehot_k: usize,
    positions_per_block: usize,
    blocks_per_poly: usize,
    num_polys: usize,
    hot_stride: usize,
) {
    use super::super::column_sweep::OneHotSweep;

    type F = Prime128Offset275;
    let n_a = 4;
    let field_elems_per_poly = blocks_per_poly * positions_per_block * D;
    let num_chunks = field_elems_per_poly / onehot_k;
    assert!(field_elems_per_poly.is_power_of_two());
    assert_eq!(num_chunks * onehot_k, field_elems_per_poly);
    let polys = (0..num_polys)
        .map(|poly| {
            let indices = (0..num_chunks)
                .map(|chunk| {
                    (chunk + poly)
                        .is_multiple_of(hot_stride)
                        .then_some((chunk * 17 + poly * 29) % onehot_k)
                })
                .collect();
            OneHotPoly::<F>::new(onehot_k, indices).unwrap()
        })
        .collect::<Vec<_>>();
    let sources = polys
        .iter()
        .map(|poly| OneHotView { poly })
        .collect::<Vec<_>>();
    let a_rows = vec![CyclotomicRing::<F, D>::zero(); n_a * positions_per_block];
    let a_flat = FlatMatrix::from_ring_slice(&a_rows);
    let a_view = a_flat.ring_view::<D>(n_a, positions_per_block).unwrap();

    let bucketed = sweep_median_ms(
        &a_view,
        &sources,
        n_a,
        positions_per_block,
        OneHotSweep::Bucketed,
    );
    let merge = sweep_median_ms(
        &a_view,
        &sources,
        n_a,
        positions_per_block,
        OneHotSweep::Merge,
    );
    println!(
        "{label}: D={D} K={onehot_k} positions={positions_per_block} blocks={blocks_per_poly} polys={num_polys} hot_stride={hot_stride} bucketed={bucketed:.3}ms merge={merge:.3}ms"
    );
}

/// Manual release-mode matrix for choosing production sweep regions.
#[test]
#[ignore = "manual one hot sweep benchmark"]
fn benchmark_production_sweep_matrix() {
    benchmark_sweep_case::<64>("tiny_single", 256, 4, 1, 1, 1);
    benchmark_sweep_case::<64>("tiny_pair", 256, 4, 2, 1, 1);
    benchmark_sweep_case::<64>("tiny_4", 256, 4, 4, 1, 1);
    benchmark_sweep_case::<64>("tiny_8", 256, 4, 8, 1, 1);
    benchmark_sweep_case::<64>("small_single", 256, 64, 16, 1, 1);
    benchmark_sweep_case::<64>("small_sparse", 256, 64, 64, 1, 16);
    benchmark_sweep_case::<64>("large_single", 256, 64, 512, 1, 1);
    benchmark_sweep_case::<64>("large_group4", 256, 64, 512, 4, 1);
    benchmark_sweep_case::<128>("equal_group2", 128, 64, 256, 2, 1);
    benchmark_sweep_case::<256>("k_lt_d_group4", 64, 64, 128, 4, 1);
    benchmark_sweep_case::<256>("equal_group8", 256, 64, 128, 8, 1);
    benchmark_sweep_case::<64>("wide_group29", 256, 64, 64, 29, 1);
    benchmark_sweep_case::<64>("dense_columns", 64, 4096, 64, 4, 1);
    benchmark_sweep_case::<64>("sparse_wide_columns", 256, 4096, 64, 4, 64);
}
