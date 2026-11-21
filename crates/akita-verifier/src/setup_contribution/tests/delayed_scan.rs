//! The delayed-reduction scan against the row-by-row oracle.
//!
//! The other scan tests run over fp128, where `SUM_IS_EXACT` is false, so
//! they only exercise `CanonicalProductSum`. Every fp32 and fp64 verification
//! takes `DelayedProductSum` instead; this test covers that version's slot
//! resets, block traversal, and nested slot and job sums.

use super::*;
use jolt_field::{FpExt4, Prime32Offset99, Ring, Unreduced};

type P = Prime32Offset99;
type E = FpExt4<P>;

fn extension(seed: u64) -> E {
    E::from_base_fn(|coordinate| P::from_u64(seed + coordinate as u64 * 17))
}

fn extensions(seed: u64, len: usize) -> Vec<E> {
    (0..len)
        .map(|index| extension(seed + 31 * index as u64))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn group(
    seed: u64,
    d_col_range: std::ops::Range<usize>,
    t_cols: usize,
    z_cols: usize,
    n_a: usize,
    n_b: usize,
) -> (SetupContributionGroupPlan<E>, DirectScanWeights<E>) {
    let e_len = d_col_range.len();
    test_group_plan(
        d_col_range,
        t_cols,
        z_cols,
        n_a,
        n_b,
        extensions(seed, e_len),
        extensions(seed + 100_000, t_cols),
        extensions(seed + 200_000, z_cols),
        extensions(seed + 300_000, n_a),
        extensions(seed + 400_000, n_b),
    )
}

#[test]
fn delayed_scan_matches_row_oracle_across_jobs_with_projected_roles() {
    const { assert!(<E as Unreduced>::SUM_IS_EXACT) };
    // A is 128-dimensional over a 64-dimensional base, so its role rings
    // project onto two base rings each.
    const D_A: usize = 128;
    const D_B: usize = 64;
    const D_D: usize = 64;
    let scan = finalize_test_plan(
        2,
        1100,
        vec![
            group(1, 0..700, 4, 600, 2, 2),
            group(2, 700..1100, 8, 300, 2, 3),
        ],
        CommitmentRingDims {
            inner: D_A,
            outer: D_B,
            opening: D_D,
        },
    );
    let geometry = scan.plan().projection_geometry();
    assert_eq!(geometry.base_ring_dim(), D_B);
    assert_eq!(geometry.a_ratio(), 2);
    let required = geometry.required();
    let job_rings = super::super::scan::SETUP_SCAN_JOB_RINGS;
    // Several jobs, a partial last job, and blocks that cut through segments.
    assert!(required > job_rings);
    assert_ne!(required % job_rings, 0);

    let setup_elements = required.div_ceil(2) * D_A;
    let setup = coefficient_setup(
        (0..setup_elements)
            .map(|index| P::from_u64(211 + 7_919 * index as u64))
            .collect(),
    );
    let alpha = E::from_u64(3);
    let alpha_pows_a = scalar_powers(alpha, D_A);
    let alpha_pows_b = scalar_powers(alpha, D_B);
    let alpha_pows_d = scalar_powers(alpha, D_D);
    let expected = scan
        .evaluate_direct_by_rows::<P>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d, D_A)
        .unwrap();
    assert_ne!(expected, E::zero());
    assert_eq!(scan.evaluate_direct::<P>(&setup).unwrap(), expected);
}
