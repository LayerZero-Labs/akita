//! Verifier α-eval cost model for tiered-root candidate scoring.
//!
//! `specs/tiered_commit.md` §12 lists the components a planner-side
//! cost model must include. This module implements the dominant
//! pieces — the `setup_contribution` α-eval rectangle (the term §10
//! singles out as the load-bearing optimisation target) and the F α-
//! eval rectangle — exactly the way they appear in the spec's
//! Performance table. Sub-dominant terms (gadget-structured
//! contribution, r-tail growth, sumcheck-domain growth) are left to a
//! follow-up so the first landing of this helper has a verifiable
//! numeric baseline.
//!
//! The unit tests in this module pin the helper against the
//! checked-in `onehot_d32_nv32` Performance table from
//! `specs/tiered_commit.md` so any future refactor that drifts the
//! cost model also drifts these tests, surfacing the mismatch loudly.

use crate::sis_security::SisModulusFamily;
use akita_field::AkitaError;

/// Inputs describing one tiered candidate's verifier-cost shape.
///
/// `n_b_prime` and `n_f` are the SIS ranks chosen by
/// [`tiered_b_prime_rank`] /
/// [`tiered_f_rank`](akita_types::layout::sis_derivation::tiered_f_rank)
/// at the candidate's `(split_factor, outer_log_basis, num_digits_outer)`.
/// The cost model takes them as inputs so the planner can supply them
/// directly and avoid an extra floor-table lookup on the hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TieredAlphaEvalCostInputs {
    /// Inner SIS rank `n_a`.
    pub n_a: usize,
    /// SIS rank of `B'` (= `n_b` in the legacy `split_factor = 1` case).
    pub n_b_prime: usize,
    /// Inner SIS rank `n_d`.
    pub n_d: usize,
    /// SIS rank of `F`. `0` for legacy `split_factor = 1` candidates.
    pub n_f: usize,
    /// Number of committed blocks `B = 2^r_vars`.
    pub num_blocks: usize,
    /// Ring elements per block.
    pub block_len: usize,
    /// `δ_commit`.
    pub num_digits_commit: usize,
    /// `δ_open`.
    pub num_digits_open: usize,
    /// `δ_outer`. `0` for legacy.
    pub num_digits_outer: usize,
    /// Splitting factor `f`. `1` for legacy.
    pub split_factor: usize,
    /// Max bundle size across opening points (i.e. `max_group_polys`),
    /// used for the B-side width per
    /// `crates/akita-types/src/layout/params.rs::LevelParams::with_decomp`.
    pub max_group_polys: usize,
    /// Total number of batched claims (drives the W-side width).
    pub num_claims: usize,
    /// Ring dimension `D`.
    pub ring_dimension: usize,
}

impl TieredAlphaEvalCostInputs {
    fn validate(&self) -> Result<(), AkitaError> {
        if self.split_factor == 0 {
            return Err(AkitaError::InvalidInput(
                "tiered cost: split_factor must be ≥ 1".to_string(),
            ));
        }
        if self.split_factor > 1 && self.n_f == 0 {
            return Err(AkitaError::InvalidInput(
                "tiered cost: split_factor > 1 requires n_f ≥ 1".to_string(),
            ));
        }
        if self.num_blocks == 0 || self.block_len == 0 || self.ring_dimension == 0 {
            return Err(AkitaError::InvalidInput(
                "tiered cost: num_blocks, block_len, and ring_dimension must be ≥ 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Returns the `(setup_cells, f_cells)` pair that `setup_contribution`'s
/// per-row α-eval loop would emit on a tiered candidate.
///
/// `setup_cells` is the sum over `r ∈ [0, r_max)` of the per-row
/// `max(e_w, e_t, e_z)` cell count, exactly matching the dispatch in
/// `compute_setup_contribution`'s inner loop. `f_cells` is the
/// independent F α-eval rectangle (0 for legacy).
///
/// # Errors
///
/// Returns an error when `inputs.validate()` fails (zero split-factor,
/// missing F-rank for tiered, or zero block geometry).
pub fn tiered_setup_contribution_alpha_eval_cells(
    inputs: &TieredAlphaEvalCostInputs,
) -> Result<(usize, usize), AkitaError> {
    inputs.validate()?;

    // Per-spec §Performance:
    //   n_cols_w        = num_claims · num_blocks · δ_open
    //   n_cols_t (full) = max_group_polys · n_a · num_blocks · δ_open
    //   z_range         = block_len · δ_commit
    //
    // The chunked-B path divides n_cols_t by `split_factor`; the legacy
    // path leaves it unchanged (split_factor = 1).
    let n_cols_w = inputs
        .num_claims
        .checked_mul(inputs.num_blocks)
        .and_then(|v| v.checked_mul(inputs.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidInput("n_cols_w overflow".to_string()))?;
    let n_cols_t_full = inputs
        .max_group_polys
        .checked_mul(inputs.n_a)
        .and_then(|v| v.checked_mul(inputs.num_blocks))
        .and_then(|v| v.checked_mul(inputs.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidInput("n_cols_t overflow".to_string()))?;
    let n_cols_t_chunked = n_cols_t_full / inputs.split_factor;
    let z_range = inputs
        .block_len
        .checked_mul(inputs.num_digits_commit)
        .ok_or_else(|| AkitaError::InvalidInput("z_range overflow".to_string()))?;

    let r_max = inputs.n_d.max(inputs.n_b_prime).max(inputs.n_a);

    let setup_cells: usize = (0..r_max)
        .map(|row| {
            let mut e = 0usize;
            if row < inputs.n_d {
                e = e.max(n_cols_w);
            }
            if row < inputs.n_b_prime {
                e = e.max(n_cols_t_chunked);
            }
            if row < inputs.n_a {
                e = e.max(z_range);
            }
            e
        })
        .sum();

    let f_cells = if inputs.split_factor > 1 {
        inputs
            .n_f
            .checked_mul(inputs.n_b_prime)
            .and_then(|v| v.checked_mul(inputs.split_factor))
            .and_then(|v| v.checked_mul(inputs.num_digits_outer))
            .ok_or_else(|| AkitaError::InvalidInput("F α-eval width overflow".to_string()))?
    } else {
        0
    };

    Ok((setup_cells, f_cells))
}

/// Convenience wrapper: returns total α-eval *ops*
/// (`(setup_cells + f_cells) · ring_dimension`).
///
/// # Errors
///
/// Same as [`tiered_setup_contribution_alpha_eval_cells`].
pub fn tiered_setup_contribution_alpha_eval_ops(
    inputs: &TieredAlphaEvalCostInputs,
) -> Result<u128, AkitaError> {
    let (setup_cells, f_cells) = tiered_setup_contribution_alpha_eval_cells(inputs)?;
    Ok((setup_cells as u128 + f_cells as u128) * inputs.ring_dimension as u128)
}

/// Result of a single planner candidate evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TieredCandidate {
    /// Splitting factor evaluated (`1` for the legacy baseline).
    pub split_factor: usize,
    /// `δ_outer`. `0` for legacy.
    pub num_digits_outer: usize,
    /// Outer gadget basis. `0` for legacy.
    pub outer_log_basis: u32,
    /// SIS rank chosen for B'.
    pub n_b_prime: usize,
    /// SIS rank chosen for F. `0` for legacy.
    pub n_f: usize,
    /// Total α-eval ops the candidate would charge.
    pub setup_contribution_ops: u128,
}

/// Compute the SIS ranks `(n_b', n_F)` for a tiered candidate.
///
/// `inner_t_inf_bound` is the existing inner gadget bound on `t̂`
/// entries (i.e. `(1 << log_basis) - 1` for the legacy balanced i8
/// convention). The two-case binding proof in `specs/tiered_commit.md`
/// §6 doubles it for the `Δt` collision bound, which
/// [`tiered_b_prime_rank`](akita_types::layout::sis_derivation::tiered_b_prime_rank)
/// applies internally.
///
/// # Errors
///
/// Returns an error if the floor table does not cover the requested
/// `(family, D, collision, width)` tuple.
pub fn size_tiered_candidate(
    family: SisModulusFamily,
    ring_dimension: u32,
    inner_t_inf_bound: u32,
    full_outer_width: usize,
    split_factor: usize,
    outer_log_basis: u32,
    num_digits_outer: usize,
) -> Result<(u32, u32), AkitaError> {
    let n_b_prime = akita_types::layout::sis_derivation::tiered_b_prime_rank(
        family,
        ring_dimension,
        inner_t_inf_bound,
        full_outer_width,
        split_factor,
    )?;
    let n_f = if split_factor > 1 {
        akita_types::layout::sis_derivation::tiered_f_rank(
            family,
            ring_dimension,
            outer_log_basis,
            n_b_prime,
            split_factor,
            num_digits_outer,
        )?
    } else {
        // Legacy mirrors today's behaviour: no F key, no F-rank.
        0
    };
    Ok((n_b_prime, n_f))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `onehot_d32_nv32` root baseline parameters from
    /// `specs/tiered_commit.md` Performance section.
    fn onehot_d32_nv32_baseline(
        n_b_prime: usize,
        n_f: usize,
        split_factor: usize,
        num_digits_outer: usize,
    ) -> TieredAlphaEvalCostInputs {
        TieredAlphaEvalCostInputs {
            n_a: 3,
            n_b_prime,
            n_d: 2,
            n_f,
            num_blocks: 2048,
            block_len: 65_536,
            num_digits_commit: 1,
            num_digits_open: 64,
            num_digits_outer,
            split_factor,
            max_group_polys: 1,
            num_claims: 1,
            ring_dimension: 32,
        }
    }

    /// Legacy baseline: 27,262,976 α-evals per the spec's Performance
    /// table (`f = 1` row).
    #[test]
    fn legacy_baseline_matches_spec_performance_table() {
        let inputs = onehot_d32_nv32_baseline(2, 0, 1, 0);
        let ops = tiered_setup_contribution_alpha_eval_ops(&inputs).unwrap();
        assert_eq!(
            ops, 27_262_976,
            "legacy f=1 α-eval count must match spec/tiered_commit.md Performance table"
        );
    }

    /// f = 2 tiered: 14,685,696 α-evals (with δ_outer = 22, n_F = 2,
    /// n_b' = 2). 1.86× speedup vs legacy per the spec table.
    #[test]
    fn tiered_f2_matches_spec_performance_table() {
        let inputs = onehot_d32_nv32_baseline(2, 2, 2, 22);
        let ops = tiered_setup_contribution_alpha_eval_ops(&inputs).unwrap();
        assert_eq!(
            ops, 14_685_696,
            "tiered f=2 α-eval count must match spec/tiered_commit.md Performance table"
        );
    }

    /// f = 4 tiered: 10,497,024 α-evals. 2.60× speedup vs legacy.
    /// Beyond f = 4 the W-side becomes the bottleneck and the cost
    /// plateaus, so f = 4 is the planner's sweet spot for this preset.
    #[test]
    fn tiered_f4_matches_spec_performance_table_and_hits_w_bottleneck() {
        let f4 = onehot_d32_nv32_baseline(2, 2, 4, 22);
        let f8 = onehot_d32_nv32_baseline(2, 2, 8, 22);
        let ops_f4 = tiered_setup_contribution_alpha_eval_ops(&f4).unwrap();
        let ops_f8 = tiered_setup_contribution_alpha_eval_ops(&f8).unwrap();
        assert_eq!(
            ops_f4, 10_497_024,
            "tiered f=4 α-eval count must match spec/tiered_commit.md Performance table"
        );
        assert_eq!(
            ops_f8, 10_508_288,
            "tiered f=8 α-eval count must match spec/tiered_commit.md Performance table"
        );
        // f=8 should not be meaningfully faster than f=4 because W is
        // already the bottleneck. The ~11k delta is the additional F
        // α-evals at the larger split — exactly what the spec calls
        // out as the "marginal returns" plateau.
        assert!(
            ops_f8 > ops_f4,
            "f=8 must be slightly worse than f=4 once W bottlenecks (extra F cells dominate)"
        );
    }

    /// Speedup ratio sanity check at f=4 vs legacy: 27.26M / 10.49M ≈ 2.60.
    /// Pin to two decimals so accidental cost-model drift fails loudly.
    #[test]
    fn tiered_speedup_at_f4_matches_spec_estimate() {
        let legacy = onehot_d32_nv32_baseline(2, 0, 1, 0);
        let tiered = onehot_d32_nv32_baseline(2, 2, 4, 22);
        let legacy_ops = tiered_setup_contribution_alpha_eval_ops(&legacy).unwrap() as f64;
        let tiered_ops = tiered_setup_contribution_alpha_eval_ops(&tiered).unwrap() as f64;
        let speedup = legacy_ops / tiered_ops;
        assert!(
            (speedup - 2.60).abs() < 0.01,
            "expected ~2.60× speedup at f=4 per spec; got {speedup:.3}"
        );
    }

    #[test]
    fn rejects_zero_split_factor() {
        let mut inputs = onehot_d32_nv32_baseline(2, 0, 0, 0);
        inputs.split_factor = 0;
        let err = tiered_setup_contribution_alpha_eval_ops(&inputs).expect_err("0 split rejected");
        assert!(format!("{err:?}").contains("split_factor"));
    }

    #[test]
    fn rejects_tiered_with_zero_f_rank() {
        let mut inputs = onehot_d32_nv32_baseline(2, 0, 4, 22);
        inputs.n_f = 0;
        let err =
            tiered_setup_contribution_alpha_eval_ops(&inputs).expect_err("tiered requires n_f ≥ 1");
        assert!(format!("{err:?}").contains("n_f"));
    }

    #[test]
    fn size_tiered_candidate_returns_valid_ranks_for_q128() {
        // Smaller widths than the full onehot_d32_nv32 root so the
        // floor table covers them. We just check the call succeeds and
        // returns positive ranks; the spec's Performance illustrative
        // `n_b' = n_b = 2`, `n_F = 2` choices reflect upper-bound SIS
        // sizing, not exact returned values.
        let (n_b_prime, n_f) = size_tiered_candidate(
            SisModulusFamily::Q128,
            32,
            (1u32 << 6) - 1,
            393_216,
            4,
            6,
            22,
        )
        .expect("ranks should resolve");
        assert!(n_b_prime >= 1);
        assert!(n_f >= 1);
    }

    #[test]
    fn size_tiered_candidate_with_split_factor_one_returns_zero_nf() {
        let (_n_b_prime, n_f) = size_tiered_candidate(
            SisModulusFamily::Q128,
            32,
            (1u32 << 6) - 1,
            393_216,
            1,
            0,
            0,
        )
        .expect("legacy ranks should resolve");
        assert_eq!(n_f, 0, "legacy candidate has no F key");
    }
}
