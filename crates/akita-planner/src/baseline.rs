use std::collections::HashMap;

use super::proof_size::{
    baseline_packed_digits_bytes, baseline_ring_vec_bytes, baseline_sumcheck_bytes, field_bytes,
    sumcheck_rounds, FIELD_BITS,
};
use akita_types::layout::digit_math::{
    baseline_optimal_m_r_split, compute_num_digits_fold_with_claims, num_digits_for_bound,
};

/// Baseline planner result.
#[derive(Debug, Clone)]
pub struct BaselineResult {
    pub total: usize,
    pub num_levels: usize,
    pub tail_bytes: usize,
    pub final_w_len: usize,
    pub final_lb: u32,
}

/// Parameters for the baseline planner (fixed D, na, nb, nd).
pub struct BaselineParams {
    pub d: u32,
    pub n_a: u32,
    pub n_b: u32,
    pub n_d: u32,
    pub challenge_l1_mass: usize,
    pub log_commit_bound: u32,
    pub field_bits: u32,
    pub num_vars: usize,
    pub min_lb: u32,
    pub max_lb: u32,
}

/// Returns
/// `(m, r, d_open, d_commit, d_fold, w_ring, next_w_len, rounds, terminal_rounds)`.
///
/// `terminal_rounds` uses the terminal M-row layout (D-block dropped from
/// `r_ct`), matching `terminal_level_bytes`'s stage-2 sumcheck round count.
#[allow(clippy::type_complexity)]
fn compute_level(
    bp: &BaselineParams,
    level: usize,
    current_w_len: usize,
    lb: u32,
) -> (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
) {
    let alpha = bp.d.trailing_zeros() as usize;

    let (reduced, log_cb) = if level == 0 {
        (bp.num_vars - alpha, bp.log_commit_bound)
    } else {
        let num_ring = current_w_len / bp.d as usize;
        let rp2 = num_ring.next_power_of_two();
        (rp2.trailing_zeros() as usize, lb)
    };

    let (m, r) = baseline_optimal_m_r_split(
        bp.n_a,
        bp.challenge_l1_mass,
        log_cb,
        lb,
        reduced,
        bp.field_bits,
    );
    let op = log_cb.max(bp.field_bits);
    let d_open = num_digits_for_bound(op, bp.field_bits, lb);
    let d_commit = num_digits_for_bound(log_cb, bp.field_bits, lb);
    let d_fold = compute_num_digits_fold_with_claims(r, bp.challenge_l1_mass, lb, 1, bp.field_bits);
    let bl = 1usize << m;
    let iw = bl * d_commit;
    let w_hat = (1usize << r) * d_open;
    let t_hat = (1usize << r) * bp.n_a as usize * d_open;
    let z_pre = iw * d_fold;
    let digits_field = num_digits_for_bound(bp.field_bits, bp.field_bits, lb);
    let r_ct = (bp.n_d as usize + bp.n_b as usize + 2 + bp.n_a as usize) * digits_field;
    let r_ct_terminal = (bp.n_b as usize + 2 + bp.n_a as usize) * digits_field;
    let w_ring = w_hat + t_hat + z_pre + r_ct;
    let w_ring_terminal = w_hat + t_hat + z_pre + r_ct_terminal;
    let nw = w_ring * bp.d as usize;
    let nw_terminal = w_ring_terminal * bp.d as usize;
    let rnds = sumcheck_rounds(bp.d, nw);
    let rnds_terminal = sumcheck_rounds(bp.d, nw_terminal);
    (
        m,
        r,
        d_open,
        d_commit,
        d_fold,
        w_ring,
        nw,
        rnds,
        rnds_terminal,
    )
}

fn level_bytes(bp: &BaselineParams, lb: u32, rounds: usize) -> usize {
    let s1_deg = ((1u32 << lb) / 2 + 1) as usize;
    baseline_ring_vec_bytes(1, bp.d, bp.field_bits)
        + baseline_ring_vec_bytes(bp.n_d as usize, bp.d, bp.field_bits)
        + baseline_sumcheck_bytes(rounds, s1_deg, bp.field_bits)
        + field_bytes(bp.field_bits)
        + baseline_sumcheck_bytes(rounds, 3, bp.field_bits)
        + baseline_ring_vec_bytes(bp.n_b as usize, bp.d, bp.field_bits)
        + field_bytes(bp.field_bits)
}

/// Bytes for a terminal fold level: ships only `y` and the (relation-only)
/// stage-2 sumcheck. No stage-1, no next-witness commitment, no next-witness
/// evaluation claim, and no D-block `v` (the terminal M-row layout drops it);
/// the cleartext final witness is accounted for separately via
/// [`baseline_packed_digits_bytes`].
fn terminal_level_bytes(bp: &BaselineParams, rounds: usize) -> usize {
    baseline_ring_vec_bytes(1, bp.d, bp.field_bits)
        + baseline_sumcheck_bytes(rounds, 3, bp.field_bits)
}

/// Run the baseline planner matching the existing Rust `best_recursive_suffix` logic.
pub fn run_baseline_planner(bp: &BaselineParams) -> Option<BaselineResult> {
    type MemoKey = (usize, usize, u32);
    type LevelEntry = (u32, usize, usize, usize); // (lb, bytes, next_w, rounds)
    type MemoVal = (usize, Vec<LevelEntry>, u32); // (cost, levels, tail_lb)
    let mut memo: HashMap<MemoKey, MemoVal> = HashMap::new();

    fn best_suffix(
        bp: &BaselineParams,
        memo: &mut HashMap<MemoKey, MemoVal>,
        level: usize,
        w_len: usize,
        lb: u32,
    ) -> MemoVal {
        let key = (level, w_len, lb);
        if let Some(existing) = memo.get(&key) {
            return existing.clone();
        }

        let tail = baseline_packed_digits_bytes(w_len, lb);
        let mut best: MemoVal = (tail, Vec::new(), lb);

        let (_, _, _, _, _, _, nw, rnds, rnds_terminal) = compute_level(bp, level, w_len, lb);
        if nw < w_len {
            for nlb in lb.max(bp.min_lb)..=bp.max_lb {
                let (sb, sl, stlb) = best_suffix(bp, memo, level + 1, nw, nlb);
                // If the recursion's best is "ship direct" (no further folds),
                // this fold is the terminal one and pays the cheaper
                // `terminal_level_bytes`, with stage-2 rounds derived from
                // the terminal-layout witness length.
                let lbytes = if sl.is_empty() {
                    terminal_level_bytes(bp, rnds_terminal)
                } else {
                    level_bytes(bp, lb, rnds)
                };
                let cand = lbytes + sb;
                if cand < best.0 {
                    let mut levels = Vec::with_capacity(1 + sl.len());
                    levels.push((lb, lbytes, nw, rnds));
                    levels.extend(sl);
                    best = (cand, levels, stlb);
                }
            }
        }

        memo.insert(key, best.clone());
        best
    }

    let root_w = 1usize << bp.num_vars;
    let mut overall: Option<MemoVal> = None;

    for rlb in bp.min_lb..=bp.max_lb {
        let (_, _, _, _, _, _, nw, rnds, rnds_terminal) = compute_level(bp, 0, root_w, rlb);
        if nw >= root_w {
            continue;
        }
        for nlb in rlb.max(bp.min_lb)..=bp.max_lb {
            let (sb, sl, stlb) = best_suffix(bp, &mut memo, 1, nw, nlb);
            // Root is the terminal fold when the recursive suffix is just
            // "ship direct" with zero further fold levels; stage-2 rounds
            // come from the terminal-layout witness length.
            let rb = if sl.is_empty() {
                terminal_level_bytes(bp, rnds_terminal)
            } else {
                level_bytes(bp, rlb, rnds)
            };
            let total = rb + sb;
            let is_better = overall.as_ref().is_none_or(|(best, _, _)| total < *best);
            if is_better {
                let mut levels = Vec::with_capacity(1 + sl.len());
                levels.push((rlb, rb, nw, rnds));
                levels.extend(sl);
                overall = Some((total, levels, stlb));
            }
        }
    }

    let (total_no_wrapper, levels, tail_lb) = overall?;
    let total = total_no_wrapper + 4;
    let term_w = levels.last()?.2;
    let tail_bytes = baseline_packed_digits_bytes(term_w, tail_lb);

    Some(BaselineResult {
        total,
        num_levels: levels.len(),
        tail_bytes,
        final_w_len: term_w,
        final_lb: tail_lb,
    })
}

/// Known-good baseline results. Single source of truth for tests and CLI validation.
/// Update these when the cost model intentionally changes, then re-run
/// `cargo test` or `cargo run -p akita-planner --bin akita-planner -- --validate`.
pub const BASELINE_CASES: &[(&str, u32, u32, usize, usize)] = &[
    //  (name,   d,  lcb, nv,  expected_total)
    ("onehot", 64, 1, 32, 90_413),
    ("full128", 128, 128, 25, 154_861),
    ("full128", 128, 128, 32, 161_445),
];

/// Build [`BaselineParams`] from a `BASELINE_CASES` entry.
pub fn baseline_params_for(d: u32, lcb: u32, nv: usize) -> BaselineParams {
    let l1 = if d == 64 { 54 } else { 31 };
    BaselineParams {
        d,
        n_a: 1,
        n_b: 1,
        n_d: 1,
        challenge_l1_mass: l1,
        log_commit_bound: lcb,
        field_bits: FIELD_BITS,
        num_vars: nv,
        min_lb: 2,
        max_lb: 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_matches_expected() {
        for &(name, d, lcb, nv, expected) in BASELINE_CASES {
            let bp = baseline_params_for(d, lcb, nv);
            let r = run_baseline_planner(&bp).unwrap();
            assert_eq!(
                r.total, expected,
                "{name} nv={nv}: got {}, expected {expected}",
                r.total
            );
        }
    }

    #[test]
    fn tail_lb_matches_terminal_packing() {
        let configs: &[(u32, u32, &[usize])] =
            &[(64, 1, &[20, 25, 30, 32]), (128, 128, &[20, 25, 32])];
        for &(d, lcb, nvs) in configs {
            for &nv in nvs {
                let bp = baseline_params_for(d, lcb, nv);
                let Some(r) = run_baseline_planner(&bp) else {
                    continue;
                };
                assert_eq!(
                    r.tail_bytes,
                    baseline_packed_digits_bytes(r.final_w_len, r.final_lb),
                    "tail_bytes inconsistent at nv={nv}"
                );
            }
        }
    }
}
