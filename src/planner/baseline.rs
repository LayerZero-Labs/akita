use std::collections::HashMap;

use super::digit_math::{baseline_optimal_m_r_split, compute_num_digits, compute_num_digits_fold};
use super::proof_size::{
    baseline_packed_digits_bytes, baseline_ring_vec_bytes, baseline_sumcheck_bytes, elem_bytes,
    sumcheck_rounds,
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
    pub max_num_vars: usize,
    pub min_lb: u32,
    pub max_lb: u32,
}

fn compute_level(
    bp: &BaselineParams,
    level: usize,
    current_w_len: usize,
    lb: u32,
) -> (usize, usize, usize, usize, usize, usize, usize, usize) {
    let alpha = bp.d.trailing_zeros() as usize;

    let (reduced, log_cb) = if level == 0 {
        (bp.max_num_vars - alpha, bp.log_commit_bound)
    } else {
        let num_ring = current_w_len / bp.d as usize;
        let rp2 = num_ring.next_power_of_two();
        (rp2.trailing_zeros() as usize, lb)
    };

    let (m, r) = baseline_optimal_m_r_split(bp.n_a, bp.challenge_l1_mass, log_cb, lb, reduced);
    let op = if log_cb < 128 { 128 } else { log_cb };
    let d_open = compute_num_digits(op, lb);
    let d_commit = compute_num_digits(log_cb, lb);
    let d_fold = compute_num_digits_fold(r, bp.challenge_l1_mass, lb);
    let bl = 1usize << m;
    let iw = bl * d_commit;
    let w_hat = (1usize << r) * d_open;
    let t_hat = (1usize << r) * bp.n_a as usize * d_open;
    let z_pre = iw * d_fold;
    let r_ct =
        (bp.n_d as usize + bp.n_b as usize + 2 + bp.n_a as usize) * compute_num_digits(128, lb);
    let w_ring = w_hat + t_hat + z_pre + r_ct;
    let nw = w_ring * bp.d as usize;
    let rnds = sumcheck_rounds(bp.d, nw);
    (m, r, d_open, d_commit, d_fold, w_ring, nw, rnds)
}

fn level_bytes(bp: &BaselineParams, lb: u32, rounds: usize) -> usize {
    let s1_deg = ((1u32 << lb) / 2 + 1) as usize;
    baseline_ring_vec_bytes(1, bp.d)
        + baseline_ring_vec_bytes(bp.n_d as usize, bp.d)
        + baseline_sumcheck_bytes(rounds, s1_deg)
        + elem_bytes()
        + baseline_sumcheck_bytes(rounds, 3)
        + baseline_ring_vec_bytes(bp.n_b as usize, bp.d)
        + elem_bytes()
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

        let (_, _, _, _, _, _, nw, rnds) = compute_level(bp, level, w_len, lb);
        if nw < w_len {
            for nlb in lb.max(bp.min_lb)..=bp.max_lb {
                let lbytes = level_bytes(bp, lb, rnds);
                let (sb, sl, stlb) = best_suffix(bp, memo, level + 1, nw, nlb);
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

    let root_w = 1usize << bp.max_num_vars;
    let mut overall: Option<MemoVal> = None;

    for rlb in bp.min_lb..=bp.max_lb {
        let (_, _, _, _, _, _, nw, rnds) = compute_level(bp, 0, root_w, rlb);
        if nw >= root_w {
            continue;
        }
        for nlb in rlb.max(bp.min_lb)..=bp.max_lb {
            let rb = level_bytes(bp, rlb, rnds);
            let (sb, sl, stlb) = best_suffix(bp, &mut memo, 1, nw, nlb);
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
/// `cargo test` or `cargo run --bin hachi-planner -- --validate`.
pub const BASELINE_CASES: &[(&str, u32, u32, usize, usize)] = &[
    //  (name,   d,  lcb, nv,  expected_total)
    ("onehot", 64, 1, 32, 97_277),
    ("full128", 128, 128, 25, 164_053),
    ("full128", 128, 128, 32, 170_637),
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
        max_num_vars: nv,
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
