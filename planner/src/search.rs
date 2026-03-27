use std::collections::HashMap;

use crate::digit_math::{compute_num_digits, compute_num_digits_fold, r_decomp_levels};
use crate::proof_size::{
    elem_bytes, packed_digits_bytes, ring_vec_bytes, stage1_bytes_optimized, sumcheck_bytes,
    sumcheck_rounds,
};
use crate::sis_security::min_rank_for_secure_width;

// ── Ring configurations ────────────────────────────────────────────────────

/// A candidate ring configuration for one folding level.
#[derive(Clone, Debug)]
pub struct RingConfig {
    pub d: u32,
    pub n_a: u32,
    pub challenge_l1_mass: usize,
    pub label: &'static str,
}

/// All 9 ring configurations available to the planner.
pub const ALL_RING_CONFIGS: &[RingConfig] = &[
    RingConfig {
        d: 64,
        n_a: 1,
        challenge_l1_mass: 54,
        label: "D64-na1",
    },
    RingConfig {
        d: 64,
        n_a: 2,
        challenge_l1_mass: 54,
        label: "D64-na2",
    },
    RingConfig {
        d: 32,
        n_a: 1,
        challenge_l1_mass: 256,
        label: "D32-na1",
    },
    RingConfig {
        d: 32,
        n_a: 2,
        challenge_l1_mass: 256,
        label: "D32-na2",
    },
    RingConfig {
        d: 32,
        n_a: 3,
        challenge_l1_mass: 256,
        label: "D32-na3",
    },
    RingConfig {
        d: 16,
        n_a: 1,
        challenge_l1_mass: 2048,
        label: "D16-na1",
    },
    RingConfig {
        d: 16,
        n_a: 2,
        challenge_l1_mass: 2048,
        label: "D16-na2",
    },
    RingConfig {
        d: 16,
        n_a: 3,
        challenge_l1_mass: 2048,
        label: "D16-na3",
    },
    RingConfig {
        d: 16,
        n_a: 4,
        challenge_l1_mass: 2048,
        label: "D16-na4",
    },
];

const MIN_LB: u32 = 2;
const MAX_LB: u32 = 7;

const HALF_FIELD_BOUND_P275: u128 = (u128::MAX - 274) / 2; // (2^128 - 275) / 2

// ── Witness computation ────────────────────────────────────────────────────

struct LevelComputation {
    m_vars: u32,
    r_vars: u32,
    delta_commit: usize,
    delta_open: usize,
    delta_fold: usize,
    w_ring_elems: usize,
    next_w_len: usize,
    rounds: usize,
}

fn optimal_m_r_split(
    n_a: u32,
    challenge_l1_mass: usize,
    log_commit_bound: u32,
    log_basis: u32,
    reduced_vars: usize,
    num_ring: usize,
) -> (usize, usize) {
    if reduced_vars <= 2 || reduced_vars >= 53 {
        let r = reduced_vars / 2;
        return (reduced_vars - r, r);
    }

    let open_bound = if log_commit_bound < 128 {
        128
    } else {
        log_commit_bound
    };
    let delta_open = compute_num_digits(open_bound, log_basis) as u64;
    let delta_commit = compute_num_digits(log_commit_bound, log_basis) as u64;
    let c1 = delta_open + n_a as u64 * delta_commit;

    let mut best_r = reduced_vars / 2;
    let mut best_cost = u64::MAX;

    for r in 1..reduced_vars {
        let m = reduced_vars - r;
        let delta_fold = compute_num_digits_fold(r, challenge_l1_mass, log_basis) as u64;
        let m_eff = if num_ring > 0 {
            num_ring.div_ceil(1usize << r) as u64
        } else {
            1u64 << m
        };
        let cost = c1.saturating_mul(1u64 << r)
            + delta_commit
                .saturating_mul(delta_fold)
                .saturating_mul(m_eff);
        if cost < best_cost {
            best_cost = cost;
            best_r = r;
        }
    }

    (reduced_vars - best_r, best_r)
}

struct LevelWitnessArgs {
    level: usize,
    current_w_len: usize,
    max_num_vars: usize,
    log_basis: u32,
    half_field_bound: u128,
    nb: u32,
    nd: u32,
    log_commit_bound: u32,
    tight_zpre: bool,
}

fn compute_level_witness(cfg: &RingConfig, a: &LevelWitnessArgs) -> LevelComputation {
    let level = a.level;
    let current_w_len = a.current_w_len;
    let max_num_vars = a.max_num_vars;
    let log_basis = a.log_basis;
    let half_field_bound = a.half_field_bound;
    let nb = a.nb;
    let nd = a.nd;
    let log_commit_bound = a.log_commit_bound;
    let tight_zpre = a.tight_zpre;
    let d = cfg.d;
    let alpha = d.trailing_zeros();

    let (reduced_vars, log_cb, num_ring_actual) = if level == 0 {
        let rv = max_num_vars - alpha as usize;
        (rv, log_commit_bound, 1usize << rv)
    } else {
        let num_ring = current_w_len / d as usize;
        let ring_pow2 = num_ring.next_power_of_two();
        let rv = ring_pow2.trailing_zeros() as usize;
        (rv, log_basis, num_ring)
    };

    let nr_arg = if tight_zpre { num_ring_actual } else { 0 };
    let (m_vars, r_vars) = optimal_m_r_split(
        cfg.n_a,
        cfg.challenge_l1_mass,
        log_cb,
        log_basis,
        reduced_vars,
        nr_arg,
    );

    let open_bound = if log_cb < 128 { 128 } else { log_cb };
    let delta_open = compute_num_digits(open_bound, log_basis);
    let delta_commit = compute_num_digits(log_cb, log_basis);
    let delta_fold = compute_num_digits_fold(r_vars, cfg.challenge_l1_mass, log_basis);

    let num_blocks = 1usize << r_vars;
    let m_actual = if tight_zpre {
        num_ring_actual.div_ceil(num_blocks)
    } else {
        1usize << m_vars
    };
    let inner_width = m_actual * delta_commit;

    let w_hat = num_blocks * delta_open;
    let t_hat = num_blocks * cfg.n_a as usize * delta_open;
    let z_pre = inner_width * delta_fold;
    let m_row = nd as usize + nb as usize + 2 + cfg.n_a as usize;
    let r_ct = m_row * r_decomp_levels(128, half_field_bound, log_basis);
    let w_ring_elems = w_hat + t_hat + z_pre + r_ct;
    let next_w_len = w_ring_elems * d as usize;
    let rounds = sumcheck_rounds(d, next_w_len);

    LevelComputation {
        m_vars: m_vars as u32,
        r_vars: r_vars as u32,
        delta_commit,
        delta_open,
        delta_fold,
        w_ring_elems,
        next_w_len,
        rounds,
    }
}

// ── Output types ───────────────────────────────────────────────────────────

/// Per-level parameters in the planned schedule.
#[derive(Clone, Debug)]
pub struct PlannedLevel {
    pub d: u32,
    pub lb: u32,
    pub m_vars: u32,
    pub r_vars: u32,
    pub na: u32,
    pub nb: u32,
    pub nd: u32,
    pub delta_open: usize,
    pub delta_fold: usize,
    pub delta_commit: usize,
    pub w_ring: usize,
    pub next_w_len: usize,
    pub level_bytes: usize,
    pub label: &'static str,
}

/// Complete planned schedule.
#[derive(Clone, Debug)]
pub struct Schedule {
    pub levels: Vec<PlannedLevel>,
    pub tail_bytes: usize,
    pub total_bytes: usize,
    pub final_w_len: usize,
    pub final_lb: u32,
}

// ── Planner options ────────────────────────────────────────────────────────

/// Configuration knobs for the planner.
pub struct PlannerOptions {
    pub log_commit_bound: u32,
    pub max_num_vars: usize,
    pub ring_configs: &'static [RingConfig],
    pub half_field_bound: u128,
    pub opt_sumcheck: bool,
    pub monotone_d: bool,
    pub tight_zpre: bool,
}

impl PlannerOptions {
    pub fn new(log_commit_bound: u32, max_num_vars: usize) -> Self {
        Self {
            log_commit_bound,
            max_num_vars,
            ring_configs: ALL_RING_CONFIGS,
            half_field_bound: HALF_FIELD_BOUND_P275,
            opt_sumcheck: true,
            monotone_d: true,
            tight_zpre: true,
        }
    }

    pub fn with_tight_zpre(mut self, v: bool) -> Self {
        self.tight_zpre = v;
        self
    }

    #[allow(dead_code)]
    pub fn with_opt_sumcheck(mut self, v: bool) -> Self {
        self.opt_sumcheck = v;
        self
    }
}

// ── Planner internals ──────────────────────────────────────────────────────

type MemoKey = (usize, u32, u32); // (w_len, cur_D, prev_lb)

#[derive(Clone)]
struct BestSuffix {
    cost: usize,
    levels: Vec<PlannedLevel>,
    tail_lb: u32,
}

/// Internal planner state bundling options and precomputed data.
struct Planner {
    opts: PlannerOptions,
    unique_ds: Vec<u32>,
    memo: HashMap<MemoKey, BestSuffix>,
}

impl Planner {
    fn new(opts: PlannerOptions) -> Self {
        let mut ds: Vec<u32> = opts.ring_configs.iter().map(|c| c.d).collect();
        ds.sort_unstable();
        ds.dedup();
        ds.reverse();
        Self {
            opts,
            unique_ds: ds,
            memo: HashMap::new(),
        }
    }

    fn cfgs_for_d(&self, d: u32) -> impl Iterator<Item = &RingConfig> {
        self.opts.ring_configs.iter().filter(move |c| c.d == d)
    }

    fn level_prefix(&self, cfg: &RingConfig, lb: u32, rounds: usize, nd: u32) -> usize {
        let s1 = if self.opts.opt_sumcheck {
            stage1_bytes_optimized(rounds, lb)
        } else {
            let deg = ((1u32 << lb) / 2 + 1) as usize;
            sumcheck_bytes(rounds, deg)
        };
        ring_vec_bytes(1, cfg.d)
            + ring_vec_bytes(nd as usize, cfg.d)
            + s1
            + elem_bytes()
            + sumcheck_bytes(rounds, 3)
            + elem_bytes()
    }

    fn try_level(
        &self,
        cfg: &RingConfig,
        level: usize,
        w_len: usize,
        lb: u32,
        log_cb: u32,
    ) -> Option<(usize, LevelComputation, u32, u32)> {
        let args = LevelWitnessArgs {
            level,
            current_w_len: w_len,
            max_num_vars: self.opts.max_num_vars,
            log_basis: lb,
            half_field_bound: self.opts.half_field_bound,
            nb: 1,
            nd: 1,
            log_commit_bound: log_cb,
            tight_zpre: self.opts.tight_zpre,
        };
        let lc = compute_level_witness(cfg, &args);
        if lc.next_w_len >= w_len {
            return None;
        }

        let num_ring = if level > 0 {
            w_len / cfg.d as usize
        } else {
            1usize << (self.opts.max_num_vars - cfg.d.trailing_zeros() as usize)
        };
        let inner_width = if self.opts.tight_zpre {
            num_ring.div_ceil(1usize << lc.r_vars) * lc.delta_commit
        } else {
            (1usize << lc.m_vars) * lc.delta_commit
        };
        let a_collision = if level == 0 && log_cb == 1 {
            2
        } else {
            (1u32 << lb) - 1
        };
        let na_needed = min_rank_for_secure_width(cfg.d, a_collision, inner_width)?;
        if na_needed > cfg.n_a {
            return None;
        }

        let bd_collision = (1u32 << lb) - 1;
        let outer = cfg.n_a as usize * lc.delta_open * (1usize << lc.r_vars);
        let d_mat = lc.delta_open * (1usize << lc.r_vars);
        let nb = min_rank_for_secure_width(cfg.d, bd_collision, outer)?;
        let nd = min_rank_for_secure_width(cfg.d, bd_collision, d_mat)?;

        let args = LevelWitnessArgs { nb, nd, ..args };
        let lc = compute_level_witness(cfg, &args);
        if lc.next_w_len >= w_len {
            return None;
        }
        let prefix = self.level_prefix(cfg, lb, lc.rounds, nd);
        Some((prefix, lc, nb, nd))
    }

    fn tail_entry_nb(&self, w_len: usize, d: u32, tail_lb: u32) -> Option<u32> {
        let ring_elems = w_len.div_ceil(d as usize);
        min_rank_for_secure_width(d, (1u32 << tail_lb) - 1, ring_elems)
    }

    fn best_from(&mut self, w_len: usize, cur_d: u32, prev_lb: u32) -> BestSuffix {
        let key = (w_len, cur_d, prev_lb);
        if let Some(existing) = self.memo.get(&key) {
            return existing.clone();
        }

        let mut best = BestSuffix {
            cost: usize::MAX,
            levels: Vec::new(),
            tail_lb: prev_lb,
        };

        if let Some(tnb) = self.tail_entry_nb(w_len, cur_d, prev_lb) {
            let t = ring_vec_bytes(tnb as usize, cur_d) + packed_digits_bytes(w_len, prev_lb);
            best = BestSuffix {
                cost: t,
                levels: Vec::new(),
                tail_lb: prev_lb,
            };
        }

        // Recursion terminates naturally: try_level requires next_w_len < w_len,
        // so w_len strictly decreases at each level. Memoization prevents
        // revisiting the same (w_len, D, lb) state.
        let cfgs: Vec<RingConfig> = self.cfgs_for_d(cur_d).cloned().collect();
        let unique_ds = self.unique_ds.clone();
        let monotone_d = self.opts.monotone_d;

        for cfg in &cfgs {
            for lb in MIN_LB..=MAX_LB {
                let result = self.try_level(cfg, 1, w_len, lb, lb);
                let Some((prefix, lc, nb_self, nd_self)) = result else {
                    continue;
                };
                let entry_commit = ring_vec_bytes(nb_self as usize, cur_d);

                for &next_d in &unique_ds {
                    if monotone_d && next_d > cur_d {
                        continue;
                    }
                    let suffix = self.best_from(lc.next_w_len, next_d, lb);
                    if suffix.cost == usize::MAX {
                        continue;
                    }
                    let total = entry_commit + prefix + suffix.cost;
                    if total < best.cost {
                        let mut levels = Vec::with_capacity(1 + suffix.levels.len());
                        levels.push(PlannedLevel {
                            d: cfg.d,
                            lb,
                            m_vars: lc.m_vars,
                            r_vars: lc.r_vars,
                            na: cfg.n_a,
                            nb: nb_self,
                            nd: nd_self,
                            delta_open: lc.delta_open,
                            delta_fold: lc.delta_fold,
                            delta_commit: lc.delta_commit,
                            w_ring: lc.w_ring_elems,
                            next_w_len: lc.next_w_len,
                            level_bytes: entry_commit + prefix,
                            label: cfg.label,
                        });
                        levels.extend_from_slice(&suffix.levels);
                        best = BestSuffix {
                            cost: total,
                            levels,
                            tail_lb: suffix.tail_lb,
                        };
                    }
                }
            }
        }

        self.memo.insert(key, best.clone());
        best
    }
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Run the universal planner.
pub fn run_universal_planner(opts: &PlannerOptions) -> Schedule {
    let owned_opts = PlannerOptions {
        log_commit_bound: opts.log_commit_bound,
        max_num_vars: opts.max_num_vars,
        ring_configs: opts.ring_configs,
        half_field_bound: opts.half_field_bound,
        opt_sumcheck: opts.opt_sumcheck,
        monotone_d: opts.monotone_d,
        tight_zpre: opts.tight_zpre,
    };
    let mut planner = Planner::new(owned_opts);

    let root_w_len = 1usize << opts.max_num_vars;
    let mut overall_best: Option<(usize, Vec<PlannedLevel>, u32)> = None;

    let all_cfgs: Vec<RingConfig> = opts.ring_configs.to_vec();
    let unique_ds = planner.unique_ds.clone();

    for root_cfg in &all_cfgs {
        for root_lb in MIN_LB..=MAX_LB {
            let result = planner.try_level(root_cfg, 0, root_w_len, root_lb, opts.log_commit_bound);
            let Some((root_prefix, root_lc, root_nb, root_nd)) = result else {
                continue;
            };

            for &next_d in &unique_ds {
                if opts.monotone_d && next_d > root_cfg.d {
                    continue;
                }
                let suffix = planner.best_from(root_lc.next_w_len, next_d, root_lb);
                if suffix.cost == usize::MAX {
                    continue;
                }
                let total = root_prefix + suffix.cost + 4;
                let is_better = overall_best
                    .as_ref()
                    .map_or(true, |(best_total, _, _)| total < *best_total);
                if is_better {
                    let mut levels = Vec::with_capacity(1 + suffix.levels.len());
                    levels.push(PlannedLevel {
                        d: root_cfg.d,
                        lb: root_lb,
                        m_vars: root_lc.m_vars,
                        r_vars: root_lc.r_vars,
                        na: root_cfg.n_a,
                        nb: root_nb,
                        nd: root_nd,
                        delta_open: root_lc.delta_open,
                        delta_fold: root_lc.delta_fold,
                        delta_commit: root_lc.delta_commit,
                        w_ring: root_lc.w_ring_elems,
                        next_w_len: root_lc.next_w_len,
                        level_bytes: root_prefix,
                        label: root_cfg.label,
                    });
                    levels.extend_from_slice(&suffix.levels);
                    overall_best = Some((total, levels, suffix.tail_lb));
                }
            }
        }
    }

    match overall_best {
        Some((total, levels, tail_lb)) => {
            let final_w_len = levels.last().map_or(0, |l| l.next_w_len);
            let tail_bytes = packed_digits_bytes(final_w_len, tail_lb);
            Schedule {
                levels,
                tail_bytes,
                total_bytes: total,
                final_w_len,
                final_lb: tail_lb,
            }
        }
        None => Schedule {
            levels: Vec::new(),
            tail_bytes: 0,
            total_bytes: usize::MAX,
            final_w_len: 0,
            final_lb: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onehot_32_produces_schedule() {
        let opts = PlannerOptions::new(1, 32);
        let sched = run_universal_planner(&opts);
        assert!(!sched.levels.is_empty());
        assert!(sched.total_bytes < 100_000);
    }

    #[test]
    fn onehot_32_optimal() {
        let opts = PlannerOptions::new(1, 32);
        let sched = run_universal_planner(&opts);
        // Improved over Python's 54,116 B (which had depth-bound memo bug)
        assert_eq!(sched.total_bytes, 52_548, "onehot nv=32");
    }

    #[test]
    fn full_32_optimal() {
        let opts = PlannerOptions::new(128, 32);
        let sched = run_universal_planner(&opts);
        assert_eq!(sched.total_bytes, 55_572, "full nv=32");
    }

    #[test]
    fn full_25_optimal() {
        let opts = PlannerOptions::new(128, 25);
        let sched = run_universal_planner(&opts);
        assert_eq!(sched.total_bytes, 52_428, "full nv=25");
    }

    #[test]
    fn onehot_44_optimal() {
        let opts = PlannerOptions::new(1, 44);
        let sched = run_universal_planner(&opts);
        assert_eq!(sched.total_bytes, 59_236, "onehot nv=44");
    }
}
