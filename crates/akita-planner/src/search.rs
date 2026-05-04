use std::collections::HashMap;

use super::proof_size::{
    elem_bytes, packed_digits_bytes, ring_vec_bytes, stage1_bytes_optimized, sumcheck_bytes,
    sumcheck_rounds, FIELD_BITS,
};
use super::sis_security::{ceil_supported_collision, min_rank_for_secure_width};
use akita_types::layout::digit_math::{
    compute_num_digits_fold_with_claims, num_digits_for_bound, optimal_m_r_split,
};

// Ring configurations.

/// A candidate ring configuration for one folding level.
#[derive(Clone, Debug)]
pub struct RingConfig {
    pub d: u32,
    pub n_a: u32,
    pub challenge_l1_mass: usize,
    pub max_abs_challenge_coeff: u32,
    pub label: &'static str,
}

/// All 7 ring configurations available to the planner.
pub const ALL_RING_CONFIGS: &[RingConfig] = &[
    RingConfig {
        d: 128,
        n_a: 1,
        challenge_l1_mass: 31,
        max_abs_challenge_coeff: 1,
        label: "D128-na1",
    },
    RingConfig {
        d: 128,
        n_a: 2,
        challenge_l1_mass: 31,
        max_abs_challenge_coeff: 1,
        label: "D128-na2",
    },
    RingConfig {
        d: 64,
        n_a: 1,
        challenge_l1_mass: 54,
        max_abs_challenge_coeff: 2,
        label: "D64-na1",
    },
    RingConfig {
        d: 64,
        n_a: 2,
        challenge_l1_mass: 54,
        max_abs_challenge_coeff: 2,
        label: "D64-na2",
    },
    RingConfig {
        d: 32,
        n_a: 1,
        challenge_l1_mass: 256,
        max_abs_challenge_coeff: 8,
        label: "D32-na1",
    },
    RingConfig {
        d: 32,
        n_a: 2,
        challenge_l1_mass: 256,
        max_abs_challenge_coeff: 8,
        label: "D32-na2",
    },
    RingConfig {
        d: 32,
        n_a: 3,
        challenge_l1_mass: 256,
        max_abs_challenge_coeff: 8,
        label: "D32-na3",
    },
];

const MIN_LB: u32 = 2;
const MAX_LB: u32 = 6;

// Witness computation.

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

struct WitnessArgs {
    m_vars: usize,
    r_vars: usize,
    log_basis: u32,
    log_cb: u32,
    nb: u32,
    nd: u32,
    num_ring_actual: usize,
    tight_zpre: bool,
}

fn compute_level_witness(cfg: &RingConfig, a: &WitnessArgs) -> LevelComputation {
    let WitnessArgs {
        m_vars,
        r_vars,
        log_basis,
        log_cb,
        nb,
        nd,
        num_ring_actual,
        tight_zpre,
    } = *a;
    let d = cfg.d;

    let open_bound = if log_cb < 128 { 128 } else { log_cb };
    let delta_open = num_digits_for_bound(open_bound, log_basis);
    let delta_commit = num_digits_for_bound(log_cb, log_basis);
    let delta_fold =
        compute_num_digits_fold_with_claims(r_vars, cfg.challenge_l1_mass, log_basis, 1);

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
    let r_ct = m_row * num_digits_for_bound(128, log_basis);
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

// Output types.

/// Serialized direct-witness shape chosen at a terminal step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DirectWitnessShape {
    PackedDigits {
        num_elems: usize,
        bits_per_elem: u32,
    },
    FieldElements {
        num_elems: usize,
    },
}

impl DirectWitnessShape {
    pub fn witness_bytes(&self) -> usize {
        match self {
            Self::PackedDigits {
                num_elems,
                bits_per_elem,
            } => packed_digits_bytes(*num_elems, *bits_per_elem),
            Self::FieldElements { num_elems } => num_elems.saturating_mul(elem_bytes()),
        }
    }
}

/// One planned folding step.
#[derive(Clone, Debug)]
pub struct PlannedFoldStep {
    pub current_w_len: usize,
    pub d: u32,
    pub lb: u32,
    pub challenge_l1_mass: usize,
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

/// One planned direct terminal step.
#[derive(Clone, Debug)]
pub struct PlannedDirectStep {
    pub current_w_len: usize,
    pub witness_shape: DirectWitnessShape,
    pub entry_d: Option<u32>,
    pub entry_nb: Option<u32>,
    pub direct_bytes: usize,
    pub total_bytes: usize,
}

/// One planned schedule step.
#[derive(Clone, Debug)]
pub enum PlannedStep {
    Fold(PlannedFoldStep),
    Direct(PlannedDirectStep),
}

/// Complete planned schedule.
#[derive(Clone, Debug)]
pub struct Schedule {
    pub steps: Vec<PlannedStep>,
    pub total_bytes: usize,
}

impl Schedule {
    pub fn fold_steps(&self) -> impl Iterator<Item = &PlannedFoldStep> {
        self.steps.iter().filter_map(|step| match step {
            PlannedStep::Fold(level) => Some(level),
            PlannedStep::Direct(_) => None,
        })
    }

    pub fn num_fold_levels(&self) -> usize {
        self.fold_steps().count()
    }

    pub fn direct_step(&self) -> Option<&PlannedDirectStep> {
        self.steps.iter().find_map(|step| match step {
            PlannedStep::Direct(direct) => Some(direct),
            PlannedStep::Fold(_) => None,
        })
    }

    pub fn direct_bytes(&self) -> usize {
        self.direct_step().map_or(0, |step| step.total_bytes)
    }

    pub fn final_w_len(&self) -> usize {
        self.direct_step().map_or(0, |step| step.current_w_len)
    }

    pub fn final_lb(&self) -> Option<u32> {
        match self.direct_step().map(|step| &step.witness_shape) {
            Some(DirectWitnessShape::PackedDigits { bits_per_elem, .. }) => Some(*bits_per_elem),
            _ => None,
        }
    }
}

// Planner options.

/// Configuration knobs for the planner.
#[derive(Clone)]
pub struct PlannerOptions {
    pub log_commit_bound: u32,
    pub max_num_vars: usize,
    pub ring_configs: Vec<RingConfig>,
    pub opt_sumcheck: bool,
    pub monotone_d: bool,
    pub tight_zpre: bool,
}

impl PlannerOptions {
    pub fn new(log_commit_bound: u32, max_num_vars: usize) -> Self {
        Self {
            log_commit_bound,
            max_num_vars,
            ring_configs: ALL_RING_CONFIGS.to_vec(),
            opt_sumcheck: true,
            monotone_d: true,
            tight_zpre: true,
        }
    }

    pub fn with_tight_zpre(mut self, v: bool) -> Self {
        self.tight_zpre = v;
        self
    }
}

// Planner internals.

type MemoKey = (usize, u32, u32); // (w_len, cur_D, current_lb)

#[derive(Clone)]
struct BestSuffix {
    cost: usize,
    steps: Vec<PlannedStep>,
}

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

    /// Return the supported SIS collision bucket used to size the `A` role.
    ///
    /// For the root onehot level the raw digit difference is bounded by `2`;
    /// otherwise the raw collision bound is the balanced-digit width
    /// `2^lb - 1`. The `A` reduction pays one extra challenge factor, so we
    /// scale by the maximum absolute coefficient in the level's stage-1
    /// challenge family before rounding up to the next precomputed SIS bucket.
    fn a_role_sis_collision_bucket(
        &self,
        cfg: &RingConfig,
        level: usize,
        log_cb: u32,
        lb: u32,
    ) -> Option<u32> {
        let raw_collision = if level == 0 && log_cb == 1 {
            2
        } else {
            (1u32 << lb) - 1
        };
        ceil_supported_collision(cfg.d, raw_collision * cfg.max_abs_challenge_coeff)
    }

    /// Try a specific (cfg, lb, m, r) combination at a given level/witness.
    #[allow(clippy::too_many_arguments)]
    fn try_level_mr(
        &self,
        cfg: &RingConfig,
        level: usize,
        w_len: usize,
        lb: u32,
        log_cb: u32,
        m_vars: usize,
        r_vars: usize,
    ) -> Option<(usize, LevelComputation, u32, u32)> {
        let num_ring = if level > 0 {
            w_len / cfg.d as usize
        } else {
            let alpha = cfg.d.trailing_zeros() as usize;
            if self.opts.max_num_vars <= alpha {
                return None;
            }
            1usize << (self.opts.max_num_vars - alpha)
        };

        // Bit-width of one input element: 128-bit field elements at the root,
        // lb-bit packed digits at recursive levels. The product
        // w_len * input_elem_bits must fit u64, which holds for nv < 57.
        let input_elem_bits: u64 = if level == 0 {
            FIELD_BITS as u64
        } else {
            lb as u64
        };

        let base_args = WitnessArgs {
            m_vars,
            r_vars,
            log_basis: lb,
            log_cb,
            nb: 1,
            nd: 1,
            num_ring_actual: num_ring,
            tight_zpre: self.opts.tight_zpre,
        };
        let lc = compute_level_witness(cfg, &base_args);
        if (lc.next_w_len as u64) * (lb as u64) >= (w_len as u64) * input_elem_bits {
            return None;
        }

        let inner_width = if self.opts.tight_zpre {
            num_ring.div_ceil(1usize << r_vars) * lc.delta_commit
        } else {
            (1usize << m_vars) * lc.delta_commit
        };
        let a_sis_collision = self.a_role_sis_collision_bucket(cfg, level, log_cb, lb)?;
        let na_needed =
            min_rank_for_secure_width(cfg.d, a_sis_collision, u64::try_from(inner_width).ok()?)?;
        if na_needed > cfg.n_a {
            return None;
        }

        let bd_collision = (1u32 << lb) - 1;
        let outer = cfg.n_a as usize * lc.delta_open * (1usize << r_vars);
        let d_mat = lc.delta_open * (1usize << r_vars);
        let nb = min_rank_for_secure_width(cfg.d, bd_collision, u64::try_from(outer).ok()?)?;
        let nd = min_rank_for_secure_width(cfg.d, bd_collision, u64::try_from(d_mat).ok()?)?;

        let lc = compute_level_witness(
            cfg,
            &WitnessArgs {
                nb,
                nd,
                ..base_args
            },
        );
        if (lc.next_w_len as u64) * (lb as u64) >= (w_len as u64) * input_elem_bits {
            return None;
        }
        let prefix = self.level_prefix(cfg, lb, lc.rounds, nd);
        Some((prefix, lc, nb, nd))
    }

    /// Try a level using the locally optimal (m, r) from `optimal_m_r_split`.
    fn try_level(
        &self,
        cfg: &RingConfig,
        level: usize,
        w_len: usize,
        lb: u32,
        log_cb: u32,
    ) -> Option<(usize, LevelComputation, u32, u32)> {
        let d = cfg.d;
        let alpha = d.trailing_zeros() as usize;

        let (reduced_vars, num_ring) = if level == 0 {
            if self.opts.max_num_vars <= alpha {
                return None;
            }
            let rv = self.opts.max_num_vars - alpha;
            (rv, 1usize << rv)
        } else {
            let nr = w_len / d as usize;
            let rv = nr.next_power_of_two().trailing_zeros() as usize;
            (rv, nr)
        };

        let nr_arg = if self.opts.tight_zpre { num_ring } else { 0 };
        let (m, r) = optimal_m_r_split(
            cfg.n_a,
            cfg.challenge_l1_mass,
            log_cb,
            lb,
            reduced_vars,
            nr_arg,
        );
        self.try_level_mr(cfg, level, w_len, lb, log_cb, m, r)
    }

    fn tail_entry_nb(&self, w_len: usize, d: u32, tail_lb: u32) -> Option<u32> {
        let ring_elems = w_len.div_ceil(d as usize);
        min_rank_for_secure_width(d, (1u32 << tail_lb) - 1, u64::try_from(ring_elems).ok()?)
    }

    fn best_from(&mut self, w_len: usize, cur_d: u32, current_lb: u32) -> BestSuffix {
        let key = (w_len, cur_d, current_lb);
        if let Some(existing) = self.memo.get(&key) {
            return existing.clone();
        }

        let mut best = BestSuffix {
            cost: usize::MAX,
            steps: Vec::new(),
        };

        if let Some(tnb) = self.tail_entry_nb(w_len, cur_d, current_lb) {
            let witness_shape = DirectWitnessShape::PackedDigits {
                num_elems: w_len,
                bits_per_elem: current_lb,
            };
            let direct_bytes = witness_shape.witness_bytes();
            let total_bytes = ring_vec_bytes(tnb as usize, cur_d) + direct_bytes;
            best = BestSuffix {
                cost: total_bytes,
                steps: vec![PlannedStep::Direct(PlannedDirectStep {
                    current_w_len: w_len,
                    witness_shape,
                    entry_d: Some(cur_d),
                    entry_nb: Some(tnb),
                    direct_bytes,
                    total_bytes,
                })],
            };
        }

        let cfgs: Vec<RingConfig> = self.cfgs_for_d(cur_d).cloned().collect();
        let unique_ds = self.unique_ds.clone();
        let monotone_d = self.opts.monotone_d;

        for cfg in &cfgs {
            for lb in MIN_LB..=MAX_LB {
                let result = self.try_level(cfg, 1, w_len, lb, current_lb);
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
                        let mut steps = Vec::with_capacity(1 + suffix.steps.len());
                        steps.push(PlannedStep::Fold(PlannedFoldStep {
                            current_w_len: w_len,
                            d: cfg.d,
                            lb,
                            challenge_l1_mass: cfg.challenge_l1_mass,
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
                        }));
                        steps.extend_from_slice(&suffix.steps);
                        best = BestSuffix { cost: total, steps };
                    }
                }
            }
        }

        self.memo.insert(key, best.clone());
        best
    }
}

// Public API.

/// Run the universal planner.
///
/// The root level enumerates all feasible (m, r) splits to find the globally
/// optimal starting point. Recursive levels use the corrected
/// `optimal_m_r_split` heuristic (which matches the actual witness
/// construction) and rely on the DP across configs, lb, and D-transitions
/// for global optimality.
///
/// Header stripping (optimization #5) is modeled: the total does NOT include
/// the 4-byte `num_levels` wrapper that the current serialization adds.
pub fn run_universal_planner(opts: &PlannerOptions) -> Schedule {
    let mut planner = Planner::new(opts.clone());

    let root_w_len = 1usize << opts.max_num_vars;
    let root_direct_shape = DirectWitnessShape::FieldElements {
        num_elems: root_w_len,
    };
    let root_direct_bytes = root_direct_shape.witness_bytes();
    let mut overall_best = Some((
        root_direct_bytes,
        vec![PlannedStep::Direct(PlannedDirectStep {
            current_w_len: root_w_len,
            witness_shape: root_direct_shape,
            entry_d: None,
            entry_nb: None,
            direct_bytes: root_direct_bytes,
            total_bytes: root_direct_bytes,
        })],
    ));

    let all_cfgs: Vec<RingConfig> = opts.ring_configs.to_vec();
    let unique_ds = planner.unique_ds.clone();

    for root_cfg in &all_cfgs {
        let alpha = root_cfg.d.trailing_zeros() as usize;
        if opts.max_num_vars <= alpha {
            continue;
        }
        let rv = opts.max_num_vars - alpha;
        let num_ring = 1usize << rv;

        for root_lb in MIN_LB..=MAX_LB {
            // Enumerate all (m, r) splits at the root for global optimality.
            for root_r in 1..rv {
                let root_m = rv - root_r;
                let nr_arg = if opts.tight_zpre { num_ring } else { 0 };
                // Early pruning: skip (m,r) splits whose local witness cost
                // is far from optimal. This avoids trying clearly bad splits
                // at the root level.
                let (_, opt_r) = optimal_m_r_split(
                    root_cfg.n_a,
                    root_cfg.challenge_l1_mass,
                    opts.log_commit_bound,
                    root_lb,
                    rv,
                    nr_arg,
                );
                if root_r.abs_diff(opt_r) > 4 {
                    continue;
                }

                let result = planner.try_level_mr(
                    root_cfg,
                    0,
                    root_w_len,
                    root_lb,
                    opts.log_commit_bound,
                    root_m,
                    root_r,
                );
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
                    let root_entry_commit = ring_vec_bytes(root_nb as usize, root_cfg.d);
                    let total = root_entry_commit + root_prefix + suffix.cost;
                    let is_better = overall_best
                        .as_ref()
                        .is_none_or(|(best_total, _)| total < *best_total);
                    if is_better {
                        let mut steps = Vec::with_capacity(1 + suffix.steps.len());
                        steps.push(PlannedStep::Fold(PlannedFoldStep {
                            current_w_len: root_w_len,
                            d: root_cfg.d,
                            lb: root_lb,
                            challenge_l1_mass: root_cfg.challenge_l1_mass,
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
                            level_bytes: root_entry_commit + root_prefix,
                            label: root_cfg.label,
                        }));
                        steps.extend_from_slice(&suffix.steps);
                        overall_best = Some((total, steps));
                    }
                }
            }
        }
    }

    match overall_best {
        Some((total, steps)) => Schedule {
            steps,
            total_bytes: total,
        },
        None => Schedule {
            steps: Vec::new(),
            total_bytes: usize::MAX,
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
        assert!(sched.num_fold_levels() > 0);
        assert!(sched.total_bytes < 100_000);
    }

    #[test]
    fn onehot_32_beats_baseline() {
        let opts = PlannerOptions::new(1, 32);
        let sched = run_universal_planner(&opts);
        assert!(
            sched.total_bytes < 97_277,
            "onehot nv=32: {} should stay below the D=64 baseline",
            sched.total_bytes
        );
    }

    #[test]
    fn full_32_beats_baseline() {
        let opts = PlannerOptions::new(128, 32);
        let sched = run_universal_planner(&opts);
        assert!(
            sched.total_bytes < 170_637,
            "full nv=32: {} should stay below the D=128 baseline",
            sched.total_bytes
        );
    }

    #[test]
    fn full_25_produces_schedule() {
        let opts = PlannerOptions::new(128, 25);
        let sched = run_universal_planner(&opts);
        assert!(sched.num_fold_levels() > 0);
        assert!(sched.total_bytes < 166_613);
    }

    #[test]
    fn d128_only_configs_produce_schedule() {
        let mut opts = PlannerOptions::new(128, 25);
        let d128_cfgs: Vec<_> = ALL_RING_CONFIGS
            .iter()
            .filter(|cfg| cfg.d == 128)
            .cloned()
            .collect();
        assert!(!d128_cfgs.is_empty());
        opts.ring_configs = d128_cfgs;
        let sched = run_universal_planner(&opts);
        assert!(sched.num_fold_levels() > 0);
        assert!(sched.fold_steps().all(|level| level.d == 128));
    }

    #[test]
    fn onehot_44_produces_schedule() {
        let opts = PlannerOptions::new(1, 44);
        let sched = run_universal_planner(&opts);
        assert!(sched.num_fold_levels() > 0);
        assert!(sched.total_bytes < 106_533);
    }

    #[test]
    fn no_header_wrapper() {
        let opts = PlannerOptions::new(1, 20);
        let sched = run_universal_planner(&opts);
        let level_sum: usize = sched.fold_steps().map(|l| l.level_bytes).sum();
        let direct = sched
            .direct_step()
            .expect("schedule should end in direct step");
        let overhead = sched.total_bytes - (level_sum + direct.direct_bytes);
        let matched_tail_entry = match (&direct.entry_d, &direct.entry_nb) {
            (Some(d), Some(rank)) => overhead == ring_vec_bytes(*rank as usize, *d),
            _ => overhead == 0,
        };
        assert!(
            matched_tail_entry,
            "tail overhead {overhead} should be one valid entry commitment"
        );
    }
}
