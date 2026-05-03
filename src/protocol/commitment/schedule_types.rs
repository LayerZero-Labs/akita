//! Runtime schedule shapes shared by configs, prover, verifier, and planner.

use crate::protocol::commitment::digit_math::compute_num_digits_fold_with_claims;
use crate::protocol::commitment::{HachiPlannedStep, HachiSchedulePlan};
use crate::protocol::config::CommitmentConfig;
use akita_types::DirectWitnessShape;
use akita_types::LevelParams;

/// Parameters for one fold level in the computed schedule.
#[derive(Clone, Debug)]
pub struct FoldStep {
    /// Unified level parameters (ring dimension, Ajtai keys, block geometry,
    /// digit depths, challenge config).
    pub params: LevelParams,
    /// Witness length entering this level.
    pub current_w_len: usize,
    /// Per-polynomial fold digits (`num_claims=1`). Equal to
    /// `params.num_digits_fold` for singleton schedules; smaller for batched
    /// roots where the layout uses the batched bound.
    pub delta_fold_per_poly: usize,
    /// Ring-element count in the witness after ring-switching.
    pub w_ring: usize,
    /// Witness length leaving this level.
    pub next_w_len: usize,
    /// Proof bytes for this level.
    pub level_bytes: usize,
}

/// Terminal direct-send step.
#[derive(Clone, Debug)]
pub struct DirectStep {
    /// Witness length entering the direct step.
    pub current_w_len: usize,
    /// Packed bits per witness element.
    pub bits_per_elem: u32,
    /// Direct witness bytes.
    pub direct_bytes: usize,
}

/// A single step in the schedule.
#[derive(Clone, Debug)]
pub enum Step {
    /// Fold through one recursive level.
    Fold(FoldStep),
    /// Send the terminal witness directly.
    Direct(DirectStep),
}

/// Complete schedule with step-by-step parameters.
#[derive(Clone, Debug)]
pub struct Schedule {
    /// Ordered proof schedule steps.
    pub steps: Vec<Step>,
    /// Exact total proof bytes for the schedule.
    pub total_bytes: usize,
}

/// Aggregate witness-shape inputs that determine root-level sizing.
///
/// The root-level witness ring count is, for any `(K, G, P)`:
///
/// ```text
///   W(lp; K, G, P) = K · 2^r · δ_open                       // |ŵ|
///                  + K · 2^r · n_A · δ_open                 // |t̂|
///                  + P · 2^m · δ_commit · δ_fold            // |z_pre|
///                  + (n_D + n_B·G + P + 1 + n_A) · δ_R(b)   // |r|
/// ```
///
/// Singleton openings are simply the `K = G = P = 1` special case of this
/// formula; the planner does not need to branch on "batched vs non-batched"
/// — only on this aggregate shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WitnessShape {
    /// `K` — total number of polynomial claims (drives `|ŵ|`, `|t̂|`).
    pub num_claims: usize,
    /// `G` — number of commitment groups (drives the `n_B·G` term in `|r|).
    pub num_commitment_groups: usize,
    /// `P` — number of distinct opening points (drives `|z_pre|` and the
    /// `+P` term in `|r|).
    pub num_points: usize,
}

impl WitnessShape {
    /// Build a witness shape from explicit `(K, G, P)`.
    pub const fn new(num_claims: usize, num_commitment_groups: usize, num_points: usize) -> Self {
        Self {
            num_claims,
            num_commitment_groups,
            num_points,
        }
    }

    /// Singleton shape: one polynomial, one group, one point.
    pub const fn singleton() -> Self {
        Self {
            num_claims: 1,
            num_commitment_groups: 1,
            num_points: 1,
        }
    }

    /// Build a witness shape from per-group opening-point counts.
    ///
    /// Interprets `points_per_group[g]` as the number of distinct opening
    /// points associated with commitment group `g`. The aggregates are:
    ///
    /// * `G = points_per_group.len()`
    /// * `P = sum(points_per_group)`  (treats each group's points as
    ///   distinct from other groups')
    /// * `K = sum(points_per_group)`  (one claim per `(group, point)` pair)
    pub fn from_points_per_group(points_per_group: &[usize]) -> Self {
        let num_commitment_groups = points_per_group.len();
        let total_points: usize = points_per_group.iter().copied().sum();
        Self {
            num_claims: total_points,
            num_commitment_groups,
            num_points: total_points,
        }
    }
}

/// Translate an offline [`HachiSchedulePlan`] into the runtime [`Schedule`]
/// format.
///
/// The offline schedule tables in `src/protocol/commitment/generated/*` are
/// the authoritative source of pre-computed optimal schedules for shipped
/// `(Cfg, max_num_vars, WitnessShape)` cases. Runtime config code converts each
/// entry into a [`HachiSchedulePlan`] through `Cfg::schedule_plan`, then maps it
/// into this shared schedule representation.
pub(crate) fn schedule_from_plan<Cfg: CommitmentConfig>(plan: &HachiSchedulePlan) -> Schedule {
    let field_bits_u32 = Cfg::decomposition().field_bits();
    let mut steps = Vec::with_capacity(plan.steps.len());
    for step in &plan.steps {
        match step {
            HachiPlannedStep::Fold(level) => {
                let lp = level.lp.clone();
                let delta_fold_per_poly = compute_num_digits_fold_with_claims(
                    lp.r_vars,
                    lp.challenge_l1_mass(),
                    lp.log_basis,
                    1,
                );
                let ring_dim = lp.ring_dimension;
                let next_w_len = level.next_inputs.current_w_len;
                let w_ring = next_w_len / ring_dim;
                steps.push(Step::Fold(FoldStep {
                    params: lp,
                    current_w_len: level.inputs.current_w_len,
                    delta_fold_per_poly,
                    w_ring,
                    next_w_len,
                    level_bytes: level.level_bytes,
                }));
            }
            HachiPlannedStep::Direct(direct) => {
                let bits_per_elem = match direct.witness_shape {
                    DirectWitnessShape::PackedDigits((_, bits)) => bits,
                    DirectWitnessShape::FieldElements(_) => field_bits_u32,
                };
                steps.push(Step::Direct(DirectStep {
                    current_w_len: direct.state.current_w_len,
                    bits_per_elem,
                    direct_bytes: direct.direct_bytes,
                }));
            }
        }
    }
    Schedule {
        steps,
        total_bytes: plan.exact_proof_bytes,
    }
}
