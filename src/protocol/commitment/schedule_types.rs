//! Runtime conversion from planned schedules to shared schedule shapes.

use crate::protocol::commitment::{HachiPlannedStep, HachiSchedulePlan};
use crate::protocol::config::CommitmentConfig;
use akita_types::digit_math::compute_num_digits_fold_with_claims;
use akita_types::{DirectStep, DirectWitnessShape, FoldStep, Schedule, Step};

/// Translate an offline [`HachiSchedulePlan`] into the runtime [`Schedule`]
/// format.
///
/// The offline schedule tables are the authoritative source of pre-computed
/// optimal schedules for shipped `(Cfg, max_num_vars, WitnessShape)` cases.
/// Runtime config code converts each entry into a [`HachiSchedulePlan`] through
/// `Cfg::schedule_plan`, then maps it into the shared schedule representation.
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
