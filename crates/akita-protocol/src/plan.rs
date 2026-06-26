//! Per-level protocol plan.
//!
//! [`plan_level`] is a pure function of `(const D, LevelParams, next_w_len,
//! ProtocolGates, LevelRole)` that both the prover and the verifier call to
//! obtain the identical ordered sumcheck schedule, so Fiat-Shamir ordering,
//! batching, and per-instance proof format agree by construction. The plan is
//! field-free: it describes schedule structure only, and the evaluation field
//! is chosen when a descriptor is later evaluated.
//!
//! Today [`plan_level`] emits only the baseline stage-2 schedule: one regular
//! instance, fused at intermediate levels and relation-only at the terminal
//! cleartext-witness level ([`LevelRole`]). Optional gates select future
//! extensions (y-ring trace fusion, folded-witness L2 certificate, setup-claim
//! offloading); when a gate is enabled but not yet implemented, planning
//! fails rather than emitting an incomplete schedule. [`BatchingScheme`] is
//! the single place cross-instance gamma powers are allocated so extensions
//! cannot collide on the same exponent.

use akita_field::AkitaError;
use akita_sumcheck::descriptor::{ClaimSlot, InstanceKind, SumcheckInstanceDescriptor};
use akita_types::LevelParams;

use crate::ids::{AkitaChallengeId, AkitaOpeningId, AkitaPublicId};
use crate::stage2::stage2_descriptor;

/// Feature gates that select which instances a level's plan contains.
///
/// `plan_level` is a pure function of these gates plus the level parameters and
/// the schedule's `next_w_len`, so the prover and verifier obtain identical
/// plans for identical inputs.
///
/// Each non-`zk` gate is an extension point. Today `plan_level` rejects an
/// enabled gate until that extension wires its instances and transcript events
/// into this plan. Gates live here so gamma-power batching stays centralized
/// in [`BatchingScheme`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProtocolGates {
    /// Fuse the y-ring trace term into the stage-2 instance.
    pub trace: bool,
    /// Emit folded-witness L2 certificate sumcheck instances for this level.
    pub l2_certificate: bool,
    /// Append a stage-3 setup product-sumcheck instance (setup-claim offloading).
    pub setup_offload: bool,
    /// Use ZK committed-round sinks. Does not change the instance list or the
    /// transcript schedule emitted here; only the prover's sink differs.
    pub zk: bool,
}

/// Whether a level folds into a further committed level or is the cleartext tail.
///
/// This selects which stage-2 descriptor a level emits. It mirrors the
/// `Intermediate`/`Terminal` split the prover (`prove_terminal_*` /
/// `AkitaProofStep::Terminal`) and verifier (`RootStageInput::Terminal`)
/// already encode: the caller knows its position in the fold schedule, so the
/// role is an explicit input rather than something `plan_level` derives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelRole {
    /// A non-final fold. The next witness stays committed, so the carried
    /// norm/range virtual claim is fused into stage 2 (the fused descriptor).
    Intermediate,
    /// The cleartext-witness tail. The witness is opened in the clear, so there
    /// is no carried virtual claim and stage 2 is relation-only.
    Terminal,
}

/// How gamma powers are allocated across the instances batched in one stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchingScheme {
    /// One instance proven on its own; there is no cross-instance batching
    /// coefficient. Intra-instance fusion (for example stage 2's virtual
    /// sub-claim weighted by gamma) is carried by
    /// [`akita_sumcheck::descriptor::SubClaim::weight`] on that
    /// instance's summand, not by this scheme.
    Standalone,
    /// Several regular instances linearly combined into one batched sumcheck;
    /// `gamma_powers[i]` is the exponent of the batching challenge applied to
    /// instance `i`.
    GammaPowers {
        /// Per-instance gamma exponent, allocated centrally.
        gamma_powers: Vec<usize>,
    },
}

/// A group of instances proven together in one (possibly batched) sumcheck.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagePlan<O, P, C> {
    /// Instances batched together in this stage.
    pub instances: Vec<SumcheckInstanceDescriptor<O, P, C>>,
    /// How gamma powers are allocated across `instances`.
    pub batching: BatchingScheme,
}

impl<O, P, C> StagePlan<O, P, C> {
    /// Whether the eq-factored proof-size optimization is retained for the
    /// instance at `index`.
    ///
    /// The eq-factored wire format (sending the inner `q` with its linear term
    /// omitted) is valid only for an [`InstanceKind::EqFactored`] instance
    /// proven standalone. Any batching linearly combines round polynomials and
    /// the combined polynomial no longer shares a single eq factor, so a
    /// batched eq-factored instance demotes to the regular compressed format
    /// (the prover still computes the eq factor with Gruen split-eq). An
    /// out-of-range `index` is reported as `false` rather than panicking.
    pub fn retains_eq_factored_format(&self, index: usize) -> bool {
        match self.instances.get(index) {
            Some(instance) => {
                matches!(instance.kind, InstanceKind::EqFactored)
                    && self.instances.len() == 1
                    && matches!(self.batching, BatchingScheme::Standalone)
            }
            None => false,
        }
    }
}

/// The carried-opening claims handed from this level to the next fold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CarriedOpeningPlan {
    /// A single carried opening claim (the common case).
    Singleton,
    /// Multiple carried opening claims (setup-claim offloading).
    List {
        /// Number of carried opening claims.
        count: usize,
    },
}

/// One ordered Fiat-Shamir event in a level's transcript schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptEvent {
    /// Absorb a chained input claim before a stage's rounds.
    AbsorbInputClaim(ClaimSlot),
    /// Run `num_rounds` boolean-hypercube rounds: per round append the round
    /// polynomial and squeeze the round challenge.
    SumcheckRounds {
        /// Number of rounds.
        num_rounds: usize,
    },
    /// Absorb a chained output claim after a stage's rounds.
    AbsorbOutputClaim(ClaimSlot),
}

/// The ordered Fiat-Shamir schedule for a level, derived from its stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptSchedule {
    /// Ordered events; identical on the prover and verifier.
    pub events: Vec<TranscriptEvent>,
}

/// The full per-level protocol plan: stages, carried openings, and schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelProtocolPlan<O, P, C> {
    /// Ordered stages, each a batched sumcheck.
    pub stages: Vec<StagePlan<O, P, C>>,
    /// Carried opening claims handed to the next fold.
    pub carried_openings: CarriedOpeningPlan,
    /// Ordered Fiat-Shamir schedule.
    pub transcript_schedule: TranscriptSchedule,
}

/// Derive the per-level protocol plan from level parameters and feature gates.
///
/// Field-free: the plan describes only the schedule's *structure* (round
/// counts, batching, instance kinds, claim chaining, transcript order). The
/// evaluation field is chosen later, when the verifier evaluates a descriptor.
/// The ring dimension is the const `D` the whole prove/verify stack is
/// dispatched at; `params.ring_dimension` must agree (mirroring the verifier's
/// `lp.ring_dimension == D` boundary check), so the plan cannot derive a round
/// count against a ring dimension the rest of the call is not using.
///
/// Pure and panic-free: identical inputs yield identical plans on both sides.
/// This stub emits the baseline stage-2 schedule (one regular instance: fused
/// at [`LevelRole::Intermediate`], relation-only at [`LevelRole::Terminal`])
/// and rejects unimplemented trace, L2-certificate, and setup-offload gates.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidInput`] for a not-yet-supported gate and
/// [`AkitaError::InvalidSetup`] when `D` is not a power of two, when
/// `params.ring_dimension` disagrees with `D`, or when `next_w_len` is not a
/// positive multiple of `D`.
pub fn plan_level<const D: usize>(
    params: &LevelParams,
    next_w_len: usize,
    gates: &ProtocolGates,
    role: LevelRole,
) -> Result<LevelProtocolPlan<AkitaOpeningId, AkitaPublicId, AkitaChallengeId>, AkitaError> {
    if gates.trace || gates.l2_certificate || gates.setup_offload {
        return Err(AkitaError::InvalidInput(
            "plan_level currently emits only the baseline stage-2 schedule; enable \
             trace, L2-certificate, or setup-offload only after those extensions \
             are implemented"
                .to_string(),
        ));
    }

    let num_rounds = stage2_num_rounds::<D>(params, next_w_len)?;
    let descriptor = stage2_descriptor(num_rounds, role);
    let input_claim = descriptor.input_claim;
    let output_claim = descriptor.output_claim;

    let stage = StagePlan {
        instances: vec![descriptor],
        batching: BatchingScheme::Standalone,
    };

    let transcript_schedule = TranscriptSchedule {
        events: vec![
            TranscriptEvent::AbsorbInputClaim(input_claim),
            TranscriptEvent::SumcheckRounds { num_rounds },
            TranscriptEvent::AbsorbOutputClaim(output_claim),
        ],
    };

    Ok(LevelProtocolPlan {
        stages: vec![stage],
        carried_openings: CarriedOpeningPlan::Singleton,
        transcript_schedule,
    })
}

/// Stage-2 round count for a level, matching prover/verifier sumcheck drivers.
///
/// `col_bits` is `log2` of the next power of two of `next_w_len / D`, and
/// `ring_bits` is `log2(D)`, same formula as [`sumcheck_rounds`] but with
/// overflow-checked column padding so verifier-reachable planning cannot panic.
/// Validates `D` against `params.ring_dimension` so the plan's round count
/// cannot diverge from the const ring dimension the rest of the call uses.
fn stage2_num_rounds<const D: usize>(
    params: &LevelParams,
    next_w_len: usize,
) -> Result<usize, AkitaError> {
    if D == 0 || !D.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "stage-2 plan requires a power-of-two ring dimension".to_string(),
        ));
    }
    if params.ring_dimension != D {
        return Err(AkitaError::InvalidSetup(
            "stage-2 plan ring dimension does not match dispatched D".to_string(),
        ));
    }
    if next_w_len == 0 || !next_w_len.is_multiple_of(D) {
        return Err(AkitaError::InvalidSetup(
            "stage-2 plan requires next_w_len to be a positive multiple of the ring dimension"
                .to_string(),
        ));
    }
    let ring_bits = D.trailing_zeros() as usize;
    let num_ring_elems = next_w_len / D;
    let col_bits = num_ring_elems
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("stage-2 plan column count overflow".to_string()))?
        .trailing_zeros() as usize;
    Ok(col_bits + ring_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_sumcheck::descriptor::Source;
    use akita_types::{sumcheck_rounds, SisModulusFamily};

    // Dispatched ring dimension for the sample level (matches `sample_params`).
    const D: usize = 64;

    fn sample_params() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
            D,
            3,
            2,
            4,
            3,
            SparseChallengeConfig::Uniform {
                weight: 3,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(4, 2, 2, 2, 0)
        .expect("valid layout")
    }

    fn sample_next_w_len(params: &LevelParams) -> usize {
        // Same witness-shape convention as schedule/proof-size tests.
        params.ring_dimension * 8
    }

    #[test]
    fn plan_level_is_deterministic_for_fixed_inputs() {
        let params = sample_params();
        let next_w_len = sample_next_w_len(&params);
        let gates = ProtocolGates::default();
        let first =
            plan_level::<D>(&params, next_w_len, &gates, LevelRole::Intermediate).expect("plan");
        let second =
            plan_level::<D>(&params, next_w_len, &gates, LevelRole::Intermediate).expect("plan");
        assert_eq!(first, second);
    }

    #[test]
    fn plan_level_intermediate_emits_single_fused_stage2_instance() {
        let params = sample_params();
        let next_w_len = sample_next_w_len(&params);
        let expected_rounds = sumcheck_rounds(D, next_w_len);
        let plan = plan_level::<D>(
            &params,
            next_w_len,
            &ProtocolGates::default(),
            LevelRole::Intermediate,
        )
        .expect("plan");

        assert_eq!(plan.stages.len(), 1);
        let stage = &plan.stages[0];
        assert_eq!(stage.instances.len(), 1);
        assert_eq!(stage.instances[0].kind, InstanceKind::Regular);
        assert_eq!(stage.instances[0].num_rounds, expected_rounds);
        // Intermediate level fuses the virtual half: two sub-claims (virtual +
        // relation).
        assert_eq!(stage.instances[0].summand.subclaims.len(), 2);
        assert_eq!(stage.batching, BatchingScheme::Standalone);
        assert!(matches!(
            plan.carried_openings,
            CarriedOpeningPlan::Singleton
        ));
        assert_eq!(plan.transcript_schedule.events.len(), 3);
    }

    #[test]
    fn plan_level_terminal_emits_relation_only_stage2_instance() {
        let params = sample_params();
        let next_w_len = sample_next_w_len(&params);
        let expected_rounds = sumcheck_rounds(D, next_w_len);
        let plan = plan_level::<D>(
            &params,
            next_w_len,
            &ProtocolGates::default(),
            LevelRole::Terminal,
        )
        .expect("plan");

        assert_eq!(plan.stages.len(), 1);
        let stage = &plan.stages[0];
        assert_eq!(stage.instances.len(), 1);
        let instance = &stage.instances[0];
        assert_eq!(instance.kind, InstanceKind::Regular);
        assert_eq!(instance.num_rounds, expected_rounds);
        assert_eq!(instance.label, "stage2-relation-only");
        // Terminal level is relation-only: a single unweighted relation
        // sub-claim, and no batching-coeff challenge anywhere in the summand
        // (neither as a sub-claim weight nor as a body factor).
        assert_eq!(instance.summand.subclaims.len(), 1);
        assert_eq!(instance.summand.subclaims[0].weight, None);
        let has_challenge = instance.summand.subclaims.iter().any(|subclaim| {
            subclaim.weight.is_some()
                || subclaim.body.terms.iter().any(|term| {
                    term.factors
                        .iter()
                        .any(|factor| matches!(factor, Source::Challenge(_)))
                })
        });
        assert!(!has_challenge, "terminal summand must not fuse a gamma");
        assert_eq!(stage.batching, BatchingScheme::Standalone);
    }

    #[test]
    fn plan_level_rejects_unsupported_gates() {
        let params = sample_params();
        let next_w_len = sample_next_w_len(&params);
        for gates in [
            ProtocolGates {
                trace: true,
                ..ProtocolGates::default()
            },
            ProtocolGates {
                l2_certificate: true,
                ..ProtocolGates::default()
            },
            ProtocolGates {
                setup_offload: true,
                ..ProtocolGates::default()
            },
        ] {
            let err = plan_level::<D>(&params, next_w_len, &gates, LevelRole::Intermediate)
                .expect_err("gate not yet supported");
            assert!(matches!(err, AkitaError::InvalidInput(_)));
        }
    }

    #[test]
    fn plan_level_allows_zk_gate() {
        let params = sample_params();
        let next_w_len = sample_next_w_len(&params);
        let gates = ProtocolGates {
            zk: true,
            ..ProtocolGates::default()
        };
        // ZK only switches the prover sink; the emitted plan is unchanged.
        let with_zk =
            plan_level::<D>(&params, next_w_len, &gates, LevelRole::Intermediate).expect("zk plan");
        let baseline = plan_level::<D>(
            &params,
            next_w_len,
            &ProtocolGates::default(),
            LevelRole::Intermediate,
        )
        .expect("baseline plan");
        assert_eq!(with_zk, baseline);
    }

    #[test]
    fn plan_level_rejects_non_power_of_two_ring_dimension() {
        let mut params = sample_params();
        params.ring_dimension = 3;
        let err = plan_level::<3>(
            &params,
            24,
            &ProtocolGates::default(),
            LevelRole::Intermediate,
        )
        .expect_err("non-power-of-two ring dimension rejected");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn plan_level_rejects_ring_dimension_mismatch() {
        let mut params = sample_params();
        // params disagrees with the dispatched const D.
        params.ring_dimension = 32;
        let err = plan_level::<D>(
            &params,
            sample_next_w_len(&params),
            &ProtocolGates::default(),
            LevelRole::Intermediate,
        )
        .expect_err("ring dimension mismatch rejected");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn plan_level_round_count_follows_sumcheck_rounds_not_outer_vars() {
        let params = sample_params();
        // Five ring elements pad to col_bits = 3, not outer_vars() = 6.
        let next_w_len = params.ring_dimension * 5;
        let expected_rounds = sumcheck_rounds(D, next_w_len);
        assert_ne!(expected_rounds, params.outer_vars() + 6);

        let plan = plan_level::<D>(
            &params,
            next_w_len,
            &ProtocolGates::default(),
            LevelRole::Intermediate,
        )
        .expect("plan");
        assert_eq!(plan.stages[0].instances[0].num_rounds, expected_rounds);
        assert_eq!(
            plan.transcript_schedule.events[1],
            TranscriptEvent::SumcheckRounds {
                num_rounds: expected_rounds
            }
        );
    }

    #[test]
    fn retains_eq_factored_format_only_for_standalone_eq_factored() {
        let eq_factored_instance = || {
            let mut descriptor = stage2_descriptor(4, LevelRole::Intermediate);
            descriptor.kind = InstanceKind::EqFactored;
            descriptor
        };

        let standalone = StagePlan {
            instances: vec![eq_factored_instance()],
            batching: BatchingScheme::Standalone,
        };
        assert!(standalone.retains_eq_factored_format(0));

        let regular = StagePlan {
            instances: vec![stage2_descriptor(4, LevelRole::Intermediate)],
            batching: BatchingScheme::Standalone,
        };
        assert!(!regular.retains_eq_factored_format(0));

        let batched = StagePlan {
            instances: vec![
                eq_factored_instance(),
                stage2_descriptor(4, LevelRole::Intermediate),
            ],
            batching: BatchingScheme::GammaPowers {
                gamma_powers: vec![0, 1],
            },
        };
        assert!(!batched.retains_eq_factored_format(0));
        assert!(!batched.retains_eq_factored_format(99));
    }
}
