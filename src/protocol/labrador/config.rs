//! Labrador parameter-selection and security checks.

use crate::error::HachiError;
use crate::primitives::serialization::Compress;
use crate::protocol::commitment::utils::norm::detect_field_modulus;
use crate::protocol::labrador::guardrails::LABRADOR_MAX_LEVELS;
use crate::protocol::labrador::types::{LabradorReductionConfig, LabradorWitness};
use crate::protocol::proof::PackedCoeffRow;
use crate::{CanonicalField, FieldCore, HachiSerialize};
use std::f64::consts::{E, PI};
const LABRADOR_LOGDELTA: f64 = 0.00639138757765197; // log2(1.00444)
const LABRADOR_T: f64 = 14.0;
const LABRADOR_SLACK: f64 = 2.0;
const LABRADOR_TAU1: f64 = 32.0;
const LABRADOR_TAU2: f64 = 8.0;

/// Full fold-level plan: security parameters plus witness reshaping layout.
#[derive(Debug, Clone)]
pub struct LabradorFoldPlan {
    /// Security parameters (formerly `f`, `b`, `fu`, `bu`, `kappa`,
    /// `kappa1`, `tail`).
    pub config: LabradorReductionConfig,
    /// Virtual row length after reshaping (formerly `nn`).
    pub virtual_row_len: usize,
    /// Per-original-row split count. `0` = continuation (concatenate with next
    /// row), `>0` = boundary that terminates a group and splits it into this
    /// many virtual rows of length `virtual_row_len` (formerly `nu`).
    pub row_split_counts: Vec<usize>,
}

const MAX_WITNESS_DIGIT_PARTS: usize = 8;
const MAX_COMMITMENT_RANK: usize = 32;

#[derive(Debug, Clone)]
struct LabradorWitnessPlanningProfile {
    row_lengths: Vec<usize>,
    row_coeff_bits: Vec<usize>,
    norm_sum: f64,
    coeff_bit_bound: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct LabradorFoldEstimate {
    pub plan: LabradorFoldPlan,
    pub level_payload_bytes: usize,
    pub next_witness_bytes: usize,
    pub transition_bytes: usize,
    next_row_lengths: Vec<usize>,
    next_row_coeff_bits: Vec<usize>,
    next_norm_sum: f64,
}

impl LabradorFoldEstimate {
    fn next_profile(&self) -> Result<LabradorWitnessPlanningProfile, HachiError> {
        LabradorWitnessPlanningProfile::new(
            self.next_row_lengths.clone(),
            self.next_row_coeff_bits.clone(),
            self.next_norm_sum,
            None,
        )
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LabradorRecursiveSizeEstimate {
    pub initial_plan: LabradorFoldPlan,
    pub proof_bytes: usize,
    pub final_witness_bytes: usize,
    pub level_count: usize,
}

impl LabradorWitnessPlanningProfile {
    fn new(
        row_lengths: Vec<usize>,
        row_coeff_bits: Vec<usize>,
        norm_sum: f64,
        coeff_bit_bound: Option<usize>,
    ) -> Result<Self, HachiError> {
        if row_lengths.is_empty() {
            return Err(HachiError::InvalidInput(
                "cannot select config for empty Labrador witness".to_string(),
            ));
        }
        if row_lengths.iter().sum::<usize>() == 0 {
            return Err(HachiError::InvalidInput(
                "cannot select config for zero-length Labrador witness".to_string(),
            ));
        }
        if row_lengths.len() != row_coeff_bits.len() {
            return Err(HachiError::InvalidInput(
                "Labrador witness profile row_bits length mismatch".to_string(),
            ));
        }
        if row_coeff_bits.iter().any(|&bits| bits == 0 || bits > 128) {
            return Err(HachiError::InvalidInput(
                "Labrador witness profile coeff_bits out of range".to_string(),
            ));
        }
        if !norm_sum.is_finite() || norm_sum < 0.0 {
            return Err(HachiError::InvalidInput(
                "cannot select config for non-finite Labrador witness norm".to_string(),
            ));
        }
        Ok(Self {
            row_lengths,
            row_coeff_bits,
            norm_sum,
            coeff_bit_bound,
        })
    }

    fn from_witness<F: FieldCore + CanonicalField, const D: usize>(
        witness: &LabradorWitness<F, D>,
    ) -> Result<Self, HachiError> {
        let row_lengths = witness.rows().iter().map(|r| r.len()).collect();
        let row_coeff_bits = witness
            .rows()
            .iter()
            .map(|row| {
                PackedCoeffRow::detect_coeff_bits(row)
                    .map_err(|err| HachiError::InvalidInput(err.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(row_lengths, row_coeff_bits, witness.norm() as f64, None)
    }

    fn from_handoff_witness<F: FieldCore + CanonicalField, const D: usize>(
        witness: &LabradorWitness<F, D>,
        coeff_bit_bound: usize,
    ) -> Result<Self, HachiError> {
        let row_lengths = witness.rows().iter().map(|r| r.len()).collect();
        let row_coeff_bits = witness
            .rows()
            .iter()
            .map(|row| {
                PackedCoeffRow::detect_coeff_bits(row)
                    .map_err(|err| HachiError::InvalidInput(err.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(
            row_lengths,
            row_coeff_bits,
            witness.norm() as f64,
            Some(coeff_bit_bound.max(1)),
        )
    }

    fn total_len(&self) -> usize {
        self.row_lengths.iter().sum()
    }
}

/// Euclidean SIS estimate for a flattened Module-SIS instance.
///
/// This mirrors the `norm == 2` path in the sibling lattice-estimator:
/// flatten rank-`rank` over ring degree `D` to SIS with `n = rank * D`,
/// flatten `width_ring_elems` ring columns to `m = width_ring_elems * D`,
/// solve for the required root-Hermite factor at the optimizer's preferred
/// attack dimension, and convert that to an approximate BKZ block size using
/// the Chen-style `delta(beta)` relation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SisEuclideanLatticeEstimate {
    /// Exact field modulus used for the estimate.
    pub modulus: u128,
    /// `log2(modulus)` from the exact modulus value.
    pub logq: f64,
    /// Flattened SIS row dimension `n = rank * D`.
    pub sis_dimension: usize,
    /// Flattened SIS width `m = width_ring_elems * D`.
    pub sis_width: usize,
    /// Euclidean norm bound supplied to the estimator.
    pub norm: f64,
    /// Optimized attack sublattice dimension `d_att <= m`.
    pub attack_dimension: usize,
    /// Required root-Hermite factor `delta_req`.
    pub required_delta: f64,
    /// BKZ block size implied by `delta_req`, rounded up.
    pub bkz_beta: usize,
    /// Whether the estimator considers lattice reduction feasible.
    pub reduction_possible: bool,
    /// `log2(lb)` for the estimator's lower-bound predicate.
    pub log2_solution_lower_bound: f64,
    /// Whether the supplied norm exceeds the estimator's lower bound.
    pub solution_exists: bool,
    /// Approximate `log2(rop)` under the BDGL16 asymptotic cost model.
    ///
    /// This is `+inf` when the Euclidean estimator would reject the instance
    /// as not attackable under its feasibility predicate.
    pub log2_rop_bdgl16: f64,
}

/// Module-SIS security check used by the C reference.
///
/// Returns `true` when `log2(norm) < min(LOGQ, 2*sqrt(LOGQ*LOGDELTA*N)*sqrt(rank))`.
pub fn sis_secure<F: CanonicalField, const D: usize>(rank: usize, norm: f64) -> bool {
    sis_secure_with_params(rank, norm, logq_bits::<F>() as f64, D as f64)
}

/// Approximate the sibling lattice-estimator's Euclidean SIS attack model for
/// a flattened Module-SIS instance.
///
/// The input `width_ring_elems` is the number of ring columns before
/// flattening. Internally this becomes SIS width `m = width_ring_elems * D`.
///
/// # Errors
///
/// Returns an error on zero dimensions, non-positive / non-finite norms,
/// modulus detection failure, or when the bound is trivially large compared to
/// the modulus (matching the estimator's Euclidean guardrail).
pub fn estimate_module_sis_euclidean<F: CanonicalField, const D: usize>(
    rank: usize,
    width_ring_elems: usize,
    norm: f64,
) -> Result<SisEuclideanLatticeEstimate, HachiError> {
    if rank == 0 {
        return Err(HachiError::InvalidInput(
            "SIS estimate requires rank > 0".to_string(),
        ));
    }
    if width_ring_elems == 0 {
        return Err(HachiError::InvalidInput(
            "SIS estimate requires width_ring_elems > 0".to_string(),
        ));
    }
    if !norm.is_finite() || norm <= 0.0 {
        return Err(HachiError::InvalidInput(
            "SIS estimate requires a finite positive norm".to_string(),
        ));
    }

    let modulus = detect_field_modulus::<F>();
    if modulus <= 1 {
        return Err(HachiError::InvalidInput(
            "SIS estimate requires modulus > 1".to_string(),
        ));
    }
    let modulus_f = modulus as f64;
    let logq = modulus_f.log2();
    let sis_dimension = rank
        .checked_mul(D)
        .ok_or_else(|| HachiError::InvalidInput("SIS estimate dimension overflow".to_string()))?;
    let sis_width = width_ring_elems
        .checked_mul(D)
        .ok_or_else(|| HachiError::InvalidInput("SIS estimate width overflow".to_string()))?;

    if norm >= (modulus_f - 1.0) / 2.0 {
        return Err(HachiError::InvalidInput(
            "SIS estimate expects norm < (q-1)/2".to_string(),
        ));
    }

    let log2_norm = norm.log2();
    let log_delta = if log2_norm == 0.0 {
        0.0
    } else {
        (log2_norm * log2_norm) / (4.0 * sis_dimension as f64 * logq)
    };
    let opt_attack_dimension = if log_delta > 0.0 {
        ((sis_dimension as f64 * logq) / log_delta).sqrt().floor() as usize
    } else {
        sis_width
    };
    let attack_dimension = opt_attack_dimension.clamp(2, sis_width.max(2));

    let root_volume = sis_dimension as f64 * logq / attack_dimension as f64;
    let required_delta_log2 =
        (log2_norm - root_volume) / (attack_dimension.saturating_sub(1) as f64);
    let required_delta = 2f64.powf(required_delta_log2);
    let beta = beta_from_root_hermite(required_delta).unwrap_or(usize::MAX);
    let reduction_possible = required_delta >= 1.0 && beta <= attack_dimension;
    let bkz_beta = if reduction_possible {
        beta
    } else {
        attack_dimension
    };

    // Matches the Euclidean estimator's lower-bound feasibility gate:
    // lb = min(sqrt(n * ln(q)), sqrt(d) * q^(n/d)).
    let log2_lb_gaussian = 0.5 * ((sis_dimension as f64) * modulus_f.ln()).log2();
    let log2_lb_qary = 0.5 * (attack_dimension as f64).log2() + root_volume;
    let log2_solution_lower_bound = log2_lb_gaussian.min(log2_lb_qary);
    let solution_exists = log2_norm > log2_solution_lower_bound;

    let log2_rop_bdgl16 = if reduction_possible && solution_exists {
        let repeat = if bkz_beta < attack_dimension {
            8.0 * attack_dimension as f64
        } else {
            1.0
        };
        let lll_log2 = 3.0 * (attack_dimension as f64).log2();
        let sieve_log2 = 0.292 * bkz_beta as f64 + 16.4 + repeat.log2();
        log2_add_exp(lll_log2, sieve_log2)
    } else {
        f64::INFINITY
    };

    Ok(SisEuclideanLatticeEstimate {
        modulus,
        logq,
        sis_dimension,
        sis_width,
        norm,
        attack_dimension,
        required_delta,
        bkz_beta,
        reduction_possible,
        log2_solution_lower_bound,
        solution_exists,
        log2_rop_bdgl16,
    })
}

/// Select a linear-only Labrador fold plan (non-tail mode).
///
/// Mirrors the C `init_proof` parameter selection path with `quadratic=0`,
/// including the row-split k-loop that determines the optimal
/// `virtual_row_len`.
///
/// # Errors
///
/// Returns an error if witness metadata is empty/invalid or if no secure
/// commitment ranks are found within supported bounds.
pub fn select_config<F: FieldCore + CanonicalField + HachiSerialize, const D: usize>(
    witness: &LabradorWitness<F, D>,
) -> Result<LabradorReductionConfig, HachiError> {
    plan_fold::<F, D>(witness, false).map(|p| p.config)
}

/// Select a linear-only Labrador fold plan with explicit tail flag.
///
/// # Errors
///
/// Returns an error if witness metadata is empty/invalid or if no secure
/// commitment ranks are found within supported bounds.
pub fn select_config_with_mode<F: FieldCore + CanonicalField + HachiSerialize, const D: usize>(
    witness: &LabradorWitness<F, D>,
    tail: bool,
) -> Result<LabradorReductionConfig, HachiError> {
    plan_fold::<F, D>(witness, tail).map(|p| p.config)
}

/// Compute a full Labrador fold plan (config + reshaping layout).
///
/// Mirrors the C `init_proof` algorithm with `quadratic=0`: all input rows
/// are placed in a single group (boundary at the last row only). The k-loop
/// searches from k=15 down to k=1, keeps every secure candidate whose
/// commitment overhead fits within `1.1 × virtual_row_len`, and returns the candidate with
/// the smallest carried witness size for the next transition.
///
/// # Errors
///
/// Returns an error if the witness is empty or no secure parameters exist.
#[tracing::instrument(skip_all, name = "labrador::plan_fold")]
pub fn plan_fold<F: FieldCore + CanonicalField + HachiSerialize, const D: usize>(
    witness: &LabradorWitness<F, D>,
    tail: bool,
) -> Result<LabradorFoldPlan, HachiError> {
    let profile = LabradorWitnessPlanningProfile::from_witness(witness)?;
    plan_fold_with_profile::<F, D>(&profile, tail)
}

fn plan_fold_with_profile<F: FieldCore + CanonicalField + HachiSerialize, const D: usize>(
    profile: &LabradorWitnessPlanningProfile,
    tail: bool,
) -> Result<LabradorFoldPlan, HachiError> {
    search_best_estimate_with_profile::<F, D>(profile, tail).map(|estimate| estimate.plan)
}

fn coeff_varz_cap(coeff_bit_bound: usize) -> Option<f64> {
    let exp = coeff_bit_bound.checked_mul(2)?;
    if exp > i32::MAX as usize {
        return None;
    }
    let cap = 2f64.powi(exp as i32) / 12.0 * (LABRADOR_TAU1 + 4.0 * LABRADOR_TAU2);
    (cap.is_finite() && cap > 0.0).then_some(cap)
}

fn search_best_estimate_with_profile<
    F: FieldCore + CanonicalField + HachiSerialize,
    const D: usize,
>(
    profile: &LabradorWitnessPlanningProfile,
    tail: bool,
) -> Result<LabradorFoldEstimate, HachiError> {
    let row_lengths = &profile.row_lengths;
    let r = row_lengths.len();
    let total_len = profile.total_len();
    let logq_bits = logq_bits::<F>();
    let logq = logq_bits as f64;
    let d = D as f64;

    let mut last_plan = None;
    let mut best_estimate = None;
    let mut best_score = usize::MAX;

    for k in (1..=15usize).rev() {
        let virtual_row_len = total_len.div_ceil(k);
        let virtual_row_count = k;
        let virtual_row_count_f = virtual_row_count as f64;

        let mut varz = profile.norm_sum / (virtual_row_len as f64 * d);
        varz *= LABRADOR_TAU1 + 4.0 * LABRADOR_TAU2;
        if let Some(coeff_bit_bound) = profile.coeff_bit_bound {
            if let Some(varz_cap) = coeff_varz_cap(coeff_bit_bound) {
                varz = varz.min(varz_cap);
            }
        }
        if !varz.is_finite() || varz <= 0.0 {
            varz = 1.0;
        }

        let witness_digit_part_range = if tail {
            1..=1
        } else {
            1..=MAX_WITNESS_DIGIT_PARTS
        };

        for witness_digit_parts in witness_digit_part_range {
            let mut witness_digit_bits = ((12.0f64.log2() + varz.log2())
                / (2.0 * witness_digit_parts as f64))
                .round() as isize;
            witness_digit_bits = witness_digit_bits.clamp(1, logq_bits as isize);

            let (aux_digit_parts, aux_digit_bits) = if tail {
                (1usize, logq_bits.max(1))
            } else {
                let aux_digit_parts = ((logq_bits + 2 * (witness_digit_bits as usize) / 3)
                    / (witness_digit_bits as usize))
                    .max(1);
                let aux_digit_bits = ((logq_bits + aux_digit_parts / 2) / aux_digit_parts).max(1);
                (aux_digit_parts, aux_digit_bits)
            };

            let mut found_inner_commit_rank = None;
            let mut last_normsq = 0.0f64;

            for inner_commit_rank in 1..=MAX_COMMITMENT_RANK {
                let mut normsq = (2f64.powi(2 * witness_digit_bits as i32) / 12.0
                    * ((witness_digit_parts - 1) as f64)
                    + varz
                        / 2f64.powi(
                            2 * (witness_digit_parts - 1) as i32 * witness_digit_bits as i32,
                        ))
                    * virtual_row_len as f64;
                if !tail {
                    let hi_exp = logq_bits as isize
                        - (aux_digit_parts.saturating_sub(1) * aux_digit_bits) as isize;
                    let hi_exp = hi_exp.max(0) as i32;
                    normsq += (2f64.powi(2 * aux_digit_bits as i32)
                        * ((aux_digit_parts - 1) as f64)
                        + 2f64.powi(2 * hi_exp))
                        / 12.0
                        * (virtual_row_count_f * inner_commit_rank as f64
                            + (virtual_row_count_f * virtual_row_count_f + virtual_row_count_f)
                                / 2.0);
                }
                normsq *= d;
                last_normsq = normsq;

                if sis_secure_with_params(
                    inner_commit_rank,
                    6.0 * LABRADOR_T
                        * LABRADOR_SLACK
                        * 2f64
                            .powi(((witness_digit_parts - 1) * witness_digit_bits as usize) as i32)
                        * normsq.sqrt(),
                    logq,
                    d,
                ) {
                    found_inner_commit_rank = Some(inner_commit_rank);
                    break;
                }
            }

            let inner_commit_rank = match found_inner_commit_rank {
                Some(rank) => rank,
                None => {
                    last_plan = Some(build_plan(
                        LabradorReductionConfig {
                            witness_digit_parts,
                            witness_digit_bits: witness_digit_bits as usize,
                            aux_digit_parts,
                            aux_digit_bits,
                            inner_commit_rank: MAX_COMMITMENT_RANK,
                            outer_commit_rank: 0,
                            tail,
                        },
                        virtual_row_len,
                        r,
                        virtual_row_count,
                    ));
                    continue;
                }
            };

            let outer_commit_rank = if tail {
                0
            } else {
                match (1..=MAX_COMMITMENT_RANK).find(|&rank| {
                    sis_secure_with_params(rank, 2.0 * LABRADOR_SLACK * last_normsq.sqrt(), logq, d)
                }) {
                    Some(rank) => rank,
                    None => {
                        last_plan = Some(build_plan(
                            LabradorReductionConfig {
                                witness_digit_parts,
                                witness_digit_bits: witness_digit_bits as usize,
                                aux_digit_parts,
                                aux_digit_bits,
                                inner_commit_rank,
                                outer_commit_rank: MAX_COMMITMENT_RANK,
                                tail,
                            },
                            virtual_row_len,
                            r,
                            virtual_row_count,
                        ));
                        continue;
                    }
                }
            };

            let plan = build_plan(
                LabradorReductionConfig {
                    witness_digit_parts,
                    witness_digit_bits: witness_digit_bits as usize,
                    aux_digit_parts,
                    aux_digit_bits,
                    inner_commit_rank,
                    outer_commit_rank,
                    tail,
                },
                virtual_row_len,
                r,
                virtual_row_count,
            );
            let estimate = estimate_plan_with_profile::<F, D>(profile, &plan, last_normsq)?;
            let score = estimate.transition_bytes;
            last_plan = Some(plan);
            maybe_take_better_estimate(&mut best_estimate, &mut best_score, score, estimate);
        }
    }

    if let Some(estimate) = best_estimate {
        return Ok(estimate);
    }

    last_plan.map_or_else(
        || {
            Err(HachiError::InvalidInput(
                "failed to find secure Labrador fold parameters".to_string(),
            ))
        },
        |plan| estimate_plan_with_profile::<F, D>(profile, &plan, profile.norm_sum.max(1.0)),
    )
}

fn estimate_plan_with_profile<F: FieldCore + CanonicalField + HachiSerialize, const D: usize>(
    profile: &LabradorWitnessPlanningProfile,
    plan: &LabradorFoldPlan,
    next_norm_sum: f64,
) -> Result<LabradorFoldEstimate, HachiError> {
    let virtual_row_count: usize = plan.row_split_counts.iter().sum();
    let next_row_lengths =
        estimate_next_row_lengths(virtual_row_count, plan.virtual_row_len, &plan.config);
    let next_row_coeff_bits = estimate_next_row_coeff_bits(&plan.config);
    let level_payload_bytes = estimate_level_payload_bytes::<F, D>(
        profile.row_lengths.len(),
        virtual_row_count,
        plan.virtual_row_len,
        &plan.config,
    );
    let next_witness_bytes =
        estimate_witness_bytes_from_row_lengths::<D>(&next_row_lengths, &next_row_coeff_bits);
    Ok(LabradorFoldEstimate {
        plan: plan.clone(),
        level_payload_bytes,
        next_witness_bytes,
        transition_bytes: level_payload_bytes + next_witness_bytes,
        next_row_lengths,
        next_row_coeff_bits,
        next_norm_sum: next_norm_sum.max(1.0),
    })
}

fn estimate_next_norm_sum_for_config<F: CanonicalField, const D: usize>(
    profile: &LabradorWitnessPlanningProfile,
    virtual_row_len: usize,
    virtual_row_count: usize,
    config: &LabradorReductionConfig,
) -> f64 {
    let logq_bits = logq_bits::<F>();
    let d = D as f64;
    let mut varz = profile.norm_sum / (virtual_row_len as f64 * d);
    varz *= LABRADOR_TAU1 + 4.0 * LABRADOR_TAU2;
    if let Some(coeff_bit_bound) = profile.coeff_bit_bound {
        if let Some(varz_cap) = coeff_varz_cap(coeff_bit_bound) {
            varz = varz.min(varz_cap);
        }
    }
    if !varz.is_finite() || varz <= 0.0 {
        varz = 1.0;
    }

    let mut normsq = (2f64.powi(2 * config.witness_digit_bits as i32) / 12.0
        * ((config.witness_digit_parts - 1) as f64)
        + varz
            / 2f64.powi(
                2 * (config.witness_digit_parts - 1) as i32 * config.witness_digit_bits as i32,
            ))
        * virtual_row_len as f64;
    if !config.tail {
        let hi_exp = logq_bits as isize
            - (config.aux_digit_parts.saturating_sub(1) * config.aux_digit_bits) as isize;
        let hi_exp = hi_exp.max(0) as i32;
        let virtual_row_count_f = virtual_row_count as f64;
        normsq += (2f64.powi(2 * config.aux_digit_bits as i32)
            * ((config.aux_digit_parts - 1) as f64)
            + 2f64.powi(2 * hi_exp))
            / 12.0
            * (virtual_row_count_f * config.inner_commit_rank as f64
                + (virtual_row_count_f * virtual_row_count_f + virtual_row_count_f) / 2.0);
    }
    (normsq * d).max(1.0)
}

pub(crate) fn estimate_fold_step<F: FieldCore + CanonicalField + HachiSerialize, const D: usize>(
    witness: &LabradorWitness<F, D>,
    tail: bool,
) -> Result<LabradorFoldEstimate, HachiError> {
    let profile = LabradorWitnessPlanningProfile::from_witness(witness)?;
    search_best_estimate_with_profile::<F, D>(&profile, tail)
}

pub(crate) fn estimate_selected_fold_step<
    F: FieldCore + CanonicalField + HachiSerialize,
    const D: usize,
>(
    witness: &LabradorWitness<F, D>,
    plan: &LabradorFoldPlan,
) -> Result<LabradorFoldEstimate, HachiError> {
    let profile = LabradorWitnessPlanningProfile::from_witness(witness)?;
    let virtual_row_count: usize = plan.row_split_counts.iter().sum();
    let next_norm_sum = estimate_next_norm_sum_for_config::<F, D>(
        &profile,
        plan.virtual_row_len,
        virtual_row_count,
        &plan.config,
    );
    estimate_plan_with_profile::<F, D>(&profile, plan, next_norm_sum)
}

#[cfg(test)]
pub(crate) fn estimate_recursive_proof_with_plan<
    F: FieldCore + CanonicalField + HachiSerialize,
    const D: usize,
>(
    witness: &LabradorWitness<F, D>,
    initial_plan: &LabradorFoldPlan,
) -> Result<LabradorRecursiveSizeEstimate, HachiError> {
    let profile = LabradorWitnessPlanningProfile::from_witness(witness)?;
    let virtual_row_count: usize = initial_plan.row_split_counts.iter().sum();
    let next_norm_sum = estimate_next_norm_sum_for_config::<F, D>(
        &profile,
        initial_plan.virtual_row_len,
        virtual_row_count,
        &initial_plan.config,
    );
    let initial_estimate =
        estimate_plan_with_profile::<F, D>(&profile, initial_plan, next_norm_sum)?;
    let (proof_bytes, final_witness_bytes, level_count) =
        simulate_recursive_proof_bytes::<F, D>(profile, Some(initial_estimate))?;
    Ok(LabradorRecursiveSizeEstimate {
        initial_plan: initial_plan.clone(),
        proof_bytes,
        final_witness_bytes,
        level_count,
    })
}

pub(crate) fn estimate_handoff_recursive_proof<
    F: FieldCore + CanonicalField + HachiSerialize,
    const D: usize,
>(
    witness: &LabradorWitness<F, D>,
    coeff_bit_bound: usize,
) -> Result<LabradorRecursiveSizeEstimate, HachiError> {
    let profile = LabradorWitnessPlanningProfile::from_handoff_witness(witness, coeff_bit_bound)?;
    let initial_estimate = search_best_estimate_with_profile::<F, D>(&profile, false)?;
    let initial_plan = initial_estimate.plan.clone();
    let (proof_bytes, final_witness_bytes, level_count) =
        simulate_recursive_proof_bytes::<F, D>(profile, Some(initial_estimate))?;
    Ok(LabradorRecursiveSizeEstimate {
        initial_plan,
        proof_bytes,
        final_witness_bytes,
        level_count,
    })
}

fn simulate_recursive_proof_bytes<
    F: FieldCore + CanonicalField + HachiSerialize,
    const D: usize,
>(
    mut profile: LabradorWitnessPlanningProfile,
    mut first_non_tail: Option<LabradorFoldEstimate>,
) -> Result<(usize, usize, usize), HachiError> {
    let mut level_payload_total = 0usize;
    let mut level_count = 0usize;

    while level_count + 1 < LABRADOR_MAX_LEVELS {
        let before_bytes = estimate_witness_bytes_from_row_lengths::<D>(
            &profile.row_lengths,
            &profile.row_coeff_bits,
        );
        if before_bytes == 0 || profile.row_lengths.len() <= 1 {
            break;
        }
        let estimate = match first_non_tail.take() {
            Some(estimate) => estimate,
            None => search_best_estimate_with_profile::<F, D>(&profile, false)?,
        };
        if estimate.transition_bytes >= before_bytes {
            break;
        }
        level_payload_total += estimate.level_payload_bytes;
        profile = estimate.next_profile()?;
        level_count += 1;
    }

    if level_count + 1 < LABRADOR_MAX_LEVELS {
        let before_bytes = estimate_witness_bytes_from_row_lengths::<D>(
            &profile.row_lengths,
            &profile.row_coeff_bits,
        );
        if before_bytes > 0 && profile.row_lengths.len() > 1 {
            let tail_estimate = search_best_estimate_with_profile::<F, D>(&profile, true)?;
            if tail_estimate.transition_bytes < before_bytes {
                level_payload_total += tail_estimate.level_payload_bytes;
                profile = tail_estimate.next_profile()?;
                level_count += 1;
            }
        }
    }

    let final_witness_bytes =
        estimate_witness_bytes_from_row_lengths::<D>(&profile.row_lengths, &profile.row_coeff_bits);
    Ok((
        4 + level_payload_total + final_witness_bytes,
        final_witness_bytes,
        level_count,
    ))
}

fn estimate_next_row_lengths(
    virtual_row_count: usize,
    virtual_row_len: usize,
    config: &LabradorReductionConfig,
) -> Vec<usize> {
    let mut row_lengths = vec![virtual_row_len; config.witness_digit_parts];
    if !config.tail {
        row_lengths.push(
            virtual_row_count * config.inner_commit_rank * config.aux_digit_parts
                + virtual_row_count * (virtual_row_count + 1) / 2 * config.aux_digit_parts,
        );
    }
    row_lengths
}

fn estimate_next_row_coeff_bits(config: &LabradorReductionConfig) -> Vec<usize> {
    let mut row_coeff_bits = vec![config.witness_digit_bits.max(1); config.witness_digit_parts];
    if !config.tail {
        row_coeff_bits.push(config.aux_digit_bits.max(1));
    }
    row_coeff_bits
}

fn estimate_witness_bytes_from_row_lengths<const D: usize>(
    row_lengths: &[usize],
    row_coeff_bits: &[usize],
) -> usize {
    debug_assert_eq!(row_lengths.len(), row_coeff_bits.len());
    4 + row_lengths
        .iter()
        .zip(row_coeff_bits.iter())
        .map(|(&ring_elems, &coeff_bits)| {
            estimate_packed_coeff_row_bytes::<D>(ring_elems, coeff_bits)
        })
        .sum::<usize>()
}

fn estimate_level_payload_bytes<F: FieldCore + CanonicalField + HachiSerialize, const D: usize>(
    input_row_count: usize,
    virtual_row_count: usize,
    virtual_row_len: usize,
    config: &LabradorReductionConfig,
) -> usize {
    let inner_payload_ring_elems = if config.tail || config.outer_commit_rank == 0 {
        virtual_row_count * config.inner_commit_rank * config.aux_digit_parts
    } else {
        config.outer_commit_rank
    };
    let linear_payload_ring_elems = if config.tail || config.outer_commit_rank == 0 {
        virtual_row_count * (virtual_row_count + 1) / 2 * config.aux_digit_parts
    } else {
        config.outer_commit_rank
    };

    1 + estimate_vec_usize_bytes(input_row_count)
        + config.serialized_size(Compress::No)
        + virtual_row_len.serialized_size(Compress::No)
        + estimate_vec_usize_bytes(input_row_count)
        + estimate_flat_ring_vec_bytes::<F, D>(inner_payload_ring_elems)
        + estimate_flat_ring_vec_bytes::<F, D>(linear_payload_ring_elems)
        + jl_projection_bytes()
        + 8
        + estimate_flat_ring_vec_bytes::<F, D>(jl_lifts::<F>())
        + 16
}

fn estimate_flat_ring_vec_bytes<F: FieldCore + HachiSerialize, const D: usize>(
    ring_elems: usize,
) -> usize {
    4 + 8 + ring_elems * D * F::zero().serialized_size(Compress::No)
}

fn estimate_packed_coeff_row_bytes<const D: usize>(ring_elems: usize, coeff_bits: usize) -> usize {
    debug_assert!((1..=128).contains(&coeff_bits));
    4 + 8 + 1 + (ring_elems * D * coeff_bits).div_ceil(8)
}

fn estimate_vec_usize_bytes(len: usize) -> usize {
    8 + len * 8
}

fn jl_projection_bytes() -> usize {
    256 * std::mem::size_of::<i64>()
}

fn maybe_take_better_estimate(
    best_estimate: &mut Option<LabradorFoldEstimate>,
    best_score: &mut usize,
    score: usize,
    candidate: LabradorFoldEstimate,
) {
    if score < *best_score
        || (score == *best_score
            && best_estimate.as_ref().is_none_or(|best| {
                candidate.plan.row_split_counts.iter().sum::<usize>()
                    < best.plan.row_split_counts.iter().sum::<usize>()
            }))
    {
        *best_score = score;
        *best_estimate = Some(candidate);
    }
}

fn build_plan(
    config: LabradorReductionConfig,
    virtual_row_len: usize,
    input_row_count: usize,
    virtual_row_count: usize,
) -> LabradorFoldPlan {
    let mut row_split_counts = vec![0usize; input_row_count];
    if !row_split_counts.is_empty() {
        row_split_counts[input_row_count - 1] = virtual_row_count;
    }
    LabradorFoldPlan {
        config,
        virtual_row_len,
        row_split_counts,
    }
}

/// Build a trivial fold plan (no reshaping) from a config and row lengths.
///
/// All rows keep their original lengths; `virtual_row_len = max(row_lengths)`
/// and `row_split_counts`
/// marks each row as its own virtual row.
pub fn trivial_plan(config: LabradorReductionConfig, row_lengths: &[usize]) -> LabradorFoldPlan {
    let virtual_row_len = row_lengths.iter().copied().max().unwrap_or(0);
    let row_split_counts: Vec<usize> = row_lengths.iter().map(|_| 1).collect();
    LabradorFoldPlan {
        config,
        virtual_row_len,
        row_split_counts,
    }
}

/// Compute a full Labrador fold plan for the Hachi→Labrador handoff witness.
///
/// Unlike the generic recursive planner, the handoff planner is seeded from the
/// actual witness rows and their squared norm, rather than collapsing the input
/// to only `(row_count, max_row_len)`.
///
/// # Errors
///
/// Returns an error if the witness is empty or no secure parameter
/// combination exists within the supported bounds.
pub fn plan_handoff<F: FieldCore + CanonicalField + HachiSerialize, const D: usize>(
    witness: &LabradorWitness<F, D>,
    coeff_bit_bound: usize,
) -> Result<LabradorFoldPlan, HachiError> {
    let profile = LabradorWitnessPlanningProfile::from_handoff_witness(witness, coeff_bit_bound)?;
    plan_fold_with_profile::<F, D>(&profile, false)
}

/// Select Labrador reduction config for the Hachi→Labrador handoff witness.
///
/// # Errors
///
/// Returns an error if the witness is empty or no secure parameter
/// combination exists within the supported bounds.
pub fn select_handoff_config<F: FieldCore + CanonicalField + HachiSerialize, const D: usize>(
    witness: &LabradorWitness<F, D>,
    coeff_bit_bound: usize,
) -> Result<LabradorReductionConfig, HachiError> {
    plan_handoff::<F, D>(witness, coeff_bit_bound).map(|plan| plan.config)
}

pub(crate) fn logq_bits<F: CanonicalField>() -> usize {
    let modulus = detect_field_modulus::<F>();
    if modulus <= 1 {
        return 1;
    }
    128 - (modulus.saturating_sub(1)).leading_zeros() as usize
}

pub(crate) fn jl_lifts<F: CanonicalField>() -> usize {
    128_usize.div_ceil(logq_bits::<F>().max(1))
}

fn sis_secure_with_params(rank: usize, norm: f64, logq: f64, ring_degree: f64) -> bool {
    if rank == 0 || !norm.is_finite() || norm <= 0.0 {
        return false;
    }
    let mut maxlog = 2.0 * (logq * LABRADOR_LOGDELTA * ring_degree).sqrt() * (rank as f64).sqrt();
    maxlog = maxlog.min(logq);
    norm.log2() < maxlog
}

fn root_hermite_from_beta(beta: f64) -> f64 {
    ((beta / (2.0 * PI * E)) * (PI * beta).powf(1.0 / beta)).powf(1.0 / (2.0 * (beta - 1.0)))
}

fn beta_from_root_hermite(delta: f64) -> Option<usize> {
    const MIN_BETA: usize = 40;
    const MAX_BETA: usize = 1 << 16;

    if !delta.is_finite() || delta <= 1.0 {
        return None;
    }
    if root_hermite_from_beta(MIN_BETA as f64) < delta {
        return Some(MIN_BETA);
    }

    let mut beta = MIN_BETA;
    while beta < MAX_BETA / 2 && root_hermite_from_beta((2 * beta) as f64) > delta {
        beta *= 2;
    }
    while beta + 10 < MAX_BETA && root_hermite_from_beta((beta + 10) as f64) > delta {
        beta += 10;
    }
    while beta < MAX_BETA && root_hermite_from_beta(beta as f64) >= delta {
        beta += 1;
    }

    (beta < MAX_BETA).then_some(beta)
}

fn log2_add_exp(a: f64, b: f64) -> f64 {
    let hi = a.max(b);
    let lo = a.min(b);
    hi + (1.0 + 2f64.powf(lo - hi)).log2()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp128;
    use crate::algebra::fields::Fp64;
    use crate::algebra::ring::CyclotomicRing;
    use crate::protocol::commitment::Fp128FullCommitmentConfig;
    use crate::protocol::CommitmentConfig;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn row(len: usize) -> Vec<CyclotomicRing<F, D>> {
        (0..len)
            .map(|i| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|j| {
                    F::from_i64(((i + j) as i64 % 5) - 2)
                }))
            })
            .collect()
    }

    #[test]
    fn sis_secure_rejects_non_positive_norm() {
        assert!(!sis_secure::<F, D>(4, 0.0));
        assert!(!sis_secure::<F, D>(4, -1.0));
    }

    #[test]
    fn select_config_returns_valid_ranges() {
        let witness = LabradorWitness::new(vec![row(32), row(32), row(32)]);
        let cfg = select_config::<F, D>(&witness).unwrap();
        assert!((1..=MAX_WITNESS_DIGIT_PARTS).contains(&cfg.witness_digit_parts));
        assert!(cfg.witness_digit_bits > 0);
        assert!(cfg.aux_digit_parts > 0);
        assert!(cfg.aux_digit_bits > 0);
        assert!((1..=MAX_COMMITMENT_RANK).contains(&cfg.inner_commit_rank));
        assert!((1..=MAX_COMMITMENT_RANK).contains(&cfg.outer_commit_rank));
        assert!(!cfg.tail);
    }

    #[test]
    fn handoff_estimate_is_not_worse_than_generic_plan_on_small_coeffs() {
        let witness = LabradorWitness::new(vec![row(48), row(48), row(48)]);

        let generic_plan = plan_fold::<F, D>(&witness, false).unwrap();
        let generic_estimate =
            estimate_recursive_proof_with_plan::<F, D>(&witness, &generic_plan).unwrap();
        let handoff_estimate = estimate_handoff_recursive_proof::<F, D>(&witness, 3).unwrap();

        assert!(handoff_estimate.proof_bytes <= generic_estimate.proof_bytes);
    }

    #[test]
    fn planner_can_search_more_than_two_z_parts() {
        let row_lengths = vec![512usize, 512usize, 512usize];
        let row_coeff_bits = vec![8usize; row_lengths.len()];
        let found = (20..=80).any(|exp| {
            let profile = LabradorWitnessPlanningProfile::new(
                row_lengths.clone(),
                row_coeff_bits.clone(),
                2f64.powi(exp),
                None,
            )
            .unwrap();
            plan_fold_with_profile::<F, D>(&profile, false)
                .map(|plan| plan.config.witness_digit_parts > 2)
                .unwrap_or(false)
        });
        assert!(
            found,
            "expected planner search to reach witness_digit_parts > 2"
        );
    }

    #[test]
    fn sis_estimate_rejects_invalid_inputs() {
        assert!(estimate_module_sis_euclidean::<F, D>(0, 10, 1.0).is_err());
        assert!(estimate_module_sis_euclidean::<F, D>(1, 0, 1.0).is_err());
        assert!(estimate_module_sis_euclidean::<F, D>(1, 10, 0.0).is_err());
    }

    #[test]
    fn print_profile_style_role_sis_summary() {
        type F128 = Fp128<0xfffffffffffffffffffffffffffffeed>;
        type Cfg = Fp128FullCommitmentConfig;
        const D128: usize = Cfg::D;
        const MAX_NUM_VARS: usize = 25;

        let layout = Cfg::commitment_layout(MAX_NUM_VARS).unwrap();
        let roles = [
            ("A", Cfg::N_A, layout.inner_width),
            ("B", Cfg::N_B, layout.outer_width),
            ("D", Cfg::N_D, layout.d_matrix_width),
        ];
        let collision_inf = 7.0;

        for (role, rank, width_ring_elems) in roles {
            let width_coords = width_ring_elems * D128;
            let l2_bound = (width_coords as f64).sqrt() * collision_inf;
            let heuristic_secure =
                sis_secure_with_params(rank, l2_bound, logq_bits::<F128>() as f64, D128 as f64);
            let heuristic_max_log2 = (2.0
                * (logq_bits::<F128>() as f64 * LABRADOR_LOGDELTA * D128 as f64).sqrt()
                * (rank as f64).sqrt())
            .min(logq_bits::<F128>() as f64);
            let estimate =
                estimate_module_sis_euclidean::<F128, D128>(rank, width_ring_elems, l2_bound)
                    .unwrap();

            eprintln!(
                "[labrador::config] profile-style role SIS summary: \
                 max_num_vars={MAX_NUM_VARS}, D={D128}, role={role}, r_vars={}, m_vars={}, \
                 width_ring_elems={}, width_coords={}, rank={}, \
                 log2(bound)={:.2}, heuristic_max_log2={:.2}, heuristic_secure={}, \
                 d_att={}, delta_req={:.6}, beta={}, solution_exists={}, log2(lb)={:.2}, log2(rop_bdgl16)={:.2}",
                layout.r_vars,
                layout.m_vars,
                width_ring_elems,
                width_coords,
                rank,
                l2_bound.log2(),
                heuristic_max_log2,
                heuristic_secure,
                estimate.attack_dimension,
                estimate.required_delta,
                estimate.bkz_beta,
                estimate.solution_exists,
                estimate.log2_solution_lower_bound,
                estimate.log2_rop_bdgl16,
            );

            assert_eq!(estimate.sis_dimension, rank * D128);
            assert_eq!(estimate.sis_width, width_coords);
            assert!(estimate.required_delta.is_finite());
        }
    }
}
