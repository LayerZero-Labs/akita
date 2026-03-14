//! Labrador parameter-selection and security checks.

use crate::error::HachiError;
use crate::protocol::commitment::utils::norm::detect_field_modulus;
use crate::protocol::labrador::types::{LabradorReductionConfig, LabradorWitness};
use crate::{CanonicalField, FieldCore};
use std::f64::consts::{E, PI};
const LABRADOR_LOGDELTA: f64 = 0.00639138757765197; // log2(1.00444)
const LABRADOR_T: f64 = 14.0;
const LABRADOR_SLACK: f64 = 2.0;
const LABRADOR_TAU1: f64 = 32.0;
const LABRADOR_TAU2: f64 = 8.0;

/// Full fold-level plan: security parameters plus witness reshaping layout.
#[derive(Debug, Clone)]
pub struct LabradorFoldPlan {
    /// Security parameters (f, b, fu, bu, kappa, kappa1, tail).
    pub config: LabradorReductionConfig,
    /// Virtual row length after nu-reshaping.
    pub nn: usize,
    /// Per-original-row split count. `0` = continuation (concatenate with next
    /// row), `>0` = boundary that terminates a group and splits it into this
    /// many virtual rows of length `nn`.
    pub nu: Vec<usize>,
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
/// including the nu-partitioning k-loop that determines the optimal virtual
/// row length `nn`.
///
/// # Errors
///
/// Returns an error if witness metadata is empty/invalid or if no secure
/// commitment ranks are found within supported bounds.
pub fn select_config<F: FieldCore + CanonicalField, const D: usize>(
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
pub fn select_config_with_mode<F: FieldCore + CanonicalField, const D: usize>(
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
/// commitment overhead fits within `1.1 × nn`, and returns the candidate with
/// the smallest carried witness size for the next transition.
///
/// # Errors
///
/// Returns an error if the witness is empty or no secure parameters exist.
#[tracing::instrument(skip_all, name = "labrador::plan_fold")]
pub fn plan_fold<F: FieldCore + CanonicalField, const D: usize>(
    witness: &LabradorWitness<F, D>,
    tail: bool,
) -> Result<LabradorFoldPlan, HachiError> {
    if witness.rows().is_empty() {
        return Err(HachiError::InvalidInput(
            "cannot select config for empty Labrador witness".to_string(),
        ));
    }

    let row_lengths: Vec<usize> = witness.rows().iter().map(|r| r.len()).collect();
    let r = row_lengths.len();
    let total_len: usize = row_lengths.iter().sum();
    if total_len == 0 {
        return Err(HachiError::InvalidInput(
            "cannot select config for zero-length Labrador witness".to_string(),
        ));
    }

    let norm_sum: f64 = witness.norm() as f64;
    let logq_bits = logq_bits::<F>();
    let logq = logq_bits as f64;
    let d = D as f64;

    // For quadratic=0: single group with boundary at last row.
    // k-loop: enumerate all secure candidates and keep the cheapest witness
    // carry-forward plan instead of the first aggressive split that passes.
    let mut last_config = None;
    let mut best_plan = None;
    let mut best_score = usize::MAX;

    for k in (1..=15usize).rev() {
        let nn = total_len.div_ceil(k);
        let rr = k;
        let rr_f = rr as f64;

        let mut varz = norm_sum / (nn as f64 * d);
        varz *= LABRADOR_TAU1 + 4.0 * LABRADOR_TAU2;
        if !varz.is_finite() || varz <= 0.0 {
            varz = 1.0;
        }

        let decompose = !tail
            && (!sis_secure_with_params(
                13,
                6.0 * LABRADOR_T
                    * LABRADOR_SLACK
                    * (2.0 * (LABRADOR_TAU1 + 4.0 * LABRADOR_TAU2) * varz * nn as f64 * d).sqrt(),
                logq,
                d,
            ) || 64.0 * varz > (1u64 << 28) as f64);

        let f: usize = if decompose { 2 } else { 1 };
        let mut b = if decompose {
            ((12.0f64.log2() + varz.log2()) / 4.0).round() as isize
        } else {
            ((12.0f64.log2() + varz.log2()) / 2.0).round() as isize
        };
        b = b.clamp(1, logq_bits as isize);

        let (fu, bu) = if tail {
            (1usize, logq_bits.max(1))
        } else {
            let fu = ((logq_bits + 2 * (b as usize) / 3) / (b as usize)).max(1);
            let bu = ((logq_bits + fu / 2) / fu).max(1);
            (fu, bu)
        };

        let fg: usize = 0; // quadratic=0

        let mut found_kappa = None;
        let mut last_normsq = 0.0f64;

        for kappa in 1..=32usize {
            let mut normsq = (2f64.powi(2 * b as i32) / 12.0 * ((f - 1) as f64)
                + varz / 2f64.powi(2 * (f - 1) as i32 * b as i32))
                * nn as f64;
            if !tail {
                let hi_exp = logq_bits as isize - (fu.saturating_sub(1) * bu) as isize;
                let hi_exp = hi_exp.max(0) as i32;
                normsq += (2f64.powi(2 * bu as i32) * ((fu - 1) as f64) + 2f64.powi(2 * hi_exp))
                    / 12.0
                    * (rr_f * kappa as f64 + (rr_f * rr_f + rr_f) / 2.0);
            }
            normsq *= d;
            last_normsq = normsq;

            if sis_secure_with_params(
                kappa,
                6.0 * LABRADOR_T
                    * LABRADOR_SLACK
                    * 2f64.powi(((f - 1) * b as usize) as i32)
                    * normsq.sqrt(),
                logq,
                d,
            ) {
                found_kappa = Some(kappa);
                break;
            }
        }

        let kappa = match found_kappa {
            Some(k) => k,
            None => {
                let c = LabradorReductionConfig {
                    f,
                    b: b as usize,
                    fu,
                    bu,
                    kappa: 32,
                    kappa1: 0,
                    tail,
                };
                last_config = Some(build_plan(c, nn, r, rr));
                continue;
            }
        };

        if tail {
            let u1len = rr * kappa;
            let u2len = 2 * rr - 1;
            let varz_log = if varz > 0.0 { varz.log2() } else { 0.0 };
            let tc = LabradorReductionConfig {
                f,
                b: b as usize,
                fu,
                bu,
                kappa,
                kappa1: 0,
                tail: true,
            };
            if kappa <= 32
                && (u1len + u2len) as f64 * logq <= 1.1 * nn as f64 * (varz_log / 2.0 + 2.05)
            {
                let score = transition_carry_ring_elems(nn, rr, &tc);
                maybe_take_better_plan(
                    &mut best_plan,
                    &mut best_score,
                    score,
                    build_plan(tc, nn, r, rr),
                );
            }
            last_config = Some(build_plan(tc, nn, r, rr));
        } else {
            let kappa1 = (1..=32usize).find(|&k1| {
                sis_secure_with_params(k1, 2.0 * LABRADOR_SLACK * last_normsq.sqrt(), logq, d)
            });
            let kappa1 = match kappa1 {
                Some(k1) => k1,
                None => {
                    let c = LabradorReductionConfig {
                        f,
                        b: b as usize,
                        fu,
                        bu,
                        kappa,
                        kappa1: 32,
                        tail: false,
                    };
                    last_config = Some(build_plan(c, nn, r, rr));
                    continue;
                }
            };

            let c = LabradorReductionConfig {
                f,
                b: b as usize,
                fu,
                bu,
                kappa,
                kappa1,
                tail: false,
            };
            let lab_m = fu * rr * kappa + (fu + fg) * (rr * rr + rr) / 2;
            if (lab_m as f64) <= 1.1 * nn as f64 {
                let score = transition_carry_ring_elems(nn, rr, &c);
                maybe_take_better_plan(
                    &mut best_plan,
                    &mut best_score,
                    score,
                    build_plan(c, nn, r, rr),
                );
            }
            last_config = Some(build_plan(c, nn, r, rr));
        }
    }

    if let Some(plan) = best_plan {
        return Ok(plan);
    }

    last_config.ok_or_else(|| {
        HachiError::InvalidInput("failed to find secure Labrador fold parameters".to_string())
    })
}

fn transition_carry_ring_elems(nn: usize, rr: usize, config: &LabradorReductionConfig) -> usize {
    let z_rows = config.f * nn;
    z_rows + rr * config.kappa * config.fu + rr * (rr + 1) / 2 * config.fu
}

fn maybe_take_better_plan(
    best_plan: &mut Option<LabradorFoldPlan>,
    best_score: &mut usize,
    score: usize,
    candidate: LabradorFoldPlan,
) {
    if score < *best_score
        || (score == *best_score
            && best_plan
                .as_ref()
                .is_none_or(|best| candidate.nu.iter().sum::<usize>() < best.nu.iter().sum()))
    {
        *best_score = score;
        *best_plan = Some(candidate);
    }
}

fn build_plan(config: LabradorReductionConfig, nn: usize, r: usize, rr: usize) -> LabradorFoldPlan {
    let mut nu = vec![0usize; r];
    if !nu.is_empty() {
        nu[r - 1] = rr;
    }
    LabradorFoldPlan { config, nn, nu }
}

/// Build a trivial fold plan (no reshaping) from a config and row lengths.
///
/// All rows keep their original lengths; `nn = max(row_lengths)` and `nu`
/// marks each row as its own virtual row.
pub fn trivial_plan(config: LabradorReductionConfig, row_lengths: &[usize]) -> LabradorFoldPlan {
    let nn = row_lengths.iter().copied().max().unwrap_or(0);
    let nu: Vec<usize> = row_lengths.iter().map(|_| 1).collect();
    LabradorFoldPlan { config, nn, nu }
}

/// Parameter selection for the Hachi→Labrador handoff witness.
///
/// Given fixed matrix dimensions `m` (rows) and `n` (columns), searches over
/// `f` (2..=8) and `kappa` (1..=32) to find the smallest SIS-secure parameter
/// set for the polynomial commitment.
///
/// # Errors
///
/// Returns an error if no secure parameter combination exists within the
/// supported bounds.
pub fn select_handoff_config<F: CanonicalField, const D: usize>(
    m: usize,
    n: usize,
) -> Result<LabradorReductionConfig, HachiError> {
    let logq = logq_bits::<F>();
    let d = D as f64;
    let mf = m as f64;
    let nf = n as f64;

    const GH_T: f64 = 14.0;
    const GH_SLACK: f64 = 2.0;
    const GH_TAU1: f64 = 32.0;
    const GH_TAU2: f64 = 8.0;

    for f in 2..=8usize {
        let b = (logq + f / 2) / f;

        let varz = 2f64.powi(2 * b as i32) / 12.0 * nf * (GH_TAU1 + 4.0 * GH_TAU2);
        let bu = ((0.25 * (12.0 * varz).log2()).round() as usize)
            .max(1)
            .min(logq);
        let fu = ((logq as f64 / bu as f64).round() as usize).max(1);

        let mut found = None;
        for kappa in 1..=32usize {
            let mut normsq =
                (2f64.powi(2 * bu as i32) / 12.0 + varz / 2f64.powi(2 * bu as i32)) * mf * f as f64;
            let hi_exp = logq as i32 - (fu.saturating_sub(1) * bu) as i32;
            normsq += (2f64.powi(2 * bu as i32) * (fu as f64 - 1.0) + 2f64.powi(2 * hi_exp.max(0)))
                / 12.0
                * (kappa as f64 + 1.0)
                * nf;
            normsq *= d;

            if sis_secure::<F, D>(
                kappa,
                6.0 * GH_T * GH_SLACK * 2f64.powi(bu as i32) * normsq.sqrt(),
            ) {
                found = Some((kappa, normsq));
                break;
            }
        }

        let (kappa, _normsq) = match found {
            Some(v) => v,
            None => continue,
        };

        let kappa1 =
            (1..=32usize).find(|&k1| sis_secure::<F, D>(k1, 2.0 * GH_SLACK * _normsq.sqrt()));
        let kappa1 = match kappa1 {
            Some(k1) => k1,
            None => continue,
        };

        return Ok(LabradorReductionConfig {
            f,
            b,
            fu,
            bu,
            kappa,
            kappa1,
            tail: false,
        });
    }

    Err(HachiError::InvalidInput(
        "select_handoff_config: no secure parameters found".to_string(),
    ))
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
        assert!(cfg.f >= 1 && cfg.f <= 2);
        assert!(cfg.b > 0);
        assert!(cfg.fu > 0);
        assert!(cfg.bu > 0);
        assert!((1..=32).contains(&cfg.kappa));
        assert!((1..=32).contains(&cfg.kappa1));
        assert!(!cfg.tail);
    }

    #[test]
    fn sis_estimate_rejects_invalid_inputs() {
        assert!(estimate_module_sis_euclidean::<F, D>(0, 10, 1.0).is_err());
        assert!(estimate_module_sis_euclidean::<F, D>(1, 0, 1.0).is_err());
        assert!(estimate_module_sis_euclidean::<F, D>(1, 10, 0.0).is_err());
    }

    #[test]
    fn print_profile_style_handoff_sis_summary() {
        type F128 = Fp128<0xfffffffffffffffffffffffffffffeed>;
        type Cfg = Fp128FullCommitmentConfig;
        const D128: usize = Cfg::D;
        const MAX_NUM_VARS: usize = 25;

        let layout = Cfg::commitment_layout(MAX_NUM_VARS).unwrap();
        let rank = Cfg::N_D + Cfg::N_B + 2 + Cfg::N_A;
        let width_ring_elems = layout.d_matrix_width
            + layout.outer_width
            + layout.inner_width * layout.num_digits_fold;
        let beta_inf = (1usize << layout.r_vars)
            * Cfg::challenge_weight_for_ring_dim(D128)
            * (1usize << (layout.log_basis - 1));
        let collision_inf = (2 * beta_inf) as f64;
        let width_coords = width_ring_elems * D128;
        let l2_bound = (width_coords as f64).sqrt() * collision_inf;

        let heuristic_secure =
            sis_secure_with_params(rank, l2_bound, logq_bits::<F128>() as f64, D128 as f64);
        let heuristic_max_log2 = (2.0
            * (logq_bits::<F128>() as f64 * LABRADOR_LOGDELTA * D128 as f64).sqrt()
            * (rank as f64).sqrt())
        .min(logq_bits::<F128>() as f64);
        let estimate =
            estimate_module_sis_euclidean::<F128, D128>(rank, width_ring_elems, l2_bound).unwrap();

        eprintln!(
            "[labrador::config] profile-style handoff SIS summary: \
             max_num_vars={MAX_NUM_VARS}, D={D128}, r_vars={}, m_vars={}, \
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
