//! Ring switching logic for the Hachi PCS (Section 4.3).
//!
//! Handles the transition from the ring-based quadratic equation to field-based
//! sumcheck instances by expanding the ring elements into their coefficient
//! vectors and setting up the evaluation tables.

use crate::algebra::CyclotomicRing;
use crate::cfg_into_iter;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::norm::detect_field_modulus;
use crate::protocol::commitment::{
    CommitmentConfig, HachiCommitmentCore, HachiCommitmentLayout, HachiExpandedSetup,
    RingCommitment, RingCommitmentScheme,
};
use crate::protocol::quadratic_equation::{
    compute_m_a_streaming, compute_r_split_eq, QuadraticEquation,
};
use crate::protocol::sumcheck::eq_poly::EqPolynomial;
use crate::protocol::transcript::labels::{
    ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling};

/// Output of the ring switch protocol, containing everything needed for sumchecks.
pub struct RingSwitchOutput<F: FieldCore, const D: usize> {
    /// The witness vector w (concatenation of z and r coefficients).
    pub w: Vec<F>,
    /// Commitment to w.
    pub w_commitment: RingCommitment<F, D>,
    /// Compact evaluation table of w (all entries in [-8, 7], reordered for sumcheck).
    /// Populated by the prover; empty on the verifier side.
    pub w_evals: Vec<i8>,
    /// Evaluation table of M_alpha(x) (tau1-weighted).
    pub m_evals_x: Vec<F>,
    /// Evaluation table of alpha powers (y dimension).
    pub alpha_evals_y: Vec<F>,
    /// Number of upper variable bits.
    pub num_u: usize,
    /// Number of lower variable bits.
    pub num_l: usize,
    /// Challenge tau0 for F_0 sumcheck.
    pub tau0: Vec<F>,
    /// Challenge tau1 for F_alpha sumcheck.
    pub tau1: Vec<F>,
    /// Basis size b = 2^LOG_BASIS.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: F,
}

/// Execute the prover side of the ring switching protocol (Section 4.3).
///
/// # Errors
///
/// Returns an error if z_pre/w_hat is missing, commitment fails, or matrix expansion fails.
pub fn ring_switch_prover<F, T, const D: usize, Cfg>(
    quad_eq: &mut QuadraticEquation<F, D, Cfg>,
    setup: &HachiExpandedSetup<F, D>,
    transcript: &mut T,
) -> Result<RingSwitchOutput<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let w_hat = quad_eq
        .w_hat()
        .ok_or_else(|| HachiError::InvalidInput("missing w_hat in prover".to_string()))?;
    let z_pre = quad_eq
        .z_pre()
        .ok_or_else(|| HachiError::InvalidInput("missing z_pre in prover".to_string()))?;
    let hint = quad_eq
        .hint()
        .ok_or_else(|| HachiError::InvalidInput("missing hint in prover".to_string()))?;
    let t_hat = &hint.t_hat;

    let r = compute_r_split_eq::<F, D, Cfg>(
        setup,
        quad_eq.opening_point(),
        &quad_eq.challenges,
        w_hat,
        t_hat,
        z_pre,
        quad_eq.y(),
    )?;
    let w = build_w_coeffs::<F, D, Cfg>(w_hat, t_hat, z_pre, &r);

    let w_commitment = commit_w::<F, D, Cfg>(&w)?;
    transcript.append_serde(ABSORB_SUMCHECK_W, &w_commitment);

    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);

    let m_a = compute_m_a_streaming::<F, D, Cfg>(
        setup,
        quad_eq.opening_point(),
        &quad_eq.challenges,
        &alpha,
    )?;
    let m_a_vec = expand_m_a::<F, D, Cfg>(&m_a, alpha)?;
    let m_rows = m_row_count::<Cfg>();
    let m_cols = if m_a.is_empty() {
        0
    } else {
        m_a_vec.len() / m_a.len()
    };

    let (w_evals, num_u, num_l) = build_w_evals_compact::<F>(&w, D)?;
    let alpha_evals_y = build_alpha_evals_y(alpha, D);

    let num_sc_vars = num_u + num_l;
    let tau0 = sample_tau::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);

    let num_i = m_rows.next_power_of_two().trailing_zeros() as usize;
    let tau1 = sample_tau::<F, T>(transcript, CHALLENGE_TAU1, num_i);

    let m_evals_x = build_m_evals_x::<F>(&m_a_vec, m_rows, m_cols, &tau1);

    Ok(RingSwitchOutput {
        w,
        w_commitment,
        w_evals,
        m_evals_x,
        alpha_evals_y,
        num_u,
        num_l,
        tau0,
        tau1,
        b: 1usize << Cfg::LOG_BASIS,
        alpha,
    })
}

/// Replay the verifier side of ring switching to reconstruct evaluation tables.
///
/// # Errors
///
/// Returns an error if matrix expansion fails.
pub fn ring_switch_verifier<F, T, const D: usize, Cfg>(
    quad_eq: &QuadraticEquation<F, D, Cfg>,
    setup: &HachiExpandedSetup<F, D>,
    w: &[F],
    w_commitment: &RingCommitment<F, D>,
    transcript: &mut T,
) -> Result<RingSwitchOutput<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    transcript.append_serde(ABSORB_SUMCHECK_W, w_commitment);

    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);

    let m_a = compute_m_a_streaming::<F, D, Cfg>(
        setup,
        quad_eq.opening_point(),
        &quad_eq.challenges,
        &alpha,
    )?;
    let m_a_vec = expand_m_a::<F, D, Cfg>(&m_a, alpha)?;
    let m_rows = m_row_count::<Cfg>();
    let m_cols = if m_a.is_empty() {
        0
    } else {
        m_a_vec.len() / m_a.len()
    };

    let num_ring_elems = w.len() / D;
    let num_u = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let num_l = D.trailing_zeros() as usize;
    let alpha_evals_y = build_alpha_evals_y(alpha, D);

    let num_sc_vars = num_u + num_l;
    let tau0 = sample_tau::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);

    let num_i = m_rows.next_power_of_two().trailing_zeros() as usize;
    let tau1 = sample_tau::<F, T>(transcript, CHALLENGE_TAU1, num_i);

    let m_evals_x = build_m_evals_x::<F>(&m_a_vec, m_rows, m_cols, &tau1);

    Ok(RingSwitchOutput {
        w: w.to_vec(),
        w_commitment: w_commitment.clone(),
        w_evals: Vec::new(),
        m_evals_x,
        alpha_evals_y,
        num_u,
        num_l,
        tau0,
        tau1,
        b: 1usize << Cfg::LOG_BASIS,
        alpha,
    })
}

#[cfg(test)]
pub(crate) fn compute_r_via_poly_division<F: FieldCore + CanonicalField, const D: usize>(
    m: &[Vec<CyclotomicRing<F, D>>],
    z: &[CyclotomicRing<F, D>],
    y: &[CyclotomicRing<F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    let poly_len = 2 * D - 1;
    let out = m
        .iter()
        .zip(y.iter())
        .map(|(row, y_i)| {
            let column_contribution =
                |m_ij: &CyclotomicRing<F, D>, z_j: &CyclotomicRing<F, D>| -> Vec<F> {
                    let mut local = vec![F::zero(); poly_len];
                    if m_ij.is_zero() {
                        return local;
                    }
                    let a = m_ij.coefficients();
                    let b = z_j.coefficients();
                    let is_scalar = a[1..].iter().all(|c| c.is_zero());
                    if is_scalar {
                        let scalar = a[0];
                        for s in 0..D {
                            local[s] = scalar * b[s];
                        }
                    } else {
                        for t in 0..D {
                            for s in 0..D {
                                local[t + s] = local[t + s] + a[t] * b[s];
                            }
                        }
                    }
                    local
                };

            let pointwise_add = |mut a: Vec<F>, b: Vec<F>| -> Vec<F> {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai = *ai + *bi;
                }
                a
            };

            #[cfg(feature = "parallel")]
            let mut poly = row
                .par_iter()
                .zip(z.par_iter())
                .fold(
                    || vec![F::zero(); poly_len],
                    |acc, (m_ij, z_j)| pointwise_add(acc, column_contribution(m_ij, z_j)),
                )
                .reduce(|| vec![F::zero(); poly_len], pointwise_add);

            #[cfg(not(feature = "parallel"))]
            let mut poly = row
                .iter()
                .zip(z.iter())
                .fold(vec![F::zero(); poly_len], |acc, (m_ij, z_j)| {
                    pointwise_add(acc, column_contribution(m_ij, z_j))
                });
            let y_coeffs = y_i.coefficients();
            for k in 0..D {
                poly[k] = poly[k] - y_coeffs[k];
            }
            let mut quotient = vec![F::zero(); D];
            for k in (D..poly_len).rev() {
                let q = poly[k];
                quotient[k - D] = q;
                poly[k - D] = poly[k - D] - q;
            }
            let coeffs: [F; D] = std::array::from_fn(|k| quotient[k]);
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
struct WCommitmentConfig<const D: usize, Cfg: CommitmentConfig> {
    _cfg: std::marker::PhantomData<Cfg>,
}

impl<const D: usize, Cfg: CommitmentConfig> CommitmentConfig for WCommitmentConfig<D, Cfg> {
    const D: usize = D;
    const N_A: usize = Cfg::N_A;
    const N_B: usize = Cfg::N_B;
    const N_D: usize = Cfg::N_D;
    const LOG_BASIS: u32 = Cfg::LOG_BASIS;
    const DELTA: usize = Cfg::DELTA;
    const TAU: usize = Cfg::TAU;
    const CHALLENGE_WEIGHT: usize = Cfg::CHALLENGE_WEIGHT;

    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        let alpha = D.trailing_zeros() as usize;
        let m_vars = max_num_vars.checked_sub(alpha).ok_or_else(|| {
            HachiError::InvalidSetup("max_num_vars is smaller than alpha".to_string())
        })?;
        HachiCommitmentLayout::new::<Self>(m_vars, 0)
    }
}

fn commit_w<F, const D: usize, Cfg>(w: &[F]) -> Result<RingCommitment<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig,
{
    type WCfg<const D: usize, C> = WCommitmentConfig<D, C>;

    let ring_elems: Vec<CyclotomicRing<F, D>> = w
        .chunks(D)
        .map(|chunk| {
            let coeffs: [F; D] =
                std::array::from_fn(|i| if i < chunk.len() { chunk[i] } else { F::zero() });
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();

    let block_len = ring_elems.len().next_power_of_two().max(1);
    let mut padded = ring_elems;
    padded.resize(block_len, CyclotomicRing::<F, D>::zero());
    let m_vars = block_len.trailing_zeros() as usize;
    let max_num_vars = m_vars + D.trailing_zeros() as usize;
    let blocks = vec![padded];

    let (w_setup, _) =
        <HachiCommitmentCore as RingCommitmentScheme<F, D, WCfg<D, Cfg>>>::setup(max_num_vars)?;

    let w = <HachiCommitmentCore as RingCommitmentScheme<F, D, WCfg<D, Cfg>>>::commit_ring_blocks(
        &blocks, &w_setup,
    )?;

    Ok(w.commitment)
}

pub(crate) fn eval_ring_at<F: FieldCore, const D: usize>(r: &CyclotomicRing<F, D>, alpha: &F) -> F {
    let mut acc = F::zero();
    let mut power = F::one();
    for coeff in r.coefficients() {
        acc = acc + (*coeff * power);
        power = power * *alpha;
    }
    acc
}

pub(crate) fn r_decomp_levels<F: CanonicalField, Cfg: CommitmentConfig>() -> usize {
    let modulus = detect_field_modulus::<F>();
    let bits = 128 - (modulus.saturating_sub(1)).leading_zeros() as usize;
    let log_basis = Cfg::LOG_BASIS as usize;
    let mut levels = (bits + log_basis.saturating_sub(1)) / log_basis.max(1);
    if levels == 0 {
        levels = 1;
    }

    let b = 1u128 << Cfg::LOG_BASIS;
    let half_q = modulus / 2;
    let half_b_minus_1 = b / 2 - 1;
    let b_minus_1 = b - 1;
    let mut b_pow = 1u128;
    for _ in 0..levels {
        b_pow = b_pow.saturating_mul(b);
    }
    let max_positive = half_b_minus_1.saturating_mul((b_pow - 1) / b_minus_1);
    if max_positive < half_q {
        levels += 1;
    }

    levels
}

pub(crate) fn expand_m_a<F: CanonicalField, const D: usize, Cfg: CommitmentConfig>(
    m_a: &[Vec<F>],
    alpha: F,
) -> Result<Vec<F>, HachiError> {
    if m_a.is_empty() {
        return Ok(Vec::new());
    }
    let rows = m_a.len();
    let cols = m_a[0].len();
    if cols == 0 {
        return Ok(vec![F::zero(); rows]);
    }
    for row in m_a.iter() {
        if row.len() != cols {
            return Err(HachiError::InvalidSize {
                expected: cols,
                actual: row.len(),
            });
        }
    }

    let levels = r_decomp_levels::<F, Cfg>();
    let total_cols = cols
        .checked_add(
            rows.checked_mul(levels)
                .ok_or_else(|| HachiError::InvalidSetup("expanded M width overflow".to_string()))?,
        )
        .ok_or_else(|| HachiError::InvalidSetup("expanded M width overflow".to_string()))?;

    let base = F::from_canonical_u128_reduced(1u128 << Cfg::LOG_BASIS);
    let mut gadget_row = Vec::with_capacity(levels);
    let mut power = F::one();
    for _ in 0..levels {
        gadget_row.push(power);
        power = power * base;
    }

    let mut alpha_pow = F::one();
    for _ in 0..D {
        alpha_pow = alpha_pow * alpha;
    }
    let denom = alpha_pow + F::one();

    let mut out = vec![F::zero(); rows * total_cols];
    for (i, m_a_row) in m_a.iter().enumerate() {
        let row_start = i * total_cols;
        out[row_start..row_start + cols].copy_from_slice(m_a_row);
        let r_start = row_start + cols + i * levels;
        for (j, g) in gadget_row.iter().enumerate() {
            out[r_start + j] = -denom * *g;
        }
    }
    Ok(out)
}

/// # Errors
///
/// Returns an error if `w.len()` is not a multiple of `d`.
pub(crate) fn build_w_evals<F: FieldCore>(
    w: &[F],
    d: usize,
) -> Result<(Vec<F>, usize, usize), HachiError> {
    if d == 0 || w.len() % d != 0 {
        return Err(HachiError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let num_l = d.trailing_zeros() as usize;
    let num_ring_elems = w.len() / d;
    let num_u = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let x_len = 1usize << num_u;
    let n = x_len << num_l;

    let evals: Vec<F> = cfg_into_iter!(0..n)
        .map(|dst| {
            let x = dst & (x_len - 1);
            let y = dst >> num_u;
            let src = y + (x << num_l);
            if src < w.len() {
                w[src]
            } else {
                F::zero()
            }
        })
        .collect();
    Ok((evals, num_u, num_l))
}

/// Compact variant of `build_w_evals` returning `Vec<i8>`.
///
/// All entries in `w` must have canonical values in `[-half_b, half_b - 1]`
/// where `half_b = 2^(LOG_BASIS-1)`. This holds when `w` is produced by
/// `build_w_coeffs` (all components go through `balanced_decompose_pow2`).
pub(crate) fn build_w_evals_compact<F: FieldCore + CanonicalField>(
    w: &[F],
    d: usize,
) -> Result<(Vec<i8>, usize, usize), HachiError> {
    if d == 0 || w.len() % d != 0 {
        return Err(HachiError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let num_l = d.trailing_zeros() as usize;
    let num_ring_elems = w.len() / d;
    let num_u = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let x_len = 1usize << num_u;
    let n = x_len << num_l;

    let q = (-F::one()).to_canonical_u128() + 1;
    let half_q = q / 2;

    let evals: Vec<i8> = (0..n)
        .map(|dst| {
            let x = dst & (x_len - 1);
            let y = dst >> num_u;
            let src = y + (x << num_l);
            if src < w.len() {
                let canonical = w[src].to_canonical_u128();
                if canonical <= half_q {
                    canonical as i8
                } else {
                    (canonical as i128 - q as i128) as i8
                }
            } else {
                0i8
            }
        })
        .collect();
    Ok((evals, num_u, num_l))
}

pub(crate) fn m_row_count<Cfg: CommitmentConfig>() -> usize {
    Cfg::N_D + Cfg::N_B + 1 + 1 + Cfg::N_A
}

pub(crate) fn build_m_evals_x<F: FieldCore + CanonicalField>(
    m_a_flat: &[F],
    rows: usize,
    cols: usize,
    tau1: &[F],
) -> Vec<F> {
    let eq_tau1 = EqPolynomial::evals(tau1);
    let x_len = cols.next_power_of_two();
    cfg_into_iter!(0..x_len)
        .map(|x| {
            let mut acc = F::zero();
            for i in 0..eq_tau1.len() {
                let row_val = if i < rows && x < cols {
                    m_a_flat[i * cols + x]
                } else {
                    F::zero()
                };
                acc = acc + eq_tau1[i] * row_val;
            }
            acc
        })
        .collect()
}

pub(crate) fn build_alpha_evals_y<F: FieldCore>(alpha: F, d: usize) -> Vec<F> {
    let mut out = vec![F::zero(); d];
    let mut power = F::one();
    for val in out.iter_mut() {
        *val = power;
        power = power * alpha;
    }
    out
}

pub(crate) fn sample_tau<F: FieldCore + CanonicalField, T: Transcript<F>>(
    transcript: &mut T,
    label: &[u8],
    n: usize,
) -> Vec<F> {
    (0..n).map(|_| transcript.challenge_scalar(label)).collect()
}

pub(crate) fn build_w_coeffs<F: CanonicalField, const D: usize, Cfg: CommitmentConfig>(
    w_hat: &[Vec<CyclotomicRing<F, D>>],
    t_hat: &[Vec<CyclotomicRing<F, D>>],
    z_pre: &[CyclotomicRing<F, D>],
    r: &[CyclotomicRing<F, D>],
) -> Vec<F> {
    let levels = r_decomp_levels::<F, Cfg>();
    let r_hat: Vec<CyclotomicRing<F, D>> = r
        .iter()
        .flat_map(|ri| ri.balanced_decompose_pow2(levels, Cfg::LOG_BASIS))
        .collect();

    let w_hat_flat = w_hat.iter().flat_map(|v| v.iter());
    let t_hat_flat = t_hat.iter().flat_map(|v| v.iter());
    let z_hat_iter = z_pre
        .iter()
        .flat_map(|z_j| z_j.balanced_decompose_pow2(Cfg::TAU, Cfg::LOG_BASIS));

    let z_count = w_hat.iter().map(|v| v.len()).sum::<usize>()
        + t_hat.iter().map(|v| v.len()).sum::<usize>()
        + z_pre.len() * Cfg::TAU;
    let mut out = Vec::with_capacity((z_count + r_hat.len()) * D);
    for elem in w_hat_flat
        .chain(t_hat_flat)
        .chain(z_hat_iter.collect::<Vec<_>>().iter())
        .chain(r_hat.iter())
    {
        out.extend_from_slice(elem.coefficients());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::compute_r_via_poly_division;
    use crate::algebra::{CyclotomicRing, Fp64};
    use crate::{FieldCore, FromSmallInt};

    fn compute_r_schoolbook<F: FieldCore, const D: usize>(
        m: &[Vec<CyclotomicRing<F, D>>],
        z: &[CyclotomicRing<F, D>],
        y: &[CyclotomicRing<F, D>],
    ) -> Vec<CyclotomicRing<F, D>> {
        let poly_len = 2 * D - 1;
        m.iter()
            .zip(y.iter())
            .map(|(row, y_i)| {
                let mut poly = vec![F::zero(); poly_len];
                for (m_ij, z_j) in row.iter().zip(z.iter()) {
                    if m_ij.is_zero() {
                        continue;
                    }
                    let a = m_ij.coefficients();
                    let b = z_j.coefficients();
                    let is_scalar = a[1..].iter().all(|c| c.is_zero());
                    if is_scalar {
                        let scalar = a[0];
                        for s in 0..D {
                            poly[s] = poly[s] + scalar * b[s];
                        }
                    } else {
                        for t in 0..D {
                            for s in 0..D {
                                poly[t + s] = poly[t + s] + a[t] * b[s];
                            }
                        }
                    }
                }
                let y_coeffs = y_i.coefficients();
                for k in 0..D {
                    poly[k] = poly[k] - y_coeffs[k];
                }
                let mut quotient = vec![F::zero(); D];
                for k in (D..poly_len).rev() {
                    let q = poly[k];
                    quotient[k - D] = q;
                    poly[k - D] = poly[k - D] - q;
                }
                let coeffs: [F; D] = std::array::from_fn(|k| quotient[k]);
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect()
    }

    #[test]
    fn compute_r_matches_schoolbook_reference() {
        type F = Fp64<4294967197>;
        const D: usize = 64;

        let m: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        if (i + j) % 3 == 0 {
                            let mut coeffs = [F::zero(); D];
                            coeffs[0] = F::from_u64((i * 5 + j + 1) as u64);
                            CyclotomicRing::from_coefficients(coeffs)
                        } else {
                            let coeffs = std::array::from_fn(|k| {
                                F::from_u64((i as u64 * 1000 + j as u64 * 100 + k as u64 + 1) % 97)
                            });
                            CyclotomicRing::from_coefficients(coeffs)
                        }
                    })
                    .collect()
            })
            .collect();
        let z: Vec<CyclotomicRing<F, D>> = (0..4)
            .map(|j| {
                let coeffs =
                    std::array::from_fn(|k| F::from_u64((j as u64 * 37 + k as u64 + 5) % 89));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();
        let y: Vec<CyclotomicRing<F, D>> = (0..3)
            .map(|i| {
                let coeffs =
                    std::array::from_fn(|k| F::from_u64((i as u64 * 29 + k as u64 + 7) % 83));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();

        let expected = compute_r_schoolbook(&m, &z, &y);
        let got = compute_r_via_poly_division::<F, D>(&m, &z, &y)
            .expect("ring-switch CRT+NTT path should dispatch for D=64");
        assert_eq!(got, expected);
    }
}
