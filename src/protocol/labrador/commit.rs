//! Two-tier commitment helpers for Labrador.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::linear::{
    decompose_rows_with_carry, mat_vec_mul_crt_ntt_i8_many,
};
use crate::protocol::labrador::comkey::{derive_extendable_comkey_matrix, LabradorComKeySeed};
use crate::protocol::labrador::types::{LabradorReductionConfig, LabradorWitness};
use crate::protocol::labrador::utils::{mat_vec_mul, try_centered_i8_rows};
use crate::{cfg_iter, CanonicalField, FieldCore, FieldSampling};

/// Commitment artifacts needed by downstream Labrador flows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorCommitmentArtifacts<F: FieldCore, const D: usize> {
    /// Per-row inner commitments.
    pub u_inner: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Opening-side payload (formerly `u1`).
    pub inner_opening_payload: Vec<CyclotomicRing<F, D>>,
    /// Linear-garbage-side payload (formerly `u2`).
    pub linear_garbage_payload: Vec<CyclotomicRing<F, D>>,
    /// Decomposed witness rows.
    pub decomposed_witness: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Decomposed inner commitments.
    pub decomposed_inner: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Linear garbage terms `h_{ij}` (always present in linear-only mode).
    pub linear_garbage: Vec<CyclotomicRing<F, D>>,
}

/// Commit witness rows in linear-only Labrador mode.
///
/// # Errors
///
/// Returns an error if dimensions/config are invalid.
pub fn commit_linear_only<F, const D: usize>(
    witness: &LabradorWitness<F, D>,
    config: &LabradorReductionConfig,
    comkey_seed: &LabradorComKeySeed,
) -> Result<LabradorCommitmentArtifacts<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
{
    if witness.rows().is_empty() {
        return Err(HachiError::InvalidInput(
            "cannot commit empty Labrador witness".to_string(),
        ));
    }
    if config.aux_digit_parts == 0 || config.aux_digit_bits == 0 || config.inner_commit_rank == 0 {
        return Err(HachiError::InvalidInput(
            "invalid Labrador commitment config".to_string(),
        ));
    }

    #[allow(clippy::type_complexity)]
    let per_row: Vec<(
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    )> = cfg_iter!(witness.rows())
        .map(|row| {
            let a = derive_extendable_comkey_matrix::<F, D>(
                config.inner_commit_rank,
                row.len(),
                comkey_seed,
                b"labrador/comkey/A",
            );
            let t = mat_vec_mul(&a, row);
            let inner_opening_digits =
                decompose_rows_with_carry(&t, config.aux_digit_parts, config.aux_digit_bits as u32);
            let witness_digits = decompose_rows_with_carry(
                row,
                config.witness_digit_parts,
                config.witness_digit_bits as u32,
            );
            (t, inner_opening_digits, witness_digits)
        })
        .collect();

    let mut u_inner = Vec::with_capacity(per_row.len());
    let mut decomposed_inner = Vec::with_capacity(per_row.len());
    let mut decomposed_witness = Vec::with_capacity(per_row.len());
    for (t, inner_opening_digits, witness_digits) in per_row {
        if t.is_empty() {
            return Err(HachiError::InvalidInput(
                "inner commitment row produced empty vector".to_string(),
            ));
        }
        u_inner.push(t);
        decomposed_inner.push(inner_opening_digits);
        decomposed_witness.push(witness_digits);
    }

    let mut inner_opening_digits_flat = Vec::new();
    for inner_opening_digits in &decomposed_inner {
        inner_opening_digits_flat.extend(inner_opening_digits.iter().copied());
    }

    let inner_opening_payload = if config.tail || config.outer_commit_rank == 0 {
        u_inner.iter().flat_map(|v| v.iter().copied()).collect()
    } else {
        let b = derive_extendable_comkey_matrix::<F, D>(
            config.outer_commit_rank,
            inner_opening_digits_flat.len(),
            comkey_seed,
            b"labrador/comkey/B",
        );
        mat_vec_mul(&b, &inner_opening_digits_flat)
    };

    let linear_garbage = build_linear_garbage(witness);
    let linear_garbage_payload = if config.tail || config.outer_commit_rank == 0 {
        linear_garbage.clone()
    } else {
        let b2 = derive_extendable_comkey_matrix::<F, D>(
            config.outer_commit_rank,
            linear_garbage.len(),
            comkey_seed,
            b"labrador/comkey/U2",
        );
        mat_vec_mul(&b2, &linear_garbage)
    };

    Ok(LabradorCommitmentArtifacts {
        u_inner,
        inner_opening_payload,
        linear_garbage_payload,
        decomposed_witness,
        decomposed_inner,
        linear_garbage,
    })
}

type RingVec<F, const D: usize> = Vec<CyclotomicRing<F, D>>;
type TwoTierResult<F, const D: usize> = Result<(RingVec<F, D>, RingVec<F, D>), HachiError>;
pub(crate) const OUTER_NTT_LOG_BASIS: u32 = 4;

fn max_centered_coeff_bits<F: CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
) -> usize {
    let q = (-F::one()).to_canonical_u128() + 1;
    let half_q = q / 2;
    let mut max_abs = 0u128;

    for row in rows {
        for coeff in row.coeffs.iter() {
            let canonical = coeff.to_canonical_u128();
            let signed = if canonical > half_q {
                -((q - canonical) as i128)
            } else {
                canonical as i128
            };
            let abs = signed.unsigned_abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }
    }

    if max_abs == 0 {
        1
    } else {
        (u128::BITS - max_abs.leading_zeros()) as usize
    }
}

pub(crate) fn outer_ntt_digit_levels<F: CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
) -> usize {
    let coeff_bits = max_centered_coeff_bits(rows);
    coeff_bits.div_ceil(OUTER_NTT_LOG_BASIS as usize) + 1
}

fn witness_ntt_digit_levels<F: CanonicalField, const D: usize>(
    witness: &[Vec<CyclotomicRing<F, D>>],
) -> usize {
    witness
        .iter()
        .map(|row| outer_ntt_digit_levels(row))
        .max()
        .unwrap_or(1)
}

fn pow2_field<F: FieldCore>(exp: u32) -> F {
    let two = F::one() + F::one();
    let mut acc = F::one();
    for _ in 0..exp {
        acc = acc * two;
    }
    acc
}

pub(crate) fn expand_matrix_for_i8_digits<F: FieldCore, const D: usize>(
    matrix: &[Vec<CyclotomicRing<F, D>>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let scale_step = pow2_field::<F>(log_basis);
    let mut scales = Vec::with_capacity(num_digits);
    let mut scale = F::one();
    for _ in 0..num_digits {
        scales.push(scale);
        scale = scale * scale_step;
    }

    cfg_iter!(matrix)
        .map(|row| {
            let mut expanded = Vec::with_capacity(row.len() * num_digits);
            for entry in row {
                for scale in &scales {
                    expanded.push(entry.scale(scale));
                }
            }
            expanded
        })
        .collect()
}

pub(crate) fn decompose_rows_ntt_i8_exact<F: CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<[i8; D]> {
    let mut out = Vec::with_capacity(rows.len() * num_digits);
    for row in rows {
        out.extend(row.balanced_decompose_pow2_i8(num_digits, log_basis));
    }
    out
}

#[tracing::instrument(skip_all, name = "labrador::commit_witness_ntt")]
fn commit_witness_ntt<F: FieldCore + CanonicalField, const D: usize>(
    matrix: &[Vec<CyclotomicRing<F, D>>],
    witness: &[Vec<CyclotomicRing<F, D>>],
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError> {
    if matrix.is_empty() {
        return Ok(vec![vec![]; witness.len()]);
    }

    if let Some(witness_i8) = try_centered_i8_rows(witness) {
        return mat_vec_mul_crt_ntt_i8_many(matrix, &witness_i8);
    }

    // Large decomposed witness rows can exceed the safe reconstruction range of
    // the generic ring-element NTT multiply. Re-expand them into balanced i8
    // planes and scale A by powers of two so the shared CRT backend stays exact.
    let witness_digit_levels = witness_ntt_digit_levels(witness);
    let expanded_matrix =
        expand_matrix_for_i8_digits(matrix, witness_digit_levels, OUTER_NTT_LOG_BASIS);
    let witness_digits: Vec<Vec<[i8; D]>> = cfg_iter!(witness)
        .map(|row| decompose_rows_ntt_i8_exact(row, witness_digit_levels, OUTER_NTT_LOG_BASIS))
        .collect();
    mat_vec_mul_crt_ntt_i8_many(&expanded_matrix, &witness_digits)
}

#[tracing::instrument(skip_all, name = "labrador::commit_inner_ntt")]
fn commit_inner_ntt<F: FieldCore + CanonicalField, const D: usize>(
    matrix: &[Vec<CyclotomicRing<F, D>>],
    inner_commitment: &[Vec<CyclotomicRing<F, D>>],
    num_digits: usize,
    decompose_modulus: u32,
) -> TwoTierResult<F, D> {
    let inner_opening_digits_per_row: Vec<Vec<CyclotomicRing<F, D>>> = cfg_iter!(inner_commitment)
        .map(|t| decompose_rows_with_carry(t, num_digits, decompose_modulus))
        .collect();
    let inner_opening_digits: Vec<CyclotomicRing<F, D>> =
        inner_opening_digits_per_row.into_iter().flatten().collect();

    if matrix.is_empty() {
        return Ok((inner_opening_digits.clone(), inner_opening_digits));
    }

    // The outer B-multiply sees arbitrary Labrador key coefficients times
    // decomposed carry digits. Re-expand those digits into small i8 planes and
    // expand B by powers of two so the product stays within the conservative
    // CRT range used by the shared NTT backend.
    let outer_digit_levels = outer_ntt_digit_levels(&inner_opening_digits);
    let expanded_matrix =
        expand_matrix_for_i8_digits(matrix, outer_digit_levels, OUTER_NTT_LOG_BASIS);
    let inner_opening_digits_i8 = decompose_rows_ntt_i8_exact(
        &inner_opening_digits,
        outer_digit_levels,
        OUTER_NTT_LOG_BASIS,
    );
    let u0 = mat_vec_mul_crt_ntt_i8_many(&expanded_matrix, &[inner_opening_digits_i8])?
        .into_iter()
        .next()
        .unwrap_or_default();
    Ok((inner_opening_digits, u0))
}

/// NTT-accelerated two-tier commitment: `witness → t = A·w → t̂ → u = B·t̂`.
///
/// Returns `(inner_opening_digits, payload)` where `inner_opening_digits` is
/// the flattened decomposed inner commitment
/// and `u` is the outer commitment.
///
/// # Errors
///
/// Propagates NTT or matrix shape errors.
#[tracing::instrument(skip_all, name = "labrador::ntt_two_tier_commit")]
pub fn ntt_two_tier_commit<F: FieldCore + CanonicalField, const D: usize>(
    a_mat: &[Vec<CyclotomicRing<F, D>>],
    b_mat: &[Vec<CyclotomicRing<F, D>>],
    witness: &[Vec<CyclotomicRing<F, D>>],
    num_digits: usize,
    decompose_modulus: u32,
) -> TwoTierResult<F, D> {
    let t = commit_witness_ntt(a_mat, witness)?;
    commit_inner_ntt(b_mat, &t, num_digits, decompose_modulus)
}

fn build_linear_garbage<F: FieldCore, const D: usize>(
    witness: &LabradorWitness<F, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let r = witness.rows().len();
    let pairs: Vec<(usize, usize)> = (0..r).flat_map(|i| (i..r).map(move |j| (i, j))).collect();
    cfg_iter!(pairs)
        .map(|&(i, j)| {
            let len = witness.rows()[i].len().min(witness.rows()[j].len());
            let mut acc = CyclotomicRing::<F, D>::zero();
            for k in 0..len {
                acc += witness.rows()[i][k] * witness.rows()[j][k];
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::{Fp128, Fp64};
    use crate::protocol::commitment::utils::linear::mat_vec_mul_crt_ntt_many;
    use crate::protocol::labrador::setup::LabradorSetup;
    use crate::protocol::labrador::types::LabradorReductionConfig;
    use crate::protocol::labrador::utils::mat_vec_mul;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn sample_witness() -> LabradorWitness<F, D> {
        let row = |len: usize| -> Vec<CyclotomicRing<F, D>> {
            (0..len)
                .map(|i| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|j| {
                        F::from_i64(((i + j) as i64 % 9) - 4)
                    }))
                })
                .collect()
        };
        LabradorWitness::new(vec![row(4), row(4), row(4)])
    }

    #[test]
    fn commit_linear_only_is_deterministic() {
        let witness = sample_witness();
        let cfg = LabradorReductionConfig {
            witness_digit_parts: 1,
            witness_digit_bits: 8,
            aux_digit_parts: 2,
            aux_digit_bits: 10,
            inner_commit_rank: 3,
            outer_commit_rank: 2,
            tail: false,
        };
        let seed = [3u8; 32];
        let a = commit_linear_only(&witness, &cfg, &seed).unwrap();
        let b = commit_linear_only(&witness, &cfg, &seed).unwrap();
        assert_eq!(a, b);
        assert!(
            !a.linear_garbage_payload.is_empty(),
            "linear garbage payload must exist"
        );
    }

    #[test]
    fn ntt_two_tier_commit_matches_schoolbook_fp128_non_tail() {
        type F128 = Fp128<0xfffffffffffffffffffffffffffffeed>;
        const D128: usize = 256;

        let row = |seed: i64, len: usize| -> Vec<CyclotomicRing<F128, D128>> {
            (0..len)
                .map(|j| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                        let raw = (seed + j as i64 * 7 + k as i64 * 11) % 17;
                        F128::from_i64(raw - 8)
                    }))
                })
                .collect()
        };
        let mut second_row = row(2, 36);
        second_row.resize(48, CyclotomicRing::<F128, D128>::zero());
        let witness = vec![row(1, 48), second_row];

        let cfg = LabradorReductionConfig {
            witness_digit_parts: 1,
            witness_digit_bits: 35,
            aux_digit_parts: 4,
            aux_digit_bits: 32,
            inner_commit_rank: 3,
            outer_commit_rank: 3,
            tail: false,
        };
        let comkey_seed = [17u8; 32];
        let setup = LabradorSetup::<F128, D128>::new(&cfg, witness.len(), 48, &comkey_seed);

        let t_schoolbook: Vec<Vec<CyclotomicRing<F128, D128>>> = witness
            .iter()
            .map(|row| mat_vec_mul(&setup.matrices.a_mat, row))
            .collect();
        let t_direct_ntt = mat_vec_mul_crt_ntt_many(&setup.matrices.a_mat, &witness).unwrap();
        assert_eq!(t_direct_ntt, t_schoolbook);

        let t_cached_ntt = commit_witness_ntt(&setup.matrices.a_mat, &witness).unwrap();
        assert_eq!(t_cached_ntt, t_schoolbook);

        let inner_opening_digits_schoolbook: Vec<CyclotomicRing<F128, D128>> = t_schoolbook
            .iter()
            .flat_map(|row| {
                decompose_rows_with_carry(row, cfg.aux_digit_parts, cfg.aux_digit_bits as u32)
            })
            .collect();
        let inner_opening_payload_schoolbook =
            mat_vec_mul(&setup.matrices.b_mat, &inner_opening_digits_schoolbook);

        let (inner_opening_digits_ntt, inner_opening_payload_ntt) = ntt_two_tier_commit(
            &setup.matrices.a_mat,
            &setup.matrices.b_mat,
            &witness,
            cfg.aux_digit_parts,
            cfg.aux_digit_bits as u32,
        )
        .unwrap();

        assert_eq!(inner_opening_digits_ntt, inner_opening_digits_schoolbook);
        assert_eq!(inner_opening_payload_ntt, inner_opening_payload_schoolbook);
    }

    #[test]
    fn commit_witness_ntt_matches_schoolbook_on_large_tail_digits() {
        type F128 = Fp128<0xfffffffffffffffffffffffffffffeed>;
        const D128: usize = 256;

        let row = |seed: i64, scale_exp: u32| -> Vec<CyclotomicRing<F128, D128>> {
            let scale = pow2_field::<F128>(scale_exp);
            (0..28)
                .map(|j| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                        let raw = (seed + j as i64 * 7 + k as i64 * 11) % 17;
                        F128::from_i64(raw - 8) * scale
                    }))
                })
                .collect()
        };
        let witness = vec![row(1, 35), row(2, 32), row(3, 35)];

        let cfg = LabradorReductionConfig {
            witness_digit_parts: 1,
            witness_digit_bits: 39,
            aux_digit_parts: 1,
            aux_digit_bits: 128,
            inner_commit_rank: 4,
            outer_commit_rank: 0,
            tail: true,
        };
        let comkey_seed = [23u8; 32];
        let setup = LabradorSetup::<F128, D128>::new(&cfg, witness.len(), 28, &comkey_seed);

        let t_schoolbook: Vec<Vec<CyclotomicRing<F128, D128>>> = witness
            .iter()
            .map(|row| mat_vec_mul(&setup.matrices.a_mat, row))
            .collect();
        let t_cached_ntt = commit_witness_ntt(&setup.matrices.a_mat, &witness).unwrap();

        assert_eq!(t_cached_ntt, t_schoolbook);
    }
}
