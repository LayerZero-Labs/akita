//! Two-tier commitment helpers for Labrador.

use crate::algebra::ring::CyclotomicRing;
use crate::algebra::{CrtNttParamSet, CyclotomicCrtNtt, MontCoeff, PrimeWidth};
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::linear::decompose_rows_with_carry;
use crate::protocol::labrador::comkey::{derive_extendable_comkey_matrix, LabradorComKeySeed};
use crate::protocol::labrador::types::{LabradorReductionConfig, LabradorWitness};
use crate::protocol::labrador::utils::mat_vec_mul;
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

// ---------------------------------------------------------------------------
// NTT-accelerated two-tier commitment (used by Labrador fold levels)
// ---------------------------------------------------------------------------

type RingVec<F, const D: usize> = Vec<CyclotomicRing<F, D>>;
type TwoTierResult<F, const D: usize> = Result<(RingVec<F, D>, RingVec<F, D>), HachiError>;

fn mat_vec_mul_ntt_ring_many<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    vecs: &[Vec<CyclotomicRing<F, D>>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let ntt_vecs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>> = cfg_iter!(vecs)
        .map(|vec| {
            vec.iter()
                .map(|v| CyclotomicCrtNtt::from_ring_with_params(v, params))
                .collect()
        })
        .collect();

    cfg_iter!(&ntt_vecs)
        .map(|ntt_vec| {
            cfg_iter!(ntt_mat)
                .map(|row_ntt| {
                    let n = row_ntt.len().min(ntt_vec.len());
                    let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
                    for j in 0..n {
                        let prod = row_ntt[j].pointwise_mul_with_params(&ntt_vec[j], params);
                        for (k, prime) in params.primes.iter().copied().enumerate() {
                            for d in 0..D {
                                let sum = MontCoeff::from_raw(
                                    acc.limbs[k][d].raw().wrapping_add(prod.limbs[k][d].raw()),
                                );
                                acc.limbs[k][d] = prime.reduce_range(sum);
                            }
                        }
                    }
                    acc.to_ring_with_params(params)
                })
                .collect()
        })
        .collect()
}

fn ntt_slot_num_rows<const D: usize>(slot: &NttSlotCache<D>) -> usize {
    match slot {
        NttSlotCache::Q32 { neg, .. } => neg.len(),
        NttSlotCache::Q64 { neg, .. } => neg.len(),
        NttSlotCache::Q128 { neg, .. } => neg.len(),
    }
}

#[tracing::instrument(skip_all, name = "labrador::commit_witness_ntt")]
fn commit_witness_ntt<F: FieldCore + CanonicalField, const D: usize>(
    matrix: &NttSlotCache<D>,
    witness: &[Vec<CyclotomicRing<F, D>>],
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError> {
    if ntt_slot_num_rows(matrix) == 0 {
        return Ok(vec![vec![]; witness.len()]);
    }
    let out = match matrix {
        NttSlotCache::Q32 { neg, params, .. } => mat_vec_mul_ntt_ring_many(neg, witness, params),
        NttSlotCache::Q64 { neg, params, .. } => mat_vec_mul_ntt_ring_many(neg, witness, params),
        NttSlotCache::Q128 { neg, params, .. } => mat_vec_mul_ntt_ring_many(neg, witness, params),
    };
    Ok(out)
}

#[tracing::instrument(skip_all, name = "labrador::commit_inner_ntt")]
fn commit_inner_ntt<F: FieldCore + CanonicalField, const D: usize>(
    matrix: &NttSlotCache<D>,
    inner_commitment: &[Vec<CyclotomicRing<F, D>>],
    num_digits: usize,
    decompose_modulus: u32,
    outer_rows: usize,
) -> TwoTierResult<F, D> {
    let inner_opening_digits_per_row: Vec<Vec<CyclotomicRing<F, D>>> = cfg_iter!(inner_commitment)
        .map(|t| decompose_rows_with_carry(t, num_digits, decompose_modulus))
        .collect();
    let inner_opening_digits: Vec<CyclotomicRing<F, D>> =
        inner_opening_digits_per_row.into_iter().flatten().collect();

    if ntt_slot_num_rows(matrix) == 0 || outer_rows == 0 {
        return Ok((inner_opening_digits.clone(), inner_opening_digits));
    }
    let one_vec = vec![inner_opening_digits.clone()];
    let u = match matrix {
        NttSlotCache::Q32 { neg, params, .. } => mat_vec_mul_ntt_ring_many(neg, &one_vec, params),
        NttSlotCache::Q64 { neg, params, .. } => mat_vec_mul_ntt_ring_many(neg, &one_vec, params),
        NttSlotCache::Q128 { neg, params, .. } => mat_vec_mul_ntt_ring_many(neg, &one_vec, params),
    };
    let u0 = u.into_iter().next().unwrap_or_default();
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
    a_ntt: &NttSlotCache<D>,
    b_ntt: &NttSlotCache<D>,
    witness: &[Vec<CyclotomicRing<F, D>>],
    num_digits: usize,
    decompose_modulus: u32,
    outer_rows: usize,
) -> TwoTierResult<F, D> {
    let t = commit_witness_ntt(a_ntt, witness)?;
    commit_inner_ntt(b_ntt, &t, num_digits, decompose_modulus, outer_rows)
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
    use crate::algebra::fields::Fp64;
    use crate::protocol::labrador::types::LabradorReductionConfig;
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
}
