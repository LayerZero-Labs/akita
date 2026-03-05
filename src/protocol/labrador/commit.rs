//! Two-tier Ajtai commitment helpers for Labrador (linear-only mode).

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::linear::decompose_rows_with_carry;
use crate::protocol::labrador::comkey::{derive_extendable_comkey_matrix, LabradorComKeySeed};
use crate::protocol::labrador::types::{LabradorReductionConfig, LabradorWitness};
use crate::protocol::labrador::utils::mat_vec_mul;
use crate::{CanonicalField, FieldCore, FieldSampling};

/// Commitment artifacts needed by downstream Labrador/Greyhound flows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorCommitmentArtifacts<F: FieldCore, const D: usize> {
    /// Per-row inner commitments.
    pub u_inner: Vec<Vec<CyclotomicRing<F, D>>>,
    /// First outer commitment (`u1`).
    pub u1: Vec<CyclotomicRing<F, D>>,
    /// Second outer commitment (`u2`) from linear garbage terms.
    pub u2: Vec<CyclotomicRing<F, D>>,
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
    if config.fu == 0 || config.bu == 0 || config.kappa == 0 {
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
                config.kappa,
                row.len(),
                comkey_seed,
                b"labrador/comkey/A",
            );
            let t = mat_vec_mul(&a, row);
            let t_hat = decompose_rows_with_carry(&t, config.fu, config.bu as u32);
            let s_hat = decompose_rows_with_carry(row, config.f, config.b as u32);
            (t, t_hat, s_hat)
        })
        .collect();

    let mut u_inner = Vec::with_capacity(per_row.len());
    let mut decomposed_inner = Vec::with_capacity(per_row.len());
    let mut decomposed_witness = Vec::with_capacity(per_row.len());
    for (t, t_hat, s_hat) in per_row {
        if t.is_empty() {
            return Err(HachiError::InvalidInput(
                "inner commitment row produced empty vector".to_string(),
            ));
        }
        u_inner.push(t);
        decomposed_inner.push(t_hat);
        decomposed_witness.push(s_hat);
    }

    let mut t_hat_flat = Vec::new();
    for t_hat in &decomposed_inner {
        t_hat_flat.extend(t_hat.iter().copied());
    }

    let u1 = if config.tail || config.kappa1 == 0 {
        u_inner.iter().flat_map(|v| v.iter().copied()).collect()
    } else {
        let b = derive_extendable_comkey_matrix::<F, D>(
            config.kappa1,
            t_hat_flat.len(),
            comkey_seed,
            b"labrador/comkey/B",
        );
        mat_vec_mul(&b, &t_hat_flat)
    };

    let linear_garbage = build_linear_garbage(witness);
    let u2 = if config.tail || config.kappa1 == 0 {
        linear_garbage.clone()
    } else {
        let b2 = derive_extendable_comkey_matrix::<F, D>(
            config.kappa1,
            linear_garbage.len(),
            comkey_seed,
            b"labrador/comkey/U2",
        );
        mat_vec_mul(&b2, &linear_garbage)
    };

    Ok(LabradorCommitmentArtifacts {
        u_inner,
        u1,
        u2,
        decomposed_witness,
        decomposed_inner,
        linear_garbage,
    })
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
            f: 1,
            b: 8,
            fu: 2,
            bu: 10,
            kappa: 3,
            kappa1: 2,
            tail: false,
        };
        let seed = [3u8; 32];
        let a = commit_linear_only(&witness, &cfg, &seed).unwrap();
        let b = commit_linear_only(&witness, &cfg, &seed).unwrap();
        assert_eq!(a, b);
        assert!(!a.u2.is_empty(), "linear garbage commitment u2 must exist");
    }
}
