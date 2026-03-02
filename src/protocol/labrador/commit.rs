//! Two-tier Ajtai commitment helpers for Labrador (linear-only mode).

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::commitment::utils::linear::decompose_rows;
use crate::protocol::labrador::comkey::{derive_extendable_comkey_matrix, LabradorComKeySeed};
use crate::protocol::labrador::types::{LabradorReductionConfig, LabradorWitness};
use crate::protocol::prg::MatrixPrgBackendChoice;
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
    backend: MatrixPrgBackendChoice,
) -> Result<LabradorCommitmentArtifacts<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
{
    if witness.rows.is_empty() {
        return Err(HachiError::InvalidInput(
            "cannot commit empty Labrador witness".to_string(),
        ));
    }
    if config.fu == 0 || config.bu == 0 || config.kappa == 0 {
        return Err(HachiError::InvalidInput(
            "invalid Labrador commitment config".to_string(),
        ));
    }

    let mut decomposed_witness = Vec::with_capacity(witness.rows.len());
    let mut u_inner = Vec::with_capacity(witness.rows.len());
    let mut decomposed_inner = Vec::with_capacity(witness.rows.len());

    for (row_idx, row) in witness.rows.iter().enumerate() {
        let a = derive_extendable_comkey_matrix::<F, D>(
            config.kappa,
            row.s.len(),
            comkey_seed,
            b"labrador/comkey/A",
            backend,
        );
        let t = mat_vec_mul(&a, &row.s);
        if t.is_empty() {
            return Err(HachiError::InvalidInput(format!(
                "inner commitment row {row_idx} produced empty vector"
            )));
        }
        let t_hat = decompose_rows(&t, config.fu, config.bu as u32);
        let s_hat = decompose_rows(&row.s, config.f, config.b as u32);
        decomposed_witness.push(s_hat);
        decomposed_inner.push(t_hat);
        u_inner.push(t);
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
            backend,
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
            backend,
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
    let mut out =
        Vec::with_capacity((witness.rows.len() * witness.rows.len() + witness.rows.len()) / 2);
    for i in 0..witness.rows.len() {
        for j in i..witness.rows.len() {
            let len = witness.rows[i].s.len().min(witness.rows[j].s.len());
            let mut acc = CyclotomicRing::<F, D>::zero();
            for k in 0..len {
                acc += witness.rows[i].s[k] * witness.rows[j].s[k];
            }
            out.push(acc);
        }
    }
    out
}

pub(crate) fn mat_vec_mul<F: FieldCore, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    mat.iter()
        .map(|row| {
            debug_assert_eq!(row.len(), vec.len());
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (a, x) in row.iter().zip(vec.iter()) {
                acc += *a * *x;
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::protocol::labrador::types::{LabradorReductionConfig, LabradorWitnessRow};
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn sample_witness() -> LabradorWitness<F, D> {
        let row = |len: usize| LabradorWitnessRow {
            s: (0..len)
                .map(|i| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|j| {
                        F::from_i64(((i + j) as i64 % 9) - 4)
                    }))
                })
                .collect(),
            norm_sq: 1000,
        };
        LabradorWitness {
            rows: vec![row(3), row(2), row(4)],
        }
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
        let a =
            commit_linear_only(&witness, &cfg, &seed, MatrixPrgBackendChoice::Shake256).unwrap();
        let b =
            commit_linear_only(&witness, &cfg, &seed, MatrixPrgBackendChoice::Shake256).unwrap();
        assert_eq!(a, b);
        assert!(!a.u2.is_empty(), "linear garbage commitment u2 must exist");
    }
}
