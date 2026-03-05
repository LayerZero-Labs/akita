use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::ajtai::ajtai_commit::AjtaiCommitmentScheme;
use crate::protocol::commitment::utils::linear::decompose_rows_with_carry;
use crate::protocol::labrador::utils::mat_vec_mul;
use crate::{cfg_iter, CanonicalField, FieldCore};

#[cfg(feature = "parallel")]
use crate::parallel::*;

/// A coefficient-based implementation of the Ajtai commitment scheme.
///
/// This implementation performs standard matrix multiplication on ring elements
/// without NTT transforms. Suitable for testing and reference.
#[derive(Clone, Copy, Debug, Default)]
pub struct CoeffAjtai;

/// Configuration for the CoeffAjtai scheme.
#[derive(Clone, Debug)]
pub struct CoeffAjtaiConfig {
    /// Dimension of the inner matrix A (rows).
    pub inner_rows: usize,
    /// Dimension of the outer matrix B (rows).
    pub outer_rows: usize,
    /// Number of digits in the balanced decomposition.
    pub num_digits: usize,
    /// Decomposition modulus.
    pub decompose_modulus: u32,
}

impl<F: FieldCore + CanonicalField, const D: usize> AjtaiCommitmentScheme<F, D> for CoeffAjtai {
    /// For the coefficient implementation, we pass the fully derived matrix as `PublicMatrix`.
    /// The caller is responsible for deriving it.
    type PublicMatrix = Vec<Vec<CyclotomicRing<F, D>>>;

    /// The witness is a list of rows (vectors of ring elements).
    type Witness = [Vec<CyclotomicRing<F, D>>];

    /// The inner commitment result $t$ before decomposition.
    /// Since we process multiple witness rows, this is a list of vectors $t_i$.
    type InnerCommitment = Vec<Vec<CyclotomicRing<F, D>>>;

    /// The decomposed inner commitment $\hat{t}$ (flattened).
    type DecomposedInnerCommitment = Vec<CyclotomicRing<F, D>>;

    /// The final outer commitment $u$.
    type OuterCommitment = Vec<CyclotomicRing<F, D>>;

    type Params = CoeffAjtaiConfig;

    fn commit_witness(
        matrix: &Self::PublicMatrix,
        witness: &Self::Witness,
        _params: &Self::Params,
    ) -> Result<Self::InnerCommitment, HachiError> {
        if matrix.is_empty() {
            return Ok(vec![vec![]; witness.len()]);
        }

        let max_len = matrix.first().map(|r| r.len()).unwrap_or(0);

        let t_per_row: Vec<Vec<CyclotomicRing<F, D>>> = cfg_iter!(witness)
            .map(|row| {
                let mut padded = Vec::with_capacity(max_len);
                padded.extend_from_slice(row);
                padded.resize(max_len, CyclotomicRing::<F, D>::zero());
                mat_vec_mul(matrix, &padded)
            })
            .collect();

        Ok(t_per_row)
    }

    fn commit_inner(
        matrix: &Self::PublicMatrix,
        inner_commitment: &Self::InnerCommitment,
        params: &Self::Params,
    ) -> Result<(Self::DecomposedInnerCommitment, Self::OuterCommitment), HachiError> {
        let t_hat_per_row: Vec<Vec<CyclotomicRing<F, D>>> = cfg_iter!(inner_commitment)
            .map(|t| decompose_rows_with_carry(t, params.num_digits, params.decompose_modulus))
            .collect();

        let t_hat: Vec<CyclotomicRing<F, D>> = t_hat_per_row.into_iter().flatten().collect();

        let u = if !matrix.is_empty() && params.outer_rows > 0 {
            mat_vec_mul(matrix, &t_hat)
        } else {
            t_hat.clone()
        };

        Ok((t_hat, u))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::protocol::labrador::comkey::derive_extendable_comkey_matrix;
    type F = Fp64<4294967197>;
    const D: usize = 64;
    const WITNESS_SEED: u8 = 42;
    const MATRIX_A_SEED: u8 = 43;
    const MATRIX_B_SEED: u8 = 44;

    const WITNESS_LEN: usize = 50;
    const NUM_WITNESS_ROWS: usize = 20;

    fn get_test_config() -> CoeffAjtaiConfig {
        CoeffAjtaiConfig {
            inner_rows: 40,
            outer_rows: 30,
            num_digits: 4,
            decompose_modulus: 8,
        }
    }

    fn sample_random_instances(
        seed: u8,
        rows: usize,
        cols: usize,
    ) -> Vec<Vec<CyclotomicRing<F, D>>> {
        let comkey_seed = [seed; 32];
        derive_extendable_comkey_matrix::<F, D>(rows, cols, &comkey_seed, b"test/random")
    }

    #[test]
    fn test_commit_witness_manual_verification() {
        let config = get_test_config();

        let witness = sample_random_instances(WITNESS_SEED, NUM_WITNESS_ROWS, WITNESS_LEN);
        let matrix_a = sample_random_instances(MATRIX_A_SEED, config.inner_rows, WITNESS_LEN);

        let t = CoeffAjtai::commit_witness(&matrix_a, &witness, &config).unwrap();

        assert_eq!(t.len(), NUM_WITNESS_ROWS);
        for i in 0..NUM_WITNESS_ROWS {
            let expected_t = mat_vec_mul(&matrix_a, &witness[i]);
            assert_eq!(t[i], expected_t);
        }
    }

    #[test]
    fn test_commit_inner_manual_verification() {
        let config = get_test_config();

        let inner_len = config.inner_rows;
        let inner_comm = sample_random_instances(WITNESS_SEED, NUM_WITNESS_ROWS, inner_len);

        let decomp_len = NUM_WITNESS_ROWS * inner_len * config.num_digits;
        let matrix_b = sample_random_instances(MATRIX_B_SEED, config.outer_rows, decomp_len);

        let (t_hat, u) = CoeffAjtai::commit_inner(&matrix_b, &inner_comm, &config).unwrap();

        // Non-carry digits should be small: norm <= D * (B/2)^2 where B = 2^log_basis
        let num_digits = config.num_digits;
        let half_basis = 1u128 << (config.decompose_modulus - 1);
        let max_norm = (D as u128) * half_basis * half_basis;

        for (i, poly) in t_hat.iter().enumerate() {
            let is_carry = (i % num_digits) == num_digits - 1;
            if !is_carry {
                let norm = poly.coeff_norm_sq();
                assert!(
                    norm <= max_norm,
                    "Digit {} norm {} exceeds max {}",
                    i,
                    norm,
                    max_norm
                );
            }
        }

        let mut expected_t_hat_flat = Vec::new();
        for row in &inner_comm {
            let decomposed =
                decompose_rows_with_carry(row, config.num_digits, config.decompose_modulus);
            expected_t_hat_flat.extend(decomposed);
        }
        assert_eq!(t_hat, expected_t_hat_flat);
        assert_eq!(t_hat.len(), decomp_len);

        let expected_u = mat_vec_mul(&matrix_b, &expected_t_hat_flat);
        assert_eq!(u, expected_u);
    }

    #[test]
    fn test_two_tier_commit_consistency() {
        let config = get_test_config();

        let witness = sample_random_instances(WITNESS_SEED, NUM_WITNESS_ROWS, WITNESS_LEN);
        let matrix_a = sample_random_instances(MATRIX_A_SEED, config.inner_rows, WITNESS_LEN);

        let decomp_len = NUM_WITNESS_ROWS * config.inner_rows * config.num_digits;
        let matrix_b = sample_random_instances(MATRIX_B_SEED, config.outer_rows, decomp_len);

        let (t_hat_full, u_full) =
            CoeffAjtai::two_tier_commit(&matrix_a, &matrix_b, &witness, &config).unwrap();

        let t = CoeffAjtai::commit_witness(&matrix_a, &witness, &config).unwrap();
        let (t_hat_step, u_step) = CoeffAjtai::commit_inner(&matrix_b, &t, &config).unwrap();

        assert_eq!(t_hat_full, t_hat_step);
        assert_eq!(u_full, u_step);
    }
}
