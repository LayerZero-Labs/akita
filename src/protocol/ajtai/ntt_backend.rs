use crate::algebra::CyclotomicRing;
use crate::algebra::{CrtNttParamSet, CyclotomicCrtNtt, MontCoeff, PrimeWidth};
use crate::error::HachiError;
use crate::protocol::ajtai::ajtai_commit::AjtaiCommitmentScheme;
use crate::protocol::ajtai::coeff::CoeffAjtaiConfig;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::{cfg_iter, CanonicalField, FieldCore};

#[cfg(feature = "parallel")]
use crate::parallel::*;

/// NTT-based Ajtai backend.
///
/// Implements the same interface as [`AjtaiCommitmentScheme`] but with cached
/// CRT+NTT mat-vec for both witness and outer commitment multiplications.
#[derive(Clone, Copy, Debug, Default)]
pub struct NttAjtaiBackend;

fn mat_vec_mul_ntt_ring_many_with_params<
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

impl<F: FieldCore + CanonicalField, const D: usize> AjtaiCommitmentScheme<F, D>
    for NttAjtaiBackend
{
    type PublicMatrix = NttSlotCache<D>;
    type Witness = [Vec<CyclotomicRing<F, D>>];
    type InnerCommitment = Vec<Vec<CyclotomicRing<F, D>>>;
    type DecomposedInnerCommitment = Vec<CyclotomicRing<F, D>>;
    type OuterCommitment = Vec<CyclotomicRing<F, D>>;
    type Params = CoeffAjtaiConfig;

    #[tracing::instrument(skip_all, name = "NttAjtaiBackend::commit_witness")]
    fn commit_witness(
        matrix: &Self::PublicMatrix,
        witness: &Self::Witness,
        _params: &Self::Params,
    ) -> Result<Self::InnerCommitment, HachiError> {
        if ntt_slot_num_rows(matrix) == 0 {
            return Ok(vec![vec![]; witness.len()]);
        }

        let out = match matrix {
            NttSlotCache::Q32 { neg, params, .. } => {
                mat_vec_mul_ntt_ring_many_with_params(neg, witness, params)
            }
            NttSlotCache::Q64 { neg, params, .. } => {
                mat_vec_mul_ntt_ring_many_with_params(neg, witness, params)
            }
            NttSlotCache::Q128 { neg, params, .. } => {
                mat_vec_mul_ntt_ring_many_with_params(neg, witness, params)
            }
        };
        Ok(out)
    }

    #[tracing::instrument(skip_all, name = "NttAjtaiBackend::commit_inner")]
    fn commit_inner(
        matrix: &Self::PublicMatrix,
        inner_commitment: &Self::InnerCommitment,
        params: &Self::Params,
    ) -> Result<(Self::DecomposedInnerCommitment, Self::OuterCommitment), HachiError> {
        let t_hat_per_row: Vec<Vec<CyclotomicRing<F, D>>> = cfg_iter!(inner_commitment)
            .map(|t| {
                crate::protocol::commitment::utils::linear::decompose_rows_with_carry(
                    t,
                    params.num_digits,
                    params.decompose_modulus,
                )
            })
            .collect();

        let t_hat: Vec<CyclotomicRing<F, D>> = t_hat_per_row.into_iter().flatten().collect();

        if ntt_slot_num_rows(matrix) == 0 || params.outer_rows == 0 {
            return Ok((t_hat.clone(), t_hat));
        }

        let one_vec = vec![t_hat.clone()];
        let u = match matrix {
            NttSlotCache::Q32 { neg, params, .. } => {
                mat_vec_mul_ntt_ring_many_with_params(neg, &one_vec, params)
            }
            NttSlotCache::Q64 { neg, params, .. } => {
                mat_vec_mul_ntt_ring_many_with_params(neg, &one_vec, params)
            }
            NttSlotCache::Q128 { neg, params, .. } => {
                mat_vec_mul_ntt_ring_many_with_params(neg, &one_vec, params)
            }
        };

        let u0 = u.into_iter().next().unwrap_or_default();
        Ok((t_hat, u0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::{Fp64, Prime128M8M4M1M0};
    use crate::protocol::commitment::utils::crt_ntt::build_ntt_slot;
    use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
    use crate::protocol::commitment::utils::linear::decompose_rows_with_carry;
    use crate::protocol::labrador::comkey::derive_extendable_comkey_matrix;
    use crate::protocol::labrador::utils::mat_vec_mul;
    use crate::FieldSampling;

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

    fn sample_random_instances<Ff: FieldCore + FieldSampling, const DD: usize>(
        seed: u8,
        rows: usize,
        cols: usize,
    ) -> Vec<Vec<CyclotomicRing<Ff, DD>>> {
        let comkey_seed = [seed; 32];
        derive_extendable_comkey_matrix::<Ff, DD>(rows, cols, &comkey_seed, b"test/random")
    }

    fn build_slot<Ff: FieldCore + CanonicalField, const DD: usize>(
        mat: &[Vec<CyclotomicRing<Ff, DD>>],
    ) -> NttSlotCache<DD> {
        let flat = FlatMatrix::from_ring_matrix(mat);
        build_ntt_slot(flat.view::<DD>()).unwrap()
    }

    #[test]
    fn test_commit_witness_manual_verification() {
        let config = get_test_config();
        let witness = sample_random_instances::<F, D>(WITNESS_SEED, NUM_WITNESS_ROWS, WITNESS_LEN);
        let matrix_a =
            sample_random_instances::<F, D>(MATRIX_A_SEED, config.inner_rows, WITNESS_LEN);
        let a_ntt = build_slot::<F, D>(&matrix_a);

        let t = <NttAjtaiBackend as AjtaiCommitmentScheme<F, D>>::commit_witness(
            &a_ntt, &witness, &config,
        )
        .unwrap();

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
        let inner_comm = sample_random_instances::<F, D>(WITNESS_SEED, NUM_WITNESS_ROWS, inner_len);

        let decomp_len = NUM_WITNESS_ROWS * inner_len * config.num_digits;
        let matrix_b =
            sample_random_instances::<F, D>(MATRIX_B_SEED, config.outer_rows, decomp_len);
        let b_ntt = build_slot::<F, D>(&matrix_b);

        let (t_hat, u) = <NttAjtaiBackend as AjtaiCommitmentScheme<F, D>>::commit_inner(
            &b_ntt,
            &inner_comm,
            &config,
        )
        .unwrap();

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
        let witness = sample_random_instances::<F, D>(WITNESS_SEED, NUM_WITNESS_ROWS, WITNESS_LEN);
        let matrix_a =
            sample_random_instances::<F, D>(MATRIX_A_SEED, config.inner_rows, WITNESS_LEN);
        let a_ntt = build_slot::<F, D>(&matrix_a);

        let decomp_len = NUM_WITNESS_ROWS * config.inner_rows * config.num_digits;
        let matrix_b =
            sample_random_instances::<F, D>(MATRIX_B_SEED, config.outer_rows, decomp_len);
        let b_ntt = build_slot::<F, D>(&matrix_b);

        let (t_hat_full, u_full) =
            <NttAjtaiBackend as AjtaiCommitmentScheme<F, D>>::two_tier_commit(
                &a_ntt, &b_ntt, &witness, &config,
            )
            .unwrap();

        let t = <NttAjtaiBackend as AjtaiCommitmentScheme<F, D>>::commit_witness(
            &a_ntt, &witness, &config,
        )
        .unwrap();
        let (t_hat_step, u_step) =
            <NttAjtaiBackend as AjtaiCommitmentScheme<F, D>>::commit_inner(&b_ntt, &t, &config)
                .unwrap();

        assert_eq!(t_hat_full, t_hat_step);
        assert_eq!(u_full, u_step);
    }

    mod fp128 {
        use super::*;

        type F128 = Prime128M8M4M1M0;
        const D128: usize = 64;

        fn fp128_config() -> CoeffAjtaiConfig {
            CoeffAjtaiConfig {
                inner_rows: 11,
                outer_rows: 3,
                num_digits: 8,
                decompose_modulus: 16,
            }
        }

        #[test]
        fn commit_witness_ntt_matches_coeff() {
            let config = fp128_config();
            let witness = sample_random_instances::<F128, D128>(WITNESS_SEED, 8, 100);
            let matrix_a =
                sample_random_instances::<F128, D128>(MATRIX_A_SEED, config.inner_rows, 100);
            let a_ntt = build_slot::<F128, D128>(&matrix_a);

            let t = <NttAjtaiBackend as AjtaiCommitmentScheme<F128, D128>>::commit_witness(
                &a_ntt, &witness, &config,
            )
            .unwrap();

            assert_eq!(t.len(), 8);
            for i in 0..8 {
                let expected = mat_vec_mul(&matrix_a, &witness[i]);
                assert_eq!(t[i], expected, "mismatch at witness row {i}");
            }
        }

        #[test]
        fn commit_inner_ntt_matches_coeff() {
            let config = fp128_config();
            let inner_len = config.inner_rows;
            let inner_comm = sample_random_instances::<F128, D128>(WITNESS_SEED, 8, inner_len);

            let decomp_len = 8 * inner_len * config.num_digits;
            let matrix_b =
                sample_random_instances::<F128, D128>(MATRIX_B_SEED, config.outer_rows, decomp_len);
            let b_ntt = build_slot::<F128, D128>(&matrix_b);

            let (t_hat, u) = <NttAjtaiBackend as AjtaiCommitmentScheme<F128, D128>>::commit_inner(
                &b_ntt,
                &inner_comm,
                &config,
            )
            .unwrap();

            let mut expected_t_hat = Vec::new();
            for row in &inner_comm {
                expected_t_hat.extend(decompose_rows_with_carry(
                    row,
                    config.num_digits,
                    config.decompose_modulus,
                ));
            }

            assert_eq!(t_hat, expected_t_hat);

            let expected_u = mat_vec_mul(&matrix_b, &expected_t_hat);
            assert_eq!(u, expected_u);
        }

        #[test]
        fn two_tier_ntt_matches_coeff() {
            let config = fp128_config();
            let witness = sample_random_instances::<F128, D128>(WITNESS_SEED, 8, 100);
            let matrix_a =
                sample_random_instances::<F128, D128>(MATRIX_A_SEED, config.inner_rows, 100);
            let a_ntt = build_slot::<F128, D128>(&matrix_a);

            let decomp_len = 8 * config.inner_rows * config.num_digits;
            let matrix_b =
                sample_random_instances::<F128, D128>(MATRIX_B_SEED, config.outer_rows, decomp_len);
            let b_ntt = build_slot::<F128, D128>(&matrix_b);

            let (ntt_t_hat, ntt_u) =
                <NttAjtaiBackend as AjtaiCommitmentScheme<F128, D128>>::two_tier_commit(
                    &a_ntt, &b_ntt, &witness, &config,
                )
                .unwrap();

            let coeff_t = crate::protocol::ajtai::coeff::CoeffAjtai::commit_witness(
                &matrix_a, &witness, &config,
            )
            .unwrap();
            let (coeff_t_hat, coeff_u) = crate::protocol::ajtai::coeff::CoeffAjtai::commit_inner(
                &matrix_b, &coeff_t, &config,
            )
            .unwrap();

            assert_eq!(ntt_t_hat, coeff_t_hat);
            assert_eq!(ntt_u, coeff_u);
        }
    }
}
