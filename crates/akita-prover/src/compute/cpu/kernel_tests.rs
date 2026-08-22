use super::prepared_tests::{prepared, D};
use super::CpuBackend;
use crate::backend::packed_digits::PackedSignedDigits;
use crate::backend::RingSwitchRelationView;
use crate::compute::backend::{CyclicRowsComputeBackend, DigitRowsComputeBackend};
use crate::compute::{RingSwitchRelationKernel, RingSwitchRelationPlan};
use crate::kernels::linear::{
    fused_split_eq_quotients_prover_bounds, mat_vec_mul_ntt_digits_i8,
    mat_vec_mul_ntt_raw_digits_i8, mat_vec_mul_ntt_single_i8, mat_vec_mul_ntt_single_i8_cyclic,
};
use akita_error::AkitaError;
use akita_types::{NttCacheKey, NttTransformDomain};

#[test]
fn cpu_digit_rows_match_direct_kernel() {
    let prepared = prepared();
    let digits = vec![[1i8; D], [-1i8; D], [2i8; D]];
    let log_basis = 3;
    let via_backend = CpuBackend::DEFAULT
        .digit_rows::<D>(&prepared, 2, &[digits.as_slice()], log_basis)
        .expect("backend digit rows");
    let direct = prepared
        .with_shared_ntt::<D, _>(
            NttCacheKey::from_matrix_shape(D, 2, digits.len(), NttTransformDomain::Negacyclic)
                .unwrap(),
            |ntt| mat_vec_mul_ntt_single_i8(ntt, 2, digits.len(), &digits, log_basis),
        )
        .expect("direct digit rows");
    assert_eq!(via_backend, vec![direct]);
}

#[test]
fn cpu_digit_rows_accept_logical_input_longer_than_stride() {
    let prepared = prepared();
    let digits = vec![[1i8; D]; 12];
    let log_basis = 3;
    let via_backend = CpuBackend::DEFAULT
        .digit_rows::<D>(&prepared, 2, &[digits.as_slice()], log_basis)
        .expect("backend digit rows");
    let direct = prepared
        .with_shared_ntt::<D, _>(
            NttCacheKey::from_matrix_shape(D, 2, digits.len(), NttTransformDomain::Negacyclic)
                .unwrap(),
            |ntt| mat_vec_mul_ntt_single_i8(ntt, 2, digits.len(), &digits, log_basis),
        )
        .expect("direct digit rows");
    assert_eq!(via_backend, vec![direct]);
}

#[test]
fn cpu_digit_row_batch_matches_independent_single_kernels() {
    let prepared = prepared();
    let first = vec![[1i8; D], [-1i8; D], [2i8; D]];
    let second = vec![[-2i8; D], [0i8; D], [3i8; D]];
    let log_basis = 3;
    let via_backend = CpuBackend::DEFAULT
        .digit_rows::<D>(
            &prepared,
            2,
            &[first.as_slice(), second.as_slice()],
            log_basis,
        )
        .expect("backend digit row batch");
    let direct = prepared
        .with_shared_ntt::<D, _>(
            NttCacheKey::from_matrix_shape(D, 2, first.len(), NttTransformDomain::Negacyclic)
                .unwrap(),
            |ntt| {
                [first.as_slice(), second.as_slice()]
                    .into_iter()
                    .map(|digits| {
                        mat_vec_mul_ntt_single_i8(ntt, 2, digits.len(), digits, log_basis)
                    })
                    .collect::<Result<Vec<_>, _>>()
            },
        )
        .expect("independent digit rows");
    assert_eq!(via_backend, direct);
}

#[test]
fn cpu_digit_row_batch_rejects_empty_or_ragged_inputs() {
    let prepared = prepared();
    let empty: [&[[i8; D]]; 0] = [];
    assert!(matches!(
        CpuBackend::DEFAULT.digit_rows::<D>(&prepared, 2, &empty, 3),
        Err(AkitaError::InvalidInput(_))
    ));

    let wide = vec![[1i8; D]; 3];
    let narrow = vec![[1i8; D]; 2];
    assert!(matches!(
        CpuBackend::DEFAULT
            .digit_rows::<D>(&prepared, 2, &[wide.as_slice(), narrow.as_slice()], 3,),
        Err(AkitaError::InvalidInput(_))
    ));
}

#[test]
fn recursive_commit_ignores_commitment_padding_blocks() {
    let prepared = prepared();
    let coeffs = vec![[1i8; D]; 6];
    let packed =
        PackedSignedDigits::from_i8_digits(coeffs.into_iter().flatten().collect(), 2).unwrap();
    let rows = CpuBackend::DEFAULT
        .recursive_packed_witness_commit_rows::<_, D>(
            &prepared,
            packed.zero_padded(6 * D).unwrap(),
            1,
            2,
            2,
            1,
            3,
            Some(3),
        )
        .expect("recursive commit rows");

    assert_eq!(rows.len(), 2);
}

#[test]
fn packed_recursive_commit_matches_predecoded_block_parallel_kernel() {
    let prepared = prepared();
    let coeffs = (0..40)
        .map(|ring| std::array::from_fn(|coefficient| ((ring + coefficient) % 7) as i8 - 3))
        .collect::<Vec<[i8; D]>>();
    let packed =
        PackedSignedDigits::from_i8_digits(coeffs.iter().flatten().copied().collect(), 3).unwrap();
    let got = CpuBackend::DEFAULT
        .recursive_packed_witness_commit_rows::<_, D>(
            &prepared,
            packed.zero_padded(coeffs.len() * D).unwrap(),
            1,
            2,
            20,
            1,
            3,
            Some(3),
        )
        .unwrap();
    let blocks = coeffs.chunks(2).collect::<Vec<_>>();
    let expected = prepared
        .with_shared_ntt::<D, _>(
            NttCacheKey::from_matrix_shape(D, 1, 2, NttTransformDomain::Negacyclic).unwrap(),
            |ntt| mat_vec_mul_ntt_digits_i8(ntt, 1, 2, &blocks, 3),
        )
        .unwrap();
    assert_eq!(got, expected);
}

#[test]
fn packed_recursive_raw_commit_matches_predecoded_kernel() {
    let prepared = prepared();
    let coeffs = (0..40)
        .map(|ring| std::array::from_fn(|coefficient| ((ring + coefficient) % 13) as i8 - 6))
        .collect::<Vec<[i8; D]>>();
    let packed =
        PackedSignedDigits::from_i8_digits(coeffs.iter().flatten().copied().collect(), 4).unwrap();
    let got = CpuBackend::DEFAULT
        .recursive_packed_witness_commit_rows::<_, D>(
            &prepared,
            packed.zero_padded(coeffs.len() * D).unwrap(),
            1,
            2,
            20,
            1,
            3,
            None,
        )
        .unwrap();
    let blocks = coeffs.chunks(2).collect::<Vec<_>>();
    let expected = prepared
        .with_shared_ntt::<D, _>(
            NttCacheKey::from_matrix_shape(D, 1, 2, NttTransformDomain::Negacyclic).unwrap(),
            |ntt| mat_vec_mul_ntt_raw_digits_i8(ntt, 1, 2, &blocks),
        )
        .unwrap();
    assert_eq!(got, expected);
}

#[test]
fn cpu_cyclic_digit_rows_match_direct_kernel() {
    let prepared = prepared();
    let digits = vec![[1i8; D], [0i8; D], [-2i8; D], [3i8; D]];
    let log_basis = 3;
    let via_backend = CpuBackend::DEFAULT
        .cyclic_digit_rows::<D>(&prepared, 2, &digits, log_basis)
        .expect("backend cyclic digit rows");
    let direct = prepared
        .with_shared_ntt::<D, _>(
            NttCacheKey::from_matrix_shape(D, 2, digits.len(), NttTransformDomain::Cyclic).unwrap(),
            |ntt| mat_vec_mul_ntt_single_i8_cyclic(ntt, 2, digits.len(), &digits, log_basis),
        )
        .expect("direct cyclic digit rows");
    assert_eq!(via_backend, direct);
}

#[test]
fn cpu_ring_switch_relation_rows_use_distinct_open_and_outer_bases() {
    let prepared = prepared();
    let e_hat = vec![[1i8; D], [-1i8; D]];
    let t_hat = vec![[-1i8; D], [3i8; D]];
    let z_segment = vec![[1i32; D], [-2i32; D], [3i32; D]];
    let via_backend = CpuBackend::DEFAULT
        .relation_rows(
            &prepared,
            RingSwitchRelationView {
                e_hat: &e_hat,
                t_hat: &t_hat,
                z_segment: &z_segment,
                z_folded_centered_inf_norm: 3,
            },
            RingSwitchRelationPlan {
                n_d: 1,
                n_b: 1,
                n_a: 1,
                log_basis_open: 2,
                log_basis_outer: 3,
            },
        )
        .expect("backend ring-switch relation rows");
    let direct = prepared
        .with_shared_ntt::<D, _>(
            NttCacheKey::from_matrix_shape(D, 1, z_segment.len(), NttTransformDomain::Cyclic)
                .unwrap(),
            |cyclic_ntt| {
                prepared.with_shared_ntt::<D, _>(
                    NttCacheKey::from_matrix_shape(
                        D,
                        1,
                        z_segment.len(),
                        NttTransformDomain::Negacyclic,
                    )
                    .unwrap(),
                    |negacyclic_ntt| {
                        fused_split_eq_quotients_prover_bounds(
                            negacyclic_ntt,
                            cyclic_ntt,
                            1,
                            1,
                            &t_hat,
                            &z_segment,
                            3,
                            3,
                        )
                    },
                )
            },
        )
        .expect("direct fused split-eq rows");
    let expected_d_negacyclic = prepared
        .with_shared_ntt::<D, _>(
            NttCacheKey::from_matrix_shape(D, 1, e_hat.len(), NttTransformDomain::Negacyclic)
                .unwrap(),
            |ntt| mat_vec_mul_ntt_single_i8(ntt, 1, e_hat.len(), &e_hat, 2),
        )
        .expect("direct D negacyclic rows");
    assert_eq!(via_backend.d_negacyclic, expected_d_negacyclic);
    assert_eq!(via_backend.b_cyclic, direct.b_cyclic);
    assert_eq!(via_backend.a_quotients, direct.a_quotients);
}
