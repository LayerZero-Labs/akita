use super::prepared_tests::{prepared, D};
use super::CpuBackend;
use crate::compute::backend::{
    CommitmentComputeBackend, CyclicRowsComputeBackend, DigitRowsComputeBackend,
    RingSwitchComputeBackend,
};
use crate::compute::plans::{RecursiveWitnessCommitRowsPlan, RingSwitchRelationRowsPlan};
use crate::kernels::linear::{
    fused_split_eq_quotients_prover_bounds, mat_vec_mul_ntt_single_i8,
    mat_vec_mul_ntt_single_i8_cyclic,
};
use akita_types::{NttCacheKey, NttTransformDomain};

#[test]
fn cpu_digit_rows_match_direct_kernel() {
    let prepared = prepared();
    let digits = vec![[1i8; D], [-1i8; D], [2i8; D]];
    let log_basis = 3;
    let via_backend = CpuBackend
        .digit_rows::<D>(&prepared, 2, &digits, log_basis)
        .expect("backend digit rows");
    let direct = prepared
        .with_shared_ntt::<D, _>(
            NttCacheKey::from_matrix_shape(D, 2, digits.len(), NttTransformDomain::Negacyclic)
                .unwrap(),
            |ntt| mat_vec_mul_ntt_single_i8(ntt, 2, digits.len(), &digits, log_basis),
        )
        .expect("direct digit rows");
    assert_eq!(via_backend, direct);
}

#[test]
fn cpu_digit_rows_accept_logical_input_longer_than_stride() {
    let prepared = prepared();
    let digits = vec![[1i8; D]; 12];
    let log_basis = 3;
    let via_backend = CpuBackend
        .digit_rows::<D>(&prepared, 2, &digits, log_basis)
        .expect("backend digit rows");
    let direct = prepared
        .with_shared_ntt::<D, _>(
            NttCacheKey::from_matrix_shape(D, 2, digits.len(), NttTransformDomain::Negacyclic)
                .unwrap(),
            |ntt| mat_vec_mul_ntt_single_i8(ntt, 2, digits.len(), &digits, log_basis),
        )
        .expect("direct digit rows");
    assert_eq!(via_backend, direct);
}

#[test]
fn recursive_commit_ignores_commitment_padding_blocks() {
    let prepared = prepared();
    let coeffs = vec![[1i8; D]; 6];
    let rows = CpuBackend
        .recursive_witness_commit_rows(
            &prepared,
            RecursiveWitnessCommitRowsPlan {
                coeffs: &coeffs,
                n_rows: 1,
                num_positions_per_block: 2,
                num_live_blocks: 2,
                num_digits_inner: 1,
                log_basis_inner: 3,
                known_balanced_log_basis: Some(3),
            },
        )
        .expect("recursive commit rows");

    assert_eq!(rows.len(), 2);
}

#[test]
fn cpu_cyclic_digit_rows_match_direct_kernel() {
    let prepared = prepared();
    let digits = vec![[1i8; D], [0i8; D], [-2i8; D], [3i8; D]];
    let log_basis = 3;
    let via_backend = CpuBackend
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
    let via_backend = CpuBackend
        .ring_switch_relation_rows::<D>(
            &prepared,
            RingSwitchRelationRowsPlan {
                n_d: 1,
                n_b: 1,
                n_a: 1,
                e_hat: &e_hat,
                t_hat: &t_hat,
                z_segment: &z_segment,
                z_folded_centered_inf_norm: 3,
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
                            1,
                            &e_hat,
                            &t_hat,
                            &z_segment,
                            3,
                            2,
                            3,
                        )
                    },
                )
            },
        )
        .expect("direct fused split-eq rows");
    assert_eq!(
        (
            via_backend.d_cyclic,
            via_backend.b_cyclic,
            via_backend.a_quotients
        ),
        direct
    );
}
