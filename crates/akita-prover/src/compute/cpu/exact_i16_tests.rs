use super::prepared_tests::{prepared, D, F};
use super::CpuBackend;
use crate::compute::plans::DenseCommitInput;
use akita_algebra::CyclotomicRing;
use akita_types::{NttCacheKey, NttTransformDomain};

#[test]
fn recursive_commit_selects_exact_i16_from_inner_basis() {
    let prepared = prepared();
    let coeffs = vec![[1i8; D], [-1i8; D]];
    let commit = |log_basis_inner| {
        CpuBackend::DEFAULT
            .recursive_witness_commit_rows(&prepared, &coeffs, 1, 2, 1, 1, log_basis_inner, Some(2))
            .expect("recursive commit rows")
    };

    assert_eq!(commit(3), commit(11));
    assert!(prepared.shared_ntt.lock().unwrap().contains_key(
        &NttCacheKey::from_matrix_shape(
            D,
            1,
            2,
            NttTransformDomain::ExactNegacyclicI16 {
                width: 2,
                rhs_abs_bound: 1 << 10,
            },
        )
        .unwrap()
    ));
}

#[test]
fn dense_coeff_commit_selects_exact_i16_from_inner_basis() {
    let prepared = prepared();
    let block = vec![
        CyclotomicRing::from_coefficients([F::one(); D]),
        CyclotomicRing::from_coefficients([F::from_i8(-1); D]),
    ];
    let commit = |log_basis_inner| {
        CpuBackend::DEFAULT
            .dense_commit_rows(
                &prepared,
                1,
                DenseCommitInput::CoeffBlocks {
                    block_slices: vec![block.as_slice()],
                    num_digits_inner: 1,
                    log_basis_inner,
                },
            )
            .expect("dense commit rows")
    };

    assert_eq!(commit(3), commit(11));
}
