use super::prepared_tests::{prepared, D, F};
use super::CpuBackend;
use crate::backend::packed_digits::PackedSignedDigits;
use crate::compute::plans::DenseCommitInput;
use akita_algebra::CyclotomicRing;
use akita_types::sis::compute_num_digits_field_width;
use akita_types::{NttCacheKey, NttTransformDomain};
use jolt_field::{CanonicalEncoding, One, Ring};

#[test]
fn recursive_commit_selects_exact_i16_from_inner_basis() {
    run_exact_i16_test(recursive_commit_selects_exact_i16_from_inner_basis_inner);
}

fn recursive_commit_selects_exact_i16_from_inner_basis_inner() {
    let prepared = prepared();
    let coeffs = vec![[1i8; D], [-1i8; D]];
    let packed =
        PackedSignedDigits::from_i8_digits(coeffs.into_iter().flatten().collect(), 2).unwrap();
    let commit = |log_basis_inner| {
        CpuBackend::DEFAULT
            .recursive_packed_witness_commit_rows::<_, D>(
                &prepared,
                packed.zero_padded(2 * D).unwrap(),
                1,
                2,
                1,
                1,
                log_basis_inner,
            )
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
    run_exact_i16_test(dense_coeff_commit_selects_exact_i16_from_inner_basis_inner);
}

fn dense_coeff_commit_selects_exact_i16_from_inner_basis_inner() {
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

#[test]
fn dense_i16_commit_matches_schoolbook_composition() {
    run_exact_i16_test(dense_i16_commit_matches_schoolbook_composition_inner);
}

fn run_exact_i16_test(test: fn()) {
    std::thread::Builder::new()
        .name("dense-i16-schoolbook-test".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(test)
        .expect("spawn dense i16 test")
        .join()
        .expect("dense i16 test thread");
}

fn dense_i16_commit_matches_schoolbook_composition_inner() {
    let prepared = prepared();
    let mut state = 0x8f3d_71a5_c29b_4e67u64;
    let block: Vec<_> = (0..2)
        .map(|_| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                F::from_u64(state)
            }))
        })
        .collect();
    let n_a = 2;

    for log_basis_inner in [9, 10, 11] {
        let num_digits_inner =
            compute_num_digits_field_width(<F as CanonicalEncoding>::MODULUS_BITS, log_basis_inner);
        let row_width = block.len() * num_digits_inner;
        let actual = CpuBackend::DEFAULT
            .dense_commit_rows(
                &prepared,
                n_a,
                DenseCommitInput::CoeffBlocks {
                    block_slices: vec![block.as_slice()],
                    num_digits_inner,
                    log_basis_inner,
                },
            )
            .expect("dense i16 commit");

        let mut digit_planes = vec![[0i16; D]; row_width];
        for (ring_index, ring) in block.iter().enumerate() {
            let start = ring_index * num_digits_inner;
            ring.balanced_decompose_pow2_i16_into(
                &mut digit_planes[start..start + num_digits_inner],
                log_basis_inner,
            );
        }
        let matrix = prepared
            .expanded
            .shared_matrix()
            .ring_view::<D>(n_a, row_width)
            .expect("setup matrix view");
        let expected: Vec<_> = matrix
            .rows()
            .map(|row| {
                row.iter().zip(&digit_planes).fold(
                    CyclotomicRing::zero(),
                    |acc, (matrix_entry, digit_plane)| {
                        let digit_ring = CyclotomicRing::from_coefficients(
                            digit_plane.map(|digit| F::from_i64(i64::from(digit))),
                        );
                        acc + *matrix_entry * digit_ring
                    },
                )
            })
            .collect();
        assert_eq!(actual, vec![expected], "basis {log_basis_inner}");
    }
}
