#![cfg(feature = "labinius")]

mod common;

use akita_algebra::{
    binary::{BinaryField128, BinaryField192},
    MinusTrinomial, PlusTrinomial, Prime64Offset23703, TrinomialModulus,
};
use akita_error::AkitaError;
use akita_labinius_prover::{
    commit_binary_clear, commit_binary_clear_prepared, commit_kernel::pack_binary_element_i8,
    PreparedCommitMatrix,
};
use akita_labinius_verifier::{source::pack_source_column, BinaryClearSetup};
use akita_params::sis::labinius::{LabiniusCoefficientPrime, LabiniusRingDegree};
use common::{data, setup, TestHost, TestPrime};
use jolt_field::{One, Prime128OffsetA7F7, Ring, WithPacking};

fn random_equal<
    H: TestHost,
    F: TestPrime + WithPacking,
    const D: usize,
    M: TrinomialModulus + Send + Sync,
>()
where
    H::Source: Sync,
{
    for n_a in [1, 2] {
        for m in [1, 2, 4] {
            for columns in [1, 2, 4] {
                let setup = setup::<F, D, M>(n_a, m, columns);
                let prepared = PreparedCommitMatrix::prepare(&setup).unwrap();
                let (source, _, _) = data::<H>(setup.source_len(), setup.num_vars());
                let reference = commit_binary_clear::<H, F, D, M>(&setup, &source).unwrap();
                let actual =
                    commit_binary_clear_prepared::<H, F, D, M>(&prepared, &setup, &source).unwrap();
                assert_eq!(
                    actual.images, reference.images,
                    "D={D}, n_a={n_a}, m={m}, columns={columns}"
                );
            }
        }
    }
}

#[test]
fn random_p128_all_packing_ranks_and_geometries() {
    random_equal::<BinaryField128, Prime128OffsetA7F7, 162, PlusTrinomial>();
    random_equal::<BinaryField128, Prime128OffsetA7F7, 324, MinusTrinomial>();
    random_equal::<BinaryField128, Prime128OffsetA7F7, 648, MinusTrinomial>();
}

#[test]
fn random_u64_host_and_p64_coefficient_field() {
    random_equal::<BinaryField192, Prime64Offset23703, 162, PlusTrinomial>();
    random_equal::<BinaryField192, Prime64Offset23703, 324, MinusTrinomial>();
    random_equal::<BinaryField192, Prime64Offset23703, 648, MinusTrinomial>();
    random_equal::<BinaryField192, Prime128OffsetA7F7, 648, MinusTrinomial>();
}

#[test]
fn random_u128_host_and_p64_coefficient_field() {
    random_equal::<BinaryField128, Prime64Offset23703, 162, PlusTrinomial>();
    random_equal::<BinaryField128, Prime64Offset23703, 324, MinusTrinomial>();
    random_equal::<BinaryField128, Prime64Offset23703, 648, MinusTrinomial>();
}

/// One column at the benchmark geometry: 4096 ring elements accumulate into
/// each row, the longest sum the first root profile produces.
fn benchmark_column_equal<F: TestPrime + WithPacking>(n_a: usize) {
    let setup = setup::<F, 648, MinusTrinomial>(n_a, 4096, 1);
    let prepared = PreparedCommitMatrix::prepare(&setup).unwrap();
    let (source, _, _) = data::<BinaryField128>(setup.source_len(), setup.num_vars());
    assert_eq!(source.len() * 128, 2_097_152);
    let reference =
        commit_binary_clear::<BinaryField128, F, 648, MinusTrinomial>(&setup, &source).unwrap();
    let actual = commit_binary_clear_prepared::<BinaryField128, F, 648, MinusTrinomial>(
        &prepared, &setup, &source,
    )
    .unwrap();
    assert_eq!(actual.images.len(), n_a);
    assert_eq!(actual.images, reference.images, "n_a={n_a}");
}

#[test]
fn benchmark_column_p128_rank_one_matches_reference() {
    benchmark_column_equal::<Prime128OffsetA7F7>(1);
}

#[test]
fn benchmark_column_p64_rank_two_matches_reference() {
    benchmark_column_equal::<Prime64Offset23703>(2);
}

fn edge_equal<const D: usize, M: TrinomialModulus + Send + Sync>() {
    let setup = setup::<Prime128OffsetA7F7, D, M>(2, 2, 2);
    let prepared = PreparedCommitMatrix::prepare(&setup).unwrap();
    let len = setup.source_len();
    let mut cases = vec![vec![0u128; len], vec![u128::MAX; len]];
    for word in [0, len - 1] {
        for bit in [0, 127] {
            let mut source = vec![0; len];
            source[word] = 1u128 << bit;
            cases.push(source);
        }
    }
    for source in cases {
        let reference =
            commit_binary_clear::<BinaryField128, Prime128OffsetA7F7, D, M>(&setup, &source)
                .unwrap();
        let actual = commit_binary_clear_prepared::<BinaryField128, Prime128OffsetA7F7, D, M>(
            &prepared, &setup, &source,
        )
        .unwrap();
        assert_eq!(actual.images, reference.images);
    }
}

#[test]
fn edge_sources_all_packing_ranks() {
    edge_equal::<162, PlusTrinomial>();
    edge_equal::<324, MinusTrinomial>();
    edge_equal::<648, MinusTrinomial>();
}

fn packer_equal<H: TestHost, F: TestPrime, const D: usize, M: TrinomialModulus>() {
    let setup = setup::<F, D, M>(1, 4, 4);
    let (source, _, _) = data::<H>(setup.source_len(), setup.num_vars());
    for column in 0..setup.columns() {
        let packed = pack_source_column::<H, F, D, M>(&setup, &source, column).unwrap();
        let start = column * setup.scalar_rows();
        let words = &source[start..start + setup.scalar_rows()];
        for (expected, element) in packed.iter().zip(words.chunks_exact(setup.k())) {
            let digits = pack_binary_element_i8::<H, D>(element).unwrap();
            assert!(digits.iter().all(|digit| (-1..=1).contains(digit)));
            let coefficients = digits.map(|digit| {
                let magnitude = F::from_u64(u64::from(digit.unsigned_abs()));
                if digit < 0 {
                    -magnitude
                } else {
                    magnitude
                }
            });
            assert_eq!(&coefficients, expected.coefficients());
        }
    }
}

#[test]
fn signed_packer_matches_field_packing_for_both_hosts_and_primes() {
    packer_equal::<BinaryField128, Prime128OffsetA7F7, 162, PlusTrinomial>();
    packer_equal::<BinaryField128, Prime128OffsetA7F7, 324, MinusTrinomial>();
    packer_equal::<BinaryField128, Prime128OffsetA7F7, 648, MinusTrinomial>();
    packer_equal::<BinaryField192, Prime64Offset23703, 162, PlusTrinomial>();
    packer_equal::<BinaryField192, Prime64Offset23703, 324, MinusTrinomial>();
    packer_equal::<BinaryField192, Prime64Offset23703, 648, MinusTrinomial>();
}

#[test]
fn source_geometry_errors_match_reference_exactly() {
    let setup = setup::<Prime128OffsetA7F7, 648, MinusTrinomial>(2, 2, 2);
    let prepared = PreparedCommitMatrix::prepare(&setup).unwrap();
    for actual in [0, setup.source_len() - 1, setup.source_len() + 1] {
        let source = vec![0u128; actual];
        let reference =
            commit_binary_clear::<BinaryField128, Prime128OffsetA7F7, 648, MinusTrinomial>(
                &setup, &source,
            )
            .unwrap_err();
        let prepared_error = commit_binary_clear_prepared::<
            BinaryField128,
            Prime128OffsetA7F7,
            648,
            MinusTrinomial,
        >(&prepared, &setup, &source)
        .unwrap_err();
        assert_eq!(prepared_error, reference);
        assert_eq!(
            prepared_error,
            AkitaError::InvalidSize {
                expected: setup.source_len(),
                actual
            }
        );
    }
    assert!(pack_binary_element_i8::<BinaryField128, 648>(&[0; 3]).is_err());
    assert!(pack_binary_element_i8::<BinaryField128, 648>(&[0; 5]).is_err());
    assert!(pack_binary_element_i8::<BinaryField128, 486>(&[0; 3]).is_err());
}

#[test]
fn prepared_matrix_rejects_changed_matrix_or_shape_and_reuses_other_column_counts() {
    type F = Prime128OffsetA7F7;
    let original = setup::<F, 648, MinusTrinomial>(1, 2, 1);
    let prepared = PreparedCommitMatrix::prepare(&original).unwrap();
    let mut matrix = original.matrix().to_vec();
    matrix[0] += akita_algebra::TrinomialRing::one().unwrap();
    let changed = BinaryClearSetup::new(
        matrix,
        1,
        2,
        1,
        -1024,
        1024,
        128,
        common::profile(),
        LabiniusCoefficientPrime::P128OffsetA7F7,
        LabiniusRingDegree::D648,
    )
    .unwrap();
    for incompatible in [
        changed,
        setup::<F, 648, MinusTrinomial>(2, 2, 1),
        setup::<F, 648, MinusTrinomial>(1, 4, 1),
    ] {
        let source = vec![0u128; incompatible.source_len()];
        let error = commit_binary_clear_prepared::<BinaryField128, F, 648, MinusTrinomial>(
            &prepared,
            &incompatible,
            &source,
        )
        .unwrap_err();
        assert!(matches!(error, AkitaError::InvalidSetup(_)));
    }
    for columns in [1, 2, 4] {
        let compatible = setup::<F, 648, MinusTrinomial>(1, 2, columns);
        let (source, _, _) = data::<BinaryField128>(compatible.source_len(), compatible.num_vars());
        let actual = commit_binary_clear_prepared::<BinaryField128, F, 648, MinusTrinomial>(
            &prepared,
            &compatible,
            &source,
        )
        .unwrap();
        let reference =
            commit_binary_clear::<BinaryField128, F, 648, MinusTrinomial>(&compatible, &source)
                .unwrap();
        assert_eq!(actual.images, reference.images);
    }
}

#[test]
fn prepared_sizes_matrix_transform_and_minimal_table_range() {
    type F = Prime128OffsetA7F7;
    let setup = setup::<F, 648, MinusTrinomial>(2, 4, 1);
    let prepared = PreparedCommitMatrix::prepare(&setup).unwrap();
    assert_eq!(prepared.matrix_bytes(), 2 * 4 * 648 * size_of::<F>());
    assert_eq!(prepared.table_bytes(), 2 * 648 * 4 * size_of::<F>());
    assert_eq!(
        prepared.prepared_bytes(),
        prepared.matrix_bytes() + prepared.table_bytes()
    );
    assert_eq!(prepared.i8_lut().log_basis(), 2);
    let domain = prepared.domain();
    let mut workspace = domain.workspace();
    for (element, transformed) in setup.matrix().iter().zip(prepared.matrix_ntt()) {
        assert_eq!(
            transformed,
            &domain.forward_with_workspace(element, &mut workspace)
        );
    }
    let mut output = domain.zero_ntt();
    for digit in [-1i8, 0, 1] {
        domain
            .forward_i8_with_lut_into_workspace(
                &[digit; 648],
                prepared.i8_lut(),
                &mut output,
                &mut workspace,
            )
            .unwrap();
        let expected = akita_algebra::TrinomialRing::from_coefficients(
            [if digit < 0 {
                -F::one()
            } else {
                F::from_u64(digit as u64)
            }; 648],
        )
        .unwrap();
        assert_eq!(
            domain.inverse_with_workspace(&output, &mut workspace),
            expected
        );
    }
    let too_small = domain.prepare_i8_lut(1).unwrap();
    assert!(domain
        .forward_i8_with_lut_into_workspace(&[1; 648], &too_small, &mut output, &mut workspace)
        .is_err());
    for digit in [-3i8, 2] {
        assert!(domain
            .forward_i8_with_lut_into_workspace(
                &[digit; 648],
                prepared.i8_lut(),
                &mut output,
                &mut workspace
            )
            .is_err());
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_columns_equal_independent_sequential_transform_accumulation() {
    type F = Prime128OffsetA7F7;
    let setup = setup::<F, 648, MinusTrinomial>(2, 4, 4);
    let prepared = PreparedCommitMatrix::prepare(&setup).unwrap();
    let (source, _, _) = data::<BinaryField128>(setup.source_len(), setup.num_vars());
    let parallel = commit_binary_clear_prepared::<BinaryField128, F, 648, MinusTrinomial>(
        &prepared, &setup, &source,
    )
    .unwrap();
    // Ordinary field packing, field transforms and unpacked accumulation provide
    // a sequential oracle independent of the optimized source/table/packed path.
    let domain = prepared.domain();
    let mut workspace = domain.workspace();
    let mut sequential = Vec::new();
    for column in 0..setup.columns() {
        let packed =
            pack_source_column::<BinaryField128, F, 648, MinusTrinomial>(&setup, &source, column)
                .unwrap();
        let transformed: Vec<_> = packed
            .iter()
            .map(|element| domain.forward_with_workspace(element, &mut workspace))
            .collect();
        for row in prepared.matrix_ntt().chunks_exact(setup.m()) {
            let mut sum = domain.zero_ntt();
            for (matrix, source) in row.iter().zip(&transformed) {
                sum.add_assign_pointwise_mul(matrix, source);
            }
            sequential.push(domain.inverse_with_workspace(&sum, &mut workspace));
        }
    }
    assert_eq!(parallel.images, sequential);
    let reference =
        commit_binary_clear::<BinaryField128, F, 648, MinusTrinomial>(&setup, &source).unwrap();
    assert_eq!(parallel.images, reference.images);
}
