use super::*;
use crate::{AkitaProverSetup, CommitInnerWitness};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallengeConfig;
use akita_field::Fp64;
use akita_types::{OpenCommitMatrixParams, SetupMatrixCapacity, SisModulusProfileId};

type F = Fp64<4294967197>;
const D: usize = 64;

fn inner_witness(recomposed_blocks: usize, rows_per_block: usize) -> CommitInnerWitness<F> {
    CommitInnerWitness::from_rows(vec![
        vec![CyclotomicRing::<F, D>::zero(); rows_per_block];
        recomposed_blocks
    ])
}

#[test]
fn commit_inner_shape_accepts_expected_layout() {
    let inner = inner_witness(2, 3);
    validate_commit_inner_shape::<F, D>(&inner, 2, 3).expect("shape should match");
}

#[test]
fn commit_inner_shape_rejects_bad_block_count() {
    let inner = inner_witness(1, 3);
    assert!(validate_commit_inner_shape::<F, D>(&inner, 2, 3).is_err());
}

#[test]
fn commit_inner_shape_rejects_bad_row_count() {
    let inner = inner_witness(2, 2);
    assert!(validate_commit_inner_shape::<F, D>(&inner, 2, 3).is_err());
}

#[test]
fn commit_inner_shape_accepts_many_all_zero_blocks() {
    let num_live_blocks = 1024;
    let inner = inner_witness(num_live_blocks, 3);
    validate_commit_inner_shape::<F, D>(&inner, num_live_blocks, 3).expect("all-zero blocks");
}

#[test]
fn commit_level_params_reject_log_basis_above_i8_range() {
    let expanded = AkitaProverSetup::<F>::generate_with_capacity(
        5,
        1,
        SetupMatrixCapacity {
            num_field_elements: D,
        },
    )
    .unwrap()
    .expanded;
    let params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        9,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(1),
    )
    .with_decomp(2, 4, 2, 2, 2)
    .unwrap();

    assert!(matches!(
        validate_commit_level_params::<F>(&params, &expanded),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn commit_level_params_do_not_charge_unused_shared_d_footprint() {
    let expanded = AkitaProverSetup::<F>::generate_with_capacity(
        5,
        1,
        SetupMatrixCapacity {
            num_field_elements: D,
        },
    )
    .unwrap()
    .expanded;
    let mut params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        2,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(1),
    )
    .with_decomp(1, 1, 1, 1, 1)
    .unwrap();
    let d_key = params.open_commit_matrix.sis_table_key();
    params.open_commit_matrix = OpenCommitMatrixParams::new_unchecked(
        d_key.policy,
        d_key.table_digest,
        d_key.modulus_profile,
        8,
        8,
        d_key.coeff_linf_bound,
        D,
    );

    validate_commit_level_params::<F>(&params, &expanded)
        .expect("standalone commitment only materializes A and B");
}

#[test]
fn commit_b_input_len_rejects_overflow() {
    assert_eq!(checked_commit_b_input_len(3, 5).expect("fits"), 15);
    assert!(matches!(
        checked_commit_b_input_len(usize::MAX, 2),
        Err(AkitaError::InvalidInput(_))
    ));
}

#[test]
fn outer_slice_inputs_are_polynomial_major_and_zero_padded() {
    let first = akita_types::DigitBlocks::new(vec![10, 11, 12, 13, 14], vec![1; 5], 1)
        .expect("first digit blocks");
    let second = akita_types::DigitBlocks::new(vec![20, 21, 22, 23, 24], vec![1; 5], 1)
        .expect("second digit blocks");
    let geometry = akita_types::CommitmentSliceGeometry::try_new(
        akita_types::CommitmentSliceCount::TWO,
        5,
        2,
        1,
        1,
        1,
        1,
    )
    .expect("slice geometry");

    let inputs = outer_slice_inputs::<1>(&[&first, &second], &geometry).expect("slice inputs");
    assert_eq!(
        inputs,
        vec![
            vec![[10], [11], [0], [20], [21], [0]],
            vec![[12], [13], [14], [22], [23], [24]],
        ]
    );
}
