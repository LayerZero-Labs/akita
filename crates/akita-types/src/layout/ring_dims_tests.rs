use super::*;
use crate::{
    CommittedGroupParams, FoldParams, FoldSchedule, TailSegmentGroupLayout, TailSegmentLayout,
    TerminalFoldParams, TerminalResponseShape,
};
use akita_challenges::SparseChallengeConfig;

fn committed(ring_dimension: usize) -> CommittedGroupParams {
    CommittedGroupParams::params_only(
        crate::SisModulusProfileId::Q128OffsetA7F7,
        ring_dimension,
        3,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(ring_dimension.max(31)),
    )
    .with_decomp(8, 32, 2, 2, 2)
    .expect("ring-dimension test params")
}

fn schedule(root: CommittedGroupParams, terminal: CommittedGroupParams) -> FoldSchedule {
    let terminal_witness = TerminalFoldParams::from_expanded_group(terminal);
    let ring_dimension = terminal_witness.d_a();
    FoldSchedule {
        root: FoldParams {
            params: root.clone(),
            input_witness_len: root.d_a(),
            output_witness_len: ring_dimension,
        },
        recursive_folds: Vec::new(),
        terminal: TerminalFoldParams {
            fold_challenge_config: SparseChallengeConfig::pm1_only(ring_dimension.max(31)),
            response_shape: TerminalResponseShape {
                layout: TailSegmentLayout {
                    ring_dimension,
                    groups: vec![TailSegmentGroupLayout {
                        z_coords: ring_dimension,
                        e_field_elems: ring_dimension,
                        t_field_elems: ring_dimension,
                        z_linf_cap: Some(1),
                        z_payload_bytes: 1,
                        z_rice_low_bits: 0,
                    }],
                    logical_num_elems: 3 * ring_dimension,
                },
            },
            input_witness_len: ring_dimension,
            ..terminal_witness
        },
    }
}

#[test]
fn accepts_typed_root_and_terminal_ring_dimensions() {
    let schedule = schedule(committed(128), committed(64));
    validate_schedule_ring_dims(&schedule).expect("mixed dimensions are valid");
}

/// A recursive fold's shared D matrix used to be stored twice — on the fold and
/// on its witness params — and this test constructed a disagreement between the
/// two copies.
///
/// The state no longer exists: one field holds the matrix, so the mismatch is
/// unrepresentable and the runtime check that caught it has been deleted along
/// with the duplicate. The guarantee moved from validated to type-level, which is
/// strictly stronger, and there is nothing left here to construct.
#[test]
fn recursive_shared_d_matrix_has_a_single_owner() {
    let fold = FoldParams {
        params: committed(64),
        input_witness_len: 64,
        output_witness_len: 64,
    };
    // Reading the matrix through the fold accessor and through its params must
    // be the same read, because it is the same field. If a second copy is ever
    // reintroduced, these diverge and this fails.
    assert_eq!(
        fold.open_commit_matrix(),
        &fold.params.open().matrix,
        "the fold accessor must not introduce a second copy"
    );
    assert_eq!(
        fold.sparse_challenge_config(),
        fold.params.fold_challenge_config(),
        "the challenge-family accessor must not introduce a second copy"
    );
}

#[test]
fn rejects_non_power_of_two_role_dimension() {
    assert!(matches!(
        validate_role_dims(CommitmentRingDims {
            inner: 128,
            outer: 48,
            opening: 16,
        }),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn accepts_either_b_d_order_below_a() {
    for dims in [
        CommitmentRingDims {
            inner: 256,
            outer: 64,
            opening: 128,
        },
        CommitmentRingDims {
            inner: 256,
            outer: 128,
            opening: 64,
        },
    ] {
        validate_role_dims(dims).expect("B and D must not be ordered relative to each other");
    }
}

#[test]
fn rejects_b_or_d_larger_than_a() {
    for dims in [
        CommitmentRingDims {
            inner: 64,
            outer: 128,
            opening: 64,
        },
        CommitmentRingDims {
            inner: 64,
            outer: 64,
            opening: 128,
        },
    ] {
        validate_role_dims(dims).expect_err("A must remain the largest role");
    }
}

#[test]
fn common_relation_count_depends_only_on_current_roles() {
    let uniform_roles = CommitmentRingDims::uniform(128);
    assert_eq!(uniform_roles.common_relation_coeff_count(), 128);

    let mixed_roles = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 64,
    };
    assert_eq!(mixed_roles.common_relation_coeff_count(), 64);
}

#[test]
fn rejects_sub_d64_commitment_matrix_dimensions() {
    for dims in [
        CommitmentRingDims {
            inner: 32,
            outer: 32,
            opening: 32,
        },
        CommitmentRingDims {
            inner: 128,
            outer: 32,
            opening: 64,
        },
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 32,
        },
    ] {
        validate_role_dims(dims).expect_err("A/B/D dimensions below 64 must be rejected");
    }
}

#[test]
fn reduced_residue_oracle_tracks_every_admitted_commitment_dimension() {
    use jolt_field::{
        Ext2, ExtField, Field, FpExt4, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59,
    };

    fn check<F, E>(dimension: usize)
    where
        F: Field,
        E: Field + ExtField<F>,
    {
        let coefficients = (0..dimension)
            .map(|index| F::from_u64((index as u64).wrapping_mul(17).wrapping_add(11)))
            .collect::<Vec<_>>();
        let alpha = E::lift_base(F::from_u64(29));
        let mut powers = Vec::with_capacity(dimension);
        let mut power = E::one();
        for _ in 0..dimension {
            powers.push(power);
            power *= alpha;
        }
        let expected = (0..dimension)
            .map(|shift| {
                coefficients
                    .iter()
                    .enumerate()
                    .fold(E::zero(), |sum, (coefficient, &value)| {
                        let exponent = coefficient + shift;
                        let term = E::lift_base(value) * powers[exponent % dimension];
                        if exponent < dimension {
                            sum + term
                        } else {
                            sum - term
                        }
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            akita_algebra::residue_kernel(&coefficients, alpha).unwrap(),
            expected,
            "residue kernel disagrees at admitted dimension {dimension}"
        );
    }

    for dimension in SUPPORTED_COMMITMENT_RING_DIMS {
        check::<Prime32Offset99, FpExt4<Prime32Offset99>>(dimension);
        check::<Prime64Offset59, Ext2<Prime64Offset59>>(dimension);
        check::<Prime128OffsetA7F7, Prime128OffsetA7F7>(dimension);
    }
}
