//! Schedule-level coverage for tableless three-band mixed-role roots.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_pcs::test_support::three_band_role_switch_schedule;
use akita_types::{CommitmentRingDims, SisTableDigest};

fn three_band_schedule<Root>(expected_root_d: usize) -> akita_types::FoldSchedule
where
    Root: akita_config::CommitmentConfig<Field = fp128::Field, ExtField = fp128::Field>,
{
    let schedule = three_band_role_switch_schedule::<Root, fp128::D128OneHot, fp128::D64OneHot>(
        36, 1, 128, 64,
    )
    .expect("tableless three-band schedule");

    assert_eq!(
        schedule.root.params.final_group.commitment.role_dims(),
        CommitmentRingDims {
            inner: expected_root_d,
            outer: 128,
            opening: 128,
        }
    );
    assert_eq!(
        schedule.recursive_folds[0].params.witness.role_dims(),
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        }
    );
    assert_eq!(
        schedule.recursive_folds[1].params.witness.role_dims(),
        CommitmentRingDims::uniform(64)
    );
    schedule
}

#[test]
fn tableless_d256_root_uses_offline_planner() {
    let _schedule = three_band_schedule::<fp128::D256OneHot>(256);
}

#[test]
fn d512_root_uses_additive_a_role_sis_table() {
    let schedule = three_band_schedule::<fp128::D512OneHot>(512);

    assert_eq!(
        schedule
            .root
            .params
            .final_group
            .commitment
            .inner_commit_matrix
            .sis_table_key()
            .table_digest,
        SisTableDigest::Q128_INNER_D512
    );
    let root = &schedule.root.params.final_group.commitment;
    assert_eq!(
        root.flat_field_len().expect("root flat length"),
        1usize << 36
    );
    assert_eq!(
        (
            root.inner_commit_matrix.output_rank(),
            root.outer_commit_matrix.output_rank(),
            root.open_commit_matrix.output_rank(),
        ),
        (1, 1, 1)
    );
}
