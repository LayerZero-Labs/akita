use super::*;
use akita_challenges::FoldChallengeFamily;
use akita_params::sis::ajtai_key::SisModulusProfileId;
use akita_params::{
    CommittedGroupParams, InnerCommitMatrixParams, OuterCommitMatrixParams, PolynomialGroupLayout,
};
use jolt_field::{One, Ring, Zero};

type F = Prime128Offset275;

fn profile(polynomials: usize) -> GroupCommitPhaseParams {
    let params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128Offset275,
        64,
        3,
        3,
        2,
        2,
        FoldChallengeFamily::pm1_only(3),
    )
    .with_decomp(2, 4, 1, 43, 43)
    .unwrap();
    let mut profile = GroupCommitPhaseParams::from_params_unchecked_for_test(
        PolynomialGroupLayout::new(8, polynomials),
        &params,
    );
    let inner = profile.inner.matrix;
    profile.inner.matrix = InnerCommitMatrixParams::try_new(
        inner.security_policy(),
        inner.sis_table_key().unwrap().table_digest,
        inner.sis_modulus_profile(),
        5,
        inner.input_width(),
        208692,
        inner.ring_dimension(),
    )
    .unwrap();
    let old = profile.outer.matrix;
    profile.outer.matrix = OuterCommitMatrixParams::try_new(
        old.security_policy(),
        old.sis_table_key().table_digest,
        old.sis_modulus_profile(),
        old.output_rank(),
        profile
            .derive_slice_geometry()
            .unwrap()
            .physical_input_width(),
        7,
        old.ring_dimension(),
    )
    .unwrap();
    profile
}

#[test]
fn ordinary_polynomials_get_disjoint_source_evaluation_rows() {
    let profile = profile(3);
    let plan = OrdinaryNativeRootPlan::new(profile, 0).unwrap();
    let repacking = RootCoefficientRepacking::new(plan.segments().to_vec(), 64).unwrap();
    assert_eq!(plan.segments()[0].elements, 12);
    assert_eq!(plan.segments()[1].elements, 30);
    assert_eq!(plan.linear_batch_count(), 30 + 2 + 2 + 3);
    let setup = AkitaSetupDescriptor {
        max_num_vars: 8,
        max_num_batched_polys: 3,
        num_field_elements: 1 << 20,
        setup_seed: [17; 32].into(),
    };
    let point = [F::zero(); 8];
    let payload = vec![F::zero(); plan.compression.terminal_coefficients()];
    for owner in 0..3 {
        let mut batches = vec![F::zero(); plan.linear_batch_count()];
        batches[34 + owner] = F::one();
        let weights = plan
            .linear_weights(
                &repacking,
                &setup,
                &point,
                &[F::zero(); 3],
                &payload,
                F::from_u64(7),
                &batches,
            )
            .unwrap();
        let mut table = vec![F::zero(); repacking.domain_len()];
        weights.check_table(&table).unwrap();
        for other in 0..3 {
            let address = repacking.coefficient_address(0, other * 4, 0, 0).unwrap();
            table[address] = F::one();
            assert_eq!(weights.check_table(&table).is_err(), owner == other);
            table[address] = F::zero();
        }
    }
}

#[test]
fn ordinary_groups_append_all_four_segments_without_aliasing() {
    let first = OrdinaryNativeRootPlan::new(profile(2), 0).unwrap();
    let second = OrdinaryNativeRootPlan::new(profile(3), 4).unwrap();
    let mut segments = first.segments().to_vec();
    segments.extend_from_slice(second.segments());
    let repacking = RootCoefficientRepacking::new(segments, 64).unwrap();
    first.validate_repacking(&repacking).unwrap();
    second.validate_repacking(&repacking).unwrap();
    assert!(repacking.segment_range(3).unwrap().end <= repacking.segment_range(4).unwrap().start);
    let negative = first.negative_binary_addresses(&repacking).unwrap();
    let second_negative = second.negative_binary_addresses(&repacking).unwrap();
    assert!(negative.iter().max().unwrap() < second_negative.iter().min().unwrap());
    assert!(OrdinaryNativeRootPlan::new(profile(1), usize::MAX).is_err());
}

#[test]
fn ordinary_partial_source_blocks_have_authenticated_zero_tails() {
    let mut profile = profile(3);
    profile.blocks.positions_per_block = 8;
    profile.blocks.live_blocks = 1;
    profile.inner.matrix = profile.inner.matrix.try_with_input_width(8).unwrap();
    let old = profile.outer.matrix;
    profile.outer.matrix = OuterCommitMatrixParams::try_new(
        old.security_policy(),
        old.sis_table_key().table_digest,
        old.sis_modulus_profile(),
        old.output_rank(),
        profile
            .derive_slice_geometry()
            .unwrap()
            .physical_input_width(),
        7,
        old.ring_dimension(),
    )
    .unwrap();
    let plan = OrdinaryNativeRootPlan::new(profile, 0).unwrap();
    assert_eq!(plan.linear_batch_count(), 15 + 2 + 2 + 3 + 3 * 256);
    let repacking = RootCoefficientRepacking::new(plan.segments().to_vec(), 64).unwrap();
    let setup = AkitaSetupDescriptor {
        max_num_vars: 8,
        max_num_batched_polys: 3,
        num_field_elements: 1 << 20,
        setup_seed: [17; 32].into(),
    };
    let mut batches = vec![F::zero(); plan.linear_batch_count()];
    *batches.last_mut().unwrap() = F::one();
    let payload = vec![F::zero(); plan.compression.terminal_coefficients()];
    let weights = plan
        .linear_weights(
            &repacking,
            &setup,
            &[F::zero(); 8],
            &[F::zero(); 3],
            &payload,
            F::from_u64(7),
            &batches,
        )
        .unwrap();
    let mut table = vec![F::zero(); repacking.domain_len()];
    weights.check_table(&table).unwrap();
    table[repacking.coefficient_address(0, 23, 0, 63).unwrap()] = F::one();
    assert!(weights.check_table(&table).is_err());
}
