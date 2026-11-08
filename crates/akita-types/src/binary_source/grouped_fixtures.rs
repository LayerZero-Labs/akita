//! Small deterministic resources for the grouped capacity controls.

use super::*;
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::ordinary_source::OrdinarySourceWidth;
use akita_params::sis::{
    sis_table_key_for_linf_bound, InnerCommitMatrixParams, OuterCommitMatrixParams, SisMatrixRole,
    SisTableDigest, DEFAULT_SIS_SECURITY_POLICY,
};
use akita_params::{
    BlockGeometry, CommitmentSliceCount, CommitmentSliceGeometry, GadgetDigits,
    GroupCommitPhaseParams, PolynomialGroupLayout, RoleParams, SetupPrefixSlotId,
    SisModulusProfileId,
};

/// Existing binary descriptors plus the four signed widths and slot counts.
pub fn grouped_sources_fixture() -> (
    Vec<GroupedBinarySource>,
    Vec<(GroupCommitPhaseParams, OrdinarySourceWidth)>,
) {
    let field = BinaryFieldProfile::P64;
    let carrier = binary_ordinary_profile_fixture(field, None);
    let binary = [1, 16, 128, 128]
        .into_iter()
        .enumerate()
        .map(|(group, width)| {
            let index = BinaryIndexMap::new(width, 8, 8, 2).unwrap();
            let profiles =
                BinaryCommittedGroupManifest::profiles_for_field(&index, 1, group as u64, field)
                    .unwrap();
            let descriptor =
                BinarySourceDescriptor::new(index, vec![[group as u8; 32]], profiles).unwrap();
            let manifest = BinaryCommittedGroupManifest::with_field_profile(
                descriptor,
                group as u64,
                &carrier,
                field,
            )
            .unwrap();
            let prime = PrimePlaneManifest::with_field_profile(
                manifest.descriptor(),
                (0..2).map(|point| PrimeGroupClaim {
                    point: vec![[2, 3 + point].into(); 11],
                    evaluations: vec![PrimePlaneClaim {
                        polynomial: 0,
                        value: [0, 0].into(),
                    }],
                }),
                field,
            )
            .unwrap();
            GroupedBinarySource {
                manifest,
                carrier,
                setup_group: group as u64,
                word: true,
                prime,
            }
        })
        .collect();
    let ordinary = [(2, 3), (9, 3), (18, 5), (33, 1)]
        .into_iter()
        .map(|(bits, slots)| {
            (
                binary_ordinary_profile_fixture(field, Some((bits, slots))),
                OrdinarySourceWidth::new(bits).unwrap(),
            )
        })
        .collect();
    (binary, ordinary)
}

/// Table-backed complete-cube profile for a witness or full-field setup prefix.
fn profile(domain: usize, digits: usize) -> GroupCommitPhaseParams {
    let positions = 64;
    let blocks = domain / (128 * positions);
    assert!(blocks > 0);
    let key = |role, degree| {
        sis_table_key_for_linf_bound(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q64Offset59,
            role,
            degree,
            8,
        )
        .unwrap()
    };
    let inner = InnerCommitMatrixParams::try_new_with_min_rank(
        key(SisMatrixRole::Inner, 128),
        positions * digits,
    )
    .unwrap();
    let geometry = CommitmentSliceGeometry::try_new(
        CommitmentSliceCount::ONE,
        blocks,
        1,
        inner.output_rank(),
        22,
        128,
        64,
    )
    .unwrap();
    let outer = OuterCommitMatrixParams::try_new_with_min_rank(
        key(SisMatrixRole::Outer, 64),
        geometry.physical_input_width(),
    )
    .unwrap();
    GroupCommitPhaseParams {
        version: GroupCommitPhaseParams::VERSION,
        group: PolynomialGroupLayout::singleton(domain.trailing_zeros() as usize),
        blocks: BlockGeometry::new(domain / 128, positions, blocks),
        outer_slice_count: CommitmentSliceCount::ONE,
        inner: RoleParams::new(GadgetDigits::new(3, digits), inner),
        outer: RoleParams::new(GadgetDigits::new(3, 22), outer),
    }
}

/// Bind ample, independently validated resources to a supplied fixture plan.
pub fn grouped_binding_fixture(plan: &GroupedNativeRootPlan) -> GroupedTypedPrefix {
    let natural_len = plan
        .binary()
        .iter()
        .map(|binary| {
            BinaryTypedPrefix::from_plan(&binary.source().manifest, binary.local())
                .unwrap()
                .natural_len
        })
        .sum::<usize>();
    let successor = profile(plan.repacking().domain_len(), 1);
    let first = SetupPrefixSlotId {
        natural_len,
        commitment_profile: profile(natural_len.next_power_of_two(), 22),
    };
    let mut required = akita_params::setup_prefix_slot_field_elements(&first).unwrap();
    for p in std::iter::once(&successor)
        .chain(plan.binary().iter().map(|binary| &binary.source().carrier))
        .chain(
            plan.ordinary()
                .into_iter()
                .flat_map(|batch| batch.plans().iter().map(|plan| plan.profile())),
        )
    {
        required = required.max(
            akita_params::commit_only_setup_field_elements(
                &p.inner.matrix,
                &p.outer.matrix,
                p.outer_slice_count,
            )
            .unwrap(),
        );
    }
    GroupedTypedPrefix::new(
        plan,
        successor,
        first,
        crate::AkitaSetupDescriptor {
            max_num_vars: 24,
            max_num_batched_polys: 16,
            num_field_elements: required.next_power_of_two(),
            setup_seed: [41; 32].into(),
        },
    )
    .unwrap()
}
