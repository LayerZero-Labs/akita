//! Actual source inventory for a staged four-group p64 session.

use super::*;
use akita_params::{
    CommittedGroupBatchProfile, CommittedGroupParams, GroupOpenPhaseParams, GroupOpeningPlan,
    OpenCommitMatrixParams, OpeningMethod, SisMatrixRole,
};
use akita_schedules::native_root::NativeRootSources;

pub(crate) fn actual_fixture() -> &'static Fixture {
    static ACTUAL: OnceLock<Fixture> = OnceLock::new();
    ACTUAL.get_or_init(|| {
        let old = fixture();
        let original = &old.bindings[0].1;
        let (binary, signed) = sources();
        let plan = GroupedNativeRootPlan::new(
            &binary,
            &signed,
            original.plan().repacking().destination_ring_dimension(),
        )
        .unwrap();
        let capacity =
            GroupedPrefixCapacity::new(&plan, *original.successor(), original.first().clone())
                .unwrap();
        let plan = capacity.plan();
        let signed = plan.ordinary().unwrap();
        let profiles = CommittedGroupBatchProfile::from_profiles(
            plan.binary()
                .iter()
                .map(|source| source.source().carrier)
                .chain(signed.plans().iter().map(|source| *source.profile()))
                .collect(),
        )
        .unwrap();
        let widths = signed
            .plans()
            .iter()
            .map(|source| source.source_width().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let sources =
            NativeRootSources::new(profiles.clone(), widths, plan.identity().to_vec()).unwrap();
        let mut schedule = old
            .catalog
            .resolve_selection(old.bindings[0].0)
            .unwrap()
            .schedule()
            .clone();
        let opening = GroupOpeningPlan {
            opening_method: OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 64,
            },
            fold_challenge_config: Config::ring_challenge_config(64).unwrap(),
            log_basis_open: 3,
            num_digits_open: 22,
            num_digits_fold: 1,
        };
        let groups = profiles
            .precommitteds
            .iter()
            .chain(std::iter::once(&profiles.final_group))
            .map(|profile| GroupOpenPhaseParams {
                profile: *profile,
                setup_natural_len: None,
                opening,
            })
            .collect::<Vec<_>>();
        let d_width = groups
            .iter()
            .map(|group| group.d_segment_width(2, 64).unwrap())
            .sum::<usize>();
        let collision = akita_params::sis::rounded_up_collision_inf_norm(
            old.catalog.policy().sis_security_policy,
            old.catalog.policy().sis_modulus_profile,
            SisMatrixRole::Open,
            64,
            opening.log_basis_open,
        )
        .unwrap();
        let key = akita_params::sis::sis_table_key_for_linf_bound(
            old.catalog.policy().sis_security_policy,
            old.catalog.policy().sis_table_digest,
            old.catalog.policy().sis_modulus_profile,
            SisMatrixRole::Open,
            64,
            collision,
        )
        .unwrap();
        let root = &schedule.root.params;
        schedule.root.params = CommittedGroupParams::try_new(
            groups,
            OpenCommitMatrixParams::try_new_with_min_rank(key, d_width).unwrap(),
            root.payload_mode,
            root.ring_relation_mode,
            root.source_encoding,
            root.witness_chunk,
        )
        .unwrap();
        schedule.root.input_witness_len =
            akita_params::root_input_witness_len(&schedule.root.params);
        let transition = NativeRootTransition::new(
            schedule.root.output_witness_len,
            3,
            capacity.first().natural_len,
        )
        .unwrap()
        .with_sources(sources);
        let catalog = TrustedScheduleCatalog::new(
            ValidatedScheduleCatalog::try_new_native_root(
                Config::schedule_family_name(),
                old.catalog
                    .rows()
                    .filter(|row| !row.is_native_root())
                    .map(|row| (row.profiles().clone(), row.schedule().clone())),
                [(profiles, schedule, transition)],
                old.catalog.policy(),
                Config::ring_challenge_config,
            )
            .unwrap(),
            ChunkedWitnessCfg::default_non_chunked(),
        )
        .unwrap();
        let selection = catalog
            .rows()
            .find(|row| row.is_native_root())
            .unwrap()
            .selection();
        Fixture {
            catalog,
            bindings: vec![(selection, capacity)],
        }
    })
}

pub(crate) fn sources() -> (
    Vec<akita_types::binary_source::GroupedBinarySource>,
    Vec<(
        akita_params::GroupCommitPhaseParams,
        akita_params::ordinary_source::OrdinarySourceWidth,
    )>,
) {
    let (mut binary, signed) = grouped_suffix_sources_fixture();
    for (group, source) in binary.iter_mut().enumerate() {
        let words = native_source_data::words(group);
        let mut claims = source.prime.claims().to_vec();
        for claim in &mut claims {
            native_source_data::prime(
                source.manifest.descriptor().index_map().word_width(),
                &words,
                claim,
            );
        }
        source.prime = akita_types::binary_source::PrimePlaneManifest::with_field_profile(
            source.manifest.descriptor(),
            claims,
            source.manifest.field_profile(),
        )
        .unwrap();
    }
    (binary, signed)
}
