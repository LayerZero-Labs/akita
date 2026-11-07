//! Fresh Q64 producer and explicitly selected direct research row.

use akita_challenges::FoldChallengeDistribution;
use akita_config::{
    policy_of, proof_optimized::fp64::Dense, CommitmentConfig, TrustedScheduleCatalog,
};
use akita_error::AkitaError;
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::ordinary_source::OrdinarySourceWidth;
use akita_params::{
    ChunkedWitnessCfg, CommittedGroupBatchProfile, DecompositionParams, GroupCommitPhaseParams,
    PolynomialGroupLayout, ScheduleLookupKey, SetupContributionMode, SisModulusProfileId,
};
use akita_schedules::native_root::NativeRootTransition;
use akita_schedules::{ResolvedScheduleRow, ValidatedScheduleCatalog};
use akita_types::binary_source::{
    BinaryCommittedGroupManifest, BinaryIndexMap, BinarySourceDescriptor, PrimeClaims,
    PrimeGroupClaim, PrimePlaneClaim, PrimePlaneManifest, ResearchBatchPlan, ResearchTerminalMode,
};
use akita_types::opening_claims::GroupClaimKind;
use jolt_field::{Ext2, Prime64Offset59};

pub(super) type F = Prime64Offset59;
pub(super) type E = Ext2<F>;

#[derive(Clone)]
pub(super) struct Config<const BITS: u32>;

impl<const BITS: u32> CommitmentConfig for Config<BITS> {
    type Field = F;
    type ExtField = E;
    const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
        akita_schedules::RingDimensionSchedule {
            num_search_levels: 2,
            potential_a_dimensions: &[64, 128],
            potential_b_dimensions: &[64],
            potential_d_dimensions: &[64],
            suffix_dimensions: &[64],
        };
    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: BITS,
            log_open_bound: Some(64),
        }
    }
    fn ring_challenge_config(dimension: usize) -> Result<FoldChallengeDistribution, AkitaError> {
        Dense::ring_challenge_config(dimension)
    }
    fn sis_modulus_profile() -> SisModulusProfileId {
        SisModulusProfileId::Q64Offset59
    }
    fn inner_basis_range() -> (u32, u32) {
        (3, 3)
    }
    fn opening_basis_range() -> (u32, u32) {
        (3, 3)
    }
    fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
        akita_params::sis::CommittedSourceClass::BalancedSignedDigit
    }
    fn recursive_setup_planning() -> bool {
        false
    }
    fn schedule_family_name() -> &'static str {
        "route1_direct_research_fixture"
    }
}

fn producer<const BITS: u32>(slots: usize) -> TrustedScheduleCatalog<Config<BITS>> {
    let chunking = ChunkedWitnessCfg::default_non_chunked();
    let policy = policy_of::<Config<BITS>>(chunking);
    let planned = akita_planner::ordinary_source::plan_signed_ordinary_source(
        PolynomialGroupLayout::new(10, slots),
        OrdinarySourceWidth::new(BITS).unwrap(),
        &policy,
        Config::<BITS>::ring_challenge_config,
    )
    .expect("small research source producer");
    let profiles = CommittedGroupBatchProfile {
        final_group: planned.schedule.root.params.own_group().profile,
        precommitteds: Vec::new(),
    };
    TrustedScheduleCatalog::new(
        ValidatedScheduleCatalog::try_new(
            Config::<BITS>::schedule_family_name(),
            [(profiles, planned.schedule)],
            &policy,
            Config::<BITS>::ring_challenge_config,
        )
        .unwrap(),
        chunking,
    )
    .unwrap()
}

pub(super) struct Producers {
    pub(crate) binary: TrustedScheduleCatalog<Config<3>>,
    pub(crate) width2: TrustedScheduleCatalog<Config<2>>,
    pub(crate) width9: TrustedScheduleCatalog<Config<9>>,
    pub(crate) width18: TrustedScheduleCatalog<Config<18>>,
    pub(crate) width33: TrustedScheduleCatalog<Config<33>>,
}

impl Producers {
    pub(crate) fn new() -> Self {
        Self {
            binary: producer(1),
            width2: producer(3),
            width9: producer(3),
            width18: producer(5),
            width33: producer(1),
        }
    }
}

#[allow(clippy::type_complexity)]
pub(super) fn batch(
    producers: &Producers,
    with_word: bool,
    mode: ResearchTerminalMode,
) -> (
    TrustedScheduleCatalog<Config<33>>,
    Vec<BinaryCommittedGroupManifest>,
    Vec<GroupCommitPhaseParams>,
    ResearchBatchPlan,
) {
    let carrier = producers
        .binary
        .catalog()
        .rows()
        .next()
        .unwrap()
        .profiles()
        .final_group;
    let mut profiles = vec![carrier; 4];
    profiles.extend([
        producers
            .width2
            .catalog()
            .rows()
            .next()
            .unwrap()
            .profiles()
            .final_group,
        producers
            .width9
            .catalog()
            .rows()
            .next()
            .unwrap()
            .profiles()
            .final_group,
        producers
            .width18
            .catalog()
            .rows()
            .next()
            .unwrap()
            .profiles()
            .final_group,
        producers
            .width33
            .catalog()
            .rows()
            .next()
            .unwrap()
            .profiles()
            .final_group,
    ]);
    let mut manifests = Vec::with_capacity(4);
    let mut kinds = Vec::with_capacity(8);
    for group in 0..4 {
        let index = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
        let labels = BinaryCommittedGroupManifest::profiles_for_field(
            &index,
            1,
            group,
            BinaryFieldProfile::P64,
        )
        .unwrap();
        let descriptor = BinarySourceDescriptor::new(
            index,
            vec![[u8::try_from(group + 1).unwrap(); 32]],
            labels,
        )
        .unwrap();
        let manifest = BinaryCommittedGroupManifest::with_field_profile(
            descriptor,
            group,
            &carrier,
            BinaryFieldProfile::P64,
        )
        .unwrap();
        let readings = match mode {
            ResearchTerminalMode::BaseS11a => PrimeClaims::Base(PrimeGroupClaim {
                point: vec![0; 11],
                evaluations: vec![PrimePlaneClaim {
                    polynomial: 0,
                    value: u128::from((group + 1) & 1),
                }],
            }),
            ResearchTerminalMode::E64TwoPointS11a => PrimeClaims::Extension(
                [1, 2]
                    .map(|pass| PrimeGroupClaim {
                        point: vec![[0, pass]; 11],
                        evaluations: vec![PrimePlaneClaim {
                            polynomial: 0,
                            value: [0, 0],
                        }],
                    })
                    .to_vec(),
            ),
        };
        let prime = PrimePlaneManifest::with_field_profile(
            manifest.descriptor(),
            Some(readings),
            BinaryFieldProfile::P64,
        )
        .unwrap();
        kinds.push(GroupClaimKind::Binary {
            manifest: Box::new(manifest.clone()),
            claim: (with_word && group == 0).then(|| {
                super::word::claim(manifest.descriptor(), &(1..=8).collect::<Vec<u128>>())
            }),
            prime,
        });
        manifests.push(manifest);
    }
    for (bits, slots, start) in [(2, 3, 20), (9, 3, 30), (18, 5, 40), (33, 1, 50)] {
        kinds.push(GroupClaimKind::ResearchOrdinary {
            owners: (0..slots).map(|slot| [start + slot; 32]).collect(),
            width: OrdinarySourceWidth::new(bits).unwrap(),
            mode,
        });
    }
    let descriptor = akita_types::AkitaSetupDescriptor {
        max_num_vars: 10,
        max_num_batched_polys: 16,
        num_field_elements: 1 << 24,
        setup_seed: [41; 32].into(),
    };
    let plan = ResearchBatchPlan::new(&kinds, &profiles, &descriptor, 64).unwrap();
    for local in plan.binary() {
        assert!(local.plan().quotient_segment().is_some());
    }
    if with_word {
        assert_eq!(plan.binary()[0].plan().opening_segments(), vec![2, 5]);
        assert_eq!(plan.binary()[0].plan().prime_segment(), Some(5));
        assert_eq!(plan.binary()[0].plan().quotient_segment(), Some(6));
    }
    for (index, manifest) in manifests.iter().enumerate() {
        let GroupClaimKind::Binary { prime, .. } = &kinds[index] else {
            unreachable!("binary inventory");
        };
        let planes = prime.planes().len();
        let rank = manifest
            .d_label(with_word && index == 0, planes)
            .unwrap()
            .rank;
        let expected = usize::try_from(rank).unwrap() * manifest.b_ring_dimension();
        assert_eq!(plan.d_ranges()[index].len(), expected);
        if !with_word || index != 0 {
            assert_eq!(plan.binary()[index].plan().opening_segments(), vec![2]);
        }
    }
    let mut prime_only = kinds.clone();
    let GroupClaimKind::Binary { claim, .. } = &mut prime_only[0] else {
        unreachable!("first binary group");
    };
    *claim = None;
    let without_word = ResearchBatchPlan::new(&prime_only, &profiles, &descriptor, 64).unwrap();
    if with_word {
        assert!(plan.batch_count() >= without_word.batch_count() + 2);
        assert!(
            plan.binary()[0].plan().repacking().natural_end().unwrap()
                > without_word.binary()[0]
                    .plan()
                    .repacking()
                    .natural_end()
                    .unwrap()
        );
        assert!(plan.physical_cells() >= without_word.physical_cells());
    }
    let mut wrong_group = prime_only;
    let GroupClaimKind::Binary {
        manifest, claim, ..
    } = &mut wrong_group[1]
    else {
        unreachable!("second binary group");
    };
    *claim = Some(super::word::claim(manifest.descriptor(), &[2; 8]));
    assert!(matches!(
        ResearchBatchPlan::new(&wrong_group, &profiles, &descriptor, 64),
        Err(AkitaError::InvalidInput(_))
    ));
    let chunking = ChunkedWitnessCfg::default_non_chunked();
    let policy = policy_of::<Config<33>>(chunking);
    let key = ScheduleLookupKey {
        final_group: profiles[7].group,
        precommitteds: profiles[..7].to_vec(),
        root_accumulated_setup_prefix_len: 0,
    };
    let contracts = [3, 3, 3, 3, 2, 9, 18].map(|bits| {
        OrdinarySourceWidth::new(bits)
            .unwrap()
            .producer_contract(SisModulusProfileId::Q64Offset59)
            .unwrap()
    });
    let final_row = producers.width33.catalog().rows().next().unwrap();
    let seed = akita_planner::find_schedule_with_root_setup_mode(
        &key,
        Config::<33>::committed_source_contract().unwrap(),
        &contracts,
        &policy,
        Config::<33>::ring_challenge_config,
        SetupContributionMode::Direct,
        Some(&final_row.schedule().root.params),
    )
    .expect("ordered research carrier seed");
    let batch_profiles = CommittedGroupBatchProfile {
        final_group: profiles[7],
        precommitteds: profiles[..7].to_vec(),
    };
    let seed =
        ResolvedScheduleRow::try_new(batch_profiles.clone(), seed.schedule, &policy).unwrap();
    let transition = NativeRootTransition::research_direct(plan.physical_cells()).unwrap();
    let schedule = akita_planner::native_root::find_native_root_schedule(
        &seed,
        transition,
        &policy,
        Config::<33>::ring_challenge_config,
    )
    .expect("exact research native suffix");
    assert_eq!(schedule.root.params, seed.schedule().root.params);
    assert_eq!(schedule.root.output_witness_len, plan.physical_cells());
    let successor = schedule
        .recursive_folds
        .first()
        .expect("research native root requires an ordinary successor fold");
    assert_eq!(successor.input_witness_len, plan.physical_cells());
    assert!(successor.params.setup_prefix().is_none());
    assert!(
        ResolvedScheduleRow::try_new(batch_profiles.clone(), schedule.clone(), &policy).is_err()
    );
    let catalog = ValidatedScheduleCatalog::try_new_native_root(
        Config::<33>::schedule_family_name(),
        [],
        [(batch_profiles, schedule, transition)],
        &policy,
        Config::<33>::ring_challenge_config,
    )
    .unwrap();
    (
        TrustedScheduleCatalog::new(catalog, chunking).unwrap(),
        manifests,
        profiles,
        plan,
    )
}
