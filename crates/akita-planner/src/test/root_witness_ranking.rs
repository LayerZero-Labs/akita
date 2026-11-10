use super::*;
use akita_config::{policy_of, proof_optimized::fp64::Dense, CommitmentConfig};

struct RootRankingFixture {
    policy: PlannerPolicy,
    key: ScheduleLookupKey,
    contracts: Vec<CommittedSourceContract>,
    dimensions: CommitmentRingDims,
    opening: PlannerOpeningCandidate,
    inner_basis: u32,
    open_basis: u32,
}

impl RootRankingFixture {
    fn new(count: usize, dimension: usize) -> Self {
        let policy = policy_of::<Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked());
        let contract = Dense::committed_source_contract().unwrap();
        let dimensions = CommitmentRingDims {
            inner: dimension,
            outer: 128,
            opening: 64,
        };
        let opening =
            PlannerOpeningCandidate::coefficient_packing(policy.claim_ext_degree, dimensions, 64)
                .unwrap()
                .unwrap();
        let inner_basis = Dense::inner_basis_range().0;
        let open_basis = Dense::opening_basis_range().0;
        let mut profiles = Vec::new();
        for index in 0..count {
            let group = PolynomialGroupLayout::singleton(16 + index);
            let candidates = root_level_candidates_with_fresh_preparation(
                &ScheduleLookupKey::single(group),
                contract,
                &[],
                &policy,
                dimensions,
                opening,
                &[],
                inner_basis,
                open_basis,
                None,
            )
            .unwrap();
            let (params, _) = candidates.into_iter().min_by_key(|(_, len)| *len).unwrap();
            profiles.push(GroupCommitPhaseParams::try_from_params(group, &params).unwrap());
        }
        let key = ScheduleLookupKey {
            root_accumulated_setup_prefix_len: 0,
            final_group: PolynomialGroupLayout::singleton(24),
            precommitteds: profiles,
        };
        key.validate(policy.decomposition.field_bits()).unwrap();
        Self {
            policy,
            key,
            contracts: vec![contract; count],
            dimensions,
            opening,
            inner_basis,
            open_basis,
        }
    }

    fn main(
        &self,
        guide: Option<crate::schedule_params::CandidateLayoutGuide>,
    ) -> PreparedRootFinalGroup {
        PreparedRootFinalGroup::prepare(
            &self.key,
            Dense::committed_source_contract().unwrap(),
            &self.policy,
            self.dimensions,
            self.opening,
            self.inner_basis,
            self.open_basis,
            guide,
            crate::schedule_params::LevelMasking::Clear,
        )
        .unwrap()
    }
}

#[test]
fn exact_witness_ranking_matches_exhaustive_sort_without_shared_slots() {
    let mut fixture = RootRankingFixture::new(9, 256);
    fixture.key.precommitteds.push(fixture.key.precommitteds[0]);
    fixture.contracts.push(fixture.contracts[0]);
    let main = fixture.main(None);
    let actual = smallest_witness_root_candidates(
        &fixture.key,
        &fixture.contracts,
        &fixture.policy,
        fixture.dimensions,
        fixture.open_basis,
        &main,
        crate::schedule_params::LevelMasking::Clear,
    )
    .unwrap();
    let mut reference = Vec::new();
    for bits in 0..512 {
        let assignments = (0..9)
            .rev()
            .map(|i| {
                PlannerOpeningCandidate::coefficient_packing(
                    fixture.policy.claim_ext_degree,
                    fixture.dimensions,
                    64 << ((bits >> i) & 1),
                )
                .unwrap()
                .unwrap()
            })
            .collect::<Vec<_>>();
        let Some(prepared) = PreparedRootProducers::prepare(
            &fixture.key,
            &fixture.contracts,
            &fixture.policy,
            fixture.dimensions,
            fixture.opening,
            &assignments,
            fixture.open_basis,
            crate::schedule_params::LevelMasking::Clear,
        )
        .unwrap() else {
            continue;
        };
        let candidates = main.candidates(&fixture.policy, &prepared, false).unwrap();
        if let Some(score) = candidates.iter().map(|(_, len)| *len).min() {
            reference.push(((score, bits), candidates));
        }
    }
    assert!(
        reference.len() > 256,
        "fixture must exercise witness truncation"
    );
    reference.sort_by_key(|(rank, _)| *rank);
    reference.truncate(256);
    assert!(
        !reference.iter().any(|((_, bits), _)| *bits == 511),
        "the all-large shared subring must not get a reserved slot"
    );
    let expected = reference
        .into_iter()
        .flat_map(|(_, candidates)| candidates)
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    let assignments = actual
        .iter()
        .map(|(params, _)| {
            params
                .precommitted_groups()
                .iter()
                .map(|group| group.opening.opening_method)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        assignments
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        256
    );
    for (params, len) in actual {
        assert_eq!(
            params.precommitted_groups()[0],
            params.precommitted_groups()[9]
        );
        assert_eq!(
            root_batch_next_w_len(
                fixture.policy.decomposition.field_bits(),
                fixture.policy.claim_ext_degree,
                &params,
                &fixture.key.opening_layout().unwrap()
            )
            .unwrap(),
            Some(len)
        );
    }
}

#[test]
#[ignore = "manual exhaustive next-witness ranking benchmark"]
fn benchmark_exact_witness_ranking() {
    let fixture = RootRankingFixture::new(10, 1024);
    let main = fixture.main(None);
    // Pin one valid root layout so timings can be compared with the original
    // parameter-sizing benchmark. Production ranks all allowed layouts.
    let prototype = main.candidates_by_split.iter().flatten().next().unwrap();
    let guide = crate::schedule_params::CandidateLayoutGuide {
        position_index_bits: prototype.blocks().position_index_bits(),
        outer_slice_count: prototype.outer_slice_count(),
        inner_route: crate::schedule_params::CandidateInnerRoute::of(
            prototype.inner().matrix.security_route(),
        ),
        setup_prefix: None,
    };
    let main = fixture.main(Some(guide));
    let started = Instant::now();
    let candidates = smallest_witness_root_candidates(
        &fixture.key,
        &fixture.contracts,
        &fixture.policy,
        fixture.dimensions,
        fixture.open_basis,
        &main,
        crate::schedule_params::LevelMasking::Clear,
    )
    .unwrap();
    eprintln!(
        "exact witness ranking: classes=10 raw_assignments={} candidates={} elapsed={:?}",
        4usize.pow(10),
        candidates.len(),
        started.elapsed()
    );
    assert!(!std::hint::black_box(candidates).is_empty());
}
