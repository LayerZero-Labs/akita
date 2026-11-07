//! Full inventory pricing and statement mutations for the two source-batched passes.

use super::*;
use akita_types::binary_source::{
    PrimeClaims, PrimeGroupClaim, PrimePlaneClaim, PrimePlaneManifest, ResearchBatchPlan,
};
use akita_types::opening_claims::GroupClaimKind;

pub(super) fn check_geometry(
    claims: &OpeningClaims<'_, E, CommittedGroup<F>>,
    profiles: &[GroupCommitPhaseParams],
    plan: &ResearchBatchPlan,
    setup: &akita_types::AkitaSetupDescriptor,
) {
    let count = if plan.mode() == ResearchTerminalMode::BaseS11a {
        1
    } else {
        4
    };
    let mut total = 0;
    for (index, group) in claims.groups().iter().take(4).enumerate() {
        let GroupClaimKind::Binary {
            manifest,
            claim,
            prime,
        } = group.kind()
        else {
            unreachable!()
        };
        assert_eq!(prime.planes().len(), count);
        total += prime.planes().len();
        let local = plan.binary()[index].plan();
        let segment = &local.repacking().segments()[local.prime_segment().unwrap()];
        assert_eq!(
            segment.elements,
            count * manifest.descriptor().index_map().columns()
        );
        assert_eq!(segment.digits, 22);
        let rank = usize::try_from(manifest.d_label(claim.is_some(), count).unwrap().rank).unwrap();
        assert_eq!(
            plan.d_ranges()[index].len(),
            rank * manifest.b_ring_dimension()
        );
    }
    assert_eq!(total, count * 4);
    let negative = plan
        .ordinary()
        .plans()
        .iter()
        .map(|local| {
            local
                .negative_binary_addresses(plan.ordinary_repacking())
                .unwrap()
                .len()
        })
        .sum::<usize>();
    let binary = plan
        .binary()
        .iter()
        .map(|local| local.batch_count())
        .sum::<usize>();
    let extra = if count == 4 { 12 } else { 0 };
    assert_eq!(
        plan.batch_count(),
        binary + plan.ordinary().linear_batch_count() + negative + extra,
    );
    if count == 4 {
        for group in claims.groups().iter().skip(4) {
            let GroupClaimKind::ResearchOrdinary { owners, width, .. } = group.kind() else {
                unreachable!()
            };
            for pass in 0..2 {
                let mut missing = group.research_terminals().unwrap().clone();
                missing.readings[pass].evaluations.pop();
                assert!(PolynomialGroupClaims::new_research_ordinary(
                    group.commitment().clone(),
                    missing,
                    owners.clone(),
                    *width,
                )
                .is_err());
            }
        }
        let mut groups = claims.groups().to_vec();
        let group = &groups[4];
        let GroupClaimKind::ResearchOrdinary { owners, width, .. } = group.kind() else {
            unreachable!()
        };
        groups[4] = PolynomialGroupClaims::new_research_ordinary(
            group.commitment().clone(),
            ResearchTerminals {
                mode: ResearchTerminalMode::BaseS11a,
                readings: vec![ResearchTerminal {
                    point: group
                        .point()
                        .iter()
                        .map(|value| E::lift_base(value.base_coefficient(0)))
                        .collect(),
                    evaluations: group
                        .evaluations()
                        .iter()
                        .map(|value| E::lift_base(value.base_coefficient(0)))
                        .collect(),
                }],
            },
            owners.clone(),
            *width,
        )
        .unwrap();
        assert!(OpeningClaims::from_groups(groups).is_err(), "mixed modes");
        let mut base_kinds = claims
            .groups()
            .iter()
            .map(|group| group.kind().clone())
            .collect::<Vec<_>>();
        for kind in &mut base_kinds {
            match kind {
                GroupClaimKind::Binary {
                    manifest, prime, ..
                } => {
                    *prime = PrimePlaneManifest::with_field_profile(
                        manifest.descriptor(),
                        Some(PrimeClaims::Base(PrimeGroupClaim {
                            point: vec![0; 11],
                            evaluations: vec![PrimePlaneClaim {
                                polynomial: 0,
                                value: 0,
                            }],
                        })),
                        akita_params::binary_setup::BinaryFieldProfile::P64,
                    )
                    .unwrap();
                }
                GroupClaimKind::ResearchOrdinary { mode, .. } => {
                    *mode = ResearchTerminalMode::BaseS11a;
                }
                _ => unreachable!(),
            }
        }
        let base = ResearchBatchPlan::new(&base_kinds, profiles, setup, 64).unwrap();
        assert!(plan.physical_cells() >= base.physical_cells());
        assert!(plan.batch_count() >= base.batch_count() + 12 + 2 * 12);
        assert_eq!(plan.fold_coordinates(), base.fold_coordinates());
    }
}

pub(super) fn mutations(
    claims: &OpeningClaims<'static, E, CommittedGroup<F>>,
) -> Vec<OpeningClaims<'static, E, CommittedGroup<F>>> {
    let mut changed = Vec::new();
    // Retain H when changing either E64 coordinate on any binary source.
    for index in 0..4 {
        for pass in 0..2 {
            for coordinate in 0..2 {
                let delta = E::from_base_fn(|i| F::from_u64(u64::from(i == coordinate)));
                for point in [true, false] {
                    let mut groups = claims.groups().to_vec();
                    let group = &groups[index];
                    let GroupClaimKind::Binary { claim, .. } = group.kind() else {
                        unreachable!()
                    };
                    let mut terminals = group.research_terminals().unwrap().clone();
                    if point {
                        terminals.readings[pass].point[0] += delta;
                    } else {
                        terminals.readings[pass].evaluations[0] += delta;
                    }
                    groups[index] = PolynomialGroupClaims::new_research_binary_with_word(
                        group.commitment().clone(),
                        terminals,
                        claim.clone(),
                    )
                    .unwrap();
                    changed.push(OpeningClaims::from_groups(groups).unwrap());
                }
            }
        }
    }
    // Every signed slot, both readings and both coordinates must be consumed.
    for index in 4..8 {
        for pass in 0..2 {
            for coordinate in 0..2 {
                let delta = E::from_base_fn(|i| F::from_u64(u64::from(i == coordinate)));
                let slots = claims.groups()[index].evaluations().len();
                for target in 0..=slots {
                    let mut groups = claims.groups().to_vec();
                    let group = &groups[index];
                    let GroupClaimKind::ResearchOrdinary { owners, width, .. } = group.kind()
                    else {
                        unreachable!()
                    };
                    let mut terminals = group.research_terminals().unwrap().clone();
                    if target == slots {
                        terminals.readings[pass].point[0] += delta;
                    } else {
                        terminals.readings[pass].evaluations[target] += delta;
                    }
                    groups[index] = PolynomialGroupClaims::new_research_ordinary(
                        group.commitment().clone(),
                        terminals,
                        owners.clone(),
                        *width,
                    )
                    .unwrap();
                    changed.push(OpeningClaims::from_groups(groups).unwrap());
                }
            }
        }
    }
    assert_eq!(changed.len(), 96);
    changed
}
