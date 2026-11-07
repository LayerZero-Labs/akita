//! Small real four-plus-four research proof; no production admission.
#![cfg(all(feature = "binary-source-measurements", feature = "logging-transcript"))]

#[path = "binary_source_end_to_end/byte_contract.rs"]
mod byte_contract;
#[path = "research_batch/catalog.rs"]
mod catalog;
mod common;
#[path = "research_batch/e64.rs"]
mod e64;
#[path = "research_batch/word.rs"]
mod word;

use akita_config::TrustedScheduleCatalog;
use akita_cpu_backend::{CommitmentHandle, CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_params::ordinary_source::OrdinarySourceWidth;
use akita_params::{BasisMode, GroupCommitPhaseParams, PrecommittedGroupProfiles};
use akita_pcs::AkitaCommitmentScheme;
use akita_transcript::{new_prover_channel, new_verifier_channel, ProofChannel, ProtocolSiteId};
use akita_types::binary_source::{ResearchTerminal, ResearchTerminalMode, ResearchTerminals};
use akita_types::{CommittedGroup, OpeningClaims, PolynomialGroupClaims};
use catalog::{Config, E, F};
use jolt_field::{ExtField, Ring, Zero};
use std::sync::Arc;

const LABEL: &[u8] = b"route1/4+4/research/base-s11a";

fn ordinary<const BITS: u32>(
    backend: &CpuBackend<F, E>,
    producer: &TrustedScheduleCatalog<Config<BITS>>,
    profile: GroupCommitPhaseParams,
    owner_start: u8,
    mode: ResearchTerminalMode,
) -> (
    PolynomialGroupClaims<'static, E, CommittedGroup<F>>,
    CommitmentHandle<F, E>,
) {
    let width = OrdinarySourceWidth::new(BITS).unwrap();
    let offset = i128::try_from(width.offset()).unwrap();
    let values = (0..profile.group.num_polynomials())
        .map(|slot| {
            (0..1usize << profile.group.num_vars())
                .map(|index| {
                    F::from_i128(match (index + slot) % 4 {
                        0 => -offset,
                        1 => offset - 1,
                        2 => -1,
                        3 => 0,
                        _ => unreachable!("remainder modulo four"),
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let source = backend
        .import_source(
            values
                .iter()
                .map(|values| {
                    DensePoly::from_field_evals(profile.group.num_vars(), values).unwrap()
                })
                .collect(),
        )
        .unwrap();
    let committed = backend
        .commit(producer, &source, GroupContext::explicit(&profile))
        .unwrap();
    let passes = if mode == ResearchTerminalMode::BaseS11a {
        1
    } else {
        2
    };
    let readings = (0..passes)
        .map(|pass| {
            let point = (0..profile.group.num_vars())
                .map(|axis| {
                    let base = 2 + 3 * u64::try_from(axis).unwrap();
                    point_coordinate(mode, base, pass)
                })
                .collect::<Vec<_>>();
            let weights = akita_algebra::eq_poly::EqPolynomial::evals(&point).unwrap();
            ResearchTerminal {
                point,
                evaluations: values
                    .iter()
                    .map(|values| {
                        values
                            .iter()
                            .zip(&weights)
                            .fold(E::zero(), |sum, (&value, &weight)| {
                                sum + E::lift_base(value) * weight
                            })
                    })
                    .collect(),
            }
        })
        .collect();
    let owners = (0..profile.group.num_polynomials())
        .map(|slot| [owner_start + u8::try_from(slot).unwrap(); 32])
        .collect();
    (
        PolynomialGroupClaims::new_research_ordinary(
            committed.committed_group,
            ResearchTerminals { mode, readings },
            owners,
            width,
        )
        .unwrap(),
        committed.private_handle,
    )
}

#[test]
fn research_four_plus_four_commit_prove_replay_and_mutations() {
    common::init_rayon_pool();
    common::run_on_large_stack(|| run(false, ResearchTerminalMode::BaseS11a));
}

#[test]
fn research_four_plus_four_with_word_commit_prove_replay_and_mutations() {
    common::init_rayon_pool();
    common::run_on_large_stack(|| run(true, ResearchTerminalMode::BaseS11a));
}

#[test]
fn research_four_plus_four_e64_two_point_replay_and_mutations() {
    common::init_rayon_pool();
    common::run_on_large_stack(|| run(false, ResearchTerminalMode::E64TwoPointS11a));
}

#[test]
fn research_four_plus_four_e64_two_point_with_word_replay_and_mutations() {
    common::init_rayon_pool();
    common::run_on_large_stack(|| run(true, ResearchTerminalMode::E64TwoPointS11a));
}

fn point_coordinate(mode: ResearchTerminalMode, base: u64, pass: u64) -> E {
    E::from_base_fn(|index| {
        F::from_u64(if index == 0 {
            base + pass
        } else if mode == ResearchTerminalMode::BaseS11a {
            0
        } else {
            1 + pass
        })
    })
}

fn run(with_word: bool, mode: ResearchTerminalMode) {
    let producers = catalog::Producers::new();
    let (catalog, manifests, profiles, plan) = catalog::batch(&producers, with_word, mode);
    let scheme = AkitaCommitmentScheme::<Config<33>>::new(catalog);
    let setup = scheme.setup_prover(10, 16).unwrap();
    let backend = Arc::new(CpuBackend::<F, E>::new(setup.expanded.clone()).unwrap());
    let mut groups = Vec::with_capacity(8);
    let mut handles = Vec::with_capacity(8);
    for (index, manifest) in manifests.iter().enumerate() {
        let profile = profiles[index];
        let words = (0..8)
            .map(|word| 1 + u128::try_from(index + word).unwrap())
            .collect::<Vec<_>>();
        let source = backend
            .import_packed_binary_source(manifest.clone(), profile, 8, vec![words.clone()])
            .unwrap();
        let preceding = (index != 0)
            .then(|| PrecommittedGroupProfiles::from_profiles(profiles[..index].to_vec()).unwrap());
        let context = preceding.as_ref().map_or_else(
            || GroupContext::explicit(&profile),
            |preceding| GroupContext::explicit_with_precommitted_groups(&profile, preceding),
        );
        let committed = backend.commit(&producers.binary, &source, context).unwrap();
        let passes = if mode == ResearchTerminalMode::BaseS11a {
            1
        } else {
            2
        };
        let readings = (0..passes)
            .map(|pass| {
                let point = (0..11)
                    .map(|axis| point_coordinate(mode, 19 + 7 * u64::try_from(axis).unwrap(), pass))
                    .collect::<Vec<_>>();
                let equality = akita_algebra::eq_poly::EqPolynomial::evals(&point).unwrap();
                let evaluation =
                    words
                        .iter()
                        .enumerate()
                        .fold(E::zero(), |sum, (carrier, word)| {
                            (0..128).fold(sum, |sum, bit| {
                                if (word >> bit) & 1 == 0 {
                                    sum
                                } else {
                                    sum + equality[carrier * 256 + bit]
                                }
                            })
                        });
                ResearchTerminal {
                    point,
                    evaluations: vec![evaluation],
                }
            })
            .collect();
        groups.push(
            PolynomialGroupClaims::new_research_binary_with_word(
                committed.committed_group,
                ResearchTerminals { mode, readings },
                (with_word && index == 0).then(|| word::claim(manifest.descriptor(), &words)),
            )
            .unwrap(),
        );
        handles.push(committed.private_handle);
    }
    let ordinary = [
        ordinary(&backend, &producers.width2, profiles[4], 20, mode),
        ordinary(&backend, &producers.width9, profiles[5], 30, mode),
        ordinary(&backend, &producers.width18, profiles[6], 40, mode),
        ordinary(&backend, &producers.width33, profiles[7], 50, mode),
    ];
    for (group, handle) in ordinary {
        groups.push(group);
        handles.push(handle);
    }
    let claims = OpeningClaims::from_groups(groups).unwrap();
    let mut incomplete = claims.groups().to_vec();
    incomplete.pop();
    assert!(OpeningClaims::from_groups(incomplete).is_err());
    let mut reordered = claims.groups().to_vec();
    reordered.swap(0, 4);
    assert!(OpeningClaims::from_groups(reordered).is_err());
    e64::check_geometry(&claims, &profiles, &plan, setup.expanded.descriptor());
    let data =
        common::selected_prover_data::<Config<33>>(claims.clone(), handles, scheme.schedules());
    let selection = data.selection();
    let row = scheme.schedules().resolve_selection(selection).unwrap();
    assert!(row.is_native_root());
    assert!(row.schedule().recursive_folds[0]
        .params
        .setup_prefix()
        .is_none());
    assert_eq!(
        row.schedule().root.output_witness_len,
        plan.physical_cells()
    );
    assert!(scheme.schedules().catalog().to_artifact_bytes().is_err());
    let prefixes = akita_prover::SetupPrefixProverRegistry::default();
    let mut registry = akita_prover::BackendRegistry::<Config<33>>::new().unwrap();
    let executor = registry.register(backend.as_ref(), &prefixes).unwrap();
    let mut route =
        akita_prover::FixedFoldRoute::new(vec![executor; row.schedule().num_fold_levels()]);
    akita_transcript::clear_thread_events();
    let mut prover = new_prover_channel(LABEL);
    akita_prover::batched_prove_embedded::<Config<33>, _>(
        setup.expanded.descriptor(),
        scheme.schedules(),
        &registry,
        data,
        &mut prover,
        BasisMode::Lagrange,
        &mut route,
    )
    .unwrap();
    let proof = prover.narg_string().to_vec();
    byte_contract::compare_baseline(
        &proof,
        match (mode, with_word) {
            (ResearchTerminalMode::BaseS11a, false) => "research-four-plus-four.bin",
            (ResearchTerminalMode::BaseS11a, true) => "research-four-plus-four-word.bin",
            (ResearchTerminalMode::E64TwoPointS11a, false) => "research-four-plus-four-e64.bin",
            (ResearchTerminalMode::E64TwoPointS11a, true) => "research-four-plus-four-e64-word.bin",
        },
    );
    let continuation = prover
        .challenge_message(32, |bytes| Ok::<_, AkitaError>(bytes.to_vec()))
        .unwrap();
    akita_transcript::finish_proof_ranges(&prover);
    let ranges = akita_transcript::thread_proof_ranges();
    let events = akita_transcript::thread_events();
    let root_sites = events
        .iter()
        .filter_map(|event| {
            let akita_transcript::TranscriptEvent::Context(record) = event;
            let site = ProtocolSiteId::from_bytes(record.site_id);
            (site.level == 0).then_some((site, record.kind))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        root_sites
            .iter()
            .filter(|(_, kind)| {
                *kind == akita_transcript::ProtocolMessageKind::FoldResponseNonce as u32
            })
            .count(),
        1
    );
    assert_eq!(
        root_sites
            .iter()
            .filter(|(site, _)| { site.family == akita_transcript::SITE_FAMILY_OPENING_PAYLOAD })
            .count(),
        1
    );
    assert_eq!(
        root_sites
            .iter()
            .filter(|(site, _)| {
                site.family == akita_transcript::SITE_FAMILY_NEXT_WITNESS && site.stage == 1
            })
            .count(),
        1
    );
    let drawn_groups = root_sites
        .iter()
        .filter(|(site, _)| site.family == akita_transcript::SITE_FAMILY_FOLD_CHALLENGE)
        .map(|(site, _)| site.group)
        .collect::<Vec<_>>();
    assert_eq!(drawn_groups, vec![0, 1, 2, 3]);
    assert!(!root_sites.iter().any(|(site, _)| {
        site.family == akita_transcript::SITE_FAMILY_SUMCHECK
            && site.invocation == akita_params::SumcheckProtocol::Stage3.tag()
    }));
    let verifier_setup = scheme.setup_verifier(&setup).unwrap();
    let verifier = scheme.verifier(verifier_setup).unwrap();
    let verify = |proof: &[u8], claims: &OpeningClaims<'_, E, CommittedGroup<F>>| {
        let borrowed = OpeningClaims::from_groups(
            claims
                .groups()
                .iter()
                .map(|group| group.map_commitment(|_| group.commitment()))
                .collect(),
        )
        .unwrap();
        let statement = akita_types::GroupBatchStatement::new(selection, borrowed).unwrap();
        let mut replay = new_verifier_channel(LABEL, proof);
        verifier
            .batched_verify_embedded(
                &mut replay,
                statement,
                BasisMode::Lagrange,
                akita_verifier::SetupPolicy::Check(None),
            )
            .and_then(|pending| {
                let continued =
                    replay.challenge_message(32, |bytes| Ok::<_, AkitaError>(bytes.to_vec()))?;
                if continued != continuation {
                    return Err(AkitaError::InvalidProof);
                }
                pending.check_eof(replay)
            })
    };
    akita_transcript::clear_thread_events();
    verify(&proof, &claims).expect("honest research replay and exact EOF");
    assert_eq!(akita_transcript::thread_events(), events);
    if mode == ResearchTerminalMode::E64TwoPointS11a {
        for changed in e64::mutations(&claims) {
            assert!(verify(&proof, &changed).is_err(), "E64 reading mutation");
        }
    }
    if with_word {
        word::reject_mutations(&claims, |changed| verify(&proof, changed).is_err());
    }
    let payload = ranges
        .iter()
        .find(|range| {
            let site = ProtocolSiteId::from_bytes(range.context.site_id);
            site.family == akita_transcript::SITE_FAMILY_OPENING_PAYLOAD && site.level == 0
        })
        .expect("complete root D payload");
    let nonce = ranges
        .iter()
        .find(|range| {
            let site = ProtocolSiteId::from_bytes(range.context.site_id);
            site.level == 0
                && range.context.kind
                    == akita_transcript::ProtocolMessageKind::FoldResponseNonce as u32
        })
        .expect("joint response nonce");
    // The field switch is the only proof payload before the complete D inventory.
    let switch_bytes = if with_word {
        let akita_types::opening_claims::GroupClaimKind::Binary {
            manifest,
            claim: Some(word),
            ..
        } = claims.groups()[0].kind()
        else {
            panic!("hash word claim");
        };
        128 * 16 + word.carrier_point(manifest.descriptor()).unwrap().len() * 42 + 21
    } else {
        0
    };
    assert_eq!(payload.start, switch_bytes);
    assert_eq!(payload.len, plan.d_ranges().last().unwrap().end * 8);
    assert!(payload.start + payload.len <= nonce.start);
    let mut bad_d = proof.clone();
    bad_d[payload.start + plan.d_ranges()[2].start * 8] ^= 1;
    assert!(verify(&bad_d, &claims).is_err(), "third D mutation");
    let mut bad_nonce = proof.clone();
    bad_nonce[nonce.start] ^= 1;
    assert!(verify(&bad_nonce, &claims).is_err(), "joint nonce mutation");
    let mut wrong_groups = claims.groups().to_vec();
    let group = &wrong_groups[5];
    let changed = PolynomialGroupClaims::new_research_ordinary(
        group.commitment().clone(),
        group.research_terminals().unwrap().clone(),
        vec![[90; 32], [31; 32], [32; 32]],
        OrdinarySourceWidth::new(9).unwrap(),
    )
    .unwrap();
    wrong_groups[5] = changed;
    assert!(verify(&proof, &OpeningClaims::from_groups(wrong_groups).unwrap()).is_err());
    let mut trailing = proof.clone();
    trailing.push(0);
    assert!(verify(&trailing, &claims).is_err(), "outer EOF mutation");
    assert!(
        verify(&proof[..proof.len() - 1], &claims).is_err(),
        "truncated EOF mutation"
    );
    let ordinary_plan = akita_params::GrindingPlan::new(
        vec![],
        akita_params::binary_setup::BinaryFieldProfile::P64
            .challenge_order()
            .unwrap(),
    )
    .unwrap();
    let joint = claims
        .root_grinding_plan(ordinary_plan.clone(), row.schedule(), true)
        .unwrap();
    assert_eq!(
        joint.runs()[1].loss_factor(),
        u64::try_from(plan.fold_coordinates().iter().sum::<usize>()).unwrap()
    );
    assert!(joint.runs().iter().any(|run| {
        run.site()
            == akita_params::GrindingSite::SumcheckRound {
                protocol: akita_params::SumcheckProtocol::Stage2,
                level: 0,
                stage: 0,
                round: plan.domain_len().trailing_zeros() - 1,
            }
    }));
    assert!(claims
        .root_grinding_plan(ordinary_plan, row.schedule(), false)
        .is_err());
    eprintln!(
        "research=true binary_groups=4 ordinary_groups=4 witnesses=1 suffixes=1 bytes={}",
        proof.len()
    );
}
