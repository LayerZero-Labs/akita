//! Typed adapter for canonical transcript-grinding plan derivation.

use crate::CommitmentConfig;
use akita_error::AkitaError;
use akita_params::zk::OpeningSchedule;
use akita_params::{ChallengeFieldOrder, GrindingPlan, OpeningClaimsLayout};
use jolt_algebra::{CanonicalEncoding, ExtField};

/// Derive the only accepted grinding plan for one effective schedule and call.
pub fn derive_transcript_grinding_plan<'a, Cfg: CommitmentConfig>(
    schedule: impl Into<OpeningSchedule<'a>>,
    root_layout: &OpeningClaimsLayout,
) -> Result<GrindingPlan, AkitaError>
where
    Cfg::Field: CanonicalEncoding,
    Cfg::ExtField: ExtField<Cfg::Field>,
{
    Cfg::validate_sis_modulus_profile()?;
    let extension_degree = <Cfg::ExtField as ExtField<Cfg::Field>>::DEGREE;
    if Cfg::EXT_DEGREE != extension_degree {
        return Err(AkitaError::InvalidSetup(
            "grinding plan extension degree does not match the field tower".into(),
        ));
    }
    let challenge_order = ChallengeFieldOrder::from_field(
        <Cfg::Field as CanonicalEncoding>::MODULUS_BITS,
        extension_degree,
        Cfg::sis_modulus_profile().modulus(),
    )?;
    schedule
        .into()
        .grinding_plan(root_layout, challenge_order, extension_degree)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof_optimized::fp128;
    use akita_params::{GrindingQueryKind, GrindingSite, GRINDING_NONCE_SLACK_BITS};

    #[test]
    fn production_onehot_plan_is_canonical_and_fully_priced() {
        let catalog = crate::test_support::workspace_schedule_catalog::<fp128::OneHot>(
            akita_params::ChunkedWitnessCfg::default_non_chunked(),
        )
        .expect("one-hot schedule catalog");
        let layout = OpeningClaimsLayout::new(14, 1).expect("opening layout");
        let row = catalog
            .resolve_key(&akita_params::ScheduleLookupKey::single(
                layout.root_final_group_layout().expect("root group"),
            ))
            .expect("generated production row");
        let plan = derive_transcript_grinding_plan::<fp128::OneHot>(row.schedule(), &layout)
            .expect("grinding plan");

        assert_eq!(
            plan.runs().first().unwrap().site(),
            GrindingSite::FoldResponse { level: 0 }
        );
        assert_eq!(
            plan.runs()
                .iter()
                .filter(|run| matches!(run.site(), GrindingSite::EvaluationBatch { .. }))
                .count(),
            0,
            "singleton row batching draws no challenge and has no plan entry"
        );
        for run in plan.runs() {
            if run.kind() == GrindingQueryKind::ProofOfWork {
                assert!(u128::from(run.loss_factor()) <= (1u128 << run.grind_bits()));
                assert_eq!(
                    run.nonce_bits(),
                    if run.grind_bits() == 0 {
                        0
                    } else {
                        run.grind_bits() + GRINDING_NONCE_SLACK_BITS
                    }
                );
            }
        }
        assert_eq!(
            (
                plan.runs().len(),
                plan.expanded_query_count(),
                plan.total_nonce_bits(),
                plan.digest().unwrap(),
            ),
            (
                43,
                50,
                400,
                [
                    214, 212, 86, 221, 26, 115, 163, 245, 7, 27, 104, 22, 8, 173, 81, 176, 161,
                    186, 166, 148, 181, 248, 72, 43, 40, 203, 182, 47, 174, 236, 149, 251,
                ],
            )
        );
    }

    /// The five-level `fp128_onehot final=16:2` row and its opening layout:
    /// the smallest one-hot row whose masked levels meet the one-shot target.
    fn five_level_onehot_row() -> (OpeningClaimsLayout, akita_params::FoldSchedule) {
        let catalog = crate::test_support::workspace_schedule_catalog::<fp128::OneHot>(
            akita_params::ChunkedWitnessCfg::default_non_chunked(),
        )
        .expect("one-hot schedule catalog");
        let layout = OpeningClaimsLayout::new(16, 2).expect("opening layout");
        let row = catalog
            .resolve_key(&akita_params::ScheduleLookupKey::single(
                layout.root_final_group_layout().expect("root group"),
            ))
            .expect("generated production row");
        (layout, row.schedule().clone())
    }

    fn zk_schedule(
        schedule: akita_params::FoldSchedule,
        cut_level: u32,
    ) -> Result<akita_params::zk::ZkOpeningSchedule, AkitaError> {
        akita_params::zk::ZkOpeningSchedule::for_tests(
            schedule,
            cut_level,
            fp128::OneHot::committed_source_contract().unwrap(),
        )
    }

    #[test]
    fn zero_knowledge_plan_is_the_clear_plan_with_one_shot_responses_up_to_the_cut() {
        use akita_params::zk::{OpeningSchedule, ZK_CUT_LEVEL};
        use akita_params::FOLD_RESPONSE_NONCE_BITS;

        let (layout, schedule) = five_level_onehot_row();
        let clear = derive_transcript_grinding_plan::<fp128::OneHot>(&schedule, &layout)
            .expect("clear plan");
        let zk = zk_schedule(schedule.clone(), ZK_CUT_LEVEL).expect("zk schedule");
        let one_shot =
            derive_transcript_grinding_plan::<fp128::OneHot>(OpeningSchedule::Zk(&zk), &layout)
                .expect("zk plan");

        assert_eq!(clear.runs().len(), one_shot.runs().len());
        let mut fold_levels = Vec::new();
        for (clear_run, zk_run) in clear.runs().iter().zip(one_shot.runs()) {
            assert_eq!(clear_run.site(), zk_run.site());
            if let GrindingSite::FoldResponse { level } = zk_run.site() {
                fold_levels.push(level);
                assert_eq!(clear_run.nonce_bits(), FOLD_RESPONSE_NONCE_BITS);
                let expected = if level <= ZK_CUT_LEVEL {
                    0
                } else {
                    FOLD_RESPONSE_NONCE_BITS
                };
                assert_eq!(zk_run.nonce_bits(), expected, "level {level}");
            } else {
                assert_eq!(clear_run, zk_run);
            }
        }
        assert_eq!(
            fold_levels,
            (0..schedule.num_fold_levels() as u32).collect::<Vec<_>>()
        );
        assert_ne!(clear.digest().unwrap(), one_shot.digest().unwrap());
    }

    #[test]
    fn zero_knowledge_schedule_rejects_an_unsupported_cut_or_a_short_masked_level() {
        use akita_params::zk::ZK_CUT_LEVEL;

        let (_, schedule) = five_level_onehot_row();
        assert!(zk_schedule(schedule.clone(), ZK_CUT_LEVEL).is_ok());
        assert!(matches!(
            zk_schedule(schedule.clone(), ZK_CUT_LEVEL + 1),
            Err(AkitaError::UnsupportedSchedule(_))
        ));
        for level in 0..=ZK_CUT_LEVEL as usize {
            let mut short = schedule.clone();
            let params = match level {
                0 => &mut short.root.params,
                _ => &mut short.recursive_folds[level - 1].params,
            };
            params.own_group_mut().opening.num_digits_fold -= 1;
            assert!(
                matches!(
                    zk_schedule(short, ZK_CUT_LEVEL),
                    Err(AkitaError::UnsupportedSchedule(message)) if message.contains("fold digits")
                ),
                "level {level}"
            );
        }
        // A short level after the cut is not masked.
        let mut short = schedule;
        short.recursive_folds[ZK_CUT_LEVEL as usize]
            .params
            .own_group_mut()
            .opening
            .num_digits_fold -= 1;
        assert!(zk_schedule(short, ZK_CUT_LEVEL).is_ok());
    }

    /// The one-shot fold response at level 0 of a zero-knowledge plan sends
    /// no nonce and forks at counter 0, and the verifier rejects a nonce in
    /// its place.
    #[test]
    fn one_shot_fold_response_replays_without_a_nonce() {
        use akita_params::zk::{OpeningSchedule, ZK_CUT_LEVEL};
        use akita_params::FoldResponseSearch;
        use akita_types::{
            AkitaSponge, FoldDrawRecord, GrindingReplay, ProverGrinding, VerifierGrinding,
            PROOF_STREAM_PROTOCOL,
        };
        use jolt_transcript::{ProverTranscript, VerifierTranscript};
        type F = <fp128::OneHot as CommitmentConfig>::Field;
        const SESSION: &[u8] = b"one-shot-fold-response";

        let (layout, schedule) = five_level_onehot_row();
        let zk = zk_schedule(schedule, ZK_CUT_LEVEL).expect("zk schedule");
        let plan =
            derive_transcript_grinding_plan::<fp128::OneHot>(OpeningSchedule::Zk(&zk), &layout)
                .expect("zk plan");
        let site = GrindingSite::FoldResponse { level: 0 };
        let start = plan
            .runs()
            .iter()
            .position(|run| run.site() == site)
            .expect("level-0 fold response");
        // Every run before the response is a proof-of-work query.
        let earlier = &plan.runs()[..start];
        assert!(earlier
            .iter()
            .all(|run| run.kind() == GrindingQueryKind::ProofOfWork));
        let [group, next, ..] = &plan.runs()[start + 1..] else {
            panic!("plan too short");
        };
        let GrindingSite::FoldChallengeGroup { level: 0, group: 0 } = group.site() else {
            panic!("the response's challenge group follows it");
        };
        let coordinates = usize::try_from(group.fold_coordinate_count().unwrap()).unwrap();

        let mut transcript = ProverTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION);
        let mut prover = ProverGrinding::new(&mut transcript, &plan);
        for run in earlier {
            prover.grinded_ext_challenge::<F, F>(run.site()).unwrap();
        }
        assert_eq!(
            prover.begin_fold_response(site).unwrap(),
            FoldResponseSearch::OneShot
        );
        assert!(prover.fold_response_fork(site, 1).is_err());
        let sent = prover.state_mut().narg().len();
        let mut prover_fork = prover.commit_fold_response(site, 0).unwrap();
        assert_eq!(prover.state_mut().narg().len(), sent, "no nonce is sent");
        prover
            .record_fold_challenges(0, 0, coordinates, &FoldDrawRecord::default())
            .unwrap();
        prover.grinded_ext_challenge::<F, F>(next.site()).unwrap();
        let proof = transcript.finish();

        let verify = |proof: &[u8]| -> Result<[u8; 32], AkitaError> {
            let mut transcript =
                VerifierTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION, proof);
            let mut verifier = VerifierGrinding::new(&mut transcript, &plan);
            for run in earlier {
                verifier.grinded_ext_challenge::<F, F>(run.site())?;
            }
            let mut fork = verifier.read_fold_response(site)?;
            verifier.record_fold_challenges(0, 0, coordinates, &FoldDrawRecord::default())?;
            verifier.grinded_ext_challenge::<F, F>(next.site())?;
            transcript.finish()?;
            Ok(fork.squeeze::<32>())
        };
        assert_eq!(verify(&proof), Ok(prover_fork.squeeze::<32>()));
        // A clear prover's nonce for counter 0 where the response sits: after
        // the bytes of the earlier queries.
        let mut transcript = ProverTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION);
        let mut prover = ProverGrinding::new(&mut transcript, &plan);
        for run in earlier {
            prover.grinded_ext_challenge::<F, F>(run.site()).unwrap();
        }
        let at_response = prover.state_mut().narg().len();
        let mut with_nonce = proof;
        with_nonce.insert(at_response, 0);
        assert!(verify(&with_nonce).is_err());

        prover.begin_fold_response(site).unwrap();
        assert!(matches!(
            prover.commit_fold_response(site, 1),
            Err(AkitaError::InvalidInput(_))
        ));
    }

    #[test]
    fn stage1_prices_the_full_eq_factored_round_degree() {
        let catalog = crate::test_support::workspace_schedule_catalog::<fp128::OneHot>(
            akita_params::ChunkedWitnessCfg::default_non_chunked(),
        )
        .expect("one-hot schedule catalog");
        let layout = OpeningClaimsLayout::new(14, 1).expect("opening layout");
        let row = catalog
            .resolve_key(&akita_params::ScheduleLookupKey::single(
                layout.root_final_group_layout().expect("root group"),
            ))
            .expect("generated production row");
        let plan = derive_transcript_grinding_plan::<fp128::OneHot>(row.schedule(), &layout)
            .expect("grinding plan");
        let root = &row.schedule().root;
        let rounds = akita_params::sumcheck_rounds(root.params.d_a(), root.output_witness_len);
        let basis = 1usize
            .checked_shl(root.params.open().digits.log_basis)
            .expect("digit range basis");
        let range = akita_params::DigitRangePlan::new(basis).expect("digit range plan");
        let (stages, _) = range
            .proof_shapes_for_route(rounds, root.params.inner().matrix.security_route())
            .expect("Stage 1 shapes");

        for run in plan.runs() {
            let GrindingSite::SumcheckRound {
                protocol: akita_params::SumcheckProtocol::Stage1,
                level: 0,
                stage,
                ..
            } = run.site()
            else {
                continue;
            };
            let q_degree = stages[usize::try_from(stage).expect("stage index")]
                .sumcheck_proof
                .1;
            assert_eq!(run.loss_factor(), u64::try_from(q_degree + 1).unwrap());
        }
    }

    #[test]
    fn exact_field_orders_price_pseudo_mersenne_deficits() {
        fn check<Cfg: CommitmentConfig>()
        where
            Cfg::Field: CanonicalEncoding,
        {
            Cfg::validate_sis_modulus_profile().unwrap();
            let order = ChallengeFieldOrder::from_field(
                <Cfg::Field as CanonicalEncoding>::MODULUS_BITS,
                Cfg::EXT_DEGREE,
                Cfg::sis_modulus_profile().modulus(),
            )
            .unwrap();
            for exponent in 0..akita_params::MAX_GRINDING_BITS {
                assert_eq!(
                    akita_params::grind_bits_for_loss(1u64 << exponent, order).unwrap(),
                    exponent + 1,
                );
            }
            assert_eq!(akita_params::grind_bits_for_loss(3, order).unwrap(), 2);
        }
        check::<fp128::OneHot>();
        check::<crate::proof_optimized::fp64::OneHot>();
        check::<crate::proof_optimized::fp32::OneHot>();
    }

    #[test]
    fn every_generated_production_row_derives_a_complete_plan() {
        use crate::proof_optimized::{fp32, fp64};

        fn audit<Cfg: CommitmentConfig>(chunking: akita_params::ChunkedWitnessCfg)
        where
            Cfg::Field: CanonicalEncoding,
            Cfg::ExtField: ExtField<Cfg::Field>,
        {
            let catalog = crate::test_support::workspace_schedule_catalog::<Cfg>(chunking)
                .expect("workspace schedule catalog");
            for row in catalog.rows() {
                let layout = row.profiles().opening_layout().expect("opening layout");
                let plan = derive_transcript_grinding_plan::<Cfg>(row.schedule(), &layout)
                    .expect("complete grinding plan");
                let count = |kind| plan.runs().iter().filter(|run| run.kind() == kind).count();
                assert_eq!(
                    count(GrindingQueryKind::FoldResponse),
                    row.schedule().num_fold_levels()
                );
                let expected_evaluation_batch_levels =
                    std::iter::once(layout.requires_row_batch_challenge().then_some(0))
                        .flatten()
                        .chain(
                            row.schedule()
                                .recursive_folds
                                .iter()
                                .enumerate()
                                .filter_map(|(index, step)| {
                                    step.params.setup_prefix().map(|_| {
                                        u32::try_from(index + 1)
                                            .expect("recursive fold level fits u32")
                                    })
                                }),
                        )
                        .collect::<Vec<_>>();
                let evaluation_batch_levels = plan
                    .runs()
                    .iter()
                    .filter_map(|run| match run.site() {
                        GrindingSite::EvaluationBatch { level } => Some(level),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    evaluation_batch_levels, expected_evaluation_batch_levels,
                    "evaluation batching belongs only to nonterminal multi-polynomial layouts"
                );
                assert!(count(GrindingQueryKind::FoldChallengeGroup) > 0);
                assert!(plan.expanded_query_count() >= plan.runs().len() as u64);

                for run in plan.runs() {
                    let GrindingSite::L2VirtualBatch { level } = run.site() else {
                        continue;
                    };
                    let params = if level == 0 {
                        &row.schedule().root.params
                    } else {
                        &row.schedule().recursive_folds
                            [usize::try_from(level - 1).expect("fold level fits usize")]
                        .params
                    };
                    let akita_params::InnerCommitSecurityRoute::L2 {
                        norm_proof_shape, ..
                    } = params.inner().matrix.security_route()
                    else {
                        panic!("L2 virtual-batch query requires an L2 security route");
                    };
                    assert_eq!(
                        run.loss_factor(),
                        u64::try_from(norm_proof_shape.virtual_evaluation_count())
                            .expect("virtual evaluation count fits u64"),
                        "shifted virtual batching must price its highest eta degree"
                    );
                }

                // A level's Stage 3 rounds are induced by its *successor's*
                // setup prefix, and their count is the prefix group's own
                // committed width. Pin that against the stored layout so no
                // second derivation of the prefix width can reappear.
                for (index, step) in row.schedule().recursive_folds.iter().enumerate() {
                    let level = u32::try_from(index).expect("grinding level fits u32");
                    let expected = step
                        .params
                        .setup_prefix()
                        .map_or(0, |prefix| prefix.profile.group.num_vars());
                    let observed = plan
                        .runs()
                        .iter()
                        .filter(|run| {
                            matches!(
                                run.site(),
                                GrindingSite::SumcheckRound {
                                    protocol: akita_params::SumcheckProtocol::Stage3,
                                    level: run_level,
                                    ..
                                } if run_level == level
                            )
                        })
                        .count();
                    assert_eq!(
                        observed, expected,
                        "level {level} Stage 3 rounds must equal the successor prefix group width"
                    );
                }
            }
        }

        audit::<fp128::Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked());
        audit::<fp128::DenseBounded>(akita_params::ChunkedWitnessCfg::default_non_chunked());
        audit::<fp128::DenseMultiChunk>(akita_params::MultiChunkProfileId::W8R2.cfg());
        audit::<fp128::OneHot>(akita_params::ChunkedWitnessCfg::default_non_chunked());
        audit::<fp128::OneHotMultiChunk>(akita_params::MultiChunkProfileId::W8R2.cfg());
        audit::<fp128::OneHotMultiChunkW2R2>(akita_params::MultiChunkProfileId::W2R2.cfg());
        audit::<fp128::OneHotMultiChunkW4R2>(akita_params::MultiChunkProfileId::W4R2.cfg());
        audit::<fp128::OneHotMultiChunkW2R3>(akita_params::MultiChunkProfileId::W2R3.cfg());
        audit::<fp128::OneHotMultiChunkW4R3>(akita_params::MultiChunkProfileId::W4R3.cfg());
        audit::<fp128::OneHotMultiChunkW8R3>(akita_params::MultiChunkProfileId::W8R3.cfg());
        audit::<fp128::DenseMultiChunkW8R3>(akita_params::MultiChunkProfileId::W8R3.cfg());
        audit::<crate::RecursiveCommitmentConfig<fp128::OneHotMultiChunkW2R3>>(
            akita_params::MultiChunkProfileId::W2R3.cfg(),
        );
        audit::<crate::RecursiveCommitmentConfig<fp128::OneHotMultiChunkW4R3>>(
            akita_params::MultiChunkProfileId::W4R3.cfg(),
        );
        audit::<crate::RecursiveCommitmentConfig<fp128::OneHotMultiChunkW8R3>>(
            akita_params::MultiChunkProfileId::W8R3.cfg(),
        );
        audit::<fp64::Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked());
        audit::<fp64::OneHot>(akita_params::ChunkedWitnessCfg::default_non_chunked());
        audit::<fp32::Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked());
        audit::<fp32::OneHot>(akita_params::ChunkedWitnessCfg::default_non_chunked());
        audit::<crate::RecursiveCommitmentConfig<fp128::OneHot>>(
            akita_params::ChunkedWitnessCfg::default_non_chunked(),
        );
        audit::<crate::RecursiveCommitmentConfig<fp128::OneHotMultiChunk>>(
            akita_params::MultiChunkProfileId::W8R2.cfg(),
        );
    }
}
