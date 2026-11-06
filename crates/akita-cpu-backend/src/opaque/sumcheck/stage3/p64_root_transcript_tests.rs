//! Word-root replay through Stage 3, below production catalog admission.
use super::product_table::{SetupProductSource, SetupProductSumcheck};
use super::UnivariatePoly;
use crate::opaque::sumcheck::digit_range::{DigitRangeProver, DigitRangeSession};
use crate::opaque::{
    CpuStage2SessionHandle, Stage1PublicTransition, Stage1RoundPolynomial, Stage1Step,
};
use crate::sources::packed_digits::PackedSignedDigits;
use akita_algebra::eq_poly::EqPolynomial;
use akita_challenges::Challenges;
use akita_error::AkitaError;
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::{
    DigitRangePlan, FlatBooleanDomain, GrindingPlan, GrindingRun, GrindingSite, SumcheckProtocol,
};
use akita_sumcheck::{SumcheckKernel, SumcheckProverChannel, SumcheckRole, SumcheckShape};
use akita_types::binary_source::{
    binary_root_p64_batch_challenges, BinaryAOperand, BinaryAProduct, BinaryAReadGuard,
    BinaryCommittedGroupManifest, BinaryRootPublic, BinaryTypedPrefix,
};
use akita_types::{
    GrindingReplay, GrindingSumcheckProver, GrindingSumcheckVerifier, ProverGrinding,
    VerifierGrinding,
};
use jolt_field::{Ext2, ExtField, One, Prime64Offset59 as F, Ring, Zero};
use std::sync::Arc;
type E = Ext2<F>;

struct RelationRounds<'a>(&'a mut CpuStage2SessionHandle<E>);
impl SumcheckKernel<E> for RelationRounds<'_> {
    fn num_rounds(&self) -> usize {
        self.0.num_rounds()
    }
    fn degree_bound(&self) -> usize {
        3
    }
    fn input_claim(&self) -> E {
        self.0.input_claim()
    }
    fn round_polynomial(
        &mut self,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        self.0.round_polynomial(round, claim)
    }
    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        self.0.bind_challenge(round, challenge)
    }
    fn finish(&mut self) -> Result<(), AkitaError> {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn word_root_round_trip(
    manifest: &BinaryCommittedGroupManifest,
    public: &BinaryRootPublic,
    matrix: &[Vec<[i128; 648]>],
    challenges: &Challenges,
    witness: &[i8],
    b: &[F],
    d: &[F],
) {
    let variables = public.plan().repacking().domain_len().trailing_zeros() as usize;
    let typed = BinaryTypedPrefix::new(manifest, true, 0, 64).unwrap();
    let setup = typed
        .setup_coefficients::<F>(&[41; 32], manifest, typed.domain_len)
        .unwrap();
    let stage3_rounds = typed.domain_len.trailing_zeros() as usize;
    let order = BinaryFieldProfile::P64.challenge_order().unwrap();
    // Test-only transport pricing; production catalog/security admission is
    // intentionally still P128 until the later catalog stage.
    let mut runs = [
        GrindingSite::RingSwitchAlpha { level: 0 },
        GrindingSite::Tau0Point { level: 0 },
        GrindingSite::Tau1Point { level: 0 },
    ]
    .into_iter()
    .map(|site| GrindingRun::proof_of_work(site, 1, order).unwrap())
    .collect::<Vec<_>>();
    for (protocol, rounds, degree) in [
        (SumcheckProtocol::Stage1, variables, 5),
        (SumcheckProtocol::Stage2, variables, 3),
        (SumcheckProtocol::Stage3, stage3_rounds, 2),
    ] {
        if protocol == SumcheckProtocol::Stage2 {
            runs.push(
                GrindingRun::proof_of_work(GrindingSite::Stage2Batch { level: 0 }, 1, order)
                    .unwrap(),
            );
        }
        for round in 0..rounds {
            runs.push(
                GrindingRun::proof_of_work(
                    GrindingSite::SumcheckRound {
                        protocol,
                        level: 0,
                        stage: 0,
                        round: round as u32,
                    },
                    degree,
                    order,
                )
                .unwrap(),
            );
        }
    }
    let plan = GrindingPlan::new(runs, order).unwrap();
    let catalog = akita_config::test_support::workspace_schedule_catalog::<
        akita_config::proof_optimized::fp64::Dense,
    >(akita_params::ChunkedWitnessCfg::default_non_chunked())
    .unwrap();
    let slot = akita_params::SetupPrefixSlotId {
        natural_len: typed.natural_len,
        commitment_profile: catalog
            .catalog()
            .rows()
            .find_map(|row| {
                std::iter::once(&row.schedule().root)
                    .chain(&row.schedule().recursive_folds)
                    .find_map(|fold| {
                        akita_params::setup_prefix_precommitted_params(
                            &fold.params,
                            typed.domain_len,
                        )
                        .ok()
                    })
            })
            .expect("ordinary fixture prefix profile")
            .profile,
    };
    let identity = typed.identity(manifest, &slot).unwrap();
    let lift = |values: &[F]| values.iter().copied().map(E::lift_base).collect::<Vec<_>>();
    let prove = |witness: &[i8]| -> Result<(Vec<u8>, usize, usize), AkitaError> {
        let mut channel = akita_transcript::new_prover_channel(b"binary-p64-word-stages");
        let mut grinding = ProverGrinding::new(&mut channel, &plan);
        let alpha =
            grinding.grinded_ext_challenge::<E>(GrindingSite::RingSwitchAlpha { level: 0 })?;
        let tau = grinding
            .grinded_ext_challenges::<E>(GrindingSite::Tau0Point { level: 0 }, variables)?;
        let batches =
            binary_root_p64_batch_challenges::<F, E, _>(&mut grinding, public.batch_count())?;
        assert_ne!(alpha.base_coefficient(1), F::zero());
        assert!(tau
            .iter()
            .chain(&batches)
            .all(|r| r.base_coefficient(1) != F::zero()));
        let (weights, negative) = public.relation(
            challenges,
            alpha,
            &batches,
            &[],
            &[],
            &lift(b),
            &lift(d),
            None,
            Some(BinaryAOperand::new::<F>(matrix)),
        )?;
        assert!(negative.is_empty());
        let claim = weights.claim();
        let domain = FlatBooleanDomain::new(witness.len(), variables)?;
        let mut range = DigitRangeSession::new(
            DigitRangeProver::from_packed_digits(
                PackedSignedDigits::from_i8_digits_auto(witness.to_vec()),
                DigitRangePlan::new(8)?,
                domain,
                akita_types::DigitRangeEqualityPoint::from_column_then_ring_challenges(
                    &tau, variables, 0,
                )?,
            )?,
            None,
        )?;
        let mut stage1_point = Vec::with_capacity(variables);
        let mut stage1_offset = None;
        {
            let mut rounds =
                GrindingSumcheckProver::<F, E>::new(&mut grinding, SumcheckProtocol::Stage1, 0, 0);
            let claim_site = rounds.sumcheck_site(0, 0, SumcheckRole::Claim);
            akita_transcript::public_extensions::<F, E, _>(
                rounds.state_mut(),
                claim_site,
                &[E::zero()],
            )?;
            for round in 0..variables {
                let Stage1RoundPolynomial::EqFactored(poly) =
                    range.round_polynomial(Stage1Step::RangeLeaf, round, range.current_claim())?
                else {
                    return Err(AkitaError::InvalidProof);
                };
                let mut coefficients = poly.into_coefficients();
                assert!(coefficients.len() <= 4);
                coefficients.resize(4, E::zero());
                let site = rounds.sumcheck_site(0, round as u32, SumcheckRole::RoundBody);
                if round == 0 {
                    stage1_offset = Some(rounds.state_mut().narg_string().len());
                }
                akita_transcript::exchange_extension_group::<F, E, _>(
                    rounds.state_mut(),
                    site,
                    &mut coefficients,
                )?;
                let challenge = rounds.round_challenge(0, round as u32)?;
                range.bind_challenge(Stage1Step::RangeLeaf, round, challenge)?;
                stage1_point.push(challenge);
            }
        }
        let Stage1PublicTransition::Final {
            range_image_evaluation: image,
            ..
        } = range.public_transition(Stage1Step::RangeLeaf)?
        else {
            return Err(AkitaError::InvalidProof);
        };
        range.finish()?;
        akita_types::stage1_range_image::<F, E, _>(&mut grinding, 0, 0, image)?;
        let gamma = grinding.grinded_ext_challenge::<E>(GrindingSite::Stage2Batch { level: 0 })?;
        let mut relation = CpuStage2SessionHandle::from_structured_relation(
            PackedSignedDigits::from_i8_digits_auto(witness.to_vec()),
            Box::new(weights.profiles(10)?),
            10,
            claim,
            &stage1_point,
            image,
            gamma,
        )?;
        let (point, _) = akita_sumcheck::prove_sumcheck::<F, E, _, _>(
            &mut RelationRounds(&mut relation),
            &mut GrindingSumcheckProver::<F, E>::new(&mut grinding, SumcheckProtocol::Stage2, 0, 0),
            SumcheckShape::new(variables, 3)?,
            0,
        )?;
        let terminal = relation.finish()?;
        akita_types::stage2_w_eval::<F, E, _>(&mut grinding, 0, terminal.witness_evaluation())?;
        let (rho, _) = public.coefficient_inputs(&batches, &[], &[])?;
        let a = BinaryAProduct::new(public.plan(), &rho, &point, typed.domain_len)?;
        let batches = public.authentication_batches(&batches)?;
        let mut product = SetupProductSumcheck::<F, E>::structured(
            SetupProductSource::Owned(Arc::from(setup.clone())),
            Box::new(typed.prover_weight_blocks(&a, &point, alpha, batches, typed.domain_len)?),
        )?;
        akita_types::stage3_public_slot(&mut grinding, 0, &identity)?;
        let claim_offset = grinding.state_mut().narg_string().len();
        akita_types::stage3_claim::<F, E, _>(&mut grinding, 0, product.input_claim())?;
        akita_sumcheck::prove_sumcheck::<F, E, _, _>(
            &mut product,
            &mut GrindingSumcheckProver::<F, E>::new(&mut grinding, SumcheckProtocol::Stage3, 0, 0),
            SumcheckShape::new(stage3_rounds, 2)?,
            0,
        )?;
        akita_types::stage3_prefix_eval::<F, E, _>(
            &mut grinding,
            0,
            product.folded_table_value()?,
        )?;
        grinding.finish()?;
        Ok((
            channel.narg_string().to_vec(),
            stage1_offset.unwrap(),
            claim_offset,
        ))
    };
    let verify = |proof: &[u8], committed_witness: &[i8]| -> Result<(), AkitaError> {
        let _guard = BinaryAReadGuard::forbid();
        let mut channel = akita_transcript::new_verifier_channel(b"binary-p64-word-stages", proof);
        let mut grinding = VerifierGrinding::new(&mut channel, &plan);
        let alpha =
            grinding.grinded_ext_challenge::<E>(GrindingSite::RingSwitchAlpha { level: 0 })?;
        let tau = grinding
            .grinded_ext_challenges::<E>(GrindingSite::Tau0Point { level: 0 }, variables)?;
        let batches =
            binary_root_p64_batch_challenges::<F, E, _>(&mut grinding, public.batch_count())?;
        let (weights, negative) = public.relation(
            challenges,
            alpha,
            &batches,
            &[],
            &[],
            &lift(b),
            &lift(d),
            None,
            None,
        )?;
        assert!(negative.is_empty());
        let range = akita_sumcheck::verify_eq_factored_sumcheck_rounds::<F, E, _>(
            &tau,
            E::zero(),
            SumcheckShape::new(variables, 4)?,
            &mut GrindingSumcheckVerifier::<F, E>::new(
                &mut grinding,
                SumcheckProtocol::Stage1,
                0,
                0,
            ),
            0,
        )?;
        let image = akita_types::stage1_range_image::<F, E, _>(&mut grinding, 0, 0, E::zero())?;
        if range.output_claim
            != image
                * (image - E::from_u64(2))
                * (image - E::from_u64(6))
                * (image - E::from_u64(12))
        {
            return Err(AkitaError::InvalidProof);
        }
        let gamma = grinding.grinded_ext_challenge::<E>(GrindingSite::Stage2Batch { level: 0 })?;
        let relation = akita_sumcheck::verify_sumcheck_rounds::<F, E, _>(
            &mut GrindingSumcheckVerifier::<F, E>::new(
                &mut grinding,
                SumcheckProtocol::Stage2,
                0,
                0,
            ),
            0,
            weights.claim() + gamma * image,
            SumcheckShape::new(variables, 3)?,
        )?;
        let w = akita_types::stage2_w_eval::<F, E, _>(&mut grinding, 0, E::zero())?;
        // The successor also authenticates this outgoing-witness opening.
        // At this isolated root boundary use its fixed table as the oracle.
        let expected_w = committed_witness
            .iter()
            .zip(EqPolynomial::evals(&relation.challenges)?)
            .fold(E::zero(), |sum, (&digit, eq)| {
                sum + E::from_i64(i64::from(digit)) * eq
            });
        if w != expected_w {
            return Err(AkitaError::InvalidProof);
        }

        let (rho, _) = public.coefficient_inputs(&batches, &[], &[])?;
        let a = BinaryAProduct::new(public.plan(), &rho, &relation.challenges, typed.domain_len)?;
        let batches = public.authentication_batches(&batches)?;
        akita_types::stage3_public_slot(&mut grinding, 0, &identity)?;
        let claim = akita_types::stage3_claim::<F, E, _>(&mut grinding, 0, E::zero())?;
        let product = akita_sumcheck::verify_sumcheck_rounds::<F, E, _>(
            &mut GrindingSumcheckVerifier::<F, E>::new(
                &mut grinding,
                SumcheckProtocol::Stage3,
                0,
                0,
            ),
            0,
            claim,
            SumcheckShape::new(stage3_rounds, 2)?,
        )?;
        let value = akita_types::stage3_prefix_eval::<F, E, _>(&mut grinding, 0, E::zero())?;
        let weight = a.evaluate(&product.challenges)?
            + typed.digit_weight(&product.challenges, &relation.challenges, alpha, batches)?;
        if product.output_claim != value * weight {
            return Err(AkitaError::InvalidProof);
        }
        let expected = (weights.evaluate(&relation.challenges)? + claim) * w
            + gamma
                * EqPolynomial::mle(&range.challenges, &relation.challenges)?
                * w
                * (w + E::one());
        if relation.output_claim != expected {
            return Err(AkitaError::InvalidProof);
        }
        // Successor opening oracle: the base-valued setup is evaluated at the
        // extension point passed across the Stage 3 boundary.
        let expected = setup
            .iter()
            .zip(EqPolynomial::evals(&product.challenges)?)
            .fold(E::zero(), |sum, (&s, eq)| sum + E::lift_base(s) * eq);
        if value != expected {
            return Err(AkitaError::InvalidProof);
        }
        grinding.finish()?.check_eof(channel)?;
        Ok(())
    };
    let (proof, stage1_offset, claim_offset) = prove(witness).unwrap();
    verify(&proof, witness).unwrap();
    // A changed K digit still satisfies Stage 1's range check, but cannot
    // satisfy the compiled integer relation. Debug builds check it at
    // construction; release builds check the folded terminal oracle.
    let mut changed = witness.to_vec();
    let k = public.plan().quotient_segment().unwrap();
    let address = public
        .plan()
        .repacking()
        .coefficient_address(k, 0, 0, 0)
        .unwrap();
    changed[address] = if changed[address] == 3 {
        2
    } else {
        changed[address] + 1
    };
    let error = prove(&changed).expect_err("changed K must not produce a root proof");
    if cfg!(debug_assertions) {
        assert!(matches!(error, AkitaError::Internal(message)
            if message == "materialized relation weights do not match the combined relation claim"));
    } else {
        assert!(matches!(error, AkitaError::InvalidInput(message)
            if message == "relation session final claim disagrees with its folded oracle"));
    }
    assert!(matches!(
        verify(&proof, &changed),
        Err(AkitaError::InvalidProof)
    ));
    // Stage 1 body, Stage 3 product claim and successor endpoint. Replay
    // must reject changes to either coordinate at each boundary.
    for offset in [stage1_offset, claim_offset, proof.len() - 16] {
        for coordinate in [0, 8] {
            let mut changed = proof.clone();
            changed[offset + coordinate] ^= 1;
            assert!(verify(&changed, witness).is_err());
        }
    }
}
