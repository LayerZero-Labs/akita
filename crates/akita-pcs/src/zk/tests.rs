//! `AkitaResponseProver` on jolt-zk's synthetic Σ-protocol statements
//! (`specs/whiteout-akita.md` §5.6): acceptance where the statement's own
//! checker accepts (W5-a), rejection of each item's violation (W5-a), and the
//! batched weights against `H` contracted column by column through the
//! statement's own row maps (W5-b).

use super::*;
use akita_config::policy_of;
use akita_config::proof_optimized::fp128::ResponseDigits;
use akita_config::recursive_commitment::RecursiveCommitmentConfig;
use akita_params::{ChunkedWitnessCfg, CommittedGroupBatchProfile, GroupCommitPhaseParams};
use akita_schedules::ValidatedScheduleCatalog;
use jolt_algebra::cfg_into_iter;
use jolt_algebra::solinas::parallel::*;
use jolt_algebra::Zero;
use jolt_zk::whiteout::test_support::{CheckingResponseProver, SyntheticSigma};
use jolt_zk::whiteout::ResponseSegment;

type Cfg = RecursiveCommitmentConfig<ResponseDigits>;

/// A response prover whose catalog holds the planner's response schedule for
/// `e` of `len` entries.
fn akita_for(len: usize) -> AkitaResponseProver<Cfg> {
    let num_vars = ResponseShape::new(len).unwrap().num_vars;
    let chunking = ChunkedWitnessCfg::default_non_chunked();
    let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
    let policy = policy_of::<Cfg>(chunking);
    let schedule = akita_planner::find_schedule(
        &key,
        Cfg::committed_source_contract().unwrap(),
        &[],
        &policy,
        Cfg::ring_challenge_config,
    )
    .unwrap()
    .schedule;
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &schedule.root.params,
        )
        .unwrap(),
        precommitteds: Vec::new(),
    };
    let catalog = ValidatedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        [(profiles, schedule)],
        &policy,
        Cfg::ring_challenge_config,
    )
    .unwrap();
    let scheme = AkitaCommitmentScheme::<Cfg>::from_schedule_artifact(
        &catalog.to_artifact_bytes().unwrap(),
        chunking,
    )
    .unwrap();
    let setup = scheme.setup_prover(num_vars, 1).unwrap();
    AkitaResponseProver::new(scheme, setup).unwrap()
}

fn on_big_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(test)
        .unwrap()
        .join()
        .unwrap();
}

/// W5-a: honest statements at both ring dimensions verify under Akita, as
/// they do under the statement's own checker.
#[test]
fn accepts_what_the_statement_checker_accepts() {
    on_big_stack(|| {
        fn check<const D: usize>(seed: u64) {
            let sigma = SyntheticSigma::<D>::new(seed, 1);
            let checked = sigma.prove(&CheckingResponseProver, seed).unwrap();
            sigma.verify(&checked, &CheckingResponseProver).unwrap();
            let akita = akita_for(sigma.response_len());
            let proof = sigma.prove(&akita, seed).unwrap();
            sigma.verify(&proof, &akita).unwrap();
        }
        check::<64>(1);
        check::<256>(2);
    });
}

/// Changes `e` before proving it, after the statement's checker names the
/// expected violation.
struct Tampering<'a> {
    akita: &'a AkitaResponseProver<Cfg>,
    tamper: Tamper,
    expected: WhiteoutError,
}

/// An edit of `e`, given its segment positions.
type Tamper = fn(&ResponseLayoutPositions, &mut [i8]);

/// The first digit of each kind of segment of `e`.
struct ResponseLayoutPositions {
    z: usize,
    v: usize,
    theta: usize,
}

impl ResponseLayoutPositions {
    fn new<const D: usize>(statement: &ResponseStatement<'_, F, D>) -> Self {
        let mut offset = 0;
        let (mut v, mut theta) = (0, 0);
        for segment in statement.layout().segments {
            match segment {
                ResponseSegment::Response { .. } => {}
                ResponseSegment::V { .. } => v = offset,
                ResponseSegment::Theta => theta = offset,
            }
            offset += segment.num_digits();
        }
        Self { z: 0, v, theta }
    }
}

/// Moves a digit to another one in the alphabet.
fn bump(digit: &mut i8) {
    *digit = if *digit == 7 { 6 } else { *digit + 1 };
}

impl<const D: usize> ResponseProver<F, D> for Tampering<'_> {
    fn prove_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        let mut tampered = witness.clone();
        (self.tamper)(
            &ResponseLayoutPositions::new(statement),
            &mut tampered.digits,
        );
        assert_eq!(
            statement.check_witness(&tampered),
            Err(self.expected.clone())
        );
        self.akita.prove_response(statement, &tampered, transcript)
    }

    fn verify_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        self.akita.verify_response(statement, transcript)
    }
}

/// W5-a: with the prover's own check bypassed, the verifier rejects each
/// violation the statement's checker names: Items 2, 3 and 4 by tampering
/// with `e`, and Item 5 on a statement whose relation fails. Item 1 cannot
/// reach the verifier: a digit outside `[−8, 7]` is outside the response
/// schedule's commitment bound.
#[test]
fn rejects_each_item_violation() {
    on_big_stack(|| {
        let sigma = SyntheticSigma::<64>::new(3, 1);
        let akita = akita_for(sigma.response_len()).unchecked();
        let cases: [(Tamper, &'static str); 3] = [
            (|at, e| bump(&mut e[at.v]), "V commitment"),
            (|at, e| bump(&mut e[at.z]), "ring-linear rows"),
            (|at, e| bump(&mut e[at.theta]), "product quotient θ"),
        ];
        for (tamper, expected) in cases {
            let tampering = Tampering {
                akita: &akita,
                tamper,
                expected: WhiteoutError::ResponseRejected(expected),
            };
            let proof = sigma.prove(&tampering, 4).unwrap();
            assert!(sigma.verify(&proof, &akita).is_err(), "{expected}");
        }

        let alphabet = Tampering {
            akita: &akita,
            tamper: |at, e| e[at.z] = 8,
            expected: WhiteoutError::ResponseRejected("digit outside its alphabet"),
        };
        assert!(sigma.prove(&alphabet, 5).is_err());

        let violating = SyntheticSigma::<64>::violating_item_five(6, 1);
        let checked = violating.prove(&CheckingResponseProver, 7).unwrap_err();
        assert!(format!("{checked:?}").contains("ct(h)"), "{checked:?}");
        let proof = violating.prove(&akita, 7).unwrap();
        assert!(violating.verify(&proof, &akita).is_err());
    });
}

/// Compares the batched weights with `H` built column by column.
struct WeightOracle;

impl<const D: usize> ResponseProver<F, D> for WeightOracle {
    fn prove_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        let shape = ResponseShape::new(statement.layout().len()).unwrap();
        let alpha: F = transcript.challenge();
        let product_vars = ceil_log2(statement.row_shape().products.max(1));
        let product_point = challenges(transcript, product_vars);
        let eta = challenges(transcript, row_variables(statement, shape));
        let r = challenges(transcript, shape.num_vars);
        let range_point = challenges(transcript, shape.num_vars);
        let gamma: F = transcript.challenge();
        let claims = ProductClaims {
            target: transcript.challenge(),
            x: transcript.challenge(),
            y: transcript.challenge(),
        };

        let weights = RowWeights::new(statement, shape, &eta, &product_point).unwrap();
        let zero = statement.rows(&vec![0; shape.len], alpha).unwrap();
        let mut columns: Vec<F> = cfg_into_iter!(0..shape.len)
            .map(|j| {
                let mut unit = vec![0i8; shape.len];
                unit[j] = 1;
                let column = statement.rows(&unit, alpha).unwrap();
                weights.rows.dot(&column).unwrap() - weights.rows.dot(&zero).unwrap()
            })
            .collect();
        columns.extend(weights.padding.iter().copied());
        let eq_r = EqPolynomial::evals(&r).unwrap();
        let contracted: F = columns.iter().zip(&eq_r).map(|(h, e)| *h * *e).sum();
        let b = weights.target(claims) - weights.rows.dot(&zero).unwrap();

        let batch = batch_rows(statement, shape, alpha, weights, claims).unwrap();
        assert_eq!(multilinear_eval(&batch.a, &r).unwrap(), contracted);
        assert_eq!(batch.b, b);
        let b_table: F = EqPolynomial::evals(&range_point)
            .unwrap()
            .iter()
            .zip(&eq_r)
            .map(|(eq, e)| gamma * *eq * *e)
            .sum();
        assert_eq!(
            gamma * EqPolynomial::mle(&range_point, &r).unwrap(),
            b_table
        );
        assert!(statement.check_witness(witness).is_ok());
        assert_ne!(contracted, F::zero());
        Ok(())
    }

    fn verify_response<H: Sponge>(
        &self,
        _statement: &ResponseStatement<'_, F, D>,
        _transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        Ok(())
    }
}

/// W5-b: on the smallest synthetic shape, the verifier's `A(r)` and `B(r)`
/// equal `Σ_j eq(r, j)·Σ_l eq(η, l)·H_{l,j}` and `Σ_j eq(r, j)·B_j`, with
/// each column `H_{·,j}` read from the statement's own rows at the unit
/// vector `δ_j`.
#[test]
fn batched_weights_match_h_column_by_column() {
    on_big_stack(|| {
        let sigma = SyntheticSigma::<64>::new(8, 1);
        sigma.prove(&WeightOracle, 9).unwrap();
    });
}
