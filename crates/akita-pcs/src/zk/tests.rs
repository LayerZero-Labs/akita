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

/// Proves a false statement: `edit` builds, from the honest `e`, the digits
/// the prover commits and those its rows and range session read, after the
/// statement's checker gives `expected` on the committed digits. The prover
/// skips its own check and its sumcheck rounds stay consistent (`rounds.rs`),
/// so only the verifier's checks can reject.
struct Cheating<'a> {
    akita: &'a AkitaResponseProver<Cfg>,
    edit: Edit,
    expected: Result<(), WhiteoutError>,
}

/// The committed, rows and range digits of a false statement.
struct CheatDigits {
    committed: Vec<i8>,
    rows: Vec<i8>,
    range: Vec<i8>,
}

impl CheatDigits {
    /// One edited `e` in every role.
    fn everywhere(e: Vec<i8>) -> Self {
        Self {
            committed: e.clone(),
            rows: e.clone(),
            range: e,
        }
    }
}

type Edit = fn(&ResponseLayoutPositions, &[i8]) -> CheatDigits;

/// Where `e`'s segments start, and the first response segment's shape.
struct ResponseLayoutPositions {
    z_coefficients: usize,
    z_levels: usize,
    v: usize,
    theta: usize,
}

impl ResponseLayoutPositions {
    fn new<const D: usize>(statement: &ResponseStatement<'_, F, D>) -> Self {
        let mut offset = 0;
        let (mut v, mut theta, mut z) = (0, 0, None);
        for segment in statement.layout().segments {
            match segment {
                ResponseSegment::Response {
                    coefficients,
                    digits,
                    ..
                } => {
                    z.get_or_insert((coefficients, digits));
                }
                ResponseSegment::V { .. } => v = offset,
                ResponseSegment::Theta => theta = offset,
            }
            offset += segment.num_digits();
        }
        let (z_coefficients, z_levels) = z.unwrap();
        Self {
            z_coefficients,
            z_levels,
            v,
            theta,
        }
    }
}

/// Moves a digit to another one in the alphabet.
fn bumped(e: &[i8], at: usize) -> CheatDigits {
    let mut e = e.to_vec();
    e[at] = if e[at] == 7 { 6 } else { e[at] + 1 };
    CheatDigits::everywhere(e)
}

/// The first response coefficient's digits `(−8, x)` at planes `(h, h + 1)`
/// rewritten as `(8, x − 1)`: the same integer with a digit outside
/// `[−8, 7]`.
fn out_of_alphabet(at: &ResponseLayoutPositions, e: &[i8]) -> Vec<i8> {
    let n = at.z_coefficients;
    let index = (0..n * (at.z_levels - 1))
        .find(|&i| e[i] == -8 && e[i + n] > -8)
        .unwrap();
    let mut edited = e.to_vec();
    edited[index] = 8;
    edited[index + n] -= 1;
    edited
}

/// The range session reads a table other than `E`, every digit in the
/// alphabet and every affine row holding: only the range image's link to
/// `E` in the fused sumcheck can reject.
fn unlinked_range(_: &ResponseLayoutPositions, e: &[i8]) -> CheatDigits {
    let mut range = e.to_vec();
    range[0] = if range[0] == 7 { 6 } else { range[0] + 1 };
    CheatDigits {
        committed: e.to_vec(),
        rows: e.to_vec(),
        range,
    }
}

impl<const D: usize> ResponseProver<F, D> for Cheating<'_> {
    fn prove_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        let digits = (self.edit)(&ResponseLayoutPositions::new(statement), &witness.digits);
        assert_eq!(
            statement.check_witness(&ResponseWitness {
                digits: digits.committed.clone()
            }),
            self.expected.clone()
        );
        self.akita.prove(
            statement,
            ProverDigits {
                committed: &digits.committed,
                rows: &digits.rows,
                range: &digits.range,
            },
            transcript,
        )
    }

    fn verify_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        self.akita.verify_response(statement, transcript)
    }
}

/// The verifier's rejection, if it is a statement rejection.
fn rejection(error: &ZkError) -> Option<&WhiteoutError> {
    match error {
        ZkError::Whiteout(source) => source.downcast_ref::<WhiteoutError>(),
        _ => None,
    }
}

/// W5-a: the verifier rejects each violation the statement's checker names,
/// at the check that catches it: Items 2–4 by editing `e`, Item 5 on a
/// statement whose relation fails. Item 1: the response schedule refuses to
/// commit a digit outside `[−8, 7]`, the range session cannot hold one, and
/// Stage 1's leaf rejects any range image off its roots (akita-verifier's
/// stage-1 tests); here the fused sumcheck rejects a range table other than
/// the committed `E`, the link that ties Stage 1 to `E`.
#[test]
fn rejects_each_item_violation() {
    on_big_stack(|| {
        let sigma = SyntheticSigma::<64>::new(3, 1);
        let akita = akita_for(sigma.response_len());
        let rejected = |item| Err(WhiteoutError::ResponseRejected(item));
        let cases: [(Edit, Result<(), WhiteoutError>, &str); 4] = [
            (unlinked_range, Ok(()), "fused sumcheck"),
            (
                |at, e| bumped(e, at.v),
                rejected("V commitment"),
                "fused sumcheck",
            ),
            (
                |_, e| bumped(e, 0),
                rejected("ring-linear rows"),
                "product sumcheck",
            ),
            (
                |at, e| bumped(e, at.theta),
                rejected("product quotient θ"),
                "product sumcheck",
            ),
        ];
        let mut observed = Vec::new();
        for (edit, expected, _) in cases.iter().cloned() {
            let cheating = Cheating {
                akita: &akita,
                edit,
                expected,
            };
            let proof = sigma.prove(&cheating, 4).unwrap();
            observed.push(sigma.verify(&proof, &akita).unwrap_err());
        }
        let observed: Vec<_> = observed.iter().map(rejection).collect();
        let expected: Vec<_> = cases
            .iter()
            .map(|(_, _, check)| Some(WhiteoutError::ResponseRejected(check)))
            .collect();
        assert_eq!(
            observed,
            expected.iter().map(Option::as_ref).collect::<Vec<_>>()
        );

        let committing = Cheating {
            akita: &akita,
            edit: |at, e| CheatDigits::everywhere(out_of_alphabet(at, e)),
            expected: rejected("digit outside its alphabet"),
        };
        let refused = sigma.prove(&committing, 5).unwrap_err();
        assert!(
            format!("{refused:?}").contains("coefficient bound"),
            "{refused:?}"
        );

        let violating = SyntheticSigma::<64>::violating_item_five(6, 1);
        let checked = violating.prove(&CheckingResponseProver, 7).unwrap_err();
        assert!(format!("{checked:?}").contains("ct(h)"), "{checked:?}");
        let cheating = Cheating {
            akita: &akita,
            edit: |_, e| CheatDigits::everywhere(e.to_vec()),
            expected: rejected("ct(h)"),
        };
        let proof = violating.prove(&cheating, 7).unwrap();
        let error = violating.verify(&proof, &akita).unwrap_err();
        assert_eq!(
            rejection(&error),
            Some(&WhiteoutError::ResponseRejected("fused sumcheck")),
            "{error:?}"
        );
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
