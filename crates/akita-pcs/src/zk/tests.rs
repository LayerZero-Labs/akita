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
use jolt_algebra::{Field, Zero};
use jolt_zk::whiteout::test_support::{CheckingResponseProver, SyntheticSigma};
use jolt_zk::whiteout::ResponseSegment;
use std::sync::Mutex;

use akita_params::PackedNegativeBinary;
use akita_sumcheck::multilinear_eval;
use jolt_zk::whiteout::{CompressedImage, CompressedTarget};
use rand::RngCore;

type Cfg = RecursiveCommitmentConfig<ResponseDigits>;

/// A response prover whose catalog holds the planner's response schedule for
/// `e` of `len` entries.
fn akita_for(len: usize) -> AkitaResponseProver<Cfg> {
    let (scheme, num_vars) = scheme_for(len);
    let setup = AkitaResponseProver::setup(&scheme, num_vars).unwrap();
    AkitaResponseProver::new(scheme, setup).unwrap()
}

/// A scheme whose catalog holds the planner's response schedule for `e` of
/// `len` entries, and that schedule's variable count.
fn scheme_for(len: usize) -> (AkitaCommitmentScheme<Cfg>, usize) {
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
    (scheme, num_vars)
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
            let akita = akita_for(sigma.response_len(&CompressionShape::<Cfg>::default()));
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
    /// The compression the prover's images and witnesses come from.
    compression: &'a dyn ImageCompression<F>,
    edit: Edit,
    expected: Result<(), WhiteoutError>,
}

/// The committed, rows and range digits of a false statement.
struct CheatDigits {
    committed: Vec<i8>,
    rows: Vec<i8>,
    range: Vec<i8>,
    padding: Vec<i8>,
}

impl CheatDigits {
    /// One edited `e` in every role.
    fn everywhere(e: Vec<i8>) -> Self {
        Self {
            committed: e.clone(),
            rows: e.clone(),
            range: e,
            padding: Vec::new(),
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
    /// The first `ξ₂` digit of the first compressed image's first chunk.
    xi_2: usize,
}

impl ResponseLayoutPositions {
    fn new<const D: usize>(statement: &ResponseStatement<'_, F, D>) -> Self {
        let mut offset = 0;
        let (mut v, mut theta, mut z) = (0, 0, None);
        for segment in statement.layout().unwrap().segments {
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
                ResponseSegment::Compression { .. } => {}
            }
            offset += segment.num_digits();
        }
        let (z_coefficients, z_levels) = z.unwrap();
        let first = statement.compressed_targets().unwrap()[0].clone();
        let chunk = compression::chunks(Cfg::sis_modulus_profile(), first.image_len).unwrap();
        let xi_1 = chunk[0].plan.maps()[0].padded_digit_count();
        Self {
            z_coefficients,
            z_levels,
            v,
            theta,
            xi_2: first.digits.start + xi_1,
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
        padding: Vec::new(),
    }
}

/// `e` holds, but a `−1` is committed past its end: `−1` is in the digit
/// alphabet and a root of `E(E + 1)`, and the padding lies outside `I_bin`,
/// so only the padding term can reject.
fn nonzero_padding(_: &ResponseLayoutPositions, e: &[i8]) -> CheatDigits {
    CheatDigits {
        padding: vec![-1],
        ..CheatDigits::everywhere(e.to_vec())
    }
}

impl<const D: usize> ResponseProver<F, D> for Cheating<'_> {
    fn compression(&self) -> &dyn ImageCompression<F> {
        self.compression
    }

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
                padding: &digits.padding,
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
/// at the check that catches it: Items 2–4 by editing `e` (Item 2's
/// compression chain by flipping a `ξ₂` digit, which leaves the image `ξ₁`
/// reconstructs unchanged), Item 5 on a statement whose relation fails. Item 1: the response schedule refuses to
/// commit a digit outside `[−8, 7]`, the range session cannot hold one, and
/// Stage 1's leaf rejects any range image off its roots (akita-verifier's
/// stage-1 tests); here the fused sumcheck rejects a range table other than
/// the committed `E`, the link that ties Stage 1 to `E`. A nonzero entry
/// committed past `e`'s end fails the padding term.
#[test]
fn rejects_each_item_violation() {
    on_big_stack(|| {
        let sigma = SyntheticSigma::<64>::new(3, 1);
        let akita = akita_for(sigma.response_len(&CompressionShape::<Cfg>::default()));
        let rejected = |item| Err(WhiteoutError::ResponseRejected(item));
        let cases: [(Edit, Result<(), WhiteoutError>, &str); 6] = [
            (
                |at, e| {
                    let mut e = e.to_vec();
                    e[at.xi_2] = -1 - e[at.xi_2];
                    CheatDigits::everywhere(e)
                },
                rejected("compressed image"),
                "fused sumcheck",
            ),
            (unlinked_range, Ok(()), "fused sumcheck"),
            (nonzero_padding, Ok(()), "fused sumcheck"),
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
                compression: &akita,
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
            compression: &akita,
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
            compression: &akita,
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
struct WeightOracle<'a> {
    akita: &'a AkitaResponseProver<Cfg>,
}

impl<const D: usize> ResponseProver<F, D> for WeightOracle<'_> {
    fn compression(&self) -> &dyn ImageCompression<F> {
        self.akita
    }

    fn prove_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        let shape = ResponseShape::new(statement.layout().unwrap().len()).unwrap();
        let chains = self.akita.chains(statement).unwrap();
        let alpha: F = transcript.challenge();
        let product_vars = ceil_log2(statement.row_shape().products.max(1));
        let product_point = challenges(transcript, product_vars);
        let eta = challenges(transcript, row_variables(statement, &chains).unwrap());
        let padding = padding_term(transcript, shape);
        let r = challenges(transcript, shape.num_vars);
        let range_point = challenges(transcript, shape.num_vars);
        let gamma: F = transcript.challenge();
        let zeta: F = transcript.challenge();
        let claims = ProductClaims {
            target: transcript.challenge(),
            x: transcript.challenge(),
            y: transcript.challenge(),
        };

        // jolt-zk's rows and the padding term; the chains' rows have their
        // own test against their definition.
        let mut weights = RowWeights::new(statement, &eta, &product_point, &chains).unwrap();
        weights.chains.fill(F::zero());
        let zero = statement.rows(&vec![0; shape.len], alpha).unwrap();
        let mut columns: Vec<F> = cfg_into_iter!(0..shape.len)
            .map(|j| {
                let mut unit = vec![0i8; shape.len];
                unit[j] = 1;
                let column = statement.rows(&unit, alpha).unwrap();
                weights.rows.dot(&column).unwrap() - weights.rows.dot(&zero).unwrap()
            })
            .collect();
        let eq_pad = EqPolynomial::evals(&padding.point).unwrap();
        columns.extend(eq_pad[shape.len..].iter().copied());
        let eq_r = EqPolynomial::evals(&r).unwrap();
        let contracted: F = columns.iter().zip(&eq_r).map(|(h, e)| *h * *e).sum();
        let b = weights.target(claims) - weights.rows.dot(&zero).unwrap();

        let batch =
            batch_rows(statement, shape, alpha, &weights, &padding, claims, &chains).unwrap();
        assert_eq!(multilinear_eval(&batch.a, &r).unwrap(), contracted);
        let at = |weights: &RowWeights<F>| {
            batch_weight_at(statement, shape, alpha, weights, &padding, &chains, &r).unwrap()
        };
        assert_eq!(at(&weights), contracted);
        assert_eq!(batch.b, b);

        // With the chains' rows: the verifier's factored A(r) against the
        // prover's dense A.
        let weights = RowWeights::new(statement, &eta, &product_point, &chains).unwrap();
        let batch =
            batch_rows(statement, shape, alpha, &weights, &padding, claims, &chains).unwrap();
        assert_eq!(at(&weights), multilinear_eval(&batch.a, &r).unwrap());
        assert_eq!(
            batch_target(statement, alpha, &weights, claims, &chains).unwrap(),
            batch.b
        );

        let mut binary = vec![false; shape.padded_len()];
        for target in statement.compressed_targets().unwrap() {
            binary[target.digits].fill(true);
        }
        let b_table: F = EqPolynomial::evals(&range_point)
            .unwrap()
            .iter()
            .zip(&eq_r)
            .zip(&binary)
            .map(|((eq, e), &binary)| (gamma + if binary { zeta } else { F::zero() }) * *eq * *e)
            .sum();
        assert_eq!(
            range_weight_at(
                gamma,
                zeta,
                &chains.support(shape.padded_len()).unwrap(),
                &range_point,
                &r
            )
            .unwrap(),
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

/// W5-b: on the smallest synthetic shape, the verifier's factored `A(r)`,
/// the prover's dense `A` at `r`, and `B(r)` equal `Σ_j eq(r, j)·Σ_l eq(η, l)·H_{l,j}` and `Σ_j eq(r, j)·B_j`, with
/// each column `H_{·,j}` read from the statement's own rows at the unit
/// vector `δ_j`.
#[test]
fn batched_weights_match_h_column_by_column() {
    on_big_stack(|| {
        let sigma = SyntheticSigma::<64>::new(8, 1);
        let akita = akita_for(sigma.response_len(&CompressionShape::<Cfg>::default()));
        sigma.prove(&WeightOracle { akita: &akita }, 9).unwrap();
    });
}

/// The chain relation against its definition: on digits in `{−1, 0}`,
/// `⟨A, ξ⟩ + c = Σ_m w_m·row_m(ξ)(α)` with each row
/// [`compression::chain_rows`] coefficient by coefficient, and its constant
/// `−w₂·cm(α)` from a nonzero payload; and Akita's executor's witness
/// reconstructs the image and satisfies every row.
#[test]
fn chain_relation_matches_its_rows_and_vanishes_on_the_executor_witness() {
    on_big_stack(|| {
        let sigma = SyntheticSigma::<64>::new(10, 1);
        let akita = akita_for(sigma.response_len(&CompressionShape::<Cfg>::default()));
        let stream = akita.expanded().shared_matrix().as_field_slice();
        let profile = Cfg::sis_modulus_profile();
        let mut rng = rand::thread_rng();
        for image_len in [64, 512, 640] {
            let image: Vec<F> = (0..image_len).map(|_| F::random(&mut rng)).collect();
            let compressed = akita.compress(&image).unwrap();
            assert_eq!(
                akita.reconstruct(image_len, &compressed.witness).unwrap(),
                image
            );
            let chunks = compression::chunks(profile, image_len).unwrap();
            // Over A7F7 every map's digits fill its ring rows: no padding.
            assert!(chunks.iter().all(|chunk| chunk
                .plan
                .maps()
                .iter()
                .all(|map| map.padded_digit_count() == map.real_digit_count())));
            let len = compressed.witness.len();
            let padded_len = len.next_power_of_two();
            let xi: Vec<i8> = (0..len).map(|_| -((rng.next_u32() & 1) as i8)).collect();
            let alpha = F::random(&mut rng);
            let weights: Vec<F> = (0..chunks.len() * 2).map(|_| F::random(&mut rng)).collect();
            let (mut digit, mut payload) = (0, compressed.payload.as_slice());
            let mut expected = F::zero();
            for (chunk, weights) in chunks.iter().zip(weights.chunks_exact(2)) {
                let width = chunk.witness_len().unwrap();
                let (head, tail) = payload.split_at(chunk.plan.terminal_coefficients());
                payload = tail;
                let honest = compression::chain_rows(
                    stream,
                    chunk,
                    &compressed.witness[digit..digit + width],
                    head,
                )
                .unwrap();
                assert!(honest.iter().flatten().all(F::is_zero));
                let rows = compression::chain_rows(stream, chunk, &xi[digit..digit + width], head)
                    .unwrap();
                for (weight, row) in weights.iter().zip(&rows) {
                    let at_alpha = row.iter().rev().fold(F::zero(), |sum, c| sum * alpha + *c);
                    expected += *weight * at_alpha;
                }
                digit += width;
            }
            let chains = compression::Chains::of_targets(
                profile,
                akita.expanded(),
                vec![CompressedTarget {
                    digits: 0..len,
                    image_len,
                    payload: &compressed.payload,
                }],
            );
            let mut a = vec![F::zero(); padded_len];
            chains.accumulate(alpha, &weights, &mut a).unwrap();
            let constant = chains.constant(alpha, &weights).unwrap();
            assert_ne!(constant, F::zero());
            let actual: F = a
                .iter()
                .zip(&xi)
                .map(|(a, &x)| *a * F::from_i64(i64::from(x)))
                .sum::<F>()
                + constant;
            assert_eq!(actual, expected);
        }
    });
}

/// A prover-chosen compression: Akita's executor, then one edit to the
/// chain of chunk `chunk` of every image of `image_len` coefficients.
struct Forging<'a> {
    akita: &'a AkitaResponseProver<Cfg>,
    forgery: Forgery,
    image_len: usize,
    chunk: usize,
}

#[derive(Clone, Copy, Debug)]
enum Forgery {
    /// A `ξ₁` coefficient's bits `(0, −1)` at `(b, b + 1)` rewritten as
    /// `(−2, 0)`, `ξ₂ = Bin(K₁ξ₁)` and `cm = K₂ξ₂` recomputed: the image
    /// and both chain rows are unchanged, one digit leaves `{−1, 0}`.
    Carry,
    /// A `ξ₂` digit past the first coefficient flipped and `cm = K₂ξ₂`
    /// recomputed: only `K₁`'s row fails.
    FlipXi2,
    /// A payload coefficient past the constant term moved: only `K₂`'s row
    /// fails.
    BumpPayload,
    /// `cm − (X − α₀)`: `K₂`'s residual is `X − α₀`, zero at `α₀` only.
    CancelAt(u64),
}

impl Forgery {
    /// Whether each chain row holds after the forgery.
    fn holds(self) -> [bool; 2] {
        match self {
            Forgery::Carry => [true, true],
            Forgery::FlipXi2 => [false, true],
            Forgery::BumpPayload | Forgery::CancelAt(_) => [true, false],
        }
    }
}

impl Forging<'_> {
    /// Edits `compressed` in place and returns the chunk's rows after the
    /// edit.
    fn forge(&self, compressed: &mut CompressedImage<F>) -> [Vec<F>; 2] {
        let stream = self.akita.expanded().shared_matrix().as_field_slice();
        let chunks = compression::chunks(Cfg::sis_modulus_profile(), self.image_len).unwrap();
        let chunk = &chunks[self.chunk];
        let digit: usize = chunks[..self.chunk]
            .iter()
            .map(|chunk| chunk.witness_len().unwrap())
            .sum();
        let start: usize = chunks[..self.chunk]
            .iter()
            .map(|chunk| chunk.plan.terminal_coefficients())
            .sum();
        let payload = &mut compressed.payload[start..start + chunk.plan.terminal_coefficients()];
        let witness = &mut compressed.witness[digit..digit + chunk.witness_len().unwrap()];
        let [first, second] = <[_; 2]>::try_from(chunk.plan.maps()).unwrap();
        let (xi_1, xi_2) = witness.split_at_mut(first.padded_digit_count());
        let n = chunk.coefficients.len();
        let recompute_cm = |xi_2: &[i8], payload: &mut [F]| {
            let cm = compression::apply_row(stream, second.ring_dimension(), xi_2);
            payload.copy_from_slice(&cm[..payload.len()]);
        };
        match self.forgery {
            Forgery::Carry => {
                let at = (0..n * (chunk.plan.field_bits() - 1))
                    .find(|&i| xi_1[i] == 0 && xi_1[i + n] == -1)
                    .unwrap();
                xi_1[at] = -2;
                xi_1[at + n] = 0;
                let j = compression::apply_row(stream, first.ring_dimension(), xi_1);
                let packed = PackedNegativeBinary::from_coefficients(
                    second,
                    &j[..second.input_coefficients()],
                )
                .unwrap();
                xi_2.fill(0);
                for (dst, digit) in xi_2.iter_mut().zip(compression::digits(&packed)) {
                    *dst = digit;
                }
                recompute_cm(xi_2, payload);
            }
            Forgery::FlipXi2 => {
                let at = second.real_digit_count() / 2 + 1;
                xi_2[at] = -1 - xi_2[at];
                recompute_cm(xi_2, payload);
            }
            Forgery::BumpPayload => payload[3] += F::one(),
            Forgery::CancelAt(alpha) => {
                payload[0] += F::from_u64(alpha);
                payload[1] -= F::one();
            }
        }
        compression::chain_rows(stream, chunk, witness, payload).unwrap()
    }
}

impl ImageCompression<F> for Forging<'_> {
    fn payload_len(&self, image_len: usize) -> Result<usize, WhiteoutError> {
        self.akita.payload_len(image_len)
    }

    fn witness_len(&self, image_len: usize) -> Result<usize, WhiteoutError> {
        self.akita.witness_len(image_len)
    }

    fn compress(&self, image: &[F]) -> Result<CompressedImage<F>, WhiteoutError> {
        let mut compressed = self.akita.compress(image)?;
        if image.len() == self.image_len {
            self.forge(&mut compressed);
        }
        Ok(compressed)
    }

    fn reconstruct(&self, image_len: usize, witness: &[i8]) -> Result<Vec<F>, WhiteoutError> {
        self.akita.reconstruct(image_len, witness)
    }

    fn reconstruct_transpose(
        &self,
        image_len: usize,
        weights: &[F],
    ) -> Result<Vec<F>, WhiteoutError> {
        self.akita.reconstruct_transpose(image_len, weights)
    }

    fn check(&self, image_len: usize, payload: &[F], witness: &[i8]) -> Result<(), WhiteoutError> {
        self.akita.check(image_len, payload, witness)
    }
}

/// Records the image length of each compressed target.
struct ImageLens<'a> {
    akita: &'a AkitaResponseProver<Cfg>,
    lens: Mutex<Vec<usize>>,
}

impl<const D: usize> ResponseProver<F, D> for ImageLens<'_> {
    fn compression(&self) -> &dyn ImageCompression<F> {
        self.akita
    }

    fn prove_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        _: &ResponseWitness,
        _: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        *self.lens.lock().unwrap() = statement
            .compressed_targets()?
            .iter()
            .map(|target| target.image_len)
            .collect();
        Ok(())
    }

    fn verify_response<H: Sponge>(
        &self,
        _: &ResponseStatement<'_, F, D>,
        _: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        Ok(())
    }
}

/// The prover chooses `cm`, so each chain constraint is load-bearing: a
/// carry inside `ξ₁` keeps the image and both chain rows and is caught
/// only by the binary constraint `ζ`; a `ξ₂` flip with `cm` recomputed
/// fails only `K₁`'s row; a moved payload fails only `K₂`'s row, and so
/// does `cm − (X − α₀)`, whose residual vanishes at `α₀` alone. Each
/// moves the chain relation off zero on a non-first chunk (the carry
/// excepted), and the verifier rejects each at the fused sumcheck on the
/// first image's chain and on `t_D`'s. The synthetic images fit one chunk.
#[test]
fn rejects_prover_chosen_compressions() {
    on_big_stack(|| {
        let sigma = SyntheticSigma::<64>::new(11, 3);
        let akita = akita_for(sigma.response_len(&CompressionShape::<Cfg>::default()));
        let profile = Cfg::sis_modulus_profile();
        let recorder = ImageLens {
            akita: &akita,
            lens: Mutex::new(Vec::new()),
        };
        sigma.prove(&recorder, 12).unwrap();
        let lens = recorder.lens.into_inner().unwrap();
        let (first, t_d) = (lens[0], *lens.last().unwrap());
        assert_eq!(lens.iter().filter(|&&len| len == t_d).count(), 1);
        let rejected = |item| Err(WhiteoutError::ResponseRejected(item));
        let mut rng = rand::thread_rng();
        const ALPHA_0: u64 = 5;
        for forgery in [
            Forgery::Carry,
            Forgery::FlipXi2,
            Forgery::BumpPayload,
            Forgery::CancelAt(ALPHA_0),
        ] {
            // The chain relation at a random α, on a non-first chunk.
            let image_len = 640;
            let forging = Forging {
                akita: &akita,
                forgery,
                image_len,
                chunk: 1,
            };
            let image: Vec<F> = (0..image_len).map(|_| F::random(&mut rng)).collect();
            let mut compressed = akita.compress(&image).unwrap();
            let rows = forging.forge(&mut compressed);
            let holds = forgery.holds();
            assert_eq!(
                rows.map(|row| row.iter().all(F::is_zero)),
                holds,
                "{forgery:?}"
            );
            assert_eq!(
                akita.reconstruct(image_len, &compressed.witness).unwrap(),
                image
            );
            let chains = compression::Chains::of_targets(
                profile,
                akita.expanded(),
                vec![CompressedTarget {
                    digits: 0..compressed.witness.len(),
                    image_len,
                    payload: &compressed.payload,
                }],
            );
            // Each of chunk 1's rows alone, at a random α: zero exactly when
            // that row holds.
            let row_value = |row: usize, alpha: F| {
                let mut weights = vec![F::zero(); 4];
                weights[2 + row] = F::one();
                let mut a = vec![F::zero(); compressed.witness.len().next_power_of_two()];
                chains.accumulate(alpha, &weights, &mut a).unwrap();
                a.iter()
                    .zip(&compressed.witness)
                    .map(|(a, &x)| *a * F::from_i64(i64::from(x)))
                    .sum::<F>()
                    + chains.constant(alpha, &weights).unwrap()
            };
            let alpha = F::random(&mut rng);
            assert_eq!(
                [0, 1].map(|row| row_value(row, alpha).is_zero()),
                holds,
                "{forgery:?}"
            );
            if let Forgery::CancelAt(alpha_0) = forgery {
                assert!(row_value(1, F::from_u64(alpha_0)).is_zero());
            }

            // End to end, on the first image's chain and on t_D's.
            for image_len in [first, t_d] {
                let forging = Forging {
                    akita: &akita,
                    forgery,
                    image_len,
                    chunk: 0,
                };
                let cheating = Cheating {
                    akita: &akita,
                    compression: &forging,
                    edit: |_, e| CheatDigits::everywhere(e.to_vec()),
                    expected: rejected("compressed image"),
                };
                let proof = sigma.prove(&cheating, 13).unwrap();
                let error = sigma.verify(&proof, &akita).unwrap_err();
                assert_eq!(
                    rejection(&error),
                    Some(&WhiteoutError::ResponseRejected("fused sumcheck")),
                    "{forgery:?} on an image of {image_len}: {error:?}"
                );
            }
        }
    });
}

/// The response schedule's own setup is narrower than the compression
/// chains' maps; the prover refuses it, and [`AkitaResponseProver::setup`]
/// covers both. Generated without the disk cache, which may hold a wider
/// matrix.
#[test]
fn refuses_a_setup_without_the_chains_maps() {
    let (scheme, num_vars) = scheme_for(1 << 18);
    let capacity = SetupRequirements::from_catalog::<Cfg>(scheme.schedules(), num_vars, 1)
        .unwrap()
        .matrix_capacity();
    let needed = compression::setup_len(Cfg::sis_modulus_profile()).unwrap();
    assert!(capacity.num_field_elements < needed);
    let narrow = AkitaProverSetup::generate_with_capacity(num_vars, 1, capacity).unwrap();
    let error = AkitaResponseProver::new(scheme.clone(), narrow)
        .err()
        .unwrap();
    assert!(
        format!("{error:?}").contains("compression chains"),
        "{error:?}"
    );
    let setup = AkitaResponseProver::setup(&scheme, num_vars).unwrap();
    assert!(setup.expanded.shared_matrix().num_field_elements() >= needed);
    AkitaResponseProver::new(scheme, setup).unwrap();
}

/// The chains' relation at a point against the MLE of its dense weights,
/// for images of one to three chunks placed at unaligned offsets.
#[test]
fn chain_relation_at_a_point_matches_its_dense_weights() {
    on_big_stack(|| {
        let sigma = SyntheticSigma::<64>::new(14, 1);
        let akita = akita_for(sigma.response_len(&CompressionShape::<Cfg>::default()));
        let profile = Cfg::sis_modulus_profile();
        let mut rng = rand::thread_rng();
        for (image_len, offset) in [(640, 2017), (1100, 3), (1024, 4159), (513, 1)] {
            let image: Vec<F> = (0..image_len).map(|_| F::random(&mut rng)).collect();
            let compressed = akita.compress(&image).unwrap();
            let len = compressed.witness.len();
            let chains = compression::Chains::of_targets(
                profile,
                akita.expanded(),
                vec![CompressedTarget {
                    digits: offset..offset + len,
                    image_len,
                    payload: &compressed.payload,
                }],
            );
            let rows = chains.rows().unwrap();
            let weights: Vec<F> = (0..rows).map(|_| F::random(&mut rng)).collect();
            let alpha = F::random(&mut rng);
            let padded = (offset + len).next_power_of_two();
            let mut a = vec![F::zero(); padded];
            chains.accumulate(alpha, &weights, &mut a).unwrap();
            let point: Vec<F> = (0..padded.trailing_zeros())
                .map(|_| F::random(&mut rng))
                .collect();
            assert_eq!(
                chains.evaluate(alpha, &weights, &point).unwrap(),
                multilinear_eval(&a, &point).unwrap(),
                "{image_len} at {offset}"
            );
        }
    });
}

/// The padding term's dense weights are `eq(τ_pad, ·)` on `[len, 2^ν)`,
/// and its value at a point is their MLE.
#[test]
fn padding_term_matches_its_dense_weights() {
    let mut rng = rand::thread_rng();
    for len in [1, 2, 3, 63, 64, 65, 1023, 1024, 1025, 4097] {
        let shape = ResponseShape::new(len).unwrap();
        let padding = PaddingTerm {
            point: (0..shape.num_vars).map(|_| F::random(&mut rng)).collect(),
        };
        let weights = padding.weights(shape).unwrap();
        let eq = EqPolynomial::evals(&padding.point).unwrap();
        assert_eq!(weights, eq[len..], "{len}");
        let mut dense = vec![F::zero(); len];
        dense.extend(&weights);
        let point: Vec<F> = (0..shape.num_vars).map(|_| F::random(&mut rng)).collect();
        assert_eq!(
            padding.at(shape, &point).unwrap(),
            multilinear_eval(&dense, &point).unwrap(),
            "{len}"
        );
    }
}
