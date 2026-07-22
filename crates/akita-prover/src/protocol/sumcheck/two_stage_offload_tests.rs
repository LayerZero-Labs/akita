use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::poly::multilinear_eval;
use akita_algebra::uni_poly::UniPoly;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, Prime128Offset275};
use akita_sumcheck::{
    SumcheckInstanceProver, SumcheckInstanceProverExt, SumcheckInstanceVerifier,
    SumcheckInstanceVerifierExt,
};
use akita_transcript::{labels, AkitaTranscript, Transcript};
use akita_types::BatchedStage3Geometry;

type F = Prime128Offset275;

fn evaluate_coefficients<E: FieldCore>(coefficients: &[E], value: E) -> E {
    coefficients
        .iter()
        .rev()
        .fold(E::zero(), |acc, coefficient| acc * value + *coefficient)
}

fn checked_round_count(len: usize, role: &'static str) -> Result<usize, AkitaError> {
    if len < 2 || !len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "{role} dense oracle length must be a power of two greater than one"
        )));
    }
    Ok(len.trailing_zeros() as usize)
}

fn ensure_same_len<E>(tables: &[&[E]], role: &'static str) -> Result<usize, AkitaError> {
    let len = tables.first().map_or(0, |table| table.len());
    checked_round_count(len, role)?;
    if tables.iter().any(|table| table.len() != len) {
        return Err(AkitaError::InvalidInput(format!(
            "{role} dense oracle tables have inconsistent lengths"
        )));
    }
    Ok(len)
}

fn fold_table<E: FieldCore>(table: &mut Vec<E>, challenge: E) {
    let next_len = table.len() / 2;
    for index in 0..next_len {
        let left = table[2 * index];
        let right = table[2 * index + 1];
        table[index] = left + challenge * (right - left);
    }
    table.truncate(next_len);
}

fn linear_at<E: FieldCore>(left: E, right: E, point: E) -> E {
    left + point * (right - left)
}

#[derive(Clone)]
struct OffloadedStage1DenseOracle<E: FieldCore> {
    witness: Vec<E>,
    range_image: Vec<E>,
    equality: Vec<E>,
    relation_weight: Vec<E>,
    leaf_coefficients: Vec<E>,
    relation_batch: E,
    input_claim: E,
    rounds: usize,
}

impl<E: FieldCore + FromPrimitiveInt> OffloadedStage1DenseOracle<E> {
    fn new(
        witness: Vec<E>,
        equality_point: &[E],
        relation_weight: Vec<E>,
        leaf_coefficients: Vec<E>,
        relation_batch: E,
    ) -> Result<Self, AkitaError> {
        if leaf_coefficients.is_empty() {
            return Err(AkitaError::InvalidInput(
                "offloaded Stage 1 leaf polynomial is empty".into(),
            ));
        }
        let range_image = witness
            .iter()
            .map(|value| *value * (*value + E::one()))
            .collect::<Vec<_>>();
        let equality = EqPolynomial::evals(equality_point)?;
        let len = ensure_same_len(
            &[&witness, &range_image, &equality, &relation_weight],
            "offloaded Stage 1",
        )?;
        let rounds = checked_round_count(len, "offloaded Stage 1")?;
        if equality_point.len() != rounds {
            return Err(AkitaError::InvalidPointDimension {
                expected: rounds,
                actual: equality_point.len(),
            });
        }
        let range_claim = equality
            .iter()
            .zip(&range_image)
            .map(|(&eq, &value)| eq * evaluate_coefficients(&leaf_coefficients, value))
            .fold(E::zero(), |acc, value| acc + value);
        let relation_claim = witness
            .iter()
            .zip(&relation_weight)
            .map(|(&w, &weight)| w * weight)
            .fold(E::zero(), |acc, value| acc + value);
        Ok(Self {
            witness,
            range_image,
            equality,
            relation_weight,
            leaf_coefficients,
            relation_batch,
            input_claim: range_claim + relation_batch * relation_claim,
            rounds,
        })
    }

    fn evaluate_round(&self, point: E) -> E {
        (0..self.witness.len() / 2)
            .map(|pair| {
                let left = 2 * pair;
                let right = left + 1;
                let witness = linear_at(self.witness[left], self.witness[right], point);
                let range_image = linear_at(self.range_image[left], self.range_image[right], point);
                let equality = linear_at(self.equality[left], self.equality[right], point);
                let relation_weight = linear_at(
                    self.relation_weight[left],
                    self.relation_weight[right],
                    point,
                );
                equality * evaluate_coefficients(&self.leaf_coefficients, range_image)
                    + self.relation_batch * witness * relation_weight
            })
            .fold(E::zero(), |acc, value| acc + value)
    }
}

impl<E: FieldCore + FromPrimitiveInt> SumcheckInstanceProver<E> for OffloadedStage1DenseOracle<E> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        self.leaf_coefficients.len()
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let evaluations = (0..=self.leaf_coefficients.len())
            .map(|point| self.evaluate_round(E::from_u64(point as u64)))
            .collect::<Vec<_>>();
        UniPoly::from_evals(&evaluations)
    }

    fn ingest_challenge(&mut self, _round: usize, challenge: E) {
        fold_table(&mut self.witness, challenge);
        fold_table(&mut self.range_image, challenge);
        fold_table(&mut self.equality, challenge);
        fold_table(&mut self.relation_weight, challenge);
    }
}

impl<E: FieldCore + FromPrimitiveInt> SumcheckInstanceVerifier<E>
    for OffloadedStage1DenseOracle<E>
{
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        self.leaf_coefficients.len()
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
        let witness = multilinear_eval(&self.witness, challenges)?;
        let range_image = multilinear_eval(&self.range_image, challenges)?;
        let equality = multilinear_eval(&self.equality, challenges)?;
        let relation_weight = multilinear_eval(&self.relation_weight, challenges)?;
        Ok(
            equality * evaluate_coefficients(&self.leaf_coefficients, range_image)
                + self.relation_batch * witness * relation_weight,
        )
    }
}

#[derive(Clone)]
struct WitnessReductionTerm<E: FieldCore> {
    equality: Vec<E>,
    witness: Vec<E>,
    current_claim: E,
    theta: E,
    native_rounds: usize,
}

impl<E: FieldCore + FromPrimitiveInt> WitnessReductionTerm<E> {
    fn round_poly(&self, total_rounds: usize, round: usize) -> UniPoly<E> {
        let inactive_rounds = total_rounds - self.native_rounds;
        if round < inactive_rounds {
            return UniPoly::from_coeffs(vec![half(self.current_claim)]);
        }
        let evaluations = (0..=3)
            .map(|point| {
                let point = E::from_u64(point);
                (0..self.witness.len() / 2)
                    .map(|pair| {
                        let left = 2 * pair;
                        let right = left + 1;
                        let equality = linear_at(self.equality[left], self.equality[right], point);
                        let witness = linear_at(self.witness[left], self.witness[right], point);
                        equality * (self.theta * witness + witness * (witness + E::one()))
                    })
                    .fold(E::zero(), |acc, value| acc + value)
            })
            .collect::<Vec<_>>();
        let mut polynomial = UniPoly::from_evals(&evaluations);
        let scale = lift_scale::<E>(inactive_rounds);
        for coefficient in &mut polynomial.coeffs {
            *coefficient *= scale;
        }
        polynomial
    }

    fn ingest(&mut self, total_rounds: usize, round: usize, challenge: E, round_poly: &UniPoly<E>) {
        self.current_claim = round_poly.evaluate(&challenge);
        if round >= total_rounds - self.native_rounds {
            fold_table(&mut self.equality, challenge);
            fold_table(&mut self.witness, challenge);
        }
    }
}

#[derive(Clone)]
struct SetupProductTerm<E: FieldCore> {
    setup: Vec<E>,
    weight: Vec<E>,
    current_claim: E,
    native_rounds: usize,
}

impl<E: FieldCore + FromPrimitiveInt> SetupProductTerm<E> {
    fn round_poly(&self, total_rounds: usize, round: usize) -> UniPoly<E> {
        let inactive_rounds = total_rounds - self.native_rounds;
        if round < inactive_rounds {
            return UniPoly::from_coeffs(vec![half(self.current_claim)]);
        }
        let evaluations = (0..=2)
            .map(|point| {
                let point = E::from_u64(point);
                (0..self.setup.len() / 2)
                    .map(|pair| {
                        let left = 2 * pair;
                        let right = left + 1;
                        linear_at(self.setup[left], self.setup[right], point)
                            * linear_at(self.weight[left], self.weight[right], point)
                    })
                    .fold(E::zero(), |acc, value| acc + value)
            })
            .collect::<Vec<_>>();
        let mut polynomial = UniPoly::from_evals(&evaluations);
        let scale = lift_scale::<E>(inactive_rounds);
        for coefficient in &mut polynomial.coeffs {
            *coefficient *= scale;
        }
        polynomial
    }

    fn ingest(&mut self, total_rounds: usize, round: usize, challenge: E, round_poly: &UniPoly<E>) {
        self.current_claim = round_poly.evaluate(&challenge);
        if round >= total_rounds - self.native_rounds {
            fold_table(&mut self.setup, challenge);
            fold_table(&mut self.weight, challenge);
        }
    }
}

fn half<E: FieldCore + FromPrimitiveInt>(value: E) -> E {
    value
        * E::from_u64(2)
            .inverse()
            .expect("Akita test fields have odd characteristic")
}

fn lift_scale<E: FieldCore + FromPrimitiveInt>(inactive_rounds: usize) -> E {
    (0..inactive_rounds).fold(E::one(), |scale, _| half(scale))
}

fn combine_polynomials<E: FieldCore>(
    setup: &UniPoly<E>,
    witness: &UniPoly<E>,
    eta: E,
) -> UniPoly<E> {
    let len = setup.coeffs.len().max(witness.coeffs.len());
    let mut coefficients = vec![E::zero(); len];
    for (index, coefficient) in setup.coeffs.iter().enumerate() {
        coefficients[index] += *coefficient;
    }
    for (index, coefficient) in witness.coeffs.iter().enumerate() {
        coefficients[index] += eta * *coefficient;
    }
    UniPoly::from_coeffs(coefficients)
}

#[derive(Clone)]
struct OffloadedStage2DenseOracle<E: FieldCore> {
    witness: WitnessReductionTerm<E>,
    setup: SetupProductTerm<E>,
    geometry: BatchedStage3Geometry,
    r1: Vec<E>,
    eta: E,
    input_claim: E,
    pending: Option<(UniPoly<E>, UniPoly<E>)>,
}

impl<E: FieldCore + FromPrimitiveInt> OffloadedStage2DenseOracle<E> {
    fn new(
        witness: Vec<E>,
        r1: Vec<E>,
        theta: E,
        setup: Vec<E>,
        setup_weight: Vec<E>,
        eta: E,
    ) -> Result<Self, AkitaError> {
        let witness_rounds = checked_round_count(witness.len(), "offloaded Stage 2 witness")?;
        if r1.len() != witness_rounds {
            return Err(AkitaError::InvalidPointDimension {
                expected: witness_rounds,
                actual: r1.len(),
            });
        }
        let setup_len = ensure_same_len(&[&setup, &setup_weight], "offloaded Stage 2 setup")?;
        let setup_rounds = checked_round_count(setup_len, "offloaded Stage 2 setup")?;
        let equality = EqPolynomial::evals(&r1)?;
        let range_image = witness
            .iter()
            .map(|value| *value * (*value + E::one()))
            .collect::<Vec<_>>();
        let witness_claim =
            theta * multilinear_eval(&witness, &r1)? + multilinear_eval(&range_image, &r1)?;
        let setup_claim = setup
            .iter()
            .zip(&setup_weight)
            .map(|(&value, &weight)| value * weight)
            .fold(E::zero(), |acc, value| acc + value);
        let geometry = BatchedStage3Geometry::new(witness_rounds, setup_rounds)?;
        Ok(Self {
            witness: WitnessReductionTerm {
                equality,
                witness,
                current_claim: witness_claim,
                theta,
                native_rounds: witness_rounds,
            },
            setup: SetupProductTerm {
                setup,
                weight: setup_weight,
                current_claim: setup_claim,
                native_rounds: setup_rounds,
            },
            geometry,
            r1,
            eta,
            input_claim: setup_claim + eta * witness_claim,
            pending: None,
        })
    }
}

impl<E: FieldCore + FromPrimitiveInt> SumcheckInstanceProver<E> for OffloadedStage2DenseOracle<E> {
    fn num_rounds(&self) -> usize {
        self.geometry.batched_rounds()
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, round: usize, _previous_claim: E) -> UniPoly<E> {
        let total_rounds = self.geometry.batched_rounds();
        let setup = self.setup.round_poly(total_rounds, round);
        let witness = self.witness.round_poly(total_rounds, round);
        let combined = combine_polynomials(&setup, &witness, self.eta);
        self.pending = Some((setup, witness));
        combined
    }

    fn ingest_challenge(&mut self, round: usize, challenge: E) {
        let (setup, witness) = self
            .pending
            .take()
            .expect("round polynomial precedes challenge ingestion");
        let total_rounds = self.geometry.batched_rounds();
        self.setup.ingest(total_rounds, round, challenge, &setup);
        self.witness
            .ingest(total_rounds, round, challenge, &witness);
    }
}

impl<E: FieldCore + FromPrimitiveInt> SumcheckInstanceVerifier<E>
    for OffloadedStage2DenseOracle<E>
{
    fn num_rounds(&self) -> usize {
        self.geometry.batched_rounds()
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
        let witness_point = self.geometry.witness_point(challenges)?;
        let setup_point = self.geometry.setup_point(challenges)?;
        let witness = multilinear_eval(&self.witness.witness, &witness_point)?;
        let equality = EqPolynomial::mle(&self.r1, &witness_point)?;
        let witness_term =
            equality * (self.witness.theta * witness + witness * (witness + E::one()));
        let setup = multilinear_eval(&self.setup.setup, &setup_point)?;
        let setup_weight = multilinear_eval(&self.setup.weight, &setup_point)?;
        Ok(
            self.geometry.setup_lift_scale::<E>()? * setup * setup_weight
                + self.eta * self.geometry.witness_lift_scale::<E>()? * witness_term,
        )
    }
}

fn prove_and_verify<Instance>(instance: Instance) -> (Vec<F>, usize)
where
    Instance: SumcheckInstanceProver<F> + SumcheckInstanceVerifier<F> + Clone,
{
    let mut prover = instance.clone();
    let verifier = instance;
    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/two-stage-offload");
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/two-stage-offload");
    let (proof, prover_point, _final_claim) = prover
        .prove::<F, _, _>(&mut prover_transcript, |transcript| {
            transcript.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .expect("dense oracle proof");
    let verifier_point = verifier
        .verify::<F, _, _>(&proof, &mut verifier_transcript, |transcript| {
            transcript.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .expect("dense oracle verification");
    assert_eq!(prover_point, verifier_point);
    (prover_point, proof.round_polys.len())
}

fn test_witness(len: usize) -> Vec<F> {
    (0..len)
        .map(|index| F::from_i64((index as i64 % 4) - 2))
        .collect()
}

fn test_table(len: usize, multiplier: u64, offset: u64) -> Vec<F> {
    (0..len)
        .map(|index| F::from_u64(multiplier * index as u64 + offset))
        .collect()
}

#[test]
fn offloaded_stage1_final_leaf_uses_ordinary_sumcheck_equation() {
    let witness = test_witness(8);
    let equality_point = vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)];
    let relation_weight = test_table(8, 11, 13);
    let leaf_coefficients = test_table(5, 17, 19);
    let oracle = OffloadedStage1DenseOracle::new(
        witness,
        &equality_point,
        relation_weight,
        leaf_coefficients,
        F::from_u64(23),
    )
    .expect("valid Stage 1 oracle");
    assert_eq!(oracle.leaf_coefficients.len(), 5);
    let (_point, rounds) = prove_and_verify(oracle);
    assert_eq!(rounds, 3);
}

#[test]
fn offloaded_stage2_projects_shorter_setup_term_from_common_suffix() {
    let oracle = OffloadedStage2DenseOracle::new(
        test_witness(8),
        vec![F::from_u64(29), F::from_u64(31), F::from_u64(37)],
        F::from_u64(41),
        test_table(4, 43, 47),
        test_table(4, 53, 59),
        F::from_u64(61),
    )
    .expect("valid Stage 2 oracle");
    let (point, rounds) = prove_and_verify(oracle.clone());
    assert_eq!(rounds, 3);
    assert_eq!(
        oracle.geometry.setup_point(&point).expect("setup point"),
        point[1..]
    );
    assert_eq!(
        oracle
            .geometry
            .witness_point(&point)
            .expect("witness point"),
        point
    );
}

#[test]
fn offloaded_stage2_projects_shorter_witness_term_from_common_suffix() {
    let oracle = OffloadedStage2DenseOracle::new(
        test_witness(4),
        vec![F::from_u64(67), F::from_u64(71)],
        F::from_u64(73),
        test_table(8, 79, 83),
        test_table(8, 89, 97),
        F::from_u64(101),
    )
    .expect("valid Stage 2 oracle");
    let (point, rounds) = prove_and_verify(oracle.clone());
    assert_eq!(rounds, 3);
    assert_eq!(
        oracle
            .geometry
            .witness_point(&point)
            .expect("witness point"),
        point[1..]
    );
    assert_eq!(
        oracle.geometry.setup_point(&point).expect("setup point"),
        point
    );
}

#[test]
fn offloaded_dense_oracles_reject_malformed_domains() {
    assert!(OffloadedStage1DenseOracle::new(
        test_witness(3),
        &[F::from_u64(1), F::from_u64(2)],
        test_table(3, 3, 5),
        test_table(3, 7, 11),
        F::from_u64(13),
    )
    .is_err());
    assert!(OffloadedStage2DenseOracle::new(
        test_witness(4),
        vec![F::from_u64(17)],
        F::from_u64(19),
        test_table(4, 23, 29),
        test_table(4, 31, 37),
        F::from_u64(41),
    )
    .is_err());
}
