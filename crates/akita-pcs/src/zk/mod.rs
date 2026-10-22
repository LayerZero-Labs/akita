//! Whiteout's response proof on Akita (`specs/whiteout-akita.md` §5; Whiteout
//! §4.3, "Proving the response"). [`AkitaResponseProver`] proves jolt-zk's
//! response statement with an ordinary Akita commitment and opening:
//!
//! 1. commit to `e`, without randomness;
//! 2. draw `α`, send `t_×`, run the product sumcheck, send `u_×` and `w_×`;
//! 3. draw `η` and batch every affine row into `⟨A, e⟩ = b_η`;
//! 4. run Akita's digit-range reduction at the statement's radix, yielding
//!    `s_rng` at a fresh point;
//! 5. draw `γ`, run the fused sumcheck, send `v`, check
//!    `A(r)v + B(r)v(v + 1)`;
//! 6. open `E(r) = v` under the response schedule.
//!
//! The verifier evaluates `A(r)` from the transposed rows and scans the
//! setup's prefix views directly (W5a); it never builds `H`. Test support
//! until the zero-knowledge mode is published (§5.5).

mod rounds;
mod rows;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use akita_algebra::EqPolynomial;
use akita_config::{derive_digit_range_grinding_plan, CommitmentConfig};
use akita_cpu_backend::standalone::DigitRangeProver;
use akita_cpu_backend::{AkitaProverSetup, CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_params::{
    BasisMode, DigitRangePlan, FlatBooleanDomain, OpeningClaimsLayout, PolynomialGroupLayout,
    RelationWitnessGeometry, ScheduleLookupKey,
};
use akita_prover::{prove_digit_range, SelectedProverOpeningData};
use akita_sumcheck::{multilinear_eval, FusedRounds};
use akita_types::{
    AkitaVerifierSetup, Commitment, CommittedGroup, DigitRangeEqualityPoint, GroupBatchStatement,
    OpeningClaims, PolynomialGroupClaims, ProverGrinding, RingVec, VerifierGrinding,
};
use akita_verifier::{AkitaVerifier, Stage1Verifier};
use jolt_algebra::{One, Prime128OffsetA7F7, Ring, Zero};
use jolt_sumcheck::{prove_single, verify_single, SingleFormat, SingleInstance};
use jolt_transcript::{Channel, ProverTranscript, Sponge, VerifierTranscript};
use jolt_zk::masking::ZkError;
use jolt_zk::whiteout::{ResponseProver, ResponseStatement, ResponseWitness, WhiteoutError};

use crate::AkitaCommitmentScheme;
use rounds::{ClearProverRounds, ClearVerifierRounds, FusedKernel, ProductKernel};
use rows::{
    batch_rows, ceil_log2, eq_table, row_variables, zk_error, ProductClaims, ResponseShape,
    RowWeights,
};

/// The response statement's field.
type F = Prime128OffsetA7F7;

/// Grinding-plan level of the standalone digit-range reduction.
const RANGE_LEVEL: u32 = 0;

/// The digit-range plan of `e`'s alphabet. The response schedule commits
/// exactly that alphabet: its signed bound is the radix's bit width.
fn range_plan<Cfg: CommitmentConfig, const D: usize>(
    statement: &ResponseStatement<'_, F, D>,
) -> Result<DigitRangePlan, AkitaError> {
    let log_basis = statement.digit_log_basis();
    if Cfg::decomposition().log_commit_bound != log_basis {
        return Err(AkitaError::InvalidSetup(format!(
            "response schedule commits {}-bit digits, the statement's alphabet has {log_basis}",
            Cfg::decomposition().log_commit_bound
        )));
    }
    DigitRangePlan::new(1 << log_basis)
}

/// Proves jolt-zk's Whiteout response statement with Akita.
pub struct AkitaResponseProver<Cfg: CommitmentConfig> {
    scheme: AkitaCommitmentScheme<Cfg>,
    verifier: AkitaVerifier<Cfg>,
    prover: Option<ProverSetup>,
}

/// The digits each step of the prover reads. An honest prover reads `e` in
/// all three; tests split them to reach the verifier with a false
/// statement.
#[derive(Clone, Copy)]
struct ProverDigits<'a> {
    /// The committed table `E`.
    committed: &'a [i8],
    /// What the product claims and their rows are computed from.
    rows: &'a [i8],
    /// The digit-range session's table.
    range: &'a [i8],
}

impl<'a> ProverDigits<'a> {
    fn honest(digits: &'a [i8]) -> Self {
        Self {
            committed: digits,
            rows: digits,
            range: digits,
        }
    }
}

/// The prover's setup and the backend that holds it.
struct ProverSetup {
    setup: AkitaProverSetup<F>,
    backend: CpuBackend<F, F>,
}

impl<Cfg> AkitaResponseProver<Cfg>
where
    Cfg: CommitmentConfig<Field = F, ExtField = F> + 'static,
{
    /// A prover and verifier over `scheme`, whose catalog holds the response
    /// schedule of every `e` it proves, with `setup` sized for them.
    pub fn new(
        scheme: AkitaCommitmentScheme<Cfg>,
        setup: AkitaProverSetup<F>,
    ) -> Result<Self, AkitaError> {
        let verifier = scheme.verifier(scheme.setup_verifier(&setup)?)?;
        let backend = CpuBackend::new(setup.expanded.clone())?;
        Ok(Self {
            scheme,
            verifier,
            prover: Some(ProverSetup { setup, backend }),
        })
    }

    /// A verifier only.
    pub fn verifier(
        scheme: AkitaCommitmentScheme<Cfg>,
        setup: AkitaVerifierSetup<F>,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            verifier: scheme.verifier(setup)?,
            scheme,
            prover: None,
        })
    }

    /// Proves the statement on `digits`, without checking that it holds:
    /// [`ResponseProver::prove_response`] checks first.
    fn prove<H: Sponge, const D: usize>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        digits: ProverDigits<'_>,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        let side = self.prover.as_ref().ok_or_else(|| {
            zk_error(AkitaError::InvalidInput(
                "a verifier-only response prover cannot prove".into(),
            ))
        })?;
        let shape = ResponseShape::new(statement.layout().len()).map_err(zk_error)?;
        let plan = range_plan::<Cfg, D>(statement).map_err(zk_error)?;
        if [digits.committed, digits.rows, digits.range]
            .iter()
            .any(|digits| digits.len() != shape.len)
        {
            return Err(WhiteoutError::ResponseRejected("response length").into());
        }
        let mut e: Vec<F> = digits
            .committed
            .iter()
            .map(|&digit| F::from_i64(i64::from(digit)))
            .collect();
        e.resize(shape.padded_len(), F::zero());

        // 1. Commit to e.
        let poly = DensePoly::from_field_evals(shape.num_vars, &e).map_err(zk_error)?;
        let source = side.backend.import_source(vec![poly]).map_err(zk_error)?;
        let output = side
            .backend
            .commit(
                self.scheme.schedules(),
                &source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .map_err(zk_error)?;
        transcript.send_all(output.committed_group.rows().coeffs());

        // 2. The product reduction at α.
        let alpha: F = transcript.challenge();
        let rows = statement.rows(digits.rows, alpha)?;
        let product_vars = ceil_log2(rows.product_x.len().max(1));
        let mut product = ProductKernel {
            u: padded(rows.product_x, product_vars),
            w: padded(rows.product_y, product_vars),
        };
        // t_× is the product lift's right side at α (Whiteout (4)); for a
        // true statement it equals Σ_i U_i W_i.
        let target = rows.product_target;
        transcript.send(&target);
        let product_point = prove_single::<F, F, AkitaError, _>(
            &mut FusedRounds(&mut product),
            &compressed(target, product_vars, 2),
            &mut ClearProverRounds(transcript),
        )
        .map_err(zk_error)?
        .challenges;
        let claims = ProductClaims {
            target,
            x: first(&product.u)?,
            y: first(&product.w)?,
        };
        transcript.send(&claims.x);
        transcript.send(&claims.y);

        // 3. The row batch at η.
        let eta: Vec<F> = challenges(transcript, row_variables(statement, shape));
        let weights = RowWeights::new(statement, shape, &eta, &product_point)?;
        let batch = batch_rows(statement, shape, alpha, weights, claims)?;

        // 4. The digit range of e.
        let tau: Vec<F> = challenges(transcript, shape.num_vars);
        let grinding_plan =
            derive_digit_range_grinding_plan::<Cfg>(plan, shape.num_vars, RANGE_LEVEL)
                .map_err(zk_error)?;
        let equality_point =
            DigitRangeEqualityPoint::from_column_then_ring_challenges(&tau, shape.num_vars, 0)
                .map_err(zk_error)?;
        let mut session = DigitRangeProver::new(
            Arc::from(digits.range),
            plan,
            FlatBooleanDomain::new(shape.len, shape.num_vars).map_err(zk_error)?,
            equality_point.clone(),
        )
        .and_then(DigitRangeProver::into_session)
        .map_err(zk_error)?;
        let mut grinding = ProverGrinding::new(transcript, &grinding_plan);
        let range = prove_digit_range::<F, F, _, _>(
            &mut session,
            plan,
            equality_point.into_coordinates(),
            &mut grinding,
            RANGE_LEVEL,
        )
        .map_err(zk_error)?;
        grinding.finish().map_err(zk_error)?;

        // 5. The fused range-and-relation sumcheck.
        let gamma: F = transcript.challenge();
        let mut fused = FusedKernel {
            a: batch.a,
            b: eq_table(&range.point)
                .into_iter()
                .map(|weight| weight * gamma)
                .collect(),
            e,
        };
        let claim = batch.b + gamma * range.range_image;
        let point = prove_single::<F, F, AkitaError, _>(
            &mut FusedRounds(&mut fused),
            &compressed(claim, shape.num_vars, 3),
            &mut ClearProverRounds(transcript),
        )
        .map_err(zk_error)?
        .challenges;
        let v = first(&fused.e)?;
        transcript.send(&v);

        // 6. Open E(r) = v.
        let group =
            PolynomialGroupClaims::new(point, vec![v], output.committed_group).map_err(zk_error)?;
        let opening = SelectedProverOpeningData::from_committed_claims::<Cfg>(
            OpeningClaims::from_groups(vec![group]).map_err(zk_error)?,
            vec![output.private_handle],
            self.scheme.schedules(),
        )
        .map_err(zk_error)?;
        self.scheme
            .prove_on_transcript(
                &side.setup,
                opening,
                &side.backend,
                transcript,
                BasisMode::Lagrange,
            )
            .map_err(zk_error)
    }

    fn verify<H: Sponge, const D: usize>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        let shape = ResponseShape::new(statement.layout().len()).map_err(zk_error)?;
        let row = self
            .scheme
            .schedules()
            .resolve_key(&ScheduleLookupKey::single(
                PolynomialGroupLayout::singleton(shape.num_vars),
            ))
            .map_err(zk_error)?;
        let opening_layout = OpeningClaimsLayout::new(shape.num_vars, 1).map_err(zk_error)?;
        let geometry =
            RelationWitnessGeometry::for_level(&row.schedule().root.params, &opening_layout, 1)
                .map_err(zk_error)?;
        let compression = geometry
            .rhs_layout()
            .compression_plan_for_group(0)
            .map_err(zk_error)?;
        let ring_dim = compression
            .maps()
            .last()
            .ok_or_else(|| {
                zk_error(AkitaError::InvalidSetup(
                    "response commitment has no compression map".into(),
                ))
            })?
            .ring_dimension();
        let coefficients: Vec<F> = transcript.receive_n(compression.terminal_coefficients())?;
        let committed = CommittedGroup::new(
            row.profiles().final_group,
            Commitment::new(
                RingVec::from_coeffs_with_ring_dim(coefficients, ring_dim).map_err(zk_error)?,
            ),
        );

        // 2. The product reduction at α.
        let alpha: F = transcript.challenge();
        let product_vars = ceil_log2(statement.row_shape().products.max(1));
        let target: F = transcript.receive()?;
        let product = verify_single::<F, F, AkitaError, _>(
            &compressed(target, product_vars, 2),
            &mut ClearVerifierRounds(transcript),
        )
        .map_err(zk_error)?;
        let claims = ProductClaims {
            target,
            x: transcript.receive()?,
            y: transcript.receive()?,
        };
        if claims.x * claims.y != product.output_claim {
            return Err(WhiteoutError::ResponseRejected("product sumcheck").into());
        }

        // 3. The row batch at η.
        let eta: Vec<F> = challenges(transcript, row_variables(statement, shape));
        let weights = RowWeights::new(statement, shape, &eta, &product.challenges)?;
        let batch = batch_rows(statement, shape, alpha, weights, claims)?;

        // 4. The digit range of e.
        let tau: Vec<F> = challenges(transcript, shape.num_vars);
        let plan = range_plan::<Cfg, D>(statement).map_err(zk_error)?;
        let grinding_plan =
            derive_digit_range_grinding_plan::<Cfg>(plan, shape.num_vars, RANGE_LEVEL)
                .map_err(zk_error)?;
        let mut grinding = VerifierGrinding::new(transcript, &grinding_plan);
        let range = Stage1Verifier::new(
            DigitRangeEqualityPoint::from_column_then_ring_challenges(&tau, shape.num_vars, 0)
                .map_err(zk_error)?,
            plan,
        )
        .verify::<F, _>(&mut grinding, None, RANGE_LEVEL)
        .map_err(zk_error)?;
        grinding.finish().map_err(zk_error)?;

        // 5. The fused range-and-relation sumcheck.
        let gamma: F = transcript.challenge();
        let fused = verify_single::<F, F, AkitaError, _>(
            &compressed(
                batch.b + gamma * range.range_image_evaluation,
                shape.num_vars,
                3,
            ),
            &mut ClearVerifierRounds(transcript),
        )
        .map_err(zk_error)?;
        let v: F = transcript.receive()?;
        let a = multilinear_eval(&batch.a, &fused.challenges).map_err(zk_error)?;
        let b = gamma * EqPolynomial::mle(&range.point, &fused.challenges).map_err(zk_error)?;
        if a * v + b * v * (v + F::one()) != fused.output_claim {
            return Err(WhiteoutError::ResponseRejected("fused sumcheck").into());
        }

        // 6. Open E(r) = v.
        let group =
            PolynomialGroupClaims::new(fused.challenges, vec![v], &committed).map_err(zk_error)?;
        let opening = GroupBatchStatement::new(
            row.selection(),
            OpeningClaims::from_groups(vec![group]).map_err(zk_error)?,
        )
        .map_err(zk_error)?;
        self.verifier
            .batched_verify(transcript, opening, BasisMode::Lagrange)
            .map_err(zk_error)
    }
}

impl<Cfg, const D: usize> ResponseProver<F, D> for AkitaResponseProver<Cfg>
where
    Cfg: CommitmentConfig<Field = F, ExtField = F> + 'static,
{
    fn prove_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        statement.check_witness(witness)?;
        self.prove(statement, ProverDigits::honest(&witness.digits), transcript)
    }

    fn verify_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        self.verify(statement, transcript)
    }
}

/// A compressed single-instance sumcheck of `degree` over `rounds` variables.
fn compressed(input_claim: F, rounds: usize, degree: usize) -> SingleInstance<F> {
    SingleInstance {
        input_claim,
        rounds,
        degree,
        format: SingleFormat::Compressed,
        invocation: 0,
    }
}

/// `values`, zero-padded to `2^num_vars`.
fn padded(mut values: Vec<F>, num_vars: usize) -> Vec<F> {
    values.resize(1 << num_vars, F::zero());
    values
}

/// The value a table bound in every variable holds.
fn first(table: &[F]) -> Result<F, ZkError> {
    table
        .first()
        .copied()
        .ok_or_else(|| zk_error(AkitaError::Internal("a bound table is empty".into())))
}

fn challenges<C: Channel>(channel: &mut C, count: usize) -> Vec<F> {
    (0..count).map(|_| channel.challenge()).collect()
}
