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
//! setup's prefix views directly (W5a); it never builds `H`.
//!
//! The module also gathers the zero-knowledge opening's public pieces:
//! [`ZkCommitmentConfig`], the masked prefix's prover and verifier, and the
//! residual hand-off.

mod compression;
mod residual;
mod rounds;
mod rows;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use akita_algebra::EqPolynomial;
use akita_config::{derive_digit_range_grinding_plan, CommitmentConfig, SetupRequirements};
use akita_cpu_backend::standalone::DigitRangeProver;
use akita_cpu_backend::{AkitaProverSetup, CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_params::{
    BasisMode, DigitRangePlan, FlatBooleanDomain, OpeningClaimsLayout, PolynomialGroupLayout,
    RelationWitnessGeometry, ScheduleLookupKey,
};
use akita_prover::backend::ResidualExportKernel;
use akita_prover::{prove_digit_range, SelectedProverOpeningData};
use akita_sumcheck::FusedRounds;
use akita_types::{
    AkitaExpandedSetup, AkitaVerifierSetup, Commitment, CommittedGroup, DigitRangeEqualityPoint,
    GroupBatchStatement, NegativeBinarySupport, OpeningClaims, PolynomialGroupClaims,
    ProverGrinding, RingVec, VerifierGrinding,
};
use akita_verifier::{AkitaVerifier, Stage1Verifier};
use jolt_algebra::solinas::parallel::*;
use jolt_algebra::{One, Prime128OffsetA7F7};
use jolt_sumcheck::{prove_single, verify_single, SingleFormat, SingleInstance};
use jolt_transcript::{Channel, ProverTranscript, SiteId, Sponge, VerifierTranscript};
use jolt_zk::masking::ZkError;
use jolt_zk::whiteout::lift_ct;
use jolt_zk::whiteout::{
    CompressedImage, ImageCompression, ResponseLayout, ResponseProver, ResponseStatement,
    ResponseWitness, WhiteoutError,
};
use zeroize::Zeroizing;

use crate::AkitaCommitmentScheme;
pub use akita_config::ZkCommitmentConfig;
pub use akita_prover::{prove_masked_prefix, ResidualOpening};
pub use akita_verifier::{MaskedPrefix, MaskedPrefixRows, ResidualClaim};
use compression::Chains;
pub use residual::ResidualShape;
use rounds::{
    ClearProverRounds, ClearVerifierRounds, FusedE, FusedKernel, ProductKernel, SecretTable, SplitB,
};
use rows::{
    batch_rows, batch_target, batch_weight_at, ceil_log2, row_variables, zk_error, PaddingTerm,
    ProductClaims, ResponseShape, RowWeights,
};

/// The response statement's field.
type F = Prime128OffsetA7F7;

/// The `zk` feature makes the CPU backend a masked-prefix backend: it exports
/// the residual opening at the cut (`ResidualExportKernel`); `zk-prover` adds
/// the OS-drawn hiding randomness. Fails to compile if `zk` stops enabling
/// `akita-cpu-backend/zk`, which workspace dev-dependencies would hide.
const _: fn() = || {
    fn exports_the_residual<B: ResidualExportKernel<F, F>>() {}
    exports_the_residual::<CpuBackend<F, F>>();
};

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

/// Whether `setup`'s shared matrix holds every compression chain's maps.
fn covers_chains<Cfg: CommitmentConfig>(setup: &AkitaExpandedSetup<F>) -> Result<(), AkitaError> {
    let needed = compression::setup_len(Cfg::sis_modulus_profile())?;
    let held = setup.shared_matrix().num_field_elements();
    if held < needed {
        return Err(AkitaError::InvalidSetup(format!(
            "the compression chains read {needed} setup field elements, the setup holds {held}"
        )));
    }
    Ok(())
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
    /// Digits committed past `e`'s end, where the padding term asks for
    /// zeros; empty for an honest prover.
    padding: &'a [i8],
}

impl<'a> ProverDigits<'a> {
    fn honest(digits: &'a [i8]) -> Self {
        Self {
            committed: digits,
            rows: digits,
            range: digits,
            padding: &[],
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
    /// A prover setup for every `e` of up to `2^max_num_vars` entries: the
    /// response schedules' matrix, extended to the compression chains' maps.
    pub fn setup(
        scheme: &AkitaCommitmentScheme<Cfg>,
        max_num_vars: usize,
    ) -> Result<AkitaProverSetup<F>, AkitaError> {
        let requirements =
            SetupRequirements::from_catalog::<Cfg>(scheme.schedules(), max_num_vars, 1)?
                .covering(compression::setup_len(Cfg::sis_modulus_profile())?);
        akita_setup::new_prover_setup::<F>(&requirements)
    }

    /// A prover and verifier over `scheme`, whose catalog holds the response
    /// schedule of every `e` it proves, with `setup` from [`Self::setup`].
    pub fn new(
        scheme: AkitaCommitmentScheme<Cfg>,
        setup: AkitaProverSetup<F>,
    ) -> Result<Self, AkitaError> {
        let verifier = scheme.verifier(scheme.setup_verifier(&setup)?)?;
        covers_chains::<Cfg>(verifier.setup().expanded())?;
        let backend = CpuBackend::new(setup.expanded.clone())?;
        Ok(Self {
            scheme,
            verifier,
            prover: Some(ProverSetup { setup, backend }),
        })
    }

    /// The variable count of the committed table for a response vector of
    /// `layout`'s length: the `max_num_vars` its [`Self::setup`] needs and
    /// the key of its response schedule.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidInput`] for an empty layout.
    pub fn num_vars(layout: &ResponseLayout) -> Result<usize, AkitaError> {
        Ok(ResponseShape::new(layout.len())?.num_vars)
    }

    /// A verifier only.
    pub fn verifier(
        scheme: AkitaCommitmentScheme<Cfg>,
        setup: AkitaVerifierSetup<F>,
    ) -> Result<Self, AkitaError> {
        let verifier = scheme.verifier(setup)?;
        covers_chains::<Cfg>(verifier.setup().expanded())?;
        Ok(Self {
            verifier,
            scheme,
            prover: None,
        })
    }

    /// The setup whose shared matrix holds the compression maps.
    fn expanded(&self) -> &AkitaExpandedSetup<F> {
        self.verifier.setup().expanded()
    }

    /// The compression chains of `statement`'s images.
    fn chains<'s, const D: usize>(
        &'s self,
        statement: &'s ResponseStatement<'_, F, D>,
    ) -> Result<Chains<'s>, ZkError> {
        Chains::new(Cfg::sis_modulus_profile(), self.expanded(), statement)
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
        let shape = ResponseShape::new(statement.layout()?.len()).map_err(zk_error)?;
        let plan = range_plan::<Cfg, D>(statement).map_err(zk_error)?;
        if [digits.committed, digits.rows, digits.range]
            .iter()
            .any(|digits| digits.len() != shape.len)
        {
            return Err(WhiteoutError::ResponseRejected("response length").into());
        }
        // `e`'s digits and every table derived from them are wiped when
        // freed (`Zeroizing`, `SecretTable`), and enter `F` through `lift_ct`.
        let mut committed = Zeroizing::new(Vec::with_capacity(shape.padded_len()));
        committed.extend_from_slice(digits.committed);
        committed.extend_from_slice(digits.padding);
        if committed.len() > shape.padded_len() {
            return Err(WhiteoutError::ResponseRejected("response length").into());
        }
        committed.resize(shape.padded_len(), 0);

        // 1. Commit to e.
        let e = SecretTable::padded(
            cfg_iter!(committed)
                .map(|&digit| lift_ct::<F>(i64::from(digit)))
                .collect(),
            0,
        );
        let poly = DensePoly::from_field_evals(shape.num_vars, &e[..]).map_err(zk_error)?;
        drop(e);
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
            u: SecretTable::padded(rows.product_x, 1 << product_vars),
            w: SecretTable::padded(rows.product_y, 1 << product_vars),
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

        // 3. The row batch at η, and the padding term at τ_pad.
        let chains = self.chains(statement)?;
        let eta: Vec<F> = challenges(
            transcript,
            row_variables(statement, &chains).map_err(zk_error)?,
        );
        let weights = RowWeights::new(statement, &eta, &product_point, &chains)?;
        let padding = padding_term(transcript, shape);
        let batch = batch_rows(statement, shape, alpha, &weights, &padding, claims, &chains)?;

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

        // 5. The fused range-and-relation sumcheck, with the binary
        // constraint on the compression digits.
        let gamma: F = transcript.challenge();
        let zeta: F = transcript.challenge();
        let support = chains.support(shape.padded_len()).map_err(zk_error)?;
        let mut fused = FusedKernel {
            a: batch.a,
            b: SplitB::new(&range.point, gamma, zeta, support.intervals()),
            e: FusedE::Digits(committed),
        };
        let claim = batch.b + gamma * range.range_image;
        let point = prove_single::<F, F, AkitaError, _>(
            &mut FusedRounds(&mut fused),
            &compressed(claim, shape.num_vars, 3),
            &mut ClearProverRounds(transcript),
        )
        .map_err(zk_error)?
        .challenges;
        let v = fused
            .e
            .first()
            .ok_or_else(|| zk_error(AkitaError::Internal("a bound table is empty".into())))?;
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

    #[tracing::instrument(skip_all, name = "AkitaResponseProver::verify")]
    fn verify<H: Sponge, const D: usize>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        let shape = ResponseShape::new(statement.layout()?.len()).map_err(zk_error)?;
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
        let product_span = tracing::info_span!("response_product_sumcheck").entered();
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
        drop(product_span);

        // 3. The row batch at η, and the padding term at τ_pad.
        let rows_span = tracing::info_span!("response_row_batch").entered();
        let chains = self.chains(statement)?;
        let eta: Vec<F> = challenges(
            transcript,
            row_variables(statement, &chains).map_err(zk_error)?,
        );
        let weights = RowWeights::new(statement, &eta, &product.challenges, &chains)?;
        let padding = padding_term(transcript, shape);
        let b = batch_target(statement, alpha, &weights, claims, &chains)?;
        drop(rows_span);

        // 4. The digit range of e.
        let range_span = tracing::info_span!("response_digit_range").entered();
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
        .verify_clear::<F, _>(&mut grinding, None, RANGE_LEVEL)
        .map_err(zk_error)?;
        grinding.finish().map_err(zk_error)?;
        drop(range_span);

        // 5. The fused range-and-relation sumcheck, with the binary
        // constraint on the compression digits.
        let fused_span = tracing::info_span!("response_fused_sumcheck").entered();
        let gamma: F = transcript.challenge();
        let zeta: F = transcript.challenge();
        let fused = verify_single::<F, F, AkitaError, _>(
            &compressed(b + gamma * range.range_image_evaluation, shape.num_vars, 3),
            &mut ClearVerifierRounds(transcript),
        )
        .map_err(zk_error)?;
        let v: F = transcript.receive()?;
        drop(fused_span);
        let weight_span = tracing::info_span!("response_weight_at").entered();
        let a = batch_weight_at(
            statement,
            shape,
            alpha,
            &weights,
            &padding,
            &chains,
            &fused.challenges,
        )?;
        let b = range_weight_at(
            gamma,
            zeta,
            &chains.support(shape.padded_len()).map_err(zk_error)?,
            &range.point,
            &fused.challenges,
        )
        .map_err(zk_error)?;
        if a * v + b * v * (v + F::one()) != fused.output_claim {
            return Err(WhiteoutError::ResponseRejected("fused sumcheck").into());
        }
        drop(weight_span);

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
    fn compression(&self) -> &dyn ImageCompression<F> {
        self
    }

    fn prove_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        statement.check_witness(witness)?;
        transcript.site(RESPONSE_SITE);
        self.prove(statement, ProverDigits::honest(&witness.digits), transcript)
    }

    fn verify_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        transcript.site(RESPONSE_SITE);
        self.verify(statement, transcript)
    }
}

/// The diagnostic site the response proof opens with, so an event log can
/// tell its messages from the Σ-protocol's (sites never reach the sponge).
pub const RESPONSE_SITE: SiteId = SiteId::label("whiteout/response");

/// [`AkitaResponseProver`] that proves without checking the statement, so a
/// test reaches the verifier with a false one (the consistent-cheat tests,
/// `specs/whiteout-akita.md` §5.5 HT4): the unit tests' `Cheating` prover
/// with `CheatDigits::everywhere`. Under `test-support` the sumcheck rounds
/// of a false statement are shifted to stay consistent, so the proof it
/// sends is a forgery only the verifier's checks can reject. Test support
/// only.
#[cfg(feature = "test-support")]
pub struct UncheckedResponseProver<'a, Cfg: CommitmentConfig>(pub &'a AkitaResponseProver<Cfg>);

#[cfg(feature = "test-support")]
impl<Cfg, const D: usize> ResponseProver<F, D> for UncheckedResponseProver<'_, Cfg>
where
    Cfg: CommitmentConfig<Field = F, ExtField = F> + 'static,
{
    fn compression(&self) -> &dyn ImageCompression<F> {
        self.0
    }

    fn prove_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError> {
        transcript.site(RESPONSE_SITE);
        self.0
            .prove(statement, ProverDigits::honest(&witness.digits), transcript)
    }

    fn verify_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError> {
        <AkitaResponseProver<Cfg> as ResponseProver<F, D>>::verify_response(
            self.0, statement, transcript,
        )
    }
}

/// `B(r) = Σ_j (γ + ζ·1_{I_bin}(j))·eq(τ, j)·eq(r, j)`: the range image's
/// weight and the binary constraint on `I_bin`, at the range point `τ`.
fn range_weight_at(
    gamma: F,
    zeta: F,
    binary: &NegativeBinarySupport,
    range_point: &[F],
    point: &[F],
) -> Result<F, AkitaError> {
    Ok(gamma * EqPolynomial::mle(range_point, point)?
        + zeta * binary.evaluate_restricted_equality_at_point(range_point, point)?)
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

/// The padding term's `τ_pad`, drawn after `η`.
fn padding_term<C: Channel>(channel: &mut C, shape: ResponseShape) -> PaddingTerm<F> {
    PaddingTerm {
        point: challenges(channel, shape.num_vars),
    }
}

/// Akita's image compression without a prover backend: its lengths and
/// reconstruction, for sizing `e` before a prover exists. It cannot
/// compress.
pub struct CompressionShape<Cfg>(std::marker::PhantomData<Cfg>);

impl<Cfg> Default for CompressionShape<Cfg> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<Cfg: CommitmentConfig> ImageCompression<F> for CompressionShape<Cfg> {
    fn payload_len(&self, image_len: usize) -> Result<usize, WhiteoutError> {
        Ok(compression::chunks(Cfg::sis_modulus_profile(), image_len)
            .map_err(compression::whiteout_error)?
            .iter()
            .map(|chunk| chunk.plan.terminal_coefficients())
            .sum())
    }

    fn witness_len(&self, image_len: usize) -> Result<usize, WhiteoutError> {
        compression::chunks(Cfg::sis_modulus_profile(), image_len)
            .and_then(|chunks| chunks.iter().map(compression::Chunk::witness_len).sum())
            .map_err(compression::whiteout_error)
    }

    fn compress(&self, _image: &[F]) -> Result<CompressedImage<F>, WhiteoutError> {
        Err(WhiteoutError::InvalidRelation(
            "compression needs a prover backend",
        ))
    }

    fn reconstruct(&self, image_len: usize, witness: &[i8]) -> Result<Vec<F>, WhiteoutError> {
        compression::reconstruct(Cfg::sis_modulus_profile(), image_len, witness)
            .map_err(compression::whiteout_error)
    }

    fn reconstruct_transpose(
        &self,
        image_len: usize,
        weights: &[F],
    ) -> Result<Vec<F>, WhiteoutError> {
        compression::reconstruct_transpose(Cfg::sis_modulus_profile(), image_len, weights)
            .map_err(compression::whiteout_error)
    }

    fn check(
        &self,
        _image_len: usize,
        _payload: &[F],
        _witness: &[i8],
    ) -> Result<(), WhiteoutError> {
        Err(WhiteoutError::InvalidRelation(
            "checking a compression needs a prover backend",
        ))
    }
}

impl<Cfg> ImageCompression<F> for AkitaResponseProver<Cfg>
where
    Cfg: CommitmentConfig<Field = F, ExtField = F> + 'static,
{
    fn payload_len(&self, image_len: usize) -> Result<usize, WhiteoutError> {
        CompressionShape::<Cfg>::default().payload_len(image_len)
    }

    fn witness_len(&self, image_len: usize) -> Result<usize, WhiteoutError> {
        CompressionShape::<Cfg>::default().witness_len(image_len)
    }

    fn compress(&self, image: &[F]) -> Result<CompressedImage<F>, WhiteoutError> {
        let side = self.prover.as_ref().ok_or(WhiteoutError::InvalidRelation(
            "a verifier-only response prover cannot compress",
        ))?;
        compression::compress(&side.backend, Cfg::sis_modulus_profile(), image)
            .map_err(compression::whiteout_error)
    }

    fn reconstruct(&self, image_len: usize, witness: &[i8]) -> Result<Vec<F>, WhiteoutError> {
        CompressionShape::<Cfg>::default().reconstruct(image_len, witness)
    }

    fn reconstruct_transpose(
        &self,
        image_len: usize,
        weights: &[F],
    ) -> Result<Vec<F>, WhiteoutError> {
        CompressionShape::<Cfg>::default().reconstruct_transpose(image_len, weights)
    }

    /// The canonical witness only: Akita's chain is deterministic in the
    /// image, so `witness` opens `payload` exactly when compressing the
    /// image it reconstructs gives both back. The verifier enforces the
    /// chain equations and the binary digits instead (§5.7).
    fn check(&self, image_len: usize, payload: &[F], witness: &[i8]) -> Result<(), WhiteoutError> {
        let compressed = self.compress(&self.reconstruct(image_len, witness)?)?;
        if compressed.payload == payload && compressed.witness == witness {
            Ok(())
        } else {
            Err(WhiteoutError::ResponseRejected("compressed image"))
        }
    }
}
