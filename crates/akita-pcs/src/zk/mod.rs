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
use akita_cpu_backend::custom_source::SignedBlockPoly;
use akita_cpu_backend::standalone::DigitRangeProver;
use akita_cpu_backend::{AkitaProverSetup, CpuBackend, GroupContext};
use akita_error::AkitaError;
use akita_params::{
    BasisMode, DigitRangePlan, FlatBooleanDomain, OpeningClaimsLayout, PolynomialGroupLayout,
    RelationWitnessGeometry, ScheduleLookupKey,
};
use akita_prover::backend::ResidualExportKernel;
use akita_prover::{prove_digit_range, SelectedProverOpeningData};
use akita_sumcheck::{prove_sumcheck, verify_sumcheck_rounds, SumcheckShape};
use akita_transcript::{
    ext_challenge, ext_challenges, receive_field_group, send_field_group, ProverChannel,
    TranscriptSponge, VerifierChannel,
};
use akita_types::{
    AkitaExpandedSetup, AkitaVerifierSetup, Commitment, CommittedGroup, DigitRangeEqualityPoint,
    GroupBatchStatement, NegativeBinarySupport, OpeningClaims, PendingEof, PolynomialGroupClaims,
    ProverGrinding, RingVec, VerifierGrinding,
};
use akita_verifier::{AkitaVerifier, SetupPolicy, Stage1Verifier};
use jolt_algebra::{One, Prime128OffsetA7F7};
use jolt_zk::masking::ZkError;
use jolt_zk::whiteout::{
    CompressedImage, ImageCompression, ResponseLayout, ResponseProver, ResponseStatement,
    ResponseWitness, WhiteoutError,
};
use zeroize::Zeroizing;

use crate::AkitaCommitmentScheme;
pub use akita_config::ZkCommitmentConfig;
pub use akita_prover::{prove_masked_prefix, ResidualOpening};
pub use akita_verifier::{MaskedPrefix, MaskedPrefixRows, ResidualClaim};
#[cfg(feature = "test-support")]
pub use compression::BackendCompression;
#[cfg(not(feature = "test-support"))]
use compression::BackendCompression;
use compression::Chains;
pub use residual::{ResidualMatrices, ResidualShape};
use rounds::{
    response_site, FusedKernel, ProductKernel, ProverRounds, SecretTable, SplitB, VerifierRounds,
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

/// The response proof's steps, in transcript order: the `step` of each
/// message's [`response_site`].
mod step {
    pub(super) const COMMITMENT: u32 = 0;
    pub(super) const ALPHA: u32 = 1;
    pub(super) const PRODUCT_TARGET: u32 = 2;
    pub(super) const PRODUCT_SUMCHECK: u32 = 3;
    pub(super) const PRODUCT_CLAIMS: u32 = 4;
    pub(super) const ETA: u32 = 5;
    pub(super) const PADDING_POINT: u32 = 6;
    pub(super) const RANGE_POINT: u32 = 7;
    pub(super) const GAMMA: u32 = 8;
    pub(super) const ZETA: u32 = 9;
    pub(super) const FUSED_SUMCHECK: u32 = 10;
    pub(super) const OPENING_VALUE: u32 = 11;
}

/// The site of `step`'s one message.
fn site(step: u32) -> akita_transcript::ProtocolSiteId {
    response_site(step, 0, 0)
}

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
    fn prove<const D: usize>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        digits: ProverDigits<'_>,
        state: &mut ProverChannel,
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
        // freed (`Zeroizing`, `SecretTable`, `SignedBlockPoly`), and enter
        // `F` through `lift_ct`.
        let mut committed = Zeroizing::new(Vec::with_capacity(shape.padded_len()));
        committed.extend_from_slice(digits.committed);
        committed.extend_from_slice(digits.padding);
        if committed.len() > shape.padded_len() {
            return Err(WhiteoutError::ResponseRejected("response length").into());
        }
        // Every digit from `live` on is the padding's zero.
        let live = committed.len();
        committed.resize(shape.padded_len(), 0);

        // 1. Commit to e. The backend keeps the digits as `i8` until the
        // opening, read in constant time (`SignedBlockPoly`).
        let poly = SignedBlockPoly::i8(committed.to_vec(), shape.num_vars).map_err(zk_error)?;
        let source = side.backend.import_source(vec![poly]).map_err(zk_error)?;
        let output = side
            .backend
            .commit(
                self.scheme.schedules(),
                &source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .map_err(zk_error)?;
        send_field_group(
            state,
            site(step::COMMITMENT),
            output.committed_group.rows().coeffs(),
        )
        .map_err(zk_error)?;

        // 2. The product reduction at α.
        let alpha: F = ext_challenge(state, site(step::ALPHA)).map_err(zk_error)?;
        let rows = statement.rows(digits.rows, alpha)?;
        let product_vars = ceil_log2(rows.product_x.len().max(1));
        // t_× is the product lift's right side at α (Whiteout (4)); for a
        // true statement it equals Σ_i U_i W_i.
        let target = rows.product_target;
        let mut product = ProductKernel {
            u: SecretTable::padded(rows.product_x, 1 << product_vars),
            w: SecretTable::padded(rows.product_y, 1 << product_vars),
            target,
        };
        send_field_group(state, site(step::PRODUCT_TARGET), &[target]).map_err(zk_error)?;
        let (product_point, _) = prove_sumcheck::<F, F, _, _>(
            &mut product,
            &mut ProverRounds {
                state,
                step: step::PRODUCT_SUMCHECK,
            },
            SumcheckShape::new(product_vars, 2).map_err(zk_error)?,
            0,
        )
        .map_err(zk_error)?;
        let claims = ProductClaims {
            target,
            x: first(&product.u)?,
            y: first(&product.w)?,
        };
        send_field_group(state, site(step::PRODUCT_CLAIMS), &[claims.x, claims.y])
            .map_err(zk_error)?;

        // 3. The row batch at η, and the padding term at τ_pad.
        let chains = self.chains(statement)?;
        let eta: Vec<F> = ext_challenges(
            state,
            site(step::ETA),
            row_variables(statement, &chains).map_err(zk_error)?,
        )
        .map_err(zk_error)?;
        let weights = RowWeights::new(statement, &eta, &product_point, &chains)?;
        let padding = padding_term(state, shape)?;
        let batch = batch_rows(statement, shape, alpha, &weights, &padding, claims, &chains)?;

        // 4. The digit range of e.
        let tau: Vec<F> =
            ext_challenges(state, site(step::RANGE_POINT), shape.num_vars).map_err(zk_error)?;
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
        let mut grinding = ProverGrinding::new(state, &grinding_plan);
        let range = prove_digit_range::<F, F, _>(
            &mut session,
            plan,
            equality_point.into_coordinates(),
            &mut grinding,
            RANGE_LEVEL,
        )
        .map_err(zk_error)?;
        grinding.finish().map_err(zk_error)?;
        drop(session);

        // 5. The fused range-and-relation sumcheck, with the binary
        // constraint on the compression digits.
        let gamma: F = ext_challenge(state, site(step::GAMMA)).map_err(zk_error)?;
        let zeta: F = ext_challenge(state, site(step::ZETA)).map_err(zk_error)?;
        let support = chains.support(shape.padded_len()).map_err(zk_error)?;
        let claim = batch.b + gamma * range.range_image;
        let mut fused = FusedKernel::new(
            batch.a,
            SplitB::new(&range.point, gamma, zeta, support.intervals()),
            committed,
            live,
            claim,
        )
        .map_err(zk_error)?;
        let (point, _) = prove_sumcheck::<F, F, _, _>(
            &mut fused,
            &mut ProverRounds {
                state,
                step: step::FUSED_SUMCHECK,
            },
            SumcheckShape::new(shape.num_vars, 3).map_err(zk_error)?,
            0,
        )
        .map_err(zk_error)?;
        let v = fused
            .e_first()
            .ok_or_else(|| zk_error(AkitaError::Internal("a bound table is empty".into())))?;
        send_field_group(state, site(step::OPENING_VALUE), &[v]).map_err(zk_error)?;
        drop(fused);

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
            .batched_prove_embedded(
                &side.setup,
                opening,
                &side.backend,
                state,
                BasisMode::Lagrange,
            )
            .map_err(zk_error)
    }

    #[tracing::instrument(skip_all, name = "AkitaResponseProver::verify")]
    fn verify<const D: usize>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        state: &mut VerifierChannel<'_>,
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
        let coefficients: Vec<F> = receive_field_group(
            state,
            site(step::COMMITMENT),
            compression.terminal_coefficients(),
        )
        .map_err(zk_error)?;
        let committed = CommittedGroup::new(
            row.profiles().final_group,
            Commitment::new(
                RingVec::from_coeffs_with_ring_dim(coefficients, ring_dim).map_err(zk_error)?,
            ),
        )
        .map_err(|_| zk_error(AkitaError::InvalidProof))?;

        // 2. The product reduction at α.
        let product_span = tracing::info_span!("response_product_sumcheck").entered();
        let alpha: F = ext_challenge(state, site(step::ALPHA)).map_err(zk_error)?;
        let product_vars = ceil_log2(statement.row_shape().products.max(1));
        let [target] = receive_array(state, step::PRODUCT_TARGET)?;
        let product = verify_sumcheck_rounds::<F, F, _>(
            &mut VerifierRounds {
                state,
                step: step::PRODUCT_SUMCHECK,
            },
            0,
            target,
            SumcheckShape::new(product_vars, 2).map_err(zk_error)?,
        )
        .map_err(zk_error)?;
        let [x, y] = receive_array(state, step::PRODUCT_CLAIMS)?;
        let claims = ProductClaims { target, x, y };
        if claims.x * claims.y != product.output_claim {
            return Err(WhiteoutError::ResponseRejected("product sumcheck").into());
        }
        drop(product_span);

        // 3. The row batch at η, and the padding term at τ_pad.
        let rows_span = tracing::info_span!("response_row_batch").entered();
        let chains = self.chains(statement)?;
        let eta: Vec<F> = ext_challenges(
            state,
            site(step::ETA),
            row_variables(statement, &chains).map_err(zk_error)?,
        )
        .map_err(zk_error)?;
        let weights = RowWeights::new(statement, &eta, &product.challenges, &chains)?;
        let padding = padding_term(state, shape)?;
        let b = batch_target(statement, alpha, &weights, claims, &chains)?;
        drop(rows_span);

        // 4. The digit range of e.
        let range_span = tracing::info_span!("response_digit_range").entered();
        let tau: Vec<F> =
            ext_challenges(state, site(step::RANGE_POINT), shape.num_vars).map_err(zk_error)?;
        let plan = range_plan::<Cfg, D>(statement).map_err(zk_error)?;
        let grinding_plan =
            derive_digit_range_grinding_plan::<Cfg>(plan, shape.num_vars, RANGE_LEVEL)
                .map_err(zk_error)?;
        let mut grinding = VerifierGrinding::new(state, &grinding_plan);
        let range = Stage1Verifier::new(
            DigitRangeEqualityPoint::from_column_then_ring_challenges(&tau, shape.num_vars, 0)
                .map_err(zk_error)?,
            plan,
        )
        .verify_clear::<F>(&mut grinding, None, RANGE_LEVEL)
        .map_err(zk_error)?;
        // The response's later messages follow on this state; the Whiteout
        // verifier checks the end of the argument string after its tail.
        let _: PendingEof = grinding.finish().map_err(zk_error)?;
        drop(range_span);

        // 5. The fused range-and-relation sumcheck, with the binary
        // constraint on the compression digits.
        let fused_span = tracing::info_span!("response_fused_sumcheck").entered();
        let gamma: F = ext_challenge(state, site(step::GAMMA)).map_err(zk_error)?;
        let zeta: F = ext_challenge(state, site(step::ZETA)).map_err(zk_error)?;
        let fused = verify_sumcheck_rounds::<F, F, _>(
            &mut VerifierRounds {
                state,
                step: step::FUSED_SUMCHECK,
            },
            0,
            b + gamma * range.range_image_evaluation,
            SumcheckShape::new(shape.num_vars, 3).map_err(zk_error)?,
        )
        .map_err(zk_error)?;
        let [v] = receive_array(state, step::OPENING_VALUE)?;
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
        // The caller owns the transcript and its end-of-proof check.
        let _pending = self
            .verifier
            .batched_verify_embedded(
                state,
                opening,
                BasisMode::Lagrange,
                SetupPolicy::Check(None),
            )
            .map_err(zk_error)?;
        Ok(())
    }
}

impl<Cfg, const D: usize> ResponseProver<F, D> for AkitaResponseProver<Cfg>
where
    Cfg: CommitmentConfig<Field = F, ExtField = F> + 'static,
{
    type Sponge = TranscriptSponge;

    fn compression(&self) -> &dyn ImageCompression<F> {
        self
    }

    fn prove_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverChannel,
    ) -> Result<(), ZkError> {
        statement.check_witness(witness)?;
        self.prove(statement, ProverDigits::honest(&witness.digits), transcript)
    }

    fn verify_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierChannel<'_>,
    ) -> Result<(), ZkError> {
        self.verify(statement, transcript)
    }
}

/// [`AkitaResponseProver`] that proves without checking the statement, so a
/// test reaches the verifier with a false one (the consistent-cheat tests,
/// `specs/whiteout-akita.md` §5.5 HT4): the unit tests' `Cheating` prover
/// with `CheatDigits::everywhere`. Under `test-support` the sumcheck rounds
/// of a false statement are shifted to stay consistent, so the proof it
/// sends is a forgery only the verifier's checks can reject. The second
/// field is the compression the Σ prover builds `e`'s compression witnesses
/// with, so a test can forge a chain (the prover's own for an honest one).
/// Test support only.
#[cfg(feature = "test-support")]
pub struct UncheckedResponseProver<'a, Cfg: CommitmentConfig>(
    pub &'a AkitaResponseProver<Cfg>,
    pub &'a dyn ImageCompression<F>,
);

#[cfg(feature = "test-support")]
impl<Cfg, const D: usize> ResponseProver<F, D> for UncheckedResponseProver<'_, Cfg>
where
    Cfg: CommitmentConfig<Field = F, ExtField = F> + 'static,
{
    type Sponge = TranscriptSponge;

    fn compression(&self) -> &dyn ImageCompression<F> {
        self.1
    }

    fn prove_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverChannel,
    ) -> Result<(), ZkError> {
        self.0
            .prove(statement, ProverDigits::honest(&witness.digits), transcript)
    }

    fn verify_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierChannel<'_>,
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

/// Receives `step`'s message of `N` field elements.
fn receive_array<const N: usize>(
    state: &mut VerifierChannel<'_>,
    step: u32,
) -> Result<[F; N], ZkError> {
    receive_field_group(state, site(step), N)
        .map_err(zk_error)?
        .try_into()
        .map_err(|_| zk_error(AkitaError::InvalidProof))
}

/// The value a table bound in every variable holds.
fn first(table: &[F]) -> Result<F, ZkError> {
    table
        .first()
        .copied()
        .ok_or_else(|| zk_error(AkitaError::Internal("a bound table is empty".into())))
}

/// The padding term's `τ_pad`, drawn after `η`.
fn padding_term<S: akita_transcript::ProofChannel>(
    state: &mut S,
    shape: ResponseShape,
) -> Result<PaddingTerm<F>, ZkError> {
    Ok(PaddingTerm {
        point: ext_challenges(state, site(step::PADDING_POINT), shape.num_vars)
            .map_err(zk_error)?,
    })
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

impl<Cfg> AkitaResponseProver<Cfg>
where
    Cfg: CommitmentConfig<Field = F, ExtField = F> + 'static,
{
    /// The compression on this prover's backend.
    fn backend_compression(&self) -> Result<BackendCompression<'_, Cfg>, WhiteoutError> {
        let side = self.prover.as_ref().ok_or(WhiteoutError::InvalidRelation(
            "a verifier-only response prover cannot compress",
        ))?;
        Ok(BackendCompression::new(&side.backend))
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
        self.backend_compression()?.compress(image)
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

    fn check(&self, image_len: usize, payload: &[F], witness: &[i8]) -> Result<(), WhiteoutError> {
        self.backend_compression()?
            .check(image_len, payload, witness)
    }
}
