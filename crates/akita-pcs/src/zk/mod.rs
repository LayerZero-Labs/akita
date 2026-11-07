//! Whiteout's response proof on Akita (`specs/whiteout-akita.md` §5; Whiteout
//! §4.3, "Proving the response"). `AkitaResponseProver` (under `zk-prover`)
//! proves jolt-zk's response statement with an ordinary Akita commitment and
//! opening, and [`AkitaResponseVerifier`] checks it:
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
#[cfg(any(test, feature = "zk-prover"))]
mod prover;
mod residual;
#[cfg(any(test, feature = "zk-prover"))]
mod rounds;
mod rows;
#[cfg(test)]
mod tests;

use akita_algebra::EqPolynomial;
use akita_config::{derive_digit_range_grinding_plan, CommitmentConfig, TrustedScheduleCatalog};
use akita_cpu_backend::CpuBackend;
use akita_error::AkitaError;
use akita_params::{
    verifier_setup_matrix_capacity_for_schedule, BasisMode, DigitRangePlan, OpeningClaimsLayout,
    PolynomialGroupLayout, RelationWitnessGeometry, ScheduleLookupKey,
};
use akita_prover::backend::ResidualExportKernel;
use akita_sumcheck::{
    verify_sumcheck_rounds, SumcheckRole, SumcheckShape, SumcheckVerifierChannel,
};
use akita_transcript::{
    ext_challenge, ext_challenges, receive_field_group, ChallengeCodec, ProtocolSiteId,
    TranscriptSponge, VerifierChannel, SITE_FAMILY_RESPONSE,
};
use akita_types::{
    AkitaExpandedSetup, AkitaVerifierSetup, Commitment, CommittedGroup, DigitRangeEqualityPoint,
    GroupBatchStatement, NegativeBinarySupport, OpeningClaims, PendingEof, PolynomialGroupClaims,
    RingVec, VerifierGrinding,
};
use akita_verifier::{AkitaVerifier, SetupPolicy, Stage1Verifier};
use jolt_algebra::{One, Prime128OffsetA7F7};
use jolt_zk::masking::ZkError;
use jolt_zk::whiteout::{
    CompressedImage, ImageCompression, ResponseLayout, ResponseStatement, ResponseVerifier,
    WhiteoutError,
};

use crate::AkitaCommitmentScheme;
pub use akita_config::ZkCommitmentConfig;
pub use akita_prover::{prove_masked_prefix, ResidualOpening};
pub use akita_verifier::{MaskedPrefix, MaskedPrefixRows, ResidualClaim};
#[cfg(feature = "test-support")]
pub use compression::BackendCompression;
use compression::Chains;
#[cfg(any(test, feature = "zk-prover"))]
pub use prover::AkitaResponseProver;
#[cfg(test)]
use prover::ProverDigits;
#[cfg(feature = "test-support")]
pub use prover::UncheckedResponseProver;
pub use residual::{ResidualMatrices, ResidualShape};
use rows::{
    batch_target, batch_weight_at, ceil_log2, row_variables, zk_error, PaddingTerm, ProductClaims,
    ResponseShape, RowWeights,
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

/// The site of a response-proof message: `step` of the proof, and `round`
/// and `role` within a sumcheck. Sites name records for transcript
/// diagnostics; they never reach the sponge.
fn response_site(step: u32, round: u32, role: u32) -> ProtocolSiteId {
    ProtocolSiteId {
        family: SITE_FAMILY_RESPONSE,
        stage: step,
        round,
        detail: role,
        ..ProtocolSiteId::default()
    }
}

/// Clear sumcheck rounds of response-proof `step` on the caller's verifier
/// channel, without grinding.
struct VerifierRounds<'a, 'proof> {
    state: &'a mut VerifierChannel<'proof>,
    step: u32,
}

impl<'proof, E: ChallengeCodec> SumcheckVerifierChannel<'proof, E> for VerifierRounds<'_, 'proof> {
    fn state_mut(&mut self) -> &mut VerifierChannel<'proof> {
        self.state
    }

    fn sumcheck_site(&self, _invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
        response_site(self.step, round, role as u32)
    }

    fn round_challenge(&mut self, _invocation: u32, round: u32) -> Result<E, AkitaError> {
        ext_challenge(
            self.state,
            response_site(self.step, round, SumcheckRole::Challenge as u32),
        )
    }
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

/// Verifies jolt-zk's Whiteout response proof with Akita.
pub struct AkitaResponseVerifier<Cfg: CommitmentConfig> {
    scheme: AkitaCommitmentScheme<Cfg>,
    verifier: AkitaVerifier<Cfg>,
    compression: CompressionShape<Cfg>,
}

impl<Cfg> AkitaResponseVerifier<Cfg>
where
    Cfg: CommitmentConfig<Field = F, ExtField = F> + 'static,
{
    /// A verifier over `scheme`, whose catalog holds the response schedule
    /// of every `e` it checks. `setup` may be a view of a wider shared
    /// matrix (`AkitaVerifierSetup::with_capacity`) holding at least
    /// [`Self::setup_field_elements`].
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidSetup`] if `setup`'s matrix does not hold the
    /// compression chains' maps.
    pub fn new(
        scheme: AkitaCommitmentScheme<Cfg>,
        setup: AkitaVerifierSetup<F>,
    ) -> Result<Self, AkitaError> {
        let verifier = scheme.verifier(setup)?;
        covers_chains::<Cfg>(verifier.setup().expanded())?;
        Ok(Self {
            scheme,
            verifier,
            compression: CompressionShape::default(),
        })
    }

    /// The public-matrix length a verifier over `catalog` of every `e` of up
    /// to `2^max_num_vars` entries reads: each admitted response schedule's
    /// verifier matrix, and the compression chains' maps.
    ///
    /// # Errors
    ///
    /// [`AkitaError`] if the catalog admits no row at that capacity, or a
    /// row's size overflows.
    pub fn setup_field_elements(
        catalog: &TrustedScheduleCatalog<Cfg>,
        max_num_vars: usize,
    ) -> Result<usize, AkitaError> {
        let rows = catalog.rows_within_setup_capacity(max_num_vars, 1)?;
        if rows.is_empty() {
            return Err(AkitaError::InvalidSetup(format!(
                "the response catalog has no row of up to {max_num_vars} variables"
            )));
        }
        let mut needed = compression::setup_len(Cfg::sis_modulus_profile())?;
        for row in rows {
            let layout = row.profiles().opening_layout()?;
            needed = needed.max(
                verifier_setup_matrix_capacity_for_schedule(row.schedule(), &layout)?
                    .num_field_elements,
            );
        }
        Ok(needed)
    }

    /// The variable count of the committed table for a response vector of
    /// `layout`'s length: the `max_num_vars` a setup needs and the key of
    /// its response schedule.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidInput`] for an empty layout.
    pub fn num_vars(layout: &ResponseLayout) -> Result<usize, AkitaError> {
        Ok(ResponseShape::new(layout.len())?.num_vars)
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

    #[tracing::instrument(skip_all, name = "AkitaResponseVerifier::verify")]
    fn verify<const D: usize>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        state: &mut VerifierChannel<'_>,
    ) -> Result<(), ZkError> {
        // A rejected proof ends the transcript, as in the fold verifiers.
        let result = (|| -> Result<(), ZkError> {
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
            let group = PolynomialGroupClaims::new(fused.challenges, vec![v], &committed)
                .map_err(zk_error)?;
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
        })();
        if result.is_err() {
            let _ = akita_transcript::reject(state);
        }
        result
    }
}

impl<Cfg, const D: usize> ResponseVerifier<F, D> for AkitaResponseVerifier<Cfg>
where
    Cfg: CommitmentConfig<Field = F, ExtField = F> + 'static,
{
    type Sponge = TranscriptSponge;

    fn compression(&self) -> &dyn ImageCompression<F> {
        &self.compression
    }

    fn verify_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierChannel<'_>,
    ) -> Result<(), ZkError> {
        self.verify(statement, transcript)
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
