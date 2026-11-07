//! Whiteout's response prover on Akita: the verifier's
//! [`AkitaResponseVerifier`] with a prover setup and backend.

use akita_config::SetupRequirements;
use akita_cpu_backend::custom_source::SignedBlockPoly;
use akita_cpu_backend::standalone::DigitRangeProver;
use akita_cpu_backend::{AkitaProverSetup, GroupContext};
use akita_params::FlatBooleanDomain;
use akita_prover::{prove_digit_range, SelectedProverOpeningData};
use akita_sumcheck::prove_sumcheck;
use akita_transcript::{send_field_group, ProverChannel};
use akita_types::ProverGrinding;
use jolt_zk::whiteout::{ResponseProver, ResponseWitness};
use zeroize::Zeroizing;

#[cfg(not(feature = "test-support"))]
use super::compression::BackendCompression;
use super::rounds::{FusedKernel, ProductKernel, ProverRounds, SecretTable, SplitB};
use super::rows::batch_rows;
use super::*;

/// Proves jolt-zk's Whiteout response statement with Akita.
pub struct AkitaResponseProver<Cfg: CommitmentConfig> {
    verifier: AkitaResponseVerifier<Cfg>,
    setup: AkitaProverSetup<F>,
    backend: CpuBackend<F, F>,
}

/// The digits each step of the prover reads. An honest prover reads `e` in
/// all three; tests split them to reach the verifier with a false
/// statement.
#[derive(Clone, Copy)]
pub(super) struct ProverDigits<'a> {
    /// The committed table `E`.
    pub(super) committed: &'a [i8],
    /// What the product claims and their rows are computed from.
    pub(super) rows: &'a [i8],
    /// The digit-range session's table.
    pub(super) range: &'a [i8],
    /// Digits committed past `e`'s end, where the padding term asks for
    /// zeros; empty for an honest prover.
    pub(super) padding: &'a [i8],
}

impl<'a> ProverDigits<'a> {
    pub(super) fn honest(digits: &'a [i8]) -> Self {
        Self {
            committed: digits,
            rows: digits,
            range: digits,
            padding: &[],
        }
    }
}

impl<Cfg> AkitaResponseProver<Cfg>
where
    Cfg: CommitmentConfig<Field = F, ExtField = F> + 'static,
{
    /// A prover setup for every `e` of up to `2^max_num_vars` entries: the
    /// response schedules' matrix, extended to the compression chains' maps.
    ///
    /// # Errors
    ///
    /// [`AkitaError`] if the catalog cannot size the setup.
    pub fn setup(
        scheme: &AkitaCommitmentScheme<Cfg>,
        max_num_vars: usize,
    ) -> Result<AkitaProverSetup<F>, AkitaError> {
        let requirements =
            SetupRequirements::from_catalog::<Cfg>(scheme.schedules(), max_num_vars, 1)?
                .covering(compression::setup_len(Cfg::sis_modulus_profile())?);
        akita_setup::new_prover_setup::<F>(&requirements)
    }

    /// A prover over `scheme`, whose catalog holds the response schedule of
    /// every `e` it proves, with `setup` from [`Self::setup`].
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidSetup`] if `setup` does not hold the compression
    /// chains' maps.
    pub fn new(
        scheme: AkitaCommitmentScheme<Cfg>,
        setup: AkitaProverSetup<F>,
    ) -> Result<Self, AkitaError> {
        let verifier_setup = scheme.setup_verifier(&setup)?;
        let backend = CpuBackend::new(setup.expanded.clone())?;
        Ok(Self {
            verifier: AkitaResponseVerifier::new(scheme, verifier_setup)?,
            setup,
            backend,
        })
    }

    /// This prover's verifier.
    pub fn verifier(&self) -> &AkitaResponseVerifier<Cfg> {
        &self.verifier
    }

    /// Proves the statement on `digits`, without checking that it holds:
    /// [`ResponseProver::prove_response`] checks first.
    pub(super) fn prove<const D: usize>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        digits: ProverDigits<'_>,
        state: &mut ProverChannel,
    ) -> Result<(), ZkError> {
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
        committed.resize(shape.padded_len(), 0);

        // 1. Commit to e. The backend keeps the digits as `i8` until the
        // opening, read in constant time (`SignedBlockPoly`).
        let poly = SignedBlockPoly::i8(committed.to_vec(), shape.num_vars).map_err(zk_error)?;
        let source = self.backend.import_source(vec![poly]).map_err(zk_error)?;
        let output = self
            .backend
            .commit(
                self.verifier.scheme.schedules(),
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
        // t_× is the product lift's right self at α (Whiteout (4)); for a
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
        let chains = self.verifier.chains(statement)?;
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
            digits.range,
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
            self.verifier.scheme.schedules(),
        )
        .map_err(zk_error)?;
        self.verifier
            .scheme
            .batched_prove_embedded(
                &self.setup,
                opening,
                &self.backend,
                state,
                BasisMode::Lagrange,
            )
            .map_err(zk_error)
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
        self.verifier.verify(statement, transcript)
    }
}

/// [`AkitaResponseProver`] that proves without checking the statement, so a
/// test reaches the verifier with a false one (the consistent-cheat tests,
/// `specs/whiteout-akita.md` §9): the unit tests' `Cheating` prover
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
        self.0.verifier.verify(statement, transcript)
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
        BackendCompression::<Cfg>::new(&self.backend).compress(image)
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
        BackendCompression::<Cfg>::new(&self.backend).check(image_len, payload, witness)
    }
}

/// The value a table bound in every variable holds.
fn first(table: &[F]) -> Result<F, ZkError> {
    table
        .first()
        .copied()
        .ok_or_else(|| zk_error(AkitaError::Internal("a bound table is empty".into())))
}
