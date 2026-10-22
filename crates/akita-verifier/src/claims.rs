//! How a fold verifier holds the claims that depend on the witness.
//!
//! The clear verifier holds each as a field value and checks it on the spot.
//! The masked verifier of a zero-knowledge opening (`specs/whiteout-akita.md`
//! §4.5) holds each as an affine form over the pads, records the stage checks
//! on them, and lowers those checks into residual rows once the auxiliary
//! block has arrived. Both run the same fold verifier, written once over
//! [`ClaimReplay`], and both evaluate each check through the one formula in
//! `akita-types` (`check_range_leaf`, `check_stage2_oracle`).

use akita_error::AkitaError;
use akita_types::{
    check_range_leaf, check_stage2_oracle, FoldResponseReplay, GrindingReplay,
    GrindingSumcheckVerifier, RangeLeaf, Stage2OracleWeights, VerifierGrinding, FIELD_DIGEST_BYTES,
};
use jolt_algebra::{CanonicalDecode, CanonicalEncoding, ExtField, Field};
use jolt_sumcheck::{verify_single, SingleInstance, SumcheckError};
use jolt_transcript::{ClaimMismatch, ClearDomain, PrivateChannel, SiteId, Sponge};

/// A fold verifier's replay: the grinding replay it reads the proof through,
/// and how it holds and checks the claims that depend on the witness.
pub(crate) trait ClaimReplay<F, E>:
    FoldResponseReplay<State: PrivateChannel<Error: Into<SumcheckError<E>>>>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
{
    /// A witness-dependent claim: a value in the clear, an affine form over
    /// the pads when masked.
    type Claim: Clone + Send + Sync;

    /// The public constant `value`.
    fn constant(value: E) -> Self::Claim;

    /// `lhs + rhs`.
    fn add(lhs: &Self::Claim, rhs: &Self::Claim) -> Self::Claim;

    /// `value · scale` for a public `scale`.
    fn scale(value: &Self::Claim, scale: E) -> Self::Claim;

    /// A claim produced by a path only the clear verifier replays: Stage-1
    /// product stages, the physical ℓ2 norm, extension-field opening claims.
    ///
    /// # Errors
    ///
    /// A masked replay returns [`AkitaError::UnsupportedSchedule`]; admission
    /// (`ZkOpeningSchedule::admit`) already rejects these paths at masked
    /// levels.
    fn clear_path(value: E) -> Result<Self::Claim, AkitaError>;

    /// The values of claims a clear-only path consumes (extension-field
    /// opening claims).
    ///
    /// # Errors
    ///
    /// As [`clear_path`](Self::clear_path).
    fn clear_values(claims: &[Self::Claim]) -> Result<Vec<E>, AkitaError>;

    /// Receives the digest of a terminal `t` state's last block
    /// (`akita_types::receive_field_digest`), sent under `dev-protocol`.
    ///
    /// # Errors
    ///
    /// The transcript's error; a masked replay, which never reaches the
    /// terminal, returns [`AkitaError::UnsupportedSchedule`].
    fn receive_field_digest(
        &mut self,
        site: SiteId,
    ) -> Result<[u8; FIELD_DIGEST_BYTES], AkitaError>;

    /// Binds a fold level's openings at its `Openings` site
    /// (`akita_types::bind_fold_openings`).
    ///
    /// # Errors
    ///
    /// The channel's error.
    fn bind_openings(
        &mut self,
        level: u32,
        prefix: Option<&E>,
        openings: &[Self::Claim],
    ) -> Result<(), AkitaError>;

    /// Replays one single-instance sumcheck of `protocol` from `input` and
    /// returns its challenges and running output claim. The instance's own
    /// `input_claim` is ignored.
    ///
    /// # Errors
    ///
    /// The sumcheck's or the channel's error.
    fn single_sumcheck(
        &mut self,
        protocol: akita_params::SumcheckProtocol,
        level: u32,
        stage: u32,
        instance: SingleInstance<E>,
        input: Self::Claim,
    ) -> Result<(Vec<E>, Self::Claim), AkitaError>;

    /// Receives the Stage-1 range-image evaluation of `level`.
    ///
    /// # Errors
    ///
    /// The channel's error.
    fn range_image(&mut self, level: u32, stage: u32) -> Result<Self::Claim, AkitaError>;

    /// Receives the Stage-2 witness evaluation of `level`.
    ///
    /// # Errors
    ///
    /// The channel's error.
    fn witness_eval(&mut self, level: u32) -> Result<Self::Claim, AkitaError>;

    /// The check `lhs = rhs` between two claims the verifier derived from the
    /// same openings.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidProof`] if they differ.
    fn check_equal(&mut self, lhs: &Self::Claim, rhs: &Self::Claim) -> Result<(), AkitaError>;

    /// The Stage-1 leaf check, [`check_range_leaf`].
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidProof`] if the clear check fails.
    fn check_range_leaf(
        &mut self,
        leaf: RangeLeaf<E>,
        range_image: Self::Claim,
        claim: Self::Claim,
    ) -> Result<(), AkitaError>;

    /// The Stage-2 oracle check, [`check_stage2_oracle`].
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidProof`] if the clear check fails.
    fn check_stage2_oracle(
        &mut self,
        weights: Stage2OracleWeights<E>,
        witness_eval: Self::Claim,
        claim: Self::Claim,
    ) -> Result<(), AkitaError>;
}

/// The clear verifier: every claim is a value, checked when it is formed.
impl<'proof, F, E, H> ClaimReplay<F, E> for VerifierGrinding<'_, 'proof, H>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    H: Sponge,
{
    type Claim = E;

    fn constant(value: E) -> E {
        value
    }

    fn add(lhs: &E, rhs: &E) -> E {
        *lhs + *rhs
    }

    fn scale(value: &E, scale: E) -> E {
        *value * scale
    }

    fn clear_path(value: E) -> Result<E, AkitaError> {
        Ok(value)
    }

    fn clear_values(claims: &[E]) -> Result<Vec<E>, AkitaError> {
        Ok(claims.to_vec())
    }

    fn receive_field_digest(
        &mut self,
        site: SiteId,
    ) -> Result<[u8; FIELD_DIGEST_BYTES], AkitaError> {
        akita_types::receive_field_digest(self.state_mut(), site)
    }

    fn bind_openings(
        &mut self,
        level: u32,
        prefix: Option<&E>,
        openings: &[E],
    ) -> Result<(), AkitaError> {
        akita_types::bind_fold_openings::<F, E, _>(self, level, prefix, openings)
    }

    fn single_sumcheck(
        &mut self,
        protocol: akita_params::SumcheckProtocol,
        level: u32,
        stage: u32,
        instance: SingleInstance<E>,
        input: E,
    ) -> Result<(Vec<E>, E), AkitaError> {
        let mut channel = GrindingSumcheckVerifier::<F, E, _>::new(self, protocol, level, stage);
        let replay = verify_single::<F, E, AkitaError, _>(
            &SingleInstance {
                input_claim: input,
                ..instance
            },
            &mut channel,
        )?;
        Ok((replay.challenges, replay.output_claim))
    }

    fn range_image(&mut self, level: u32, stage: u32) -> Result<E, AkitaError> {
        Ok(akita_types::stage1_range_image::<F, E, _>(self, level, stage, E::zero())?.0)
    }

    fn witness_eval(&mut self, level: u32) -> Result<E, AkitaError> {
        Ok(akita_types::stage2_w_eval::<F, E, _>(self, level, E::zero())?.0)
    }

    fn check_equal(&mut self, lhs: &E, rhs: &E) -> Result<(), AkitaError> {
        if lhs == rhs {
            Ok(())
        } else {
            Err(AkitaError::InvalidProof)
        }
    }

    fn check_range_leaf(
        &mut self,
        leaf: RangeLeaf<E>,
        range_image: E,
        claim: E,
    ) -> Result<(), AkitaError> {
        check_range_leaf(&mut ClearDomain, &leaf, range_image, claim)
            .map_err(|ClaimMismatch| AkitaError::InvalidProof)
    }

    fn check_stage2_oracle(
        &mut self,
        weights: Stage2OracleWeights<E>,
        witness_eval: E,
        claim: E,
    ) -> Result<(), AkitaError> {
        check_stage2_oracle(&mut ClearDomain, &weights, witness_eval, claim)
            .map_err(|ClaimMismatch| AkitaError::InvalidProof)
    }
}
