//! How a fold verifier holds the claims that depend on the witness.
//!
//! The fold verifier is written once over [`ClaimReplay`]. The clear verifier,
//! [`VerifierGrinding`] on a [`VerifierChannel`], holds each claim as a field
//! value and checks it on the spot.
//!
//! Extension seam: zero knowledge (akita#120). A masked verifier holds each
//! witness-dependent claim as a form over its pads instead, and replays the
//! same fold verifier through its own [`ClaimReplay`].

use akita_error::AkitaError;
use akita_transcript::FIELD_DIGEST_BYTES;
use akita_transcript::{ChallengeCodec, PrivateProofChannel, ProtocolSiteId, VerifierChannel};
use akita_types::{
    check_range_leaf, check_stage2_oracle, FoldResponseReplay, GrindingReplay,
    GrindingSumcheckVerifier, RangeLeaf, Stage2OracleWeights, VerifierGrinding,
    VerifierGrindingChannel,
};
use jolt_algebra::{CanonicalEncoding, ExtField, Field};
use jolt_transcript::{ClaimMismatch, ClearDomain};

/// How a single-instance sumcheck's rounds go on the wire: the two drivers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SumcheckFormat<E> {
    /// `[c0, c2, …, c_degree]` per round, the linear coefficient omitted.
    Compressed,
    /// `[q1, …, q_degree]` of `s(X) = eq(τ_i, X) · q(X)` per round, with
    /// `τ_i = equality_point[i]` (low to high); the running claim is the
    /// normalized `q(r_i)`.
    EqFactored { equality_point: Vec<E> },
}

/// One single-instance sumcheck's public shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SumcheckInstance<E> {
    /// The invocation's index among its protocol's sumchecks at one stage.
    pub(crate) invocation: u32,
    pub(crate) rounds: usize,
    pub(crate) degree: usize,
    pub(crate) format: SumcheckFormat<E>,
}

/// A fold verifier's replay: the grinding replay it reads the proof through,
/// and how it holds and checks the claims that depend on the witness.
pub(crate) trait ClaimReplay<F, E>:
    FoldResponseReplay<State: PrivateProofChannel<F>>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + ChallengeCodec,
{
    /// A witness-dependent claim: a field value in the clear verifier.
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
    /// A replay that does not support the path.
    fn clear_path(value: E) -> Result<Self::Claim, AkitaError>;

    /// Runs `replay`, a path only the clear verifier admits (the binary
    /// root), on the clear grinding replay.
    ///
    /// # Errors
    ///
    /// `replay`'s error.
    #[cfg(feature = "labinius-trinomial")]
    fn clear_replay<R>(
        &mut self,
        replay: impl FnOnce(&mut VerifierGrinding<'_, '_>) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError>;

    /// The values of claims a clear-only path consumes: Stage-1 product
    /// stages, the physical ℓ2 norm, extension-field opening claims.
    ///
    /// # Errors
    ///
    /// As [`clear_path`](Self::clear_path).
    fn clear_values(claims: &[Self::Claim]) -> Result<Vec<E>, AkitaError>;

    /// Receives the digest of a terminal `t` state's last block
    /// ([`akita_transcript::receive_field_digest`]).
    ///
    /// # Errors
    ///
    /// The transcript's error.
    fn receive_field_digest(
        &mut self,
        site: ProtocolSiteId,
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
    /// returns its challenges and running output claim.
    ///
    /// # Errors
    ///
    /// The sumcheck's or the channel's error.
    fn single_sumcheck(
        &mut self,
        protocol: akita_params::SumcheckProtocol,
        level: u32,
        stage: u32,
        instance: SumcheckInstance<E>,
        input: Self::Claim,
    ) -> Result<(Vec<E>, Self::Claim), AkitaError>;

    /// Replays one sumcheck that is public in both modes (Stage 3):
    /// its rounds arrive clear and its claims are values.
    ///
    /// # Errors
    ///
    /// The sumcheck's or the channel's error.
    fn public_sumcheck(
        &mut self,
        protocol: akita_params::SumcheckProtocol,
        level: u32,
        stage: u32,
        instance: SumcheckInstance<E>,
        input: E,
    ) -> Result<(Vec<E>, E), AkitaError>;

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
    /// [`AkitaError::InvalidProof`] if the check fails.
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
    /// [`AkitaError::InvalidProof`] if the check fails.
    fn check_stage2_oracle(
        &mut self,
        weights: Stage2OracleWeights<E>,
        witness_eval: Self::Claim,
        claim: Self::Claim,
    ) -> Result<(), AkitaError>;
}

/// The clear verifier: every claim is a value, checked when it is formed.
impl<'proof, F, E> ClaimReplay<F, E> for VerifierGrinding<'proof, '_, VerifierChannel<'proof>>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + ChallengeCodec,
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

    #[cfg(feature = "labinius-trinomial")]
    fn clear_replay<R>(
        &mut self,
        replay: impl FnOnce(&mut VerifierGrinding<'_, '_>) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        replay(self)
    }

    fn clear_values(claims: &[E]) -> Result<Vec<E>, AkitaError> {
        Ok(claims.to_vec())
    }

    fn receive_field_digest(
        &mut self,
        site: ProtocolSiteId,
    ) -> Result<[u8; FIELD_DIGEST_BYTES], AkitaError> {
        akita_transcript::receive_field_digest(self.state_mut(), site)
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
        instance: SumcheckInstance<E>,
        input: E,
    ) -> Result<(Vec<E>, E), AkitaError> {
        clear_sumcheck::<F, E, _>(self, protocol, level, stage, instance, input)
    }

    fn public_sumcheck(
        &mut self,
        protocol: akita_params::SumcheckProtocol,
        level: u32,
        stage: u32,
        instance: SumcheckInstance<E>,
        input: E,
    ) -> Result<(Vec<E>, E), AkitaError> {
        clear_sumcheck::<F, E, _>(self, protocol, level, stage, instance, input)
    }

    fn range_image(&mut self, level: u32, stage: u32) -> Result<E, AkitaError> {
        akita_types::stage1_range_image::<F, E, _>(self, level, stage, E::zero())
    }

    fn witness_eval(&mut self, level: u32) -> Result<E, AkitaError> {
        akita_types::stage2_w_eval::<F, E, _>(self, level, E::zero())
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

/// Replays one sumcheck whose rounds arrive clear on `grinding`'s channel.
pub(crate) fn clear_sumcheck<'proof, F, E, G>(
    grinding: &mut G,
    protocol: akita_params::SumcheckProtocol,
    level: u32,
    stage: u32,
    instance: SumcheckInstance<E>,
    input: E,
) -> Result<(Vec<E>, E), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + ChallengeCodec,
    G: GrindingReplay<State: VerifierGrindingChannel<'proof>>,
{
    let shape = akita_sumcheck::SumcheckShape::new(instance.rounds, instance.degree)?;
    let mut channel = GrindingSumcheckVerifier::<F, E, _>::new(grinding, protocol, level, stage);
    let replay = match instance.format {
        SumcheckFormat::Compressed => akita_sumcheck::verify_sumcheck_rounds::<F, E, _>(
            &mut channel,
            instance.invocation,
            input,
            shape,
        )?,
        SumcheckFormat::EqFactored { equality_point } => {
            akita_sumcheck::verify_eq_factored_sumcheck_rounds::<F, E, _>(
                &equality_point,
                input,
                shape,
                &mut channel,
                instance.invocation,
            )?
        }
    };
    Ok((replay.challenges, replay.output_claim))
}
