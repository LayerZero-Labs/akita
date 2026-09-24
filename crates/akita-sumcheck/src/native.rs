//! Native Spongefish proof-stream driver for standard sumcheck.

use crate::{
    advance_eq_factored_claim, CompressedUniPoly, EqFactoredUniPoly, SumcheckInstanceVerifier,
    SumcheckKernel,
};
#[cfg(test)]
use crate::{EqFactoredSumcheckInstanceProver, SumcheckInstanceProver};
use akita_algebra::split_eq::GruenSplitEq;
use akita_error::{checked, AkitaError};
use akita_transcript::{
    prover_context, receive_native_extension, send_native_extension, verifier_context, NativeField,
    NativeProverState, NativeVerifierState, ProtocolContextRecord, ProtocolMessageKind,
    ProtocolSiteId,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// Typed discriminator for native sumcheck transcript sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum NativeSumcheckRole {
    /// Public input claim.
    Claim = 1,
    /// Proof-supplied compressed round polynomial.
    RoundBody = 3,
    /// Verifier challenge drawn after a round body.
    Challenge = 4,
}

/// Public fixed grammar of one native sumcheck invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeSumcheckShape {
    num_rounds: usize,
    degree_bound: usize,
}

impl NativeSumcheckShape {
    /// Construct a checked native sumcheck grammar.
    ///
    /// # Errors
    ///
    /// Returns an error when a proof-controlled round body would have no
    /// coefficients or the shape cannot be represented safely.
    pub fn new(num_rounds: usize, degree_bound: usize) -> Result<Self, AkitaError> {
        if degree_bound == 0
            || u32::try_from(num_rounds).is_err()
            || checked::product([degree_bound, num_rounds]).is_none()
        {
            return Err(AkitaError::InvalidSetup(
                "invalid native sumcheck shape".into(),
            ));
        }
        Ok(Self {
            num_rounds,
            degree_bound,
        })
    }

    /// Number of proof rounds.
    #[must_use]
    pub const fn num_rounds(self) -> usize {
        self.num_rounds
    }

    /// Fixed extension coefficients emitted in each compressed round body.
    #[must_use]
    pub const fn degree_bound(self) -> usize {
        self.degree_bound
    }

    fn validate_instance(self, num_rounds: usize, degree_bound: usize) -> Result<(), AkitaError> {
        if self.num_rounds != num_rounds || self.degree_bound != degree_bound {
            return Err(AkitaError::InvalidSetup(
                "native sumcheck instance disagrees with its public shape".into(),
            ));
        }
        Ok(())
    }
}

/// Prover-side native operations required by the standard sumcheck driver.
pub trait NativeSumcheckProverChannel<E> {
    /// Borrow the native state for public and proof messages.
    fn state_mut(&mut self) -> &mut NativeProverState;

    /// Return the complete public identity for one sumcheck record.
    fn sumcheck_site(
        &self,
        invocation: u32,
        round: u32,
        role: NativeSumcheckRole,
    ) -> ProtocolSiteId;

    /// Apply scheduled work and draw the challenge for `round`.
    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError>;
}

/// Verifier-side native operations required by the standard sumcheck driver.
pub trait NativeSumcheckVerifierChannel<'proof, E> {
    /// Borrow the native state for public and proof messages.
    fn state_mut(&mut self) -> &mut NativeVerifierState<'proof>;

    /// Return the complete public identity for one sumcheck record.
    fn sumcheck_site(
        &self,
        invocation: u32,
        round: u32,
        role: NativeSumcheckRole,
    ) -> ProtocolSiteId;

    /// Verify scheduled work and draw the challenge for `round`.
    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError>;
}

/// Verifier output after native round replay and before an oracle check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeSumcheckRoundResult<E: Field> {
    /// Claim obtained after replaying every round.
    pub output_claim: E,
    /// Fiat--Shamir point sampled during replay.
    pub challenges: Vec<E>,
}

fn context(
    site: ProtocolSiteId,
    role: NativeSumcheckRole,
    atom_count: usize,
    encoded_bytes: usize,
    challenge_bytes: usize,
) -> Result<ProtocolContextRecord, AkitaError> {
    let kind = match role {
        NativeSumcheckRole::Claim => ProtocolMessageKind::PublicValue,
        NativeSumcheckRole::RoundBody => ProtocolMessageKind::ProofAtoms,
        NativeSumcheckRole::Challenge => return Err(AkitaError::InvalidProof),
    };
    Ok(ProtocolContextRecord::new(
        site.to_bytes(),
        kind as u32,
        u64::try_from(atom_count).map_err(|_| AkitaError::InvalidProof)?,
        u64::try_from(encoded_bytes).map_err(|_| AkitaError::InvalidProof)?,
        u64::try_from(challenge_bytes).map_err(|_| AkitaError::InvalidProof)?,
    ))
}

fn field_bytes<F: CanonicalEncoding>(count: usize) -> Result<usize, AkitaError> {
    checked::product([count, F::NUM_BYTES]).ok_or(AkitaError::InvalidProof)
}

fn extension_atom_count<E, F>(count: usize) -> Result<usize, AkitaError>
where
    F: Field,
    E: ExtField<F>,
{
    checked::product([count, E::DEGREE]).ok_or(AkitaError::InvalidProof)
}

fn public_claim_prover<F, E>(
    state: &mut NativeProverState,
    site: ProtocolSiteId,
    claim: E,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let coefficients = claim.to_base_vec();
    prover_context(
        state,
        context(
            site,
            NativeSumcheckRole::Claim,
            coefficients.len(),
            field_bytes::<F>(coefficients.len())?,
            0,
        )?,
    );
    for coefficient in coefficients {
        state.public_message(&NativeField::new(coefficient));
    }
    Ok(())
}

fn public_claim_verifier<F, E>(
    state: &mut NativeVerifierState<'_>,
    site: ProtocolSiteId,
    claim: E,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let coefficients = claim.to_base_vec();
    verifier_context(
        state,
        context(
            site,
            NativeSumcheckRole::Claim,
            coefficients.len(),
            field_bytes::<F>(coefficients.len())?,
            0,
        )?,
    );
    for coefficient in coefficients {
        state.public_message(&NativeField::new(coefficient));
    }
    Ok(())
}

/// Prove one standard sumcheck directly into a Spongefish argument string.
///
/// `invocation` is the schedule-derived identity of this sumcheck within the
/// enclosing Akita proof. The channel owns grinding and challenge context.
pub fn prove_sumcheck_native<F, E, C, P>(
    prover: &mut P,
    channel: &mut C,
    shape: NativeSumcheckShape,
    invocation: u32,
) -> Result<(Vec<E>, E), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: NativeSumcheckProverChannel<E>,
    P: SumcheckKernel<E> + ?Sized,
{
    shape.validate_instance(prover.num_rounds(), prover.degree_bound())?;
    let num_rounds = shape.num_rounds();
    let degree_bound = shape.degree_bound();
    let mut claim = prover.input_claim();
    let claim_site = channel.sumcheck_site(invocation, 0, NativeSumcheckRole::Claim);
    public_claim_prover::<F, E>(channel.state_mut(), claim_site, claim)?;

    let mut challenges = Vec::with_capacity(num_rounds);
    for round in 0..num_rounds {
        let round_id = u32::try_from(round).map_err(|_| AkitaError::InvalidProof)?;
        let poly = prover.round_polynomial(round, claim)?;
        if poly.evaluate(&E::zero()) + poly.evaluate(&E::one()) != claim {
            return Err(AkitaError::InvalidInput(
                "sumcheck round polynomial does not match its input claim".into(),
            ));
        }
        let mut compressed = poly.compress();
        let coefficient_count = compressed.coeffs_except_linear_term.len();
        if coefficient_count == 0 || coefficient_count > degree_bound {
            return Err(AkitaError::InvalidProof);
        }
        compressed
            .coeffs_except_linear_term
            .resize(degree_bound, E::zero());
        let atom_count = extension_atom_count::<E, F>(degree_bound)?;
        let body_site = channel.sumcheck_site(invocation, round_id, NativeSumcheckRole::RoundBody);
        prover_context(
            channel.state_mut(),
            context(
                body_site,
                NativeSumcheckRole::RoundBody,
                atom_count,
                field_bytes::<F>(atom_count)?,
                0,
            )?,
        );
        for coefficient in &compressed.coeffs_except_linear_term {
            send_native_extension::<F, E>(channel.state_mut(), *coefficient);
        }
        let challenge = channel.round_challenge(invocation, round_id)?;
        claim = compressed.eval_from_hint(&claim, &challenge);
        prover.bind_challenge(round, challenge)?;
        challenges.push(challenge);
    }
    prover.finish()?;
    Ok((challenges, claim))
}

/// Verify one standard sumcheck by receiving its messages from Spongefish.
///
/// This function does not accept a structured proof. Counts are bounded by the
/// verifier's public sumcheck parameters before allocation.
pub fn verify_sumcheck_native<'proof, F, E, C, V>(
    verifier: &V,
    channel: &mut C,
    shape: NativeSumcheckShape,
    invocation: u32,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: NativeSumcheckVerifierChannel<'proof, E>,
    V: SumcheckInstanceVerifier<E> + ?Sized,
{
    shape.validate_instance(verifier.num_rounds(), verifier.degree_bound())?;
    let replay = verify_sumcheck_rounds_native::<F, E, C>(
        channel,
        invocation,
        verifier.input_claim(),
        shape,
    )?;
    if replay.output_claim != verifier.expected_output_claim(&replay.challenges)? {
        return Err(AkitaError::InvalidProof);
    }
    Ok(replay.challenges)
}

/// Receive and replay standard sumcheck rounds before the terminal oracle check.
pub fn verify_sumcheck_rounds_native<'proof, F, E, C>(
    channel: &mut C,
    invocation: u32,
    mut claim: E,
    shape: NativeSumcheckShape,
) -> Result<NativeSumcheckRoundResult<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: NativeSumcheckVerifierChannel<'proof, E>,
{
    let num_rounds = shape.num_rounds();
    let degree_bound = shape.degree_bound();
    let claim_site = channel.sumcheck_site(invocation, 0, NativeSumcheckRole::Claim);
    public_claim_verifier::<F, E>(channel.state_mut(), claim_site, claim)?;

    let mut challenges = Vec::with_capacity(num_rounds);
    for round in 0..num_rounds {
        let round_id = u32::try_from(round).map_err(|_| AkitaError::InvalidProof)?;
        let atom_count = extension_atom_count::<E, F>(degree_bound)?;
        let body_site = channel.sumcheck_site(invocation, round_id, NativeSumcheckRole::RoundBody);
        verifier_context(
            channel.state_mut(),
            context(
                body_site,
                NativeSumcheckRole::RoundBody,
                atom_count,
                field_bytes::<F>(atom_count)?,
                0,
            )?,
        );
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(degree_bound)
            .map_err(|_| AkitaError::InvalidProof)?;
        for _ in 0..degree_bound {
            coefficients.push(
                receive_native_extension::<F, E>(channel.state_mut())
                    .map_err(|_| AkitaError::InvalidProof)?,
            );
        }
        let compressed = CompressedUniPoly {
            coeffs_except_linear_term: coefficients,
        };
        let challenge = channel.round_challenge(invocation, round_id)?;
        claim = compressed.eval_from_hint(&claim, &challenge);
        challenges.push(challenge);
    }

    Ok(NativeSumcheckRoundResult {
        output_claim: claim,
        challenges,
    })
}

/// Prove one equality-factored sumcheck into the native argument stream.
pub fn prove_eq_factored_sumcheck_native<F, E, C, P>(
    prover: &mut P,
    channel: &mut C,
    shape: NativeSumcheckShape,
    invocation: u32,
) -> Result<(Vec<E>, E), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: NativeSumcheckProverChannel<E>,
    P: crate::EqFactoredSumcheckKernel<E> + ?Sized,
{
    shape.validate_instance(prover.num_rounds(), prover.degree_bound())?;
    let num_rounds = shape.num_rounds();
    let degree_bound = shape.degree_bound();
    let mut claim = prover.input_claim();
    let claim_site = channel.sumcheck_site(invocation, 0, NativeSumcheckRole::Claim);
    public_claim_prover::<F, E>(channel.state_mut(), claim_site, claim)?;
    let mut challenges = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        let round_id = u32::try_from(round).map_err(|_| AkitaError::InvalidProof)?;
        let mut poly = prover.round_polynomial(round, claim)?;
        let coefficient_count = poly.coeffs_except_constant_term.len();
        if coefficient_count > degree_bound {
            return Err(AkitaError::InvalidProof);
        }
        poly.coeffs_except_constant_term
            .resize(degree_bound, E::zero());
        let atom_count = extension_atom_count::<E, F>(degree_bound)?;
        let body_site = channel.sumcheck_site(invocation, round_id, NativeSumcheckRole::RoundBody);
        prover_context(
            channel.state_mut(),
            context(
                body_site,
                NativeSumcheckRole::RoundBody,
                atom_count,
                field_bytes::<F>(atom_count)?,
                0,
            )?,
        );
        for coefficient in &poly.coeffs_except_constant_term {
            send_native_extension::<F, E>(channel.state_mut(), *coefficient);
        }
        let challenge = channel.round_challenge(invocation, round_id)?;
        claim = advance_eq_factored_claim(claim, prover.current_tau(), &poly, challenge);
        challenges.push(challenge);
        prover.bind_challenge(round, challenge)?;
    }
    prover.finish()?;
    Ok((challenges, claim))
}

/// Verify one equality-factored sumcheck from the native argument stream.
pub fn verify_eq_factored_sumcheck_native<'proof, F, E, C, O>(
    equality_point: &[E],
    input_claim: E,
    shape: NativeSumcheckShape,
    channel: &mut C,
    invocation: u32,
    expected_output_claim: O,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: NativeSumcheckVerifierChannel<'proof, E>,
    O: FnOnce(&[E]) -> Result<E, AkitaError>,
{
    let replay = verify_eq_factored_sumcheck_rounds_native::<F, E, C>(
        equality_point,
        input_claim,
        shape,
        channel,
        invocation,
    )?;
    if replay.output_claim != expected_output_claim(&replay.challenges)? {
        return Err(AkitaError::InvalidProof);
    }
    Ok(replay.challenges)
}

/// Replay equality-factored rounds before receiving and checking a late oracle claim.
pub fn verify_eq_factored_sumcheck_rounds_native<'proof, F, E, C>(
    equality_point: &[E],
    input_claim: E,
    shape: NativeSumcheckShape,
    channel: &mut C,
    invocation: u32,
) -> Result<NativeSumcheckRoundResult<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: NativeSumcheckVerifierChannel<'proof, E>,
{
    shape.validate_instance(equality_point.len(), shape.degree_bound())?;
    let degree_bound = shape.degree_bound();
    let mut equality = GruenSplitEq::new(equality_point)?;
    let mut claim = input_claim;
    let claim_site = channel.sumcheck_site(invocation, 0, NativeSumcheckRole::Claim);
    public_claim_verifier::<F, E>(channel.state_mut(), claim_site, claim)?;
    let mut challenges = Vec::with_capacity(equality_point.len());

    for round in 0..equality_point.len() {
        let round_id = u32::try_from(round).map_err(|_| AkitaError::InvalidProof)?;
        let atom_count = extension_atom_count::<E, F>(degree_bound)?;
        let body_site = channel.sumcheck_site(invocation, round_id, NativeSumcheckRole::RoundBody);
        verifier_context(
            channel.state_mut(),
            context(
                body_site,
                NativeSumcheckRole::RoundBody,
                atom_count,
                field_bytes::<F>(atom_count)?,
                0,
            )?,
        );
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(degree_bound)
            .map_err(|_| AkitaError::InvalidProof)?;
        for _ in 0..degree_bound {
            coefficients.push(
                receive_native_extension::<F, E>(channel.state_mut())
                    .map_err(|_| AkitaError::InvalidProof)?,
            );
        }
        let poly = EqFactoredUniPoly {
            coeffs_except_constant_term: coefficients,
        };
        let challenge = channel.round_challenge(invocation, round_id)?;
        claim = advance_eq_factored_claim(claim, equality.current_tau(), &poly, challenge);
        equality.bind(challenge);
        challenges.push(challenge);
    }
    Ok(NativeSumcheckRoundResult {
        output_claim: claim,
        challenges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UniPoly;
    use akita_algebra::poly::multilinear_eval;
    use akita_transcript::{
        native_field_challenge_bytes, native_prover_field_challenge,
        native_verifier_field_challenge, new_native_prover, new_native_verifier,
        SITE_FAMILY_SUMCHECK,
    };
    use jolt_field::{CanonicalBytes, One, Prime128Offset275 as F, Ring, Zero};

    struct DenseInstance {
        evaluations: Vec<F>,
        rounds: usize,
        claim: F,
    }

    impl SumcheckInstanceProver<F> for DenseInstance {
        fn num_rounds(&self) -> usize {
            self.rounds
        }

        fn degree_bound(&self) -> usize {
            1
        }

        fn input_claim(&self) -> F {
            self.claim
        }

        fn compute_round_univariate(&mut self, _round: usize, _claim: F) -> UniPoly<F> {
            let half = self.evaluations.len() / 2;
            let (zero, one) = (0..half).fold((F::zero(), F::zero()), |(zero, one), index| {
                (
                    zero + self.evaluations[2 * index],
                    one + self.evaluations[2 * index + 1],
                )
            });
            UniPoly::from_coeffs(vec![zero, one - zero])
        }

        fn ingest_challenge(&mut self, _round: usize, challenge: F) {
            let half = self.evaluations.len() / 2;
            for index in 0..half {
                let zero = self.evaluations[2 * index];
                self.evaluations[index] =
                    zero + challenge * (self.evaluations[2 * index + 1] - zero);
            }
            self.evaluations.truncate(half);
        }
    }

    impl SumcheckInstanceVerifier<F> for DenseInstance {
        fn num_rounds(&self) -> usize {
            self.rounds
        }

        fn degree_bound(&self) -> usize {
            1
        }

        fn input_claim(&self) -> F {
            self.claim
        }

        fn expected_output_claim(&self, challenges: &[F]) -> Result<F, AkitaError> {
            multilinear_eval(&self.evaluations, challenges)
        }
    }

    struct OneRoundEq {
        tau: F,
        split: GruenSplitEq<F>,
        coefficients: Vec<F>,
    }

    impl OneRoundEq {
        fn new(tau: F, coefficients: Vec<F>) -> Self {
            Self {
                tau,
                split: GruenSplitEq::new(&[tau]).unwrap(),
                coefficients,
            }
        }

        fn evaluate(&self, point: F) -> F {
            UniPoly::from_coeffs(self.coefficients.clone()).evaluate(&point)
        }

        fn claim(&self) -> F {
            (F::one() - self.tau) * self.evaluate(F::zero()) + self.tau * self.evaluate(F::one())
        }
    }

    impl EqFactoredSumcheckInstanceProver<F> for OneRoundEq {
        fn num_rounds(&self) -> usize {
            1
        }

        fn degree_bound(&self) -> usize {
            self.coefficients.len() - 1
        }

        fn input_claim(&self) -> F {
            self.claim()
        }

        fn current_tau(&self) -> F {
            self.split.current_tau()
        }

        fn compute_round_eq_factored(&mut self, _round: usize) -> EqFactoredUniPoly<F> {
            EqFactoredUniPoly::from_q_coeffs(self.coefficients.clone())
        }

        fn ingest_challenge(&mut self, _round: usize, challenge: F) {
            self.split.bind(challenge);
        }
    }

    fn fixture() -> (Vec<F>, F) {
        let evaluations = (1..=16).map(F::from_u64).collect::<Vec<_>>();
        let claim = evaluations.iter().copied().fold(F::zero(), |a, b| a + b);
        (evaluations, claim)
    }

    struct TestProverChannel {
        state: NativeProverState,
        invocation: u32,
    }

    impl NativeSumcheckProverChannel<F> for TestProverChannel {
        fn state_mut(&mut self) -> &mut NativeProverState {
            &mut self.state
        }

        fn sumcheck_site(
            &self,
            invocation: u32,
            round: u32,
            role: NativeSumcheckRole,
        ) -> ProtocolSiteId {
            ProtocolSiteId {
                family: SITE_FAMILY_SUMCHECK,
                invocation: self.invocation,
                round,
                group: invocation,
                detail: role as u32,
                ..ProtocolSiteId::default()
            }
        }

        fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<F, AkitaError> {
            let site = self.sumcheck_site(invocation, round, NativeSumcheckRole::Challenge);
            prover_context(
                &mut self.state,
                ProtocolContextRecord::new(
                    site.to_bytes(),
                    ProtocolMessageKind::Challenge as u32,
                    0,
                    0,
                    native_field_challenge_bytes::<F>(),
                ),
            );
            native_prover_field_challenge(&mut self.state).map_err(|_| AkitaError::InvalidProof)
        }
    }

    struct TestVerifierChannel<'proof> {
        state: NativeVerifierState<'proof>,
        invocation: u32,
    }

    struct FixedProverChannel {
        state: NativeProverState,
        challenges: Vec<F>,
    }

    impl NativeSumcheckProverChannel<F> for FixedProverChannel {
        fn state_mut(&mut self) -> &mut NativeProverState {
            &mut self.state
        }

        fn sumcheck_site(
            &self,
            invocation: u32,
            round: u32,
            role: NativeSumcheckRole,
        ) -> ProtocolSiteId {
            ProtocolSiteId {
                family: SITE_FAMILY_SUMCHECK,
                invocation,
                round,
                detail: role as u32,
                ..ProtocolSiteId::default()
            }
        }

        fn round_challenge(&mut self, _invocation: u32, round: u32) -> Result<F, AkitaError> {
            self.challenges
                .get(round as usize)
                .copied()
                .ok_or(AkitaError::InvalidProof)
        }
    }

    struct FixedVerifierChannel<'proof> {
        state: NativeVerifierState<'proof>,
        challenges: Vec<F>,
    }

    impl<'proof> NativeSumcheckVerifierChannel<'proof, F> for FixedVerifierChannel<'proof> {
        fn state_mut(&mut self) -> &mut NativeVerifierState<'proof> {
            &mut self.state
        }

        fn sumcheck_site(
            &self,
            invocation: u32,
            round: u32,
            role: NativeSumcheckRole,
        ) -> ProtocolSiteId {
            ProtocolSiteId {
                family: SITE_FAMILY_SUMCHECK,
                invocation,
                round,
                detail: role as u32,
                ..ProtocolSiteId::default()
            }
        }

        fn round_challenge(&mut self, _invocation: u32, round: u32) -> Result<F, AkitaError> {
            self.challenges
                .get(round as usize)
                .copied()
                .ok_or(AkitaError::InvalidProof)
        }
    }

    struct TwoRoundEq {
        equality: [F; 2],
        split: GruenSplitEq<F>,
        coefficients: [F; 4],
        first_challenge: Option<F>,
    }

    impl TwoRoundEq {
        fn new(equality: [F; 2], coefficients: [F; 4]) -> Self {
            Self {
                equality,
                split: GruenSplitEq::new(&equality).unwrap(),
                coefficients,
                first_challenge: None,
            }
        }

        fn evaluate(&self, x: F, y: F) -> F {
            let [a, b, c, d] = self.coefficients;
            a + b * x + c * y + d * x * y
        }
    }

    impl EqFactoredSumcheckInstanceProver<F> for TwoRoundEq {
        fn num_rounds(&self) -> usize {
            2
        }

        fn degree_bound(&self) -> usize {
            1
        }

        fn input_claim(&self) -> F {
            self.evaluate(self.equality[0], self.equality[1])
        }

        fn current_tau(&self) -> F {
            self.split.current_tau()
        }

        fn compute_round_eq_factored(&mut self, round: usize) -> EqFactoredUniPoly<F> {
            let [a, b, c, d] = self.coefficients;
            let coefficients = if round == 0 {
                vec![a + c * self.equality[1], b + d * self.equality[1]]
            } else {
                let first = self.first_challenge.unwrap();
                vec![a + b * first, c + d * first]
            };
            EqFactoredUniPoly::from_q_coeffs(coefficients)
        }

        fn ingest_challenge(&mut self, round: usize, challenge: F) {
            if round == 0 {
                self.first_challenge = Some(challenge);
            }
            self.split.bind(challenge);
        }
    }

    impl<'proof> NativeSumcheckVerifierChannel<'proof, F> for TestVerifierChannel<'proof> {
        fn state_mut(&mut self) -> &mut NativeVerifierState<'proof> {
            &mut self.state
        }

        fn sumcheck_site(
            &self,
            invocation: u32,
            round: u32,
            role: NativeSumcheckRole,
        ) -> ProtocolSiteId {
            ProtocolSiteId {
                family: SITE_FAMILY_SUMCHECK,
                invocation: self.invocation,
                round,
                group: invocation,
                detail: role as u32,
                ..ProtocolSiteId::default()
            }
        }

        fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<F, AkitaError> {
            let site = self.sumcheck_site(invocation, round, NativeSumcheckRole::Challenge);
            verifier_context(
                &mut self.state,
                ProtocolContextRecord::new(
                    site.to_bytes(),
                    ProtocolMessageKind::Challenge as u32,
                    0,
                    0,
                    native_field_challenge_bytes::<F>(),
                ),
            );
            native_verifier_field_challenge(&mut self.state).map_err(|_| AkitaError::InvalidProof)
        }
    }

    #[test]
    fn native_sumcheck_roundtrip_consumes_the_argument() {
        let (evaluations, claim) = fixture();
        let mut prover_instance = DenseInstance {
            evaluations: evaluations.clone(),
            rounds: 4,
            claim,
        };
        let mut prover = TestProverChannel {
            state: new_native_prover(b"native-sumcheck", b"fixture").unwrap(),
            invocation: 7,
        };
        let shape = NativeSumcheckShape::new(4, 1).unwrap();
        let (prover_point, _) = prove_sumcheck_native(
            &mut crate::InfallibleSumcheck(&mut prover_instance),
            &mut prover,
            shape,
            7,
        )
        .unwrap();
        let proof = prover.state.narg_string().to_vec();

        let verifier_instance = DenseInstance {
            evaluations,
            rounds: 4,
            claim,
        };
        let mut verifier = TestVerifierChannel {
            state: new_native_verifier(b"native-sumcheck", b"fixture", &proof).unwrap(),
            invocation: 7,
        };
        let verifier_point =
            verify_sumcheck_native(&verifier_instance, &mut verifier, shape, 7).unwrap();
        assert_eq!(verifier_point, prover_point);
        assert!(verifier.state.check_eof().is_ok());
    }

    #[test]
    fn native_sumcheck_rejects_truncation() {
        let (evaluations, claim) = fixture();
        let mut prover_instance = DenseInstance {
            evaluations: evaluations.clone(),
            rounds: 4,
            claim,
        };
        let mut prover = TestProverChannel {
            state: new_native_prover(b"native-sumcheck", b"fixture").unwrap(),
            invocation: 7,
        };
        let shape = NativeSumcheckShape::new(4, 1).unwrap();
        prove_sumcheck_native(
            &mut crate::InfallibleSumcheck(&mut prover_instance),
            &mut prover,
            shape,
            7,
        )
        .unwrap();
        let proof = prover.state.narg_string();
        let verifier_instance = DenseInstance {
            evaluations,
            rounds: 4,
            claim,
        };

        let mut truncated = TestVerifierChannel {
            state: new_native_verifier(b"native-sumcheck", b"fixture", &proof[..proof.len() - 1])
                .unwrap(),
            invocation: 7,
        };
        assert_eq!(
            verify_sumcheck_native(&verifier_instance, &mut truncated, shape, 7),
            Err(AkitaError::InvalidProof)
        );
    }

    #[test]
    fn native_eq_factored_sumcheck_roundtrip() {
        let tau = F::from_u64(7);
        let coefficients = vec![F::from_u64(3), F::from_u64(5), F::from_u64(11)];
        let mut instance = OneRoundEq::new(tau, coefficients.clone());
        let claim = instance.claim();
        let degree = instance.degree_bound();
        let shape = NativeSumcheckShape::new(1, degree).unwrap();
        let mut prover = TestProverChannel {
            state: new_native_prover(b"native-eq-sumcheck", b"fixture").unwrap(),
            invocation: 12,
        };
        let (prover_point, _) = prove_eq_factored_sumcheck_native::<F, F, _, _>(
            &mut crate::InfallibleEqFactoredSumcheck(&mut instance),
            &mut prover,
            shape,
            12,
        )
        .unwrap();
        let proof = prover.state.narg_string().to_vec();

        let expected = OneRoundEq::new(tau, coefficients);
        let mut verifier = TestVerifierChannel {
            state: new_native_verifier(b"native-eq-sumcheck", b"fixture", &proof).unwrap(),
            invocation: 12,
        };
        let verifier_point = verify_eq_factored_sumcheck_native::<F, F, _, _>(
            &[tau],
            claim,
            shape,
            &mut verifier,
            12,
            |point| Ok(expected.evaluate(point[0])),
        )
        .unwrap();
        assert_eq!(verifier_point, prover_point);
        assert!(verifier.state.check_eof().is_ok());
    }

    #[test]
    fn native_eq_factored_rejects_old_wire_forgery_when_tau_is_zero() {
        let coefficients = vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)];
        let instance = OneRoundEq::new(F::zero(), coefficients);
        let challenge = F::from_u64(11);
        let mut proof_state = new_native_prover(b"native-eq-forgery", b"fixture").unwrap();
        send_native_extension::<F, F>(&mut proof_state, instance.claim());
        send_native_extension::<F, F>(&mut proof_state, F::from_u64(101));
        let proof = proof_state.narg_string().to_vec();
        let mut verifier = FixedVerifierChannel {
            state: new_native_verifier(b"native-eq-forgery", b"fixture", &proof).unwrap(),
            challenges: vec![challenge],
        };

        assert_eq!(
            verify_eq_factored_sumcheck_native::<F, F, _, _>(
                &[F::zero()],
                instance.claim(),
                NativeSumcheckShape::new(1, instance.degree_bound()).unwrap(),
                &mut verifier,
                19,
                |_| Ok(instance.evaluate(challenge)),
            ),
            Err(AkitaError::InvalidProof)
        );
    }

    #[test]
    fn native_eq_factored_rejects_late_tampering_after_vanished_factor() {
        let equality = [F::from_u64(2), F::from_u64(5)];
        let coefficients = [
            F::from_u64(3),
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(13),
        ];
        let point = [F::from_u64(3).inverse().unwrap(), F::from_u64(17)];
        let mut instance = TwoRoundEq::new(equality, coefficients);
        let input_claim = instance.input_claim();
        let mut prover = FixedProverChannel {
            state: new_native_prover(b"native-eq-late", b"fixture").unwrap(),
            challenges: point.to_vec(),
        };
        let shape = NativeSumcheckShape::new(2, 1).unwrap();
        prove_eq_factored_sumcheck_native::<F, F, _, _>(
            &mut crate::InfallibleEqFactoredSumcheck(&mut instance),
            &mut prover,
            shape,
            23,
        )
        .unwrap();
        let honest = prover.state.narg_string().to_vec();
        let expected = TwoRoundEq::new(equality, coefficients).evaluate(point[0], point[1]);
        let verify = |proof: &[u8]| {
            let mut verifier = FixedVerifierChannel {
                state: new_native_verifier(b"native-eq-late", b"fixture", proof).unwrap(),
                challenges: point.to_vec(),
            };
            verify_eq_factored_sumcheck_native::<F, F, _, _>(
                &equality,
                input_claim,
                shape,
                &mut verifier,
                23,
                |_| Ok(expected),
            )
        };
        assert_eq!(verify(&honest), Ok(point.to_vec()));

        let mut tampered = honest;
        let second = F::from_u64(11) + F::from_u64(13) * point[0] + F::one();
        tampered[F::NUM_BYTES..2 * F::NUM_BYTES].copy_from_slice(&second.to_bytes_le_vec());
        assert_eq!(verify(&tampered), Err(AkitaError::InvalidProof));
    }
}
