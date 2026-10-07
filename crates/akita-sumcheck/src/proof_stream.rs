//! Proof-stream driver for standard sumcheck.

use crate::{advance_eq_factored_claim, SumcheckInstanceVerifier, SumcheckKernel};
#[cfg(test)]
use crate::{EqFactoredSumcheckInstanceProver, SumcheckInstanceProver};
use akita_algebra::split_eq::GruenSplitEq;
use akita_error::{checked, AkitaError};
use jolt_field::{CanonicalDecode, CanonicalEncoding, ExtField, Field};
use jolt_poly::{CompressedPoly, OmittedConstantPoly};
use jolt_transcript::{Channel, ProverTranscript, SiteId, Sponge, VerifierTranscript};

/// Typed discriminator for sumcheck transcript sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SumcheckRole {
    /// Public input claim.
    Claim = 1,
    /// Proof-supplied compressed round polynomial.
    RoundBody = 3,
    /// Verifier challenge drawn after a round body.
    Challenge = 4,
}

/// Public fixed grammar of one sumcheck invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckShape {
    num_rounds: usize,
    degree_bound: usize,
}

impl SumcheckShape {
    /// Construct a checked sumcheck grammar.
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

/// Prover-side operations required by the standard sumcheck driver.
pub trait SumcheckProverChannel<E> {
    /// Sponge of the borrowed transcript.
    type Sponge: Sponge;

    /// Borrow the transcript for public and proof messages.
    fn state_mut(&mut self) -> &mut ProverTranscript<Self::Sponge>;

    /// Return the diagnostic site of one sumcheck operation.
    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> SiteId;

    /// Apply scheduled work and draw the challenge for `round`.
    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError>;
}

/// Verifier-side operations required by the standard sumcheck driver.
pub trait SumcheckVerifierChannel<'proof, E> {
    /// Sponge of the borrowed transcript.
    type Sponge: Sponge;

    /// Borrow the transcript for public and proof messages.
    fn state_mut(&mut self) -> &mut VerifierTranscript<'proof, Self::Sponge>;

    /// Return the diagnostic site of one sumcheck operation.
    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> SiteId;

    /// Verify scheduled work and draw the challenge for `round`.
    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError>;
}

/// Verifier output after round replay and before an oracle check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRoundResult<E: Field> {
    /// Claim obtained after replaying every round.
    pub output_claim: E,
    /// Fiat--Shamir point sampled during replay.
    pub challenges: Vec<E>,
}

fn public_claim<E: CanonicalDecode, C: Channel>(channel: &mut C, site: SiteId, claim: &E) {
    channel.site(site);
    channel.public(claim);
}

/// Prove one standard sumcheck directly into the caller's argument string.
///
/// `invocation` is the schedule-derived identity of this sumcheck within the
/// enclosing Akita proof. The channel owns grinding and challenge context.
pub fn prove_sumcheck<F, E, C, P>(
    prover: &mut P,
    channel: &mut C,
    shape: SumcheckShape,
    invocation: u32,
) -> Result<(Vec<E>, E), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    C: SumcheckProverChannel<E>,
    P: SumcheckKernel<E> + ?Sized,
{
    shape.validate_instance(prover.num_rounds(), prover.degree_bound())?;
    let num_rounds = shape.num_rounds();
    let degree_bound = shape.degree_bound();
    let mut claim = prover.input_claim();
    let claim_site = channel.sumcheck_site(invocation, 0, SumcheckRole::Claim);
    public_claim(channel.state_mut(), claim_site, &claim);

    let mut challenges = Vec::with_capacity(num_rounds);
    for round in 0..num_rounds {
        let round_id = u32::try_from(round)
            .map_err(|_| AkitaError::Internal("sumcheck round index does not fit u32".into()))?;
        let poly = prover.round_polynomial(round, claim)?;
        if poly.evaluate(E::zero()) + poly.evaluate(E::one()) != claim {
            return Err(AkitaError::InvalidInput(
                "sumcheck round polynomial does not match its input claim".into(),
            ));
        }
        let mut coefficients = poly.compress().coeffs_except_linear_term().to_vec();
        let coefficient_count = coefficients.len();
        if coefficient_count == 0 {
            return Err(AkitaError::Internal(
                "sumcheck compressed coefficient count is zero".into(),
            ));
        }
        if coefficient_count > degree_bound {
            return Err(AkitaError::Internal(
                "sumcheck compressed coefficient count exceeds degree bound".into(),
            ));
        }
        coefficients.resize(degree_bound, E::zero());
        let compressed = CompressedPoly::new(coefficients);
        let body_site = channel.sumcheck_site(invocation, round_id, SumcheckRole::RoundBody);
        let state = channel.state_mut();
        state.site(body_site);
        state.send_all(compressed.coeffs_except_linear_term());
        let challenge = channel.round_challenge(invocation, round_id)?;
        claim = compressed.eval_from_hint(&claim, &challenge);
        prover.bind_challenge(round, challenge)?;
        challenges.push(challenge);
    }
    prover.finish()?;
    Ok((challenges, claim))
}

/// Verify one standard sumcheck by receiving its messages from the transcript.
///
/// This function does not accept a structured proof. Counts are bounded by the
/// verifier's public sumcheck parameters before allocation.
pub fn verify_sumcheck<'proof, F, E, C, V>(
    verifier: &V,
    channel: &mut C,
    shape: SumcheckShape,
    invocation: u32,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    C: SumcheckVerifierChannel<'proof, E>,
    V: SumcheckInstanceVerifier<E> + ?Sized,
{
    shape.validate_instance(verifier.num_rounds(), verifier.degree_bound())?;
    let replay =
        verify_sumcheck_rounds::<F, E, C>(channel, invocation, verifier.input_claim(), shape)?;
    if replay.output_claim != verifier.expected_output_claim(&replay.challenges)? {
        return Err(AkitaError::InvalidProof);
    }
    Ok(replay.challenges)
}

/// Receive and replay standard sumcheck rounds before the terminal oracle check.
pub fn verify_sumcheck_rounds<'proof, F, E, C>(
    channel: &mut C,
    invocation: u32,
    mut claim: E,
    shape: SumcheckShape,
) -> Result<SumcheckRoundResult<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    C: SumcheckVerifierChannel<'proof, E>,
{
    let num_rounds = shape.num_rounds();
    let degree_bound = shape.degree_bound();
    let claim_site = channel.sumcheck_site(invocation, 0, SumcheckRole::Claim);
    public_claim(channel.state_mut(), claim_site, &claim);

    let mut challenges = Vec::with_capacity(num_rounds);
    for round in 0..num_rounds {
        let round_id = u32::try_from(round).map_err(|_| AkitaError::InvalidProof)?;
        let body_site = channel.sumcheck_site(invocation, round_id, SumcheckRole::RoundBody);
        let state = channel.state_mut();
        state.site(body_site);
        let coefficients = state.receive_n::<E>(degree_bound)?;
        let compressed = CompressedPoly::new(coefficients);
        let challenge = channel.round_challenge(invocation, round_id)?;
        claim = compressed.eval_from_hint(&claim, &challenge);
        challenges.push(challenge);
    }

    Ok(SumcheckRoundResult {
        output_claim: claim,
        challenges,
    })
}

/// Prove one equality-factored sumcheck into the argument stream.
pub fn prove_eq_factored_sumcheck<F, E, C, P>(
    prover: &mut P,
    channel: &mut C,
    shape: SumcheckShape,
    invocation: u32,
) -> Result<(Vec<E>, E), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    C: SumcheckProverChannel<E>,
    P: crate::EqFactoredSumcheckKernel<E> + ?Sized,
{
    shape.validate_instance(prover.num_rounds(), prover.degree_bound())?;
    let num_rounds = shape.num_rounds();
    let degree_bound = shape.degree_bound();
    let mut claim = prover.input_claim();
    let claim_site = channel.sumcheck_site(invocation, 0, SumcheckRole::Claim);
    public_claim(channel.state_mut(), claim_site, &claim);
    let mut challenges = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        let round_id = u32::try_from(round).map_err(|_| {
            AkitaError::Internal("equality-factored sumcheck round index does not fit u32".into())
        })?;
        let mut coefficients = prover.round_polynomial(round, claim)?.into_coefficients();
        let coefficient_count = coefficients.len();
        if coefficient_count > degree_bound {
            return Err(AkitaError::Internal(
                "equality-factored sumcheck coefficient count exceeds the degree bound".into(),
            ));
        }
        coefficients.resize(degree_bound, E::zero());
        let poly = OmittedConstantPoly::new(coefficients);
        let body_site = channel.sumcheck_site(invocation, round_id, SumcheckRole::RoundBody);
        let state = channel.state_mut();
        state.site(body_site);
        state.send_all(poly.coefficients());
        let challenge = channel.round_challenge(invocation, round_id)?;
        claim = advance_eq_factored_claim(claim, prover.current_tau(), &poly, challenge);
        challenges.push(challenge);
        prover.bind_challenge(round, challenge)?;
    }
    prover.finish()?;
    Ok((challenges, claim))
}

/// Verify one equality-factored sumcheck from the argument stream.
pub fn verify_eq_factored_sumcheck<'proof, F, E, C, O>(
    equality_point: &[E],
    input_claim: E,
    shape: SumcheckShape,
    channel: &mut C,
    invocation: u32,
    expected_output_claim: O,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    C: SumcheckVerifierChannel<'proof, E>,
    O: FnOnce(&[E]) -> Result<E, AkitaError>,
{
    let replay = verify_eq_factored_sumcheck_rounds::<F, E, C>(
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
pub fn verify_eq_factored_sumcheck_rounds<'proof, F, E, C>(
    equality_point: &[E],
    input_claim: E,
    shape: SumcheckShape,
    channel: &mut C,
    invocation: u32,
) -> Result<SumcheckRoundResult<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    C: SumcheckVerifierChannel<'proof, E>,
{
    shape.validate_instance(equality_point.len(), shape.degree_bound())?;
    let degree_bound = shape.degree_bound();
    let mut equality = GruenSplitEq::new(equality_point)?;
    let mut claim = input_claim;
    let claim_site = channel.sumcheck_site(invocation, 0, SumcheckRole::Claim);
    public_claim(channel.state_mut(), claim_site, &claim);
    let mut challenges = Vec::with_capacity(equality_point.len());

    for round in 0..equality_point.len() {
        let round_id = u32::try_from(round).map_err(|_| AkitaError::InvalidProof)?;
        let body_site = channel.sumcheck_site(invocation, round_id, SumcheckRole::RoundBody);
        let state = channel.state_mut();
        state.site(body_site);
        let coefficients = state.receive_n::<E>(degree_bound)?;
        let poly = OmittedConstantPoly::new(coefficients);
        let challenge = channel.round_challenge(invocation, round_id)?;
        claim = advance_eq_factored_claim(claim, equality.current_tau(), &poly, challenge);
        equality.bind(challenge);
        challenges.push(challenge);
    }
    Ok(SumcheckRoundResult {
        output_claim: claim,
        challenges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::poly::multilinear_eval;
    use jolt_field::{CanonicalBytes, One, Prime128Offset275 as F, Ring, Zero};
    use jolt_poly::UnivariatePoly;
    use jolt_transcript::{Blake2b512 as H, ProtocolId};

    const PROTOCOL: ProtocolId = ProtocolId::new::<H>("akita-sumcheck/test");

    fn new_prover(session: &[u8]) -> ProverTranscript<H> {
        ProverTranscript::new(&PROTOCOL, session)
    }

    fn new_verifier<'proof>(session: &[u8], proof: &'proof [u8]) -> VerifierTranscript<'proof, H> {
        VerifierTranscript::new(&PROTOCOL, session, proof)
    }

    fn test_site(invocation: u32, round: u32, role: SumcheckRole) -> SiteId {
        let mut site = [0u8; 32];
        site[..4].copy_from_slice(&invocation.to_le_bytes());
        site[4..8].copy_from_slice(&round.to_le_bytes());
        site[8..12].copy_from_slice(&(role as u32).to_le_bytes());
        SiteId(site)
    }

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

        fn compute_round_univariate(&mut self, _round: usize, _claim: F) -> UnivariatePoly<F> {
            let half = self.evaluations.len() / 2;
            let (zero, one) = (0..half).fold((F::zero(), F::zero()), |(zero, one), index| {
                (
                    zero + self.evaluations[2 * index],
                    one + self.evaluations[2 * index + 1],
                )
            });
            UnivariatePoly::new(vec![zero, one - zero])
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
            UnivariatePoly::new(self.coefficients.clone()).evaluate(point)
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

        fn compute_round_eq_factored(
            &mut self,
            _round: usize,
            _claim: F,
        ) -> OmittedConstantPoly<F> {
            OmittedConstantPoly::from_q_coefficients(self.coefficients.clone())
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
        state: ProverTranscript<H>,
    }

    impl SumcheckProverChannel<F> for TestProverChannel {
        type Sponge = H;

        fn state_mut(&mut self) -> &mut ProverTranscript<H> {
            &mut self.state
        }

        fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> SiteId {
            test_site(invocation, round, role)
        }

        fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<F, AkitaError> {
            let site = self.sumcheck_site(invocation, round, SumcheckRole::Challenge);
            self.state.site(site);
            Ok(self.state.challenge())
        }
    }

    struct TestVerifierChannel<'proof> {
        state: VerifierTranscript<'proof, H>,
    }

    struct FixedProverChannel {
        state: ProverTranscript<H>,
        challenges: Vec<F>,
    }

    impl SumcheckProverChannel<F> for FixedProverChannel {
        type Sponge = H;

        fn state_mut(&mut self) -> &mut ProverTranscript<H> {
            &mut self.state
        }

        fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> SiteId {
            test_site(invocation, round, role)
        }

        fn round_challenge(&mut self, _invocation: u32, round: u32) -> Result<F, AkitaError> {
            self.challenges
                .get(round as usize)
                .copied()
                .ok_or(AkitaError::InvalidProof)
        }
    }

    struct FixedVerifierChannel<'proof> {
        state: VerifierTranscript<'proof, H>,
        challenges: Vec<F>,
    }

    impl<'proof> SumcheckVerifierChannel<'proof, F> for FixedVerifierChannel<'proof> {
        type Sponge = H;

        fn state_mut(&mut self) -> &mut VerifierTranscript<'proof, H> {
            &mut self.state
        }

        fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> SiteId {
            test_site(invocation, round, role)
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

        fn compute_round_eq_factored(&mut self, round: usize, _claim: F) -> OmittedConstantPoly<F> {
            let [a, b, c, d] = self.coefficients;
            let coefficients = if round == 0 {
                vec![a + c * self.equality[1], b + d * self.equality[1]]
            } else {
                let first = self.first_challenge.unwrap();
                vec![a + b * first, c + d * first]
            };
            OmittedConstantPoly::from_q_coefficients(coefficients)
        }

        fn ingest_challenge(&mut self, round: usize, challenge: F) {
            if round == 0 {
                self.first_challenge = Some(challenge);
            }
            self.split.bind(challenge);
        }
    }

    impl<'proof> SumcheckVerifierChannel<'proof, F> for TestVerifierChannel<'proof> {
        type Sponge = H;

        fn state_mut(&mut self) -> &mut VerifierTranscript<'proof, H> {
            &mut self.state
        }

        fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> SiteId {
            test_site(invocation, round, role)
        }

        fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<F, AkitaError> {
            let site = self.sumcheck_site(invocation, round, SumcheckRole::Challenge);
            self.state.site(site);
            Ok(self.state.challenge())
        }
    }

    #[test]
    fn sumcheck_rejects_round_polynomial_that_contradicts_the_claim() {
        let (evaluations, claim) = fixture();
        let mut instance = DenseInstance {
            evaluations,
            rounds: 4,
            claim: claim + F::one(),
        };
        let mut channel = TestProverChannel {
            state: new_prover(b"native-sumcheck/fixture"),
        };
        assert!(matches!(
            prove_sumcheck(
                &mut crate::InfallibleSumcheck(&mut instance),
                &mut channel,
                SumcheckShape::new(4, 1).unwrap(),
                7,
            ),
            Err(AkitaError::InvalidInput(_))
        ));
    }

    #[test]
    fn sumcheck_roundtrip_consumes_the_argument() {
        let (evaluations, claim) = fixture();
        let mut prover_instance = DenseInstance {
            evaluations: evaluations.clone(),
            rounds: 4,
            claim,
        };
        let mut prover = TestProverChannel {
            state: new_prover(b"native-sumcheck/fixture"),
        };
        let shape = SumcheckShape::new(4, 1).unwrap();
        let (prover_point, _) = prove_sumcheck(
            &mut crate::InfallibleSumcheck(&mut prover_instance),
            &mut prover,
            shape,
            7,
        )
        .unwrap();
        let proof = prover.state.narg().to_vec();

        let verifier_instance = DenseInstance {
            evaluations,
            rounds: 4,
            claim,
        };
        let mut verifier = TestVerifierChannel {
            state: new_verifier(b"native-sumcheck/fixture", &proof),
        };
        let verifier_point = verify_sumcheck(&verifier_instance, &mut verifier, shape, 7).unwrap();
        assert_eq!(verifier_point, prover_point);
        verifier.state.finish().unwrap();
    }

    #[test]
    fn sumcheck_rejects_truncation() {
        let (evaluations, claim) = fixture();
        let mut prover_instance = DenseInstance {
            evaluations: evaluations.clone(),
            rounds: 4,
            claim,
        };
        let mut prover = TestProverChannel {
            state: new_prover(b"native-sumcheck/fixture"),
        };
        let shape = SumcheckShape::new(4, 1).unwrap();
        prove_sumcheck(
            &mut crate::InfallibleSumcheck(&mut prover_instance),
            &mut prover,
            shape,
            7,
        )
        .unwrap();
        let proof = prover.state.narg();
        let verifier_instance = DenseInstance {
            evaluations,
            rounds: 4,
            claim,
        };

        let mut truncated = TestVerifierChannel {
            state: new_verifier(b"native-sumcheck/fixture", &proof[..proof.len() - 1]),
        };
        assert_eq!(
            verify_sumcheck(&verifier_instance, &mut truncated, shape, 7),
            Err(AkitaError::InvalidProof)
        );
    }

    #[test]
    fn eq_factored_sumcheck_roundtrip() {
        let tau = F::from_u64(7);
        let coefficients = vec![F::from_u64(3), F::from_u64(5), F::from_u64(11)];
        let mut instance = OneRoundEq::new(tau, coefficients.clone());
        let claim = instance.claim();
        let degree = instance.degree_bound();
        let shape = SumcheckShape::new(1, degree).unwrap();
        let mut prover = TestProverChannel {
            state: new_prover(b"native-eq-sumcheck/fixture"),
        };
        let (prover_point, _) = prove_eq_factored_sumcheck::<F, F, _, _>(
            &mut crate::InfallibleEqFactoredSumcheck(&mut instance),
            &mut prover,
            shape,
            12,
        )
        .unwrap();
        let proof = prover.state.narg().to_vec();

        let expected = OneRoundEq::new(tau, coefficients);
        let mut verifier = TestVerifierChannel {
            state: new_verifier(b"native-eq-sumcheck/fixture", &proof),
        };
        let verifier_point = verify_eq_factored_sumcheck::<F, F, _, _>(
            &[tau],
            claim,
            shape,
            &mut verifier,
            12,
            |point| Ok(expected.evaluate(point[0])),
        )
        .unwrap();
        assert_eq!(verifier_point, prover_point);
        verifier.state.finish().unwrap();
    }

    #[test]
    fn eq_factored_rejects_old_wire_forgery_when_tau_is_zero() {
        let coefficients = vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)];
        let instance = OneRoundEq::new(F::zero(), coefficients);
        let challenge = F::from_u64(11);
        let mut proof_state = new_prover(b"native-eq-forgery/fixture");
        proof_state.send(&instance.claim());
        proof_state.send(&F::from_u64(101));
        let proof = proof_state.narg().to_vec();
        let mut verifier = FixedVerifierChannel {
            state: new_verifier(b"native-eq-forgery/fixture", &proof),
            challenges: vec![challenge],
        };

        assert_eq!(
            verify_eq_factored_sumcheck::<F, F, _, _>(
                &[F::zero()],
                instance.claim(),
                SumcheckShape::new(1, instance.degree_bound()).unwrap(),
                &mut verifier,
                19,
                |_| Ok(instance.evaluate(challenge)),
            ),
            Err(AkitaError::InvalidProof)
        );
    }

    #[test]
    fn eq_factored_rejects_late_tampering_after_vanished_factor() {
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
            state: new_prover(b"native-eq-late/fixture"),
            challenges: point.to_vec(),
        };
        let shape = SumcheckShape::new(2, 1).unwrap();
        prove_eq_factored_sumcheck::<F, F, _, _>(
            &mut crate::InfallibleEqFactoredSumcheck(&mut instance),
            &mut prover,
            shape,
            23,
        )
        .unwrap();
        let honest = prover.state.narg().to_vec();
        let expected = TwoRoundEq::new(equality, coefficients).evaluate(point[0], point[1]);
        let verify = |proof: &[u8]| {
            let mut verifier = FixedVerifierChannel {
                state: new_verifier(b"native-eq-late/fixture", proof),
                challenges: point.to_vec(),
            };
            verify_eq_factored_sumcheck::<F, F, _, _>(
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

    #[test]
    fn arithmetic_proof_and_challenges_match_their_snapshot() {
        fn hex(bytes: &[u8]) -> String {
            bytes.iter().map(|byte| format!("{byte:02x}")).collect()
        }
        fn challenge_hex(point: &[F]) -> String {
            hex(&point
                .iter()
                .flat_map(|challenge| challenge.to_bytes_le_vec())
                .collect::<Vec<_>>())
        }

        // A snapshot taken by this code at the spongefish-state port, not
        // independent ground truth: it fails on any change to the argument
        // string or the challenges, which is a proof-format change.
        let (evaluations, claim) = fixture();
        let mut standard = DenseInstance {
            evaluations,
            rounds: 4,
            claim,
        };
        let mut prover = TestProverChannel {
            state: new_prover(b"native-cutover/standard"),
        };
        let (standard_point, _) = prove_sumcheck(
            &mut crate::InfallibleSumcheck(&mut standard),
            &mut prover,
            SumcheckShape::new(4, 1).unwrap(),
            3,
        )
        .unwrap();
        let standard_proof = hex(prover.state.narg());
        let standard_point = challenge_hex(&standard_point);

        let mut normalized = TwoRoundEq::new(
            [F::from_u64(2), F::from_u64(5)],
            [
                F::from_u64(3),
                F::from_u64(7),
                F::from_u64(11),
                F::from_u64(13),
            ],
        );
        let degree = normalized.degree_bound();
        let mut prover = TestProverChannel {
            state: new_prover(b"native-cutover/normalized"),
        };
        let (normalized_point, _) = prove_eq_factored_sumcheck::<F, F, _, _>(
            &mut crate::InfallibleEqFactoredSumcheck(&mut normalized),
            &mut prover,
            SumcheckShape::new(2, degree).unwrap(),
            5,
        )
        .unwrap();
        let normalized_proof = hex(prover.state.narg());
        let normalized_point = challenge_hex(&normalized_point);

        assert_eq!(
            standard_proof,
            "4000000000000000000000000000000049227796f06b10136bda182d8328234272f0fb6d0fa1ef791c5c5b0b8b436b80d3ea6e09e85716af98d63a142c0567b1"
        );
        assert_eq!(
            standard_point,
            "bdc79d25fc1ac4c49a3646cb20ca88d0b217b0c8c5da19bcb9bb335dd26bd6371e3c9c14d8a1879c224aa3a3d9584c9c574bb72471cec11a9595506b5b34e4a6"
        );
        assert_eq!(
            normalized_proof,
            "480000000000000000000000000000004d9d14cff02a39d6e78912a62aa8c577"
        );
        assert_eq!(
            normalized_point,
            "2b82c6994d03dd3739e363e5dbd1e71c264aa30819299b27127fbfe679315e17"
        );
    }
}
