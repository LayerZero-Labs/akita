//! Native Spongefish proof-stream driver for standard sumcheck.

use crate::{
    advance_eq_factored_claim, CompressedUniPoly, EqFactoredSumcheckInstanceProver,
    EqFactoredUniPoly, SumcheckInstanceProver, SumcheckInstanceVerifier,
};
use akita_algebra::split_eq::GruenSplitEq;
use akita_error::{checked, AkitaError};
use akita_transcript::{
    prover_context, receive_native_extension, send_native_extension, verifier_context, NativeField,
    NativeProverState, NativeVerifierState, ProtocolContextRecord, ProtocolMessageKind,
    ProtocolSiteId,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

const ROLE_CLAIM: u32 = 1;
const ROLE_ROUND_LENGTH: u32 = 2;
const ROLE_ROUND_BODY: u32 = 3;

/// Prover-side native operations required by the standard sumcheck driver.
pub trait NativeSumcheckProverChannel<E> {
    /// Borrow the native state for public and proof messages.
    fn state_mut(&mut self) -> &mut NativeProverState;

    /// Return the complete public identity for one sumcheck record.
    fn sumcheck_site(&self, invocation: u32, round: u32, role: u32) -> ProtocolSiteId;

    /// Apply scheduled work and draw the challenge for `round`.
    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError>;
}

/// Verifier-side native operations required by the standard sumcheck driver.
pub trait NativeSumcheckVerifierChannel<'proof, E> {
    /// Borrow the native state for public and proof messages.
    fn state_mut(&mut self) -> &mut NativeVerifierState<'proof>;

    /// Return the complete public identity for one sumcheck record.
    fn sumcheck_site(&self, invocation: u32, round: u32, role: u32) -> ProtocolSiteId;

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
    role: u32,
    atom_count: usize,
    encoded_bytes: usize,
    challenge_bytes: usize,
) -> Result<ProtocolContextRecord, AkitaError> {
    let kind = match role {
        ROLE_CLAIM => ProtocolMessageKind::PublicValue,
        ROLE_ROUND_LENGTH => ProtocolMessageKind::ProofLength,
        ROLE_ROUND_BODY => ProtocolMessageKind::ProofAtoms,
        _ => return Err(AkitaError::InvalidProof),
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
            ROLE_CLAIM,
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
            ROLE_CLAIM,
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
    invocation: u32,
) -> Result<(Vec<E>, E), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: NativeSumcheckProverChannel<E>,
    P: SumcheckInstanceProver<E> + ?Sized,
{
    let num_rounds = prover.num_rounds();
    let degree_bound = prover.degree_bound();
    let mut claim = prover.input_claim();
    let claim_site = channel.sumcheck_site(invocation, 0, ROLE_CLAIM);
    public_claim_prover::<F, E>(channel.state_mut(), claim_site, claim)?;

    let mut challenges = Vec::with_capacity(num_rounds);
    for round in 0..num_rounds {
        let round_id = u32::try_from(round).map_err(|_| AkitaError::InvalidProof)?;
        let poly = prover.compute_round_univariate(round, claim);
        let compressed = poly.compress();
        let coefficient_count = compressed.coeffs_except_linear_term.len();
        if coefficient_count == 0 || coefficient_count > degree_bound {
            return Err(AkitaError::InvalidProof);
        }
        let coefficient_count_u32 =
            u32::try_from(coefficient_count).map_err(|_| AkitaError::InvalidProof)?;

        let length_site = channel.sumcheck_site(invocation, round_id, ROLE_ROUND_LENGTH);
        prover_context(
            channel.state_mut(),
            context(length_site, ROLE_ROUND_LENGTH, 1, 4, 0)?,
        );
        channel.state_mut().prover_message(&coefficient_count_u32);
        let atom_count = extension_atom_count::<E, F>(coefficient_count)?;
        let body_site = channel.sumcheck_site(invocation, round_id, ROLE_ROUND_BODY);
        prover_context(
            channel.state_mut(),
            context(
                body_site,
                ROLE_ROUND_BODY,
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
        prover.ingest_challenge(round, challenge);
        challenges.push(challenge);
    }
    prover.finalize();
    Ok((challenges, claim))
}

/// Verify one standard sumcheck by receiving its messages from Spongefish.
///
/// This function does not accept a structured proof. Counts are bounded by the
/// verifier's public sumcheck parameters before allocation.
pub fn verify_sumcheck_native<'proof, F, E, C, V>(
    verifier: &V,
    channel: &mut C,
    invocation: u32,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: NativeSumcheckVerifierChannel<'proof, E>,
    V: SumcheckInstanceVerifier<E> + ?Sized,
{
    let replay = verify_sumcheck_rounds_native::<F, E, C>(
        channel,
        invocation,
        verifier.input_claim(),
        verifier.num_rounds(),
        verifier.degree_bound(),
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
    num_rounds: usize,
    degree_bound: usize,
) -> Result<NativeSumcheckRoundResult<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: NativeSumcheckVerifierChannel<'proof, E>,
{
    let claim_site = channel.sumcheck_site(invocation, 0, ROLE_CLAIM);
    public_claim_verifier::<F, E>(channel.state_mut(), claim_site, claim)?;

    let mut challenges = Vec::with_capacity(num_rounds);
    for round in 0..num_rounds {
        let round_id = u32::try_from(round).map_err(|_| AkitaError::InvalidProof)?;
        let length_site = channel.sumcheck_site(invocation, round_id, ROLE_ROUND_LENGTH);
        verifier_context(
            channel.state_mut(),
            context(length_site, ROLE_ROUND_LENGTH, 1, 4, 0)?,
        );
        let coefficient_count = channel
            .state_mut()
            .prover_message::<u32>()
            .map_err(|_| AkitaError::InvalidProof)?;
        let coefficient_count =
            usize::try_from(coefficient_count).map_err(|_| AkitaError::InvalidProof)?;
        if coefficient_count == 0 || coefficient_count > degree_bound {
            return Err(AkitaError::InvalidProof);
        }
        let atom_count = extension_atom_count::<E, F>(coefficient_count)?;
        let body_site = channel.sumcheck_site(invocation, round_id, ROLE_ROUND_BODY);
        verifier_context(
            channel.state_mut(),
            context(
                body_site,
                ROLE_ROUND_BODY,
                atom_count,
                field_bytes::<F>(atom_count)?,
                0,
            )?,
        );
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(coefficient_count)
            .map_err(|_| AkitaError::InvalidProof)?;
        for _ in 0..coefficient_count {
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
    invocation: u32,
) -> Result<(Vec<E>, E), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: NativeSumcheckProverChannel<E>,
    P: EqFactoredSumcheckInstanceProver<E> + ?Sized,
{
    let num_rounds = prover.num_rounds();
    let degree_bound = prover.degree_bound();
    let mut claim = prover.input_claim();
    let claim_site = channel.sumcheck_site(invocation, 0, ROLE_CLAIM);
    public_claim_prover::<F, E>(channel.state_mut(), claim_site, claim)?;
    let mut challenges = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        let round_id = u32::try_from(round).map_err(|_| AkitaError::InvalidProof)?;
        let poly = prover.compute_round_eq_factored(round);
        let coefficient_count = poly.coeffs_except_constant_term.len();
        if coefficient_count > degree_bound {
            return Err(AkitaError::InvalidProof);
        }
        let coefficient_count_u32 =
            u32::try_from(coefficient_count).map_err(|_| AkitaError::InvalidProof)?;
        let length_site = channel.sumcheck_site(invocation, round_id, ROLE_ROUND_LENGTH);
        prover_context(
            channel.state_mut(),
            context(length_site, ROLE_ROUND_LENGTH, 1, 4, 0)?,
        );
        channel.state_mut().prover_message(&coefficient_count_u32);
        let atom_count = extension_atom_count::<E, F>(coefficient_count)?;
        let body_site = channel.sumcheck_site(invocation, round_id, ROLE_ROUND_BODY);
        prover_context(
            channel.state_mut(),
            context(
                body_site,
                ROLE_ROUND_BODY,
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
        prover.ingest_challenge(round, challenge);
    }
    prover.finalize();
    Ok((challenges, claim))
}

/// Verify one equality-factored sumcheck from the native argument stream.
pub fn verify_eq_factored_sumcheck_native<'proof, F, E, C, O>(
    equality_point: &[E],
    input_claim: E,
    degree_bound: usize,
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
        degree_bound,
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
    degree_bound: usize,
    channel: &mut C,
    invocation: u32,
) -> Result<NativeSumcheckRoundResult<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: NativeSumcheckVerifierChannel<'proof, E>,
{
    let mut equality = GruenSplitEq::new(equality_point)?;
    let mut claim = input_claim;
    let claim_site = channel.sumcheck_site(invocation, 0, ROLE_CLAIM);
    public_claim_verifier::<F, E>(channel.state_mut(), claim_site, claim)?;
    let mut challenges = Vec::with_capacity(equality_point.len());

    for round in 0..equality_point.len() {
        let round_id = u32::try_from(round).map_err(|_| AkitaError::InvalidProof)?;
        let length_site = channel.sumcheck_site(invocation, round_id, ROLE_ROUND_LENGTH);
        verifier_context(
            channel.state_mut(),
            context(length_site, ROLE_ROUND_LENGTH, 1, 4, 0)?,
        );
        let coefficient_count = channel
            .state_mut()
            .prover_message::<u32>()
            .map_err(|_| AkitaError::InvalidProof)?;
        let coefficient_count =
            usize::try_from(coefficient_count).map_err(|_| AkitaError::InvalidProof)?;
        if coefficient_count > degree_bound {
            return Err(AkitaError::InvalidProof);
        }
        let atom_count = extension_atom_count::<E, F>(coefficient_count)?;
        let body_site = channel.sumcheck_site(invocation, round_id, ROLE_ROUND_BODY);
        verifier_context(
            channel.state_mut(),
            context(
                body_site,
                ROLE_ROUND_BODY,
                atom_count,
                field_bytes::<F>(atom_count)?,
                0,
            )?,
        );
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(coefficient_count)
            .map_err(|_| AkitaError::InvalidProof)?;
        for _ in 0..coefficient_count {
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
        native_prover_field_challenge, native_verifier_field_challenge, new_native_prover,
        new_native_verifier, NATIVE_FIELD_CHALLENGE_BYTES, SITE_FAMILY_SUMCHECK,
    };
    use jolt_field::{One, Prime128Offset275 as F, Ring, Zero};

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

        fn sumcheck_site(&self, invocation: u32, round: u32, role: u32) -> ProtocolSiteId {
            ProtocolSiteId {
                family: SITE_FAMILY_SUMCHECK,
                invocation: self.invocation,
                round,
                group: invocation,
                detail: role,
                ..ProtocolSiteId::default()
            }
        }

        fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<F, AkitaError> {
            let site = self.sumcheck_site(invocation, round, 4);
            prover_context(
                &mut self.state,
                ProtocolContextRecord::new(
                    site.to_bytes(),
                    ProtocolMessageKind::Challenge as u32,
                    0,
                    0,
                    NATIVE_FIELD_CHALLENGE_BYTES,
                ),
            );
            Ok(native_prover_field_challenge(&mut self.state))
        }
    }

    struct TestVerifierChannel<'proof> {
        state: NativeVerifierState<'proof>,
        invocation: u32,
    }

    impl<'proof> NativeSumcheckVerifierChannel<'proof, F> for TestVerifierChannel<'proof> {
        fn state_mut(&mut self) -> &mut NativeVerifierState<'proof> {
            &mut self.state
        }

        fn sumcheck_site(&self, invocation: u32, round: u32, role: u32) -> ProtocolSiteId {
            ProtocolSiteId {
                family: SITE_FAMILY_SUMCHECK,
                invocation: self.invocation,
                round,
                group: invocation,
                detail: role,
                ..ProtocolSiteId::default()
            }
        }

        fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<F, AkitaError> {
            let site = self.sumcheck_site(invocation, round, 4);
            verifier_context(
                &mut self.state,
                ProtocolContextRecord::new(
                    site.to_bytes(),
                    ProtocolMessageKind::Challenge as u32,
                    0,
                    0,
                    NATIVE_FIELD_CHALLENGE_BYTES,
                ),
            );
            Ok(native_verifier_field_challenge(&mut self.state))
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
        let (prover_point, _) =
            prove_sumcheck_native(&mut prover_instance, &mut prover, 7).unwrap();
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
        let verifier_point = verify_sumcheck_native(&verifier_instance, &mut verifier, 7).unwrap();
        assert_eq!(verifier_point, prover_point);
        assert!(verifier.state.check_eof().is_ok());
    }

    #[test]
    fn native_sumcheck_rejects_truncation_and_wrong_site() {
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
        prove_sumcheck_native(&mut prover_instance, &mut prover, 7).unwrap();
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
            verify_sumcheck_native(&verifier_instance, &mut truncated, 7),
            Err(AkitaError::InvalidProof)
        );

        let mut wrong_site = TestVerifierChannel {
            state: new_native_verifier(b"native-sumcheck", b"fixture", proof).unwrap(),
            invocation: 8,
        };
        assert_eq!(
            verify_sumcheck_native(&verifier_instance, &mut wrong_site, 8),
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
        let mut prover = TestProverChannel {
            state: new_native_prover(b"native-eq-sumcheck", b"fixture").unwrap(),
            invocation: 12,
        };
        let (prover_point, _) =
            prove_eq_factored_sumcheck_native::<F, F, _, _>(&mut instance, &mut prover, 12)
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
            degree,
            &mut verifier,
            12,
            |point| Ok(expected.evaluate(point[0])),
        )
        .unwrap();
        assert_eq!(verifier_point, prover_point);
        assert!(verifier.state.check_eof().is_ok());
    }
}
