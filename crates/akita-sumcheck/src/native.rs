//! Native Spongefish proof-stream driver for standard sumcheck.

use crate::{CompressedUniPoly, SumcheckInstanceProver, SumcheckInstanceVerifier};
use akita_error::{checked, AkitaError};
use akita_transcript::{
    prover_context, receive_native_field, send_native_field, verifier_context, NativeField,
    NativeProverState, NativeVerifierState, ProtocolContextRecord, ProtocolMessageKind,
    ProtocolSiteId, SITE_FAMILY_SUMCHECK,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

const ROLE_CLAIM: u32 = 1;
const ROLE_ROUND_LENGTH: u32 = 2;
const ROLE_ROUND_BODY: u32 = 3;

/// Prover-side native operations required by the standard sumcheck driver.
pub trait NativeSumcheckProverChannel<E> {
    /// Borrow the native state for public and proof messages.
    fn state_mut(&mut self) -> &mut NativeProverState;

    /// Apply scheduled work and draw the challenge for `round`.
    fn round_challenge(&mut self, round: u32) -> Result<E, AkitaError>;
}

/// Verifier-side native operations required by the standard sumcheck driver.
pub trait NativeSumcheckVerifierChannel<'proof, E> {
    /// Borrow the native state for public and proof messages.
    fn state_mut(&mut self) -> &mut NativeVerifierState<'proof>;

    /// Verify scheduled work and draw the challenge for `round`.
    fn round_challenge(&mut self, round: u32) -> Result<E, AkitaError>;
}

fn site_id(invocation: u32, round: u32, role: u32) -> [u8; 32] {
    ProtocolSiteId {
        family: SITE_FAMILY_SUMCHECK,
        invocation,
        stage: role,
        round,
        ..ProtocolSiteId::default()
    }
    .to_bytes()
}

fn context(
    invocation: u32,
    round: u32,
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
        site_id(invocation, round, role),
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
    invocation: u32,
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
            invocation,
            0,
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
    invocation: u32,
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
            invocation,
            0,
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
    public_claim_prover::<F, E>(channel.state_mut(), invocation, claim)?;

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

        prover_context(
            channel.state_mut(),
            context(invocation, round_id, ROLE_ROUND_LENGTH, 1, 4, 0)?,
        );
        channel.state_mut().prover_message(&coefficient_count_u32);
        let atom_count = extension_atom_count::<E, F>(coefficient_count)?;
        prover_context(
            channel.state_mut(),
            context(
                invocation,
                round_id,
                ROLE_ROUND_BODY,
                atom_count,
                field_bytes::<F>(atom_count)?,
                0,
            )?,
        );
        for coefficient in &compressed.coeffs_except_linear_term {
            for base in coefficient.to_base_vec() {
                send_native_field(channel.state_mut(), base);
            }
        }
        let challenge = channel.round_challenge(round_id)?;
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
    let num_rounds = verifier.num_rounds();
    let degree_bound = verifier.degree_bound();
    let mut claim = verifier.input_claim();
    public_claim_verifier::<F, E>(channel.state_mut(), invocation, claim)?;

    let mut challenges = Vec::with_capacity(num_rounds);
    for round in 0..num_rounds {
        let round_id = u32::try_from(round).map_err(|_| AkitaError::InvalidProof)?;
        verifier_context(
            channel.state_mut(),
            context(invocation, round_id, ROLE_ROUND_LENGTH, 1, 4, 0)?,
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
        verifier_context(
            channel.state_mut(),
            context(
                invocation,
                round_id,
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
            let mut base_coefficients = Vec::new();
            base_coefficients
                .try_reserve_exact(E::DEGREE)
                .map_err(|_| AkitaError::InvalidProof)?;
            for _ in 0..E::DEGREE {
                base_coefficients.push(
                    receive_native_field::<F>(channel.state_mut())
                        .map_err(|_| AkitaError::InvalidProof)?,
                );
            }
            coefficients.push(E::from_base_slice(&base_coefficients));
        }
        let compressed = CompressedUniPoly {
            coeffs_except_linear_term: coefficients,
        };
        let challenge = channel.round_challenge(round_id)?;
        claim = compressed.eval_from_hint(&claim, &challenge);
        challenges.push(challenge);
    }

    if claim != verifier.expected_output_claim(&challenges)? {
        return Err(AkitaError::InvalidProof);
    }
    Ok(challenges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UniPoly;
    use akita_algebra::poly::multilinear_eval;
    use akita_transcript::{
        native_prover_field_challenge, native_verifier_field_challenge, new_native_prover,
        new_native_verifier, NATIVE_FIELD_CHALLENGE_BYTES,
    };
    use jolt_field::{Prime128Offset275 as F, Ring, Zero};

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

        fn round_challenge(&mut self, round: u32) -> Result<F, AkitaError> {
            prover_context(
                &mut self.state,
                ProtocolContextRecord::new(
                    site_id(self.invocation, round, 4),
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

        fn round_challenge(&mut self, round: u32) -> Result<F, AkitaError> {
            verifier_context(
                &mut self.state,
                ProtocolContextRecord::new(
                    site_id(self.invocation, round, 4),
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
}
