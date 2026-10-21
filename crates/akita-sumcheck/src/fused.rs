//! Adapters from Akita's split kernel interface to `jolt-sumcheck`'s fused
//! round API, consumed by [`jolt_sumcheck::prove_single`].

use crate::{EqFactoredSumcheckKernel, SumcheckKernel};
use akita_error::AkitaError;
use jolt_algebra::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SingleFormat, SingleInstance};

/// Drives a [`SumcheckKernel`] through the fused round API: `prove_round`
/// binds the previous round's challenge, then computes the round polynomial;
/// `finish_rounds` binds the last challenge, then finishes the kernel.
pub struct FusedRounds<'a, K: ?Sized>(pub &'a mut K);

/// The [`FusedRounds`] adapter for an [`EqFactoredSumcheckKernel`]. Each
/// round returns the full inner polynomial `q`, its constant term recovered
/// from the running claim and `τ_round` of the instance's equality point, the
/// same coordinates the driver checks the round against.
pub struct FusedEqFactoredRounds<'a, E, K: ?Sized> {
    kernel: &'a mut K,
    equality_point: &'a [E],
}

impl<'a, E: Field, K: EqFactoredSumcheckKernel<E> + ?Sized> FusedEqFactoredRounds<'a, E, K> {
    /// Adapts `kernel` to prove `instance`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::Internal`] unless `instance` is eq-factored.
    pub fn new(kernel: &'a mut K, instance: &'a SingleInstance<E>) -> Result<Self, AkitaError> {
        let SingleFormat::EqFactored { equality_point } = &instance.format else {
            return Err(AkitaError::Internal(
                "eq-factored kernel paired with a compressed sumcheck instance".into(),
            ));
        };
        Ok(Self {
            kernel,
            equality_point: equality_point.as_slice(),
        })
    }
}

fn previous_round(round: usize) -> Result<usize, AkitaError> {
    round.checked_sub(1).ok_or_else(|| {
        AkitaError::Internal("sumcheck driver bound a challenge before round zero".into())
    })
}

fn last_round(num_rounds: usize) -> Result<usize, AkitaError> {
    num_rounds.checked_sub(1).ok_or_else(|| {
        AkitaError::Internal("sumcheck driver finished a kernel with no rounds".into())
    })
}

impl<E: Field, K: SumcheckKernel<E> + ?Sized> ProveRounds<E, AkitaError> for FusedRounds<'_, K> {
    fn num_rounds(&self) -> usize {
        self.0.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<E>,
        round: usize,
        previous_claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        if let Some(challenge) = bind {
            self.0.bind_challenge(previous_round(round)?, challenge)?;
        }
        self.0.round_polynomial(round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: E) -> Result<(), AkitaError> {
        self.0
            .bind_challenge(last_round(self.0.num_rounds())?, bind)?;
        self.0.finish()
    }
}

impl<E: Field, K: EqFactoredSumcheckKernel<E> + ?Sized> ProveRounds<E, AkitaError>
    for FusedEqFactoredRounds<'_, E, K>
{
    fn num_rounds(&self) -> usize {
        self.kernel.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<E>,
        round: usize,
        previous_claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        if let Some(challenge) = bind {
            self.kernel
                .bind_challenge(previous_round(round)?, challenge)?;
        }
        let tau = *self.equality_point.get(round).ok_or_else(|| {
            AkitaError::Internal("eq-factored round exceeds the equality point".into())
        })?;
        let q = self.kernel.round_polynomial(round, previous_claim)?;
        let constant = q.constant_from_claim(previous_claim, tau);
        let mut coefficients = Vec::with_capacity(q.degree() + 1);
        coefficients.push(constant);
        coefficients.extend_from_slice(q.coefficients());
        Ok(UnivariatePoly::new(coefficients))
    }

    fn finish_rounds(&mut self, bind: E) -> Result<(), AkitaError> {
        self.kernel
            .bind_challenge(last_round(self.kernel.num_rounds())?, bind)?;
        self.kernel.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::poly::multilinear_eval;
    use jolt_algebra::{CanonicalBytes, One, Prime128Offset275 as F, Ring, Zero};
    use jolt_poly::{OmittedConstantPoly, Point};
    use jolt_sumcheck::{
        prove_single, verify_single, ProverRoundContext, SingleFormat, SingleInstance,
        SingleReduction, SumcheckSite, VerifierRoundContext,
    };
    use jolt_transcript::{
        Blake2b512 as H, Channel, ProtocolId, ProverTranscript, SiteId, VerifierTranscript,
    };

    const PROTOCOL: ProtocolId = ProtocolId::new::<H>("akita-sumcheck/test");

    fn test_site(site: SumcheckSite) -> SiteId {
        let mut bytes = [0u8; 32];
        bytes[..4].copy_from_slice(&site.invocation.to_le_bytes());
        bytes[4..8].copy_from_slice(&site.round.to_le_bytes());
        bytes[8..12].copy_from_slice(&(site.role as u32).to_le_bytes());
        SiteId(bytes)
    }

    /// Draws transcript challenges, or replays `fixed` ones when present.
    struct TestProver {
        state: ProverTranscript<H>,
        fixed: Option<Vec<F>>,
    }

    impl TestProver {
        fn new(session: &[u8], fixed: Option<Vec<F>>) -> Self {
            Self {
                state: ProverTranscript::new(&PROTOCOL, session),
                fixed,
            }
        }
    }

    impl ProverRoundContext<F> for TestProver {
        type Channel = ProverTranscript<H>;
        type Error = AkitaError;

        fn channel_mut(&mut self) -> &mut ProverTranscript<H> {
            &mut self.state
        }

        fn site(&self, site: SumcheckSite) -> Option<SiteId> {
            Some(test_site(site))
        }

        fn challenge(&mut self, site: SumcheckSite) -> Result<F, AkitaError> {
            match &self.fixed {
                Some(fixed) => fixed
                    .get(site.round as usize)
                    .copied()
                    .ok_or(AkitaError::InvalidProof),
                None => {
                    self.state.site(test_site(site));
                    Ok(self.state.challenge())
                }
            }
        }
    }

    struct TestVerifier<'proof> {
        state: VerifierTranscript<'proof, H>,
        fixed: Option<Vec<F>>,
    }

    impl<'proof> TestVerifier<'proof> {
        fn new(session: &[u8], proof: &'proof [u8], fixed: Option<Vec<F>>) -> Self {
            Self {
                state: VerifierTranscript::new(&PROTOCOL, session, proof),
                fixed,
            }
        }
    }

    impl<'proof> VerifierRoundContext<F> for TestVerifier<'proof> {
        type Channel = VerifierTranscript<'proof, H>;
        type Error = AkitaError;

        fn channel_mut(&mut self) -> &mut VerifierTranscript<'proof, H> {
            &mut self.state
        }

        fn site(&self, site: SumcheckSite) -> Option<SiteId> {
            Some(test_site(site))
        }

        fn challenge(&mut self, site: SumcheckSite) -> Result<F, AkitaError> {
            match &self.fixed {
                Some(fixed) => fixed
                    .get(site.round as usize)
                    .copied()
                    .ok_or(AkitaError::InvalidProof),
                None => {
                    self.state.site(test_site(site));
                    Ok(self.state.challenge())
                }
            }
        }
    }

    struct DenseKernel {
        evaluations: Vec<F>,
        rounds: usize,
    }

    impl DenseKernel {
        fn new(evaluations: Vec<F>) -> Self {
            let rounds = evaluations.len().trailing_zeros() as usize;
            Self {
                evaluations,
                rounds,
            }
        }
    }

    impl SumcheckKernel<F> for DenseKernel {
        fn num_rounds(&self) -> usize {
            self.rounds
        }

        fn round_polynomial(
            &mut self,
            _round: usize,
            _claim: F,
        ) -> Result<UnivariatePoly<F>, AkitaError> {
            let (zero, one) = self
                .evaluations
                .chunks_exact(2)
                .fold((F::zero(), F::zero()), |(zero, one), pair| {
                    (zero + pair[0], one + pair[1])
                });
            Ok(UnivariatePoly::new(vec![zero, one - zero]))
        }

        fn bind_challenge(&mut self, _round: usize, challenge: F) -> Result<(), AkitaError> {
            let half = self.evaluations.len() / 2;
            for index in 0..half {
                let zero = self.evaluations[2 * index];
                self.evaluations[index] =
                    zero + challenge * (self.evaluations[2 * index + 1] - zero);
            }
            self.evaluations.truncate(half);
            Ok(())
        }
    }

    fn fixture() -> (Vec<F>, F) {
        let evaluations = (1..=16).map(F::from_u64).collect::<Vec<_>>();
        let claim = evaluations.iter().copied().fold(F::zero(), |a, b| a + b);
        (evaluations, claim)
    }

    fn compressed(input_claim: F, rounds: usize, invocation: u32) -> SingleInstance<F> {
        SingleInstance {
            input_claim,
            rounds,
            degree: 1,
            format: SingleFormat::Compressed,
            invocation,
        }
    }

    fn eq_factored(
        input_claim: F,
        equality_point: &[F],
        degree: usize,
        invocation: u32,
    ) -> SingleInstance<F> {
        SingleInstance {
            input_claim,
            rounds: equality_point.len(),
            degree,
            format: SingleFormat::EqFactored {
                equality_point: Point::low_to_high(equality_point.to_vec()),
            },
            invocation,
        }
    }

    fn prove(
        kernel: &mut dyn ProveRounds<F, AkitaError>,
        instance: &SingleInstance<F>,
        prover: &mut TestProver,
    ) -> Result<SingleReduction<F>, AkitaError> {
        prove_single::<F, F, AkitaError, _>(kernel, instance, prover)
    }

    fn verify(
        instance: &SingleInstance<F>,
        verifier: &mut TestVerifier<'_>,
    ) -> Result<SingleReduction<F>, AkitaError> {
        verify_single::<F, F, AkitaError, _>(instance, verifier)
    }

    /// `q(X) = Σ coefficients[k] X^k` behind the factor `eq(τ, X)`.
    struct OneRoundEq {
        tau: F,
        coefficients: Vec<F>,
    }

    impl OneRoundEq {
        fn evaluate(&self, point: F) -> F {
            UnivariatePoly::new(self.coefficients.clone()).evaluate(point)
        }

        fn claim(&self) -> F {
            (F::one() - self.tau) * self.evaluate(F::zero()) + self.tau * self.evaluate(F::one())
        }

        fn degree(&self) -> usize {
            self.coefficients.len() - 1
        }
    }

    impl EqFactoredSumcheckKernel<F> for OneRoundEq {
        fn num_rounds(&self) -> usize {
            1
        }

        fn round_polynomial(
            &mut self,
            _round: usize,
            _claim: F,
        ) -> Result<OmittedConstantPoly<F>, AkitaError> {
            Ok(OmittedConstantPoly::from_q_coefficients(
                self.coefficients.clone(),
            ))
        }

        fn bind_challenge(&mut self, _round: usize, _challenge: F) -> Result<(), AkitaError> {
            Ok(())
        }
    }

    /// The bilinear `a + b x + c y + d x y` behind `eq(τ, (x, y))`.
    struct TwoRoundEq {
        equality: [F; 2],
        coefficients: [F; 4],
        first_challenge: Option<F>,
    }

    impl TwoRoundEq {
        fn new(equality: [F; 2], coefficients: [F; 4]) -> Self {
            Self {
                equality,
                coefficients,
                first_challenge: None,
            }
        }

        fn evaluate(&self, x: F, y: F) -> F {
            let [a, b, c, d] = self.coefficients;
            a + b * x + c * y + d * x * y
        }

        fn input_claim(&self) -> F {
            self.evaluate(self.equality[0], self.equality[1])
        }
    }

    impl EqFactoredSumcheckKernel<F> for TwoRoundEq {
        fn num_rounds(&self) -> usize {
            2
        }

        fn round_polynomial(
            &mut self,
            round: usize,
            _claim: F,
        ) -> Result<OmittedConstantPoly<F>, AkitaError> {
            let [a, b, c, d] = self.coefficients;
            let coefficients = match self.first_challenge {
                None => vec![a + c * self.equality[1], b + d * self.equality[1]],
                Some(first) if round == 1 => vec![a + b * first, c + d * first],
                Some(_) => return Err(AkitaError::Internal("unexpected round".into())),
            };
            Ok(OmittedConstantPoly::from_q_coefficients(coefficients))
        }

        fn bind_challenge(&mut self, round: usize, challenge: F) -> Result<(), AkitaError> {
            if round == 0 {
                self.first_challenge = Some(challenge);
            }
            Ok(())
        }
    }

    /// Fails its bind of round `fail_at - 1`, as a stale backend session would.
    struct FailingKernel {
        inner: DenseKernel,
        fail_at: usize,
    }

    impl SumcheckKernel<F> for FailingKernel {
        fn num_rounds(&self) -> usize {
            self.inner.num_rounds()
        }

        fn round_polynomial(
            &mut self,
            round: usize,
            claim: F,
        ) -> Result<UnivariatePoly<F>, AkitaError> {
            self.inner.round_polynomial(round, claim)
        }

        fn bind_challenge(&mut self, round: usize, challenge: F) -> Result<(), AkitaError> {
            if round + 1 == self.fail_at {
                return Err(AkitaError::InvalidInput("stale kernel session".into()));
            }
            self.inner.bind_challenge(round, challenge)
        }
    }

    #[test]
    fn kernel_error_reaches_the_caller_unchanged() {
        let (evaluations, claim) = fixture();
        for fail_at in 1..=4 {
            let mut prover = TestProver::new(b"native-sumcheck/kernel-error", None);
            let mut kernel = FailingKernel {
                inner: DenseKernel::new(evaluations.clone()),
                fail_at,
            };
            assert_eq!(
                prove(
                    &mut FusedRounds(&mut kernel),
                    &compressed(claim, 4, 7),
                    &mut prover
                ),
                Err(AkitaError::InvalidInput("stale kernel session".into())),
                "failure in bind {fail_at}"
            );
        }
    }

    #[test]
    fn prover_rejects_round_polynomial_that_contradicts_the_claim() {
        let (evaluations, claim) = fixture();
        let mut prover = TestProver::new(b"native-sumcheck/fixture", None);
        assert!(matches!(
            prove(
                &mut FusedRounds(&mut DenseKernel::new(evaluations)),
                &compressed(claim + F::one(), 4, 7),
                &mut prover,
            ),
            Err(AkitaError::Internal(_))
        ));
    }

    #[test]
    fn roundtrip_consumes_the_argument_and_reaches_the_oracle() {
        let (evaluations, claim) = fixture();
        let instance = compressed(claim, 4, 7);
        let mut prover = TestProver::new(b"native-sumcheck/fixture", None);
        let proved = prove(
            &mut FusedRounds(&mut DenseKernel::new(evaluations.clone())),
            &instance,
            &mut prover,
        )
        .unwrap();
        let proof = prover.state.finish();

        let mut verifier = TestVerifier::new(b"native-sumcheck/fixture", &proof, None);
        let replay = verify(&instance, &mut verifier).unwrap();
        assert_eq!(replay, proved);
        assert_eq!(
            replay.output_claim,
            multilinear_eval(&evaluations, &replay.challenges).unwrap()
        );
        verifier.state.finish().unwrap();

        let mut truncated =
            TestVerifier::new(b"native-sumcheck/fixture", &proof[..proof.len() - 1], None);
        assert_eq!(
            verify(&instance, &mut truncated),
            Err(AkitaError::InvalidProof)
        );
    }

    #[test]
    fn eq_factored_roundtrip_reaches_the_oracle() {
        let mut kernel = OneRoundEq {
            tau: F::from_u64(7),
            coefficients: vec![F::from_u64(3), F::from_u64(5), F::from_u64(11)],
        };
        let instance = eq_factored(kernel.claim(), &[kernel.tau], kernel.degree(), 12);
        let mut prover = TestProver::new(b"native-eq-sumcheck/fixture", None);
        let proved = prove(
            &mut FusedEqFactoredRounds::new(&mut kernel, &instance).unwrap(),
            &instance,
            &mut prover,
        )
        .unwrap();
        let proof = prover.state.finish();

        let mut verifier = TestVerifier::new(b"native-eq-sumcheck/fixture", &proof, None);
        let replay = verify(&instance, &mut verifier).unwrap();
        assert_eq!(replay, proved);
        assert_eq!(replay.output_claim, kernel.evaluate(replay.challenges[0]));
        verifier.state.finish().unwrap();
    }

    #[test]
    fn eq_factored_rejects_old_wire_forgery_when_tau_is_zero() {
        let kernel = OneRoundEq {
            tau: F::zero(),
            coefficients: vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)],
        };
        let challenge = F::from_u64(11);
        let mut forged = ProverTranscript::<H>::new(&PROTOCOL, b"native-eq-forgery/fixture");
        forged.send(&kernel.claim());
        forged.send(&F::from_u64(101));
        let proof = forged.finish();
        let mut verifier =
            TestVerifier::new(b"native-eq-forgery/fixture", &proof, Some(vec![challenge]));
        let replay = verify(
            &eq_factored(kernel.claim(), &[F::zero()], kernel.degree(), 19),
            &mut verifier,
        )
        .unwrap();
        assert_ne!(replay.output_claim, kernel.evaluate(challenge));
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
        let mut kernel = TwoRoundEq::new(equality, coefficients);
        let instance = eq_factored(kernel.input_claim(), &equality, 1, 23);
        let mut prover = TestProver::new(b"native-eq-late/fixture", Some(point.to_vec()));
        prove(
            &mut FusedEqFactoredRounds::new(&mut kernel, &instance).unwrap(),
            &instance,
            &mut prover,
        )
        .unwrap();
        let honest = prover.state.finish();
        let expected = kernel.evaluate(point[0], point[1]);
        let replay = |proof: &[u8]| {
            let mut verifier =
                TestVerifier::new(b"native-eq-late/fixture", proof, Some(point.to_vec()));
            verify(&instance, &mut verifier).unwrap()
        };
        let accepted = replay(&honest);
        assert_eq!(accepted.challenges, point.to_vec());
        assert_eq!(accepted.output_claim, expected);

        let mut tampered = honest;
        let second = F::from_u64(11) + F::from_u64(13) * point[0] + F::one();
        tampered[F::NUM_BYTES..2 * F::NUM_BYTES].copy_from_slice(&second.to_bytes_le_vec());
        assert_ne!(replay(&tampered).output_claim, expected);
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

        // A snapshot taken by Akita's own driver at the spongefish-state port,
        // not independent ground truth: it fails on any change to the argument
        // string or the challenges, which is a proof-format change.
        let (evaluations, claim) = fixture();
        let mut prover = TestProver::new(b"native-cutover/standard", None);
        let standard = prove(
            &mut FusedRounds(&mut DenseKernel::new(evaluations)),
            &compressed(claim, 4, 3),
            &mut prover,
        )
        .unwrap();
        let standard_proof = hex(prover.state.narg());
        let standard_point = challenge_hex(&standard.challenges);

        let equality = [F::from_u64(2), F::from_u64(5)];
        let mut normalized = TwoRoundEq::new(
            equality,
            [
                F::from_u64(3),
                F::from_u64(7),
                F::from_u64(11),
                F::from_u64(13),
            ],
        );
        let instance = eq_factored(normalized.input_claim(), &equality, 1, 5);
        let mut prover = TestProver::new(b"native-cutover/normalized", None);
        let normalized = prove(
            &mut FusedEqFactoredRounds::new(&mut normalized, &instance).unwrap(),
            &instance,
            &mut prover,
        )
        .unwrap();
        let normalized_proof = hex(prover.state.narg());
        let normalized_point = challenge_hex(&normalized.challenges);

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
