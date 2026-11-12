//! A product of twenty fixed cubics, weighted by equality at a fixed point.

use akita_error::AkitaError;
use akita_sumcheck::{
    EqFactoredSumcheckKernel, SumcheckProverChannel, SumcheckRole, SumcheckVerifierChannel,
};
use akita_transcript::{
    ext_challenge, ProtocolSiteId, ProverChannel, VerifierChannel, SITE_FAMILY_SUMCHECK,
};
use jolt_field::{ExtField, Field, Prime128Offset6A23801F as F};
use jolt_poly::OmittedConstantPoly;
use rand::{rngs::StdRng, SeedableRng};

pub(super) const ROUNDS: usize = 20;
pub(super) const WIDTH: usize = 3;
pub(super) const SESSION: &[u8] = b"akita-ea0b-clear-driver-v1";

#[derive(Clone)]
pub(super) struct Instance<E: Field> {
    pub(super) tau: [E; ROUNDS],
    factors: [[E; WIDTH + 1]; ROUNDS],
    suffix: [E; ROUNDS + 1],
    prefix: E,
    round: usize,
}

fn evaluate<E: Field>(coefficients: &[E; WIDTH + 1], x: E) -> E {
    coefficients
        .iter()
        .rev()
        .fold(E::zero(), |value, &coefficient| value * x + coefficient)
}

impl<E: ExtField<F>> Instance<E> {
    pub(super) fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(0x00ea_0b20);
        let tau = std::array::from_fn(|_| E::lift_base(F::random(&mut rng)));
        let factors: [[E; WIDTH + 1]; ROUNDS] =
            std::array::from_fn(|_| std::array::from_fn(|_| E::lift_base(F::random(&mut rng))));
        let mut suffix = [E::one(); ROUNDS + 1];
        for round in (0..ROUNDS).rev() {
            let at_zero = factors[round][0];
            let at_one: E = factors[round].iter().copied().sum();
            suffix[round] = suffix[round + 1] * (at_zero + tau[round] * (at_one - at_zero));
        }
        Self {
            tau,
            factors,
            suffix,
            prefix: E::one(),
            round: 0,
        }
    }

    pub(super) fn claim(&self) -> E {
        self.suffix[0]
    }

    pub(super) fn expected(&self, challenges: &[E]) -> E {
        assert_eq!(challenges.len(), ROUNDS);
        self.factors
            .iter()
            .zip(challenges)
            .map(|(q, &r)| evaluate(q, r))
            .product()
    }
}

impl<E: ExtField<F>> EqFactoredSumcheckKernel<E> for Instance<E> {
    fn num_rounds(&self) -> usize {
        ROUNDS
    }
    fn degree_bound(&self) -> usize {
        WIDTH
    }
    fn input_claim(&self) -> E {
        self.claim()
    }
    fn current_tau(&self) -> E {
        self.tau[self.round]
    }
    fn round_polynomial(
        &mut self,
        round: usize,
        _claim: E,
    ) -> Result<OmittedConstantPoly<E>, AkitaError> {
        assert_eq!(round, self.round);
        let scale = self.prefix * self.suffix[round + 1];
        // Construct the native omitted vector directly, at the fixed public width.
        Ok(OmittedConstantPoly::new(
            self.factors[round][1..]
                .iter()
                .map(|&x| scale * x)
                .collect(),
        ))
    }
    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        assert_eq!(round, self.round);
        self.prefix *= evaluate(&self.factors[round], challenge);
        self.round += 1;
        Ok(())
    }
    fn finish(&mut self) -> Result<(), AkitaError> {
        assert_eq!(self.round, ROUNDS);
        Ok(())
    }
}

fn site(invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
    ProtocolSiteId {
        family: SITE_FAMILY_SUMCHECK,
        invocation,
        round,
        detail: role as u32,
        ..ProtocolSiteId::default()
    }
}

pub(super) struct Prover(pub(super) ProverChannel);
impl<E: ExtField<F>> SumcheckProverChannel<E> for Prover {
    fn state_mut(&mut self) -> &mut ProverChannel {
        &mut self.0
    }
    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
        site(invocation, round, role)
    }
    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError> {
        ext_challenge::<F, _>(
            &mut self.0,
            site(invocation, round, SumcheckRole::Challenge),
        )
        .map(E::lift_base)
    }
}

pub(super) struct Verifier<'a>(pub(super) VerifierChannel<'a>);
impl<'a, E: ExtField<F>> SumcheckVerifierChannel<'a, E> for Verifier<'a> {
    fn state_mut(&mut self) -> &mut VerifierChannel<'a> {
        &mut self.0
    }
    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
        site(invocation, round, role)
    }
    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError> {
        ext_challenge::<F, _>(
            &mut self.0,
            site(invocation, round, SumcheckRole::Challenge),
        )
        .map(E::lift_base)
    }
}
