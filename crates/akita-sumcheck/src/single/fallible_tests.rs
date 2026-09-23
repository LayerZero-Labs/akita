use super::*;
use akita_algebra::uni_poly::UniPoly;
use akita_transcript::AkitaTranscript;
use jolt_field::{One, Prime128Offset275 as F, Zero};

#[derive(Clone, Copy, PartialEq)]
enum Failure {
    Round,
    Bind,
    Finish,
    Claim,
    Degree,
}

struct RejectingKernel {
    failure: Failure,
    calls: Vec<&'static str>,
}
impl crate::SumcheckKernel<F> for RejectingKernel {
    fn num_rounds(&self) -> usize {
        1
    }
    fn degree_bound(&self) -> usize {
        1
    }
    fn input_claim(&self) -> F {
        F::one()
    }
    fn round_polynomial(&mut self, _: usize, _: F) -> Result<UniPoly<F>, AkitaError> {
        self.calls.push("round");
        match self.failure {
            Failure::Round => Err(AkitaError::InvalidInput("round failure".into())),
            Failure::Claim => Ok(UniPoly::from_coeffs(vec![F::zero()])),
            Failure::Degree => Ok(UniPoly::from_coeffs(vec![F::zero(), F::zero(), F::one()])),
            _ => Ok(UniPoly::from_coeffs(vec![F::zero(), F::one()])),
        }
    }
    fn bind_challenge(&mut self, _: usize, _: F) -> Result<(), AkitaError> {
        self.calls.push("bind");
        if self.failure == Failure::Bind {
            return Err(AkitaError::InvalidInput("bind failure".into()));
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<(), AkitaError> {
        self.calls.push("finish");
        Err(AkitaError::InvalidInput("finish failure".into()))
    }
}

#[test]
fn kernel_failures_propagate_without_fabricating_rounds_or_advancing_after_failure() {
    for (failure, expected_calls, expected_samples, message) in [
        (Failure::Round, vec!["round"], 0, "round failure"),
        (Failure::Bind, vec!["round", "bind"], 1, "bind failure"),
        (
            Failure::Finish,
            vec!["round", "bind", "finish"],
            1,
            "finish failure",
        ),
        (
            Failure::Claim,
            vec!["round"],
            0,
            "sumcheck round polynomial does not match its input claim",
        ),
        (
            Failure::Degree,
            vec!["round"],
            0,
            "sumcheck round poly degree 2 exceeds bound 1",
        ),
    ] {
        let mut kernel = RejectingKernel {
            failure,
            calls: Vec::new(),
        };
        let mut transcript = AkitaTranscript::<F>::prover(b"fallible-sumcheck", b"fixture");
        let mut samples = 0;
        let result = prove_sumcheck::<F, _, F, _, _>(&mut kernel, &mut transcript, |_| {
            samples += 1;
            Ok(F::one())
        });
        assert!(matches!(result, Err(AkitaError::InvalidInput(ref actual)) if actual == message));
        assert_eq!(kernel.calls, expected_calls);
        assert_eq!(samples, expected_samples);
    }
}
