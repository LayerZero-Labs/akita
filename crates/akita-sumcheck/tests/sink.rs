#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

use akita_algebra::poly::fold_evals_in_place;
use akita_algebra::split_eq::GruenSplitEq;
use akita_field::FieldCore;
use akita_field::Prime128Offset275;
use akita_sumcheck::{
    prove_clear_eq_factored, prove_clear_regular, EqFactoredSumcheckInstanceProver,
    EqFactoredSumcheckInstanceProverExt, EqFactoredUniPoly, SumcheckInstanceProver,
    SumcheckInstanceProverExt, UniPoly,
};
use akita_transcript::labels as tr_labels;
use akita_transcript::{AkitaTranscript, Transcript};

type F = Prime128Offset275;

fn new_transcript() -> AkitaTranscript<F> {
    <AkitaTranscript<F> as Transcript<F>>::new(tr_labels::DOMAIN_AKITA_PROTOCOL)
}

fn sample_round(tr: &mut AkitaTranscript<F>) -> F {
    tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND)
}

struct DenseSumcheckProver {
    evals: Vec<F>,
}

impl SumcheckInstanceProver<F> for DenseSumcheckProver {
    fn num_rounds(&self) -> usize {
        self.evals.len().ilog2() as usize
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn input_claim(&self) -> F {
        self.evals.iter().copied().fold(F::zero(), |a, b| a + b)
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        let half = self.evals.len() / 2;
        let mut eval_0 = F::zero();
        let mut eval_1 = F::zero();
        for i in 0..half {
            eval_0 += self.evals[2 * i];
            eval_1 += self.evals[2 * i + 1];
        }
        UniPoly::from_coeffs(vec![eval_0, eval_1 - eval_0])
    }

    fn ingest_challenge(&mut self, _round: usize, r: F) {
        fold_evals_in_place(&mut self.evals, r);
    }
}

#[test]
fn clear_regular_sink_matches_driver_extension() {
    let evals: Vec<F> = (0..8).map(F::from_u64).collect();
    let mut via_sink = DenseSumcheckProver {
        evals: evals.clone(),
    };
    let mut via_driver = DenseSumcheckProver { evals };

    let mut tr_sink = new_transcript();
    let (proof_sink, r_sink, claim_sink) =
        prove_clear_regular(&mut via_sink, &mut tr_sink, sample_round).unwrap();

    let mut tr_driver = new_transcript();
    let (proof_driver, r_driver, claim_driver) = via_driver
        .prove::<F, _, _>(&mut tr_driver, sample_round)
        .unwrap();

    assert_eq!(proof_sink, proof_driver);
    assert_eq!(r_sink, r_driver);
    assert_eq!(claim_sink, claim_driver);
}

struct ToyEqFactoredInstance {
    tau: F,
    split_eq: GruenSplitEq<F>,
    q_coeffs: Vec<F>,
}

impl ToyEqFactoredInstance {
    fn new(tau: F, q_coeffs: Vec<F>) -> Self {
        Self {
            tau,
            split_eq: GruenSplitEq::new(&[tau]).unwrap(),
            q_coeffs,
        }
    }

    fn input_claim_from_tau(&self) -> F {
        let g = GruenSplitEq::new(&[self.tau])
            .unwrap()
            .gruen_mul(&UniPoly::from_coeffs(self.q_coeffs.clone()));
        g.evaluate(&F::zero()) + g.evaluate(&F::one())
    }
}

impl EqFactoredSumcheckInstanceProver<F> for ToyEqFactoredInstance {
    fn num_rounds(&self) -> usize {
        1
    }

    fn degree_bound(&self) -> usize {
        self.q_coeffs.len().saturating_sub(1)
    }

    fn input_claim(&self) -> F {
        self.input_claim_from_tau()
    }

    fn current_linear_factor_evals(&self) -> (F, F) {
        self.split_eq.linear_factor_evals()
    }

    fn compute_round_eq_factored(&mut self, _round: usize) -> EqFactoredUniPoly<F> {
        EqFactoredUniPoly::from_q_coeffs(self.q_coeffs.clone())
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: F) {
        self.split_eq.bind(r_round);
    }
}

#[test]
fn clear_eq_factored_sink_matches_driver_extension() {
    let tau = F::from_u64(17);
    let q_coeffs = vec![
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
    ];

    let mut via_sink = ToyEqFactoredInstance::new(tau, q_coeffs.clone());
    let mut via_driver = ToyEqFactoredInstance::new(tau, q_coeffs);

    let mut tr_sink = new_transcript();
    let (proof_sink, r_sink, claim_sink) =
        prove_clear_eq_factored(&mut via_sink, &mut tr_sink, sample_round).unwrap();

    let mut tr_driver = new_transcript();
    let (proof_driver, r_driver, claim_driver) = via_driver
        .prove::<F, _, _>(&mut tr_driver, sample_round)
        .unwrap();

    assert_eq!(proof_sink, proof_driver);
    assert_eq!(r_sink, r_driver);
    assert_eq!(claim_sink, claim_driver);
}
