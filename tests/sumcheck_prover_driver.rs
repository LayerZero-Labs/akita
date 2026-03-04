#![allow(missing_docs)]

use hachi_pcs::algebra::Fp64;
use hachi_pcs::protocol::transcript::labels;
use hachi_pcs::protocol::{
    prove_sumcheck, Blake2bTranscript, SumcheckInstanceProver, Transcript, UniPoly,
};
use hachi_pcs::{FieldCore, FieldSampling};
use rand::rngs::StdRng;
use rand::SeedableRng;

type F = Fp64<4294967197>;

/// A tiny prover-side sumcheck instance for a multilinear function in evaluation-table form.
///
/// Variable order convention: the current round binds the least-significant index bit first,
/// i.e. pairs are `(i<<1)|0` and `(i<<1)|1` (matches the common LSB-first sumcheck table fold).
struct DenseTableSumcheck {
    table: Vec<F>,
}

impl DenseTableSumcheck {
    fn new(table: Vec<F>) -> Self {
        assert!(table.len().is_power_of_two());
        Self { table }
    }
}

impl SumcheckInstanceProver<F> for DenseTableSumcheck {
    fn num_rounds(&self) -> usize {
        self.table.len().trailing_zeros() as usize
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn input_claim(&self) -> F {
        self.table.iter().copied().fold(F::zero(), |a, b| a + b)
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        let half = self.table.len() / 2;
        let mut s0 = F::zero();
        let mut s1 = F::zero();
        for i in 0..half {
            s0 += self.table[i << 1];
            s1 += self.table[(i << 1) | 1];
        }
        UniPoly::from_coeffs(vec![s0, s1 - s0])
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: F) {
        let half = self.table.len() / 2;
        let mut next = Vec::with_capacity(half);
        let one_minus = F::one() - r_round;
        for i in 0..half {
            let v0 = self.table[i << 1];
            let v1 = self.table[(i << 1) | 1];
            next.push(one_minus * v0 + r_round * v1);
        }
        self.table = next;
    }
}

#[test]
fn prover_driver_produces_proof_that_verifier_replays() {
    let mut rng = StdRng::seed_from_u64(2026);
    let num_rounds = 8usize;
    let n = 1usize << num_rounds;

    let table: Vec<F> = (0..n).map(|_| F::sample(&mut rng)).collect();
    let mut prover_inst = DenseTableSumcheck::new(table.clone());
    let mut prover_t = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let (proof, r_vec, final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover_inst, &mut prover_t, |tr| {
            tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .unwrap();

    // After folding all variables, the table should be a single value equal to f(r*).
    assert_eq!(prover_inst.table.len(), 1);
    assert_eq!(final_claim, prover_inst.table[0]);

    // Verifier replay must derive the same (final_claim, r_vec).
    let initial_claim = table.iter().copied().fold(F::zero(), |acc, x| acc + x);
    let mut verifier_t = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    verifier_t.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &initial_claim);
    let (final_claim_v, r_vec_v) = proof
        .verify::<F, _, _>(initial_claim, num_rounds, 1, &mut verifier_t, |tr| {
            tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .unwrap();

    assert_eq!(r_vec_v, r_vec);
    assert_eq!(final_claim_v, final_claim);
}
