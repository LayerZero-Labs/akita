#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use hachi_pcs::algebra::Fp128;
use hachi_pcs::protocol::sumcheck::norm_sumcheck::NormSumcheckProver;
use hachi_pcs::protocol::sumcheck::split_eq::GruenSplitEq;
use hachi_pcs::protocol::sumcheck::{
    fold_evals_in_place, prove_sumcheck, range_check_eval, SumcheckInstanceProver, UniPoly,
};
use hachi_pcs::protocol::transcript::labels;
use hachi_pcs::protocol::Blake2bTranscript;
use hachi_pcs::{FieldCore, FromSmallInt, Transcript};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::time::Duration;

type F = Fp128<0xfffffffffffffffffffffffffffffeed>;

/// Baseline prover keeps the pre-dispatch point-eval kernel for apples-to-apples benchmarks.
/// It is intentionally local to this bench and should not be used in production code.
struct BaselineNormSumcheckProver<E: FieldCore> {
    split_eq: GruenSplitEq<E>,
    w_table: Vec<E>,
    num_vars: usize,
    b: usize,
}

impl<E: FieldCore + FromSmallInt> BaselineNormSumcheckProver<E> {
    fn new(tau: &[E], w_evals: Vec<E>, b: usize) -> Self {
        let num_vars = tau.len();
        assert_eq!(w_evals.len(), 1 << num_vars);
        Self {
            split_eq: GruenSplitEq::new(tau),
            w_table: w_evals,
            num_vars,
            b,
        }
    }
}

impl<E: FieldCore + FromSmallInt> SumcheckInstanceProver<E> for BaselineNormSumcheckProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        2 * self.b
    }

    fn input_claim(&self) -> E {
        E::zero()
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let half = self.w_table.len() / 2;
        let degree_q = 2 * self.b - 1;
        let num_points_q = degree_q + 1;

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let b = self.b;

        #[cfg(feature = "parallel")]
        let q_evals = {
            (0..half)
                .into_par_iter()
                .fold(
                    || vec![E::zero(); num_points_q],
                    |mut evals, j| {
                        let j_low = j & (num_first - 1);
                        let j_high = j >> first_bits;
                        let eq_rem = e_first[j_low] * e_second[j_high];
                        let w_0 = self.w_table[2 * j];
                        let w_1 = self.w_table[2 * j + 1];
                        for (t, eval) in evals.iter_mut().enumerate() {
                            let t_e = E::from_u64(t as u64);
                            let w_t = w_0 + t_e * (w_1 - w_0);
                            *eval += eq_rem * range_check_eval(w_t, b);
                        }
                        evals
                    },
                )
                .reduce(
                    || vec![E::zero(); num_points_q],
                    |mut a, b_vec| {
                        for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                            *ai += *bi;
                        }
                        a
                    },
                )
        };
        #[cfg(not(feature = "parallel"))]
        let q_evals = {
            let mut q_evals = vec![E::zero(); num_points_q];
            for j in 0..half {
                let j_low = j & (num_first - 1);
                let j_high = j >> first_bits;
                let eq_rem = e_first[j_low] * e_second[j_high];
                let w_0 = self.w_table[2 * j];
                let w_1 = self.w_table[2 * j + 1];
                for (t, eval) in q_evals.iter_mut().enumerate() {
                    let t_e = E::from_u64(t as u64);
                    let w_t = w_0 + t_e * (w_1 - w_0);
                    *eval = *eval + eq_rem * range_check_eval(w_t, b);
                }
            }
            q_evals
        };

        let q_poly = UniPoly::from_evals(&q_evals);
        self.split_eq.gruen_mul(&q_poly)
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        self.split_eq.bind(r);
        fold_evals_in_place(&mut self.w_table, r);
    }
}

#[derive(Clone)]
struct NormCase {
    num_vars: usize,
    b: usize,
    tau: Vec<F>,
    w_evals: Vec<F>,
}

fn build_case(num_vars: usize, b: usize, seed: u64) -> NormCase {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = 1usize << num_vars;
    let tau: Vec<F> = (0..num_vars)
        .map(|_| F::from_u64(rng.gen_range(0u64..(1u64 << 24))))
        .collect();
    let w_evals: Vec<F> = (0..n)
        .map(|_| F::from_u64(rng.gen_range(0u64..(1u64 << 24))))
        .collect();
    NormCase {
        num_vars,
        b,
        tau,
        w_evals,
    }
}

fn bench_norm_sumcheck(c: &mut Criterion) {
    let cases = [
        build_case(10, 4, 0xA11CE001),
        build_case(10, 8, 0xA11CE002),
        build_case(14, 4, 0xA11CE003),
        build_case(14, 8, 0xA11CE004),
        build_case(14, 16, 0xA11CE005),
        build_case(18, 8, 0xA11CE006),
    ];

    let mut group = c.benchmark_group("norm_sumcheck");
    group.warm_up_time(Duration::from_secs(8));
    group.measurement_time(Duration::from_secs(24));
    group.sample_size(35);

    for case in &cases {
        let case_tag = format!("nv{}_b{}", case.num_vars, case.b);
        group.bench_function(BenchmarkId::new("baseline", &case_tag), |bencher| {
            bencher.iter_batched(
                || BaselineNormSumcheckProver::new(&case.tau, case.w_evals.clone(), case.b),
                |mut prover| {
                    let mut transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
                    black_box(
                        prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut transcript, |tr| {
                            tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
                        })
                        .unwrap(),
                    )
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function(BenchmarkId::new("dispatched", &case_tag), |bencher| {
            bencher.iter_batched(
                || NormSumcheckProver::new(&case.tau, case.w_evals.clone(), case.b),
                |mut prover| {
                    let mut transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
                    black_box(
                        prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut transcript, |tr| {
                            tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
                        })
                        .unwrap(),
                    )
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_norm_sumcheck);
criterion_main!(benches);
