//! Fixed-seed first-round baselines. Fixture preparation is outside timing.

use akita_algebra::eq_poly::EqPolynomial;
use akita_cpu_backend::benchmark_support::{
    DenseRelationWeights, LowBasisRangeCheckProver, PackedSignedDigits, PreparedProverLinearTerms,
    RelationRangeImageProver, RelationWeightOracle,
};
use akita_params::DigitRangePlan;
use akita_sumcheck::{EqFactoredSumcheckInstanceProver, SumcheckInstanceProver};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use jolt_field::{Field, One, Prime128Offset6A23801F as F, Ring, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;

const LANE_BITS: usize = 6;
const COEFFICIENT_BITS: usize = 6;
const NUM_VARS: usize = LANE_BITS + COEFFICIENT_BITS;
const BASIS: usize = 8;

struct Fixture {
    digits: Vec<i8>,
    point: Vec<F>,
    common: Vec<F>,
    lanes: Vec<F>,
    range_claim: F,
    relation_claim: F,
}

impl Fixture {
    fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(0xea0b);
        let digits: Vec<i8> = (0..1 << NUM_VARS).map(|_| rng.gen_range(-4..4)).collect();
        let point: Vec<F> = (0..NUM_VARS).map(|_| F::random(&mut rng)).collect();
        let common: Vec<F> = (0..1 << COEFFICIENT_BITS)
            .map(|_| F::random(&mut rng))
            .collect();
        let lanes: Vec<F> = (0..1 << LANE_BITS).map(|_| F::random(&mut rng)).collect();
        let eq = EqPolynomial::evals(&point).unwrap();
        let mut range_claim = F::zero();
        let mut relation_claim = F::zero();
        for (index, &digit) in digits.iter().enumerate() {
            let w = F::from_i64(i64::from(digit));
            range_claim += eq[index] * w * (w + F::one());
            relation_claim += w * common[index % common.len()] * lanes[index / common.len()];
        }
        Self {
            digits,
            point,
            common,
            lanes,
            range_claim,
            relation_claim,
        }
    }

    fn digit(&self) -> LowBasisRangeCheckProver<F> {
        LowBasisRangeCheckProver::new(
            PackedSignedDigits::from_i8_digits_auto(self.digits.clone()),
            &self.point,
            DigitRangePlan::new(BASIS).unwrap(),
            1 << LANE_BITS,
            LANE_BITS,
            COEFFICIENT_BITS,
        )
        .unwrap()
    }

    fn relation(&self) -> RelationRangeImageProver<F> {
        RelationRangeImageProver::new(
            F::from_u64(7),
            PackedSignedDigits::from_i8_digits_auto(self.digits.clone()),
            &self.point,
            self.range_claim,
            BASIS,
            RelationWeightOracle::ReducedDense(
                DenseRelationWeights::new(
                    self.lanes
                        .iter()
                        .flat_map(|&lane| self.common.iter().map(move |&x| lane * x))
                        .collect(),
                    self.digits.len(),
                )
                .unwrap(),
            ),
            1 << LANE_BITS,
            LANE_BITS,
            COEFFICIENT_BITS,
            self.relation_claim,
            PreparedProverLinearTerms::zero(1 << LANE_BITS, 1 << COEFFICIENT_BITS),
            None,
        )
        .unwrap()
    }
}

fn rounds(c: &mut Criterion) {
    let fixture = Fixture::new();
    // Check each first-round identity before timing the arithmetic kernel.
    let mut digit = fixture.digit();
    let q = digit.compute_round_eq_factored(0, F::zero());
    assert_eq!(q.nonconstant_term_sum_at_one(), F::zero());
    let mut relation = fixture.relation();
    let claim = relation.input_claim();
    let p = relation.compute_round_univariate(0, claim);
    assert_eq!(p.evaluate(F::zero()) + p.evaluate(F::one()), claim);

    let mut group = c.benchmark_group("range_sumcheck_rounds");
    group.bench_function("digit_range/direct_leaf_round_0", |b| {
        b.iter_batched(
            || fixture.digit(),
            |mut prover| {
                let poly = prover.compute_round_eq_factored(0, F::zero());
                (prover, black_box(poly))
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("relation_range_image/round_0", |b| {
        b.iter_batched(
            || fixture.relation(),
            |mut prover| {
                let claim = prover.input_claim();
                let poly = prover.compute_round_univariate(0, claim);
                (prover, black_box(poly))
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, rounds);
criterion_main!(benches);
