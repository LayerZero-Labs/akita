use akita_field::Prime128Offset275;
use akita_prover::DigitRangeProver;
use akita_types::{DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain};
use std::sync::Arc;

pub(crate) type BenchmarkField = Prime128Offset275;

const NUM_VARIABLES: usize = 18;
const LOW_VARIABLE_COUNT: usize = 6;

#[derive(Clone, Copy)]
pub(crate) enum DigitDistribution {
    Uniform,
    ZeroHeavy,
    AlternatingEndpoints,
}

impl DigitDistribution {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::ZeroHeavy => "zero-heavy",
            Self::AlternatingEndpoints => "alternating-endpoints",
        }
    }

    fn witness(self, basis: usize, live_len: usize) -> Arc<[i8]> {
        let half = i16::try_from(basis / 2).expect("supported basis fits i16");
        (0..live_len)
            .map(|index| {
                let digit = match self {
                    Self::Uniform => i16::try_from(index % basis).expect("basis fits i16") - half,
                    Self::ZeroHeavy => {
                        if index % 16 == 0 {
                            half - 1
                        } else {
                            0
                        }
                    }
                    Self::AlternatingEndpoints => {
                        if index & 1 == 0 {
                            -half
                        } else {
                            half - 1
                        }
                    }
                };
                i8::try_from(digit).expect("supported balanced digit fits i8")
            })
            .collect::<Vec<_>>()
            .into()
    }
}

#[derive(Clone)]
pub(crate) struct BenchmarkCase {
    pub(crate) plan: DigitRangePlan,
    pub(crate) domain: FlatBooleanDomain,
    pub(crate) equality_point: DigitRangeEqualityPoint<BenchmarkField>,
    digit_witness: Arc<[i8]>,
}

pub(crate) struct ProverInput {
    plan: DigitRangePlan,
    domain: FlatBooleanDomain,
    equality_point: DigitRangeEqualityPoint<BenchmarkField>,
    digit_witness: Arc<[i8]>,
}

impl ProverInput {
    pub(crate) fn build(self) -> DigitRangeProver<BenchmarkField> {
        DigitRangeProver::new(
            self.digit_witness,
            self.plan,
            self.domain,
            self.equality_point,
        )
        .expect("benchmark prover")
    }
}

impl BenchmarkCase {
    pub(crate) fn new(
        basis: usize,
        live_numerator: usize,
        distribution: DigitDistribution,
    ) -> Self {
        let domain_len = 1usize << NUM_VARIABLES;
        let live_len = domain_len * live_numerator / 4;
        let raw_challenges = (0..NUM_VARIABLES)
            .map(|index| BenchmarkField::from_u64(u64::try_from(index + 2).expect("small index")))
            .collect::<Vec<_>>();
        let high_variable_count = NUM_VARIABLES - LOW_VARIABLE_COUNT;
        let equality_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
            &raw_challenges,
            high_variable_count,
            LOW_VARIABLE_COUNT,
        )
        .expect("benchmark point");
        Self {
            plan: DigitRangePlan::new(basis).expect("supported benchmark basis"),
            domain: FlatBooleanDomain::new(live_len, NUM_VARIABLES)
                .expect("aligned benchmark domain"),
            equality_point,
            digit_witness: distribution.witness(basis, live_len),
        }
    }

    pub(crate) fn prover_input(&self) -> ProverInput {
        ProverInput {
            plan: self.plan,
            domain: self.domain,
            equality_point: self.equality_point.clone(),
            digit_witness: Arc::clone(&self.digit_witness),
        }
    }
}
