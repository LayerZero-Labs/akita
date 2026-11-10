//! Fixed-seed first-round baselines. Fixture preparation is outside timing.
//!
//! The module declarations below compile the existing private CPU kernels with
//! their original ancestor paths. No production visibility or implementation changes.

#[path = "../src/arithmetic/mod.rs"]
#[allow(dead_code, unused_imports, unused_macros, unreachable_pub)]
pub(crate) mod arithmetic;
#[path = "range_sumcheck_rounds/cache/mod.rs"]
#[allow(dead_code, unused_imports, unused_macros, unreachable_pub)]
mod cache;
#[path = "../src/commitment/mod.rs"]
#[allow(dead_code, unused_imports, unused_macros, unreachable_pub)]
pub(crate) mod commitment;
#[path = "../src/kernels/mod.rs"]
#[allow(dead_code, unused_imports, unused_macros, unreachable_pub)]
pub(crate) mod kernels;
#[path = "../src/opaque/mod.rs"]
#[allow(dead_code, unused_imports, unused_macros, unreachable_pub)]
pub(crate) mod opaque;
#[path = "../src/portable_witness.rs"]
#[allow(dead_code, unused_imports, unused_macros, unreachable_pub)]
mod portable_witness;
#[path = "../src/setup/mod.rs"]
#[allow(dead_code, unused_imports, unused_macros, unreachable_pub)]
pub(crate) mod setup;
#[path = "../src/sources/mod.rs"]
#[allow(dead_code, unused_imports, unused_macros, unreachable_pub)]
pub(crate) mod sources;
#[path = "../src/validation.rs"]
#[allow(dead_code, unused_imports, unused_macros, unreachable_pub)]
mod validation;
pub use portable_witness::{CompressionWitnessFragment, PortableWitnessPatch};

use akita_algebra::CyclotomicRing;
use akita_error::{checked, AkitaError};
use akita_types::RingVec;
use jolt_field::Field;

pub(crate) use akita_prover::protocol;
pub use commitment::{
    GroupContext, PortableCompressionOutput, PortableCompressionState, SetupPrefixProverRegistry,
    SetupPrefixSlot,
};
pub use opaque::standalone;
pub use opaque::{
    evaluate_root_polynomial, CpuBackend, CpuExportPacket, CpuImportPacket, CpuPacketDescriptor,
    CpuPacketSections, PreparedCrtNttProfile, PreparedNttCacheMetric, RootPolyMeta, RootPolyShape,
    RootPolynomialEvaluator, SuccessorEncoding, SuccessorSection, SuccessorSectionDescriptor,
};
pub use opaque::{CommitOutput, CommitmentHandle, CpuSource, SourceHandle};
pub use setup::AkitaProverSetup;
pub use sources::{DensePoly, OneHotIndex, OneHotPoly, OneHotSource};

// Root imports required by the existing source modules' cfg(test) helpers.
#[cfg(test)]
#[allow(unused_imports)]
mod custom_source {
    pub(crate) use crate::sources::poly::SourceCoefficients;
    pub(crate) use crate::sources::signed_blocks::SignedBlockPoly;
}

/// Prover-side output of the inner Ajtai commit step.
///
/// Ring dimension is stored by the single flat A-ring buffer. Public commit
/// parameters own the source-block and row boundaries.
#[allow(dead_code)]
pub(crate) fn typed_inner_rows<F: Field, const D: usize>(
    recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
) -> Result<RingVec<F>, AkitaError> {
    let row_count = checked::sum(recomposed_inner_rows.iter().map(Vec::len))
        .ok_or_else(|| AkitaError::Internal("inner commitment row count overflow".into()))?;
    let coefficient_count = checked::product([row_count, D]).ok_or_else(|| {
        AkitaError::Internal("inner commitment coefficient count overflow".into())
    })?;
    let mut coefficients = Vec::with_capacity(coefficient_count);
    for block in recomposed_inner_rows {
        for row in block {
            coefficients.extend_from_slice(row.coefficients());
        }
    }
    RingVec::from_coeffs_with_ring_dim(coefficients, D)
}

#[cfg(test)]
#[path = "../src/tests/mod.rs"]
#[allow(dead_code, unused_imports, unused_macros, unreachable_pub)]
mod tests;

use akita_algebra::eq_poly::EqPolynomial;
use akita_params::DigitRangePlan;
use akita_sumcheck::{EqFactoredSumcheckInstanceProver, SumcheckInstanceProver};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use jolt_field::{One, Prime128OffsetA7F7 as F, Ring, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;

use opaque::sumcheck::digit_range::LowBasisRangeCheckProver;
use opaque::sumcheck::relation_range_image::{
    DenseRelationWeights, PreparedProverLinearTerms, RelationRangeImageProver, RelationWeightOracle,
};
use sources::packed_digits::PackedSignedDigits;

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
