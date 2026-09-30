//! Stage-1 range-check tree prover for the Akita PCS.
//!
//! For `b <= 8`, stage 1 is still a single eq-factored sumcheck over
//! `Q(range_image(z))`, where `range_image(z) = w(z)(w(z)+1)` and `Q` is the
//! full range polynomial.
//! For larger supported bases, stage 1 is written as a short root-to-leaf tree:
//!
//! - a root stage proves the product of `2` or `4` quartic leaf factors,
//! - the prover sends those child-node claims at the sampled root point,
//! - a leaf stage proves a random linear combination of the quartic factors
//!   directly from `range_image`.
//!
//! This matches the proof-size study's current tree cutover for `log_basis <= 6`
//! without widening the recursive witness encoding beyond the existing runtime
//! bound.

mod class_indexed_product;
pub(crate) mod class_indexed_range_leaf;
mod class_indexed_state;
mod compact_digit_source;
pub(crate) mod direct_range_leaf;
pub(crate) mod exact_prefix;
mod range_class_tables;
pub(crate) mod range_poly;
mod round_accumulation;
mod session;

pub use direct_range_leaf::LowBasisRangeCheckProver;
pub(crate) use session::DigitRangeSession;

use crate::sources::packed_digits::PackedSignedDigits;
use akita_error::AkitaError;
use akita_types::{DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain};
use class_indexed_product::ClassIndexedProductSubcheckProver;
use class_indexed_range_leaf::ClassIndexedRangeLeafProver;
use compact_digit_source::CompactDigitSource;
use jolt_field::{Field, Ring};

const MAX_TREE_STAGE_Q_DEGREE: usize = 4;
const MAX_QUARTET_TABLE_CLASS_COUNT: usize = 8;

struct ProductSubcheckInput<'a, E: Field> {
    source: CompactDigitSource,
    plan: DigitRangePlan,
    leaf_polynomials: &'a [Vec<E>],
    stage_index: usize,
    parent_weights: Vec<E>,
    equality_point: &'a [E],
    input_claim: E,
}

fn compose_small_poly_with_affine<E: Field>(coeffs: &[E], offset: E, slope: E) -> [E; 5] {
    debug_assert!(coeffs.len() <= MAX_TREE_STAGE_Q_DEGREE + 1);
    let [constant, linear, quadratic, cubic, quartic] = match coeffs {
        [] => return [E::zero(); 5],
        [c0] => return [*c0, E::zero(), E::zero(), E::zero(), E::zero()],
        [c0, c1] => {
            return [
                *c0 + *c1 * offset,
                *c1 * slope,
                E::zero(),
                E::zero(),
                E::zero(),
            ]
        }
        [c0, c1, c2] => [*c0, *c1, *c2, E::zero(), E::zero()],
        [c0, c1, c2, c3] => [*c0, *c1, *c2, *c3, E::zero()],
        [c0, c1, c2, c3, c4] => [*c0, *c1, *c2, *c3, *c4],
        _ => unreachable!("range polynomial degree is at most four"),
    };

    let two_quadratic = quadratic + quadratic;
    let three_cubic = cubic + cubic + cubic;
    let four_quartic = (quartic + quartic) + (quartic + quartic);
    let six_quartic = four_quartic + quartic + quartic;

    let value =
        constant + offset * (linear + offset * (quadratic + offset * (cubic + offset * quartic)));
    let first_derivative =
        linear + offset * (two_quadratic + offset * (three_cubic + offset * four_quartic));
    let second_divided_derivative = quadratic + offset * (three_cubic + offset * six_quartic);
    let third_divided_derivative = cubic + offset * four_quartic;
    let slope_squared = slope * slope;

    [
        value,
        slope * first_derivative,
        slope_squared * second_divided_derivative,
        slope_squared * slope * third_divided_derivative,
        slope_squared * slope_squared * quartic,
    ]
}

/// Stage-1 range-check prover, including the root/leaf tree choreography.
pub struct DigitRangeProver<E: Field> {
    digit_source: CompactDigitSource,
    equality_point: Vec<E>,
    plan: DigitRangePlan,
    live_block_count: usize,
    high_variable_count: usize,
    low_variable_count: usize,
}

impl<E: Field + Ring> DigitRangeProver<E> {
    /// Build a standalone prover from a compact digit witness.
    pub fn new(
        digit_witness: std::sync::Arc<[i8]>,
        plan: DigitRangePlan,
        domain: FlatBooleanDomain,
        equality_point: DigitRangeEqualityPoint<E>,
    ) -> Result<Self, AkitaError> {
        Self::from_packed_digits(
            PackedSignedDigits::from_i8_digits_auto(digit_witness.as_ref().to_vec()),
            plan,
            domain,
            equality_point,
        )
    }

    pub(crate) fn from_packed_digits(
        digit_witness: PackedSignedDigits,
        plan: DigitRangePlan,
        domain: FlatBooleanDomain,
        equality_point: DigitRangeEqualityPoint<E>,
    ) -> Result<Self, AkitaError> {
        equality_point.validate_domain(domain)?;
        let low_variable_count = equality_point.low_variable_count();
        let high_variable_count = domain.num_vars() - low_variable_count;
        let live_block_count = domain.live_block_count(low_variable_count)?;
        let coordinates = equality_point.into_coordinates();
        let digit_source = {
            let _span = tracing::info_span!(
                "digit_range_prepare_compact_source",
                basis = plan.basis(),
                live_len = domain.live_len(),
                domain_len = domain.domain_len(),
            )
            .entered();
            CompactDigitSource::new(digit_witness, domain, plan)?
        };
        Ok(Self {
            digit_source,
            equality_point: coordinates,
            plan,
            live_block_count,
            high_variable_count,
            low_variable_count,
        })
    }
}
