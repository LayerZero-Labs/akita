//! Structured public weight families on one root's outgoing digit table.
//!
//! A family keeps the public inputs of one linear identity instead of one
//! term per table address. The verifier evaluates its multilinear extension
//! at the Stage 2 point through [`RootEq`]; the prover streams exactly the same
//! field elements into its dense Stage 2 table. Both read one construction,
//! so the two sides cannot disagree about a weight.

use akita_algebra::eq_poly::EqPolynomial;
use akita_error::{checked, AkitaError};
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::Field;

use super::profiles::ProfileBuilder;
use crate::binary_source::{
    DigitAuthenticationWeights, DirectAWeights, ImageWeights, ParityRowWeights, PrimePlaneWeights,
};
use crate::root_repacking::RootCoefficientRepacking;

/// Low-block width of the split equality table: one 1024-coefficient lane.
const LOW_BITS: usize = 10;
/// Bound on the high table, which has one entry per 1024 table addresses.
const MAX_HIGH_BITS: usize = 22;

/// Equality weights at one point, split as `eq(r,a) = low[a mod 2^b] * high[a >> b]`.
///
/// Every root coefficient lane has a power-of-two stride of at most 1024 and
/// starts at a multiple of that stride, so one lane never crosses a low block.
/// The tables hold `2^b + 2^(n-b)` elements rather than `2^n`.
pub(crate) struct RootEq<E: Field> {
    low_bits: usize,
    low: Vec<E>,
    high: Vec<E>,
}

impl<E: Field> RootEq<E> {
    pub(crate) fn new(point: &[E]) -> Result<Self, AkitaError> {
        let low_bits = point.len().min(LOW_BITS);
        let (low_point, high_point) = point.split_at(low_bits);
        if high_point.len() > MAX_HIGH_BITS {
            return Err(AkitaError::InvalidInput(
                "root weight domain exceeds the split equality bound".into(),
            ));
        }
        Ok(Self {
            low_bits,
            low: EqPolynomial::evals(low_point)?,
            high: EqPolynomial::evals(high_point)?,
        })
    }

    /// Equality weights of the low address bits.
    pub(crate) fn low(&self) -> &[E] {
        &self.low
    }

    /// Offset of `address` inside its low block.
    pub(crate) fn low_offset(&self, address: usize) -> usize {
        address & (self.low.len() - 1)
    }

    /// High equality factor of the block that contains `address`.
    pub(crate) fn high(&self, address: usize) -> Result<E, AkitaError> {
        self.high
            .get(address >> self.low_bits)
            .copied()
            .ok_or(AkitaError::InvalidProof)
    }
}

/// Lane addresses of one single-plane repacking segment.
///
/// A lane is one digit of one element: `stride` consecutive addresses whose
/// first `degree` hold natural coefficients and the rest are zero padding.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SegmentLanes {
    start: usize,
    stride: usize,
    degree: usize,
    digits: usize,
    elements: usize,
}

impl SegmentLanes {
    pub(crate) fn new(
        repacking: &RootCoefficientRepacking,
        segment: usize,
    ) -> Result<Self, AkitaError> {
        let shape = repacking
            .segments()
            .get(segment)
            .ok_or(AkitaError::InvalidProof)?;
        let start = repacking.segment_range(segment)?.start;
        let stride = shape.geometry.padded_coefficient_count();
        if shape.geometry.coordinate_plane_count() != 1
            || stride > 1 << LOW_BITS
            || !start.is_multiple_of(stride)
        {
            return Err(AkitaError::InvalidSetup(
                "structured root weights require aligned single-plane lanes".into(),
            ));
        }
        Ok(Self {
            start,
            stride,
            degree: shape.geometry.polynomial_modulus_dimension(),
            digits: shape.digits,
            elements: shape.elements,
        })
    }

    pub(crate) fn stride(&self) -> usize {
        self.stride
    }
    pub(crate) fn degree(&self) -> usize {
        self.degree
    }
    pub(crate) fn digits(&self) -> usize {
        self.digits
    }
    pub(crate) fn elements(&self) -> usize {
        self.elements
    }

    /// Address of natural coefficient zero in one lane.
    pub(crate) fn base(&self, element: usize, digit: usize) -> Result<usize, AkitaError> {
        if element >= self.elements || digit >= self.digits {
            return Err(AkitaError::InvalidInput(
                "root lane index is invalid".into(),
            ));
        }
        checked::mul_add(element, self.digits, digit)
            .and_then(|lane| checked::mul_add(lane, self.stride, self.start))
            .ok_or(AkitaError::InvalidProof)
    }

    /// Stream `8^d * value` onto natural coefficient `offset` of every digit
    /// lane `d` of `element`: the recomposed digits of one coefficient.
    pub(crate) fn recompose<E: Field>(
        &self,
        element: usize,
        offset: usize,
        value: E,
        visit: &mut impl FnMut(usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        if offset >= self.degree {
            return Err(AkitaError::InvalidProof);
        }
        let mut scale = value;
        for digit in 0..self.digits {
            visit(self.base(element, digit)? + offset, scale)?;
            scale *= E::from_u64(8);
        }
        Ok(())
    }

    /// `sum_d 8^d * high(lane d)` for one element whose lanes start a block.
    ///
    /// This is the digit recomposition of every coefficient of the element,
    /// contracted with the high equality factors. It requires each lane to
    /// start a low block, so that all lanes share the same low factors.
    pub(crate) fn block_digit_weight<E: Field>(
        &self,
        eq: &RootEq<E>,
        element: usize,
    ) -> Result<E, AkitaError> {
        let mut sum = E::zero();
        let mut power = E::one();
        for digit in 0..self.digits {
            let base = self.base(element, digit)?;
            if eq.low_offset(base) != 0 || eq.low().len() < self.stride {
                return Err(AkitaError::InvalidSetup(
                    "root lane does not start an equality block".into(),
                ));
            }
            sum += power * eq.high(base)?;
            power *= E::from_u64(8);
        }
        Ok(sum)
    }
}

/// Units a family weighs at once before streaming them; bounds the buffer.
pub(crate) const WEIGH_BATCH: usize = 4096;

/// Weigh the units in `range`, in order, in parallel under `parallel`.
///
/// Computing a unit's weights dominates a family's prover stream. Streaming
/// them into one table stays serial, so a family weighs a batch of units
/// with this and then visits their addresses.
pub(crate) fn weigh_units<W: Send>(
    range: std::ops::Range<usize>,
    weigh: impl Fn(usize) -> Result<W, AkitaError> + Sync + Send,
) -> Result<Vec<W>, AkitaError> {
    #[cfg(feature = "parallel")]
    let weights = range.into_par_iter().map(weigh).collect();
    #[cfg(not(feature = "parallel"))]
    let weights = range.map(weigh).collect();
    weights
}

/// The structured public weight families of a binary root.
#[derive(Clone, Debug)]
pub(crate) enum RootWeightFamily<E: Field> {
    DirectA(DirectAWeights<E>),
    Image(ImageWeights<E>),
    Prime(PrimePlaneWeights<E>),
    ParityRows(ParityRowWeights<E>),
    Authentication(DigitAuthenticationWeights<E>),
}

impl<E: Field> RootWeightFamily<E> {
    pub(crate) fn table_len(&self) -> usize {
        match self {
            Self::DirectA(direct) => direct.table_len(),
            Self::Image(image) => image.table_len(),
            Self::Prime(prime) => prime.table_len(),
            Self::ParityRows(rows) => rows.table_len(),
            Self::Authentication(authentication) => authentication.table_len(),
        }
    }

    pub(crate) fn evaluate(&self, eq: &RootEq<E>) -> Result<E, AkitaError> {
        match self {
            Self::DirectA(direct) => direct.evaluate(eq),
            Self::Image(image) => {
                let _span = tracing::info_span!("root_weights_evaluate_image").entered();
                image.evaluate(eq)
            }
            Self::Prime(prime) => {
                let _span = tracing::info_span!("root_weights_evaluate_prime").entered();
                prime.evaluate(eq)
            }
            Self::ParityRows(rows) => {
                let _span = tracing::info_span!("root_weights_evaluate_parity_rows").entered();
                rows.evaluate(eq)
            }
            Self::Authentication(authentication) => {
                let _span = tracing::info_span!("root_weights_evaluate_authentication").entered();
                authentication.evaluate(eq)
            }
        }
    }

    /// Stream every `(address, weight)` term; addresses may repeat.
    pub(crate) fn visit(
        &self,
        mut visit: impl FnMut(usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        let mut recompose = |lanes: &SegmentLanes, element, offset, value| {
            lanes.recompose(element, offset, value, &mut visit)
        };
        match self {
            Self::DirectA(direct) => {
                let _span = tracing::info_span!("root_weights_visit_direct_a").entered();
                direct.profiles(&mut recompose)
            }
            Self::Image(image) => {
                let _span = tracing::info_span!("root_weights_visit_image").entered();
                image.profiles(&mut recompose)
            }
            Self::Prime(prime) => {
                let _span = tracing::info_span!("root_weights_visit_prime").entered();
                prime.profiles(&mut recompose)
            }
            Self::ParityRows(rows) => {
                let _span = tracing::info_span!("root_weights_visit_parity_rows").entered();
                rows.visit(visit)
            }
            Self::Authentication(authentication) => {
                let _span = tracing::info_span!("root_weights_visit_authentication").entered();
                authentication.visit(visit)
            }
        }
    }

    /// Add this family to block profiles. Recomposed families add one
    /// profile entry per element coefficient; the others add every term.
    pub(crate) fn emit(&self, profiles: &mut ProfileBuilder<E>) -> Result<(), AkitaError> {
        let mut recomposed = |lanes: &SegmentLanes, element, offset, value| {
            profiles.add_recomposed(lanes, element, offset, value)
        };
        match self {
            Self::DirectA(direct) => {
                let _span = tracing::info_span!("root_weights_emit_direct_a").entered();
                direct.profiles(&mut recomposed)
            }
            Self::Image(image) => {
                let _span = tracing::info_span!("root_weights_emit_image").entered();
                image.profiles(&mut recomposed)
            }
            Self::Prime(prime) => {
                let _span = tracing::info_span!("root_weights_emit_prime").entered();
                prime.profiles(&mut recomposed)
            }
            Self::ParityRows(rows) => {
                let _span = tracing::info_span!("root_weights_emit_parity_rows").entered();
                rows.visit(|address, value| profiles.add(address, value))
            }
            Self::Authentication(authentication) => {
                let _span = tracing::info_span!("root_weights_emit_authentication").entered();
                authentication.emit(profiles)
            }
        }
    }
}
