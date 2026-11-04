//! Structured public weight families on one root's outgoing digit table.
//!
//! A family keeps the public inputs of one linear identity instead of one
//! term per table address. The verifier evaluates its multilinear extension
//! at the Stage 2 point through [`RootEq`]; the prover streams exactly the same
//! field elements into its dense Stage 2 table. Both read one construction,
//! so the two sides cannot disagree about a weight.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_error::{checked, AkitaError};
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::Field;

use crate::binary_source::{
    DigitAuthenticationWeights, DirectAWeights, ImageWeights, ParityRowWeights, PrimePlaneWeights,
};
use crate::root_repacking::RootCoefficientRepacking;

/// Low-block width of the split equality table.
const LOW_BITS: usize = 10;
/// Bound on the high table, which has one entry per 1024 table addresses.
const MAX_HIGH_BITS: usize = 22;

/// Equality weights at one point, split as `eq(r,a) = low[a mod 2^b] * high[a >> b]`.
///
/// This evaluates `eq(r, a)` at any single address in one multiplication. The
/// tables hold `2^b + 2^(n-b)` elements rather than `2^n`. Interleaved lanes
/// are evaluated through [`SegmentEq`], which this keeps the point for.
pub(crate) struct RootEq<E: Field> {
    point: Vec<E>,
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
            point: point.to_vec(),
            low_bits,
            low: EqPolynomial::evals(low_point)?,
            high: EqPolynomial::evals(high_point)?,
        })
    }

    /// `eq(r, address)`.
    pub(crate) fn at(&self, address: usize) -> Result<E, AkitaError> {
        let low = self
            .low
            .get(address & (self.low.len() - 1))
            .ok_or(AkitaError::InvalidProof)?;
        let high = self
            .high
            .get(address >> self.low_bits)
            .ok_or(AkitaError::InvalidProof)?;
        Ok(*low * *high)
    }

    /// The lane and coefficient factors of one interleaved segment.
    pub(crate) fn segment(&self, lanes: &SegmentLanes) -> Result<SegmentEq<E>, AkitaError> {
        SegmentEq::new(&self.point, lanes)
    }
}

/// Lane addresses of one single-plane repacking segment.
///
/// A lane is one digit of one element, numbered `element * digits + digit`.
/// Lanes form runs of `run` lanes, a power of two, and coefficient `c` of a
/// lane lives at its first address plus `c * run`
/// ([`RootCoefficientRepacking`] states the layout).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SegmentLanes {
    start: usize,
    run: usize,
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
        let run = repacking.run_lanes(segment)?;
        if shape.geometry.coordinate_plane_count() != 1 || !start.is_multiple_of(run) {
            return Err(AkitaError::InvalidSetup(
                "structured root weights require aligned single-plane lanes".into(),
            ));
        }
        Ok(Self {
            start,
            run,
            degree: shape.geometry.polynomial_modulus_dimension(),
            digits: shape.digits,
            elements: shape.elements,
        })
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
    /// Address distance between consecutive coefficients of one lane.
    pub(crate) fn step(&self) -> usize {
        self.run
    }

    /// Lane number of one digit of one element.
    pub(crate) fn lane(&self, element: usize, digit: usize) -> Result<usize, AkitaError> {
        if element >= self.elements || digit >= self.digits {
            return Err(AkitaError::InvalidInput(
                "root lane index is invalid".into(),
            ));
        }
        checked::mul_add(element, self.digits, digit).ok_or(AkitaError::InvalidProof)
    }

    /// Address of natural coefficient zero in one lane; coefficient `c` is at
    /// this address plus `c * step`.
    pub(crate) fn base(&self, element: usize, digit: usize) -> Result<usize, AkitaError> {
        let lane = self.lane(element, digit)?;
        checked::product([lane / self.run, self.run, self.degree])
            .and_then(|offset| checked::sum([self.start, offset, lane % self.run]))
            .ok_or(AkitaError::InvalidProof)
    }

    /// Address of one natural coefficient of one lane.
    pub(crate) fn address(
        &self,
        element: usize,
        digit: usize,
        coefficient: usize,
    ) -> Result<usize, AkitaError> {
        if coefficient >= self.degree {
            return Err(AkitaError::InvalidInput(
                "root lane coefficient is invalid".into(),
            ));
        }
        checked::mul_add(coefficient, self.run, self.base(element, digit)?)
            .ok_or(AkitaError::InvalidProof)
    }
}

/// Lanes a prover stream writes per coefficient before moving to the next.
const VISIT_LANES: usize = 256;

impl SegmentLanes {
    /// Stream `weight(element, digit, coefficient)` to every lane of the
    /// elements in `elements`, for the first `coefficients` coefficients.
    ///
    /// Groups of about [`VISIT_LANES`] lanes are written coefficient by
    /// coefficient, so consecutive writes land on consecutive addresses of
    /// one run instead of `step` apart.
    pub(crate) fn visit_elements<E>(
        &self,
        elements: std::ops::Range<usize>,
        coefficients: usize,
        mut weight: impl FnMut(usize, usize, usize) -> Result<E, AkitaError>,
        mut visit: impl FnMut(usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        if coefficients > self.degree || elements.end > self.elements {
            return Err(AkitaError::InvalidInput(
                "root lane visit exceeds the segment".into(),
            ));
        }
        let group = (VISIT_LANES / self.digits).max(1);
        let mut bases = Vec::new();
        bases
            .try_reserve_exact(group * self.digits)
            .map_err(|_| AkitaError::InvalidProof)?;
        for first in elements.clone().step_by(group) {
            let last = elements.end.min(first + group);
            bases.clear();
            for element in first..last {
                for digit in 0..self.digits {
                    bases.push((element, digit, self.base(element, digit)?));
                }
            }
            for coefficient in 0..coefficients {
                let offset = coefficient * self.run;
                for &(element, digit, base) in &bases {
                    visit(base + offset, weight(element, digit, coefficient)?)?;
                }
            }
        }
        Ok(())
    }
}

/// Equality weights of one segment's lanes at one point.
///
/// Lane `l`'s coefficient `c` sits at `start + (l / W) W N + c W + l mod W`,
/// and `start` is a multiple of `W`, so
/// `eq(r, address) = eq(r_[0,w), l mod W) * eq(r_[w,n), start / W + (l / W) N + c)`.
/// The first factor is the lane factor; the second is the coefficient vector
/// shared by every lane of run `l / W`. Building both costs `W` multiplications
/// for the lane table and about `N + n` per run.
pub(crate) struct SegmentEq<E: Field> {
    lanes: SegmentLanes,
    lane_eq: Vec<E>,
    coefficients: Vec<Vec<E>>,
}

impl<E: Field> SegmentEq<E> {
    pub(crate) fn new(point: &[E], lanes: &SegmentLanes) -> Result<Self, AkitaError> {
        let lane_bits = lanes.run.trailing_zeros() as usize;
        let (lane_point, rest) = (point.get(..lane_bits), point.get(lane_bits..));
        let (lane_point, rest) = lane_point.zip(rest).ok_or(AkitaError::InvalidProof)?;
        let low_bits = rest.len().min(LOW_BITS);
        let (low_point, high_point) = rest.split_at(low_bits);
        let low = EqPolynomial::evals(low_point)?;
        let mask = low.len() - 1;
        let lane_count =
            checked::product([lanes.elements, lanes.digits]).ok_or(AkitaError::InvalidProof)?;
        let runs = lane_count / lanes.run;
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(runs)
            .map_err(|_| AkitaError::InvalidProof)?;
        for run in 0..runs {
            let first = checked::mul_add(run, lanes.degree, lanes.start / lanes.run)
                .ok_or(AkitaError::InvalidProof)?;
            let mut vector = Vec::new();
            vector
                .try_reserve_exact(lanes.degree)
                .map_err(|_| AkitaError::InvalidProof)?;
            let mut block = None;
            for index in
                first..checked::sum([first, lanes.degree]).ok_or(AkitaError::InvalidProof)?
            {
                let high = index >> low_bits;
                let factor = match block {
                    Some((current, factor)) if current == high => factor,
                    _ => {
                        if checked::pow2(high_point.len()).is_some_and(|bound| high >= bound) {
                            return Err(AkitaError::InvalidProof);
                        }
                        let factor = eq_eval_at_index(high_point, high);
                        block = Some((high, factor));
                        factor
                    }
                };
                vector.push(*low.get(index & mask).ok_or(AkitaError::InvalidProof)? * factor);
            }
            coefficients.push(vector);
        }
        Ok(Self {
            lanes: *lanes,
            lane_eq: EqPolynomial::evals(lane_point)?,
            coefficients,
        })
    }

    /// Run and lane factor of one digit of one element.
    pub(crate) fn lane(&self, element: usize, digit: usize) -> Result<(usize, E), AkitaError> {
        let lane = self.lanes.lane(element, digit)?;
        let factor = self
            .lane_eq
            .get(lane % self.lanes.run)
            .copied()
            .ok_or(AkitaError::InvalidProof)?;
        Ok((lane / self.lanes.run, factor))
    }

    /// The coefficient factors shared by every lane of one run.
    pub(crate) fn coefficients(&self, run: usize) -> Result<&[E], AkitaError> {
        self.coefficients
            .get(run)
            .map(Vec::as_slice)
            .ok_or(AkitaError::InvalidProof)
    }

    pub(crate) fn runs(&self) -> usize {
        self.coefficients.len()
    }

    /// Visit `(run, sum_d 8^d * lane factor)` over one element's digit lanes,
    /// once per run they touch: the digit recomposition of every coefficient,
    /// contracted with the lane factors.
    pub(crate) fn digit_weights(
        &self,
        element: usize,
        mut visit: impl FnMut(usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        let mut current: Option<(usize, E)> = None;
        let mut power = E::one();
        for digit in 0..self.lanes.digits {
            let (run, factor) = self.lane(element, digit)?;
            current = match current {
                Some((open, sum)) if open == run => Some((open, sum + power * factor)),
                Some((open, sum)) => {
                    visit(open, sum)?;
                    Some((run, power * factor))
                }
                None => Some((run, power * factor)),
            };
            power *= E::from_u64(8);
        }
        match current {
            Some((run, sum)) => visit(run, sum),
            None => Ok(()),
        }
    }
}

/// `scale * 8^d` for each of `digits` balanced octal digits.
pub(crate) fn digit_powers<E: Field>(digits: usize, scale: E) -> Vec<E> {
    std::iter::successors(Some(scale), |&power| Some(power * E::from_u64(8)))
        .take(digits)
        .collect()
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
        visit: impl FnMut(usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        match self {
            Self::DirectA(direct) => {
                let _span = tracing::info_span!("root_weights_visit_direct_a").entered();
                direct.visit(visit)
            }
            Self::Image(image) => {
                let _span = tracing::info_span!("root_weights_visit_image").entered();
                image.visit(visit)
            }
            Self::Prime(prime) => {
                let _span = tracing::info_span!("root_weights_visit_prime").entered();
                prime.visit(visit)
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
}
