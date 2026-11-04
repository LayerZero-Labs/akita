//! Structured public weight families on one root's outgoing digit table.
//!
//! A family keeps the public inputs of one linear identity instead of one
//! term per table address. The verifier evaluates its multilinear extension
//! at the Stage 2 point through [`RootEq`]; the prover streams exactly the same
//! field elements into its dense Stage 2 table. Both read one construction,
//! so the two sides cannot disagree about a weight.

use std::borrow::Cow;

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::ResidueKernelPoint;
use akita_error::{checked, AkitaError};
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::{cfg_chunks, Field};

use super::profiles::ProfileBuilder;
use crate::binary_source::{
    DigitAuthenticationWeights, DirectAWeights, ImageWeights, ParityRowWeights,
    PartialColumnWeights, PrimePlaneWeights, SourceEvaluationWeights,
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

    /// The equality point, low coordinate first.
    pub(crate) fn point(&self) -> &[E] {
        &self.point
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    pub(crate) fn start(&self) -> usize {
        self.start
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
    PartialColumns(PartialColumnWeights<E>),
    Authentication(DigitAuthenticationWeights<E>),
    Translated(TranslatedLanes<E>),
    SourceEvaluation(SourceEvaluationWeights<E>),
}

impl<E: Field> RootWeightFamily<E> {
    pub(crate) fn table_len(&self) -> usize {
        match self {
            Self::DirectA(direct) => direct.table_len(),
            Self::Image(image) => image.table_len(),
            Self::Prime(prime) => prime.table_len(),
            Self::ParityRows(rows) => rows.table_len(),
            Self::PartialColumns(columns) => columns.table_len(),
            Self::Authentication(authentication) => authentication.table_len(),
            Self::Translated(lanes) => lanes.table_len(),
            Self::SourceEvaluation(source) => source.table_len(),
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
            Self::PartialColumns(columns) => {
                let _span = tracing::info_span!("root_weights_evaluate_partial_columns").entered();
                columns.evaluate(eq)
            }
            Self::Authentication(authentication) => {
                let _span = tracing::info_span!("root_weights_evaluate_authentication").entered();
                authentication.evaluate(eq)
            }
            Self::Translated(lanes) => {
                let _span = tracing::info_span!("root_weights_evaluate_translated").entered();
                lanes.evaluate(eq)
            }
            Self::SourceEvaluation(source) => {
                let _span = tracing::info_span!("root_weights_evaluate_source").entered();
                source.evaluate(eq)
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
            Self::PartialColumns(columns) => {
                let _span = tracing::info_span!("root_weights_visit_partial_columns").entered();
                columns.visit(visit)
            }
            Self::Translated(lanes) => {
                let _span = tracing::info_span!("root_weights_visit_translated").entered();
                lanes.visit(visit)
            }
            Self::SourceEvaluation(source) => {
                let _span = tracing::info_span!("root_weights_visit_source").entered();
                source.visit(visit)
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
                authentication.visit(|address, value| profiles.add(address, value))
            }
            Self::PartialColumns(columns) => {
                let _span = tracing::info_span!("root_weights_emit_partial_columns").entered();
                columns.visit(|address, value| profiles.add(address, value))
            }
            Self::Translated(lanes) => {
                let _span = tracing::info_span!("root_weights_emit_translated").entered();
                lanes.visit(|address, value| profiles.add(address, value))
            }
            Self::SourceEvaluation(source) => {
                let _span = tracing::info_span!("root_weights_emit_source").entered();
                source.visit(|address, value| profiles.add(address, value))
            }
        }
    }
}

/// One public lane pattern copied to several table addresses.
///
/// Copy `(base, scale)` and lane `(offset, weights)` put `scale * weights[c]`
/// at `base + offset + c`. A row family that repeats one public pattern, such
/// as one matrix row against every block or one recomposition against every
/// bit, stores the pattern once with one scale per copy.
///
/// Copies whose bases agree modulo the low block width read the same low
/// equality factors. One pass over the pattern per such class gives a partial
/// sum per high block, and every copy in the class combines them with its own
/// high factors. Verification costs `classes * pattern + copies * span / 2^b`
/// operations instead of `copies * pattern`.
///
/// With a residue point, each lane holds the coefficients `a` of one
/// negacyclic ring element and its weights are the residue kernel
/// `w_j = (X^j a mod X^d + 1)(alpha)`. That kernel is a Hankel product, so the
/// part of a lane inside one low block reads `<a, K(x)>`, where `x` holds the
/// lane's low factors in that block and zeros elsewhere. These kernels depend
/// only on the lane's low offset, so the verifier computes them once per
/// offset instead of one kernel per lane.
#[derive(Clone, Debug)]
pub(crate) struct TranslatedLanes<E: Field> {
    table_len: usize,
    copies: Vec<(usize, E)>,
    lanes: Vec<(usize, Vec<E>)>,
    residue: Option<ResidueKernelPoint<E>>,
    /// One past the largest lane address relative to a base.
    span: usize,
}

/// Lanes per parallel chunk of one pattern pass.
const PATTERN_CHUNK_LANES: usize = 16;

impl<E: Field> TranslatedLanes<E> {
    /// Check that every copy of every lane stays inside the table, and that
    /// residue lanes hold one ring element of a power-of-two degree.
    pub(crate) fn new(
        table_len: usize,
        copies: Vec<(usize, E)>,
        lanes: Vec<(usize, Vec<E>)>,
        residue: Option<E>,
    ) -> Result<Self, AkitaError> {
        let residue = match (residue, lanes.first()) {
            (Some(alpha), Some((_, ring))) => {
                if lanes.iter().any(|(_, lane)| lane.len() != ring.len()) {
                    return Err(AkitaError::InvalidProof);
                }
                Some(ResidueKernelPoint::new(alpha, ring.len())?)
            }
            _ => None,
        };
        let span = lanes
            .iter()
            .map(|(offset, weights)| checked::sum([*offset, weights.len()]))
            .try_fold(0usize, |span, end| end.map(|end| span.max(end)))
            .ok_or(AkitaError::InvalidProof)?;
        let last = copies
            .iter()
            .map(|&(base, _)| checked::sum([base, span]))
            .try_fold(0usize, |last, end| end.map(|end| last.max(end)))
            .ok_or(AkitaError::InvalidProof)?;
        if last > table_len {
            return Err(AkitaError::InvalidSetup(
                "translated root lanes leave the weight table".into(),
            ));
        }
        Ok(Self {
            table_len,
            copies,
            lanes,
            residue,
            span,
        })
    }

    pub(crate) fn table_len(&self) -> usize {
        self.table_len
    }

    pub(crate) fn evaluate(&self, eq: &RootEq<E>) -> Result<E, AkitaError> {
        let block = eq.low().len();
        let mut classes = self
            .copies
            .iter()
            .map(|&(base, scale)| (eq.low_offset(base), base, scale))
            .collect::<Vec<_>>();
        classes.sort_unstable_by_key(|&(class, base, _)| (class, base));
        // The largest relative address is `block - 1 + span - 1`.
        let blocks = checked::sum([block - 1, self.span])
            .map(|end| end.div_ceil(block))
            .ok_or(AkitaError::InvalidProof)?;
        // Residue kernels of the low windows that lanes read, one per low
        // block a lane meets, by low offset; every class shares them.
        let mut windows = Vec::new();
        if let Some(point) = &self.residue {
            windows.resize(block, None);
            for copies in classes.chunk_by(|left, right| left.0 == right.0) {
                let class = copies.first().ok_or(AkitaError::InvalidProof)?.0;
                for (offset, _) in &self.lanes {
                    let start = (class + offset) % block;
                    let slot = windows.get_mut(start).ok_or(AkitaError::InvalidProof)?;
                    if slot.is_none() {
                        *slot = Some(residue_windows(point, eq.low(), start)?);
                    }
                }
            }
        }
        let mut sum = E::zero();
        for copies in classes.chunk_by(|left, right| left.0 == right.0) {
            let class = copies.first().ok_or(AkitaError::InvalidProof)?.0;
            let partials = self.pattern_partials(eq, class, blocks, &windows)?;
            for &(_, base, scale) in copies {
                let mut copy = E::zero();
                for (index, &partial) in partials.iter().enumerate() {
                    if partial.is_zero() {
                        continue;
                    }
                    let address = checked::mul_add(index, block, base - class)
                        .ok_or(AkitaError::InvalidProof)?;
                    copy += partial * eq.high(address)?;
                }
                sum += scale * copy;
            }
        }
        Ok(sum)
    }

    /// Low-weighted pattern sums per high block, for bases `class mod 2^b`.
    fn pattern_partials(
        &self,
        eq: &RootEq<E>,
        class: usize,
        blocks: usize,
        windows: &[Option<Vec<Vec<E>>>],
    ) -> Result<Vec<E>, AkitaError> {
        let low = eq.low();
        let block = low.len();
        cfg_chunks!(self.lanes, PATTERN_CHUNK_LANES)
            .map(|lanes| {
                let mut partials = vec![E::zero(); blocks];
                for (offset, lane) in lanes {
                    let mut address = class + offset;
                    if self.residue.is_some() {
                        let window = windows
                            .get(address % block)
                            .and_then(Option::as_ref)
                            .ok_or(AkitaError::InvalidProof)?;
                        for (index, kernel) in window.iter().enumerate() {
                            *partials
                                .get_mut(address / block + index)
                                .ok_or(AkitaError::InvalidProof)? += lane
                                .iter()
                                .zip(kernel)
                                .fold(E::zero(), |dot, (&weight, &factor)| dot + weight * factor);
                        }
                        continue;
                    }
                    let mut weights = lane.as_slice();
                    while !weights.is_empty() {
                        let start = address % block;
                        let (run, rest) = weights.split_at(weights.len().min(block - start));
                        let factors = low.get(start..).ok_or(AkitaError::InvalidProof)?;
                        let dot = run
                            .iter()
                            .zip(factors)
                            .fold(E::zero(), |dot, (&weight, &factor)| dot + weight * factor);
                        *partials
                            .get_mut(address / block)
                            .ok_or(AkitaError::InvalidProof)? += dot;
                        address += run.len();
                        weights = rest;
                    }
                }
                Ok(partials)
            })
            .collect::<Result<Vec<_>, AkitaError>>()?
            .into_iter()
            .try_fold(vec![E::zero(); blocks], |mut sum, partials| {
                for (total, partial) in sum.iter_mut().zip(partials) {
                    *total += partial;
                }
                Ok(sum)
            })
    }

    pub(crate) fn visit(
        &self,
        mut visit: impl FnMut(usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        let kernels = match &self.residue {
            Some(point) => Cow::Owned(
                self.lanes
                    .iter()
                    .map(|(offset, ring)| Ok((*offset, point.field_kernel(ring)?)))
                    .collect::<Result<Vec<_>, AkitaError>>()?,
            ),
            None => Cow::Borrowed(&self.lanes),
        };
        for &(base, scale) in &self.copies {
            for (offset, weights) in kernels.iter() {
                for (coefficient, &weight) in weights.iter().enumerate() {
                    visit(base + offset + coefficient, scale * weight)?;
                }
            }
        }
        Ok(())
    }
}

/// Residue kernels of the low factors a degree-`d` lane at low offset
/// `start` reads, one per low block it meets, each zero outside its run.
fn residue_windows<E: Field>(
    point: &ResidueKernelPoint<E>,
    low: &[E],
    start: usize,
) -> Result<Vec<Vec<E>>, AkitaError> {
    let degree = point.dimension();
    let mut windows = Vec::new();
    let (mut position, mut low_start) = (0, start);
    while position < degree {
        let run = (degree - position).min(low.len() - low_start);
        let factors = low
            .get(low_start..low_start + run)
            .ok_or(AkitaError::InvalidProof)?;
        let mut window = vec![E::zero(); degree];
        window
            .get_mut(position..position + run)
            .ok_or(AkitaError::InvalidProof)?
            .copy_from_slice(factors);
        windows.push(point.field_kernel(&window)?);
        position += run;
        low_start = 0;
    }
    Ok(windows)
}
