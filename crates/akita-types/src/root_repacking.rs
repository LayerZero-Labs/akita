//! Coefficient addresses shared by a mixed root and its ordinary successor.
//!
//! Repacking is a change of coefficient layout, not a ring homomorphism. The
//! root relation must read the digit table that the successor commits. Its
//! linear coefficient weights below address that same table; there is no
//! separately supplied source buffer whose equality could be assumed.
//! The caller still proves the root equations and digit ranges in the shared
//! stages. Constructing or locally validating a table is not a proof of those
//! statements.

use core::ops::Range;

use akita_error::{checked, AkitaError};
use akita_params::RelationRowGeometry;
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::Field;

/// Alignment of the committed live table length.
///
/// The padded-stride layout always ended on a multiple of the 256-cell scalar
/// stride. Keeping that alignment admits every successor coefficient block it
/// admitted.
pub const LIVE_ALIGNMENT: usize = 256;

/// One bounded coefficient family in the root witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootRepackingSegment {
    pub geometry: RelationRowGeometry,
    pub elements: usize,
    /// Number of balanced base-eight digits per coefficient.
    pub digits: usize,
}

/// Checked, source-independent addresses for one ordinary successor digit table.
///
/// Segments follow one another in order. A lane is one digit of one element,
/// numbered `element * digits + digit`, and holds the element's natural
/// coefficients, coordinate plane first. A segment whose row geometry pads its
/// coefficients is *interleaved*: its lanes form runs of `W`, the largest power
/// of two dividing the lane count, and coefficient `c` of lane `l` lives at
/// `start + (l / W) * W * N + c * W + l mod W`, where `N` is the natural lane
/// width. The segment start is a multiple of `W`. The table therefore holds no
/// padded coefficient, and the equality weight of an interleaved address
/// factors into a low factor of `l mod W` and a high factor of
/// `start / W + (l / W) * N + c` (`specs/binary-source-dense-lanes.md`). A
/// segment without padding keeps each lane contiguous at `start + l * N + c`.
/// Destination ring elements are consecutive chunks of the resulting table;
/// their boundaries have no effect on any root polynomial identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootCoefficientRepacking {
    segments: Vec<RootRepackingSegment>,
    ranges: Vec<Range<usize>>,
    destination_ring_dimension: usize,
    live_len: usize,
    domain_len: usize,
}

impl RootCoefficientRepacking {
    pub fn new(
        segments: Vec<RootRepackingSegment>,
        destination_ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if segments.is_empty() || !destination_ring_dimension.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "root repacking needs segments and an ordinary power-of-two destination ring"
                    .into(),
            ));
        }
        let mut live_len = 0;
        let mut ranges = Vec::new();
        ranges.try_reserve_exact(segments.len()).map_err(|_| {
            AkitaError::InvalidSetup("root repacking segment allocation is too large".into())
        })?;
        for segment in &segments {
            if segment.elements == 0 || segment.digits == 0 {
                return Err(AkitaError::InvalidSetup(
                    "root repacking segments require nonzero elements and digits".into(),
                ));
            }
            let width =
                checked::product([segment.elements, segment.digits, natural_width(segment)?])
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("root repacking width overflow".into())
                    })?;
            let start = checked::align_up(live_len, run_lanes(segment)?)
                .ok_or_else(|| AkitaError::InvalidSetup("root repacking range overflow".into()))?;
            let range = checked::range(start, width)
                .ok_or_else(|| AkitaError::InvalidSetup("root repacking range overflow".into()))?;
            live_len = range.end;
            ranges.push(range);
        }
        let live_len = checked::align_up(live_len, LIVE_ALIGNMENT)
            .ok_or_else(|| AkitaError::InvalidSetup("root repacking range overflow".into()))?;
        let domain_len = checked::ceil_log2(live_len.max(destination_ring_dimension))
            .and_then(checked::pow2)
            .ok_or_else(|| AkitaError::InvalidSetup("root repacking domain overflow".into()))?;
        Ok(Self {
            segments,
            ranges,
            destination_ring_dimension,
            live_len,
            domain_len,
        })
    }

    /// Enlarge the authenticated table to a scheduled ordinary witness domain.
    ///
    /// Every added address is padding, like the tail of the minimal domain, and
    /// an honest prover fills it with zeros. A binary root places no constraint
    /// on padding values (`specs/binary-source-zero-padding.md`); only the
    /// generic `NativeRootRelation::ring_switch` compiler gives padding public
    /// weights. Live coefficient addresses never move. The domain must remain a power of two and contain the complete
    /// natural table and at least one ordinary destination ring element.
    pub fn with_domain_len(mut self, domain_len: usize) -> Result<Self, AkitaError> {
        if !domain_len.is_power_of_two() || domain_len < self.domain_len {
            return Err(AkitaError::InvalidSetup(
                "scheduled root domain is smaller than its coefficient table".into(),
            ));
        }
        self.domain_len = domain_len;
        Ok(self)
    }

    pub fn segments(&self) -> &[RootRepackingSegment] {
        &self.segments
    }

    pub const fn destination_ring_dimension(&self) -> usize {
        self.destination_ring_dimension
    }

    /// Committed live table length: the natural end rounded up to
    /// [`LIVE_ALIGNMENT`].
    ///
    /// The successor's relation reads the live witness in coefficient blocks
    /// of its smallest role dimension. An interleaved table can end at any
    /// cell, so the live witness keeps the padding up to the next aligned end.
    /// The length does not depend on the destination ring dimension.
    pub const fn live_len(&self) -> usize {
        self.live_len
    }

    /// Full digit table length, including plane tails and final ordinary padding.
    pub const fn domain_len(&self) -> usize {
        self.domain_len
    }

    /// Segment range including each coordinate plane's virtual tail.
    pub fn segment_range(&self, segment: usize) -> Result<Range<usize>, AkitaError> {
        self.ranges.get(segment).cloned().ok_or_else(|| {
            AkitaError::InvalidInput("root repacking segment index is invalid".into())
        })
    }

    /// Address one natural coefficient's digit in the shared committed table.
    ///
    /// `coefficient` flattens the natural coordinate planes: plane times natural
    /// polynomial degree, followed by its polynomial coefficient. Padding has
    /// no natural coefficient address.
    pub fn coefficient_address(
        &self,
        segment: usize,
        element: usize,
        digit: usize,
        coefficient: usize,
    ) -> Result<usize, AkitaError> {
        let shape = self.segments.get(segment).ok_or_else(|| {
            AkitaError::InvalidInput("root repacking segment index is invalid".into())
        })?;
        let natural_width = natural_width(shape)?;
        if element >= shape.elements || digit >= shape.digits || coefficient >= natural_width {
            return Err(AkitaError::InvalidInput(
                "root repacking coefficient index is invalid".into(),
            ));
        }
        let lane = checked::mul_add(element, shape.digits, digit)
            .ok_or_else(|| AkitaError::InvalidSetup("root repacking lane overflow".into()))?;
        let range = self.ranges.get(segment).ok_or(AkitaError::InvalidProof)?;
        let run = run_lanes(shape)?;
        checked::product([lane / run, run, natural_width])
            .and_then(|offset| checked::mul_add(coefficient, run, offset))
            .and_then(|offset| checked::sum([range.start, offset, lane % run]))
            .filter(|&address| address < range.end)
            .ok_or_else(|| AkitaError::InvalidSetup("root coefficient address overflow".into()))
    }

    /// Lanes per interleaved run of one segment: the largest power of two
    /// dividing its lane count, or one for a segment that keeps lanes contiguous.
    pub fn run_lanes(&self, segment: usize) -> Result<usize, AkitaError> {
        self.segments
            .get(segment)
            .ok_or_else(|| {
                AkitaError::InvalidInput("root repacking segment index is invalid".into())
            })
            .and_then(run_lanes)
    }

    /// Whether a table address holds no natural coefficient.
    ///
    /// Segments hold natural coefficients only, so padding is the alignment
    /// gap before an interleaved segment and the domain tail. Honest
    /// construction writes zero at these addresses. The binary root compiles
    /// no row on them; [`NativeRootRelation::ring_switch`] adds one zero row
    /// per address, including the final ordinary-ring padding.
    ///
    /// [`NativeRootRelation::ring_switch`]: crate::native_root_relation::NativeRootRelation::ring_switch
    pub fn is_padding_address(&self, address: usize) -> Result<bool, AkitaError> {
        if address >= self.domain_len {
            return Err(AkitaError::InvalidInput(
                "root repacking table address is invalid".into(),
            ));
        }
        Ok(!self.ranges.iter().any(|range| range.contains(&address)))
    }

    /// One past the largest natural coefficient address.
    ///
    /// Every address at or beyond this bound is a padding address.
    pub fn natural_end(&self) -> Result<usize, AkitaError> {
        self.ranges
            .last()
            .map(|range| range.end)
            .ok_or(AkitaError::InvalidProof)
    }

    /// Construct a prover table from natural, already balanced base-eight digits.
    ///
    /// Inputs follow segment/element/digit/natural-coordinate order. Every
    /// supplied digit lies in `[-4, 3]`; virtual tails are constructed as zero.
    pub fn repack(&self, natural_digits: &[&[i8]]) -> Result<Vec<i8>, AkitaError> {
        if natural_digits.len() != self.segments.len() {
            return Err(AkitaError::InvalidSize {
                expected: self.segments.len(),
                actual: natural_digits.len(),
            });
        }
        for (shape, digits) in self.segments.iter().zip(natural_digits) {
            let expected = checked::product([shape.elements, shape.digits, natural_width(shape)?])
                .ok_or_else(|| AkitaError::InvalidSetup("root digit count overflow".into()))?;
            if digits.len() != expected {
                return Err(AkitaError::InvalidSize {
                    expected,
                    actual: digits.len(),
                });
            }
            if digits.iter().any(|value| !(-4..=3).contains(value)) {
                return Err(AkitaError::InvalidInput(
                    "root repacking digit is outside the balanced base-eight interval".into(),
                ));
            }
        }
        let mut table = Vec::new();
        table.try_reserve_exact(self.domain_len).map_err(|_| {
            AkitaError::InvalidSetup("root repacking table allocation is too large".into())
        })?;
        table.resize(self.domain_len, 0);
        for (segment, (shape, digits)) in self.segments.iter().zip(natural_digits).enumerate() {
            let width = natural_width(shape)?;
            let run = run_lanes(shape)?;
            let region = self
                .ranges
                .get(segment)
                .and_then(|range| table.get_mut(range.clone()))
                .ok_or(AkitaError::InvalidProof)?;
            if run == 1 {
                region.copy_from_slice(digits);
                continue;
            }
            let run_len = checked::product([run, width]).ok_or(AkitaError::InvalidProof)?;
            for (run_table, run_digits) in region
                .chunks_exact_mut(run_len)
                .zip(digits.chunks_exact(run_len))
            {
                transpose_run(run_table, run_digits, run, width)?;
            }
        }
        Ok(table)
    }

    /// Check a complete field-encoded digit table as the honest prover builds it.
    ///
    /// Every digit lies in `[-4, 3]` and every padding address holds zero. This
    /// is a local construction check. The verifier enforces only the digit
    /// range on the authenticated witness. The binary root's equations do not
    /// read padding, so a nonzero in-range padding digit is not a proof
    /// failure there.
    pub fn validate_digit_table<F: Field>(&self, table: &[F]) -> Result<(), AkitaError> {
        if table.len() != self.domain_len {
            return Err(AkitaError::InvalidSize {
                expected: self.domain_len,
                actual: table.len(),
            });
        }
        let allowed = [-4, -3, -2, -1, 0, 1, 2, 3].map(F::from_i64);
        for (address, value) in table.iter().enumerate() {
            if !allowed.contains(value) || (self.is_padding_address(address)? && !value.is_zero()) {
                return Err(AkitaError::InvalidInput(
                    "root digit table has an invalid digit or nonzero padding".into(),
                ));
            }
        }
        Ok(())
    }

    /// Public linear weights for recomposing one root coefficient in Stage 2.
    ///
    /// These addresses are the ordinary successor's digit witness addresses.
    /// A root equation compiled from them authenticates coefficient repacking
    /// against that witness without a second coefficient table.
    pub fn coefficient_weights<F: Field>(
        &self,
        segment: usize,
        element: usize,
        coefficient: usize,
        scale: F,
    ) -> Result<Vec<(usize, F)>, AkitaError> {
        let shape = self.segments.get(segment).ok_or_else(|| {
            AkitaError::InvalidInput("root repacking segment index is invalid".into())
        })?;
        let mut weights = Vec::new();
        weights.try_reserve_exact(shape.digits).map_err(|_| {
            AkitaError::InvalidSetup("root coefficient weight allocation is too large".into())
        })?;
        let mut power = scale;
        for digit in 0..shape.digits {
            weights.push((
                self.coefficient_address(segment, element, digit, coefficient)?,
                power,
            ));
            power *= F::from_u64(8);
        }
        Ok(weights)
    }

    /// Evaluate the same linear coefficient map used by the root relation.
    ///
    /// The table must pass the shared digit-range and zero-padding constraints.
    /// This evaluation checks its length and addresses, without repeating that
    /// full boundary validation for every coefficient.
    pub fn recompose_coefficient<F: Field>(
        &self,
        table: &[F],
        segment: usize,
        element: usize,
        coefficient: usize,
    ) -> Result<F, AkitaError> {
        if table.len() != self.domain_len {
            return Err(AkitaError::InvalidSize {
                expected: self.domain_len,
                actual: table.len(),
            });
        }
        self.coefficient_weights(segment, element, coefficient, F::one())?
            .into_iter()
            .try_fold(F::zero(), |sum, (address, weight)| {
                Ok(sum + *table.get(address).ok_or(AkitaError::InvalidProof)? * weight)
            })
    }
}

/// Natural coefficients of one lane: every coordinate plane's degree.
fn natural_width(shape: &RootRepackingSegment) -> Result<usize, AkitaError> {
    checked::product([
        shape.geometry.polynomial_modulus_dimension(),
        shape.geometry.coordinate_plane_count(),
    ])
    .ok_or_else(|| AkitaError::InvalidSetup("root natural width overflow".into()))
}

/// Lanes per run: one when the geometry has no padding, so lanes stay
/// contiguous, and otherwise the largest power of two dividing the lane count.
fn run_lanes(shape: &RootRepackingSegment) -> Result<usize, AkitaError> {
    if shape.geometry.padded_coefficient_count() == shape.geometry.polynomial_modulus_dimension() {
        return Ok(1);
    }
    let lanes = checked::product([shape.elements, shape.digits])
        .filter(|&lanes| lanes > 0)
        .ok_or_else(|| AkitaError::InvalidSetup("root repacking lane overflow".into()))?;
    checked::pow2(lanes.trailing_zeros() as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("root repacking lane overflow".into()))
}

/// Coefficient rows one transpose task writes; 64 one-byte digits of a lane
/// share a cache line.
const TRANSPOSE_ROWS: usize = 64;

/// Write one run's lane-major digits coefficient-major: lane `j`, coefficient
/// `c` moves from `digits[j * width + c]` to `table[c * run + j]`.
fn transpose_run(
    table: &mut [i8],
    digits: &[i8],
    run: usize,
    width: usize,
) -> Result<(), AkitaError> {
    let rows = checked::product([TRANSPOSE_ROWS, run]).ok_or(AkitaError::InvalidProof)?;
    let transpose = |(chunk, block): (usize, &mut [i8])| -> Result<(), AkitaError> {
        let first = checked::product([chunk, TRANSPOSE_ROWS]).ok_or(AkitaError::InvalidProof)?;
        let count = block.len() / run;
        for lane in 0..run {
            let source = checked::mul_add(lane, width, first)
                .and_then(|start| digits.get(start..start.checked_add(count)?))
                .ok_or(AkitaError::InvalidProof)?;
            for (row, &value) in block.chunks_exact_mut(run).zip(source) {
                *row.get_mut(lane).ok_or(AkitaError::InvalidProof)? = value;
            }
        }
        Ok(())
    };
    #[cfg(feature = "parallel")]
    let result = table
        .par_chunks_mut(rows)
        .enumerate()
        .try_for_each(transpose);
    #[cfg(not(feature = "parallel"))]
    let result = table.chunks_mut(rows).enumerate().try_for_each(transpose);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_params::RelationPolynomial;
    use jolt_field::{Prime128OffsetA7F7, Ring, Zero};
    use std::collections::BTreeSet;

    type F = Prime128OffsetA7F7;

    fn plan() -> RootCoefficientRepacking {
        RootCoefficientRepacking::new(
            vec![
                RootRepackingSegment {
                    geometry: RelationRowGeometry::for_polynomial(
                        RelationPolynomial::minus_trinomial(648).unwrap(),
                        1,
                        1024,
                    )
                    .unwrap(),
                    elements: 2,
                    digits: 4,
                },
                RootRepackingSegment {
                    geometry: RelationRowGeometry::new(8, 2).unwrap(),
                    elements: 1,
                    digits: 2,
                },
            ],
            256,
        )
        .unwrap()
    }

    #[test]
    fn mixed_ring_repacking_matches_i128_and_has_no_address_aliases() {
        let plan = plan();
        let binary = (0..2 * 4 * 648)
            .map(|index| (index % 8) as i8 - 4)
            .collect::<Vec<_>>();
        let ordinary = (0..2 * 16)
            .map(|index| (index % 7) as i8 - 3)
            .collect::<Vec<_>>();
        let natural = [&binary[..], &ordinary[..]];
        let table = plan.repack(&natural).unwrap();
        let field_table = table
            .iter()
            .map(|&value| F::from_i64(i64::from(value)))
            .collect::<Vec<_>>();
        plan.validate_digit_table(&field_table).unwrap();
        let mut addresses = BTreeSet::new();
        for (segment, shape) in plan.segments().iter().enumerate() {
            let width = shape.geometry.polynomial_modulus_dimension()
                * shape.geometry.coordinate_plane_count();
            for element in 0..shape.elements {
                for coefficient in 0..width {
                    let mut reference = 0i128;
                    let mut power = 1i128;
                    for digit in 0..shape.digits {
                        let address = plan
                            .coefficient_address(segment, element, digit, coefficient)
                            .unwrap();
                        assert!(addresses.insert(address));
                        assert!(!plan.is_padding_address(address).unwrap());
                        reference += i128::from(
                            natural[segment]
                                [(element * shape.digits + digit) * width + coefficient],
                        ) * power;
                        power *= 8;
                    }
                    assert_eq!(
                        plan.recompose_coefficient(&field_table, segment, element, coefficient)
                            .unwrap(),
                        F::from_i64(i64::try_from(reference).unwrap())
                    );
                }
            }
        }
        // Eight binary lanes interleave coefficient-major; the ordinary
        // segment has no padding and stays lane-major.
        assert_eq!(plan.run_lanes(0).unwrap(), 8);
        assert_eq!(plan.run_lanes(1).unwrap(), 1);
        assert_eq!(plan.coefficient_address(0, 1, 2, 13).unwrap(), 13 * 8 + 6);
        assert_eq!(plan.coefficient_address(1, 0, 1, 3).unwrap(), 5184 + 16 + 3);
        assert_eq!(plan.segment_range(0).unwrap(), 0..5184);
        assert_eq!(plan.segment_range(1).unwrap(), 5184..5216);
        assert_eq!(plan.domain_len(), 8192);
        for (address, &value) in table.iter().enumerate() {
            if !addresses.contains(&address) {
                assert!(plan.is_padding_address(address).unwrap());
                assert_eq!(value, 0);
            }
        }
    }

    #[test]
    fn digit_and_padding_mutations_are_rejected() {
        let plan = plan();
        let binary = vec![0; 2 * 4 * 648];
        let ordinary = vec![0; 2 * 16];
        let mut table = plan
            .repack(&[&binary, &ordinary])
            .unwrap()
            .iter()
            .map(|&value| F::from_i64(i64::from(value)))
            .collect::<Vec<_>>();
        for address in [5216, 6000, plan.domain_len() - 1] {
            assert!(plan.is_padding_address(address).unwrap());
            table[address] = F::from_u64(1);
            assert!(plan.validate_digit_table(&table).is_err());
            table[address] = F::zero();
        }
        let address = plan.coefficient_address(0, 1, 2, 13).unwrap();
        table[address] = F::from_i64(4);
        assert!(plan.validate_digit_table(&table).is_err());
        table[address] = F::from_i64(-5);
        assert!(plan.validate_digit_table(&table).is_err());
        table[address] = F::from_i64(3);
        plan.validate_digit_table(&table).unwrap();
        assert_eq!(
            plan.recompose_coefficient(&table, 0, 1, 13).unwrap(),
            F::from_u64(3 * 64)
        );
        let weights = plan.coefficient_weights(0, 1, 13, F::from_u64(11)).unwrap();
        let relation = weights
            .into_iter()
            .map(|(address, weight)| table[address] * weight)
            .sum::<F>();
        assert_eq!(relation, F::from_u64(11 * 3 * 64));
    }

    #[test]
    fn interleaved_runs_align_and_their_gaps_are_padding() {
        let scalar = RelationRowGeometry::for_polynomial(
            RelationPolynomial::plus_trinomial(162).unwrap(),
            1,
            256,
        )
        .unwrap();
        let plan = RootCoefficientRepacking::new(
            vec![
                RootRepackingSegment {
                    geometry: scalar,
                    elements: 1,
                    digits: 1,
                },
                RootRepackingSegment {
                    geometry: scalar,
                    elements: 6,
                    digits: 2,
                },
            ],
            256,
        )
        .unwrap();
        // Twelve lanes form three runs of four, aligned past the first lane.
        assert_eq!(plan.run_lanes(1).unwrap(), 4);
        assert_eq!(plan.segment_range(0).unwrap(), 0..162);
        assert_eq!(plan.segment_range(1).unwrap(), 164..164 + 12 * 162);
        assert_eq!(
            plan.coefficient_address(1, 5, 1, 161).unwrap(),
            164 + 2 * 4 * 162 + 161 * 4 + 3
        );
        let natural = [
            vec![1i8; 162],
            (0..12 * 162).map(|i| (i % 7) as i8 - 3).collect(),
        ];
        let table = plan
            .repack(&[&natural[0], &natural[1]])
            .unwrap()
            .iter()
            .map(|&value| F::from_i64(i64::from(value)))
            .collect::<Vec<_>>();
        plan.validate_digit_table(&table).unwrap();
        for element in 0..6 {
            for coefficient in [0, 77, 161] {
                let expected = (0..2).rev().fold(0i64, |acc, digit| {
                    8 * acc + i64::from(natural[1][(element * 2 + digit) * 162 + coefficient])
                });
                assert_eq!(
                    plan.recompose_coefficient(&table, 1, element, coefficient)
                        .unwrap(),
                    F::from_i64(expected)
                );
            }
        }
        for address in [162, 163] {
            assert!(plan.is_padding_address(address).unwrap());
            let mut changed = table.clone();
            changed[address] = F::from_u64(1);
            assert!(plan.validate_digit_table(&changed).is_err());
        }
    }

    #[test]
    fn natural_end_matches_the_address_predicate() {
        let plan = plan();
        let natural_end = (0..plan.domain_len())
            .rev()
            .find(|&address| !plan.is_padding_address(address).unwrap())
            .unwrap()
            + 1;
        assert_eq!(plan.natural_end().unwrap(), natural_end);
        assert_eq!(natural_end, 5216);
    }

    #[test]
    fn malformed_shapes_addresses_and_input_digits_reject() {
        let plan = plan();
        assert!(plan.coefficient_address(2, 0, 0, 0).is_err());
        assert!(plan.coefficient_address(0, 2, 0, 0).is_err());
        assert!(plan.coefficient_address(0, 0, 4, 0).is_err());
        assert!(plan.coefficient_address(0, 0, 0, 648).is_err());
        assert!(plan.is_padding_address(plan.domain_len()).is_err());
        assert!(plan.repack(&[]).is_err());
        assert!(plan.repack(&[&[], &[]]).is_err());
        let mut bad = vec![0; 2 * 4 * 648];
        bad[0] = 4;
        assert!(plan.repack(&[&bad, &[0; 32]]).is_err());
        assert!(plan.validate_digit_table::<F>(&[]).is_err());
        let segment = RootRepackingSegment {
            geometry: RelationRowGeometry::native(8).unwrap(),
            elements: 1,
            digits: 1,
        };
        assert!(RootCoefficientRepacking::new(vec![], 256).is_err());
        assert!(RootCoefficientRepacking::new(vec![segment], 648).is_err());
        for (elements, digits) in [(0, 1), (1, 0), (usize::MAX, 2)] {
            assert!(RootCoefficientRepacking::new(
                vec![RootRepackingSegment {
                    elements,
                    digits,
                    ..segment
                }],
                256
            )
            .is_err());
        }
    }
}
