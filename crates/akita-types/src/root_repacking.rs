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

use akita_error::AkitaError;
use akita_params::RelationRowGeometry;
use jolt_algebra::checked;
#[cfg(feature = "parallel")]
use jolt_algebra::solinas::parallel::*;
use jolt_algebra::{cfg_iter, Field};

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
/// Segment order is followed by element, digit, coordinate plane, and natural
/// polynomial coefficient. Each plane uses the row geometry's padded stride.
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
            let width = checked::product([
                segment.elements,
                segment.digits,
                segment.geometry.physical_coefficient_width(),
            ])
            .ok_or_else(|| AkitaError::InvalidSetup("root repacking width overflow".into()))?;
            let range = checked::range(live_len, width)
                .ok_or_else(|| AkitaError::InvalidSetup("root repacking range overflow".into()))?;
            live_len = range.end;
            ranges.push(range);
        }
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
        let degree = shape.geometry.polynomial_modulus_dimension();
        let natural_width = checked::product([degree, shape.geometry.coordinate_plane_count()])
            .ok_or_else(|| AkitaError::InvalidSetup("root natural width overflow".into()))?;
        if element >= shape.elements || digit >= shape.digits || coefficient >= natural_width {
            return Err(AkitaError::InvalidInput(
                "root repacking coefficient index is invalid".into(),
            ));
        }
        let physical = shape
            .geometry
            .physical_coefficient_index(coefficient / degree, coefficient % degree)?;
        let lane = checked::mul_add(element, shape.digits, digit)
            .ok_or_else(|| AkitaError::InvalidSetup("root repacking lane overflow".into()))?;
        let offset = checked::mul_add(lane, shape.geometry.physical_coefficient_width(), physical)
            .ok_or_else(|| AkitaError::InvalidSetup("root coefficient offset overflow".into()))?;
        let range = self.ranges.get(segment).ok_or(AkitaError::InvalidProof)?;
        checked::sum([range.start, offset])
            .filter(|&address| address < range.end)
            .ok_or_else(|| AkitaError::InvalidSetup("root coefficient address overflow".into()))
    }

    /// Whether a table address holds no natural coefficient.
    ///
    /// Honest construction writes zero at these addresses. The binary root
    /// compiles no row on them; [`NativeRootRelation::ring_switch`] adds one
    /// zero row per address, including the final ordinary-ring padding.
    ///
    /// [`NativeRootRelation::ring_switch`]: crate::native_root_relation::NativeRootRelation::ring_switch
    pub fn is_padding_address(&self, address: usize) -> Result<bool, AkitaError> {
        if address >= self.domain_len {
            return Err(AkitaError::InvalidInput(
                "root repacking table address is invalid".into(),
            ));
        }
        if address >= self.live_len {
            return Ok(true);
        }
        for (shape, range) in self.segments.iter().zip(&self.ranges) {
            if range.contains(&address) {
                return Ok(
                    (address - range.start) % shape.geometry.padded_coefficient_count()
                        >= shape.geometry.polynomial_modulus_dimension(),
                );
            }
        }
        Err(AkitaError::InvalidProof)
    }

    /// One past the largest natural coefficient address.
    ///
    /// Every address at or beyond this bound is a padding address.
    pub fn natural_end(&self) -> Result<usize, AkitaError> {
        let (shape, range) = self
            .segments
            .last()
            .zip(self.ranges.last())
            .ok_or(AkitaError::InvalidProof)?;
        range
            .end
            .checked_sub(shape.geometry.padded_coefficient_count())
            .and_then(|start| checked::sum([start, shape.geometry.polynomial_modulus_dimension()]))
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
            let natural_width = checked::product([
                shape.geometry.polynomial_modulus_dimension(),
                shape.geometry.coordinate_plane_count(),
            ])
            .ok_or_else(|| AkitaError::InvalidSetup("root natural width overflow".into()))?;
            let expected = checked::product([shape.elements, shape.digits, natural_width])
                .ok_or_else(|| AkitaError::InvalidSetup("root digit count overflow".into()))?;
            if digits.len() != expected {
                return Err(AkitaError::InvalidSize {
                    expected,
                    actual: digits.len(),
                });
            }
            // Reject bad digits before allocating the table.
            if !cfg_iter!(digits).all(|value| (-4..=3).contains(value)) {
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
        // Lane `l` of a segment is the block `[l * width, (l + 1) * width)`
        // past the segment start, laid out as lane zero, so
        // [`Self::coefficient_address`] runs once per coefficient of lane
        // zero and the lanes fill their disjoint blocks in parallel.
        for (segment, (shape, digits)) in self.segments.iter().zip(natural_digits).enumerate() {
            let natural_width = checked::product([
                shape.geometry.polynomial_modulus_dimension(),
                shape.geometry.coordinate_plane_count(),
            ])
            .ok_or_else(|| AkitaError::InvalidSetup("root natural width overflow".into()))?;
            if shape.elements == 0 || shape.digits == 0 {
                continue;
            }
            let range = self.segment_range(segment)?;
            let width = shape.geometry.physical_coefficient_width();
            let offsets = (0..natural_width)
                .map(|coefficient| {
                    self.coefficient_address(segment, 0, 0, coefficient)?
                        .checked_sub(range.start)
                        .filter(|&offset| offset < width)
                        .ok_or(AkitaError::InvalidProof)
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            let end = checked::product([shape.elements, shape.digits, width])
                .and_then(|len| checked::sum([range.start, len]))
                .filter(|&end| end <= range.end)
                .ok_or(AkitaError::InvalidProof)?;
            let lanes = table
                .get_mut(range.start..end)
                .ok_or(AkitaError::InvalidProof)?;
            let fill = |(lane, lane_digits): (&mut [i8], &[i8])| {
                for (&offset, &value) in offsets.iter().zip(lane_digits) {
                    lane[offset] = value;
                }
            };
            #[cfg(feature = "parallel")]
            lanes
                .par_chunks_exact_mut(width)
                .zip(digits.par_chunks_exact(natural_width))
                .for_each(fill);
            #[cfg(not(feature = "parallel"))]
            lanes
                .chunks_exact_mut(width)
                .zip(digits.chunks_exact(natural_width))
                .for_each(fill);
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
    ///
    /// The digit lanes of one element are consecutive, so digit `d` sits `d`
    /// lane widths after digit zero. Both end addresses are checked here; the
    /// weights are then yielded without allocation.
    pub fn coefficient_weights<F: Field>(
        &self,
        segment: usize,
        element: usize,
        coefficient: usize,
        scale: F,
    ) -> Result<impl Iterator<Item = (usize, F)>, AkitaError> {
        let shape = self.segments.get(segment).ok_or_else(|| {
            AkitaError::InvalidInput("root repacking segment index is invalid".into())
        })?;
        let first = match shape.digits.checked_sub(1) {
            Some(last) => {
                self.coefficient_address(segment, element, last, coefficient)?;
                self.coefficient_address(segment, element, 0, coefficient)?
            }
            None => 0,
        };
        let width = shape.geometry.physical_coefficient_width();
        let mut power = scale;
        Ok((0..shape.digits).map(move |digit| {
            let weight = power;
            power *= F::from_u64(8);
            (first + digit * width, weight)
        }))
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
            .try_fold(F::zero(), |sum, (address, weight)| {
                Ok(sum + *table.get(address).ok_or(AkitaError::InvalidProof)? * weight)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_params::RelationPolynomial;
    use jolt_algebra::{Prime128OffsetA7F7, Ring, Zero};
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
        assert_eq!(plan.segment_range(0).unwrap(), 0..8192);
        assert_eq!(plan.segment_range(1).unwrap(), 8192..8224);
        assert_eq!(plan.domain_len(), 16384);
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
        for address in [648, 8191, 8224, plan.domain_len() - 1] {
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
            .map(|(address, weight)| table[address] * weight)
            .sum::<F>();
        assert_eq!(relation, F::from_u64(11 * 3 * 64));
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
        assert_eq!(natural_end, 8224);
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
        // A bad digit is rejected before the table is allocated, even for a
        // domain far too large to allocate.
        let huge = plan.clone().with_domain_len(1 << 60).unwrap();
        assert!(matches!(
            huge.repack(&[bad.as_slice(), &[0; 32]]),
            Err(AkitaError::InvalidInput(_))
        ));
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
