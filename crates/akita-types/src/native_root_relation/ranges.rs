//! Borrowed coefficients on contiguous ranges of a root table.

use std::ops::Range;

use akita_algebra::offset_eq::OffsetEqWindow;
use akita_error::{checked, AkitaError};
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::{cfg_iter, Field};

/// Ordered range contributions. Overlaps add, just as explicit root terms do.
/// Coefficients borrow the challenge vector; verification needs no address/value
/// expansion. The prover can consume the same ranges as explicit terms.
pub struct RootWeightRanges<'a, E: Field> {
    table_len: usize,
    ranges: Vec<(Range<usize>, &'a [E])>,
    term_count: usize,
}

impl<'a, E: Field> RootWeightRanges<'a, E> {
    pub(crate) fn new(
        table_len: usize,
        ranges: Vec<(Range<usize>, &'a [E])>,
    ) -> Result<Self, AkitaError> {
        if !table_len.is_power_of_two()
            || ranges.iter().any(|(range, coefficients)| {
                range.start > range.end
                    || range.end > table_len
                    || range.len() != coefficients.len()
            })
        {
            return Err(AkitaError::InvalidProof);
        }
        let term_count = checked::sum(ranges.iter().map(|(_, coefficients)| coefficients.len()))
            .ok_or(AkitaError::InvalidProof)?;
        Ok(Self {
            table_len,
            ranges,
            term_count,
        })
    }

    pub fn term_count(&self) -> usize {
        self.term_count
    }

    /// Expand for a prover backend that owns explicit address/value terms.
    pub fn into_terms(self) -> Result<Vec<(usize, E)>, AkitaError> {
        let mut terms = Vec::new();
        terms
            .try_reserve_exact(self.term_count)
            .map_err(|_| AkitaError::InvalidProof)?;
        for (addresses, coefficients) in self.ranges {
            #[cfg(feature = "parallel")]
            terms.par_extend(
                addresses
                    .into_par_iter()
                    .zip(coefficients.par_iter().copied()),
            );
            #[cfg(not(feature = "parallel"))]
            terms.extend(addresses.zip(coefficients.iter().copied()));
        }
        Ok(terms)
    }

    /// Evaluate the same MLE as the expanded terms, retaining every coefficient.
    pub fn evaluate(&self, point: &[E]) -> Result<E, AkitaError> {
        if point.len() != self.table_len.trailing_zeros() as usize {
            return Err(AkitaError::InvalidSize {
                expected: self.table_len.trailing_zeros() as usize,
                actual: point.len(),
            });
        }
        let equality = OffsetEqWindow::new(point)?;
        Ok(cfg_iter!(self.ranges)
            .map(|(addresses, coefficients)| {
                addresses
                    .clone()
                    .zip(coefficients.iter().copied())
                    .fold(E::zero(), |sum, (address, coefficient)| {
                        sum + coefficient * equality.eval(address)
                    })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(E::zero(), |sum, range| sum + range))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_root_relation::NativeRootWeights;
    use jolt_field::{Ext2, Prime128Offset275, Prime64Offset59, Ring, Zero};

    fn compare<E: Field>(value: impl Fn(u64) -> E) {
        for len in [0, 1, 63, 64, 65, 16383, 16384, 16385] {
            let coefficients = (0..len)
                .map(|i| value((i % 23) as u64) - value(11))
                .collect::<Vec<_>>();
            // Gaps, overlapping contributions, zero terms and a nonzero start.
            let ranges = vec![
                (7..7 + len, coefficients.as_slice()),
                (0..len, &coefficients),
            ];
            let expected = ranges
                .iter()
                .flat_map(|(range, values)| range.clone().zip(values.iter().copied()))
                .collect::<Vec<_>>();
            let weights = RootWeightRanges::new(32768, ranges).unwrap();
            assert_eq!(weights.term_count(), 2 * len);
            let oracle = NativeRootWeights::new(32768, expected.clone(), E::zero()).unwrap();
            for point in [
                vec![E::zero(); 15],
                vec![E::one(); 15],
                (0..15).map(|i| value(i + 2)).collect(),
            ] {
                assert_eq!(
                    weights.evaluate(&point).unwrap(),
                    oracle.evaluate(&point).unwrap()
                );
            }
            assert!(weights.evaluate(&[E::zero(); 14]).is_err());
            assert_eq!(weights.into_terms().unwrap(), expected);
        }
    }

    #[test]
    fn borrowed_root_ranges_match_explicit_terms() {
        compare::<Prime128Offset275>(Prime128Offset275::from_u64);
        compare::<Ext2<Prime64Offset59>>(|i| {
            Ext2::new(
                Prime64Offset59::from_u64(i),
                Prime64Offset59::from_u64(i + 1),
            )
        });
        let values = [Ext2::new(
            Prime64Offset59::from_u64(3),
            Prime64Offset59::from_u64(5),
        )];
        let weights = RootWeightRanges::new(1, vec![(0..1, &values)]).unwrap();
        assert_eq!(weights.evaluate(&[]).unwrap(), values[0]);
        assert_eq!(
            RootWeightRanges::<Prime64Offset59>::new(1, vec![])
                .unwrap()
                .evaluate(&[])
                .unwrap(),
            Prime64Offset59::zero()
        );
    }

    #[test]
    fn borrowed_root_ranges_reject_invalid_geometry() {
        let values = [Prime64Offset59::from_u64(1); 2];
        for (table_len, range, values) in [
            (0, 0..0, &values[..0]),
            (3, 0..2, &values[..]),
            (8, Range { start: 3, end: 2 }, &values[..0]),
            (8, 7..9, &values[..]),
            (8, 0..1, &values[..]),
            (8, usize::MAX..usize::MAX, &values[..0]),
        ] {
            assert!(RootWeightRanges::new(table_len, vec![(range, values)]).is_err());
        }
    }
}
