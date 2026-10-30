//! Bounded quotient witness for the exact q24 commitment equation.

use super::*;
use akita_types::binary_source::BinaryQuotientBounds;
use akita_types::native_root_relation::{NativeRootRelation, NativeRootTerm};
use jolt_field::Zero;

/// Construction data for `rem_Psi(A z - sum c T) - q KA = 0`.
///
/// The digits must be committed before relation challenges and authenticated
/// through the shared range, relation and successor-opening checks. This
/// arithmetic object is not a proof or a security-admitted opening profile.
/// No quotient or residual table is a public message.
pub struct BinaryQuotientWitness {
    pub relation: NativeRootRelation<Prime128Offset275>,
    pub digits: Vec<i8>,
    pub bounds: BinaryQuotientBounds,
}

impl BinaryRootRelation {
    /// Add the committed quotient segment to the response/image coefficient map.
    ///
    /// Segments zero and one retain their response and image addresses. Segment
    /// two holds one quotient ring element per polynomial and matrix row. The
    /// native A rows have zero targets and read this same digit table, rather
    /// than trusting the construction residual as a public target.
    pub fn committed_quotient(
        &self,
        descriptor: &BinarySourceDescriptor,
        matrix: &[Vec<[i128; D]>],
        challenges: &Challenges,
        destination_ring_dimension: usize,
    ) -> Result<BinaryQuotientWitness, AkitaError> {
        let (_, columns, source_rows) = dimensions(descriptor, matrix)?;
        let modulus = (-Prime128Offset275::one())
            .to_u128_checked()
            .and_then(|x| x.checked_add(1))
            .ok_or(AkitaError::InvalidProof)?;
        let bounds = BinaryQuotientBounds::new(source_rows, columns, modulus)?;
        // The retained construction fields are public. Revalidate them against
        // the caller's replayed challenges before relying on Gamma or admitting
        // an exact integer lift; a field identity alone cannot do that.
        let canonical = Self::from_response(
            descriptor,
            matrix,
            self.response.clone(),
            &self.images,
            challenges,
            FoldChallengeFamily::BinaryRoot,
        )?;
        if self.embedded_challenges != canonical.embedded_challenges
            || self.rows.len() != canonical.rows.len()
            || self.rows.iter().zip(&canonical.rows).any(|(left, right)| {
                left.remainder != right.remainder || left.quotient != right.quotient
            })
        {
            return Err(invalid(
                "binary quotient construction differs from its replayed relation",
            ));
        }
        let (base, mut digits) = self.successor_table(destination_ring_dimension)?;
        let mut segments = base.segments().to_vec();
        let geometry = segments.first().ok_or(AkitaError::InvalidProof)?.geometry;
        segments.push(RootRepackingSegment {
            geometry,
            elements: self.rows.len(),
            digits: bounds.digit_count(),
        });
        let plan = RootCoefficientRepacking::new(segments, destination_ring_dimension)?;
        // Final domain padding may now be occupied by quotient digits; all
        // existing live response/image addresses are unchanged.
        let additional = plan
            .domain_len()
            .checked_sub(digits.len())
            .ok_or_else(|| invalid("binary quotient domain shrank"))?;
        digits
            .try_reserve_exact(additional)
            .map_err(|_| invalid("binary quotient digit allocation failed"))?;
        digits.resize(plan.domain_len(), 0);
        let offset = i128::try_from(bounds.carry_bound())
            .map_err(|_| invalid("binary quotient offset exceeds signed arithmetic"))?;
        let bits = checked::product([3, bounds.digit_count()])
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| invalid("binary quotient digit width overflow"))?;
        let word_limit = 1u128
            .checked_shl(bits)
            .ok_or_else(|| invalid("binary quotient digit width overflow"))?;
        for (element, row) in self.rows.iter().enumerate() {
            for (coefficient, &residual) in row.remainder.iter().enumerate() {
                let q = i128::from(Q24);
                if residual % q != 0 {
                    return Err(invalid("binary quotient residual is not divisible by q24"));
                }
                let word = (residual / q)
                    .checked_add(offset)
                    .and_then(|x| u128::try_from(x).ok())
                    .filter(|&x| x < word_limit)
                    .ok_or_else(|| invalid("binary quotient exceeds its balanced digit range"))?;
                for digit in 0..bounds.digit_count() {
                    let shift = checked::product([3, digit])
                        .ok_or_else(|| invalid("binary quotient shift overflow"))?;
                    let address = plan.coefficient_address(2, element, digit, coefficient)?;
                    *digits.get_mut(address).ok_or(AkitaError::InvalidProof)? =
                        ((word >> shift) & 7) as i8 - 4;
                }
            }
        }
        let mut group = self.commitment_group::<Prime128Offset275>(descriptor, matrix)?;
        for (element, row) in group.rows.iter_mut().enumerate() {
            row.target.fill(Prime128Offset275::zero());
            let mut multiplier = vec![Prime128Offset275::zero(); D];
            *multiplier.first_mut().ok_or(AkitaError::InvalidProof)? =
                -Prime128Offset275::from_u64(u64::from(Q24));
            row.terms.push(NativeRootTerm {
                segment: 2,
                element,
                multiplier,
            });
        }
        Ok(BinaryQuotientWitness {
            relation: NativeRootRelation::new(plan, vec![group])?,
            digits,
            bounds,
        })
    }
}
