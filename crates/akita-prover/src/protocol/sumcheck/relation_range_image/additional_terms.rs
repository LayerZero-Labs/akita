//! Sparse compact-geometry relation and restricted-binary terms.

use akita_algebra::UniPoly;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};
use std::collections::BTreeMap;
use std::ops::Range;

#[derive(Clone, Copy)]
struct SparseWeight<E: FieldCore> {
    index: usize,
    linear: E,
    binary: E,
}

/// Sparse Stage-2 addend over the canonical witness table.
///
/// Only the compression relation and negative-binary weights are retained.
/// Witness values are read from `RelationRangeImageProver`'s existing compact
/// or folded table, avoiding a full-domain field copy and keeping the addend's
/// work proportional to its live support.
pub(crate) struct AdditionalRelationTerms<E: FieldCore> {
    weights: Vec<SparseWeight<E>>,
    binary_batching: E,
    input_claim: E,
    domain_len: usize,
}

impl<E: FieldCore + FromPrimitiveInt> AdditionalRelationTerms<E> {
    pub(crate) fn new(
        compact_witness: &[i8],
        domain_len: usize,
        linear_weights: Vec<(usize, E)>,
        binary_intervals: &[Range<usize>],
        binary_batching: E,
    ) -> Result<Self, AkitaError> {
        if !domain_len.is_power_of_two() || compact_witness.len() > domain_len {
            return Err(AkitaError::InvalidSize {
                expected: domain_len,
                actual: compact_witness.len(),
            });
        }
        let mut combined = BTreeMap::<usize, (E, E)>::new();
        for (index, value) in linear_weights {
            if index >= domain_len {
                return Err(AkitaError::InvalidSize {
                    expected: domain_len,
                    actual: index.saturating_add(1),
                });
            }
            combined.entry(index).or_insert((E::zero(), E::zero())).0 += value;
        }
        let mut previous_end = 0usize;
        for interval in binary_intervals {
            if interval.start >= interval.end
                || interval.start < previous_end
                || interval.end > domain_len
            {
                return Err(AkitaError::InvalidInput(
                    "negative-binary support interval is malformed".into(),
                ));
            }
            for index in interval.clone() {
                combined.entry(index).or_insert((E::zero(), E::zero())).1 += E::one();
            }
            previous_end = interval.end;
        }
        let weights = combined
            .into_iter()
            .filter_map(|(index, (linear, binary))| {
                (!linear.is_zero() || !binary.is_zero()).then_some(SparseWeight {
                    index,
                    linear,
                    binary,
                })
            })
            .collect::<Vec<_>>();
        let input_claim = weights.iter().fold(E::zero(), |sum, weight| {
            let witness = compact_witness
                .get(weight.index)
                .map_or_else(E::zero, |&value| E::from_i64(i64::from(value)));
            sum + witness * weight.linear
                + binary_batching * weight.binary * witness * (witness + E::one())
        });
        Ok(Self {
            weights,
            binary_batching,
            input_claim,
            domain_len,
        })
    }

    pub(crate) fn input_claim(&self) -> E {
        self.input_claim
    }

    fn round_polynomial_with(&self, witness_at: impl Fn(usize) -> E) -> UniPoly<E> {
        let mut evaluations = [E::zero(); 4];
        let mut cursor = 0usize;
        while cursor < self.weights.len() {
            let parent = self.weights[cursor].index >> 1;
            let mut linear = [E::zero(); 2];
            let mut binary = [E::zero(); 2];
            while cursor < self.weights.len() && self.weights[cursor].index >> 1 == parent {
                let weight = self.weights[cursor];
                let side = weight.index & 1;
                linear[side] = weight.linear;
                binary[side] = weight.binary;
                cursor += 1;
            }
            let witness = [witness_at(2 * parent), witness_at(2 * parent + 1)];
            let dw = witness[1] - witness[0];
            let d_linear = linear[1] - linear[0];
            let d_binary = binary[1] - binary[0];
            for (point, evaluation) in evaluations.iter_mut().enumerate() {
                let t = E::from_u64(point as u64);
                let witness_t = witness[0] + t * dw;
                let linear_t = linear[0] + t * d_linear;
                let binary_t = binary[0] + t * d_binary;
                *evaluation += witness_t * linear_t
                    + self.binary_batching * binary_t * witness_t * (witness_t + E::one());
            }
        }
        UniPoly::from_evals(&evaluations)
    }

    pub(crate) fn round_polynomial_compact(
        &self,
        compact_witness: &[i8],
        first_challenge: Option<E>,
    ) -> UniPoly<E> {
        self.round_polynomial_with(|index| {
            let compact_value = |source_index| {
                compact_witness
                    .get(source_index)
                    .map_or_else(E::zero, |&value| E::from_i64(i64::from(value)))
            };
            if let Some(challenge) = first_challenge {
                let left = compact_value(2 * index);
                left + challenge * (compact_value(2 * index + 1) - left)
            } else {
                compact_value(index)
            }
        })
    }

    pub(crate) fn round_polynomial_folded(&self, folded_witness: &[E]) -> UniPoly<E> {
        self.round_polynomial_with(|index| {
            folded_witness.get(index).copied().unwrap_or_else(E::zero)
        })
    }

    pub(crate) fn bind(&mut self, challenge: E) {
        let even_scale = E::one() - challenge;
        let mut folded: Vec<SparseWeight<E>> = Vec::with_capacity(self.weights.len());
        for weight in self.weights.drain(..) {
            let scale = if weight.index & 1 == 0 {
                even_scale
            } else {
                challenge
            };
            let parent = weight.index >> 1;
            let linear = scale * weight.linear;
            let binary = scale * weight.binary;
            if let Some(last) = folded.last_mut() {
                if last.index == parent {
                    last.linear += linear;
                    last.binary += binary;
                    continue;
                }
            }
            folded.push(SparseWeight {
                index: parent,
                linear,
                binary,
            });
        }
        folded.retain(|weight| !weight.linear.is_zero() || !weight.binary.is_zero());
        self.weights = folded;
        self.domain_len /= 2;
    }

    pub(crate) fn final_claim(&self, witness: E) -> Result<E, AkitaError> {
        if self.domain_len != 1
            || self.weights.len() > 1
            || self.weights.first().is_some_and(|weight| weight.index != 0)
        {
            return Err(AkitaError::InvalidProof);
        }
        let Some(weight) = self.weights.first() else {
            return Ok(E::zero());
        };
        Ok(witness * weight.linear
            + self.binary_batching * weight.binary * witness * (witness + E::one()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7 as F;

    #[test]
    fn round_polynomial_matches_boolean_sum_and_fold() {
        let witness = [-1, 0, 2, -2];
        let linear = vec![
            (0, F::from_u64(3)),
            (1, F::from_u64(5)),
            (2, F::from_u64(7)),
            (3, F::from_u64(11)),
        ];
        let rho = F::from_u64(13);
        let claim = witness.iter().zip([3, 5, 7, 11]).enumerate().fold(
            F::zero(),
            |sum, (index, (&witness, linear))| {
                let witness = F::from_i64(i64::from(witness));
                let binary = F::from_u64(u64::from(index < 2));
                sum + witness * F::from_u64(linear) + rho * binary * witness * (witness + F::one())
            },
        );
        let binary_interval = 0..2;
        let mut prover = AdditionalRelationTerms::new(
            &witness,
            4,
            linear,
            std::slice::from_ref(&binary_interval),
            rho,
        )
        .unwrap();
        assert_eq!(prover.input_claim(), claim);
        let polynomial = prover.round_polynomial_compact(&witness, None);
        assert_eq!(
            polynomial.evaluate(&F::zero()) + polynomial.evaluate(&F::one()),
            claim
        );
        let challenge = F::from_u64(17);
        let next_claim = polynomial.evaluate(&challenge);
        prover.bind(challenge);
        let next = prover.round_polynomial_compact(&witness, Some(challenge));
        assert_eq!(
            next.evaluate(&F::zero()) + next.evaluate(&F::one()),
            next_claim
        );
    }

    #[test]
    fn nonbinary_digit_inside_support_contributes_a_nonzero_constraint() {
        let rho = F::from_u64(13);
        let binary_interval = 0..1;
        let support = std::slice::from_ref(&binary_interval);
        let invalid = AdditionalRelationTerms::new(&[2, 0], 2, Vec::new(), support, rho).unwrap();
        assert_eq!(invalid.input_claim(), rho * F::from_u64(6));

        let valid = AdditionalRelationTerms::new(&[-1, 0], 2, Vec::new(), support, rho).unwrap();
        assert_eq!(valid.input_claim(), F::zero());
    }
}
