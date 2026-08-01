//! Additive compact-geometry relation and restricted-binary terms.

use akita_algebra::UniPoly;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};

pub(crate) struct AdditionalRelationTerms<E: FieldCore> {
    witness: Vec<E>,
    linear_weights: Vec<E>,
    binary_weights: Vec<E>,
    binary_batching: E,
    input_claim: E,
}

impl<E: FieldCore + FromPrimitiveInt> AdditionalRelationTerms<E> {
    pub(crate) fn new(
        compact_witness: &[i8],
        linear_weights: Vec<E>,
        binary_weights: Vec<E>,
        binary_batching: E,
    ) -> Result<Self, AkitaError> {
        if linear_weights.len() != binary_weights.len()
            || !linear_weights.len().is_power_of_two()
            || compact_witness.len() > linear_weights.len()
        {
            return Err(AkitaError::InvalidSize {
                expected: linear_weights.len(),
                actual: compact_witness.len(),
            });
        }
        let mut witness = compact_witness
            .iter()
            .map(|&value| E::from_i64(i64::from(value)))
            .collect::<Vec<_>>();
        witness.resize(linear_weights.len(), E::zero());
        let input_claim = witness
            .iter()
            .zip(&linear_weights)
            .zip(&binary_weights)
            .fold(E::zero(), |sum, ((&w, &linear), &binary)| {
                sum + w * linear + binary_batching * binary * w * (w + E::one())
            });
        Ok(Self {
            witness,
            linear_weights,
            binary_weights,
            binary_batching,
            input_claim,
        })
    }

    pub(crate) fn input_claim(&self) -> E {
        self.input_claim
    }

    pub(crate) fn round_polynomial(&self) -> UniPoly<E> {
        let mut evaluations = [E::zero(); 4];
        for ((w, linear), binary) in self
            .witness
            .chunks_exact(2)
            .zip(self.linear_weights.chunks_exact(2))
            .zip(self.binary_weights.chunks_exact(2))
        {
            let dw = w[1] - w[0];
            let d_linear = linear[1] - linear[0];
            let d_binary = binary[1] - binary[0];
            for (point, evaluation) in evaluations.iter_mut().enumerate() {
                let t = E::from_u64(point as u64);
                let w_t = w[0] + t * dw;
                let linear_t = linear[0] + t * d_linear;
                let binary_t = binary[0] + t * d_binary;
                *evaluation +=
                    w_t * linear_t + self.binary_batching * binary_t * w_t * (w_t + E::one());
            }
        }
        UniPoly::from_evals(&evaluations)
    }

    pub(crate) fn bind(&mut self, challenge: E) {
        fn fold<E: FieldCore>(evaluations: &mut Vec<E>, challenge: E) {
            let half = evaluations.len() / 2;
            for index in 0..half {
                let left = evaluations[2 * index];
                evaluations[index] = left + challenge * (evaluations[2 * index + 1] - left);
            }
            evaluations.truncate(half);
        }
        fold(&mut self.witness, challenge);
        fold(&mut self.linear_weights, challenge);
        fold(&mut self.binary_weights, challenge);
    }

    pub(crate) fn final_claim(&self) -> Result<E, AkitaError> {
        if self.witness.len() != 1
            || self.linear_weights.len() != 1
            || self.binary_weights.len() != 1
        {
            return Err(AkitaError::InvalidProof);
        }
        let witness = self.witness[0];
        Ok(witness * self.linear_weights[0]
            + self.binary_batching * self.binary_weights[0] * witness * (witness + E::one()))
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
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
        ];
        let binary = vec![F::one(), F::one(), F::zero(), F::zero()];
        let rho = F::from_u64(13);
        let claim = witness.iter().zip(&linear).zip(&binary).fold(
            F::zero(),
            |sum, ((&w, &linear), &binary)| {
                let w = F::from_i64(i64::from(w));
                sum + w * linear + rho * binary * w * (w + F::one())
            },
        );
        let mut prover =
            AdditionalRelationTerms::new(&witness, linear, binary, rho).expect("additional terms");
        assert_eq!(prover.input_claim(), claim);
        let polynomial = prover.round_polynomial();
        assert_eq!(
            polynomial.evaluate(&F::zero()) + polynomial.evaluate(&F::one()),
            claim
        );
        let challenge = F::from_u64(17);
        let next_claim = polynomial.evaluate(&challenge);
        prover.bind(challenge);
        let next = prover.round_polynomial();
        assert_eq!(
            next.evaluate(&F::zero()) + next.evaluate(&F::one()),
            next_claim
        );
    }

    #[test]
    fn nonbinary_digit_inside_support_contributes_a_nonzero_constraint() {
        let rho = F::from_u64(13);
        let invalid = AdditionalRelationTerms::new(
            &[2, 0],
            vec![F::zero(); 2],
            vec![F::one(), F::zero()],
            rho,
        )
        .unwrap();
        assert_eq!(invalid.input_claim(), rho * F::from_u64(6));

        let valid = AdditionalRelationTerms::new(
            &[-1, 0],
            vec![F::zero(); 2],
            vec![F::one(), F::zero()],
            rho,
        )
        .unwrap();
        assert_eq!(valid.input_claim(), F::zero());
    }
}
