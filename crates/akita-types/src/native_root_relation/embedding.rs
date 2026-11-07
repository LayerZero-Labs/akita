//! Aligned embeddings preserve a local oracle's exact multilinear extension.

use super::{NativeRootWeights, RootWeightFamily};
use akita_error::{checked, AkitaError};
use jolt_field::Field;

#[derive(Clone, Debug)]
pub(crate) struct EmbeddedWeights<E: Field> {
    pub(super) local: Box<NativeRootWeights<E>>,
    pub(super) offset: usize,
    table_len: usize,
}

impl<E: Field> NativeRootWeights<E> {
    /// Translate complete local oracles, including their structured families.
    /// Each local power-of-two table occupies an aligned, disjoint block.
    pub fn concatenate(
        table_len: usize,
        components: impl IntoIterator<Item = (usize, Self)>,
    ) -> Result<Self, AkitaError> {
        if !table_len.is_power_of_two() {
            return Err(AkitaError::InvalidProof);
        }
        let mut families = Vec::new();
        let mut claim = E::zero();
        let mut end = 0;
        for (offset, local) in components {
            let range = checked::range(offset, local.table_len).ok_or(AkitaError::InvalidProof)?;
            if offset < end || !offset.is_multiple_of(local.table_len) || range.end > table_len {
                return Err(AkitaError::InvalidInput(
                    "overlapping root weight blocks".into(),
                ));
            }
            end = range.end;
            claim += local.claim;
            families
                .try_reserve(1)
                .map_err(|_| AkitaError::InvalidProof)?;
            families.push(RootWeightFamily::Embedded(EmbeddedWeights {
                local: Box::new(local),
                offset,
                table_len,
            }));
        }
        Self::structured(table_len, Vec::new(), families, claim)
    }
}

impl<E: Field> EmbeddedWeights<E> {
    pub(super) fn table_len(&self) -> usize {
        self.table_len
    }

    pub(super) fn evaluate(&self, point: &[E]) -> Result<E, AkitaError> {
        let local_vars = usize::try_from(self.local.table_len.trailing_zeros())
            .map_err(|_| AkitaError::InvalidProof)?;
        let low = point.get(..local_vars).ok_or(AkitaError::InvalidProof)?;
        let high = point.get(local_vars..).ok_or(AkitaError::InvalidProof)?;
        let mut address = self.offset / self.local.table_len;
        let mut factor = E::one();
        for &coordinate in high {
            factor *= if address & 1 == 0 {
                E::one() - coordinate
            } else {
                coordinate
            };
            address >>= 1;
        }
        if address != 0 {
            return Err(AkitaError::InvalidProof);
        }
        Ok(factor * self.local.evaluate(low)?)
    }

    // Erase the visitor type here so nested embeddings reuse one instantiation.
    pub(super) fn visit(
        &self,
        visit: &mut dyn FnMut(usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        let mut translated = |address, value| {
            let address = checked::sum([self.offset, address]).ok_or(AkitaError::InvalidProof)?;
            visit(address, value)
        };
        for &(address, value) in &self.local.terms {
            translated(address, value)?;
        }
        for family in &self.local.families {
            family.visit(&mut translated)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Ext2, ExtField, One, Prime64Offset59 as F, Ring, Zero};

    #[test]
    fn nested_root_embeddings_preserve_weights_and_visitor_errors() {
        let leaf = NativeRootWeights::new(4, vec![(1, F::from_u64(3))], F::from_u64(5)).unwrap();
        let inner = NativeRootWeights::concatenate(16, [(4, leaf)]).unwrap();
        let outer = NativeRootWeights::concatenate(64, [(32, inner)]).unwrap();
        let expected =
            NativeRootWeights::new(64, vec![(37, F::from_u64(3))], F::from_u64(5)).unwrap();
        assert_eq!(outer.claim(), expected.claim());
        assert_eq!(
            outer.materialize().unwrap(),
            expected.materialize().unwrap()
        );
        assert_eq!(
            outer.profiles(2).unwrap().take_dense(),
            expected.materialize().unwrap()
        );
        let point = [2, 3, 5, 7, 11, 13].map(F::from_u64);
        assert_eq!(
            outer.evaluate(&point).unwrap(),
            expected.evaluate(&point).unwrap()
        );
        assert!(matches!(
            outer.families[0].visit(|_, _| Err(AkitaError::InvalidProof)),
            Err(AkitaError::InvalidProof)
        ));
    }

    #[test]
    fn translated_root_weights_preserve_dense_and_mle_evaluations() {
        type E = Ext2<F>;
        let ext = |value| E::from_base_fn(|coordinate| F::from_u64(value + coordinate as u64));
        let a = NativeRootWeights::new(4, vec![(1, ext(3))], E::zero()).unwrap();
        let b = NativeRootWeights::new(8, vec![(6, ext(7))], E::zero()).unwrap();
        let joined = NativeRootWeights::concatenate(32, [(0, a), (16, b)]).unwrap();
        let table = joined.materialize().unwrap();
        assert_eq!(table[1], ext(3));
        assert_eq!(table[22], ext(7));
        let point = [2, 3, 5, 7, 11].map(ext);
        let eq = akita_algebra::eq_poly::EqPolynomial::evals(&point).unwrap();
        let dense = table
            .iter()
            .zip(eq)
            .fold(E::zero(), |sum, (&w, e)| sum + w * e);
        assert_eq!(joined.evaluate(&point).unwrap(), dense);
        let nested = NativeRootWeights::concatenate(128, [(32, joined.clone())]).unwrap();
        // Explicit terms before the embedding force additive overlap; a second
        // embedding also overlaps the first. Neither may overwrite existing rows.
        let overlapping = NativeRootWeights::merge(
            32,
            [
                NativeRootWeights::new(32, vec![(1, ext(19))], E::zero()).unwrap(),
                joined.clone(),
                joined,
            ],
            E::zero(),
        )
        .unwrap();
        for weights in [nested, overlapping] {
            for bits in [0, 1, 3, 5] {
                let mut expected = weights.materialize().unwrap();
                let mut profiles = weights.profiles(bits).unwrap();
                assert_eq!(profiles.clone().take_dense(), expected);
                for round in 0..bits {
                    for left in (0..expected.len()).step_by(2) {
                        assert_eq!(profiles.pair(left), (expected[left], expected[left + 1]));
                    }
                    let challenge = match round % 3 {
                        0 => E::zero(),
                        1 => E::one(),
                        _ => ext(23),
                    };
                    profiles.bind(challenge);
                    akita_algebra::poly::fold_evals_in_place(&mut expected, challenge);
                }
                assert_eq!(profiles.take_dense(), expected);
            }
        }
        let local = NativeRootWeights::new(8, vec![(0, F::from_u64(1))], F::zero()).unwrap();
        assert!(NativeRootWeights::concatenate(32, [(3, local.clone())]).is_err());
        assert!(NativeRootWeights::concatenate(4, [(0, local.clone())]).is_err());
        assert!(NativeRootWeights::concatenate(32, [(8, local.clone()), (8, local)]).is_err());
    }
}
