//! Word evaluations over GF(2^128), before the carrier-level field switch.

use akita_algebra::binary::field_switch::{SwitchField, SwitchPartials};
use akita_algebra::binary::BinaryField128;
use akita_error::{checked, AkitaError};

use super::BinarySourceDescriptor;

/// One logical word MLE evaluation, in canonical low-degree-first H encodings.
///
/// H is GF(2^128), with modulus `X^128 + X^7 + X^2 + X + 1`.
/// The descriptor fixes the point split: low coordinates select a word within
/// a carrier, and the remaining coordinates select the carrier address.
/// A point may use either `ceil(log2(logical_len))` coordinates or the complete
/// padded address space. Missing high coordinates in a logical point are zero;
/// the public claim retains exactly the coordinates supplied by the caller.
/// Applications reduce multiple points on one polynomial before opening;
/// an opening rejects a second word claim for the same polynomial identifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryWordClaim {
    polynomial: [u8; 32],
    point: Vec<[u8; 16]>,
    value: [u8; 16],
}

impl BinaryWordClaim {
    /// Validate the polynomial identity and logical or complete padded point shape.
    /// Every 16-byte polynomial-basis encoding is a unique element of H.
    pub fn new(
        descriptor: &BinarySourceDescriptor,
        polynomial: [u8; 32],
        point: Vec<[u8; 16]>,
        value: [u8; 16],
    ) -> Result<Self, AkitaError> {
        let claim = Self {
            polynomial,
            point,
            value,
        };
        claim.validate(descriptor)?;
        Ok(claim)
    }

    pub fn polynomial(&self) -> &[u8; 32] {
        &self.polynomial
    }

    pub fn point(&self) -> &[[u8; 16]] {
        &self.point
    }

    pub fn value(&self) -> &[u8; 16] {
        &self.value
    }

    pub fn validate(&self, descriptor: &BinarySourceDescriptor) -> Result<(), AkitaError> {
        let split = descriptor.index_map().claim_variable_split();
        let expected = checked::sum([split.intra_carrier, split.carrier_address])
            .ok_or_else(|| AkitaError::InvalidInput("binary word point size overflow".into()))?;
        let logical = checked::ceil_log2(descriptor.index_map().logical_len())
            .ok_or_else(|| AkitaError::InvalidInput("binary logical point size overflow".into()))?;
        if !descriptor.polynomials().contains(&self.polynomial)
            || (self.point.len() != logical && self.point.len() != expected)
        {
            return Err(AkitaError::InvalidInput(
                "invalid binary word claim identity or coordinate split".into(),
            ));
        }
        Ok(())
    }

    /// Validate the group-wide one-terminal-per-polynomial contract.
    /// Each polynomial keeps its own field switch and authenticated source view.
    pub fn validate_group(
        descriptor: &BinarySourceDescriptor,
        claims: &[Self],
    ) -> Result<(), AkitaError> {
        if claims.len() > descriptor.polynomials().len() {
            return Err(AkitaError::InvalidInput(
                "one binary word claim per polynomial; reduce multiple points before opening"
                    .into(),
            ));
        }
        let mut owners = std::collections::BTreeSet::new();
        for claim in claims {
            claim.validate(descriptor)?;
            if !owners.insert(*claim.polynomial()) {
                return Err(AkitaError::InvalidInput(
                    "one binary word claim per polynomial; reduce multiple points before opening"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    /// Sole order-zero encoding, binding the identifier, point arity and value.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"AkitaBinaryWord\x01".to_vec();
        bytes.extend_from_slice(&self.polynomial);
        bytes.extend_from_slice(&(self.point.len() as u64).to_le_bytes());
        for coordinate in &self.point {
            bytes.extend_from_slice(coordinate);
        }
        bytes.extend_from_slice(&self.value);
        bytes
    }

    /// Carrier-address point, after removing the descriptor's word-slot axes.
    /// Missing high logical axes are fixed to zero in the padded carrier table.
    pub fn carrier_point(
        &self,
        descriptor: &BinarySourceDescriptor,
    ) -> Result<Vec<BinaryField128>, AkitaError> {
        self.validate(descriptor)?;
        let split = descriptor.index_map().claim_variable_split();
        let mut point = Vec::new();
        point
            .try_reserve_exact(split.carrier_address)
            .map_err(|_| AkitaError::InvalidInput("binary word point allocation failed".into()))?;
        for axis in 0..split.carrier_address {
            let logical_axis = checked::sum([split.intra_carrier, axis]).ok_or_else(|| {
                AkitaError::InvalidInput("binary word point size overflow".into())
            })?;
            point.push(
                self.point
                    .get(logical_axis)
                    .map_or(BinaryField128::ZERO, decode),
            );
        }
        Ok(point)
    }

    pub fn decoded_value(&self) -> BinaryField128 {
        decode(&self.value)
    }

    /// Reconstruct `sum_k X^k L(q[k])` from ordinary carrier partials.
    ///
    /// For width ell, `L(x) = sum_s bit_s(x) eq_H(r_intra, s/ell) X^(s%ell)`.
    /// L is F2-linear and acts only in this public bit-weight check. The
    /// authenticated binary terminal remains a whole-carrier evaluation.
    pub fn reconstructed_value(
        &self,
        descriptor: &BinarySourceDescriptor,
        partials: &SwitchPartials<BinaryField128>,
    ) -> Result<BinaryField128, AkitaError> {
        self.validate(descriptor)?;
        let index = descriptor.index_map();
        let split = index.claim_variable_split();
        let basis = BinaryField128::ONE.basis_products();
        let mut coefficients = [BinaryField128::ZERO; 128];
        for (s, coefficient) in coefficients.iter_mut().enumerate() {
            let slot = s / index.word_width();
            let weight = (0..split.intra_carrier).fold(BinaryField128::ONE, |weight, axis| {
                let r = self.point.get(axis).map_or(BinaryField128::ZERO, decode);
                weight
                    * if (slot >> axis) & 1 == 0 {
                        BinaryField128::ONE + r
                    } else {
                        r
                    }
            });
            *coefficient = weight
                * *basis.get(s % index.word_width()).ok_or_else(|| {
                    AkitaError::Internal("validated binary word basis missing".into())
                })?;
        }
        Ok(partials.values().iter().zip(basis).fold(
            BinaryField128::ZERO,
            |sum, (&partial, basis)| {
                let image = coefficients.iter().enumerate().fold(
                    BinaryField128::ZERO,
                    |image, (bit, &coefficient)| {
                        if (partial >> bit) & 1 == 1 {
                            image + coefficient
                        } else {
                            image
                        }
                    },
                );
                sum + basis * image
            },
        ))
    }
}

fn decode(bytes: &[u8; 16]) -> BinaryField128 {
    let bits = u128::from_le_bytes(*bytes);
    BinaryField128::from_words([bits as u64, (bits >> 64) as u64])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap};
    use akita_algebra::binary::field_switch::partial_evaluations;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn descriptor(ell: usize, carriers: usize, columns: usize) -> BinarySourceDescriptor {
        BinarySourceDescriptor::new(
            BinaryIndexMap::new(ell, carriers * (128 / ell), carriers, columns).unwrap(),
            vec![[1; 32]],
            BinaryCommitmentProfiles {
                a: [2; 32],
                b: [3; 32],
            },
        )
        .unwrap()
    }

    #[test]
    fn field_switch_word_claim_validates_descriptor_split_and_identity() {
        let descriptor = descriptor(8, 8, 2);
        let claim = BinaryWordClaim::new(&descriptor, [1; 32], vec![[0; 16]; 7], [0; 16]).unwrap();
        assert_eq!(claim.carrier_point(&descriptor).unwrap().len(), 3);
        assert!(BinaryWordClaim::new(&descriptor, [2; 32], vec![[0; 16]; 7], [0; 16]).is_err());
        assert!(BinaryWordClaim::new(&descriptor, [1; 32], vec![[0; 16]; 6], [0; 16]).is_err());
    }

    #[test]
    fn field_switch_linear_word_map_matches_independent_word_mle() {
        let mut rng = StdRng::seed_from_u64(0x61978b);
        for ell in [1, 8, 32, 128] {
            for (carriers, columns) in [(8usize, 1), (8, 2), (16, 4)] {
                let descriptor = descriptor(ell, carriers, columns);
                let split = descriptor.index_map().claim_variable_split();
                let point: Vec<_> = (0..split.intra_carrier + split.carrier_address)
                    .map(|_| rng.gen::<u128>().to_le_bytes())
                    .collect();
                let source: Vec<_> = (0..carriers).map(|_| rng.gen::<u128>()).collect();
                let mut words = Vec::new();
                for &carrier in &source {
                    for slot in 0..128 / ell {
                        let mask = u128::MAX >> (128 - ell);
                        words.push(decode(&((carrier >> (slot * ell)) & mask).to_le_bytes()));
                    }
                }
                // Independent pairwise folding of logical words, low axis first.
                for coordinate in &point {
                    let r = decode(coordinate);
                    words = words
                        .chunks_exact(2)
                        .map(|pair| pair[0] * (BinaryField128::ONE + r) + pair[1] * r)
                        .collect();
                }
                let claim = BinaryWordClaim::new(&descriptor, [1; 32], point, [0; 16]).unwrap();
                let partials = partial_evaluations(
                    &source,
                    &claim.carrier_point(&descriptor).unwrap(),
                    &mut Vec::new(),
                )
                .unwrap();
                assert_eq!(
                    claim.reconstructed_value(&descriptor, &partials).unwrap(),
                    words[0]
                );
            }
        }
    }

    #[test]
    fn logical_points_match_small_zero_padded_word_mles() {
        let mut rng = StdRng::seed_from_u64(0x05a1_110c);
        for ell in [1usize, 2, 4, 8, 16, 32, 64, 128] {
            let per_carrier = 128 / ell;
            for logical_len in [1, 3, per_carrier, per_carrier + 1, 5 * per_carrier + 1] {
                let descriptor = BinarySourceDescriptor::new(
                    BinaryIndexMap::new(ell, logical_len, 8, 2).unwrap(),
                    vec![[1; 32]],
                    BinaryCommitmentProfiles {
                        a: [2; 32],
                        b: [3; 32],
                    },
                )
                .unwrap();
                let split = descriptor.index_map().claim_variable_split();
                let padded_log =
                    checked::sum([split.intra_carrier, split.carrier_address]).unwrap();
                let logical_log = checked::ceil_log2(logical_len).unwrap();
                let point = (0..logical_log)
                    .map(|_| rng.gen::<u128>().to_le_bytes())
                    .collect::<Vec<_>>();
                let words = (0..logical_len)
                    .map(|_| rng.gen::<u128>() & (u128::MAX >> (128 - ell)))
                    .collect::<Vec<_>>();
                // Evaluate the caller's logical table independently, without
                // adding the source ring's physical padding dimensions.
                let mut mle = words
                    .iter()
                    .map(|word| decode(&word.to_le_bytes()))
                    .collect::<Vec<_>>();
                mle.resize(checked::pow2(logical_log).unwrap(), BinaryField128::ZERO);
                for coordinate in &point {
                    let r = decode(coordinate);
                    mle = mle
                        .chunks_exact(2)
                        .map(|pair| pair[0] * (BinaryField128::ONE + r) + pair[1] * r)
                        .collect();
                }
                assert_eq!(mle.len(), 1);
                let mut carriers = vec![0u128; 8];
                for (word, value) in words.iter().enumerate() {
                    carriers[word / per_carrier] |= value << ((word % per_carrier) * ell);
                }
                let logical_claim =
                    BinaryWordClaim::new(&descriptor, [1; 32], point.clone(), [0; 16]).unwrap();
                assert_eq!(logical_claim.point(), &point);
                let carrier_point = logical_claim.carrier_point(&descriptor).unwrap();
                assert_eq!(carrier_point.len(), split.carrier_address);
                let partials =
                    partial_evaluations(&carriers, &carrier_point, &mut Vec::new()).unwrap();
                assert_eq!(
                    logical_claim
                        .reconstructed_value(&descriptor, &partials)
                        .unwrap(),
                    mle[0]
                );
                let mut padded_point = point.clone();
                padded_point.resize(padded_log, [0; 16]);
                let padded_claim =
                    BinaryWordClaim::new(&descriptor, [1; 32], padded_point, [0; 16]).unwrap();
                assert_eq!(
                    padded_claim.carrier_point(&descriptor).unwrap(),
                    carrier_point
                );
                assert_eq!(
                    padded_claim
                        .reconstructed_value(&descriptor, &partials)
                        .unwrap(),
                    mle[0]
                );
                for arity in 0..=padded_log + 1 {
                    assert_eq!(
                        BinaryWordClaim::new(&descriptor, [1; 32], vec![[0; 16]; arity], [0; 16])
                            .is_ok(),
                        arity == logical_log || arity == padded_log,
                        "width={ell}, logical_len={logical_len}, arity={arity}",
                    );
                }
            }
        }
    }
}
