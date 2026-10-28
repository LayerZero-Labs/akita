//! Binary terminal claims consumed by the native root relations.

use akita_algebra::binary::BinaryField162;
use akita_error::{checked, AkitaError};

use super::BinarySourceDescriptor;

/// A binary terminal from the caller's frontend.
///
/// The future field switch produces this exact interface. Coordinates and value
/// use the unique 21-byte polynomial-basis encoding of GF(2^162). The row point
/// addresses scalar carriers within a column; the column point addresses the
/// low carrier-address bits. The complete claim belongs in transcript order zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryTerminalClaim {
    polynomial: [u8; 32],
    rho_row: Vec<[u8; 21]>,
    rho_col: Vec<[u8; 21]>,
    value: [u8; 21],
}

impl BinaryTerminalClaim {
    pub fn new(
        descriptor: &BinarySourceDescriptor,
        polynomial: [u8; 32],
        rho_row: Vec<[u8; 21]>,
        rho_col: Vec<[u8; 21]>,
        value: [u8; 21],
    ) -> Result<Self, AkitaError> {
        let claim = Self {
            polynomial,
            rho_row,
            rho_col,
            value,
        };
        claim.validate(descriptor)?;
        Ok(claim)
    }

    pub fn polynomial(&self) -> &[u8; 32] {
        &self.polynomial
    }
    pub fn rho_row(&self) -> &[[u8; 21]] {
        &self.rho_row
    }
    pub fn rho_col(&self) -> &[[u8; 21]] {
        &self.rho_col
    }
    pub fn value(&self) -> &[u8; 21] {
        &self.value
    }

    pub fn validate(&self, descriptor: &BinarySourceDescriptor) -> Result<(), AkitaError> {
        let index = descriptor.index_map();
        let rows = checked::exact_div(index.padded_carriers(), index.columns())
            .ok_or(AkitaError::InvalidProof)?;
        if !descriptor.polynomials().contains(&self.polynomial)
            || self.rho_row.len() != rows.trailing_zeros() as usize
            || self.rho_col.len() != index.columns().trailing_zeros() as usize
            || self
                .rho_row
                .iter()
                .chain(&self.rho_col)
                .chain(std::iter::once(&self.value))
                .any(|bytes| BinaryField162::from_bytes(bytes).is_none())
        {
            return Err(AkitaError::InvalidInput(
                "invalid binary terminal claim".into(),
            ));
        }
        Ok(())
    }

    /// Sole transcript encoding, with distinct row and column coordinate lengths.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"AkitaBinaryTerminal\x01".to_vec();
        bytes.extend_from_slice(&self.polynomial);
        for point in [&self.rho_row, &self.rho_col] {
            bytes.extend_from_slice(&(point.len() as u64).to_le_bytes());
            for coordinate in point {
                bytes.extend_from_slice(coordinate);
            }
        }
        bytes.extend_from_slice(&self.value);
        bytes
    }

    /// Expand the public row and column equality tensors directly.
    pub fn equality_weights(
        &self,
        descriptor: &BinarySourceDescriptor,
    ) -> Result<(Vec<BinaryField162>, Vec<BinaryField162>), AkitaError> {
        self.validate(descriptor)?;
        Ok((tensor(&self.rho_row)?, tensor(&self.rho_col)?))
    }
}

fn tensor(point: &[[u8; 21]]) -> Result<Vec<BinaryField162>, AkitaError> {
    let count = checked::pow2(point.len()).ok_or(AkitaError::InvalidProof)?;
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidProof)?;
    weights.push(BinaryField162::ONE);
    for bytes in point {
        let coordinate = BinaryField162::from_bytes(bytes).ok_or(AkitaError::InvalidProof)?;
        let old_len = weights.len();
        for index in 0..old_len {
            let old = *weights.get(index).ok_or(AkitaError::InvalidProof)?;
            *weights.get_mut(index).ok_or(AkitaError::InvalidProof)? =
                old * (BinaryField162::ONE + coordinate);
            weights.push(old * coordinate);
        }
    }
    Ok(weights)
}

/// Canonical integer lift of a binary scalar into the degree-162 scalar ring.
pub fn binary_scalar_lift(value: BinaryField162) -> [i128; 162] {
    let [low, middle, high] = value.to_words();
    std::array::from_fn(|index| {
        let word = match index / 64 {
            0 => low,
            1 => middle,
            _ => high,
        };
        i128::from((word >> (index % 64)) & 1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap};

    #[test]
    fn terminal_claim_validates_shape_and_canonical_bits() {
        let descriptor = BinarySourceDescriptor::new(
            BinaryIndexMap::new(128, 8, 8, 2).unwrap(),
            vec![[1; 32]],
            BinaryCommitmentProfiles {
                a: [2; 32],
                b: [3; 32],
            },
        )
        .unwrap();
        let claim = BinaryTerminalClaim::new(
            &descriptor,
            [1; 32],
            vec![[0; 21]; 2],
            vec![[0; 21]],
            [0; 21],
        )
        .unwrap();
        let (rows, columns) = claim.equality_weights(&descriptor).unwrap();
        assert_eq!(
            rows,
            vec![
                BinaryField162::ONE,
                BinaryField162::ZERO,
                BinaryField162::ZERO,
                BinaryField162::ZERO
            ]
        );
        assert_eq!(columns, vec![BinaryField162::ONE, BinaryField162::ZERO]);
        assert!(BinaryTerminalClaim::new(
            &descriptor,
            [2; 32],
            vec![[0; 21]; 2],
            vec![[0; 21]],
            [0; 21]
        )
        .is_err());
        assert!(BinaryTerminalClaim::new(
            &descriptor,
            [1; 32],
            vec![[0; 21]],
            vec![[0; 21]],
            [0; 21]
        )
        .is_err());
        let mut bad = [0; 21];
        bad[20] = 4;
        assert!(BinaryTerminalClaim::new(
            &descriptor,
            [1; 32],
            vec![[0; 21]; 2],
            vec![[0; 21]],
            bad
        )
        .is_err());
        assert!(BinaryTerminalClaim::new(
            &descriptor,
            [1; 32],
            vec![bad; 2],
            vec![[0; 21]],
            [0; 21]
        )
        .is_err());
    }
}
