//! Exactly uniform q24 projection rows drawn inside the caller's transcript.

use akita_error::AkitaError;

use akita_transcript::{ProofChannel, ProtocolSiteId, SITE_FAMILY_ROOT_STATEMENT};
use jolt_algebra::checked;
use jolt_algebra::Field;

use super::{challenge_block, BinaryResidualBounds, BINARY_PROJECTION_COUNT, BINARY_SOURCE_Q};

/// Seven independent rows, each covering every polynomial/rank/coefficient residual entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryProjectionRows {
    rows: Vec<Vec<[u32; 648]>>,
    rank: usize,
}

impl BinaryProjectionRows {
    /// Draw after the outgoing commitment and accepted response nonce are fixed.
    /// Every three-byte candidate is accepted iff it is strictly below q.
    pub fn draw(
        channel: &mut impl ProofChannel,
        bounds: &BinaryResidualBounds,
    ) -> Result<Self, AkitaError> {
        if bounds.quotient().is_some() {
            return Err(AkitaError::InvalidInput(
                "quotient profile has no J rows".into(),
            ));
        }
        akita_transcript::public_bytes(
            channel,
            ProtocolSiteId {
                family: SITE_FAMILY_ROOT_STATEMENT,
                stage: 5,
                ..ProtocolSiteId::default()
            },
            b"binary exact divisibility rows",
        )?;
        Self::from_blocks(bounds.residual_rows(), || challenge_block(channel))
    }

    fn from_blocks(
        rank: usize,
        mut block: impl FnMut() -> Result<[u8; 32], AkitaError>,
    ) -> Result<Self, AkitaError> {
        checked::product([BINARY_PROJECTION_COUNT, rank, 648]).ok_or(AkitaError::InvalidProof)?;
        let mut rows = Vec::with_capacity(BINARY_PROJECTION_COUNT);
        let mut bytes = [0; 32];
        let mut offset = 30;
        for _ in 0..BINARY_PROJECTION_COUNT {
            let mut row = Vec::new();
            row.try_reserve_exact(rank)
                .map_err(|_| AkitaError::InvalidProof)?;
            for _ in 0..rank {
                let mut coefficients = [0; 648];
                for coefficient in &mut coefficients {
                    let mut attempts = 0u16;
                    loop {
                        if attempts == 256 {
                            return Err(AkitaError::InvalidProof);
                        }
                        attempts += 1;
                        // Ten disjoint three-byte candidates per block. The final
                        // two bytes are discarded, so candidates never overlap.
                        if offset == 30 {
                            bytes = block()?;
                            offset = 0;
                        }
                        let candidate = bytes
                            .get(offset..offset + 3)
                            .ok_or(AkitaError::InvalidProof)?;
                        let candidate = u32::from_le_bytes([
                            candidate.first().copied().ok_or(AkitaError::InvalidProof)?,
                            candidate.get(1).copied().ok_or(AkitaError::InvalidProof)?,
                            candidate.get(2).copied().ok_or(AkitaError::InvalidProof)?,
                            0,
                        ]);
                        offset += 3;
                        if candidate < BINARY_SOURCE_Q {
                            *coefficient = candidate;
                            break;
                        }
                    }
                }
                row.push(coefficients);
            }
            rows.push(row);
        }
        Ok(Self { rows, rank })
    }

    /// Compute the seven integer dot products, checking the admitted residual.
    pub fn project(
        &self,
        residual: &[[i128; 648]],
        bounds: &BinaryResidualBounds,
    ) -> Result<[i128; BINARY_PROJECTION_COUNT], AkitaError> {
        if bounds.quotient().is_some()
            || residual.len() != self.rank
            || bounds.residual_rows() != self.rank
            || residual
                .iter()
                .flatten()
                .any(|x| x.unsigned_abs() > bounds.residual_bound())
        {
            return Err(AkitaError::InvalidInput(
                "binary projection residual shape or bound".into(),
            ));
        }
        let mut values = [0i128; BINARY_PROJECTION_COUNT];
        for (row, value) in self.rows.iter().zip(&mut values) {
            for (weights, residual) in row.iter().zip(residual) {
                for (&weight, &coefficient) in weights.iter().zip(residual) {
                    *value = coefficient
                        .checked_mul(i128::from(weight))
                        .and_then(|term| value.checked_add(term))
                        .ok_or_else(|| {
                            AkitaError::Internal("binary projection integer overflow".into())
                        })?;
                }
            }
            if value.unsigned_abs() > bounds.projection_bound() {
                return Err(AkitaError::Internal(
                    "binary projection envelope violated".into(),
                ));
            }
        }
        Ok(values)
    }

    /// Form the explicit Stage-2 coefficient weights `rho = sum_l mu_l J_l`.
    pub fn batch<E: Field>(
        &self,
        scalars: &[E; BINARY_PROJECTION_COUNT],
    ) -> Result<Vec<[E; 648]>, AkitaError> {
        let mut result = Vec::new();
        result
            .try_reserve_exact(self.rank)
            .map_err(|_| AkitaError::InvalidProof)?;
        result.resize(self.rank, [E::zero(); 648]);
        for (row, &scalar) in self.rows.iter().zip(scalars) {
            for (result, weights) in result.iter_mut().zip(row) {
                for (result, &weight) in result.iter_mut().zip(weights) {
                    *result += scalar * E::from_u64(u64::from(weight));
                }
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_sampler_exhaustion_rejects_without_looping_forever() {
        let mut calls = 0;
        assert!(BinaryProjectionRows::from_blocks(1, || {
            calls += 1;
            Ok([0xff; 32])
        })
        .is_err());
        assert_eq!(calls, 26);
    }

    #[test]
    fn projection_integer_dot_products_check_shape_and_envelope() {
        let bounds = BinaryResidualBounds::new(1, 2, 1, 1, u128::MAX - 274).unwrap();
        let rows = BinaryProjectionRows::from_blocks(1, || {
            let mut bytes = [0; 32];
            for chunk in bytes.chunks_exact_mut(3) {
                chunk[0] = 1;
            }
            Ok(bytes)
        })
        .unwrap();
        let mut residual = [[0; 648]];
        residual[0][0] = i128::from(BINARY_SOURCE_Q);
        assert_eq!(
            rows.project(&residual, &bounds).unwrap(),
            [i128::from(BINARY_SOURCE_Q); 7]
        );
        assert!(rows.project(&[], &bounds).is_err());
        residual[0][0] = bounds.residual_bound() as i128 + 1;
        assert!(rows.project(&residual, &bounds).is_err());
    }

    #[test]
    fn projection_rows_join_every_polynomial_before_integer_batching() {
        let bounds = BinaryResidualBounds::new(1, 2, 3, 2, u128::MAX - 274).unwrap();
        let rows = BinaryProjectionRows::from_blocks(bounds.residual_rows(), || {
            let mut bytes = [0; 32];
            for chunk in bytes.chunks_exact_mut(3) {
                chunk[0] = 1;
            }
            Ok(bytes)
        })
        .unwrap();
        let mut residual = [[0; 648]; 6];
        residual[0][0] = i128::from(BINARY_SOURCE_Q);
        residual[5][647] = 2 * i128::from(BINARY_SOURCE_Q);
        assert_eq!(
            rows.project(&residual, &bounds).unwrap(),
            [3 * i128::from(BINARY_SOURCE_Q); 7]
        );
        assert!(rows.project(&residual[..3], &bounds).is_err());
    }

    #[test]
    fn projection_sampler_rejects_q_and_byte_range_top() {
        let mut calls = 0;
        let rows = BinaryProjectionRows::from_blocks(1, || {
            calls += 1;
            let mut bytes = [0; 32];
            if calls == 1 {
                for (chunk, value) in bytes.chunks_exact_mut(3).zip([
                    BINARY_SOURCE_Q,
                    0xff_ffff,
                    BINARY_SOURCE_Q - 1,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                ]) {
                    chunk.copy_from_slice(&value.to_le_bytes()[..3]);
                }
            }
            Ok(bytes)
        })
        .unwrap();
        assert_eq!(
            rows.rows[0][0][..8],
            [BINARY_SOURCE_Q - 1, 0, 1, 2, 3, 4, 5, 6]
        );
        assert_eq!(calls, (7usize * 648 + 2).div_ceil(10));
    }
}
