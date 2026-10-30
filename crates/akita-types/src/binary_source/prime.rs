use super::{BinarySourceDescriptor, PRIME_POINT_CAP};
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BINARY_SETUP_P;
use jolt_field::Field;

/// Number of balanced base-eight digits for a centered prime coefficient.
/// The interval contains `[-(P-1)/2, (P-1)/2]` for `P = 2^128-275`.
pub const PRIME_PLANE_DIGITS: usize = 43;

/// One prime-field tensor-component reading of a committed polynomial.
///
/// Coordinates and value are canonical residues modulo `2^128-275`. The point
/// order is eight low coefficient bits, low carrier-column bits, then scalar-row
/// bits. The tensor component is the caller's decomposition tag in `0..4`;
/// it distinguishes requests but does not change the separable equality weights.
/// One plane reads the full source at its point. No general nonseparable weight
/// functional is inferred from the tag. Virtual coefficients 162..255 are zero.
/// Different polynomials never share a plane; identical requests do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimePlaneClaim {
    pub polynomial: usize,
    pub point: Vec<u128>,
    pub tensor_component: u8,
    pub value: u128,
}

impl PrimePlaneClaim {
    /// Expand the public scalar-row, carrier-column and coefficient weights.
    #[allow(clippy::type_complexity)]
    pub fn weights<E: Field>(
        &self,
        descriptor: &BinarySourceDescriptor,
    ) -> Result<(Vec<E>, Vec<E>, Vec<E>), AkitaError> {
        self.validate(descriptor)?;
        let column_bits = descriptor.index_map().columns().trailing_zeros() as usize;
        let split = checked::sum([8, column_bits]).ok_or(AkitaError::InvalidProof)?;
        let omega = tensor(self.point.get(..8).ok_or(AkitaError::InvalidProof)?)?;
        let columns = tensor(self.point.get(8..split).ok_or(AkitaError::InvalidProof)?)?;
        let rows = tensor(self.point.get(split..).ok_or(AkitaError::InvalidProof)?)?;
        Ok((rows, columns, omega))
    }

    fn validate(&self, descriptor: &BinarySourceDescriptor) -> Result<(), AkitaError> {
        let arity = checked::ceil_log2(descriptor.index_map().padded_carriers())
            .and_then(|n| checked::sum([n, 8]))
            .ok_or(AkitaError::InvalidProof)?;
        if self.polynomial >= descriptor.polynomials().len() || self.tensor_component >= 4 {
            return Err(AkitaError::InvalidInput(
                "invalid prime plane owner or component".into(),
            ));
        }
        if self.point.len() != arity {
            return Err(AkitaError::InvalidInput(
                "prime point dimension differs from the source".into(),
            ));
        }
        if self.value >= BINARY_SETUP_P || self.point.iter().any(|&x| x >= BINARY_SETUP_P) {
            return Err(AkitaError::InvalidInput(
                "noncanonical prime point or value".into(),
            ));
        }
        Ok(())
    }
}

/// Checked assignment of each request to its first-occurrence plane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimePlaneManifest {
    planes: Vec<PrimePlaneClaim>,
    assignments: Vec<usize>,
}
impl PrimePlaneManifest {
    /// Enforce five distinct points and five total polynomial/tensor planes.
    /// Duplicate functionals with different claimed values are invalid input.
    pub fn new(
        source: &BinarySourceDescriptor,
        claims: &[PrimePlaneClaim],
    ) -> Result<Self, AkitaError> {
        if claims.len() > 4096 {
            return Err(AkitaError::InvalidInput(
                "too many prime plane requests".into(),
            ));
        }
        let mut planes: Vec<PrimePlaneClaim> = Vec::new();
        let mut assignments = Vec::new();
        assignments
            .try_reserve_exact(claims.len())
            .map_err(|_| AkitaError::InvalidInput("prime request allocation failed".into()))?;
        for claim in claims {
            claim.validate(source)?;
            if let Some(plane) = planes.iter().position(|p| {
                p.polynomial == claim.polynomial
                    && p.point == claim.point
                    && p.tensor_component == claim.tensor_component
            }) {
                if planes.get(plane).ok_or(AkitaError::InvalidProof)?.value != claim.value {
                    return Err(AkitaError::InvalidInput(
                        "duplicate prime functional has conflicting values".into(),
                    ));
                }
                assignments.push(plane);
            } else {
                if planes.len() == PRIME_POINT_CAP {
                    return Err(AkitaError::InvalidInput(
                        "prime group exceeds its five-plane cap".into(),
                    ));
                }
                assignments.push(planes.len());
                planes.push(claim.clone());
            }
        }
        Ok(Self {
            planes,
            assignments,
        })
    }
    pub fn planes(&self) -> &[PrimePlaneClaim] {
        &self.planes
    }
    pub fn assignments(&self) -> &[usize] {
        &self.assignments
    }

    /// Order-zero encoding of the unique planes, in first-occurrence order.
    /// Duplicate requests do not alter the transcript or authenticated witness.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"AkitaPrimePlanes\x01".to_vec();
        bytes.extend_from_slice(&(self.planes.len() as u64).to_le_bytes());
        for plane in &self.planes {
            bytes.extend_from_slice(&(plane.polynomial as u64).to_le_bytes());
            bytes.extend_from_slice(&(plane.point.len() as u64).to_le_bytes());
            for coordinate in &plane.point {
                bytes.extend_from_slice(&coordinate.to_le_bytes());
            }
            bytes.push(plane.tensor_component);
            bytes.extend_from_slice(&plane.value.to_le_bytes());
        }
        bytes
    }
}

fn tensor<E: Field>(point: &[u128]) -> Result<Vec<E>, AkitaError> {
    let count = checked::pow2(point.len()).ok_or(AkitaError::InvalidProof)?;
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidProof)?;
    weights.push(E::one());
    for &coordinate in point {
        let coordinate = E::from_u128(coordinate);
        let old_len = weights.len();
        for index in 0..old_len {
            let old = *weights.get(index).ok_or(AkitaError::InvalidProof)?;
            *weights.get_mut(index).ok_or(AkitaError::InvalidProof)? =
                old * (E::one() - coordinate);
            weights.push(old * coordinate);
        }
    }
    Ok(weights)
}

#[cfg(test)]
#[path = "prime_tests.rs"]
mod tests;
