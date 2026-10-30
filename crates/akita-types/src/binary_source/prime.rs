use super::{BinarySourceDescriptor, PRIME_POINT_CAP};
use akita_error::AkitaError;

/// One requested base-field prime plane. Point coordinates are canonical
/// residues modulo 2^128-275, in the descriptor's scalar MLE variable order.
/// A rank-r functional lists components 0..r explicitly. Different polynomials
/// never share a plane; exact duplicate requests do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimePlaneClaim {
    pub polynomial: usize,
    pub point: Vec<u128>,
    pub tensor_component: u8,
}
/// Checked assignment of each request to its first-occurrence plane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimePlaneManifest {
    planes: Vec<PrimePlaneClaim>,
    assignments: Vec<usize>,
}
impl PrimePlaneManifest {
    /// Enforce both five distinct points and five total polynomial/tensor planes.
    /// The allocation limit is independent of proof-controlled duplicate counts.
    pub fn new(
        source: &BinarySourceDescriptor,
        claims: &[PrimePlaneClaim],
    ) -> Result<Self, AkitaError> {
        if claims.len() > 4096 {
            return Err(AkitaError::InvalidSize {
                expected: 4096,
                actual: claims.len(),
            });
        }
        let arity = akita_error::checked::ceil_log2(source.index_map().padded_carriers())
            .and_then(|n| n.checked_add(8))
            .ok_or(AkitaError::InvalidProof)?;
        let mut planes: Vec<PrimePlaneClaim> = Vec::new();
        let mut assignments = Vec::with_capacity(claims.len());
        for claim in claims {
            if claim.polynomial >= source.polynomials().len() || claim.tensor_component >= 4 {
                return Err(AkitaError::InvalidInput(
                    "invalid prime plane owner or component".into(),
                ));
            }
            if claim.point.len() != arity {
                return Err(AkitaError::InvalidPointDimension {
                    expected: arity,
                    actual: claim.point.len(),
                });
            }
            if claim.point.iter().any(|&x| x >= u128::MAX - 274) {
                return Err(AkitaError::InvalidInput(
                    "noncanonical prime coordinate".into(),
                ));
            }
            if let Some(plane) = planes.iter().position(|p| p == claim) {
                assignments.push(plane);
            } else {
                if planes.len() == PRIME_POINT_CAP {
                    return Err(AkitaError::InvalidSize {
                        expected: PRIME_POINT_CAP,
                        actual: planes.len() + 1,
                    });
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
}
