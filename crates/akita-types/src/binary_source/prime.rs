use super::BinarySourceDescriptor;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BinaryFieldProfile;
use jolt_field::Field;

/// Number of balanced base-eight digits for a centered prime coefficient.
/// The interval contains `[-(P-1)/2, (P-1)/2]` for `P = 2^128-275`.
pub const PRIME_PLANE_DIGITS: usize = BinaryFieldProfile::P128.prime_plane_digits();

/// One claimed polynomial value at the group's prime-field point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimePlaneClaim {
    pub polynomial: usize,
    pub value: u128,
}

/// One prime-field opening point and at most one value per committed polynomial.
///
/// Coordinates and values are canonical base residues in the selected profile.
/// Extension points and values have no encoding in this API: each claim owns
/// exactly one base-coordinate plane. An extension claim would require two
/// planes and separate resource admission. The point
/// order is eight low coefficient bits, low carrier-column bits, then scalar-row
/// bits. Virtual coefficients 162..255 are zero. Each claimed polynomial owns
/// one evaluation plane, as in an ordinary grouped opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeGroupClaim {
    pub point: Vec<u128>,
    pub evaluations: Vec<PrimePlaneClaim>,
}

impl PrimeGroupClaim {
    /// Expand the shared scalar-row, carrier-column and coefficient weights.
    #[allow(clippy::type_complexity)]
    pub fn weights<E: Field>(
        &self,
        descriptor: &BinarySourceDescriptor,
    ) -> Result<(Vec<E>, Vec<E>, Vec<E>), AkitaError> {
        let field = [BinaryFieldProfile::P128, BinaryFieldProfile::P64]
            .into_iter()
            .find(|field| E::from_u128(field.modulus()) == E::zero())
            .ok_or_else(|| AkitaError::InvalidInput("unsupported prime plane field".into()))?;
        self.validate(descriptor, field)?;
        let column_bits = descriptor.index_map().columns().trailing_zeros() as usize;
        let split = checked::sum([8, column_bits]).ok_or(AkitaError::InvalidProof)?;
        let omega = tensor(self.point.get(..8).ok_or(AkitaError::InvalidProof)?)?;
        let columns = tensor(self.point.get(8..split).ok_or(AkitaError::InvalidProof)?)?;
        let rows = tensor(self.point.get(split..).ok_or(AkitaError::InvalidProof)?)?;
        Ok((rows, columns, omega))
    }

    fn validate(
        &self,
        descriptor: &BinarySourceDescriptor,
        field: BinaryFieldProfile,
    ) -> Result<(), AkitaError> {
        let arity = checked::ceil_log2(descriptor.index_map().padded_carriers())
            .and_then(|n| checked::sum([n, 8]))
            .ok_or(AkitaError::InvalidProof)?;
        if self.evaluations.is_empty() || self.evaluations.len() > descriptor.polynomials().len() {
            return Err(AkitaError::InvalidInput("invalid prime value count".into()));
        }
        if self.point.len() != arity {
            return Err(AkitaError::InvalidInput(
                "prime point dimension differs from the source".into(),
            ));
        }
        if self.point.iter().any(|&x| x >= field.modulus()) {
            return Err(AkitaError::InvalidInput("noncanonical prime point".into()));
        }
        for (index, claim) in self.evaluations.iter().enumerate() {
            if claim.polynomial >= descriptor.polynomials().len() || claim.value >= field.modulus()
            {
                return Err(AkitaError::InvalidInput(
                    "invalid prime owner or value".into(),
                ));
            }
            if self
                .evaluations
                .iter()
                .take(index)
                .any(|earlier| earlier.polynomial == claim.polynomial)
            {
                return Err(AkitaError::InvalidInput(
                    "duplicate prime polynomial value".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Checked single-point statement, with evaluation planes in polynomial order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimePlaneManifest {
    claim: Option<PrimeGroupClaim>,
    field: BinaryFieldProfile,
}
impl PrimePlaneManifest {
    pub fn new(
        source: &BinarySourceDescriptor,
        claim: Option<PrimeGroupClaim>,
    ) -> Result<Self, AkitaError> {
        Self::with_field_profile(source, claim, BinaryFieldProfile::P128)
    }

    pub fn with_field_profile(
        source: &BinarySourceDescriptor,
        mut claim: Option<PrimeGroupClaim>,
        field: BinaryFieldProfile,
    ) -> Result<Self, AkitaError> {
        if let Some(claim) = &mut claim {
            claim.validate(source, field)?;
            claim
                .evaluations
                .sort_unstable_by_key(|value| value.polynomial);
        }
        Ok(Self { claim, field })
    }

    /// Revalidate against the commitment's descriptor without copying the claim.
    pub fn validate(&self, source: &BinarySourceDescriptor) -> Result<(), AkitaError> {
        if let Some(claim) = &self.claim {
            claim.validate(source, self.field)?;
        }
        Ok(())
    }

    pub fn claim(&self) -> Option<&PrimeGroupClaim> {
        self.claim.as_ref()
    }

    pub fn field_profile(&self) -> BinaryFieldProfile {
        self.field
    }

    pub fn planes(&self) -> &[PrimePlaneClaim] {
        self.claim.as_ref().map_or(&[], |claim| &claim.evaluations)
    }

    /// Order zero binds the shared point once, then polynomial/value pairs.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = match self.field {
            BinaryFieldProfile::P128 => b"AkitaPrimePlanes\x02".to_vec(),
            BinaryFieldProfile::P64 => b"AkitaPrimePlanes/p64\x03".to_vec(),
        };
        let coordinate_bytes = match self.field {
            BinaryFieldProfile::P128 => 16,
            BinaryFieldProfile::P64 => 8,
        };
        let point = self
            .claim
            .as_ref()
            .map_or(&[][..], |claim| claim.point.as_slice());
        bytes.extend_from_slice(&(point.len() as u64).to_le_bytes());
        for coordinate in point {
            bytes.extend_from_slice(&coordinate.to_le_bytes()[..coordinate_bytes]);
        }
        bytes.extend_from_slice(&(self.planes().len() as u64).to_le_bytes());
        for plane in self.planes() {
            bytes.extend_from_slice(&(plane.polynomial as u64).to_le_bytes());
            bytes.extend_from_slice(&plane.value.to_le_bytes()[..coordinate_bytes]);
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
