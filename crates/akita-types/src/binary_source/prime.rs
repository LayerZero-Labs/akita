use super::BinarySourceDescriptor;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BinaryFieldProfile;
use jolt_field::Field;

/// Number of balanced base-eight digits for a centered prime coefficient.
/// The interval contains `[-(P-1)/2, (P-1)/2]` for `P = 2^128-275`.
pub const PRIME_PLANE_DIGITS: usize = BinaryFieldProfile::P128.prime_plane_digits();

/// Canonical coordinates of a prime claim element in its paired testing field.
/// P128 uses one base residue; P64 uses `c0 + c1*u`, with `u^2 = 2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimeClaimValue {
    Base(u128),
    Extension([u128; 2]),
}
impl From<u128> for PrimeClaimValue {
    fn from(value: u128) -> Self {
        Self::Base(value)
    }
}
impl From<[u128; 2]> for PrimeClaimValue {
    fn from(value: [u128; 2]) -> Self {
        Self::Extension(value)
    }
}
impl PrimeClaimValue {
    pub fn coordinates(&self) -> &[u128] {
        match self {
            Self::Base(value) => std::slice::from_ref(value),
            Self::Extension(values) => values,
        }
    }
    fn validate(&self, field: BinaryFieldProfile) -> bool {
        self.coordinates().len() == field.extension_degree()
            && self.coordinates().iter().all(|&x| x < field.modulus())
    }
}

/// One base-coordinate evaluation plane, in polynomial/coordinate order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeCoordinatePlane {
    pub polynomial: usize,
    pub coordinate: usize,
    pub value: u128,
}

/// One claimed polynomial value at the group's prime point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimePlaneClaim {
    pub polynomial: usize,
    pub value: PrimeClaimValue,
}

/// One prime opening point and at most one value per committed polynomial.
///
/// P128 points and values are base residues. P64 points and values are E64
/// elements, each represented by two canonical base coordinates. The point
/// order is eight low coefficient bits, low carrier-column bits, then scalar-row
/// bits. Virtual coefficients 162..255 are zero. Each value owns one plane per
/// base coordinate, ordered by polynomial and then coordinate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeGroupClaim {
    pub point: Vec<PrimeClaimValue>,
    pub evaluations: Vec<PrimePlaneClaim>,
}

impl PrimeGroupClaim {
    /// Expand the shared row, column and coefficient tensors into base coordinates.
    /// P128 returns one triple; P64 returns the c0 and c1 triples in that order.
    #[allow(clippy::type_complexity)]
    pub fn weights<E: Field>(
        &self,
        descriptor: &BinarySourceDescriptor,
    ) -> Result<Vec<(Vec<E>, Vec<E>, Vec<E>)>, AkitaError> {
        let field = [BinaryFieldProfile::P128, BinaryFieldProfile::P64]
            .into_iter()
            .find(|field| E::from_u128(field.modulus()) == E::zero())
            .ok_or_else(|| AkitaError::InvalidInput("unsupported prime plane field".into()))?;
        self.validate(descriptor, field)?;
        let column_bits = descriptor.index_map().columns().trailing_zeros() as usize;
        let split = checked::sum([8, column_bits]).ok_or(AkitaError::InvalidProof)?;
        let omega = tensor(self.point.get(..8).ok_or(AkitaError::InvalidProof)?, field)?;
        let columns = tensor(
            self.point.get(8..split).ok_or(AkitaError::InvalidProof)?,
            field,
        )?;
        let rows = tensor(
            self.point.get(split..).ok_or(AkitaError::InvalidProof)?,
            field,
        )?;
        Ok(rows
            .into_iter()
            .zip(columns)
            .zip(omega)
            .map(|((r, c), o)| (r, c, o))
            .collect())
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
        if self.point.iter().any(|x| !x.validate(field)) {
            return Err(AkitaError::InvalidInput("noncanonical prime point".into()));
        }
        for (index, claim) in self.evaluations.iter().enumerate() {
            if claim.polynomial >= descriptor.polynomials().len() || !claim.value.validate(field) {
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
    planes: Vec<PrimeCoordinatePlane>,
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
        let planes = claim
            .as_ref()
            .into_iter()
            .flat_map(|claim| &claim.evaluations)
            .flat_map(|claim| {
                claim
                    .value
                    .coordinates()
                    .iter()
                    .enumerate()
                    .map(|(coordinate, &value)| PrimeCoordinatePlane {
                        polynomial: claim.polynomial,
                        coordinate,
                        value,
                    })
            })
            .collect();
        Ok(Self {
            claim,
            field,
            planes,
        })
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

    pub fn planes(&self) -> &[PrimeCoordinatePlane] {
        &self.planes
    }

    /// Decode only the selected profile's canonical statement encoding.
    /// Counts are bounded by the source before allocating, and trailing bytes reject.
    pub fn from_bytes(
        source: &BinarySourceDescriptor,
        field: BinaryFieldProfile,
        bytes: &[u8],
    ) -> Result<Self, akita_serialization::SerializationError> {
        use akita_serialization::SerializationError;
        use std::io::{Cursor, Read};
        let bad = || SerializationError::InvalidData("noncanonical prime manifest".into());
        let tag: &[u8] = match field {
            BinaryFieldProfile::P128 => b"AkitaPrimePlanes\x02",
            BinaryFieldProfile::P64 => b"AkitaPrimePlanes/p64\x04",
        };
        let mut reader = Cursor::new(bytes.strip_prefix(tag).ok_or_else(bad)?);
        let read_u64 = |reader: &mut Cursor<&[u8]>| -> Result<u64, SerializationError> {
            let mut bytes = [0; 8];
            reader.read_exact(&mut bytes)?;
            Ok(u64::from_le_bytes(bytes))
        };
        let arity = checked::ceil_log2(source.index_map().padded_carriers())
            .and_then(|n| checked::sum([n, 8]))
            .ok_or_else(bad)?;
        let count = usize::try_from(read_u64(&mut reader)?).map_err(|_| bad())?;
        if count != 0 && count != arity {
            return Err(bad());
        }
        let read_element =
            |reader: &mut Cursor<&[u8]>| -> Result<PrimeClaimValue, SerializationError> {
                if field == BinaryFieldProfile::P128 {
                    let mut bytes = [0; 16];
                    reader.read_exact(&mut bytes)?;
                    Ok(u128::from_le_bytes(bytes).into())
                } else {
                    Ok([u128::from(read_u64(reader)?), u128::from(read_u64(reader)?)].into())
                }
            };
        let mut point = Vec::with_capacity(count);
        for _ in 0..count {
            point.push(read_element(&mut reader)?);
        }
        let claims = usize::try_from(read_u64(&mut reader)?).map_err(|_| bad())?;
        if claims > source.polynomials().len() || ((claims == 0) != (count == 0)) {
            return Err(bad());
        }
        let mut evaluations = Vec::with_capacity(claims);
        for _ in 0..claims {
            evaluations.push(PrimePlaneClaim {
                polynomial: usize::try_from(read_u64(&mut reader)?).map_err(|_| bad())?,
                value: read_element(&mut reader)?,
            });
        }
        let claim = (claims != 0).then_some(PrimeGroupClaim { point, evaluations });
        let manifest = Self::with_field_profile(source, claim, field).map_err(|_| bad())?;
        if manifest.canonical_bytes() != bytes {
            return Err(bad());
        }
        Ok(manifest)
    }

    /// Order zero binds the shared point once, then polynomial/value pairs.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = match self.field {
            BinaryFieldProfile::P128 => b"AkitaPrimePlanes\x02".to_vec(),
            BinaryFieldProfile::P64 => b"AkitaPrimePlanes/p64\x04".to_vec(),
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
            for coordinate in coordinate.coordinates() {
                bytes.extend_from_slice(&coordinate.to_le_bytes()[..coordinate_bytes]);
            }
        }
        let claims = self
            .claim
            .as_ref()
            .map_or(&[][..], |claim| claim.evaluations.as_slice());
        bytes.extend_from_slice(&(claims.len() as u64).to_le_bytes());
        for claim in claims {
            bytes.extend_from_slice(&(claim.polynomial as u64).to_le_bytes());
            for value in claim.value.coordinates() {
                bytes.extend_from_slice(&value.to_le_bytes()[..coordinate_bytes]);
            }
        }
        bytes
    }
}

// Coordinate arithmetic in E64, performed over E so the same compiler accepts
// base setup fields and extension sumcheck fields without an integer lift.
fn tensor<E: Field>(
    point: &[PrimeClaimValue],
    field: BinaryFieldProfile,
) -> Result<Vec<Vec<E>>, AkitaError> {
    let count = checked::pow2(point.len()).ok_or(AkitaError::InvalidProof)?;
    let degree = field.extension_degree();
    let mut weights = vec![Vec::new(); degree];
    for coordinate in &mut weights {
        coordinate
            .try_reserve_exact(count)
            .map_err(|_| AkitaError::InvalidProof)?;
        coordinate.push(E::zero());
    }
    weights[0][0] = E::one();
    for coordinate in point {
        let c = coordinate.coordinates();
        let c0 = E::from_u128(*c.first().ok_or(AkitaError::InvalidProof)?);
        let c1 = E::from_u128(c.get(1).copied().unwrap_or(0));
        let old_len = weights[0].len();
        for index in 0..old_len {
            let a = weights[0][index];
            let b = if degree == 2 {
                weights[1][index]
            } else {
                E::zero()
            };
            let product0 = a * c0 + E::from_u64(2) * b * c1;
            weights[0][index] = a - product0;
            weights[0].push(product0);
            if degree == 2 {
                let product1 = a * c1 + b * c0;
                weights[1][index] = b - product1;
                weights[1].push(product1);
            }
        }
    }
    Ok(weights)
}

#[cfg(test)]
#[path = "prime_tests.rs"]
mod tests;
