use super::BinarySourceDescriptor;
use akita_algebra::eq_poly::EqPolynomial;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BinaryFieldProfile;
use jolt_field::{CanonicalEncoding, Ext2, ExtField, Field, Prime64Offset59, Ring};

pub(super) type E64 = Ext2<Prime64Offset59>;

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

/// One base-coordinate evaluation plane, in point/polynomial/coordinate order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeCoordinatePlane {
    pub point: usize,
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

    /// Native row, column and coefficient tensors for the two field identities.
    #[allow(clippy::type_complexity)]
    pub(super) fn field_weights<F: Field, E: ExtField<F>>(
        &self,
        descriptor: &BinarySourceDescriptor,
    ) -> Result<(Vec<E>, Vec<E>, Vec<E>), AkitaError> {
        let coordinates = self.weights::<F>(descriptor)?;
        if coordinates.len() > E::DEGREE {
            return Err(AkitaError::InvalidProof);
        }
        if coordinates.len() == 2 {
            // P64 coordinates are in the canonical E64 basis, even when E is
            // supplied through the generic extension-field interface.
            let basis = E::from_base_fn(|i| F::from_bool(i == 1));
            if basis.square() != E::lift_base(F::from_u64(2)) {
                return Err(AkitaError::InvalidProof);
            }
        }
        let (rows, columns, omega) = coordinates.first().ok_or(AkitaError::InvalidProof)?;
        let mut native = (
            vec![E::zero(); rows.len()],
            vec![E::zero(); columns.len()],
            vec![E::zero(); omega.len()],
        );
        for (coordinate, (rows, columns, omega)) in coordinates.iter().enumerate() {
            let basis = E::from_base_fn(|i| F::from_bool(i == coordinate));
            for (target, source) in [
                (&mut native.0, rows),
                (&mut native.1, columns),
                (&mut native.2, omega),
            ] {
                for (target, &value) in target.iter_mut().zip(source) {
                    *target += basis * E::lift_base(value);
                }
            }
        }
        Ok(native)
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
        let mut owners = std::collections::BTreeSet::new();
        for claim in &self.evaluations {
            if claim.polynomial >= descriptor.polynomials().len() || !claim.value.validate(field) {
                return Err(AkitaError::InvalidInput(
                    "invalid prime owner or value".into(),
                ));
            }
            if !owners.insert(claim.polynomial) {
                return Err(AkitaError::InvalidInput(
                    "duplicate prime polynomial value".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Checked points, with planes ordered by point, polynomial and coordinate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimePlaneManifest {
    claims: Vec<PrimeGroupClaim>,
    field: BinaryFieldProfile,
    planes: Vec<PrimeCoordinatePlane>,
}
impl PrimePlaneManifest {
    /// P128 accepts at most one point; P64 accepts at most two distinct E64 points.
    pub fn with_field_profile(
        source: &BinarySourceDescriptor,
        claims: impl IntoIterator<Item = PrimeGroupClaim>,
        field: BinaryFieldProfile,
    ) -> Result<Self, AkitaError> {
        let mut claims = claims
            .into_iter()
            .take(field.maximum_prime_points() + 1)
            .collect::<Vec<_>>();
        if claims.len() > field.maximum_prime_points() {
            return Err(AkitaError::InvalidInput("too many prime points".into()));
        }
        for claim in &mut claims {
            claim.validate(source, field)?;
            claim
                .evaluations
                .sort_unstable_by_key(|value| value.polynomial);
        }
        if claims.len() == 2 && claims[0].point == claims[1].point {
            return Err(AkitaError::InvalidInput("duplicate prime point".into()));
        }
        let mut planes = Vec::new();
        let capacity = checked::product([
            checked::sum(claims.iter().map(|claim| claim.evaluations.len())).ok_or(
                AkitaError::InvalidInput("prime plane allocation too large".into()),
            )?,
            field.extension_degree(),
        ])
        .ok_or(AkitaError::InvalidInput(
            "prime plane allocation too large".into(),
        ))?;
        planes
            .try_reserve_exact(capacity)
            .map_err(|_| AkitaError::InvalidInput("prime plane allocation too large".into()))?;
        for (point, claim) in claims.iter().enumerate() {
            for claim in &claim.evaluations {
                for (coordinate, &value) in claim.value.coordinates().iter().enumerate() {
                    planes.push(PrimeCoordinatePlane {
                        point,
                        polynomial: claim.polynomial,
                        coordinate,
                        value,
                    });
                }
            }
        }
        Ok(Self {
            claims,
            field,
            planes,
        })
    }

    /// Revalidate against the commitment's descriptor without copying the claims.
    pub fn validate(&self, source: &BinarySourceDescriptor) -> Result<(), AkitaError> {
        for claim in &self.claims {
            claim.validate(source, self.field)?;
        }
        Ok(())
    }

    pub fn claims(&self) -> &[PrimeGroupClaim] {
        &self.claims
    }

    pub fn field_profile(&self) -> BinaryFieldProfile {
        self.field
    }

    pub fn planes(&self) -> &[PrimeCoordinatePlane] {
        &self.planes
    }

    /// Decode bounded point records and require their exact canonical encoding.
    pub fn from_bytes(
        source: &BinarySourceDescriptor,
        field: BinaryFieldProfile,
        bytes: &[u8],
    ) -> Result<Self, akita_serialization::SerializationError> {
        use akita_serialization::SerializationError;
        use std::io::{Cursor, Read};
        let bad = || SerializationError::InvalidData("noncanonical prime manifest".into());
        let two =
            field == BinaryFieldProfile::P64 && bytes.starts_with(b"AkitaPrimePlanes/p64\x05");
        let tag: &[u8] = match (field, two) {
            (BinaryFieldProfile::P128, _) => b"AkitaPrimePlanes\x02",
            (BinaryFieldProfile::P64, false) => b"AkitaPrimePlanes/p64\x04",
            (BinaryFieldProfile::P64, true) => b"AkitaPrimePlanes/p64\x05",
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
        let read_element = |reader: &mut Cursor<&[u8]>| {
            if field == BinaryFieldProfile::P128 {
                let mut bytes = [0; 16];
                reader.read_exact(&mut bytes)?;
                Ok::<PrimeClaimValue, SerializationError>(u128::from_le_bytes(bytes).into())
            } else {
                Ok([u128::from(read_u64(reader)?), u128::from(read_u64(reader)?)].into())
            }
        };
        let mut points = Vec::new();
        for _ in 0..if two { 2 } else { 1 } {
            let count = usize::try_from(read_u64(&mut reader)?).map_err(|_| bad())?;
            if (count != 0 && count != arity) || (two && count == 0) {
                return Err(bad());
            }
            let mut point = Vec::with_capacity(count);
            for _ in 0..count {
                point.push(read_element(&mut reader)?);
            }
            let count_values = usize::try_from(read_u64(&mut reader)?).map_err(|_| bad())?;
            if count_values > source.polynomials().len() || ((count_values == 0) != (count == 0)) {
                return Err(bad());
            }
            let mut evaluations = Vec::new();
            evaluations
                .try_reserve_exact(count_values)
                .map_err(|_| bad())?;
            for _ in 0..count_values {
                evaluations.push(PrimePlaneClaim {
                    polynomial: usize::try_from(read_u64(&mut reader)?).map_err(|_| bad())?,
                    value: read_element(&mut reader)?,
                });
            }
            if count_values != 0 {
                points.push(PrimeGroupClaim { point, evaluations });
            }
        }
        let manifest = Self::with_field_profile(source, points, field).map_err(|_| bad())?;
        if manifest.canonical_bytes() != bytes {
            return Err(bad());
        }
        Ok(manifest)
    }

    /// Keep the single-point encoding; the two-point tag binds two ordered records.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = match (self.field, self.claims.len()) {
            (BinaryFieldProfile::P128, _) => b"AkitaPrimePlanes\x02".to_vec(),
            (BinaryFieldProfile::P64, 2) => b"AkitaPrimePlanes/p64\x05".to_vec(),
            (BinaryFieldProfile::P64, _) => b"AkitaPrimePlanes/p64\x04".to_vec(),
        };
        let coordinate_bytes = match self.field {
            BinaryFieldProfile::P128 => 16,
            BinaryFieldProfile::P64 => 8,
        };
        if self.claims.is_empty() {
            bytes.extend_from_slice(&[0; 16]);
        }
        for claim in &self.claims {
            bytes.extend_from_slice(&(claim.point.len() as u64).to_le_bytes());
            for coordinate in &claim.point {
                for coordinate in coordinate.coordinates() {
                    bytes.extend_from_slice(&coordinate.to_le_bytes()[..coordinate_bytes]);
                }
            }
            bytes.extend_from_slice(&(claim.evaluations.len() as u64).to_le_bytes());
            for claim in &claim.evaluations {
                bytes.extend_from_slice(&(claim.polynomial as u64).to_le_bytes());
                for value in claim.value.coordinates() {
                    bytes.extend_from_slice(&value.to_le_bytes()[..coordinate_bytes]);
                }
            }
        }
        bytes
    }
}

/// Native E64 tensor of the public claim point, before coordinate extraction.
/// The same constructor supplies EP row weights and public reading products.
pub(super) fn e64_tensor(point: &[PrimeClaimValue]) -> Result<Vec<E64>, AkitaError> {
    let point = point
        .iter()
        .map(|value| {
            if !value.validate(BinaryFieldProfile::P64) {
                return Err(AkitaError::InvalidInput(
                    "noncanonical E64 prime point".into(),
                ));
            }
            let PrimeClaimValue::Extension([c0, c1]) = *value else {
                return Err(AkitaError::InvalidProof);
            };
            Ok(E64::from_base_fn(|i| {
                Prime64Offset59::from_u128(if i == 0 { c0 } else { c1 })
            }))
        })
        .collect::<Result<Vec<_>, _>>()?;
    EqPolynomial::evals(&point)
}

fn tensor<E: Field>(
    point: &[PrimeClaimValue],
    field: BinaryFieldProfile,
) -> Result<Vec<Vec<E>>, AkitaError> {
    match field {
        BinaryFieldProfile::P128 => {
            let point = point
                .iter()
                .map(|value| match value {
                    PrimeClaimValue::Base(value) => Ok(E::from_u128(*value)),
                    PrimeClaimValue::Extension(_) => Err(AkitaError::InvalidProof),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(vec![EqPolynomial::evals(&point)?])
        }
        BinaryFieldProfile::P64 => {
            let weights = e64_tensor(point)?;
            (0..2)
                .map(|coordinate| {
                    weights
                        .iter()
                        .map(|value| {
                            value
                                .base_coefficient(coordinate)
                                .to_u128_checked()
                                .map(E::from_u128)
                                .ok_or(AkitaError::InvalidProof)
                        })
                        .collect()
                })
                .collect()
        }
    }
}

#[cfg(test)]
#[path = "prime_tests.rs"]
mod tests;
