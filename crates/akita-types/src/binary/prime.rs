use super::prime_encoding::{append_claim, read_claim};
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
        self.validate_shape(
            prime_point_arity(descriptor)?,
            descriptor.polynomials().len(),
            field,
        )
    }

    pub(super) fn validate_shape(
        &self,
        arity: usize,
        owners: usize,
        field: BinaryFieldProfile,
    ) -> Result<(), AkitaError> {
        if self.evaluations.is_empty() || self.evaluations.len() > owners {
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
        let mut seen = std::collections::BTreeSet::new();
        for claim in &self.evaluations {
            if claim.polynomial >= owners || !claim.value.validate(field) {
                return Err(AkitaError::InvalidInput(
                    "invalid prime owner or value".into(),
                ));
            }
            if !seen.insert(claim.polynomial) {
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
        let bad = || SerializationError::InvalidData("noncanonical prime manifest".into());
        let tag: &[u8] = match field {
            BinaryFieldProfile::P128 => b"AkitaPrimePlanes\x02",
            BinaryFieldProfile::P64 => b"AkitaPrimePlanes/p64\x04",
        };
        let mut reader = bytes.strip_prefix(tag).ok_or_else(bad)?;
        let claim = read_claim(
            &mut reader,
            field,
            prime_point_arity(source).map_err(|_| bad())?,
            source.polynomials().len(),
            None,
        )
        .map_err(|error| match error {
            SerializationError::InvalidData(_) => bad(),
            error => error,
        })?;
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
        append_claim(&mut bytes, self.claim.as_ref(), self.field);
        bytes
    }
}

pub(super) fn prime_point_arity(source: &BinarySourceDescriptor) -> Result<usize, AkitaError> {
    checked::ceil_log2(source.index_map().padded_carriers())
        .and_then(|n| checked::sum([n, 8]))
        .ok_or(AkitaError::InvalidProof)
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

#[cfg(test)]
#[path = "prime_compatibility_tests.rs"]
mod compatibility_tests;
