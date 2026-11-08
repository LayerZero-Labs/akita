use super::{BinarySourceDescriptor, PrimeGroupClaim, PrimePlaneManifest};
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BinaryFieldProfile;
use jolt_field::{CanonicalEncoding, Ext2, ExtField, Field, Prime64Offset59, Ring};
use std::sync::Arc;

/// Claim shape selects the encoding and coordinate-plane count, without a flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrimeClaims {
    Base(PrimeGroupClaim),
    /// One or two ordered, distinct E64 points. Each value owns two planes.
    Extension(Vec<PrimeGroupClaim<[u64; 2]>>),
}

impl From<PrimeGroupClaim> for PrimeClaims {
    fn from(claim: PrimeGroupClaim) -> Self {
        Self::Base(claim)
    }
}

impl PrimeClaims {
    pub(super) fn validate(
        &self,
        source: &BinarySourceDescriptor,
        field: BinaryFieldProfile,
    ) -> Result<(), AkitaError> {
        let points = match self {
            Self::Base(claim) => return claim.validate(source, field),
            Self::Extension(points) => points,
        };
        if field != BinaryFieldProfile::P64 || points.is_empty() || points.len() > 2 {
            return Err(AkitaError::InvalidInput(
                "invalid extension prime point count or field".into(),
            ));
        }
        let arity = checked::ceil_log2(source.index_map().padded_carriers())
            .and_then(|n| checked::sum([n, 8]))
            .ok_or(AkitaError::InvalidProof)?;
        for (index, point) in points.iter().enumerate() {
            if point.point.len() != arity
                || point.evaluations.is_empty()
                || point.evaluations.len() > source.polynomials().len()
                || point
                    .point
                    .iter()
                    .flatten()
                    .any(|&x| u128::from(x) >= field.modulus())
                || points
                    .iter()
                    .take(index)
                    .any(|earlier| earlier.point == point.point)
            {
                return Err(AkitaError::InvalidInput(
                    "invalid or duplicate extension prime point".into(),
                ));
            }
            for (index, value) in point.evaluations.iter().enumerate() {
                if value.polynomial >= source.polynomials().len()
                    || value
                        .value
                        .iter()
                        .any(|&x| u128::from(x) >= field.modulus())
                    || point
                        .evaluations
                        .iter()
                        .take(index)
                        .any(|earlier| earlier.polynomial == value.polynomial)
                {
                    return Err(AkitaError::InvalidInput(
                        "invalid extension prime owner or value".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Factored weights of one coordinate plane. Rows are base-coordinate weights;
/// columns and coefficient weights retain both coordinates of the E64 product.
/// Only rows are public for CPU plane construction; reading factors stay internal.
#[derive(Clone, Debug)]
pub struct PrimePlaneFactors<E: Field> {
    pub rows: Arc<[E]>,
    pub(crate) columns: Arc<[[E; 2]]>,
    pub(crate) point_index: usize,
    pub(crate) omega: Arc<[[E; 2]]>,
    /// Coordinate within the reading (zero for a base claim).
    coordinate: usize,
    /// First plane of this value, for locating its independent reading batches.
    first_plane: usize,
    extension: bool,
}

impl PrimePlaneManifest {
    /// Compile public tensors once per point, sharing them across owners.
    /// Coordinate projection precedes later extension-field evaluation.
    #[tracing::instrument(level = "info", skip_all, name = "prime_plane_factors", fields(planes = self.planes().len()))]
    pub fn plane_factors<E: Field>(
        &self,
        source: &BinarySourceDescriptor,
    ) -> Result<Vec<PrimePlaneFactors<E>>, AkitaError> {
        self.validate(source)?;
        if E::from_u128(self.field.modulus()) != E::zero() {
            return Err(AkitaError::InvalidProof);
        }
        let mut factors = Vec::new();
        factors
            .try_reserve_exact(self.planes().len())
            .map_err(|_| AkitaError::InvalidProof)?;
        let Some(claim) = self.claim() else {
            if !self.planes().is_empty() {
                return Err(AkitaError::Internal(
                    "prime planes exist without a claim".into(),
                ));
            }
            return Ok(factors);
        };
        match claim {
            PrimeClaims::Base(claim) => {
                let (rows, columns, mut omega) = claim.weights::<E>(source)?;
                omega.truncate(162);
                let rows: Arc<[E]> = rows.into();
                let columns: Arc<[[E; 2]]> = columns.into_iter().map(|x| [x, E::zero()]).collect();
                let omega: Arc<[[E; 2]]> = omega.into_iter().map(|x| [x, E::zero()]).collect();
                for first_plane in 0..claim.evaluations.len() {
                    factors.push(PrimePlaneFactors {
                        rows: rows.clone(),
                        columns: columns.clone(),
                        omega: omega.clone(),
                        point_index: 0,
                        coordinate: 0,
                        first_plane,
                        extension: false,
                    });
                }
            }
            PrimeClaims::Extension(points) => {
                let column_bits = source.index_map().columns().trailing_zeros() as usize;
                let split = checked::sum([8, column_bits]).ok_or(AkitaError::InvalidProof)?;
                let expand = |point: &[[u64; 2]]| -> Result<Vec<[E; 2]>, AkitaError> {
                    // Project coefficients of the public E64 tensor, not limbs
                    // of its evaluation at a later verifier point.
                    let point = point
                        .iter()
                        .map(|x| {
                            Ext2::<Prime64Offset59>::from_base_fn(|i| {
                                Prime64Offset59::from_u64(if i == 0 { x[0] } else { x[1] })
                            })
                        })
                        .collect::<Vec<_>>();
                    let weights = akita_algebra::eq_poly::EqPolynomial::evals(&point)?;
                    weights
                        .into_iter()
                        .map(|x| {
                            let convert = |i| {
                                x.base_coefficient(i)
                                    .to_u128_checked()
                                    .map(E::from_u128)
                                    .ok_or(AkitaError::InvalidProof)
                            };
                            Ok([convert(0)?, convert(1)?])
                        })
                        .collect()
                };
                for (point_index, point) in points.iter().enumerate() {
                    let rows = expand(point.point.get(split..).ok_or(AkitaError::InvalidProof)?)?;
                    let rows: [Arc<[E]>; 2] = [
                        rows.iter().map(|x| x[0]).collect(),
                        rows.iter().map(|x| x[1]).collect(),
                    ];
                    let columns: Arc<[[E; 2]]> =
                        expand(point.point.get(8..split).ok_or(AkitaError::InvalidProof)?)?.into();
                    let mut omega = expand(point.point.get(..8).ok_or(AkitaError::InvalidProof)?)?;
                    omega.truncate(162);
                    let omega: Arc<[[E; 2]]> = omega.into();
                    for _ in &point.evaluations {
                        let first_plane = factors.len();
                        for (coordinate, rows) in rows.iter().enumerate() {
                            factors.push(PrimePlaneFactors {
                                rows: rows.clone(),
                                columns: columns.clone(),
                                omega: omega.clone(),
                                point_index,
                                coordinate,
                                first_plane,
                                extension: true,
                            });
                        }
                    }
                }
            }
        }
        // Every consumer zips these factors with the manifest planes. Reject any
        // construction drift here instead of silently dropping a relation.
        if factors.len() != self.planes().len() {
            return Err(AkitaError::InvalidSize {
                expected: self.planes().len(),
                actual: factors.len(),
            });
        }
        Ok(factors)
    }
}

impl<E: Field> PrimePlaneFactors<E> {
    /// Combine both scalar coordinate equations into two separable tensor terms.
    /// Apply owner-specific batches lazily, retaining the shared unbatched columns.
    pub(crate) fn reading_columns(
        &self,
        batches: &[E],
    ) -> Result<impl Iterator<Item = [E; 2]> + '_, AkitaError> {
        let combine = self.reading_combiner(batches)?;
        Ok(self.columns.iter().copied().map(combine))
    }

    /// Read one column directly, without walking the preceding columns.
    pub(crate) fn reading_column(
        &self,
        batches: &[E],
        column: usize,
    ) -> Result<[E; 2], AkitaError> {
        let weights = *self.columns.get(column).ok_or(AkitaError::InvalidProof)?;
        Ok(self.reading_combiner(batches)?(weights))
    }

    fn reading_combiner(
        &self,
        batches: &[E],
    ) -> Result<impl Fn([E; 2]) -> [E; 2] + '_, AkitaError> {
        let start = checked::mul_add(self.first_plane, 2, 1).ok_or(AkitaError::InvalidProof)?;
        let b0 = *batches.get(start).ok_or(AkitaError::InvalidProof)?;
        let b1 = if self.extension {
            *batches
                .get(checked::sum([start, 2]).ok_or(AkitaError::InvalidProof)?)
                .ok_or(AkitaError::InvalidProof)?
        } else {
            E::zero()
        };
        let two = E::from_u64(2);
        Ok(move |[c0, c1]: [E; 2]| {
            if self.coordinate == 0 {
                [b0 * c0 + b1 * c1, two * b0 * c1 + b1 * c0]
            } else {
                [two * b0 * c1 + b1 * c0, two * (b0 * c0 + b1 * c1)]
            }
        })
    }
}
