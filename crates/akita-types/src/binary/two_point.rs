//! Explicit two-point statements; construction does not select a proof path.

use super::prime::prime_point_arity;
use super::prime_encoding::{append_claim, read_claim};
use super::{BinarySourceDescriptor, BinaryWordClaim, PrimeGroupClaim};
use akita_error::AkitaError;
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::{GroupCommitPhaseParams, SisModulusProfileId};
use akita_serialization::SerializationError;

const BINARY_TAG: &[u8] = b"AkitaBinaryPoints/p64\x01";
const ORDINARY_TAG: &[u8] = b"AkitaOrdinaryPoints/p64\x01";

/// External source context, bound separately by the commitment.
#[derive(Clone, Copy, Debug)]
pub enum TwoPointClaimSource<'a> {
    Binary(&'a BinarySourceDescriptor),
    Ordinary(&'a GroupCommitPhaseParams),
}

impl TwoPointClaimSource<'_> {
    fn shape(self) -> Result<(usize, usize), AkitaError> {
        match self {
            Self::Binary(source) => Ok((prime_point_arity(source)?, source.polynomials().len())),
            Self::Ordinary(profile) => {
                // Frozen-profile validation also enforces these matrix fields.
                if profile.inner.matrix.sis_modulus_profile() != SisModulusProfileId::Q64Offset59
                    || profile.outer.matrix.sis_modulus_profile()
                        != SisModulusProfileId::Q64Offset59
                {
                    return Err(AkitaError::InvalidInput(
                        "two-point ordinary claims require a P64 profile".into(),
                    ));
                }
                profile.validate_frozen_precommit(64)?;
                Ok((profile.group.num_vars(), profile.group.num_polynomials()))
            }
        }
    }

    fn tag(self) -> &'static [u8] {
        match self {
            Self::Binary(_) => BINARY_TAG,
            Self::Ordinary(_) => ORDINARY_TAG,
        }
    }
}

/// Two E64 point claims together with the source's required readings.
/// Binary groups always carry a word claim; ordinary groups use a frozen P64 profile.
/// The word claim is validated only against the source descriptor; the opening
/// that consumes this statement must bind it to the prime owners and points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TwoPointGroupClaim {
    Binary {
        word: BinaryWordClaim,
        points: [PrimeGroupClaim; 2],
    },
    Ordinary {
        points: [PrimeGroupClaim; 2],
    },
}

impl TwoPointGroupClaim {
    pub fn points(&self) -> &[PrimeGroupClaim; 2] {
        match self {
            Self::Binary { points, .. } | Self::Ordinary { points } => points,
        }
    }
}

/// One base-coordinate plane, identified by point, polynomial and coordinate.
/// An owner present at both E64 points has four planes, ordered c0/c1 per point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwoPointCoordinatePlane {
    pub point: usize,
    pub polynomial: usize,
    pub coordinate: usize,
    pub value: u128,
}

/// Validated statement at exactly two distinct E64 points.
///
/// Point order is preserved; owners are sorted within each point. Binary points
/// may read owner subsets; ordinary points read every slot. Exact integer ranges
/// of ordinary source values remain the caller's contract.
/// This statement cannot be passed to the existing single-point opening driver.
/// Authentication of both points requires a separate arithmetic integration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwoPointClaimManifest {
    claim: TwoPointGroupClaim,
}

impl TwoPointClaimManifest {
    pub fn new(
        source: TwoPointClaimSource<'_>,
        claim: TwoPointGroupClaim,
    ) -> Result<Self, AkitaError> {
        let mut manifest = Self { claim };
        manifest.validate(source)?;
        let points = match &mut manifest.claim {
            TwoPointGroupClaim::Binary { points, .. } | TwoPointGroupClaim::Ordinary { points } => {
                points
            }
        };
        for point in points {
            point
                .evaluations
                .sort_unstable_by_key(|value| value.polynomial);
        }
        Ok(manifest)
    }

    /// Recheck shape, source kind and word identity at a commitment boundary.
    pub fn validate(&self, source: TwoPointClaimSource<'_>) -> Result<(), AkitaError> {
        let (arity, owners) = source.shape()?;
        match (&self.claim, source) {
            (TwoPointGroupClaim::Binary { word, .. }, TwoPointClaimSource::Binary(descriptor)) => {
                word.validate(descriptor)?;
            }
            (TwoPointGroupClaim::Ordinary { .. }, TwoPointClaimSource::Ordinary(_)) => {}
            _ => {
                return Err(AkitaError::InvalidInput(
                    "two-point source kind mismatch".into(),
                ))
            }
        }
        for point in self.claim.points() {
            point.validate_shape(arity, owners, BinaryFieldProfile::P64)?;
            if matches!(source, TwoPointClaimSource::Ordinary(_))
                && point.evaluations.len() != owners
            {
                return Err(AkitaError::InvalidInput(
                    "ordinary points require every polynomial value".into(),
                ));
            }
        }
        let [first, second] = self.claim.points();
        if first.point == second.point {
            return Err(AkitaError::InvalidInput("duplicate E64 point".into()));
        }
        Ok(())
    }

    pub fn claim(&self) -> &TwoPointGroupClaim {
        &self.claim
    }

    /// Point-major, owner-major, then c0/c1, without allocating a plane vector.
    pub fn planes(&self) -> impl Iterator<Item = TwoPointCoordinatePlane> + '_ {
        self.claim
            .points()
            .iter()
            .enumerate()
            .flat_map(|(point, claim)| {
                claim.evaluations.iter().flat_map(move |claim| {
                    claim
                        .value
                        .coordinates()
                        .iter()
                        .enumerate()
                        .map(move |(coordinate, &value)| TwoPointCoordinatePlane {
                            point,
                            polynomial: claim.polynomial,
                            coordinate,
                            value,
                        })
                })
            })
    }

    /// Versioned source tag, mandatory binary word record, then two point records.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = match &self.claim {
            TwoPointGroupClaim::Binary { word, .. } => {
                let mut bytes = BINARY_TAG.to_vec();
                bytes.extend_from_slice(&word.canonical_bytes());
                bytes
            }
            TwoPointGroupClaim::Ordinary { .. } => ORDINARY_TAG.to_vec(),
        };
        for point in self.claim.points() {
            append_claim(&mut bytes, Some(point), BinaryFieldProfile::P64);
        }
        bytes
    }

    /// Reject wrong tags, malformed lengths, noncanonical records and trailing bytes.
    /// Every allocation is bounded by both the source shape and remaining input.
    pub fn from_bytes(
        source: TwoPointClaimSource<'_>,
        bytes: &[u8],
    ) -> Result<Self, SerializationError> {
        let bad = || SerializationError::InvalidData("noncanonical two-point statement".into());
        let (arity, owners) = source.shape().map_err(|_| bad())?;
        let mut reader = bytes.strip_prefix(source.tag()).ok_or_else(bad)?;
        let claim = match source {
            TwoPointClaimSource::Binary(descriptor) => TwoPointGroupClaim::Binary {
                word: BinaryWordClaim::read_from(descriptor, &mut reader)?,
                points: read_points(&mut reader, arity, owners)?,
            },
            TwoPointClaimSource::Ordinary(_) => TwoPointGroupClaim::Ordinary {
                points: read_points(&mut reader, arity, owners)?,
            },
        };
        let manifest = Self::new(source, claim).map_err(|_| bad())?;
        if !reader.is_empty() {
            return Err(SerializationError::InvalidData(
                "trailing bytes in two-point statement".into(),
            ));
        }
        if manifest.canonical_bytes() != bytes {
            return Err(bad());
        }
        Ok(manifest)
    }
}

fn read_points(
    reader: &mut &[u8],
    arity: usize,
    owners: usize,
) -> Result<[PrimeGroupClaim; 2], SerializationError> {
    let mut read = || {
        read_claim(
            reader,
            BinaryFieldProfile::P64,
            arity,
            owners,
            Some(|| SerializationError::InvalidData("truncated prime record".into())),
        )?
        .ok_or_else(|| {
            SerializationError::InvalidData("two-point statement requires both points".into())
        })
    };
    Ok([read()?, read()?])
}

#[cfg(test)]
#[path = "two_point_tests.rs"]
mod tests;
