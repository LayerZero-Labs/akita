//! Optional scalar views in the outgoing witness and their compact D layout.

use super::{binary_scalar_geometry, BinaryRootWitnessPlan};
use crate::root_repacking::{RootCoefficientRepacking, RootRepackingSegment};
use akita_error::AkitaError;
use jolt_algebra::checked;

impl BinaryRootWitnessPlan {
    /// Select the word views and append every prime plane before ordinary groups.
    /// An empty prime list with a word preserves the original coefficient map.
    pub fn with_prime_planes(mut self, word: bool, planes: usize) -> Result<Self, AkitaError> {
        if planes > self.descriptor().polynomials().len() || (!word && planes == 0) {
            return Err(AkitaError::InvalidInput(
                "invalid binary plane selection".into(),
            ));
        }
        if self.prime_planes != 0 || self.repacking.segments().len() != 5 {
            return Err(AkitaError::InvalidInput(
                "prime planes must precede ordinary segments".into(),
            ));
        }
        let mut segments = self.repacking.segments().to_vec();
        if !word {
            segments.truncate(2);
        }
        if planes > 0 {
            segments.push(RootRepackingSegment {
                geometry: binary_scalar_geometry()?,
                elements: checked::product([planes, self.descriptor().index_map().columns()])
                    .ok_or(AkitaError::InvalidProof)?,
                digits: self.field_profile().prime_plane_digits(),
            });
        }
        self.repacking =
            RootCoefficientRepacking::new(segments, self.repacking.destination_ring_dimension())?;
        self.word = word;
        self.prime_planes = planes;
        Ok(self)
    }

    pub fn prime_segment(&self) -> Option<usize> {
        (self.prime_planes > 0).then_some(if self.word { 5 } else { 2 })
    }

    /// Natural digit concatenation committed by D, with no inter-plane padding.
    pub fn opening_segments(&self) -> Vec<usize> {
        let mut segments = Vec::new();
        if self.word {
            segments.push(2);
        }
        segments.extend(self.prime_segment());
        segments
    }
}
