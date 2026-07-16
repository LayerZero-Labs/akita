//! Key type for runtime ring-dimension NTT prepared-setup caches.

use akita_error::AkitaError;
use jolt_field::FieldCore;

use crate::proof::AkitaExpandedSetup;

/// Identifies one full-envelope NTT cache entry at a concrete ring degree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NttCacheKey {
    /// Ring dimension `D` for the cached transform family.
    pub ring_d: usize,
    /// Number of ring elements in the cached matrix view at `ring_d`.
    pub num_ring_elements: usize,
}

impl NttCacheKey {
    /// Build the full-envelope cache key for `ring_d` on `expanded`.
    ///
    /// # Errors
    ///
    /// Returns an error when `ring_d` does not divide the setup envelope or the
    /// matrix view length cannot be computed.
    pub fn from_envelope<F: FieldCore>(
        expanded: &AkitaExpandedSetup<F>,
        ring_d: usize,
    ) -> Result<Self, AkitaError> {
        let num_ring_elements = expanded
            .shared_matrix()
            .total_ring_elements_at_dyn(ring_d)?;
        Ok(Self {
            ring_d,
            num_ring_elements,
        })
    }
}
