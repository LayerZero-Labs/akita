//! Key type for runtime ring-dimension NTT prepared-setup caches.

/// Identifies one full-envelope NTT cache entry at a concrete ring degree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NttCacheKey {
    /// Ring dimension `D` for the cached transform family.
    pub ring_d: usize,
    /// Number of ring elements in the cached matrix view at `ring_d`.
    pub num_ring_elements: usize,
}
