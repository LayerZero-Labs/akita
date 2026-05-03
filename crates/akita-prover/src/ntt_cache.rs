//! Multi-D NTT cache management.
//!
//! Wraps per-D [`NttSlotCache`] bundles with lazy computation and memoization.
//! A single [`MultiDNttCaches`] can hold NTT caches for any subset of supported
//! ring dimensions, built on demand from the shared [`FlatMatrix`].

use akita_field::{CanonicalField, FieldCore, HachiError};
use akita_types::FlatMatrix;

use crate::crt_ntt::{build_ntt_slot, NttSlotCache};

/// Per-matrix NTT caches for multiple ring dimensions.
///
/// Each field is lazily populated by the `get_or_build_*` methods.
/// Fields use `Box<NttSlotCache<D>>` to keep the struct's inline size
/// small: `NttSlotCache<1024>` alone is ~80 KB due to inline twiddle
/// arrays, so storing them unboxed would make this struct ~155 KB.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiDNttCaches {
    /// Cache for D=32.
    pub d32: Option<Box<NttSlotCache<32>>>,
    /// Cache for D=64.
    pub d64: Option<Box<NttSlotCache<64>>>,
    /// Cache for D=128.
    pub d128: Option<Box<NttSlotCache<128>>>,
    /// Cache for D=256.
    pub d256: Option<Box<NttSlotCache<256>>>,
    /// Cache for D=512.
    pub d512: Option<Box<NttSlotCache<512>>>,
    /// Cache for D=1024.
    pub d1024: Option<Box<NttSlotCache<1024>>>,
}

macro_rules! impl_get_or_build {
    ($fn_name:ident, $field:ident, $d_val:expr) => {
        /// Get (or build and memoize) the NTT cache for this ring dimension.
        ///
        /// # Errors
        ///
        /// Returns an error if no CRT+NTT parameter set matches the field and D.
        pub fn $fn_name<F: FieldCore + CanonicalField>(
            &mut self,
            mat: &FlatMatrix<F>,
        ) -> Result<&NttSlotCache<$d_val>, HachiError> {
            if self.$field.is_none() {
                self.$field = Some(Box::new(build_ntt_slot(
                    mat.ring_view::<$d_val>(1, mat.total_ring_elements_at::<$d_val>()),
                )?));
            }
            Ok(self.$field.as_deref().unwrap())
        }
    };
}

impl MultiDNttCaches {
    /// Empty cache set.
    pub fn new() -> Self {
        Self {
            d32: None,
            d64: None,
            d128: None,
            d256: None,
            d512: None,
            d1024: None,
        }
    }

    impl_get_or_build!(get_or_build_32, d32, 32);
    impl_get_or_build!(get_or_build_64, d64, 64);
    impl_get_or_build!(get_or_build_128, d128, 128);
    impl_get_or_build!(get_or_build_256, d256, 256);
    impl_get_or_build!(get_or_build_512, d512, 512);
    impl_get_or_build!(get_or_build_1024, d1024, 1024);

    /// Check if a cache for dimension `d` is already populated.
    pub fn has(&self, d: usize) -> bool {
        match d {
            32 => self.d32.is_some(),
            64 => self.d64.is_some(),
            128 => self.d128.is_some(),
            256 => self.d256.is_some(),
            512 => self.d512.is_some(),
            1024 => self.d1024.is_some(),
            _ => false,
        }
    }
}

impl Default for MultiDNttCaches {
    fn default() -> Self {
        Self::new()
    }
}
