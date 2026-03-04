//! Multi-D NTT cache management.
//!
//! Wraps per-D [`NttSlotCache`] bundles with lazy computation and memoization.
//! A single [`MultiDNttCaches`] can hold NTT caches for any subset of supported
//! ring dimensions, built on demand from a shared [`FlatMatrix`].

use super::crt_ntt::{build_ntt_slot, NttSlotCache};
use super::flat_matrix::FlatMatrix;
use crate::error::HachiError;
use crate::{CanonicalField, FieldCore};

/// Per-matrix NTT caches for multiple ring dimensions.
///
/// Each field is lazily populated by the `get_or_build_*` methods.
/// Fields use `Box<NttSlotCache<D>>` to keep the struct's inline size
/// small: `NttSlotCache<1024>` alone is ~80 KB due to inline twiddle
/// arrays, so storing them unboxed would make this struct ~155 KB.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiDNttCaches {
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
                self.$field = Some(Box::new(build_ntt_slot(mat.view::<$d_val>())?));
            }
            Ok(self.$field.as_deref().unwrap())
        }
    };
}

impl MultiDNttCaches {
    /// Empty cache set.
    pub fn new() -> Self {
        Self {
            d64: None,
            d128: None,
            d256: None,
            d512: None,
            d1024: None,
        }
    }

    impl_get_or_build!(get_or_build_64, d64, 64);
    impl_get_or_build!(get_or_build_128, d128, 128);
    impl_get_or_build!(get_or_build_256, d256, 256);
    impl_get_or_build!(get_or_build_512, d512, 512);
    impl_get_or_build!(get_or_build_1024, d1024, 1024);

    /// Check if a cache for dimension `d` is already populated.
    pub fn has(&self, d: usize) -> bool {
        match d {
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

/// Bundle of three multi-D NTT caches for the A, B, and D matrices.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[allow(non_snake_case)]
pub struct MultiDNttBundle {
    /// NTT caches for the A matrix at various ring dimensions.
    pub A: MultiDNttCaches,
    /// NTT caches for the B matrix at various ring dimensions.
    pub B: MultiDNttCaches,
    /// NTT caches for the D matrix at various ring dimensions.
    pub D_mat: MultiDNttCaches,
}

impl MultiDNttBundle {
    /// Empty bundle.
    pub fn new() -> Self {
        Self::default()
    }
}
