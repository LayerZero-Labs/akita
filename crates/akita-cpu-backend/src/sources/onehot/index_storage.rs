//! Ownership for resident or externally backed one-hot index slices.
use super::OneHotIndex;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) enum OneHotIndices<I: OneHotIndex> {
    Owned(Vec<Option<I>>),
    Shared(Arc<dyn AsRef<[Option<I>]> + Send + Sync>),
}

impl<I: OneHotIndex> Deref for OneHotIndices<I> {
    type Target = [Option<I>];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(indices) => indices,
            Self::Shared(indices) => indices.as_ref().as_ref(),
        }
    }
}

impl<I: OneHotIndex> fmt::Debug for OneHotIndices<I> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OneHotIndices")
            .field("len", &self.len())
            .finish()
    }
}

// Internal tests deliberately mutate malformed fixtures and test clone ownership.
#[cfg(test)]
impl<I: OneHotIndex> std::ops::DerefMut for OneHotIndices<I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Owned(indices) => indices,
            Self::Shared(indices) => {
                let owned = indices.as_ref().as_ref().to_vec();
                *self = Self::Owned(owned);
                self.deref_mut()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OneHotPoly;
    use jolt_field::Prime128OffsetA7F7 as F;

    #[test]
    fn shared_indices_preserve_values_and_ownership() {
        let backing = Arc::new(vec![Some(0u8), Some(7), None, Some(15)]);
        let poly = OneHotPoly::<F, u8>::from_shared_indices(16, backing.clone()).unwrap();
        assert_eq!(poly.indices(), backing.as_slice());
        let mut cloned = poly.clone();
        cloned.indices[0] = None;
        assert_eq!(poly.indices()[0], Some(0));
        assert_eq!(cloned.indices()[0], None);
    }

    #[test]
    fn shared_indices_validate_positions() {
        assert!(
            OneHotPoly::<F, u8>::from_shared_indices(16, Arc::new(vec![Some(16u8), None])).is_err()
        );
    }
}
