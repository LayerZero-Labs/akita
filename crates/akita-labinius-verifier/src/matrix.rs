//! Bounded pages of canonical root coefficients, with no implicit flattening.
use crate::admitted::{MAX_MATRIX_BYTES, MAX_PREFIX_BYTES};
use akita_error::{checked, AkitaError};
use akita_types::proof::{
    derive_public_matrix_page, AkitaSetupSeed, MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
};
use jolt_field::{CanonicalEncoding, Field};

/// At degree 648 this is 512 elements, and exactly 81 canonical SHAKE pages.
const MATRIX_PAGE_COEFFICIENTS: usize = 512 * 648;

#[derive(Clone, Debug)]
pub(crate) struct RootMatrix {
    pages: Vec<Vec<u32>>,
    page_coefficients: usize,
    len: usize,
}

impl RootMatrix {
    pub(crate) fn dense(matrix: Vec<u32>) -> Self {
        Self {
            page_coefficients: matrix.len().max(1),
            len: matrix.len(),
            pages: vec![matrix],
        }
    }
    pub(crate) fn derive<P: Field + CanonicalEncoding>(
        seed: &AkitaSetupSeed,
        rows: usize,
        columns: usize,
        degree: usize,
        q: u32,
    ) -> Result<Self, AkitaError> {
        let invalid = || AkitaError::InvalidSetup("invalid paged root matrix extent".into());
        let len = checked::product([rows, columns, degree]).ok_or_else(invalid)?;
        let page_coefficients = MATRIX_PAGE_COEFFICIENTS;
        let page_bytes =
            checked::product([page_coefficients, size_of::<u32>()]).ok_or_else(invalid)?;
        let stream_page = seed.derivation.page_field_elements();
        // The same generic coefficient cap remains in force. A page must satisfy
        // the original dense matrix budget, and stream scratch its prefix budget.
        if rows == 0
            || columns == 0
            || degree == 0
            || !degree.is_multiple_of(2)
            || !page_coefficients.is_multiple_of(degree)
            || q == 0
            || len > MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS
            || P::NUM_BYTES > size_of::<u128>()
            || page_bytes > MAX_MATRIX_BYTES
            || !page_coefficients.is_multiple_of(stream_page)
            || checked::product([stream_page, size_of::<P>()]).is_none_or(|n| n > MAX_PREFIX_BYTES)
        {
            return Err(invalid());
        }
        akita_params::field_modulus_be_bytes::<P>()?;
        let page_count = len.div_ceil(page_coefficients);
        let mut pages = Vec::new();
        pages.try_reserve_exact(page_count).map_err(|_| invalid())?;
        let mut consumed = 0usize;
        while consumed < len {
            let count = (len - consumed).min(page_coefficients);
            let mut page = Vec::new();
            page.try_reserve_exact(count).map_err(|_| invalid())?;
            while page.len() < count {
                let offset = checked::sum([consumed, page.len()]).ok_or_else(invalid)?;
                let take = (count - page.len()).min(stream_page);
                let values = derive_public_matrix_page::<P>(seed, offset / stream_page, take)?;
                for value in values {
                    let mut bytes = [0u8; 16];
                    value.to_bytes_le(bytes.get_mut(..P::NUM_BYTES).ok_or_else(invalid)?);
                    page.push((u128::from_le_bytes(bytes) % u128::from(q)) as u32);
                }
            }
            consumed = checked::sum([consumed, count]).ok_or_else(invalid)?;
            pages.push(page);
        }
        Ok(Self {
            pages,
            page_coefficients,
            len,
        })
    }
    pub(crate) fn len(&self) -> usize {
        self.len
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = &u32> {
        self.pages.iter().flatten()
    }
    pub(crate) fn element(&self, index: usize, degree: usize) -> Option<&[u32]> {
        let start = checked::product([index, degree])?;
        if start >= self.len {
            return None;
        }
        let page = self.pages.get(start / self.page_coefficients)?;
        page.get(checked::range(start % self.page_coefficients, degree)?)
    }
    pub(crate) fn page_count(&self) -> usize {
        self.pages.len()
    }
    pub(crate) fn maximum_page_bytes(&self) -> usize {
        self.pages
            .iter()
            .map(|p| p.len() * size_of::<u32>())
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use jolt_field::Prime128Offset275;

    #[test]
    fn paged_coefficients_match_dense_at_every_boundary() {
        let seed = AkitaSetupSeed::shake256_paged_v1([0x28; 32]);
        for degree in [162, 324, 648] {
            let columns = MATRIX_PAGE_COEFFICIENTS / degree + 1;
            let paged =
                RootMatrix::derive::<Prime128Offset275>(&seed, 1, columns, degree, 33_568_993)
                    .unwrap();
            let dense = crate::admitted::derive_trinomial_matrix::<Prime128Offset275>(
                &seed, 1, columns, degree, 33_568_993,
            )
            .unwrap();
            assert_eq!(paged.page_count(), 2);
            assert_eq!(paged.maximum_page_bytes(), MATRIX_PAGE_COEFFICIENTS * 4);
            assert!(paged.iter().eq(dense.iter()));
            for index in [0, columns - 2, columns - 1] {
                assert_eq!(
                    paged.element(index, degree).unwrap(),
                    &dense[index * degree..(index + 1) * degree]
                );
            }
            assert!(paged.element(columns, degree).is_none());
        }
    }

    #[test]
    fn paged_and_dense_caps_are_not_relaxed() {
        let seed = AkitaSetupSeed::shake256_paged_v1([0x28; 32]);
        // 32K geometry remains rejected by the old monolithic API before allocation.
        assert!(
            crate::admitted::derive_trinomial_matrix::<Prime128Offset275>(
                &seed, 2, 32768, 648, 33_568_993
            )
            .is_err()
        );
        assert!(RootMatrix::derive::<Prime128Offset275>(&seed, 2, 65536, 648, 33_568_993).is_err());
        assert!(RootMatrix::derive::<Prime128Offset275>(&seed, 1, 1, 0, 33_568_993).is_err());
    }
}
