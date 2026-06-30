//! Prover-private typed digit-plane storage (`FlatDigitBlocks<const D>`).
//!
//! # Why this lives in the prover crate (runtime-ring cutover, Slice A)
//!
//! S4 demoted the protocol-facing digit storage in `akita-types` to the D-free
//! [`akita_types::DigitBlocks`] (a flat `Vec<i8>` plane stream plus an explicit
//! runtime `digit_stride`). The prover's commit/decompose/ring-switch kernels,
//! however, operate on `&[[i8; D]]` digit planes where `D` is a compile-time
//! constant. Rather than rewrite every kernel to index a flat `&[i8]` by a
//! runtime stride (which would also lose the `[i8; D]` array ergonomics the
//! decomposition routines rely on), the typed plane representation is re-homed
//! here as a **kernel-internal** type. Kernels keep using `FlatDigitBlocks<D>`;
//! conversion to/from the D-free [`akita_types::DigitBlocks`] happens only at the
//! protocol-storage boundary (see [`FlatDigitBlocks::into_digit_blocks`] /
//! [`FlatDigitBlocks::from_digit_blocks`]).
//!
//! This carries no serialization impls: at-rest serialization is the job of the
//! D-free `DigitBlocks`. This type is a prover-side compute carrier only.

use akita_field::AkitaError;
use akita_types::DigitBlocks;

/// Flat digit-plane storage plus explicit block boundaries, keyed on the
/// compile-time ring dimension `D`.
///
/// Each plane is a `[i8; D]` row of signed digits; `block_sizes` gives the
/// per-block plane count, and `flat_digits.len() == sum(block_sizes)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatDigitBlocks<const D: usize> {
    flat_digits: Vec<[i8; D]>,
    block_sizes: Vec<usize>,
}

/// Iterator over logical blocks inside [`FlatDigitBlocks`].
pub struct FlatDigitBlockIter<'a, const D: usize> {
    flat_digits: &'a [[i8; D]],
    block_sizes: &'a [usize],
    offset: usize,
}

impl<'a, const D: usize> Iterator for FlatDigitBlockIter<'a, D> {
    type Item = &'a [[i8; D]];

    fn next(&mut self) -> Option<Self::Item> {
        let (&size, rest) = self.block_sizes.split_first()?;
        self.block_sizes = rest;
        let block = &self.flat_digits[self.offset..self.offset + size];
        self.offset += size;
        Some(block)
    }
}

impl<const D: usize> FlatDigitBlocks<D> {
    /// Construct an empty digit-block collection.
    pub fn empty() -> Self {
        Self {
            flat_digits: Vec::new(),
            block_sizes: Vec::new(),
        }
    }

    /// Construct zero-initialized flat digits for explicit block sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the block sizes overflow the total flat length.
    pub fn zeroed(block_sizes: Vec<usize>) -> Result<Self, AkitaError> {
        let total_planes = block_sizes.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                AkitaError::InvalidInput("flat digit block size overflow".to_string())
            })
        })?;
        Ok(Self {
            flat_digits: vec![[0i8; D]; total_planes],
            block_sizes,
        })
    }

    /// Construct from flat digits and explicit block sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the block sizes do not sum to the flat digit count.
    pub fn new(flat_digits: Vec<[i8; D]>, block_sizes: Vec<usize>) -> Result<Self, AkitaError> {
        let expected = block_sizes.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                AkitaError::InvalidInput("flat digit block size overflow".to_string())
            })
        })?;
        if expected != flat_digits.len() {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: flat_digits.len(),
            });
        }
        Ok(Self {
            flat_digits,
            block_sizes,
        })
    }

    /// Flatten a block-owned representation into canonical storage.
    pub fn from_blocks(blocks: Vec<Vec<[i8; D]>>) -> Self {
        let block_sizes: Vec<usize> = blocks.iter().map(Vec::len).collect();
        let total_planes: usize = block_sizes.iter().sum();
        let mut flat_digits = Vec::with_capacity(total_planes);
        for block in blocks {
            flat_digits.extend(block);
        }
        Self {
            flat_digits,
            block_sizes,
        }
    }

    /// Number of logical blocks.
    pub fn block_count(&self) -> usize {
        self.block_sizes.len()
    }

    /// Number of logical blocks.
    pub fn len(&self) -> usize {
        self.block_count()
    }

    /// Whether there are no logical blocks.
    pub fn is_empty(&self) -> bool {
        self.block_sizes.is_empty()
    }

    /// Per-block digit-plane counts.
    pub fn block_sizes(&self) -> &[usize] {
        &self.block_sizes
    }

    /// Flat digit stream in plane-major block order.
    pub fn flat_digits(&self) -> &[[i8; D]] {
        &self.flat_digits
    }

    /// Mutable flat digit stream in plane-major block order.
    pub fn flat_digits_mut(&mut self) -> &mut [[i8; D]] {
        &mut self.flat_digits
    }

    /// Split the flat digit stream into disjoint mutable block slices.
    pub fn split_blocks_mut(&mut self) -> Vec<&mut [[i8; D]]> {
        let mut blocks = Vec::with_capacity(self.block_sizes.len());
        let mut tail = self.flat_digits.as_mut_slice();
        for &block_size in &self.block_sizes {
            let (head, rest) = tail.split_at_mut(block_size);
            blocks.push(head);
            tail = rest;
        }
        blocks
    }

    /// Iterate over blocks as slices into the flat digit stream.
    pub fn iter_blocks(&self) -> FlatDigitBlockIter<'_, D> {
        FlatDigitBlockIter {
            flat_digits: &self.flat_digits,
            block_sizes: &self.block_sizes,
            offset: 0,
        }
    }

    /// Iterate over logical blocks.
    pub fn iter(&self) -> FlatDigitBlockIter<'_, D> {
        self.iter_blocks()
    }

    /// Append the flat digit stream to `dst`.
    pub fn extend_flat_digits(&self, dst: &mut Vec<[i8; D]>) {
        dst.extend_from_slice(&self.flat_digits);
    }

    /// Truncate every block to at most `block_len` digit planes.
    pub fn truncate_each_block(&mut self, block_len: usize) {
        if self.block_sizes.iter().all(|&size| size <= block_len) {
            return;
        }

        let total_planes: usize = self
            .block_sizes
            .iter()
            .map(|&size| size.min(block_len))
            .sum();
        let mut new_flat = Vec::with_capacity(total_planes);
        let mut offset = 0usize;
        for size in &mut self.block_sizes {
            let keep = (*size).min(block_len);
            new_flat.extend_from_slice(&self.flat_digits[offset..offset + keep]);
            offset += *size;
            *size = keep;
        }
        self.flat_digits = new_flat;
    }

    /// Consume the storage and rebuild owned blocks.
    pub fn into_blocks(self) -> Vec<Vec<[i8; D]>> {
        let mut blocks = Vec::with_capacity(self.block_sizes.len());
        let mut offset = 0usize;
        for size in self.block_sizes {
            blocks.push(self.flat_digits[offset..offset + size].to_vec());
            offset += size;
        }
        blocks
    }

    /// Consume into the flat digits and block sizes.
    pub fn into_parts(self) -> (Vec<[i8; D]>, Vec<usize>) {
        (self.flat_digits, self.block_sizes)
    }

    /// Convert into the D-free protocol-storage [`DigitBlocks`].
    ///
    /// The per-plane stride is the compile-time `D`. This is the prover→protocol
    /// boundary conversion: the typed `[i8; D]` planes are flattened into the
    /// D-free `Vec<i8>` plane stream.
    pub fn into_digit_blocks(self) -> DigitBlocks {
        let mut digits = Vec::with_capacity(self.flat_digits.len() * D);
        for plane in &self.flat_digits {
            digits.extend_from_slice(plane);
        }
        // Stride `D` and `block_sizes` are consistent by construction
        // (`flat_digits.len() == sum(block_sizes)`), so `new` cannot fail.
        DigitBlocks::new(digits, self.block_sizes, D)
            .expect("typed flat digit blocks always satisfy the DigitBlocks invariant")
    }

    /// Borrow as the D-free protocol-storage [`DigitBlocks`].
    pub fn to_digit_blocks(&self) -> DigitBlocks {
        self.clone().into_digit_blocks()
    }

    /// Rebuild typed planes from the D-free [`DigitBlocks`].
    ///
    /// # Errors
    ///
    /// Returns an error if the digit stride does not match `D` or the flat
    /// digit stream is not an exact multiple of `D`.
    pub fn from_digit_blocks(blocks: &DigitBlocks) -> Result<Self, AkitaError> {
        if blocks.digit_stride() != D {
            return Err(AkitaError::InvalidInput(format!(
                "digit blocks stride {} does not match prover ring dimension D={D}",
                blocks.digit_stride()
            )));
        }
        let (planes, remainder) = blocks.digits().as_chunks::<D>();
        if !remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: blocks.digits().len(),
            });
        }
        Self::new(planes.to_vec(), blocks.block_sizes().to_vec())
    }
}
