//! Storage access for digit-based commitment and folding kernels.

use crate::sources::packed_digits::PackedSignedDigitView;
use crate::sources::poly_helpers::{
    I8DigitSpan, SignedDigitBasis, ValidatedSparseChallenge, WidePlan, WindowedChallenge,
};
use akita_error::AkitaError;
use akita_params::SignedDigitKernel;
use jolt_algebra::lift_ct;
use jolt_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use jolt_algebra::CyclotomicRing;
use jolt_algebra::{CanonicalEncoding, Field};
use jolt_crypto::challenge::SparseChallenge;
use std::ops::Range;

pub(crate) mod evaluate;

#[derive(Clone, Copy)]
pub(crate) struct DigitShape {
    pub num_digits: usize,
    pub log_basis: u32,
}

pub(crate) trait RingValues<F: Field, const D: usize> {
    /// Whether readers may skip zero coefficients. Signed values are a
    /// secret witness's (Whiteout's response digits,
    /// `specs/whiteout-akita.md` A45), so they are lifted without a branch on
    /// their sign and never skipped.
    const SKIPS_ZEROS: bool;
    fn coeff(&self, k: usize) -> F;
}

impl<F: Field, const D: usize> RingValues<F, D> for &[F; D] {
    const SKIPS_ZEROS: bool = true;
    fn coeff(&self, k: usize) -> F {
        self[k]
    }
}

impl<F: Field, const D: usize> RingValues<F, D> for [i8; D] {
    const SKIPS_ZEROS: bool = false;
    fn coeff(&self, k: usize) -> F {
        lift_ct(i64::from(self[k]))
    }
}

impl<F: Field, const D: usize> RingValues<F, D> for [i16; D] {
    const SKIPS_ZEROS: bool = false;
    fn coeff(&self, k: usize) -> F {
        lift_ct(i64::from(self[k]))
    }
}

pub(crate) trait RingValueSource<F: Field, const D: usize>: Sync {
    type Values<'a>: RingValues<F, D> + Sync
    where
        Self: 'a;

    fn num_rings(&self) -> usize;
    fn values(&self, ring: usize) -> Result<Self::Values<'_>, AkitaError>;
}

impl<F: Field, const D: usize> RingValueSource<F, D> for &[CyclotomicRing<F, D>] {
    type Values<'a>
        = &'a [F; D]
    where
        Self: 'a;

    fn num_rings(&self) -> usize {
        self.len()
    }
    fn values(&self, ring: usize) -> Result<Self::Values<'_>, AkitaError> {
        self.get(ring)
            .map(CyclotomicRing::coefficients)
            .ok_or_else(|| {
                AkitaError::InvalidInput("ring value index is outside the source".into())
            })
    }
}

pub(crate) trait BlockDigits<const D: usize>: Sync {
    type Scratch: Send;
    type Planes<'a>: DigitPlaneSet<D>
    where
        Self: 'a;

    fn num_rings(&self) -> usize;
    fn scratch(&self, shape: DigitShape, max_rings: usize) -> Self::Scratch;
    fn digit_abs_bound(&self, shape: DigitShape) -> u64;
    fn i8_digit_span(&self, shape: DigitShape) -> Option<I8DigitSpan>;
    fn digit_planes<'a>(
        &'a self,
        ring: usize,
        shape: DigitShape,
        s: &'a mut Self::Scratch,
    ) -> Result<Self::Planes<'a>, AkitaError>;
}

pub(crate) trait I8Rows<const D: usize>: Sync {
    type I8Scratch: Send;

    fn i8_scratch(&self, shape: DigitShape, max_rings: usize) -> Self::I8Scratch;
    fn digit_abs_bound(&self, shape: DigitShape) -> u64;
    /// `rings` is local to the block; `block_start` locates it in a flat source.
    fn i8_rows<'a>(
        &'a self,
        block_index: usize,
        block_start: usize,
        rings: Range<usize>,
        shape: DigitShape,
        s: &'a mut Self::I8Scratch,
    ) -> Result<&'a [[i8; D]], AkitaError>;
}

pub(crate) trait I16Rows<const D: usize>: Sync {
    type I16Scratch: Send;

    fn i16_scratch(&self, shape: DigitShape, max_rings: usize) -> Self::I16Scratch;
    fn i16_rows<'a>(
        &'a self,
        rings: Range<usize>,
        shape: DigitShape,
        s: &'a mut Self::I16Scratch,
    ) -> Result<&'a [[i16; D]], AkitaError>;
}

fn bounded_i8_digit_span(digit_abs_bound: u64) -> I8DigitSpan {
    if digit_abs_bound <= 63 {
        I8DigitSpan::Half
    } else {
        I8DigitSpan::Full
    }
}

pub(crate) struct CachedDigits<'a, const D: usize> {
    pub(crate) digit_planes: &'a [[i8; D]],
    pub(crate) num_rings: usize,
    pub(crate) digit_abs_bound: u64,
}

impl<'a, const D: usize> CachedDigits<'a, D> {
    pub(crate) fn new(
        digit_planes: &'a [[i8; D]],
        num_digits: usize,
        basis: SignedDigitBasis,
    ) -> Result<Self, AkitaError> {
        let num_rings = digit_planes.len().checked_div(num_digits).ok_or_else(|| {
            AkitaError::InvalidInput("cached fold digit count must be nonzero".into())
        })?;
        Ok(Self {
            digit_planes,
            num_rings,
            digit_abs_bound: basis.abs_bound.min(u64::from(i8::MIN.unsigned_abs())),
        })
    }
}

impl<const D: usize> BlockDigits<D> for CachedDigits<'_, D> {
    type Scratch = ();
    type Planes<'a>
        = DigitPlanes<'a, D>
    where
        Self: 'a;
    fn num_rings(&self) -> usize {
        self.num_rings
    }
    fn digit_abs_bound(&self, _shape: DigitShape) -> u64 {
        self.digit_abs_bound
    }
    fn i8_digit_span(&self, _shape: DigitShape) -> Option<I8DigitSpan> {
        Some(bounded_i8_digit_span(self.digit_abs_bound))
    }
    fn scratch(&self, _shape: DigitShape, _max_rings: usize) {}
    fn digit_planes<'a>(
        &'a self,
        ring: usize,
        shape: DigitShape,
        _s: &'a mut (),
    ) -> Result<Self::Planes<'a>, AkitaError> {
        let start = ring * shape.num_digits;
        Ok(DigitPlanes::I8(
            &self.digit_planes[start..start + shape.num_digits],
        ))
    }
}

pub(crate) struct LiveRings<'a, F: Field + CanonicalEncoding, const D: usize> {
    pub(crate) coeffs: &'a [CyclotomicRing<F, D>],
    pub(crate) params: &'a BalancedDecomposePow2Params<F>,
    pub(crate) basis: SignedDigitBasis,
}

pub(crate) enum LiveDigitScratch<const D: usize> {
    I8(Vec<[i8; D]>),
    I16(Vec<[i16; D]>),
}

impl<F: Field + CanonicalEncoding, const D: usize> RingValueSource<F, D> for LiveRings<'_, F, D> {
    type Values<'a>
        = &'a [F; D]
    where
        Self: 'a;
    fn num_rings(&self) -> usize {
        self.coeffs.len()
    }
    fn values(&self, ring: usize) -> Result<Self::Values<'_>, AkitaError> {
        self.coeffs.values(ring)
    }
}

impl<F: Field + CanonicalEncoding, const D: usize> BlockDigits<D> for LiveRings<'_, F, D> {
    type Scratch = LiveDigitScratch<D>;
    type Planes<'a>
        = DigitPlanes<'a, D>
    where
        Self: 'a;
    fn num_rings(&self) -> usize {
        self.coeffs.len()
    }
    fn digit_abs_bound(&self, _shape: DigitShape) -> u64 {
        self.basis.abs_bound
    }
    fn i8_digit_span(&self, _shape: DigitShape) -> Option<I8DigitSpan> {
        matches!(self.basis.kernel, SignedDigitKernel::I8).then(|| {
            if self.basis.abs_bound <= 64 {
                I8DigitSpan::Half
            } else {
                I8DigitSpan::Full
            }
        })
    }
    fn scratch(&self, shape: DigitShape, max_rings: usize) -> Self::Scratch {
        let width = shape.num_digits * max_rings;
        match self.basis.kernel {
            SignedDigitKernel::I8 => LiveDigitScratch::I8(vec![[0; D]; width]),
            SignedDigitKernel::I16 => LiveDigitScratch::I16(vec![[0; D]; width]),
        }
    }

    fn digit_planes<'a>(
        &'a self,
        ring: usize,
        shape: DigitShape,
        s: &'a mut Self::Scratch,
    ) -> Result<Self::Planes<'a>, AkitaError> {
        match s {
            LiveDigitScratch::I8(buffer) => {
                let planes = &mut buffer[..shape.num_digits];
                self.coeffs[ring].balanced_decompose_pow2_i8_into_with_params(planes, self.params);
                Ok(DigitPlanes::I8(planes))
            }
            LiveDigitScratch::I16(buffer) => {
                let planes = &mut buffer[..shape.num_digits];
                self.coeffs[ring].balanced_decompose_pow2_i16_into(planes, self.params);
                Ok(DigitPlanes::I16(planes))
            }
        }
    }
}

impl<F: Field + CanonicalEncoding, const D: usize> I8Rows<D> for LiveRings<'_, F, D> {
    type I8Scratch = Vec<[i8; D]>;
    fn i8_scratch(&self, shape: DigitShape, max_rings: usize) -> Self::I8Scratch {
        vec![[0; D]; shape.num_digits * max_rings]
    }
    fn digit_abs_bound(&self, _shape: DigitShape) -> u64 {
        self.basis.abs_bound
    }
    fn i8_rows<'a>(
        &'a self,
        _block_index: usize,
        block_start: usize,
        rings: Range<usize>,
        shape: DigitShape,
        s: &'a mut Self::I8Scratch,
    ) -> Result<&'a [[i8; D]], AkitaError> {
        let width = rings.len() * shape.num_digits;
        for (ring, dst) in self.coeffs[block_start + rings.start..block_start + rings.end]
            .iter()
            .zip(s[..width].chunks_mut(shape.num_digits))
        {
            ring.balanced_decompose_pow2_i8_into_with_params(dst, self.params);
        }
        Ok(&s[..width])
    }
}

impl<F: Field + CanonicalEncoding, const D: usize> I16Rows<D> for LiveRings<'_, F, D> {
    type I16Scratch = Vec<[i16; D]>;
    fn i16_scratch(&self, shape: DigitShape, max_rings: usize) -> Self::I16Scratch {
        vec![[0; D]; shape.num_digits * max_rings]
    }
    fn i16_rows<'a>(
        &'a self,
        rings: Range<usize>,
        shape: DigitShape,
        s: &'a mut Self::I16Scratch,
    ) -> Result<&'a [[i16; D]], AkitaError> {
        let width = rings.len() * shape.num_digits;
        for (ring, dst) in self.coeffs[rings]
            .iter()
            .zip(s[..width].chunks_mut(shape.num_digits))
        {
            ring.balanced_decompose_pow2_i16_into(dst, self.params);
        }
        s[width..].fill([0; D]);
        Ok(s)
    }
}

pub(crate) struct PackedDigits<'a> {
    pub(crate) digits: PackedSignedDigitView<'a>,
    pub(crate) num_rings: usize,
}

impl<F: Field, const D: usize> RingValueSource<F, D> for PackedDigits<'_> {
    type Values<'a>
        = [i8; D]
    where
        Self: 'a;
    fn num_rings(&self) -> usize {
        self.num_rings
    }
    fn values(&self, ring: usize) -> Result<Self::Values<'_>, AkitaError> {
        self.digits.decode_array(ring * D)
    }
}

impl<const D: usize> BlockDigits<D> for PackedDigits<'_> {
    type Scratch = [i8; D];
    type Planes<'a>
        = &'a [i8; D]
    where
        Self: 'a;
    fn num_rings(&self) -> usize {
        self.num_rings
    }
    fn digit_abs_bound(&self, _shape: DigitShape) -> u64 {
        let bounds = self.digits.bounds();
        u64::from(bounds.negative_abs_max().max(bounds.positive_max()))
    }
    fn i8_digit_span(&self, shape: DigitShape) -> Option<I8DigitSpan> {
        Some(bounded_i8_digit_span(
            <Self as BlockDigits<D>>::digit_abs_bound(self, shape),
        ))
    }
    fn scratch(&self, _shape: DigitShape, _max_rings: usize) -> Self::Scratch {
        [0; D]
    }
    fn digit_planes<'a>(
        &'a self,
        ring: usize,
        shape: DigitShape,
        s: &'a mut Self::Scratch,
    ) -> Result<Self::Planes<'a>, AkitaError> {
        debug_assert_eq!(shape.num_digits, 1);
        self.digits.decode_range(ring * D, s)?;
        Ok(s)
    }
}

impl<const D: usize> I8Rows<D> for PackedDigits<'_> {
    type I8Scratch = Vec<[i8; D]>;
    fn i8_scratch(&self, _shape: DigitShape, max_rings: usize) -> Self::I8Scratch {
        vec![[0; D]; max_rings]
    }
    fn digit_abs_bound(&self, shape: DigitShape) -> u64 {
        <Self as BlockDigits<D>>::digit_abs_bound(self, shape)
    }
    fn i8_rows<'a>(
        &'a self,
        _block_index: usize,
        block_start: usize,
        rings: Range<usize>,
        shape: DigitShape,
        s: &'a mut Self::I8Scratch,
    ) -> Result<&'a [[i8; D]], AkitaError> {
        if shape.num_digits != 1 {
            return Err(AkitaError::InvalidInput(
                "packed rows require one inner digit".into(),
            ));
        }
        let rows = &mut s[..rings.len()];
        self.digits
            .decode_range((block_start + rings.start) * D, rows.as_flattened_mut())?;
        Ok(rows)
    }
}

impl<const D: usize> I16Rows<D> for PackedDigits<'_> {
    type I16Scratch = (Vec<[i8; D]>, Vec<[i16; D]>);
    fn i16_scratch(&self, shape: DigitShape, max_rings: usize) -> Self::I16Scratch {
        (
            <Self as I8Rows<D>>::i8_scratch(self, shape, max_rings),
            vec![[0; D]; max_rings],
        )
    }
    fn i16_rows<'a>(
        &'a self,
        rings: Range<usize>,
        shape: DigitShape,
        s: &'a mut Self::I16Scratch,
    ) -> Result<&'a [[i16; D]], AkitaError> {
        let rows = <Self as I8Rows<D>>::i8_rows(self, 0, 0, rings, shape, &mut s.0)?;
        s.1.fill([0; D]);
        for (dst, src) in s.1.iter_mut().zip(rows) {
            *dst = std::array::from_fn(|k| i16::from(src[k]));
        }
        Ok(&s.1)
    }
}

/// Packed small integers split into `num_digits` balanced planes.
pub(crate) struct DecomposedPackedDigits<'a, F: Field + CanonicalEncoding> {
    pub(crate) packed: PackedDigits<'a>,
    pub(crate) params: BalancedDecomposePow2Params<F>,
}

impl<F: Field + CanonicalEncoding> DecomposedPackedDigits<'_, F> {
    fn decode<'s, const D: usize>(
        &self,
        rings: Range<usize>,
        decoded: &'s mut [[i8; D]],
    ) -> Result<&'s [[i8; D]], AkitaError> {
        let decoded = &mut decoded[..rings.len()];
        self.packed
            .digits
            .decode_range(rings.start * D, decoded.as_flattened_mut())?;
        Ok(decoded)
    }
}

impl<F: Field + CanonicalEncoding, const D: usize> I8Rows<D> for DecomposedPackedDigits<'_, F> {
    type I8Scratch = (Vec<[i8; D]>, Vec<[i8; D]>);
    fn i8_scratch(&self, shape: DigitShape, max_rings: usize) -> Self::I8Scratch {
        (
            vec![[0; D]; max_rings],
            vec![[0; D]; shape.num_digits * max_rings],
        )
    }
    fn digit_abs_bound(&self, shape: DigitShape) -> u64 {
        1 << (shape.log_basis - 1)
    }
    fn i8_rows<'a>(
        &'a self,
        _block_index: usize,
        block_start: usize,
        rings: Range<usize>,
        shape: DigitShape,
        s: &'a mut Self::I8Scratch,
    ) -> Result<&'a [[i8; D]], AkitaError> {
        let width = rings.len() * shape.num_digits;
        let decoded = self.decode(block_start + rings.start..block_start + rings.end, &mut s.0)?;
        for (ring, planes) in decoded
            .iter()
            .zip(s.1[..width].chunks_mut(shape.num_digits))
        {
            self.params.decompose_small_signed_into(ring, planes);
        }
        Ok(&s.1[..width])
    }
}

impl<F: Field + CanonicalEncoding, const D: usize> I16Rows<D> for DecomposedPackedDigits<'_, F> {
    type I16Scratch = (Vec<[i8; D]>, Vec<[i16; D]>);
    fn i16_scratch(&self, shape: DigitShape, max_rings: usize) -> Self::I16Scratch {
        (
            vec![[0; D]; max_rings],
            vec![[0; D]; shape.num_digits * max_rings],
        )
    }
    fn i16_rows<'a>(
        &'a self,
        rings: Range<usize>,
        shape: DigitShape,
        s: &'a mut Self::I16Scratch,
    ) -> Result<&'a [[i16; D]], AkitaError> {
        let decoded = self.decode(rings, &mut s.0)?;
        s.1.fill([0; D]);
        for (ring, planes) in decoded.iter().zip(s.1.chunks_mut(shape.num_digits)) {
            self.params.decompose_small_signed_into(ring, planes);
        }
        Ok(&s.1)
    }
}

pub(crate) struct DigitRows<'a, const D: usize> {
    pub(crate) blocks: &'a [&'a [[i8; D]]],
    pub(crate) digit_abs_bound: u64,
}

impl<const D: usize> I8Rows<D> for DigitRows<'_, D> {
    type I8Scratch = ();
    fn i8_scratch(&self, _shape: DigitShape, _max_rings: usize) {}
    fn digit_abs_bound(&self, _shape: DigitShape) -> u64 {
        self.digit_abs_bound
    }
    fn i8_rows<'a>(
        &'a self,
        block_index: usize,
        _block_start: usize,
        rings: Range<usize>,
        _shape: DigitShape,
        _s: &'a mut (),
    ) -> Result<&'a [[i8; D]], AkitaError> {
        Ok(&self.blocks[block_index][rings])
    }
}
/// A ring's digit planes, independent of how the source supplied them.
pub(crate) enum DigitPlanes<'a, const D: usize> {
    I8(&'a [[i8; D]]),
    I16(&'a [[i16; D]]),
}

pub(crate) trait DigitPlaneSet<const D: usize> {
    fn accumulate_wide(
        self,
        acc: &mut [[i32; D]],
        challenge: ValidatedSparseChallenge<'_, D>,
        plan: &WidePlan<D>,
    );
    fn accumulate_narrow(
        self,
        acc: &mut [[i16; D]],
        challenge: &SparseChallenge,
        windowed: Option<&WindowedChallenge>,
    );
    fn accumulate_chunked_narrow(
        self,
        narrow: &mut [[i16; D]],
        wide: &mut [[i32; D]],
        challenge: &SparseChallenge,
        term_ranges: &[Range<usize>],
    );
}
