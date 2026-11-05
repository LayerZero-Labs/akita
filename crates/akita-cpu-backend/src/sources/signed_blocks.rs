//! Signed coefficient storage for shared commitment and opening kernels.

use crate::commitment::{AcceptedReach, SignedBlockDecoder, SignedValues};
use crate::sources::block_digits::{
    BlockDigits, DigitPlanes, DigitShape, I16Rows, I8Rows, RingValueSource, RingValues,
};
use crate::sources::poly_helpers::{I8DigitSpan, SignedDigitBasis};
use akita_error::AkitaError;
use akita_params::SignedDigitKernel;
use jolt_algebra::checked;
use jolt_algebra::ring::cyclotomic::{BalancedDecomposePow2Params, SmallSignedValue};
use jolt_algebra::{CanonicalEncoding, Field};
use std::marker::PhantomData;
use std::ops::Range;
use zeroize::Zeroize;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

mod kernels;
mod owned;
pub use owned::SignedBlockPoly;

pub(crate) struct I8Slice<'a>(pub(crate) &'a [i8]);
pub(crate) struct I16Slice<'a>(pub(crate) &'a [i16]);
pub(crate) struct Decoded<'a>(pub(crate) &'a dyn SignedBlockDecoder);

pub(crate) trait SignedStorage: Sync {
    type Value: Copy + Default + Ord + Into<i64> + SmallSignedValue + Send + Sync;
    const NEGATIVE_ABS: u128;
    const POSITIVE: u128;
    /// Every stored value, when the storage holds them in memory.
    fn resident(&self) -> Option<&[Self::Value]>;
    /// Writes values `start..start + out.len()`.
    fn load(&self, start: usize, out: &mut [Self::Value]);
    fn borrowed_i8<const D: usize>(&self, _rings: Range<usize>) -> Option<&[[i8; D]]> {
        None
    }
}

pub(crate) struct Scan {
    min: i64,
    max: i64,
    negative_abs: u128,
    positive: u128,
    accepted: Option<AcceptedReach>,
}

/// Values per task of the extrema pass.
const EXTREMA_CHUNK: usize = 1 << 15;

/// The least and greatest of `values` and zero, in the carrier type.
fn carrier_extrema<V: Copy + Default + Ord>(values: &[V]) -> (V, V) {
    values
        .iter()
        .fold((V::default(), V::default()), |(min, max), &value| {
            (min.min(value), max.max(value))
        })
}

fn slice_extrema<V: Copy + Default + Ord + Into<i64> + Send + Sync>(values: &[V]) -> (i64, i64) {
    #[cfg(feature = "parallel")]
    let (min, max) = values
        .par_chunks(EXTREMA_CHUNK)
        .map(carrier_extrema)
        .reduce(
            || (V::default(), V::default()),
            |a, b| (a.0.min(b.0), a.1.max(b.1)),
        );
    #[cfg(not(feature = "parallel"))]
    let (min, max) = carrier_extrema(values);
    (min.into(), max.into())
}

/// The least and greatest coefficient of a zero-extended signed table: one
/// pass in the carrier type, decoding in chunks when the table is encoded.
pub(crate) fn signed_extrema(values: SignedValues<'_>, len: usize) -> (i64, i64) {
    match values {
        SignedValues::I8(values) => slice_extrema(&values[..len]),
        SignedValues::I16(values) => slice_extrema(&values[..len]),
        SignedValues::DecodedI8(decoder) => {
            let chunk = |index: usize| {
                let mut buffer = [0i8; EXTREMA_CHUNK];
                let start = index * EXTREMA_CHUNK;
                let values = &mut buffer[..EXTREMA_CHUNK.min(len - start)];
                decoder.decode_i8(start, values);
                carrier_extrema(values)
            };
            let chunks = len.div_ceil(EXTREMA_CHUNK);
            #[cfg(feature = "parallel")]
            let (min, max) = (0..chunks)
                .into_par_iter()
                .map(chunk)
                .reduce(|| (0, 0), |a, b| (a.0.min(b.0), a.1.max(b.1)));
            #[cfg(not(feature = "parallel"))]
            let (min, max) = (0..chunks)
                .map(chunk)
                .fold((0, 0), |a, b| (a.0.min(b.0), a.1.max(b.1)));
            (min.into(), max.into())
        }
    }
}

/// Folds `values` into `scan`. The extrema pass stays in the carrier type so
/// it vectorizes; only an interval that wraps the carrier visits each value.
fn track<V: Copy + Ord + Into<i64>>(scan: &mut Option<Scan>, values: &[V]) {
    let (Some(scan), Some(&first)) = (scan.as_mut(), values.first()) else {
        return;
    };
    let (min, max) = values.iter().fold((first, first), |(min, max), &value| {
        (min.min(value), max.max(value))
    });
    scan.min = scan.min.min(min.into());
    scan.max = scan.max.max(max.into());
    if let Some(bounds) = scan.accepted {
        for &value in values {
            let (negative_abs, positive) = bounds.centered(value.into());
            scan.negative_abs = scan.negative_abs.max(negative_abs);
            scan.positive = scan.positive.max(positive);
        }
    }
}

impl AcceptedReach {
    pub(crate) fn centered(self, value: i64) -> (u128, u128) {
        let magnitude = u128::from(value.unsigned_abs());
        let magnitude = if magnitude < self.modulus {
            magnitude
        } else {
            magnitude % self.modulus
        };
        let canonical = if value < 0 && magnitude != 0 {
            self.modulus - magnitude
        } else {
            magnitude
        };
        if canonical > self.centering_threshold {
            (self.modulus - canonical, 0)
        } else {
            (0, canonical)
        }
    }

    fn preserves_carrier(self, negative_abs: u128, positive: u128) -> bool {
        self.centering_threshold >= positive
            && self.modulus - self.centering_threshold > negative_abs
    }

    fn covers_carrier(self, negative_abs: u128, positive: u128) -> bool {
        if self.modulus <= negative_abs || self.centering_threshold >= self.modulus {
            return false;
        }
        let negative_start = self.modulus - negative_abs;
        let positive_reach = positive.min(self.centering_threshold).max(
            if negative_start <= self.centering_threshold {
                self.centering_threshold
            } else {
                0
            },
        );
        let negative_reach = if positive > self.centering_threshold {
            self.modulus - self.centering_threshold - 1
        } else {
            negative_abs.min(self.modulus - self.centering_threshold - 1)
        };
        self.negative_abs
            .is_none_or(|bound| bound >= negative_reach)
            && self.positive.is_none_or(|bound| bound >= positive_reach)
    }
}

macro_rules! slice_storage {
    ($storage:ident, $value:tt, $negative:expr, $positive:expr) => {
        impl SignedStorage for $storage<'_> {
            type Value = $value;
            const NEGATIVE_ABS: u128 = $negative;
            const POSITIVE: u128 = $positive;
            fn resident(&self) -> Option<&[Self::Value]> {
                Some(self.0)
            }
            fn load(&self, start: usize, out: &mut [Self::Value]) {
                out.copy_from_slice(&self.0[start..start + out.len()]);
            }
            slice_storage!(@borrow $value);
        }
    };
    (@borrow i8) => {
        fn borrowed_i8<const D: usize>(&self, rings: Range<usize>) -> Option<&[[i8; D]]> {
            self.0.as_chunks::<D>().0.get(rings)
        }
    };
    (@borrow i16) => {};
}
slice_storage!(I8Slice, i8, 128, 127);
slice_storage!(I16Slice, i16, 32768, 32767);

impl SignedStorage for Decoded<'_> {
    type Value = i8;
    const NEGATIVE_ABS: u128 = 128;
    const POSITIVE: u128 = 127;
    fn resident(&self) -> Option<&[i8]> {
        None
    }
    fn load(&self, start: usize, out: &mut [i8]) {
        self.0.decode_i8(start, out);
    }
}

/// Whole-source extrema known when the source is built for a commit.
#[derive(Clone, Copy)]
struct Known {
    min: i64,
    max: i64,
    within_bounds: bool,
}

pub(crate) struct SignedBlockDigits<'a, F, St, const D: usize> {
    storage: St,
    len: usize,
    num_rings: usize,
    accepted: Option<AcceptedReach>,
    check_required: bool,
    known: Option<Known>,
    marker: PhantomData<&'a F>,
}

impl<'a, F, St, const D: usize> SignedBlockDigits<'a, F, St, D>
where
    F: Field + CanonicalEncoding,
    St: SignedStorage,
{
    /// `extrema`, when given, are the least and greatest stored value; with
    /// an accepted interval that keeps the carrier centered they settle the
    /// bound check once instead of per row request.
    pub(crate) fn new(
        storage: St,
        num_vars: usize,
        len: usize,
        accepted: Option<AcceptedReach>,
        extrema: Option<(i64, i64)>,
    ) -> Result<Self, AkitaError> {
        if D == 0 || !D.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "signed block ring dimension must be a power of two".into(),
            ));
        }
        let domain = checked::pow2(num_vars)
            .ok_or_else(|| AkitaError::InvalidInput("signed block extent overflow".into()))?;
        if len > domain {
            return Err(AkitaError::InvalidInput(
                "signed block length exceeds its domain".into(),
            ));
        }
        let check_required =
            accepted.is_some_and(|bounds| !bounds.covers_carrier(St::NEGATIVE_ABS, St::POSITIVE));
        let mut source = Self {
            storage,
            len,
            num_rings: domain.div_ceil(D),
            accepted,
            check_required,
            known: None,
            marker: PhantomData,
        };
        // An interval that wraps the carrier needs every value, per request.
        if let (Some(_), Some((min, max)), None) = (accepted, extrema, source.per_value_bounds()) {
            let scan = Some(Scan {
                min,
                max,
                negative_abs: 0,
                positive: 0,
                accepted: None,
            });
            source.known = Some(Known {
                min,
                max,
                within_bounds: source.check(&scan).is_ok(),
            });
        }
        Ok(source)
    }

    /// The accepted interval when it wraps the carrier, so that only each
    /// value's centered representative decides the check.
    fn per_value_bounds(&self) -> Option<AcceptedReach> {
        self.accepted.filter(|bounds| {
            self.check_required && !bounds.preserves_carrier(St::NEGATIVE_ABS, St::POSITIVE)
        })
    }

    /// A per-request scan, unless the known extrema already cover it.
    fn scan(&self, digits: bool) -> Option<Scan> {
        if self.known.is_some() || !(self.check_required || digits) {
            return None;
        }
        Some(Scan {
            min: 0,
            max: 0,
            negative_abs: 0,
            positive: 0,
            accepted: self.per_value_bounds(),
        })
    }

    fn check(&self, scan: &Option<Scan>) -> Result<(), AkitaError> {
        if self.known.is_some_and(|known| !known.within_bounds) {
            return Err(AkitaError::InvalidInput(
                "committed source exceeds the scheduled coefficient bound".into(),
            ));
        }
        if let (Some(scan), Some(bounds)) = (scan, self.accepted.filter(|_| self.check_required)) {
            let (negative_abs, positive) = if scan.accepted.is_some() {
                (scan.negative_abs, scan.positive)
            } else {
                (u128::from(scan.min.unsigned_abs()), scan.max as u128)
            };
            if bounds
                .negative_abs
                .is_some_and(|bound| negative_abs > bound)
                || bounds.positive.is_some_and(|bound| positive > bound)
            {
                return Err(AkitaError::InvalidInput(
                    "committed source exceeds the scheduled coefficient bound".into(),
                ));
            }
        }
        Ok(())
    }

    fn single_fits(&self, shape: DigitShape) -> bool {
        let half = 1u128 << (shape.log_basis - 1);
        (St::NEGATIVE_ABS <= half && St::POSITIVE < half)
            || self.accepted.is_some_and(|bounds| {
                bounds.preserves_carrier(St::NEGATIVE_ABS, St::POSITIVE)
                    && bounds.negative_abs.is_some_and(|bound| bound <= half)
                    && bounds.positive.is_some_and(|bound| bound < half)
            })
    }

    /// Whether one-digit rows equal the values themselves: the carrier or
    /// the accepted interval proves it, or the scanned extrema show it.
    fn digits_fit(&self, scan: &Option<Scan>, shape: DigitShape) -> bool {
        let half = 1i64 << (shape.log_basis - 1);
        let extrema = self
            .known
            .map(|known| (known.min, known.max))
            .or(scan.as_ref().map(|scan| (scan.min, scan.max)));
        self.single_fits(shape) || extrema.is_some_and(|(min, max)| min >= -half && max < half)
    }

    /// Rings `rings` of the zero-extended source, borrowed from resident
    /// storage when fully live and otherwise staged in `raw`.
    fn source_rings<'s>(
        &'s self,
        rings: Range<usize>,
        raw: &'s mut [[St::Value; D]],
        scan: &mut Option<Scan>,
    ) -> Result<&'s [[St::Value; D]], AkitaError> {
        if rings.start > rings.end || rings.end > self.num_rings || raw.len() < rings.len() {
            return Err(AkitaError::InvalidInput(
                "signed block range exceeds its source".into(),
            ));
        }
        if rings.end <= self.len / D {
            if let Some(values) = self.storage.resident() {
                let rows = &values.as_chunks::<D>().0[rings];
                track(scan, rows.as_flattened());
                self.check(scan)?;
                return Ok(rows);
            }
        }
        let raw = &mut raw[..rings.len()];
        let start = rings.start * D;
        let live = self.len.saturating_sub(start).min(rings.len() * D);
        let (live_raw, tail) = raw.as_flattened_mut().split_at_mut(live);
        if live != 0 {
            self.storage.load(start, live_raw);
            track(scan, live_raw);
        }
        tail.fill(St::Value::default());
        self.check(scan)?;
        Ok(raw)
    }

    /// Writes the digit rows of `rings` into `out`, `num_digits` per ring.
    fn write_rows<O: Copy>(
        &self,
        rings: Range<usize>,
        shape: DigitShape,
        raw: &mut [[St::Value; D]],
        out: &mut [[O; D]],
        narrow: impl Fn(i32) -> O,
        decompose: impl Fn(&[St::Value; D], &mut [[O; D]]),
    ) -> Result<(), AkitaError> {
        let single = shape.num_digits == 1;
        let mut scan = self.scan(single && !self.single_fits(shape));
        let source = self.source_rings(rings, raw, &mut scan)?;
        if single && self.digits_fit(&scan, shape) {
            for (dst, &value) in out.as_flattened_mut().iter_mut().zip(source.as_flattened()) {
                *dst = narrow(value.widen());
            }
        } else {
            for (ring, planes) in source.iter().zip(out.chunks_mut(shape.num_digits)) {
                decompose(ring, planes);
            }
        }
        Ok(())
    }
}

pub(crate) struct SignedScratch<F: Field + CanonicalEncoding, V, const D: usize> {
    params: BalancedDecomposePow2Params<F>,
    raw: Vec<[V; D]>,
    /// Digit rows per request; the row buffers are allocated on first use.
    width: usize,
    i8: Vec<[i8; D]>,
    i16: Vec<[i16; D]>,
}

impl<F: Field + CanonicalEncoding, V: Copy + Default, const D: usize> SignedScratch<F, V, D> {
    fn new(shape: DigitShape, max_rings: usize) -> Self {
        Self {
            params: BalancedDecomposePow2Params::new(shape.num_digits, shape.log_basis),
            raw: vec![[V::default(); D]; max_rings],
            width: max_rings * shape.num_digits,
            i8: Vec::new(),
            i16: Vec::new(),
        }
    }
}

/// The staged values and digit rows copy the source, which may be a secret
/// witness (`SignedBlockPoly`, A45): wiped over their whole capacity.
/// Borrowed rows never pass through here, so on that path they hold zeros.
impl<F: Field + CanonicalEncoding, V, const D: usize> Drop for SignedScratch<F, V, D> {
    fn drop(&mut self) {
        wipe_scratch(&mut self.raw);
        wipe_scratch(&mut self.i8);
        wipe_scratch(&mut self.i16);
    }
}

/// Zeroes `values`' whole capacity and empties it.
fn wipe_scratch<T>(values: &mut Vec<T>) {
    values.clear();
    values.spare_capacity_mut().zeroize();
}

fn row_buffer<T: Copy + Default, const D: usize>(
    rows: &mut Vec<[T; D]>,
    width: usize,
) -> &mut [[T; D]] {
    if rows.len() < width {
        // Growing frees the old buffer, which may hold secret rows.
        wipe_scratch(rows);
        rows.resize(width, [T::default(); D]);
    }
    rows
}

impl<F, St, const D: usize> RingValueSource<F, D> for SignedBlockDigits<'_, F, St, D>
where
    F: Field,
    St: SignedStorage,
    [St::Value; D]: RingValues<F, D>,
{
    type Values<'a>
        = [St::Value; D]
    where
        Self: 'a;
    fn num_rings(&self) -> usize {
        self.num_rings
    }
    fn values(&self, ring: usize) -> Result<Self::Values<'_>, AkitaError> {
        if ring >= self.num_rings {
            return Err(AkitaError::InvalidInput(
                "ring value index is outside the source".into(),
            ));
        }
        let mut values = [St::Value::default(); D];
        let start = ring * D;
        let live = self.len.saturating_sub(start).min(D);
        if live != 0 {
            self.storage.load(start, &mut values[..live]);
        }
        Ok(values)
    }
}

impl<F, St, const D: usize> BlockDigits<D> for SignedBlockDigits<'_, F, St, D>
where
    F: Field + CanonicalEncoding,
    St: SignedStorage,
    [St::Value; D]: RingValues<F, D>,
{
    type Scratch = SignedScratch<F, St::Value, D>;
    type Planes<'a>
        = DigitPlanes<'a, D>
    where
        Self: 'a;
    fn num_rings(&self) -> usize {
        self.num_rings
    }
    fn stored_rings(&self) -> usize {
        self.len.div_ceil(D)
    }
    fn scratch(&self, shape: DigitShape, max_rings: usize) -> Self::Scratch {
        SignedScratch::new(shape, max_rings)
    }
    fn digit_abs_bound(&self, shape: DigitShape) -> u64 {
        1 << (shape.log_basis - 1)
    }
    fn i8_digit_span(&self, shape: DigitShape) -> Option<I8DigitSpan> {
        matches!(
            SignedDigitKernel::for_log_basis(shape.log_basis),
            Some(SignedDigitKernel::I8)
        )
        .then_some(if shape.log_basis <= 7 {
            I8DigitSpan::Half
        } else {
            I8DigitSpan::Full
        })
    }
    fn digit_planes<'a>(
        &'a self,
        ring: usize,
        shape: DigitShape,
        s: &'a mut Self::Scratch,
    ) -> Result<Self::Planes<'a>, AkitaError> {
        match SignedDigitBasis::new(shape.log_basis)?.kernel {
            SignedDigitKernel::I8 => Ok(DigitPlanes::I8(self.rows_i8(ring..ring + 1, shape, s)?)),
            SignedDigitKernel::I16 => Ok(DigitPlanes::I16(
                &self.rows_i16(ring..ring + 1, shape, s)?[..shape.num_digits],
            )),
        }
    }
}

impl<F, St, const D: usize> SignedBlockDigits<'_, F, St, D>
where
    F: Field + CanonicalEncoding,
    St: SignedStorage,
    [St::Value; D]: RingValues<F, D>,
{
    fn rows_i8<'a>(
        &'a self,
        rings: Range<usize>,
        shape: DigitShape,
        s: &'a mut SignedScratch<F, St::Value, D>,
    ) -> Result<&'a [[i8; D]], AkitaError> {
        if shape.num_digits == 1 && rings.start <= rings.end && rings.end <= self.len / D {
            if let Some(rows) = self.storage.borrowed_i8::<D>(rings.clone()) {
                let mut scan = self.scan(!self.single_fits(shape));
                track(&mut scan, rows.as_flattened());
                self.check(&scan)?;
                if self.digits_fit(&scan, shape) {
                    return Ok(rows);
                }
            }
        }
        let width = rings.len() * shape.num_digits;
        let SignedScratch {
            params, raw, i8, ..
        } = s;
        let out = &mut row_buffer(i8, width)[..width];
        self.write_rows(
            rings,
            shape,
            raw,
            out,
            |value| value as i8,
            |ring, planes| params.decompose_small_signed_into(ring, planes),
        )?;
        Ok(out)
    }

    /// The `I16Rows` contract: every row of the scratch, zero past `rings`.
    fn rows_i16<'a>(
        &'a self,
        rings: Range<usize>,
        shape: DigitShape,
        s: &'a mut SignedScratch<F, St::Value, D>,
    ) -> Result<&'a [[i16; D]], AkitaError> {
        let width = rings.len() * shape.num_digits;
        let SignedScratch {
            params,
            raw,
            width: rows,
            i16,
            ..
        } = s;
        let all = row_buffer(i16, (*rows).max(width));
        let (out, tail) = all.split_at_mut(width);
        self.write_rows(
            rings,
            shape,
            raw,
            out,
            |value| value as i16,
            |ring, planes| params.decompose_small_signed_into(ring, planes),
        )?;
        tail.fill([0; D]);
        Ok(all)
    }
}

impl<F, St, const D: usize> I8Rows<D> for SignedBlockDigits<'_, F, St, D>
where
    F: Field + CanonicalEncoding,
    St: SignedStorage,
    [St::Value; D]: RingValues<F, D>,
{
    type I8Scratch = SignedScratch<F, St::Value, D>;
    fn i8_scratch(&self, shape: DigitShape, max_rings: usize) -> Self::I8Scratch {
        SignedScratch::new(shape, max_rings)
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
        self.rows_i8(block_start + rings.start..block_start + rings.end, shape, s)
    }
}

impl<F, St, const D: usize> I16Rows<D> for SignedBlockDigits<'_, F, St, D>
where
    F: Field + CanonicalEncoding,
    St: SignedStorage,
    [St::Value; D]: RingValues<F, D>,
{
    type I16Scratch = SignedScratch<F, St::Value, D>;
    fn i16_scratch(&self, shape: DigitShape, max_rings: usize) -> Self::I16Scratch {
        SignedScratch::new(shape, max_rings)
    }
    fn i16_rows<'a>(
        &'a self,
        rings: Range<usize>,
        shape: DigitShape,
        s: &'a mut Self::I16Scratch,
    ) -> Result<&'a [[i16; D]], AkitaError> {
        self.rows_i16(rings, shape, s)
    }
}
