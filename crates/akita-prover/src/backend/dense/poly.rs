//! Dense polynomial storage and constructors.

use crate::backend::poly_helpers::try_small_i8_cache_from_ring_coeffs;
use crate::kernels::linear::try_centered_i8;
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_types::{tensor_opening_split, RingVec, TensorColumnSource};
use jolt_field::parallel::*;
use jolt_field::{CanonicalField, ExtField, FieldCore};
use std::mem::size_of;
use std::sync::OnceLock;

const MAX_DENSE_DIGIT_CACHE_BYTES: usize = 512 * 1024 * 1024;

/// Minimum physical flat coefficient length.
///
/// Physical storage is zero-padded to `max(1 << num_vars, 256)` so that for
/// every supported ring dimension `D ∈ {32, 64, 128, 256}` the live view slice
/// `num_ring_elems(D) * D = div_ceil(2^num_vars, D) * D` is in bounds: when
/// `2^num_vars >= D` the slice is exactly `2^num_vars` coefficients, and when
/// `2^num_vars < D` it is `D <= 256` coefficients. The old per-`D` storage
/// zero-padded the tail of the last ring element; the physical zero padding
/// reproduces those coefficients exactly.
const MIN_FLAT_COEFF_LEN: usize = 256;

/// D-free digit-plane cache: `num_digits` planes of `ring_d` bytes per ring
/// element, flattened in ring-major order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DenseDigitCache {
    ring_d: usize,
    num_digits: usize,
    log_basis: u32,
    planes: Vec<i8>,
}

/// Dense polynomial: all ring coefficients materialized in memory.
///
/// Storage is D-free: coefficients are a flat field-element buffer, and the
/// ring dimension is a view selected at kernel entry (each ring-shaped method
/// takes it as a const generic).
#[derive(Debug)]
pub struct DensePoly<F: FieldCore> {
    /// Actual multilinear variable count of the source witness.
    pub(super) num_vars: usize,
    /// Ring-element count at the CONSTRUCTION dimension; metadata, not
    /// authority — kernels compute their ring count at their own dimension.
    meta_ring_elems: usize,
    /// Flat field coefficients in sequential block order (untagged compact
    /// [`RingVec`]; see [`MIN_FLAT_COEFF_LEN`] for the physical padding).
    coeffs: RingVec<F>,
    /// Flat centered-`i8` mirror of `coeffs` (same physical length and
    /// padding), present only when every live coefficient is small.
    pub(super) small_i8_coeffs: Option<Vec<i8>>,
    digit_cache: OnceLock<DenseDigitCache>,
}

impl<F: FieldCore + Clone> Clone for DensePoly<F> {
    fn clone(&self) -> Self {
        Self {
            num_vars: self.num_vars,
            meta_ring_elems: self.meta_ring_elems,
            coeffs: self.coeffs.clone(),
            small_i8_coeffs: self.small_i8_coeffs.clone(),
            digit_cache: OnceLock::new(),
        }
    }
}

impl<F: FieldCore + PartialEq> PartialEq for DensePoly<F> {
    fn eq(&self, other: &Self) -> bool {
        self.num_vars == other.num_vars
            && self.coeffs == other.coeffs
            && self.small_i8_coeffs == other.small_i8_coeffs
    }
}

impl<F: FieldCore + Eq> Eq for DensePoly<F> {}

/// Reinterpret a flat coefficient slice as ring elements of dimension `D`.
///
/// This is the sub-slice counterpart of [`RingVec::as_ring_slice_trusted`]:
/// callers slice the live prefix for their view dimension first, then
/// reinterpret.
#[inline]
fn as_ring_view<F: FieldCore, const D: usize>(flat: &[F]) -> &[CyclotomicRing<F, D>] {
    debug_assert!(D > 0);
    debug_assert!(flat.len().is_multiple_of(D));
    // SAFETY: `CyclotomicRing<F, D>` is `#[repr(transparent)]` over `[F; D]`,
    // and the length is a multiple of `D`.
    unsafe {
        std::slice::from_raw_parts(flat.as_ptr() as *const CyclotomicRing<F, D>, flat.len() / D)
    }
}

impl<F: FieldCore> DensePoly<F> {
    /// Full physical (zero-padded) flat coefficient buffer.
    pub fn field_coeffs(&self) -> &[F] {
        self.coeffs.coeffs()
    }

    /// Ring-element count at the construction dimension (metadata only).
    pub(super) fn meta_ring_elems(&self) -> usize {
        self.meta_ring_elems
    }

    /// Ring-element count viewed at dimension `ring_d`.
    #[inline]
    pub(super) fn num_ring_elems_at(&self, ring_d: usize) -> usize {
        (1usize << self.num_vars).div_ceil(ring_d)
    }

    /// Live view of the coefficients as ring elements of dimension `D`.
    ///
    /// The view covers `num_ring_elems_at(D)` ring elements read from the flat
    /// prefix `[..num_ring_elems_at(D) * D]`; the physical zero padding
    /// supplies the tail of the last ring exactly as the old per-`D` storage
    /// did.
    ///
    /// # Errors
    ///
    /// Returns an error if `D` is not a power of two or the view exceeds the
    /// physical buffer (unsupported ring dimension for this arity).
    pub fn ring_coeffs<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        if D == 0 || !D.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "ring degree D={D} is not a power of two"
            )));
        }
        let needed = self
            .num_ring_elems_at(D)
            .checked_mul(D)
            .ok_or_else(|| AkitaError::InvalidInput("dense ring view overflow".to_string()))?;
        let flat = self.field_coeffs();
        let live = flat.get(..needed).ok_or_else(|| {
            AkitaError::InvalidInput(format!(
                "dense ring view at D={D} needs {needed} coefficients but only {} are stored",
                flat.len()
            ))
        })?;
        Ok(as_ring_view::<F, D>(live))
    }

    /// Live small-i8 mirror viewed as per-ring coefficient planes at `D`.
    pub(super) fn small_i8_ring_coeffs<const D: usize>(&self) -> Option<&[[i8; D]]> {
        let flat = self.small_i8_coeffs.as_deref()?;
        let needed = self.num_ring_elems_at(D).checked_mul(D)?;
        let (chunks, remainder) = flat.get(..needed)?.as_chunks::<D>();
        debug_assert!(remainder.is_empty());
        Some(chunks)
    }

    /// Live (unpadded) flat coefficient count, `1 << num_vars`.
    pub(super) fn live_coeff_len(&self) -> Result<usize, AkitaError> {
        1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })
    }

    pub(super) fn tensor_shape<E, const D: usize>(
        &self,
        logical_point: Option<&[E]>,
    ) -> Result<(usize, usize), AkitaError>
    where
        E: ExtField<F>,
    {
        let (split_bits, width) = tensor_opening_split::<F, E>()?;
        if split_bits > self.num_vars {
            return Err(AkitaError::InvalidInput(
                "extension-opening tensor split exceeds polynomial arity".to_string(),
            ));
        }
        if width > D || !D.is_multiple_of(width) {
            return Err(AkitaError::InvalidInput(format!(
                "extension degree {width} does not evenly pack into dense ring degree {D}"
            )));
        }
        if let Some(point) = logical_point {
            if point.len() != self.num_vars {
                return Err(AkitaError::InvalidPointDimension {
                    expected: self.num_vars,
                    actual: point.len(),
                });
            }
        }
        Ok((split_bits, width))
    }
}

impl<F: FieldCore + CanonicalField> DensePoly<F> {
    /// Pack field-element evaluations into flat dense storage.
    ///
    /// `ring_d` is the caller's configured ring dimension. It is recorded as
    /// construction metadata (the [`crate::compute::RootPolyMeta`] ring-element
    /// count) only; the storage itself is D-free and kernels select their view
    /// dimension at entry. At dimension `D` the first `α = log₂(D)` variables
    /// become coefficient slots within each ring element; the remaining
    /// variables index ring elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `ring_d` is not a power of two or if
    /// `evals.len() != 2^num_vars`.
    pub fn from_field_evals(
        num_vars: usize,
        ring_d: usize,
        evals: &[F],
    ) -> Result<Self, AkitaError> {
        if ring_d == 0 || !ring_d.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "ring degree D={ring_d} is not a power of two"
            )));
        }
        let expected_len = 1usize
            .checked_shl(num_vars as u32)
            .ok_or_else(|| AkitaError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if evals.len() != expected_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: evals.len(),
            });
        }

        let physical_len = expected_len.max(MIN_FLAT_COEFF_LEN);
        let mut coeffs = Vec::with_capacity(physical_len);
        coeffs.extend_from_slice(evals);
        coeffs.resize(physical_len, F::zero());

        // Padding zeros are centered-0 (trivially small-i8), so a poly whose
        // live coefficients are all small stays all-small — identical to the
        // old per-ring check over the zero-padded last ring.
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;
        let mut small_i8_coeffs = Vec::with_capacity(physical_len);
        let mut all_small_i8 = true;
        for coeff in evals {
            if let Some(centered) = try_centered_i8(*coeff, q, half_q) {
                small_i8_coeffs.push(centered);
            } else {
                all_small_i8 = false;
                break;
            }
        }
        if all_small_i8 {
            small_i8_coeffs.resize(physical_len, 0);
        }

        Ok(Self {
            num_vars,
            meta_ring_elems: expected_len.div_ceil(ring_d),
            coeffs: RingVec::from_coeffs(coeffs),
            small_i8_coeffs: all_small_i8.then_some(small_i8_coeffs),
            digit_cache: OnceLock::new(),
        })
    }

    /// Flatten an existing vector of ring elements into dense storage.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs.len() * D` overflows `usize`.
    pub fn from_ring_coeffs<const D: usize>(coeffs: Vec<CyclotomicRing<F, D>>) -> Self {
        let total = coeffs
            .len()
            .checked_mul(D)
            .expect("ring elems * D overflow");
        let physical_len = total.max(MIN_FLAT_COEFF_LEN);

        let small_i8_coeffs = try_small_i8_cache_from_ring_coeffs(&coeffs).map(|planes| {
            let mut flat = Vec::with_capacity(physical_len);
            for plane in &planes {
                flat.extend_from_slice(plane);
            }
            flat.resize(physical_len, 0i8);
            flat
        });

        let mut flat = Vec::with_capacity(physical_len);
        for ring in &coeffs {
            flat.extend_from_slice(ring.coefficients());
        }
        flat.resize(physical_len, F::zero());

        Self {
            num_vars: total.trailing_zeros() as usize,
            meta_ring_elems: coeffs.len(),
            coeffs: RingVec::from_coeffs(flat),
            small_i8_coeffs,
            digit_cache: OnceLock::new(),
        }
    }

    pub(super) fn digit_planes_for<const D: usize>(
        &self,
        num_digits: usize,
        log_basis: u32,
    ) -> Option<&[[i8; D]]> {
        if let Some(cache) = self.digit_cache.get() {
            // A cache built at another dimension is not reused: returning
            // `None` falls back to the uncached path, exactly like a
            // too-large cache does. Under uniform-D this never triggers.
            return (cache.ring_d == D
                && cache.num_digits == num_digits
                && cache.log_basis == log_basis)
                .then(|| {
                    let (chunks, remainder) = cache.planes.as_chunks::<D>();
                    debug_assert!(remainder.is_empty());
                    chunks
                });
        }

        let num_rings = self.num_ring_elems_at(D);
        let cache_bytes = num_rings
            .checked_mul(num_digits)?
            .checked_mul(size_of::<[i8; D]>())?;
        if cache_bytes > MAX_DENSE_DIGIT_CACHE_BYTES {
            return None;
        }

        let rings = self.ring_coeffs::<D>().ok()?;
        let q = (-F::one()).to_canonical_u128() + 1;
        let params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);
        let mut planes = vec![0i8; num_rings * num_digits * D];
        cfg_chunks_mut!(planes, num_digits * D)
            .zip(cfg_iter!(rings))
            .for_each(|(dst, ring)| {
                let (dst_planes, remainder) = dst.as_chunks_mut::<D>();
                debug_assert!(remainder.is_empty());
                ring.balanced_decompose_pow2_i8_into_with_params(dst_planes, &params);
            });
        let _ = self.digit_cache.set(DenseDigitCache {
            ring_d: D,
            num_digits,
            log_basis,
            planes,
        });
        let cache = self.digit_cache.get()?;
        (cache.ring_d == D && cache.num_digits == num_digits && cache.log_basis == log_basis).then(
            || {
                let (chunks, remainder) = cache.planes.as_chunks::<D>();
                debug_assert!(remainder.is_empty());
                chunks
            },
        )
    }
}

/// Column source over the flat dense coefficient buffer: `row(tail)` is the
/// `width`-length base-field run at flat index `tail*width`.
///
/// The old ring-typed source computed `ring_idx = tail*width / D` and sliced
/// within that ring; runs are `width`-aligned and `width` divides `D`, so a
/// run never crossed a ring boundary and flat indexing reads the identical
/// coefficients. `flat` is bounded to the LIVE length (`1 << num_vars`); the
/// tensor fold's tails cover exactly that range.
pub(super) struct DenseColumnSource<'a, F: FieldCore> {
    pub(super) flat: &'a [F],
    pub(super) width: usize,
}

impl<F: FieldCore> TensorColumnSource<F> for DenseColumnSource<'_, F> {
    #[inline]
    fn row(&self, tail: usize) -> &[F] {
        &self.flat[tail * self.width..][..self.width]
    }
}
