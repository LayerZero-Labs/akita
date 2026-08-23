//! Dense polynomial storage and constructors.

use super::prepared::{PreparedDenseStorage, PreparedDenseWitness};
use crate::backend::packed_digits::{
    PackedSignedDigitView, PackedSignedDigitWriter, PackedSignedDigits, VECTOR_LOAD_PADDING,
};
use crate::backend::poly_helpers::try_small_i8_cache_from_ring_coeffs;
use crate::kernels::linear::try_centered_i8;
use crate::validation::is_i8_log_basis;
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore};
use akita_types::{RingVec, SUPPORTED_COMMITMENT_RING_DIMS};
use std::borrow::Cow;
use std::sync::OnceLock;

/// Bound the unpacked parallel staging area while building the persistent
/// packed dense decomposition. The packed writer has its own bounded staging
/// buffer, so peak conversion scratch remains independent of witness size.
const DENSE_DECOMPOSITION_BATCH_BYTES: usize = 8 * 1024 * 1024;

/// Minimum physical flat coefficient length.
///
/// Physical storage is zero-padded to `max(1 << num_vars, 1024)` so that for
/// every supported commitment ring dimension through `D=1024` the live view slice
/// `num_ring_elems(D) * D = div_ceil(2^num_vars, D) * D` is in bounds: when
/// `2^num_vars >= D` the slice is exactly `2^num_vars` coefficients, and when
/// `2^num_vars < D` it is `D <= 1024` coefficients. The old per-`D` storage
/// zero-padded the tail of the last ring element; the physical zero padding
/// reproduces those coefficients exactly.
const MIN_FLAT_COEFF_LEN: usize =
    SUPPORTED_COMMITMENT_RING_DIMS[SUPPORTED_COMMITMENT_RING_DIMS.len() - 1];

/// Schedule-bound packed digit planes in ring-major, then digit-major order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DenseDigitCache {
    ring_d: usize,
    num_digits: usize,
    log_basis: u32,
    planes: PackedSignedDigits,
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
/// This is the private sub-slice counterpart of [`RingVec::as_ring_slice`]:
/// callers slice the live prefix for their view dimension first, then
/// reinterpret after checking divisibility.
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
}

impl<F: FieldCore + CanonicalField> DensePoly<F> {
    /// Pack field-element evaluations into flat dense storage.
    ///
    /// At dimension `D` the first `α = log₂(D)` variables
    /// become coefficient slots within each ring element; the remaining
    /// variables index ring elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `evals.len() != 2^num_vars`.
    pub fn from_field_evals<'a>(
        num_vars: usize,
        evals: impl Into<Cow<'a, [F]>>,
    ) -> Result<Self, AkitaError>
    where
        F: 'a,
    {
        let evals = evals.into();
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

        // Padding zeros are centered-0 (trivially small-i8), so a poly whose
        // live coefficients are all small stays all-small — identical to the
        // old per-ring check over the zero-padded last ring.
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;
        let mut small_i8_coeffs = Vec::with_capacity(physical_len);
        let mut all_small_i8 = true;
        for coeff in evals.iter() {
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

        // Reuse an owned evaluation vector. Borrowed inputs pay the same
        // single copy as before, while profile and application builders can
        // transfer large buffers without doubling their resident memory.
        let mut coeffs = evals.into_owned();
        coeffs.resize(physical_len, F::zero());

        Ok(Self {
            num_vars,
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
            coeffs: RingVec::from_coeffs(flat),
            small_i8_coeffs,
            digit_cache: OnceLock::new(),
        }
    }

    pub(super) fn digit_planes_for<const D: usize>(
        &self,
        num_digits: usize,
        log_basis: u32,
    ) -> Option<PackedSignedDigitView<'_>> {
        if !is_i8_log_basis(log_basis) {
            return None;
        }
        if let Some(cache) = self.digit_cache.get() {
            // A cache built at another dimension is not reused: returning
            // `None` falls back to the uncached path, exactly like a
            // too-large cache does. Under uniform-D this never triggers.
            return (cache.ring_d == D
                && cache.num_digits == num_digits
                && cache.log_basis == log_basis)
                .then(|| cache.planes.view());
        }

        let num_rings = self.num_ring_elems_at(D);
        let num_plane_coeffs = num_rings.checked_mul(num_digits)?.checked_mul(D)?;

        let _span = tracing::info_span!(
            "dense_digit_cache_build",
            packed_bytes = num_plane_coeffs
                .checked_mul(log_basis as usize)
                .and_then(|bits| bits.checked_add(7))
                .map(|bits| bits / 8),
            num_rings,
            num_digits,
            ring_dimension = D,
        )
        .entered();
        let rings = self.ring_coeffs::<D>().ok()?;
        let q = (-F::one()).to_canonical_u128() + 1;
        let params = BalancedDecomposePow2Params::new(num_digits, log_basis, q);
        let coeffs_per_ring = num_digits.checked_mul(D)?;
        if log_basis == 8 {
            let capacity = num_plane_coeffs.checked_add(VECTOR_LOAD_PADDING)?;
            let mut planes = Vec::with_capacity(capacity);
            planes.resize(num_plane_coeffs, 0i8);
            cfg_chunks_mut!(planes, coeffs_per_ring)
                .zip(cfg_iter!(rings))
                .for_each(|(dst, ring)| {
                    let (dst_planes, remainder) = dst.as_chunks_mut::<D>();
                    debug_assert!(remainder.is_empty());
                    ring.balanced_decompose_pow2_i8_into_with_params(dst_planes, &params);
                });
            let planes = PackedSignedDigits::from_balanced_i8_digits(planes, log_basis).ok()?;
            let _ = self.digit_cache.set(DenseDigitCache {
                ring_d: D,
                num_digits,
                log_basis,
                planes,
            });
            let cache = self.digit_cache.get()?;
            return (cache.ring_d == D
                && cache.num_digits == num_digits
                && cache.log_basis == log_basis)
                .then(|| cache.planes.view());
        }

        let mut writer = PackedSignedDigitWriter::new_balanced(num_plane_coeffs, log_basis).ok()?;
        let rings_per_batch = DENSE_DECOMPOSITION_BATCH_BYTES
            .checked_div(coeffs_per_ring)?
            .max(1);
        for (batch_index, ring_batch) in rings.chunks(rings_per_batch).enumerate() {
            let mut batch_planes = vec![0i8; ring_batch.len() * coeffs_per_ring];
            cfg_chunks_mut!(batch_planes, coeffs_per_ring)
                .zip(cfg_iter!(ring_batch))
                .for_each(|(dst, ring)| {
                    let (dst_planes, remainder) = dst.as_chunks_mut::<D>();
                    debug_assert!(remainder.is_empty());
                    ring.balanced_decompose_pow2_i8_into_with_params(dst_planes, &params);
                });
            debug_assert_eq!(
                writer.position(),
                batch_index * rings_per_batch * coeffs_per_ring
            );
            writer.write_at(writer.position(), &batch_planes).ok()?;
        }
        let planes = writer.finish().ok()?;
        let _ = self.digit_cache.set(DenseDigitCache {
            ring_d: D,
            num_digits,
            log_basis,
            planes,
        });
        let cache = self.digit_cache.get()?;
        (cache.ring_d == D && cache.num_digits == num_digits && cache.log_basis == log_basis)
            .then(|| cache.planes.view())
    }

    /// Consume a committed dense polynomial into its schedule-bound opening witness.
    ///
    /// Commitment selects the ring dimension and balanced decomposition, so
    /// this conversion is available only after a successful dense commitment
    /// has populated that exact schedule-bound representation. Fast bounded
    /// spans keep only packed digits; wider spans retain canonical storage.
    ///
    /// # Errors
    ///
    /// Returns an error when this polynomial has not yet been committed with
    /// an i8 balanced-digit schedule.
    pub fn into_prepared_witness(self) -> Result<PreparedDenseWitness<F>, AkitaError> {
        let cache = self.digit_cache.get().ok_or_else(|| {
            AkitaError::InvalidInput(
                "dense polynomial must be committed before preparing its opening witness".into(),
            )
        })?;
        let num_vars = self.num_vars;
        let ring_d = cache.ring_d;
        let num_digits = cache.num_digits;
        let log_basis = cache.log_basis;
        let digit_span = num_digits
            .checked_mul(log_basis as usize)
            .ok_or_else(|| AkitaError::InvalidInput("dense digit span overflow".into()))?;
        let storage = if digit_span <= 126 {
            PreparedDenseStorage::Packed(
                self.digit_cache
                    .into_inner()
                    .expect("dense digit cache was checked above")
                    .planes,
            )
        } else {
            PreparedDenseStorage::Canonical(self)
        };
        Ok(PreparedDenseWitness {
            num_vars,
            ring_d,
            num_digits,
            log_basis,
            storage,
        })
    }

    #[cfg(test)]
    pub(super) fn cached_digit_storage(&self) -> Option<(usize, u8)> {
        self.digit_cache
            .get()
            .map(|cache| (cache.planes.encoded_bytes().len(), cache.planes.bit_width()))
    }
}
