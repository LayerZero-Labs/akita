//! Dense polynomial storage and constructors.

use crate::sources::poly_helpers::SignedDigitBasis;
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_error::{checked, AkitaError};
use akita_params::SUPPORTED_COMMITMENT_RING_DIMS;
use akita_types::{AkitaExpandedSetup, RingVec};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};
use std::borrow::Cow;
use std::mem::size_of;
use std::sync::{Arc, OnceLock};

const MAX_DENSE_DIGIT_CACHE_BYTES: usize = 512 * 1024 * 1024;

fn nonzero_prefix_len<F: Field>(values: &[F]) -> usize {
    // Dense inputs usually end in a nonzero coefficient. Avoid launching a
    // parallel search for that common case or for small sources.
    if values.last().is_some_and(|value| *value != F::zero()) {
        return values.len();
    }
    #[cfg(feature = "parallel")]
    if values.len() > 4096 {
        return values
            .par_iter()
            .position_last(|value| *value != F::zero())
            .map_or(0, |i| i + 1);
    }
    values
        .iter()
        .rposition(|value| *value != F::zero())
        .map_or(0, |i| i + 1)
}

/// Alignment needed for a view at every supported commitment ring dimension.
/// Full sources allocate at least this many coefficients; compact sources
/// round their nonzero prefix up to this alignment.
const MIN_FLAT_COEFF_LEN: usize =
    SUPPORTED_COMMITMENT_RING_DIMS[SUPPORTED_COMMITMENT_RING_DIMS.len() - 1];

/// D-free digit-plane cache: `num_digits` planes of `ring_d` bytes per ring
/// element, flattened in ring-major order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DenseDigitCache {
    ring_d: usize,
    num_digits: usize,
    log_basis: u32,
    basis: SignedDigitBasis,
    planes: Vec<i8>,
}

/// Dense polynomial with an optional implicit zero suffix.
///
/// Storage is D-free: coefficients are a flat field-element buffer, and the
/// ring dimension is a view selected at kernel entry (each ring-shaped method
/// takes it as a const generic).
#[derive(Debug)]
pub struct DensePoly<F: Field> {
    /// Actual multilinear variable count of the source witness.
    pub(super) num_vars: usize,
    /// Flat field coefficients in sequential block order (untagged compact
    /// [`RingVec`]; see [`MIN_FLAT_COEFF_LEN`] for the physical padding).
    coeffs: DenseCoefficients<F>,
    nonzero_coeff_len: usize,
    digit_cache: OnceLock<DenseDigitCache>,
}

// Setup-prefix polynomials retain a bounded view of the backend's single setup.
// Ordinary polynomials continue to own their coefficients.
#[derive(Debug, Clone)]
enum DenseCoefficients<F: Field> {
    Owned(RingVec<F>),
    SetupPrefix {
        setup: Arc<AkitaExpandedSetup<F>>,
        len: usize,
    },
}

impl<F: Field + Clone> Clone for DensePoly<F> {
    fn clone(&self) -> Self {
        Self {
            num_vars: self.num_vars,
            coeffs: self.coeffs.clone(),
            nonzero_coeff_len: self.nonzero_coeff_len,
            digit_cache: OnceLock::new(),
        }
    }
}

impl<F: Field + PartialEq> PartialEq for DensePoly<F> {
    fn eq(&self, other: &Self) -> bool {
        let left = self.field_coeffs();
        let right = other.field_coeffs();
        let common = left.len().min(right.len());
        self.num_vars == other.num_vars
            && left[..common] == right[..common]
            && left[common..]
                .iter()
                .chain(&right[common..])
                .all(|x| *x == F::zero())
    }
}

impl<F: Field + Eq> Eq for DensePoly<F> {}

/// Reinterpret a flat coefficient slice as ring elements of dimension `D`.
///
/// This is the private sub-slice counterpart of [`RingVec::as_ring_slice`]:
/// callers slice the live prefix for their view dimension first, then
/// reinterpret after checking divisibility.
#[inline]
fn as_ring_view<F: Field, const D: usize>(flat: &[F]) -> &[CyclotomicRing<F, D>] {
    debug_assert!(D > 0);
    debug_assert!(flat.len().is_multiple_of(D));
    // SAFETY: `CyclotomicRing<F, D>` is `#[repr(transparent)]` over `[F; D]`,
    // and the length is a multiple of `D`.
    unsafe {
        std::slice::from_raw_parts(flat.as_ptr() as *const CyclotomicRing<F, D>, flat.len() / D)
    }
}

impl<F: Field> DensePoly<F> {
    /// Stored flat coefficients, including ring-alignment padding.
    ///
    /// Sources built with [`Self::from_field_prefix`] may store fewer than
    /// `2^num_vars` coefficients; the remaining evaluations are zero.
    pub fn field_coeffs(&self) -> &[F] {
        match &self.coeffs {
            DenseCoefficients::Owned(coeffs) => coeffs.coeffs(),
            // The constructor checks this bound; the setup is immutable.
            DenseCoefficients::SetupPrefix { setup, len } => {
                &setup.shared_matrix().as_field_slice()[..*len]
            }
        }
    }

    pub(crate) const fn nonzero_coeff_len(&self) -> usize {
        self.nonzero_coeff_len
    }

    pub(crate) fn nonzero_ring_coeffs<const D: usize>(
        &self,
    ) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ring_prefix::<D>(self.nonzero_coeff_len)
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
    /// physical buffer. A compact source may omit this view's zero suffix;
    /// opening kernels consume its stored prefix instead.
    pub fn ring_coeffs<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ring_prefix::<D>(1usize << self.num_vars)
    }

    fn ring_prefix<const D: usize>(
        &self,
        coefficient_len: usize,
    ) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        if D == 0 || !D.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "ring degree D={D} is not a power of two"
            )));
        }
        let needed = checked::align_up(coefficient_len, D)
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
}

impl<F: Field + CanonicalEncoding> DensePoly<F> {
    pub(crate) fn from_setup_prefix(
        setup: Arc<AkitaExpandedSetup<F>>,
        num_vars: usize,
    ) -> Result<Self, AkitaError> {
        let len = checked::pow2(num_vars)
            .ok_or_else(|| AkitaError::InvalidInput("setup-prefix domain overflow".into()))?;
        let coefficients = setup
            .shared_matrix()
            .as_field_slice()
            .get(..len)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup prefix exceeds backend capacity".into())
            })?;
        if len < MIN_FLAT_COEFF_LEN {
            // Preserve zero padding beyond the polynomial domain for small rings.
            return Self::from_field_evals(num_vars, coefficients);
        }
        Ok(Self {
            num_vars,
            coeffs: DenseCoefficients::SetupPrefix { setup, len },
            nonzero_coeff_len: len,
            digit_cache: OnceLock::new(),
        })
    }

    /// Pack field-element evaluations into flat dense storage.
    ///
    /// At dimension `D` the first `α = log₂(D)` variables
    /// become coefficient slots within each ring element; the remaining
    /// variables index ring elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `2^num_vars` does not fit `usize` or
    /// `evals.len() != 2^num_vars`.
    pub fn from_field_evals<'a>(
        num_vars: usize,
        evals: impl Into<Cow<'a, [F]>>,
    ) -> Result<Self, AkitaError>
    where
        F: 'a,
    {
        let evals = evals.into();
        let expected_len = checked::pow2(num_vars)
            .ok_or_else(|| AkitaError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if evals.len() != expected_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: evals.len(),
            });
        }

        let nonzero_coeff_len = nonzero_prefix_len(&evals);
        let physical_len = expected_len.max(MIN_FLAT_COEFF_LEN);

        // Reuse an owned evaluation vector. Borrowed inputs pay the same
        // single copy as before, while profile and application builders can
        // transfer large buffers without doubling their resident memory.
        let mut coeffs = evals.into_owned();
        coeffs.resize(physical_len, F::zero());

        Ok(Self {
            num_vars,
            coeffs: DenseCoefficients::Owned(RingVec::from_coeffs(coeffs)),
            nonzero_coeff_len,
            digit_cache: OnceLock::new(),
        })
    }

    /// Store a prefix of a `2^num_vars` evaluation table, with an implicit zero suffix.
    ///
    /// The logical polynomial and commitment geometry retain the full domain.
    /// Storage is rounded only to the largest supported ring dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if the domain overflows `usize` or the prefix exceeds it.
    pub fn from_field_prefix(num_vars: usize, mut values: Vec<F>) -> Result<Self, AkitaError> {
        let domain_len = checked::pow2(num_vars)
            .ok_or_else(|| AkitaError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if values.len() > domain_len {
            return Err(AkitaError::InvalidSize {
                expected: domain_len,
                actual: values.len(),
            });
        }
        let nonzero_coeff_len = nonzero_prefix_len(&values);
        let physical_len = checked::align_up(nonzero_coeff_len, MIN_FLAT_COEFF_LEN)
            .ok_or_else(|| AkitaError::InvalidInput("dense prefix alignment overflow".into()))?;
        values.truncate(nonzero_coeff_len);
        values.reserve_exact(physical_len - values.len());
        values.resize(physical_len, F::zero());
        values.shrink_to_fit();
        Ok(Self {
            num_vars,
            coeffs: DenseCoefficients::Owned(RingVec::from_coeffs(values)),
            nonzero_coeff_len,
            digit_cache: OnceLock::new(),
        })
    }

    /// Flatten an existing vector of ring elements into dense storage.
    ///
    /// The total coefficient count must be a positive power of two. All
    /// supplied coefficients belong to the live polynomial; storage padding
    /// for larger ring views does not change its variable count.
    ///
    /// # Errors
    ///
    /// Returns an error if `coeffs.len() * D` overflows `usize`, is zero, or
    /// is not a power of two.
    pub fn from_ring_coeffs<const D: usize>(
        coeffs: Vec<CyclotomicRing<F, D>>,
    ) -> Result<Self, AkitaError> {
        let total = checked::product([coeffs.len(), D])
            .ok_or_else(|| AkitaError::InvalidInput("ring elems * D overflow".to_string()))?;
        if !total.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "dense coefficient count {total} must be a positive power of two"
            )));
        }
        let physical_len = total.max(MIN_FLAT_COEFF_LEN);

        let mut flat = Vec::with_capacity(physical_len);
        for ring in &coeffs {
            flat.extend_from_slice(ring.coefficients());
        }
        let nonzero_coeff_len = nonzero_prefix_len(&flat);
        flat.resize(physical_len, F::zero());

        Ok(Self {
            num_vars: total.trailing_zeros() as usize,
            coeffs: DenseCoefficients::Owned(RingVec::from_coeffs(flat)),
            nonzero_coeff_len,
            digit_cache: OnceLock::new(),
        })
    }

    pub(super) fn digit_planes_for<const D: usize>(
        &self,
        num_digits: usize,
        log_basis: u32,
    ) -> Option<(&[[i8; D]], SignedDigitBasis)> {
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
                    (chunks, cache.basis)
                });
        }

        let basis = SignedDigitBasis::new(log_basis).ok()?;
        if basis.kernel() != akita_params::SignedDigitKernel::I8 {
            return None;
        }
        // Preserve the cache admission limit on the logical domain: skipping
        // zeros must not introduce a cache where the full source used none.
        let logical_cache_bytes = self
            .num_ring_elems_at(D)
            .checked_mul(num_digits)?
            .checked_mul(size_of::<[i8; D]>())?;
        if logical_cache_bytes > MAX_DENSE_DIGIT_CACHE_BYTES {
            return None;
        }
        let num_rings = self.nonzero_coeff_len.div_ceil(D);
        let cache_bytes = num_rings
            .checked_mul(num_digits)?
            .checked_mul(size_of::<[i8; D]>())?;

        let _span = tracing::info_span!(
            "dense_digit_cache_build",
            cache_bytes,
            num_rings,
            num_digits,
            ring_dimension = D,
        )
        .entered();
        let rings = self.nonzero_ring_coeffs::<D>().ok()?;
        let params = BalancedDecomposePow2Params::new(num_digits, log_basis);
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
            basis,
            planes,
        });
        let cache = self.digit_cache.get()?;
        (cache.ring_d == D && cache.num_digits == num_digits && cache.log_basis == log_basis).then(
            || {
                let (chunks, remainder) = cache.planes.as_chunks::<D>();
                debug_assert!(remainder.is_empty());
                (chunks, cache.basis)
            },
        )
    }
}
