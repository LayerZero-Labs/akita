//! Field-reduction helpers for extension-field claims embedded in rings.
//!
//! These utilities model the algebraic trace subgroup used by the paper's
//! `F_{q^k}` to `R_q` reduction. They are intentionally standalone so the
//! mathematical contract can be tested independently of the prover API.

use crate::dispatch_for_field;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_serialization::Valid;
use jolt_field::{
    CanonicalField, Ext2, ExtField, FieldCore, FpExt4, FpExt4MulBackend, FpExt8, FpExt8MulBackend,
    FromPrimitiveInt, Invertible,
};
use std::array::from_fn;

/// Extension fields whose `ExtField::to_base_vec` coordinates are the
/// ring-subfield coordinates consumed by [`psi_embed`] and [`embed_subfield`].
pub trait FpExtEncoding<F: FieldCore>: ExtField<F> {
    /// Return coordinates in the ring-subfield basis.
    fn to_ext_coords(&self) -> Vec<F>;

    /// Return the underlying base scalar when this encoding is degree one.
    fn degree_one_base(&self) -> Option<F> {
        None
    }
}

impl<F> FpExtEncoding<F> for F
where
    F: FieldCore + FromPrimitiveInt,
{
    #[inline]
    fn to_ext_coords(&self) -> Vec<F> {
        vec![*self]
    }

    #[inline]
    fn degree_one_base(&self) -> Option<F> {
        Some(*self)
    }
}

impl<F> FpExtEncoding<F> for Ext2<F>
where
    F: FieldCore + FromPrimitiveInt + Valid,
{
    #[inline]
    fn to_ext_coords(&self) -> Vec<F> {
        self.coeffs.to_vec()
    }
}

impl<F> FpExtEncoding<F> for FpExt4<F>
where
    F: FieldCore + FromPrimitiveInt + Valid + FpExt4MulBackend,
{
    #[inline]
    fn to_ext_coords(&self) -> Vec<F> {
        self.coeffs.to_vec()
    }
}

impl<F> FpExtEncoding<F> for FpExt8<F>
where
    F: FieldCore + FromPrimitiveInt + Valid + FpExt8MulBackend,
{
    #[inline]
    fn to_ext_coords(&self) -> Vec<F> {
        self.coeffs.to_vec()
    }
}

/// Validation witness for the subgroup `H = <sigma_-1, sigma_(4K+1)>` of the
/// `R_q = Z_q[X] / (X^D + 1)` Galois action.
///
/// Both the ring dimension `D` and extension degree `K` are compile-time
/// constants, which lets all loop bounds in [`psi_embed`] and [`trace_h`]
/// monomorphize and unroll. The struct is zero-sized and only exists to make
/// "validated `(D, K)`" explicit in function signatures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubfieldParams<const D: usize, const K: usize> {
    _private: (),
}

impl<const D: usize, const K: usize> SubfieldParams<D, K> {
    /// Validate `(D, K)` and return the witness.
    ///
    /// # Errors
    ///
    /// Returns an error when `D == 0`, `D` is not a power of two, `K == 0`,
    /// `K` does not divide `D / 2`, or the generator `4K + 1` is not
    /// invertible modulo `2D`.
    pub fn new() -> Result<Self, AkitaError> {
        let two_d = D
            .checked_mul(2)
            .ok_or_else(|| AkitaError::InvalidInput("ring dimension is too large".to_string()))?;

        if D == 0 {
            return Err(AkitaError::InvalidInput(
                "ring dimension must be non-zero".to_string(),
            ));
        }
        if !D.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "ring dimension D={D} must be a power of two",
            )));
        }
        if !D.is_multiple_of(2) {
            return Err(AkitaError::InvalidInput(format!(
                "ring dimension D={D} must be even",
            )));
        }
        if K == 0 {
            return Err(AkitaError::InvalidInput(
                "extension degree K must be non-zero".to_string(),
            ));
        }
        if K > D / 2 || !(D / 2).is_multiple_of(K) {
            return Err(AkitaError::InvalidInput(format!(
                "extension degree K={K} must divide D/2 for D={D}",
            )));
        }
        let sigma_step = K
            .checked_mul(4)
            .and_then(|step| step.checked_add(1))
            .ok_or_else(|| AkitaError::InvalidInput("extension degree is too large".to_string()))?;
        if gcd(sigma_step, two_d) != 1 {
            return Err(AkitaError::InvalidInput(format!(
                "subgroup generator {sigma_step} must be invertible modulo 2D={two_d}",
            )));
        }

        Ok(Self { _private: () })
    }

    /// Extension degree `K`.
    #[inline]
    pub const fn extension_degree(&self) -> usize {
        K
    }

    /// Automorphism exponents generating `H`, modulo `2D`.
    #[inline]
    pub const fn h_generators(&self) -> (usize, usize) {
        let two_d = D.saturating_mul(2);
        let sigma_step = K.saturating_mul(4).saturating_add(1);
        (two_d.saturating_sub(1), sigma_step)
    }

    /// Enumerate the distinct odd exponents in `H`.
    pub fn h_exponents(&self) -> Vec<usize> {
        let two_d = D.saturating_mul(2);
        let (sigma_m1, sigma_step) = self.h_generators();
        let mut exponents = Vec::with_capacity(D / K);
        let mut power = 1usize;

        for _ in 0..two_d {
            push_unique(&mut exponents, power);
            push_unique(&mut exponents, mul_mod(power, sigma_m1, two_d));

            power = mul_mod(power, sigma_step, two_d);
            if power == 1 {
                exponents.sort_unstable();
                return exponents;
            }
        }

        unreachable!("validated subgroup generator must have finite order modulo 2D")
    }

    /// Number of base-field coordinates in the paper's packed representative.
    #[inline]
    pub const fn packed_len(&self) -> usize {
        D / K
    }
}

/// Compute `Tr_H(x) = sum_{sigma in H} sigma(x)`.
///
/// # Panics
///
/// Panics if the generated subgroup contains an invalid automorphism exponent.
pub fn trace_h<F: FieldCore, const D: usize, const K: usize>(
    params: SubfieldParams<D, K>,
    x: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let mut out = CyclotomicRing::zero();
    for exponent in params.h_exponents() {
        out += x.sigma(exponent);
    }
    out
}

/// Embed a vector of `D / K` ring-subfield elements into one element of `R_q`.
///
/// Each subfield element is given by `K` base-field coordinates in the basis
/// `[1, e_1, ..., e_{K-1}]`, where `e_j = X^(j*D/(2K)) + X^(-j*D/(2K))` viewed
/// inside `R_q = Z_q[X] / (X^D + 1)`. The disambiguating shifts used to place
/// the `D / K` elements are
/// `T = {0, ..., D/(2K) - 1} ∪ {D/2, ..., D/2 + D/(2K) - 1}`.
///
/// `coords` has length `D` with the layout
/// `[s_0[0], s_0[1], ..., s_0[K-1], s_1[0], ..., s_{D/K - 1}[K-1]]`,
/// so `coords[i*K + j]` is the `j`-th basis coordinate of the `i`-th
/// subfield slot.
///
/// For `K = 1` this reduces to placing one base-field value per ring position
/// in the canonical shift order, since the subfield basis collapses to `[1]`
/// and the shift set covers all of `[0, D)`.
///
/// The resulting embedding `psi : (R_q^H)^{D/K} -> R_q` is invertible whenever
/// `2` is a unit in `F` (i.e., for any odd prime characteristic), and is the
/// production-side packing used by the trace inner-product relation
/// `Tr_H(psi(s) * sigma_{-1}(psi(r_in))) = (D/K) * embed_subfield(<s, s>)`,
/// where `r_in` is the subfield coordinate vector and the right-hand side is
/// built with [`embed_subfield`].
///
/// Both `D` and `K` are compile-time constants, so all loop bounds in this
/// function are const-bounded and unroll. The implementation splits into two
/// branchless inner loops. For low shifts `shift in [0, D/(2K))`, both `e_j`
/// terms always land in `[0, D)` without wrapping, so the positive term
/// contributes `+c_j` at `shift + j*step` and the negative term contributes
/// `-c_j` at `shift + D - j*step`. For high shifts
/// `shift in [D/2, D/2 + D/(2K))`, the positive term still does not wrap,
/// while the negative term wraps exactly once around `X^D = -1`, flipping its
/// sign back to `+c_j` at `shift - j*step`.
///
/// # Errors
///
/// Returns an error when `coords.len() != D`.
pub fn psi_embed<F: FieldCore, const D: usize, const K: usize>(
    _params: SubfieldParams<D, K>,
    coords: &[F],
) -> Result<CyclotomicRing<F, D>, AkitaError> {
    if coords.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: coords.len(),
        });
    }

    let step = D / (2 * K);
    let half = D / (2 * K);
    let m = D / K;
    let mut out = [F::zero(); D];

    for idx in 0..half {
        let shift = idx;
        let base = idx * K;
        out[shift] += coords[base];
        for j in 1..K {
            let cj = coords[base + j];
            let pos_offset = j * step;
            out[shift + pos_offset] += cj;
            out[shift + D - pos_offset] -= cj;
        }
    }

    for idx in half..m {
        let shift = idx - half + D / 2;
        let base = idx * K;
        out[shift] += coords[base];
        for j in 1..K {
            let cj = coords[base + j];
            let pos_offset = j * step;
            out[shift + pos_offset] += cj;
            out[shift - pos_offset] += cj;
        }
    }

    Ok(CyclotomicRing::from_coefficients(out))
}

/// Embed a vector of ring-subfield elements into one base-field ring.
///
/// `values.len()` must be `D / K`, where `K = [E : F]` in the ring-subfield
/// basis. This is the typed entry point used by protocol code so that it does
/// not accidentally treat arbitrary extension coordinates as cyclotomic
/// subfield coordinates.
///
/// # Errors
///
/// Returns an error if `K` is unsupported by the current dispatcher, if
/// `SubfieldParams<D, K>` rejects the pair, or if the vector length is not
/// exactly `D / K`.
pub fn embed_ring_subfield_vector<F, E, const D: usize>(
    values: &[E],
    error: AkitaError,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F>,
{
    macro_rules! arm {
        ($k:expr) => {{
            let params = SubfieldParams::<D, $k>::new().map_err(|_| error.clone())?;
            let expected = params.packed_len();
            if values.len() != expected {
                return Err(error);
            }
            let mut coords = Vec::with_capacity(D);
            for value in values {
                let limbs = value.to_ext_coords();
                if limbs.len() != $k {
                    return Err(error);
                }
                coords.extend(limbs);
            }
            psi_embed::<F, D, $k>(params, &coords).map_err(|_| error)
        }};
    }

    match E::EXT_DEGREE {
        1 => arm!(1),
        2 => arm!(2),
        4 => arm!(4),
        8 => arm!(8),
        _ => Err(error),
    }
}

/// Embed one ring-subfield scalar into the base-field ring.
///
/// This is the scalar counterpart of [`embed_ring_subfield_vector`]: the
/// scalar's ring-subfield basis coordinates are placed in one subfield slot and
/// embedded through the same `psi` boundary as packed vectors.
///
/// # Errors
///
/// Returns an error if the extension degree is unsupported or the scalar does
/// not expose exactly `K = [E:F]` ring-subfield coordinates.
pub fn embed_ring_subfield_scalar<F, E, const D: usize>(
    value: E,
    error: AkitaError,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F>,
{
    macro_rules! arm {
        ($k:expr) => {{
            let params = SubfieldParams::<D, $k>::new().map_err(|_| error.clone())?;
            let limbs = value.to_ext_coords();
            let coords: [F; $k] = limbs.try_into().map_err(|_| error.clone())?;
            Ok(embed_subfield::<F, D, $k>(params, &coords))
        }};
    }

    match E::EXT_DEGREE {
        1 => arm!(1),
        2 => arm!(2),
        4 => arm!(4),
        8 => arm!(8),
        _ => Err(error),
    }
}

/// Runtime-dimension form of [`embed_ring_subfield_scalar`]: returns the
/// embedded element as `ring_d` flat coefficients.
///
/// # Errors
///
/// Returns an error if `ring_d` is unsupported, the extension degree is
/// unsupported, or the scalar does not expose exactly `K = [E:F]`
/// ring-subfield coordinates.
pub fn embed_ring_subfield_scalar_flat<F, E>(
    ring_d: usize,
    value: E,
    error: AkitaError,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt + CanonicalField,
    E: FpExtEncoding<F>,
{
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Outer),
        F,
        ring_d,
        |D| {
            embed_ring_subfield_scalar::<F, E, D>(value, error.clone())
                .map(|ring| ring.coefficients().to_vec())
        }
    )
}

/// Pack a base-field digit evaluation table into the canonical tensor
/// extension ring-subfield representation.
///
/// The logical table is ordered with the packed head variables fastest:
/// `digits[tail * width + head]`. For each tail slot this constructs the
/// extension value whose first `width` canonical basis coordinates are those
/// head digits, then applies the same `psi` layout as
/// [`embed_ring_subfield_vector`]. For `width = K = [L:F]`, the physical
/// output length matches the logical input length.
///
/// # Errors
///
/// Returns an error if the width is invalid for the extension degree, if the
/// input is not whole head-slices, or if a packed coefficient would overflow
/// `i8`.
pub fn pack_tensor_base_lift_i8_digits<const D: usize>(
    digits: &[i8],
    extension_degree: usize,
    width: usize,
) -> Result<Vec<i8>, AkitaError> {
    if width == 0 || width > extension_degree {
        return Err(AkitaError::InvalidInput(
            "tensor pack width must be in 1..=extension_degree".to_string(),
        ));
    }
    if extension_degree == 1 {
        if width != 1 {
            return Err(AkitaError::InvalidInput(
                "degree-one tensor pack must have width 1".to_string(),
            ));
        }
        return Ok(digits.to_vec());
    }
    if !digits.len().is_multiple_of(width) {
        return Err(AkitaError::InvalidSize {
            expected: width,
            actual: digits.len(),
        });
    }

    macro_rules! arm {
        ($k:expr) => {{
            let _params = SubfieldParams::<D, $k>::new()?;
            let packed_len = D / $k;
            let half = D / (2 * $k);
            let step = D / (2 * $k);
            let tail_len = digits.len() / width;
            let mut out = Vec::with_capacity(tail_len.div_ceil(packed_len) * D);

            for tail_start in (0..tail_len).step_by(packed_len) {
                let mut packed = [0i16; D];
                for idx in 0..packed_len {
                    let tail = tail_start + idx;
                    if tail >= tail_len {
                        break;
                    }
                    if idx < half {
                        let shift = idx;
                        let coord0 = digits[tail * width] as i16;
                        packed[shift] = packed[shift].checked_add(coord0).ok_or_else(|| {
                            AkitaError::InvalidInput("packed tensor digit overflow".to_string())
                        })?;
                        for j in 1..width {
                            let cj = digits[tail * width + j] as i16;
                            let pos_offset = j * step;
                            packed[shift + pos_offset] =
                                packed[shift + pos_offset].checked_add(cj).ok_or_else(|| {
                                    AkitaError::InvalidInput(
                                        "packed tensor digit overflow".to_string(),
                                    )
                                })?;
                            packed[shift + D - pos_offset] = packed[shift + D - pos_offset]
                                .checked_sub(cj)
                                .ok_or_else(|| {
                                    AkitaError::InvalidInput(
                                        "packed tensor digit overflow".to_string(),
                                    )
                                })?;
                        }
                    } else {
                        let shift = idx - half + D / 2;
                        let coord0 = digits[tail * width] as i16;
                        packed[shift] = packed[shift].checked_add(coord0).ok_or_else(|| {
                            AkitaError::InvalidInput("packed tensor digit overflow".to_string())
                        })?;
                        for j in 1..width {
                            let cj = digits[tail * width + j] as i16;
                            let pos_offset = j * step;
                            packed[shift + pos_offset] =
                                packed[shift + pos_offset].checked_add(cj).ok_or_else(|| {
                                    AkitaError::InvalidInput(
                                        "packed tensor digit overflow".to_string(),
                                    )
                                })?;
                            packed[shift - pos_offset] =
                                packed[shift - pos_offset].checked_add(cj).ok_or_else(|| {
                                    AkitaError::InvalidInput(
                                        "packed tensor digit overflow".to_string(),
                                    )
                                })?;
                        }
                    }
                }
                for coeff in packed {
                    let coeff = i8::try_from(coeff).map_err(|_| {
                        AkitaError::InvalidInput("packed tensor digit overflow".to_string())
                    })?;
                    out.push(coeff);
                }
            }
            Ok(out)
        }};
    }

    match extension_degree {
        2 => arm!(2),
        4 => arm!(4),
        8 => arm!(8),
        _ => Err(AkitaError::InvalidInput(format!(
            "unsupported ring-subfield extension degree {extension_degree}"
        ))),
    }
}

/// Validate that an extension field can be represented by the ring-subfield
/// dispatcher for ring dimension `D`.
///
/// This is the early scheme/config boundary check for field roles that will
/// later be consumed by [`embed_ring_subfield_vector`] and
/// [`dispatch_trace_inner_product_check`]. Today the supported monomorphized
/// extension degrees are `1, 2, 4, 8`; each accepted degree must also satisfy
/// [`SubfieldParams::new`] for the active ring dimension.
///
/// # Errors
///
/// Returns an invalid-setup error if the extension degree is unsupported or if
/// the ring-subfield subgroup parameters reject `(D, [E:F])`.
pub fn validate_ring_subfield_role<F, E, const D: usize>(
    role: &'static str,
) -> Result<(), AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F>,
{
    let error = || {
        AkitaError::InvalidSetup(format!(
            "{role} extension degree {} is not supported for ring dimension {D}",
            E::EXT_DEGREE
        ))
    };
    macro_rules! arm {
        ($k:expr) => {{
            SubfieldParams::<D, $k>::new().map_err(|_| error())?;
            Ok(())
        }};
    }

    match E::EXT_DEGREE {
        1 => arm!(1),
        2 => arm!(2),
        4 => arm!(4),
        8 => arm!(8),
        _ => Err(error()),
    }
}

/// Recover one ring-subfield inner product from a ring-level folded output.
///
/// This is the value-level counterpart of [`check_trace_inner_product`]. It is
/// used when a verifier needs the extension-field opening value itself, rather
/// than only checking it against a supplied scalar.
///
/// # Errors
///
/// Returns an error if the extension degree is unsupported for `D`, the trace
/// scale is not invertible, or the encoded coordinate count is malformed.
pub fn recover_ring_subfield_inner_product<F, E, const D: usize>(
    y_ring: &CyclotomicRing<F, D>,
    packed_inner_point: &CyclotomicRing<F, D>,
) -> Result<E, AkitaError>
where
    F: FieldCore + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F>,
{
    let trace_input = *y_ring * packed_inner_point.sigma_m1();
    macro_rules! arm {
        ($k:expr) => {{
            let params = SubfieldParams::<D, $k>::new().map_err(|_| {
                AkitaError::InvalidInput(
                    "claim-field degree must divide the ring dimension".to_string(),
                )
            })?;
            let traced = trace_h::<F, D, $k>(params, &trace_input);
            decode_traced_to_extension::<F, E, D, $k>(params, &traced)
        }};
    }

    match E::EXT_DEGREE {
        1 => arm!(1),
        2 => arm!(2),
        4 => arm!(4),
        8 => arm!(8),
        _ => Err(AkitaError::InvalidInput(
            "unsupported ring-subfield extension degree".to_string(),
        )),
    }
}

#[inline]
fn decode_traced_to_extension<F, E, const D: usize, const K: usize>(
    params: SubfieldParams<D, K>,
    traced: &CyclotomicRing<F, D>,
) -> Result<E, AkitaError>
where
    F: FieldCore + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F>,
{
    let scale_inv = F::from_u64(params.packed_len() as u64)
        .inverse()
        .ok_or_else(|| AkitaError::InvalidInput("trace scale is not invertible".to_string()))?;
    let coeffs = traced.coefficients();
    if K == 1 {
        return Ok(E::from_base_slice(&[coeffs[0] * scale_inv]));
    }
    let step = D / (2 * K);
    let mut coords = Vec::with_capacity(K);
    coords.push(coeffs[0] * scale_inv);
    let mut j = 1usize;
    while j < K {
        coords.push(coeffs[j * step] * scale_inv);
        j += 1;
    }
    Ok(E::from_base_slice(&coords))
}

/// `TraceOpen(ring · X^c)` for every ring coordinate `c`, sharing one ring product.
pub(crate) fn trace_open_ring_row<F, E, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    packed_inner_point: &CyclotomicRing<F, D>,
    ring_bits: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F>,
{
    let ring_len = 1usize
        .checked_shl(ring_bits as u32)
        .ok_or_else(|| AkitaError::InvalidInput("trace-open row length overflow".to_string()))?;
    let trace_partner = packed_inner_point.sigma_m1();
    let trace_product = *ring * trace_partner;
    macro_rules! arm {
        ($k:expr) => {{
            let params = SubfieldParams::<D, $k>::new().map_err(|_| {
                AkitaError::InvalidInput(
                    "claim-field degree must divide the ring dimension".to_string(),
                )
            })?;
            if $k == 1 {
                let mut row = Vec::with_capacity(ring_len);
                for coord in 0..ring_len {
                    let trace_input = trace_product.negacyclic_shift(coord);
                    row.push(E::from_base_slice(&[trace_input.coefficients()[0]]));
                }
                return Ok(row);
            }
            let mut row = Vec::with_capacity(ring_len);
            for coord in 0..ring_len {
                let trace_input = trace_product.negacyclic_shift(coord);
                let traced = trace_h::<F, D, $k>(params, &trace_input);
                row.push(decode_traced_to_extension::<F, E, D, $k>(params, &traced)?);
            }
            Ok(row)
        }};
    }

    match E::EXT_DEGREE {
        1 => arm!(1),
        2 => arm!(2),
        4 => arm!(4),
        8 => arm!(8),
        _ => Err(AkitaError::InvalidInput(
            "unsupported ring-subfield extension degree".to_string(),
        )),
    }
}

fn lift_ring_to_extension<F, E, const D: usize>(ring: &CyclotomicRing<F, D>) -> CyclotomicRing<E, D>
where
    F: FieldCore,
    E: ExtField<F>,
{
    CyclotomicRing::from_coefficients(from_fn(|idx| E::lift_base(ring.coefficients()[idx])))
}

fn weighted_negacyclic_shift_sum<E, const D: usize>(
    ring: &CyclotomicRing<E, D>,
    eq_coords: &[E],
) -> CyclotomicRing<E, D>
where
    E: FieldCore,
{
    let mut out = CyclotomicRing::<E, D>::zero();
    for (coord, weight) in eq_coords.iter().copied().enumerate() {
        if weight.is_zero() {
            continue;
        }
        ring.shift_scale_accumulate_into(&mut out, coord, weight);
    }
    out
}

fn decode_extension_linear_trace<F, E, const D: usize, const K: usize>(
    params: SubfieldParams<D, K>,
    trace_input: &CyclotomicRing<E, D>,
) -> Result<E, AkitaError>
where
    F: FieldCore + FromPrimitiveInt + Invertible,
    E: ExtField<F>,
{
    if K == 1 {
        return Ok(trace_input.coefficients()[0]);
    }

    let traced = trace_h::<E, D, K>(params, trace_input);
    let scale_inv = F::from_u64(params.packed_len() as u64)
        .inverse()
        .ok_or_else(|| AkitaError::InvalidInput("trace scale is not invertible".to_string()))?;
    let scale_inv = E::lift_base(scale_inv);
    let coeffs = traced.coefficients();
    let step = D / (2 * K);
    let mut out = coeffs[0] * scale_inv;

    let mut j = 1usize;
    while j < K {
        let mut basis_coords = [F::zero(); K];
        basis_coords[j] = F::one();
        let basis = E::from_base_slice(&basis_coords);
        out += coeffs[j * step] * scale_inv * basis;
        j += 1;
    }

    Ok(out)
}

/// `Σ_c eq_coords[c] · TraceOpen(folded · X^c)` for a pre-folded extension ring.
///
/// `folded` is the block-weighted, lifted fold-block ring for one trace term:
/// `Σ_block col_factor(block) · lift(block_ring)`. Because the whole trace-open
/// pipeline (shift sum, ring product, `Tr_H`, decode) is `E`-linear in the ring
/// argument, summing the per-block trace opens equals one trace open of the
/// folded ring. The caller therefore pays a single `Tr_H` of one ring product
/// per term instead of one per fold block.
pub(crate) fn trace_open_folded_ring_mle_dot<F, E, const D: usize>(
    folded: &CyclotomicRing<E, D>,
    eq_coords: &[E],
    packed_inner_point: &CyclotomicRing<F, D>,
    ring_bits: usize,
) -> Result<E, AkitaError>
where
    F: FieldCore + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F>,
{
    let ring_bits = u32::try_from(ring_bits).map_err(|_| {
        AkitaError::InvalidInput("trace-open ring bits exceed platform width".to_string())
    })?;
    let ring_len = 1usize
        .checked_shl(ring_bits)
        .ok_or_else(|| AkitaError::InvalidInput("trace-open eq length overflow".to_string()))?;
    if ring_len != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: ring_len,
        });
    }
    if eq_coords.len() != ring_len {
        return Err(AkitaError::InvalidSize {
            expected: ring_len,
            actual: eq_coords.len(),
        });
    }
    macro_rules! arm {
        ($k:expr) => {{
            let params = SubfieldParams::<D, $k>::new().map_err(|_| {
                AkitaError::InvalidInput(
                    "claim-field degree must divide the ring dimension".to_string(),
                )
            })?;
            let shifted = weighted_negacyclic_shift_sum::<E, D>(folded, eq_coords);
            let trace_partner = lift_ring_to_extension::<F, E, D>(&packed_inner_point.sigma_m1());
            let trace_input = shifted * trace_partner;
            decode_extension_linear_trace::<F, E, D, $k>(params, &trace_input)
        }};
    }

    match E::EXT_DEGREE {
        1 => arm!(1),
        2 => arm!(2),
        4 => arm!(4),
        8 => arm!(8),
        _ => Err(AkitaError::InvalidInput(
            "unsupported ring-subfield extension degree".to_string(),
        )),
    }
}

/// Verifier-side check of the ring-subfield trace inner-product identity:
/// `trace_h(trace_input) == (D / K) * embed_subfield(opening_coords)` in `R_q`.
///
/// This is the K-generic, ring-element form of the K = 1 scalar shortcut
/// `trace_h(trace_input).coefficients()[0] == D * opening`. Caller supplies the
/// claim opening as `K` base-field coordinates in the canonical basis; for
/// `K = 1` this is just the single scalar opening.
///
/// Internally dispatches on the const-generic `K`: at `K = 1` the identity
/// collapses to a single scalar comparison `D * trace_input[0] == D * opening`
/// (the existing fast path, `O(1)` work), since `trace_h` over the full Galois
/// group always produces a constant ring element. At `K > 1` it materializes
/// both sides of the ring identity. Both branches are mathematically equivalent
/// at `K = 1`; the dispatch only avoids the `O(D^2 / K)` ring trace work that
/// would otherwise be redundant on the degree-one path.
///
/// Returns `true` iff the identity holds. The `params` argument is the
/// validation witness for `(D, K)`; constructing it elsewhere is the only
/// way to enter this function with a sound `(D, K)` pair.
pub fn check_trace_inner_product<F, const D: usize, const K: usize>(
    params: SubfieldParams<D, K>,
    trace_input: &CyclotomicRing<F, D>,
    opening_coords: &[F; K],
) -> bool
where
    F: FieldCore + FromPrimitiveInt,
{
    if K == 1 {
        // trace_h for K = 1 is a constant ring element with
        // coefficient[0] = D * trace_input.coefficients()[0], compared against
        // `(D / 1) * embed_subfield(&[opening]) = D * opening_coords[0]`. The
        // common factor `D` cancels, so we compare the scalars directly. This
        // also avoids a latent footgun: multiplying both sides by `D` would
        // collapse to `0 == 0` (accepting any opening) for any modulus dividing
        // `D` — not the case for the production primes, but free to rule out.
        return trace_input.coefficients()[0] == opening_coords[0];
    }

    let lhs = trace_h(params, trace_input);
    let scale = F::from_u64(params.packed_len() as u64);
    let rhs = embed_subfield(params, opening_coords).scale(&scale);
    lhs == rhs
}

/// Dispatch a trace inner-product check at runtime on the coordinate count.
///
/// Used at the verifier-extension boundary: `opening_coords` is produced by
/// `ExtField::to_base_vec` on a typed extension-field value, so its length is
/// the runtime extension degree `K`. This helper picks the matching
/// monomorphization of [`check_trace_inner_product`] for `K ∈ {1, 2, 4, 8}`,
/// which are the extension degrees the workspace currently exercises. Higher
/// `K` would need a new arm, so unsupported values are rejected explicitly.
///
/// # Errors
///
/// Returns `error` when `opening_coords.len()` is not a supported extension
/// degree, or when [`SubfieldParams::new`] rejects the resulting `(D, K)`.
pub fn dispatch_trace_inner_product_check<F, const D: usize>(
    trace_input: &CyclotomicRing<F, D>,
    opening_coords: &[F],
    error: AkitaError,
) -> Result<bool, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
{
    macro_rules! arm {
        ($k:expr) => {{
            let coords: &[F; $k] = opening_coords.try_into().map_err(|_| error.clone())?;
            let params = SubfieldParams::<D, $k>::new().map_err(|_| error.clone())?;
            Ok(check_trace_inner_product::<F, D, $k>(
                params,
                trace_input,
                coords,
            ))
        }};
    }
    match opening_coords.len() {
        1 => arm!(1),
        2 => arm!(2),
        4 => arm!(4),
        8 => arm!(8),
        _ => Err(error),
    }
}

/// Embed a single subfield element into `R_q` at shift `X^0`.
///
/// Given `coords = [c_0, ..., c_{K-1}]` interpreted in the basis
/// `[1, e_1, ..., e_{K-1}]` with `e_j = X^(j*D/(2K)) + X^(-j*D/(2K))`, this
/// returns `c_0 + sum_{j=1}^{K-1} c_j * e_j` in `R_q`.
///
/// Mathematically equal to the slot-0 specialization of [`psi_embed`], i.e.
/// `psi_embed(constants_only(coords))` where every other slot is zero. It is
/// kept as a separate entry point because verifier-side checks only need to
/// embed a single claimed inner product into the ring, and writing the
/// `2K - 1` nonzero coefficients directly avoids the `O(D)` loop and the
/// zero-padding the full vector form pays.
///
/// `K` is a const generic so the caller's `coords` can be a fixed-size array
/// and the `2K - 1` writes unroll completely.
pub fn embed_subfield<F: FieldCore, const D: usize, const K: usize>(
    _params: SubfieldParams<D, K>,
    coords: &[F; K],
) -> CyclotomicRing<F, D> {
    let step = D / (2 * K);
    let mut out = [F::zero(); D];
    out[0] = coords[0];
    for j in 1..K {
        let cj = coords[j];
        out[j * step] = cj;
        out[D - j * step] = -cj;
    }
    CyclotomicRing::from_coefficients(out)
}

fn push_unique(values: &mut Vec<usize>, value: usize) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn mul_mod(a: usize, b: usize, modulus: usize) -> usize {
    ((a as u128 * b as u128) % modulus as u128) as usize
}

#[cfg(test)]
mod tests;
