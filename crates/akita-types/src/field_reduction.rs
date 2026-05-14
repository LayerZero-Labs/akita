//! Field-reduction helpers for extension-field claims embedded in rings.
//!
//! These utilities model the algebraic trace subgroup used by the paper's
//! `F_{q^k}` to `R_q` reduction. They are intentionally standalone so the
//! mathematical contract can be tested independently of the prover API.

use akita_algebra::CyclotomicRing;
use akita_field::{
    AkitaError, Ext2, ExtField, FieldCore, FromPrimitiveInt, Invertible, RingSubfieldFp4,
    RingSubfieldFp4MulBackend,
};
use akita_serialization::Valid;

/// Extension fields whose `ExtField::to_base_vec` coordinates are the
/// ring-subfield coordinates consumed by [`psi_embed`] and [`embed_subfield`].
///
/// This intentionally does not blanket-implement every extension field over
/// `F`: tower/power coordinates are not automatically the same basis as the
/// cyclotomic subfield basis used by the trace reduction.
pub trait RingSubfieldEncoding<F: FieldCore>: ExtField<F> {
    /// Return coordinates in the ring-subfield basis.
    fn to_ring_subfield_coords(&self) -> Vec<F>;
}

impl<F> RingSubfieldEncoding<F> for F
where
    F: FieldCore + FromPrimitiveInt,
{
    #[inline]
    fn to_ring_subfield_coords(&self) -> Vec<F> {
        vec![*self]
    }
}

impl<F> RingSubfieldEncoding<F> for Ext2<F>
where
    F: FieldCore + FromPrimitiveInt + Valid,
{
    #[inline]
    fn to_ring_subfield_coords(&self) -> Vec<F> {
        self.coeffs.to_vec()
    }
}

impl<F> RingSubfieldEncoding<F> for RingSubfieldFp4<F>
where
    F: FieldCore + FromPrimitiveInt + Valid + RingSubfieldFp4MulBackend,
{
    #[inline]
    fn to_ring_subfield_coords(&self) -> Vec<F> {
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
pub struct SubfieldParams<const D: usize, const K: usize>;

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
        if D % 2 != 0 {
            return Err(AkitaError::InvalidInput(format!(
                "ring dimension D={D} must be even",
            )));
        }
        if K == 0 {
            return Err(AkitaError::InvalidInput(
                "extension degree K must be non-zero".to_string(),
            ));
        }
        if K > D / 2 || (D / 2) % K != 0 {
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

        Ok(Self)
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
/// `Tr_H(psi(s) * sigma_{-1}(psi(v))) = (D/K) * embed_subfield(<s, v>)`,
/// where the right-hand side is built with [`embed_subfield`].
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
    E: RingSubfieldEncoding<F>,
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
                let limbs = value.to_ring_subfield_coords();
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
    E: RingSubfieldEncoding<F>,
{
    macro_rules! arm {
        ($k:expr) => {{
            let params = SubfieldParams::<D, $k>::new().map_err(|_| error.clone())?;
            let limbs = value.to_ring_subfield_coords();
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
    E: RingSubfieldEncoding<F>,
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
    inner_reduction: &CyclotomicRing<F, D>,
) -> Result<E, AkitaError>
where
    F: FieldCore + FromPrimitiveInt + Invertible,
    E: RingSubfieldEncoding<F>,
{
    let trace_input = *y_ring * inner_reduction.sigma_m1();
    macro_rules! arm {
        ($k:expr) => {{
            let params = SubfieldParams::<D, $k>::new().map_err(|_| {
                AkitaError::InvalidInput(
                    "claim-field degree must divide the ring dimension".to_string(),
                )
            })?;
            let traced = trace_h::<F, D, $k>(params, &trace_input);
            let scale_inv = F::from_u64(params.packed_len() as u64)
                .inverse()
                .ok_or_else(|| {
                    AkitaError::InvalidInput("trace scale is not invertible".to_string())
                })?;
            let coeffs = traced.coefficients();
            let step = D / (2 * $k);
            let mut coords = Vec::with_capacity($k);
            coords.push(coeffs[0] * scale_inv);
            let mut j = 1usize;
            while j < $k {
                coords.push(coeffs[j * step] * scale_inv);
                j += 1;
            }
            Ok(E::from_base_slice(&coords))
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
        // trace_h for K = 1 is always a constant ring element with
        // coefficient[0] = D * trace_input.coefficients()[0]; comparing it
        // against `(D / 1) * embed_subfield(&[opening])` reduces to a single
        // scalar equality.
        let d_field = F::from_u64(D as u64);
        return d_field * trace_input.coefficients()[0] == d_field * opening_coords[0];
    }

    let lhs = trace_h(params, trace_input);
    let scale = F::from_u64(params.packed_len() as u64);
    let rhs = embed_subfield(params, opening_coords).scale(&scale);
    lhs == rhs
}

/// Dispatch a trace inner-product check at runtime on the coordinate count.
///
/// Used at the verifier-extension boundary: `opening_coords` is produced by
/// `ExtField::to_base_vec` on a typed `ClaimField` value, so its length is
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
            let coords: &[F; $k] = opening_coords.try_into().expect("checked length");
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
mod tests {
    use super::*;
    use crate::{reduce_inner_opening_to_ring_element, BasisMode};
    use akita_field::{ExtField, Fp32, RingSubfieldFp4, TowerBasisFp4, TwoNr, UnitNr};

    type F = Fp32<251>;
    type AkitaF32 = Fp32<4294967197>;

    fn ring_from_i64s<const D: usize>(values: [i64; D]) -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(values.map(F::from_i64))
    }

    fn ring_from_index<const D: usize>() -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|i| F::from_u64((i + 1) as u64)))
    }

    fn ring_subfield_basis<Fq: FieldCore, const D: usize, const K: usize>(
        _params: SubfieldParams<D, K>,
    ) -> Vec<CyclotomicRing<Fq, D>> {
        let step = D / (2 * K);
        let mut basis = Vec::with_capacity(K);
        basis.push(CyclotomicRing::one());
        for i in 1..K {
            let pos = i * step;
            let mut coeffs = [Fq::zero(); D];
            coeffs[pos] = Fq::one();
            coeffs[D - pos] = -Fq::one();
            basis.push(CyclotomicRing::from_coefficients(coeffs));
        }
        basis
    }

    fn ring_subfield_coords<Fq: FieldCore, const D: usize, const K: usize>(
        _params: SubfieldParams<D, K>,
        x: &CyclotomicRing<Fq, D>,
    ) -> Vec<Fq> {
        let step = D / (2 * K);
        let coeffs = x.coefficients();
        let mut coords = vec![Fq::zero(); K];
        coords[0] = coeffs[0];

        for (i, coord) in coords.iter_mut().enumerate().take(K).skip(1) {
            let pos = i * step;
            *coord = coeffs[pos];
            assert_eq!(
                coeffs[D - pos],
                -*coord,
                "subfield coordinate {i} has wrong inverse coefficient"
            );
        }

        for (idx, coeff) in coeffs.iter().enumerate() {
            let is_basis_slot = idx == 0
                || (1..K).any(|i| {
                    let pos = i * step;
                    idx == pos || idx == D - pos
                });
            if !is_basis_slot {
                assert!(
                    coeff.is_zero(),
                    "unexpected nonzero coefficient at ring exponent {idx}"
                );
            }
        }

        coords
    }

    fn embed_tower_in_ring_subfield<const D: usize>(
        x: TowerBasisFp4<AkitaF32, TwoNr, UnitNr>,
    ) -> CyclotomicRing<AkitaF32, D> {
        let params = SubfieldParams::<D, 4>::new().unwrap();
        let basis = ring_subfield_basis::<AkitaF32, D, 4>(params);

        // Over 2^32 - 99, i is a square root of -1 and a satisfies
        // a^2 = 1 / (2 * (1 + i)). Thus v = a*e1 + a*i*e3 has v^2 = e2.
        let a = AkitaF32::from_u64(1_492_342_050);
        let ai = a * AkitaF32::from_u64(3_311_696_422);
        let v = basis[1].scale(&a) + basis[3].scale(&ai);
        let u = basis[2];
        let vu = v * u;
        let power_basis = [basis[0], v, u, vu];
        let coeffs = x.to_base_vec();

        coeffs
            .into_iter()
            .zip(power_basis)
            .fold(CyclotomicRing::zero(), |acc, (coeff, basis_elem)| {
                acc + basis_elem.scale(&coeff)
            })
    }

    #[test]
    fn subfield_params_validate_extension_degree() {
        assert!(SubfieldParams::<8, 1>::new().is_ok());
        assert!(SubfieldParams::<8, 4>::new().is_ok());

        assert!(matches!(
            SubfieldParams::<8, 0>::new(),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            SubfieldParams::<8, 3>::new(),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            SubfieldParams::<9, 1>::new(),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            SubfieldParams::<6, 1>::new(),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            SubfieldParams::<10, 1>::new(),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            SubfieldParams::<{ usize::MAX - 1 }, 1>::new(),
            Err(AkitaError::InvalidInput(_))
        ));
    }

    #[test]
    fn h_exponents_match_power_of_two_subgroups() {
        assert_eq!(
            SubfieldParams::<8, 1>::new().unwrap().h_exponents().len(),
            8
        );
        assert_eq!(
            SubfieldParams::<8, 2>::new().unwrap().h_exponents().len(),
            4
        );
        assert_eq!(
            SubfieldParams::<8, 4>::new().unwrap().h_exponents().len(),
            2
        );

        assert_eq!(
            SubfieldParams::<8, 2>::new().unwrap().h_exponents(),
            vec![1, 7, 9, 15]
        );
    }

    #[test]
    fn h_exponents_cover_production_ring_subgroups() {
        assert_eq!(
            SubfieldParams::<64, 1>::new().unwrap().h_exponents().len(),
            64
        );
        assert_eq!(
            SubfieldParams::<64, 8>::new().unwrap().h_exponents().len(),
            8
        );
        assert_eq!(
            SubfieldParams::<128, 1>::new().unwrap().h_exponents().len(),
            128
        );
        assert_eq!(
            SubfieldParams::<128, 16>::new()
                .unwrap()
                .h_exponents()
                .len(),
            8
        );
    }

    #[test]
    fn trace_h_k_one_matches_constant_coefficient_trace() {
        const D: usize = 8;
        let params = SubfieldParams::<D, 1>::new().unwrap();
        let x = ring_from_i64s([3, 5, 7, 11, 13, 17, 19, 23]);
        let trace = trace_h(params, &x);
        let coeffs = trace.coefficients();

        assert_eq!(coeffs[0], F::from_u64(D as u64) * x.coefficients()[0]);
        assert!(coeffs[1..].iter().all(|coeff| coeff.is_zero()));
    }

    #[test]
    fn trace_h_k_one_matches_constant_coefficient_trace_at_production_sizes() {
        let params_64 = SubfieldParams::<64, 1>::new().unwrap();
        let x_64 = ring_from_index::<64>();
        let trace_64 = trace_h(params_64, &x_64);
        assert_eq!(
            trace_64.coefficients()[0],
            F::from_u64(64) * x_64.coefficients()[0]
        );
        assert!(trace_64.coefficients()[1..]
            .iter()
            .all(|coeff| coeff.is_zero()));

        let params_128 = SubfieldParams::<128, 1>::new().unwrap();
        let x_128 = ring_from_index::<128>();
        let trace_128 = trace_h(params_128, &x_128);
        assert_eq!(
            trace_128.coefficients()[0],
            F::from_u64(128) * x_128.coefficients()[0]
        );
        assert!(trace_128.coefficients()[1..]
            .iter()
            .all(|coeff| coeff.is_zero()));
    }

    #[test]
    fn trace_h_k_one_matches_inner_opening_reduction_shortcut() {
        const D: usize = 8;
        let params = SubfieldParams::<D, 1>::new().unwrap();
        let y_ring = ring_from_i64s([2, 3, 5, 7, 11, 13, 17, 19]);
        let inner_point = [F::from_u64(3), F::from_u64(5), F::from_u64(7)];

        for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
            let v = reduce_inner_opening_to_ring_element::<F, D>(&inner_point, basis).unwrap();
            let product = y_ring * v.sigma_m1();
            let trace = trace_h(params, &product);
            let coeffs = trace.coefficients();
            let current_shortcut = F::from_u64(D as u64) * product.coefficients()[0];

            assert_eq!(coeffs[0], current_shortcut);
            assert!(coeffs[1..].iter().all(|coeff| coeff.is_zero()));
        }
    }

    #[test]
    fn trace_h_matches_direct_generator_sum() {
        const D: usize = 8;
        let params = SubfieldParams::<D, 2>::new().unwrap();
        let x = ring_from_i64s([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut expected = CyclotomicRing::zero();

        for exponent in [1, 7, 9, 15] {
            expected += x.sigma(exponent);
        }

        assert_eq!(trace_h(params, &x), expected);
    }

    /// Build a flat `psi_embed` input from `D / K` subfield elements where
    /// only the constant (`e_0 = 1`) coordinate is set.
    fn constants_only_coords<const D: usize, const K: usize>(
        params: SubfieldParams<D, K>,
        values: &[F],
    ) -> Vec<F> {
        assert_eq!(values.len(), params.packed_len());
        let mut coords = vec![F::zero(); D];
        for (i, value) in values.iter().enumerate() {
            coords[i * K] = *value;
        }
        coords
    }

    #[test]
    fn psi_embed_constants_only_matches_paper_positions() {
        const D: usize = 8;
        let params = SubfieldParams::<D, 2>::new().unwrap();
        let coords = constants_only_coords(
            params,
            &[
                F::from_u64(1),
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(4),
            ],
        );
        let packed = psi_embed::<F, D, 2>(params, &coords).unwrap();
        let expected = ring_from_i64s([1, 2, 0, 0, 3, 4, 0, 0]);

        assert_eq!(packed, expected);
    }

    #[test]
    fn psi_embed_constants_only_at_production_ring_size() {
        const D: usize = 64;
        let params = SubfieldParams::<D, 8>::new().unwrap();
        let values: Vec<F> = (0..params.packed_len())
            .map(|i| F::from_u64((i + 1) as u64))
            .collect();
        let coords = constants_only_coords(params, &values);
        let packed = psi_embed::<F, D, 8>(params, &coords).unwrap();
        let coeffs = packed.coefficients();
        let half = params.packed_len() / 2;

        assert_eq!(&coeffs[..half], &values[..half]);
        assert!(coeffs[half..D / 2].iter().all(|coeff| coeff.is_zero()));
        assert_eq!(&coeffs[D / 2..D / 2 + half], &values[half..]);
        assert!(coeffs[D / 2 + half..].iter().all(|coeff| coeff.is_zero()));
    }

    #[test]
    fn psi_embed_k_one_is_identity_placement() {
        const D: usize = 8;
        let params = SubfieldParams::<D, 1>::new().unwrap();
        let coords: Vec<F> = (0..D).map(|i| F::from_u64((i + 1) as u64)).collect();
        let packed = psi_embed::<F, D, 1>(params, &coords).unwrap();
        let expected = ring_from_i64s([1, 2, 3, 4, 5, 6, 7, 8]);

        assert_eq!(packed, expected);
    }

    #[test]
    fn psi_embed_rejects_wrong_length() {
        let params = SubfieldParams::<8, 2>::new().unwrap();
        assert!(matches!(
            psi_embed::<F, 8, 2>(params, &[F::one()]),
            Err(AkitaError::InvalidSize {
                expected: 8,
                actual: 1
            })
        ));
    }

    /// `psi_embed` of "first slot only, rest zero" must agree with
    /// [`embed_subfield`], which is the slot-0 fast path used by the verifier.
    fn assert_psi_embed_slot_zero_matches_embed_subfield<const D: usize, const K: usize>() {
        let params = SubfieldParams::<D, K>::new().unwrap();
        let single: [AkitaF32; K] = std::array::from_fn(|j| AkitaF32::from_u64(2 + 7 * j as u64));

        let mut coords = vec![AkitaF32::zero(); D];
        coords[..K].copy_from_slice(&single);

        let packed = psi_embed::<AkitaF32, D, K>(params, &coords).unwrap();
        let direct = embed_subfield::<AkitaF32, D, K>(params, &single);

        assert_eq!(packed, direct);
    }

    #[test]
    fn embed_subfield_matches_psi_embed_first_slot() {
        assert_psi_embed_slot_zero_matches_embed_subfield::<8, 2>();
        assert_psi_embed_slot_zero_matches_embed_subfield::<8, 4>();
        assert_psi_embed_slot_zero_matches_embed_subfield::<64, 4>();
        assert_psi_embed_slot_zero_matches_embed_subfield::<64, 8>();
        assert_psi_embed_slot_zero_matches_embed_subfield::<128, 16>();
    }

    fn assert_embed_subfield_scales_packed_slots<const D: usize, const K: usize>() {
        let params = SubfieldParams::<D, K>::new().unwrap();
        let packed_len = D / K;
        let slot_coords = (0..D)
            .map(|idx| AkitaF32::from_u64((3 * idx + 5) as u64))
            .collect::<Vec<_>>();
        let gamma: [AkitaF32; K] =
            std::array::from_fn(|idx| AkitaF32::from_u64((7 * idx + 2) as u64));

        let basis = ring_subfield_basis::<AkitaF32, D, K>(params);
        let gamma_ring = embed_subfield::<AkitaF32, D, K>(params, &gamma);
        let packed = psi_embed::<AkitaF32, D, K>(params, &slot_coords).unwrap();
        let scaled = gamma_ring * packed;

        let mut expected_coords = Vec::with_capacity(D);
        for slot in 0..packed_len {
            let slot_ring = slot_coords[(slot * K)..((slot + 1) * K)]
                .iter()
                .zip(basis.iter())
                .fold(
                    CyclotomicRing::<AkitaF32, D>::zero(),
                    |acc, (&coord, basis)| acc + basis.scale(&coord),
                );
            let product = gamma_ring * slot_ring;
            expected_coords.extend(ring_subfield_coords(params, &product));
        }
        let expected = psi_embed::<AkitaF32, D, K>(params, &expected_coords).unwrap();
        assert_eq!(scaled, expected);
    }

    #[test]
    fn embed_subfield_scales_packed_slots() {
        assert_embed_subfield_scales_packed_slots::<8, 2>();
        assert_embed_subfield_scales_packed_slots::<8, 4>();
    }

    #[test]
    fn ring_subfield_k4_basis_has_chebyshev_multiplication_table() {
        const D: usize = 8;
        let params = SubfieldParams::<D, 4>::new().unwrap();
        let basis = ring_subfield_basis::<AkitaF32, D, 4>(params);
        let two = AkitaF32::from_u64(2);

        assert_eq!(
            ring_subfield_coords(params, &(basis[1] * basis[1])),
            vec![two, AkitaF32::zero(), AkitaF32::one(), AkitaF32::zero()]
        );
        assert_eq!(
            ring_subfield_coords(params, &(basis[1] * basis[2])),
            vec![
                AkitaF32::zero(),
                AkitaF32::one(),
                AkitaF32::zero(),
                AkitaF32::one()
            ]
        );
        assert_eq!(
            ring_subfield_coords(params, &(basis[1] * basis[3])),
            vec![
                AkitaF32::zero(),
                AkitaF32::zero(),
                AkitaF32::one(),
                AkitaF32::zero()
            ]
        );
        assert_eq!(
            ring_subfield_coords(params, &(basis[2] * basis[2])),
            vec![two, AkitaF32::zero(), AkitaF32::zero(), AkitaF32::zero()]
        );
        assert_eq!(
            ring_subfield_coords(params, &(basis[2] * basis[3])),
            vec![
                AkitaF32::zero(),
                AkitaF32::one(),
                AkitaF32::zero(),
                -AkitaF32::one()
            ]
        );
        assert_eq!(
            ring_subfield_coords(params, &(basis[3] * basis[3])),
            vec![two, AkitaF32::zero(), -AkitaF32::one(), AkitaF32::zero()]
        );
    }

    #[test]
    fn naive_k4_basis_is_not_the_current_tower_power_basis() {
        const D: usize = 8;
        let params = SubfieldParams::<D, 4>::new().unwrap();
        let basis = ring_subfield_basis::<AkitaF32, D, 4>(params);

        assert_ne!(basis[1] * basis[1], basis[2]);
        assert_eq!(
            ring_subfield_coords(params, &(basis[1] * basis[1])),
            vec![
                AkitaF32::from_u64(2),
                AkitaF32::zero(),
                AkitaF32::one(),
                AkitaF32::zero()
            ]
        );
    }

    #[test]
    fn ring_subfield_k4_contains_current_tower_after_base_change() {
        const D: usize = 8;
        type E = TowerBasisFp4<AkitaF32, TwoNr, UnitNr>;

        let params = SubfieldParams::<D, 4>::new().unwrap();
        let basis = ring_subfield_basis::<AkitaF32, D, 4>(params);
        let a = AkitaF32::from_u64(1_492_342_050);
        let ai = a * AkitaF32::from_u64(3_311_696_422);
        let v = basis[1].scale(&a) + basis[3].scale(&ai);
        let u = basis[2];

        assert_eq!(v * v, u);
        assert_eq!(u * u, basis[0].scale(&AkitaF32::from_u64(2)));
        assert_eq!(v * v * v * v, basis[0].scale(&AkitaF32::from_u64(2)));

        let x = E::from_base_slice(&[
            AkitaF32::from_u64(3),
            AkitaF32::from_u64(5),
            AkitaF32::from_u64(7),
            AkitaF32::from_u64(11),
        ]);
        let y = E::from_base_slice(&[
            AkitaF32::from_u64(13),
            AkitaF32::from_u64(17),
            AkitaF32::from_u64(19),
            AkitaF32::from_u64(23),
        ]);

        assert_eq!(
            embed_tower_in_ring_subfield::<D>(x * y),
            embed_tower_in_ring_subfield::<D>(x) * embed_tower_in_ring_subfield::<D>(y)
        );
    }

    fn assert_ring_subfield_fp4_embedding_is_multiplicative<const D: usize>() {
        let params = SubfieldParams::<D, 4>::new().unwrap();
        let x = RingSubfieldFp4::new([
            AkitaF32::from_u64(3),
            AkitaF32::from_u64(5),
            AkitaF32::from_u64(7),
            AkitaF32::from_u64(11),
        ]);
        let y = RingSubfieldFp4::new([
            AkitaF32::from_u64(13),
            AkitaF32::from_u64(17),
            AkitaF32::from_u64(19),
            AkitaF32::from_u64(23),
        ]);

        assert_eq!(
            embed_subfield::<AkitaF32, D, 4>(params, &(x * y).coeffs),
            embed_subfield::<AkitaF32, D, 4>(params, &x.coeffs)
                * embed_subfield::<AkitaF32, D, 4>(params, &y.coeffs)
        );
    }

    #[test]
    fn ring_subfield_fp4_embedding_places_coefficients_in_ring_subfield_basis() {
        const D: usize = 8;
        let params = SubfieldParams::<D, 4>::new().unwrap();
        let x = RingSubfieldFp4::new([
            AkitaF32::from_u64(2),
            AkitaF32::from_u64(3),
            AkitaF32::from_u64(5),
            AkitaF32::from_u64(7),
        ]);
        let embedded = embed_subfield::<AkitaF32, D, 4>(params, &x.coeffs);
        let coeffs = embedded.coefficients();

        assert_eq!(coeffs[0], AkitaF32::from_u64(2));
        assert_eq!(coeffs[1], AkitaF32::from_u64(3));
        assert_eq!(coeffs[7], -AkitaF32::from_u64(3));
        assert_eq!(coeffs[2], AkitaF32::from_u64(5));
        assert_eq!(coeffs[6], -AkitaF32::from_u64(5));
        assert_eq!(coeffs[3], AkitaF32::from_u64(7));
        assert_eq!(coeffs[5], -AkitaF32::from_u64(7));
        assert!(coeffs[4].is_zero());
    }

    #[test]
    fn ring_subfield_fp4_embedding_is_multiplicative_across_ring_dimensions() {
        assert_ring_subfield_fp4_embedding_is_multiplicative::<8>();
        assert_ring_subfield_fp4_embedding_is_multiplicative::<64>();
        assert_ring_subfield_fp4_embedding_is_multiplicative::<128>();
    }

    /// Generate `D / 4` deterministic `RingSubfieldFp4` elements seeded by `tag`.
    fn deterministic_subfield_fp4_vector<const D: usize>(
        tag: u64,
    ) -> Vec<RingSubfieldFp4<AkitaF32>> {
        let m = D / 4;
        (0..m)
            .map(|i| {
                let i = i as u64;
                RingSubfieldFp4::new([
                    AkitaF32::from_u64(2 + 7 * i + 11 * tag),
                    AkitaF32::from_u64(3 + 13 * i + 17 * tag),
                    AkitaF32::from_u64(5 + 19 * i + 23 * tag),
                    AkitaF32::from_u64(7 + 29 * i + 31 * tag),
                ])
            })
            .collect()
    }

    /// Flatten `D / 4` typed `RingSubfieldFp4` slots into the
    /// `[s_0[0], s_0[1], s_0[2], s_0[3], s_1[0], ...]` layout consumed by
    /// [`psi_embed`].
    fn flatten_subfield_fp4_vector<const D: usize>(
        elements: &[RingSubfieldFp4<AkitaF32>],
    ) -> Vec<AkitaF32> {
        assert_eq!(elements.len(), D / 4);
        let mut coords = vec![AkitaF32::zero(); D];
        for (i, elem) in elements.iter().enumerate() {
            coords[i * 4..i * 4 + 4].copy_from_slice(&elem.coeffs);
        }
        coords
    }

    /// Verify the trace inner-product relation
    /// `Tr_H(psi(s) * sigma_{-1}(psi(v))) = (D / k) * embed_subfield(<s, v>)`
    /// for the typed `k = 4` ring-subfield representation.
    fn assert_psi_trace_inner_product_identity_fp4<const D: usize>() {
        let params = SubfieldParams::<D, 4>::new().unwrap();
        let s = deterministic_subfield_fp4_vector::<D>(0);
        let v = deterministic_subfield_fp4_vector::<D>(1);

        // y = <s, v> in the ring-subfield.
        let y = s
            .iter()
            .zip(v.iter())
            .fold(RingSubfieldFp4::zero(), |acc, (si, vi)| acc + (*si * *vi));

        let s_flat = flatten_subfield_fp4_vector::<D>(&s);
        let v_flat = flatten_subfield_fp4_vector::<D>(&v);
        let big_y = psi_embed::<AkitaF32, D, 4>(params, &s_flat).unwrap();
        let big_v = psi_embed::<AkitaF32, D, 4>(params, &v_flat).unwrap();
        let traced = trace_h(params, &(big_y * big_v.sigma_m1()));

        let scale = AkitaF32::from_u64(params.packed_len() as u64);
        let scaled = embed_subfield::<AkitaF32, D, 4>(params, &y.coeffs).scale(&scale);

        assert_eq!(traced, scaled);
    }

    #[test]
    fn psi_trace_inner_product_identity_fp4() {
        assert_psi_trace_inner_product_identity_fp4::<8>();
        assert_psi_trace_inner_product_identity_fp4::<64>();
        assert_psi_trace_inner_product_identity_fp4::<128>();
    }

    /// Subfield multiplication for `k = 2`: `e_1^2 = 2` for any valid `D`,
    /// so `R_q^H ≅ F_q[sqrt(2)]`.
    fn fp2_subfield_mul(a: [AkitaF32; 2], b: [AkitaF32; 2]) -> [AkitaF32; 2] {
        let two = AkitaF32::from_u64(2);
        [a[0] * b[0] + two * a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
    }

    fn assert_psi_trace_inner_product_identity_fp2<const D: usize>() {
        let params = SubfieldParams::<D, 2>::new().unwrap();
        let m = params.packed_len();

        let s: Vec<[AkitaF32; 2]> = (0..m)
            .map(|i| {
                let i = i as u64;
                [
                    AkitaF32::from_u64(2 + 7 * i),
                    AkitaF32::from_u64(3 + 13 * i),
                ]
            })
            .collect();
        let v: Vec<[AkitaF32; 2]> = (0..m)
            .map(|i| {
                let i = i as u64;
                [
                    AkitaF32::from_u64(11 + 19 * i),
                    AkitaF32::from_u64(17 + 23 * i),
                ]
            })
            .collect();

        let y = s
            .iter()
            .zip(v.iter())
            .fold([AkitaF32::zero(); 2], |acc, (si, vi)| {
                let prod = fp2_subfield_mul(*si, *vi);
                [acc[0] + prod[0], acc[1] + prod[1]]
            });

        let mut s_flat = vec![AkitaF32::zero(); D];
        let mut v_flat = vec![AkitaF32::zero(); D];
        for (i, (sc, vc)) in s.iter().zip(v.iter()).enumerate() {
            s_flat[i * 2] = sc[0];
            s_flat[i * 2 + 1] = sc[1];
            v_flat[i * 2] = vc[0];
            v_flat[i * 2 + 1] = vc[1];
        }

        let big_y = psi_embed::<AkitaF32, D, 2>(params, &s_flat).unwrap();
        let big_v = psi_embed::<AkitaF32, D, 2>(params, &v_flat).unwrap();
        let traced = trace_h(params, &(big_y * big_v.sigma_m1()));

        let scale = AkitaF32::from_u64(m as u64);
        let scaled = embed_subfield::<AkitaF32, D, 2>(params, &y).scale(&scale);

        assert_eq!(traced, scaled);
    }

    #[test]
    fn check_trace_inner_product_k_one_accepts_correct_opening() {
        const D: usize = 8;
        let params = SubfieldParams::<D, 1>::new().unwrap();
        let y_ring = ring_from_i64s([2, 3, 5, 7, 11, 13, 17, 19]);
        let inner_point = [F::from_u64(3), F::from_u64(5), F::from_u64(7)];

        for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
            let v = reduce_inner_opening_to_ring_element::<F, D>(&inner_point, basis).unwrap();
            let product = y_ring * v.sigma_m1();
            let opening = product.coefficients()[0];

            assert!(check_trace_inner_product::<F, D, 1>(
                params,
                &product,
                &[opening]
            ));
        }
    }

    #[test]
    fn check_trace_inner_product_k_one_rejects_wrong_opening() {
        const D: usize = 8;
        let params = SubfieldParams::<D, 1>::new().unwrap();
        let y_ring = ring_from_i64s([2, 3, 5, 7, 11, 13, 17, 19]);
        let v = ring_from_i64s([1, 1, 1, 1, 1, 1, 1, 1]);
        let product = y_ring * v.sigma_m1();
        let wrong = product.coefficients()[0] + F::one();

        assert!(!check_trace_inner_product::<F, D, 1>(
            params,
            &product,
            &[wrong]
        ));
    }

    /// Verify [`check_trace_inner_product`] against the K-generic ring
    /// identity for `K = 4`, both on a true witness and on a perturbed
    /// witness, across all production ring sizes.
    fn assert_check_trace_inner_product_fp4<const D: usize>() {
        let params = SubfieldParams::<D, 4>::new().unwrap();
        let s = deterministic_subfield_fp4_vector::<D>(0);
        let v = deterministic_subfield_fp4_vector::<D>(1);

        let y = s
            .iter()
            .zip(v.iter())
            .fold(RingSubfieldFp4::zero(), |acc, (si, vi)| acc + (*si * *vi));
        let s_flat = flatten_subfield_fp4_vector::<D>(&s);
        let v_flat = flatten_subfield_fp4_vector::<D>(&v);
        let big_y = psi_embed::<AkitaF32, D, 4>(params, &s_flat).unwrap();
        let big_v = psi_embed::<AkitaF32, D, 4>(params, &v_flat).unwrap();
        let trace_input = big_y * big_v.sigma_m1();

        assert!(check_trace_inner_product::<AkitaF32, D, 4>(
            params,
            &trace_input,
            &y.coeffs
        ));

        let mut wrong = y.coeffs;
        wrong[0] += AkitaF32::one();
        assert!(!check_trace_inner_product::<AkitaF32, D, 4>(
            params,
            &trace_input,
            &wrong
        ));
    }

    #[test]
    fn check_trace_inner_product_fp4_across_ring_dimensions() {
        assert_check_trace_inner_product_fp4::<8>();
        assert_check_trace_inner_product_fp4::<64>();
        assert_check_trace_inner_product_fp4::<128>();
    }

    #[test]
    fn psi_trace_inner_product_identity_fp2() {
        assert_psi_trace_inner_product_identity_fp2::<8>();
        assert_psi_trace_inner_product_identity_fp2::<64>();
        assert_psi_trace_inner_product_identity_fp2::<128>();
    }
}
