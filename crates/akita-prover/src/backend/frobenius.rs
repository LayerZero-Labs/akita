//! Frobenius-conjugate transformations for base-coefficient dense polynomials.

use akita_field::{
    canonical_frobenius_thetas, solve_frobenius_moore, validate_canonical_frobenius_thetas,
    AkitaError, CanonicalField, ExtField, FrobeniusExtField, FromPrimitiveInt, PseudoMersenneField,
};
use akita_types::{
    basis_weights, embed_ring_subfield_vector, pack_frobenius_base_lift_i8_digits, BasisMode,
    RingSubfieldEncoding,
};

use crate::{AkitaPolyOps, DensePoly, RecursiveWitnessFlat};

/// Prover-side dense Frobenius transform output.
#[derive(Debug, Clone)]
pub struct DenseFrobeniusTransform<
    F: akita_field::FieldCore,
    E: akita_field::FieldCore,
    const D: usize,
> {
    /// Transformed polynomial `g`, committed through the usual Akita ring path.
    pub polynomial: DensePoly<F, D>,
    /// Number of Boolean variables in the original base-field table.
    pub original_num_vars: usize,
    /// Number of head variables packed into the theta basis.
    pub split_bits: usize,
    /// Number of packed head slices, equal to `2^split_bits`.
    pub width: usize,
    /// Number of Boolean variables in the extension-valued tail table `g`.
    pub extension_num_vars: usize,
    /// Number of scalar `F` protocol variables after ring-subfield packing.
    pub protocol_num_vars: usize,
    /// Original public opening point.
    pub original_point: Vec<E>,
    /// Original public claimed evaluation reconstructed from internal claims.
    pub original_claim: E,
    /// Extension-domain Frobenius-conjugate tail points.
    pub extension_points: Vec<Vec<E>>,
    /// Protocol opening points after ring-subfield packing coordinates are exposed.
    pub protocol_points: Vec<Vec<E>>,
    /// Claimed openings `s_j = g(x_tail^(q^j))`.
    pub internal_claims: Vec<E>,
    /// Deterministic theta family used for the head-slice packing.
    pub thetas: Vec<E>,
}

/// Frobenius-conjugate opening plan for a logical base-field witness.
#[derive(Debug, Clone)]
pub struct FrobeniusOpeningPlan<E: akita_field::FieldCore> {
    /// Number of logical head variables packed into canonical extension slots.
    pub split_bits: usize,
    /// Number of internal openings, equal to `2^split_bits`.
    pub width: usize,
    /// Number of variables in the extension-valued tail table.
    pub extension_num_vars: usize,
    /// Protocol opening points for the committed packed table.
    pub protocol_points: Vec<Vec<E>>,
}

/// Build the per-level Frobenius opening plan.
///
/// For `E::EXT_DEGREE == 1`, this returns the degree-one plan:
/// `split_bits = 0`, `width = 1`, and the protocol point is the logical point.
///
/// # Errors
///
/// Returns an error if the extension degree is unsupported or if the logical
/// arity is too small for the full canonical Frobenius split.
pub fn frobenius_opening_plan<F, E, const D: usize>(
    logical_point: &[E],
) -> Result<FrobeniusOpeningPlan<E>, AkitaError>
where
    F: akita_field::FieldCore,
    E: FrobeniusExtField<F>,
{
    let split_bits = E::EXT_DEGREE.trailing_zeros() as usize;
    let width = 1usize
        .checked_shl(split_bits as u32)
        .ok_or_else(|| AkitaError::InvalidInput("Frobenius width overflow".to_string()))?;
    if width != E::EXT_DEGREE || !E::EXT_DEGREE.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "Frobenius opening requires power-of-two extension degree".to_string(),
        ));
    }
    if split_bits > logical_point.len() {
        return Err(AkitaError::InvalidPointDimension {
            expected: split_bits,
            actual: logical_point.len(),
        });
    }
    let extension_num_vars = logical_point.len() - split_bits;
    let tail_point = &logical_point[split_bits..];
    let mut protocol_points = Vec::with_capacity(width);
    for power in 0..width {
        let conjugate_tail = tail_point
            .iter()
            .copied()
            .map(|coord| E::frobenius_pow(coord, power))
            .collect::<Vec<_>>();
        protocol_points.push(ring_subfield_packed_extension_opening_point::<F, E, D>(
            extension_num_vars,
            &conjugate_tail,
        )?);
    }
    Ok(FrobeniusOpeningPlan {
        split_bits,
        width,
        extension_num_vars,
        protocol_points,
    })
}

/// Reconstruct the logical witness opening from internal Frobenius openings.
///
/// # Errors
///
/// Returns an error if the opening count does not match the canonical split or
/// if the Moore-type system is singular.
pub fn reconstruct_frobenius_opening<F, E>(
    logical_point: &[E],
    split_bits: usize,
    internal_claims: &[E],
) -> Result<E, AkitaError>
where
    F: PseudoMersenneField + FromPrimitiveInt,
    E: FrobeniusExtField<F>,
{
    let width = 1usize
        .checked_shl(split_bits as u32)
        .ok_or_else(|| AkitaError::InvalidInput("Frobenius width overflow".to_string()))?;
    if internal_claims.len() != width || logical_point.len() < split_bits {
        return Err(AkitaError::InvalidInput(
            "Frobenius reconstruction shape mismatch".to_string(),
        ));
    }
    let thetas = canonical_frobenius_thetas::<F, E>(width)?;
    let r = internal_claims
        .iter()
        .enumerate()
        .map(|(idx, &claim)| E::frobenius_inv_pow(claim, idx))
        .collect::<Vec<_>>();
    let z = solve_frobenius_moore::<F, E>(&thetas, &r)?;
    let head_weights = basis_weights(&logical_point[..split_bits], BasisMode::Lagrange);
    Ok(head_weights
        .into_iter()
        .zip(z)
        .fold(E::zero(), |acc, (weight, z_h)| acc + weight * z_h))
}

/// Pack a logical recursive digit witness using the canonical Frobenius plan.
///
/// For degree-one fields this is the identity. For small fields this stores
/// the transformed extension-valued table in the same ring-subfield layout used
/// by root extension openings.
///
/// # Errors
///
/// Returns an error if the logical witness length is not compatible with the
/// full Frobenius split or if ring-subfield packing fails.
pub fn frobenius_pack_recursive_witness<F, E, const D: usize>(
    logical_w: &RecursiveWitnessFlat,
) -> Result<RecursiveWitnessFlat, AkitaError>
where
    F: akita_field::FieldCore,
    E: ExtField<F>,
{
    let split_bits = E::EXT_DEGREE.trailing_zeros() as usize;
    let width = 1usize
        .checked_shl(split_bits as u32)
        .ok_or_else(|| AkitaError::InvalidInput("Frobenius width overflow".to_string()))?;
    if width != E::EXT_DEGREE || !E::EXT_DEGREE.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "Frobenius packing requires power-of-two extension degree".to_string(),
        ));
    }
    let packed =
        pack_frobenius_base_lift_i8_digits::<D>(logical_w.as_i8_digits(), E::EXT_DEGREE, width)?;
    Ok(RecursiveWitnessFlat::from_i8_digits(packed))
}

fn dense_lagrange_opening_from_ext_evals<E>(evals: &[E], point: &[E]) -> Result<E, AkitaError>
where
    E: akita_field::FieldCore,
{
    let expected = 1usize
        .checked_shl(point.len() as u32)
        .ok_or_else(|| AkitaError::InvalidInput("opening point dimension overflow".to_string()))?;
    if evals.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: evals.len(),
        });
    }
    let mut layer = evals.to_vec();
    for &r in point {
        let one_minus_r = E::one() - r;
        let next_len = layer.len() / 2;
        for idx in 0..next_len {
            layer[idx] = layer[2 * idx] * one_minus_r + layer[2 * idx + 1] * r;
        }
        layer.truncate(next_len);
    }
    Ok(layer[0])
}

fn pack_extension_evals<F, E, const D: usize>(evals: &[E]) -> Result<DensePoly<F, D>, AkitaError>
where
    F: CanonicalField + FromPrimitiveInt,
    E: RingSubfieldEncoding<F>,
{
    if E::EXT_DEGREE == 1 {
        let base_evals = evals
            .iter()
            .map(|value| {
                value
                    .to_ring_subfield_coords()
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "degree-one extension value had no coordinate".into(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        return DensePoly::<F, D>::from_field_evals(
            base_evals.len().trailing_zeros() as usize,
            &base_evals,
        );
    }
    if D % E::EXT_DEGREE != 0 {
        return Err(AkitaError::InvalidInput(
            "extension degree must divide ring dimension".to_string(),
        ));
    }
    let packed_len = D / E::EXT_DEGREE;
    let mut rings = Vec::with_capacity(evals.len().div_ceil(packed_len));
    let error = AkitaError::InvalidInput(
        "Frobenius transform failed to psi-pack extension evaluations".to_string(),
    );
    for chunk in evals.chunks(packed_len) {
        let mut values = chunk.to_vec();
        values.resize(packed_len, E::zero());
        rings.push(embed_ring_subfield_vector::<F, E, D>(
            &values,
            error.clone(),
        )?);
    }
    Ok(DensePoly::from_ring_coeffs(rings))
}

/// Convert an extension-domain opening point into the protocol point expected
/// by the current ring-subfield-packed folded root path.
///
/// The returned point has `extension_num_vars + log2([E:F])` coordinates. The
/// extra coordinates expose the extension basis slots inside the root inner
/// ring, matching the existing lifted baseline layout.
///
/// # Errors
///
/// Returns an error when the extension degree is not a power of two, does not
/// divide `D`, or the point is too short for the packed root layout.
pub fn ring_subfield_packed_extension_opening_point<F, E, const D: usize>(
    extension_num_vars: usize,
    point: &[E],
) -> Result<Vec<E>, AkitaError>
where
    F: akita_field::FieldCore,
    E: ExtField<F>,
{
    let k = E::EXT_DEGREE;
    if k == 1 {
        return Ok(point.to_vec());
    }
    if !k.is_power_of_two() || D % k != 0 {
        return Err(AkitaError::InvalidInput(
            "extension degree must be a power of two dividing D".to_string(),
        ));
    }
    if point.len() != extension_num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: extension_num_vars,
            actual: point.len(),
        });
    }
    let alpha_bits = D.trailing_zeros() as usize;
    let kappa_bits = k.trailing_zeros() as usize;
    let packed_inner_bits = alpha_bits.checked_sub(kappa_bits).ok_or_else(|| {
        AkitaError::InvalidInput("extension degree exceeds ring dimension".to_string())
    })?;
    if extension_num_vars < packed_inner_bits {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: extension_num_vars,
        });
    }

    let mut transformed = Vec::with_capacity(extension_num_vars + kappa_bits);
    transformed.extend_from_slice(&point[..packed_inner_bits]);
    transformed.resize(alpha_bits, E::zero());
    transformed.extend_from_slice(&point[packed_inner_bits..]);
    Ok(transformed)
}

/// Build the Frobenius-conjugate packed dense polynomial and its internal
/// opening claims for one original base-field opening.
///
/// This is the first reusable optimization boundary: it constructs the
/// transformed `g` polynomial and the same-commitment multipoint claims that a
/// later protocol wrapper will bind to the original public `(x, y)`.
///
/// # Errors
///
/// Returns an error for malformed table sizes, unsupported split widths,
/// singular canonical theta matrices, or unsupported ring-subfield packing parameters.
pub fn dense_frobenius_transform<F, E, const D: usize>(
    original_num_vars: usize,
    split_bits: usize,
    evals: &[F],
    original_point: &[E],
) -> Result<DenseFrobeniusTransform<F, E, D>, AkitaError>
where
    F: CanonicalField + FromPrimitiveInt + PseudoMersenneField,
    E: FrobeniusExtField<F> + RingSubfieldEncoding<F>,
{
    if split_bits > original_num_vars {
        return Err(AkitaError::InvalidInput(
            "Frobenius split exceeds polynomial arity".to_string(),
        ));
    }
    if original_point.len() != original_num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: original_num_vars,
            actual: original_point.len(),
        });
    }
    let width = 1usize
        .checked_shl(split_bits as u32)
        .ok_or_else(|| AkitaError::InvalidInput("Frobenius split width overflow".to_string()))?;
    if width > E::EXT_DEGREE {
        return Err(AkitaError::InvalidInput(format!(
            "Frobenius split width {width} exceeds extension degree {}",
            E::EXT_DEGREE
        )));
    }
    validate_canonical_frobenius_thetas::<F, E>(width)?;
    let expected_len = 1usize
        .checked_shl(original_num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidInput("dense table length overflow".to_string()))?;
    if evals.len() != expected_len {
        return Err(AkitaError::InvalidSize {
            expected: expected_len,
            actual: evals.len(),
        });
    }

    let thetas = canonical_frobenius_thetas::<F, E>(width)?;
    let extension_num_vars = original_num_vars - split_bits;
    let tail_len = 1usize << extension_num_vars;
    let mut transformed_evals = Vec::with_capacity(tail_len);
    for tail in 0..tail_len {
        let base = tail * width;
        let value = (0..width).fold(E::zero(), |acc, head| {
            acc + thetas[head].mul_base(evals[base + head])
        });
        transformed_evals.push(value);
    }

    let polynomial = pack_extension_evals::<F, E, D>(&transformed_evals)?;
    let protocol_num_vars = polynomial.num_vars();
    let tail_point = &original_point[split_bits..];
    let mut extension_points = Vec::with_capacity(width);
    let mut protocol_points = Vec::with_capacity(width);
    let mut internal_claims = Vec::with_capacity(width);
    for power in 0..width {
        let conjugate_tail = tail_point
            .iter()
            .copied()
            .map(|coord| <E as FrobeniusExtField<F>>::frobenius_pow(coord, power))
            .collect::<Vec<_>>();
        let claim = dense_lagrange_opening_from_ext_evals(&transformed_evals, &conjugate_tail)?;
        let protocol_point = ring_subfield_packed_extension_opening_point::<F, E, D>(
            extension_num_vars,
            &conjugate_tail,
        )?;
        extension_points.push(conjugate_tail);
        protocol_points.push(protocol_point);
        internal_claims.push(claim);
    }

    let r = internal_claims
        .iter()
        .enumerate()
        .map(|(idx, &claim)| <E as FrobeniusExtField<F>>::frobenius_inv_pow(claim, idx))
        .collect::<Vec<_>>();
    let z = solve_frobenius_moore::<F, E>(&thetas, &r)?;
    let head_weights = basis_weights(&original_point[..split_bits], BasisMode::Lagrange);
    let original_claim = head_weights
        .into_iter()
        .zip(z)
        .fold(E::zero(), |acc, (weight, z_h)| acc + weight * z_h);

    Ok(DenseFrobeniusTransform {
        polynomial,
        original_num_vars,
        split_bits,
        width,
        extension_num_vars,
        protocol_num_vars,
        original_point: original_point.to_vec(),
        original_claim,
        extension_points,
        protocol_points,
        internal_claims,
        thetas,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Ext2, FieldCore, Prime32Offset99, Prime64Offset59, RingSubfieldFp4};

    fn base_dense_opening<F, E>(evals: &[F], point: &[E]) -> E
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        let mut layer = evals.iter().copied().map(E::lift_base).collect::<Vec<_>>();
        for &r in point {
            let one_minus_r = E::one() - r;
            let next_len = layer.len() / 2;
            for idx in 0..next_len {
                layer[idx] = layer[2 * idx] * one_minus_r + layer[2 * idx + 1] * r;
            }
            layer.truncate(next_len);
        }
        layer[0]
    }

    #[test]
    fn dense_frobenius_fp64_recovers_original_claim() {
        type F = Prime64Offset59;
        type E = Ext2<F>;
        const D: usize = 32;
        let num_vars = 6;
        let split_bits = 1;
        let evals = (0..(1usize << num_vars))
            .map(|idx| F::from_u64((idx as u64 * 17 + 5) % 97))
            .collect::<Vec<_>>();
        let point = (0..num_vars)
            .map(|idx| {
                E::from_base_slice(&[F::from_u64(idx as u64 + 2), F::from_u64(3 * idx as u64 + 1)])
            })
            .collect::<Vec<_>>();

        let transformed =
            dense_frobenius_transform::<F, E, D>(num_vars, split_bits, &evals, &point).unwrap();
        assert_eq!(transformed.width, 2);
        assert_eq!(transformed.extension_num_vars, num_vars - split_bits);
        assert_eq!(transformed.protocol_num_vars, num_vars - split_bits + 1);
        assert_eq!(
            transformed.original_claim,
            base_dense_opening(&evals, &point)
        );
    }

    #[test]
    fn dense_frobenius_fp32_ring_subfield_uses_canonical_basis() {
        type F = Prime32Offset99;
        type E = RingSubfieldFp4<F>;
        const D: usize = 32;
        let num_vars = 6;
        let split_bits = 2;
        let evals = (0..(1usize << num_vars))
            .map(|idx| F::from_u64((idx as u64 * 11 + 9) % 101))
            .collect::<Vec<_>>();
        let point = (0..num_vars)
            .map(|idx| {
                E::from_base_slice(&[
                    F::from_u64(idx as u64 + 1),
                    F::from_u64(idx as u64 + 2),
                    F::from_u64(idx as u64 + 3),
                    F::from_u64(idx as u64 + 4),
                ])
            })
            .collect::<Vec<_>>();

        let transformed =
            dense_frobenius_transform::<F, E, D>(num_vars, split_bits, &evals, &point).unwrap();
        let basis = canonical_frobenius_thetas::<F, E>(4).unwrap();
        assert_eq!(transformed.thetas, basis);
        assert_eq!(transformed.width, 4);
        assert_eq!(transformed.extension_num_vars, num_vars - split_bits);
        assert_eq!(transformed.protocol_num_vars, num_vars - split_bits + 2);
        assert_eq!(
            transformed.original_claim,
            base_dense_opening(&evals, &point)
        );
    }

    #[test]
    fn frobenius_reconstruction_binds_each_internal_claim() {
        type F = Prime64Offset59;
        type E = Ext2<F>;
        const D: usize = 32;
        let num_vars = 5;
        let split_bits = 1;
        let evals = (0..(1usize << num_vars))
            .map(|idx| F::from_u64((idx as u64 * 13 + 19) % 127))
            .collect::<Vec<_>>();
        let point = (0..num_vars)
            .map(|idx| {
                E::from_base_slice(&[F::from_u64(idx as u64 + 7), F::from_u64(5 * idx as u64 + 3)])
            })
            .collect::<Vec<_>>();

        let transformed =
            dense_frobenius_transform::<F, E, D>(num_vars, split_bits, &evals, &point).unwrap();
        let reconstructed =
            reconstruct_frobenius_opening::<F, E>(&point, split_bits, &transformed.internal_claims)
                .unwrap();
        assert_eq!(reconstructed, transformed.original_claim);

        let mut wrong_claims = transformed.internal_claims.clone();
        wrong_claims[1] += E::one();
        let wrong_reconstructed =
            reconstruct_frobenius_opening::<F, E>(&point, split_bits, &wrong_claims).unwrap();
        assert_ne!(wrong_reconstructed, transformed.original_claim);
    }

    #[test]
    fn frobenius_plan_rejects_logical_point_shorter_than_split() {
        type F = Prime32Offset99;
        type E = RingSubfieldFp4<F>;
        const D: usize = 32;
        let point = [E::one()];

        let err = frobenius_opening_plan::<F, E, D>(&point).unwrap_err();
        assert!(matches!(
            err,
            AkitaError::InvalidPointDimension {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn dense_frobenius_rejects_split_wider_than_extension_degree() {
        type F = Prime64Offset59;
        type E = Ext2<F>;
        const D: usize = 32;
        let num_vars = 4;
        let split_bits = 2;
        let evals = vec![F::zero(); 1usize << num_vars];
        let point = vec![E::zero(); num_vars];

        let err =
            dense_frobenius_transform::<F, E, D>(num_vars, split_bits, &evals, &point).unwrap_err();
        assert!(
            matches!(err, AkitaError::InvalidInput(msg) if msg.contains("exceeds extension degree"))
        );
    }

    #[test]
    fn recursive_frobenius_pack_rejects_non_divisible_digit_count() {
        type F = Prime32Offset99;
        type E = RingSubfieldFp4<F>;
        const D: usize = 32;
        let witness = RecursiveWitnessFlat::from_i8_digits(vec![1, 2, 3]);

        let err = frobenius_pack_recursive_witness::<F, E, D>(&witness).unwrap_err();
        assert!(matches!(
            err,
            AkitaError::InvalidSize {
                expected: 4,
                actual: 3
            }
        ));
    }
}
