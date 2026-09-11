//! Validated compact coordinates for ring-subfield opening multipliers.

use crate::{embed_subfield, FpExtEncoding, SubfieldParams};
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use jolt_field::{ExtField, Field};

/// A validated pair of compact ring-subfield multiplier vectors.
///
/// Construction checks the extension/ring embedding once. Private storage then
/// keeps the coordinate lengths, extension degree, and ring dimension in sync.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubfieldMultiplierOpeningPoint<F: Field> {
    position_coordinates: Vec<F>,
    live_block_coordinates: Vec<F>,
    extension_degree: usize,
    ring_dim: usize,
}

impl<F: Field> SubfieldMultiplierOpeningPoint<F> {
    pub(super) fn new<E, const D: usize>(
        position_weights: &[E],
        live_block_weights: &[E],
        error: AkitaError,
    ) -> Result<Self, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        validate_subfield_shape::<D>(E::DEGREE, error.clone())?;
        Ok(Self {
            position_coordinates: collect_subfield_coordinates(
                position_weights,
                E::DEGREE,
                error.clone(),
            )?,
            live_block_coordinates: collect_subfield_coordinates(
                live_block_weights,
                E::DEGREE,
                error,
            )?,
            extension_degree: E::DEGREE,
            ring_dim: D,
        })
    }

    pub(super) const fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// Check that these coordinates were validated for ring dimension `D`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when `D` differs from the stored dimension.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dim == D {
            Ok(())
        } else {
            Err(AkitaError::InvalidInput(format!(
                "ring multiplier ring_d={} does not match requested D={D}",
                self.ring_dim
            )))
        }
    }

    /// Number of compact position multipliers.
    pub fn position_len(&self) -> usize {
        self.position_coordinates.len() / self.extension_degree
    }

    /// Number of compact live-block multipliers.
    pub fn fold_len(&self) -> usize {
        self.live_block_coordinates.len() / self.extension_degree
    }

    pub(super) fn position_coordinates_flat(&self) -> &[F] {
        &self.position_coordinates
    }

    pub(super) const fn extension_degree(&self) -> usize {
        self.extension_degree
    }

    pub(super) fn is_constant(&self) -> bool {
        self.position_coordinates
            .chunks_exact(self.extension_degree)
            .all(|value| subfield_constant(value).is_some())
            && self
                .live_block_coordinates
                .chunks_exact(self.extension_degree)
                .all(|value| subfield_constant(value).is_some())
    }

    /// Materialize the position multipliers for an arbitrary-ring source kernel.
    ///
    /// # Errors
    ///
    /// Returns an error when `D` differs from the validated ring dimension.
    pub fn materialize_position_rings<const D: usize>(
        &self,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.ensure_ring_dim::<D>()?;
        materialize_subfield_rings::<F, D>(&self.position_coordinates, self.extension_degree)
    }

    /// Materialize the live-block multipliers for an arbitrary-ring source kernel.
    ///
    /// # Errors
    ///
    /// Returns an error when `D` differs from the validated ring dimension.
    pub fn materialize_fold_rings<const D: usize>(
        &self,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.ensure_ring_dim::<D>()?;
        materialize_subfield_rings::<F, D>(&self.live_block_coordinates, self.extension_degree)
    }

    pub(super) fn eval_position_at<E>(&self, idx: usize, alpha_pows: &[E]) -> Result<E, AkitaError>
    where
        E: ExtField<F>,
    {
        eval_subfield_at_pows(self.position_coordinates(idx)?, self.ring_dim, alpha_pows)
    }

    pub(super) fn fold_subfield_value<E>(&self, idx: usize) -> Result<E, AkitaError>
    where
        E: ExtField<F>,
    {
        if E::DEGREE != self.extension_degree {
            return Err(AkitaError::InvalidProof);
        }
        Ok(E::from_base_slice(self.fold_coordinates(idx)?))
    }

    pub(super) fn accumulate_position_product<const D: usize>(
        &self,
        idx: usize,
        rhs: &CyclotomicRing<F, D>,
        output: &mut CyclotomicRing<F, D>,
    ) -> Result<(), AkitaError> {
        self.ensure_ring_dim::<D>()?;
        add_subfield_product(
            self.position_coordinates(idx)?,
            self.extension_degree,
            rhs,
            output,
        )
    }

    pub(super) fn accumulate_position_product_high_half<const D: usize>(
        &self,
        idx: usize,
        rhs: &CyclotomicRing<F, D>,
        output: &mut [F],
    ) -> Result<(), AkitaError> {
        self.ensure_ring_dim::<D>()?;
        if output.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: output.len(),
            });
        }
        add_subfield_product_high_half(
            self.position_coordinates(idx)?,
            self.extension_degree,
            rhs,
            output,
        )
    }

    /// Add `fold[idx] * rhs` to `output` without materializing the multiplier.
    ///
    /// # Errors
    ///
    /// Returns an error for a mismatched ring dimension or out-of-range index.
    pub fn accumulate_fold_product<const D: usize>(
        &self,
        idx: usize,
        rhs: &CyclotomicRing<F, D>,
        output: &mut CyclotomicRing<F, D>,
    ) -> Result<(), AkitaError> {
        self.ensure_ring_dim::<D>()?;
        add_subfield_product(
            self.fold_coordinates(idx)?,
            self.extension_degree,
            rhs,
            output,
        )
    }

    /// Add `scale * position[idx] * X^shift` without materializing the multiplier.
    ///
    /// # Errors
    ///
    /// Returns an error for a mismatched ring dimension, index, or shift.
    pub fn accumulate_position_monomial<const D: usize>(
        &self,
        idx: usize,
        shift: usize,
        scale: F,
        output: &mut CyclotomicRing<F, D>,
    ) -> Result<(), AkitaError> {
        self.ensure_ring_dim::<D>()?;
        add_shifted_subfield_monomial(
            self.position_coordinates(idx)?,
            self.extension_degree,
            shift,
            scale,
            output,
        )
    }

    pub(super) fn fold_constant_coeff(&self, idx: usize) -> Option<F> {
        subfield_constant(self.fold_coordinates(idx).ok()?)
    }

    fn position_coordinates(&self, idx: usize) -> Result<&[F], AkitaError> {
        coordinate_chunk(&self.position_coordinates, self.extension_degree, idx)
    }

    fn fold_coordinates(&self, idx: usize) -> Result<&[F], AkitaError> {
        coordinate_chunk(&self.live_block_coordinates, self.extension_degree, idx)
    }
}

fn coordinate_chunk<F>(coordinates: &[F], degree: usize, idx: usize) -> Result<&[F], AkitaError> {
    let start = idx.checked_mul(degree).ok_or(AkitaError::InvalidProof)?;
    let end = start.checked_add(degree).ok_or(AkitaError::InvalidProof)?;
    coordinates.get(start..end).ok_or(AkitaError::InvalidProof)
}

fn subfield_constant<F: Field>(coordinates: &[F]) -> Option<F> {
    let (&constant, rest) = coordinates.split_first()?;
    rest.iter()
        .all(|coordinate| coordinate.is_zero())
        .then_some(constant)
}

fn validate_subfield_shape<const D: usize>(
    extension_degree: usize,
    error: AkitaError,
) -> Result<(), AkitaError> {
    let valid = match extension_degree {
        1 => SubfieldParams::<D, 1>::new().is_ok(),
        2 => SubfieldParams::<D, 2>::new().is_ok(),
        4 => SubfieldParams::<D, 4>::new().is_ok(),
        8 => SubfieldParams::<D, 8>::new().is_ok(),
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(error)
    }
}

fn collect_subfield_coordinates<F, E>(
    values: &[E],
    extension_degree: usize,
    error: AkitaError,
) -> Result<Vec<F>, AkitaError>
where
    F: Field,
    E: FpExtEncoding<F>,
{
    let coordinate_len = values
        .len()
        .checked_mul(extension_degree)
        .ok_or_else(|| error.clone())?;
    let mut coordinates = Vec::new();
    coordinates
        .try_reserve_exact(coordinate_len)
        .map_err(|_| error.clone())?;
    for value in values {
        let value_coordinates = value.ext_coords();
        if value_coordinates.len() != extension_degree {
            return Err(error);
        }
        coordinates.extend_from_slice(value_coordinates);
    }
    Ok(coordinates)
}

fn materialize_subfield_rings<F: Field, const D: usize>(
    coordinates: &[F],
    extension_degree: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    macro_rules! arm {
        ($k:expr) => {{
            let params = SubfieldParams::<D, $k>::new()?;
            coordinates
                .chunks_exact($k)
                .map(|value| {
                    let value: &[F; $k] = value.try_into().map_err(|_| AkitaError::InvalidProof)?;
                    Ok(embed_subfield(params, value))
                })
                .collect()
        }};
    }
    match extension_degree {
        1 => arm!(1),
        2 => arm!(2),
        4 => arm!(4),
        8 => arm!(8),
        _ => Err(AkitaError::InvalidProof),
    }
}

fn eval_subfield_at_pows<F, E>(
    coordinates: &[F],
    ring_dim: usize,
    alpha_pows: &[E],
) -> Result<E, AkitaError>
where
    F: Field,
    E: ExtField<F>,
{
    let extension_degree = coordinates.len();
    if extension_degree != E::DEGREE || alpha_pows.len() != ring_dim {
        return Err(AkitaError::InvalidProof);
    }
    let (&constant, nonconstant) = coordinates.split_first().ok_or(AkitaError::InvalidProof)?;
    let basis_pairs = subfield_basis_pairs(ring_dim, extension_degree)?;
    let mut value = E::lift_base(constant);
    for (&coordinate, &(basis_index, inverse_index)) in nonconstant.iter().zip(&basis_pairs) {
        let positive = alpha_pows
            .get(basis_index)
            .ok_or(AkitaError::InvalidProof)?;
        let negative = alpha_pows
            .get(inverse_index)
            .ok_or(AkitaError::InvalidProof)?;
        value += (*positive - *negative).mul_base(coordinate);
    }
    Ok(value)
}

/// Canonical positive/inverse basis indices for nonconstant subfield coordinates.
pub(super) fn subfield_basis_pairs(
    ring_dim: usize,
    extension_degree: usize,
) -> Result<Vec<(usize, usize)>, AkitaError> {
    let denominator = 2usize
        .checked_mul(extension_degree)
        .filter(|&value| value != 0 && ring_dim.is_multiple_of(value))
        .ok_or(AkitaError::InvalidProof)?;
    let stride = ring_dim
        .checked_div(denominator)
        .filter(|&value| value != 0)
        .ok_or(AkitaError::InvalidProof)?;
    (1..extension_degree)
        .map(|coordinate| {
            let basis_index = coordinate
                .checked_mul(stride)
                .filter(|&index| index < ring_dim)
                .ok_or(AkitaError::InvalidProof)?;
            let inverse_index = ring_dim
                .checked_sub(basis_index)
                .ok_or(AkitaError::InvalidProof)?;
            Ok((basis_index, inverse_index))
        })
        .collect()
}

fn add_subfield_product<F: Field, const D: usize>(
    coordinates: &[F],
    extension_degree: usize,
    rhs: &CyclotomicRing<F, D>,
    output: &mut CyclotomicRing<F, D>,
) -> Result<(), AkitaError> {
    let stride = D / (2 * extension_degree);
    for (index, &coordinate) in coordinates.iter().enumerate() {
        if coordinate.is_zero() {
            continue;
        }
        let shift = index.checked_mul(stride).ok_or(AkitaError::InvalidProof)?;
        if shift >= D {
            return Err(AkitaError::InvalidProof);
        }
        rhs.shift_scale_accumulate_into(output, shift, coordinate);
        if shift != 0 {
            rhs.shift_scale_accumulate_into(output, D - shift, -coordinate);
        }
    }
    Ok(())
}

fn add_shifted_subfield_monomial<F: Field, const D: usize>(
    coordinates: &[F],
    extension_degree: usize,
    shift: usize,
    scale: F,
    output: &mut CyclotomicRing<F, D>,
) -> Result<(), AkitaError> {
    if coordinates.len() != extension_degree || shift >= D {
        return Err(AkitaError::InvalidProof);
    }
    if scale.is_zero() {
        return Ok(());
    }
    let stride = D / (2 * extension_degree);
    let output = output.coefficients_mut();
    let mut accumulate = |position: usize, coefficient: F| {
        let target = position + shift;
        if target < D {
            output[target] += coefficient * scale;
        } else {
            output[target - D] -= coefficient * scale;
        }
    };
    if !coordinates[0].is_zero() {
        accumulate(0, coordinates[0]);
    }
    for (index, &coordinate) in coordinates.iter().enumerate().skip(1) {
        if coordinate.is_zero() {
            continue;
        }
        let position = index * stride;
        accumulate(position, coordinate);
        accumulate(D - position, -coordinate);
    }
    Ok(())
}

fn add_subfield_product_high_half<F: Field, const D: usize>(
    coordinates: &[F],
    extension_degree: usize,
    rhs: &CyclotomicRing<F, D>,
    output: &mut [F],
) -> Result<(), AkitaError> {
    let stride = D / (2 * extension_degree);
    let rhs = rhs.coefficients();
    for (index, &coordinate) in coordinates.iter().enumerate().skip(1) {
        if coordinate.is_zero() {
            continue;
        }
        let shift = index.checked_mul(stride).ok_or(AkitaError::InvalidProof)?;
        if shift == 0 || shift >= D {
            return Err(AkitaError::InvalidProof);
        }
        for rhs_index in (D - shift)..D {
            output[shift + rhs_index - D] += coordinate * rhs[rhs_index];
        }
        let inverse_shift = D - shift;
        for rhs_index in shift..D {
            output[inverse_shift + rhs_index - D] -= coordinate * rhs[rhs_index];
        }
    }
    Ok(())
}
