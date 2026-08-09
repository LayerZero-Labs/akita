//! Validated compact coordinates for ring-subfield opening multipliers.

use crate::{embed_subfield, FpExtEncoding, SubfieldParams};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, ExtField, FieldCore};

/// A validated pair of compact ring-subfield multiplier vectors.
///
/// Construction checks the extension/ring embedding once. Private storage then
/// keeps the coordinate lengths, extension degree, and ring dimension in sync.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubfieldMultiplierOpeningPoint<F: FieldCore> {
    position_coordinates: Vec<F>,
    live_block_coordinates: Vec<F>,
    extension_degree: usize,
    ring_dim: usize,
}

impl<F: FieldCore> SubfieldMultiplierOpeningPoint<F> {
    pub(super) fn new<E, const D: usize>(
        position_weights: &[E],
        live_block_weights: &[E],
        error: AkitaError,
    ) -> Result<Self, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        validate_subfield_shape::<D>(E::EXT_DEGREE, error.clone())?;
        Ok(Self {
            position_coordinates: collect_subfield_coordinates(
                position_weights,
                E::EXT_DEGREE,
                error.clone(),
            )?,
            live_block_coordinates: collect_subfield_coordinates(
                live_block_weights,
                E::EXT_DEGREE,
                error,
            )?,
            extension_degree: E::EXT_DEGREE,
            ring_dim: D,
        })
    }

    pub(super) const fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    pub(super) fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dim == D {
            Ok(())
        } else {
            Err(AkitaError::InvalidInput(format!(
                "ring multiplier ring_d={} does not match requested D={D}",
                self.ring_dim
            )))
        }
    }

    pub(super) fn position_len(&self) -> usize {
        self.position_coordinates.len() / self.extension_degree
    }

    pub(super) fn fold_len(&self) -> usize {
        self.live_block_coordinates.len() / self.extension_degree
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

    pub(super) fn materialize_position_rings<const D: usize>(
        &self,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.ensure_ring_dim::<D>()?;
        materialize_subfield_rings::<F, D>(&self.position_coordinates, self.extension_degree)
    }

    pub(super) fn materialize_fold_rings<const D: usize>(
        &self,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.ensure_ring_dim::<D>()?;
        materialize_subfield_rings::<F, D>(&self.live_block_coordinates, self.extension_degree)
    }

    pub(super) fn eval_position_at<const D: usize, E>(
        &self,
        idx: usize,
        alpha_pows: &[E],
    ) -> Result<E, AkitaError>
    where
        E: ExtField<F>,
    {
        self.ensure_ring_dim::<D>()?;
        eval_subfield_at_pows(self.position_coordinates(idx)?, self.ring_dim, alpha_pows)
    }

    pub(super) fn fold_subfield_value<E>(&self, idx: usize) -> Result<E, AkitaError>
    where
        E: ExtField<F>,
    {
        if E::EXT_DEGREE != self.extension_degree {
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

    pub(super) fn position_constant_coeff(&self, idx: usize) -> Option<F> {
        subfield_constant(self.position_coordinates(idx).ok()?)
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

fn subfield_constant<F: FieldCore>(coordinates: &[F]) -> Option<F> {
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
    F: FieldCore,
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

fn materialize_subfield_rings<F: FieldCore, const D: usize>(
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
    F: FieldCore,
    E: ExtField<F>,
{
    let extension_degree = coordinates.len();
    if extension_degree != E::EXT_DEGREE || alpha_pows.len() != ring_dim {
        return Err(AkitaError::InvalidProof);
    }
    let (&constant, nonconstant) = coordinates.split_first().ok_or(AkitaError::InvalidProof)?;
    let stride = ring_dim / (2 * extension_degree);
    let mut value = E::lift_base(constant);
    for (offset, &coordinate) in nonconstant.iter().enumerate() {
        let basis_index = offset
            .checked_add(1)
            .and_then(|index| index.checked_mul(stride))
            .ok_or(AkitaError::InvalidProof)?;
        let inverse_index = ring_dim
            .checked_sub(basis_index)
            .ok_or(AkitaError::InvalidProof)?;
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

fn add_subfield_product<F: FieldCore, const D: usize>(
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
