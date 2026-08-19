use super::SubringCoefficientPackingGeometry;
use akita_challenges::SparseChallenge;
use akita_field::{AkitaError, ExtField, FieldCore, FromPrimitiveInt};
use std::mem;

const MAX_REFERENCE_ALLOCATION_BYTES: usize = 1 << 30;

fn checked_product(label: &str, factors: &[usize]) -> Result<usize, AkitaError> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product.checked_mul(factor).ok_or_else(|| {
            AkitaError::InvalidInput(format!("subring packing {label} length overflow"))
        })
    })
}

fn require_len(label: &str, actual: usize, expected: usize) -> Result<(), AkitaError> {
    if actual != expected {
        return Err(AkitaError::InvalidInput(format!(
            "subring packing {label} length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn zero_vec<T: FieldCore>(label: &str, len: usize) -> Result<Vec<T>, AkitaError> {
    let bytes = len.checked_mul(mem::size_of::<T>().max(1)).ok_or_else(|| {
        AkitaError::InvalidInput(format!("subring packing {label} allocation overflow"))
    })?;
    if bytes > MAX_REFERENCE_ALLOCATION_BYTES {
        return Err(AkitaError::InvalidInput(format!(
            "subring packing {label} requires {bytes} bytes, exceeding the reference allocation limit"
        )));
    }
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| {
        AkitaError::InvalidInput(format!(
            "subring packing {label} allocation failed for {len} elements"
        ))
    })?;
    values.resize(len, T::zero());
    Ok(values)
}

fn validate_extension_degree<F, E>(
    geometry: SubringCoefficientPackingGeometry,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    if E::EXT_DEGREE != geometry.extension_degree() {
        return Err(AkitaError::InvalidSetup(format!(
            "subring packing extension degree mismatch: geometry has {}, field has {}",
            geometry.extension_degree(),
            E::EXT_DEGREE
        )));
    }
    Ok(())
}

/// Apply the coefficient packing map to one A-ring coefficient vector.
///
/// `a_ring_coefficients` uses physical A-ring order. `packing_weights[a]`
/// contracts the low coefficient index `a < k h`. The output contains `s`
/// extension-field coefficients in increasing subring coefficient order.
///
/// # Errors
///
/// Returns an error when the extension degree or either input length disagrees
/// with `geometry`, or when the bounded reference allocation fails.
#[cfg(any(test, feature = "test-support"))]
pub(super) fn coefficient_packing_map<F, E>(
    geometry: SubringCoefficientPackingGeometry,
    a_ring_coefficients: &[F],
    packing_weights: &[E],
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    validate_extension_degree::<F, E>(geometry)?;
    require_len(
        "A-ring coefficient vector",
        a_ring_coefficients.len(),
        geometry.a_ring_dimension(),
    )?;
    require_len(
        "low-coefficient weight vector",
        packing_weights.len(),
        geometry.subring_embedding_stride(),
    )?;

    let mut packed = zero_vec::<E>(
        "coefficient packing map",
        geometry.challenge_subring_dimension(),
    )?;
    for (subring_coefficient_index, packed_coefficient) in packed.iter_mut().enumerate() {
        for (low_coefficient_index, &weight) in packing_weights.iter().enumerate() {
            let source_index = geometry
                .a_ring_coefficient_index(low_coefficient_index, subring_coefficient_index)?;
            let source = *a_ring_coefficients.get(source_index).ok_or_else(|| {
                AkitaError::InvalidInput("subring packing source index is out of bounds".into())
            })?;
            *packed_coefficient += weight.mul_base(source);
        }
    }
    Ok(packed)
}

/// Compute canonical coefficient packing partial openings for one claim.
///
/// The source layout is `[live position][A-ring coefficient]`, with live
/// positions split into blocks of `num_positions_per_block`. The final block
/// may contain fewer live positions than that fixed position-weight domain.
/// The returned base-field layout is
/// `[block][extension coordinate][subring coefficient]` and contains exactly
/// `k s` coordinates per block.
///
/// # Errors
///
/// Returns an error for an inconsistent extension degree, malformed source or
/// weight lengths, overflow, or a bounded reference allocation failure.
#[cfg(any(test, feature = "test-support"))]
pub fn coefficient_packing_partials<F, E>(
    geometry: SubringCoefficientPackingGeometry,
    num_live_positions: usize,
    num_positions_per_block: usize,
    a_ring_coefficients: &[F],
    position_weights: &[E],
    packing_weights: &[E],
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    validate_extension_degree::<F, E>(geometry)?;
    if num_live_positions == 0 || !num_positions_per_block.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "subring packing requires live positions and a nonzero power-of-two position domain"
                .into(),
        ));
    }
    require_len(
        "position-weight vector",
        position_weights.len(),
        num_positions_per_block,
    )?;
    let expected_source = checked_product(
        "source coefficient",
        &[num_live_positions, geometry.a_ring_dimension()],
    )?;
    require_len(
        "source coefficient vector",
        a_ring_coefficients.len(),
        expected_source,
    )?;
    require_len(
        "low-coefficient weight vector",
        packing_weights.len(),
        geometry.subring_embedding_stride(),
    )?;

    let num_blocks = num_live_positions.div_ceil(num_positions_per_block);
    let output_len = checked_product(
        "partial opening",
        &[num_blocks, geometry.partial_base_field_width()],
    )?;
    let mut output = zero_vec::<F>("partial opening", output_len)?;
    let mut source_offset = 0usize;
    for block_index in 0..num_blocks {
        let first_position = block_index
            .checked_mul(num_positions_per_block)
            .ok_or_else(|| {
                AkitaError::InvalidInput("subring packing block position overflow".into())
            })?;
        let remaining_positions =
            num_live_positions
                .checked_sub(first_position)
                .ok_or_else(|| {
                    AkitaError::InvalidInput("subring packing block position underflow".into())
                })?;
        let num_positions = remaining_positions.min(num_positions_per_block);
        let mut packed = zero_vec::<E>("packed block", geometry.challenge_subring_dimension())?;
        for position_index in 0..num_positions {
            let source_end = source_offset
                .checked_add(geometry.a_ring_dimension())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("subring packing source offset overflow".into())
                })?;
            let source_ring = a_ring_coefficients
                .get(source_offset..source_end)
                .ok_or_else(|| {
                    AkitaError::InvalidInput("subring packing source ring is out of bounds".into())
                })?;
            let position_weight = *position_weights.get(position_index).ok_or_else(|| {
                AkitaError::InvalidInput("subring packing position weight is missing".into())
            })?;
            let position_packed =
                coefficient_packing_map::<F, E>(geometry, source_ring, packing_weights)?;
            for (accumulator, coefficient) in packed.iter_mut().zip(position_packed) {
                *accumulator += position_weight * coefficient;
            }
            source_offset = source_end;
        }

        let block_offset = block_index
            .checked_mul(geometry.partial_base_field_width())
            .ok_or_else(|| {
                AkitaError::InvalidInput("subring packing block output offset overflow".into())
            })?;
        let block_end = block_offset
            .checked_add(geometry.partial_base_field_width())
            .ok_or_else(|| {
                AkitaError::InvalidInput("subring packing block output extent overflow".into())
            })?;
        let block_output = output.get_mut(block_offset..block_end).ok_or_else(|| {
            AkitaError::InvalidInput("subring packing block output is out of bounds".into())
        })?;
        for (subring_coefficient_index, coefficient) in packed.iter().enumerate() {
            let coordinates = coefficient.to_base_vec();
            require_len(
                "extension coefficient vector",
                coordinates.len(),
                geometry.extension_degree(),
            )?;
            for (extension_coordinate, coordinate) in coordinates.into_iter().enumerate() {
                let output_index = geometry.partial_base_field_coordinate_index(
                    extension_coordinate,
                    subring_coefficient_index,
                )?;
                *block_output.get_mut(output_index).ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "subring packing partial coordinate is out of bounds".into(),
                    )
                })? = coordinate;
            }
        }
    }
    Ok(output)
}

/// Reconstruct one group's scalar opening from canonical partial coordinates.
///
/// Each entry in `partial_coordinates_by_claim` uses
/// `[block][extension coordinate][subring coefficient]`. The three weight
/// arrays apply the claim batch, block point, and subring-tail point.
///
/// # Errors
///
/// Returns an error for zero claim/block counts, an inconsistent extension
/// degree, malformed coordinate or weight lengths, or overflow.
pub fn coefficient_packing_scalar_opening<F, E>(
    geometry: SubringCoefficientPackingGeometry,
    num_blocks: usize,
    partial_coordinates_by_claim: &[impl AsRef<[F]>],
    claim_weights: &[E],
    block_weights: &[E],
    tail_weights: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    validate_extension_degree::<F, E>(geometry)?;
    let num_claims = partial_coordinates_by_claim.len();
    if num_claims == 0 || num_blocks == 0 {
        return Err(AkitaError::InvalidInput(
            "subring packing scalar opening requires nonzero claims and blocks".into(),
        ));
    }
    require_len("claim-weight vector", claim_weights.len(), num_claims)?;
    require_len("block-weight vector", block_weights.len(), num_blocks)?;
    require_len(
        "tail-weight vector",
        tail_weights.len(),
        geometry.challenge_subring_dimension(),
    )?;
    let expected_claim_partials = checked_product(
        "scalar opening partial",
        &[num_blocks, geometry.partial_base_field_width()],
    )?;
    for partial_coordinates in partial_coordinates_by_claim {
        require_len(
            "scalar opening claim partial vector",
            partial_coordinates.as_ref().len(),
            expected_claim_partials,
        )?;
    }

    let mut opening = E::zero();
    let mut coordinates = zero_vec::<F>("extension coefficient", geometry.extension_degree())?;
    for (claim_index, (partial_coordinates, &claim_weight)) in partial_coordinates_by_claim
        .iter()
        .zip(claim_weights)
        .enumerate()
    {
        let partial_coordinates = partial_coordinates.as_ref();
        for (block_index, &block_weight) in block_weights.iter().enumerate() {
            let partial_offset = block_index
                .checked_mul(geometry.partial_base_field_width())
                .ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "subring packing scalar opening offset overflow".into(),
                    )
                })?;
            for (subring_coefficient_index, &tail_weight) in tail_weights.iter().enumerate() {
                for (extension_coordinate, coordinate) in coordinates.iter_mut().enumerate() {
                    let local_index = geometry.partial_base_field_coordinate_index(
                        extension_coordinate,
                        subring_coefficient_index,
                    )?;
                    let source_index =
                        partial_offset.checked_add(local_index).ok_or_else(|| {
                            AkitaError::InvalidInput(
                                "subring packing scalar opening source index overflow".into(),
                            )
                        })?;
                    *coordinate = *partial_coordinates.get(source_index).ok_or_else(|| {
                        AkitaError::InvalidInput(
                            format!(
                                "subring packing scalar opening source for claim {claim_index} is out of bounds"
                            ),
                        )
                    })?;
                }
                let coefficient = E::from_base_slice(&coordinates);
                opening += claim_weight * block_weight * tail_weight * coefficient;
            }
        }
    }
    Ok(opening)
}

/// Embed challenge-subring coefficients into the A ring through
/// `Y -> X^(k h)`.
///
/// # Errors
///
/// Returns an error when the challenge length is not `s` or the bounded
/// reference allocation fails.
#[cfg(test)]
pub(super) fn embed_subring_challenge_in_a_ring<F: FieldCore + FromPrimitiveInt>(
    geometry: SubringCoefficientPackingGeometry,
    challenge: &SparseChallenge,
) -> Result<Vec<F>, AkitaError> {
    challenge.validate_dyn(geometry.challenge_subring_dimension())?;
    let mut embedded = zero_vec::<F>("embedded challenge", geometry.a_ring_dimension())?;
    for (&subring_coefficient_index, &coefficient) in
        challenge.positions.iter().zip(&challenge.coeffs)
    {
        let index = geometry.a_ring_coefficient_index(0, subring_coefficient_index as usize)?;
        *embedded.get_mut(index).ok_or_else(|| {
            AkitaError::InvalidInput("embedded challenge index is out of bounds".into())
        })? = F::from_i64(i64::from(coefficient));
    }
    Ok(embedded)
}

#[cfg(test)]
fn negacyclic_product_reference<T: FieldCore>(
    dimension: usize,
    lhs: &[T],
    rhs: &[T],
) -> Result<Vec<T>, AkitaError> {
    require_len("left ring operand", lhs.len(), dimension)?;
    require_len("right ring operand", rhs.len(), dimension)?;
    let mut product = zero_vec::<T>("negacyclic product", dimension)?;
    for (lhs_index, &lhs_coefficient) in lhs.iter().enumerate() {
        if lhs_coefficient == T::zero() {
            continue;
        }
        for (rhs_index, &rhs_coefficient) in rhs.iter().enumerate() {
            if rhs_coefficient == T::zero() {
                continue;
            }
            let ordinary_index = lhs_index.checked_add(rhs_index).ok_or_else(|| {
                AkitaError::InvalidInput("subring packing product index overflow".into())
            })?;
            let (index, wraps) = if ordinary_index >= dimension {
                (ordinary_index - dimension, true)
            } else {
                (ordinary_index, false)
            };
            let destination = product.get_mut(index).ok_or_else(|| {
                AkitaError::InvalidInput("subring packing product index is out of bounds".into())
            })?;
            let term = lhs_coefficient * rhs_coefficient;
            if wraps {
                *destination -= term;
            } else {
                *destination += term;
            }
        }
    }
    Ok(product)
}

/// Multiply an A-ring vector by a challenge embedded through `Y -> X^(k h)`.
///
/// # Errors
///
/// Returns an error when either input length disagrees with `geometry` or a
/// bounded reference allocation fails.
#[cfg(test)]
pub(super) fn multiply_a_ring_by_subring_challenge<F: FieldCore + FromPrimitiveInt>(
    geometry: SubringCoefficientPackingGeometry,
    challenge: &SparseChallenge,
    a_ring_coefficients: &[F],
) -> Result<Vec<F>, AkitaError> {
    require_len(
        "A-ring coefficient vector",
        a_ring_coefficients.len(),
        geometry.a_ring_dimension(),
    )?;
    let embedded = embed_subring_challenge_in_a_ring(geometry, challenge)?;
    negacyclic_product_reference(geometry.a_ring_dimension(), &embedded, a_ring_coefficients)
}

/// Canonical product data for one logical coefficient packing relation row.
///
/// Both slices use physical layout
/// `[extension coordinate][subring coefficient]`. Their length is `k s`, but
/// the polynomial modulus used to derive them is `Y^s + 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoefficientPackingFoldProduct<F: FieldCore> {
    geometry: SubringCoefficientPackingGeometry,
    reduced_base_field_coordinates: Vec<F>,
    quotient_high_half_base_field_coordinates: Vec<F>,
}

impl<F: FieldCore> CoefficientPackingFoldProduct<F> {
    /// Geometry under which both paired outputs were constructed.
    #[must_use]
    pub fn geometry(&self) -> SubringCoefficientPackingGeometry {
        self.geometry
    }

    /// Negacyclic reduction of `sum_i c_i(Y) e_i(Y)` modulo `Y^s + 1`.
    #[must_use]
    pub fn reduced_base_field_coordinates(&self) -> &[F] {
        &self.reduced_base_field_coordinates
    }

    /// Positive ordinary-product high half `Q_pack`.
    #[must_use]
    pub fn quotient_high_half_base_field_coordinates(&self) -> &[F] {
        &self.quotient_high_half_base_field_coordinates
    }

    /// Consume the paired reduction and positive high half without separating
    /// their construction authority.
    #[must_use]
    pub fn into_geometry_and_base_field_coordinates(
        self,
    ) -> (SubringCoefficientPackingGeometry, Vec<F>, Vec<F>) {
        (
            self.geometry,
            self.reduced_base_field_coordinates,
            self.quotient_high_half_base_field_coordinates,
        )
    }
}

fn accumulate_small_signed_product<F: FieldCore + FromPrimitiveInt>(
    destination: &mut F,
    value: F,
    coefficient: i8,
) {
    match coefficient {
        1 => *destination += value,
        -1 => *destination -= value,
        2 => {
            *destination += value;
            *destination += value;
        }
        -2 => {
            *destination -= value;
            *destination -= value;
        }
        _ => *destination += value * F::from_i64(i64::from(coefficient)),
    }
}

/// Fold coefficient packing partials by sparse subring challenges.
///
/// `partial_coordinates` uses
/// `[challenge][extension coordinate][subring coefficient]`. The returned
/// reduction and quotient each use
/// `[extension coordinate][subring coefficient]` and contain exactly `k s`
/// base-field coordinates. The two outputs are accumulated together so their
/// negacyclic wrap sign and positive high-half convention cannot drift.
///
/// # Errors
///
/// Returns an error when a challenge is malformed at dimension `s`, the
/// partial length disagrees with the challenge count and geometry, or a
/// bounded reference allocation fails.
pub fn fold_coefficient_packing_partials<F: FieldCore + FromPrimitiveInt>(
    geometry: SubringCoefficientPackingGeometry,
    challenges: &[SparseChallenge],
    partial_coordinates: &[F],
) -> Result<CoefficientPackingFoldProduct<F>, AkitaError> {
    for challenge in challenges {
        challenge.validate_dyn(geometry.challenge_subring_dimension())?;
    }
    let partial_len = checked_product(
        "folded partial",
        &[challenges.len(), geometry.partial_base_field_width()],
    )?;
    require_len(
        "folded partial vector",
        partial_coordinates.len(),
        partial_len,
    )?;

    let mut reduced = zero_vec::<F>(
        "reduced packing product",
        geometry.partial_base_field_width(),
    )?;
    let mut quotient = zero_vec::<F>("packing quotient", geometry.partial_base_field_width())?;
    let s = geometry.challenge_subring_dimension();
    for (term_index, challenge) in challenges.iter().enumerate() {
        let partial_offset = term_index
            .checked_mul(geometry.partial_base_field_width())
            .ok_or_else(|| {
                AkitaError::InvalidInput("subring packing fold offset overflow".into())
            })?;
        for extension_coordinate in 0..geometry.extension_degree() {
            let coordinate_offset = extension_coordinate.checked_mul(s).ok_or_else(|| {
                AkitaError::InvalidInput("subring packing plane offset overflow".into())
            })?;
            let plane_offset = partial_offset
                .checked_add(coordinate_offset)
                .ok_or_else(|| {
                    AkitaError::InvalidInput("subring packing source offset overflow".into())
                })?;
            let plane_end = plane_offset.checked_add(s).ok_or_else(|| {
                AkitaError::InvalidInput("subring packing source extent overflow".into())
            })?;
            let partial = partial_coordinates
                .get(plane_offset..plane_end)
                .ok_or_else(|| {
                    AkitaError::InvalidInput("subring packing source plane is out of bounds".into())
                })?;
            let coordinate_end = coordinate_offset.checked_add(s).ok_or_else(|| {
                AkitaError::InvalidInput("subring packing coordinate extent overflow".into())
            })?;
            let reduced_plane = reduced
                .get_mut(coordinate_offset..coordinate_end)
                .ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "subring packing reduced plane is out of bounds".into(),
                    )
                })?;
            let quotient_plane = quotient
                .get_mut(coordinate_offset..coordinate_end)
                .ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "subring packing quotient plane is out of bounds".into(),
                    )
                })?;
            for (&challenge_position, &challenge_coefficient) in
                challenge.positions.iter().zip(&challenge.coeffs)
            {
                let challenge_index = challenge_position as usize;
                for (partial_index, &partial_coefficient) in partial.iter().enumerate() {
                    let ordinary_index =
                        challenge_index.checked_add(partial_index).ok_or_else(|| {
                            AkitaError::InvalidInput(
                                "subring packing product index overflow".into(),
                            )
                        })?;
                    let (output_index, wraps) = if ordinary_index >= s {
                        (ordinary_index - s, true)
                    } else {
                        (ordinary_index, false)
                    };
                    let reduced_destination =
                        reduced_plane.get_mut(output_index).ok_or_else(|| {
                            AkitaError::InvalidInput(
                                "subring packing reduced index is out of bounds".into(),
                            )
                        })?;
                    if wraps {
                        accumulate_small_signed_product(
                            reduced_destination,
                            -partial_coefficient,
                            challenge_coefficient,
                        );
                        let quotient_destination =
                            quotient_plane.get_mut(output_index).ok_or_else(|| {
                                AkitaError::InvalidInput(
                                    "subring packing quotient index is out of bounds".into(),
                                )
                            })?;
                        accumulate_small_signed_product(
                            quotient_destination,
                            partial_coefficient,
                            challenge_coefficient,
                        );
                    } else {
                        accumulate_small_signed_product(
                            reduced_destination,
                            partial_coefficient,
                            challenge_coefficient,
                        );
                    }
                }
            }
        }
    }
    Ok(CoefficientPackingFoldProduct {
        geometry,
        reduced_base_field_coordinates: reduced,
        quotient_high_half_base_field_coordinates: quotient,
    })
}
