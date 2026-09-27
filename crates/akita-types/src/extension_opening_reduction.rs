//! Extension-opening-reduction tensor and output helpers.
//!
//! These helpers are pure tensor algebra shared by prover and verifier code.
//! The concrete EOR prover instance lives in `akita-prover`; the prover's
//! witness and factor tables live in `akita-cpu-backend`.

use akita_algebra::EqPolynomial;
use akita_error::AkitaError;
use jolt_field::{ExtField, Field};

/// Degree bound for one witness factor times one transparent reduction factor.
pub const EXTENSION_OPENING_REDUCTION_DEGREE: usize = 2;

/// Full-split tensor opening shape for an extension `E/F`.
///
/// # Errors
///
/// Returns an error if `[E:F]` is not a power of two.
pub fn tensor_opening_split<F, E>() -> Result<(usize, usize), AkitaError>
where
    F: Field,
    E: ExtField<F>,
{
    let width = E::DEGREE;
    if width == 0 || !width.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "extension-opening tensor reduction requires power-of-two extension degree, got {width}"
        )));
    }
    Ok((width.trailing_zeros() as usize, width))
}

/// Transpose the tensor partial object from column view to row view.
///
/// Input `column_partials[v]` is decomposed in the fixed `F`-basis of `E`.
/// The returned row `u` is `sum_v coeff_{u,v} beta_v`.
///
/// # Errors
///
/// Returns an error if the partial count or any coordinate vector is malformed.
pub fn tensor_row_partials_from_columns<F, E>(column_partials: &[E]) -> Result<Vec<E>, AkitaError>
where
    F: Field,
    E: ExtField<F>,
{
    let (_split_bits, width) = tensor_opening_split::<F, E>()?;
    if column_partials.len() != width {
        return Err(AkitaError::InvalidSize {
            expected: width,
            actual: column_partials.len(),
        });
    }

    Ok((0..width)
        .map(|row| E::from_base_fn(|column| column_partials[column].base_coefficient(row)))
        .collect())
}

/// Derive a logical opening claim from column-view tensor partials.
///
/// # Errors
///
/// Returns an error if the logical point or partial vector is malformed.
pub fn derive_tensor_extension_opening_claim_from_partials<F, E>(
    logical_point: &[E],
    column_partials: &[E],
) -> Result<E, AkitaError>
where
    F: Field,
    E: ExtField<F>,
{
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    if logical_point.len() < split_bits {
        return Err(AkitaError::InvalidPointDimension {
            expected: split_bits,
            actual: logical_point.len(),
        });
    }
    if column_partials.len() != width {
        return Err(AkitaError::InvalidSize {
            expected: width,
            actual: column_partials.len(),
        });
    }
    Ok(EqPolynomial::evals(&logical_point[..split_bits])?
        .into_iter()
        .zip(column_partials.iter().copied())
        .fold(E::zero(), |acc, (weight, partial)| acc + weight * partial))
}

/// Compute the row-batched tensor reduction claim
/// `c_eta = sum_u eq(u, eta) * row_partials[u]`.
///
/// # Errors
///
/// Returns an error if the row partial or challenge shape is malformed.
pub fn tensor_reduction_claim_from_rows<F, E>(
    row_partials: &[E],
    eta: &[E],
) -> Result<E, AkitaError>
where
    F: Field,
    E: ExtField<F>,
{
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    if eta.len() != split_bits {
        return Err(AkitaError::InvalidSize {
            expected: split_bits,
            actual: eta.len(),
        });
    }
    if row_partials.len() != width {
        return Err(AkitaError::InvalidSize {
            expected: width,
            actual: row_partials.len(),
        });
    }
    Ok(EqPolynomial::evals(eta)?
        .into_iter()
        .zip(row_partials.iter().copied())
        .fold(E::zero(), |acc, (weight, partial)| acc + weight * partial))
}

/// Evaluate the transparent tensor equality factor `A_eta` at one point.
///
/// This is the verifier-side counterpart to the prover's dense
/// `tensor_equality_factor_evals` table.
/// It intentionally evaluates the transparent factor table, rather than
/// projecting coordinates after evaluating `eq(r_tail, rho)`: coordinate
/// extraction is only `F`-linear, while `rho` is extension-valued.
///
/// # Errors
///
/// Returns an error if the challenge dimensions or basis coordinates are
/// malformed.
pub fn tensor_equality_factor_eval_at_point<F, E>(
    tail_point: &[E],
    eta: &[E],
    rho: &[E],
) -> Result<E, AkitaError>
where
    F: Field,
    E: ExtField<F>,
{
    if rho.len() != tail_point.len() {
        return Err(AkitaError::InvalidPointDimension {
            expected: tail_point.len(),
            actual: rho.len(),
        });
    }
    let (split_bits, _width) = tensor_opening_split::<F, E>()?;
    if eta.len() != split_bits {
        return Err(AkitaError::InvalidSize {
            expected: split_bits,
            actual: eta.len(),
        });
    }

    let eta_weights = EqPolynomial::evals(eta)?;
    let one_coords = E::one().to_base_vec();
    if one_coords.len() != eta_weights.len() {
        return Err(AkitaError::InvalidSize {
            expected: eta_weights.len(),
            actual: one_coords.len(),
        });
    }
    let basis = (0..eta_weights.len())
        .map(|idx| {
            let mut coords = vec![F::zero(); eta_weights.len()];
            coords[idx] = F::one();
            E::from_base_slice(&coords)
        })
        .collect::<Vec<_>>();
    let mut state = one_coords.into_iter().map(E::lift_base).collect::<Vec<_>>();

    for (&tail, &rho_i) in tail_point.iter().zip(rho.iter()) {
        let tail_zero = E::one() - tail;
        let tail_one = tail;
        let rho_zero = E::one() - rho_i;
        let rho_one = rho_i;
        let mut next = vec![E::zero(); eta_weights.len()];
        for (src_idx, &src) in state.iter().enumerate() {
            if src == E::zero() {
                continue;
            }
            let zero_coords = (basis[src_idx] * tail_zero).to_base_vec();
            let one_coords = (basis[src_idx] * tail_one).to_base_vec();
            if zero_coords.len() != eta_weights.len() || one_coords.len() != eta_weights.len() {
                return Err(AkitaError::InvalidSize {
                    expected: eta_weights.len(),
                    actual: zero_coords.len().max(one_coords.len()),
                });
            }
            for dst_idx in 0..eta_weights.len() {
                let transition =
                    rho_zero.mul_base(zero_coords[dst_idx]) + rho_one.mul_base(one_coords[dst_idx]);
                next[dst_idx] += src * transition;
            }
        }
        state = next;
    }

    Ok(state
        .into_iter()
        .zip(eta_weights)
        .fold(E::zero(), |acc, (coord_eval, eta_weight)| {
            acc + coord_eval * eta_weight
        }))
}
