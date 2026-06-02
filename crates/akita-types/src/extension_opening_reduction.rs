//! Extension-opening-reduction tensor and output helpers.
//!
//! These helpers are pure tensor algebra shared by prover and verifier code.
//! The concrete EOR prover instance and its witness-bearing state live in
//! `akita-prover`.

use akita_algebra::poly::multilinear_eval;
use akita_algebra::EqPolynomial;
use akita_field::{AkitaError, ExtField, FieldCore};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Degree bound for one witness factor times one transparent reduction factor.
pub const EXTENSION_OPENING_REDUCTION_DEGREE: usize = 2;

/// Tensor-algebra data for one extension-opening reduction instance.
///
/// The `column_partials` are the column view of
/// `S = phi1(g)(phi0(r_tail))` in `E tensor_F E`:
/// `column_partials[v] = f(v, r_tail)`. The `row_partials` are the
/// deterministic tensor transpose used for row batching in the sumcheck.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionOpeningTensorPartials<E: FieldCore> {
    /// Column-view tensor partials `S_v = f(v, r_tail)`.
    pub column_partials: Vec<E>,
    /// Row-view tensor partials after basis transpose.
    pub row_partials: Vec<E>,
}

/// Full-split tensor opening shape for an extension `E/F`.
///
/// # Errors
///
/// Returns an error if `[E:F]` is not a power of two.
pub fn tensor_opening_split<F, E>() -> Result<(usize, usize), AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let width = E::EXT_DEGREE;
    if width == 0 || !width.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "extension-opening tensor reduction requires power-of-two extension degree, got {width}"
        )));
    }
    Ok((width.trailing_zeros() as usize, width))
}

/// Pack a base-field witness table into the extension-valued tail table
/// `g(w) = sum_v f(v, w) * beta_v`.
///
/// The first `log2([E:F])` variables are the packed head variables and use the
/// repository's little-endian Lagrange table order, so entry
/// `tail * [E:F] + head` is `f(head, tail)`.
///
/// # Errors
///
/// Returns an error if the extension degree is unsupported or the table shape
/// does not match `original_num_vars`.
pub fn tensor_packed_witness_evals<F, E>(
    original_num_vars: usize,
    base_evals: &[F],
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    if split_bits > original_num_vars {
        return Err(AkitaError::InvalidInput(
            "extension-opening tensor split exceeds polynomial arity".to_string(),
        ));
    }
    let expected_len = 1usize
        .checked_shl(original_num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidInput("witness table length overflow".to_string()))?;
    if base_evals.len() != expected_len {
        return Err(AkitaError::InvalidSize {
            expected: expected_len,
            actual: base_evals.len(),
        });
    }

    let tail_len = 1usize << (original_num_vars - split_bits);
    // Pure order-preserving map; the indexed parallel collect yields the same
    // ordering as the serial loop, so the packed table is byte-identical.
    #[cfg(feature = "parallel")]
    let packed: Vec<E> = {
        const PARALLEL_PACK_THRESHOLD: usize = 1 << 14;
        if tail_len >= PARALLEL_PACK_THRESHOLD {
            base_evals[..tail_len * width]
                .par_chunks_exact(width)
                .map(E::from_base_slice)
                .collect()
        } else {
            (0..tail_len)
                .map(|tail| {
                    let base = tail * width;
                    E::from_base_slice(&base_evals[base..base + width])
                })
                .collect()
        }
    };
    #[cfg(not(feature = "parallel"))]
    let packed: Vec<E> = (0..tail_len)
        .map(|tail| {
            let base = tail * width;
            E::from_base_slice(&base_evals[base..base + width])
        })
        .collect();
    Ok(packed)
}

/// Compute the column-view tensor partials `f(v, r_tail)`.
///
/// # Errors
///
/// Returns an error if the point/table shape is malformed.
pub fn tensor_column_partials_from_base_evals<F, E>(
    original_num_vars: usize,
    base_evals: &[F],
    logical_point: &[E],
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    if split_bits > original_num_vars {
        return Err(AkitaError::InvalidInput(
            "extension-opening tensor split exceeds polynomial arity".to_string(),
        ));
    }
    if logical_point.len() != original_num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: original_num_vars,
            actual: logical_point.len(),
        });
    }
    let expected_len = 1usize
        .checked_shl(original_num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidInput("witness table length overflow".to_string()))?;
    if base_evals.len() != expected_len {
        return Err(AkitaError::InvalidSize {
            expected: expected_len,
            actual: base_evals.len(),
        });
    }

    let tail_point = &logical_point[split_bits..];
    let tail_len = 1usize << tail_point.len();
    let tail_eq = EqPolynomial::evals(tail_point)?;

    // `partials[head] = sum_tail eq(tail_point, tail) * base_evals[tail*width + head]`.
    // Field addition is exact and associative, so a parallel chunk/reduce yields
    // the identical canonical value as the serial fold below.
    #[cfg(feature = "parallel")]
    let partials = {
        const PARALLEL_PARTIALS_THRESHOLD: usize = 1 << 14;
        if tail_len >= PARALLEL_PARTIALS_THRESHOLD {
            tail_eq[..tail_len]
                .par_iter()
                .zip(base_evals[..tail_len * width].par_chunks_exact(width))
                .fold(
                    || vec![E::zero(); width],
                    |mut acc, (&weight, chunk)| {
                        for (slot, &coeff) in acc.iter_mut().zip(chunk.iter()) {
                            *slot += weight.mul_base(coeff);
                        }
                        acc
                    },
                )
                .reduce(
                    || vec![E::zero(); width],
                    |mut acc, other| {
                        for (slot, value) in acc.iter_mut().zip(other) {
                            *slot += value;
                        }
                        acc
                    },
                )
        } else {
            tensor_column_partials_fold_serial::<F, E>(&tail_eq, base_evals, tail_len, width)
        }
    };
    #[cfg(not(feature = "parallel"))]
    let partials =
        tensor_column_partials_fold_serial::<F, E>(&tail_eq, base_evals, tail_len, width);
    Ok(partials)
}

/// Serial fold helper shared by the parallel and non-parallel
/// [`tensor_column_partials_from_base_evals`] paths.
#[inline]
fn tensor_column_partials_fold_serial<F, E>(
    tail_eq: &[E],
    base_evals: &[F],
    tail_len: usize,
    width: usize,
) -> Vec<E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let mut partials = vec![E::zero(); width];
    for (tail, &weight) in tail_eq.iter().enumerate().take(tail_len) {
        let base = tail * width;
        for head in 0..width {
            partials[head] += weight.mul_base(base_evals[base + head]);
        }
    }
    partials
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
    F: FieldCore,
    E: ExtField<F>,
{
    let (_split_bits, width) = tensor_opening_split::<F, E>()?;
    if column_partials.len() != width {
        return Err(AkitaError::InvalidSize {
            expected: width,
            actual: column_partials.len(),
        });
    }

    let mut rows = vec![vec![F::zero(); width]; width];
    for (column, partial) in column_partials.iter().enumerate() {
        let coords = partial.to_base_vec();
        if coords.len() != width {
            return Err(AkitaError::InvalidSize {
                expected: width,
                actual: coords.len(),
            });
        }
        for (row, coord) in coords.into_iter().enumerate() {
            rows[row][column] = coord;
        }
    }
    Ok(rows
        .into_iter()
        .map(|coords| E::from_base_slice(&coords))
        .collect())
}

/// Compute and transpose tensor partials from a base-field witness table.
///
/// # Errors
///
/// Returns an error if the point/table shape is malformed.
pub fn tensor_partials_from_base_evals<F, E>(
    original_num_vars: usize,
    base_evals: &[F],
    logical_point: &[E],
) -> Result<ExtensionOpeningTensorPartials<E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let column_partials = tensor_column_partials_from_base_evals::<F, E>(
        original_num_vars,
        base_evals,
        logical_point,
    )?;
    let row_partials = tensor_row_partials_from_columns::<F, E>(&column_partials)?;
    Ok(ExtensionOpeningTensorPartials {
        column_partials,
        row_partials,
    })
}

/// Recombine column-view tensor partials into the logical opening claim.
///
/// # Errors
///
/// Returns an error if the logical point or partial vector is malformed.
pub fn tensor_logical_claim_from_partials<F, E>(
    logical_point: &[E],
    column_partials: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore,
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
    let head_weights = EqPolynomial::evals(&logical_point[..split_bits])?;
    Ok(head_weights
        .into_iter()
        .zip(column_partials.iter().copied())
        .fold(E::zero(), |acc, (weight, partial)| acc + weight * partial))
}

/// Check column-view tensor partials against a logical opening claim.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] if the recomposed claim differs.
pub fn check_tensor_extension_opening_claim<F, E>(
    logical_point: &[E],
    logical_claim: E,
    column_partials: &[E],
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let expected = tensor_logical_claim_from_partials::<F, E>(logical_point, column_partials)?;
    if expected != logical_claim {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
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
    F: FieldCore,
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

#[doc(hidden)]
pub fn project_tensor_factor_value<F, E>(
    value: E,
    eta_weights: &[E],
    width: usize,
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let coords = value.to_base_vec();
    if coords.len() != width {
        return Err(AkitaError::InvalidSize {
            expected: width,
            actual: coords.len(),
        });
    }
    Ok(coords
        .into_iter()
        .zip(eta_weights.iter().copied())
        .fold(E::zero(), |acc, (coord, weight)| {
            acc + weight.mul_base(coord)
        }))
}

/// Dense evaluations of the FRI-Binius tensor equality factor
/// `A_eta(w) = sum_u eq(u, eta) * coord_u(eq(r_tail, w))`.
///
/// # Errors
///
/// Returns an error if `eta` does not match `log2([E:F])` or if an equality
/// value decomposes into the wrong number of base coordinates.
pub fn tensor_equality_factor_evals<F, E>(tail_point: &[E], eta: &[E]) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    if eta.len() != split_bits {
        return Err(AkitaError::InvalidSize {
            expected: split_bits,
            actual: eta.len(),
        });
    }
    let eta_weights = EqPolynomial::evals(eta)?;
    let mut out = EqPolynomial::evals(tail_point)?;
    let project = |value: &mut E| {
        *value = project_tensor_factor_value::<F, E>(*value, &eta_weights, width)?;
        Ok::<(), AkitaError>(())
    };
    #[cfg(feature = "parallel")]
    out.par_iter_mut().try_for_each(project)?;
    #[cfg(not(feature = "parallel"))]
    out.iter_mut().try_for_each(project)?;
    Ok(out)
}

/// Evaluate the transparent tensor equality factor `A_eta` at one point.
///
/// This is the verifier-side counterpart to [`tensor_equality_factor_evals`].
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
    F: FieldCore,
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

/// Verifier replay output for an extension-opening reduction sumcheck.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionOpeningReductionRoundResult<E: FieldCore> {
    /// Final sumcheck claim after all verifier challenges have been bound.
    pub final_claim: E,
    /// Sumcheck challenge point `rho`.
    pub challenges: Vec<E>,
}

/// One row-local term in a transparent extension-opening reduction factor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionOpeningFactorTerm<E: FieldCore> {
    /// Opening point for the transformed committed witness.
    pub point: Vec<E>,
    /// Row-local batching coefficient multiplying this equality factor.
    pub coeff: E,
}

impl<E: FieldCore> ExtensionOpeningFactorTerm<E> {
    /// Construct a term `coeff * eq(point, x)`.
    pub fn new(point: Vec<E>, coeff: E) -> Self {
        Self { point, coeff }
    }
}

/// Transparent reduction factor `A(x) = sum_i coeff_i * eq(point_i, x)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionOpeningReductionFactor<E: FieldCore> {
    num_vars: usize,
    terms: Vec<ExtensionOpeningFactorTerm<E>>,
}

impl<E: FieldCore> ExtensionOpeningReductionFactor<E> {
    /// Construct a singleton factor `eq(point, x)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `point.len()` is too large for an evaluation table.
    pub fn singleton(point: Vec<E>) -> Result<Self, AkitaError> {
        Self::from_terms(vec![ExtensionOpeningFactorTerm::new(point, E::one())])
    }

    /// Construct a row-local linear combination of equality factors.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no terms, if term arities differ, or if
    /// `2^num_vars` overflows `usize`.
    pub fn from_terms(terms: Vec<ExtensionOpeningFactorTerm<E>>) -> Result<Self, AkitaError> {
        let first = terms.first().ok_or_else(|| {
            AkitaError::InvalidInput(
                "extension-opening reduction factor requires at least one term".to_string(),
            )
        })?;
        let num_vars = first.point.len();
        if num_vars >= usize::BITS as usize {
            return Err(AkitaError::InvalidInput(format!(
                "extension-opening reduction factor has too many variables: {num_vars}"
            )));
        }
        for term in &terms {
            if term.point.len() != num_vars {
                return Err(AkitaError::InvalidSize {
                    expected: num_vars,
                    actual: term.point.len(),
                });
            }
        }
        Ok(Self { num_vars, terms })
    }

    /// Number of Boolean variables in this factor.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Row-local equality terms.
    pub fn terms(&self) -> &[ExtensionOpeningFactorTerm<E>] {
        &self.terms
    }

    /// Compute the transparent factor evaluation table.
    ///
    /// # Errors
    ///
    /// Returns an error if the factor point arity overflows the equality table.
    pub fn evals(&self) -> Result<Vec<E>, AkitaError> {
        let mut out = vec![E::zero(); 1usize << self.num_vars];
        for term in &self.terms {
            let term_evals = EqPolynomial::evals_with_scaling(&term.point, Some(term.coeff))?;
            for (dst, value) in out.iter_mut().zip(term_evals) {
                *dst += value;
            }
        }
        Ok(out)
    }

    /// Evaluate the transparent factor at an arbitrary point.
    ///
    /// # Errors
    ///
    /// Returns an error if `point.len()` does not match the factor arity.
    pub fn evaluate(&self, point: &[E]) -> Result<E, AkitaError> {
        if point.len() != self.num_vars {
            return Err(AkitaError::InvalidSize {
                expected: self.num_vars,
                actual: point.len(),
            });
        }
        let mut acc = E::zero();
        for term in &self.terms {
            acc += term.coeff * EqPolynomial::mle(&term.point, point)?;
        }
        Ok(acc)
    }

    /// Compute the reduction claim induced by this factor and witness table.
    ///
    /// # Errors
    ///
    /// Returns an error if `witness_evals.len() != 2^num_vars`.
    pub fn claim_for_witness(&self, witness_evals: &[E]) -> Result<E, AkitaError> {
        let expected = 1usize << self.num_vars;
        if witness_evals.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: witness_evals.len(),
            });
        }
        extension_opening_reduction_claim(witness_evals, &self.evals()?)
    }
}

/// Compute `sum_x witness(x) * factor(x)` from Boolean-hypercube evaluation
/// tables.
///
/// # Errors
///
/// Returns an error if the tables do not have the same nonzero power-of-two
/// length.
pub fn extension_opening_reduction_claim<E: FieldCore>(
    witness_evals: &[E],
    factor_evals: &[E],
) -> Result<E, AkitaError> {
    validate_reduction_tables(witness_evals, factor_evals)?;
    // `sum_x witness(x) * factor(x)`. Field addition is exact and associative,
    // so the parallel reduction yields the identical canonical value as the
    // serial fold.
    #[cfg(feature = "parallel")]
    {
        const PARALLEL_CLAIM_THRESHOLD: usize = 1 << 14;
        if witness_evals.len() >= PARALLEL_CLAIM_THRESHOLD {
            return Ok(witness_evals
                .par_iter()
                .zip(factor_evals.par_iter())
                .map(|(&w, &a)| w * a)
                .reduce(E::zero, |acc, term| acc + term));
        }
    }
    Ok(witness_evals
        .iter()
        .zip(factor_evals.iter())
        .fold(E::zero(), |acc, (&w, &a)| acc + w * a))
}

/// Evaluate the final reduction oracle `witness(point) * factor(point)`.
///
/// # Errors
///
/// Returns an error if either table length is malformed or inconsistent with
/// `point.len()`.
pub fn extension_opening_reduction_eval_at_point<E: FieldCore>(
    witness_evals: &[E],
    factor_evals: &[E],
    point: &[E],
) -> Result<E, AkitaError> {
    validate_reduction_tables(witness_evals, factor_evals)?;
    let witness_eval = multilinear_eval(witness_evals, point)?;
    let factor_eval = multilinear_eval(factor_evals, point)?;
    Ok(witness_eval * factor_eval)
}

/// Check the final extension-opening reduction equality.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] if the final sumcheck claim does not
/// match the product of the ordinary witness opening and transparent factor
/// evaluation at the sumcheck challenge point.
pub fn check_extension_opening_reduction_output<E: FieldCore>(
    final_claim: E,
    witness_eval: E,
    factor_eval: E,
) -> Result<(), AkitaError> {
    if final_claim != witness_eval * factor_eval {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

pub fn validate_reduction_tables<E: FieldCore>(
    witness_evals: &[E],
    factor_evals: &[E],
) -> Result<(), AkitaError> {
    if witness_evals.len() != factor_evals.len() {
        return Err(AkitaError::InvalidSize {
            expected: witness_evals.len(),
            actual: factor_evals.len(),
        });
    }
    num_rounds_from_table_len(witness_evals.len()).map(|_| ())
}

pub fn checked_table_len(num_vars: usize) -> Result<usize, AkitaError> {
    if num_vars >= usize::BITS as usize {
        return Err(AkitaError::InvalidInput(format!(
            "extension-opening reduction table has too many variables: {num_vars}"
        )));
    }
    Ok(1usize << num_vars)
}

pub fn num_rounds_from_table_len(len: usize) -> Result<usize, AkitaError> {
    if len == 0 || !len.is_power_of_two() {
        return Err(AkitaError::InvalidSize {
            expected: len.max(1).next_power_of_two(),
            actual: len,
        });
    }
    Ok(len.trailing_zeros() as usize)
}
