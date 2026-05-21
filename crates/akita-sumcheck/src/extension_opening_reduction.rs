//! Extension-opening reduction sumcheck instances.
//!
//! This module implements the degree-two reduction used to collapse a logical
//! extension-field opening into one ordinary opening of the transformed
//! committed witness. The dense-table instance here proves claims of the form
//! `sum_x witness(x) * factor(x)`, where `factor` is the transparent
//! extension-opening factor. Later small-digit folded-witness code should plug
//! in at this boundary instead of treating the protocol as an arbitrary product
//! gadget.

use crate::drivers::SumcheckInstanceProverExt;
#[cfg(feature = "zk")]
use crate::drivers::ZkSumcheckInstanceProverExt;
use crate::traits::{SumcheckInstanceProver, SumcheckInstanceVerifier};
use crate::types::SumcheckProof;
#[cfg(feature = "zk")]
use crate::types::{FullUniPoly, SumcheckProofMasked};
use akita_algebra::poly::{fold_evals_in_place, multilinear_eval};
use akita_algebra::uni_poly::UniPoly;
use akita_algebra::EqPolynomial;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
#[cfg(feature = "zk")]
use akita_r1cs::{ZkR1csLinearCombination, ZkRelationAccumulator};
use akita_serialization::AkitaSerialize;
use akita_transcript::{labels, Transcript};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Degree bound for one witness factor times one transparent reduction factor.
pub const EXTENSION_OPENING_REDUCTION_DEGREE: usize = 2;

/// Maximum number of sparse low-index rounds to keep in the lazy tensor factor.
///
/// The lazy factor caches one small state per low-bit assignment, avoiding a
/// full dense factor table while the sparse witness still has large support.
pub const SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS: usize = 12;

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
    let mut packed = Vec::with_capacity(tail_len);
    for tail in 0..tail_len {
        let base = tail * width;
        packed.push(E::from_base_slice(&base_evals[base..base + width]));
    }
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
    let mut partials = vec![E::zero(); width];
    for (tail, &weight) in tail_eq.iter().enumerate().take(tail_len) {
        let base = tail * width;
        for head in 0..width {
            partials[head] += weight.mul_base(base_evals[base + head]);
        }
    }
    Ok(partials)
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

fn project_tensor_factor_value<F, E>(
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

#[cfg(feature = "parallel")]
const DENSE_PARALLEL_PAIR_THRESHOLD: usize = 1 << 14;

fn accumulate_dense_round<E: FieldCore>(
    witness_evals: &[E],
    factor_evals: &[E],
    coeff: E,
) -> (E, E, E) {
    let _span = tracing::trace_span!(
        "dense_extension_reduction_accumulate_round",
        table_len = witness_evals.len()
    )
    .entered();
    debug_assert_eq!(witness_evals.len(), factor_evals.len());
    let half = witness_evals.len() / 2;
    if coeff == E::zero() {
        return (E::zero(), E::zero(), E::zero());
    }

    #[cfg(feature = "parallel")]
    {
        if half >= DENSE_PARALLEL_PAIR_THRESHOLD {
            let (constant, linear, quadratic) = (0..half)
                .into_par_iter()
                .fold(
                    || (E::zero(), E::zero(), E::zero()),
                    |(mut constant, mut linear, mut quadratic), i| {
                        let w0 = witness_evals[2 * i];
                        let w1 = witness_evals[2 * i + 1];
                        let a0 = factor_evals[2 * i];
                        let a1 = factor_evals[2 * i + 1];
                        let dw = w1 - w0;
                        let da = a1 - a0;

                        constant += w0 * a0;
                        linear += dw * a0 + w0 * da;
                        quadratic += dw * da;
                        (constant, linear, quadratic)
                    },
                )
                .reduce(
                    || (E::zero(), E::zero(), E::zero()),
                    |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2),
                );
            return (coeff * constant, coeff * linear, coeff * quadratic);
        }
    }

    let mut constant = E::zero();
    let mut linear = E::zero();
    let mut quadratic = E::zero();
    for i in 0..half {
        let w0 = witness_evals[2 * i];
        let w1 = witness_evals[2 * i + 1];
        let a0 = factor_evals[2 * i];
        let a1 = factor_evals[2 * i + 1];
        let dw = w1 - w0;
        let da = a1 - a0;

        constant += w0 * a0;
        linear += dw * a0 + w0 * da;
        quadratic += dw * da;
    }
    (coeff * constant, coeff * linear, coeff * quadratic)
}

fn fold_dense_reduction_tables_in_place<E: FieldCore>(
    witness_evals: &mut Vec<E>,
    factor_evals: &mut Vec<E>,
    r_round: E,
) {
    let _span = tracing::trace_span!(
        "fold_dense_reduction_tables_in_place",
        table_len = witness_evals.len()
    )
    .entered();
    debug_assert_eq!(witness_evals.len(), factor_evals.len());
    debug_assert!(witness_evals.len().is_power_of_two());
    debug_assert!(witness_evals.len() >= 2);
    let half = witness_evals.len() / 2;
    #[cfg(feature = "parallel")]
    {
        if half >= DENSE_PARALLEL_PAIR_THRESHOLD {
            let fold_pair = |pair: &[E]| pair[0] + r_round * (pair[1] - pair[0]);
            let (folded_witness, folded_factor) = rayon::join(
                || witness_evals.par_chunks_exact(2).map(fold_pair).collect(),
                || factor_evals.par_chunks_exact(2).map(fold_pair).collect(),
            );
            *witness_evals = folded_witness;
            *factor_evals = folded_factor;
            return;
        }
    }
    for i in 0..half {
        witness_evals[i] =
            witness_evals[2 * i] + r_round * (witness_evals[2 * i + 1] - witness_evals[2 * i]);
        factor_evals[i] =
            factor_evals[2 * i] + r_round * (factor_evals[2 * i + 1] - factor_evals[2 * i]);
    }
    witness_evals.truncate(half);
    factor_evals.truncate(half);
}

/// Prover state for the degree-two extension-opening reduction sumcheck.
#[derive(Debug, Clone)]
pub struct ExtensionOpeningReductionProver<E: FieldCore> {
    current_witness_evals: Vec<E>,
    current_factor_evals: Vec<E>,
    input_claim: E,
    num_rounds: usize,
}

impl<E: FieldCore> ExtensionOpeningReductionProver<E> {
    /// Construct a prover from transformed-witness and transparent-factor
    /// Boolean-hypercube evaluation tables.
    ///
    /// # Errors
    ///
    /// Returns an error if the tables do not have the same nonzero power-of-two
    /// length.
    pub fn new(witness_evals: Vec<E>, factor_evals: Vec<E>) -> Result<Self, AkitaError> {
        let input_claim = extension_opening_reduction_claim(&witness_evals, &factor_evals)?;
        let num_rounds = num_rounds_from_table_len(witness_evals.len())?;
        Ok(Self {
            current_witness_evals: witness_evals,
            current_factor_evals: factor_evals,
            input_claim,
            num_rounds,
        })
    }

    /// Override the transcript-visible input claim.
    ///
    /// This is used by ZK masked openings where the committed table sum remains
    /// the true claim, but the transcript absorbs the masked claim.
    #[must_use]
    pub fn with_input_claim(mut self, input_claim: E) -> Self {
        self.input_claim = input_claim;
        self
    }

    /// Return the final folded witness and factor evaluations after all
    /// challenges have been ingested.
    pub fn final_witness_and_factor_evals(&self) -> Option<(E, E)> {
        (self.current_witness_evals.len() == 1 && self.current_factor_evals.len() == 1)
            .then(|| (self.current_witness_evals[0], self.current_factor_evals[0]))
    }
}

impl<E: FieldCore> SumcheckInstanceProver<E> for ExtensionOpeningReductionProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        EXTENSION_OPENING_REDUCTION_DEGREE
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, round: usize, previous_claim: E) -> UniPoly<E> {
        debug_assert_eq!(
            self.current_witness_evals.len(),
            1usize << (self.num_rounds - round)
        );
        debug_assert_eq!(
            self.current_factor_evals.len(),
            self.current_witness_evals.len()
        );

        let (constant, linear, quadratic) = accumulate_dense_round(
            &self.current_witness_evals,
            &self.current_factor_evals,
            E::one(),
        );

        let poly = UniPoly::from_coeffs(vec![constant, linear, quadratic]);
        debug_assert_eq!(
            poly.evaluate(&E::zero()) + poly.evaluate(&E::one()),
            previous_claim
        );
        poly
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: E) {
        if self.current_witness_evals.len() > 1 {
            fold_dense_reduction_tables_in_place(
                &mut self.current_witness_evals,
                &mut self.current_factor_evals,
                r_round,
            );
        }
    }
}

/// One term in a batched extension-opening reduction sumcheck.
#[derive(Debug, Clone)]
pub struct BatchedExtensionOpeningReductionTerm<E: FieldCore> {
    current_witness_evals: BatchedExtensionOpeningWitness<E>,
    current_factor: BatchedExtensionOpeningFactor<E>,
    coeff: E,
}

/// Sparse transformed-witness evaluations for extension-opening reduction.
#[derive(Debug, Clone)]
pub struct SparseExtensionOpeningWitness<E: FieldCore> {
    table_len: usize,
    entries: Vec<(usize, E)>,
}

#[cfg(feature = "parallel")]
const SPARSE_PARALLEL_ENTRY_THRESHOLD: usize = 1 << 14;
#[cfg(feature = "parallel")]
const SPARSE_PARALLEL_CHUNKS_PER_THREAD: usize = 4;

impl<E: FieldCore> SparseExtensionOpeningWitness<E> {
    /// Construct a sparse witness table from `(index, value)` entries.
    ///
    /// Duplicate indices are combined, and zero entries are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if `table_len` is not a nonzero power of two or if an
    /// entry index is out of range.
    pub fn new(table_len: usize, mut entries: Vec<(usize, E)>) -> Result<Self, AkitaError> {
        let _span = tracing::debug_span!(
            "SparseExtensionOpeningWitness::new",
            table_len,
            entries_len = entries.len()
        )
        .entered();
        entries.sort_unstable_by_key(|(idx, _)| *idx);
        Self::from_sorted_entries(table_len, entries)
    }

    /// Construct a sparse witness table from entries already sorted by index.
    ///
    /// Duplicate indices are combined, and zero entries are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if `table_len` is not a nonzero power of two, if an
    /// entry index is out of range, or if entries are not sorted by index.
    pub fn from_sorted_entries(
        table_len: usize,
        entries: Vec<(usize, E)>,
    ) -> Result<Self, AkitaError> {
        let _span = tracing::debug_span!(
            "SparseExtensionOpeningWitness::from_sorted_entries",
            table_len,
            entries_len = entries.len()
        )
        .entered();
        if table_len == 0 || !table_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "sparse extension-opening witness length must be a nonzero power of two"
                    .to_string(),
            ));
        }
        let mut combined: Vec<(usize, E)> = Vec::with_capacity(entries.len());
        let mut previous_idx = None;
        for (idx, value) in entries {
            if idx >= table_len {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness index out of range".to_string(),
                ));
            }
            if previous_idx.is_some_and(|previous| idx < previous) {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness sorted constructor received unsorted entries"
                        .to_string(),
                ));
            }
            previous_idx = Some(idx);
            if value == E::zero() {
                continue;
            }
            if let Some((last_idx, last_value)) = combined.last_mut() {
                if *last_idx == idx {
                    *last_value += value;
                    if *last_value == E::zero() {
                        combined.pop();
                    }
                    continue;
                }
            }
            combined.push((idx, value));
        }
        Ok(Self {
            table_len,
            entries: combined,
        })
    }

    /// Construct a sparse witness table from entries already normalized as
    /// strictly sorted, unique, nonzero `(index, value)` pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if `table_len` is not a nonzero power of two, if an
    /// entry index is out of range, if an entry is zero, or if entries are not
    /// strictly sorted by index.
    pub fn from_sorted_unique_entries(
        table_len: usize,
        entries: Vec<(usize, E)>,
    ) -> Result<Self, AkitaError> {
        let _span = tracing::debug_span!(
            "SparseExtensionOpeningWitness::from_sorted_unique_entries",
            table_len,
            entries_len = entries.len()
        )
        .entered();
        if table_len == 0 || !table_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "sparse extension-opening witness length must be a nonzero power of two"
                    .to_string(),
            ));
        }
        let mut previous_idx = None;
        for &(idx, value) in &entries {
            if idx >= table_len {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness index out of range".to_string(),
                ));
            }
            if previous_idx.is_some_and(|previous| idx <= previous) {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness unique constructor received duplicate or unsorted entries"
                        .to_string(),
                ));
            }
            if value == E::zero() {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness unique constructor received a zero entry"
                        .to_string(),
                ));
            }
            previous_idx = Some(idx);
        }
        Ok(Self { table_len, entries })
    }

    /// Dense table length represented by this sparse witness.
    pub fn table_len(&self) -> usize {
        self.table_len
    }

    /// Nonzero sparse entries, sorted by table index.
    pub fn entries(&self) -> &[(usize, E)] {
        &self.entries
    }

    /// Combine sparse witnesses over the same table domain.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no terms or if the sparse witnesses have
    /// different table lengths.
    pub fn linear_combination<'a, I>(terms: I) -> Result<Self, AkitaError>
    where
        I: IntoIterator<Item = (E, &'a Self)>,
        E: 'a,
    {
        let _span =
            tracing::debug_span!("SparseExtensionOpeningWitness::linear_combination").entered();
        let mut table_len = None;
        let mut entries = Vec::new();
        {
            let _span = tracing::debug_span!("sparse_extension_witness_lc_collect").entered();
            for (coeff, witness) in terms {
                match table_len {
                    Some(len) if len != witness.table_len() => {
                        return Err(AkitaError::InvalidSize {
                            expected: len,
                            actual: witness.table_len(),
                        });
                    }
                    None => table_len = Some(witness.table_len()),
                    Some(_) => {}
                }
                entries.extend(
                    witness
                        .entries()
                        .iter()
                        .map(|&(idx, value)| (idx, value * coeff)),
                );
            }
        }
        let table_len = table_len.ok_or_else(|| {
            AkitaError::InvalidInput(
                "sparse extension-opening witness combination requires at least one term"
                    .to_string(),
            )
        })?;
        let _span = tracing::debug_span!(
            "sparse_extension_witness_lc_normalize",
            table_len,
            entries_len = entries.len()
        )
        .entered();
        Self::new(table_len, entries)
    }

    fn claim_with_factor(&self, factor_evals: &[E]) -> Result<E, AkitaError> {
        if factor_evals.len() != self.table_len {
            return Err(AkitaError::InvalidSize {
                expected: self.table_len,
                actual: factor_evals.len(),
            });
        }
        Ok(self.entries.iter().fold(E::zero(), |acc, &(idx, value)| {
            acc + value * factor_evals[idx]
        }))
    }

    fn claim_with_factor_fn<P>(&self, factor_at: P) -> E
    where
        P: Fn(usize) -> E,
    {
        self.entries
            .iter()
            .fold(E::zero(), |acc, &(idx, value)| acc + value * factor_at(idx))
    }

    fn final_eval(&self) -> Option<E> {
        if self.table_len != 1 {
            return None;
        }
        Some(
            self.entries
                .first()
                .map(|(_, value)| *value)
                .unwrap_or(E::zero()),
        )
    }

    fn accumulate_entries_with_factor<P>(
        entries: &[(usize, E)],
        coeff: E,
        factor_pair: &P,
    ) -> (E, E, E)
    where
        P: Fn(usize) -> (E, E) + Sync,
    {
        let mut constant = E::zero();
        let mut linear = E::zero();
        let mut quadratic = E::zero();
        let mut i = 0;
        while i < entries.len() {
            let pair = entries[i].0 / 2;
            let mut w0 = E::zero();
            let mut w1 = E::zero();
            while i < entries.len() && entries[i].0 / 2 == pair {
                let (idx, value) = entries[i];
                if idx & 1 == 0 {
                    w0 += value;
                } else {
                    w1 += value;
                }
                i += 1;
            }

            let (a0, a1) = factor_pair(pair);
            let da = a1 - a0;
            if w0 == E::zero() {
                linear += w1 * a0;
                quadratic += w1 * da;
            } else if w1 == E::zero() {
                let w0_a0 = w0 * a0;
                let w0_da = w0 * da;
                constant += w0_a0;
                linear += w0_da - w0_a0;
                quadratic -= w0_da;
            } else {
                let dw = w1 - w0;
                constant += w0 * a0;
                linear += dw * a0 + w0 * da;
                quadratic += dw * da;
            }
        }

        (coeff * constant, coeff * linear, coeff * quadratic)
    }

    fn accumulate_entries(entries: &[(usize, E)], factor_evals: &[E], coeff: E) -> (E, E, E) {
        Self::accumulate_entries_with_factor(entries, coeff, &|pair| {
            (factor_evals[2 * pair], factor_evals[2 * pair + 1])
        })
    }

    fn fold_entries(entries: &[(usize, E)], r_round: E) -> Vec<(usize, E)> {
        let one_minus = E::one() - r_round;
        let mut folded = Vec::with_capacity(entries.len());
        let mut i = 0;
        while i < entries.len() {
            let pair = entries[i].0 / 2;
            let mut value = E::zero();
            while i < entries.len() && entries[i].0 / 2 == pair {
                let (idx, entry_value) = entries[i];
                value += if idx & 1 == 0 {
                    entry_value * one_minus
                } else {
                    entry_value * r_round
                };
                i += 1;
            }
            if value != E::zero() {
                folded.push((pair, value));
            }
        }
        folded
    }

    #[cfg(feature = "parallel")]
    fn pair_aligned_ranges(&self) -> Vec<(usize, usize)> {
        let len = self.entries.len();
        let target_chunks = rayon::current_num_threads() * SPARSE_PARALLEL_CHUNKS_PER_THREAD;
        let chunk_size = len
            .div_ceil(target_chunks)
            .max(SPARSE_PARALLEL_ENTRY_THRESHOLD);
        let mut ranges = Vec::with_capacity(target_chunks.min(len.div_ceil(chunk_size)));
        let mut start = 0;
        while start < len {
            let mut end = (start + chunk_size).min(len);
            if end < len {
                let split_pair = self.entries[end].0 / 2;
                while end < len && self.entries[end].0 / 2 == split_pair {
                    end += 1;
                }
            }
            ranges.push((start, end));
            start = end;
        }
        ranges
    }

    fn accumulate_round(
        &self,
        factor_evals: &[E],
        coeff: E,
        constant: &mut E,
        linear: &mut E,
        quadratic: &mut E,
    ) {
        let _span = tracing::trace_span!(
            "SparseExtensionOpeningWitness::accumulate_round",
            table_len = self.table_len,
            entries_len = self.entries.len()
        )
        .entered();
        #[cfg(feature = "parallel")]
        let (round_constant, round_linear, round_quadratic) =
            if self.entries.len() >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
                self.pair_aligned_ranges()
                    .into_par_iter()
                    .map(|(start, end)| {
                        Self::accumulate_entries(&self.entries[start..end], factor_evals, coeff)
                    })
                    .reduce(
                        || (E::zero(), E::zero(), E::zero()),
                        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2),
                    )
            } else {
                Self::accumulate_entries(&self.entries, factor_evals, coeff)
            };
        #[cfg(not(feature = "parallel"))]
        let (round_constant, round_linear, round_quadratic) =
            Self::accumulate_entries(&self.entries, factor_evals, coeff);
        *constant += round_constant;
        *linear += round_linear;
        *quadratic += round_quadratic;
    }

    fn accumulate_round_with_factor<P>(
        &self,
        coeff: E,
        constant: &mut E,
        linear: &mut E,
        quadratic: &mut E,
        factor_pair: P,
    ) where
        P: Fn(usize) -> (E, E) + Sync,
    {
        let _span = tracing::trace_span!(
            "SparseExtensionOpeningWitness::accumulate_round_with_factor",
            table_len = self.table_len,
            entries_len = self.entries.len()
        )
        .entered();
        #[cfg(feature = "parallel")]
        let (round_constant, round_linear, round_quadratic) =
            if self.entries.len() >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
                self.pair_aligned_ranges()
                    .into_par_iter()
                    .map(|(start, end)| {
                        Self::accumulate_entries_with_factor(
                            &self.entries[start..end],
                            coeff,
                            &factor_pair,
                        )
                    })
                    .reduce(
                        || (E::zero(), E::zero(), E::zero()),
                        |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2),
                    )
            } else {
                Self::accumulate_entries_with_factor(&self.entries, coeff, &factor_pair)
            };
        #[cfg(not(feature = "parallel"))]
        let (round_constant, round_linear, round_quadratic) =
            Self::accumulate_entries_with_factor(&self.entries, coeff, &factor_pair);
        *constant += round_constant;
        *linear += round_linear;
        *quadratic += round_quadratic;
    }

    fn fold_in_place(&mut self, r_round: E) {
        let _span = tracing::trace_span!(
            "SparseExtensionOpeningWitness::fold_in_place",
            table_len = self.table_len,
            entries_len = self.entries.len()
        )
        .entered();
        if self.table_len <= 1 {
            return;
        }
        #[cfg(feature = "parallel")]
        let folded = if self.entries.len() >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
            let chunks = self
                .pair_aligned_ranges()
                .into_par_iter()
                .map(|(start, end)| Self::fold_entries(&self.entries[start..end], r_round))
                .collect::<Vec<_>>();
            let len = chunks.iter().map(Vec::len).sum();
            let mut folded = Vec::with_capacity(len);
            for chunk in chunks {
                folded.extend(chunk);
            }
            folded
        } else {
            Self::fold_entries(&self.entries, r_round)
        };
        #[cfg(not(feature = "parallel"))]
        let folded = Self::fold_entries(&self.entries, r_round);
        self.table_len /= 2;
        self.entries = folded;
    }
}

#[derive(Debug, Clone)]
struct TensorFactorTransition<E: FieldCore> {
    zero: Vec<Vec<E>>,
    one: Vec<Vec<E>>,
}

/// Lazy transparent tensor factor for sparse extension-opening terms.
///
/// This stores the exact multilinear folding state for
/// `A_eta(w) = sum_u eq(u, eta) * coord_u(eq(r_tail, w))` without relying on
/// `coord_u` being extension-linear. Once the sparse low block has been folded,
/// it materializes into the ordinary dense factor table and rejoins the shared
/// reduction path.
#[derive(Debug, Clone)]
struct TensorEqualityFactor<E: FieldCore> {
    table_vars: usize,
    round: usize,
    materialize_at: usize,
    prefix_state: Vec<E>,
    transitions: Vec<TensorFactorTransition<E>>,
    suffix_tables: Vec<Vec<E>>,
    low_states: Vec<Vec<E>>,
}

impl<E: FieldCore> TensorEqualityFactor<E> {
    fn new<F>(tail_point: Vec<E>, eta: Vec<E>, materialize_at: usize) -> Result<Self, AkitaError>
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
        if materialize_at > tail_point.len() {
            return Err(AkitaError::InvalidSize {
                expected: tail_point.len(),
                actual: materialize_at,
            });
        }
        checked_table_len(tail_point.len())?;
        checked_table_len(tail_point.len() - materialize_at)?;

        let eta_weights = EqPolynomial::evals(&eta)?;
        let basis = (0..width)
            .map(|idx| {
                let mut coords = vec![F::zero(); width];
                coords[idx] = F::one();
                E::from_base_slice(&coords)
            })
            .collect::<Vec<_>>();
        let one_coords = E::one().to_base_vec();
        if one_coords.len() != width {
            return Err(AkitaError::InvalidSize {
                expected: width,
                actual: one_coords.len(),
            });
        }
        let prefix_state = one_coords.into_iter().map(E::lift_base).collect::<Vec<_>>();

        let transitions = tail_point[..materialize_at]
            .iter()
            .copied()
            .map(|tail| Self::transition::<F>(&basis, tail, width))
            .collect::<Result<Vec<_>, _>>()?;
        let suffix_eq = EqPolynomial::evals(&tail_point[materialize_at..])?;
        let suffix_tables = basis
            .iter()
            .map(|&basis_elem| {
                suffix_eq
                    .iter()
                    .copied()
                    .map(|suffix| {
                        project_tensor_factor_value::<F, E>(
                            basis_elem * suffix,
                            &eta_weights,
                            width,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut factor = Self {
            table_vars: tail_point.len(),
            round: 0,
            materialize_at,
            prefix_state,
            transitions,
            suffix_tables,
            low_states: Vec::new(),
        };
        factor.rebuild_low_states();
        Ok(factor)
    }

    fn transition<F>(
        basis: &[E],
        tail: E,
        width: usize,
    ) -> Result<TensorFactorTransition<E>, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        let tail_zero = E::one() - tail;
        let tail_one = tail;
        let mut zero = vec![vec![E::zero(); width]; width];
        let mut one = vec![vec![E::zero(); width]; width];
        for (src_idx, &basis_elem) in basis.iter().enumerate() {
            let zero_coords = (basis_elem * tail_zero).to_base_vec();
            let one_coords = (basis_elem * tail_one).to_base_vec();
            if zero_coords.len() != width || one_coords.len() != width {
                return Err(AkitaError::InvalidSize {
                    expected: width,
                    actual: zero_coords.len().max(one_coords.len()),
                });
            }
            for dst_idx in 0..width {
                zero[src_idx][dst_idx] = E::lift_base(zero_coords[dst_idx]);
                one[src_idx][dst_idx] = E::lift_base(one_coords[dst_idx]);
            }
        }
        Ok(TensorFactorTransition { zero, one })
    }

    fn len(&self) -> usize {
        1usize << (self.table_vars - self.round)
    }

    fn is_ready_to_materialize(&self) -> bool {
        self.round >= self.materialize_at
    }

    fn apply_transition(
        state: &[E],
        transition: &TensorFactorTransition<E>,
        challenge: E,
    ) -> Vec<E> {
        let width = state.len();
        let one_minus = E::one() - challenge;
        let mut next = vec![E::zero(); width];
        for (src_idx, &src) in state.iter().enumerate() {
            if src == E::zero() {
                continue;
            }
            for (dst_idx, dst) in next.iter_mut().enumerate() {
                let step = transition.zero[src_idx][dst_idx] * one_minus
                    + transition.one[src_idx][dst_idx] * challenge;
                *dst += src * step;
            }
        }
        next
    }

    fn apply_boolean_transition(
        state: &[E],
        transition: &TensorFactorTransition<E>,
        bit: usize,
    ) -> Vec<E> {
        let width = state.len();
        let matrix = if bit == 0 {
            &transition.zero
        } else {
            &transition.one
        };
        let mut next = vec![E::zero(); width];
        for (src_idx, &src) in state.iter().enumerate() {
            if src == E::zero() {
                continue;
            }
            for (dst_idx, dst) in next.iter_mut().enumerate() {
                *dst += src * matrix[src_idx][dst_idx];
            }
        }
        next
    }

    fn rebuild_low_states(&mut self) {
        let low_bits = self.materialize_at.saturating_sub(self.round);
        if low_bits == 0 {
            self.low_states.clear();
            return;
        }
        let count = 1usize << low_bits;
        let mut low_states = Vec::with_capacity(count);
        for low in 0..count {
            let mut state = self.prefix_state.clone();
            for bit_idx in 0..low_bits {
                let bit = (low >> bit_idx) & 1;
                state = Self::apply_boolean_transition(
                    &state,
                    &self.transitions[self.round + bit_idx],
                    bit,
                );
            }
            low_states.push(state);
        }
        self.low_states = low_states;
    }

    fn eval_state_at_suffix(&self, state: &[E], suffix_index: usize) -> E {
        self.suffix_tables
            .iter()
            .zip(state.iter().copied())
            .fold(E::zero(), |acc, (table, coeff)| {
                acc + coeff * table[suffix_index]
            })
    }

    fn factor_at_index(&self, index: usize) -> E {
        let low_bits = self.materialize_at.saturating_sub(self.round);
        if low_bits == 0 {
            return self.eval_state_at_suffix(&self.prefix_state, index);
        }
        let low_mask = (1usize << low_bits) - 1;
        let low = index & low_mask;
        let suffix_index = index >> low_bits;
        self.eval_state_at_suffix(&self.low_states[low], suffix_index)
    }

    fn factor_pair(&self, pair: usize) -> (E, E) {
        let low_bits = self.materialize_at - self.round;
        debug_assert!(low_bits > 0);
        let rest_low_bits = low_bits - 1;
        let low_mask = (1usize << rest_low_bits).saturating_sub(1);
        let low_rest = pair & low_mask;
        let suffix_index = pair >> rest_low_bits;
        let low_zero = low_rest << 1;
        let low_one = low_zero | 1;
        (
            self.eval_state_at_suffix(&self.low_states[low_zero], suffix_index),
            self.eval_state_at_suffix(&self.low_states[low_one], suffix_index),
        )
    }

    fn fold_in_place(&mut self, r_round: E) {
        if self.len() <= 1 {
            return;
        }
        debug_assert!(self.round < self.materialize_at);
        self.prefix_state =
            Self::apply_transition(&self.prefix_state, &self.transitions[self.round], r_round);
        self.round += 1;
        self.rebuild_low_states();
    }

    fn materialize_dense(&self) -> Vec<E> {
        debug_assert!(self.is_ready_to_materialize());
        let suffix_len = self.suffix_tables.first().map(Vec::len).unwrap_or(0);
        let _span = tracing::debug_span!(
            "TensorEqualityFactor::materialize_dense",
            suffix_len,
            width = self.prefix_state.len()
        )
        .entered();
        #[cfg(feature = "parallel")]
        {
            (0..suffix_len)
                .into_par_iter()
                .map(|idx| self.eval_state_at_suffix(&self.prefix_state, idx))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..suffix_len)
                .map(|idx| self.eval_state_at_suffix(&self.prefix_state, idx))
                .collect()
        }
    }
}

#[derive(Debug, Clone)]
enum BatchedExtensionOpeningWitness<E: FieldCore> {
    Dense(Vec<E>),
    Sparse(SparseExtensionOpeningWitness<E>),
}

#[derive(Debug, Clone)]
enum BatchedExtensionOpeningFactor<E: FieldCore> {
    Dense(Vec<E>),
    Tensor(TensorEqualityFactor<E>),
}

impl<E: FieldCore> BatchedExtensionOpeningFactor<E> {
    fn len(&self) -> usize {
        match self {
            Self::Dense(evals) => evals.len(),
            Self::Tensor(factor) => factor.len(),
        }
    }
}

impl<E: FieldCore> BatchedExtensionOpeningWitness<E> {
    fn len(&self) -> usize {
        match self {
            Self::Dense(evals) => evals.len(),
            Self::Sparse(witness) => witness.table_len(),
        }
    }

    fn claim_with_factor(
        &self,
        factor: &BatchedExtensionOpeningFactor<E>,
    ) -> Result<E, AkitaError> {
        match self {
            Self::Dense(evals) => match factor {
                BatchedExtensionOpeningFactor::Dense(factor_evals) => {
                    extension_opening_reduction_claim(evals, factor_evals)
                }
                BatchedExtensionOpeningFactor::Tensor(_) => Err(AkitaError::InvalidInput(
                    "lazy tensor extension-opening factor requires a sparse witness".to_string(),
                )),
            },
            Self::Sparse(witness) => match factor {
                BatchedExtensionOpeningFactor::Dense(factor_evals) => {
                    witness.claim_with_factor(factor_evals)
                }
                BatchedExtensionOpeningFactor::Tensor(factor) => {
                    if witness.table_len() != factor.len() {
                        return Err(AkitaError::InvalidSize {
                            expected: witness.table_len(),
                            actual: factor.len(),
                        });
                    }
                    Ok(witness.claim_with_factor_fn(|idx| factor.factor_at_index(idx)))
                }
            },
        }
    }

    fn final_eval(&self) -> Option<E> {
        match self {
            Self::Dense(evals) => (evals.len() == 1).then_some(evals[0]),
            Self::Sparse(witness) => witness.final_eval(),
        }
    }

    fn accumulate_round(
        &self,
        factor: &BatchedExtensionOpeningFactor<E>,
        coeff: E,
        constant: &mut E,
        linear: &mut E,
        quadratic: &mut E,
    ) {
        match (self, factor) {
            (Self::Dense(witness_evals), BatchedExtensionOpeningFactor::Dense(factor_evals)) => {
                let (round_constant, round_linear, round_quadratic) =
                    accumulate_dense_round(witness_evals, factor_evals, coeff);
                *constant += round_constant;
                *linear += round_linear;
                *quadratic += round_quadratic;
            }
            (Self::Sparse(witness), BatchedExtensionOpeningFactor::Dense(factor_evals)) => {
                witness.accumulate_round(factor_evals, coeff, constant, linear, quadratic);
            }
            (Self::Sparse(witness), BatchedExtensionOpeningFactor::Tensor(factor)) => {
                witness.accumulate_round_with_factor(coeff, constant, linear, quadratic, |pair| {
                    factor.factor_pair(pair)
                });
            }
            (Self::Dense(_), BatchedExtensionOpeningFactor::Tensor(_)) => {
                unreachable!("lazy tensor factor is only constructed for sparse witnesses")
            }
        }
    }

    fn fold_with_factor_in_place(
        &mut self,
        factor: &mut BatchedExtensionOpeningFactor<E>,
        r_round: E,
    ) {
        match self {
            Self::Dense(witness_evals) => match factor {
                BatchedExtensionOpeningFactor::Dense(factor_evals) => {
                    fold_dense_reduction_tables_in_place(witness_evals, factor_evals, r_round);
                }
                BatchedExtensionOpeningFactor::Tensor(_) => {
                    unreachable!("lazy tensor factor is only constructed for sparse witnesses")
                }
            },
            Self::Sparse(witness) => {
                witness.fold_in_place(r_round);
                match factor {
                    BatchedExtensionOpeningFactor::Dense(factor_evals) => {
                        fold_evals_in_place(factor_evals, r_round);
                    }
                    BatchedExtensionOpeningFactor::Tensor(tensor_factor) => {
                        tensor_factor.fold_in_place(r_round);
                        if tensor_factor.is_ready_to_materialize() {
                            let dense = tensor_factor.materialize_dense();
                            *factor = BatchedExtensionOpeningFactor::Dense(dense);
                        }
                    }
                }
            }
        }
    }
}

impl<E: FieldCore> BatchedExtensionOpeningReductionTerm<E> {
    /// Construct one term `coeff * sum_x witness(x) * factor(x)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the witness/factor tables are malformed.
    pub fn new(witness_evals: Vec<E>, factor_evals: Vec<E>, coeff: E) -> Result<Self, AkitaError> {
        validate_reduction_tables(&witness_evals, &factor_evals)?;
        Ok(Self {
            current_witness_evals: BatchedExtensionOpeningWitness::Dense(witness_evals),
            current_factor: BatchedExtensionOpeningFactor::Dense(factor_evals),
            coeff,
        })
    }

    /// Construct one sparse-witness term `coeff * sum_x witness(x) * factor(x)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the sparse witness and factor table shapes differ.
    pub fn new_sparse(
        witness_evals: SparseExtensionOpeningWitness<E>,
        factor_evals: Vec<E>,
        coeff: E,
    ) -> Result<Self, AkitaError> {
        if witness_evals.table_len() != factor_evals.len() {
            return Err(AkitaError::InvalidSize {
                expected: witness_evals.table_len(),
                actual: factor_evals.len(),
            });
        }
        Ok(Self {
            current_witness_evals: BatchedExtensionOpeningWitness::Sparse(witness_evals),
            current_factor: BatchedExtensionOpeningFactor::Dense(factor_evals),
            coeff,
        })
    }

    /// Construct one sparse-witness term with a lazy transparent tensor factor.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor factor shape and sparse witness domain
    /// differ, or if the tensor opening parameters are malformed.
    pub fn new_sparse_tensor_factor<F>(
        witness_evals: SparseExtensionOpeningWitness<E>,
        tail_point: Vec<E>,
        eta: Vec<E>,
        coeff: E,
        materialize_at: usize,
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        let factor = TensorEqualityFactor::new::<F>(tail_point, eta, materialize_at)?;
        if witness_evals.table_len() != factor.len() {
            return Err(AkitaError::InvalidSize {
                expected: witness_evals.table_len(),
                actual: factor.len(),
            });
        }
        let current_factor = if factor.is_ready_to_materialize() {
            BatchedExtensionOpeningFactor::Dense(factor.materialize_dense())
        } else {
            BatchedExtensionOpeningFactor::Tensor(factor)
        };
        Ok(Self {
            current_witness_evals: BatchedExtensionOpeningWitness::Sparse(witness_evals),
            current_factor,
            coeff,
        })
    }

    /// Batching coefficient multiplying this term.
    pub fn coeff(&self) -> E {
        self.coeff
    }

    /// Return final folded witness/factor evaluations after all challenges.
    pub fn final_witness_and_factor_evals(&self) -> Option<(E, E)> {
        match &self.current_factor {
            BatchedExtensionOpeningFactor::Dense(factor_evals) => (factor_evals.len() == 1)
                .then(|| self.current_witness_evals.final_eval())
                .flatten()
                .map(|witness| (witness, factor_evals[0])),
            BatchedExtensionOpeningFactor::Tensor(_) => None,
        }
    }
}

/// Prover state for one batched degree-two extension-opening reduction.
#[derive(Debug, Clone)]
pub struct BatchedExtensionOpeningReductionProver<E: FieldCore> {
    terms: Vec<BatchedExtensionOpeningReductionTerm<E>>,
    input_claim: E,
    num_rounds: usize,
}

impl<E: FieldCore> BatchedExtensionOpeningReductionProver<E> {
    /// Construct a batched prover from terms sharing one Boolean domain.
    ///
    /// The caller supplies the claimed input sum. This avoids recomputing it
    /// in protocol paths that already derived the claim while preparing the
    /// transcript-bound reduction.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no terms or their table lengths differ.
    pub fn new(
        terms: Vec<BatchedExtensionOpeningReductionTerm<E>>,
        input_claim: E,
    ) -> Result<Self, AkitaError> {
        let first = terms.first().ok_or_else(|| {
            AkitaError::InvalidInput(
                "batched extension-opening reduction requires at least one term".to_string(),
            )
        })?;
        let table_len = first.current_witness_evals.len();
        let num_rounds = num_rounds_from_table_len(table_len)?;
        for term in &terms {
            if term.current_witness_evals.len() != table_len
                || term.current_factor.len() != table_len
            {
                return Err(AkitaError::InvalidSize {
                    expected: table_len,
                    actual: term
                        .current_witness_evals
                        .len()
                        .max(term.current_factor.len()),
                });
            }
        }
        Ok(Self {
            terms,
            input_claim,
            num_rounds,
        })
    }

    /// Override the transcript-visible input claim.
    ///
    /// This is used by ZK masked openings where the term tables remain true,
    /// but the transcript absorbs the masked claim.
    #[must_use]
    pub fn with_input_claim(mut self, input_claim: E) -> Self {
        self.input_claim = input_claim;
        self
    }

    /// Compute the input sum represented by a set of batched terms.
    ///
    /// This is useful for tests and standalone callers that do not already
    /// have an independently derived input claim.
    ///
    /// # Errors
    ///
    /// Returns an error if any term has malformed witness/factor tables.
    pub fn input_claim_from_terms(
        terms: &[BatchedExtensionOpeningReductionTerm<E>],
    ) -> Result<E, AkitaError> {
        terms.iter().try_fold(E::zero(), |acc, term| {
            term.current_witness_evals
                .claim_with_factor(&term.current_factor)
                .map(|claim| acc + term.coeff * claim)
        })
    }

    /// Final folded `(coeff, witness(rho), factor(rho))` tuples.
    pub fn final_terms(&self) -> Option<Vec<(E, E, E)>> {
        self.terms
            .iter()
            .map(|term| {
                term.final_witness_and_factor_evals()
                    .map(|(witness, factor)| (term.coeff, witness, factor))
            })
            .collect()
    }
}

impl<E: FieldCore> SumcheckInstanceProver<E> for BatchedExtensionOpeningReductionProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        EXTENSION_OPENING_REDUCTION_DEGREE
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, round: usize, previous_claim: E) -> UniPoly<E> {
        let mut constant = E::zero();
        let mut linear = E::zero();
        let mut quadratic = E::zero();

        for term in &self.terms {
            debug_assert_eq!(
                term.current_witness_evals.len(),
                1usize << (self.num_rounds - round)
            );
            debug_assert_eq!(term.current_factor.len(), term.current_witness_evals.len());

            term.current_witness_evals.accumulate_round(
                &term.current_factor,
                term.coeff,
                &mut constant,
                &mut linear,
                &mut quadratic,
            );
        }

        let poly = UniPoly::from_coeffs(vec![constant, linear, quadratic]);
        debug_assert_eq!(
            poly.evaluate(&E::zero()) + poly.evaluate(&E::one()),
            previous_claim
        );
        poly
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: E) {
        for term in &mut self.terms {
            if term.current_witness_evals.len() > 1 {
                term.current_witness_evals
                    .fold_with_factor_in_place(&mut term.current_factor, r_round);
            }
        }
    }
}

/// Verifier state for the degree-two extension-opening reduction sumcheck.
#[derive(Debug, Clone)]
pub struct ExtensionOpeningReductionVerifier<E: FieldCore> {
    witness_evals: Vec<E>,
    factor_evals: Vec<E>,
    input_claim: E,
    num_rounds: usize,
}

impl<E: FieldCore> ExtensionOpeningReductionVerifier<E> {
    /// Construct a verifier from transformed-witness and transparent-factor
    /// Boolean-hypercube evaluation tables.
    ///
    /// # Errors
    ///
    /// Returns an error if the tables do not have the same nonzero power-of-two
    /// length.
    pub fn new(witness_evals: Vec<E>, factor_evals: Vec<E>) -> Result<Self, AkitaError> {
        let input_claim = extension_opening_reduction_claim(&witness_evals, &factor_evals)?;
        let num_rounds = num_rounds_from_table_len(witness_evals.len())?;
        Ok(Self {
            witness_evals,
            factor_evals,
            input_claim,
            num_rounds,
        })
    }
}

impl<E: FieldCore> SumcheckInstanceVerifier<E> for ExtensionOpeningReductionVerifier<E> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        EXTENSION_OPENING_REDUCTION_DEGREE
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
        extension_opening_reduction_eval_at_point(
            &self.witness_evals,
            &self.factor_evals,
            challenges,
        )
    }
}

/// Transcript driver for an extension-opening reduction sumcheck.
///
/// Unlike [`ExtensionOpeningReductionVerifier`], this object only verifies the
/// round chain and returns the detached final claim. The caller must still check
/// that final claim against the separately opened transformed witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtensionOpeningReductionSumcheck<E: FieldCore> {
    input_claim: E,
    num_rounds: usize,
}

impl<E: FieldCore> ExtensionOpeningReductionSumcheck<E> {
    /// Construct a detached extension-opening reduction sumcheck driver.
    #[must_use]
    pub fn new(input_claim: E, num_rounds: usize) -> Self {
        Self {
            input_claim,
            num_rounds,
        }
    }

    /// Initial transcript-visible claim.
    #[must_use]
    pub fn input_claim(&self) -> E {
        self.input_claim
    }

    /// Number of sumcheck rounds.
    #[must_use]
    pub fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    /// Degree bound for each round polynomial.
    #[must_use]
    pub fn degree_bound(&self) -> usize {
        EXTENSION_OPENING_REDUCTION_DEGREE
    }

    /// Prove an extension-opening reduction sumcheck.
    ///
    /// # Errors
    ///
    /// Returns an error if the prover instance shape does not match this driver
    /// or any produced round polynomial exceeds the fixed degree bound.
    pub fn prove<F, T, S>(
        &self,
        prover: &mut ExtensionOpeningReductionProver<E>,
        transcript: &mut T,
        sample_challenge: S,
    ) -> Result<(SumcheckProof<E>, ExtensionOpeningReductionRoundResult<E>), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        self.check_prover_shape(prover)?;
        let (proof, challenges, final_claim) =
            SumcheckInstanceProverExt::prove::<F, T, S>(prover, transcript, sample_challenge)?;
        Ok((
            proof,
            ExtensionOpeningReductionRoundResult {
                final_claim,
                challenges,
            },
        ))
    }

    /// Replay extension-opening reduction sumcheck rounds without doing the
    /// final witness-opening check.
    ///
    /// # Errors
    ///
    /// Returns an error if the proof shape is inconsistent or a round polynomial
    /// exceeds the fixed degree bound.
    pub fn verify<F, T, S>(
        &self,
        proof: &SumcheckProof<E>,
        transcript: &mut T,
        mut sample_challenge: S,
    ) -> Result<ExtensionOpeningReductionRoundResult<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        if proof.round_polys.len() != self.num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: self.num_rounds,
                actual: proof.round_polys.len(),
            });
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &self.input_claim);
        let mut claim = self.input_claim;
        let mut challenges = Vec::with_capacity(self.num_rounds);
        for poly in &proof.round_polys {
            if poly.degree() > self.degree_bound() {
                return Err(AkitaError::InvalidInput(format!(
                    "extension-opening reduction round poly exceeds degree bound {}",
                    self.degree_bound()
                )));
            }
            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
            let r_i = sample_challenge(transcript);
            challenges.push(r_i);
            claim = poly.eval_from_hint(&claim, &r_i);
        }

        Ok(ExtensionOpeningReductionRoundResult {
            final_claim: claim,
            challenges,
        })
    }

    fn check_prover_shape(
        &self,
        prover: &ExtensionOpeningReductionProver<E>,
    ) -> Result<(), AkitaError> {
        if prover.num_rounds() != self.num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: self.num_rounds,
                actual: prover.num_rounds(),
            });
        }
        if prover.input_claim() != self.input_claim {
            return Err(AkitaError::InvalidInput(
                "extension-opening reduction prover input claim mismatch".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "zk")]
impl<E: FieldCore> ExtensionOpeningReductionSumcheck<E> {
    /// Prove an extension-opening reduction sumcheck with ZK round masks.
    ///
    /// # Errors
    ///
    /// Returns an error if the prover/pad shape is invalid or any produced round
    /// polynomial exceeds the fixed degree bound.
    pub fn prove_zk<F, T, S>(
        &self,
        prover: &mut ExtensionOpeningReductionProver<E>,
        transcript: &mut T,
        sample_challenge: S,
        pre_sampled_pads: Vec<FullUniPoly<E>>,
    ) -> Result<
        (
            SumcheckProofMasked<E>,
            ExtensionOpeningReductionRoundResult<E>,
        ),
        AkitaError,
    >
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        self.check_prover_shape(prover)?;
        let (proof, challenges) = ZkSumcheckInstanceProverExt::prove_zk::<F, T, S>(
            prover,
            transcript,
            sample_challenge,
            pre_sampled_pads,
        )?;
        let (final_witness, final_factor) =
            prover.final_witness_and_factor_evals().ok_or_else(|| {
                AkitaError::InvalidInput(
                    "extension-opening reduction has not reached a final point".to_string(),
                )
            })?;
        Ok((
            proof,
            ExtensionOpeningReductionRoundResult {
                final_claim: final_witness * final_factor,
                challenges,
            },
        ))
    }

    /// Replay masked extension-opening reduction sumcheck rounds and return the
    /// unmasked final claim as a deferred R1CS linear combination.
    ///
    /// # Errors
    ///
    /// Returns an error if the proof shape is inconsistent or any round
    /// polynomial exceeds the fixed degree bound.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_zk<F, T, S>(
        &self,
        proof: &SumcheckProofMasked<E>,
        input_claim_mask: ZkR1csLinearCombination<E>,
        transcript: &mut T,
        mut sample_challenge: S,
        relations: &mut ZkRelationAccumulator<E>,
        hiding_cursor: &mut usize,
    ) -> Result<(ZkR1csLinearCombination<E>, Vec<E>), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize + ExtField<F>,
        S: FnMut(&mut T) -> E,
    {
        if proof.masked_round_polys.len() != self.num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: self.num_rounds,
                actual: proof.masked_round_polys.len(),
            });
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &self.input_claim);
        let mut masked_claim = self.input_claim;
        let mut claim_mask = input_claim_mask;
        let mut challenges = Vec::with_capacity(self.num_rounds);
        for masked_poly in &proof.masked_round_polys {
            if masked_poly.degree() > self.degree_bound() {
                return Err(AkitaError::InvalidInput(format!(
                    "extension-opening reduction round poly exceeds degree bound {}",
                    self.degree_bound()
                )));
            }
            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, masked_poly);
            let r_i = sample_challenge(transcript);
            challenges.push(r_i);
            let (next_claim_mask, _round_sum_mask) = relations
                .push_masked_full_round_relation::<F>(
                    "masked extension-opening reduction round chain",
                    masked_claim,
                    &claim_mask,
                    masked_poly.coeffs(),
                    r_i,
                    hiding_cursor,
                );
            masked_claim = masked_poly.evaluate(&r_i);
            claim_mask = next_claim_mask;
        }

        Ok((
            relations.push_masked_claim_relation(
                "extension-opening reduction final claim",
                masked_claim,
                &claim_mask,
            ),
            challenges,
        ))
    }
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

fn validate_reduction_tables<E: FieldCore>(
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

fn checked_table_len(num_vars: usize) -> Result<usize, AkitaError> {
    if num_vars >= usize::BITS as usize {
        return Err(AkitaError::InvalidInput(format!(
            "extension-opening reduction table has too many variables: {num_vars}"
        )));
    }
    Ok(1usize << num_vars)
}

fn num_rounds_from_table_len(len: usize) -> Result<usize, AkitaError> {
    if len == 0 || !len.is_power_of_two() {
        return Err(AkitaError::InvalidSize {
            expected: len.max(1).next_power_of_two(),
            actual: len,
        });
    }
    Ok(len.trailing_zeros() as usize)
}
