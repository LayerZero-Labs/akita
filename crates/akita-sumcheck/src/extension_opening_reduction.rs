//! Extension-opening reduction sumcheck instances.
//!
//! This module implements the degree-two reduction used to collapse a logical
//! extension-field opening into one ordinary opening of the transformed
//! committed witness. The dense-table instance here proves claims of the form
//! `sum_x witness(x) * factor(x)`, where `factor` is the transparent
//! extension-opening factor. Later small-digit folded-witness code should plug
//! in at this boundary instead of treating the protocol as an arbitrary product
//! gadget.

use crate::drivers::prove_sumcheck;
use crate::traits::{SumcheckInstanceProver, SumcheckInstanceVerifier};
use crate::types::SumcheckProof;
use akita_algebra::poly::{fold_evals_in_place, multilinear_eval};
use akita_algebra::uni_poly::UniPoly;
use akita_algebra::EqPolynomial;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_transcript::{labels, Transcript};
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
    let tail_eq = EqPolynomial::evals(tail_point);
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
    let head_weights = EqPolynomial::evals(&logical_point[..split_bits]);
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
    Ok(EqPolynomial::evals(eta)
        .into_iter()
        .zip(row_partials.iter().copied())
        .fold(E::zero(), |acc, (weight, partial)| acc + weight * partial))
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
    let eta_weights = EqPolynomial::evals(eta);
    let mut out = EqPolynomial::evals(tail_point);
    let project = |value: &mut E| {
        let coords = value.to_base_vec();
        if coords.len() != width {
            return Err(AkitaError::InvalidSize {
                expected: width,
                actual: coords.len(),
            });
        }
        *value = coords
            .into_iter()
            .zip(eta_weights.iter().copied())
            .fold(E::zero(), |acc, (coord, weight)| {
                acc + weight.mul_base(coord)
            });
        Ok(())
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

    let eta_weights = EqPolynomial::evals(eta);
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
    pub fn evals(&self) -> Vec<E> {
        let mut out = vec![E::zero(); 1usize << self.num_vars];
        for term in &self.terms {
            let term_evals = EqPolynomial::evals_with_scaling(&term.point, Some(term.coeff));
            for (dst, value) in out.iter_mut().zip(term_evals) {
                *dst += value;
            }
        }
        out
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
        Ok(self.terms.iter().fold(E::zero(), |acc, term| {
            acc + term.coeff * EqPolynomial::mle(&term.point, point)
        }))
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
        extension_opening_reduction_claim(witness_evals, &self.evals())
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

        let half = self.current_witness_evals.len() / 2;
        let mut constant = E::zero();
        let mut linear = E::zero();
        let mut quadratic = E::zero();

        for i in 0..half {
            let w0 = self.current_witness_evals[2 * i];
            let w1 = self.current_witness_evals[2 * i + 1];
            let a0 = self.current_factor_evals[2 * i];
            let a1 = self.current_factor_evals[2 * i + 1];
            let dw = w1 - w0;
            let da = a1 - a0;

            constant += w0 * a0;
            linear += dw * a0 + w0 * da;
            quadratic += dw * da;
        }

        let poly = UniPoly::from_coeffs(vec![constant, linear, quadratic]);
        debug_assert_eq!(
            poly.evaluate(&E::zero()) + poly.evaluate(&E::one()),
            previous_claim
        );
        poly
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: E) {
        if self.current_witness_evals.len() > 1 {
            fold_evals_in_place(&mut self.current_witness_evals, r_round);
            fold_evals_in_place(&mut self.current_factor_evals, r_round);
        }
    }
}

/// One term in a batched extension-opening reduction sumcheck.
#[derive(Debug, Clone)]
pub struct BatchedExtensionOpeningReductionTerm<E: FieldCore> {
    current_witness_evals: BatchedExtensionOpeningWitness<E>,
    current_factor_evals: Vec<E>,
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
        if table_len == 0 || !table_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "sparse extension-opening witness length must be a nonzero power of two"
                    .to_string(),
            ));
        }
        for (idx, _) in &entries {
            if *idx >= table_len {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness index out of range".to_string(),
                ));
            }
        }
        entries.sort_unstable_by_key(|(idx, _)| *idx);
        let mut combined: Vec<(usize, E)> = Vec::with_capacity(entries.len());
        for (idx, value) in entries {
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

    fn accumulate_entries(entries: &[(usize, E)], factor_evals: &[E], coeff: E) -> (E, E, E) {
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

            let a0 = factor_evals[2 * pair];
            let a1 = factor_evals[2 * pair + 1];
            let dw = w1 - w0;
            let da = a1 - a0;

            constant += coeff * w0 * a0;
            linear += coeff * (dw * a0 + w0 * da);
            quadratic += coeff * dw * da;
        }
        (constant, linear, quadratic)
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
enum BatchedExtensionOpeningWitness<E: FieldCore> {
    Dense(Vec<E>),
    Sparse(SparseExtensionOpeningWitness<E>),
}

impl<E: FieldCore> BatchedExtensionOpeningWitness<E> {
    fn len(&self) -> usize {
        match self {
            Self::Dense(evals) => evals.len(),
            Self::Sparse(witness) => witness.table_len(),
        }
    }

    fn claim_with_factor(&self, factor_evals: &[E]) -> Result<E, AkitaError> {
        match self {
            Self::Dense(evals) => extension_opening_reduction_claim(evals, factor_evals),
            Self::Sparse(witness) => witness.claim_with_factor(factor_evals),
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
        factor_evals: &[E],
        coeff: E,
        constant: &mut E,
        linear: &mut E,
        quadratic: &mut E,
    ) {
        match self {
            Self::Dense(witness_evals) => {
                let half = witness_evals.len() / 2;
                for i in 0..half {
                    let w0 = witness_evals[2 * i];
                    let w1 = witness_evals[2 * i + 1];
                    let a0 = factor_evals[2 * i];
                    let a1 = factor_evals[2 * i + 1];
                    let dw = w1 - w0;
                    let da = a1 - a0;

                    *constant += coeff * w0 * a0;
                    *linear += coeff * (dw * a0 + w0 * da);
                    *quadratic += coeff * dw * da;
                }
            }
            Self::Sparse(witness) => {
                witness.accumulate_round(factor_evals, coeff, constant, linear, quadratic);
            }
        }
    }

    fn fold_in_place(&mut self, r_round: E) {
        match self {
            Self::Dense(evals) => fold_evals_in_place(evals, r_round),
            Self::Sparse(witness) => witness.fold_in_place(r_round),
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
            current_factor_evals: factor_evals,
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
            current_factor_evals: factor_evals,
            coeff,
        })
    }

    /// Batching coefficient multiplying this term.
    pub fn coeff(&self) -> E {
        self.coeff
    }

    /// Return final folded witness/factor evaluations after all challenges.
    pub fn final_witness_and_factor_evals(&self) -> Option<(E, E)> {
        (self.current_factor_evals.len() == 1)
            .then(|| self.current_witness_evals.final_eval())
            .flatten()
            .map(|witness| (witness, self.current_factor_evals[0]))
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
    /// # Errors
    ///
    /// Returns an error if there are no terms or their table lengths differ.
    pub fn new(terms: Vec<BatchedExtensionOpeningReductionTerm<E>>) -> Result<Self, AkitaError> {
        let first = terms.first().ok_or_else(|| {
            AkitaError::InvalidInput(
                "batched extension-opening reduction requires at least one term".to_string(),
            )
        })?;
        let table_len = first.current_witness_evals.len();
        let num_rounds = num_rounds_from_table_len(table_len)?;
        for term in &terms {
            if term.current_witness_evals.len() != table_len
                || term.current_factor_evals.len() != table_len
            {
                return Err(AkitaError::InvalidSize {
                    expected: table_len,
                    actual: term
                        .current_witness_evals
                        .len()
                        .max(term.current_factor_evals.len()),
                });
            }
        }
        let input_claim = terms.iter().try_fold(E::zero(), |acc, term| {
            term.current_witness_evals
                .claim_with_factor(&term.current_factor_evals)
                .map(|claim| acc + term.coeff * claim)
        })?;
        Ok(Self {
            terms,
            input_claim,
            num_rounds,
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
            debug_assert_eq!(
                term.current_factor_evals.len(),
                term.current_witness_evals.len()
            );

            term.current_witness_evals.accumulate_round(
                &term.current_factor_evals,
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
                term.current_witness_evals.fold_in_place(r_round);
                fold_evals_in_place(&mut term.current_factor_evals, r_round);
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

/// Prove an extension-opening reduction sumcheck.
///
/// # Errors
///
/// Returns an error if any produced round polynomial exceeds the fixed
/// extension-opening reduction degree bound.
pub fn prove_extension_opening_reduction<F, T, E, S>(
    prover: &mut ExtensionOpeningReductionProver<E>,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<(SumcheckProof<E>, ExtensionOpeningReductionRoundResult<E>), AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
{
    let (proof, challenges, final_claim) =
        prove_sumcheck::<F, T, E, S, _>(prover, transcript, sample_challenge)?;
    Ok((
        proof,
        ExtensionOpeningReductionRoundResult {
            final_claim,
            challenges,
        },
    ))
}

/// Replay extension-opening reduction sumcheck rounds without doing the final
/// witness-opening check.
///
/// The caller must check
/// `result.final_claim == opened_witness_at_rho * factor.evaluate(rho)` after
/// the ordinary single-point opening supplies `opened_witness_at_rho`.
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent or a round polynomial
/// exceeds the fixed extension-opening reduction degree bound.
pub fn verify_extension_opening_reduction_rounds<F, T, E, S>(
    proof: &SumcheckProof<E>,
    input_claim: E,
    num_rounds: usize,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<ExtensionOpeningReductionRoundResult<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
{
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &input_claim);
    let (final_claim, challenges) = proof.verify::<F, T, S>(
        input_claim,
        num_rounds,
        EXTENSION_OPENING_REDUCTION_DEGREE,
        transcript,
        sample_challenge,
    )?;
    Ok(ExtensionOpeningReductionRoundResult {
        final_claim,
        challenges,
    })
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

fn num_rounds_from_table_len(len: usize) -> Result<usize, AkitaError> {
    if len == 0 || !len.is_power_of_two() {
        return Err(AkitaError::InvalidSize {
            expected: len.max(1).next_power_of_two(),
            actual: len,
        });
    }
    Ok(len.trailing_zeros() as usize)
}
