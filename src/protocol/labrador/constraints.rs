//! Labrador constraint types and shared recursive builders.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::labrador::setup::LabradorSetup;
use crate::protocol::labrador::types::LabradorReductionConfig;
use crate::{CanonicalField, FieldCore, FromSmallInt};
use std::ops::Range;

/// One sparse linear term in a Labrador constraint.
///
/// This encodes the paper-style contribution `<phi_i, s_i>`, except that
/// `offset` allows the term to start inside a packed witness row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorConstraintTerm<F: FieldCore, const D: usize> {
    /// Witness row used by this term.
    pub row: usize,
    /// Starting column within the witness row.
    pub offset: usize,
    /// Coefficients dotted against the witness row slice.
    pub coefficients: Vec<CyclotomicRing<F, D>>,
}

impl<F: FieldCore, const D: usize> LabradorConstraintTerm<F, D> {
    /// Build one sparse term `<coefficients, witness[row][offset..]>`.
    pub fn new(row: usize, offset: usize, coefficients: Vec<CyclotomicRing<F, D>>) -> Self {
        Self {
            row,
            offset,
            coefficients,
        }
    }
}

/// One scalar Labrador linear constraint.
///
/// Ignoring the quadratic paper term `a_ij`, this stores one equation of the form
/// `sum_terms <phi_i, s_i> = b`, where `target` is the single ring element `b`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorConstraint<F: FieldCore, const D: usize> {
    /// Sparse row terms contributing to the constraint.
    pub terms: Vec<LabradorConstraintTerm<F, D>>,
    /// Right-hand side ring element.
    pub target: CyclotomicRing<F, D>,
}

impl<F: FieldCore, const D: usize> LabradorConstraint<F, D> {
    /// Build a scalar Labrador constraint.
    pub fn new(terms: Vec<LabradorConstraintTerm<F, D>>, target: CyclotomicRing<F, D>) -> Self {
        Self { terms, target }
    }
}

/// Layout of the next-level witness emitted by one standard Labrador fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NextWitnessLayout {
    /// Number of rows used by the decomposed `z` witness.
    pub z_part_rows: usize,
    /// Row index holding `t_hat || h_hat`.
    pub aux_row: usize,
    /// Number of decomposed inner-commitment entries.
    pub t_hat_len: usize,
    /// Number of decomposed linear-garbage entries.
    pub h_hat_len: usize,
}

impl NextWitnessLayout {
    /// Derive the next-witness layout from input row count and config.
    pub(crate) fn new(input_rows: usize, config: &LabradorReductionConfig) -> Self {
        let t_hat_len = input_rows * config.kappa * config.fu;
        let h_hat_len = input_rows * (input_rows + 1) / 2 * config.fu;
        Self {
            z_part_rows: config.f,
            aux_row: config.f,
            t_hat_len,
            h_hat_len,
        }
    }

    /// Total number of witness rows at the next recursion level.
    pub(crate) fn num_rows(self) -> usize {
        self.z_part_rows + 1
    }

    /// Total length of the auxiliary row.
    pub(crate) fn aux_row_len(self) -> usize {
        self.t_hat_len + self.h_hat_len
    }

    /// Slice of the auxiliary row occupied by `t_hat`.
    pub(crate) fn t_hat_range(self) -> Range<usize> {
        0..self.t_hat_len
    }

    /// Slice of the auxiliary row occupied by `h_hat`.
    pub(crate) fn h_hat_range(self) -> Range<usize> {
        self.t_hat_len..self.aux_row_len()
    }
}

/// Build the recursive target relation for the next Labrador level.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_next_constraints<F, const D: usize>(
    phi_total: &[Vec<CyclotomicRing<F, D>>],
    b_total: &CyclotomicRing<F, D>,
    challenges: &[CyclotomicRing<F, D>],
    row_lengths: &[usize],
    max_len: usize,
    config: &LabradorReductionConfig,
    u1: &[CyclotomicRing<F, D>],
    u2: &[CyclotomicRing<F, D>],
    setup: &LabradorSetup<F, D>,
) -> Result<Vec<LabradorConstraint<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
{
    let r = row_lengths.len();
    if r == 0 || challenges.len() != r {
        return Err(HachiError::InvalidInput(
            "challenge row count mismatch".to_string(),
        ));
    }
    if config.f == 0 {
        return Err(HachiError::InvalidInput(
            "cannot build next constraints with f=0".to_string(),
        ));
    }

    let layout = NextWitnessLayout::new(r, config);
    let pow_b: Vec<F> = (0..config.f)
        .map(|idx| pow2_field::<F>(config.b * idx))
        .collect();
    let pow_bu: Vec<F> = (0..config.fu)
        .map(|idx| pow2_field::<F>(config.bu * idx))
        .collect();
    let combined_phi = combine_phi(phi_total, challenges, max_len);

    let mut constraints = Vec::new();
    if config.kappa1 > 0 {
        if u1.len() != config.kappa1 || u2.len() != config.kappa1 {
            return Err(HachiError::InvalidInput(
                "u1/u2 length mismatch for next statement".to_string(),
            ));
        }
        constraints.extend(build_outer_commitment_constraints(layout, setup, u1));
        constraints.extend(build_linear_garbage_commitment_constraints(
            layout, setup, u2,
        ));
    }
    constraints.extend(build_amortized_opening_constraints(
        layout, challenges, config, &pow_b, &pow_bu, setup,
    ));
    constraints.push(build_linear_garbage_constraint(
        layout,
        challenges,
        config,
        &pow_b,
        &pow_bu,
        &combined_phi,
    ));
    constraints.push(build_diagonal_constraint(
        layout, b_total, r, config, &pow_bu,
    ));
    Ok(constraints)
}

fn build_outer_commitment_constraints<F: FieldCore, const D: usize>(
    layout: NextWitnessLayout,
    setup: &LabradorSetup<F, D>,
    u1: &[CyclotomicRing<F, D>],
) -> Vec<LabradorConstraint<F, D>> {
    setup
        .b_mat
        .iter()
        .zip(u1.iter())
        .map(|(b_row, target)| {
            LabradorConstraint::new(
                vec![LabradorConstraintTerm::new(
                    layout.aux_row,
                    layout.t_hat_range().start,
                    b_row.clone(),
                )],
                *target,
            )
        })
        .collect()
}

fn build_linear_garbage_commitment_constraints<F: FieldCore, const D: usize>(
    layout: NextWitnessLayout,
    setup: &LabradorSetup<F, D>,
    u2: &[CyclotomicRing<F, D>],
) -> Vec<LabradorConstraint<F, D>> {
    setup
        .d_mat
        .iter()
        .zip(u2.iter())
        .map(|(d_row, target)| {
            LabradorConstraint::new(
                vec![LabradorConstraintTerm::new(
                    layout.aux_row,
                    layout.h_hat_range().start,
                    d_row.clone(),
                )],
                *target,
            )
        })
        .collect()
}

fn build_amortized_opening_constraints<F: FieldCore, const D: usize>(
    layout: NextWitnessLayout,
    challenges: &[CyclotomicRing<F, D>],
    config: &LabradorReductionConfig,
    pow_b: &[F],
    pow_bu: &[F],
    setup: &LabradorSetup<F, D>,
) -> Vec<LabradorConstraint<F, D>> {
    (0..config.kappa)
        .map(|output_idx| {
            let mut terms = Vec::with_capacity(config.f + 1);
            for (part_idx, scale) in pow_b.iter().copied().enumerate() {
                let coeffs = setup.a_mat[output_idx]
                    .iter()
                    .map(|elem| elem.scale(&scale))
                    .collect();
                terms.push(LabradorConstraintTerm::new(part_idx, 0, coeffs));
            }

            let mut aux_coeffs = vec![CyclotomicRing::<F, D>::zero(); layout.t_hat_len];
            for (row_idx, challenge) in challenges.iter().enumerate() {
                for (part_idx, &scale) in pow_bu.iter().enumerate() {
                    let idx =
                        row_idx * config.kappa * config.fu + output_idx * config.fu + part_idx;
                    aux_coeffs[idx] = -(challenge.scale(&scale));
                }
            }
            terms.push(LabradorConstraintTerm::new(
                layout.aux_row,
                layout.t_hat_range().start,
                aux_coeffs,
            ));

            LabradorConstraint::new(terms, CyclotomicRing::<F, D>::zero())
        })
        .collect()
}

fn build_linear_garbage_constraint<F: FieldCore, const D: usize>(
    layout: NextWitnessLayout,
    challenges: &[CyclotomicRing<F, D>],
    config: &LabradorReductionConfig,
    pow_b: &[F],
    pow_bu: &[F],
    combined_phi: &[CyclotomicRing<F, D>],
) -> LabradorConstraint<F, D> {
    let mut terms = Vec::with_capacity(config.f + 1);
    for (part_idx, scale) in pow_b.iter().copied().enumerate() {
        let coeffs = combined_phi.iter().map(|elem| elem.scale(&scale)).collect();
        terms.push(LabradorConstraintTerm::new(part_idx, 0, coeffs));
    }

    let mut h_coeffs = vec![CyclotomicRing::<F, D>::zero(); layout.h_hat_len];
    for i in 0..challenges.len() {
        for j in i..challenges.len() {
            let coeff = challenges[i] * challenges[j];
            let pair = pair_index(i, j, challenges.len());
            for (part_idx, &scale) in pow_bu.iter().enumerate() {
                let idx = pair * config.fu + part_idx;
                h_coeffs[idx] = -(coeff.scale(&scale));
            }
        }
    }
    terms.push(LabradorConstraintTerm::new(
        layout.aux_row,
        layout.h_hat_range().start,
        h_coeffs,
    ));

    LabradorConstraint::new(terms, CyclotomicRing::<F, D>::zero())
}

fn build_diagonal_constraint<F: FieldCore, const D: usize>(
    layout: NextWitnessLayout,
    b_total: &CyclotomicRing<F, D>,
    input_rows: usize,
    config: &LabradorReductionConfig,
    pow_bu: &[F],
) -> LabradorConstraint<F, D> {
    let mut diag_coeffs = vec![CyclotomicRing::<F, D>::zero(); layout.h_hat_len];
    for i in 0..input_rows {
        let pair = pair_index(i, i, input_rows);
        for (part_idx, &scale) in pow_bu.iter().enumerate() {
            let idx = pair * config.fu + part_idx;
            diag_coeffs[idx] = constant_poly(scale);
        }
    }

    LabradorConstraint::new(
        vec![LabradorConstraintTerm::new(
            layout.aux_row,
            layout.h_hat_range().start,
            diag_coeffs,
        )],
        *b_total,
    )
}

fn combine_phi<F: FieldCore, const D: usize>(
    phi_total: &[Vec<CyclotomicRing<F, D>>],
    challenges: &[CyclotomicRing<F, D>],
    max_len: usize,
) -> Vec<CyclotomicRing<F, D>> {
    let mut combined_phi = vec![CyclotomicRing::<F, D>::zero(); max_len];
    for (row_idx, row_phi) in phi_total.iter().enumerate() {
        let c = challenges[row_idx];
        for (j, elem) in row_phi.iter().enumerate() {
            combined_phi[j] += c * *elem;
        }
    }
    combined_phi
}

fn pow2_field<F: FieldCore + FromSmallInt>(exp: usize) -> F {
    let two = F::from_u64(2);
    let mut acc = F::one();
    for _ in 0..exp {
        acc = acc * two;
    }
    acc
}

fn constant_poly<F: FieldCore, const D: usize>(value: F) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(
        |i| {
            if i == 0 {
                value
            } else {
                F::zero()
            }
        },
    ))
}

pub(crate) fn pair_index(i: usize, j: usize, r: usize) -> usize {
    debug_assert!(i <= j && j < r);
    i * (2 * r - i + 1) / 2 + (j - i)
}
