//! Labrador constraint types and shared recursive builders.

use crate::algebra::ring::CyclotomicRing;
use crate::algebra::SparseChallenge;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::labrador::setup::LabradorSetupMatrices;
use crate::protocol::labrador::types::{LabradorReducedConstraintPlan, LabradorReductionConfig};
use crate::protocol::labrador::utils::pow2_field;
use crate::{cfg_into_iter, CanonicalField, FieldCore, FromSmallInt};
use std::ops::Range;
use std::sync::Arc;

type PreparedNextConstraintInputs<F, const D: usize> =
    (NextWitnessLayout, Vec<F>, Vec<F>, Vec<CyclotomicRing<F, D>>);

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
    /// Row index holding `inner_opening_digits || linear_garbage_digits`.
    pub aux_row: usize,
    /// Number of decomposed inner-commitment entries.
    pub inner_opening_digits_len: usize,
    /// Number of decomposed linear-garbage entries.
    pub linear_garbage_digits_len: usize,
}

impl NextWitnessLayout {
    /// Derive the next-witness layout from input row count and config.
    pub(crate) fn new(input_rows: usize, config: &LabradorReductionConfig) -> Self {
        let inner_opening_digits_len =
            input_rows * config.inner_commit_rank * config.aux_digit_parts;
        let linear_garbage_digits_len = input_rows * (input_rows + 1) / 2 * config.aux_digit_parts;
        Self {
            z_part_rows: config.witness_digit_parts,
            aux_row: config.witness_digit_parts,
            inner_opening_digits_len,
            linear_garbage_digits_len,
        }
    }

    /// Total number of witness rows at the next recursion level.
    pub(crate) fn num_rows(self) -> usize {
        self.z_part_rows + 1
    }

    /// Total length of the auxiliary row.
    pub(crate) fn aux_row_len(self) -> usize {
        self.inner_opening_digits_len + self.linear_garbage_digits_len
    }

    /// Slice of the auxiliary row occupied by `inner_opening_digits`.
    pub(crate) fn inner_opening_digits_range(self) -> Range<usize> {
        0..self.inner_opening_digits_len
    }

    /// Slice of the auxiliary row occupied by `linear_garbage_digits`.
    pub(crate) fn linear_garbage_digits_range(self) -> Range<usize> {
        self.inner_opening_digits_len..self.aux_row_len()
    }
}

/// Build the recursive target relation for the next Labrador level.
fn prepare_next_constraint_inputs<F, const D: usize>(
    phi_total: &[Vec<CyclotomicRing<F, D>>],
    challenges: &[SparseChallenge],
    row_lengths: &[usize],
    max_len: usize,
    config: &LabradorReductionConfig,
) -> Result<PreparedNextConstraintInputs<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
{
    let r = row_lengths.len();
    if r == 0 || challenges.len() != r {
        return Err(HachiError::InvalidInput(
            "challenge row count mismatch".to_string(),
        ));
    }
    if config.witness_digit_parts == 0 {
        return Err(HachiError::InvalidInput(
            "cannot build next constraints with witness_digit_parts=0".to_string(),
        ));
    }

    let layout = NextWitnessLayout::new(r, config);
    let pow_b: Vec<F> = (0..config.witness_digit_parts)
        .map(|idx| pow2_field::<F>(config.witness_digit_bits * idx))
        .collect();
    let pow_bu: Vec<F> = (0..config.aux_digit_parts)
        .map(|idx| pow2_field::<F>(config.aux_digit_bits * idx))
        .collect();
    let amortized_phi = combine_phi(phi_total, challenges, max_len);
    Ok((layout, pow_b, pow_bu, amortized_phi))
}

#[allow(clippy::too_many_arguments)]
fn build_constraints_from_prepared<F, const D: usize>(
    layout: NextWitnessLayout,
    config: &LabradorReductionConfig,
    challenges: &[SparseChallenge],
    pow_b: &[F],
    pow_bu: &[F],
    amortized_phi: &[CyclotomicRing<F, D>],
    aggregated_rhs: &CyclotomicRing<F, D>,
    inner_opening_payload: &[CyclotomicRing<F, D>],
    linear_garbage_payload: &[CyclotomicRing<F, D>],
    setup: &LabradorSetupMatrices<F, D>,
) -> Result<Vec<LabradorConstraint<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
{
    let mut constraints = Vec::new();
    let dense_challenges: Vec<CyclotomicRing<F, D>> = challenges
        .iter()
        .map(|challenge| {
            challenge
                .to_dense::<F, D>()
                .expect("sampler outputs valid challenges")
        })
        .collect();
    if config.outer_commit_rank > 0 {
        if inner_opening_payload.len() != config.outer_commit_rank
            || linear_garbage_payload.len() != config.outer_commit_rank
        {
            return Err(HachiError::InvalidInput(
                "payload length mismatch for next statement".to_string(),
            ));
        }
        constraints.extend(build_outer_commitment_constraints(
            layout,
            setup,
            inner_opening_payload,
        ));
        constraints.extend(build_linear_garbage_commitment_constraints(
            layout,
            setup,
            linear_garbage_payload,
        ));
    }
    constraints.extend(build_amortized_opening_constraints(
        layout,
        &dense_challenges,
        config,
        pow_b,
        pow_bu,
        setup,
    ));
    constraints.push(build_linear_garbage_constraint(
        layout,
        &dense_challenges,
        config,
        pow_b,
        pow_bu,
        amortized_phi,
    ));
    constraints.push(build_diagonal_constraint(
        layout,
        aggregated_rhs,
        challenges.len(),
        config,
        pow_bu,
    ));
    Ok(constraints)
}

/// Build the recursive target relation for the next Labrador level.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
#[tracing::instrument(skip_all, name = "labrador::build_next_constraints")]
pub(crate) fn build_next_constraints<F, const D: usize>(
    phi_total: &[Vec<CyclotomicRing<F, D>>],
    aggregated_rhs: &CyclotomicRing<F, D>,
    challenges: &[SparseChallenge],
    row_lengths: &[usize],
    max_len: usize,
    config: &LabradorReductionConfig,
    inner_opening_payload: &[CyclotomicRing<F, D>],
    linear_garbage_payload: &[CyclotomicRing<F, D>],
    setup: &LabradorSetupMatrices<F, D>,
) -> Result<Vec<LabradorConstraint<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
{
    let (layout, pow_b, pow_bu, amortized_phi) =
        prepare_next_constraint_inputs(phi_total, challenges, row_lengths, max_len, config)?;
    build_constraints_from_prepared(
        layout,
        config,
        challenges,
        &pow_b,
        &pow_bu,
        &amortized_phi,
        aggregated_rhs,
        inner_opening_payload,
        linear_garbage_payload,
        setup,
    )
}

#[tracing::instrument(skip_all, name = "labrador::build_next_constraint_plan")]
pub(crate) fn build_next_constraint_plan<F, const D: usize>(
    phi_total: &[Vec<CyclotomicRing<F, D>>],
    aggregated_rhs: &CyclotomicRing<F, D>,
    challenges: &[SparseChallenge],
    row_lengths: &[usize],
    max_len: usize,
    config: &LabradorReductionConfig,
    setup: Arc<LabradorSetupMatrices<F, D>>,
) -> Result<LabradorReducedConstraintPlan<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
{
    let (_layout, _pow_b, _pow_bu, amortized_phi) =
        prepare_next_constraint_inputs(phi_total, challenges, row_lengths, max_len, config)?;
    Ok(LabradorReducedConstraintPlan {
        row_count: row_lengths.len(),
        max_len,
        config: *config,
        challenges: challenges.to_vec(),
        amortized_phi,
        aggregated_rhs: *aggregated_rhs,
        setup,
    })
}

#[tracing::instrument(skip_all, name = "labrador::materialize_reduced_constraints")]
pub(crate) fn materialize_reduced_constraints<F, const D: usize>(
    plan: &LabradorReducedConstraintPlan<F, D>,
    inner_opening_payload: &[CyclotomicRing<F, D>],
    linear_garbage_payload: &[CyclotomicRing<F, D>],
) -> Result<Vec<LabradorConstraint<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
{
    let layout = NextWitnessLayout::new(plan.row_count, &plan.config);
    let pow_b: Vec<F> = (0..plan.config.witness_digit_parts)
        .map(|idx| pow2_field::<F>(plan.config.witness_digit_bits * idx))
        .collect();
    let pow_bu: Vec<F> = (0..plan.config.aux_digit_parts)
        .map(|idx| pow2_field::<F>(plan.config.aux_digit_bits * idx))
        .collect();
    build_constraints_from_prepared(
        layout,
        &plan.config,
        &plan.challenges,
        &pow_b,
        &pow_bu,
        &plan.amortized_phi,
        &plan.aggregated_rhs,
        inner_opening_payload,
        linear_garbage_payload,
        plan.setup.as_ref(),
    )
}

/// Build the paper's outer-commitment check (Fig. 3, line 19)
/// `inner_opening_payload = B * inner_opening_digits`, with the quadratic
/// `g_ij` contribution omitted.
///
/// The paper writes this as one vector equation. Here it is scalarized into one
/// `LabradorConstraint` per row of `B` / entry of the opening-side payload, all
/// reading the `inner_opening_digits` prefix of the auxiliary witness row.
fn build_outer_commitment_constraints<F: FieldCore, const D: usize>(
    layout: NextWitnessLayout,
    setup: &LabradorSetupMatrices<F, D>,
    inner_opening_payload: &[CyclotomicRing<F, D>],
) -> Vec<LabradorConstraint<F, D>> {
    setup
        .b_mat
        .iter()
        .zip(inner_opening_payload.iter())
        .map(|(b_row, target)| {
            LabradorConstraint::new(
                vec![LabradorConstraintTerm::new(
                    layout.aux_row,
                    layout.inner_opening_digits_range().start,
                    b_row.clone(),
                )],
                *target,
            )
        })
        .collect()
}

/// Build the linear-garbage commitment check (Fig. 3, line 20)
/// `linear_garbage_payload = D * linear_garbage_digits`.
///
/// As with the opening-side payload, the paper presents a vector equation; this
/// implementation expands it into one scalar constraint per row of `D` / entry
/// of the linear-garbage-side payload, reading the
/// `linear_garbage_digits` suffix of the auxiliary witness row.
fn build_linear_garbage_commitment_constraints<F: FieldCore, const D: usize>(
    layout: NextWitnessLayout,
    setup: &LabradorSetupMatrices<F, D>,
    linear_garbage_payload: &[CyclotomicRing<F, D>],
) -> Vec<LabradorConstraint<F, D>> {
    setup
        .d_mat
        .iter()
        .zip(linear_garbage_payload.iter())
        .map(|(d_row, target)| {
            LabradorConstraint::new(
                vec![LabradorConstraintTerm::new(
                    layout.aux_row,
                    layout.linear_garbage_digits_range().start,
                    d_row.clone(),
                )],
                *target,
            )
        })
        .collect()
}

/// Build the amortized opening relation (Fig. 3, line 15)
/// `A * z_tilde = sum_i c_i * t_tilde_i`.
///
/// The paper's equation is `inner_commit_rank`-dimensional, so this function
/// emits one scalar constraint per row of `A`. The first
/// `witness_digit_parts` witness rows reconstruct the decomposed
/// `z_tilde = sum_k 2^(k * witness_digit_bits) z^(k)`, while the
/// `inner_opening_digits` slice of the
/// auxiliary row reconstructs each decomposed `t_tilde_i`.
fn build_amortized_opening_constraints<F: FieldCore, const D: usize>(
    layout: NextWitnessLayout,
    challenges: &[CyclotomicRing<F, D>],
    config: &LabradorReductionConfig,
    pow_b: &[F],
    pow_bu: &[F],
    setup: &LabradorSetupMatrices<F, D>,
) -> Vec<LabradorConstraint<F, D>> {
    (0..config.inner_commit_rank)
        .map(|output_idx| {
            let mut terms = Vec::with_capacity(config.witness_digit_parts + 1);
            for (part_idx, scale) in pow_b.iter().copied().enumerate() {
                let coeffs = setup.a_mat[output_idx]
                    .iter()
                    .map(|elem| elem.scale(&scale))
                    .collect();
                terms.push(LabradorConstraintTerm::new(part_idx, 0, coeffs));
            }

            let mut aux_coeffs =
                vec![CyclotomicRing::<F, D>::zero(); layout.inner_opening_digits_len];
            for (row_idx, challenge) in challenges.iter().enumerate() {
                for (part_idx, &scale) in pow_bu.iter().enumerate() {
                    let idx = row_idx * config.inner_commit_rank * config.aux_digit_parts
                        + output_idx * config.aux_digit_parts
                        + part_idx;
                    aux_coeffs[idx] = -(challenge.scale(&scale));
                }
            }
            terms.push(LabradorConstraintTerm::new(
                layout.aux_row,
                layout.inner_opening_digits_range().start,
                aux_coeffs,
            ));

            LabradorConstraint::new(terms, CyclotomicRing::<F, D>::zero())
        })
        .collect()
}

/// Build the linear-only garbage relation (Fig. 3, line 17)
/// `sum_i c_i * <phi_i, z_tilde> = sum_{i <= j} c_i c_j * h_ij`.
///
/// `amortized_phi` already equals `sum_i c_i * phi_i`, so the left-hand side is
/// reconstructed from the decomposed `z_tilde` rows. The right-hand side is
/// reconstructed from the packed upper-triangular `linear_garbage_digits`
/// entries stored in the
/// auxiliary row.
fn build_linear_garbage_constraint<F: FieldCore, const D: usize>(
    layout: NextWitnessLayout,
    challenges: &[CyclotomicRing<F, D>],
    config: &LabradorReductionConfig,
    pow_b: &[F],
    pow_bu: &[F],
    amortized_phi: &[CyclotomicRing<F, D>],
) -> LabradorConstraint<F, D> {
    let mut terms = Vec::with_capacity(config.witness_digit_parts + 1);
    for (part_idx, scale) in pow_b.iter().copied().enumerate() {
        let coeffs = amortized_phi
            .iter()
            .map(|elem| elem.scale(&scale))
            .collect();
        terms.push(LabradorConstraintTerm::new(part_idx, 0, coeffs));
    }

    let mut h_coeffs = vec![CyclotomicRing::<F, D>::zero(); layout.linear_garbage_digits_len];
    for i in 0..challenges.len() {
        for j in i..challenges.len() {
            let coeff = challenges[i] * challenges[j];
            let pair = pair_index(i, j, challenges.len());
            for (part_idx, &scale) in pow_bu.iter().enumerate() {
                let idx = pair * config.aux_digit_parts + part_idx;
                h_coeffs[idx] = -(coeff.scale(&scale));
            }
        }
    }
    terms.push(LabradorConstraintTerm::new(
        layout.aux_row,
        layout.linear_garbage_digits_range().start,
        h_coeffs,
    ));

    LabradorConstraint::new(terms, CyclotomicRing::<F, D>::zero())
}

/// Build the linear-only diagonal relation (Fig. 3, line 18 with no `a_ij`
/// or `g_ij`)
/// `sum_i h_ii = aggregated_rhs`.
///
/// Only diagonal packed `linear_garbage_digits` entries contribute here. Their
/// decomposed digits are reweighted by powers of `2^aux_digit_bits` to
/// reconstruct each `h_ii`.
fn build_diagonal_constraint<F: FieldCore, const D: usize>(
    layout: NextWitnessLayout,
    aggregated_rhs: &CyclotomicRing<F, D>,
    input_rows: usize,
    config: &LabradorReductionConfig,
    pow_bu: &[F],
) -> LabradorConstraint<F, D> {
    let mut diag_coeffs = vec![CyclotomicRing::<F, D>::zero(); layout.linear_garbage_digits_len];
    for i in 0..input_rows {
        let pair = pair_index(i, i, input_rows);
        for (part_idx, &scale) in pow_bu.iter().enumerate() {
            let idx = pair * config.aux_digit_parts + part_idx;
            diag_coeffs[idx] = constant_poly(scale);
        }
    }

    LabradorConstraint::new(
        vec![LabradorConstraintTerm::new(
            layout.aux_row,
            layout.linear_garbage_digits_range().start,
            diag_coeffs,
        )],
        *aggregated_rhs,
    )
}

fn combine_phi<F: FieldCore + CanonicalField, const D: usize>(
    phi_total: &[Vec<CyclotomicRing<F, D>>],
    challenges: &[SparseChallenge],
    max_len: usize,
) -> Vec<CyclotomicRing<F, D>> {
    cfg_into_iter!(0..max_len)
        .map(|j| {
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (row_phi, challenge) in phi_total.iter().zip(challenges.iter()) {
                if let Some(elem) = row_phi.get(j) {
                    elem.mul_by_sparse_into(challenge, &mut acc);
                }
            }
            acc
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::protocol::labrador::comkey::LabradorComKeySeed;
    use crate::protocol::labrador::types::LabradorReductionConfig;

    type F = Fp64<4294967197>;
    const D: usize = 64;
    const COMKEY_SEED: LabradorComKeySeed = [9u8; 32];

    fn test_config() -> LabradorReductionConfig {
        LabradorReductionConfig {
            witness_digit_parts: 2,
            witness_digit_bits: 2,
            aux_digit_parts: 2,
            aux_digit_bits: 3,
            inner_commit_rank: 2,
            outer_commit_rank: 2,
            tail: false,
        }
    }

    fn unit_sparse_challenge(position: u32) -> SparseChallenge {
        SparseChallenge {
            positions: vec![position],
            coeffs: vec![1],
        }
    }

    fn constant_ring(value: u64) -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
            if idx == 0 {
                F::from_u64(value)
            } else {
                F::zero()
            }
        }))
    }

    #[test]
    fn next_witness_layout_ranges_are_disjoint() {
        let layout = NextWitnessLayout::new(3, &test_config());

        assert_eq!(layout.num_rows(), 3);
        assert_eq!(layout.aux_row, 2);
        assert_eq!(
            layout.inner_opening_digits_range().end,
            layout.linear_garbage_digits_range().start
        );
        assert_eq!(
            layout.linear_garbage_digits_range().end,
            layout.aux_row_len()
        );
    }

    #[test]
    fn pair_index_covers_upper_triangle_without_gaps() {
        let row_count = 4;
        let mut seen = Vec::new();
        for i in 0..row_count {
            for j in i..row_count {
                seen.push(pair_index(i, j, row_count));
            }
        }
        seen.sort_unstable();
        assert_eq!(
            seen,
            (0..row_count * (row_count + 1) / 2).collect::<Vec<_>>()
        );
    }

    #[test]
    fn reduced_constraint_plan_materializes_same_constraints() {
        let config = test_config();
        let row_lengths = vec![3usize, 2usize];
        let max_len = 3usize;
        let setup = Arc::new(LabradorSetupMatrices::new(
            &config,
            row_lengths.len(),
            max_len,
            &COMKEY_SEED,
        ));
        let phi_total = vec![
            vec![constant_ring(1), constant_ring(2), constant_ring(0)],
            vec![constant_ring(3), constant_ring(0), constant_ring(4)],
        ];
        let challenges = vec![unit_sparse_challenge(0), unit_sparse_challenge(1)];
        let aggregated_rhs = constant_ring(5);
        let inner_opening_payload = vec![constant_ring(6), constant_ring(7)];
        let linear_garbage_payload = vec![constant_ring(8), constant_ring(9)];

        let direct = build_next_constraints(
            &phi_total,
            &aggregated_rhs,
            &challenges,
            &row_lengths,
            max_len,
            &config,
            &inner_opening_payload,
            &linear_garbage_payload,
            setup.as_ref(),
        )
        .unwrap();

        let plan = build_next_constraint_plan(
            &phi_total,
            &aggregated_rhs,
            &challenges,
            &row_lengths,
            max_len,
            &config,
            Arc::clone(&setup),
        )
        .unwrap();

        let materialized =
            materialize_reduced_constraints(&plan, &inner_opening_payload, &linear_garbage_payload)
                .unwrap();

        assert_eq!(materialized, direct);
    }
}
