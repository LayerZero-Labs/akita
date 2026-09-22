//! Quotient witnesses for trinomial A relations.
//!
//! This constructs the polynomial quotient of `A v - Y c`, before the
//! relation-evaluation challenge is sampled. It neither commits that witness
//! nor authenticates a binary opening. The root protocol owns those phases.

use std::{fmt::Debug, marker::PhantomData};

use akita_algebra::{
    fft::{primitive_nth_root, FftWorkspace, SmoothDomain, SmoothFftField},
    TrinomialModulus, TrinomialRing,
};
use akita_error::{checked, AkitaError};
use akita_types::{RelationCoefficientRole, RelationPolynomial};

/// Reusable linear-convolution plan for a row of `A v = Y c` in a trinomial ring.
///
/// A length-`2D` transform contains the entire degree-`2D-2` product without
/// cyclic aliasing. If the field lacks that root, the plan uses length `3D`.
/// Accumulation happens in transform space, followed by one
/// inverse transform and the algebra crate's canonical polynomial division.
/// Response and challenge transforms are cached once for the entire matrix;
/// their storage is `(response.len() + challenges.len()) * domain_length`
/// field elements. Scratch is reused across rows; only the returned quotient
/// and division scratch allocate per row. This is a prover kernel, not a proof verifier.
pub struct TrinomialRelationQuotientBuilder<F: SmoothFftField, const D: usize, M: TrinomialModulus>
{
    polynomial: RelationPolynomial,
    domain: SmoothDomain<F>,
    workspace: FftWorkspace<F>,
    padded: Vec<F>,
    left: Vec<F>,
    sum: Vec<F>,
    response_transforms: Vec<Vec<F>>,
    challenge_transforms: Vec<Vec<F>>,
    modulus: PhantomData<M>,
}

impl<F: SmoothFftField + Debug, const D: usize, M: TrinomialModulus>
    TrinomialRelationQuotientBuilder<F, D, M>
{
    /// Prepare a plan and cache the operands common to every row.
    ///
    /// Rejects unsupported degree/field pairs before allocation. Rebuild the
    /// plan when the response or challenges change; row construction cannot
    /// accidentally mix cached and fresh common operands.
    pub fn new(
        response: &[TrinomialRing<F, D, M>],
        challenges: &[TrinomialRing<F, D, M>],
    ) -> Result<Self, AkitaError> {
        let polynomial = if M::MIDDLE_COEFFICIENT == 1 {
            RelationPolynomial::plus_trinomial(D)?
        } else {
            RelationPolynomial::minus_trinomial(D)?
        };
        let double_degree = checked::product([2, D]).ok_or_else(|| {
            AkitaError::InvalidSetup("trinomial convolution size overflow".into())
        })?;
        let length = if F::SMOOTH_SUBGROUP_ORDER.is_multiple_of(double_degree) {
            double_degree
        } else {
            checked::product([3, D]).ok_or_else(|| {
                AkitaError::InvalidSetup("trinomial convolution size overflow".into())
            })?
        };
        if !F::SMOOTH_SUBGROUP_ORDER.is_multiple_of(length) {
            return Err(AkitaError::InvalidSetup(
                "coefficient field does not support the trinomial convolution domain".into(),
            ));
        }
        let domain = SmoothDomain::new(primitive_nth_root::<F>(length), length);
        let mut workspace = domain.workspace();
        let mut padded = vec![F::zero(); length];
        let mut transform_operands = |operands: &[TrinomialRing<F, D, M>]| {
            operands
                .iter()
                .map(|operand| {
                    padded[..D].copy_from_slice(operand.coefficients());
                    let mut output = vec![F::zero(); length];
                    domain.forward_into(&padded, &mut output, &mut workspace);
                    output
                })
                .collect()
        };
        let response_transforms = transform_operands(response);
        let challenge_transforms = transform_operands(challenges);
        Ok(Self {
            polynomial,
            domain,
            workspace,
            padded,
            left: vec![F::zero(); length],
            sum: vec![F::zero(); length],
            response_transforms,
            challenge_transforms,
            modulus: PhantomData,
        })
    }

    /// Exact polynomial identity used by this plan.
    pub const fn polynomial(&self) -> RelationPolynomial {
        self.polynomial
    }

    /// Construct `Q` satisfying `sum_j A_j v_j - sum_t Y_t c_t = Phi_D Q`.
    ///
    /// Inputs use the same field, degree, and modulus, with challenges already
    /// embedded in the commitment ring. The result has exactly `D-1`
    /// coefficients. An invalid modular relation is rejected rather than
    /// returning a quotient that silently discards its nonzero remainder.
    pub fn build(
        &mut self,
        matrix_row: &[TrinomialRing<F, D, M>],
        images: &[TrinomialRing<F, D, M>],
    ) -> Result<Vec<F>, AkitaError> {
        if matrix_row.len() != self.response_transforms.len()
            || images.len() != self.challenge_transforms.len()
        {
            return Err(AkitaError::InvalidInput(
                "trinomial relation product lengths disagree".into(),
            ));
        }
        self.sum.fill(F::zero());
        let products = matrix_row
            .iter()
            .zip(&self.response_transforms)
            .map(|pair| (pair, false))
            .chain(
                images
                    .iter()
                    .zip(&self.challenge_transforms)
                    .map(|pair| (pair, true)),
            );
        for ((left, right), subtract) in products {
            self.padded[..D].copy_from_slice(left.coefficients());
            self.padded[D..].fill(F::zero());
            self.domain
                .forward_into(&self.padded, &mut self.left, &mut self.workspace);
            for ((sum, &left), &right) in self.sum.iter_mut().zip(&self.left).zip(right) {
                let right = if subtract { -right } else { right };
                *sum = left.mul_add(right, *sum);
            }
        }
        self.domain
            .inverse_into(&self.sum, &mut self.padded, &mut self.workspace);
        let (remainder, quotient) = TrinomialRing::<F, D, M>::reduce_product_with_quotient(
            &self.padded[..self.polynomial.product_coefficient_len()?],
        )
        .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        if remainder
            .coefficients()
            .iter()
            .any(|value| !value.is_zero())
        {
            return Err(AkitaError::InvalidInput(
                "trinomial relation has a nonzero remainder".into(),
            ));
        }
        self.polynomial
            .validate_coefficient_len(RelationCoefficientRole::Quotient, quotient.len())?;
        Ok(quotient)
    }
}

#[cfg(test)]
mod tests;
