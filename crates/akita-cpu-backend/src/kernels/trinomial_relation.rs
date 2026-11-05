//! Quotient witnesses for trinomial A relations.
//!
//! This constructs the polynomial quotient of `A v - Y c`, before the
//! relation-evaluation challenge is sampled. It neither commits that witness
//! nor authenticates a binary opening. The root protocol owns those phases.

use std::{fmt::Debug, marker::PhantomData};

use akita_algebra::{TrinomialField, TrinomialModulus, TrinomialRing};

use jolt_algebra::fft::{FftWorkspace, SmoothDomain};
use akita_error::AkitaError;
use jolt_algebra::checked;
use akita_types::{RelationCoefficientRole, RelationPolynomial};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Every transform's input, output, and workspace come from one domain of
/// length `n`, so the shared FFT's length checks cannot fail.
const DOMAIN_SIZED: &str = "the quotient builder sizes every transform to its domain";

/// Reusable linear-convolution plan for a row of `A v = Y c` in a trinomial ring.
///
/// A length-`2D` transform contains the entire degree-`2D-2` product without
/// cyclic aliasing. If the field lacks that root, the plan uses length `3D`.
/// Accumulation happens in transform space, followed by one
/// inverse transform and the algebra crate's canonical polynomial division.
/// Response and challenge transforms are cached once for the entire matrix;
/// their storage is `(response.len() + challenges.len()) * domain_length`
/// field elements. Operand transforms and their products run in parallel
/// under the `parallel` feature; field sums are exact, so the result does not
/// depend on the schedule. This is a prover kernel, not a proof verifier.
pub struct TrinomialRelationQuotientBuilder<F: TrinomialField, const D: usize, M: TrinomialModulus>
{
    polynomial: RelationPolynomial,
    domain: SmoothDomain<F>,
    workspace: FftWorkspace<F>,
    padded: Vec<F>,
    sum: Vec<F>,
    response_transforms: Vec<Vec<F>>,
    challenge_transforms: Vec<Vec<F>>,
    modulus: PhantomData<M>,
}

impl<F: TrinomialField + Debug, const D: usize, M: TrinomialModulus>
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
        let root = F::primitive_root(length)
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let domain = SmoothDomain::new(root, length)
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let response_transforms = transform_operands(&domain, response);
        let challenge_transforms = transform_operands(&domain, challenges);
        Ok(Self {
            polynomial,
            workspace: domain.workspace(),
            padded: vec![F::zero(); length],
            sum: vec![F::zero(); length],
            domain,
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
        let (remainder, quotient) = self.decompose(matrix_row, images)?;
        if remainder
            .coefficients()
            .iter()
            .any(|value| !value.is_zero())
        {
            return Err(AkitaError::InvalidInput(
                "trinomial relation has a nonzero remainder".into(),
            ));
        }
        Ok(quotient)
    }

    /// Extension seam: decompose a row into its native remainder and quotient.
    ///
    /// Tracking issue: <https://github.com/LayerZero-Labs/akita/issues/285>.
    ///
    /// Returns `(r, Q)` satisfying
    /// `sum_j A_j v_j - sum_t Y_t c_t = r + Phi_D Q` over the coefficient field.
    /// A caller proving a relation modulo another commitment modulus may use
    /// the remainder as an explicit bounded residual. This method does not
    /// assert that the remainder vanishes; [`Self::build`] enforces that
    /// condition for an exact native-ring relation.
    pub fn decompose(
        &mut self,
        matrix_row: &[TrinomialRing<F, D, M>],
        images: &[TrinomialRing<F, D, M>],
    ) -> Result<(TrinomialRing<F, D, M>, Vec<F>), AkitaError> {
        if matrix_row.len() != self.response_transforms.len()
            || images.len() != self.challenge_transforms.len()
        {
            return Err(AkitaError::InvalidInput(
                "trinomial relation product lengths disagree".into(),
            ));
        }
        let added = transform_products(&self.domain, matrix_row, &self.response_transforms);
        let subtracted = transform_products(&self.domain, images, &self.challenge_transforms);
        for ((sum, &added), &subtracted) in self.sum.iter_mut().zip(&added).zip(&subtracted) {
            *sum = added - subtracted;
        }
        self.domain
            .inverse_into(&self.sum, &mut self.padded, &mut self.workspace)
            .expect(DOMAIN_SIZED);
        let (remainder, quotient) = TrinomialRing::<F, D, M>::reduce_product_with_quotient(
            &self.padded[..self.polynomial.product_coefficient_len()?],
        )
        .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        self.polynomial
            .validate_coefficient_len(RelationCoefficientRole::Quotient, quotient.len())?;
        Ok((remainder, quotient))
    }
}

/// Transform scratch owned by one worker.
struct TransformScratch<F> {
    workspace: FftWorkspace<F>,
    padded: Vec<F>,
    transform: Vec<F>,
}

impl<F: TrinomialField + Debug> TransformScratch<F> {
    fn new(domain: &SmoothDomain<F>) -> Self {
        Self {
            workspace: domain.workspace(),
            padded: vec![F::zero(); domain.n],
            transform: vec![F::zero(); domain.n],
        }
    }

    /// Transform one operand; the padded tail beyond `D` is never written.
    fn forward<const D: usize, M: TrinomialModulus>(
        &mut self,
        domain: &SmoothDomain<F>,
        operand: &TrinomialRing<F, D, M>,
    ) -> &[F] {
        self.padded[..D].copy_from_slice(operand.coefficients());
        domain
            .forward_into(&self.padded, &mut self.transform, &mut self.workspace)
            .expect(DOMAIN_SIZED);
        &self.transform
    }
}

/// Forward transforms of the zero-padded operands, in operand order.
fn transform_operands<F: TrinomialField + Debug, const D: usize, M: TrinomialModulus>(
    domain: &SmoothDomain<F>,
    operands: &[TrinomialRing<F, D, M>],
) -> Vec<Vec<F>> {
    let transform =
        |scratch: &mut TransformScratch<F>, operand| scratch.forward(domain, operand).to_vec();
    #[cfg(feature = "parallel")]
    let transforms = operands
        .par_iter()
        .map_init(|| TransformScratch::new(domain), transform)
        .collect();
    #[cfg(not(feature = "parallel"))]
    let transforms = {
        let mut scratch = TransformScratch::new(domain);
        operands
            .iter()
            .map(|operand| transform(&mut scratch, operand))
            .collect()
    };
    transforms
}

/// `sum_j transform(left_j) * right_j` in transform space.
fn transform_products<F: TrinomialField + Debug, const D: usize, M: TrinomialModulus>(
    domain: &SmoothDomain<F>,
    lefts: &[TrinomialRing<F, D, M>],
    rights: &[Vec<F>],
) -> Vec<F> {
    let accumulate = |(mut scratch, mut sum): (TransformScratch<F>, Vec<F>),
                      (left, right): (&TrinomialRing<F, D, M>, &Vec<F>)| {
        for ((sum, &left), &right) in sum.iter_mut().zip(scratch.forward(domain, left)).zip(right) {
            *sum = left.mul_add(right, *sum);
        }
        (scratch, sum)
    };
    let identity = || (TransformScratch::new(domain), vec![F::zero(); domain.n]);
    #[cfg(feature = "parallel")]
    let sum = lefts
        .par_iter()
        .zip(rights)
        .fold(identity, accumulate)
        .map(|(_, sum)| sum)
        .reduce(
            || vec![F::zero(); domain.n],
            |mut lhs, rhs| {
                for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
                    *lhs += rhs;
                }
                lhs
            },
        );
    #[cfg(not(feature = "parallel"))]
    let (_, sum) = lefts.iter().zip(rights).fold(identity(), accumulate);
    sum
}

#[cfg(test)]
mod tests;
