//! Fold stage checks written once over [`ClaimDomain`], so the clear
//! verifier ([`jolt_transcript::ClearDomain`]), the masked verifier and the
//! prover's auxiliary values all run the same formula
//! (`specs/whiteout-akita.md` §4.4–4.5).

use akita_error::AkitaError;
use akita_params::DigitRangePlan;
use jolt_algebra::Field;
use jolt_transcript::ClaimDomain;

/// The batched range-leaf polynomial `Σ_{k≤4} c_k·X^k`, zero-padded to
/// [`DigitRangePlan::MAX_LEAF_DEGREE`], the largest leaf of any supported basis.
pub struct RangeLeaf<E>([E; DigitRangePlan::MAX_LEAF_DEGREE + 1]);

impl<E: Field> RangeLeaf<E> {
    /// The leaf of `plan` with `coefficients` in ascending degree.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidSetup`] if `coefficients` exceed the degree of
    /// `plan`'s leaf ([`DigitRangePlan::leaf_degree`]).
    pub fn new(plan: DigitRangePlan, coefficients: &[E]) -> Result<Self, AkitaError> {
        let mut padded = [E::zero(); DigitRangePlan::MAX_LEAF_DEGREE + 1];
        padded
            .get_mut(..coefficients.len())
            .filter(|_| coefficients.len() <= plan.leaf_degree() + 1)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "stage-1 range leaf has {} coefficients; the basis-{} leaf has degree {}",
                    coefficients.len(),
                    plan.basis(),
                    plan.leaf_degree()
                ))
            })?
            .copy_from_slice(coefficients);
        Ok(Self(padded))
    }
}

/// The stage-1 leaf check `claim = Σ_{k≤4} c_k·ri^k` on the range image `ri`,
/// written once for every claim domain (`specs/whiteout-akita.md` §4.4). It
/// lowers as `u = ri²` and `u·(c₄u + c₃ri + c₂) = claim − c₁ri − c₀`: two
/// products and one auxiliary slot in a masked domain, a comparison in the
/// clear one.
pub fn check_range_leaf<E: Field, D: ClaimDomain<E>>(
    domain: &mut D,
    leaf: &RangeLeaf<E>,
    range_image: D::Value,
    claim: D::Value,
) -> Result<(), D::Error> {
    let [c0, c1, c2, c3, c4] = leaf.0;
    let square = domain.product(range_image.clone(), range_image.clone())?;
    let factor = domain.add(
        &domain.add(&domain.scale(&square, c4), &domain.scale(&range_image, c3)),
        &domain.constant(c2),
    );
    let target = domain.add(
        &domain.add(&claim, &domain.scale(&range_image, -c1)),
        &domain.constant(-c0),
    );
    domain.product_into(square, factor, target)
}

/// The public weights of the stage-2 oracle `w·(B(w+1) + R)` at one point.
/// `B` collects both `w(w+1)` terms (the batched virtual claim and the
/// compression binary constraint); `R` collects every term linear in `w`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2OracleWeights<E> {
    /// `B`, the weight of `w(w+1)`.
    pub binary: E,
    /// `R`, the weight of `w`.
    pub linear: E,
}

/// The stage-2 output check `claim = w·(B(w+1) + R)`, written once for every
/// claim domain (`specs/whiteout-akita.md` §4.4): one product row with no
/// auxiliary slot in a masked domain, a comparison in the clear one.
pub fn check_stage2_oracle<E: Field, D: ClaimDomain<E>>(
    domain: &mut D,
    weights: &Stage2OracleWeights<E>,
    witness_eval: D::Value,
    claim: D::Value,
) -> Result<(), D::Error> {
    let factor = domain.add(
        &domain.scale(&witness_eval, weights.binary),
        &domain.constant(weights.binary + weights.linear),
    );
    domain.product_into(witness_eval, factor, claim)
}

#[cfg(test)]
#[path = "stage_checks/tests.rs"]
mod tests;
