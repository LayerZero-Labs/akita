//! Verifier-side sumcheck instance for the extension-opening reduction (EOR).
//!
//! Both the root (multi-row incidence) and recursive (single-row) sites drive
//! their EOR sumcheck through this one instance via the generic
//! [`SumcheckInstanceVerifier`] driver. The final oracle is the batched
//! ring-subfield opening of the witness rows recovered from the `y_ring` proof
//! data, weighted by the transparent tensor-equality factor evaluated at the
//! sumcheck point.
//!
//! This is the non-zk realization. In zk mode the EOR final relation consumes
//! the post-round y-ring hiding masks, which are a shared resource with the
//! downstream ring-switch binding (`zk_relation_claim_mask_from_y_masks`), so
//! that path keeps its final relation in the outer level flow rather than
//! inside the sumcheck driver.

use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, ExtField, FieldCore, FromPrimitiveInt, Invertible};
use akita_sumcheck::SumcheckInstanceVerifier;
use akita_types::{
    recover_ring_subfield_inner_product, tensor_equality_factor_eval_at_point,
    RingSubfieldEncoding, EXTENSION_OPENING_REDUCTION_DEGREE,
};

/// Closure that recomputes the inner ring-slot reduction `v` at the sumcheck
/// point `rho`. The root and recursive sites capture their own opening-point
/// preparation (`prepare_root_opening_point_ext` / `prepare_recursive_*`), so
/// the generic instance stays free of the divergent point-prep bounds.
type InnerReductionFn<'a, F, C, const D: usize> =
    Box<dyn Fn(&[C]) -> Result<CyclotomicRing<F, D>, AkitaError> + Send + Sync + 'a>;

struct EorRow<'a, F: FieldCore, C: FieldCore, const D: usize> {
    y_ring: &'a CyclotomicRing<F, D>,
    /// Tail (post-split) of the row's padded opening point; the tensor factor
    /// is evaluated against this and the batch challenges `eta`.
    point_tail: Vec<C>,
}

/// EOR sumcheck verifier instance shared by the root and recursive levels.
pub(crate) struct ExtensionOpeningReductionVerifier<'a, F: FieldCore, C: FieldCore, const D: usize>
{
    num_rounds: usize,
    input_claim: C,
    eta: Vec<C>,
    rows: Vec<EorRow<'a, F, C, D>>,
    inner_reduction: InnerReductionFn<'a, F, C, D>,
}

impl<'a, F: FieldCore, C: FieldCore, const D: usize>
    ExtensionOpeningReductionVerifier<'a, F, C, D>
{
    /// Build the instance from the per-row `(y_ring, point_tail)` data, the
    /// batch challenges `eta`, and the point-prep closure.
    pub(crate) fn new(
        num_rounds: usize,
        input_claim: C,
        eta: Vec<C>,
        rows: Vec<(&'a CyclotomicRing<F, D>, Vec<C>)>,
        inner_reduction: InnerReductionFn<'a, F, C, D>,
    ) -> Self {
        Self {
            num_rounds,
            input_claim,
            eta,
            rows: rows
                .into_iter()
                .map(|(y_ring, point_tail)| EorRow { y_ring, point_tail })
                .collect(),
            inner_reduction,
        }
    }
}

impl<F, C, const D: usize> SumcheckInstanceVerifier<C>
    for ExtensionOpeningReductionVerifier<'_, F, C, D>
where
    F: FieldCore + FromPrimitiveInt + Invertible,
    C: FieldCore + RingSubfieldEncoding<F> + ExtField<F>,
{
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        EXTENSION_OPENING_REDUCTION_DEGREE
    }

    fn input_claim(&self) -> C {
        self.input_claim
    }

    fn expected_output_claim(&self, challenges: &[C]) -> Result<C, AkitaError> {
        let inner_reduction = (self.inner_reduction)(challenges)?;
        let mut acc = C::zero();
        for row in &self.rows {
            let opening =
                recover_ring_subfield_inner_product::<F, C, D>(row.y_ring, &inner_reduction)?;
            let factor = tensor_equality_factor_eval_at_point::<F, C>(
                &row.point_tail,
                &self.eta,
                challenges,
            )?;
            acc += opening * factor;
        }
        Ok(acc)
    }
}
