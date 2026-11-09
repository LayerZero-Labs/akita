//! Host-side evaluation of setup functionals from the materialized prefix they use.
use super::*;
use crate::AkitaExpandedSetup;
use akita_algebra::offset_eq::{materialize_eq_tensor_left, OffsetEqWindow};
use akita_algebra::row_batch::dot_base_rows_batched;
use jolt_field::MulBaseUnreduced;

impl<E: Field> SetupQuery<E> {
    /// Compute an advice answer without allocating the unused tail of a setup domain.
    /// This does not authenticate the answer; deferred acceptance still requires the setup batch.
    pub fn evaluate_materialized<F>(
        &self,
        polynomial: &SetupPolynomialId,
        setup: &AkitaExpandedSetup<F>,
    ) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: crate::FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F>,
    {
        polynomial.validate::<F, E>(setup)?;
        self.validate(polynomial)?;
        let source = setup.shared_matrix().as_field_slice();
        if let Query::Evaluation(point) = &self.0 {
            let source = source
                .get(..polynomial.field_len())
                .ok_or(AkitaError::InvalidProof)?;
            let weights = EqPolynomial::evals(point)?;
            return Ok(E::dot_product(
                &weights,
                &source.iter().copied().map(E::lift_base).collect::<Vec<_>>(),
            ));
        }
        let mut value = E::zero();
        for term in self.native_terms()? {
            let dimension = term.coefficient.len();
            let active = term.tensors.iter().try_fold(0usize, |max_end, tensor| {
                let end = tensor
                    .axes
                    .iter()
                    .try_fold(tensor.left_offset, |offset, axis| {
                        checked::mul_add(axis.left_stride, axis.len.checked_sub(1)?, offset)
                    })
                    .and_then(|last| last.checked_add(1))
                    .ok_or(AkitaError::InvalidProof)?;
                Ok::<_, AkitaError>(max_end.max(end))
            })?;
            let count = checked::product([active, dimension]).ok_or(AkitaError::InvalidProof)?;
            let prefix = source.get(..count).ok_or(AkitaError::InvalidProof)?;
            let equality = OffsetEqWindow::new(&term.point)?;
            let index = materialize_eq_tensor_left(&equality, &term.tensors, active)?;
            let columns =
                dot_base_rows_batched(&term.coefficient, prefix.chunks_exact(dimension).map(Ok))?;
            value += E::dot_product(&index, &columns);
        }
        Ok(value)
    }
}
