use super::*;
use akita_algebra::ring::eval_flat_ring_at_pows_fast;

#[cfg(test)]
impl<E: FieldCore> SetupContributionPlan<E> {
    pub(crate) fn evaluate_direct_by_rows<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
        d_a: usize,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let alpha = alpha_pows_a
            .get(1)
            .or_else(|| alpha_pows_b.get(1))
            .or_else(|| alpha_pows_d.get(1))
            .copied()
            .ok_or(AkitaError::InvalidProof)?;
        let weights = self.materialize_setup_index_weights(alpha)?;
        let base_d = self.projection_geometry.base_ring_dim();
        let base_pows = alpha_pows_d.get(..base_d).ok_or(AkitaError::InvalidProof)?;
        let view = setup
            .shared_matrix
            .ring_view_dyn(1, self.required(), base_d)?;
        let row = view.row_flat(0)?;
        let mut acc = E::zero();
        for (setup_idx, &weight) in weights.iter().enumerate() {
            if !weight.is_zero() {
                let start = setup_idx
                    .checked_mul(base_d)
                    .ok_or(AkitaError::InvalidProof)?;
                let end = start.checked_add(base_d).ok_or(AkitaError::InvalidProof)?;
                let ring = row.get(start..end).ok_or(AkitaError::InvalidProof)?;
                acc += weight * eval_flat_ring_at_pows_fast::<F, E>(ring, base_pows);
            }
        }
        let _ = (alpha_pows_a, alpha_pows_b, d_a);
        Ok(acc)
    }
}
