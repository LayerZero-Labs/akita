use crate::protocol::ring_switch::RelationMatrixEvaluator;
use akita_algebra::offset_eq::{OffsetEqWindow, MAX_COMPACT_STRIDE_TERMS};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};

/// Compute the `r`-tail contribution.
///
/// Physical `r` addresses are mapped into the canonical opening domain before
/// applying their equality weights.
pub(crate) fn compute_r_contribution<F, E>(
    prepared: &RelationMatrixEvaluator<E>,
    full_vec_randomness: &[E],
    offset_r: usize,
    denom: E,
    r_gadget: &[F],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let levels = r_gadget.len();
    let rows = prepared.setup_rows()?;
    let terms = rows.checked_mul(levels).ok_or(AkitaError::InvalidProof)?;
    if terms > MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual: terms,
        });
    }
    // Share a bounded low equality table across every canonical r address
    // instead of recomputing a full-width equality product per (row, level).
    let eq_window = OffsetEqWindow::new(full_vec_randomness)?;
    let mut contribution = E::zero();
    for row_idx in 0..rows {
        let row_weight = prepared
            .eq_tau1
            .get(row_idx)
            .copied()
            .ok_or(AkitaError::InvalidProof)?;
        for (level_idx, &gadget) in r_gadget.iter().enumerate() {
            let physical_index = row_idx
                .checked_mul(levels)
                .and_then(|row| row.checked_add(level_idx))
                .and_then(|local| offset_r.checked_add(local))
                .ok_or(AkitaError::InvalidProof)?;
            let opening_index = akita_types::checked_opening_source_index(
                prepared.opening_source_len()?,
                physical_index,
            )?;
            contribution -=
                eq_window.eval(opening_index) * row_weight * E::lift_base(gadget) * denom;
        }
    }
    Ok(contribution)
}
