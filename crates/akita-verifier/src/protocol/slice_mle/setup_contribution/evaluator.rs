use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBaseUnreduced};
use akita_types::AkitaExpandedSetup;

use crate::protocol::ring_switch::RelationMatrixEvaluator;

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_setup_contribution_direct<F, E, const D: usize>(
    relation_matrix_evaluator: &RelationMatrixEvaluator<E>,
    full_vec_randomness: &[E],
    alpha_pows_a: &[E],
    alpha_pows_b: &[E],
    alpha_pows_d: &[E],
    fold_gadget: &[F],
    setup: &AkitaExpandedSetup<F>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    if alpha_pows_a.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: alpha_pows_a.len(),
        });
    }
    validate_role_alpha_pows(
        relation_matrix_evaluator,
        alpha_pows_a,
        alpha_pows_b,
        alpha_pows_d,
    )?;
    let plan = relation_matrix_evaluator.setup_contribution_plan::<F>(
        full_vec_randomness,
        (!fold_gadget.is_empty()).then_some(fold_gadget),
    )?;
    plan.evaluate_direct::<F>(setup, alpha_pows_a, alpha_pows_b, alpha_pows_d)
}

fn validate_role_alpha_pows<E: FieldCore>(
    relation_matrix_evaluator: &RelationMatrixEvaluator<E>,
    alpha_pows_a: &[E],
    alpha_pows_b: &[E],
    alpha_pows_d: &[E],
) -> Result<(), AkitaError> {
    let d_a = relation_matrix_evaluator.role_dims.d_a();
    if alpha_pows_a.len() != d_a {
        return Err(AkitaError::InvalidSize {
            expected: d_a,
            actual: alpha_pows_a.len(),
        });
    }
    let d_d = relation_matrix_evaluator.role_dims.d_d();
    if alpha_pows_d.len() != d_d {
        return Err(AkitaError::InvalidSize {
            expected: d_d,
            actual: alpha_pows_d.len(),
        });
    }
    let d_b = relation_matrix_evaluator.role_dims.d_b();
    if alpha_pows_b.len() != d_b {
        return Err(AkitaError::InvalidSize {
            expected: d_b,
            actual: alpha_pows_b.len(),
        });
    }
    Ok(())
}
