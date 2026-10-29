//! Public-weight boundary consumed by the ordinary fused Stage 2.

use super::{PreparedRelationGroups, RelationMatrixEvaluator};
use akita_error::AkitaError;
use akita_types::{AkitaExpandedSetup, FpExtEncoding};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};

/// Extension seam: relation weights for the shared Stage 2, including groups
/// with different native polynomial identities.
/// <https://github.com/LayerZero-Labs/akita/issues/285>
///
/// Implementations derive weights from the fixed statement and public
/// challenges. `setup_claim`, when present, is authenticated by Stage 3.
/// Witness values must never determine these weights.
pub trait Stage2RelationWeights<F: Field, E: Field>: Sync {
    fn evaluate(
        &self,
        point: &[E],
        setup: &AkitaExpandedSetup<F>,
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError>;

    fn is_reduced_evaluation(&self) -> bool;
}

impl<F, E> Stage2RelationWeights<F, E> for RelationMatrixEvaluator<E>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F>,
{
    fn evaluate(
        &self,
        point: &[E],
        setup: &AkitaExpandedSetup<F>,
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError> {
        match setup_claim {
            Some(claim) => self.eval_flat_at_point_with_deferred_setup::<F>(point, alpha, claim),
            None => self.eval_flat_at_point::<F>(point, setup, alpha),
        }
    }

    fn is_reduced_evaluation(&self) -> bool {
        matches!(self.groups, PreparedRelationGroups::ReducedEvaluation(_))
    }
}
