//! Mixed native root public weights consumed by the shared Stage 2.

use super::Stage2RelationWeights;
use akita_error::AkitaError;
use akita_types::{native_root_relation::NativeRootWeights, AkitaExpandedSetup};
use jolt_field::Field;

impl<F: Field, E: Field> Stage2RelationWeights<F, E> for NativeRootWeights<E> {
    fn evaluate(
        &self,
        point: &[E],
        _setup: &AkitaExpandedSetup<F>,
        _alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError> {
        if setup_claim.is_some() {
            return Err(AkitaError::InvalidSetup(
                "native root weights require an explicit typed setup contribution".into(),
            ));
        }
        self.evaluate(point)
    }

    fn is_reduced_evaluation(&self) -> bool {
        true
    }
}

/// One fused relation oracle for ordinary rows and additional native root rows.
/// The native scalar claim must be added to the ordinary ring-switch claim.
/// Both sides address the same successor witness and use the same Stage 2 point.
pub struct MixedRootRelationWeights<'a, F: Field, E: Field> {
    pub ordinary: &'a dyn Stage2RelationWeights<F, E>,
    pub native: &'a NativeRootWeights<E>,
}

impl<F: Field, E: Field> Stage2RelationWeights<F, E> for MixedRootRelationWeights<'_, F, E> {
    fn evaluate(
        &self,
        point: &[E],
        setup: &AkitaExpandedSetup<F>,
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError> {
        Ok(self.ordinary.evaluate(point, setup, alpha, setup_claim)?
            + self.native.evaluate(point)?)
    }

    fn is_reduced_evaluation(&self) -> bool {
        true
    }
}
