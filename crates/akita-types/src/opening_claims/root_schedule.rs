use super::OpeningClaims;
use akita_error::AkitaError;
use akita_params::{FoldSchedule, GrindingPlan};

impl<E: Clone, C> OpeningClaims<'_, E, C> {
    /// Extension seam: select root query geometry while retaining the ordinary suffix cursor.
    /// Tracking issue: https://github.com/LayerZero-Labs/akita/issues/285.
    pub fn root_grinding_plan(
        &self,
        ordinary: GrindingPlan,
        _schedule: &FoldSchedule,
    ) -> Result<GrindingPlan, AkitaError> {
        Ok(ordinary)
    }
}
