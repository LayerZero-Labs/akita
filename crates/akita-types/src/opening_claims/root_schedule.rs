use super::OpeningClaims;
use akita_error::AkitaError;
use akita_params::{FoldSchedule, GrindingPlan};

impl<E: Clone, C> OpeningClaims<'_, E, C> {
    /// Extension seam: select root query geometry while retaining the ordinary suffix cursor.
    /// Tracking issue: <https://github.com/LayerZero-Labs/akita/issues/285>.
    pub fn root_grinding_plan(
        &self,
        ordinary: GrindingPlan,
        schedule: &FoldSchedule,
    ) -> Result<GrindingPlan, AkitaError> {
        #[cfg(feature = "labinius-trinomial")]
        if let Some(super::GroupClaimKind::Binary { manifest, .. }) =
            self.groups().first().map(|group| group.kind())
        {
            let next = schedule.recursive_folds.first().ok_or_else(|| {
                AkitaError::InvalidSetup("binary root requires an ordinary successor".into())
            })?;
            let domain_len = akita_params::witness_commitment_domain_len(
                schedule.root.output_witness_len,
                next.params.d_a(),
            )?;
            return crate::binary_source::binary_root_grinding_plan(
                &ordinary,
                &[akita_error::checked::product([
                    manifest.descriptor().index_map().columns(),
                    manifest.descriptor().polynomials().len(),
                ])
                .ok_or(AkitaError::InvalidProof)?],
                domain_len,
                648,
            );
        }
        let _ = schedule;
        Ok(ordinary)
    }
}
