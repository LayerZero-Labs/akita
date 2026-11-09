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
        native_root: bool,
    ) -> Result<GrindingPlan, AkitaError> {
        #[cfg(feature = "labinius-trinomial")]
        if let Some(super::GroupClaimKind::Binary { manifest, .. }) =
            self.groups().first().map(|group| group.kind())
        {
            manifest.field_profile().admit_root(native_root)?;
            let next = schedule.recursive_folds.first().ok_or_else(|| {
                AkitaError::InvalidSetup("binary root requires an ordinary successor".into())
            })?;
            let domain_len = akita_params::witness_commitment_domain_len(
                schedule.root.output_witness_len,
                next.params.d_a(),
            )?;
            let mut coordinates = Vec::new();
            let binary_count = if self.num_groups() == 8 { 4 } else { 1 };
            for group in self.groups().iter().take(binary_count) {
                let super::GroupClaimKind::Binary {
                    manifest: source, ..
                } = group.kind()
                else {
                    return Err(AkitaError::InvalidProof);
                };
                if source.field_profile() != manifest.field_profile() {
                    return Err(AkitaError::InvalidProof);
                }
                coordinates.push(
                    akita_error::checked::product([
                        source.descriptor().index_map().columns(),
                        source.descriptor().polynomials().len(),
                    ])
                    .ok_or(AkitaError::InvalidProof)?,
                );
            }
            return crate::binary_source::binary_root_grinding_plan(
                &ordinary,
                manifest.field_profile(),
                &coordinates,
                domain_len,
                648,
            );
        }
        let _ = (schedule, native_root);
        Ok(ordinary)
    }
}
