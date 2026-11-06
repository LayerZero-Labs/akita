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
        research_native: bool,
    ) -> Result<GrindingPlan, AkitaError> {
        #[cfg(feature = "labinius-trinomial")]
        if let Some(super::GroupClaimKind::Binary { manifest, .. }) =
            self.groups().first().map(|group| group.kind())
        {
            manifest.field_profile().admit_root(research_native)?;
            let next = schedule.recursive_folds.first().ok_or_else(|| {
                AkitaError::InvalidSetup("binary root requires an ordinary successor".into())
            })?;
            let domain_len = akita_params::witness_commitment_domain_len(
                schedule.root.output_witness_len,
                next.params.d_a(),
            )?;
            if self.num_groups() == 8 {
                if !research_native || next.params.setup_prefix().is_some() {
                    return Err(AkitaError::UnsupportedSchedule(
                        "research batch requires explicit direct native admission".into(),
                    ));
                }
                let mut coordinates = Vec::with_capacity(4);
                for (index, group) in self.groups().iter().take(4).enumerate() {
                    let super::GroupClaimKind::Binary {
                        manifest, claim, ..
                    } = group.kind()
                    else {
                        return Err(AkitaError::InvalidProof);
                    };
                    if manifest.field_profile()
                        != akita_params::binary_setup::BinaryFieldProfile::P64
                        || manifest.descriptor().polynomials().len() != 1
                        || (index != 0 && claim.is_some())
                    {
                        return Err(AkitaError::InvalidProof);
                    }
                    coordinates.push(manifest.descriptor().index_map().columns());
                }
                // Direct setup consumes no root Stage 3. Keep every suffix run.
                let suffix = GrindingPlan::new(
                    ordinary
                        .runs()
                        .iter()
                        .copied()
                        .filter(|run| run.site().level() != 0)
                        .collect(),
                    manifest.field_profile().challenge_order()?,
                )?;
                return crate::binary_source::binary_root_grinding_plan(
                    &suffix,
                    manifest.field_profile(),
                    &coordinates,
                    domain_len,
                    648,
                );
            }
            return crate::binary_source::binary_root_grinding_plan(
                &ordinary,
                manifest.field_profile(),
                &[akita_error::checked::product([
                    manifest.descriptor().index_map().columns(),
                    manifest.descriptor().polynomials().len(),
                ])
                .ok_or(AkitaError::InvalidProof)?],
                domain_len,
                648,
            );
        }
        let _ = (schedule, research_native);
        Ok(ordinary)
    }
}
