//! Fold-challenge draw for every opening group of one recursive level.

use akita_challenges::VerifierFoldDraw;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::GrindingReplay;
use akita_types::{
    draw_group_fold_challenges, CommittedGroupParams, GroupFoldChallenges, OpeningClaimsLayout,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// Spongefish replay of all sparse fold roots for one recursive level.
pub(crate) fn derive_multi_group_stage1_challenges<F, E>(
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    level: u32,
    opening_batch: &OpeningClaimsLayout,
    lp: &CommittedGroupParams,
) -> Result<Vec<GroupFoldChallenges>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    E: ExtField<F>,
{
    let mut group_challenges = Vec::with_capacity(opening_batch.num_groups());
    for group_index in 0..opening_batch.num_groups() {
        let group_lp = lp.group_params_geometry(opening_batch, group_index)?;
        let k_g = opening_batch.group_layout(group_index)?.num_polynomials();
        let group = u32::try_from(group_index).map_err(|_| AkitaError::InvalidProof)?;
        let drawn = {
            let mut live = VerifierFoldDraw::new(grinding.state_mut(), level, group);
            draw_group_fold_challenges::<F, E, _>(&mut live, &group_lp, group_index, k_g)?
        };
        let coordinate_count = group_lp
            .num_live_blocks()
            .checked_mul(k_g)
            .ok_or(AkitaError::InvalidProof)?;
        grinding.record_fold_challenges(level, group, coordinate_count)?;
        group_challenges.push(drawn);
    }
    Ok(group_challenges)
}
