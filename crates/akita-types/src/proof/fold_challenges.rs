//! Canonical method-aware fold-challenge dispatch shared by prover and verifier.

use akita_challenges::{Challenges, FoldChallengeDrawDomain, FoldDraw};
use akita_field::{AkitaError, ExtField, FieldCore};

use crate::{
    CoefficientPackingChallenges, InnerCommitSecurityRoute, LevelParamsLike, OpeningFamily,
    OpeningMethod, SubringCoefficientPackingGeometry,
};

/// One sampled fold challenge with its method-specific algebraic views.
pub type GroupFoldChallenges = OpeningFamily<Challenges, CoefficientPackingChallenges>;

impl OpeningFamily<Challenges, CoefficientPackingChallenges> {
    /// Challenges acting in the committed group's ambient A ring.
    #[must_use]
    pub const fn ambient_a(&self) -> &Challenges {
        match self {
            Self::EvaluationTrace(challenges) => challenges,
            Self::SubringCoefficientPacking(challenges) => challenges.ambient_a(),
        }
    }
}

/// Draw one authenticated group's fold challenges under its scheduled opening
/// method and certified A-security route.
pub fn draw_group_fold_challenges<F, E, D>(
    draw: &mut D,
    params: &(impl LevelParamsLike + ?Sized),
    group_index: usize,
    num_claims: usize,
    grind_nonce: u32,
) -> Result<GroupFoldChallenges, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
    D: FoldDraw,
{
    let d_a = params.inner_commit_matrix_params().ring_dimension();
    let config = params.fold_challenge_config();
    match params.opening_method() {
        OpeningMethod::EvaluationTrace => {
            let rejection = matches!(
                params.inner_commit_matrix_params().security_route(),
                InnerCommitSecurityRoute::L2 { .. }
            )
            .then(|| akita_challenges::selective_l2_operator_norm_rejection(d_a, &config))
            .flatten();
            draw.draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::EvaluationTrace,
                d_a,
                group_index,
                params.num_live_blocks(),
                num_claims,
                &config,
                grind_nonce,
                rejection,
            )
            .map(OpeningFamily::EvaluationTrace)
        }
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } => {
            if !matches!(
                params.inner_commit_matrix_params().security_route(),
                InnerCommitSecurityRoute::Linf(_)
            ) {
                return Err(AkitaError::InvalidSetup(
                    "coefficient packing requires an L-infinity A security route".into(),
                ));
            }
            let geometry = SubringCoefficientPackingGeometry::try_new(
                E::EXT_DEGREE,
                d_a,
                challenge_subring_dimension,
            )?;
            if config != geometry.fold_challenge_config() {
                return Err(AkitaError::InvalidSetup(
                    "coefficient-packing challenge config is not the audited production family"
                        .into(),
                ));
            }
            let subring = draw.draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::SubringCoefficientPacking {
                    challenge_subring_dimension,
                },
                challenge_subring_dimension,
                group_index,
                params.num_live_blocks(),
                num_claims,
                &config,
                grind_nonce,
                None,
            )?;
            Ok(OpeningFamily::SubringCoefficientPacking(
                CoefficientPackingChallenges::new(geometry, subring)?,
            ))
        }
    }
}
