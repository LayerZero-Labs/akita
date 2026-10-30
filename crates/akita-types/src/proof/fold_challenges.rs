//! Canonical method-aware fold-challenge dispatch shared by prover and verifier.

use akita_challenges::{Challenges, FoldChallengeDrawDomain, FoldDraw};
use akita_error::AkitaError;
use jolt_field::{ExtField, Field};

use crate::CoefficientPackingChallenges;
use akita_params::SubringCoefficientPackingGeometry;

/// Runtime value carried by one of Akita's two opening methods.
///
/// The schedule chooses an [`OpeningMethod`](akita_params::OpeningMethod); this family preserves that same
/// method distinction while each protocol stage supplies its own payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpeningFamily<Trace, Packing> {
    /// Value belonging to the full-A evaluation-trace method.
    EvaluationTrace(Trace),
    /// Value belonging to subring coefficient packing.
    SubringCoefficientPacking(Packing),
}

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
///
/// Returns `None` when the group's operator-norm pool runs out; see
/// [`FoldDraw::draw_fold_challenges`].
pub fn draw_group_fold_challenges<F, E, D>(
    draw: &mut D,
    params: &akita_params::GroupOpenPhaseParams,
    group_index: usize,
    num_claims: usize,
) -> Result<Option<GroupFoldChallenges>, AkitaError>
where
    F: Field,
    E: ExtField<F>,
    D: FoldDraw,
{
    let family = params.fold_challenge_family()?;
    let packing = match family.domain() {
        FoldChallengeDrawDomain::EvaluationTrace => None,
        FoldChallengeDrawDomain::SubringCoefficientPacking {
            challenge_subring_dimension,
        } => {
            let geometry = SubringCoefficientPackingGeometry::try_new(
                E::DEGREE,
                params.inner_commit_matrix_params().ring_dimension(),
                challenge_subring_dimension,
            )?;
            if family.config() != geometry.fold_challenge_config() {
                return Err(AkitaError::InvalidSetup(
                    "coefficient-packing challenge config is not the audited production family"
                        .into(),
                ));
            }
            Some(geometry)
        }
    };
    let Some(challenges) =
        draw.draw_fold_challenges(&family, group_index, params.num_live_blocks(), num_claims)?
    else {
        return Ok(None);
    };
    Ok(Some(match packing {
        None => OpeningFamily::EvaluationTrace(challenges),
        Some(geometry) => OpeningFamily::SubringCoefficientPacking(
            CoefficientPackingChallenges::new(geometry, challenges)?,
        ),
    }))
}
