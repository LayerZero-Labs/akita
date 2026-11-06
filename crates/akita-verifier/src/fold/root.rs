use super::FoldVerifyOutput;
use akita_error::AkitaError;
use akita_params::{
    BasisMode, CommittedGroupParams, FoldParams, OpeningClaimsLayout, TerminalFoldParams,
};
use akita_serialization::AkitaSerialize;
use akita_transcript::ChallengeCodec;
use akita_types::CommittedGroup;
use akita_types::{AkitaVerifierSetup, FpExtEncoding, OpeningClaims};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

/// Extension seam: dispatch a root by its declared group relation kind.
/// Tracking issue: <https://github.com/LayerZero-Labs/akita/issues/285>.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_root<F, E>(
    setup: &AkitaVerifierSetup<F>,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    claims: &OpeningClaims<'_, E, &CommittedGroup<F>>,
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    root_lp: &CommittedGroupParams,
    next_fold_params: Option<&FoldParams>,
    terminal: &TerminalFoldParams,
    setup_policy: &mut crate::SetupPolicy<'_, '_, E>,
) -> Result<FoldVerifyOutput<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Ring
        + AkitaSerialize
        + MulBaseUnreduced<F>
        + ChallengeCodec,
{
    match claims.groups().first().map(|group| group.kind()) {
        #[cfg(feature = "labinius-trinomial")]
        Some(akita_types::opening_claims::GroupClaimKind::Binary { .. }) => {
            if basis != BasisMode::Lagrange {
                return Err(AkitaError::InvalidProof);
            }
            if claims.num_groups() == 8 {
                return super::research_batch::verify_root(
                    setup, grinding, claims, opening_batch, root_lp, next_fold_params,
                );
            }
            super::binary_root::verify_root(
                setup,
                grinding,
                claims,
                opening_batch,
                root_lp,
                next_fold_params,
            )
        }
        #[cfg(feature = "labinius-trinomial")]
        Some(akita_types::opening_claims::GroupClaimKind::ResearchOrdinary { .. }) => {
            Err(AkitaError::InvalidInput("research ordinary requires its batch root".into()))
        }
        Some(akita_types::opening_claims::GroupClaimKind::Ordinary) | None => {
            super::ordinary_root::verify_root(
                setup,
                grinding,
                claims,
                opening_batch,
                basis,
                root_lp,
                next_fold_params,
                terminal,
                setup_policy,
            )
        }
        // A dependency may expose another kind without enabling this crate's handler.
        #[allow(unreachable_patterns)]
        _ => Err(AkitaError::InvalidInput(
            "root group kind is not enabled in this protocol engine".into(),
        )),
    }
}
