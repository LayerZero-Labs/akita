use super::FoldVerifyOutput;
use akita_error::AkitaError;
use akita_params::{
    BasisMode, CommittedGroupParams, FoldParams, OpeningClaimsLayout, TerminalFoldParams,
};
use akita_serialization::AkitaSerialize;
use akita_types::Commitment;
use akita_types::{AkitaVerifierSetup, FpExtEncoding, OpeningClaims};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

/// Extension seam: dispatch a root by its declared group relation kind.
/// Tracking issue: https://github.com/LayerZero-Labs/akita/issues/285.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_root<F, E>(
    setup: &AkitaVerifierSetup<F>,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    claims: &OpeningClaims<'_, E, &Commitment<F>>,
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    root_lp: &CommittedGroupParams,
    next_fold_params: Option<&FoldParams>,
    terminal: &TerminalFoldParams,
) -> Result<FoldVerifyOutput<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    match claims.groups().first().map(|group| group.kind()) {
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
            )
        }
    }
}
