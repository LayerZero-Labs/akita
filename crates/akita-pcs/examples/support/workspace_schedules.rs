//! Workspace-only schedule loading for examples, benches, and tests.

use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::FpExtEncoding;
use jolt_field::{CanonicalEncoding, ExtField, Field, Fold, PseudoMersenne, Ring, Unreduced};

/// Extension used only by repository-owned development targets.
pub(crate) trait WorkspaceScheduleArtifactExt: Sized {
    fn from_workspace_schedule_artifact() -> Result<Self, AkitaError>;
}

impl<Cfg> WorkspaceScheduleArtifactExt for AkitaCommitmentScheme<Cfg>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field + CanonicalEncoding + Unreduced + PseudoMersenne + Valid + AkitaSerialize,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: ExtField<Cfg::Field> + Ring + Unreduced + Fold + AkitaSerialize,
{
    fn from_workspace_schedule_artifact() -> Result<Self, AkitaError> {
        Self::new(akita_config::test_support::workspace_schedule_catalog::<Cfg>()?)
    }
}
