//! Workspace-only schedule loading for examples, benches, and tests.

use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::FpExtEncoding;
use jolt_field::{CanonicalEncoding, ExtField, Field, Fold, PseudoMersenne, Ring, Unreduced};

/// Load and bind the checked-in workspace artifact at a test or tooling boundary.
pub(crate) fn load_workspace_scheme<Cfg>() -> Result<AkitaCommitmentScheme<Cfg>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field + CanonicalEncoding + Unreduced + PseudoMersenne + Valid + AkitaSerialize,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: ExtField<Cfg::Field> + Ring + Unreduced + Fold + AkitaSerialize,
{
    AkitaCommitmentScheme::new(akita_config::test_support::workspace_schedule_catalog::<Cfg>()?)
}
