//! Fixed-field boundary to the native, catalog-typed grouped verifier.

use crate::{config::DigitConfig, F};
use akita_error::AkitaError;
use akita_params::BasisMode;
use akita_types::{AkitaVerifierSetup, GroupBatchStatement};
use akita_verifier::AkitaVerifier;

// Storing AkitaVerifier<C> directly in RootPcsVerifier<C> triggers Rustdoc 1.95
// auto-trait synthesis on C::Field. Erase only the prepared verifier's type;
// keep the catalog, setup, public API and native verification logic unchanged.
pub(super) trait GroupedVerifier: Send + Sync {
    fn setup(&self) -> &AkitaVerifierSetup<F>;
    fn verify(
        &self,
        proof: &[u8],
        session: &[u8],
        statement: GroupBatchStatement<'_, F, F>,
    ) -> Result<(), AkitaError>;
}
impl<C: DigitConfig> GroupedVerifier for AkitaVerifier<C> {
    fn setup(&self) -> &AkitaVerifierSetup<F> {
        self.setup()
    }
    fn verify(
        &self,
        proof: &[u8],
        session: &[u8],
        statement: GroupBatchStatement<'_, F, F>,
    ) -> Result<(), AkitaError> {
        self.batched_verify(proof, session, statement, BasisMode::Lagrange)
    }
}
