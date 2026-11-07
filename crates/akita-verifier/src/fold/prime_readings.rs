//! Authenticated readings with an application-owned source-agreement contract.

use crate::{AkitaVerifier, SetupPolicy};
use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_params::{descriptor_bytes::digest_descriptor_bytes, BasisMode};
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::VerifierChannel;
use akita_types::binary_source::{
    BinaryOpening, BinaryWordClaim, PrimeGroupClaim, SourceAgreementBinding,
};
use akita_types::opening_claims::GroupClaimKind;
use akita_types::{FpExtEncoding, GroupBatchStatement, PendingEof};
use jolt_field::{CanonicalEncoding, ExtField, Field, PseudoMersenne, Ring};

impl<Cfg> AkitaVerifier<Cfg>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field + CanonicalEncoding + AkitaSerialize + PseudoMersenne + Valid,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + ExtField<Cfg::Field> + Ring + AkitaSerialize + Valid,
{
    /// Verify one embedded opening and return its separately authenticated
    /// binary and prime readings under the conditional source-agreement contract.
    ///
    /// `session_id` and `group_id` identify context that the caller has already
    /// authenticated inside the caller's transcript for this exact committed
    /// source. The terminal digest also binds the commitment payload. This method does not add
    /// messages or check agreement between the two readings. The application
    /// must discharge the contract through its own source-agreement proof.
    /// As for [`Self::batched_verify_embedded`], only the returned
    /// [`PendingEof`] accepts the caller's transcript.
    #[allow(clippy::type_complexity)]
    pub fn batched_verify_readings_embedded(
        &self,
        state: &mut VerifierChannel<'_>,
        statement: GroupBatchStatement<'_, Cfg::ExtField, Cfg::Field>,
        basis: BasisMode,
        mut setup_policy: SetupPolicy<'_, '_, Cfg::ExtField>,
        session_id: [u8; 32],
        group_id: [u8; 32],
    ) -> Result<
        (
            PendingEof,
            BinaryOpening<Option<BinaryWordClaim>, Vec<PrimeGroupClaim>>,
        ),
        AkitaError,
    > {
        let pending_len = setup_policy.pending_len();
        let result = self.verify_on_state(state, statement, basis, &mut setup_policy, |claims| {
            let group = claims.groups().first().ok_or_else(|| {
                AkitaError::InvalidInput("binary readings require a binary group".into())
            })?;
            let GroupClaimKind::Binary {
                manifest,
                claim,
                prime,
            } = group.kind()
            else {
                return Err(AkitaError::InvalidInput(
                    "binary readings require a binary group first".into(),
                ));
            };
            let source = manifest.descriptor().clone();
            let source_bytes = source.canonical_bytes();
            let prime_bytes = prime.canonical_bytes();
            let mut bytes = b"akita/binary-source/authenticated-prime-readings/v1".to_vec();
            bytes.extend_from_slice(&akita_transcript::field_digest(
                group.commitment().rows().coeffs(),
            ));
            bytes.extend_from_slice(&(source_bytes.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&source_bytes);
            bytes.extend_from_slice(&(prime_bytes.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&prime_bytes);
            let binding = SourceAgreementBinding {
                session: session_id,
                group: group_id,
                source,
                terminal_prime_claims: digest_descriptor_bytes(&bytes),
            };
            Ok(BinaryOpening::from_authenticated_readings(
                claim.clone(),
                prime.claims().to_vec(),
                binding,
            ))
        });
        setup_policy.settle(pending_len, result.is_ok());
        result
    }
}
