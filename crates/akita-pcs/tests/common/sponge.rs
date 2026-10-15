//! The sponge the transcript-semantics suites run on.
//!
//! CI runs those suites twice: on Akita's standalone sponge, and with the
//! `keccak-test-sponge` feature on Keccak, the other sponge a caller may run
//! Akita on. Proofs here are standalone-shaped (one argument string bound to a
//! session) under the selected sponge.

use akita_config::CommitmentConfig;
use akita_cpu_backend::{AkitaProverSetup, CommitmentHandle, CpuBackend};
use akita_error::AkitaError;
use akita_params::BasisMode;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ProverBackend, SelectedProverOpeningData};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::{FpExtEncoding, GroupBatchStatement, PROOF_STREAM_PROTOCOL_NAME};
use akita_verifier::AkitaVerifier;
use jolt_field::{
    AdditiveGroup, CanonicalEncoding, ExtField, Field, Fold, MulBaseUnreduced, PseudoMersenne,
    Ring, Unreduced, WithCommitAccumulator,
};
use jolt_transcript::{ProtocolId, ProverTranscript, VerifierTranscript};

#[cfg(not(feature = "keccak-test-sponge"))]
pub(crate) type TestSponge = akita_types::AkitaSponge;
#[cfg(feature = "keccak-test-sponge")]
pub(crate) type TestSponge = jolt_transcript::Keccak;

/// The standalone proof-stream protocol under [`TestSponge`].
pub(crate) const TEST_PROTOCOL: ProtocolId =
    ProtocolId::new::<TestSponge>(PROOF_STREAM_PROTOCOL_NAME);

/// Prove `opening` on a fresh [`TestSponge`] transcript bound to `session` and
/// return its argument string.
#[allow(clippy::type_complexity)]
pub(crate) fn prove_on_test_sponge<'a, Cfg>(
    scheme: &AkitaCommitmentScheme<Cfg>,
    setup: &AkitaProverSetup<Cfg::Field>,
    opening: SelectedProverOpeningData<
        'a,
        Cfg::ExtField,
        CommitmentHandle<Cfg::Field, Cfg::ExtField>,
        Cfg::Field,
    >,
    backend: &CpuBackend<Cfg::Field, Cfg::ExtField>,
    session: &[u8],
    basis: BasisMode,
) -> Result<Vec<u8>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding
        + AkitaSerialize
        + Valid
        + Unreduced
        + PseudoMersenne
        + Ring
        + WithCommitAccumulator
        + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<Cfg::Field>
        + AkitaSerialize
        + 'static,
    CpuBackend<Cfg::Field, Cfg::ExtField>: ProverBackend<
        Cfg::Field,
        Cfg::ExtField,
        CommitmentHandle = CommitmentHandle<Cfg::Field, Cfg::ExtField>,
    >,
{
    let mut transcript = ProverTranscript::<TestSponge>::new(&TEST_PROTOCOL, session);
    super::caller_transcript::prove_on_transcript(
        scheme,
        setup,
        opening,
        backend,
        &mut transcript,
        basis,
    )?;
    Ok(transcript.finish())
}

/// Verify a [`prove_on_test_sponge`] proof, consuming it exactly.
pub(crate) fn verify_on_test_sponge<Cfg>(
    verifier: &AkitaVerifier<Cfg>,
    proof: &[u8],
    session: &[u8],
    statement: GroupBatchStatement<'_, Cfg::ExtField, Cfg::Field>,
    basis: BasisMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field + CanonicalEncoding + AkitaSerialize + PseudoMersenne + Valid,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + ExtField<Cfg::Field> + Ring + AkitaSerialize + Valid,
{
    let mut transcript = VerifierTranscript::<TestSponge>::new(&TEST_PROTOCOL, session, proof);
    verifier.batched_verify(&mut transcript, statement, basis)?;
    Ok(transcript.finish()?)
}
