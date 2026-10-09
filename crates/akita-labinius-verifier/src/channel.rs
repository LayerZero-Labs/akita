//! The sole dependency boundary to Akita's current transcript transport.
use akita_challenges::{BinaryChallenge, BinaryChallengeSampler, ProverFoldDraw, VerifierFoldDraw};
use akita_error::AkitaError;
use jolt_field::CanonicalEncoding;

/// Minimal role-generic channel for the standalone clear protocol.
/// Site labels are diagnostic; protocol domains are explicitly absorbed bytes.
pub trait ClearChannel {
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError>;
    /// Emit the fixed-length buffer, or overwrite it with received bytes.
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError>;
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError>;
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
    ) -> Result<Vec<BinaryChallenge>, AkitaError>;
}

impl ClearChannel for akita_transcript::ProverChannel {
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        self.public_message(bytes);
        Ok(())
    }
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError> {
        akita_transcript::send_bytes(self, bytes);
        Ok(())
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        Ok(self.verifier_message())
    }
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        sampler.sample_challenges(&mut ProverFoldDraw::new(self, 0, 0), label, count)
    }
}

impl ClearChannel for akita_transcript::VerifierChannel<'_> {
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        self.public_message(bytes);
        Ok(())
    }
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError> {
        let received = akita_transcript::receive_bytes(self, bytes.len())?;
        bytes.copy_from_slice(&received);
        Ok(())
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        self.verifier_message()
    }
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        sampler.sample_challenges(&mut VerifierFoldDraw::new(self, 0, 0), label, count)
    }
}

/// Bind a versioned geometry header and a canonical coefficient digest.
/// The backend's framed session/instance construction makes the split unique.
pub fn matrix_digest<F: CanonicalEncoding>(
    tagged_bytes: &[u8],
    coefficients: &[F],
) -> Result<[u8; 32], AkitaError> {
    let digest = akita_transcript::field_digest(coefficients);
    let mut channel = akita_transcript::new_prover_channel(tagged_bytes, &digest)?;
    ClearChannel::challenge_block(&mut channel)
}

pub fn new_prover() -> Result<akita_transcript::ProverChannel, AkitaError> {
    akita_transcript::new_prover_channel(b"akita/labinius/clear-binary-opening/v1", b"")
}

pub fn finish_prover(channel: akita_transcript::ProverChannel) -> Vec<u8> {
    channel.narg_string().to_vec()
}

pub fn new_verifier(proof: &[u8]) -> Result<akita_transcript::VerifierChannel<'_>, AkitaError> {
    akita_transcript::new_verifier_channel(b"akita/labinius/clear-binary-opening/v1", b"", proof)
}

pub fn finish_verifier(channel: akita_transcript::VerifierChannel<'_>) -> Result<(), AkitaError> {
    channel.check_eof()
}

/// Borrow the existing prover channel for coefficient-field root sumchecks.
///
/// This adapter only transports rounds and samples their field challenges. The
/// enclosing protocol must bind its statement and select distinct invocation
/// identifiers before using the generic sumcheck drivers.
pub struct RootSumcheckProverChannel<'state> {
    state: &'state mut akita_transcript::ProverChannel,
}

impl<'state> RootSumcheckProverChannel<'state> {
    pub fn new(state: &'state mut akita_transcript::ProverChannel) -> Self {
        Self { state }
    }
}

/// Borrow the existing verifier channel for coefficient-field root sumchecks.
pub struct RootSumcheckVerifierChannel<'state, 'proof> {
    state: &'state mut akita_transcript::VerifierChannel<'proof>,
}

impl<'state, 'proof> RootSumcheckVerifierChannel<'state, 'proof> {
    pub fn new(state: &'state mut akita_transcript::VerifierChannel<'proof>) -> Self {
        Self { state }
    }
}

fn root_sumcheck_site(
    invocation: u32,
    round: u32,
    role: akita_sumcheck::SumcheckRole,
) -> akita_transcript::ProtocolSiteId {
    akita_transcript::ProtocolSiteId {
        // "LRSC": a diagnostic family separate from the Akita stage sumchecks.
        family: 0x4c52_5343,
        invocation,
        round,
        detail: role as u32,
        ..akita_transcript::ProtocolSiteId::default()
    }
}

impl<F> akita_sumcheck::SumcheckProverChannel<F> for RootSumcheckProverChannel<'_>
where
    F: akita_algebra::fft::SmoothFftField + jolt_field::ExtField<F>,
{
    fn state_mut(&mut self) -> &mut akita_transcript::ProverChannel {
        self.state
    }

    fn sumcheck_site(
        &self,
        invocation: u32,
        round: u32,
        role: akita_sumcheck::SumcheckRole,
    ) -> akita_transcript::ProtocolSiteId {
        root_sumcheck_site(invocation, round, role)
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<F, AkitaError> {
        akita_transcript::ext_challenge::<F, F, _>(
            self.state,
            root_sumcheck_site(invocation, round, akita_sumcheck::SumcheckRole::Challenge),
        )
    }
}

impl<'proof, F> akita_sumcheck::SumcheckVerifierChannel<'proof, F>
    for RootSumcheckVerifierChannel<'_, 'proof>
where
    F: akita_algebra::fft::SmoothFftField + jolt_field::ExtField<F>,
{
    fn state_mut(&mut self) -> &mut akita_transcript::VerifierChannel<'proof> {
        self.state
    }

    fn sumcheck_site(
        &self,
        invocation: u32,
        round: u32,
        role: akita_sumcheck::SumcheckRole,
    ) -> akita_transcript::ProtocolSiteId {
        root_sumcheck_site(invocation, round, role)
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<F, AkitaError> {
        akita_transcript::ext_challenge::<F, F, _>(
            self.state,
            root_sumcheck_site(invocation, round, akita_sumcheck::SumcheckRole::Challenge),
        )
    }
}
