//! The masked channels Akita's grinding replay runs over at the levels up to a
//! zero-knowledge cut (`specs/whiteout-akita.md` §4.5): jolt-zk's
//! [`MaskedProver`] and [`MaskedVerifier`]. Their proof-of-work nonces and
//! fold-response counters are public, so they go to the transcript
//! underneath, unmasked.

use super::{ProverGrindingChannel, VerifierGrindingChannel};
use jolt_algebra::{CanonicalEncoding, Field};
use jolt_transcript::{ProverTranscript, Sponge, VerifierTranscript};
use jolt_zk::masking::{MaskedProver, MaskedVerifier};

impl<P: Field + CanonicalEncoding, O, H: Sponge> ProverGrindingChannel
    for MaskedProver<'_, P, O, H>
{
    fn transcript_mut(&mut self) -> &mut ProverTranscript<H> {
        Self::transcript_mut(self)
    }
}

impl<'proof, H: Sponge> VerifierGrindingChannel<'proof> for MaskedVerifier<'_, 'proof, H> {
    fn transcript_mut(&mut self) -> &mut VerifierTranscript<'proof, H> {
        Self::transcript_mut(self)
    }
}
