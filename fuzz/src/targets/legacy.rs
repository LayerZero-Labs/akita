//! The original cargo-fuzz targets, moved behind engine-independent entry
//! points with their assertions unchanged.

use akita_error::AkitaError;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use jolt_algebra::{CanonicalBytes, Prime128Offset275 as F, Ring};
use jolt_sumcheck::{
    verify_single, SingleFormat, SingleInstance, SumcheckSite, VerifierRoundContext,
};
use akita_transcript::{
    ext_challenge, new_prover_channel, new_verifier_channel, public_bytes, ProtocolSiteId,
    VerifierChannel, SITE_FAMILY_SUMCHECK,
};
use jolt_algebra::{CanonicalBytes, Prime128Offset275 as F, Ring};

pub fn serialization_vec(data: &[u8]) {
    let _ = Vec::<u8>::deserialize_compressed(data, &());
    let _ = Vec::<bool>::deserialize_compressed(data, &());

    if let Ok(decoded) = Vec::<u8>::deserialize_compressed(data, &()) {
        let mut encoded = Vec::new();
        if decoded.serialize_compressed(&mut encoded).is_ok() {
            let reparsed = Vec::<u8>::deserialize_compressed(&encoded[..], &());
            assert_eq!(reparsed.ok(), Some(decoded));
        }
    }
}

pub fn transcript_labels(data: &[u8]) {
    let split = data.len().min(255);
    let (label, bytes) = data.split_at(split);

    let mut transcript = new_prover_channel(label);
    let _ = public_bytes(&mut transcript, ProtocolSiteId::default(), bytes);
    let _: F = ext_challenge(&mut transcript, ProtocolSiteId::default()).unwrap();
}

struct FuzzVerifierChannel<'proof> {
    state: VerifierTranscript<'proof, Blake2b512>,
    challenges: usize,
}

impl<'proof> VerifierRoundContext<F> for FuzzVerifierChannel<'proof> {
    type Channel = VerifierTranscript<'proof, Blake2b512>;
    type Error = AkitaError;

    fn channel_mut(&mut self) -> &mut VerifierTranscript<'proof, Blake2b512> {
        &mut self.state
    }

    fn site(&self, site: SumcheckSite) -> Option<SiteId> {
        let mut bytes = [0u8; 32];
        bytes[..4].copy_from_slice(&site.round.to_le_bytes());
        bytes[4..8].copy_from_slice(&(site.role as u32).to_le_bytes());
        Some(SiteId(bytes))
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<F, AkitaError> {
        let site = self.sumcheck_site(invocation, round, SumcheckRole::Challenge);
        let challenge = ext_challenge::<F, _>(&mut self.state, site)?;
        self.challenges += 1;
        Ok(self.state.challenge())
    }
}

pub fn sumcheck_rounds(data: &[u8]) {
    let [rounds, degree, claim, data @ ..] = data else {
        return;
    };
    let num_rounds = usize::from(rounds % 5);
    let degree_bound = usize::from(degree % 5);
    let Ok(shape) = SumcheckShape::new(num_rounds, degree_bound) else {
        return;
    };
    let state = new_verifier_channel(b"fuzz/sumcheck-rounds", data);
    let mut channel = FuzzVerifierChannel {
        state: VerifierTranscript::new(&PROTOCOL, b"fixture", data),
        challenges: 0,
    };
    let instance = SingleInstance {
        input_claim: F::from_u64(u64::from(*claim)),
        rounds: num_rounds,
        degree: degree_bound,
        format: SingleFormat::Compressed,
        invocation: 0,
    };
    let result = verify_single::<F, F, AkitaError, _>(&instance, &mut channel);
    assert!(channel.challenges <= num_rounds);
    let round_bytes = degree_bound * F::NUM_BYTES;
    let complete_input_rounds = if round_bytes == 0 {
        0
    } else {
        (data.len() / round_bytes).min(num_rounds)
    };
    assert!(channel.challenges <= complete_input_rounds);
    if degree_bound == 0 {
        assert!(result.is_err());
    }
    if let Ok(replay) = result {
        assert_eq!(replay.challenges.len(), num_rounds);
        assert_eq!(channel.challenges, num_rounds);
        let expected_bytes = num_rounds * round_bytes;
        assert_eq!(channel.state.finish().is_ok(), data.len() == expected_bytes);
    }
}
