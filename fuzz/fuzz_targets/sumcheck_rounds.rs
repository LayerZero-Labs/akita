#![no_main]

use akita_error::AkitaError;
use akita_sumcheck::{
    verify_sumcheck_rounds, SumcheckRole, SumcheckShape, SumcheckVerifierChannel,
};
use jolt_field::{CanonicalBytes, Prime128Offset275 as F, Ring};
use jolt_transcript::{Blake2b512, Channel, ProtocolId, SiteId, VerifierTranscript};
use libfuzzer_sys::fuzz_target;

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("akita-fuzz/sumcheck-rounds");

struct FuzzVerifierChannel<'proof> {
    state: VerifierTranscript<'proof, Blake2b512>,
    challenges: usize,
}

impl<'proof> SumcheckVerifierChannel<'proof, F> for FuzzVerifierChannel<'proof> {
    type Sponge = Blake2b512;

    fn state_mut(&mut self) -> &mut VerifierTranscript<'proof, Blake2b512> {
        &mut self.state
    }

    fn sumcheck_site(&self, _invocation: u32, round: u32, role: SumcheckRole) -> SiteId {
        let mut site = [0u8; 32];
        site[..4].copy_from_slice(&round.to_le_bytes());
        site[4..8].copy_from_slice(&(role as u32).to_le_bytes());
        SiteId(site)
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<F, AkitaError> {
        let site = self.sumcheck_site(invocation, round, SumcheckRole::Challenge);
        self.state.site(site);
        self.challenges += 1;
        Ok(self.state.challenge())
    }
}

fuzz_target!(|data: &[u8]| {
    let [rounds, degree, claim, data @ ..] = data else {
        return;
    };
    let num_rounds = usize::from(rounds % 5);
    let degree_bound = usize::from(degree % 5);
    let Ok(shape) = SumcheckShape::new(num_rounds, degree_bound) else {
        return;
    };
    let mut channel = FuzzVerifierChannel {
        state: VerifierTranscript::new(&PROTOCOL, b"fixture", data),
        challenges: 0,
    };
    let result =
        verify_sumcheck_rounds::<F, F, _>(&mut channel, 0, F::from_u64(u64::from(*claim)), shape);
    assert!(channel.challenges <= num_rounds);
    let round_bytes = degree_bound * F::NUM_BYTES;
    let complete_input_rounds = if round_bytes == 0 {
        0
    } else {
        (data.len() / round_bytes).min(num_rounds)
    };
    assert!(channel.challenges <= complete_input_rounds);
    if let Ok(replay) = result {
        assert_eq!(replay.challenges.len(), num_rounds);
        assert_eq!(channel.challenges, num_rounds);
        let expected_bytes = num_rounds * round_bytes;
        assert_eq!(channel.state.finish().is_ok(), data.len() == expected_bytes);
    }
});
