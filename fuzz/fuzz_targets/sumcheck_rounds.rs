#![no_main]

use akita_error::AkitaError;
use akita_sumcheck::{verify_sumcheck_rounds_native, NativeSumcheckVerifierChannel};
use akita_transcript::{
    native_field_challenge_bytes, native_verifier_field_challenge, new_native_verifier,
    verifier_context, NativeVerifierState, ProtocolContextRecord, ProtocolMessageKind,
    ProtocolSiteId, SITE_FAMILY_SUMCHECK,
};
use jolt_field::{Prime128Offset275 as F, Ring};
use libfuzzer_sys::fuzz_target;

struct FuzzVerifierChannel<'proof> {
    state: NativeVerifierState<'proof>,
}

impl<'proof> NativeSumcheckVerifierChannel<'proof, F> for FuzzVerifierChannel<'proof> {
    fn state_mut(&mut self) -> &mut NativeVerifierState<'proof> {
        &mut self.state
    }

    fn sumcheck_site(&self, invocation: u32, round: u32, role: u32) -> ProtocolSiteId {
        ProtocolSiteId {
            family: SITE_FAMILY_SUMCHECK,
            invocation,
            round,
            detail: role,
            ..ProtocolSiteId::default()
        }
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<F, AkitaError> {
        let site = self.sumcheck_site(invocation, round, 4);
        verifier_context(
            &mut self.state,
            ProtocolContextRecord::new(
                site.to_bytes(),
                ProtocolMessageKind::Challenge as u32,
                0,
                0,
                native_field_challenge_bytes::<F>(),
            ),
        );
        Ok(native_verifier_field_challenge(&mut self.state))
    }
}

fuzz_target!(|data: &[u8]| {
    let [rounds, degree, claim, data @ ..] = data else {
        return;
    };
    let num_rounds = usize::from(rounds % 5);
    let degree_bound = usize::from(degree % 5);
    let Ok(state) = new_native_verifier(b"fuzz/sumcheck-rounds", b"fixture", data) else {
        return;
    };
    let mut channel = FuzzVerifierChannel { state };
    let _ = verify_sumcheck_rounds_native::<F, F, _>(
        &mut channel,
        0,
        F::from_u64(u64::from(*claim)),
        num_rounds,
        degree_bound,
    );
});
