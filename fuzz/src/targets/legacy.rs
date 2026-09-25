//! The original cargo-fuzz targets, moved behind engine-independent entry
//! points with their assertions unchanged.

use akita_error::AkitaError;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_sumcheck::{
    verify_sumcheck_rounds_native, NativeSumcheckRole, NativeSumcheckShape,
    NativeSumcheckVerifierChannel,
};
use akita_transcript::{
    native_field_challenge_bytes, native_prover_field_challenge, native_verifier_field_challenge,
    new_native_prover, new_native_verifier, public_native_bytes_prover, verifier_context,
    NativeVerifierState, ProtocolContextRecord, ProtocolMessageKind, ProtocolSiteId,
    SITE_FAMILY_SUMCHECK,
};
use jolt_field::{CanonicalBytes, Prime128Offset275 as F, Ring};

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

    let Ok(mut transcript) = new_native_prover(b"akita-fuzz", label) else {
        return;
    };
    let _ = public_native_bytes_prover(&mut transcript, ProtocolSiteId::default(), bytes);
    let _: F = native_prover_field_challenge(&mut transcript).unwrap();
}

struct FuzzVerifierChannel<'proof> {
    state: NativeVerifierState<'proof>,
    challenges: usize,
}

impl<'proof> NativeSumcheckVerifierChannel<'proof, F> for FuzzVerifierChannel<'proof> {
    fn state_mut(&mut self) -> &mut NativeVerifierState<'proof> {
        &mut self.state
    }

    fn sumcheck_site(
        &self,
        invocation: u32,
        round: u32,
        role: NativeSumcheckRole,
    ) -> ProtocolSiteId {
        ProtocolSiteId {
            family: SITE_FAMILY_SUMCHECK,
            invocation,
            round,
            detail: role as u32,
            ..ProtocolSiteId::default()
        }
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<F, AkitaError> {
        let site = self.sumcheck_site(invocation, round, NativeSumcheckRole::Challenge);
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
        let challenge = native_verifier_field_challenge(&mut self.state)
            .map_err(|_| AkitaError::InvalidProof)?;
        self.challenges += 1;
        Ok(challenge)
    }
}

pub fn sumcheck_rounds(data: &[u8]) {
    let [rounds, degree, claim, data @ ..] = data else {
        return;
    };
    let num_rounds = usize::from(rounds % 5);
    let degree_bound = usize::from(degree % 5);
    let Ok(shape) = NativeSumcheckShape::new(num_rounds, degree_bound) else {
        return;
    };
    let Ok(state) = new_native_verifier(b"fuzz/sumcheck-rounds", b"fixture", data) else {
        return;
    };
    let mut channel = FuzzVerifierChannel {
        state,
        challenges: 0,
    };
    let result = verify_sumcheck_rounds_native::<F, F, _>(
        &mut channel,
        0,
        F::from_u64(u64::from(*claim)),
        shape,
    );
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
        assert_eq!(
            channel.state.check_eof().is_ok(),
            data.len() == expected_bytes
        );
    }
}
