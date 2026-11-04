//! The masked channels Akita's grinding replay runs over at the levels up to a
//! zero-knowledge cut (`specs/whiteout-akita.md` §4.5).
//!
//! A masked channel wraps a clear v8 channel and jolt-zk's pads. Its private
//! messages ([`PrivateProofChannel`]) go out as m̂ = m + p, at the clear
//! message's position and in its encoding; its public absorbs, challenges,
//! proof-of-work nonces and fold-response counters pass through. A plain
//! [`ProofChannel::exchange`] is refused, so a private value cannot leave a
//! masked level in the clear; messages public in both modes use
//! [`PrivateProofChannel::exchange_public`].

#[cfg(feature = "prover")]
use super::ProverGrindingChannel;
use super::VerifierGrindingChannel;
use akita_error::AkitaError;
#[cfg(feature = "prover")]
use akita_transcript::ProverChannel;
use akita_transcript::{
    exchange_extension_group, ChallengeCodec, PrivateProofChannel, ProofChannel,
    ProtocolContextRecord, ProtocolSiteId, VerifierChannel,
};
use jolt_algebra::{CanonicalEncoding, ExtField, Field};
use jolt_transcript::PadSlots;
#[cfg(feature = "prover")]
use jolt_zk::masking::CommittedPads;
use jolt_zk::masking::CommittedPadsView;
use spongefish::{Encoding, FromNarg};

fn clear_message_refused() -> AkitaError {
    AkitaError::UnsupportedSchedule(
        "a clear message was sent on a masked channel; private messages go out masked".into(),
    )
}

/// The prover's masked channel over the clear channel `state`, masking with
/// `pads`.
#[cfg(feature = "prover")]
pub struct MaskedProverChannel<'a, F, O> {
    state: &'a mut ProverChannel,
    pads: &'a mut CommittedPads<F, O>,
}

#[cfg(feature = "prover")]
impl<'a, F, O> MaskedProverChannel<'a, F, O> {
    /// The masked channel over `state` and `pads`.
    pub fn new(state: &'a mut ProverChannel, pads: &'a mut CommittedPads<F, O>) -> Self {
        Self { state, pads }
    }
}

#[cfg(feature = "prover")]
impl<F, O> ProofChannel for MaskedProverChannel<'_, F, O> {
    fn context(&mut self, record: ProtocolContextRecord) {
        self.state.context(record);
    }

    fn public(&mut self, bytes: &[u8]) {
        self.state.public(bytes);
    }

    fn exchange<T: Encoding + FromNarg + Copy>(&mut self, _: &mut [T]) -> Result<(), AkitaError> {
        Err(clear_message_refused())
    }

    fn challenges<E: ChallengeCodec>(&mut self, len: usize) -> Result<Vec<E>, AkitaError> {
        self.state.challenges(len)
    }

    fn challenge_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<T, AkitaError>,
    ) -> Result<T, AkitaError> {
        self.state.challenge_message(len, decode)
    }

    fn guard<T>(&mut self, result: Result<T, AkitaError>) -> Result<T, AkitaError> {
        self.state.guard(result)
    }
}

/// `value` as a `T`, if `S` is `T`: the pad field and the message's base
/// field are one field, checked at run time because the channel's methods are
/// generic over the message field.
#[cfg(feature = "prover")]
fn same_field<S: Copy + 'static, T: Copy + 'static>(value: S) -> Result<T, AkitaError> {
    (&value as &dyn std::any::Any)
        .downcast_ref::<T>()
        .copied()
        .ok_or_else(|| {
            AkitaError::InvalidSetup("a masked message's field is not the pad field".into())
        })
}

#[cfg(feature = "prover")]
impl<F, P, O> PrivateProofChannel<F> for MaskedProverChannel<'_, P, O>
where
    F: Field + CanonicalEncoding + 'static,
    P: Field + 'static,
{
    fn exchange_private_extension_group<E: ExtField<F>>(
        &mut self,
        site: ProtocolSiteId,
        values: &mut [E],
    ) -> Result<PadSlots, AkitaError> {
        let coordinates = values
            .iter()
            .flat_map(|value| (0..E::DEGREE).map(|index| value.base_coefficient(index)))
            .map(same_field::<F, P>)
            .collect::<Result<Vec<P>, AkitaError>>()?;
        let (slots, masked) = self.pads.mask(&coordinates)?;
        let masked = masked
            .into_iter()
            .map(same_field::<P, F>)
            .collect::<Result<Vec<F>, AkitaError>>()?;
        #[expect(
            clippy::indexing_slicing,
            reason = "each chunk holds E::DEGREE coordinates and from_base_fn asks for 0..E::DEGREE"
        )]
        let mut masked: Vec<E> = masked
            .chunks_exact(E::DEGREE)
            .map(|chunk| E::from_base_fn(|index| chunk[index]))
            .collect();
        exchange_extension_group::<F, E, _>(self.state, site, &mut masked)?;
        #[cfg(feature = "logging-transcript")]
        akita_transcript::mark_masked(slots.start, slots.len);
        Ok(slots)
    }

    fn exchange_public<T: Encoding + FromNarg + Copy>(
        &mut self,
        messages: &mut [T],
    ) -> Result<(), AkitaError> {
        self.state.exchange(messages)
    }

    fn is_masked(&self) -> bool {
        true
    }
}

#[cfg(feature = "prover")]
impl<F, O> ProverGrindingChannel for MaskedProverChannel<'_, F, O> {
    fn clear_state(&mut self) -> &mut ProverChannel {
        self.state
    }

    fn clear_state_ref(&self) -> &ProverChannel {
        self.state
    }

    fn masks(&self) -> bool {
        true
    }
}

/// The verifier's masked channel over the clear channel `state`: it reads m̂
/// in place of each private value and records the pads it took in `view`.
pub struct MaskedVerifierChannel<'a, 'proof, C> {
    state: &'a mut VerifierChannel<'proof>,
    view: &'a mut CommittedPadsView<C>,
}

impl<'a, 'proof, C> MaskedVerifierChannel<'a, 'proof, C> {
    /// The masked channel over `state`, recording into `view`.
    pub fn new(state: &'a mut VerifierChannel<'proof>, view: &'a mut CommittedPadsView<C>) -> Self {
        Self { state, view }
    }
}

impl<C> ProofChannel for MaskedVerifierChannel<'_, '_, C> {
    fn context(&mut self, record: ProtocolContextRecord) {
        self.state.context(record);
    }

    fn public(&mut self, bytes: &[u8]) {
        self.state.public(bytes);
    }

    fn exchange<T: Encoding + FromNarg + Copy>(&mut self, _: &mut [T]) -> Result<(), AkitaError> {
        let error = clear_message_refused();
        self.state.guard(Err(error))
    }

    fn challenges<E: ChallengeCodec>(&mut self, len: usize) -> Result<Vec<E>, AkitaError> {
        self.state.challenges(len)
    }

    fn challenge_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<T, AkitaError>,
    ) -> Result<T, AkitaError> {
        self.state.challenge_message(len, decode)
    }

    fn guard<T>(&mut self, result: Result<T, AkitaError>) -> Result<T, AkitaError> {
        self.state.guard(result)
    }
}

impl<F: Field + CanonicalEncoding, C> PrivateProofChannel<F> for MaskedVerifierChannel<'_, '_, C> {
    fn exchange_private_extension_group<E: ExtField<F>>(
        &mut self,
        site: ProtocolSiteId,
        values: &mut [E],
    ) -> Result<PadSlots, AkitaError> {
        exchange_extension_group::<F, E, _>(self.state, site, values)?;
        let slots = self.view.take_ext(values.len(), E::DEGREE);
        self.state.guard(slots.map_err(AkitaError::from))
    }

    fn exchange_public<T: Encoding + FromNarg + Copy>(
        &mut self,
        messages: &mut [T],
    ) -> Result<(), AkitaError> {
        self.state.exchange(messages)
    }

    fn is_masked(&self) -> bool {
        true
    }
}

impl<'proof, C> VerifierGrindingChannel<'proof> for MaskedVerifierChannel<'_, 'proof, C> {
    fn clear_state(&mut self) -> &mut VerifierChannel<'proof> {
        self.state
    }
}
