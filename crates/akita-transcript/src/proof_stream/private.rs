//! The private-message seam of Akita's proof channel.
//!
//! A zero-knowledge opening masks every witness-dependent message of its
//! levels up to the cut (`specs/whiteout-akita.md` §4.4): the prover sends
//! m̂ = m + p at the clear message's position, in the clear message's
//! encoding, and the verifier records which pads the message took. Protocol
//! code sends those messages through [`PrivateProofChannel`] instead of
//! [`ProofChannel::exchange`], so a clear channel sends them as they are and a
//! masked channel masks them. A masked channel refuses a plain
//! [`ProofChannel::exchange`], so a private value cannot go out clear by
//! mistake; messages public in both modes go through
//! [`PrivateProofChannel::exchange_public`].

use super::{
    exchange_extension_atoms, exchange_extension_group, exchange_field_atoms, ProofChannel,
    ProtocolSiteId,
};
use akita_error::AkitaError;
use jolt_algebra::{CanonicalEncoding, ExtField, Field};
use jolt_transcript::PadSlots;
use spongefish::{Encoding, FromNarg};

use crate::{ProverChannel, VerifierChannel};

/// One end of the proof channel with private messages over the base field
/// `F`: clear (both [`ProverChannel`] and [`VerifierChannel`]) or masked.
pub trait PrivateProofChannel<F: Field + CanonicalEncoding>: ProofChannel {
    /// Exchanges one group of witness-dependent extension values at `site`,
    /// as [`exchange_extension_group`] does on a clear channel. Returns the pad
    /// slots the message took, empty on a clear channel.
    ///
    /// # Errors
    ///
    /// The channel's error on a malformed message or an exhausted pad
    /// schedule.
    fn exchange_private_extension_group<E: ExtField<F>>(
        &mut self,
        site: ProtocolSiteId,
        values: &mut [E],
    ) -> Result<PadSlots, AkitaError>;

    /// Exchanges a message that is public in both modes (the D image, the
    /// cut's payload, public grinding rounds), unmasked.
    ///
    /// # Errors
    ///
    /// As [`ProofChannel::exchange`].
    fn exchange_public<T: Encoding + FromNarg + Copy>(
        &mut self,
        messages: &mut [T],
    ) -> Result<(), AkitaError>;

    /// Whether this channel masks its private messages.
    fn is_masked(&self) -> bool;
}

macro_rules! clear_private_channel {
    ($($channel:tt)*) => {
        impl<F: Field + CanonicalEncoding> PrivateProofChannel<F> for $($channel)* {
            fn exchange_private_extension_group<E: ExtField<F>>(
                &mut self,
                site: ProtocolSiteId,
                values: &mut [E],
            ) -> Result<PadSlots, AkitaError> {
                exchange_extension_group::<F, E, _>(self, site, values)?;
                Ok(PadSlots::default())
            }

            fn exchange_public<T: Encoding + FromNarg + Copy>(
                &mut self,
                messages: &mut [T],
            ) -> Result<(), AkitaError> {
                self.exchange(messages)
            }

            fn is_masked(&self) -> bool {
                false
            }
        }
    };
}

clear_private_channel!(ProverChannel);
clear_private_channel!(VerifierChannel<'_>);

/// [`exchange_extension_group`] for a message that is public in both modes:
/// the same atoms, sent unmasked even on a masked channel.
///
/// # Errors
///
/// As [`exchange_extension_group`].
pub fn exchange_public_extension_group<F, E, S>(
    state: &mut S,
    site: ProtocolSiteId,
    values: &mut [E],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    S: PrivateProofChannel<F>,
{
    exchange_extension_atoms::<F, E, S>(state, site, values, |state, atoms| {
        state.exchange_public(atoms)
    })
}

/// A schedule-fixed group of base-field proof atoms that is public in both
/// modes (an opening payload): [`receive_field_group`](super::receive_field_group)
/// and [`send_field_group`](super::send_field_group) on a clear channel, sent
/// unmasked even on a masked one.
///
/// # Errors
///
/// As [`PrivateProofChannel::exchange_public`].
pub fn exchange_public_field_group<F, S>(
    state: &mut S,
    site: ProtocolSiteId,
    values: &mut [F],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    S: PrivateProofChannel<F>,
{
    exchange_field_atoms::<F, S>(state, site, values, |state, atoms| {
        state.exchange_public(atoms)
    })
}
