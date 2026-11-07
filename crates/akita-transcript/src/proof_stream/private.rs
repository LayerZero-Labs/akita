//! The private-message seam of Akita's proof channel.
//!
//! Extension seam: zero knowledge (akita#120). A zero-knowledge opening masks every
//! witness-dependent message of its masked levels. Protocol code sends those
//! messages through [`PrivateProofChannel`] instead of
//! [`ProofChannel::exchange`], so a clear channel sends them as they are and a
//! masking channel can mask them; messages that are public in both modes go
//! through [`PrivateProofChannel::exchange_public`]. The clear prover and
//! verifier channels are the only implementations here.

use super::{
    exchange_extension_atoms, exchange_extension_group, exchange_field_atoms, ProofChannel,
    ProtocolSiteId,
};
use akita_error::AkitaError;
use jolt_algebra::{CanonicalEncoding, ExtField, Field};
use spongefish::{Encoding, FromNarg};

use crate::{ProverChannel, VerifierChannel};

/// One end of the proof channel with private messages over the base field
/// `F`.
pub trait PrivateProofChannel<F: Field + CanonicalEncoding>: ProofChannel {
    /// Exchanges one group of witness-dependent extension values at `site`,
    /// as [`exchange_extension_group`] does on a clear channel.
    ///
    /// # Errors
    ///
    /// The channel's error on a malformed message.
    fn exchange_private_extension_group<E: ExtField<F>>(
        &mut self,
        site: ProtocolSiteId,
        values: &mut [E],
    ) -> Result<(), AkitaError>;

    /// Exchanges a message that is public in both modes, unmasked.
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
            ) -> Result<(), AkitaError> {
                exchange_extension_group::<F, E, _>(self, site, values)
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
/// the same atoms, sent unmasked on any channel.
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
/// modes: [`send_field_group`](super::send_field_group) and
/// [`receive_field_group`](super::receive_field_group) on a clear channel,
/// sent unmasked on any channel.
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
