//! Verifier-side standalone clear binary openings for Akita (#45).
//!
//! This opt-in extension is a differential oracle: it performs no SIS
//! width-table/security lookup and admits no production parameter set. It is
//! not on the Akita proof path. There is no outer commitment, setup offloading,
//! recursion or zero knowledge. Verifier-reachable code follows
//! `docs/verifier-contract.md`; this crate never depends on a prover crate.
#![cfg(feature = "labinius")]
#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

pub mod channel;
pub mod codec;
pub mod commitment;
pub mod endpoint;
pub mod frontend;
pub mod lowered;
pub mod profile;
pub mod source;

pub use channel::ClearChannel;
pub use commitment::BinaryClearCommitment;
pub use frontend::BinaryEvaluationClaim;
pub use profile::BinaryClearSetup;

use akita_algebra::{
    binary::{field_switch::SwitchField, BinaryField162 as B},
    fft::SmoothFftField,
    ring::trinomial::TrinomialModulus,
};
use akita_challenges::BinaryChallengeSampler;
use akita_error::{checked, AkitaError};

/// Bind the full public statement before any frontend challenges.
pub fn bind_statement<
    H: SwitchField,
    F: SmoothFftField,
    const D: usize,
    M: TrinomialModulus,
    S: ClearChannel,
>(
    setup: &BinaryClearSetup<F, D, M>,
    commitment: &BinaryClearCommitment<F, D, M>,
    point: &[H],
    value: H,
    channel: &mut S,
) -> Result<(), AkitaError> {
    if point.len() != setup.num_vars() {
        return Err(AkitaError::InvalidPointDimension {
            expected: setup.num_vars(),
            actual: point.len(),
        });
    }
    let expected = checked::product([setup.n_a(), setup.columns()])
        .ok_or_else(|| AkitaError::InvalidSetup("commitment size overflow".into()))?;
    if commitment.images.len() != expected {
        return Err(AkitaError::InvalidInput(
            "commitment image count mismatch".into(),
        ));
    }
    let mut domain = Vec::new();
    codec::length_prefixed(&mut domain, b"akita/labinius/clear-binary-opening/v1")?;
    channel.public(&domain)?;
    channel.public(&setup.identity_bytes::<H>()?)?;
    // The setup fixes every count and width. No proof-supplied shape is read.
    let mut coefficient = vec![0u8; F::NUM_BYTES];
    for image in &commitment.images {
        for field in image.coefficients() {
            field.to_bytes_le(&mut coefficient);
            channel.public(&coefficient)?;
        }
    }
    // Host polynomial coordinates are 16 bytes for F128 and 24 for F192.
    for host in point.iter().copied().chain(std::iter::once(value)) {
        for word in host.coordinates().iter().take(H::ROWS / 64) {
            channel.public(&word.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Verify the clear opening on an existing channel, without finishing it.
pub fn verify_binary_clear<
    H: SwitchField,
    F: SmoothFftField,
    const D: usize,
    M: TrinomialModulus,
    S: ClearChannel,
>(
    setup: &BinaryClearSetup<F, D, M>,
    commitment: &BinaryClearCommitment<F, D, M>,
    point: &[H],
    value: H,
    channel: &mut S,
) -> Result<(), AkitaError> {
    bind_statement(setup, commitment, point, value, channel)?;
    let binary = frontend::verify_frontend(point, value, channel)?;
    let mut u = Vec::new();
    u.try_reserve_exact(setup.columns())
        .map_err(|_| AkitaError::InvalidSetup("left expansion allocation failed".into()))?;
    for _ in 0..setup.columns() {
        let mut element = B::ZERO;
        codec::exchange_binary(channel, &mut element)?;
        u.push(element);
    }
    endpoint::verify_left_expansion(setup, &binary, &u)?;
    let mut sampler = BinaryChallengeSampler::new(setup.profile().clone());
    let challenges = channel.fold_challenges(
        &mut sampler,
        b"akita/labinius/clear-fold/v1",
        setup.columns(),
    )?;
    let mut response = Vec::new();
    response
        .try_reserve_exact(setup.scalar_rows())
        .map_err(|_| AkitaError::InvalidSetup("response allocation failed".into()))?;
    response.resize(setup.scalar_rows(), [0; 162]);
    codec::exchange_response(channel, setup, &mut response)?;
    endpoint::verify_endpoints(setup, commitment, &binary, &u, &challenges, &response)
}

/// Verify one complete proof byte string, rejecting truncation and trailing bytes.
pub fn verify_binary_clear_bytes<
    H: SwitchField,
    F: SmoothFftField,
    const D: usize,
    M: TrinomialModulus,
>(
    setup: &BinaryClearSetup<F, D, M>,
    commitment: &BinaryClearCommitment<F, D, M>,
    point: &[H],
    value: H,
    proof: &[u8],
) -> Result<(), AkitaError> {
    let mut channel = channel::new_verifier(proof)?;
    verify_binary_clear(setup, commitment, point, value, &mut channel)?;
    channel::finish_verifier(channel)
}
