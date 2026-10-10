//! Verifier-side binary-root components for Akita (#45).
//!
//! This opt-in extension is not on the Akita proof path. It holds the
//! standalone clear opening, which is a differential oracle; the seed-derived
//! and SIS-admitted root setup (`admitted`); and the lowered root relation
//! over the canonical response layout (`lowered`).
//! `root` reduces a statement (`statement`: a binary claim, a prime claim or
//! both) to evaluation claims on the committed tables.
//!
//! `BinaryClearSetup::new` takes an explicit matrix and checks geometry,
//! fold entropy and integer no-wrap only: it performs no SIS width-table
//! lookup. `AdmittedRootSetup::derive` is the construction that derives the
//! shape from a root profile, takes the certified rank, and expands the matrix
//! from a public seed. There is no outer commitment, setup offloading,
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

pub mod admitted;
pub mod channel;
pub mod codec;
pub mod commitment;
pub mod endpoint;
pub mod frontend;
pub mod grinding;
pub mod lowered;
pub mod profile;
pub mod root;
pub mod root_sumcheck;
pub mod source;
pub mod statement;

pub use admitted::{derive_trinomial_matrix, AdmittedRootSetup};
pub use channel::ClearChannel;
pub use commitment::BinaryClearCommitment;
pub use frontend::BinaryEvaluationClaim;
pub use profile::BinaryClearSetup;
pub use root::{
    verify_root_reduction, verify_root_reduction_bytes, RootEvaluationClaims, RootProverOracle,
    RootVerifierOracle,
};
pub use statement::{PrimeClaim, RootOpeningMode, RootStatement};

use akita_algebra::{
    binary::{field_switch::SwitchField, BinaryField162 as B},
    ring::trinomial::TrinomialModulus,
};
use akita_challenges::BinaryChallengeSampler;
use akita_error::{checked, AkitaError};

/// Bind the full public statement before any frontend challenges.
pub fn bind_statement<H: SwitchField, const D: usize, M: TrinomialModulus, S: ClearChannel>(
    setup: &BinaryClearSetup<D, M>,
    commitment: &BinaryClearCommitment,
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
    let expected = checked::product([setup.n_a(), setup.columns(), D])
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
    // Each image is absorbed as its D little-endian `u32` residues.
    let mut image_bytes = Vec::new();
    image_bytes
        .try_reserve_exact(checked::product([D, size_of::<u32>()]).ok_or(AkitaError::InvalidProof)?)
        .map_err(|_| AkitaError::InvalidSetup("commitment encoding allocation failed".into()))?;
    for image in commitment.images.chunks_exact(D) {
        image_bytes.clear();
        for residue in image {
            image_bytes.extend_from_slice(&residue.to_le_bytes());
        }
        channel.public(&image_bytes)?;
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
pub fn verify_binary_clear<H: SwitchField, const D: usize, M: TrinomialModulus, S: ClearChannel>(
    setup: &BinaryClearSetup<D, M>,
    commitment: &BinaryClearCommitment,
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
    // The nonce only selects the challenges; the interval check on the clear
    // response is what binds the prover.
    let challenges = channel.fold_challenges(
        &mut sampler,
        b"akita/labinius/clear-fold/v1",
        setup.columns(),
        &mut 0,
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
pub fn verify_binary_clear_bytes<H: SwitchField, const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<D, M>,
    commitment: &BinaryClearCommitment,
    point: &[H],
    value: H,
    proof: &[u8],
) -> Result<(), AkitaError> {
    let mut channel = channel::new_verifier(proof)?;
    verify_binary_clear(setup, commitment, point, value, &mut channel)?;
    channel::finish_verifier(channel)
}
