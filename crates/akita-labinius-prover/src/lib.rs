//! Prover-side binary-root components for the opt-in LaBinius extension.
//!
//! This crate commits binary host words and proves their host-field multilinear
//! evaluation with a clear integer response, which provides no zero knowledge,
//! and builds the lowered root witness (`lowered`). Setup admission lives in
//! the verifier crate: these functions accept any admitted
//! `BinaryClearSetup`, explicit or seed-derived.
//! The `root` reduction returns two committed-table evaluation claims through an oracle seam.

#![cfg(feature = "labinius")]

pub mod commit_kernel;
pub mod fold_kernel;
pub mod lowered;
pub mod root;
pub mod root_sumcheck;

pub use root::{prove_root_reduction, prove_root_reduction_bytes, TransparentRootProverOracle};

pub use commit_kernel::{commit_binary_clear_prepared, PreparedCommitMatrix};

use akita_algebra::{binary::field_switch::SwitchField, SmoothFftField, TrinomialModulus};
use akita_challenges::BinaryChallengeSampler;
use akita_error::AkitaError;
use akita_labinius_verifier::{
    bind_statement,
    channel::{finish_prover, new_prover, ClearChannel},
    codec::{exchange_binary, exchange_response},
    commitment::{apply_matrix, BinaryClearCommitment},
    endpoint::{fold_integer, left_expansion},
    frontend::prove_frontend,
    profile::BinaryClearSetup,
    source::pack_source_column,
};

/// Commit each packed source column under the caller's explicit Ajtai matrix.
pub fn commit_binary_clear<H, F, const D: usize, M>(
    setup: &BinaryClearSetup<F, D, M>,
    source: &[H::Source],
) -> Result<BinaryClearCommitment<F, D, M>, AkitaError>
where
    H: SwitchField,
    F: SmoothFftField,
    M: TrinomialModulus,
{
    if source.len() != setup.source_len() {
        return Err(AkitaError::InvalidSize {
            expected: setup.source_len(),
            actual: source.len(),
        });
    }
    let mut images = Vec::new();
    for column in 0..setup.columns() {
        let packed = pack_source_column::<H, F, D, M>(setup, source, column)?;
        images.extend(apply_matrix(setup, &packed)?);
    }
    Ok(BinaryClearCommitment { images })
}

/// Prove a clear binary opening without finishing the caller's channel.
///
/// Any integer response coefficient outside the admitted interval returns an
/// error. The prover never retries challenges to obtain an accepted response.
pub fn prove_binary_clear<H, F, const D: usize, M, S>(
    setup: &BinaryClearSetup<F, D, M>,
    source: &[H::Source],
    commitment: &BinaryClearCommitment<F, D, M>,
    point: &[H],
    claim: H,
    channel: &mut S,
) -> Result<(), AkitaError>
where
    H: SwitchField,
    F: SmoothFftField,
    M: TrinomialModulus,
    S: ClearChannel,
{
    if source.len() != setup.source_len() {
        return Err(AkitaError::InvalidSize {
            expected: setup.source_len(),
            actual: source.len(),
        });
    }
    bind_statement(setup, commitment, point, claim, channel)?;
    let binary = prove_frontend::<H, S>(source, point, claim, channel)?;
    let mut expansion =
        left_expansion::<H>(source, &binary.point, setup.scalar_rows(), setup.columns())?;
    for value in &mut expansion {
        exchange_binary(channel, value)?;
    }
    let mut sampler = BinaryChallengeSampler::new(setup.profile().clone());
    let challenges = channel.fold_challenges(
        &mut sampler,
        b"akita/labinius/clear-fold/v1",
        setup.columns(),
    )?;
    let mut response = fold_integer::<H>(
        source,
        setup.scalar_rows(),
        setup.columns(),
        &challenges,
        setup.profile(),
    )?;
    if response
        .iter()
        .flatten()
        .any(|&coefficient| coefficient < setup.lower() || coefficient > setup.upper())
    {
        return Err(AkitaError::InvalidInput(
            "clear binary response leaves the accepted interval".into(),
        ));
    }
    exchange_response(channel, setup, &mut response)
}

/// Create and finish a standalone prover channel, returning canonical proof bytes.
pub fn prove_binary_clear_bytes<H, F, const D: usize, M>(
    setup: &BinaryClearSetup<F, D, M>,
    source: &[H::Source],
    commitment: &BinaryClearCommitment<F, D, M>,
    point: &[H],
    claim: H,
) -> Result<Vec<u8>, AkitaError>
where
    H: SwitchField,
    F: SmoothFftField,
    M: TrinomialModulus,
{
    let mut channel = new_prover()?;
    prove_binary_clear::<H, F, D, M, _>(setup, source, commitment, point, claim, &mut channel)?;
    Ok(finish_prover(channel))
}
