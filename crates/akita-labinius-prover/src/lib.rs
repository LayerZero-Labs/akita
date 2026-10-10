//! Prover-side binary-root components for the opt-in LaBinius extension.
//!
//! This crate commits binary host words and proves their host-field multilinear
//! evaluation with a clear integer response, which provides no zero knowledge,
//! and builds the lowered root witness (`lowered`). Setup admission lives in
//! the verifier crate: these functions accept any admitted
//! `BinaryClearSetup`, explicit or seed-derived.
//! The `root` reduction proves a binary claim, a prime claim or both about one
//! commitment and returns committed-table evaluation claims through an oracle
//! seam.

#![cfg(feature = "labinius")]

pub mod combined_kernel;
pub mod fold_kernel;
pub mod limb_commit_kernel;
pub mod lowered;
pub mod root;
pub mod root_sumcheck;

pub use root::{prove_root_reduction, prove_root_reduction_bytes, TransparentRootProverOracle};

pub use limb_commit_kernel::{
    commit_binary_clear_limb_prepared, commit_binary_clear_prepared, PreparedLimbCommitMatrix,
};

use akita_algebra::{binary::field_switch::SwitchField, TrinomialModulus};
use akita_challenges::{BinaryChallenge, BinaryChallengeSampler};
use akita_error::AkitaError;
use akita_labinius_verifier::{
    bind_statement,
    channel::{finish_prover, new_prover, FoldSearchChannel},
    codec::{exchange_binary, exchange_response},
    commitment::{apply_matrix, BinaryClearCommitment},
    endpoint::{fold_integer, left_expansion},
    frontend::prove_frontend,
    profile::BinaryClearSetup,
    source::pack_source_column,
};
use akita_params::FOLD_RESPONSE_ATTEMPTS;

/// Commit each packed source column under the caller's explicit Ajtai matrix.
///
/// This is the reference commitment for every admitted degree: one matrix
/// product modulo `(q, Phi)` per column.
pub fn commit_binary_clear<H, const D: usize, M>(
    setup: &BinaryClearSetup<D, M>,
    source: &[H::Source],
) -> Result<BinaryClearCommitment, AkitaError>
where
    H: SwitchField,
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
        let packed = pack_source_column::<H, D, M>(setup, source, column)?;
        images.extend(apply_matrix(setup, &packed)?);
    }
    Ok(BinaryClearCommitment { images })
}

/// Find and emit the fold-response nonce: the first value of the search domain
/// whose fold challenges give a response inside the setup's interval.
///
/// This is Akita's fold grinding (`specs/fold-linf-rejection.md`). Candidates
/// are previewed on a copy of the public transcript state, and only the
/// accepted nonce is written to the proof, directly before the fold-challenge
/// draw. The interval is what the response encoding can represent, as in
/// Akita, where the acceptance bounds are the balanced-digit range and not the
/// honest cap it was sized from. Exhausting the domain is an error.
pub(crate) fn search_fold_response<const D: usize, M, S>(
    setup: &BinaryClearSetup<D, M>,
    label: &[u8],
    channel: &mut S,
    mut fold: impl FnMut(&[BinaryChallenge]) -> Result<Vec<[i64; 162]>, AkitaError>,
) -> Result<(Vec<BinaryChallenge>, Vec<[i64; 162]>), AkitaError>
where
    M: TrinomialModulus,
    S: FoldSearchChannel,
{
    let mut sampler = BinaryChallengeSampler::new(setup.profile().clone());
    for mut nonce in 0..FOLD_RESPONSE_ATTEMPTS {
        let candidate =
            channel.preview_fold_challenges(&mut sampler, label, setup.columns(), nonce)?;
        let response = fold(&candidate)?;
        if response
            .iter()
            .flatten()
            .any(|&coefficient| coefficient < setup.lower() || coefficient > setup.upper())
        {
            continue;
        }
        let challenges =
            channel.fold_challenges(&mut sampler, label, setup.columns(), &mut nonce)?;
        if challenges != candidate {
            return Err(AkitaError::Internal(
                "fold-response preview differs from the committed challenges".into(),
            ));
        }
        tracing::info!(accepted_nonce = nonce, "labinius fold response nonce");
        return Ok((challenges, response));
    }
    Err(AkitaError::InvalidInput(format!(
        "fold grind exceeded {FOLD_RESPONSE_ATTEMPTS} attempts"
    )))
}

/// Prove a clear binary opening without finishing the caller's channel.
///
/// The fold-response nonce is searched as in the root reduction; the response
/// is sent in the clear and lies in the setup's interval.
pub fn prove_binary_clear<H, const D: usize, M, S>(
    setup: &BinaryClearSetup<D, M>,
    source: &[H::Source],
    commitment: &BinaryClearCommitment,
    point: &[H],
    claim: H,
    channel: &mut S,
) -> Result<(), AkitaError>
where
    H: SwitchField,
    M: TrinomialModulus,
    S: FoldSearchChannel,
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
    let (_, mut response) = search_fold_response(
        setup,
        b"akita/labinius/clear-fold/v1",
        channel,
        |challenges| {
            fold_integer::<H>(
                source,
                setup.scalar_rows(),
                setup.columns(),
                challenges,
                setup.profile(),
            )
        },
    )?;
    exchange_response(channel, setup, &mut response)
}

/// Create and finish a standalone prover channel, returning canonical proof bytes.
pub fn prove_binary_clear_bytes<H, const D: usize, M>(
    setup: &BinaryClearSetup<D, M>,
    source: &[H::Source],
    commitment: &BinaryClearCommitment,
    point: &[H],
    claim: H,
) -> Result<Vec<u8>, AkitaError>
where
    H: SwitchField,
    M: TrinomialModulus,
{
    let mut channel = new_prover()?;
    prove_binary_clear::<H, D, M, _>(setup, source, commitment, point, claim, &mut channel)?;
    Ok(finish_prover(channel))
}
