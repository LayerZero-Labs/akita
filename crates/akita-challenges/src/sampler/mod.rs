//! Sparse challenge sampling via Fiat-Shamir with PRG expansion.
//!
//! Challenges are derived by absorbing context into the transcript once,
//! drawing a 32-byte PRG seed, and expanding it via SHAKE256 XOF
//! ([`xof::XofCursor`]) into all per-challenge randomness. This replaces the
//! previous per-challenge hash chain with a single seed derivation followed
//! by fast XOF expansion, providing ~6x speedup for large batch sizes (e.g.
//! 4096 challenges).
//!
//! Position and shell sampling use bitmask rejection sampling to achieve
//! zero modulo bias, ensuring ≥128-bit security in the Fiat-Shamir challenge
//! distribution.
//!
//! The dispatcher in [`parse_challenge`] routes each [`SparseChallengeConfig`]
//! variant to its dedicated submodule:
//!
//! - [`SparseChallengeConfig::Uniform`] → [`uniform::sample_uniform_sparse`]
//! - [`SparseChallengeConfig::ExactShell`] → [`exact_shell::sample_exact_shell_sparse`]
//! - [`SparseChallengeConfig::BoundedL1Ball`] → [`bounded_l1::sample_bounded_l1_into`]

mod bounded_l1;
mod exact_shell;
mod uniform;
mod xof;

use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use akita_transcript::Transcript;

use crate::{SparseChallenge, SparseChallengeConfig};

use bounded_l1::{
    build_ways_table, sample_bounded_l1_into, OwnedWaysTable, WaysTableRef, PRESET_D32_M8_B121_B,
    PRESET_D32_M8_B121_D, PRESET_D32_M8_B121_M, PRESET_D32_M8_B121_TABLE,
};
use exact_shell::sample_exact_shell_sparse;
use uniform::{sample_uniform_sparse, MAX_STACK_RING_DIM};
use xof::XofCursor;

/// Per-batch precomputed state for the bounded-`L1` sampler.
///
/// For [`SparseChallengeConfig::BoundedL1Ball`] this holds the WAYS table
/// view: a borrowed `&'static` for the production `(D=32, M=8, B=121)` preset
/// (no table construction at all), or an owned `Vec`-backed table for any
/// other triple. For other variants the scratch carries no state.
struct SamplerScratch {
    /// Owned WAYS table for the runtime-built path. Held alongside
    /// `bounded_l1_view` so the view's borrow stays valid for the lifetime
    /// of `Self`.
    _bounded_l1_owned: Option<OwnedWaysTable>,
    /// `Some(view)` iff the active config is `BoundedL1Ball`. Borrows from
    /// either `_bounded_l1_owned` (runtime build) or the static
    /// [`PRESET_D32_M8_B121_TABLE`] (production preset).
    bounded_l1_view: Option<WaysTableRef<'static>>,
}

impl SamplerScratch {
    fn new<const D: usize>(cfg: &SparseChallengeConfig) -> Result<Self, AkitaError> {
        let (owned, view) = match cfg {
            SparseChallengeConfig::BoundedL1Ball {
                max_abs_coeff,
                l1_bound,
            } => {
                let m = *max_abs_coeff as usize;
                let b = *l1_bound as usize;
                if D == PRESET_D32_M8_B121_D
                    && m == PRESET_D32_M8_B121_M
                    && b == PRESET_D32_M8_B121_B
                {
                    // Production preset: skip table construction entirely;
                    // the static `PRESET_D32_M8_B121_TABLE` already lives in
                    // `.rodata` and the borrow is genuinely `'static`.
                    (None, Some(PRESET_D32_M8_B121_TABLE))
                } else {
                    let owned = build_ways_table(D, m, b)?;
                    // The truncated-`2^128` sampler requires
                    // `WAYS[D][B] >= 2^128` so that every top-level draw
                    // `r in [0, 2^128)` lands in some valid descent path. A
                    // `Wide` value is `>= 2^128` iff its high half is non-zero.
                    let total = owned.view().at(D, b);
                    if total.hi == 0 {
                        return Err(AkitaError::InvalidInput(format!(
                            "BoundedL1Ball: support |WAYS[{D}][{b}]| < 2^128 \
                             cannot drive the truncated-2^128 sampler; \
                             use a larger l1_bound or implement an exact-uniform fallback"
                        )));
                    }
                    // Safety: the owned table is stored in `Self` alongside
                    // the view; the borrow is alive for as long as `Self` is
                    // live. The `'static` is a uniform field shape with the
                    // const-preset case; we never expose this view past
                    // `Self`'s lifetime.
                    let view = unsafe {
                        std::mem::transmute::<WaysTableRef<'_>, WaysTableRef<'static>>(owned.view())
                    };
                    (Some(owned), Some(view))
                }
            }
            _ => (None, None),
        };
        Ok(Self {
            _bounded_l1_owned: owned,
            bounded_l1_view: view,
        })
    }
}

/// Parse a single sparse challenge from a streaming XOF cursor.
fn parse_challenge<const D: usize>(
    cursor: &mut XofCursor,
    cfg: &SparseChallengeConfig,
    scratch: &SamplerScratch,
) -> SparseChallenge {
    match cfg {
        SparseChallengeConfig::Uniform {
            weight,
            nonzero_coeffs,
        } => sample_uniform_sparse(cursor, D, *weight, nonzero_coeffs),
        SparseChallengeConfig::ExactShell {
            count_mag1,
            count_mag2,
        } => sample_exact_shell_sparse(cursor, D, *count_mag1, *count_mag2),
        SparseChallengeConfig::BoundedL1Ball {
            max_abs_coeff,
            l1_bound,
        } => {
            let table = scratch
                .bounded_l1_view
                .expect("BoundedL1Ball requires a precomputed WAYS view in SamplerScratch");
            // The output `SparseChallenge` owns its `Vec`s, so each call
            // ultimately needs its own allocation. We still avoid the prior
            // 2-Vec-grow pattern (`Vec::with_capacity` inside the inner loop
            // followed by repeated `push`) by sizing both buffers to the
            // tight upper bound `D.min(B)` once and letting `push` grow into
            // the reserved capacity without further reallocs.
            let cap = D.min(*l1_bound as usize);
            let mut positions: Vec<u32> = Vec::with_capacity(cap);
            let mut coeffs: Vec<i16> = Vec::with_capacity(cap);
            sample_bounded_l1_into::<D>(
                cursor,
                table,
                *max_abs_coeff as usize,
                *l1_bound as usize,
                &mut positions,
                &mut coeffs,
            );
            SparseChallenge { positions, coeffs }
        }
    }
}

#[inline]
fn sparse_challenge_absorb_buf<const D: usize>(
    label: &[u8],
    instance_tag: u64,
    cfg: &SparseChallengeConfig,
) -> Vec<u8> {
    let domain_sep = cfg.domain_separator_bytes();
    let mut absorb_buf = Vec::with_capacity(label.len() + 8 + 8 + domain_sep.len());
    absorb_buf.extend_from_slice(label);
    absorb_buf.extend_from_slice(&instance_tag.to_le_bytes());
    absorb_buf.extend_from_slice(&(D as u64).to_le_bytes());
    absorb_buf.extend_from_slice(&domain_sep);
    absorb_buf
}

/// Absorb context into the transcript, derive a PRG seed, and create a
/// streaming XOF cursor for challenge randomness.
fn derive_xof_cursor<F, T>(transcript: &mut T, absorb_data: &[u8]) -> XofCursor
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, absorb_data);
    let seed = transcript.challenge_bytes(CHALLENGE_SPARSE_CHALLENGE, 32);
    XofCursor::from_seed(&seed)
}

/// Sample `n` sparse challenges from a transcript, returning the sparse
/// representation directly.
///
/// Absorbs the context (label, count, ring degree, config) into the
/// transcript once, derives a single 32-byte PRG seed, and expands it
/// via SHAKE256 XOF into all per-challenge randomness in one streaming
/// pass.
///
/// # Errors
///
/// Returns an error if challenge sampling fails.
#[tracing::instrument(skip_all, name = "sample_sparse_challenges")]
pub fn sample_sparse_challenges<F, T, const D: usize>(
    transcript: &mut T,
    label: &[u8],
    n: usize,
    cfg: &SparseChallengeConfig,
) -> Result<Vec<SparseChallenge>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    if D > MAX_STACK_RING_DIM {
        return Err(AkitaError::InvalidInput(format!(
            "ring dimension {D} exceeds sampling stack-buffer limit ({MAX_STACK_RING_DIM})"
        )));
    }
    cfg.validate::<D>()
        .map_err(|e| AkitaError::InvalidInput(format!("invalid sparse challenge config: {e}")))?;

    let scratch = SamplerScratch::new::<D>(cfg)?;
    let absorb_buf = sparse_challenge_absorb_buf::<D>(label, n as u64, cfg);
    let mut cursor = derive_xof_cursor::<F, T>(transcript, &absorb_buf);
    let mut challenges = Vec::with_capacity(n);
    for _ in 0..n {
        challenges.push(parse_challenge::<D>(&mut cursor, cfg, &scratch));
    }
    Ok(challenges)
}
