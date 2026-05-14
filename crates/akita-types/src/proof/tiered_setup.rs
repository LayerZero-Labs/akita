//! Tiered commitment data shapes for the Phase D-full recursive `S` opening.
//!
//! The shared setup polynomial `S` is committed at proving time via the
//! tiered design from book §5.4: `S` is split into `k = f²` row-major
//! chunks; each chunk is committed under per-chunk shared matrices whose
//! column width is `1/f` of the baseline. A tier-3 meta-commitment binds
//! the collection of per-chunk commitments via a standard Akita commit.
//!
//! `TieredSetupCommitments` holds the precomputed B-side material per
//! book §5.3 ("`B_S t̂_S = u_S` precomputed during setup") plus the
//! tier-3 meta material. The D-side commitments are computed at proving
//! time because they involve the witness `w` jointly with `S`; only the
//! B-side material can be precomputed.
//!
//! `TieredSetupProverExtras` carries the prover-only material used to
//! fold and open `S` recursively: per-chunk and meta-tier digit
//! decompositions, recomposed `t` rows, and recursive witness handles.
//! The verifier never touches this data.
//!
//! Derivation lives in `akita-prover::api::tiered_setup`; this module
//! defines the types and their basic invariants only.

use crate::{AkitaCommitmentHint, FlatRingVec};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, FieldCore};

/// Parameters of the tiered commitment design (book §5.4).
///
/// Production constants are `f = 8`, `k = 64`; see `PRODUCTION`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TieredSetupParams {
    /// Shrink factor `f`. Per-chunk shared matrices have `1/f` the column
    /// width of the baseline.
    pub shrink_factor: usize,
    /// Number of chunks `k = f²`. Constant given `shrink_factor`.
    pub num_chunks: usize,
}

impl TieredSetupParams {
    /// Production tiered params per book §5.4 Table 1: `f = 8`, `k = 64`.
    ///
    /// At `f = 8` the setup matrix shrinks 36× (nv = 32) or 8× (nv = 44),
    /// the Technique-2 cascade ratio drops to `≈ 1`, and the witness
    /// growth is `1.8`-`3.0×` — book's sweet spot.
    pub const PRODUCTION: Self = Self {
        shrink_factor: 8,
        num_chunks: 64,
    };

    /// Un-tiered (`f = 1`, `k = 1`) shape: the Slice F (book §5.3
    /// un-tiered split commitment) baseline. Distinguished structurally
    /// from `None` so call sites can carry an explicit
    /// [`TieredSetupParams`] without forcing every consumer to handle
    /// the optional case.
    pub const fn un_tiered() -> Self {
        Self {
            shrink_factor: 1,
            num_chunks: 1,
        }
    }

    /// Whether this tier triggers the tiered commit path. Returns
    /// `false` exactly when `shrink_factor == 1` (un-tiered Slice F
    /// shape) and `true` for `f ≥ 2` (Slice G tiered shape).
    #[inline]
    pub const fn is_tiered(self) -> bool {
        self.shrink_factor > 1
    }

    /// Build params from a shrink factor `f`.
    ///
    /// # Errors
    ///
    /// Returns an error if `f == 0` or `f²` overflows `usize`.
    pub fn new(shrink_factor: usize) -> Result<Self, AkitaError> {
        if shrink_factor == 0 {
            return Err(AkitaError::InvalidInput(
                "tiered shrink factor must be >= 1".to_string(),
            ));
        }
        let num_chunks = shrink_factor.checked_mul(shrink_factor).ok_or_else(|| {
            AkitaError::InvalidInput("tiered shrink factor squared overflows usize".to_string())
        })?;
        Ok(Self {
            shrink_factor,
            num_chunks,
        })
    }

    /// `log₂(f)`. Required to be exact; per the book design `f` is a
    /// power of two so the chunked layout aligns with the level's
    /// `(m_vars, r_vars)` split.
    ///
    /// # Errors
    ///
    /// Returns an error if `shrink_factor` is not a power of two.
    pub fn log2_shrink(self) -> Result<u32, AkitaError> {
        if !self.shrink_factor.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "tiered shrink factor {} must be a power of two",
                self.shrink_factor
            )));
        }
        Ok(self.shrink_factor.trailing_zeros())
    }

    /// `log₂(k) = 2 log₂(f)`. Number of variables consumed by chunk
    /// indexing in the tiered layout.
    ///
    /// # Errors
    ///
    /// Returns an error if `shrink_factor` is not a power of two.
    pub fn log2_num_chunks(self) -> Result<u32, AkitaError> {
        Ok(2 * self.log2_shrink()?)
    }
}

impl Default for TieredSetupParams {
    fn default() -> Self {
        Self::PRODUCTION
    }
}

/// Precomputed B-side commitments for the shared setup polynomial `S`
/// under the tiered design.
///
/// Per book §5.3 split-commitment rule, only the B-side material can be
/// precomputed at setup time. The D-side `D · ê_S` is computed at
/// proving time because it concatenates with the witness digits.
///
/// Layout:
///
/// - `chunk_b_commitments[j]` = `u_{S,j} = B_chunk · t̂_{S,j}` for chunk
///   `j ∈ 0..k`, each of length `n_{B,chunk}` ring elements.
/// - `meta_b_commitment` = `u_meta = B_meta · t̂_meta` where `t̂_meta` is
///   the digit decomposition of the concatenated `(u_{S,j})_j` rows,
///   per book §5.4 check group 7. Length `n_{B,meta}`.
/// - `params` is the parameter set used to derive these commitments;
///   the next-level verifier must derive under the same params.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TieredSetupCommitments<F: FieldCore, const D: usize> {
    /// Per-chunk B-side commitment vectors. `len() == params.num_chunks`.
    pub chunk_b_commitments: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Tier-3 meta B-side commitment.
    pub meta_b_commitment: Vec<CyclotomicRing<F, D>>,
    /// Parameters used to derive these commitments. Subsequent
    /// derivation calls under different params must produce a different
    /// `TieredSetupCommitments`.
    pub params: TieredSetupParams,
}

impl<F: FieldCore, const D: usize> TieredSetupCommitments<F, D> {
    /// Number of chunks in this tiered commitment. Matches
    /// `params.num_chunks` and `chunk_b_commitments.len()`.
    pub fn num_chunks(&self) -> usize {
        self.chunk_b_commitments.len()
    }

    /// Per-chunk B-side commitment as a `FlatRingVec` view for transcript
    /// absorption.
    ///
    /// # Errors
    ///
    /// Returns an error if `chunk_idx` is out of range.
    pub fn chunk_commitment_flat(&self, chunk_idx: usize) -> Result<FlatRingVec<F>, AkitaError> {
        let chunk = self.chunk_b_commitments.get(chunk_idx).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "tiered chunk index {chunk_idx} out of range (num_chunks = {})",
                self.params.num_chunks
            ))
        })?;
        Ok(FlatRingVec::from_ring_elems::<D>(chunk))
    }

    /// Tier-3 meta B-side commitment as a `FlatRingVec` view for
    /// transcript absorption.
    pub fn meta_commitment_flat(&self) -> FlatRingVec<F> {
        FlatRingVec::from_ring_elems::<D>(&self.meta_b_commitment)
    }

    /// Validate internal shape consistency: chunk count and uniformity
    /// of per-chunk commitment width.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored chunk count diverges from
    /// `params.num_chunks`, or if per-chunk commitment widths are not
    /// uniform across the `k` chunks.
    pub fn validate_shape(&self) -> Result<(), AkitaError> {
        if self.chunk_b_commitments.len() != self.params.num_chunks {
            return Err(AkitaError::InvalidProof);
        }
        if let Some(first) = self.chunk_b_commitments.first() {
            let n_b_chunk = first.len();
            for (j, chunk) in self.chunk_b_commitments.iter().enumerate() {
                if chunk.len() != n_b_chunk {
                    return Err(AkitaError::InvalidSetup(format!(
                        "tiered setup chunk {j} has width {} but chunk 0 has width {n_b_chunk}",
                        chunk.len()
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Prover-only material accompanying [`TieredSetupCommitments`].
///
/// The recursive opening of `S` at the next fold level requires the
/// digit-decomposed per-chunk and meta-tier inner witnesses, plus the
/// recomposed `t` rows used by the multi-claim quadratic equation.
/// `TieredSetupProverExtras` carries this material in the same per-chunk
/// + meta layout as [`TieredSetupCommitments`].
///
/// `chunk_hints[j]` is the [`AkitaCommitmentHint`] for chunk `j`'s
/// commitment under the chunk-tier `LevelParams`. `meta_hint` is the
/// hint for the meta-tier commitment under the meta-tier `LevelParams`.
///
/// The hints are constructed at setup time via `derive_tiered_setup_full_commitments`
/// in the prover crate. They are not serialized; the prover regenerates
/// them on demand on the first proof.
#[derive(Debug, Clone)]
pub struct TieredSetupProverExtras<F: FieldCore, const D: usize> {
    /// Per-chunk commitment hints, parallel to `chunk_b_commitments`.
    pub chunk_hints: Vec<AkitaCommitmentHint<F, D>>,
    /// Tier-3 meta commitment hint.
    pub meta_hint: AkitaCommitmentHint<F, D>,
}

impl<F: FieldCore, const D: usize> TieredSetupProverExtras<F, D> {
    /// Number of chunk hints. Must match the paired
    /// `TieredSetupCommitments::num_chunks()`.
    pub fn num_chunks(&self) -> usize {
        self.chunk_hints.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_constants_match_book() {
        let p = TieredSetupParams::PRODUCTION;
        assert_eq!(p.shrink_factor, 8);
        assert_eq!(p.num_chunks, 64);
        assert_eq!(p.log2_shrink().unwrap(), 3);
        assert_eq!(p.log2_num_chunks().unwrap(), 6);
    }

    #[test]
    fn new_rejects_zero_shrink() {
        assert!(TieredSetupParams::new(0).is_err());
    }

    #[test]
    fn new_computes_k_as_f_squared() {
        let p = TieredSetupParams::new(4).unwrap();
        assert_eq!(p.num_chunks, 16);
        assert_eq!(p.log2_shrink().unwrap(), 2);
        assert_eq!(p.log2_num_chunks().unwrap(), 4);
    }

    #[test]
    fn log2_shrink_rejects_non_power_of_two() {
        let p = TieredSetupParams::new(3).unwrap();
        assert!(p.log2_shrink().is_err());
    }

    #[test]
    fn default_is_production() {
        assert_eq!(TieredSetupParams::default(), TieredSetupParams::PRODUCTION);
    }
}
