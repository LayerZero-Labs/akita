//! Tiered setup commitment derivation for Phase D-full.
//!
//! Per book §5.3, the B-side commitment to `S` is precomputed during
//! setup: `B_S · t̂_S = u_S`. Under the tiered design (book §5.4), `S`
//! is split into `k = f²` row-major chunks; each chunk is committed
//! independently under shared per-chunk matrices, then a tier-3
//! meta-commitment binds the collection.
//!
//! This module owns the prover-side derivation. The output
//! [`TieredSetupCommitments`] will be consumed by Phase D-full slice G's
//! tiered (`k = 64`) extension to the multi-claim recursive open; the
//! verifier mirrors via [`AkitaVerifierSetup`].
//!
//! Derivation uses the existing commitment kernels (`commit_with_params`)
//! by wrapping each chunk and the meta-collection as a [`DensePoly`].
//! This module is shape-agnostic and accepts whatever `LevelParams` the
//! caller supplies for each tier; slice G picks per-chunk and meta-tier
//! parameters via the cascade-aware planner.

use crate::api::commitment::commit_with_params;
use crate::api::setup::AkitaProverSetup;
use crate::backend::DensePoly;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::{
    AkitaCommitmentHint, FlatRingVec, LevelParams, TieredSetupCommitments, TieredSetupParams,
    TieredSetupProverExtras,
};

/// Derive the tiered B-side commitments for the shared setup matrix `S`.
///
/// The shared matrix is interpreted as `n_S` ring elements at dimension
/// `D` (read via `setup.expanded.shared_matrix.ring_view::<D>(1, n_S)`).
/// `n_S` must be a multiple of `tier.num_chunks` (the row-major chunk
/// partition) and the chunk size must be a power of two (required by
/// [`DensePoly::from_ring_coeffs`]'s `num_vars` recovery).
///
/// `chunk_params` describes the commitment shape used for each chunk;
/// `meta_params` describes the tier-3 meta-commitment shape. Slice E
/// will compute these from a cascade-aware planner search; for now the
/// caller (typically slice C's opening protocol) supplies them.
///
/// Returns the precomputed B-side commitments. The D-side commitments
/// remain proving-time work because they involve the witness `w` jointly
/// with `S` per book §5.3.
///
/// # Errors
///
/// Returns an error if `n_S` is not a multiple of `k`, if either tier
/// produces a non-power-of-two chunk size, or if the underlying commit
/// kernel rejects the chunk layout.
#[tracing::instrument(
    skip_all,
    name = "derive_tiered_setup_commitments",
    fields(
        n_chunks = tier.num_chunks,
        shrink = tier.shrink_factor,
    )
)]
pub fn derive_tiered_setup_commitments<F, const D: usize>(
    setup: &AkitaProverSetup<F, D>,
    chunk_params: &LevelParams,
    meta_params: &LevelParams,
    tier: TieredSetupParams,
) -> Result<TieredSetupCommitments<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let n_s = setup.expanded.shared_matrix.total_ring_elements_at::<D>();
    if tier.num_chunks == 0 {
        return Err(AkitaError::InvalidInput(
            "tiered setup requires num_chunks >= 1".to_string(),
        ));
    }
    if !n_s.is_multiple_of(tier.num_chunks) {
        return Err(AkitaError::InvalidInput(format!(
            "shared matrix has {n_s} ring elements but tier num_chunks = {} \
             requires a multiple",
            tier.num_chunks
        )));
    }
    let chunk_n = n_s / tier.num_chunks;
    if chunk_n == 0 || !chunk_n.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "tiered chunk size {chunk_n} must be a non-zero power of two"
        )));
    }

    let (commitments, _extras) =
        derive_tiered_setup_full_commitments(setup, chunk_params, meta_params, tier)?;
    Ok(commitments)
}

/// Derive the tiered B-side commitments together with the prover-only
/// material required to fold and open `S` recursively.
///
/// Performs the same work as [`derive_tiered_setup_commitments`] and
/// additionally retains the per-chunk and meta-tier
/// [`AkitaCommitmentHint`]s produced by [`commit_with_params`]. The
/// hints carry the digit-decomposed inner witnesses and the recomposed
/// `t` rows; the recursive Hachi PCS step at the next fold level needs
/// both to discharge `S(r_setup) = y_setup` against the meta-bound
/// chunk commitments.
///
/// The split into [`TieredSetupCommitments`] (verifier-derivable) and
/// [`TieredSetupProverExtras`] (prover-only) keeps the verifier-facing
/// type lean while exposing the prover-side material that the
/// recursive opening protocol consumes.
///
/// # Errors
///
/// Returns the same errors as [`derive_tiered_setup_commitments`].
#[tracing::instrument(
    skip_all,
    name = "derive_tiered_setup_full_commitments",
    fields(
        n_chunks = tier.num_chunks,
        shrink = tier.shrink_factor,
    )
)]
pub fn derive_tiered_setup_full_commitments<F, const D: usize>(
    setup: &AkitaProverSetup<F, D>,
    chunk_params: &LevelParams,
    meta_params: &LevelParams,
    tier: TieredSetupParams,
) -> Result<(TieredSetupCommitments<F, D>, TieredSetupProverExtras<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let bundle = derive_tiered_setup_handle_bundle(setup, chunk_params, meta_params, tier)?;
    let commitments = TieredSetupCommitments {
        chunk_b_commitments: bundle.chunk_commitments_typed,
        meta_b_commitment: bundle.meta_commitment_typed,
        params: tier,
    };
    commitments.validate_shape()?;
    let extras = TieredSetupProverExtras {
        chunk_hints: bundle.chunk_hints_typed,
        meta_hint: bundle.meta_hint_typed,
    };
    Ok((commitments, extras))
}

/// Full tiered routing bundle for one routed `S` handle (book §5.4).
///
/// Returned by [`derive_tiered_setup_handle_bundle`]. Contains everything
/// the next fold level needs to expand the routed S claim into
/// `k + 1` claims:
/// - `chunk_polys`: per-chunk polynomial coefficients (`FlatRingVec`).
/// - `chunk_commitments_typed` / `chunk_commitments_flat`: per-chunk
///   B-side commitments `u_{S,j}` (typed for `TieredSetupCommitments`,
///   flat for the recursive handle wire).
/// - `chunk_hints_typed`: typed `AkitaCommitmentHint` per chunk for
///   the chunk commit (used by both `TieredSetupProverExtras` and the
///   recursive handle path via `RecursiveCommitmentHintCache`).
/// - `meta_input_poly`: the concatenated `(u_{S,j})_j` polynomial
///   padded to a power of two (`FlatRingVec`).
/// - `meta_commitment_typed` / `meta_commitment_flat`: tier-3 meta
///   B-side commitment `u_meta`.
/// - `meta_hint_typed`: typed `AkitaCommitmentHint` for the meta commit.
pub struct TieredSetupHandleBundle<F: FieldCore, const D: usize> {
    /// Per-chunk polynomial coefficients (each chunk has `chunk_n` ring
    /// elements, derived from the shared matrix's row-major linearization).
    pub chunk_polys: Vec<FlatRingVec<F>>,
    /// Per-chunk B-side commitment vectors typed as ring elements.
    pub chunk_commitments_typed: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Per-chunk B-side commitment vectors flattened for handle plumbing.
    pub chunk_commitments_flat: Vec<FlatRingVec<F>>,
    /// Per-chunk typed commitment hints.
    pub chunk_hints_typed: Vec<AkitaCommitmentHint<F, D>>,
    /// Meta-tier input polynomial (concatenated chunk commitments,
    /// padded to next power-of-two ring elements).
    pub meta_input_poly: FlatRingVec<F>,
    /// Meta-tier B-side commitment typed as ring elements.
    pub meta_commitment_typed: Vec<CyclotomicRing<F, D>>,
    /// Meta-tier B-side commitment flattened for handle plumbing.
    pub meta_commitment_flat: FlatRingVec<F>,
    /// Meta-tier typed commitment hint.
    pub meta_hint_typed: AkitaCommitmentHint<F, D>,
}

/// Derive the full tiered routing bundle in one pass.
///
/// Performs the same per-chunk + meta commit work as
/// [`derive_tiered_setup_full_commitments`] and additionally returns
/// the chunk and meta polynomial coefficient material so the next fold
/// level can re-evaluate them at the routed setup opening point.
///
/// The verifier mirror (`derive_tiered_setup_material_for_verifier` in
/// the verifier crate) must use the same chunk partition rule:
/// row-major linearization of the shared matrix at dimension `D`,
/// chunked into `tier.num_chunks` equal pieces of `chunk_n = n_s / k`
/// ring elements each.
///
/// # Errors
///
/// Returns the same errors as [`derive_tiered_setup_full_commitments`].
pub fn derive_tiered_setup_handle_bundle<F, const D: usize>(
    setup: &AkitaProverSetup<F, D>,
    chunk_params: &LevelParams,
    meta_params: &LevelParams,
    tier: TieredSetupParams,
) -> Result<TieredSetupHandleBundle<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let n_s = setup.expanded.shared_matrix.total_ring_elements_at::<D>();
    if tier.num_chunks == 0 {
        return Err(AkitaError::InvalidInput(
            "tiered setup requires num_chunks >= 1".to_string(),
        ));
    }
    if !n_s.is_multiple_of(tier.num_chunks) {
        return Err(AkitaError::InvalidInput(format!(
            "shared matrix has {n_s} ring elements but tier num_chunks = {} \
             requires a multiple",
            tier.num_chunks
        )));
    }
    let chunk_n = n_s / tier.num_chunks;
    if chunk_n == 0 || !chunk_n.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "tiered chunk size {chunk_n} must be a non-zero power of two"
        )));
    }

    let view = setup.expanded.shared_matrix.ring_view::<D>(1, n_s);
    let s_rings: &[CyclotomicRing<F, D>] = view.row(0);

    let mut chunk_polys: Vec<FlatRingVec<F>> = Vec::with_capacity(tier.num_chunks);
    let mut chunk_commitments_typed: Vec<Vec<CyclotomicRing<F, D>>> =
        Vec::with_capacity(tier.num_chunks);
    let mut chunk_commitments_flat: Vec<FlatRingVec<F>> = Vec::with_capacity(tier.num_chunks);
    let mut chunk_hints_typed: Vec<AkitaCommitmentHint<F, D>> = Vec::with_capacity(tier.num_chunks);
    for j in 0..tier.num_chunks {
        let start = j * chunk_n;
        let end = start + chunk_n;
        let chunk_slice = &s_rings[start..end];
        let chunk_poly = DensePoly::<F, D>::from_ring_coeffs(chunk_slice.to_vec());
        let (commitment, hint) =
            commit_with_params(std::slice::from_ref(&chunk_poly), setup, chunk_params)?;
        chunk_polys.push(FlatRingVec::from_ring_elems::<D>(chunk_slice));
        chunk_commitments_flat.push(FlatRingVec::from_ring_elems::<D>(&commitment.u));
        chunk_commitments_typed.push(commitment.u);
        chunk_hints_typed.push(hint);
    }

    let meta_len = chunk_commitments_typed.iter().map(Vec::len).sum::<usize>();
    if meta_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "tiered meta commitment input is empty".to_string(),
        ));
    }
    let next_pow2 = meta_len.next_power_of_two();
    let mut meta_input: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(next_pow2);
    for chunk in &chunk_commitments_typed {
        meta_input.extend_from_slice(chunk);
    }
    meta_input.resize(next_pow2, CyclotomicRing::<F, D>::zero());

    let meta_input_poly = FlatRingVec::from_ring_elems::<D>(&meta_input);
    let meta_poly = DensePoly::<F, D>::from_ring_coeffs(meta_input);
    let (meta_commitment, meta_hint) =
        commit_with_params(std::slice::from_ref(&meta_poly), setup, meta_params)?;
    let meta_commitment_flat = FlatRingVec::from_ring_elems::<D>(&meta_commitment.u);
    Ok(TieredSetupHandleBundle {
        chunk_polys,
        chunk_commitments_typed,
        chunk_commitments_flat,
        chunk_hints_typed,
        meta_input_poly,
        meta_commitment_typed: meta_commitment.u,
        meta_commitment_flat,
        meta_hint_typed: meta_hint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::{SparseChallengeConfig, Stage1ChallengeShape};
    use akita_field::Prime128OffsetA7F7;
    use akita_types::AjtaiKeyParams;

    type F = Prime128OffsetA7F7;
    const D_TEST: usize = 32;

    /// Tiny setup: `n_S = max_rows × max_stride = 2` ring elements at
    /// `D = 32` (one allocated row × stride 2). Choose values small
    /// enough that `k > n_S` fails the divisibility check.
    fn tiny_setup() -> AkitaProverSetup<F, D_TEST> {
        AkitaProverSetup::<F, D_TEST>::generate_with_capacity(4, 1, 1, 1, 2).expect("setup")
    }

    /// Manually construct a `LevelParams` valid for one-chunk commit of a
    /// single-block 1-ring-element polynomial. This is the smallest shape
    /// the kernel accepts; it lets us exercise the chunk loop and the
    /// meta-tier accumulation without depending on planner output.
    fn one_block_level_params() -> LevelParams {
        LevelParams {
            ring_dimension: D_TEST,
            log_basis: 2,
            a_key: AjtaiKeyParams::new_unchecked(1, 1, 0, D_TEST),
            b_key: AjtaiKeyParams::new_unchecked(1, 1, 0, D_TEST),
            d_key: AjtaiKeyParams::new_unchecked(1, 1, 0, D_TEST),
            num_blocks: 1,
            block_len: 1,
            m_vars: 0,
            r_vars: 0,
            stage1_config: SparseChallengeConfig::ExactShell {
                count_mag1: 1,
                count_mag2: 0,
            },
            stage1_challenge_shape: Stage1ChallengeShape::Flat,
            use_setup_claim_reduction: false,
            num_digits_commit: 1,
            num_digits_open: 1,
            num_digits_fold: 1,
            groups: None,
        }
    }

    #[test]
    fn rejects_uneven_chunking() {
        let setup = tiny_setup();
        let lp = one_block_level_params();
        // n_S = 2 ring elements; tier k = 16 does not divide 2.
        let tier = TieredSetupParams::new(4).unwrap();
        let err = derive_tiered_setup_commitments(&setup, &lp, &lp, tier)
            .expect_err("k = 16 does not divide n_S = 2");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }

    #[test]
    fn rejects_chunk_size_zero() {
        let setup = tiny_setup();
        let lp = one_block_level_params();
        let custom_tier = TieredSetupParams {
            shrink_factor: 0,
            num_chunks: 0,
        };
        let err = derive_tiered_setup_commitments(&setup, &lp, &lp, custom_tier)
            .expect_err("num_chunks = 0 rejected");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }
}
