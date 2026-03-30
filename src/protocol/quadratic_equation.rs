//! Quadratic equation builder for the Hachi PCS (§4.2).
//!
//! This module encapsulates the stage-1 prover logic and the generation of
//! the quadratic equation components M, y, z, and v.

use crate::algebra::{CyclotomicRing, SparseChallenge};
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::challenges::sparse::sample_sparse_challenges;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::linear::{
    flatten_i8_blocks, fused_split_eq_quotients, mat_vec_mul_ntt_single_i8,
    mat_vec_mul_ntt_single_i8_cyclic,
};
use crate::protocol::commitment::{
    CommitmentConfig, HachiCommitmentLayout, HachiExpandedSetup, HachiLevelParams, RingCommitment,
};
use crate::protocol::hachi_poly_ops::{DecomposeFoldWitness, HachiPolyOps, RecursiveWitnessView};
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::proof::{
    HachiBatchedCommitmentHint, HachiCommitmentHint, RingSliceSerializer,
};
use crate::protocol::transcript::labels::{ABSORB_PROVER_V, CHALLENGE_STAGE1_FOLD};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};
use std::iter::repeat_n;
use std::marker::PhantomData;
use std::time::Instant;

fn beta_linf_fold_bound_with_num_claims(
    r: usize,
    challenge_l1_mass: usize,
    log_basis: u32,
    num_claims: usize,
) -> Result<u128, HachiError> {
    let beta = crate::protocol::commitment::beta_linf_fold_bound(r, challenge_l1_mass, log_basis)?;
    beta.checked_mul(num_claims as u128)
        .ok_or_else(|| HachiError::InvalidSetup("batched beta bound overflow".to_string()))
}

fn validate_decompose_fold<F: FieldCore + CanonicalField, const D: usize>(
    z: DecomposeFoldWitness<F, D>,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> Result<DecomposeFoldWitness<F, D>, HachiError> {
    validate_decompose_fold_with_num_claims(z, level_params, layout, 1)
}

fn validate_decompose_fold_with_num_claims<F: FieldCore + CanonicalField, const D: usize>(
    z: DecomposeFoldWitness<F, D>,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    num_claims: usize,
) -> Result<DecomposeFoldWitness<F, D>, HachiError> {
    let norm = u128::from(z.centered_inf_norm);
    let beta = beta_linf_fold_bound_with_num_claims(
        layout.r_vars,
        level_params.challenge_l1_mass,
        layout.log_basis,
        num_claims,
    )?;
    if norm > beta {
        return Err(HachiError::InvalidInput(format!(
            "prover abort: ||z||_inf = {norm} > beta = {beta}"
        )));
    }
    Ok(z)
}

fn aggregate_decompose_fold_witnesses<F: FieldCore, const D: usize>(
    witnesses: Vec<DecomposeFoldWitness<F, D>>,
) -> Result<DecomposeFoldWitness<F, D>, HachiError> {
    let Some((first, rest)) = witnesses.split_first() else {
        return Err(HachiError::InvalidInput(
            "batched decompose_fold requires at least one witness".to_string(),
        ));
    };
    let z_len = first.z_pre.len();
    let coeff_len = first.centered_coeffs.len();
    let mut z_pre = first.z_pre.clone();
    let mut centered_coeffs = first.centered_coeffs.clone();

    for witness in rest {
        if witness.z_pre.len() != z_len || witness.centered_coeffs.len() != coeff_len {
            return Err(HachiError::InvalidInput(
                "batched decompose_fold witness length mismatch".to_string(),
            ));
        }
        for (dst, src) in z_pre.iter_mut().zip(witness.z_pre.iter()) {
            *dst += *src;
        }
        for (dst, src) in centered_coeffs
            .iter_mut()
            .zip(witness.centered_coeffs.iter())
        {
            for k in 0..D {
                dst[k] = dst[k].checked_add(src[k]).ok_or_else(|| {
                    HachiError::InvalidInput(
                        "batched decompose_fold centered coefficient overflow".to_string(),
                    )
                })?;
            }
        }
    }

    let centered_inf_norm = centered_coeffs
        .iter()
        .flat_map(|coeffs| coeffs.iter())
        .map(|coeff| coeff.unsigned_abs())
        .max()
        .unwrap_or(0);

    Ok(DecomposeFoldWitness {
        z_pre,
        centered_coeffs,
        centered_inf_norm,
    })
}

/// Stage-1 quadratic equation state for the Hachi protocol.
///
/// Encapsulates the relation $M(x) \cdot z = y(x) + (X^D + 1) \cdot r(x)$
/// along with intermediate prover witness data (`w_hat`, `z_pre`, `hint`).
///
/// M and z are never materialized on the hot path — split-eq factoring computes
/// their products on-the-fly via `compute_r_split_eq`, while debug/test code
/// can reconstruct reference `M_a` rows when needed.
pub struct QuadraticEquation<F: FieldCore, const D: usize, Cfg: CommitmentConfig> {
    /// Stage-1 proof vector `v = D · ŵ`.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// Stage-1 folding challenges (sparse representation).
    pub challenges: Vec<SparseChallenge>,
    /// Vector `y`.
    y: Vec<CyclotomicRing<F, D>>,
    /// Opening point (a, b) Lagrange weights.
    opening_point: RingOpeningPoint<F>,
    /// Pre-decomposition folded witness `z_pre = Σ c_i · s_i` (prover only).
    /// Replaces both `z_hat` and `z`: `z_hat = J^{-1}(z_pre)`.
    z_pre: Option<DecomposeFoldWitness<F, D>>,
    /// Decomposed `ŵ_i = G_1^{-1}(w_i)` as i8 digit planes (prover only).
    w_hat: Option<Vec<Vec<[i8; D]>>>,
    /// Flattened `w_hat` as i8 digit planes (prover only, computed once and reused).
    w_hat_flat: Option<Vec<[i8; D]>>,
    /// Pre-decomposition folded ring elements (prover only, avoids recompose roundtrip).
    w_folded: Option<Vec<CyclotomicRing<F, D>>>,
    /// Commitment hint (prover only).
    hint: Option<HachiCommitmentHint<F, D>>,
    /// Number of flattened public claims per commitment group.
    claim_group_sizes: Vec<usize>,

    _marker: PhantomData<Cfg>,
}

impl<F, const D: usize, Cfg> QuadraticEquation<F, D, Cfg>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    /// Prover constructor: runs §4.2 stage 1 and builds all equation components.
    ///
    /// `poly` provides the ring-level polynomial data for fold/decompose ops.
    /// `hint` carries `t_hat` from the commitment phase.
    ///
    /// # Errors
    ///
    /// Returns an error if the norm check, challenge sampling, or matrix
    /// generation fails.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "QuadraticEquation::new_prover")]
    #[inline(never)]
    pub fn new_prover<T: Transcript<F>, P: HachiPolyOps<F, D>>(
        ntt_d: &NttSlotCache<D>,
        ring_opening_point: RingOpeningPoint<F>,
        poly: &P,
        pre_folded: Vec<CyclotomicRing<F, D>>,
        level_params: HachiLevelParams,
        mut hint: HachiCommitmentHint<F, D>,
        transcript: &mut T,
        commitment: &RingCommitment<F, D>,
        y_ring: &CyclotomicRing<F, D>,
        layout: HachiCommitmentLayout,
    ) -> Result<Self, HachiError> {
        {
            let x: u8 = 0;
            tracing::trace!(
                stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
                "QuadraticEquation::new_prover"
            );
        }
        let (w_hat, w_hat_flat) = {
            let _span = tracing::info_span!("decompose_w_hat").entered();
            let depth_open = layout.num_digits_open;
            let log_basis = layout.log_basis;
            let w_hat: Vec<Vec<[i8; D]>> = cfg_iter!(pre_folded)
                .map(|w_i| w_i.balanced_decompose_pow2_i8(depth_open, log_basis))
                .collect();
            let w_hat_flat = flatten_i8_blocks(&w_hat);
            (w_hat, w_hat_flat)
        };
        hint.ensure_t_recomposed(layout.num_digits_open, layout.log_basis)?;

        let v = {
            let _span =
                tracing::info_span!("compute_v", w_hat_flat_len = w_hat_flat.len()).entered();
            mat_vec_mul_ntt_single_i8(ntt_d, level_params.n_d, &w_hat_flat)
        };

        transcript.append_serde(ABSORB_PROVER_V, &v);

        let challenges = sample_sparse_challenges::<F, T, D>(
            transcript,
            CHALLENGE_STAGE1_FOLD,
            layout.num_blocks,
            &level_params.stage1_config,
        )?;

        let z_pre = {
            let _span = tracing::info_span!("compute_z_pre").entered();
            let z = poly.decompose_fold(
                &challenges,
                layout.block_len,
                layout.num_digits_commit,
                layout.log_basis,
            );
            validate_decompose_fold(z, &level_params, layout)?
        };

        let y = generate_y::<F, D>(
            &v,
            &commitment.u,
            std::slice::from_ref(y_ring),
            level_params.n_d,
            level_params.n_b,
            level_params.n_a,
        )?;

        Ok(Self {
            v,
            challenges,
            y,
            opening_point: ring_opening_point,
            z_pre: Some(z_pre),
            w_hat: Some(w_hat),
            w_hat_flat: Some(w_hat_flat),
            w_folded: Some(pre_folded),
            hint: Some(hint),
            claim_group_sizes: vec![1],
            _marker: PhantomData,
        })
    }

    /// Batched prover constructor for multiple claims at one opening point.
    ///
    /// Flattens the per-claim witness blocks into the same D-erased root-witness
    /// format used by later ring-switch stages.
    ///
    /// # Errors
    ///
    /// Returns an error if the batched hints, folded witnesses, or decomposed
    /// aggregate witness are malformed.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "QuadraticEquation::new_batched_prover")]
    #[inline(never)]
    pub fn new_batched_prover<T: Transcript<F>, P: HachiPolyOps<F, D>>(
        ntt_d: &NttSlotCache<D>,
        ring_opening_point: RingOpeningPoint<F>,
        polys: &[&P],
        pre_folded_by_poly: Vec<Vec<CyclotomicRing<F, D>>>,
        claim_group_sizes: &[usize],
        level_params: HachiLevelParams,
        hints: Vec<HachiBatchedCommitmentHint<F, D>>,
        transcript: &mut T,
        commitments: &[RingCommitment<F, D>],
        y_rings: &[CyclotomicRing<F, D>],
        layout: HachiCommitmentLayout,
    ) -> Result<Self, HachiError> {
        if polys.is_empty() || claim_group_sizes.is_empty() {
            return Err(HachiError::InvalidInput(
                "batched prover requires at least one polynomial".to_string(),
            ));
        }
        if claim_group_sizes.contains(&0) {
            return Err(HachiError::InvalidInput(
                "batched prover requires nonempty commitment groups".to_string(),
            ));
        }
        let num_claims = claim_group_sizes
            .iter()
            .try_fold(0usize, |acc, &group_size| {
                acc.checked_add(group_size).ok_or_else(|| {
                    HachiError::InvalidInput("batched prover claim count overflow".to_string())
                })
            })?;
        if polys.len() != pre_folded_by_poly.len()
            || polys.len() != num_claims
            || y_rings.len() != num_claims
            || hints.len() != claim_group_sizes.len()
            || commitments.len() != claim_group_sizes.len()
        {
            return Err(HachiError::InvalidInput(
                "batched prover input lengths do not match".to_string(),
            ));
        }
        for commitment in commitments {
            if commitment.u.len() != level_params.n_b {
                return Err(HachiError::InvalidInput(
                    "batched prover received a commitment with the wrong length".to_string(),
                ));
            }
        }

        let (w_hat, w_hat_flat) = {
            let _span = tracing::info_span!("decompose_batched_w_hat").entered();
            let depth_open = layout.num_digits_open;
            let log_basis = layout.log_basis;
            let mut w_hat = Vec::new();
            for folded_rows in &pre_folded_by_poly {
                for w_i in folded_rows {
                    w_hat.push(w_i.balanced_decompose_pow2_i8(depth_open, log_basis));
                }
            }
            let w_hat_flat = flatten_i8_blocks(&w_hat);
            (w_hat, w_hat_flat)
        };
        let flattened_hint = {
            let mut inner_opening_digits = Vec::new();
            let mut t_rows = Vec::new();
            for (hint, &group_size) in hints.into_iter().zip(claim_group_sizes.iter()) {
                if hint.inner_opening_digits.len() != group_size {
                    return Err(HachiError::InvalidInput(
                        "batched prover hint group sizes do not match polynomial groups"
                            .to_string(),
                    ));
                }
                let mut hint = hint.into_flattened();
                hint.ensure_t_recomposed(layout.num_digits_open, layout.log_basis)?;
                let (digits, rows) = hint.into_parts();
                inner_opening_digits.extend(digits);
                let rows = rows.ok_or_else(|| {
                    HachiError::InvalidInput(
                        "missing recomposed t rows in batched prover hint".to_string(),
                    )
                })?;
                t_rows.extend(rows);
            }
            HachiCommitmentHint::with_t(inner_opening_digits, t_rows)
        };

        let v = {
            let _span = tracing::info_span!("compute_batched_v", w_hat_flat_len = w_hat_flat.len())
                .entered();
            mat_vec_mul_ntt_single_i8(ntt_d, level_params.n_d, &w_hat_flat)
        };

        transcript.append_serde(ABSORB_PROVER_V, &v);

        let total_blocks = layout.num_blocks.checked_mul(num_claims).ok_or_else(|| {
            HachiError::InvalidSetup("batched challenge count overflow".to_string())
        })?;
        let challenges = sample_sparse_challenges::<F, T, D>(
            transcript,
            CHALLENGE_STAGE1_FOLD,
            total_blocks,
            &level_params.stage1_config,
        )?;

        let z_pre = {
            let _span = tracing::info_span!("compute_batched_z_pre").entered();
            let z = if let Some(z) = P::decompose_fold_batched(
                polys,
                &challenges,
                layout.block_len,
                layout.num_digits_commit,
                layout.log_basis,
            ) {
                z
            } else {
                let witnesses: Vec<DecomposeFoldWitness<F, D>> = polys
                    .iter()
                    .zip(challenges.chunks(layout.num_blocks))
                    .map(|(poly, poly_challenges)| {
                        poly.decompose_fold(
                            poly_challenges,
                            layout.block_len,
                            layout.num_digits_commit,
                            layout.log_basis,
                        )
                    })
                    .collect();
                aggregate_decompose_fold_witnesses(witnesses)?
            };
            validate_decompose_fold_with_num_claims(z, &level_params, layout, num_claims)?
        };

        let commitment_rows: Vec<CyclotomicRing<F, D>> = commitments
            .iter()
            .flat_map(|commitment| commitment.u.iter().copied())
            .collect();
        let y = generate_y::<F, D>(
            &v,
            &commitment_rows,
            y_rings,
            level_params.n_d,
            level_params.n_b,
            level_params.n_a,
        )?;
        let w_folded = pre_folded_by_poly.into_iter().flatten().collect();

        Ok(Self {
            v,
            challenges,
            y,
            opening_point: ring_opening_point,
            z_pre: Some(z_pre),
            w_hat: Some(w_hat),
            w_hat_flat: Some(w_hat_flat),
            w_folded: Some(w_folded),
            hint: Some(flattened_hint),
            claim_group_sizes: claim_group_sizes.to_vec(),
            _marker: PhantomData,
        })
    }

    /// Recursive prover constructor: same as [`Self::new_prover`] but driven by
    /// the dedicated recursive witness view instead of the root polynomial trait.
    ///
    /// # Errors
    ///
    /// Returns an error if the norm check, challenge sampling, or matrix
    /// generation fails.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "QuadraticEquation::new_recursive_prover")]
    #[inline(never)]
    pub(crate) fn new_recursive_prover<T: Transcript<F>>(
        ntt_d: &NttSlotCache<D>,
        ring_opening_point: RingOpeningPoint<F>,
        witness: &RecursiveWitnessView<'_, F, D>,
        pre_folded: Vec<CyclotomicRing<F, D>>,
        level_params: HachiLevelParams,
        mut hint: HachiCommitmentHint<F, D>,
        transcript: &mut T,
        commitment: &[CyclotomicRing<F, D>],
        y_ring: &CyclotomicRing<F, D>,
        layout: HachiCommitmentLayout,
    ) -> Result<Self, HachiError> {
        {
            let x: u8 = 0;
            tracing::trace!(
                stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
                "QuadraticEquation::new_recursive_prover"
            );
        }
        let (w_hat, w_hat_flat) = {
            let _span = tracing::info_span!("decompose_w_hat").entered();
            let depth_open = layout.num_digits_open;
            let log_basis = layout.log_basis;
            let w_hat: Vec<Vec<[i8; D]>> = cfg_iter!(pre_folded)
                .map(|w_i| w_i.balanced_decompose_pow2_i8(depth_open, log_basis))
                .collect();
            let w_hat_flat = flatten_i8_blocks(&w_hat);
            (w_hat, w_hat_flat)
        };
        hint.ensure_t_recomposed(layout.num_digits_open, layout.log_basis)?;

        let v = {
            let _span =
                tracing::info_span!("compute_v", w_hat_flat_len = w_hat_flat.len()).entered();
            mat_vec_mul_ntt_single_i8(ntt_d, level_params.n_d, &w_hat_flat)
        };

        transcript.append_serde(ABSORB_PROVER_V, &v);

        let challenges = sample_sparse_challenges::<F, T, D>(
            transcript,
            CHALLENGE_STAGE1_FOLD,
            layout.num_blocks,
            &level_params.stage1_config,
        )?;

        let z_pre = {
            let _span = tracing::info_span!("compute_z_pre").entered();
            let z = witness.decompose_fold(
                &challenges,
                layout.block_len,
                layout.num_digits_commit,
                layout.log_basis,
            );
            validate_decompose_fold(z, &level_params, layout)?
        };

        let y = generate_y::<F, D>(
            &v,
            commitment,
            std::slice::from_ref(y_ring),
            level_params.n_d,
            level_params.n_b,
            level_params.n_a,
        )?;

        Ok(Self {
            v,
            challenges,
            y,
            opening_point: ring_opening_point,
            z_pre: Some(z_pre),
            w_hat: Some(w_hat),
            w_hat_flat: Some(w_hat_flat),
            w_folded: Some(pre_folded),
            hint: Some(hint),
            claim_group_sizes: vec![1],
            _marker: PhantomData,
        })
    }

    /// Get the vector y.
    pub fn y(&self) -> &[CyclotomicRing<F, D>] {
        &self.y
    }

    /// Get the vector v.
    pub fn v(&self) -> &[CyclotomicRing<F, D>] {
        &self.v
    }

    /// Get the opening point (a, b) Lagrange weights.
    pub fn opening_point(&self) -> &RingOpeningPoint<F> {
        &self.opening_point
    }

    /// Number of flattened public claims carried by each commitment group.
    pub fn claim_group_sizes(&self) -> &[usize] {
        &self.claim_group_sizes
    }

    /// Get the pre-decomposition folded witness `z_pre` (prover only).
    pub fn z_pre(&self) -> Option<&[CyclotomicRing<F, D>]> {
        self.z_pre.as_ref().map(|witness| witness.z_pre.as_slice())
    }

    /// Get centered coefficients for each `z_pre` row (prover only).
    pub fn z_pre_centered(&self) -> Option<&[[i32; D]]> {
        self.z_pre
            .as_ref()
            .map(|witness| witness.centered_coeffs.as_slice())
    }

    /// Get `||z_pre||_inf` from the centered witness representation.
    pub fn z_pre_centered_inf_norm(&self) -> Option<u32> {
        self.z_pre.as_ref().map(|witness| witness.centered_inf_norm)
    }

    /// Take ownership of the `z_pre` witness, leaving `None` in its place.
    pub fn take_z_pre(&mut self) -> Option<DecomposeFoldWitness<F, D>> {
        self.z_pre.take()
    }

    /// Get the decomposed witness `ŵ` as i8 digit planes (prover only).
    pub fn w_hat(&self) -> Option<&[Vec<[i8; D]>]> {
        self.w_hat.as_deref()
    }

    /// Get the pre-flattened `w_hat` as i8 digit planes (prover only).
    pub fn w_hat_flat(&self) -> Option<&[[i8; D]]> {
        self.w_hat_flat.as_deref()
    }

    /// Take ownership of `w_hat`, leaving `None` in its place.
    pub fn take_w_hat(&mut self) -> Option<Vec<Vec<[i8; D]>>> {
        self.w_hat.take()
    }

    /// Take ownership of the flattened witness digits, leaving `None` in its place.
    pub fn take_w_hat_flat(&mut self) -> Option<Vec<[i8; D]>> {
        self.w_hat_flat.take()
    }

    /// Get the pre-decomposition folded ring elements (prover only).
    pub fn w_folded(&self) -> Option<&[CyclotomicRing<F, D>]> {
        self.w_folded.as_deref()
    }

    /// Take ownership of the pre-decomposition folded witness, leaving `None`
    /// in its place.
    pub fn take_w_folded(&mut self) -> Option<Vec<CyclotomicRing<F, D>>> {
        self.w_folded.take()
    }

    /// Get the commitment hint (prover only).
    pub fn hint(&self) -> Option<&HachiCommitmentHint<F, D>> {
        self.hint.as_ref()
    }

    /// Take ownership of the hint, leaving `None` in its place.
    pub fn take_hint(&mut self) -> Option<HachiCommitmentHint<F, D>> {
        self.hint.take()
    }
}

pub(crate) fn derive_stage1_challenges<F, T, const D: usize>(
    transcript: &mut T,
    v: &[CyclotomicRing<F, D>],
    num_blocks: usize,
    level_params: &HachiLevelParams,
) -> Result<Vec<SparseChallenge>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_serde(ABSORB_PROVER_V, &RingSliceSerializer(v));
    sample_sparse_challenges::<F, T, D>(
        transcript,
        CHALLENGE_STAGE1_FOLD,
        num_blocks,
        &level_params.stage1_config,
    )
}

/// Add only the high-half quotient contribution of `challenge * ring`.
///
/// Skips the first `D - pos` coefficients per challenge term that cannot
/// contribute (degree < D), cutting iteration count roughly in half.
#[inline(always)]
fn add_sparse_ring_product_high_half<F: FieldCore + CanonicalField, const D: usize>(
    quotient: &mut [F],
    challenge: &SparseChallenge,
    ring: &CyclotomicRing<F, D>,
) {
    let rc = ring.coefficients();
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let c = F::from_i64(coeff as i64);
        let p = pos as usize;
        for s in (D - p)..D {
            quotient[p + s - D] += c * rc[s];
        }
    }
}

/// Parallel high-half accumulation over blocks.
///
/// Replaces `for (i, ring) in rings { add_sparse_ring_product_high_half(...) }`
/// with a fold-reduce giving block-level parallelism.
fn parallel_high_half_accumulate<F: FieldCore + CanonicalField, const D: usize>(
    challenges: &[SparseChallenge],
    rings: &[CyclotomicRing<F, D>],
) -> Vec<F> {
    cfg_fold_reduce!(
        0..rings.len(),
        || vec![F::zero(); D],
        |mut acc: Vec<F>, i: usize| {
            add_sparse_ring_product_high_half::<F, D>(&mut acc, &challenges[i], &rings[i]);
            acc
        },
        |mut a: Vec<F>, b: Vec<F>| {
            for (ai, bi) in a.iter_mut().zip(b.iter()) {
                *ai += *bi;
            }
            a
        }
    )
}

fn quotient_from_cyclic_and_reduced<F: FieldCore, const D: usize>(
    cyclic: &CyclotomicRing<F, D>,
    reduced: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let cyc_c = cyclic.coefficients();
    let red_c = reduced.coefficients();
    let quotient = std::array::from_fn(|k| (cyc_c[k] - red_c[k]) * F::TWO_INV);
    CyclotomicRing::from_coefficients(quotient)
}

fn repeated_b_commitment_rows<F: FieldCore + CanonicalField, const D: usize>(
    ntt_shared: &NttSlotCache<D>,
    n_b: usize,
    t_hat: &[Vec<[i8; D]>],
    claim_group_sizes: &[usize],
    blocks_per_claim: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    if claim_group_sizes.is_empty() || blocks_per_claim == 0 {
        return Err(HachiError::InvalidProof);
    }
    let num_claims = claim_group_sizes
        .iter()
        .try_fold(0usize, |acc, &group_size| {
            if group_size == 0 {
                return Err(HachiError::InvalidProof);
            }
            acc.checked_add(group_size).ok_or(HachiError::InvalidProof)
        })?;
    if t_hat.len() != num_claims * blocks_per_claim {
        return Err(HachiError::InvalidProof);
    }
    let mut rows = Vec::with_capacity(claim_group_sizes.len() * n_b);
    let mut offset = 0usize;
    for &group_size in claim_group_sizes {
        let group_block_count = group_size
            .checked_mul(blocks_per_claim)
            .ok_or(HachiError::InvalidProof)?;
        let next_offset = offset
            .checked_add(group_block_count)
            .ok_or(HachiError::InvalidProof)?;
        let group = t_hat
            .get(offset..next_offset)
            .ok_or(HachiError::InvalidProof)?;
        let group_digits = flatten_i8_blocks(group);
        rows.extend(mat_vec_mul_ntt_single_i8_cyclic(
            ntt_shared,
            n_b,
            &group_digits,
        ));
        offset = next_offset;
    }
    if offset != t_hat.len() {
        return Err(HachiError::InvalidProof);
    }
    Ok(rows)
}

/// Split-eq replacement for `generate_m` + `compute_r_via_poly_division`.
///
/// Computes `r` such that `M·z = y + (X^D+1)·r` without materializing M or z.
/// Uses split-eq factoring: `kron(left, gadget) · decomposed = left · pre_decomp`.
#[allow(clippy::too_many_arguments, clippy::needless_borrow)]
#[tracing::instrument(skip_all, name = "compute_r_split_eq")]
pub(crate) fn compute_r_split_eq<F, const D: usize>(
    level_params: &HachiLevelParams,
    _setup: &HachiExpandedSetup<F>,
    challenges: &[SparseChallenge],
    w_hat_flat: &[[i8; D]],
    t_hat: &[Vec<[i8; D]>],
    t: &[Vec<CyclotomicRing<F, D>>],
    w_folded: &[CyclotomicRing<F, D>],
    z_pre_centered: &[[i32; D]],
    z_pre_centered_inf_norm: u32,
    y: &[CyclotomicRing<F, D>],
    claim_group_sizes: &[usize],
    blocks_per_claim: usize,
    ntt_shared: &NttSlotCache<D>,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField,
{
    if claim_group_sizes.is_empty() || claim_group_sizes.contains(&0) {
        return Err(HachiError::InvalidProof);
    }
    let num_public_outputs = claim_group_sizes
        .iter()
        .try_fold(0usize, |acc, &group_size| {
            acc.checked_add(group_size).ok_or(HachiError::InvalidProof)
        })?;
    if num_public_outputs == 0 {
        return Err(HachiError::InvalidProof);
    }
    let num_commitment_groups = claim_group_sizes.len();
    let commitment_row_count = level_params
        .n_b
        .checked_mul(num_commitment_groups)
        .ok_or(HachiError::InvalidProof)?;
    let num_rows = level_params
        .m_row_count_with_commitments_and_public_outputs(num_commitment_groups, num_public_outputs);
    if y.len() != num_rows {
        return Err(HachiError::InvalidProof);
    }
    let row3_start = level_params.n_d + commitment_row_count;
    let row4_idx = row3_start + num_public_outputs;
    let a_row_start = row4_idx + 1;

    let t_hat_flat = flatten_i8_blocks(t_hat);

    let (d_cyclic, b_cyclic, a_quotients) = fused_split_eq_quotients::<F, D>(
        ntt_shared,
        level_params.n_d,
        level_params.n_b,
        level_params.n_a,
        w_hat_flat,
        &t_hat_flat,
        z_pre_centered,
        z_pre_centered_inf_norm,
    );
    let commitment_cyclic_rows =
        if commitment_row_count == level_params.n_b && num_commitment_groups == 1 {
            b_cyclic
        } else {
            repeated_b_commitment_rows(
                ntt_shared,
                level_params.n_b,
                t_hat,
                claim_group_sizes,
                blocks_per_claim,
            )?
        };
    if commitment_cyclic_rows.len() != commitment_row_count {
        return Err(HachiError::InvalidProof);
    }

    let mut result = Vec::with_capacity(num_rows);
    let mut other_time = 0.0f64;

    for row_idx in 0..num_rows {
        if row_idx < level_params.n_d {
            result.push(quotient_from_cyclic_and_reduced(
                &d_cyclic[row_idx],
                &y[row_idx],
            ));
        } else if row_idx < level_params.n_d + commitment_row_count {
            result.push(quotient_from_cyclic_and_reduced(
                &commitment_cyclic_rows[row_idx - level_params.n_d],
                &y[row_idx],
            ));
        } else if row_idx >= a_row_start {
            let t_row = Instant::now();
            let _span = tracing::info_span!("A_row").entered();
            let a_idx = row_idx - a_row_start;

            let mut quotient = cfg_fold_reduce!(
                0..t.len(),
                || vec![F::zero(); D],
                |mut acc: Vec<F>, i: usize| {
                    if let Some(t_row_i) = t[i].get(a_idx) {
                        add_sparse_ring_product_high_half::<F, D>(
                            &mut acc,
                            &challenges[i],
                            t_row_i,
                        );
                    }
                    acc
                },
                |mut a: Vec<F>, b: Vec<F>| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += *bi;
                    }
                    a
                }
            );

            let a_q = a_quotients[a_idx].coefficients();
            for k in 0..D {
                quotient[k] -= a_q[k];
            }
            result.push(CyclotomicRing::from_slice(&quotient));
            other_time += t_row.elapsed().as_secs_f64();
        } else {
            let t_row = Instant::now();

            if row_idx < row4_idx {
                let _span = tracing::info_span!("bTw_row").entered();
                // Each public `y` row is degree < D, so its quotient is zero.
                result.push(CyclotomicRing::<F, D>::zero());
            } else {
                let _span = tracing::info_span!("challenge_fold_row").entered();
                let quotient = parallel_high_half_accumulate::<F, D>(challenges, w_folded);
                result.push(CyclotomicRing::from_slice(&quotient));
            }
            other_time += t_row.elapsed().as_secs_f64();
        }
    }

    tracing::debug!(other_s = other_time, "compute_r breakdown");

    Ok(result)
}

pub(crate) fn generate_y<F, const D: usize>(
    v: &[CyclotomicRing<F, D>],
    commitment_rows: &[CyclotomicRing<F, D>],
    public_outputs: &[CyclotomicRing<F, D>],
    n_d: usize,
    n_b: usize,
    n_a: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore,
{
    if v.len() != n_d {
        return Err(HachiError::InvalidSize {
            expected: n_d,
            actual: v.len(),
        });
    }
    if commitment_rows.is_empty() || commitment_rows.len() % n_b != 0 {
        return Err(HachiError::InvalidSize {
            expected: n_b,
            actual: commitment_rows.len(),
        });
    }
    if public_outputs.is_empty() {
        return Err(HachiError::InvalidInput(
            "generate_y requires at least one public output".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(n_d + commitment_rows.len() + public_outputs.len() + 1 + n_a);
    out.extend_from_slice(v);
    out.extend_from_slice(commitment_rows);
    out.extend_from_slice(public_outputs);
    out.push(CyclotomicRing::<F, D>::zero());
    out.extend(repeat_n(CyclotomicRing::<F, D>::zero(), n_a));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::array::from_fn;

    use crate::algebra::CyclotomicRing;
    use crate::protocol::challenges::sparse::sample_sparse_challenges;
    use crate::protocol::commitment::HachiProverSetup;
    use crate::protocol::commitment::{
        HachiCommitmentCore, HachiScheduleInputs, RingCommitmentScheme,
    };
    use crate::protocol::hachi_poly_ops::DensePoly;
    use crate::protocol::proof::HachiCommitmentHint;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::test_utils::*;
    use crate::FromSmallInt;
    use crate::Transcript;

    const TRANSCRIPT_SEED: &[u8] = b"test/prover-relation";

    fn replay_challenges(v: &Vec<CyclotomicRing<F, D>>) -> Vec<CyclotomicRing<F, D>> {
        let mut transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_SEED);
        transcript.append_serde(ABSORB_PROVER_V, v);

        let challenge_cfg = TinyConfig::stage1_challenge_config(D);
        let sparse = sample_sparse_challenges::<F, Blake2bTranscript<F>, D>(
            &mut transcript,
            CHALLENGE_STAGE1_FOLD,
            NUM_BLOCKS,
            &challenge_cfg,
        )
        .unwrap();
        sparse
            .iter()
            .map(|c| c.to_dense::<F, D>().unwrap())
            .collect()
    }

    struct Fixture {
        setup: HachiProverSetup<F, D>,
        commitment_u: Vec<CyclotomicRing<F, D>>,
        point: RingOpeningPoint<F>,
        blocks: Vec<Vec<CyclotomicRing<F, D>>>,
        quad_eq: QuadraticEquation<F, D, TinyConfig>,
        /// Challenges re-derived via transcript replay (cross-check).
        challenges: Vec<CyclotomicRing<F, D>>,
    }

    fn build_fixture() -> Fixture {
        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::setup(16, 1).unwrap();

        let blocks = sample_blocks();
        let w =
            <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::commit_ring_blocks(
                &blocks, &setup,
            )
            .unwrap();

        let point = RingOpeningPoint {
            a: sample_a(),
            b: sample_b(),
        };

        let ring_coeffs: Vec<CyclotomicRing<F, D>> =
            blocks.iter().flat_map(|b| b.iter().copied()).collect();
        let poly = DensePoly::from_ring_coeffs(ring_coeffs);
        let hint = HachiCommitmentHint::new(w.t_hat);
        let mut transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_SEED);
        let y_ring = CyclotomicRing::<F, D>::zero();
        let layout = setup.layout();
        let w_folded = poly.fold_blocks(&point.a, layout.block_len);
        let level_params = TinyConfig::level_params(HachiScheduleInputs {
            max_num_vars: setup.expanded.seed.max_num_vars,
            level: 0,
            current_w_len: layout.num_blocks * layout.block_len * D,
        });
        let quad_eq = QuadraticEquation::<F, D, TinyConfig>::new_prover(
            &setup.ntt_shared,
            point.clone(),
            &poly,
            w_folded,
            level_params,
            hint,
            &mut transcript,
            &w.commitment,
            &y_ring,
            layout,
        )
        .unwrap();

        let challenges = replay_challenges(&quad_eq.v);

        Fixture {
            setup,
            commitment_u: w.commitment.u.clone(),
            point,
            blocks,
            quad_eq,
            challenges,
        }
    }

    fn i8_to_ring(digits: &[[i8; D]]) -> Vec<CyclotomicRing<F, D>> {
        digits
            .iter()
            .map(|d| {
                let coeffs: [F; D] = from_fn(|i| F::from_i64(d[i] as i64));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect()
    }

    /// Row 1: D · ŵ = v
    #[test]
    fn row1_d_times_w_hat_equals_v() {
        let f = build_fixture();

        let w_hat = f.quad_eq.w_hat().unwrap();
        let w_hat_flat: Vec<CyclotomicRing<F, D>> = i8_to_ring(
            &w_hat
                .iter()
                .flat_map(|v| v.iter().copied())
                .collect::<Vec<_>>(),
        );
        let lhs = mat_vec_mul(&f.setup.expanded.shared_matrix, &w_hat_flat);

        assert_eq!(lhs, f.quad_eq.v(), "Row 1 failed: D · ŵ ≠ v");
    }

    /// Row 2: B · inner opening digits = u (commitment vector)
    #[test]
    fn row2_b_times_inner_opening_digits_equals_u_commitment() {
        let f = build_fixture();

        let hint = f.quad_eq.hint().unwrap();
        let inner_opening_digits_flat_ring: Vec<CyclotomicRing<F, D>> = hint
            .inner_opening_digits
            .iter()
            .flat_map(|v| v.iter())
            .map(|plane| {
                let coeffs: [F; D] = from_fn(|k| F::from_i64(plane[k] as i64));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();
        let lhs = mat_vec_mul(
            &f.setup.expanded.shared_matrix,
            &inner_opening_digits_flat_ring,
        );

        assert_eq!(
            lhs, f.commitment_u,
            "Row 2 failed: B · inner opening digits ≠ u"
        );
    }

    /// Row 3: b^T · G_{2^r} · ŵ = u_eval
    #[test]
    fn row3_bt_gadget_w_hat_equals_u_eval() {
        let f = build_fixture();

        let w_hat = f.quad_eq.w_hat().unwrap();
        let w_recomposed: Vec<CyclotomicRing<F, D>> = w_hat
            .iter()
            .map(|w_hat_i| CyclotomicRing::gadget_recompose_pow2_i8(w_hat_i, log_basis()))
            .collect();

        let u_eval = w_recomposed
            .iter()
            .zip(f.point.b.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (w_i, b_i)| {
                acc + w_i.scale(b_i)
            });

        let u_eval_direct = f.blocks.iter().zip(f.point.b.iter()).fold(
            CyclotomicRing::<F, D>::zero(),
            |acc, (block_i, b_i)| {
                let inner: CyclotomicRing<F, D> = block_i
                    .iter()
                    .zip(f.point.a.iter())
                    .fold(CyclotomicRing::<F, D>::zero(), |acc2, (f_ij, a_j)| {
                        acc2 + f_ij.scale(a_j)
                    });
                acc + inner.scale(b_i)
            },
        );

        assert_eq!(
            u_eval, u_eval_direct,
            "Row 3 failed: b^T G ŵ ≠ Σ b_i (a^T f_i)"
        );
    }

    /// Derive z_hat from z_pre for test assertions.
    fn derive_z_hat(z_pre: &[CyclotomicRing<F, D>]) -> Vec<CyclotomicRing<F, D>> {
        z_pre
            .iter()
            .flat_map(|z_j| z_j.balanced_decompose_pow2(num_digits_fold(), log_basis()))
            .collect()
    }

    /// Row 4: (c^T ⊗ G_1) · ŵ = a^T · G_{2^m} · J · ẑ
    #[test]
    fn row4_challenge_fold_w_equals_a_gadget_j_z_hat() {
        let f = build_fixture();

        let w_hat = f.quad_eq.w_hat().unwrap();
        let w: Vec<CyclotomicRing<F, D>> = w_hat
            .iter()
            .map(|w_hat_i| CyclotomicRing::gadget_recompose_pow2_i8(w_hat_i, log_basis()))
            .collect();

        let lhs = f
            .challenges
            .iter()
            .zip(w.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (c_i, w_i)| {
                acc + (*c_i * *w_i)
            });

        let z_hat = derive_z_hat(f.quad_eq.z_pre().unwrap());
        let z_recovered = recompose_z_hat(&z_hat);
        let rhs = a_transpose_gadget_times_vec(&f.point.a, &z_recovered);

        assert_eq!(lhs, rhs, "Row 4 failed: (c^T ⊗ G_1)ŵ ≠ a^T G J ẑ");
    }

    /// Row 5: (c^T ⊗ G_{n_A}) · inner opening digits = A · J · ẑ
    #[test]
    fn row5_challenge_fold_inner_opening_digits_equals_a_j_z_hat() {
        let f = build_fixture();

        let hint = f.quad_eq.hint().unwrap();
        let mut lhs = vec![CyclotomicRing::<F, D>::zero(); N_A];
        for (c_i, inner_opening_digits_i) in
            f.challenges.iter().zip(hint.inner_opening_digits.iter())
        {
            let t_i = gadget_recompose_vec_i8(inner_opening_digits_i);
            assert_eq!(t_i.len(), N_A);
            for (lhs_j, t_ij) in lhs.iter_mut().zip(t_i.iter()) {
                *lhs_j += *c_i * *t_ij;
            }
        }

        let z_hat = derive_z_hat(f.quad_eq.z_pre().unwrap());
        let z_recovered = recompose_z_hat(&z_hat);
        let rhs = mat_vec_mul(&f.setup.expanded.shared_matrix, &z_recovered);

        assert_eq!(
            lhs, rhs,
            "Row 5 failed: (c^T ⊗ G_nA)inner opening digits ≠ A · J · ẑ"
        );
    }

    #[test]
    fn prove_output_shapes_are_correct() {
        let f = build_fixture();

        assert_eq!(f.quad_eq.v().len(), TinyConfig::envelope(0).max_n_d);

        let w_hat = f.quad_eq.w_hat().unwrap();
        assert_eq!(w_hat.len(), NUM_BLOCKS);
        assert!(w_hat.iter().all(|v| v.len() == num_digits_open()));

        let hint = f.quad_eq.hint().unwrap();
        assert_eq!(hint.inner_opening_digits.len(), NUM_BLOCKS);
        assert!(hint
            .inner_opening_digits
            .iter()
            .all(|v| v.len() == N_A * num_digits_open()));

        assert_eq!(
            f.quad_eq.z_pre().unwrap().len(),
            BLOCK_LEN * num_digits_commit()
        );
    }
}
