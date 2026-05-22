//! Tensor-shaped sparse-challenge sampling.
//!
//! For protocols that sample a length-`num_blocks` sparse-challenge vector per
//! claim, the tensor variant samples two factor vectors of length
//! `√num_blocks` and presents the logical fold challenge at block `(p, q)` as
//! the negacyclic tensor product `left[p] · right[q]`. This shrinks transcript
//! challenge sampling from `O(num_blocks)` to `O(√num_blocks)` per claim
//! while leaving the downstream fold semantics unchanged: callers see a
//! uniform flat view through [`TensorChallenges::expand_integer`] /
//! [`TensorChallenges::evals_at_pows`].
//!
//! Sampling labels are taken as a [`TensorChallengeLabels`] parameter so this
//! module is not coupled to any specific protocol stage.

use crate::{sample_sparse_challenges, IntegerChallenge, SparseChallenge, SparseChallengeConfig};
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase};
use akita_transcript::Transcript;
use sha3::{Digest, Sha3_256};
use std::ops::Range;

const TENSOR_LEFT_DIGEST_DOMAIN: &[u8] = b"akita/tensor-left-digest/v1";

/// Shape of a tensor-vs-flat sparse-challenge round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TensorChallengeShape {
    /// Sample one independent challenge for every logical block.
    #[default]
    Flat,
    /// Split each logical block index into two balanced dimensions and sample
    /// independent left/right challenge vectors.
    Tensor,
}

impl TensorChallengeShape {
    /// Effective per-logical-block integer L1 mass for this shape.
    ///
    /// Flat folds inherit the configured per-challenge L1 norm directly;
    /// tensor folds materialise `α_p · β_q` whose L1 envelope is bounded by
    /// the product of the two factors' L1 norms.
    #[inline]
    #[must_use]
    pub fn effective_l1_mass(&self, cfg: &SparseChallengeConfig) -> usize {
        match self {
            Self::Flat => cfg.l1_norm(),
            Self::Tensor => cfg.l1_norm().saturating_mul(cfg.l1_norm()),
        }
    }
}

/// Tensor-structured sparse challenges for one folding round.
///
/// `left` and `right` are laid out per claim: claim `c`'s left factor occupies
/// `left[c * left_len .. (c + 1) * left_len]` and similarly for `right`. The
/// logical challenge at block `(p, q)` of claim `c` is the tensor product
/// `left[c, p] · right[c, q]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorChallengeSet {
    /// Left vector entries, grouped by claim.
    left: Vec<SparseChallenge>,
    /// Right vector entries, grouped by claim.
    right: Vec<SparseChallenge>,
    /// Number of left entries per claim.
    left_len: usize,
    /// Number of right entries per claim.
    right_len: usize,
    /// Number of claims represented by this tensor challenge set.
    num_claims: usize,
}

/// Canonical dimensions for a [`TensorChallengeSet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorChallengeDims {
    /// Number of left entries per claim.
    pub left_len: usize,
    /// Number of right entries per claim.
    pub right_len: usize,
    /// Number of claims represented by the set.
    pub num_claims: usize,
}

impl TensorChallengeDims {
    /// Number of logical blocks per claim.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension product overflows.
    #[inline]
    pub fn blocks_per_claim(&self) -> Result<usize, AkitaError> {
        self.left_len
            .checked_mul(self.right_len)
            .ok_or_else(|| AkitaError::InvalidSetup("tensor challenge count overflow".to_string()))
    }

    /// Total number of logical blocks across every claim.
    ///
    /// # Errors
    ///
    /// Returns an error if the block count overflows.
    #[inline]
    pub fn total_blocks(&self) -> Result<usize, AkitaError> {
        self.num_claims
            .checked_mul(self.blocks_per_claim()?)
            .ok_or_else(|| AkitaError::InvalidSetup("tensor challenge count overflow".to_string()))
    }

    /// Split a claim-local block index into `(left_idx, right_idx)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `local_block_idx` is outside the logical block range.
    #[inline]
    pub fn factor_indices(&self, local_block_idx: usize) -> Result<(usize, usize), AkitaError> {
        let blocks_per_claim = self.blocks_per_claim()?;
        if local_block_idx >= blocks_per_claim {
            return Err(AkitaError::InvalidInput(format!(
                "tensor local block index {local_block_idx} out of range for {blocks_per_claim} blocks"
            )));
        }
        Ok((
            local_block_idx / self.right_len,
            local_block_idx % self.right_len,
        ))
    }
}

/// Folding challenges, either flat or tensor-structured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorChallenges {
    /// Flat challenge vector indexed as `claim * num_blocks + block`.
    Flat(Vec<SparseChallenge>),
    /// Tensor-structured vectors indexed as `(claim, p, q)`.
    Tensor(TensorChallengeSet),
}

/// Transcript labels consumed by [`sample_tensor_challenges`].
///
/// Bundling them in a struct keeps the call sites self-describing and prevents
/// accidental left/right swaps. Callers pick label byte strings appropriate
/// for their protocol stage.
#[derive(Debug, Clone, Copy)]
pub struct TensorChallengeLabels<'a> {
    /// Label used for the flat shape's single sampling step.
    pub flat: &'a [u8],
    /// Label used for sampling the tensor shape's left factor.
    pub tensor_left: &'a [u8],
    /// Label under which the canonical digest of the sampled left factor is
    /// absorbed back into the transcript before sampling the right factor.
    pub tensor_left_digest: &'a [u8],
    /// Label used for sampling the tensor shape's right factor.
    pub tensor_right: &'a [u8],
}

impl TensorChallenges {
    /// Number of logical flat challenges represented by this value.
    #[inline]
    pub fn logical_len(&self) -> Result<usize, AkitaError> {
        match self {
            Self::Flat(challenges) => Ok(challenges.len()),
            Self::Tensor(tensor) => tensor.total_blocks(),
        }
    }

    /// Expand to integer ring challenges for prover-side fold kernels.
    ///
    /// Flat challenges widen coefficients without changing the distribution;
    /// tensor challenges materialise `left[p] · right[q]` per logical block.
    ///
    /// # Errors
    ///
    /// Returns an error if a tensor product has malformed inputs or overflows
    /// its integer coefficient representation.
    pub fn expand_integer<const D: usize>(&self) -> Result<Vec<IntegerChallenge>, AkitaError> {
        match self {
            Self::Flat(challenges) => Ok(challenges
                .iter()
                .map(IntegerChallenge::from_sparse)
                .collect()),
            Self::Tensor(tensor) => tensor.expand_integer::<D>(),
        }
    }

    /// Evaluate all logical challenges at a ring-switch point, in flat order.
    ///
    /// # Errors
    ///
    /// Returns an error if challenge expansion or evaluation fails.
    pub fn evals_at_pows<F, E, const D: usize>(
        &self,
        alpha_pows: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: FieldCore + MulBase<F>,
    {
        match self {
            Self::Flat(challenges) => challenges
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E, D>(alpha_pows))
                .collect(),
            Self::Tensor(tensor) => tensor.evals_at_pows::<F, E, D>(alpha_pows),
        }
    }
}

impl TensorChallengeSet {
    /// Build a tensor challenge set and validate its factor counts.
    ///
    /// # Errors
    ///
    /// Returns an error if `left`/`right` lengths do not match the declared
    /// dimensions or if count arithmetic overflows.
    pub fn new(
        left: Vec<SparseChallenge>,
        right: Vec<SparseChallenge>,
        left_len: usize,
        right_len: usize,
        num_claims: usize,
    ) -> Result<Self, AkitaError> {
        let out = Self {
            left,
            right,
            left_len,
            right_len,
            num_claims,
        };
        out.validate_lengths()?;
        out.total_blocks()?;
        Ok(out)
    }

    /// Left vector entries, grouped by claim.
    #[inline]
    #[must_use]
    pub fn left(&self) -> &[SparseChallenge] {
        &self.left
    }

    /// Right vector entries, grouped by claim.
    #[inline]
    #[must_use]
    pub fn right(&self) -> &[SparseChallenge] {
        &self.right
    }

    /// Number of left entries per claim.
    #[inline]
    #[must_use]
    pub fn left_len(&self) -> usize {
        self.left_len
    }

    /// Number of right entries per claim.
    #[inline]
    #[must_use]
    pub fn right_len(&self) -> usize {
        self.right_len
    }

    /// Number of claims represented by this set.
    #[inline]
    #[must_use]
    pub fn num_claims(&self) -> usize {
        self.num_claims
    }

    /// Return the canonical dimensions for this tensor set.
    #[inline]
    #[must_use]
    pub fn dims(&self) -> TensorChallengeDims {
        TensorChallengeDims {
            left_len: self.left_len,
            right_len: self.right_len,
            num_claims: self.num_claims,
        }
    }

    /// Number of logical blocks represented per claim.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension product overflows.
    #[inline]
    pub fn blocks_per_claim(&self) -> Result<usize, AkitaError> {
        self.dims().blocks_per_claim()
    }

    /// Total number of logical blocks represented by this set.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension product overflows.
    #[inline]
    pub fn total_blocks(&self) -> Result<usize, AkitaError> {
        self.dims().total_blocks()
    }

    /// Validate factor counts and all sparse factor structural invariants.
    ///
    /// # Errors
    ///
    /// Returns an error if any tensor factor is malformed for ring dimension `D`.
    pub fn validate<const D: usize>(&self) -> Result<(), AkitaError> {
        self.validate_lengths()?;
        for challenge in self.left.iter().chain(self.right.iter()) {
            validate_sparse_factor::<D>(challenge)?;
        }
        Ok(())
    }

    /// Return the flat expanded range owned by `claim_idx`.
    ///
    /// # Errors
    ///
    /// Returns an error if `claim_idx` is out of range or arithmetic overflows.
    pub fn claim_logical_range(&self, claim_idx: usize) -> Result<Range<usize>, AkitaError> {
        let blocks_per_claim = self.blocks_per_claim()?;
        if claim_idx >= self.num_claims {
            return Err(AkitaError::InvalidInput(format!(
                "tensor claim index {claim_idx} out of range for {} claims",
                self.num_claims
            )));
        }
        let start = claim_idx.checked_mul(blocks_per_claim).ok_or_else(|| {
            AkitaError::InvalidSetup("tensor logical claim offset overflow".to_string())
        })?;
        let end = start.checked_add(blocks_per_claim).ok_or_else(|| {
            AkitaError::InvalidSetup("tensor logical claim end overflow".to_string())
        })?;
        Ok(start..end)
    }

    /// Return the left/right factor slices for one claim.
    ///
    /// # Errors
    ///
    /// Returns an error if the claim is out of range, lengths are inconsistent,
    /// or slice arithmetic overflows.
    pub fn factor_slices_for_claim(
        &self,
        claim_idx: usize,
    ) -> Result<(&[SparseChallenge], &[SparseChallenge]), AkitaError> {
        let left_range = self.factor_range(claim_idx, self.left_len, "tensor-left")?;
        let right_range = self.factor_range(claim_idx, self.right_len, "tensor-right")?;
        Ok((
            self.left
                .get(left_range.clone())
                .ok_or(AkitaError::InvalidSize {
                    expected: left_range.end,
                    actual: self.left.len(),
                })?,
            self.right
                .get(right_range.clone())
                .ok_or(AkitaError::InvalidSize {
                    expected: right_range.end,
                    actual: self.right.len(),
                })?,
        ))
    }

    /// Return tensor factors for a global logical block index.
    ///
    /// The returned indices are `(claim_idx, local_block_idx, left_factor, right_factor)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `block_idx` is outside the represented logical range
    /// or if factor lengths are inconsistent.
    pub fn factors_for_logical_block(
        &self,
        block_idx: usize,
    ) -> Result<(usize, usize, &SparseChallenge, &SparseChallenge), AkitaError> {
        let dims = self.dims();
        let blocks_per_claim = dims.blocks_per_claim()?;
        let total_blocks = dims.total_blocks()?;
        if block_idx >= total_blocks {
            return Err(AkitaError::InvalidInput(format!(
                "tensor block index {block_idx} out of range for {total_blocks} blocks"
            )));
        }
        let claim_idx = block_idx / blocks_per_claim;
        let local_block_idx = block_idx % blocks_per_claim;
        let (left_idx, right_idx) = dims.factor_indices(local_block_idx)?;
        let (left, right) = self.factor_slices_for_claim(claim_idx)?;
        Ok((
            claim_idx,
            local_block_idx,
            &left[left_idx],
            &right[right_idx],
        ))
    }

    /// Return a new set containing the selected claims in the requested order.
    ///
    /// # Errors
    ///
    /// Returns an error if any claim index is out of range or if this set's
    /// factor lengths are inconsistent.
    pub fn select_claims(&self, claim_indices: &[usize]) -> Result<Self, AkitaError> {
        let mut left = Vec::with_capacity(claim_indices.len().saturating_mul(self.left_len));
        let mut right = Vec::with_capacity(claim_indices.len().saturating_mul(self.right_len));
        for &claim_idx in claim_indices {
            let (left_slice, right_slice) = self.factor_slices_for_claim(claim_idx)?;
            left.extend_from_slice(left_slice);
            right.extend_from_slice(right_slice);
        }
        Self::new(
            left,
            right,
            self.left_len,
            self.right_len,
            claim_indices.len(),
        )
    }

    /// Validate tensor dimensions against an expected block count and return
    /// `(left_bits, right_bits)`.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are not power-of-two, do not multiply to
    /// `num_blocks`, or if count arithmetic overflows.
    pub fn validate_power_of_two_dimensions(
        &self,
        num_blocks: usize,
    ) -> Result<(usize, usize), AkitaError> {
        let blocks_per_claim = self.blocks_per_claim()?;
        if blocks_per_claim != num_blocks {
            return Err(AkitaError::InvalidSize {
                expected: num_blocks,
                actual: blocks_per_claim,
            });
        }
        if !self.left_len.is_power_of_two() || !self.right_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "tensor challenge dimensions must be powers of two".to_string(),
            ));
        }
        Ok((
            self.left_len.trailing_zeros() as usize,
            self.right_len.trailing_zeros() as usize,
        ))
    }

    /// Expand tensor products into logical flat order.
    ///
    /// # Errors
    ///
    /// Returns an error if any tensor product has malformed inputs or overflows
    /// its integer coefficient representation.
    pub fn expand_integer<const D: usize>(&self) -> Result<Vec<IntegerChallenge>, AkitaError> {
        self.validate_lengths()?;
        let mut out = Vec::with_capacity(self.total_blocks()?);
        for claim_idx in 0..self.num_claims {
            let (left, right) = self.factor_slices_for_claim(claim_idx)?;
            for left_factor in left {
                for right_factor in right {
                    out.push(IntegerChallenge::tensor_product::<D>(
                        left_factor,
                        right_factor,
                    )?);
                }
            }
        }
        Ok(out)
    }

    /// Evaluate reduced tensor products in logical flat order.
    ///
    /// # Errors
    ///
    /// Returns an error if challenge shape validation or evaluation fails.
    pub fn evals_at_pows<F, E, const D: usize>(
        &self,
        alpha_pows: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: FieldCore + MulBase<F>,
    {
        if alpha_pows.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: alpha_pows.len(),
            });
        }
        if D < 2 {
            return Err(AkitaError::InvalidInput(
                "tensor evaluation requires D >= 2".to_string(),
            ));
        }
        self.validate_lengths()?;

        // For `α ∈ E`, `α^D + 1 ∈ E` is the negacyclic-reduction scalar.
        // Tensor products commute with this reduction up to subtraction of a
        // quotient contribution; we precompute the scalar once and reuse it.
        let alpha_pow_d_plus_one = alpha_pows[D - 1] * alpha_pows[1] + E::one();
        let mut out = Vec::with_capacity(self.total_blocks()?);
        for claim_idx in 0..self.num_claims {
            let (left, right) = self.factor_slices_for_claim(claim_idx)?;
            let left_evals = left
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E, D>(alpha_pows))
                .collect::<Result<Vec<_>, _>>()?;
            let right_evals = right
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E, D>(alpha_pows))
                .collect::<Result<Vec<_>, _>>()?;

            for (p, left_challenge) in left.iter().enumerate() {
                for (q, right_challenge) in right.iter().enumerate() {
                    let quotient_eval = tensor_product_quotient_eval::<F, E, D>(
                        left_challenge,
                        right_challenge,
                        alpha_pows,
                    )?;
                    out.push(left_evals[p] * right_evals[q] - alpha_pow_d_plus_one * quotient_eval);
                }
            }
        }
        Ok(out)
    }

    /// Evaluate one weighted tensor aggregate exactly at a ring-switch point.
    ///
    /// Computes
    ///
    /// ```text
    /// Σ_{p,q} u[p] · v[q] · eval(reduce(L_p · R_q), α)
    /// ```
    ///
    /// without materialising every reduced tensor product. The negacyclic
    /// correction term is explicit, so the result is exact at every
    /// ring-switch point where `α^D + 1` is non-zero.
    ///
    /// # Errors
    ///
    /// Returns an error if weights, powers, claim routing, or sparse challenge
    /// representations are inconsistent with this tensor challenge set.
    pub fn eval_factored_aggregate_at_pows<F, E, const D: usize>(
        &self,
        claim_idx: usize,
        u_weights: &[E],
        v_weights: &[E],
        alpha_pows: &[E],
        alpha_pow_d_plus_one: E,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: FieldCore + MulBase<F>,
    {
        if alpha_pows.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: alpha_pows.len(),
            });
        }
        if u_weights.len() != self.left_len {
            return Err(AkitaError::InvalidSize {
                expected: self.left_len,
                actual: u_weights.len(),
            });
        }
        if v_weights.len() != self.right_len {
            return Err(AkitaError::InvalidSize {
                expected: self.right_len,
                actual: v_weights.len(),
            });
        }
        if claim_idx >= self.num_claims {
            return Err(AkitaError::InvalidInput(format!(
                "tensor claim index {claim_idx} out of range for {} claims",
                self.num_claims
            )));
        }
        self.validate_lengths()?;

        let (left, right) = self.factor_slices_for_claim(claim_idx)?;

        // Build the weighted dense factors in E directly so the product
        // evaluation never materialises a length-O(left_len · right_len) buffer.
        let mut left_bar = [E::zero(); D];
        let mut right_bar = [E::zero(); D];

        for (p, &weight) in u_weights.iter().enumerate() {
            if !weight.is_zero() {
                accumulate_sparse_scaled::<F, E, D>(&mut left_bar, &left[p], weight)?;
            }
        }
        for (q, &weight) in v_weights.iter().enumerate() {
            if !weight.is_zero() {
                accumulate_sparse_scaled::<F, E, D>(&mut right_bar, &right[q], weight)?;
            }
        }

        let product_eval =
            eval_dense_at_pows(&left_bar, alpha_pows) * eval_dense_at_pows(&right_bar, alpha_pows);
        let mut quotient_eval = E::zero();
        for (i, &left_coeff) in left_bar.iter().enumerate() {
            if left_coeff.is_zero() {
                continue;
            }
            for (j, &right_coeff) in right_bar.iter().enumerate() {
                if right_coeff.is_zero() {
                    continue;
                }
                if i + j >= D {
                    quotient_eval += left_coeff * right_coeff * alpha_pows[i + j - D];
                }
            }
        }

        Ok(product_eval - alpha_pow_d_plus_one * quotient_eval)
    }

    fn factor_range(
        &self,
        claim_idx: usize,
        factor_len: usize,
        label: &str,
    ) -> Result<Range<usize>, AkitaError> {
        if claim_idx >= self.num_claims {
            return Err(AkitaError::InvalidInput(format!(
                "tensor claim index {claim_idx} out of range for {} claims",
                self.num_claims
            )));
        }
        let start = claim_idx
            .checked_mul(factor_len)
            .ok_or_else(|| AkitaError::InvalidSetup(format!("{label} offset overflow")))?;
        let end = start
            .checked_add(factor_len)
            .ok_or_else(|| AkitaError::InvalidSetup(format!("{label} end overflow")))?;
        Ok(start..end)
    }

    fn validate_lengths(&self) -> Result<(), AkitaError> {
        let expected_left = self
            .num_claims
            .checked_mul(self.left_len)
            .ok_or_else(|| AkitaError::InvalidSetup("tensor-left count overflow".to_string()))?;
        if self.left.len() != expected_left {
            return Err(AkitaError::InvalidSize {
                expected: expected_left,
                actual: self.left.len(),
            });
        }
        let expected_right = self
            .num_claims
            .checked_mul(self.right_len)
            .ok_or_else(|| AkitaError::InvalidSetup("tensor-right count overflow".to_string()))?;
        if self.right.len() != expected_right {
            return Err(AkitaError::InvalidSize {
                expected: expected_right,
                actual: self.right.len(),
            });
        }
        Ok(())
    }
}

fn accumulate_sparse_scaled<F, E, const D: usize>(
    out: &mut [E],
    challenge: &SparseChallenge,
    scale: E,
) -> Result<(), AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FieldCore + MulBase<F>,
{
    if out.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: out.len(),
        });
    }
    if challenge.positions.len() != challenge.coeffs.len() {
        return Err(AkitaError::InvalidInput(
            "sparse challenge positions/coeffs length mismatch".to_string(),
        ));
    }

    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let idx = pos as usize;
        if idx >= D {
            return Err(AkitaError::InvalidInput(format!(
                "sparse challenge position {idx} out of range for D={D}"
            )));
        }
        if coeff == 0 {
            return Err(AkitaError::InvalidInput(
                "sparse challenge coefficients must be non-zero".to_string(),
            ));
        }
        out[idx] += scale.mul_base(F::from_i64(coeff as i64));
    }
    Ok(())
}

fn validate_sparse_factor<const D: usize>(challenge: &SparseChallenge) -> Result<(), AkitaError> {
    if challenge.positions.len() != challenge.coeffs.len() {
        return Err(AkitaError::InvalidInput(
            "sparse challenge positions/coeffs length mismatch".to_string(),
        ));
    }
    let mut terms = challenge
        .positions
        .iter()
        .copied()
        .zip(challenge.coeffs.iter().copied())
        .collect::<Vec<_>>();
    terms.sort_by_key(|&(pos, _)| pos);
    let mut previous_pos = None;
    for (pos, coeff) in terms {
        if pos as usize >= D {
            return Err(AkitaError::InvalidInput(format!(
                "sparse challenge position {pos} out of range for D={D}"
            )));
        }
        if coeff == 0 {
            return Err(AkitaError::InvalidInput(
                "sparse challenge coefficients must be non-zero".to_string(),
            ));
        }
        if previous_pos == Some(pos) {
            return Err(AkitaError::InvalidInput(
                "sparse challenge positions must be unique".to_string(),
            ));
        }
        previous_pos = Some(pos);
    }
    Ok(())
}

fn eval_dense_at_pows<E: FieldCore>(coeffs: &[E], alpha_pows: &[E]) -> E {
    coeffs
        .iter()
        .zip(alpha_pows.iter())
        .fold(E::zero(), |acc, (&coeff, &power)| acc + coeff * power)
}

fn tensor_product_quotient_eval<F, E, const D: usize>(
    left: &SparseChallenge,
    right: &SparseChallenge,
    alpha_pows: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FieldCore + MulBase<F>,
{
    if left.positions.len() != left.coeffs.len() || right.positions.len() != right.coeffs.len() {
        return Err(AkitaError::InvalidInput(
            "tensor challenge positions/coeffs length mismatch".to_string(),
        ));
    }
    let mut quotient_eval = E::zero();
    for (&left_pos, &left_coeff) in left.positions.iter().zip(left.coeffs.iter()) {
        let left_idx = left_pos as usize;
        if left_idx >= D {
            return Err(AkitaError::InvalidInput(format!(
                "tensor-left challenge position {left_idx} out of range for D={D}"
            )));
        }
        if left_coeff == 0 {
            return Err(AkitaError::InvalidInput(
                "tensor-left challenge coefficients must be non-zero".to_string(),
            ));
        }
        for (&right_pos, &right_coeff) in right.positions.iter().zip(right.coeffs.iter()) {
            let right_idx = right_pos as usize;
            if right_idx >= D {
                return Err(AkitaError::InvalidInput(format!(
                    "tensor-right challenge position {right_idx} out of range for D={D}"
                )));
            }
            if right_coeff == 0 {
                return Err(AkitaError::InvalidInput(
                    "tensor-right challenge coefficients must be non-zero".to_string(),
                ));
            }
            let degree = left_idx + right_idx;
            if degree >= D {
                let term = i64::from(left_coeff) * i64::from(right_coeff);
                quotient_eval += alpha_pows[degree - D].mul_base(F::from_i64(term));
            }
        }
    }
    Ok(quotient_eval)
}

/// Split `num_blocks = 2^r` into balanced tensor dimensions.
///
/// # Errors
///
/// Returns an error if `num_blocks` is not a power of two.
#[inline]
pub fn tensor_split(num_blocks: usize) -> Result<(usize, usize), AkitaError> {
    if !num_blocks.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "tensor challenges require a power-of-two block count".to_string(),
        ));
    }
    let r = num_blocks.trailing_zeros() as usize;
    let left_bits = r / 2;
    let right_bits = r - left_bits;
    Ok((1usize << left_bits, 1usize << right_bits))
}

/// Compute the canonical digest absorbed between tensor-left and tensor-right
/// challenge sampling.
///
/// Binding the right vector's transcript challenge to the exact left vector
/// blocks any adaptive ground-out attempt on the right factor.
///
/// # Errors
///
/// Returns an error if the tensor-left vector length is inconsistent with the
/// supplied shape or if any sparse challenge violates structural invariants.
pub fn tensor_left_digest<const D: usize>(
    left: &[SparseChallenge],
    left_len: usize,
    num_claims: usize,
) -> Result<[u8; 32], AkitaError> {
    let expected = left_len
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("tensor-left digest count overflow".to_string()))?;
    if left.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: left.len(),
        });
    }

    let mut hasher = Sha3_256::new();
    hasher.update(TENSOR_LEFT_DIGEST_DOMAIN);
    hasher.update((D as u64).to_le_bytes());
    hasher.update((num_claims as u64).to_le_bytes());
    hasher.update((left_len as u64).to_le_bytes());
    hasher.update((left.len() as u64).to_le_bytes());

    for challenge in left {
        if challenge.positions.len() != challenge.coeffs.len() {
            return Err(AkitaError::InvalidInput(
                "tensor-left digest positions/coeffs length mismatch".to_string(),
            ));
        }
        hasher.update((challenge.positions.len() as u64).to_le_bytes());

        let mut terms: Vec<(u32, i8)> = challenge
            .positions
            .iter()
            .copied()
            .zip(challenge.coeffs.iter().copied())
            .collect();
        terms.sort_by_key(|&(pos, _)| pos);
        let mut previous_pos = None;
        for (pos, coeff) in terms {
            if pos as usize >= D {
                return Err(AkitaError::InvalidInput(format!(
                    "tensor-left digest position {pos} out of range for D={D}"
                )));
            }
            if coeff == 0 {
                return Err(AkitaError::InvalidInput(
                    "tensor-left digest coefficients must be non-zero".to_string(),
                ));
            }
            if previous_pos == Some(pos) {
                return Err(AkitaError::InvalidInput(
                    "tensor-left digest positions must be unique".to_string(),
                ));
            }
            previous_pos = Some(pos);
            hasher.update(pos.to_le_bytes());
            hasher.update(coeff.to_le_bytes());
        }
    }

    Ok(hasher.finalize().into())
}

/// Sample folding challenges using the configured shape.
///
/// # Errors
///
/// Returns an error if count arithmetic overflows, if tensor splitting is
/// invalid, or if sparse challenge sampling fails.
pub fn sample_tensor_challenges<F, T, const D: usize>(
    transcript: &mut T,
    num_blocks: usize,
    num_claims: usize,
    cfg: &SparseChallengeConfig,
    shape: &TensorChallengeShape,
    labels: TensorChallengeLabels<'_>,
) -> Result<TensorChallenges, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    match shape {
        TensorChallengeShape::Flat => {
            let total = num_blocks.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("tensor challenge count overflow".to_string())
            })?;
            sample_sparse_challenges::<F, T, D>(transcript, labels.flat, total, cfg)
                .map(TensorChallenges::Flat)
        }
        TensorChallengeShape::Tensor => {
            let (left_len, right_len) = tensor_split(num_blocks)?;
            let left_total = left_len.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("tensor-left challenge count overflow".to_string())
            })?;
            let right_total = right_len.checked_mul(num_claims).ok_or_else(|| {
                AkitaError::InvalidSetup("tensor-right challenge count overflow".to_string())
            })?;
            let left = sample_sparse_challenges::<F, T, D>(
                transcript,
                labels.tensor_left,
                left_total,
                cfg,
            )?;
            let left_digest = tensor_left_digest::<D>(&left, left_len, num_claims)?;
            transcript.append_bytes(labels.tensor_left_digest, &left_digest);
            let right = sample_sparse_challenges::<F, T, D>(
                transcript,
                labels.tensor_right,
                right_total,
                cfg,
            )?;
            TensorChallengeSet::new(left, right, left_len, right_len, num_claims)
                .map(TensorChallenges::Tensor)
        }
    }
}
