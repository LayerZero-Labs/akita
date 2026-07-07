//! Shared public statement for the per-fold negacyclic-ring relation `M * z = y + (X^D + 1) * r`.

use super::OpeningClaimsLayout;
use crate::layout::{
    CommitmentRingDims, LevelParams, MRowLayout, RelationQuotientLayout, RelationRowLayout,
    RingRole,
};
use crate::validate_role_dispatch;
use crate::witness::{WitnessChunkLayout, WitnessChunkLengths, WitnessLayout};
use crate::FpExtEncoding;
use crate::{
    embed_ring_subfield_scalar, r_decomp_levels, RingMultiplierOpeningPoint, RingOpeningPoint,
    RingVec,
};
use akita_algebra::CyclotomicRing;
use akita_challenges::Challenges;
use akita_field::{AkitaError, FieldCore};
use akita_field::{CanonicalField, ExtField, FromPrimitiveInt};

/// Ring-column counts per witness segment in emission order (`z ‖ e ‖ t ‖ …`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingRelationSegmentLengths {
    pub z_len: usize,
    pub e_len: usize,
    pub t_len: usize,
}

/// Opening-batch counts that determine witness segment widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingRelationOpeningCounts {
    pub num_claims: usize,
    pub num_t_vectors: usize,
}

/// Witness segment lengths shared by prover emission, layout offsets, and M-table sizing.
pub fn ring_relation_segment_lengths<F: FieldCore + CanonicalField>(
    lp: &LevelParams,
    opening_counts: RingRelationOpeningCounts,
    _m_row_layout: MRowLayout,
) -> Result<RingRelationSegmentLengths, AkitaError> {
    let num_blocks = lp.num_blocks;
    if num_blocks == 0 || !num_blocks.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "num_blocks must be a non-zero power of two".to_string(),
        ));
    }
    let depth_open = lp.num_digits_open;
    let depth_commit = lp.num_digits_commit;
    let RingRelationOpeningCounts {
        num_claims,
        num_t_vectors,
    } = opening_counts;
    let depth_fold = lp.num_digits_fold(num_t_vectors, lp.field_bits_for_cache())?;
    if depth_open == 0 || depth_commit == 0 || depth_fold == 0 {
        return Err(AkitaError::InvalidSetup(
            "prepared ring-switch layout has zero width".to_string(),
        ));
    }
    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("total block count overflow".to_string()))?;
    let t_total_blocks = num_blocks
        .checked_mul(num_t_vectors)
        .ok_or_else(|| AkitaError::InvalidSetup("T block count overflow".to_string()))?;

    let e_len = depth_open
        .checked_mul(total_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("e-hat segment length overflow".to_string()))?;
    let t_len = depth_open
        .checked_mul(lp.a_key.row_len())
        .and_then(|len| len.checked_mul(t_total_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("T segment length overflow".to_string()))?;
    let z_len = depth_fold
        .checked_mul(depth_commit)
        .and_then(|len| len.checked_mul(lp.block_len))
        .ok_or_else(|| AkitaError::InvalidSetup("Z segment length overflow".to_string()))?;

    Ok(RingRelationSegmentLengths {
        z_len,
        e_len,
        t_len,
    })
}

/// Public statement of the negacyclic-ring matrix relation at one fold level.
///
/// Ring dimension is stored at runtime; hot paths inside `dispatch_ring_dim`
/// closures borrow typed ring rows via [`Self::y_trusted`], [`Self::v_trusted`],
/// and [`Self::row_coefficient_rings_trusted`].
#[derive(Debug, Clone)]
pub struct RingRelationInstance<F: FieldCore> {
    m_row_layout: MRowLayout,
    pub challenges: Challenges,
    opening_point: RingOpeningPoint<F>,
    ring_multiplier_point: RingMultiplierOpeningPoint<F>,
    opening_batch: OpeningClaimsLayout,
    gamma: Vec<F>,
    row_coefficient_rings: RingVec<F>,
    y: RingVec<F>,
    v: RingVec<F>,
    role_dims: CommitmentRingDims,
}

impl<F: FieldCore + CanonicalField> RingRelationInstance<F> {
    /// Construct a validated ring-relation statement from D-free ring storage.
    ///
    /// Does not sample from the transcript; callers must absorb/sample before calling.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m_row_layout: MRowLayout,
        challenges: Challenges,
        opening_point: RingOpeningPoint<F>,
        ring_multiplier_point: RingMultiplierOpeningPoint<F>,
        opening_batch: OpeningClaimsLayout,
        gamma: Vec<F>,
        row_coefficient_rings: RingVec<F>,
        y: RingVec<F>,
        v: RingVec<F>,
        role_dims: CommitmentRingDims,
    ) -> Result<Self, AkitaError> {
        opening_batch.check()?;
        if gamma.len() != opening_batch.num_total_polynomials()
            || row_coefficient_rings.count() != opening_batch.num_total_polynomials()
        {
            return Err(AkitaError::InvalidInput(
                "ring relation gamma/row coefficients length mismatch".to_string(),
            ));
        }
        if y.coeff_len() < role_dims.d_a() {
            return Err(AkitaError::InvalidInput(
                "ring relation y must contain at least the consistency row".to_string(),
            ));
        }
        if role_dims.d_a() == 0 || role_dims.d_b() == 0 || role_dims.d_d() == 0 {
            return Err(AkitaError::InvalidSize {
                expected: 1,
                actual: 0,
            });
        }
        if !row_coefficient_rings.can_decode_vec(role_dims.d_a()) {
            return Err(AkitaError::InvalidSize {
                expected: role_dims.d_a(),
                actual: row_coefficient_rings.coeff_len(),
            });
        }
        if !v.coeffs().is_empty() && !v.can_decode_vec(role_dims.d_d()) {
            return Err(AkitaError::InvalidSize {
                expected: role_dims.d_d(),
                actual: v.coeff_len(),
            });
        }
        if let Ok(uniform) = role_dims.uniform_dim() {
            if !y.can_decode_vec(uniform) {
                return Err(AkitaError::InvalidSize {
                    expected: uniform,
                    actual: y.coeff_len(),
                });
            }
        }
        for (idx, chunk) in row_coefficient_rings
            .coeffs()
            .chunks_exact(role_dims.d_a())
            .enumerate()
        {
            if gamma.get(idx) != Some(&chunk[0]) {
                return Err(AkitaError::InvalidInput(
                    "ring relation gamma does not match row coefficient rings".to_string(),
                ));
            }
        }
        Ok(Self {
            m_row_layout,
            challenges,
            opening_point,
            ring_multiplier_point,
            opening_batch,
            gamma,
            row_coefficient_rings,
            y,
            v,
            role_dims,
        })
    }

    /// Construct from typed kernel outputs at a ring-relation boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts<const D: usize>(
        m_row_layout: MRowLayout,
        challenges: Challenges,
        opening_point: RingOpeningPoint<F>,
        ring_multiplier_point: RingMultiplierOpeningPoint<F>,
        opening_batch: OpeningClaimsLayout,
        gamma: Vec<F>,
        row_coefficient_rings: &[CyclotomicRing<F, D>],
        y: &[CyclotomicRing<F, D>],
        v: &[CyclotomicRing<F, D>],
    ) -> Result<Self, AkitaError> {
        Self::new(
            m_row_layout,
            challenges,
            opening_point,
            ring_multiplier_point,
            opening_batch,
            gamma,
            RingVec::from_ring_elems(row_coefficient_rings),
            RingVec::from_ring_elems(y),
            RingVec::from_ring_elems(v),
            CommitmentRingDims::uniform(D),
        )
    }

    /// Per-role ring dimensions for this relation statement.
    pub fn role_dims(&self) -> CommitmentRingDims {
        self.role_dims
    }

    /// A-role fold dimension (`d_a`).
    pub fn ring_dim(&self) -> usize {
        self.role_dims.d_a()
    }

    pub fn m_row_layout(&self) -> MRowLayout {
        self.m_row_layout
    }

    /// Canonical semantic row layout for this relation statement.
    ///
    /// # Errors
    ///
    /// Returns an error for grouped-root layouts or malformed parameters.
    pub fn relation_row_layout(&self, lp: &LevelParams) -> Result<RelationRowLayout, AkitaError> {
        RelationRowLayout::for_scalar_level::<F>(
            lp,
            self.role_dims,
            self.m_row_layout,
            &self.opening_batch,
            self.opening_batch.num_groups(),
        )
    }

    pub fn opening_batch(&self) -> &OpeningClaimsLayout {
        &self.opening_batch
    }

    pub fn opening_point(&self) -> &RingOpeningPoint<F> {
        &self.opening_point
    }

    /// Per-group opening point for grouped-root replay.
    ///
    /// Singleton relations expose group `0` only.
    pub fn group_opening_point(&self, g: usize) -> Result<&RingOpeningPoint<F>, AkitaError> {
        if self.opening_batch.num_groups() != 1 || g != 0 {
            return Err(AkitaError::InvalidProof);
        }
        Ok(&self.opening_point)
    }

    pub fn ring_multiplier_point(&self) -> &RingMultiplierOpeningPoint<F> {
        &self.ring_multiplier_point
    }

    /// Per-group ring-multiplier opening point for grouped-root replay.
    pub fn group_ring_multiplier_point(
        &self,
        g: usize,
    ) -> Result<&RingMultiplierOpeningPoint<F>, AkitaError> {
        if self.opening_batch.num_groups() != 1 || g != 0 {
            return Err(AkitaError::InvalidProof);
        }
        Ok(&self.ring_multiplier_point)
    }

    /// Per-group stage-1 challenges for grouped-root replay.
    pub fn group_challenges(&self) -> Vec<&Challenges> {
        vec![&self.challenges]
    }

    pub fn gamma(&self) -> &[F] {
        &self.gamma
    }

    /// Public D-block rows in flat ring storage.
    pub fn v(&self) -> &RingVec<F> {
        &self.v
    }

    /// Relation RHS rows in flat ring storage.
    pub fn y(&self) -> &RingVec<F> {
        &self.y
    }

    /// Row-coefficient rings embedded in flat ring storage.
    pub fn row_coefficient_rings(&self) -> &RingVec<F> {
        &self.row_coefficient_rings
    }

    /// Validate that all role carriers match a single uniform dimension `D`.
    ///
    /// Required by fused ring-switch paths that borrow the full `y` vector as
    /// typed rings at one dimension (Slice 1 splits this).
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        let uniform = self.role_dims.uniform_dim()?;
        if uniform != D {
            return Err(AkitaError::InvalidInput(format!(
                "ring relation uniform dim {uniform} does not match requested D={D}"
            )));
        }
        validate_role_dispatch::<D>(self.role_dims, RingRole::Inner)?;
        if !self.row_coefficient_rings.can_decode_vec(D)
            || !self.y.can_decode_vec(D)
            || !self.v.can_decode_vec(D)
        {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.y.coeff_len(),
            });
        }
        self.ring_multiplier_point.ensure_ring_dim::<D>()
    }

    /// Validate one role carrier against dispatch `D`.
    pub fn ensure_role_dim<const D: usize>(&self, role: RingRole) -> Result<(), AkitaError> {
        validate_role_dispatch::<D>(self.role_dims, role).map(|_| ())
    }

    /// Borrow `y` rows when all roles share one dimension.
    pub fn y_trusted<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        self.y.as_ring_slice::<D>()
    }

    /// Borrow `v` rows at the D-role dimension (`d_d`).
    pub fn v_trusted<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_role_dim::<D>(RingRole::Opening)?;
        self.v.as_ring_slice::<D>()
    }

    /// Borrow row-coefficient rings at the A-role dimension (`d_a`).
    pub fn row_coefficient_rings_trusted<const D: usize>(
        &self,
    ) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_role_dim::<D>(RingRole::Inner)?;
        self.row_coefficient_rings.as_ring_slice::<D>()
    }

    /// Validate layout-dependent D-row payload shape.
    pub fn check_v_shape_for_level(&self, lp: &LevelParams) -> Result<(), AkitaError> {
        let expected = match self.m_row_layout {
            MRowLayout::WithDBlock => lp.d_key.row_len(),
            MRowLayout::WithoutDBlock => 0,
        };
        let d_d = self.role_dims.d_d();
        let actual = if self.v.coeff_len() == 0 {
            0
        } else if !self.v.can_decode_vec(d_d) {
            return Err(AkitaError::InvalidSize {
                expected: d_d,
                actual: self.v.coeff_len(),
            });
        } else {
            self.v.coeff_len() / d_d
        };
        if actual != expected {
            return Err(AkitaError::InvalidInput(
                "ring relation v rows do not match M-row layout".to_string(),
            ));
        }
        Ok(())
    }

    /// Build base-field `gamma` and embedded row rings from transcript-sampled coefficients.
    pub fn gamma_and_row_rings_from_coefficients<const D: usize, E>(
        row_coefficients: &[E],
    ) -> Result<(Vec<F>, RingVec<F>), AkitaError>
    where
        F: FromPrimitiveInt,
        E: FpExtEncoding<F> + ExtField<F>,
    {
        let mut gamma = Vec::with_capacity(row_coefficients.len());
        let mut row_coefficient_rings = Vec::with_capacity(row_coefficients.len());
        for &coefficient in row_coefficients {
            let ring =
                embed_ring_subfield_scalar::<F, E, D>(coefficient, AkitaError::InvalidProof)?;
            gamma.push(ring.coefficients()[0]);
            row_coefficient_rings.push(ring);
        }
        Ok((gamma, RingVec::from_ring_elems(&row_coefficient_rings)))
    }

    /// Resolve the layout-agnostic [`WitnessLayout`] for this level's witness,
    /// validating shape and (when supplied) capacity at the boundary.
    ///
    /// This is the **single source of truth** for witness column offsets shared
    /// by the distributed prover's emission and the verifier's row-MLE
    /// evaluation. `lp.witness_chunk.num_chunks = 1` yields a single chunk with
    /// the historical `z ‖ e ‖ t ‖ u ‖ r` offsets; `num_chunks = W` lays out `W`
    /// contiguous `[zᵢ | eᵢ | t̂ᵢ]` strides (z-first, `zᵢ` replicated, `eᵢ`/`t̂ᵢ`
    /// partitioned) followed by one shared `r̂` tail sized at the single-machine
    /// row count. Pass `witness_ring_len = Some(w_len / D)` to enforce the
    /// no-panic capacity bound at this boundary.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] (never panics) for a malformed chunk
    /// count (`0`, non-power-of-two, `> num_blocks`, or `∤ num_blocks`), a
    /// non-power-of-two block window, or any offset/length arithmetic overflow, or a
    /// layout whose `r̂` tail would exceed the committed witness capacity.
    pub fn segment_layout(
        &self,
        lp: &LevelParams,
        witness_ring_len: Option<usize>,
    ) -> Result<WitnessLayout, AkitaError> {
        let num_claims = self.opening_batch.num_total_polynomials();
        let lens = ring_relation_segment_lengths::<F>(
            lp,
            RingRelationOpeningCounts {
                num_claims,
                num_t_vectors: num_claims,
            },
            self.m_row_layout,
        )?;
        let RingRelationSegmentLengths {
            z_len,
            e_len,
            t_len,
        } = lens;

        let num_blocks = lp.num_blocks;
        if lp.witness_chunk.num_chunks == 0 {
            return Err(AkitaError::InvalidSetup(
                "witness chunk count must be >= 1".to_string(),
            ));
        }
        let num_chunks = lp.witness_chunk.num_chunks;

        // Shared, single-machine quotient tail: never scales with the chunk count.
        let r_levels = r_decomp_levels::<F>(lp.log_basis);
        let _n_d_active = match self.m_row_layout {
            MRowLayout::WithDBlock => lp.d_key.row_len(),
            MRowLayout::WithoutDBlock => 0,
        };
        let y_layout = crate::proof::relation::relation_y_layout_for(
            lp,
            &self.opening_batch,
            self.m_row_layout,
        )?;
        let expected_y_len =
            crate::proof::relation::relation_y_coeff_len(self.role_dims, &y_layout)?;
        if self.y.coeff_len() != expected_y_len {
            return Err(AkitaError::InvalidSetup(format!(
                "ring relation y coefficient length {} does not match per-role layout (expected {expected_y_len})",
                self.y.coeff_len()
            )));
        }
        let relation_row_layout = RelationRowLayout::for_scalar_level::<F>(
            lp,
            self.role_dims,
            self.m_row_layout,
            &self.opening_batch,
            1,
        )?;
        let quotient_layout =
            RelationQuotientLayout::from_row_layout(&relation_row_layout, r_levels);
        quotient_layout.validate()?;
        let r_len_total = quotient_layout.total_coeffs();

        // The single-chunk layout is the `num_chunks = 1` case of the chunked
        // construction below; only multi-chunk needs the extra well-formedness checks.
        if num_chunks > 1 {
            if !num_chunks.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(
                    "witness chunk count must be a power of two".to_string(),
                ));
            }
            if num_chunks > num_blocks {
                return Err(AkitaError::InvalidSetup(
                    "witness chunk count exceeds num_blocks".to_string(),
                ));
            }
            if !num_blocks.is_multiple_of(num_chunks) {
                return Err(AkitaError::InvalidSetup(
                    "witness chunk count must divide num_blocks".to_string(),
                ));
            }
            if !(num_blocks / num_chunks).is_power_of_two() {
                return Err(AkitaError::InvalidSetup(
                    "witness chunk block window must be a power of two".to_string(),
                ));
            }
            if !e_len.is_multiple_of(num_chunks) || !t_len.is_multiple_of(num_chunks) {
                return Err(AkitaError::InvalidSetup(
                    "partitioned witness segment lengths must divide evenly across chunks"
                        .to_string(),
                ));
            }
        }

        // `ê`/`t̂` are partitioned across windows; `ẑ` is replicated full-width
        // in every window. The shared `r̂` tails the last window.
        let blocks_per_chunk = num_blocks / num_chunks;
        let e_len_j = e_len / num_chunks;
        let t_len_j = t_len / num_chunks;
        let stride = z_len
            .checked_add(e_len_j)
            .and_then(|s| s.checked_add(t_len_j))
            .ok_or_else(|| AkitaError::InvalidSetup("chunk stride overflow".to_string()))?;

        let mut chunks = Vec::with_capacity(num_chunks);
        let mut chunk_lengths = Vec::with_capacity(num_chunks);
        for j in 0..num_chunks {
            let is_last = j == num_chunks - 1;
            let base = j
                .checked_mul(stride)
                .ok_or_else(|| AkitaError::InvalidSetup("chunk base overflow".to_string()))?;
            let offset_e = base
                .checked_add(z_len)
                .ok_or_else(|| AkitaError::InvalidSetup("chunk e offset overflow".to_string()))?;
            let offset_t = offset_e
                .checked_add(e_len_j)
                .ok_or_else(|| AkitaError::InvalidSetup("chunk t offset overflow".to_string()))?;
            let after_t = offset_t
                .checked_add(t_len_j)
                .ok_or_else(|| AkitaError::InvalidSetup("chunk r offset overflow".to_string()))?;
            let offset_r = if is_last { Some(after_t) } else { None };
            let global_block_base = j.checked_mul(blocks_per_chunk).ok_or_else(|| {
                AkitaError::InvalidSetup("global block base overflow".to_string())
            })?;
            chunks.push(WitnessChunkLayout {
                offset_z: base,
                offset_e,
                offset_t,
                offset_u: None,
                offset_r,
                global_block_base,
            });
            chunk_lengths.push(WitnessChunkLengths {
                z_len,
                e_len: e_len_j,
                t_len: t_len_j,
                u_len: None,
                r_len: is_last.then_some(r_len_total),
            });
        }
        let layout = WitnessLayout {
            blocks_per_chunk,
            chunks,
            chunk_lengths,
            quotient_layout,
        };

        if let Some(witness_ring_len) = witness_ring_len {
            let r_offset = layout.r_offset()?;
            let needed = r_offset
                .checked_add(r_len_total)
                .ok_or_else(|| AkitaError::InvalidSetup("witness capacity overflow".to_string()))?;
            if needed > witness_ring_len {
                return Err(AkitaError::InvalidSetup(format!(
                    "resolved witness layout requires {needed} ring columns but only {witness_ring_len} are committed"
                )));
            }
        }

        Ok(layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::{SparseChallenge, SparseChallengeConfig};
    use akita_field::Fp32;

    type F = Fp32<251>;
    const D: usize = 32;

    fn stage1_config() -> SparseChallengeConfig {
        SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![1],
        }
    }

    fn opening_point(lp: &LevelParams) -> RingOpeningPoint<F> {
        RingOpeningPoint {
            a: vec![F::zero(); lp.block_len],
            b: vec![F::zero(); lp.num_blocks],
        }
    }

    fn test_level_params() -> LevelParams {
        LevelParams::params_only(crate::SisModulusFamily::Q32, D, 2, 1, 1, 1, stage1_config())
            .with_decomp(2, 1, 1, 2, 0)
            .expect("test params")
    }

    fn test_challenges(lp: &LevelParams, num_claims: usize) -> Challenges {
        let total = lp.num_blocks * num_claims;
        Challenges::from_sparse(
            vec![
                SparseChallenge {
                    positions: vec![0],
                    coeffs: vec![1],
                };
                total
            ],
            lp.num_blocks,
            num_claims,
        )
        .expect("challenges")
    }

    #[test]
    fn relation_instance_rejects_empty_y() {
        let lp = test_level_params();
        let opening_batch = OpeningClaimsLayout::new(2, 1).expect("valid opening batch");
        let opening_point = opening_point(&lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let err = RingRelationInstance::<F>::new(
            MRowLayout::WithoutDBlock,
            test_challenges(&lp, opening_batch.num_total_polynomials()),
            opening_point,
            ring_multiplier_point,
            opening_batch,
            vec![F::one()],
            RingVec::from_ring_elems::<D>(&[CyclotomicRing::one()]),
            RingVec::from_ring_elems::<D>(&[]),
            RingVec::from_ring_elems::<D>(&[]),
            CommitmentRingDims::uniform(D),
        )
        .expect_err("empty y must be rejected");
        assert!(
            format!("{err:?}")
                .contains("ring relation y must contain at least the consistency row"),
            "unexpected error: {err:?}"
        );
    }

    fn chunk_test_level_params(r_vars: usize) -> LevelParams {
        // num_blocks = 2^r_vars, block_len = 2^m_vars, single-tier.
        LevelParams::params_only(crate::SisModulusFamily::Q32, D, 2, 1, 1, 1, stage1_config())
            .with_decomp(2, r_vars, 1, 2, 0)
            .expect("test params")
    }

    /// Build a minimal `WithDBlock` relation instance whose layout-relevant
    /// shape is `opening_batch.num_total_polynomials() = num_claims` and `y.len() =
    /// num_rows` (the only fields [`RingRelationInstance::segment_layout`] reads).
    fn build_instance(
        lp: &LevelParams,
        num_claims: usize,
        num_rows: usize,
    ) -> RingRelationInstance<F> {
        let opening_batch = OpeningClaimsLayout::new(8, num_claims).expect("opening batch");
        let opening_point = opening_point(lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        RingRelationInstance::<F>::new(
            MRowLayout::WithDBlock,
            test_challenges(lp, num_claims),
            opening_point,
            ring_multiplier_point,
            opening_batch,
            vec![F::one(); num_claims],
            RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::one(); num_claims]),
            RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::zero(); num_rows]),
            RingVec::from_ring_elems::<D>(&[]),
            CommitmentRingDims::uniform(D),
        )
        .expect("instance")
    }

    #[test]
    fn resolve_single_chunk_matches_legacy_offsets() {
        let lp = chunk_test_level_params(1);
        assert_eq!(lp.witness_chunk.num_chunks, 1);
        let num_claims = 3;
        let lens = ring_relation_segment_lengths::<F>(
            &lp,
            RingRelationOpeningCounts {
                num_claims,
                num_t_vectors: num_claims,
            },
            MRowLayout::WithDBlock,
        )
        .expect("lengths");

        let resolved = build_instance(&lp, num_claims, 4)
            .segment_layout(&lp, None)
            .expect("resolved layout");
        assert_eq!(resolved.num_chunks(), 1);
        let chunk = resolved.chunks[0];
        // Legacy single-chunk offsets: z-first, then e, t, (u), r.
        assert_eq!(chunk.offset_z, 0);
        assert_eq!(chunk.offset_e, lens.z_len);
        assert_eq!(chunk.offset_t, lens.z_len + lens.e_len);
        // Single-tier fixture: u segment absent, r tails z‖e‖t.
        assert_eq!(chunk.offset_r, Some(lens.z_len + lens.e_len + lens.t_len));
        assert_eq!(chunk.global_block_base, 0);
        assert_eq!(resolved.blocks_per_chunk, lp.num_blocks);
    }

    #[test]
    fn resolve_multi_chunk_offsets_contiguous_and_cover_blocks() {
        let num_claims = 2;
        for w in [1usize, 2, 4, 8] {
            let mut lp = chunk_test_level_params(3); // num_blocks = 8
            if w > 1 {
                lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
                    num_chunks: w,
                    num_activated_levels: 1,
                };
            }
            let lens = ring_relation_segment_lengths::<F>(
                &lp,
                RingRelationOpeningCounts {
                    num_claims,
                    num_t_vectors: num_claims,
                },
                MRowLayout::WithDBlock,
            )
            .expect("lengths");
            let layout = build_instance(&lp, num_claims, 4)
                .segment_layout(&lp, None)
                .expect("layout");
            assert_eq!(layout.num_chunks(), w);
            assert_eq!(layout.blocks_per_chunk, lp.num_blocks / w);

            // Partitioned e/t lengths sum to the single-machine totals; z replicated.
            let e_sum: usize = layout.chunk_lengths.iter().map(|l| l.e_len).sum();
            let t_sum: usize = layout.chunk_lengths.iter().map(|l| l.t_len).sum();
            assert_eq!(e_sum, lens.e_len);
            assert_eq!(t_sum, lens.t_len);
            for l in &layout.chunk_lengths {
                assert_eq!(l.z_len, lens.z_len);
            }

            // Offsets are contiguous z-first per chunk; only the last chunk has r̂.
            let stride = lens.z_len + lens.e_len / w + lens.t_len / w;
            for (j, chunk) in layout.chunks.iter().enumerate() {
                let base = j * stride;
                assert_eq!(chunk.offset_z, base);
                assert_eq!(chunk.offset_e, base + lens.z_len);
                assert_eq!(chunk.offset_t, base + lens.z_len + lens.e_len / w);
                assert_eq!(chunk.global_block_base, j * (lp.num_blocks / w));
                assert_eq!(chunk.offset_u, None);
                if j + 1 == w {
                    assert_eq!(chunk.offset_r, Some(w * stride));
                } else {
                    assert_eq!(chunk.offset_r, None);
                }
            }
            // Block windows tile [0, num_blocks).
            assert_eq!(
                layout.chunks.last().unwrap().global_block_base + layout.blocks_per_chunk,
                lp.num_blocks
            );
        }
    }

    #[test]
    fn resolve_rejects_bad_chunk_count() {
        let num_claims = 2;
        // num_chunks = 3 is not a power of two.
        let mut lp = chunk_test_level_params(3);
        lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
            num_chunks: 3,
            num_activated_levels: 1,
        };
        assert!(build_instance(&lp, num_claims, 4)
            .segment_layout(&lp, None)
            .is_err());

        // num_chunks = 16 exceeds num_blocks = 8.
        let mut lp = chunk_test_level_params(3);
        lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
            num_chunks: 16,
            num_activated_levels: 1,
        };
        assert!(build_instance(&lp, num_claims, 4)
            .segment_layout(&lp, None)
            .is_err());
    }

    #[test]
    fn resolve_rejects_capacity_overflow() {
        let num_claims = 2;
        let lp = chunk_test_level_params(3);
        // A witness ring capacity of 1 is far smaller than offset_r + r_len.
        assert!(
            build_instance(&lp, num_claims, 4)
                .segment_layout(&lp, Some(1))
                .is_err(),
            "tiny witness capacity must be rejected"
        );
        // A generous capacity passes.
        build_instance(&lp, num_claims, 4)
            .segment_layout(&lp, Some(1 << 20))
            .expect("ample capacity");
    }

    #[test]
    fn relation_segment_layout_uses_same_axis_contract() {
        use crate::proof::relation::{relation_y_row_count, RelationYLayout};

        let lp = test_level_params();
        let opening_batch = OpeningClaimsLayout::new(2, 3).expect("valid batch");
        let opening_point = opening_point(&lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let y_layout = RelationYLayout::uniform(
            lp.d_key.row_len(),
            lp.a_key.row_len(),
            lp.b_key.row_len(),
            0,
            opening_batch.num_groups(),
        );
        let y_rows = relation_y_row_count(&y_layout);
        let v_zeros = vec![CyclotomicRing::zero(); y_layout.n_d];
        let y_zeros = vec![CyclotomicRing::zero(); y_rows];
        let instance = RingRelationInstance::<F>::from_parts::<D>(
            MRowLayout::WithDBlock,
            test_challenges(&lp, opening_batch.num_total_polynomials()),
            opening_point,
            ring_multiplier_point,
            opening_batch,
            vec![F::one(); 3],
            &[CyclotomicRing::one(); 3],
            &y_zeros,
            &v_zeros,
        )
        .expect("same-axis relation");

        let layout = instance.segment_layout(&lp, None).expect("layout");
        let chunk = layout.chunks[0];
        let lens = ring_relation_segment_lengths::<F>(
            &lp,
            RingRelationOpeningCounts {
                num_claims: instance.opening_batch().num_total_polynomials(),
                num_t_vectors: instance.opening_batch().num_total_polynomials(),
            },
            instance.m_row_layout(),
        )
        .expect("segment lengths");
        assert_eq!(layout.num_chunks(), 1);
        assert_eq!(chunk.offset_z, 0);
        assert_eq!(chunk.offset_e, lens.z_len);
        assert_eq!(chunk.offset_t, lens.z_len + lens.e_len);
        assert_eq!(chunk.offset_r, Some(lens.z_len + lens.e_len + lens.t_len));
        instance
            .check_v_shape_for_level(&lp)
            .expect("v rows match layout");
    }
}
