//! Shared public statement for the per-fold negacyclic-ring relation `M * z = y + (X^D + 1) * r`.

use super::OpeningClaimsLayout;
use crate::layout::{CommitmentRingDims, RingRole};
use crate::validate_role_dispatch;
use crate::witness::WitnessLayout;
use crate::FpExtEncoding;
use crate::{
    embed_ring_subfield_scalar, r_decomp_levels, CommittedGroupParams, RingMultiplierOpeningPoint,
    RingOpeningPoint, RingVec,
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
    lp: &CommittedGroupParams,
    opening_counts: RingRelationOpeningCounts,
) -> Result<RingRelationSegmentLengths, AkitaError> {
    let num_live_blocks = lp.num_live_blocks;
    if num_live_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "num_live_blocks must be positive".to_string(),
        ));
    }
    let depth_open = lp.num_digits_open;
    let depth_inner = lp.num_digits_inner;
    let depth_outer = lp.num_digits_outer;
    let RingRelationOpeningCounts {
        num_claims,
        num_t_vectors,
    } = opening_counts;
    let depth_fold = lp.num_digits_fold(num_t_vectors, lp.field_bits_for_cache())?;
    if depth_open == 0 || depth_inner == 0 || depth_outer == 0 || depth_fold == 0 {
        return Err(AkitaError::InvalidSetup(
            "prepared ring-switch layout has zero width".to_string(),
        ));
    }
    let total_blocks = num_live_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("total block count overflow".to_string()))?;
    let t_total_blocks = num_live_blocks
        .checked_mul(num_t_vectors)
        .ok_or_else(|| AkitaError::InvalidSetup("T block count overflow".to_string()))?;

    let e_len = depth_open
        .checked_mul(total_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("e-hat segment length overflow".to_string()))?;
    let t_len = depth_outer
        .checked_mul(lp.inner_commit_matrix.output_rank())
        .and_then(|len| len.checked_mul(t_total_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("T segment length overflow".to_string()))?;
    let z_len = depth_fold
        .checked_mul(depth_inner)
        .and_then(|len| len.checked_mul(lp.num_positions_per_block))
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
/// closures borrow typed ring rows via [`Self::rhs_trusted`], [`Self::v_trusted`],
/// and [`Self::row_coefficient_rings_trusted`].
#[derive(Debug, Clone)]
pub struct RingRelationInstance<F: FieldCore> {
    group_challenges: Vec<Challenges>,
    group_opening_points: Vec<RingOpeningPoint<F>>,
    group_ring_multiplier_points: Vec<RingMultiplierOpeningPoint<F>>,
    opening_batch: OpeningClaimsLayout,
    gamma: Vec<F>,
    row_coefficient_rings: RingVec<F>,
    rhs: RingVec<F>,
    v: RingVec<F>,
    role_dims: CommitmentRingDims,
}

impl<F: FieldCore + CanonicalField> RingRelationInstance<F> {
    /// Construct a validated ring-relation statement from D-free ring storage.
    ///
    /// Does not sample from the transcript; callers must absorb/sample before calling.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        group_challenges: Vec<Challenges>,
        group_opening_points: Vec<RingOpeningPoint<F>>,
        group_ring_multiplier_points: Vec<RingMultiplierOpeningPoint<F>>,
        opening_batch: OpeningClaimsLayout,
        gamma: Vec<F>,
        row_coefficient_rings: RingVec<F>,
        rhs: RingVec<F>,
        v: RingVec<F>,
        role_dims: CommitmentRingDims,
    ) -> Result<Self, AkitaError> {
        opening_batch.check()?;
        let num_groups = opening_batch.num_groups();
        if group_challenges.len() != num_groups
            || group_opening_points.len() != num_groups
            || group_ring_multiplier_points.len() != num_groups
        {
            return Err(AkitaError::InvalidInput(
                "ring relation group carrier count does not match opening batch".to_string(),
            ));
        }
        for g in 0..num_groups {
            let group_layout = opening_batch.group_layout(g)?;
            let k_g = group_layout.num_polynomials();
            let challenges = &group_challenges[g];
            if challenges.num_claims() != k_g {
                return Err(AkitaError::InvalidInput(format!(
                    "ring relation group {g} challenges claim count {} does not match K_g={k_g}",
                    challenges.num_claims()
                )));
            }
            let num_live_blocks_g = challenges.num_live_blocks_per_claim();
            if group_opening_points[g].live_block_weights.len() != num_live_blocks_g {
                return Err(AkitaError::InvalidInput(format!(
                    "ring relation group {g} opening point block count does not match challenges"
                )));
            }
            if group_ring_multiplier_points[g].fold_len() != num_live_blocks_g {
                return Err(AkitaError::InvalidInput(format!(
                    "ring relation group {g} ring multiplier block count does not match challenges"
                )));
            }
        }
        if gamma.len() != opening_batch.num_total_polynomials()
            || row_coefficient_rings.count() != opening_batch.num_total_polynomials()
        {
            return Err(AkitaError::InvalidInput(
                "ring relation gamma/row coefficients length mismatch".to_string(),
            ));
        }
        if rhs.coeff_len() < role_dims.d_a() {
            return Err(AkitaError::InvalidInput(
                "ring relation rhs must contain at least the consistency row".to_string(),
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
            if !rhs.can_decode_vec(uniform) {
                return Err(AkitaError::InvalidSize {
                    expected: uniform,
                    actual: rhs.coeff_len(),
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
            group_challenges,
            group_opening_points,
            group_ring_multiplier_points,
            opening_batch,
            gamma,
            row_coefficient_rings,
            rhs,
            v,
            role_dims,
        })
    }

    /// Construct from typed kernel outputs at a ring-relation boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts<const D: usize>(
        group_challenges: Vec<Challenges>,
        group_opening_points: Vec<RingOpeningPoint<F>>,
        group_ring_multiplier_points: Vec<RingMultiplierOpeningPoint<F>>,
        opening_batch: OpeningClaimsLayout,
        gamma: Vec<F>,
        row_coefficient_rings: &[CyclotomicRing<F, D>],
        rhs: &[CyclotomicRing<F, D>],
        v: &[CyclotomicRing<F, D>],
    ) -> Result<Self, AkitaError> {
        Self::new(
            group_challenges,
            group_opening_points,
            group_ring_multiplier_points,
            opening_batch,
            gamma,
            RingVec::from_ring_elems(row_coefficient_rings),
            RingVec::from_ring_elems(rhs),
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

    pub fn opening_batch(&self) -> &OpeningClaimsLayout {
        &self.opening_batch
    }

    pub fn group_challenges(&self) -> &[Challenges] {
        &self.group_challenges
    }

    pub fn group_opening_point(&self, g: usize) -> Result<&RingOpeningPoint<F>, AkitaError> {
        self.group_opening_points.get(g).ok_or_else(|| {
            AkitaError::InvalidInput(format!(
                "ring relation opening point group index {g} out of range ({} groups)",
                self.group_opening_points.len()
            ))
        })
    }

    pub fn group_ring_multiplier_point(
        &self,
        g: usize,
    ) -> Result<&RingMultiplierOpeningPoint<F>, AkitaError> {
        self.group_ring_multiplier_points.get(g).ok_or_else(|| {
            AkitaError::InvalidInput(format!(
                "ring relation ring multiplier group index {g} out of range ({} groups)",
                self.group_ring_multiplier_points.len()
            ))
        })
    }

    pub fn gamma(&self) -> &[F] {
        &self.gamma
    }

    /// Public D-block rows in flat ring storage.
    pub fn v(&self) -> &RingVec<F> {
        &self.v
    }

    /// Relation RHS rows in flat ring storage.
    pub fn rhs(&self) -> &RingVec<F> {
        &self.rhs
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
            || !self.rhs.can_decode_vec(D)
            || !self.v.can_decode_vec(D)
        {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.rhs.coeff_len(),
            });
        }
        for point in &self.group_ring_multiplier_points {
            point.ensure_ring_dim::<D>()?;
        }
        Ok(())
    }

    /// Validate one role carrier against dispatch `D`.
    pub fn ensure_role_dim<const D: usize>(&self, role: RingRole) -> Result<(), AkitaError> {
        validate_role_dispatch::<D>(self.role_dims, role).map(|_| ())
    }

    /// Borrow `y` rows when all roles share one dimension.
    pub fn rhs_trusted<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        self.rhs.as_ring_slice::<D>()
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

    /// Validate the mandatory D-row payload shape.
    pub fn check_v_shape_for_level(&self, lp: &CommittedGroupParams) -> Result<(), AkitaError> {
        let expected = lp.open_commit_matrix.output_rank();
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
                "ring relation v rows do not match the open-commit matrix".to_string(),
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

    /// Resolve the canonical [`WitnessLayout`] for this level's witness,
    /// validating shape and (when supplied) capacity at the boundary.
    ///
    /// This is the **single source of truth** for witness column offsets shared
    /// by the distributed prover's emission and the verifier's row-MLE
    /// evaluation. `lp.witness_chunk.num_chunks = 1` yields one ownership unit
    /// with compact `[z | e | t]` ranges; `num_chunks = W` lays out `W`
    /// contiguous `[zᵢ | eᵢ | tᵢ]` ownership units (`zᵢ` replicated,
    /// `eᵢ`/`tᵢ` partitioned) followed by one shared `r` tail sized at the
    /// single-machine row count. Pass `witness_ring_len = Some(witness_len / D)` to
    /// enforce the no-panic capacity bound at this boundary.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] (never panics) for malformed
    /// ownership geometry, offset or length arithmetic overflow, or a layout
    /// whose shared `r` tail would exceed the committed witness capacity.
    pub fn segment_layout(
        &self,
        lp: &CommittedGroupParams,
        witness_ring_len: Option<usize>,
    ) -> Result<WitnessLayout, AkitaError> {
        lp.witness_chunk.validate()?;
        let num_chunks = lp.witness_chunk.num_chunks;
        let relation_rhs_layout =
            crate::proof::relation::relation_rhs_layout_for(lp, &self.opening_batch)?;
        let expected_rhs_coeff_len =
            crate::proof::relation::relation_rhs_coeff_len(self.role_dims, &relation_rhs_layout)?;
        if self.rhs.coeff_len() != expected_rhs_coeff_len {
            return Err(AkitaError::InvalidSetup(format!(
                "ring relation rhs coefficient length {} does not match per-role layout (expected {expected_rhs_coeff_len})",
                self.rhs.coeff_len()
            )));
        }
        // `EvaluationTrace` is a logical relation row used by Stage 2. It is
        // not materialized in the quotient witness's shared `r` tail.
        let relation_rhs_rows =
            crate::proof::relation::relation_rhs_row_count(&relation_rhs_layout);
        let r_levels = r_decomp_levels::<F>(lp.log_basis_open);
        let layout = WitnessLayout::new(
            lp,
            &self.opening_batch,
            num_chunks,
            relation_rhs_rows,
            r_levels,
        )?;
        if let Some(capacity) = witness_ring_len {
            if layout.total_len() > capacity {
                return Err(AkitaError::InvalidSetup(format!(
                    "resolved witness layout requires {} ring columns but only {capacity} are committed",
                    layout.total_len(),
                )));
            }
        }
        Ok(layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::PrecommittedLevelParams;
    use crate::{
        emit_witness_e_planes, emit_witness_r_planes, emit_witness_t_planes, emit_witness_z_planes,
        InnerCommitMatrixParams, OuterCommitMatrixParams, PolynomialGroupLayout,
    };
    use akita_challenges::{SparseChallenge, SparseChallengeConfig};
    use akita_field::Fp32;

    type F = Fp32<251>;
    const D: usize = 32;

    fn marker(index: usize) -> [i8; 2] {
        let value = (index % 100 + 1) as i8;
        [value, -value]
    }

    fn flatten_markers(markers: impl IntoIterator<Item = [i8; 2]>) -> Vec<i8> {
        markers.into_iter().flatten().collect()
    }

    fn certify_test_sis_bounds(lp: &mut CommittedGroupParams) {
        const BOUND: u128 = 1;
        lp.inner_commit_matrix = InnerCommitMatrixParams::new_unchecked(
            lp.inner_commit_matrix.security_policy(),
            lp.inner_commit_matrix.sis_table_key().table_digest,
            lp.inner_commit_matrix.sis_modulus_profile(),
            lp.inner_commit_matrix.output_rank(),
            lp.inner_commit_matrix.input_width(),
            BOUND,
            lp.d_a(),
        );
        lp.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
            lp.outer_commit_matrix.security_policy(),
            lp.outer_commit_matrix.sis_table_key().table_digest,
            lp.outer_commit_matrix.sis_modulus_profile(),
            lp.outer_commit_matrix.output_rank(),
            lp.outer_commit_matrix.input_width(),
            BOUND,
            lp.d_a(),
        );
    }

    fn fold_challenge_config() -> SparseChallengeConfig {
        SparseChallengeConfig::pm1_only(1)
    }

    fn opening_point(lp: &CommittedGroupParams) -> RingOpeningPoint<F> {
        RingOpeningPoint {
            position_weights: vec![F::zero(); lp.num_positions_per_block],
            live_block_weights: vec![F::zero(); lp.num_live_blocks],
        }
    }

    fn test_level_params() -> CommittedGroupParams {
        CommittedGroupParams::params_only(
            crate::SisModulusProfileId::Q32Offset99,
            D,
            2,
            1,
            1,
            1,
            fold_challenge_config(),
        )
        .with_decomp(4, 8, 1, 2, 2)
        .expect("test params")
    }

    fn test_challenges(lp: &CommittedGroupParams, num_claims: usize) -> Challenges {
        let total = lp.num_live_blocks * num_claims;
        Challenges::from_sparse(
            vec![
                SparseChallenge {
                    positions: vec![0],
                    coeffs: vec![1],
                };
                total
            ],
            lp.num_live_blocks,
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
            vec![test_challenges(&lp, opening_batch.num_total_polynomials())],
            vec![opening_point],
            vec![ring_multiplier_point],
            opening_batch,
            vec![F::one()],
            RingVec::from_ring_elems::<D>(&[CyclotomicRing::one()]),
            RingVec::from_ring_elems::<D>(&[]),
            RingVec::from_ring_elems::<D>(&[]),
            CommitmentRingDims::uniform(D),
        )
        .expect_err("empty rhs must be rejected");
        assert!(
            format!("{err:?}")
                .contains("ring relation rhs must contain at least the consistency row"),
            "unexpected error: {err:?}"
        );
    }

    fn chunk_test_level_params(block_index_bits: usize) -> CommittedGroupParams {
        // num_live_blocks = 2^block_index_bits, num_positions_per_block = 2^position_index_bits, single-tier.
        CommittedGroupParams::params_only(
            crate::SisModulusProfileId::Q32Offset99,
            D,
            2,
            1,
            1,
            1,
            fold_challenge_config(),
        )
        .with_decomp(4, 1usize << (2 + block_index_bits), 1, 2, 2)
        .expect("test params")
    }

    /// Build a minimal `WithDBlock` relation instance whose layout-relevant
    /// shape is `opening_batch.num_total_polynomials() = num_claims` and `y.len() =
    /// num_rows` (the only fields [`RingRelationInstance::segment_layout`] reads).
    fn build_instance(
        lp: &CommittedGroupParams,
        num_claims: usize,
        num_rows: usize,
    ) -> RingRelationInstance<F> {
        let opening_batch = OpeningClaimsLayout::new(8, num_claims).expect("opening batch");
        let opening_point = opening_point(lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        RingRelationInstance::<F>::new(
            vec![test_challenges(lp, num_claims)],
            vec![opening_point],
            vec![ring_multiplier_point],
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
        )
        .expect("lengths");

        let resolved = build_instance(&lp, num_claims, 4)
            .segment_layout(&lp, None)
            .expect("resolved layout");
        assert_eq!(resolved.num_chunks_for_group(0), 1);
        let unit = &resolved.units()[0];
        // Single-unit compact offsets: z first, then e, t, and the shared r tail.
        assert_eq!(unit.z_range().start, 0);
        assert_eq!(unit.e_range().start, lens.z_len);
        assert_eq!(unit.t_range().start, lens.z_len + lens.e_len);
        // The shared r tail follows the unit's compact z, e, and t ranges.
        assert_eq!(
            resolved.r_range().start,
            lens.z_len + lens.e_len + lens.t_len
        );
        assert_eq!(unit.global_block_start(), 0);
        assert_eq!(unit.num_live_blocks(), lp.num_live_blocks);
    }

    #[test]
    fn resolve_multi_chunk_offsets_contiguous_and_cover_blocks() {
        let num_claims = 2;
        for w in [1usize, 2, 4, 8] {
            let mut lp = chunk_test_level_params(3); // num_live_blocks = 8
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
            )
            .expect("lengths");
            let layout = build_instance(&lp, num_claims, 4)
                .segment_layout(&lp, None)
                .expect("layout");
            assert_eq!(layout.num_chunks_for_group(0), w);
            let blocks_per_chunk = lp.num_live_blocks / w;

            // Partitioned e/t lengths sum to the single-machine totals; z replicated.
            let e_sum: usize = layout.units().iter().map(|unit| unit.e_range().len()).sum();
            let t_sum: usize = layout.units().iter().map(|unit| unit.t_range().len()).sum();
            assert_eq!(e_sum, lens.e_len);
            assert_eq!(t_sum, lens.t_len);
            for unit in layout.units() {
                assert_eq!(unit.z_range().len(), lens.z_len);
            }

            // Ownership units are contiguous and z-first; the shared r tail follows all units.
            let stride = lens.z_len + lens.e_len / w + lens.t_len / w;
            for (j, unit) in layout.units().iter().enumerate() {
                let base = j * stride;
                assert_eq!(unit.z_range().start, base);
                assert_eq!(unit.e_range().start, base + lens.z_len);
                assert_eq!(unit.t_range().start, base + lens.z_len + lens.e_len / w);
                assert_eq!(unit.global_block_start(), j * blocks_per_chunk);
            }
            assert_eq!(layout.r_range().start, w * stride);
            // Block windows tile [0, num_live_blocks).
            assert_eq!(
                layout.units().last().unwrap().global_block_start() + blocks_per_chunk,
                lp.num_live_blocks
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

        // num_chunks = 16 exceeds num_live_blocks = 8.
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
        use crate::proof::relation::RelationRhsLayout;

        let lp = test_level_params();
        let opening_batch = OpeningClaimsLayout::new(2, 3).expect("valid batch");
        let opening_point = opening_point(&lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let relation_rhs_layout = RelationRhsLayout::uniform(
            lp.open_commit_matrix.output_rank(),
            lp.inner_commit_matrix.output_rank(),
            lp.outer_commit_matrix.output_rank(),
            0,
            1,
        );
        let relation_rhs_rows = lp.relation_matrix_row_count(1).expect("row count");
        let v_zeros = vec![CyclotomicRing::zero(); relation_rhs_layout.n_d];
        let y_zeros = vec![CyclotomicRing::zero(); relation_rhs_rows];
        let instance = RingRelationInstance::<F>::from_parts::<D>(
            vec![test_challenges(&lp, opening_batch.num_total_polynomials())],
            vec![opening_point],
            vec![ring_multiplier_point],
            opening_batch,
            vec![F::one(); 3],
            &[CyclotomicRing::one(); 3],
            &y_zeros,
            &v_zeros,
        )
        .expect("same-axis relation");

        let layout = instance.segment_layout(&lp, None).expect("layout");
        let unit = &layout.units()[0];
        let lens = ring_relation_segment_lengths::<F>(
            &lp,
            RingRelationOpeningCounts {
                num_claims: instance.opening_batch().num_total_polynomials(),
                num_t_vectors: instance.opening_batch().num_total_polynomials(),
            },
        )
        .expect("segment lengths");
        assert_eq!(layout.num_chunks_for_group(0), 1);
        assert_eq!(unit.z_range().start, 0);
        assert_eq!(unit.e_range().start, lens.z_len);
        assert_eq!(unit.t_range().start, lens.z_len + lens.e_len);
        assert_eq!(layout.r_range().start, lens.z_len + lens.e_len + lens.t_len);
        instance
            .check_v_shape_for_level(&lp)
            .expect("v rows match layout");
    }

    fn multi_group_one_three_fixture() -> (CommittedGroupParams, OpeningClaimsLayout) {
        use crate::schedule::PrecommittedGroupDescriptor;
        let lp = CommittedGroupParams::params_only(
            crate::SisModulusProfileId::Q128OffsetA7F7,
            D,
            3,
            2,
            4,
            3,
            fold_challenge_config(),
        )
        .with_decomp(4, 16, 2, 2, 2)
        .expect("multi-group main params");
        let mut precommit_lp = CommittedGroupParams::params_only(
            crate::SisModulusProfileId::Q128OffsetA7F7,
            D,
            3,
            2,
            4,
            3,
            fold_challenge_config(),
        )
        .with_decomp(4, 16, 2, 2, 2)
        .expect("multi-group precommit params");
        certify_test_sis_bounds(&mut precommit_lp);
        let precommit = PrecommittedLevelParams {
            layout: PrecommittedGroupDescriptor::from_params(
                PolynomialGroupLayout::new(4, 1),
                &precommit_lp,
            ),
            inner_commit_matrix: precommit_lp.inner_commit_matrix.clone(),
            outer_commit_matrix: precommit_lp.outer_commit_matrix.clone(),
            log_basis_open: precommit_lp.log_basis_open,
            num_digits_inner: precommit_lp.num_digits_inner,
            num_digits_outer: precommit_lp.num_digits_outer,
            num_digits_open: precommit_lp.num_digits_open,
            num_digits_fold_one: precommit_lp.num_digits_fold_one,
        };
        let mut multi_group_lp = lp;
        multi_group_lp.precommitted_groups = vec![precommit];
        let batch = OpeningClaimsLayout::from_root_groups(
            &[PolynomialGroupLayout::new(4, 1)],
            PolynomialGroupLayout::new(4, 1),
        )
        .expect("multi-group opening batch");
        (multi_group_lp, batch)
    }

    #[test]
    fn multi_group_segment_layout_total_matches_next_w_len() {
        let (lp, opening_batch) = multi_group_one_three_fixture();
        let relation_rhs_rows = lp
            .relation_matrix_row_count(opening_batch.num_groups())
            .expect("row count");
        let opening_point_pre = opening_point(&lp);
        let opening_point_final = opening_point(&lp);
        let ring_multiplier_pre = RingMultiplierOpeningPoint::from_base(&opening_point_pre);
        let ring_multiplier_final = RingMultiplierOpeningPoint::from_base(&opening_point_final);
        let instance = RingRelationInstance::<F>::new(
            vec![test_challenges(&lp, 1), test_challenges(&lp, 1)],
            vec![opening_point_pre, opening_point_final],
            vec![ring_multiplier_pre, ring_multiplier_final],
            opening_batch.clone(),
            vec![F::one(); opening_batch.num_total_polynomials()],
            RingVec::from_ring_elems::<D>(&vec![
                CyclotomicRing::one();
                opening_batch.num_total_polynomials()
            ]),
            RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::zero(); relation_rhs_rows]),
            RingVec::from_ring_elems::<D>(&vec![
                CyclotomicRing::zero();
                lp.open_commit_matrix.output_rank()
            ]),
            CommitmentRingDims::uniform(D),
        )
        .expect("multi-group instance");

        let layout = instance
            .segment_layout(&lp, None)
            .expect("multi-group segment layout");
        let num_groups = opening_batch.num_groups();
        // Group-major: one ownership unit per group, each holding a contiguous
        // `[z_g | e_g | t_g]` stride; only the shared `r` tail follows all units.
        assert_eq!(layout.units().len(), num_groups);
        let r_len_total = relation_rhs_rows * r_decomp_levels::<F>(lp.log_basis_open);

        let mut base = 0usize;
        for (p, unit) in layout.units().iter().enumerate() {
            let z_g = unit.z_range().len();
            let e_g = unit.e_range().len();
            let t_g = unit.t_range().len();
            assert_eq!(unit.z_range().start, base);
            assert_eq!(unit.e_range().start, base + z_g);
            assert_eq!(unit.t_range().start, base + z_g + e_g);
            if p + 1 == num_groups {
                assert_eq!(layout.r_range().start, base + z_g + e_g + t_g);
                assert_eq!(layout.r_range().len(), r_len_total);
            }
            base += z_g + e_g + t_g;
        }

        let witness_ring_cols = base + r_len_total;
        let expected_witness_len = lp
            .output_witness_len::<F>(&opening_batch)
            .expect("next w len");
        assert_eq!(witness_ring_cols * D, expected_witness_len);
    }

    #[test]
    fn multi_group_segment_layout_resolves_group_shard_product() {
        let (mut lp, opening_batch) = multi_group_one_three_fixture();
        lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
            num_chunks: 2,
            num_activated_levels: 1,
        };
        let relation_rhs_rows = lp
            .relation_matrix_row_count(opening_batch.num_groups())
            .expect("row count");
        let opening_point_pre = opening_point(&lp);
        let opening_point_final = opening_point(&lp);
        let ring_multiplier_pre = RingMultiplierOpeningPoint::from_base(&opening_point_pre);
        let ring_multiplier_final = RingMultiplierOpeningPoint::from_base(&opening_point_final);
        let gamma_len = opening_batch.num_total_polynomials();
        let instance = RingRelationInstance::<F>::new(
            vec![test_challenges(&lp, 1), test_challenges(&lp, 1)],
            vec![opening_point_pre, opening_point_final],
            vec![ring_multiplier_pre, ring_multiplier_final],
            opening_batch,
            vec![F::one(); gamma_len],
            RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::one(); gamma_len]),
            RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::zero(); relation_rhs_rows]),
            RingVec::from_ring_elems::<D>(&vec![
                CyclotomicRing::zero();
                lp.open_commit_matrix.output_rank()
            ]),
            CommitmentRingDims::uniform(D),
        )
        .expect("multi-group instance");
        let layout = instance
            .segment_layout(&lp, None)
            .expect("multi-group multi-chunk layout");
        assert_eq!(layout.units().len(), 4);
        assert_eq!(
            layout
                .units()
                .iter()
                .map(|unit| (unit.group_index(), unit.chunk_index()))
                .collect::<Vec<_>>(),
            vec![(1, 0), (1, 1), (0, 0), (0, 1)]
        );
        for group_index in [1, 0] {
            let units = layout.units_for_group(group_index).expect("group units");
            assert_eq!(units[0].global_block_range(), 0..2);
            assert_eq!(units[1].global_block_range(), 2..4);
            assert_eq!(units[0].t_range().end, units[1].z_range().start);
        }
        assert_eq!(
            layout.units().last().expect("last unit").t_range().end,
            layout.r_range().start
        );

        // Independent dense emitter oracle: each physical range must contain
        // the corresponding semantic source planes in digit-innermost order.
        let mut emitted = vec![0i8; layout.total_len() * 2];
        for group_index in [1, 0] {
            let params = lp
                .group_params(instance.opening_batch(), group_index)
                .expect("group params");
            let num_claims = instance
                .opening_batch()
                .group_layout(group_index)
                .expect("group layout")
                .num_polynomials();
            let num_live_blocks = params.num_live_blocks();
            let depth_witness = params.num_digits_inner();
            let depth_commit = params.num_digits_outer();
            let depth_open = params.num_digits_open();
            let n_a = params.a_rows_len();
            let e_source = (0..num_claims * num_live_blocks * depth_open)
                .map(|index| marker(100 * group_index + index))
                .collect::<Vec<_>>();
            let t_source = (0..num_claims * num_live_blocks * n_a * depth_commit)
                .map(|index| marker(300 * group_index + index))
                .collect::<Vec<_>>();
            emit_witness_e_planes(
                &mut emitted,
                &layout,
                group_index,
                num_claims,
                depth_open,
                &e_source,
                num_live_blocks,
            )
            .expect("emit E");
            emit_witness_t_planes(
                &mut emitted,
                &layout,
                group_index,
                num_claims,
                n_a,
                depth_commit,
                &t_source,
                num_live_blocks,
            )
            .expect("emit T");

            let depth_fold = lp
                .num_digits_fold_for_params(params, num_claims, lp.field_bits_for_cache())
                .expect("fold depth");
            for unit in layout.units_for_group(group_index).expect("units") {
                let z_source = (0..params.num_positions_per_block() * depth_witness * depth_fold)
                    .map(|index| marker(500 * group_index + 100 * unit.chunk_index() + index))
                    .collect::<Vec<_>>();
                emit_witness_z_planes(
                    &mut emitted,
                    unit,
                    params.num_positions_per_block(),
                    depth_witness,
                    depth_fold,
                    &z_source,
                )
                .expect("emit Z");
                let z_range = unit.z_range();
                assert_eq!(
                    &emitted[z_range.start * 2..z_range.end * 2],
                    flatten_markers(z_source).as_slice()
                );

                let mut expected_e = Vec::new();
                for claim in 0..num_claims {
                    for block_idx in unit.global_block_range() {
                        for digit in 0..depth_open {
                            expected_e.push(
                                e_source
                                    [(claim * num_live_blocks + block_idx) * depth_open + digit],
                            );
                        }
                    }
                }
                let e_range = unit.e_range();
                assert_eq!(
                    &emitted[e_range.start * 2..e_range.end * 2],
                    flatten_markers(expected_e).as_slice()
                );

                let mut expected_t = Vec::new();
                for claim in 0..num_claims {
                    for block_idx in unit.global_block_range() {
                        for a_row in 0..n_a {
                            for digit in 0..depth_open {
                                expected_t.push(
                                    t_source[((claim * num_live_blocks + block_idx) * n_a + a_row)
                                        * depth_open
                                        + digit],
                                );
                            }
                        }
                    }
                }
                let t_range = unit.t_range();
                assert_eq!(
                    &emitted[t_range.start * 2..t_range.end * 2],
                    flatten_markers(expected_t).as_slice()
                );
            }
        }
        let quotient_depth = r_decomp_levels::<F>(lp.log_basis_open);
        let r_source = (0..layout.r_range().len())
            .map(|index| marker(900 + index))
            .collect::<Vec<_>>();
        emit_witness_r_planes(&mut emitted, &layout, quotient_depth, &r_source).expect("emit R");
        let r_range = layout.r_range();
        assert_eq!(
            &emitted[r_range.start * 2..r_range.end * 2],
            flatten_markers(r_source).as_slice()
        );
    }
}
