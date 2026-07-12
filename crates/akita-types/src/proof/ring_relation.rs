//! Shared public statement for the per-fold negacyclic-ring relation `M * z = y + (X^D + 1) * r`.

use super::OpeningClaimsLayout;
use crate::layout::relation::RelationLayout;
use crate::layout::{CommitmentRingDims, RingRole};
use crate::validate_role_dispatch;
use crate::FpExtEncoding;
use crate::{
    embed_ring_subfield_scalar, LevelParams, RelationMatrixRowLayout, RingMultiplierOpeningPoint,
    RingOpeningPoint, RingVec,
};
use akita_algebra::CyclotomicRing;
use akita_challenges::Challenges;
use akita_field::{AkitaError, FieldCore};
use akita_field::{CanonicalField, ExtField, FromPrimitiveInt};

/// Public statement of the negacyclic-ring matrix relation at one fold level.
///
/// Ring dimension is stored at runtime; hot paths inside `dispatch_ring_dim`
/// closures borrow typed ring rows via [`Self::rhs_trusted`], [`Self::v_trusted`],
/// and [`Self::row_coefficient_rings_trusted`].
#[derive(Debug, Clone)]
pub struct RingRelationInstance<F: FieldCore> {
    relation_layout: RelationLayout,
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
        lp: &LevelParams,
        relation_matrix_row_layout: RelationMatrixRowLayout,
        group_challenges: Vec<Challenges>,
        group_opening_points: Vec<RingOpeningPoint<F>>,
        group_ring_multiplier_points: Vec<RingMultiplierOpeningPoint<F>>,
        opening_batch: OpeningClaimsLayout,
        gamma: Vec<F>,
        row_coefficient_rings: RingVec<F>,
        commitment_rows: RingVec<F>,
        v: RingVec<F>,
    ) -> Result<Self, AkitaError> {
        opening_batch.check()?;
        let relation_layout = RelationLayout::from_authenticated_statement(
            lp,
            &opening_batch,
            relation_matrix_row_layout,
            F::modulus_bits(),
        )?;
        let role_dims = CommitmentRingDims {
            inner: lp.a_key.sis_table_key().ring_dimension as usize,
            outer: lp.b_key.sis_table_key().ring_dimension as usize,
            opening: lp.d_key.sis_table_key().ring_dimension as usize,
        };
        let rhs = crate::proof::relation::assemble_relation_rhs(
            relation_layout.row_plan(),
            &v,
            &commitment_rows,
        )?;
        let num_groups = opening_batch.num_groups();
        if group_challenges.len() != num_groups
            || group_opening_points.len() != num_groups
            || group_ring_multiplier_points.len() != num_groups
        {
            return Err(AkitaError::InvalidInput(
                "ring relation group carrier count does not match opening batch".to_string(),
            ));
        }
        for (g, ((challenges, opening_point), multiplier_point)) in group_challenges
            .iter()
            .zip(&group_opening_points)
            .zip(&group_ring_multiplier_points)
            .enumerate()
        {
            let group_layout = opening_batch.group_layout(g)?;
            let k_g = group_layout.num_polynomials();
            if challenges.num_claims() != k_g {
                return Err(AkitaError::InvalidInput(format!(
                    "ring relation group {g} challenges claim count {} does not match K_g={k_g}",
                    challenges.num_claims()
                )));
            }
            let num_blocks_g = challenges.num_blocks_per_claim();
            if opening_point.b.len() != num_blocks_g {
                return Err(AkitaError::InvalidInput(format!(
                    "ring relation group {g} opening point block count does not match challenges"
                )));
            }
            if multiplier_point.b_len() != num_blocks_g {
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
            if gamma.get(idx) != chunk.first() {
                return Err(AkitaError::InvalidInput(
                    "ring relation gamma does not match row coefficient rings".to_string(),
                ));
            }
        }
        Ok(Self {
            relation_layout,
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
        lp: &LevelParams,
        relation_matrix_row_layout: RelationMatrixRowLayout,
        group_challenges: Vec<Challenges>,
        group_opening_points: Vec<RingOpeningPoint<F>>,
        group_ring_multiplier_points: Vec<RingMultiplierOpeningPoint<F>>,
        opening_batch: OpeningClaimsLayout,
        gamma: Vec<F>,
        row_coefficient_rings: &[CyclotomicRing<F, D>],
        commitment_rows: &[CyclotomicRing<F, D>],
        v: &[CyclotomicRing<F, D>],
    ) -> Result<Self, AkitaError> {
        Self::new(
            lp,
            relation_matrix_row_layout,
            group_challenges,
            group_opening_points,
            group_ring_multiplier_points,
            opening_batch,
            gamma,
            RingVec::from_ring_elems(row_coefficient_rings),
            RingVec::from_ring_elems(commitment_rows),
            RingVec::from_ring_elems(v),
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

    pub fn relation_matrix_row_layout(&self) -> RelationMatrixRowLayout {
        if self
            .relation_layout
            .row_plan()
            .families()
            .iter()
            .any(|family| matches!(family.id(), crate::RelationRowId::D))
        {
            RelationMatrixRowLayout::WithDBlock
        } else {
            RelationMatrixRowLayout::WithoutDBlock
        }
    }

    pub fn relation_layout(&self) -> &RelationLayout {
        &self.relation_layout
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
        self.relation_layout
            .row_plan()
            .validate_uniform_execution(D)?;
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

    /// Validate layout-dependent D-row payload shape.
    pub fn check_v_shape_for_level(&self, _lp: &LevelParams) -> Result<(), AkitaError> {
        let expected = self
            .relation_layout
            .row_plan()
            .families()
            .iter()
            .find(|family| matches!(family.id(), crate::RelationRowId::D))
            .map_or(0, |family| family.rows().len());
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
                "ring relation v rows do not match relation-matrix row layout".to_string(),
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
            gamma.push(*ring.coefficients().first().ok_or_else(|| {
                AkitaError::InvalidInput("row coefficient ring has no coefficients".into())
            })?);
            row_coefficient_rings.push(ring);
        }
        Ok((gamma, RingVec::from_ring_elems(&row_coefficient_rings)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::PrecommittedLevelParams;
    use crate::r_decomp_levels;
    use crate::PolynomialGroupLayout;
    use akita_challenges::{SparseChallenge, SparseChallengeConfig};
    use akita_field::Fp32;

    type F = Fp32<251>;
    const D: usize = 32;

    fn fold_challenge_config() -> SparseChallengeConfig {
        SparseChallengeConfig::pm1_only(1)
    }

    fn opening_point(lp: &LevelParams) -> RingOpeningPoint<F> {
        RingOpeningPoint {
            a: vec![F::zero(); lp.block_len],
            b: vec![F::zero(); lp.num_blocks],
        }
    }

    fn test_level_params() -> LevelParams {
        LevelParams::params_only(
            crate::SisModulusFamily::Q32,
            D,
            2,
            1,
            1,
            1,
            fold_challenge_config(),
        )
        .with_decomp(2, 1, 1, 2, 0)
        .map(|mut lp| {
            lp.field_bits_hint = F::modulus_bits();
            lp
        })
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
            &lp,
            RelationMatrixRowLayout::WithoutDBlock,
            vec![test_challenges(&lp, opening_batch.num_total_polynomials())],
            vec![opening_point],
            vec![ring_multiplier_point],
            opening_batch,
            vec![F::one()],
            RingVec::from_ring_elems::<D>(&[CyclotomicRing::one()]),
            RingVec::from_ring_elems::<D>(&[]),
            RingVec::from_ring_elems::<D>(&[]),
        )
        .expect_err("empty rhs must be rejected");
        assert!(matches!(
            err,
            AkitaError::InvalidProof | AkitaError::InvalidSize { .. }
        ));
    }

    fn chunk_test_level_params(r_vars: usize) -> LevelParams {
        // num_blocks = 2^r_vars, block_len = 2^m_vars, single-tier.
        LevelParams::params_only(
            crate::SisModulusFamily::Q32,
            D,
            2,
            1,
            1,
            1,
            fold_challenge_config(),
        )
        .with_decomp(2, r_vars, 1, 2, 0)
        .map(|mut lp| {
            lp.field_bits_hint = F::modulus_bits();
            lp
        })
        .expect("test params")
    }

    /// Build a minimal `WithDBlock` relation instance.
    fn build_instance(
        lp: &LevelParams,
        num_claims: usize,
        _num_rows: usize,
    ) -> RingRelationInstance<F> {
        try_build_instance(lp, num_claims, _num_rows).expect("instance")
    }

    fn try_build_instance(
        lp: &LevelParams,
        num_claims: usize,
        _num_rows: usize,
    ) -> Result<RingRelationInstance<F>, AkitaError> {
        let opening_batch = OpeningClaimsLayout::new(8, num_claims).expect("opening batch");
        let opening_point = opening_point(lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        RingRelationInstance::<F>::new(
            lp,
            RelationMatrixRowLayout::WithDBlock,
            vec![test_challenges(lp, num_claims)],
            vec![opening_point],
            vec![ring_multiplier_point],
            opening_batch,
            vec![F::one(); num_claims],
            RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::one(); num_claims]),
            RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::zero(); lp.b_key.row_len()]),
            RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::zero(); lp.d_key.row_len()]),
        )
    }

    #[test]
    fn resolve_single_chunk_matches_legacy_offsets() {
        let lp = chunk_test_level_params(1);
        assert_eq!(lp.witness_chunk.num_chunks, 1);
        let num_claims = 3;
        let instance = build_instance(&lp, num_claims, 4);
        let resolved = instance
            .relation_layout()
            .witness_layout(None)
            .expect("resolved layout");
        assert_eq!(resolved.num_chunks(), 1);
        let lens = resolved.chunk_lengths[0];
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
            let instance = build_instance(&lp, num_claims, 4);
            let layout = instance
                .relation_layout()
                .witness_layout(None)
                .expect("layout");
            assert_eq!(layout.num_chunks(), w);
            assert_eq!(layout.blocks_per_chunk, lp.num_blocks / w);
            let z_len = layout.chunk_lengths[0].z_len;
            let e_len: usize = layout.chunk_lengths.iter().map(|l| l.e_len).sum();
            let t_len: usize = layout.chunk_lengths.iter().map(|l| l.t_len).sum();

            // Partitioned e/t lengths sum to the single-machine totals; z replicated.
            let e_sum: usize = layout.chunk_lengths.iter().map(|l| l.e_len).sum();
            let t_sum: usize = layout.chunk_lengths.iter().map(|l| l.t_len).sum();
            assert_eq!(e_sum, e_len);
            assert_eq!(t_sum, t_len);
            for l in &layout.chunk_lengths {
                assert_eq!(l.z_len, z_len);
            }

            // Offsets are contiguous z-first per chunk; only the last chunk has r̂.
            let stride = z_len + e_len / w + t_len / w;
            for (j, chunk) in layout.chunks.iter().enumerate() {
                let base = j * stride;
                assert_eq!(chunk.offset_z, base);
                assert_eq!(chunk.offset_e, base + z_len);
                assert_eq!(chunk.offset_t, base + z_len + e_len / w);
                assert_eq!(chunk.global_block_base, j * (lp.num_blocks / w));
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
        assert!(try_build_instance(&lp, num_claims, 4).is_err());

        // num_chunks = 16 exceeds num_blocks = 8.
        let mut lp = chunk_test_level_params(3);
        lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
            num_chunks: 16,
            num_activated_levels: 1,
        };
        assert!(try_build_instance(&lp, num_claims, 4).is_err());
    }

    #[test]
    fn resolve_rejects_capacity_overflow() {
        let num_claims = 2;
        let lp = chunk_test_level_params(3);
        // A witness ring capacity of 1 is far smaller than offset_r + r_len.
        assert!(
            build_instance(&lp, num_claims, 4)
                .relation_layout()
                .witness_layout(Some(1))
                .is_err(),
            "tiny witness capacity must be rejected"
        );
        // A generous capacity passes.
        build_instance(&lp, num_claims, 4)
            .relation_layout()
            .witness_layout(Some(1 << 20))
            .expect("ample capacity");
    }

    #[test]
    fn relation_segment_layout_uses_same_axis_contract() {
        let lp = test_level_params();
        let opening_batch = OpeningClaimsLayout::new(2, 3).expect("valid batch");
        let opening_point = opening_point(&lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let v_zeros = vec![CyclotomicRing::zero(); lp.d_key.row_len()];
        let commitment_zeros = vec![CyclotomicRing::zero(); lp.b_key.row_len()];
        let instance = RingRelationInstance::<F>::from_parts::<D>(
            &lp,
            RelationMatrixRowLayout::WithDBlock,
            vec![test_challenges(&lp, opening_batch.num_total_polynomials())],
            vec![opening_point],
            vec![ring_multiplier_point],
            opening_batch,
            vec![F::one(); 3],
            &[CyclotomicRing::one(); 3],
            &commitment_zeros,
            &v_zeros,
        )
        .expect("same-axis relation");

        let layout = instance
            .relation_layout()
            .witness_layout(None)
            .expect("layout");
        let chunk = layout.chunks[0];
        let lens = layout.chunk_lengths[0];
        assert_eq!(layout.num_chunks(), 1);
        assert_eq!(chunk.offset_z, 0);
        assert_eq!(chunk.offset_e, lens.z_len);
        assert_eq!(chunk.offset_t, lens.z_len + lens.e_len);
        assert_eq!(chunk.offset_r, Some(lens.z_len + lens.e_len + lens.t_len));
        instance
            .check_v_shape_for_level(&lp)
            .expect("v rows match layout");
    }

    fn multi_group_one_three_fixture() -> (LevelParams, OpeningClaimsLayout) {
        use crate::schedule::PrecommittedGroupParams;
        let lp = LevelParams::params_only(
            crate::SisModulusFamily::Q128,
            D,
            3,
            2,
            4,
            3,
            fold_challenge_config(),
        )
        .with_decomp(2, 2, 2, 2, 0)
        .map(|mut lp| {
            lp.field_bits_hint = F::modulus_bits();
            lp
        })
        .expect("multi-group main params");
        let precommit_lp = LevelParams::params_only(
            crate::SisModulusFamily::Q128,
            D,
            3,
            2,
            4,
            3,
            fold_challenge_config(),
        )
        .with_decomp(2, 2, 2, 2, 0)
        .map(|mut lp| {
            lp.field_bits_hint = F::modulus_bits();
            lp
        })
        .expect("multi-group precommit params");
        let precommit = PrecommittedLevelParams {
            layout: PrecommittedGroupParams::from_params(
                PolynomialGroupLayout::new(4, 3),
                &precommit_lp,
            ),
            a_key: precommit_lp.a_key.clone(),
            b_key: precommit_lp.b_key.clone(),
            num_blocks: precommit_lp.num_blocks,
            block_len: precommit_lp.block_len,
            num_digits_commit: precommit_lp.num_digits_commit,
            num_digits_open: precommit_lp.num_digits_open,
            num_digits_fold_one: precommit_lp.num_digits_fold_one,
        };
        let mut multi_group_lp = lp;
        multi_group_lp.precommitted_groups = vec![precommit];
        let batch = OpeningClaimsLayout::from_root_groups(
            &[PolynomialGroupLayout::new(4, 3)],
            PolynomialGroupLayout::new(4, 1),
        )
        .expect("multi-group opening batch");
        (multi_group_lp, batch)
    }

    #[test]
    fn multi_group_segment_layout_total_matches_root_next_w_len() {
        let (lp, opening_batch) = multi_group_one_three_fixture();
        let commitment_rows = lp.b_key.row_len()
            + lp.precommitted_groups
                .iter()
                .map(|group| group.b_key.row_len())
                .sum::<usize>();
        let opening_point_pre = opening_point(&lp);
        let opening_point_final = opening_point(&lp);
        let ring_multiplier_pre = RingMultiplierOpeningPoint::from_base(&opening_point_pre);
        let ring_multiplier_final = RingMultiplierOpeningPoint::from_base(&opening_point_final);
        let instance = RingRelationInstance::<F>::new(
            &lp,
            RelationMatrixRowLayout::WithDBlock,
            vec![test_challenges(&lp, 3), test_challenges(&lp, 1)],
            vec![opening_point_pre, opening_point_final],
            vec![ring_multiplier_pre, ring_multiplier_final],
            opening_batch.clone(),
            vec![F::one(); 4],
            RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::one(); 4]),
            RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::zero(); commitment_rows]),
            RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::zero(); lp.d_key.row_len()]),
        )
        .expect("multi-group instance");

        let layout = instance
            .relation_layout()
            .witness_layout(None)
            .expect("multi-group segment layout");
        let plan = instance.relation_layout().row_plan();
        assert_eq!(
            plan.families()
                .iter()
                .map(|family| family.id())
                .collect::<Vec<_>>(),
            vec![
                crate::RelationRowId::Consistency,
                crate::RelationRowId::A {
                    group: crate::RelationGroupId::Current
                },
                crate::RelationRowId::B {
                    group: crate::RelationGroupId::Current
                },
                crate::RelationRowId::A {
                    group: crate::RelationGroupId::Precommitted { index: 0 }
                },
                crate::RelationRowId::B {
                    group: crate::RelationGroupId::Precommitted { index: 0 }
                },
                crate::RelationRowId::D,
            ]
        );
        let expected_geometry = [(0, 1), (1, 2), (3, 4), (7, 2), (9, 4), (13, 3)];
        for (family, (start, len)) in plan.families().iter().zip(expected_geometry) {
            assert_eq!(family.rows().start(), start);
            assert_eq!(family.rows().len(), len);
            assert_eq!(family.native_ring_dim(), D);
        }
        assert_eq!(plan.trace_row(), 16);
        assert_eq!(plan.padded_row_count(), 32);
        let num_groups = layout.num_chunks();
        // Group-major: one chunk per group, each holding a contiguous
        // `[z_g | e_g | t_g]` stride; only the last chunk carries the single
        // shared `r` tail.
        assert_eq!(layout.num_chunks(), num_groups);
        let z_total: usize = layout.chunk_lengths.iter().map(|lens| lens.z_len).sum();
        let e_total: usize = layout.chunk_lengths.iter().map(|lens| lens.e_len).sum();
        let t_total: usize = layout.chunk_lengths.iter().map(|lens| lens.t_len).sum();
        let r_len_total =
            instance.relation_layout().row_plan().trace_row() * r_decomp_levels::<F>(lp.log_basis);

        let mut base = 0usize;
        for (p, (chunk, lengths)) in layout
            .chunks
            .iter()
            .zip(layout.chunk_lengths.iter())
            .enumerate()
        {
            let z_g = lengths.z_len;
            let e_g = lengths.e_len;
            let t_g = lengths.t_len;
            assert_eq!(chunk.offset_z, base);
            assert_eq!(chunk.offset_e, base + z_g);
            assert_eq!(chunk.offset_t, base + z_g + e_g);
            if p + 1 == num_groups {
                assert_eq!(chunk.offset_r, Some(base + z_g + e_g + t_g));
                assert_eq!(lengths.r_len, Some(r_len_total));
            } else {
                assert_eq!(chunk.offset_r, None);
                assert_eq!(lengths.r_len, None);
            }
            base += z_g + e_g + t_g;
        }
        assert_eq!(base, z_total + e_total + t_total);

        let witness_ring_cols = z_total + e_total + t_total + r_len_total;
        let expected_w_len = layout.ring_len().expect("root next w len") * D;
        assert_eq!(witness_ring_cols * D, expected_w_len);
    }

    #[test]
    fn uniform_execution_rejects_divisible_mixed_precommitted_dimension() {
        use crate::schedule::PrecommittedGroupParams;
        const CURRENT_D: usize = 64;
        const PRE_D: usize = 32;
        let make = |d| {
            LevelParams::params_only(
                crate::SisModulusFamily::Q32,
                d,
                3,
                2,
                4,
                3,
                fold_challenge_config(),
            )
            .with_decomp(2, 2, 2, 2, 0)
            .map(|mut lp| {
                lp.field_bits_hint = F::modulus_bits();
                lp
            })
            .unwrap()
        };
        let mut lp = make(CURRENT_D);
        let pre_lp = make(PRE_D);
        lp.precommitted_groups.push(PrecommittedLevelParams {
            layout: PrecommittedGroupParams::from_params(PolynomialGroupLayout::new(4, 1), &pre_lp),
            a_key: pre_lp.a_key.clone(),
            b_key: pre_lp.b_key.clone(),
            num_blocks: pre_lp.num_blocks,
            block_len: pre_lp.block_len,
            num_digits_commit: pre_lp.num_digits_commit,
            num_digits_open: pre_lp.num_digits_open,
            num_digits_fold_one: pre_lp.num_digits_fold_one,
        });
        let opening = OpeningClaimsLayout::from_root_groups(
            &[PolynomialGroupLayout::new(4, 1)],
            PolynomialGroupLayout::new(4, 1),
        )
        .unwrap();
        let point = opening_point(&lp);
        let multiplier = RingMultiplierOpeningPoint::from_base(&point);
        let commitment_coeffs =
            vec![F::zero(); lp.b_key.row_len() * CURRENT_D + pre_lp.b_key.row_len() * PRE_D];
        assert!(commitment_coeffs.len().is_multiple_of(CURRENT_D));
        let instance = RingRelationInstance::<F>::new(
            &lp,
            RelationMatrixRowLayout::WithDBlock,
            vec![test_challenges(&lp, 1), test_challenges(&lp, 1)],
            vec![point.clone(), point],
            vec![multiplier.clone(), multiplier],
            opening,
            vec![F::one(); 2],
            RingVec::from_ring_elems::<CURRENT_D>(&[CyclotomicRing::one(); 2]),
            RingVec::from_coeffs(commitment_coeffs),
            RingVec::from_ring_elems::<CURRENT_D>(&vec![
                CyclotomicRing::zero();
                lp.d_key.row_len()
            ]),
        )
        .unwrap();
        assert!(instance.ensure_ring_dim::<CURRENT_D>().is_err());
    }

    #[test]
    fn multi_group_without_d_omits_only_opening_family() {
        let (lp, opening) = multi_group_one_three_fixture();
        let with_d = crate::RelationRowPlan::compile_base(
            &lp,
            &opening,
            RelationMatrixRowLayout::WithDBlock,
        )
        .unwrap();
        let without_d = crate::RelationRowPlan::compile_base(
            &lp,
            &opening,
            RelationMatrixRowLayout::WithoutDBlock,
        )
        .unwrap();
        assert!(without_d.family(crate::RelationRowId::D).is_err());
        assert_eq!(
            with_d.trace_row() - without_d.trace_row(),
            lp.d_key.row_len()
        );
        assert_eq!(without_d.trace_row(), 13);
        assert_eq!(without_d.padded_row_count(), 16);
        assert_eq!(
            &with_d.families()[..with_d.families().len() - 1],
            without_d.families()
        );
    }

    #[test]
    fn multi_group_segment_layout_rejects_multi_chunk() {
        let (mut lp, opening_batch) = multi_group_one_three_fixture();
        lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
            num_chunks: 2,
            num_activated_levels: 1,
        };
        let commitment_rows = lp.b_key.row_len()
            + lp.precommitted_groups
                .iter()
                .map(|group| group.b_key.row_len())
                .sum::<usize>();
        let opening_point_pre = opening_point(&lp);
        let opening_point_final = opening_point(&lp);
        let ring_multiplier_pre = RingMultiplierOpeningPoint::from_base(&opening_point_pre);
        let ring_multiplier_final = RingMultiplierOpeningPoint::from_base(&opening_point_final);
        let err = RingRelationInstance::<F>::new(
            &lp,
            RelationMatrixRowLayout::WithDBlock,
            vec![test_challenges(&lp, 3), test_challenges(&lp, 1)],
            vec![opening_point_pre, opening_point_final],
            vec![ring_multiplier_pre, ring_multiplier_final],
            opening_batch,
            vec![F::one(); 4],
            RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::one(); 4]),
            RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::zero(); commitment_rows]),
            RingVec::from_ring_elems::<D>(&vec![CyclotomicRing::zero(); lp.d_key.row_len()]),
        )
        .expect_err("multi-group multi-chunk must reject");
        assert!(
            format!("{err:?}").contains(crate::MULTI_GROUP_ROOT_MULTI_CHUNK_UNSUPPORTED),
            "unexpected error: {err:?}"
        );
    }
}
