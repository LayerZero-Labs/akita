//! Shared public statement for the per-fold negacyclic-ring relation `M * z = y + (X^D + 1) * r`.

use super::OpeningBatchShape;
use crate::witness::{
    witness_chunk_lengths, WitnessChunkLayout, WitnessChunkLengths, WitnessLayout,
};
use crate::FpExtEncoding;
use crate::{
    embed_ring_subfield_scalar, LevelParams, MRowLayout, RingMultiplierOpeningPoint,
    RingOpeningPoint,
};
use akita_algebra::CyclotomicRing;
use akita_challenges::Challenges;
use akita_field::{AkitaError, FieldCore};
use akita_field::{CanonicalField, ExtField, FromPrimitiveInt};

/// Witness-column segment offsets for ring-switch evaluation.
///
/// Produced only by [`RingRelationInstance::segment_layout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingRelationSegmentLayout {
    pub offset_e: usize,
    pub offset_t: usize,
    /// Witness column offset of the tiered `û_concat` segment (flat, contiguous,
    /// immediately after `t̂`). Equals `offset_t + t_len`; for single-tier
    /// levels the segment is empty (`u_len == 0`) but the offset is still valid.
    pub offset_u: usize,
    pub offset_z: usize,
    pub offset_r: usize,
}

/// Ring-column counts per witness segment in emission order (`z ‖ e ‖ t ‖ …`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingRelationSegmentLengths {
    pub z_len: usize,
    pub e_len: usize,
    pub t_len: usize,
    pub u_len: usize,
}

/// Opening-batch counts that determine witness segment widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingRelationOpeningCounts {
    pub num_claims: usize,
    pub num_t_vectors: usize,
}

/// Witness segment lengths shared by prover emission, layout offsets, and M-table sizing.
pub fn ring_relation_segment_lengths<F: FieldCore + CanonicalField, const D: usize>(
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
    let depth_fold = lp.num_digits_fold(num_t_vectors, F::modulus_bits())?;
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

    let u_len = lp.u_concat_ring_len_per_group();

    Ok(RingRelationSegmentLengths {
        z_len,
        e_len,
        t_len,
        u_len,
    })
}

/// Public statement of the negacyclic-ring matrix relation at one fold level.
#[derive(Debug, Clone)]
pub struct RingRelationInstance<F: FieldCore, const D: usize> {
    m_row_layout: MRowLayout,
    pub challenges: Challenges,
    opening_point: RingOpeningPoint<F>,
    ring_multiplier_point: RingMultiplierOpeningPoint<F, D>,
    opening_batch: OpeningBatchShape,
    gamma: Vec<F>,
    row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
    y: Vec<CyclotomicRing<F, D>>,
    pub v: Vec<CyclotomicRing<F, D>>,
}

impl<F: FieldCore + CanonicalField, const D: usize> RingRelationInstance<F, D> {
    /// Construct a validated ring-relation statement from already-sampled inputs.
    ///
    /// Does not sample from the transcript; callers must absorb/sample before calling.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m_row_layout: MRowLayout,
        challenges: Challenges,
        opening_point: RingOpeningPoint<F>,
        ring_multiplier_point: RingMultiplierOpeningPoint<F, D>,
        opening_batch: OpeningBatchShape,
        gamma: Vec<F>,
        row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
        y: Vec<CyclotomicRing<F, D>>,
        v: Vec<CyclotomicRing<F, D>>,
    ) -> Result<Self, AkitaError> {
        opening_batch.check()?;
        if gamma.len() != opening_batch.num_polynomials()
            || row_coefficient_rings.len() != opening_batch.num_polynomials()
        {
            return Err(AkitaError::InvalidInput(
                "ring relation gamma/row coefficients length mismatch".to_string(),
            ));
        }
        if y.is_empty() {
            return Err(AkitaError::InvalidInput(
                "ring relation y must contain at least the consistency row".to_string(),
            ));
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
        })
    }

    pub fn m_row_layout(&self) -> MRowLayout {
        self.m_row_layout
    }

    pub fn opening_batch(&self) -> &OpeningBatchShape {
        &self.opening_batch
    }

    pub fn opening_point(&self) -> &RingOpeningPoint<F> {
        &self.opening_point
    }

    pub fn ring_multiplier_point(&self) -> &RingMultiplierOpeningPoint<F, D> {
        &self.ring_multiplier_point
    }

    pub fn gamma(&self) -> &[F] {
        &self.gamma
    }

    pub fn row_coefficient_rings(&self) -> &[CyclotomicRing<F, D>] {
        &self.row_coefficient_rings
    }

    pub fn y(&self) -> &[CyclotomicRing<F, D>] {
        &self.y
    }

    /// Validate layout-dependent D-row payload shape.
    pub fn check_v_shape_for_level(&self, lp: &LevelParams) -> Result<(), AkitaError> {
        let expected = match self.m_row_layout {
            MRowLayout::WithDBlock => lp.d_key.row_len(),
            MRowLayout::WithoutDBlock => 0,
        };
        if self.v.len() != expected {
            return Err(AkitaError::InvalidInput(
                "ring relation v rows do not match M-row layout".to_string(),
            ));
        }
        Ok(())
    }

    /// Build base-field `gamma` and embedded row rings from transcript-sampled coefficients.
    pub fn gamma_and_row_rings_from_coefficients<L>(
        row_coefficients: &[L],
    ) -> Result<(Vec<F>, Vec<CyclotomicRing<F, D>>), AkitaError>
    where
        F: FromPrimitiveInt,
        L: FpExtEncoding<F> + ExtField<F>,
    {
        let mut gamma = Vec::with_capacity(row_coefficients.len());
        let mut row_coefficient_rings = Vec::with_capacity(row_coefficients.len());
        for &coefficient in row_coefficients {
            let ring =
                embed_ring_subfield_scalar::<F, L, D>(coefficient, AkitaError::InvalidProof)?;
            gamma.push(ring.coefficients()[0]);
            row_coefficient_rings.push(ring);
        }
        Ok((gamma, row_coefficient_rings))
    }

    /// Witness-column segment layout shared by prover and verifier ring-switch paths.
    pub fn segment_layout(
        &self,
        lp: &LevelParams,
    ) -> Result<RingRelationSegmentLayout, AkitaError> {
        let lens = ring_relation_segment_lengths::<F, D>(
            lp,
            RingRelationOpeningCounts {
                num_claims: self.opening_batch.num_polynomials(),
                num_t_vectors: self.opening_batch.num_polynomials(),
            },
            self.m_row_layout,
        )?;
        let RingRelationSegmentLengths {
            z_len,
            e_len,
            t_len,
            u_len,
        } = lens;

        let offset_z = 0;
        let offset_e = z_len;
        let offset_t = z_len
            .checked_add(e_len)
            .ok_or_else(|| AkitaError::InvalidSetup("T offset overflow".to_string()))?;
        let offset_u = offset_t
            .checked_add(t_len)
            .ok_or_else(|| AkitaError::InvalidSetup("U offset overflow".to_string()))?;
        let offset_r = offset_u
            .checked_add(u_len)
            .ok_or_else(|| AkitaError::InvalidSetup("r-tail offset overflow".to_string()))?;

        Ok(RingRelationSegmentLayout {
            offset_e,
            offset_t,
            offset_u,
            offset_z,
            offset_r,
        })
    }

    /// Witness layout for the multi-chunk layout.
    pub fn witness_layout(&self, lp: &LevelParams) -> Result<WitnessLayout, AkitaError> {
        let chunk_lengths = witness_chunk_lengths::<F, D>(
            lp,
            RingRelationOpeningCounts {
                num_claims: self.opening_batch.num_polynomials(),
                num_t_vectors: self.opening_batch.num_polynomials(),
            },
            self.m_row_layout,
        )?;
        let WitnessChunkLengths {
            z_chunk_len,
            e_chunk_len,
            t_chunk_len,
            u_chunk_len,
            r_chunk_len: _,
        } = chunk_lengths;

        if u_chunk_len != 0 {
            return Err(AkitaError::InvalidSetup(
                "witness layout: û segment is not supported for multi-chunk layout".to_string(),
            ));
        }

        let overflow =
            || AkitaError::InvalidSetup("witness layout: chunk offset overflow".to_string());
        let num_chunks = lp.witness_chunk.num_chunks;
        let blocks_per_chunk = lp.num_blocks / num_chunks;

        // One chunk is [ẑ, ê, t̂] laid contiguously. The stride is constant across all chunks except the last one.
        // The last chunk is [ẑ, ê, t̂, r̂] laid contiguously.
        let chunk_stride = z_chunk_len
            .checked_add(e_chunk_len)
            .and_then(|n| n.checked_add(t_chunk_len))
            .ok_or_else(overflow)?;
        let offset_r = num_chunks.checked_mul(chunk_stride).ok_or_else(overflow)?;

        let chunks = (0..num_chunks)
            .map(|j| {
                let is_last = j == num_chunks - 1;
                let chunk_offset_base = j.checked_mul(chunk_stride).ok_or_else(overflow)?;
                let offset_z = chunk_offset_base;
                let offset_e = offset_z.checked_add(z_chunk_len).ok_or_else(overflow)?;
                let offset_t = offset_e.checked_add(e_chunk_len).ok_or_else(overflow)?;
                let offset_r = if is_last { Some(offset_r) } else { None };
                Ok(WitnessChunkLayout {
                    global_block_base: j.checked_mul(blocks_per_chunk).ok_or_else(overflow)?,
                    offset_z,
                    offset_e,
                    offset_t,
                    offset_u: None,
                    offset_r,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;

        Ok(WitnessLayout {
            blocks_per_chunk,
            chunks,
            chunk_lengths,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::w_ring_element_count_for_chunks;
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
        let opening_batch = OpeningBatchShape::new(2, 1).expect("valid opening batch");
        let opening_point = opening_point(&lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let err = RingRelationInstance::<F, D>::new(
            MRowLayout::WithoutDBlock,
            test_challenges(&lp, opening_batch.num_polynomials()),
            opening_point,
            ring_multiplier_point,
            opening_batch,
            vec![F::one()],
            vec![CyclotomicRing::one()],
            Vec::new(),
            Vec::new(),
        )
        .expect_err("empty y must be rejected");
        assert!(
            format!("{err:?}")
                .contains("ring relation y must contain at least the consistency row"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn relation_segment_layout_uses_same_axis_contract() {
        let lp = test_level_params();
        let opening_batch = OpeningBatchShape::new(2, 3).expect("valid batch");
        let opening_point = opening_point(&lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let instance = RingRelationInstance::<F, D>::new(
            MRowLayout::WithDBlock,
            test_challenges(&lp, opening_batch.num_polynomials()),
            opening_point,
            ring_multiplier_point,
            opening_batch,
            vec![F::one(); 3],
            vec![CyclotomicRing::one(); 3],
            vec![CyclotomicRing::zero(); 2],
            vec![CyclotomicRing::zero(); lp.d_key.row_len()],
        )
        .expect("same-axis relation");

        let layout = instance.segment_layout(&lp).expect("layout");
        let lens = ring_relation_segment_lengths::<F, D>(
            &lp,
            RingRelationOpeningCounts {
                num_claims: instance.opening_batch().num_polynomials(),
                num_t_vectors: instance.opening_batch().num_polynomials(),
            },
            instance.m_row_layout(),
        )
        .expect("segment lengths");
        assert_eq!(layout.offset_z, 0);
        assert_eq!(layout.offset_e, lens.z_len);
        assert_eq!(layout.offset_t, lens.z_len + lens.e_len);
        assert_eq!(
            layout.offset_r,
            lens.z_len + lens.e_len + lens.t_len + lens.u_len
        );
        instance
            .check_v_shape_for_level(&lp)
            .expect("v rows match layout");
    }

    #[test]
    fn check_capacity_accepts_planner_sized_witness() {
        let lp = test_level_params();
        let opening_batch = OpeningBatchShape::new(2, 3).expect("valid batch");
        let opening_point = opening_point(&lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let instance = RingRelationInstance::<F, D>::new(
            MRowLayout::WithDBlock,
            test_challenges(&lp, opening_batch.num_polynomials()),
            opening_point,
            ring_multiplier_point,
            opening_batch,
            vec![F::one(); 3],
            vec![CyclotomicRing::one(); 3],
            vec![CyclotomicRing::zero(); 2],
            vec![CyclotomicRing::zero(); lp.d_key.row_len()],
        )
        .expect("relation instance");

        let layout = instance.witness_layout(&lp).expect("witness layout");
        layout
            .check_capacity(
                F::modulus_bits(),
                &lp,
                instance.opening_batch().num_polynomials(),
                instance.m_row_layout(),
            )
            .expect("layout span matches planner pricing");
    }

    #[test]
    fn check_capacity_rejects_too_short_witness() {
        let lp = test_level_params();
        let opening_batch = OpeningBatchShape::new(2, 1).expect("valid batch");
        let opening_point = opening_point(&lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let instance = RingRelationInstance::<F, D>::new(
            MRowLayout::WithoutDBlock,
            test_challenges(&lp, opening_batch.num_polynomials()),
            opening_point,
            ring_multiplier_point,
            opening_batch,
            vec![F::one()],
            vec![CyclotomicRing::one()],
            vec![CyclotomicRing::zero(); 2],
            Vec::new(),
        )
        .expect("relation instance");

        let mut layout = instance.witness_layout(&lp).expect("witness layout");
        layout.chunks.last_mut().unwrap().offset_r = Some(0);
        let err = layout
            .check_capacity(
                F::modulus_bits(),
                &lp,
                instance.opening_batch().num_polynomials(),
                instance.m_row_layout(),
            )
            .expect_err("corrupt layout must be rejected");
        assert!(
            format!("{err:?}").contains("witness capacity mismatch"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn witness_ring_len_matches_planner_pricing_at_single_chunk() {
        let lp = test_level_params();
        assert_eq!(lp.witness_chunk.num_chunks, 1);
        let opening_batch = OpeningBatchShape::new(2, 1).expect("valid batch");
        let opening_point = opening_point(&lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let instance = RingRelationInstance::<F, D>::new(
            MRowLayout::WithoutDBlock,
            test_challenges(&lp, opening_batch.num_polynomials()),
            opening_point,
            ring_multiplier_point,
            opening_batch,
            vec![F::one()],
            vec![CyclotomicRing::one()],
            vec![CyclotomicRing::zero(); 2],
            Vec::new(),
        )
        .expect("relation instance");

        let layout = instance.witness_layout(&lp).expect("witness layout");
        let legacy_required = w_ring_element_count_for_chunks(
            F::modulus_bits(),
            &lp,
            instance.opening_batch().num_polynomials(),
            instance.m_row_layout(),
            lp.witness_chunk.num_chunks,
        )
        .expect("planner witness width");
        assert_eq!(
            legacy_required,
            layout.witness_ring_len().expect("layout ring span"),
        );
    }

    /// Build a relation instance for `lp` without repeating the construction
    /// boilerplate in every test.
    fn instance_for(lp: &LevelParams, layout: MRowLayout) -> RingRelationInstance<F, D> {
        let opening_batch = OpeningBatchShape::new(2, 3).expect("valid batch");
        let n = opening_batch.num_polynomials();
        let opening_point = opening_point(lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let v = match layout {
            MRowLayout::WithDBlock => vec![CyclotomicRing::zero(); lp.d_key.row_len()],
            MRowLayout::WithoutDBlock => Vec::new(),
        };
        RingRelationInstance::<F, D>::new(
            layout,
            test_challenges(lp, n),
            opening_point,
            ring_multiplier_point,
            opening_batch,
            vec![F::one(); n],
            vec![CyclotomicRing::one(); n],
            vec![CyclotomicRing::zero(); 2],
            v,
        )
        .expect("relation instance")
    }

    /// `r_vars = 3` ⇒ `num_blocks = 8`, so the witness splits into W ∈ {1,2,4,8}.
    fn multi_chunk_level_params(num_chunks: usize) -> LevelParams {
        let mut lp =
            LevelParams::params_only(crate::SisModulusFamily::Q32, D, 2, 1, 1, 1, stage1_config())
                .with_decomp(2, 3, 1, 2, 0)
                .expect("test params");
        lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
            num_chunks,
            num_activated_levels: 1,
        };
        lp
    }

    /// The key Stage-1 invariant: single-chunk offsets are byte-identical to the
    /// legacy `segment_layout`. Everything downstream relies on this.
    #[test]
    fn num_chunks_one_matches_legacy_segment_layout() {
        let lp = test_level_params();
        let instance = instance_for(&lp, MRowLayout::WithDBlock);

        let legacy = instance.segment_layout(&lp).expect("legacy layout");
        let layout = instance.witness_layout(&lp).expect("witness layout");

        assert_eq!(layout.chunks.len(), 1);
        let c = &layout.chunks[0];
        assert_eq!(c.offset_z, legacy.offset_z);
        assert_eq!(c.offset_e, legacy.offset_e);
        assert_eq!(c.offset_t, legacy.offset_t);
        assert_eq!(c.offset_r, Some(legacy.offset_r));
        assert_eq!(c.global_block_base, 0);
    }

    /// Multi-chunk geometry: `[z|e|t]` per chunk at a constant stride, `r` only on
    /// the last chunk, and `global_block_base` walking the block axis.
    #[test]
    fn multi_chunk_offsets_are_contiguous_and_cover_blocks() {
        for w in [2usize, 4, 8] {
            let lp = multi_chunk_level_params(w);
            let instance = instance_for(&lp, MRowLayout::WithDBlock);
            let layout = instance.witness_layout(&lp).expect("witness layout");

            assert_eq!(layout.chunks.len(), w);
            assert_eq!(layout.blocks_per_chunk, 8 / w);

            let lens = layout.chunk_lengths;
            let stride = lens.z_chunk_len + lens.e_chunk_len + lens.t_chunk_len;
            for (j, c) in layout.chunks.iter().enumerate() {
                assert_eq!(c.offset_z, j * stride);
                assert_eq!(c.offset_e, j * stride + lens.z_chunk_len);
                assert_eq!(c.offset_t, j * stride + lens.z_chunk_len + lens.e_chunk_len);
                assert_eq!(c.global_block_base, j * (8 / w));
                assert_eq!(c.offset_r, if j == w - 1 { Some(w * stride) } else { None });
            }
        }
    }

    /// No-panic boundary: malformed chunk counts return `Err`, never panic.
    /// `3 ∤ 8` and `16 > 8` are both rejected by the divisibility check.
    #[test]
    fn rejects_bad_chunk_counts() {
        for bad in [3usize, 16] {
            let lp = multi_chunk_level_params(bad);
            let instance = instance_for(&lp, MRowLayout::WithDBlock);
            assert!(
                instance.witness_layout(&lp).is_err(),
                "num_chunks = {bad} should be rejected"
            );
        }
    }
}
