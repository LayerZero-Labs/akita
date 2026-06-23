//! Shared public statement for the per-fold negacyclic-ring relation `M * z = y + (X^D + 1) * r`.

use super::OpeningBatch;
use crate::FpExtEncoding;
use crate::{
    embed_ring_subfield_scalar, LevelParams, MRowLayout, RingMultiplierOpeningPoint,
    RingOpeningPoint,
};
use akita_algebra::CyclotomicRing;
use akita_challenges::Challenges;
use akita_field::{AkitaError, FieldCore, Zero};
use akita_field::{CanonicalField, ExtField, FromPrimitiveInt};

/// Column ordering for the ring-switch row MLE: `m_vars >= r_vars` places ẑ
/// before ê/t̂; otherwise ê/t̂ precede ẑ (see
/// `book/src/how/verifying/matrix_evaluation.md`).
#[inline]
pub fn ring_column_z_first(lp: &LevelParams) -> bool {
    lp.m_vars >= lp.r_vars
}

/// Witness-column segment offsets for ring-switch evaluation.
///
/// Produced only by [`RingRelationInstance::segment_layout`] (or
/// [`ring_relation_segment_layout_for_opening_shape`] in tests).
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
    #[cfg(feature = "zk")]
    pub b_blinding_offset: usize,
    #[cfg(feature = "zk")]
    pub d_blinding_offset: usize,
}

/// Public statement of the negacyclic-ring matrix relation at one fold level.
#[derive(Debug, Clone)]
pub struct RingRelationInstance<F: FieldCore, const D: usize> {
    m_row_layout: MRowLayout,
    pub challenges: Challenges,
    opening_point: RingOpeningPoint<F>,
    ring_multiplier_point: RingMultiplierOpeningPoint<F, D>,
    opening_batch: OpeningBatch,
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
        opening_batch: OpeningBatch,
        gamma: Vec<F>,
        row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
        y: Vec<CyclotomicRing<F, D>>,
        v: Vec<CyclotomicRing<F, D>>,
    ) -> Result<Self, AkitaError> {
        opening_batch.check()?;
        if gamma.len() != opening_batch.num_claims()
            || row_coefficient_rings.len() != opening_batch.num_claims()
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

    pub fn opening_batch(&self) -> &OpeningBatch {
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
        let num_blocks = lp.num_blocks;
        if num_blocks == 0 || !num_blocks.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "num_blocks must be a non-zero power of two".to_string(),
            ));
        }
        let depth_open = lp.num_digits_open;
        let depth_commit = lp.num_digits_commit;

        let num_claims = self.opening_batch.num_claims();
        let total_blocks = num_blocks
            .checked_mul(num_claims)
            .ok_or_else(|| AkitaError::InvalidSetup("total block count overflow".to_string()))?;
        let num_t_vectors = self.opening_batch.num_polynomials();
        let depth_fold = lp.num_digits_fold(num_t_vectors, F::modulus_bits())?;
        if depth_open == 0 || depth_commit == 0 || depth_fold == 0 {
            return Err(AkitaError::InvalidSetup(
                "prepared ring-switch layout has zero width".to_string(),
            ));
        }
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
            .and_then(|len| len.checked_mul(1))
            .and_then(|len| len.checked_mul(lp.block_len))
            .ok_or_else(|| AkitaError::InvalidSetup("Z segment length overflow".to_string()))?;

        #[cfg(feature = "zk")]
        let d_blinding_segment_len = match self.m_row_layout {
            MRowLayout::WithDBlock => {
                crate::zk::blinding_digit_plane_count::<F>(lp.d_key.row_len(), D, lp.log_basis)
            }
            MRowLayout::WithoutDBlock => 0,
        };
        #[cfg(not(feature = "zk"))]
        let d_blinding_segment_len = 0usize;
        #[cfg(not(feature = "zk"))]
        let b_blinding_segment_len = 0usize;
        #[cfg(feature = "zk")]
        let b_blinding_digit_planes_per_point =
            crate::zk::blinding_digit_plane_count::<F>(lp.b_key.row_len(), D, lp.log_basis);
        #[cfg(feature = "zk")]
        let b_blinding_segment_len = b_blinding_digit_planes_per_point;

        // Tiered `û_concat` segment length (per the single commitment bundle);
        // `0` for single-tier levels.
        let u_len = lp.u_concat_ring_len_per_group();
        let z_first = ring_column_z_first(lp);
        let offset_z = if z_first {
            0
        } else {
            e_len
                .checked_add(t_len)
                .and_then(|offset| offset.checked_add(u_len))
                .and_then(|offset| offset.checked_add(b_blinding_segment_len))
                .and_then(|offset| offset.checked_add(d_blinding_segment_len))
                .ok_or_else(|| AkitaError::InvalidSetup("Z offset overflow".to_string()))?
        };
        let offset_e = if z_first { z_len } else { 0 };
        let offset_t = if z_first {
            z_len
                .checked_add(e_len)
                .ok_or_else(|| AkitaError::InvalidSetup("T offset overflow".to_string()))?
        } else {
            e_len
        };
        // `û_concat` is emitted immediately after `t̂` in both orderings.
        let offset_u = offset_t
            .checked_add(t_len)
            .ok_or_else(|| AkitaError::InvalidSetup("U offset overflow".to_string()))?;
        let b_blinding_offset = offset_u
            .checked_add(u_len)
            .ok_or_else(|| AkitaError::InvalidSetup("B blinding offset overflow".to_string()))?;
        let d_blinding_offset = b_blinding_offset
            .checked_add(b_blinding_segment_len)
            .ok_or_else(|| AkitaError::InvalidSetup("D blinding offset overflow".to_string()))?;
        let offset_r_base = d_blinding_offset
            .checked_add(d_blinding_segment_len)
            .ok_or_else(|| AkitaError::InvalidSetup("r-tail offset overflow".to_string()))?;
        let offset_r = if z_first {
            offset_r_base
        } else {
            offset_r_base
                .checked_add(z_len)
                .ok_or_else(|| AkitaError::InvalidSetup("r-tail offset overflow".to_string()))?
        };

        Ok(RingRelationSegmentLayout {
            offset_e,
            offset_t,
            offset_u,
            offset_z,
            offset_r,
            #[cfg(feature = "zk")]
            b_blinding_offset,
            #[cfg(feature = "zk")]
            d_blinding_offset,
        })
    }
}

/// Derive witness segment layout for unit tests from level params and opening shape.
///
/// Production code must use [`RingRelationInstance::segment_layout`].
///
/// # Errors
///
/// Same as [`RingRelationInstance::segment_layout`].
pub fn ring_relation_segment_layout_for_opening_shape<
    F: FieldCore + CanonicalField + Zero,
    const D: usize,
>(
    lp: &LevelParams,
    m_row_layout: MRowLayout,
    num_polys: usize,
) -> Result<RingRelationSegmentLayout, AkitaError> {
    let opening_batch = OpeningBatch::same_point(32, num_polys)?;
    let opening_point = RingOpeningPoint {
        a: vec![F::zero(); lp.block_len],
        b: vec![F::zero(); lp.num_blocks],
    };
    let ring_multiplier_point: RingMultiplierOpeningPoint<F, D> =
        RingMultiplierOpeningPoint::from_base(&opening_point);
    let num_claims = opening_batch.num_claims();
    let challenges = Challenges::Sparse {
        challenges: Vec::new(),
        num_blocks_per_claim: lp.num_blocks,
        num_claims,
    };
    let instance = RingRelationInstance::new(
        m_row_layout,
        challenges,
        opening_point,
        ring_multiplier_point,
        opening_batch,
        vec![F::zero(); num_claims],
        vec![CyclotomicRing::<F, D>::zero(); num_claims],
        vec![CyclotomicRing::<F, D>::zero(); num_claims],
        Vec::new(),
    )?;
    instance.segment_layout(lp)
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
        let opening_batch = OpeningBatch::same_point(2, 1).expect("valid opening batch");
        let opening_point = opening_point(&lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let err = RingRelationInstance::<F, D>::new(
            MRowLayout::WithoutDBlock,
            test_challenges(&lp, opening_batch.num_claims()),
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
        let opening_batch = OpeningBatch::same_point(2, 3).expect("valid batch");
        let opening_point = opening_point(&lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let instance = RingRelationInstance::<F, D>::new(
            MRowLayout::WithDBlock,
            test_challenges(&lp, opening_batch.num_claims()),
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
        assert!(ring_column_z_first(&lp));
        assert_eq!(layout.offset_z, 0);
        let num_t_vectors = instance.opening_batch().num_polynomials();
        let depth_fold = lp
            .num_digits_fold(num_t_vectors, F::modulus_bits())
            .unwrap();
        let num_claims = instance.opening_batch().num_claims();
        let z_len = depth_fold * lp.num_digits_commit * lp.block_len;
        let e_len = lp.num_digits_open * lp.num_blocks * num_claims;
        assert_eq!(layout.offset_e, z_len);
        assert_eq!(layout.offset_t, z_len + e_len);
        #[cfg(not(feature = "zk"))]
        {
            let t_len = lp.num_digits_open * lp.a_key.row_len() * lp.num_blocks * num_t_vectors;
            assert_eq!(layout.offset_r, z_len + e_len + t_len);
        }
        instance
            .check_v_shape_for_level(&lp)
            .expect("v rows match layout");
    }
}
