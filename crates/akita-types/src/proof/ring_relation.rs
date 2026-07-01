//! Shared public statement for the per-fold negacyclic-ring relation `M * z = y + (X^D + 1) * r`.

use super::OpeningBatchShape;
use crate::FpExtEncoding;
use crate::{
    embed_ring_subfield_scalar, LevelParams, MRowLayout, RingMultiplierOpeningPoint,
    RingOpeningPoint, RingVec,
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
    opening_batch: OpeningBatchShape,
    gamma: Vec<F>,
    row_coefficient_rings: RingVec<F>,
    y: RingVec<F>,
    v: RingVec<F>,
    ring_dim: usize,
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
        opening_batch: OpeningBatchShape,
        gamma: Vec<F>,
        row_coefficient_rings: RingVec<F>,
        y: RingVec<F>,
        v: RingVec<F>,
        ring_dim: usize,
    ) -> Result<Self, AkitaError> {
        opening_batch.check()?;
        if gamma.len() != opening_batch.num_polynomials()
            || row_coefficient_rings.count() != opening_batch.num_polynomials()
        {
            return Err(AkitaError::InvalidInput(
                "ring relation gamma/row coefficients length mismatch".to_string(),
            ));
        }
        if y.count() == 0 {
            return Err(AkitaError::InvalidInput(
                "ring relation y must contain at least the consistency row".to_string(),
            ));
        }
        if ring_dim == 0 {
            return Err(AkitaError::InvalidSize {
                expected: 1,
                actual: 0,
            });
        }
        if !row_coefficient_rings.can_decode_vec(ring_dim)
            || !y.can_decode_vec(ring_dim)
            || !v.can_decode_vec(ring_dim)
        {
            return Err(AkitaError::InvalidSize {
                expected: ring_dim,
                actual: row_coefficient_rings.coeff_len(),
            });
        }
        for (idx, chunk) in row_coefficient_rings
            .coeffs()
            .chunks_exact(ring_dim)
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
            ring_dim,
        })
    }

    /// Construct from typed kernel outputs at a ring-relation boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts<const D: usize>(
        m_row_layout: MRowLayout,
        challenges: Challenges,
        opening_point: RingOpeningPoint<F>,
        ring_multiplier_point: RingMultiplierOpeningPoint<F>,
        opening_batch: OpeningBatchShape,
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
            D,
        )
    }

    /// Stored ring dimension (coefficients per ring element).
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
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

    pub fn ring_multiplier_point(&self) -> &RingMultiplierOpeningPoint<F> {
        &self.ring_multiplier_point
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

    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match storage.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dim != D {
            return Err(AkitaError::InvalidInput(format!(
                "ring relation instance ring_d={} does not match requested D={D}",
                self.ring_dim
            )));
        }
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

    /// Borrow `y` rows after [`Self::ensure_ring_dim`].
    pub fn y_trusted<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        self.y.as_ring_slice::<D>()
    }

    /// Borrow `v` rows after [`Self::ensure_ring_dim`].
    pub fn v_trusted<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        self.v.as_ring_slice::<D>()
    }

    /// Borrow row-coefficient rings after [`Self::ensure_ring_dim`].
    pub fn row_coefficient_rings_trusted<const D: usize>(
        &self,
    ) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        self.row_coefficient_rings.as_ring_slice::<D>()
    }

    /// Validate layout-dependent D-row payload shape.
    pub fn check_v_shape_for_level(&self, lp: &LevelParams) -> Result<(), AkitaError> {
        let expected = match self.m_row_layout {
            MRowLayout::WithDBlock => lp.d_key.row_len(),
            MRowLayout::WithoutDBlock => 0,
        };
        if self.v.count() != expected {
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

    /// Witness-column segment layout shared by prover and verifier ring-switch paths.
    pub fn segment_layout(
        &self,
        lp: &LevelParams,
    ) -> Result<RingRelationSegmentLayout, AkitaError> {
        let lens = ring_relation_segment_lengths::<F>(
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
        let opening_batch = OpeningBatchShape::new(2, 1).expect("valid opening batch");
        let opening_point = opening_point(&lp);
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&opening_point);
        let err = RingRelationInstance::<F>::new(
            MRowLayout::WithoutDBlock,
            test_challenges(&lp, opening_batch.num_polynomials()),
            opening_point,
            ring_multiplier_point,
            opening_batch,
            vec![F::one()],
            RingVec::from_ring_elems::<D>(&[CyclotomicRing::one()]),
            RingVec::from_ring_elems::<D>(&[]),
            RingVec::from_ring_elems::<D>(&[]),
            D,
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
        let v_zeros = vec![CyclotomicRing::zero(); lp.d_key.row_len()];
        let instance = RingRelationInstance::<F>::from_parts::<D>(
            MRowLayout::WithDBlock,
            test_challenges(&lp, opening_batch.num_polynomials()),
            opening_point,
            ring_multiplier_point,
            opening_batch,
            vec![F::one(); 3],
            &[CyclotomicRing::one(); 3],
            &[CyclotomicRing::zero(); 2],
            &v_zeros,
        )
        .expect("same-axis relation");

        let layout = instance.segment_layout(&lp).expect("layout");
        let lens = ring_relation_segment_lengths::<F>(
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
}
