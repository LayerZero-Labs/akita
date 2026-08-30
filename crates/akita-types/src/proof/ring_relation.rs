//! Shared public statement for the per-fold negacyclic-ring relation `M * z = y + (X^D + 1) * r`.

use super::OpeningClaimsLayout;
use crate::layout::{CommitmentRingDims, RingRole};
use crate::validate_role_dispatch;
use crate::witness::WitnessLayout;
use crate::FpExtEncoding;
use crate::{
    embed_ring_subfield_scalar, CommittedGroupParams, OpeningFamily, RingMultiplierOpeningPoint,
    RingVec, SubringCoefficientPackingGeometry,
};
use akita_algebra::CyclotomicRing;
use akita_challenges::Challenges;
use akita_error::AkitaError;
use challenge_validation::validate_packing_challenge_weights;
use jolt_field::Field;
use jolt_field::{CanonicalEncoding, ExtField, Ring};

mod challenge_validation;

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

/// Method-typed fold challenge and opening-point material for one relation group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingRelationGroupOpening<F: Field> {
    kind: OpeningFamily<EvaluationTraceGroupOpening<F>, CoefficientPackingChallenges>,
}

/// Borrowed, method-typed view of one relation group's opening material.
#[derive(Debug, Clone, Copy)]
pub enum RingRelationGroupOpeningView<'a, F: Field> {
    /// EvaluationTrace challenges and its ring-multiplier point.
    EvaluationTrace {
        challenges: &'a Challenges,
        ring_multiplier_point: &'a RingMultiplierOpeningPoint<F>,
    },
    /// Canonical subring challenges and their derived ambient-A embedding.
    SubringCoefficientPacking {
        geometry: SubringCoefficientPackingGeometry,
        canonical_subring_challenges: &'a Challenges,
        ambient_a_challenges: &'a Challenges,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EvaluationTraceGroupOpening<F: Field> {
    challenges: Challenges,
    ring_multiplier_point: RingMultiplierOpeningPoint<F>,
}

/// Checked canonical and ambient views of one coefficient-packing challenge batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoefficientPackingChallenges {
    geometry: SubringCoefficientPackingGeometry,
    subring_challenges: Challenges,
    embedded_a_challenges: Challenges,
}

impl CoefficientPackingChallenges {
    /// Validate canonical subring challenges and derive their ambient-A embedding once.
    pub fn new(
        geometry: SubringCoefficientPackingGeometry,
        subring_challenges: Challenges,
    ) -> Result<Self, AkitaError> {
        let config = geometry.fold_challenge_config();
        validate_packing_challenge_weights(&subring_challenges, &config)?;
        let embedded_a_challenges = subring_challenges.embed_subring_positions(
            geometry.challenge_subring_dimension(),
            geometry.subring_embedding_stride(),
            geometry.a_ring_dimension(),
        )?;
        Ok(Self {
            geometry,
            subring_challenges,
            embedded_a_challenges,
        })
    }

    /// Geometry authenticated by both challenge views.
    #[must_use]
    pub const fn geometry(&self) -> SubringCoefficientPackingGeometry {
        self.geometry
    }

    /// Canonical challenges sampled in the packing subring.
    #[must_use]
    pub const fn canonical(&self) -> &Challenges {
        &self.subring_challenges
    }

    /// The same challenges embedded in the ambient A ring.
    #[must_use]
    pub const fn ambient_a(&self) -> &Challenges {
        &self.embedded_a_challenges
    }
}

impl<F: Field> RingRelationGroupOpening<F> {
    /// Borrow this opening without erasing its algebraic method.
    #[must_use]
    pub fn view(&self) -> RingRelationGroupOpeningView<'_, F> {
        match &self.kind {
            OpeningFamily::EvaluationTrace(opening) => {
                RingRelationGroupOpeningView::EvaluationTrace {
                    challenges: &opening.challenges,
                    ring_multiplier_point: &opening.ring_multiplier_point,
                }
            }
            OpeningFamily::SubringCoefficientPacking(opening) => {
                RingRelationGroupOpeningView::SubringCoefficientPacking {
                    geometry: opening.geometry,
                    canonical_subring_challenges: &opening.subring_challenges,
                    ambient_a_challenges: &opening.embedded_a_challenges,
                }
            }
        }
    }

    /// Construct the current full-A evaluation-trace carrier.
    pub fn evaluation_trace(
        challenges: Challenges,
        ring_multiplier_point: RingMultiplierOpeningPoint<F>,
    ) -> Self {
        Self {
            kind: OpeningFamily::EvaluationTrace(EvaluationTraceGroupOpening {
                challenges,
                ring_multiplier_point,
            }),
        }
    }

    /// Construct the relation carrier from one checked packing challenge batch.
    #[must_use]
    pub fn coefficient_packing(challenges: CoefficientPackingChallenges) -> Self {
        Self {
            kind: OpeningFamily::SubringCoefficientPacking(challenges),
        }
    }

    /// Canonical challenges for this opening relation.
    pub fn canonical_challenges(&self) -> &Challenges {
        match &self.kind {
            OpeningFamily::EvaluationTrace(opening) => &opening.challenges,
            OpeningFamily::SubringCoefficientPacking(opening) => &opening.subring_challenges,
        }
    }

    /// Challenges embedded in the ambient A ring.
    pub fn ambient_a_challenges(&self) -> &Challenges {
        match &self.kind {
            OpeningFamily::EvaluationTrace(opening) => &opening.challenges,
            OpeningFamily::SubringCoefficientPacking(opening) => &opening.embedded_a_challenges,
        }
    }

    /// Evaluation-trace multiplier point, rejecting coefficient packing.
    pub fn evaluation_trace_multiplier_point(
        &self,
    ) -> Result<&RingMultiplierOpeningPoint<F>, AkitaError> {
        match &self.kind {
            OpeningFamily::EvaluationTrace(opening) => Ok(&opening.ring_multiplier_point),
            OpeningFamily::SubringCoefficientPacking(_) => Err(AkitaError::InvalidSetup(
                "coefficient packing has no evaluation-trace multiplier point".into(),
            )),
        }
    }

    /// Checked coefficient-packing geometry, when this group uses packing.
    #[must_use]
    pub fn coefficient_packing_geometry(&self) -> Option<SubringCoefficientPackingGeometry> {
        match self.kind {
            OpeningFamily::SubringCoefficientPacking(ref opening) => Some(opening.geometry),
            OpeningFamily::EvaluationTrace(_) => None,
        }
    }
}

/// Witness segment lengths shared by prover emission, layout offsets, and M-table sizing.
pub fn ring_relation_segment_lengths<F: Field + CanonicalEncoding>(
    lp: &CommittedGroupParams,
    opening_counts: RingRelationOpeningCounts,
) -> Result<RingRelationSegmentLengths, AkitaError> {
    let num_live_blocks = lp.blocks().live_blocks;
    if num_live_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "num_live_blocks must be positive".to_string(),
        ));
    }
    let depth_open = lp.open().digits.num_digits;
    let depth_inner = lp.inner().digits.num_digits;
    let depth_outer = lp.outer().digits.num_digits;
    let RingRelationOpeningCounts {
        num_claims,
        num_t_vectors,
    } = opening_counts;
    let depth_fold = lp.num_digits_fold();
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
        .checked_mul(lp.inner().matrix.output_rank())
        .and_then(|len| len.checked_mul(t_total_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("T segment length overflow".to_string()))?;
    let z_len = depth_fold
        .checked_mul(depth_inner)
        .and_then(|len| len.checked_mul(lp.blocks().positions_per_block))
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
/// closures borrow typed role-local ring rows via [`Self::v_trusted`],
/// and [`Self::row_coefficient_rings_trusted`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingRelationInstance<F: Field> {
    group_openings: Vec<RingRelationGroupOpening<F>>,
    extension_degree: usize,
    opening_batch: OpeningClaimsLayout,
    gamma: Vec<F>,
    row_coefficient_rings: RingVec<F>,
    rhs: RingVec<F>,
    v: RingVec<F>,
    role_dims: CommitmentRingDims,
}

impl<F: Field + CanonicalEncoding> RingRelationInstance<F> {
    /// Construct a validated ring-relation statement from D-free ring storage.
    ///
    /// Does not sample from the transcript; callers must absorb/sample before calling.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        group_openings: Vec<RingRelationGroupOpening<F>>,
        extension_degree: usize,
        opening_batch: OpeningClaimsLayout,
        gamma: Vec<F>,
        row_coefficient_rings: RingVec<F>,
        rhs: RingVec<F>,
        v: RingVec<F>,
        role_dims: CommitmentRingDims,
    ) -> Result<Self, AkitaError> {
        opening_batch.check()?;
        if extension_degree == 0 {
            return Err(AkitaError::InvalidInput(
                "ring relation extension degree must be nonzero".into(),
            ));
        }
        let num_groups = opening_batch.num_groups();
        if group_openings.len() != num_groups {
            return Err(AkitaError::InvalidInput(
                "ring relation group carrier count does not match opening batch".to_string(),
            ));
        }
        for (g, group_opening) in group_openings.iter().enumerate() {
            let group_layout = opening_batch.group_layout(g)?;
            let k_g = group_layout.num_polynomials();
            let challenges = group_opening.canonical_challenges();
            if challenges.num_claims() != k_g {
                return Err(AkitaError::InvalidInput(format!(
                    "ring relation group {g} challenges claim count {} does not match K_g={k_g}",
                    challenges.num_claims()
                )));
            }
            let num_live_blocks_g = challenges.num_live_blocks_per_claim();
            if let OpeningFamily::EvaluationTrace(opening) = &group_opening.kind {
                if opening.ring_multiplier_point.fold_len() != num_live_blocks_g {
                    return Err(AkitaError::InvalidInput(format!(
                        "ring relation group {g} ring multiplier block count does not match challenges"
                    )));
                }
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
            group_openings,
            extension_degree,
            opening_batch,
            gamma,
            row_coefficient_rings,
            rhs,
            v,
            role_dims,
        })
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

    pub fn group_openings(&self) -> &[RingRelationGroupOpening<F>] {
        &self.group_openings
    }

    /// Method-typed opening material for one group.
    pub fn group_opening_view(
        &self,
        group: usize,
    ) -> Result<RingRelationGroupOpeningView<'_, F>, AkitaError> {
        self.group_openings
            .get(group)
            .map(RingRelationGroupOpening::view)
            .ok_or_else(|| AkitaError::InvalidInput("relation group index is invalid".into()))
    }

    /// Protocol field extension degree used to resolve packing coordinate planes.
    #[must_use]
    pub fn extension_degree(&self) -> usize {
        self.extension_degree
    }

    pub fn group_ambient_a_challenges(&self, group: usize) -> Result<&Challenges, AkitaError> {
        self.group_openings
            .get(group)
            .map(RingRelationGroupOpening::ambient_a_challenges)
            .ok_or(AkitaError::InvalidProof)
    }

    pub fn group_ring_multiplier_point(
        &self,
        g: usize,
    ) -> Result<&RingMultiplierOpeningPoint<F>, AkitaError> {
        self.group_openings
            .get(g)
            .ok_or_else(|| {
                AkitaError::InvalidInput(format!(
                    "ring relation ring multiplier group index {g} out of range ({} groups)",
                    self.group_openings.len()
                ))
            })?
            .evaluation_trace_multiplier_point()
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
    /// The heterogeneous RHS is intentionally excluded: compression rows use
    /// native dimensions even when A/B/D share `D`.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        let uniform = self.role_dims.uniform_dim()?;
        if uniform != D {
            return Err(AkitaError::InvalidInput(format!(
                "ring relation uniform dim {uniform} does not match requested D={D}"
            )));
        }
        validate_role_dispatch::<D>(self.role_dims, RingRole::Inner)?;
        if !self.row_coefficient_rings.can_decode_vec(D) || !self.v.can_decode_vec(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.v.coeff_len(),
            });
        }
        for opening in &self.group_openings {
            match opening.view() {
                RingRelationGroupOpeningView::EvaluationTrace {
                    ring_multiplier_point,
                    ..
                } => ring_multiplier_point.ensure_ring_dim::<D>()?,
                RingRelationGroupOpeningView::SubringCoefficientPacking { geometry, .. } => {
                    if geometry.a_ring_dimension() != D {
                        return Err(AkitaError::InvalidInput(format!(
                            "coefficient-packing ambient dimension {} does not match requested D={D}",
                            geometry.a_ring_dimension(),
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Validate one role carrier against dispatch `D`.
    pub fn ensure_role_dim<const D: usize>(&self, role: RingRole) -> Result<(), AkitaError> {
        validate_role_dispatch::<D>(self.role_dims, role).map(|_| ())
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
        let expected = lp.open().matrix.output_rank();
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
        F: Ring,
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
    /// single-machine row count. Pass `witness_coeff_len = Some(witness_len)`
    /// to enforce the no-panic capacity bound at this boundary.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] (never panics) for malformed
    /// ownership geometry, offset or length arithmetic overflow, or a layout
    /// whose shared `r` tail would exceed the committed witness capacity.
    pub fn segment_layout(
        &self,
        lp: &CommittedGroupParams,
        witness_coeff_len: Option<usize>,
    ) -> Result<WitnessLayout, AkitaError> {
        lp.witness_chunk.validate()?;
        let num_chunks = lp.witness_chunk.num_chunks;
        let relation_geometry = crate::RelationWitnessGeometry::for_level(
            lp,
            &self.opening_batch,
            self.extension_degree,
        )?;
        for (group_index, opening) in self.group_openings.iter().enumerate() {
            let expected_method = relation_geometry.group_opening_method(group_index)?;
            let group_params = lp.group_params_geometry(&self.opening_batch, group_index)?;
            let expected_blocks = group_params.num_live_blocks();
            if opening.canonical_challenges().num_live_blocks_per_claim() != expected_blocks {
                return Err(AkitaError::InvalidSetup(
                    "relation opening block count disagrees with the schedule".into(),
                ));
            }
            match (expected_method, opening.coefficient_packing_geometry()) {
                (crate::OpeningMethod::EvaluationTrace, None) => {}
                (
                    crate::OpeningMethod::SubringCoefficientPacking {
                        challenge_subring_dimension,
                    },
                    Some(actual),
                ) => {
                    let expected = SubringCoefficientPackingGeometry::try_new(
                        self.extension_degree,
                        group_params.inner_commit_matrix_params().ring_dimension(),
                        challenge_subring_dimension,
                    )?;
                    if actual != expected {
                        return Err(AkitaError::InvalidSetup(
                            "relation opening geometry disagrees with the schedule".into(),
                        ));
                    }
                }
                _ => {
                    return Err(AkitaError::InvalidSetup(
                        "relation opening method disagrees with the schedule".into(),
                    ));
                }
            }
        }
        let relation_rhs_layout = relation_geometry.rhs_layout();
        let expected_rhs_coeff_len =
            crate::proof::relation::relation_rhs_coeff_len(relation_rhs_layout)?;
        if self.rhs.coeff_len() != expected_rhs_coeff_len {
            return Err(AkitaError::InvalidSetup(format!(
                "ring relation rhs coefficient length {} does not match per-role layout (expected {expected_rhs_coeff_len})",
                self.rhs.coeff_len()
            )));
        }
        // `EvaluationTrace` is a logical relation row used by Stage 2. It is
        // not materialized in the quotient witness's shared `r` tail.
        let layout = WitnessLayout::new(
            lp,
            &self.opening_batch,
            &relation_geometry,
            num_chunks,
            crate::RelationQuotientPlan::for_field_bits(lp, F::MODULUS_BITS)?,
        )?;
        if let Some(capacity) = witness_coeff_len {
            if layout.live_coeff_len() > capacity {
                return Err(AkitaError::InvalidSetup(format!(
                    "resolved witness layout requires {} coefficients but only {capacity} are committed",
                    layout.live_coeff_len(),
                )));
            }
        }
        Ok(layout)
    }
}

#[cfg(test)]
#[path = "ring_relation/tests.rs"]
mod tests;
