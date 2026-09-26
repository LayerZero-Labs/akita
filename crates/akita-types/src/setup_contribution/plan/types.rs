use crate::{
    CommitmentRingDims, CommitmentSliceGeometry, CommittedGroupParams, OpeningClaimsLayout,
    SetupProjectionGeometry, WitnessLayout,
};
use akita_algebra::offset_eq::{EqPairTensorFamily, OffsetEqWindow};
use akita_error::AkitaError;
use jolt_field::Field;
use std::{ops::Range, sync::Arc};

#[derive(Clone)]
pub struct SetupContributionGroupInputs {
    pub group_id: usize,
    pub num_claims: usize,
    pub depth_fold: usize,
    pub a_row_start: usize,
    pub b_row_start: usize,
}

pub(crate) fn validate_setup_inputs(
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    witness_layout: &WitnessLayout,
    groups: &[SetupContributionGroupInputs],
) -> Result<(), AkitaError> {
    if groups.is_empty() || groups.len() != witness_layout.num_groups() {
        return Err(AkitaError::InvalidSetup(
            "setup groups disagree with witness layout".into(),
        ));
    }
    let group_count = groups.len();
    if witness_layout
        .units()
        .iter()
        .enumerate()
        .any(|(index, unit)| {
            let expected_group = groups[index % group_count].group_id;
            unit.group_index() != expected_group || unit.chunk_index() != index / group_count
        })
    {
        return Err(AkitaError::InvalidSetup(
            "setup witness units do not follow chunk-major relation order".into(),
        ));
    }
    for group in groups {
        group.validate_against(level_params, opening_batch)?;
        validate_group_witness_layout(
            witness_layout,
            group.group_id,
            group.num_live_blocks_for(level_params, opening_batch)?,
        )?;
    }
    validate_setup_group_ids(groups, witness_layout.num_groups())
}

fn validate_setup_group_ids(
    groups: &[SetupContributionGroupInputs],
    num_groups: usize,
) -> Result<(), AkitaError> {
    let mut seen = vec![false; num_groups];
    for group in groups {
        let slot = seen
            .get_mut(group.group_id)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D group id out of range".into()))?;
        if std::mem::replace(slot, true) {
            return Err(AkitaError::InvalidSetup(
                "setup D group id appears more than once".into(),
            ));
        }
    }
    if seen.iter().any(|present| !present) {
        return Err(AkitaError::InvalidSetup(
            "setup D group ids are not contiguous".into(),
        ));
    }
    Ok(())
}

fn validate_group_witness_layout(
    layout: &WitnessLayout,
    group_id: usize,
    num_live_blocks: usize,
) -> Result<(), AkitaError> {
    let units = layout.units_for_group(group_id)?;
    let mut next_fold = 0usize;
    for unit in units {
        if unit.global_block_start() != next_fold {
            return Err(AkitaError::InvalidSetup(
                "setup witness units do not form a contiguous fold tiling".into(),
            ));
        }
        next_fold = next_fold
            .checked_add(unit.num_live_blocks())
            .ok_or_else(|| AkitaError::InvalidSetup("setup fold coverage overflow".into()))?;
    }
    if next_fold != num_live_blocks {
        return Err(AkitaError::InvalidSetup(
            "setup group dimensions disagree with witness layout".into(),
        ));
    }
    Ok(())
}

impl SetupContributionGroupInputs {
    fn group_params_for(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<crate::GroupOpenPhaseParams, AkitaError> {
        level_params.group_params(opening_batch, self.group_id)
    }

    fn validate_against(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<(), AkitaError> {
        let group_layout = opening_batch.group_layout(self.group_id)?;
        if self.num_claims != group_layout.num_polynomials() {
            return Err(AkitaError::InvalidSetup(
                "setup group claim count disagrees with opening batch".into(),
            ));
        }
        let n_a = self.n_a_for(level_params, opening_batch)?;
        let n_b = self.n_b(level_params, opening_batch)?;
        let a_range = level_params.a_row_range(opening_batch, self.group_id)?;
        let b_range = level_params.commitment_row_range(opening_batch, self.group_id)?;
        if a_range.start != self.a_row_start || a_range.len() != n_a {
            return Err(AkitaError::InvalidSetup(
                "setup group A row range disagrees with level params".into(),
            ));
        }
        if b_range.start != self.b_row_start || b_range.len() != n_b {
            return Err(AkitaError::InvalidSetup(
                "setup group B row range disagrees with level params".into(),
            ));
        }
        Ok(())
    }

    fn num_live_blocks_for(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .num_live_blocks())
    }

    fn n_a_for(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .a_rows_len())
    }

    pub(crate) fn num_live_blocks(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .num_live_blocks())
    }

    pub(crate) fn num_positions_per_block(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .num_positions_per_block())
    }

    pub(crate) fn depth_witness(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .num_digits_inner())
    }

    pub(crate) fn depth_commit(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .num_digits_outer())
    }

    pub(crate) fn depth_open(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .num_digits_open())
    }

    pub(crate) fn log_basis_open(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<u32, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .log_basis_open())
    }

    pub(crate) fn n_a(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        Ok(self
            .group_params_for(level_params, opening_batch)?
            .a_rows_len())
    }

    pub(crate) fn n_b(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        self.group_params_for(level_params, opening_batch)?
            .logical_b_rows_len()
    }

    pub(crate) fn t_vector_width(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        let n_a = self.n_a(level_params, opening_batch)?;
        let depth_commit = self.depth_commit(level_params, opening_batch)?;
        let num_live_blocks = self.num_live_blocks(level_params, opening_batch)?;
        n_a.checked_mul(depth_commit)
            .and_then(|n| n.checked_mul(num_live_blocks))
            .ok_or_else(|| AkitaError::InvalidSetup("setup B vector width overflow".into()))
    }

    pub(crate) fn d_active_cols(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        let num_live_blocks = self.num_live_blocks(level_params, opening_batch)?;
        let depth_open = self.depth_open(level_params, opening_batch)?;
        self.num_claims
            .checked_mul(num_live_blocks)
            .and_then(|cols| cols.checked_mul(depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("setup D active width overflow".into()))
    }
}

/// Checked relation-address point and its bounded equality window.
///
/// Clones share both allocations. Keeping the point and window in one value
/// prevents setup planning from consuming a window prepared for different
/// challenges.
#[derive(Clone)]
pub struct PreparedRelationAddress<E: Field> {
    pub(crate) point: Arc<[E]>,
    pub(crate) equality_window: Arc<OffsetEqWindow<E>>,
}

impl<E: Field> PreparedRelationAddress<E> {
    /// Prepare the reusable equality state for one relation-address point.
    ///
    /// # Errors
    ///
    /// Returns an error if the bounded equality window cannot be constructed.
    pub fn new(point: &[E]) -> Result<Self, AkitaError> {
        Ok(Self {
            point: point.to_vec().into(),
            equality_window: Arc::new(OffsetEqWindow::new(point)?),
        })
    }

    /// Relation lane-and-column challenges in LSB-first order.
    #[must_use]
    pub fn point(&self) -> &[E] {
        &self.point
    }

    /// Equality window prepared from [`Self::point`].
    #[must_use]
    pub fn equality_window(&self) -> &OffsetEqWindow<E> {
        &self.equality_window
    }
}

pub struct SetupContributionPlan<E: Field> {
    pub(crate) groups: Vec<SetupContributionGroupPlan<E>>,
    pub(crate) d_rows: usize,
    pub(crate) d_physical_cols: usize,
    pub(crate) d_weights: Arc<[E]>,
    pub(crate) relation_address: PreparedRelationAddress<E>,
    pub(crate) relation_address_geometry: crate::RelationAddressGeometry,
    pub(crate) projection_geometry: SetupProjectionGeometry,
}

/// One logical B source feeding a [`PhysicalBWeightSegment`].
#[derive(Clone, Copy)]
pub struct PhysicalBWeightTerm<E> {
    /// First logical B column read by this term.
    pub logical_start: usize,
    /// Logical B row weight applied to every column of the term.
    pub row_weight: E,
}

/// One contiguous physical B column run and its logical sources.
#[derive(Clone)]
pub struct PhysicalBWeightSegment<E> {
    /// First physical B ring index of the run.
    pub physical_start: usize,
    /// Number of physical B rings in the run.
    pub len: usize,
    /// Logical slices whose weights add onto this run.
    pub terms: Arc<[PhysicalBWeightTerm<E>]>,
}

/// One canonical owner for the physical B matrix and its logical sliced image.
pub struct PhysicalBSetupPlan<E: Field> {
    pub(super) geometry: CommitmentSliceGeometry,
    pub(super) physical_rows: usize,
    pub(super) logical_row_weights: Arc<[E]>,
    pub(super) weight_segments: Arc<[PhysicalBWeightSegment<E>]>,
    pub(super) relation_tensors: Vec<EqPairTensorFamily<E>>,
}

impl<E: Field> PhysicalBSetupPlan<E> {
    pub fn new(
        geometry: CommitmentSliceGeometry,
        physical_rows: usize,
        logical_row_weights: Arc<[E]>,
    ) -> Result<Self, AkitaError> {
        let logical_rows = geometry.logical_output_rows(physical_rows)?;
        if logical_row_weights.len() != logical_rows {
            return Err(AkitaError::InvalidSetup(
                "logical B row weights disagree with slice geometry".into(),
            ));
        }
        let weight_segments = super::physical_b::build_physical_b_weight_segments(
            &geometry,
            physical_rows,
            &logical_row_weights,
        )?;
        Ok(Self {
            geometry,
            physical_rows,
            logical_row_weights,
            weight_segments: weight_segments.into(),
            relation_tensors: Vec::new(),
        })
    }

    pub fn logical_rows(&self) -> Result<usize, AkitaError> {
        self.geometry.logical_output_rows(self.physical_rows)
    }

    pub const fn geometry(&self) -> &CommitmentSliceGeometry {
        &self.geometry
    }

    pub const fn physical_rows(&self) -> usize {
        self.physical_rows
    }

    pub fn logical_row_weights(&self) -> &[E] {
        &self.logical_row_weights
    }

    pub fn weight_segments(&self) -> &[PhysicalBWeightSegment<E>] {
        &self.weight_segments
    }

    /// Relation-column tensors of the logical B role, one per active unit
    /// and claim.
    pub fn relation_tensors(&self) -> &[EqPairTensorFamily<E>] {
        &self.relation_tensors
    }

    pub fn logical_input_width(&self) -> usize {
        self.geometry.logical_input_width()
    }

    pub fn physical_input_width(&self) -> usize {
        self.geometry.physical_input_width()
    }

    pub fn physical_footprint(&self) -> Result<usize, AkitaError> {
        self.geometry
            .physical_matrix_ring_elements(self.physical_rows)
    }
}

/// Coefficient-functional-free per-group setup-contribution geometry and
/// weights.
///
/// Built only by [`SetupContributionPlan::prepare`] (or the test-support
/// fixture constructor); consumers read it through
/// [`SetupContributionPlan::groups`].
pub struct SetupContributionGroupPlan<E: Field> {
    pub(crate) group_id: usize,
    pub(crate) opening_method: crate::OpeningMethod,
    pub(crate) role_dims: CommitmentRingDims,
    pub(crate) a_ratio: usize,
    pub(crate) b_ratio: usize,
    pub(crate) d_ratio: usize,
    pub(crate) a_relation_ratio: usize,
    pub(crate) b_relation_ratio: usize,
    pub(crate) d_relation_ratio: usize,
    pub(crate) opening_subcolumns: usize,
    pub(crate) consistency_weight: E,
    pub(crate) num_claims: usize,
    pub(crate) num_live_blocks: usize,
    pub(crate) num_positions_per_block: usize,
    pub(crate) depth_witness: usize,
    pub(crate) depth_commit: usize,
    pub(crate) depth_open: usize,
    pub(crate) log_basis_inner: u32,
    pub(crate) log_basis_outer: u32,
    pub(crate) log_basis_open: u32,
    pub(crate) d_col_range: Range<usize>,
    pub(crate) z_cols: usize,
    pub(crate) n_a: usize,
    pub(crate) physical_b: PhysicalBSetupPlan<E>,
    pub(crate) a_row_weights: Arc<[E]>,
    pub(crate) fold_gadget: Arc<[E]>,
    /// The non-empty witness units of this group, in layout order. The E and
    /// T roles are partitioned by them, and the verifier's Stage-3 B tensors
    /// are rebuilt from them, so no second layout is ever consulted.
    pub(crate) active_units: Arc<[crate::WitnessUnitLayout]>,
    /// All physical units, including empty chunks that retain replicated Z.
    pub(crate) num_physical_units: usize,
    pub(crate) d_tensors: Vec<EqPairTensorFamily<E>>,
    pub(crate) a_tensors: Vec<EqPairTensorFamily<E>>,
}

#[cfg(any(test, feature = "test-support"))]
impl<E: Field> SetupContributionGroupPlan<E> {
    /// Minimal fixture group: evaluation-trace opening, uniform 64-dimensional
    /// roles with unit ratios, no claims, live blocks, units or tensors.
    #[must_use]
    pub fn from_test_parts(
        d_col_range: Range<usize>,
        z_cols: usize,
        n_a: usize,
        physical_b: PhysicalBSetupPlan<E>,
        a_row_weights: Arc<[E]>,
    ) -> Self {
        Self {
            group_id: 0,
            opening_method: crate::OpeningMethod::EvaluationTrace,
            role_dims: CommitmentRingDims::uniform(64),
            a_ratio: 1,
            b_ratio: 1,
            d_ratio: 1,
            a_relation_ratio: 1,
            b_relation_ratio: 1,
            d_relation_ratio: 1,
            opening_subcolumns: 1,
            consistency_weight: E::one(),
            num_claims: 0,
            num_live_blocks: 0,
            num_positions_per_block: z_cols,
            depth_witness: 1,
            depth_commit: 1,
            depth_open: 1,
            log_basis_inner: 1,
            log_basis_outer: 1,
            log_basis_open: 1,
            d_col_range,
            z_cols,
            n_a,
            physical_b,
            a_row_weights,
            fold_gadget: vec![E::one()].into(),
            active_units: Vec::new().into(),
            num_physical_units: 0,
            d_tensors: Vec::new(),
            a_tensors: Vec::new(),
        }
    }

    /// Mutable A-row weights, for tests that perturb them.
    pub fn a_row_weights_mut_for_test(&mut self) -> &mut [E] {
        Arc::make_mut(&mut self.a_row_weights)
    }

    /// Overwrites the consistency-row weight, for tests that perturb it.
    pub fn set_consistency_weight_for_test(&mut self, weight: E) {
        self.consistency_weight = weight;
    }

    /// Mutable D, B and A role tensor families, for tests that perturb them.
    pub fn role_tensors_mut_for_test(&mut self) -> [&mut [EqPairTensorFamily<E>]; 3] {
        [
            &mut self.d_tensors,
            &mut self.physical_b.relation_tensors,
            &mut self.a_tensors,
        ]
    }
}

impl<E: Field> SetupContributionGroupPlan<E> {
    /// Index of this group in the opening batch.
    #[must_use]
    pub const fn group_id(&self) -> usize {
        self.group_id
    }

    /// Opening method of the consuming fold.
    #[must_use]
    pub const fn opening_method(&self) -> crate::OpeningMethod {
        self.opening_method
    }

    /// Ring dimensions of the A, B and D roles.
    #[must_use]
    pub const fn role_dims(&self) -> CommitmentRingDims {
        self.role_dims
    }

    /// `d_a` over the shared setup base ring dimension.
    #[must_use]
    pub const fn a_ratio(&self) -> usize {
        self.a_ratio
    }

    /// `d_b` over the shared setup base ring dimension.
    #[must_use]
    pub const fn b_ratio(&self) -> usize {
        self.b_ratio
    }

    /// `d_d` over the shared setup base ring dimension.
    #[must_use]
    pub const fn d_ratio(&self) -> usize {
        self.d_ratio
    }

    /// `d_a` over the relation coefficient block length.
    #[must_use]
    pub const fn a_relation_ratio(&self) -> usize {
        self.a_relation_ratio
    }

    /// `d_b` over the relation coefficient block length.
    #[must_use]
    pub const fn b_relation_ratio(&self) -> usize {
        self.b_relation_ratio
    }

    /// `d_d` over the relation coefficient block length.
    #[must_use]
    pub const fn d_relation_ratio(&self) -> usize {
        self.d_relation_ratio
    }

    /// D-role subcolumns per opening column.
    #[must_use]
    pub const fn opening_subcolumns(&self) -> usize {
        self.opening_subcolumns
    }

    /// `eq(tau_1)` weight of this group's consistency row.
    #[must_use]
    pub const fn consistency_weight(&self) -> E {
        self.consistency_weight
    }

    /// Number of opening claims in this group.
    #[must_use]
    pub const fn num_claims(&self) -> usize {
        self.num_claims
    }

    /// Number of live witness blocks.
    #[must_use]
    pub const fn num_live_blocks(&self) -> usize {
        self.num_live_blocks
    }

    /// Positions per witness block.
    #[must_use]
    pub const fn num_positions_per_block(&self) -> usize {
        self.num_positions_per_block
    }

    /// Inner (witness) gadget depth.
    #[must_use]
    pub const fn depth_witness(&self) -> usize {
        self.depth_witness
    }

    /// Commitment gadget depth.
    #[must_use]
    pub const fn depth_commit(&self) -> usize {
        self.depth_commit
    }

    /// Opening gadget depth.
    #[must_use]
    pub const fn depth_open(&self) -> usize {
        self.depth_open
    }

    /// Inner decomposition log-basis.
    #[must_use]
    pub const fn log_basis_inner(&self) -> u32 {
        self.log_basis_inner
    }

    /// Outer decomposition log-basis.
    #[must_use]
    pub const fn log_basis_outer(&self) -> u32 {
        self.log_basis_outer
    }

    /// Opening decomposition log-basis.
    #[must_use]
    pub const fn log_basis_open(&self) -> u32 {
        self.log_basis_open
    }

    /// Z columns: positions per block times the witness depth.
    #[must_use]
    pub const fn z_cols(&self) -> usize {
        self.z_cols
    }

    /// Number of A rows of this group.
    #[must_use]
    pub const fn n_a(&self) -> usize {
        self.n_a
    }

    /// All physical units, including empty chunks that retain replicated Z.
    #[must_use]
    pub const fn num_physical_units(&self) -> usize {
        self.num_physical_units
    }

    /// This group's column range in the shared physical D matrix.
    #[must_use]
    pub fn d_col_range(&self) -> Range<usize> {
        self.d_col_range.clone()
    }

    /// Physical B setup plan for this group.
    #[must_use]
    pub const fn physical_b(&self) -> &PhysicalBSetupPlan<E> {
        &self.physical_b
    }

    /// `eq(tau_1)` weights of this group's A rows.
    #[must_use]
    pub fn a_row_weights(&self) -> &[E] {
        &self.a_row_weights
    }

    /// The first `depth_fold` fold-gadget scalars, lifted to `E`.
    #[must_use]
    pub fn fold_gadget(&self) -> &[E] {
        &self.fold_gadget
    }

    /// The non-empty witness units of this group, in layout order.
    #[must_use]
    pub fn active_units(&self) -> &[crate::WitnessUnitLayout] {
        &self.active_units
    }

    /// D-role eq-pair tensor families.
    #[must_use]
    pub fn d_tensors(&self) -> &[EqPairTensorFamily<E>] {
        &self.d_tensors
    }

    /// A-role eq-pair tensor families.
    #[must_use]
    pub fn a_tensors(&self) -> &[EqPairTensorFamily<E>] {
        &self.a_tensors
    }

    pub(crate) fn set_projection_ratios(
        &mut self,
        setup_base_ring_dim: usize,
        relation_base_ring_dim: usize,
    ) -> Result<(), AkitaError> {
        let ratio = |name: &'static str, dimension: usize, base_ring_dim: usize| {
            dimension
                .checked_div(base_ring_dim)
                .filter(|ratio| {
                    base_ring_dim != 0
                        && dimension.is_multiple_of(base_ring_dim)
                        && ratio.is_power_of_two()
                })
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "setup {name} dimension does not project to the shared base"
                    ))
                })
        };
        self.a_ratio = ratio("A", self.role_dims.d_a(), setup_base_ring_dim)?;
        self.b_ratio = ratio("B", self.role_dims.d_b(), setup_base_ring_dim)?;
        self.d_ratio = ratio("D", self.role_dims.d_d(), setup_base_ring_dim)?;
        self.a_relation_ratio = ratio("relation A", self.role_dims.d_a(), relation_base_ring_dim)?;
        self.b_relation_ratio = ratio("relation B", self.role_dims.d_b(), relation_base_ring_dim)?;
        self.d_relation_ratio = ratio("relation D", self.role_dims.d_d(), relation_base_ring_dim)?;
        Ok(())
    }
}
