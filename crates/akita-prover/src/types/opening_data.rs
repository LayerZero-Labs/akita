use crate::backend::RecursiveFoldSource;
use crate::commitment::{
    InnerRelationState, InnerRelationStateMaterial, OuterCompressionState, PortableCompressionState,
};
use crate::compute::RootPolyMeta;
use crate::protocol::core::RootProverGroupMeta;
use crate::{ErasedPreparedProverGroup, PreparedProverGroup};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::{
    AkitaCommitmentHint, Commitment, CommittedGroup, CommittedGroupBatchProfile,
    CommittedGroupParams, CompressionChainPlan, FpExtEncoding, OpeningClaims, OpeningClaimsLayout,
    OpeningScheduleSelection, PolynomialGroupClaims, PolynomialGroupLayout, RingRelationMode,
    SetupPrefixSlot,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};

/// Exact top-level row selection paired with its prover opening material.
#[derive(Debug, Clone)]
pub struct SelectedProverOpeningData<
    'a,
    PointF: Clone,
    G,
    CommitF: Field,
    S = AkitaCommitmentHint<CommitF>,
> {
    selection: OpeningScheduleSelection,
    opening_data: ProverOpeningData<'a, PointF, G, CommitF, S>,
}

impl<'a, PointF: Clone, G, CommitF: Field, S> SelectedProverOpeningData<'a, PointF, G, CommitF, S> {
    /// Exact catalog row identity selected for this complete opening batch.
    pub const fn selection(&self) -> OpeningScheduleSelection {
        self.selection
    }

    pub(crate) fn into_low_level_parts(
        self,
    ) -> (
        OpeningScheduleSelection,
        ProverOpeningData<'a, PointF, G, CommitF, S>,
    ) {
        (self.selection, self.opening_data)
    }
}

#[derive(Debug, Clone)]
struct ProverGroupInput<G, S> {
    state: S,
    group: G,
}

impl<G, S> ProverGroupInput<G, S> {
    fn new(state: S, group: G) -> Self {
        Self { state, group }
    }

    fn state(&self) -> &S {
        &self.state
    }

    fn group(&self) -> &G {
        &self.group
    }
}

fn bind_group_inputs<G, S>(
    states: Vec<S>,
    groups: Vec<G>,
) -> Result<Vec<ProverGroupInput<G, S>>, AkitaError> {
    if states.len() != groups.len() {
        return Err(AkitaError::InvalidInput(
            "prover state and prepared-source counts are misaligned".to_string(),
        ));
    }
    Ok(states
        .into_iter()
        .zip(groups)
        .map(|(state, group)| ProverGroupInput::new(state, group))
        .collect())
}

fn opening_data_from_committed_groups<'a, PointF, G, CommitF, S>(
    opening_claims: OpeningClaims<'a, PointF, CommittedGroup<CommitF>>,
    states: Vec<S>,
    groups: Vec<G>,
) -> Result<ProverOpeningData<'a, PointF, G, CommitF, S>, AkitaError>
where
    PointF: Clone,
    CommitF: Field,
    G: RootProverGroupMeta<CommitF>,
{
    let opening_layout = opening_claims.committed_layout()?;
    if opening_claims.num_groups() != groups.len() {
        return Err(AkitaError::InvalidInput(
            "committed claims and prover source groups are misaligned".into(),
        ));
    }
    for (claims_group, source_group) in opening_claims.groups().iter().zip(&groups) {
        let actual =
            PolynomialGroupLayout::new(source_group.num_vars()?, source_group.num_polynomials());
        if claims_group.commitment().profile().group != actual {
            return Err(AkitaError::InvalidInput(
                "committed group geometry does not match the prover polynomials".into(),
            ));
        }
    }
    let raw_groups = opening_claims
        .groups()
        .iter()
        .map(|group| {
            PolynomialGroupClaims::new(
                group.point().to_vec(),
                group.evaluations().to_vec(),
                group.commitment().commitment().clone(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let group_inputs = bind_group_inputs(states, groups)?;
    let data = ProverOpeningData {
        opening_claims: OpeningClaims::from_groups(raw_groups)?,
        opening_layout,
        group_inputs,
    };
    data.validate_claims()?;
    Ok(data)
}

fn opening_layout_for_groups<PointF: Clone, G, CommitF: Field>(
    opening_claims: &OpeningClaims<'_, PointF, Commitment<CommitF>>,
    groups: &[G],
) -> Result<OpeningClaimsLayout, AkitaError>
where
    G: RootProverGroupMeta<CommitF>,
{
    if opening_claims.num_groups() != groups.len() {
        return Err(AkitaError::InvalidInput(
            "opening claims and prover source groups are misaligned".into(),
        ));
    }
    let layouts = groups
        .iter()
        .map(|group| {
            let num_vars = group.num_vars()?;
            Ok(PolynomialGroupLayout::new(
                num_vars,
                group.num_polynomials(),
            ))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    OpeningClaimsLayout::from_groups(layouts)
}

/// Prover opening input: public claims plus ordered group-local prover material.
///
/// Constructors validate point arities, evaluation counts, and source geometry
/// against the stored layout. Private fields preserve this alignment during proving.
#[derive(Debug, Clone)]
pub struct ProverOpeningData<'a, PointF: Clone, G, CommitF: Field, S = AkitaCommitmentHint<CommitF>>
{
    opening_claims: OpeningClaims<'a, PointF, Commitment<CommitF>>,
    opening_layout: OpeningClaimsLayout,
    group_inputs: Vec<ProverGroupInput<G, S>>,
}

/// Exact, validated commitment state consumed by one ring-relation group.
pub(crate) struct PreparedCommitmentRelationMaterial<F: Field> {
    pub(crate) inner: InnerRelationStateMaterial<F>,
    pub(crate) compression: Option<PortableCompressionState<F>>,
}

/// Private state bound to one recursive opening group.
///
/// Setup-prefix persistence remains portable until its Stage 6 cutover, while
/// the recursively committed witness may remain in its selected resident form.
pub(crate) enum ProverOpeningState<F: Field, S> {
    Portable(AkitaCommitmentHint<F>),
    Selected(S),
}

impl<F, S> InnerRelationState<F> for ProverOpeningState<F, S>
where
    F: Field,
    S: InnerRelationState<F>,
{
    fn preflight_inner_relation(
        &self,
        plan: &crate::compute::CommitInnerPlan,
        source_count: usize,
    ) -> Result<(), AkitaError> {
        match self {
            Self::Portable(hint) => hint.preflight_inner_relation(plan, source_count),
            Self::Selected(state) => state.preflight_inner_relation(plan, source_count),
        }
    }

    fn inner_relation_material(
        &self,
        plan: &crate::compute::CommitInnerPlan,
        source_count: usize,
    ) -> Result<InnerRelationStateMaterial<F>, AkitaError> {
        match self {
            Self::Portable(hint) => hint.inner_relation_material(plan, source_count),
            Self::Selected(state) => state.inner_relation_material(plan, source_count),
        }
    }
}

impl<F, S> OuterCompressionState<F> for ProverOpeningState<F, S>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    S: OuterCompressionState<F>,
{
    fn preflight_outer_compression(
        &self,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<(), AkitaError> {
        match self {
            Self::Portable(hint) => hint.preflight_outer_compression(plan, relation_mode),
            Self::Selected(state) => state.preflight_outer_compression(plan, relation_mode),
        }
    }

    fn outer_compression_material(
        &self,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<PortableCompressionState<F>, AkitaError> {
        match self {
            Self::Portable(hint) => hint.outer_compression_material(plan, relation_mode),
            Self::Selected(state) => state.outer_compression_material(plan, relation_mode),
        }
    }
}

impl<'a, PointF: Clone, P, CommitF: Field, S>
    ProverOpeningData<'a, PointF, PreparedProverGroup<'a, P>, CommitF, S>
where
    P: RootPolyMeta<CommitF>,
{
    fn new_internal(
        opening_claims: OpeningClaims<'a, PointF, Commitment<CommitF>>,
        states: Vec<S>,
        polynomials: Vec<&'a [&'a P]>,
    ) -> Result<Self, AkitaError> {
        let groups = polynomials
            .into_iter()
            .map(PreparedProverGroup::from_refs)
            .collect::<Result<Vec<_>, _>>()?;
        let opening_layout = opening_layout_for_groups(&opening_claims, &groups)?;
        let group_inputs = bind_group_inputs(states, groups)?;
        let data = Self {
            opening_claims,
            opening_layout,
            group_inputs,
        };
        data.validate_claims()?;
        Ok(data)
    }

    /// Bundle public claims with matching prover hints and polynomial groups.
    pub fn new(
        opening_claims: OpeningClaims<'a, PointF, CommittedGroup<CommitF>>,
        states: Vec<S>,
        polynomials: Vec<&'a [&'a P]>,
    ) -> Result<Self, AkitaError> {
        let groups = polynomials
            .into_iter()
            .map(PreparedProverGroup::from_refs)
            .collect::<Result<Vec<_>, _>>()?;
        opening_data_from_committed_groups(opening_claims, states, groups)
    }
}

impl<'a, PointF, CommitF, S, O>
    SelectedProverOpeningData<
        'a,
        PointF,
        ErasedPreparedProverGroup<'a, CommitF, PointF, O>,
        CommitF,
        S,
    >
where
    PointF: Clone
        + FpExtEncoding<CommitF>
        + ExtField<CommitF>
        + MulBaseUnreduced<CommitF>
        + AkitaSerialize,
    CommitF: Field
        + CanonicalEncoding
        + jolt_field::Ring
        + jolt_field::Unreduced
        + AkitaSerialize
        + 'static,
    <CommitF as jolt_field::Unreduced>::Wide: From<CommitF> + jolt_field::AdditiveGroup,
    O: crate::compute::ComputeBackendSetup<CommitF>
        + crate::compute::DigitRowsComputeBackend<CommitF>,
{
    /// Select a catalog row for an ordered batch of already prepared groups.
    pub fn from_prepared_groups<Cfg>(
        opening_claims: OpeningClaims<'a, PointF, CommittedGroup<CommitF>>,
        states: Vec<S>,
        groups: Vec<ErasedPreparedProverGroup<'a, CommitF, PointF, O>>,
        schedules: &TrustedScheduleCatalog<Cfg>,
    ) -> Result<Self, AkitaError>
    where
        Cfg: CommitmentConfig<Field = CommitF, ExtField = PointF>,
    {
        let batch_profile = CommittedGroupBatchProfile::from_ordered_groups(
            opening_claims
                .groups()
                .iter()
                .map(PolynomialGroupClaims::commitment),
        )?;
        let selection = schedules.resolve_profiles(&batch_profile)?.selection();
        let opening_data = opening_data_from_committed_groups(opening_claims, states, groups)?;
        Ok(Self {
            selection,
            opening_data,
        })
    }
}

impl<'a, PointF, P, CommitF, S>
    SelectedProverOpeningData<'a, PointF, PreparedProverGroup<'a, P>, CommitF, S>
where
    PointF: Clone,
    CommitF: Field,
    P: RootPolyMeta<CommitF>,
{
    /// Atomically select the exact batch row before stripping commitment profiles.
    pub fn from_committed_claims<Cfg>(
        opening_claims: OpeningClaims<'a, PointF, CommittedGroup<CommitF>>,
        states: Vec<S>,
        polynomial_groups: Vec<&'a [&'a P]>,
        schedules: &TrustedScheduleCatalog<Cfg>,
    ) -> Result<Self, AkitaError>
    where
        Cfg: CommitmentConfig<Field = CommitF, ExtField = PointF>,
    {
        let batch_profile = CommittedGroupBatchProfile::from_ordered_groups(
            opening_claims
                .groups()
                .iter()
                .map(PolynomialGroupClaims::commitment),
        )?;
        let selection = schedules.resolve_profiles(&batch_profile)?.selection();
        let opening_data = ProverOpeningData::new(opening_claims, states, polynomial_groups)?;
        Ok(Self {
            selection,
            opening_data,
        })
    }
}

#[allow(private_bounds)]
impl<'a, PointF: Clone, G, CommitF: Field, S> ProverOpeningData<'a, PointF, G, CommitF, S>
where
    G: RootProverGroupMeta<CommitF>,
{
    fn validate_claims(&self) -> Result<(), AkitaError> {
        if self.opening_claims.num_groups() != self.group_inputs.len() {
            return Err(AkitaError::InvalidInput(
                "prover opening data group counts are misaligned".to_string(),
            ));
        }
        for (claims, layout) in self
            .opening_claims
            .groups()
            .iter()
            .zip(self.opening_layout.groups())
        {
            if claims.point().len() != layout.num_vars() {
                return Err(AkitaError::InvalidPointDimension {
                    expected: layout.num_vars(),
                    actual: claims.point().len(),
                });
            }
            if claims.evaluations().len() != layout.num_polynomials() {
                return Err(AkitaError::InvalidInput(
                    "prover opening data polynomial/evaluation counts are misaligned".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Largest natural root arity across all polynomial groups.
    pub fn num_vars(&self) -> Result<usize, AkitaError> {
        self.group_inputs
            .iter()
            .map(|input| input.group.num_vars())
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "prover opening data requires at least one polynomial".to_string(),
                )
            })
    }

    /// Public claims carried by this prover input.
    pub fn opening_claims(&self) -> &OpeningClaims<'a, PointF, Commitment<CommitF>> {
        &self.opening_claims
    }

    /// Opening geometry checked against claims and polynomial groups at construction.
    pub fn opening_layout(&self) -> &OpeningClaimsLayout {
        &self.opening_layout
    }

    /// Borrow one prover hint.
    pub fn group_state(&self, index: usize) -> Result<&S, AkitaError> {
        self.group_inputs
            .get(index)
            .map(ProverGroupInput::state)
            .ok_or(AkitaError::InvalidProof)
    }

    /// Prepare each group state once, before transcript mutation, under the
    /// exact relation plan that will consume it.
    pub(crate) fn prepare_commitment_relation_material(
        &self,
        params: &CommittedGroupParams,
        extension_degree: usize,
        mut prepared: Option<(usize, PreparedCommitmentRelationMaterial<CommitF>)>,
    ) -> Result<Vec<PreparedCommitmentRelationMaterial<CommitF>>, AkitaError>
    where
        CommitF: CanonicalEncoding + AkitaSerialize,
        S: InnerRelationState<CommitF> + OuterCompressionState<CommitF>,
    {
        let relation_geometry = akita_types::RelationWitnessGeometry::for_level(
            params,
            &self.opening_layout,
            extension_degree,
        )?;
        let result = (0..self.opening_layout.num_groups())
            .map(|group_index| {
                let group = params.group_params(&self.opening_layout, group_index)?;
                let plan = crate::commitment::CommitmentExecutionPlan::for_root(&group.profile)?;
                let source_count = self
                    .opening_layout
                    .group_layout(group_index)?
                    .num_polynomials();
                if prepared
                    .as_ref()
                    .is_some_and(|(prepared_index, _)| *prepared_index == group_index)
                {
                    let (_, material) = prepared.take().ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "prepared commitment material was consumed more than once".into(),
                        )
                    })?;
                    material.inner.validate(plan.inner(), source_count)?;
                    match (
                        params.payload_mode.is_compressed(),
                        material.compression.as_ref(),
                    ) {
                        (true, Some(compression)) => {
                            let chain = relation_geometry
                                .rhs_layout()
                                .compression_plan_for_group(group_index)?;
                            compression.validate(chain, params.ring_relation_mode)?;
                        }
                        (false, None) => {}
                        _ => {
                            return Err(AkitaError::InvalidInput(
                                "prepared commitment material disagrees with payload mode".into(),
                            ));
                        }
                    }
                    return Ok(material);
                }
                let state = self.group_state(group_index)?;
                let inner = state.inner_relation_material(plan.inner(), source_count)?;
                inner.validate(plan.inner(), source_count)?;
                let compression = if params.payload_mode.is_compressed() {
                    let chain = relation_geometry
                        .rhs_layout()
                        .compression_plan_for_group(group_index)?;
                    let material =
                        state.outer_compression_material(chain, params.ring_relation_mode)?;
                    material.validate(chain, params.ring_relation_mode)?;
                    Some(material)
                } else {
                    None
                };
                Ok(PreparedCommitmentRelationMaterial { inner, compression })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        if prepared.is_some() {
            return Err(AkitaError::InvalidInput(
                "prepared commitment material group index is out of range".into(),
            ));
        }
        Ok(result)
    }

    /// Borrow one polynomial group.
    pub(crate) fn group(&self, index: usize) -> Result<&G, AkitaError> {
        self.group_inputs
            .get(index)
            .map(ProverGroupInput::group)
            .ok_or(AkitaError::InvalidProof)
    }

    /// Commitments in commitment-group order.
    pub fn commitments(&self) -> Vec<&Commitment<CommitF>> {
        self.opening_claims
            .groups()
            .iter()
            .map(PolynomialGroupClaims::commitment)
            .collect()
    }

    /// Bind the actual root commitments and opening points as framed native
    /// public messages. The descriptor separately commits to batch geometry.
    pub(crate) fn append_to_native(
        &self,
        root_params: &CommittedGroupParams,
        grinding: &mut akita_types::NativeProverGrinding<'_>,
    ) -> Result<(), AkitaError>
    where
        CommitF: CanonicalEncoding,
        PointF: ExtField<CommitF>,
    {
        let layout = self.opening_layout();
        let relation_geometry =
            akita_types::RelationWitnessGeometry::for_level(root_params, layout, PointF::DEGREE)?;
        let relation_layout = relation_geometry.rhs_layout();
        for (group_index, commitment) in self.commitments().into_iter().enumerate() {
            let compression = relation_layout.compression_plan_for_group(group_index)?;
            if commitment.rows().coeff_len() != compression.terminal_coefficients() {
                return Err(AkitaError::InvalidInput(
                    "root compressed commitment does not match scheduled root params".into(),
                ));
            }
            let ring_dim = compression
                .maps()
                .last()
                .ok_or(AkitaError::InvalidProof)?
                .ring_dimension();
            let group = u32::try_from(group_index)
                .map_err(|_| AkitaError::InvalidSetup("group index exceeds u32".into()))?;
            akita_transcript::public_native_fields_prover(
                grinding.state_mut(),
                akita_transcript::ProtocolSiteId {
                    family: akita_transcript::SITE_FAMILY_ROOT_STATEMENT,
                    stage: 1,
                    group,
                    detail: u32::try_from(ring_dim).map_err(|_| {
                        AkitaError::InvalidSetup("ring dimension exceeds u32".into())
                    })?,
                    ..akita_transcript::ProtocolSiteId::default()
                },
                commitment.rows().coeffs(),
            )
            .map_err(|_| AkitaError::InvalidProof)?;
        }
        for (group_index, group_claims) in self.opening_claims.groups().iter().enumerate() {
            akita_transcript::public_native_extensions_prover::<CommitF, PointF>(
                grinding.state_mut(),
                akita_transcript::ProtocolSiteId {
                    family: akita_transcript::SITE_FAMILY_ROOT_STATEMENT,
                    stage: 2,
                    group: u32::try_from(group_index)
                        .map_err(|_| AkitaError::InvalidSetup("group index exceeds u32".into()))?,
                    ..akita_transcript::ProtocolSiteId::default()
                },
                group_claims.point(),
            )
            .map_err(|_| AkitaError::InvalidProof)?;
        }
        Ok(())
    }
}

impl<'a, PointF, CommitF, S>
    ProverOpeningData<
        'a,
        PointF,
        PreparedProverGroup<'a, RecursiveFoldSource<CommitF>>,
        CommitF,
        ProverOpeningState<CommitF, S>,
    >
where
    PointF: Field,
    CommitF: Field,
{
    /// Build recursive suffix opening data, with an optional setup-prefix group.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_recursive_suffix_fold(
        opening_point: &[PointF],
        recursive_num_vars: usize,
        setup_prefix_opening: Option<(Vec<PointF>, PointF)>,
        setup_slot: Option<&'a SetupPrefixSlot<CommitF>>,
        setup_polys: Option<&'a [&'a RecursiveFoldSource<CommitF>]>,
        witness_eval: PointF,
        witness_polys: &'a [&'a RecursiveFoldSource<CommitF>],
        witness_commitment: (Commitment<CommitF>, S),
    ) -> Result<Self, AkitaError> {
        if opening_point.len() > recursive_num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: recursive_num_vars,
                actual: opening_point.len(),
            });
        }
        let witness_group = PolynomialGroupClaims::new(
            opening_point.to_vec(),
            vec![witness_eval],
            witness_commitment.0,
        )?;

        match (setup_prefix_opening, setup_slot, setup_polys) {
            (
                Some((setup_prefix_point, setup_prefix_eval)),
                Some(setup_slot),
                Some(setup_polys),
            ) => {
                let setup_commitment_rows =
                    setup_slot.commitment.rows.first().cloned().ok_or_else(|| {
                        AkitaError::InvalidSetup("setup-prefix slot has no commitment rows".into())
                    })?;
                let setup_group = PolynomialGroupClaims::new(
                    setup_prefix_point,
                    vec![setup_prefix_eval],
                    Commitment::new(setup_commitment_rows),
                )?;
                ProverOpeningData::new_internal(
                    OpeningClaims::from_groups(vec![setup_group, witness_group])?,
                    vec![
                        ProverOpeningState::Portable(setup_slot.hint.clone()),
                        ProverOpeningState::Selected(witness_commitment.1),
                    ],
                    vec![setup_polys, witness_polys],
                )
            }
            (None, None, None) => ProverOpeningData::new_internal(
                OpeningClaims::from_groups(vec![witness_group])?,
                vec![ProverOpeningState::Selected(witness_commitment.1)],
                vec![witness_polys],
            ),
            _ => Err(AkitaError::InvalidInput(
                "setup-prefix suffix inputs are incomplete".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_types::{GroupCommitPhaseParams, GroupOpenPhaseParams, RingVec, SisModulusProfileId};
    use jolt_field::{Fp32, Zero};

    type F = Fp32<251>;

    #[derive(Clone)]
    struct MockPoly {
        num_vars: usize,
    }

    impl RootPolyMeta<F> for MockPoly {
        fn num_vars(&self) -> usize {
            self.num_vars
        }
    }

    fn empty_hint() -> AkitaCommitmentHint<F> {
        AkitaCommitmentHint::new(1, Vec::new()).expect("empty test hint")
    }

    fn commitment() -> Commitment<F> {
        Commitment::new(RingVec::from_coeffs(vec![F::zero(); 64]))
    }

    fn synthetic_profile(
        group: PolynomialGroupLayout,
        params: &CommittedGroupParams,
    ) -> GroupCommitPhaseParams {
        GroupCommitPhaseParams {
            version: GroupCommitPhaseParams::VERSION,
            group,

            blocks: akita_types::BlockGeometry::new(
                params.blocks().live_ring_elements_per_claim,
                params.blocks().positions_per_block,
                params.blocks().live_blocks,
            ),

            outer_slice_count: params.outer_slice_count(),
            inner: akita_types::RoleParams::new(
                akita_types::GadgetDigits::new(
                    params.inner().digits.log_basis,
                    params.inner().digits.num_digits,
                ),
                params.inner().matrix,
            ),
            outer: akita_types::RoleParams::new(
                akita_types::GadgetDigits::new(
                    params.outer().digits.log_basis,
                    params.outer().digits.num_digits,
                ),
                params.outer().matrix,
            ),
        }
    }

    fn multi_group_params() -> CommittedGroupParams {
        let pre_layout = PolynomialGroupLayout::new(2, 1);
        let mut pre = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            1,
            1,
            1,
            1,
            SparseChallengeConfig::production_for_ring_dim(64)
                .expect("test ring has a production challenge"),
        )
        .with_decomp(1, 1, 1, 1, 1)
        .expect("precommitted params");
        let inner = &pre.inner().matrix;
        pre.own_group_mut().profile.inner.matrix =
            akita_types::InnerCommitMatrixParams::new_unchecked(
                inner.security_policy(),
                inner
                    .sis_table_key()
                    .expect("test matrix is L infinity")
                    .table_digest,
                inner.sis_modulus_profile(),
                inner.output_rank(),
                inner.input_width(),
                1_484,
                inner.ring_dimension(),
            );
        let outer = &pre.outer().matrix;
        pre.own_group_mut().profile.outer.matrix =
            akita_types::OuterCommitMatrixParams::new_unchecked(
                outer.security_policy(),
                outer.sis_table_key().table_digest,
                outer.sis_modulus_profile(),
                outer.output_rank(),
                outer.input_width(),
                3,
                outer.ring_dimension(),
            );
        let mut root = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            1,
            1,
            1,
            1,
            SparseChallengeConfig::production_for_ring_dim(64)
                .expect("test ring has a production challenge"),
        )
        .with_decomp(1, 1, 1, 1, 1)
        .expect("root params");
        root.insert_precommitted_group(GroupOpenPhaseParams {
            setup_natural_len: None,
            profile: synthetic_profile(pre_layout, &pre),
            opening: akita_types::GroupOpeningPlan::evaluation_trace(
                pre.fold_challenge_config(),
                pre.open().digits.log_basis,
                pre.open().digits.num_digits,
                pre.num_digits_fold(),
            ),
        })
        .unwrap();
        root
    }

    fn multi_group_data<'a>(
        pre_refs: &'a [&'a MockPoly],
        final_refs: &'a [&'a MockPoly],
    ) -> Result<ProverOpeningData<'a, F, PreparedProverGroup<'a, MockPoly>, F>, AkitaError> {
        let layout = OpeningClaimsLayout::from_groups(vec![
            PolynomialGroupLayout::new(2, 1),
            PolynomialGroupLayout::new(4, 2),
        ])
        .expect("fixture layout");
        let geometry =
            akita_types::RelationWitnessGeometry::for_level(&multi_group_params(), &layout, 1)
                .expect("fixture geometry");
        let commitment_for_group = |index| {
            let plan = geometry
                .rhs_layout()
                .compression_plan_for_group(index)
                .expect("fixture compression");
            Commitment::new(RingVec::from_coeffs(vec![
                F::zero();
                plan.terminal_coefficients()
            ]))
        };
        let claims = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(
                vec![F::zero(); 2],
                vec![F::zero()],
                commitment_for_group(0),
            )
            .expect("pre group"),
            PolynomialGroupClaims::new(
                vec![F::zero(); 4],
                vec![F::zero(), F::zero()],
                commitment_for_group(1),
            )
            .expect("final group"),
        ])
        .expect("claims");
        ProverOpeningData::new_internal(
            claims,
            vec![empty_hint(), empty_hint()],
            vec![pre_refs, final_refs],
        )
    }

    #[test]
    fn opening_layout_preserves_precise_group_arities() {
        let pre_poly = MockPoly { num_vars: 2 };
        let final_a = MockPoly { num_vars: 4 };
        let final_b = MockPoly { num_vars: 4 };
        let pre_refs = [&pre_poly];
        let final_refs = [&final_a, &final_b];
        let data = multi_group_data(&pre_refs, &final_refs).expect("prover data");

        let layout = data.opening_layout();

        assert_eq!(
            layout.groups(),
            &[
                PolynomialGroupLayout::new(2, 1),
                PolynomialGroupLayout::new(4, 2)
            ]
        );
    }

    #[test]
    fn construction_rejects_group_arity_mismatch() {
        let pre_poly = MockPoly { num_vars: 3 };
        let final_a = MockPoly { num_vars: 4 };
        let final_b = MockPoly { num_vars: 4 };
        let pre_refs = [&pre_poly];
        let final_refs = [&final_a, &final_b];
        let err = multi_group_data(&pre_refs, &final_refs)
            .err()
            .expect("pre group point vars claim two variables");

        assert!(matches!(
            err,
            AkitaError::InvalidPointDimension {
                expected: 3,
                actual: 2
            }
        ));
    }

    #[test]
    fn construction_rejects_misaligned_claim_material() {
        let poly = MockPoly { num_vars: 2 };
        let refs = [&poly];
        for (evaluations, states, sources) in [(2, 1, 1), (1, 0, 1), (1, 1, 0)] {
            let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                vec![F::zero(); 2],
                vec![F::zero(); evaluations],
                commitment(),
            )
            .expect("claims group")])
            .expect("claims");
            assert!(ProverOpeningData::new_internal(
                claims,
                vec![empty_hint(); states],
                vec![&refs[..]; sources],
            )
            .is_err());
        }
    }
}
