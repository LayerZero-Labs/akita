//! Prover-owned commitment kernels.

use crate::compute::compression::{execute_compression_chains, CompressionExecutionInput};
use crate::compute::{
    tensor_root_projection, CommitInnerPlan, OperationCtx, RootCommitSource, RootPolyMeta,
    RuntimeCommitBackendFor, RuntimeCommitSource, RuntimeRootCommitBackend, RuntimeRootCommitPoly,
    UniformProverStack,
};
use crate::validation::{signed_digit_kernel_for_setup, validate_i8_setup_log_basis};
use crate::RootTensorProjectionPoly;
use akita_config::{ensure_prover_schedule_fits_setup, CommitmentConfig};
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, HalvingField, RandomSampling,
};
use akita_types::{
    dispatch_for_field, root_tensor_projection_enabled, validate_role_dims,
    validate_role_dims_for_field, AkitaCommitmentHint, AkitaExpandedSetup, AkitaScheduleLookupKey,
    Commitment, CommitmentRingDims, CommittedGroup, CommittedGroupParams, CommittedGroupProfile,
    CompressionChainPlan, FpExtEncoding, InnerCommitMatrixParams, OpeningClaimsLayout,
    OuterCommitMatrixParams, PriorGroupProfiles, RingVec,
};
use std::borrow::Cow;

mod inner;
use inner::prepare_inner_commit_group;
pub(crate) use inner::validate_commit_inner_shape;

/// Commitment output plus prover-side hint for one committed polynomial bundle.
///
/// D-free protocol storage: a flat [`Commitment`] plus the semantic A-native
/// inner rows needed when the commitment is opened.
pub(crate) type CommitmentWithHint<F> = (Commitment<F>, AkitaCommitmentHint<F>);

/// Ordered groups committed before the current group.
#[derive(Debug, Clone, Copy)]
pub enum PriorGroupContext<'a> {
    /// The current group has no earlier groups in its opening batch.
    NoPriorGroups,
    /// Exact prior profiles in opening-claim and transcript order.
    WithPriorGroups(&'a PriorGroupProfiles),
}

/// Authority for the current group's commitment parameters.
#[derive(Debug, Clone, Copy)]
pub enum GroupParameterSource<'a> {
    /// Select an existing S or G row from the configured generated catalog.
    Scheduler,
    /// Use caller-supplied root parameters without catalog selection.
    Explicit(&'a CommittedGroupParams),
}

/// Complete context for committing one polynomial group.
#[derive(Debug, Clone, Copy)]
pub struct GroupContext<'a> {
    prior_groups: PriorGroupContext<'a>,
    parameter_source: GroupParameterSource<'a>,
}

impl<'a> GroupContext<'a> {
    /// Select the scalar S row for a group with no prior groups.
    #[must_use]
    pub const fn scheduler_without_prior_groups() -> Self {
        Self {
            prior_groups: PriorGroupContext::NoPriorGroups,
            parameter_source: GroupParameterSource::Scheduler,
        }
    }

    /// Select the grouped G row for a group after exact ordered prior profiles.
    ///
    /// # Errors
    ///
    /// Returns an error when `prior_group_profiles` is empty.
    pub fn scheduler_with_prior_groups(
        prior_group_profiles: &'a PriorGroupProfiles,
    ) -> Result<Self, AkitaError> {
        Self::with_prior_groups(prior_group_profiles, GroupParameterSource::Scheduler)
    }

    /// Use explicit scalar root parameters for a group with no prior groups.
    #[must_use]
    pub const fn explicit_without_prior_groups(params: &'a CommittedGroupParams) -> Self {
        Self {
            prior_groups: PriorGroupContext::NoPriorGroups,
            parameter_source: GroupParameterSource::Explicit(params),
        }
    }

    /// Use explicit grouped root parameters after exact ordered prior profiles.
    ///
    /// # Errors
    ///
    /// Returns an error when `prior_group_profiles` is empty.
    pub fn explicit_with_prior_groups(
        prior_group_profiles: &'a PriorGroupProfiles,
        params: &'a CommittedGroupParams,
    ) -> Result<Self, AkitaError> {
        Self::with_prior_groups(prior_group_profiles, GroupParameterSource::Explicit(params))
    }

    fn with_prior_groups(
        prior_group_profiles: &'a PriorGroupProfiles,
        parameter_source: GroupParameterSource<'a>,
    ) -> Result<Self, AkitaError> {
        if prior_group_profiles.as_slice().is_empty() {
            return Err(AkitaError::InvalidInput(
                "group context requires at least one prior group profile".to_string(),
            ));
        }
        Ok(Self {
            prior_groups: PriorGroupContext::WithPriorGroups(prior_group_profiles),
            parameter_source,
        })
    }
}

/// Result of committing one polynomial group.
#[derive(Debug)]
pub struct CommitOutput<F: FieldCore> {
    /// Self-describing committed group.
    pub committed_group: CommittedGroup<F>,
    /// Prover-only opening hint.
    pub hint: AkitaCommitmentHint<F>,
}

impl<F: FieldCore> CommitOutput<F> {
    /// Consume the named result into its committed group and prover hint.
    pub fn into_parts(self) -> (CommittedGroup<F>, AkitaCommitmentHint<F>) {
        (self.committed_group, self.hint)
    }
}

#[derive(Clone, Copy)]
struct CommitmentGeometry<'a> {
    context: &'static str,
    num_positions_per_block: usize,
    num_live_blocks: usize,
    log_basis_inner: u32,
    num_digits_inner: usize,
    inner_matrix: &'a InnerCommitMatrixParams,
    log_basis_outer: u32,
    num_digits_outer: usize,
    outer_matrix: &'a OuterCommitMatrixParams,
}

impl<'a> From<&'a CommittedGroupParams> for CommitmentGeometry<'a> {
    fn from(params: &'a CommittedGroupParams) -> Self {
        Self {
            context: "commit params",
            num_positions_per_block: params.num_positions_per_block,
            num_live_blocks: params.num_live_blocks,
            log_basis_inner: params.log_basis_inner,
            num_digits_inner: params.num_digits_inner,
            inner_matrix: &params.inner_commit_matrix,
            log_basis_outer: params.log_basis_outer,
            num_digits_outer: params.num_digits_outer,
            outer_matrix: &params.outer_commit_matrix,
        }
    }
}

fn commit_only_setup_field_elements(geometry: CommitmentGeometry<'_>) -> Result<usize, AkitaError> {
    let matrix_fields = |role: &str, output_rank: usize, input_width: usize, ring_d: usize| {
        output_rank
            .checked_mul(input_width)
            .and_then(|elements| elements.checked_mul(ring_d))
            .ok_or_else(|| AkitaError::InvalidSetup(format!("{role} setup footprint overflow")))
    };
    let a_fields = matrix_fields(
        "A",
        geometry.inner_matrix.output_rank(),
        geometry.inner_matrix.input_width(),
        geometry.inner_matrix.ring_dimension(),
    )?;
    let b_fields = matrix_fields(
        "B",
        geometry.outer_matrix.output_rank(),
        geometry.outer_matrix.input_width(),
        geometry.outer_matrix.ring_dimension(),
    )?;
    let b_output_coefficients = geometry
        .outer_matrix
        .output_rank()
        .checked_mul(geometry.outer_matrix.ring_dimension())
        .ok_or_else(|| AkitaError::InvalidSetup("B output width overflow".to_string()))?;
    let compression_fields = CompressionChainPlan::for_complete_source(
        geometry.outer_matrix.sis_table_key().modulus_profile,
        b_output_coefficients,
    )?
    .max_setup_field_elements()?;
    Ok(a_fields.max(b_fields).max(compression_fields))
}

fn validate_commitment_geometry<F>(
    geometry: CommitmentGeometry<'_>,
    setup: &AkitaExpandedSetup<F>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    signed_digit_kernel_for_setup(
        geometry.log_basis_inner,
        "for signed witness commitment decomposition",
    )?;
    validate_i8_setup_log_basis(
        geometry.log_basis_outer,
        "for i8 outer commitment decomposition",
    )?;

    // A/B geometry is independent of the D/opening matrix. Mirroring B into
    // the opening slot lets the shared role validator enforce only the two
    // dimensions represented by this borrowed view.
    let dims = CommitmentRingDims {
        inner: geometry.inner_matrix.ring_dimension(),
        outer: geometry.outer_matrix.ring_dimension(),
        opening: geometry.outer_matrix.ring_dimension(),
    };
    validate_role_dims(dims)?;
    validate_role_dims_for_field::<F>(dims)?;

    let expected_a_width = geometry
        .num_positions_per_block
        .checked_mul(geometry.num_digits_inner)
        .ok_or_else(|| AkitaError::InvalidSetup("A commit width overflow".to_string()))?;
    if geometry.inner_matrix.input_width() != expected_a_width {
        return Err(AkitaError::InvalidSetup(format!(
            "{} A width {} does not match num_positions_per_block * num_digits_inner = {expected_a_width}",
            geometry.context,
            geometry.inner_matrix.input_width()
        )));
    }
    if geometry.outer_matrix.input_width() == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "{} requires nonzero B width, got B={}",
            geometry.context,
            geometry.outer_matrix.input_width()
        )));
    }

    let required = commit_only_setup_field_elements(geometry)?;
    let available = setup.shared_matrix.num_field_elements();
    if required > available {
        return Err(AkitaError::InvalidSetup(format!(
            "{} requires {required} setup field elements for commitment, but setup has {available}",
            geometry.context
        )));
    }
    Ok(())
}

pub(crate) fn validate_commit_level_params<F>(
    params: &CommittedGroupParams,
    setup: &AkitaExpandedSetup<F>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if params.num_live_blocks == 0 || params.num_positions_per_block == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit params require nonzero num_live_blocks and num_positions_per_block".to_string(),
        ));
    }
    if params.num_digits_inner == 0 || params.num_digits_outer == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit params require nonzero A/B digit depths".to_string(),
        ));
    }
    validate_commitment_geometry::<F>(params.into(), setup)?;

    // D/opening geometry is level-only: standalone commitment profiles freeze
    // only the A/B matrices used to materialize the commitment.
    if params.num_digits_open == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit params require nonzero opening digit depth".to_string(),
        ));
    }
    validate_i8_setup_log_basis(params.log_basis_open, "for i8 opening decomposition")?;
    let dims = params.role_dims();
    validate_role_dims(dims)?;
    validate_role_dims_for_field::<F>(dims)?;
    if params.open_commit_matrix.input_width() == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params require nonzero D width, got D={}",
            params.open_commit_matrix.input_width()
        )));
    }
    // Commitment materialization uses only A and B. In particular, a
    // standalone group extracted from an approved multi-group row may retain
    // that row's shared D geometry, which is consumed only if the group later
    // participates in the selected opening schedule. Charging D here would
    // reject a setup that exactly fits the standalone commitment profile.
    Ok(())
}

pub(crate) fn validate_commit_outer_input_nonempty(active_len: usize) -> Result<(), AkitaError> {
    if active_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit B input must be nonempty".to_string(),
        ));
    }
    Ok(())
}

/// Validate a singleton commitment request against prover setup capacity.
///
/// # Errors
///
/// Returns an error if the request is empty, mixes polynomial dimensions, or
/// exceeds the prover setup capacity.
pub fn prepare_commit_inputs<F, P>(
    polys: &[P],
    setup: &AkitaExpandedSetup<F>,
) -> Result<OpeningClaimsLayout, AkitaError>
where
    F: FieldCore,
    P: RootPolyMeta<F>,
{
    if polys.is_empty() {
        return Err(AkitaError::InvalidInput(
            "commit requires at least one polynomial".to_string(),
        ));
    }
    let num_vars = polys[0].num_vars();
    if polys.iter().any(|p| p.num_vars() != num_vars) {
        return Err(AkitaError::InvalidInput(
            "all polynomials in a batched commit must have the same num_vars".to_string(),
        ));
    }
    if polys.len() > setup.seed.max_num_batched_polys {
        return Err(AkitaError::InvalidInput(format!(
            "commit received {} polynomials but setup supports at most {}",
            polys.len(),
            setup.seed.max_num_batched_polys
        )));
    }
    if num_vars > setup.seed.max_num_vars {
        return Err(AkitaError::InvalidInput(format!(
            "commit received a polynomial with {} variables but setup supports at most {}",
            num_vars, setup.seed.max_num_vars
        )));
    }

    OpeningClaimsLayout::new(num_vars, polys.len())
}

#[cfg(test)]
fn checked_commit_b_input_len(total_polys: usize, per_poly: usize) -> Result<usize, AkitaError> {
    total_polys.checked_mul(per_poly).ok_or_else(|| {
        AkitaError::InvalidInput(format!(
            "commit B digit input length overflow for {total_polys} polynomials with {per_poly} digits each"
        ))
    })
}

/// A-role root tensor projection at `transform_ring_d` when the schedule calls for it.
fn tensor_project_roots<F, P, E, B>(
    transform_ring_d: usize,
    tensor_ctx: &OperationCtx<'_, F, B>,
    polys: &[P],
) -> Result<Vec<RootTensorProjectionPoly<F>>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: FpExtEncoding<F>,
    P: RuntimeRootCommitPoly<F>,
    B: RuntimeRootCommitBackend<F, P, E>,
{
    let backend = tensor_ctx.backend();
    let prepared = tensor_ctx.prepared();
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        transform_ring_d,
        |D| {
            polys
                .iter()
                .map(|poly| tensor_root_projection::<F, P, E, B, D>(backend, Some(prepared), poly))
                .collect()
        }
    )
}

fn commit_with_validated_geometry<F, P, B>(
    polys: &[P],
    ctx: &OperationCtx<'_, F, B>,
    geometry: CommitmentGeometry<'_>,
) -> Result<CommitmentWithHint<F>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + FromPrimitiveInt
        + HalvingField
        + HasWide
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RuntimeCommitSource<F>,
    B: RuntimeCommitBackendFor<F, P>,
{
    let backend = ctx.backend();
    let prepared = ctx.prepared();
    // Per-role ring dimensions for this level: the inner commit digits are
    // A-role data, the outer `B·t̂` rows are B-role data. The mixed-row spec
    // feeds diverging dims here (uniform today).
    let dims = CommitmentRingDims {
        inner: geometry.inner_matrix.ring_dimension(),
        outer: geometry.outer_matrix.ring_dimension(),
        opening: geometry.outer_matrix.ring_dimension(),
    };
    let plan = CommitInnerPlan {
        n_a: geometry.inner_matrix.output_rank(),
        num_positions_per_block: geometry.num_positions_per_block,
        num_digits_inner: geometry.num_digits_inner,
        log_basis_inner: geometry.log_basis_inner,
    };
    let num_live_blocks = geometry.num_live_blocks;
    let num_digits_open = geometry.num_digits_outer;
    let log_basis = geometry.log_basis_outer;
    let n_b = geometry.outer_matrix.output_rank();
    let (commitment, inner_rows, compression_witness, compression_quotients) = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        dims.d_a(),
        |D_A| {
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Outer),
                F,
                dims.d_b(),
                |D_B| {
                    // The whole group multiplies the same A matrix, so the
                    // backend can stream it once across every polynomial.
                    let views = polys
                        .iter()
                        .map(|poly| RootCommitSource::<F, D_A>::commit_view(poly))
                        .collect::<Result<Vec<_>, _>>()?;
                    let prepared_polynomials = prepare_inner_commit_group::<F, _, _, D_A, D_B>(
                        backend,
                        prepared,
                        views,
                        plan,
                        num_live_blocks,
                        num_digits_open,
                        log_basis,
                    )?;
                    let total_planes =
                        prepared_polynomials
                            .iter()
                            .try_fold(0usize, |total, (_, digits)| {
                                total.checked_add(digits.total_planes()).ok_or_else(|| {
                                    AkitaError::InvalidSetup(
                                        "commit B input plane count overflow".to_string(),
                                    )
                                })
                            })?;
                    validate_commit_outer_input_nonempty(total_planes)?;
                    let mut b_input_digits = Vec::with_capacity(total_planes);
                    for (_, digits) in &prepared_polynomials {
                        b_input_digits.extend_from_slice(digits.typed_planes::<D_B>()?);
                    }
                    let u = backend.digit_rows::<D_B>(prepared, n_b, &b_input_digits, log_basis)?;
                    if u.len() != n_b {
                        return Err(AkitaError::InvalidSetup(format!(
                            "backend returned {} B commitment rows, expected {n_b}",
                            u.len(),
                        )));
                    }
                    let source = RingVec::from_ring_elems(&u);
                    let plan = CompressionChainPlan::for_complete_source(
                        geometry.outer_matrix.sis_table_key().modulus_profile,
                        source.coeff_len(),
                    )?;
                    let (mut outputs, _) = execute_compression_chains(
                        ctx,
                        vec![CompressionExecutionInput {
                            id: (),
                            plan,
                            coefficients: source.into_coeffs(),
                        }],
                    )?;
                    let output = outputs.pop().ok_or(AkitaError::InvalidProof)?;
                    let terminal_ring_dim = output
                        .witness
                        .plan()
                        .maps()
                        .last()
                        .ok_or(AkitaError::InvalidProof)?
                        .ring_dimension();
                    let payload = RingVec::from_coeffs_with_ring_dim(
                        output.terminal.coefficients().to_vec(),
                        terminal_ring_dim,
                    )?;
                    Ok::<_, AkitaError>((
                        Commitment::new(payload),
                        prepared_polynomials
                            .into_iter()
                            .map(|(rows, _)| rows)
                            .collect::<Vec<_>>(),
                        output.witness,
                        output.quotients,
                    ))
                }
            )
        }
    )?;
    let hint = AkitaCommitmentHint::new_with_outer_compression(
        dims.d_a(),
        inner_rows,
        &compression_witness,
        &compression_quotients,
    )?;
    Ok((commitment, hint))
}

fn validate_explicit_context<F>(
    group_layout: akita_types::PolynomialGroupLayout,
    prior_groups: PriorGroupContext<'_>,
    params: &CommittedGroupParams,
    expanded: &AkitaExpandedSetup<F>,
) -> Result<CommittedGroupProfile, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    validate_commit_level_params::<F>(params, expanded)?;

    match prior_groups {
        PriorGroupContext::NoPriorGroups => {
            params.require_scalar_level("explicit commitment")?;
        }
        PriorGroupContext::WithPriorGroups(prior_group_profiles) => {
            if params.setup_prefix.is_some() {
                return Err(AkitaError::InvalidSetup(
                    "explicit grouped root params must not contain a setup-prefix group"
                        .to_string(),
                ));
            }
            let profiles = prior_group_profiles.as_slice();
            if params.precommitted_groups.len() != profiles.len() {
                return Err(AkitaError::InvalidSetup(format!(
                    "explicit grouped root params contain {} prior groups, expected {}",
                    params.precommitted_groups.len(),
                    profiles.len(),
                )));
            }
            for (index, (group, profile)) in
                params.precommitted_groups.iter().zip(profiles).enumerate()
            {
                if group.layout != *profile {
                    return Err(AkitaError::InvalidSetup(format!(
                        "explicit grouped root prior profile {index} does not match its params"
                    )));
                }
            }
            let prior_layouts = profiles
                .iter()
                .map(|profile| profile.group)
                .collect::<Vec<_>>();
            let opening_layout =
                OpeningClaimsLayout::from_root_groups(&prior_layouts, group_layout)?;
            params.validate_opening_batch(&opening_layout)?;
        }
    }

    let profile = CommittedGroupProfile::from_params(group_layout, params);
    profile.validate(F::modulus_bits())?;
    profile.validate_root_geometry()?;
    Ok(profile)
}

/// Commit one homogeneous polynomial group in its complete parameter context.
///
/// Scheduler contexts select an existing S or G catalog row. Explicit
/// contexts validate caller-supplied root parameters without catalog lookup.
/// Tensor projection is determined solely from field/root geometry. Geometry
/// validation, commitment arithmetic, and result assembly are shared by every
/// context.
///
/// # Errors
///
/// Returns an error for an empty or mixed-arity group, unsupported role
/// parameters, insufficient setup, or commitment execution failure.
pub fn commit<Cfg, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    stack: &UniformProverStack<'_, Cfg::Field, B>,
    context: GroupContext<'_>,
) -> Result<CommitOutput<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + FromPrimitiveInt
        + HalvingField
        + HasWide
        + 'static,
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    P: RuntimeRootCommitPoly<Cfg::Field>,
    B: RuntimeRootCommitBackend<Cfg::Field, P, Cfg::ExtField>,
{
    let opening_layout = prepare_commit_inputs::<Cfg::Field, P>(polys, expanded)?;
    let group_layout = opening_layout.root_final_group_layout()?;

    let (params, profile): (Cow<'_, CommittedGroupParams>, CommittedGroupProfile) =
        match (context.prior_groups, context.parameter_source) {
            (PriorGroupContext::NoPriorGroups, GroupParameterSource::Scheduler) => {
                let row = Cfg::select_schedule_for_opening(&opening_layout)?;
                let params = &row.schedule().root.params.final_group.commitment;
                validate_commit_level_params::<Cfg::Field>(params, expanded)?;
                let row_profile = row.profiles().final_group;
                if row_profile.group != group_layout
                    || row_profile != CommittedGroupProfile::from_params(group_layout, params)
                {
                    return Err(AkitaError::InvalidSetup(
                    "scalar S row profile does not match its requested layout and root parameters"
                        .to_string(),
                ));
                }
                (
                    Cow::Owned(row.into_schedule().root.params.final_group.commitment),
                    row_profile,
                )
            }
            (
                PriorGroupContext::WithPriorGroups(prior_group_profiles),
                GroupParameterSource::Scheduler,
            ) => {
                let key = AkitaScheduleLookupKey {
                    final_group: group_layout,
                    prior_group_profiles: prior_group_profiles.as_slice().to_vec(),
                };
                let row = Cfg::select_schedule_for_key(&key)?;
                ensure_prover_schedule_fits_setup::<Cfg>(
                    expanded,
                    row.schedule(),
                    &key.opening_layout()?,
                )?;
                let params = &row.schedule().root.params.final_group.commitment;
                validate_commit_level_params::<Cfg::Field>(params, expanded)?;
                let row_profile = row.profiles().final_group;
                (
                    Cow::Owned(row.into_schedule().root.params.final_group.commitment),
                    row_profile,
                )
            }
            (prior_groups, GroupParameterSource::Explicit(params)) => {
                let profile = validate_explicit_context::<Cfg::Field>(
                    group_layout,
                    prior_groups,
                    params,
                    expanded,
                )?;
                (Cow::Borrowed(params), profile)
            }
        };

    let geometry: CommitmentGeometry<'_> = params.as_ref().into();
    let transform_ring_d = geometry.inner_matrix.ring_dimension();
    let (commitment, hint) = if root_tensor_projection_enabled::<Cfg::Field, Cfg::ExtField>(
        transform_ring_d,
        group_layout.num_vars(),
    ) {
        let transformed = tensor_project_roots::<Cfg::Field, P, Cfg::ExtField, B>(
            transform_ring_d,
            stack.tensor(),
            polys,
        )?;
        commit_with_validated_geometry::<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>, B>(
            &transformed,
            stack.commit(),
            geometry,
        )?
    } else {
        commit_with_validated_geometry::<Cfg::Field, P, B>(polys, stack.commit(), geometry)?
    };

    Ok(CommitOutput {
        committed_group: CommittedGroup::new(profile, commitment),
        hint,
    })
}

#[cfg(test)]
mod tests;
