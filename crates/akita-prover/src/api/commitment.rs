//! Prover-owned commitment kernels.

use crate::compute::compression::{execute_compression_chains, CompressionExecutionInput};
use crate::compute::{
    CommitInnerPlan, OperationCtx, RootCommitSource, RootPolyMeta, RuntimeCommitBackendFor,
    RuntimeCommitSource, UniformProverStack,
};
use crate::validation::{signed_digit_kernel_for_setup, validate_i8_setup_log_basis};
use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_config::{ensure_prover_schedule_fits_setup, CommitmentConfig};
#[cfg(test)]
use akita_error::checked;
use akita_error::AkitaError;
use akita_types::sis::CommittedSourceContract;
use akita_types::{
    dispatch_for_field, validate_role_dims, validate_role_dims_for_field, AkitaCommitmentHint,
    AkitaExpandedSetup, AkitaScheduleLookupKey, Commitment, CommitmentRingDims,
    CommitmentSliceCount, CommittedGroup, CommittedGroupParams, CommittedSourceEncoding,
    CompressionChainPlan, FpExtEncoding, GroupCommitPhaseParams, InnerCommitMatrixParams,
    OpeningClaimsLayout, OuterCommitMatrixParams, PrecommittedGroupProfiles, RingVec,
};
use jolt_field::Unreduced;
use jolt_field::{CanonicalEncoding, Field, Ring};

mod inner;
#[cfg(test)]
use inner::outer_slice_inputs;
use inner::prepare_inner_commit_group;
pub(crate) use inner::validate_commit_inner_shape;
pub(crate) use inner::{commit_outer_slices, for_each_outer_slice_input};

/// Commitment output plus prover-side hint for one committed polynomial bundle.
///
/// D-free protocol storage: a flat [`Commitment`] plus the semantic A-native
/// inner rows needed when the commitment is opened.
pub(crate) type CommitmentWithHint<F> = (Commitment<F>, AkitaCommitmentHint<F>);

/// Ordered groups committed before the current group.
#[derive(Debug, Clone, Copy)]
enum PrecommittedGroupContext<'a> {
    /// The current group has no earlier groups in its opening batch.
    NoPrecommittedGroups,
    /// Exact precommitted profiles in opening-claim and transcript order.
    WithPrecommittedGroups(&'a PrecommittedGroupProfiles),
}

impl PrecommittedGroupContext<'_> {
    /// Borrow the ordered precommitted profiles, empty when there are none.
    fn as_slice(&self) -> &[GroupCommitPhaseParams] {
        match self {
            Self::NoPrecommittedGroups => &[],
            Self::WithPrecommittedGroups(profiles) => profiles.as_slice(),
        }
    }
}

/// Authority for the current group's commitment parameters.
#[derive(Debug, Clone, Copy)]
enum GroupParameterSource<'a> {
    /// Select an existing scalar or grouped row from the generated catalog.
    Scheduler,
    /// Use caller-supplied root parameters without catalog selection.
    Explicit(&'a CommittedGroupParams),
}

/// Complete context for committing one polynomial group.
#[derive(Debug, Clone, Copy)]
pub struct GroupContext<'a> {
    precommitted_groups: PrecommittedGroupContext<'a>,
    parameter_source: GroupParameterSource<'a>,
}

impl<'a> GroupContext<'a> {
    /// Select the scalar row, the generated row for a group with no precommitted groups.
    #[must_use]
    pub const fn scheduler_without_precommitted_groups() -> Self {
        Self {
            precommitted_groups: PrecommittedGroupContext::NoPrecommittedGroups,
            parameter_source: GroupParameterSource::Scheduler,
        }
    }

    /// Select the grouped row keyed on these exact ordered precommitted profiles.
    #[must_use]
    pub const fn scheduler_with_precommitted_groups(
        precommitteds: &'a PrecommittedGroupProfiles,
    ) -> Self {
        Self {
            precommitted_groups: PrecommittedGroupContext::WithPrecommittedGroups(precommitteds),
            parameter_source: GroupParameterSource::Scheduler,
        }
    }

    /// Use explicit scalar root parameters for a group with no precommitted groups.
    #[must_use]
    pub const fn explicit_without_precommitted_groups(params: &'a CommittedGroupParams) -> Self {
        Self {
            precommitted_groups: PrecommittedGroupContext::NoPrecommittedGroups,
            parameter_source: GroupParameterSource::Explicit(params),
        }
    }

    /// Use explicit grouped root parameters after exact ordered precommitted profiles.
    #[must_use]
    pub const fn explicit_with_precommitted_groups(
        precommitteds: &'a PrecommittedGroupProfiles,
        params: &'a CommittedGroupParams,
    ) -> Self {
        Self {
            precommitted_groups: PrecommittedGroupContext::WithPrecommittedGroups(precommitteds),
            parameter_source: GroupParameterSource::Explicit(params),
        }
    }
}

/// Result of committing one polynomial group.
#[derive(Debug)]
pub struct CommitOutput<F: Field> {
    /// Self-describing committed group.
    pub committed_group: CommittedGroup<F>,
    /// Prover-only opening hint.
    pub hint: AkitaCommitmentHint<F>,
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
    outer_slice_count: CommitmentSliceCount,
}

impl<'a> From<&'a CommittedGroupParams> for CommitmentGeometry<'a> {
    fn from(params: &'a CommittedGroupParams) -> Self {
        Self {
            context: "commit params",
            num_positions_per_block: params.blocks().positions_per_block,
            num_live_blocks: params.blocks().live_blocks,
            log_basis_inner: params.inner().digits.log_basis,
            num_digits_inner: params.inner().digits.num_digits,
            inner_matrix: &params.inner().matrix,
            log_basis_outer: params.outer().digits.log_basis,
            num_digits_outer: params.outer().digits.num_digits,
            outer_matrix: &params.outer().matrix,
            outer_slice_count: params.outer_slice_count(),
        }
    }
}

fn validate_commitment_geometry<F>(
    geometry: CommitmentGeometry<'_>,
    setup: &AkitaExpandedSetup<F>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
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

    let required = akita_types::commit_only_setup_field_elements(
        geometry.inner_matrix,
        geometry.outer_matrix,
        geometry.outer_slice_count,
    )?;
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
    fold_level: usize,
    num_polynomials: usize,
) -> Result<akita_types::CommitmentSliceGeometry, AkitaError>
where
    F: Field + CanonicalEncoding,
{
    let slice_geometry = params.validate_commitment_request(fold_level, num_polynomials)?;
    if params.blocks().live_blocks == 0 || params.blocks().positions_per_block == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit params require nonzero num_live_blocks and num_positions_per_block".to_string(),
        ));
    }
    if params.inner().digits.num_digits == 0 || params.outer().digits.num_digits == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit params require nonzero A/B digit depths".to_string(),
        ));
    }
    validate_commitment_geometry::<F>(params.into(), setup)?;

    // D/opening geometry is level-only: standalone commitment profiles freeze
    // only the A/B matrices used to materialize the commitment.
    if params.open().digits.num_digits == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit params require nonzero opening digit depth".to_string(),
        ));
    }
    validate_i8_setup_log_basis(
        params.open().digits.log_basis,
        "for i8 opening decomposition",
    )?;
    let dims = params.role_dims();
    validate_role_dims(dims)?;
    validate_role_dims_for_field::<F>(dims)?;
    if params.open().matrix.input_width() == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params require nonzero D width, got D={}",
            params.open().matrix.input_width()
        )));
    }
    // Commitment materialization uses only A and B. In particular, a
    // standalone group extracted from an approved multi-group row may retain
    // that row's shared D geometry, which is consumed only if the group later
    // participates in the selected opening schedule. Charging D here would
    // reject a setup that exactly fits the standalone commitment profile.
    Ok(slice_geometry)
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
    F: Field,
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
    let group = akita_types::PolynomialGroupLayout::new(num_vars, polys.len());
    OpeningClaimsLayout::from_groups(vec![group])
}

#[cfg(test)]
fn checked_commit_b_input_len(total_polys: usize, per_poly: usize) -> Result<usize, AkitaError> {
    checked::product([total_polys, per_poly]).ok_or_else(|| {
        AkitaError::InvalidInput(format!(
            "commit B digit input length overflow for {total_polys} polynomials with {per_poly} digits each"
        ))
    })
}

/// Reject a group whose logical source representation differs from the class
/// whose honest-response bounds the schedule uses.
fn ensure_sources_match_declared_class<F, P>(
    polys: &[P],
    contract: CommittedSourceContract,
) -> Result<(), AkitaError>
where
    F: Field,
    P: RootPolyMeta<F>,
{
    let Some(required_chunk_size) = contract.class().required_onehot_chunk_size() else {
        return Ok(());
    };
    for poly in polys {
        match RootPolyMeta::<F>::onehot_chunk_size(poly) {
            Some(chunk_size) if chunk_size == required_chunk_size => {}
            Some(chunk_size) => {
                return Err(AkitaError::InvalidInput(format!(
                    "committed source is a unit one-hot representation with chunk size \
                     {chunk_size}, but this schedule is priced for one hot position per \
                     {required_chunk_size} coefficients"
                )))
            }
            None => {
                return Err(AkitaError::InvalidInput(format!(
                    "committed source is not a unit one-hot representation, but this schedule \
                     is priced for one hot position per {required_chunk_size} coefficients; \
                     a dense source can satisfy the digit envelope while carrying far more \
                     energy than the frozen response caps allow"
                )))
            }
        }
    }
    Ok(())
}

/// Reject coefficients outside the intersection of the source declaration and
/// the exact balanced-digit interval committed by this row.
fn ensure_sources_fit_accepted_interval<F, P, const D: usize>(
    polys: &[P],
    plan: CommitInnerPlan,
    contract: CommittedSourceContract,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    P: RootCommitSource<F, D>,
{
    let modulus = (-F::one())
        .to_u128_checked()
        .expect("Akita field element must fit in u128")
        + 1;
    let threshold =
        decompose_centering_threshold(plan.num_digits_inner, plan.log_basis_inner, modulus);
    let (negative_reach, positive_reach) =
        contract.accepted_bounds(plan.log_basis_inner, plan.num_digits_inner);
    let exceeds = |negative_abs: u128, positive: u128| {
        negative_reach.is_some_and(|reach| negative_abs > reach)
            || positive_reach.is_some_and(|reach| positive > reach)
    };
    if !exceeds(modulus.saturating_sub(threshold + 1), threshold) {
        return Ok(());
    }
    let render_reach = |reach: Option<u128>| match reach {
        Some(value) => value.to_string(),
        None => ">2^128".to_string(),
    };
    for poly in polys {
        let (negative_abs, positive) =
            RootCommitSource::<F, D>::committed_centered_reach(poly, modulus, threshold)?;
        if exceeds(negative_abs, positive) {
            return Err(AkitaError::InvalidInput(format!(
                "committed source exceeds the scheduled bound: centered coefficients reach \
                 [-{negative_abs}, {positive}] but a source declared at \
                 log_commit_bound = {} and committed as {} balanced base-2^{} digits accepts \
                 only [-{}, {}]",
                contract.decomposition().log_commit_bound,
                plan.num_digits_inner,
                plan.log_basis_inner,
                render_reach(negative_reach),
                render_reach(positive_reach),
            )));
        }
    }
    Ok(())
}

fn commit_with_validated_geometry<F, P, B>(
    polys: &[P],
    ctx: &OperationCtx<'_, F, B>,
    geometry: CommitmentGeometry<'_>,
    slice_geometry: &akita_types::CommitmentSliceGeometry,
    contract: CommittedSourceContract,
) -> Result<CommitmentWithHint<F>, AkitaError>
where
    F: Field + CanonicalEncoding + Ring + Unreduced + 'static,
    <F as Unreduced>::Wide: From<F>,
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
            ensure_sources_fit_accepted_interval::<F, P, D_A>(polys, plan, contract)?;
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
                    let u = commit_outer_slices::<F, _, D_B>(
                        backend,
                        prepared,
                        n_b,
                        prepared_polynomials.iter().map(|(_, digits)| digits),
                        slice_geometry,
                        log_basis,
                    )?;
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
                        output.terminal.into_coefficients(),
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

fn validate_explicit_context(
    group_layout: akita_types::PolynomialGroupLayout,
    precommitted_groups: PrecommittedGroupContext<'_>,
    params: &CommittedGroupParams,
) -> Result<GroupCommitPhaseParams, AkitaError> {
    match precommitted_groups {
        PrecommittedGroupContext::NoPrecommittedGroups => {
            params.require_scalar_level("explicit commitment")?;
        }
        PrecommittedGroupContext::WithPrecommittedGroups(precommitteds) => {
            if params.setup_prefix().is_some() {
                return Err(AkitaError::InvalidSetup(
                    "explicit grouped root params must not contain a setup-prefix group"
                        .to_string(),
                ));
            }
            let profiles = precommitteds.as_slice();
            if params.precommitted_groups().len() != profiles.len() {
                return Err(AkitaError::InvalidSetup(format!(
                    "explicit grouped root params contain {} precommitted groups, expected {}",
                    params.precommitted_groups().len(),
                    profiles.len(),
                )));
            }
            for (index, (group, profile)) in params
                .precommitted_groups()
                .iter()
                .zip(profiles)
                .enumerate()
            {
                if group.profile != *profile {
                    return Err(AkitaError::InvalidSetup(format!(
                        "explicit grouped root precommitted profile {index} does not match its params"
                    )));
                }
            }
            let precommitted_layouts = profiles
                .iter()
                .map(|profile| profile.group)
                .collect::<Vec<_>>();
            let opening_layout =
                OpeningClaimsLayout::from_root_groups(&precommitted_layouts, group_layout)?;
            params.validate_opening_batch(&opening_layout)?;
        }
    }

    GroupCommitPhaseParams::try_from_params(group_layout, params)
}

/// Commit one homogeneous polynomial group in its complete parameter context.
///
/// Scheduler contexts select an existing S or G catalog row. Explicit
/// contexts validate caller-supplied root parameters without catalog lookup.
/// Root commitments always consume the canonical coefficient table.
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
    Cfg::Field: Field + CanonicalEncoding + Ring + Unreduced + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    P: RuntimeCommitSource<Cfg::Field>,
    B: RuntimeCommitBackendFor<Cfg::Field, P>,
{
    let opening_layout = prepare_commit_inputs::<Cfg::Field, P>(polys, expanded)?;
    let group_layout = opening_layout.root_final_group_layout()?;

    let scheduled_row;
    let (params, profile): (&CommittedGroupParams, GroupCommitPhaseParams) =
        if let GroupParameterSource::Explicit(params) = context.parameter_source {
            let profile =
                validate_explicit_context(group_layout, context.precommitted_groups, params)?;
            (params, profile)
        } else {
            let key = AkitaScheduleLookupKey {
                final_group: group_layout,
                precommitteds: context.precommitted_groups.as_slice().to_vec(),
            };
            scheduled_row = Cfg::resolve_catalog_row_for_key(&key)?;

            // A group with precommitted groups is the final group of the batch this
            // row opens, so the setup must carry the row's whole schedule. A
            // group without precommitted groups may instead be opened later under a
            // grouped row, so it is admitted on its own A/B footprint alone.
            if matches!(
                context.precommitted_groups,
                PrecommittedGroupContext::WithPrecommittedGroups(_)
            ) {
                ensure_prover_schedule_fits_setup::<Cfg>(
                    expanded,
                    scheduled_row.schedule(),
                    &key.opening_layout()?,
                )?;
            }

            // `audit_resolved_schedule` already proved this row's profile
            // agrees with its parameters, so no re-derivation happens here.
            let params = &scheduled_row.schedule().root.params;
            (params, scheduled_row.profiles().final_group)
        };

    let contract = Cfg::committed_source_contract()?;
    ensure_sources_match_declared_class::<Cfg::Field, P>(polys, contract)?;

    let slice_geometry =
        validate_commit_level_params::<Cfg::Field>(params, expanded, 0, polys.len())?;
    let geometry: CommitmentGeometry<'_> = params.into();
    if params.source_encoding != CommittedSourceEncoding::CanonicalCoefficientTable {
        return Err(AkitaError::InvalidSetup(
            "root commitments require canonical coefficient-table source encoding".into(),
        ));
    }
    let (commitment, hint) = commit_with_validated_geometry::<Cfg::Field, P, B>(
        polys,
        stack.commit(),
        geometry,
        &slice_geometry,
        contract,
    )?;

    Ok(CommitOutput {
        committed_group: CommittedGroup::new(profile, commitment),
        hint,
    })
}

#[cfg(test)]
mod tests;
