//! Prover-owned commitment kernels.

use crate::compute::{
    RootCommitSource, RootPolyMeta, RuntimeCommitBackendFor, RuntimeCommitSource,
    UniformProverStack,
};
use crate::validation::{signed_digit_kernel_for_setup, validate_i8_setup_log_basis};
use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_config::{ensure_prover_schedule_fits_setup, CommitmentConfig, TrustedScheduleCatalog};
#[cfg(test)]
use akita_error::checked;
use akita_error::AkitaError;
use akita_types::sis::CommittedSourceContract;
use akita_types::{
    dispatch_for_field, validate_role_dims, validate_role_dims_for_field, AkitaCommitmentHint,
    AkitaExpandedSetup, AkitaScheduleLookupKey, Commitment, CommitmentRingDims, CommittedGroup,
    CommittedGroupParams, FpExtEncoding, GadgetDigits, GroupCommitPhaseParams,
    PrecommittedGroupProfiles,
};
use jolt_field::{CanonicalEncoding, Field, Ring, Unreduced};

mod compression;
mod inner_outer;
use compression::{compute_commitment_compression, CommitmentCompressionOutput};
use inner_outer::compute_inner_outer_commitment;
pub(crate) use inner_outer::validate_commit_inner_shape;
pub(crate) use inner_outer::{commit_outer_slices, for_each_outer_slice_input};
#[cfg(test)]
use inner_outer::{compute_inner_commitment, outer_slice_inputs};

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
    /// Use a caller-supplied commit profile without catalog selection.
    Explicit(&'a GroupCommitPhaseParams),
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

    /// Commit a caller-supplied frozen commit profile without catalog lookup.
    ///
    /// The profile fully determines the A/B commitment. Any opening schedule the
    /// caller intends to use later — including a grouped root over precommitted
    /// groups — is validated where the opening consumes it, not at commit time.
    #[must_use]
    pub const fn explicit(profile: &'a GroupCommitPhaseParams) -> Self {
        Self {
            precommitted_groups: PrecommittedGroupContext::NoPrecommittedGroups,
            parameter_source: GroupParameterSource::Explicit(profile),
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

fn validate_commitment_geometry<F>(
    profile: &GroupCommitPhaseParams,
    setup: &AkitaExpandedSetup<F>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
{
    signed_digit_kernel_for_setup(
        profile.inner.digits.log_basis,
        "for signed witness commitment decomposition",
    )?;
    validate_i8_setup_log_basis(
        profile.outer.digits.log_basis,
        "for i8 outer commitment decomposition",
    )?;

    // A/B geometry is independent of the D/opening matrix. Mirroring B into
    // the opening slot lets the shared role validator enforce only the two
    // dimensions represented by this borrowed view.
    let dims = CommitmentRingDims {
        inner: profile.inner.matrix.ring_dimension(),
        outer: profile.outer.matrix.ring_dimension(),
        opening: profile.outer.matrix.ring_dimension(),
    };
    validate_role_dims(dims)?;
    validate_role_dims_for_field::<F>(dims)?;

    let expected_a_width = profile
        .blocks()
        .positions_per_block
        .checked_mul(profile.inner.digits.num_digits)
        .ok_or_else(|| AkitaError::InvalidSetup("A commit width overflow".to_string()))?;
    if profile.inner.matrix.input_width() != expected_a_width {
        return Err(AkitaError::InvalidSetup(format!(
            "commit profile A width {} does not match num_positions_per_block * num_digits_inner = {expected_a_width}",
            profile.inner.matrix.input_width()
        )));
    }
    if profile.outer.matrix.input_width() == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "commit profile requires nonzero B width, got B={}",
            profile.outer.matrix.input_width()
        )));
    }

    let required = akita_types::commit_only_setup_field_elements(
        &profile.inner.matrix,
        &profile.outer.matrix,
        profile.outer_slice_count,
    )?;
    let available = setup.shared_matrix.num_field_elements();
    if required > available {
        return Err(AkitaError::InvalidSetup(format!(
            "commit profile requires {required} setup field elements for commitment, but setup has {available}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_commit_level_params<F>(
    params: &CommittedGroupParams,
    setup: &AkitaExpandedSetup<F>,
    fold_level: usize,
    num_polynomials: usize,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
{
    params.validate_commitment_request(fold_level, num_polynomials)?;
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
    validate_commitment_geometry::<F>(&params.own_group().profile, setup)?;

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
    Ok(())
}

/// Validate a singleton commitment request against prover setup capacity.
///
/// # Errors
///
/// Returns an error if the request is empty, mixes polynomial dimensions, or
/// exceeds the prover setup capacity.
pub fn resolve_polynomial_group_layout<F, P>(
    polys: &[P],
    setup: &AkitaExpandedSetup<F>,
) -> Result<akita_types::PolynomialGroupLayout, AkitaError>
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
    if polys.len() > setup.descriptor.max_num_batched_polys {
        return Err(AkitaError::InvalidInput(format!(
            "commit received {} polynomials but setup supports at most {}",
            polys.len(),
            setup.descriptor.max_num_batched_polys
        )));
    }
    if num_vars > setup.descriptor.max_num_vars {
        return Err(AkitaError::InvalidInput(format!(
            "commit received a polynomial with {} variables but setup supports at most {}",
            num_vars, setup.descriptor.max_num_vars
        )));
    }
    Ok(akita_types::PolynomialGroupLayout::new(
        num_vars,
        polys.len(),
    ))
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
    inner_digits: GadgetDigits,
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
        decompose_centering_threshold(inner_digits.num_digits, inner_digits.log_basis, modulus);
    let (negative_reach, positive_reach) =
        contract.accepted_bounds(inner_digits.log_basis, inner_digits.num_digits);
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
                inner_digits.num_digits,
                inner_digits.log_basis,
                render_reach(negative_reach),
                render_reach(positive_reach),
            )));
        }
    }
    Ok(())
}

/// Resolve the frozen commit params of one root commitment and admit its sources.
fn resolve_commit_params<Cfg, P>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    schedules: &TrustedScheduleCatalog,
    context: GroupContext<'_>,
) -> Result<GroupCommitPhaseParams, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field + CanonicalEncoding,
    P: RuntimeCommitSource<Cfg::Field>,
{
    let polynomial_group_layout =
        resolve_polynomial_group_layout::<Cfg::Field, P>(polys, expanded)?;

    // The frozen commit params fully determine the A/B commitment. D/opening
    // geometry and source encoding are validated where the opening schedule
    // consumes them, so the commit path works from the commit params alone.
    let commit_params: GroupCommitPhaseParams =
        if let GroupParameterSource::Explicit(commit_params) = context.parameter_source {
            if commit_params.group != polynomial_group_layout {
                return Err(AkitaError::InvalidSetup(
                    "explicit commit params do not match the polynomial group being committed"
                        .into(),
                ));
            }
            *commit_params
        } else {
            let key = AkitaScheduleLookupKey {
                final_group: polynomial_group_layout,
                precommitteds: context.precommitted_groups.as_slice().to_vec(),
            };
            let scheduled_row = schedules.resolve_key(&key)?;

            // A final group with precommitted groups consumes the row's whole
            // schedule. A standalone group is admitted on its own A/B footprint.
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

            scheduled_row.profiles().final_group
        };

    // The commit params fully determine the A/B commitment, so this is the
    // complete commit-time gate. D/opening geometry and source encoding are not
    // part of the commit params: they are validated wherever the opening
    // schedule consumes them, never at commit time.
    commit_params.validate_frozen_precommit(
        commit_params
            .inner
            .matrix
            .sis_modulus_profile()
            .field_bits(),
    )?;
    validate_commitment_geometry::<Cfg::Field>(&commit_params, expanded)?;

    // Both admission gates read the `Cfg` declaration rather than commitment
    // geometry, so they run here instead of inside the commit kernel.
    let contract = Cfg::committed_source_contract()?;
    ensure_sources_match_declared_class::<Cfg::Field, P>(polys, contract)?;
    dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        Cfg::Field,
        commit_params.inner.matrix.ring_dimension(),
        |D_A| ensure_sources_fit_accepted_interval::<Cfg::Field, P, D_A>(
            polys,
            commit_params.inner.digits,
            contract,
        )
    )?;

    Ok(commit_params)
}

/// Commit one homogeneous polynomial group in its complete parameter context.
///
/// Scheduler contexts select an existing S or G catalog row. Explicit
/// contexts commit a caller-supplied frozen profile without catalog lookup.
/// Root commitments always consume the canonical coefficient table.
///
/// # Errors
///
/// Returns an error for an empty or mixed-arity group, unsupported role
/// parameters, insufficient setup, or commitment execution failure.
pub fn commit<Cfg, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    schedules: &TrustedScheduleCatalog,
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
    let commit_params = resolve_commit_params::<Cfg, P>(polys, expanded, schedules, context)?;
    let ctx = stack.commit();
    let (inner_rows, uncompressed_commitment) =
        compute_inner_outer_commitment(polys, ctx, commit_params)?;
    let CommitmentCompressionOutput {
        payload,
        witness,
        quotients,
    } = compute_commitment_compression(
        ctx,
        commit_params.outer.matrix.sis_table_key().modulus_profile,
        uncompressed_commitment,
    )?;
    let hint = AkitaCommitmentHint::new_with_outer_compression(
        commit_params.inner.matrix.ring_dimension(),
        inner_rows,
        &witness,
        &quotients,
    )?;

    Ok(CommitOutput {
        committed_group: CommittedGroup::new(commit_params, Commitment::new(payload)),
        hint,
    })
}

#[cfg(test)]
mod tests;
