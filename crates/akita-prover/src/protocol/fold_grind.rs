//! Fold-l∞ Fiat–Shamir grind: preview off-sponge clones, commit the winning nonce.

#[cfg(feature = "response-model-diagnostics")]
use crate::backend::AcceptedFoldHandle;
use crate::backend::{
    FoldProbeDiagnostics, FoldProbeGeometry, FoldProbeOutcome, ValidatedFoldAcceptancePlan,
    ValidatedFoldProbePlan, ValidatedTerminalFoldProbePlan,
};
use akita_challenges::{FoldDraw, NativePreviewFoldDraw, NativeProverFoldDraw};
use akita_error::AkitaError;
use akita_types::GroupFoldChallenges;
use akita_types::{
    draw_group_fold_challenges, dyadic_block_ranges, CommittedGroupParams,
    InnerCommitSecurityRoute, OpeningClaimsLayout, TerminalFoldParams, TerminalResponseShape,
    FOLD_RESPONSE_ATTEMPTS,
};
#[cfg(test)]
use akita_types::{OpeningFamily, OpeningMethod};
use jolt_field::Unreduced;
use jolt_field::{CanonicalEncoding, Field, Ring};

#[cfg(feature = "response-model-diagnostics")]
#[inline]
fn response_model_diagnostics_enabled() -> bool {
    tracing::enabled!(
        target: "akita_prover::protocol::fold_response_model",
        tracing::Level::INFO
    )
}

/// Whether every group of every probed nonce is reported, so the grind keeps
/// probing the remaining groups after one is rejected.
#[inline]
fn trace_every_probe() -> bool {
    #[cfg(feature = "response-model-diagnostics")]
    {
        response_model_diagnostics_enabled()
    }
    #[cfg(not(feature = "response-model-diagnostics"))]
    {
        false
    }
}

/// Public acceptance policy of one probed group, reported with each probe.
#[cfg_attr(not(feature = "response-model-diagnostics"), allow(dead_code))]
#[derive(Clone, Copy)]
struct FoldProbeTrace {
    level: u32,
    group_index: usize,
    terminal: bool,
    linf_bound_negative: u128,
    linf_bound_positive: u128,
    response_l2_sq_cap: Option<u128>,
    response_coeffs: usize,
    log_basis_response: Option<u32>,
    num_digits_response: Option<usize>,
}

impl FoldProbeTrace {
    /// Report one probe outcome, accepted or rejected. `accepted` is the
    /// honest acceptance predicate, even when a fault admitted the response.
    #[cfg_attr(not(feature = "response-model-diagnostics"), allow(unused_variables))]
    fn emit(&self, nonce: u32, accepted: bool, diagnostics: FoldProbeDiagnostics) {
        #[cfg(feature = "response-model-diagnostics")]
        if response_model_diagnostics_enabled() {
            tracing::info!(
                target: "akita_prover::protocol::fold_response_model",
                level = self.level,
                group_index = self.group_index,
                nonce,
                terminal = self.terminal,
                accepted,
                observed_linf = ?diagnostics.observed_linf(),
                linf_bound_negative = self.linf_bound_negative,
                linf_bound_positive = self.linf_bound_positive,
                observed_l2_sq = ?diagnostics.observed_l2_sq(),
                response_l2_sq_cap = ?self.response_l2_sq_cap,
                response_coeffs = self.response_coeffs,
                log_basis_response = ?self.log_basis_response,
                num_digits_response = ?self.num_digits_response,
                "fold probe"
            );
        }
    }
}

/// Run one backend probe with the fault-injection admission bypass set to
/// `bypass`. Also returns whether the backend admitted a response only
/// because of that bypass.
fn probe_with_admission_bypass<T>(
    bypass: bool,
    probe: impl FnOnce() -> Result<T, AkitaError>,
) -> Result<(T, bool), AkitaError> {
    #[cfg(feature = "fault-injection")]
    {
        let (value, bypassed) = crate::fault_injection::probe_with_admission_bypass(bypass, probe);
        Ok((value?, bypassed))
    }
    #[cfg(not(feature = "fault-injection"))]
    {
        let _ = bypass;
        Ok((probe()?, false))
    }
}

/// Groups at `level` whose acceptance predicate the active fault skips:
/// `Some(None)` for every group, `Some(Some(g))` for group `g` only.
#[inline]
fn accept_rejected_target(level: u32) -> Option<Option<usize>> {
    #[cfg(feature = "fault-injection")]
    {
        crate::fault_injection::accept_rejected_target(level)
    }
    #[cfg(not(feature = "fault-injection"))]
    {
        let _ = level;
        None
    }
}

/// Outcome of one nonce in a grind that may carry an `AcceptRejectedNonce`
/// fault.
enum NonceProbe<T> {
    Rejected,
    /// Every group passed its acceptance predicate.
    Accepted(T),
    /// Every group was admitted, and a fault-selected group only because its
    /// predicate was skipped.
    FaultAccepted(T),
}

/// Commit the first nonce the grind accepts. While a fault targets this grind
/// (`faulted`), only fault-accepted nonces qualify; the first honestly
/// accepted nonce is kept as a fallback when none exists.
fn first_accepted_nonce_with_fault<T>(
    max_grind_attempts: u32,
    faulted: bool,
    mut probe: impl FnMut(u32) -> Result<NonceProbe<T>, AkitaError>,
) -> Result<(u32, T), AkitaError> {
    let mut fallback = None;
    let selected = first_jointly_accepted_nonce(max_grind_attempts, |nonce| {
        Ok(match probe(nonce)? {
            NonceProbe::Rejected => None,
            NonceProbe::Accepted(value) if faulted => {
                fallback.get_or_insert((nonce, value));
                None
            }
            NonceProbe::Accepted(value) => Some(value),
            NonceProbe::FaultAccepted(value) => {
                #[cfg(feature = "fault-injection")]
                crate::fault_injection::record_applied();
                Some(value)
            }
        })
    });
    match selected {
        Ok(selected) => Ok(selected),
        Err(error) => fallback.ok_or(error),
    }
}

pub(crate) struct FoldGrindGroup<'group, G: ?Sized> {
    pub(crate) group_index: usize,
    pub(crate) opening: &'group G,
    pub(crate) num_polynomials: usize,
    pub(crate) params: akita_types::GroupOpenPhaseParams,
}

impl<G: ?Sized> Copy for FoldGrindGroup<'_, G> {}

impl<G: ?Sized> Clone for FoldGrindGroup<'_, G> {
    fn clone(&self) -> Self {
        *self
    }
}

pub(crate) struct FoldProbeOutput<FoldHandle> {
    pub(crate) fold_handle: FoldHandle,
    pub(crate) challenges: GroupFoldChallenges,
    pub(crate) diagnostics: FoldProbeDiagnostics,
}

pub(crate) struct TerminalFoldGrindOutput {
    pub(crate) encoded_payload: Vec<u8>,
}

/// Sample the flat scalar terminal fold against its capacity-based response
/// cap. The returned witness retains centered `z` coefficients only; terminal
/// `e` and `t` are never gadget decomposed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_terminal_fold_response<F, E, H, B, const D: usize>(
    backend: &B,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: u32,
    params: &TerminalFoldParams,
    sparse: &akita_challenges::SparseChallengeConfig,
    witness: &H,
    shape: &TerminalResponseShape,
) -> Result<TerminalFoldGrindOutput, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring + Unreduced + 'static,
    <F as Unreduced>::Wide: From<F>,
    E: Field,
    H: crate::backend::RecursiveWitnessHandle,
    B: crate::backend::OpaqueTerminalFoldKernel<F, E, WitnessHandle = H>,
{
    let expected_group =
        shape.layout.groups.first().ok_or_else(|| {
            AkitaError::InvalidSetup("terminal response shape has no group".into())
        })?;
    if shape.layout.groups.len() != 1
        || expected_group.z_coords
            != params
                .inner_width()
                .checked_mul(params.d_a())
                .ok_or_else(|| AkitaError::InvalidSetup("terminal z width overflow".into()))?
    {
        return Err(AkitaError::InvalidSetup(
            "terminal response shape does not match terminal A width".into(),
        ));
    }
    let linf_cap = expected_group.z_linf_cap;
    params.validate_terminal_linf_cap(linf_cap)?;
    let response_l2_sq_cap = params.response_l2_sq_cap();
    let operator_rejection = if response_l2_sq_cap.is_some() {
        Some(
            akita_challenges::selective_l2_operator_norm_rejection(params.d_a(), sparse)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("unsupported terminal L2 challenge policy".into())
                })?,
        )
    } else {
        None
    };
    let point_indices = [0usize];
    let site = akita_types::GrindingSite::FoldResponse { level };
    let (linf_bound_negative, linf_bound_positive) = linf_cap
        .map_or((1u128 << 15, u128::from(i16::MAX.unsigned_abs())), |cap| {
            (cap, cap)
        });
    let probe_trace = FoldProbeTrace {
        level,
        group_index: 0,
        terminal: true,
        linf_bound_negative,
        linf_bound_positive,
        response_l2_sq_cap,
        response_coeffs: expected_group.z_coords,
        log_basis_response: None,
        num_digits_response: None,
    };
    let fault_target = accept_rejected_target(level);
    let bypass = fault_target.is_some_and(|group| group.is_none_or(|group| group == 0));
    let (nonce, (fold_handle, challenges, encoding, diagnostics)) =
        first_accepted_nonce_with_fault(FOLD_RESPONSE_ATTEMPTS, bypass, |nonce| {
            let mut preview_state = grinding.preview_fold_response(site, nonce)?;
            let mut preview = NativePreviewFoldDraw::new(&mut preview_state);
            let challenges = preview.draw_folding_challenges_with_rejection(
                akita_challenges::FoldChallengeDrawDomain::EvaluationTrace,
                params.d_a(),
                0,
                params.blocks.live_blocks,
                1,
                sparse,
                operator_rejection,
            )?;
            let selected = challenges.select_claims(&point_indices)?;
            let plan = ValidatedTerminalFoldProbePlan::new::<D>(
                &selected,
                params.blocks.positions_per_block,
                params.inner.digits.num_digits,
                params.inner.digits.log_basis,
                expected_group.z_coords,
                linf_cap,
                response_l2_sq_cap,
                expected_group.z_rice_low_bits,
                expected_group.z_payload_bytes,
            )?;
            let (outcome, bypassed) = probe_with_admission_bypass(bypass, || {
                <B as crate::backend::OpaqueTerminalFoldKernel<F, E>>::probe_terminal_fold(
                    backend, witness, &plan,
                )
            })?;
            match outcome {
                FoldProbeOutcome::Rejected { diagnostics } => {
                    probe_trace.emit(nonce, false, diagnostics);
                    Ok(NonceProbe::Rejected)
                }
                FoldProbeOutcome::Accepted {
                    fold_handle,
                    diagnostics,
                } => {
                    probe_trace.emit(nonce, !bypassed, diagnostics);
                    let value = (
                        fold_handle,
                        challenges,
                        crate::backend::ValidatedTerminalZEncodingPlan::from_probe(&plan),
                        diagnostics,
                    );
                    Ok(if bypassed {
                        NonceProbe::FaultAccepted(value)
                    } else {
                        NonceProbe::Accepted(value)
                    })
                }
            }
        })?;
    grinding.commit_fold_response(site, nonce)?;
    let mut live = NativeProverFoldDraw::new(grinding.state_mut(), level, 0);
    let live_challenges = live.draw_folding_challenges_with_rejection(
        akita_challenges::FoldChallengeDrawDomain::EvaluationTrace,
        params.d_a(),
        0,
        params.blocks.live_blocks,
        1,
        sparse,
        operator_rejection,
    )?;
    if live_challenges != challenges {
        return Err(AkitaError::InvalidInput(
            "terminal grind preview did not match live transcript replay".into(),
        ));
    }
    grinding.record_fold_challenges(level, 0, params.blocks.live_blocks)?;
    #[cfg(not(feature = "response-model-diagnostics"))]
    let _ = diagnostics;
    #[cfg(feature = "response-model-diagnostics")]
    if response_model_diagnostics_enabled() {
        let source_l2_sq = diagnostics.source_l2_sq();
        let conditional_mean_l2_sq =
            source_l2_sq.and_then(|energy| energy.checked_mul(sparse.challenge_l2_sq_max()));
        tracing::info!(
            target: "akita_prover::protocol::fold_response_model",
            terminal = true,
            nonce,
            attempts = nonce + 1,
            ring_dimension = params.d_a(),
            num_live_blocks = params.blocks.live_blocks,
            num_positions_per_block = params.blocks.positions_per_block,
            response_coeffs = expected_group.z_coords,
            log_basis_inner = params.inner.digits.log_basis,
            num_digits_inner = params.inner.digits.num_digits,
            challenge_weight = sparse.weight(),
            challenge_l1 = sparse.l1_norm(),
            challenge_l2_sq = sparse.challenge_l2_sq_max(),
            challenge_linf = sparse.infinity_norm(),
            source_l2_sq = ?source_l2_sq,
            conditional_mean_l2_sq = ?conditional_mean_l2_sq,
            response_l2_sq = ?diagnostics.observed_l2_sq(),
            response_l2_sq_cap = ?response_l2_sq_cap,
            response_linf = ?diagnostics.observed_linf(),
            response_linf_cap = ?linf_cap,
            "terminal fold response model sample"
        );
    }
    let encoded_payload =
        <B as crate::backend::OpaqueTerminalFoldKernel<F, E>>::encode_terminal_fold(
            backend,
            fold_handle,
            &encoding,
        )?;
    Ok(TerminalFoldGrindOutput { encoded_payload })
}

struct PreparedFoldGrindGroup<'group, G: ?Sized> {
    input: FoldGrindGroup<'group, G>,
    acceptance: ValidatedFoldAcceptancePlan,
}

fn first_jointly_accepted_nonce<T>(
    max_grind_attempts: u32,
    mut probe: impl FnMut(u32) -> Result<Option<T>, AkitaError>,
) -> Result<(u32, T), AkitaError> {
    for nonce in 0..max_grind_attempts {
        if let Some(value) = probe(nonce)? {
            return Ok((nonce, value));
        }
    }
    Err(AkitaError::InvalidInput(format!(
        "fold grind exceeded {} joint attempts",
        max_grind_attempts
    )))
}

/// Probe every group at its native A dimension as one transcript transaction
/// for each candidate nonce.
#[allow(clippy::too_many_arguments)]
fn sample_multi_group_fold_decompose_witnesses_native<F, E, B>(
    opening_ctx: &crate::backend::OperationCtx<'_, F, B>,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: u32,
    root_lp: &CommittedGroupParams,
    groups: &[PreparedFoldGrindGroup<'_, B::PreparedOpeningHandle>],
    max_grind_attempts: u32,
) -> Result<Vec<FoldProbeOutput<B::AcceptedFoldHandle>>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring + Unreduced + 'static,
    <F as Unreduced>::Wide: From<F>,
    E: akita_types::FpExtEncoding<F>
        + jolt_field::ExtField<F>
        + akita_serialization::AkitaSerialize,
    B: crate::backend::OpaqueOpeningKernel<F, E>,
{
    if groups.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "fold grind batch has no groups".to_string(),
        ));
    }
    let site = akita_types::GrindingSite::FoldResponse { level };
    let trace_probes = trace_every_probe();
    let fault_target = accept_rejected_target(level);
    let (nonce, mut candidate_outputs) =
        first_accepted_nonce_with_fault(max_grind_attempts, fault_target.is_some(), |nonce| {
            let mut candidate_outputs = Vec::with_capacity(groups.len());
            let mut jointly_accepted = true;
            let mut fault_accepted = false;
            {
                let mut preview_state = grinding.preview_fold_response(site, nonce)?;
                let mut preview = NativePreviewFoldDraw::new(&mut preview_state);
                for prepared_group in groups {
                    let group = &prepared_group.input;
                    let challenges = draw_group_fold_challenges::<F, E, _>(
                        &mut preview,
                        &group.params,
                        group.group_index,
                        group.num_polynomials,
                    )?;
                    let ranges = (root_lp.witness_chunk.num_chunks > 1)
                        .then(|| {
                            dyadic_block_ranges(
                                group.params.num_live_blocks(),
                                root_lp.witness_chunk.num_chunks,
                            )
                        })
                        .transpose()?;
                    let geometry = ranges
                        .as_deref()
                        .map_or(FoldProbeGeometry::Sparse, |chunk_ranges| {
                            FoldProbeGeometry::SparseChunked { chunk_ranges }
                        });
                    let context = opening_ctx.for_group(group.group_index);
                    let bypass = fault_target
                        .is_some_and(|target| target.is_none_or(|g| g == group.group_index));
                    let ring_dimension = group.params.inner_commit_matrix_params().ring_dimension();
                    let (outcome, bypassed) = probe_with_admission_bypass(bypass, || {
                        akita_types::dispatch_for_field!(
                            ProtocolDispatchSlot::Role(RingRole::Inner),
                            F,
                            ring_dimension,
                            |D| {
                                let plan = ValidatedFoldProbePlan::new::<D>(
                                    challenges.ambient_a(),
                                    group.num_polynomials,
                                    group.params.num_live_blocks(),
                                    geometry,
                                    group.params.num_positions_per_block(),
                                    group.params.num_digits_inner(),
                                    group.params.log_basis_inner(),
                                    group.params.opening_method(),
                                    prepared_group.acceptance,
                                )?;
                                opening_ctx.backend().probe_opening_fold(
                                    context.proof_context(),
                                    group.opening,
                                    &plan,
                                )
                            }
                        )
                    })?;
                    let probe_trace = || {
                        let response_coeffs = group
                            .params
                            .num_positions_per_block()
                            .checked_mul(group.params.num_digits_inner())
                            .and_then(|rows| rows.checked_mul(ring_dimension))
                            .and_then(|coeffs| {
                                coeffs.checked_mul(ranges.as_ref().map_or(1, Vec::len))
                            })
                            .unwrap_or(usize::MAX);
                        FoldProbeTrace {
                            level,
                            group_index: group.group_index,
                            terminal: false,
                            linf_bound_negative: prepared_group
                                .acceptance
                                .digit_negative_abs_bound(),
                            linf_bound_positive: prepared_group.acceptance.digit_positive_bound(),
                            response_l2_sq_cap: prepared_group.acceptance.response_l2_sq_cap(),
                            response_coeffs,
                            log_basis_response: Some(group.params.log_basis_open()),
                            num_digits_response: Some(group.params.num_digits_fold()),
                        }
                    };
                    match outcome {
                        FoldProbeOutcome::Rejected { diagnostics } => {
                            if trace_probes {
                                probe_trace().emit(nonce, false, diagnostics);
                            } else {
                                return Ok(NonceProbe::Rejected);
                            }
                            jointly_accepted = false;
                        }
                        FoldProbeOutcome::Accepted {
                            fold_handle,
                            diagnostics,
                        } => {
                            if trace_probes {
                                probe_trace().emit(nonce, !bypassed, diagnostics);
                            }
                            fault_accepted |= bypassed;
                            candidate_outputs.push(FoldProbeOutput {
                                fold_handle,
                                challenges,
                                diagnostics,
                            });
                        }
                    }
                }
            }
            Ok(if !jointly_accepted {
                NonceProbe::Rejected
            } else if fault_accepted {
                NonceProbe::FaultAccepted(candidate_outputs)
            } else {
                NonceProbe::Accepted(candidate_outputs)
            })
        })?;

    grinding.commit_fold_response(site, nonce)?;
    {
        let _span = tracing::info_span!("fold_grind_live_replay").entered();
        for (prepared_group, output) in groups.iter().zip(candidate_outputs.iter_mut()) {
            let group = &prepared_group.input;
            let challenges = {
                let group_index = u32::try_from(group.group_index)
                    .map_err(|_| AkitaError::InvalidSetup("fold group index exceeds u32".into()))?;
                let mut live = NativeProverFoldDraw::new(grinding.state_mut(), level, group_index);
                draw_group_fold_challenges::<F, E, _>(
                    &mut live,
                    &group.params,
                    group.group_index,
                    group.num_polynomials,
                )?
            };
            if challenges != output.challenges {
                return Err(AkitaError::InvalidInput(
                    "fold grind preview did not match live transcript replay".to_string(),
                ));
            }
            let group_index = u32::try_from(group.group_index)
                .map_err(|_| AkitaError::InvalidSetup("fold group index exceeds u32".into()))?;
            let coordinate_count = group
                .params
                .num_live_blocks()
                .checked_mul(group.num_polynomials)
                .ok_or_else(|| AkitaError::InvalidSetup("fold coordinate count overflow".into()))?;
            grinding.record_fold_challenges(level, group_index, coordinate_count)?;
            tracing::info!(
                group_index = group.group_index,
                nonce,
                attempts = nonce + 1,
                response_l2_sq = ?output.diagnostics.observed_l2_sq(),
                response_l2_sq_cap = ?prepared_group.acceptance.response_l2_sq_cap(),
                "selected physical fold response"
            );
            #[cfg(feature = "response-model-diagnostics")]
            if response_model_diagnostics_enabled() {
                let challenge_config = group.params.fold_challenge_config();
                let source_l2_sq = output.diagnostics.source_l2_sq();
                let conditional_mean_l2_sq = source_l2_sq
                    .and_then(|energy| energy.checked_mul(challenge_config.challenge_l2_sq_max()));
                let metadata = output.fold_handle.metadata();
                let response_coeffs = metadata
                    .response_coordinate_count()
                    .checked_mul(metadata.num_chunks())
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("fold diagnostic response size overflow".into())
                    })?;
                tracing::info!(
                    target: "akita_prover::protocol::fold_response_model",
                    level,
                    group_index = group.group_index,
                    nonce,
                    attempts = nonce + 1,
                    ring_dimension = metadata.ring_dimension(),
                    num_polynomials = group.num_polynomials,
                    num_live_blocks = group.params.num_live_blocks(),
                    num_positions_per_block = group.params.num_positions_per_block(),
                    num_chunks = metadata.num_chunks(),
                    response_coeffs,
                    log_basis_inner = group.params.log_basis_inner(),
                    num_digits_inner = group.params.num_digits_inner(),
                    log_basis_response = group.params.log_basis_open(),
                    num_digits_response = group.params.num_digits_fold(),
                    challenge_weight = challenge_config.weight(),
                    challenge_l1 = challenge_config.l1_norm(),
                    challenge_l2_sq = challenge_config.challenge_l2_sq_max(),
                    challenge_linf = challenge_config.infinity_norm(),
                    source_l2_sq = ?source_l2_sq,
                    conditional_mean_l2_sq = ?conditional_mean_l2_sq,
                    response_l2_sq = ?output.diagnostics.observed_l2_sq(),
                    response_l2_sq_cap = ?prepared_group.acceptance.response_l2_sq_cap(),
                    response_linf = ?output.diagnostics.observed_linf(),
                    "fold response model sample"
                );
            }
        }
    }
    Ok(candidate_outputs)
}

/// Probe all root groups off-sponge and commit the first jointly accepted nonce.
///
/// Every preset probes `nonce = 0, 1, …` and commits the minimum accepting nonce.
/// When `tail_t_vectors` is set, the terminal response must fit the exact cap
/// and Golomb-Rice byte budget carried by its scheduled response shape.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_multi_group_fold_decompose_witnesses<F, E, B>(
    opening_ctx: &crate::backend::OperationCtx<'_, F, B>,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
    level: u32,
    root_lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    groups: &[FoldGrindGroup<'_, B::PreparedOpeningHandle>],
    _tail_t_vectors: Option<usize>,
) -> Result<Vec<FoldProbeOutput<B::AcceptedFoldHandle>>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring + Unreduced + 'static,
    <F as Unreduced>::Wide: From<F>,
    E: akita_types::FpExtEncoding<F>
        + jolt_field::ExtField<F>
        + akita_serialization::AkitaSerialize,
    B: crate::backend::OpaqueOpeningKernel<F, E>,
{
    if groups.len() != opening_batch.num_groups() {
        return Err(AkitaError::InvalidSetup(
            "fold grind groups do not match the opening batch".to_string(),
        ));
    }
    let mut prepared_groups = Vec::with_capacity(groups.len());
    for (expected_group_index, group) in groups.iter().enumerate() {
        let expected_claims = opening_batch
            .group_layout(expected_group_index)?
            .num_polynomials();
        if group.group_index != expected_group_index
            || group.num_polynomials == 0
            || group.num_polynomials != expected_claims
        {
            return Err(AkitaError::InvalidSetup(
                "fold grind group descriptor is malformed".to_string(),
            ));
        }
        let delta_fold = group.params.num_digits_fold();
        let (digit_negative_abs_bound, digit_positive_bound) =
            akita_types::sis::balanced_digit_representable_bounds(
                group.params.log_basis_open(),
                delta_fold,
            );
        let response_l2_sq_cap = match group.params.inner_commit_matrix_params().security_route() {
            InnerCommitSecurityRoute::Linf(_) => None,
            InnerCommitSecurityRoute::L2 {
                response_l2_sq_cap, ..
            } => {
                if groups.len() != 1 || group.group_index != 0 {
                    return Err(AkitaError::InvalidSetup(
                        "L2 fold grinding requires one scalar group".into(),
                    ));
                }
                Some(response_l2_sq_cap)
            }
        };
        prepared_groups.push(PreparedFoldGrindGroup {
            input: *group,
            acceptance: ValidatedFoldAcceptancePlan::new(
                digit_negative_abs_bound,
                digit_positive_bound,
                response_l2_sq_cap,
            ),
        });
    }
    sample_multi_group_fold_decompose_witnesses_native::<F, E, B>(
        opening_ctx,
        grinding,
        level,
        root_lp,
        &prepared_groups,
        FOLD_RESPONSE_ATTEMPTS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_types::SisModulusProfileId;

    type F = jolt_field::Prime128Offset275;

    #[derive(Default)]
    struct FixedDraw {
        draws: usize,
    }

    impl FoldDraw for FixedDraw {
        fn absorb_and_squeeze(&mut self, _payload: &[u8]) -> Result<[u8; 32], AkitaError> {
            self.draws += 1;
            Ok([11; akita_transcript::FOLD_CHALLENGE_SEED_LEN])
        }
    }

    #[test]
    fn packing_draw_has_one_subring_value_and_derived_a_view() {
        let mut params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            128,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::production_for_ring_dim(128).unwrap(),
        )
        .with_decomp(4, 6, 2, 2, 2)
        .unwrap();
        params.own_group_mut().opening.opening_method = OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        };
        params.own_group_mut().opening.fold_challenge_config =
            SparseChallengeConfig::production_for_ring_dim(64).unwrap();

        let mut draw = FixedDraw::default();
        let challenges =
            draw_group_fold_challenges::<F, F, _>(&mut draw, &params.final_group(), 3, 2).unwrap();
        assert_eq!(draw.draws, 1);
        let OpeningFamily::SubringCoefficientPacking(challenges) = challenges else {
            panic!("expected coefficient-packing challenges");
        };
        assert_eq!(challenges.geometry().subring_embedding_stride(), 2);
        assert_eq!(challenges.canonical().len(), 4);
        assert_eq!(challenges.ambient_a().len(), challenges.canonical().len());
        for (canonical, embedded) in challenges
            .canonical()
            .as_slice()
            .iter()
            .zip(challenges.ambient_a().as_slice())
        {
            assert_eq!(canonical.coeffs, embedded.coeffs);
            assert_eq!(canonical.positions.len(), embedded.positions.len());
            for (&subring_position, &ambient_position) in
                canonical.positions.iter().zip(&embedded.positions)
            {
                assert_eq!(ambient_position, 2 * subring_position);
            }
        }
    }

    #[test]
    fn packing_draw_rejects_unaudited_family_before_squeeze() {
        let mut params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            128,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::production_for_ring_dim(128).unwrap(),
        )
        .with_decomp(4, 6, 2, 2, 2)
        .unwrap();
        params.own_group_mut().opening.opening_method = OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        };

        for config in [
            SparseChallengeConfig::pm1_only(0),
            SparseChallengeConfig::pm1_only(1),
        ] {
            params.own_group_mut().opening.fold_challenge_config = config;
            let mut draw = FixedDraw::default();
            assert!(
                draw_group_fold_challenges::<F, F, _>(&mut draw, &params.final_group(), 0, 1)
                    .is_err()
            );
            assert_eq!(draw.draws, 0);
        }
    }

    #[test]
    fn joint_grind_skips_different_group_first_nonces() {
        let group_accepts = [[0, 2], [1, 2]];
        let mut probed = Vec::new();
        let (nonce, ()) = first_jointly_accepted_nonce(4, |nonce| {
            probed.push(nonce);
            Ok(group_accepts
                .iter()
                .all(|accepted| accepted.contains(&nonce))
                .then_some(()))
        })
        .unwrap();

        assert_eq!(nonce, 2);
        assert_eq!(probed, vec![0, 1, 2]);
    }
}
