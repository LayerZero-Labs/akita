//! Commitment scheme trait implementation.

use crate::protocol::commitment::hachi_recursive_level_layout_from_params;
use crate::protocol::config::{CommitmentConfig, WCommitmentConfig};
use crate::{CanonicalField, FieldCore, FieldSampling};
use akita_algebra::fields::wide::HasWide;
use akita_algebra::fields::HasUnreducedOps;
#[allow(unused_imports)]
use akita_field::parallel::*;
use akita_field::HachiError;
use akita_prover::crt_ntt::NttSlotCache;
use akita_prover::dispatch_with_ntt;
use akita_prover::ring_switch::commit_w;
use akita_prover::{
    batched_commit_with_params, build_folded_batched_proof_with_suffix, commit_with_params,
    verify_root_direct_commitments_with_params, CommitmentProver, HachiPolyOps, HachiProverSetup,
    MultiDNttCaches, ProveLevelOutput, ProverClaims, RecursiveCommitmentHintCache,
    RecursiveProverState, RecursiveSuffixOutcome, RecursiveWitnessFlat, RecursiveWitnessView,
    RootLevelRawOutput,
};
use akita_serialization::Valid;
use akita_transcript::Transcript;
use akita_types::BasisMode;
use akita_types::LevelParams;
use akita_types::{
    checked_total_claims, checked_total_groups, prepare_root_opening_point,
    schedule_is_root_direct, CommitmentVerifier, FlatRingVec, HachiBatchedProof,
    HachiCommitmentHint, MultiPointBatchShape, PreparedRootOpeningPoint, RingCommitment, Schedule,
    Step, VerifierClaims,
};
use akita_types::{
    HachiExpandedSetup, HachiRootBatchSummary, HachiScheduleInputs, HachiScheduleLookupKey,
    HachiVerifierSetup,
};
use akita_verifier::{
    prepare_verifier_claims, verify_batched_proof_with_schedule, BatchedVerifierScheduleContext,
    FoldVerifierLayouts,
};
use std::marker::PhantomData;
use std::time::Instant;

/// End-to-end PCS wrapper, generic over ring degree `D` and config `Cfg`.
#[derive(Clone, Copy, Debug, Default)]
pub struct HachiCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

fn scheduled_next_level_params<Cfg: CommitmentConfig>(
    schedule: &Schedule,
    step_index: usize,
    inputs: HachiScheduleInputs,
) -> Result<LevelParams, HachiError> {
    match schedule.steps.get(step_index) {
        Some(Step::Fold(step)) => Ok(step.params.clone()),
        Some(Step::Direct(step)) => {
            Ok(Cfg::level_params_with_log_basis(inputs, step.bits_per_elem))
        }
        None => Err(HachiError::InvalidSetup(
            "schedule is missing successor step".to_string(),
        )),
    }
}

fn scheduled_fold_execution<Cfg: CommitmentConfig>(
    schedule: &Schedule,
    level: usize,
    inputs: HachiScheduleInputs,
    current_log_basis: u32,
) -> Result<(LevelParams, LevelParams), HachiError> {
    let Some(Step::Fold(step)) = schedule.steps.get(level) else {
        return Err(HachiError::InvalidSetup(format!(
            "schedule is missing fold step at level {level}"
        )));
    };
    if step.current_w_len != inputs.current_w_len || step.params.log_basis != current_log_basis {
        return Err(HachiError::InvalidSetup(
            "scheduled recursive level did not match runtime state".to_string(),
        ));
    }
    let next_inputs = HachiScheduleInputs {
        max_num_vars: inputs.max_num_vars,
        level: level + 1,
        current_w_len: step.next_w_len,
    };
    let next_level_params = scheduled_next_level_params::<Cfg>(schedule, level + 1, next_inputs)?;
    Ok((step.params.clone(), next_level_params))
}

/// Unified root-level prover for both the singleton (`prove`) and multi-point
/// batched (`batched_prove`) paths.
///
/// The function uses a single canonical transcript layout that matches the
/// multi-point batched Fiat–Shamir stream: it always absorbs the batch-shape
/// header, per-claim field openings, a γ challenge per claim, and then the
/// γ-combined per-point y-rings. For a trivially-singleton call (1 point,
/// 1 group, 1 claim), the same sequence degenerates to absorbing the
/// constants `[1, 1, 1]` for the shape header, a single opening field
/// element, and a single γ — this γ is still sampled (not hard-coded to 1),
/// and the single per-claim y-ring is γ-scaled into the single per-point
/// y-ring. The verifier must replay the same layout.
///
/// The selected schedule is passed in by the caller and is authoritative for
/// the root handoff: after ring-switching produces `w`, the function derives
/// the first recursive commitment params from `schedule.steps[1]`.
///
/// * **Root layout**: the function receives the batch-effective root layout.
///   For singleton calls this is the exact root layout used by the generated
///   schedule check; for larger batches it is the combined-claim layout used by
///   the root relation.
/// * **Commitment rows** for the relation claim: when `commitments.len()
///   == 1` we borrow `&commitments[0].u` directly; otherwise we
///   concatenate row vectors across the multiple commitments. Only the
///   multi-commitment path pays the clone.
///
/// Callers reshape [`RootLevelRawOutput`] into either a
/// [`HachiLevelProof`](akita_types::HachiLevelProof) or a
/// [`HachiBatchedRootProof`](akita_types::HachiBatchedRootProof).
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prove_root_level<F, T, const D: usize, Cfg, P>(
    expanded: &HachiExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    commit_ntt_cache: &mut MultiDNttCaches,
    polys: &[&P],
    batch_shape: &MultiPointBatchShape,
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    commitments: &[RingCommitment<F, D>],
    root_key: HachiScheduleLookupKey,
    schedule: &Schedule,
    hints: Vec<HachiCommitmentHint<F, D>>,
    transcript: &mut T,
    batched_lp: &LevelParams,
) -> Result<RootLevelRawOutput<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
    P: HachiPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    let Some(Step::Fold(root_step)) = schedule.steps.first() else {
        return Err(HachiError::InvalidSetup(
            "root schedule does not start with a fold".to_string(),
        ));
    };
    let next_inputs = HachiScheduleInputs {
        max_num_vars: root_key.max_num_vars,
        level: 1,
        current_w_len: root_step.next_w_len,
    };
    let next_params = scheduled_next_level_params::<Cfg>(schedule, 1, next_inputs)?;
    let next_log_basis = next_params.log_basis;
    akita_prover::prove_root_fold_with_params::<F, T, D, P, _>(
        expanded,
        ntt_shared,
        transcript,
        polys,
        batch_shape,
        prepared_points,
        commitments,
        hints,
        batched_lp,
        root_step.next_w_len,
        next_log_basis,
        |w| {
            if next_params.ring_dimension == D {
                let commit_layout =
                    hachi_recursive_level_layout_from_params::<Cfg>(&next_params, w.len())?;
                let (wc, wh) =
                    commit_w::<F, D>(w, ntt_shared, &commit_layout, expanded.seed.max_stride)?;
                Ok((
                    FlatRingVec::from_commitment(&wc),
                    RecursiveCommitmentHintCache::from_typed(wh)?,
                ))
            } else {
                dispatch_commit::<F, Cfg>(next_params.clone(), commit_ntt_cache, expanded, w)
            }
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prove_one_recursive_level<F, T, const D: usize, Cfg>(
    expanded: &HachiExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    commit_ntt_cache: &mut MultiDNttCaches,
    witness: &RecursiveWitnessView<'_, F, D>,
    opening_point: &[F],
    hint: HachiCommitmentHint<F, D>,
    transcript: &mut T,
    commitment: &FlatRingVec<F>,
    level: usize,
    lp: &LevelParams,
    next_params: LevelParams,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    let next_log_basis = next_params.log_basis;
    akita_prover::prove_recursive_fold_with_params::<F, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        witness,
        opening_point,
        hint,
        commitment,
        level,
        lp,
        next_log_basis,
        |w| {
            if next_params.ring_dimension == D {
                let commit_layout = hachi_recursive_level_layout_from_params::<
                    WCommitmentConfig<{ D }, Cfg>,
                >(&next_params, w.len())?;
                let (wc, wh) =
                    commit_w::<F, D>(w, ntt_shared, &commit_layout, expanded.seed.max_stride)?;
                Ok((
                    FlatRingVec::from_commitment(&wc),
                    RecursiveCommitmentHintCache::from_typed(wh)?,
                ))
            } else {
                dispatch_commit::<F, Cfg>(next_params.clone(), commit_ntt_cache, expanded, w)
            }
        },
    )
}

/// Dispatch a commit-w operation to the correct ring dimension.
///
/// Each match arm builds NTT caches for the target D and calls `commit_w`.
/// `#[inline(never)]` isolates the match arms in their own stack frame,
/// preventing debug-mode stack bloat from monomorphized arms.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_commit<F, Cfg>(
    commit_params: LevelParams,
    commit_ntt_cache: &mut MultiDNttCaches,
    expanded: &HachiExpandedSetup<F>,
    w: &RecursiveWitnessFlat,
) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    Cfg: CommitmentConfig<Field = F>,
{
    let commit_d = commit_params.ring_dimension;
    let stride = expanded.seed.max_stride;
    dispatch_with_ntt!(
        commit_d,
        commit_ntt_cache,
        expanded,
        |D_COMMIT, ntt_shared| {
            let commit_layout = hachi_recursive_level_layout_from_params::<
                WCommitmentConfig<{ D_COMMIT }, Cfg>,
            >(&commit_params, w.len())?;
            let (wc, wh) = commit_w::<F, { D_COMMIT }>(w, ntt_shared, &commit_layout, stride)?;
            Ok((
                FlatRingVec::from_commitment(&wc),
                RecursiveCommitmentHintCache::from_typed(wh)?,
            ))
        }
    )
}

/// Dispatch a prove-level operation to the correct ring dimension.
///
/// Handles the fast-path (`level_d == D`) and the dynamic dispatch path.
/// `#[inline(never)]` isolates the monomorphized match arms in their own
/// stack frame.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_prove_level<F, T, const D: usize, Cfg>(
    level_d: usize,
    ntt_cache: &mut MultiDNttCaches,
    expanded: &HachiExpandedSetup<F>,
    setup_ntt_shared: &NttSlotCache<D>,
    commit_ntt_cache: &mut MultiDNttCaches,
    current_state: &RecursiveProverState<F>,
    transcript: &mut T,
    level: usize,
    level_params: &LevelParams,
    next_params: LevelParams,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    if level_d == D {
        prove_subsequent_level::<F, T, D, Cfg>(
            expanded,
            setup_ntt_shared,
            commit_ntt_cache,
            current_state,
            transcript,
            level,
            level_params,
            next_params,
        )
    } else {
        dispatch_with_ntt!(level_d, ntt_cache, expanded, |D_LEVEL, ntt_shared| {
            prove_subsequent_level::<F, T, { D_LEVEL }, Cfg>(
                expanded,
                ntt_shared,
                commit_ntt_cache,
                current_state,
                transcript,
                level,
                level_params,
                next_params,
            )
        })
    }
}

/// Single subsequent (recursive) prove level, extracted so that the
/// dispatch match arms contain only a function call.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prove_subsequent_level<F, T, const D_LEVEL: usize, Cfg>(
    expanded: &HachiExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D_LEVEL>,
    commit_ntt_cache: &mut MultiDNttCaches,
    current_state: &RecursiveProverState<F>,
    transcript: &mut T,
    level: usize,
    level_params: &LevelParams,
    next_params: LevelParams,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    let _setup_span = tracing::info_span!("inter_level_setup", level).entered();

    let current_w = &current_state.w;
    let opening_point = current_state.sumcheck_challenges.clone();

    let w_lp = hachi_recursive_level_layout_from_params::<Cfg>(level_params, current_w.len())?;
    let w_view = current_w.view::<F, { D_LEVEL }>()?;
    let typed_hint: HachiCommitmentHint<F, { D_LEVEL }> =
        current_state.hint.to_typed::<{ D_LEVEL }>()?;
    drop(_setup_span);

    prove_one_recursive_level::<F, T, { D_LEVEL }, Cfg>(
        expanded,
        ntt_shared,
        commit_ntt_cache,
        &w_view,
        &opening_point,
        typed_hint,
        transcript,
        &current_state.commitment,
        level,
        &w_lp,
        next_params,
    )
}

/// Drive the recursive fold levels (after the root) and resolve the terminal
/// `log_basis` for the packed-digit direct witness.
///
/// The selected planner schedule is authoritative: it determines the fold
/// count, per-level `LevelParams`, successor params, and terminal direct
/// witness basis.
#[allow(clippy::too_many_arguments)]
fn prove_recursive_suffix<F, T, const D: usize, Cfg>(
    setup: &HachiProverSetup<F, D>,
    ntt_cache: &mut MultiDNttCaches,
    commit_ntt_cache: &mut MultiDNttCaches,
    max_num_vars: usize,
    transcript: &mut T,
    initial_state: RecursiveProverState<F>,
    schedule: &Schedule,
) -> Result<RecursiveSuffixOutcome<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide + Valid,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    akita_prover::prove_recursive_suffix_with_policy(
        max_num_vars,
        initial_state,
        schedule,
        |level, inputs, current_log_basis| {
            scheduled_fold_execution::<Cfg>(schedule, level, inputs, current_log_basis)
        },
        |level, current_state, level_params, next_params| {
            dispatch_prove_level::<F, T, D, Cfg>(
                level_params.ring_dimension,
                ntt_cache,
                &setup.expanded,
                &setup.ntt_shared,
                commit_ntt_cache,
                current_state,
                transcript,
                level,
                level_params,
                next_params,
            )
        },
    )
}

impl<F, const D: usize, Cfg> CommitmentProver<F, D> for HachiCommitmentScheme<D, Cfg>
where
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
    Cfg: CommitmentConfig<Field = F>,
{
    type ProverSetup = HachiProverSetup<F, D>;
    type CommitHint = HachiCommitmentHint<F, D>;

    fn setup_prover(
        max_num_vars: usize,
        max_num_polys_per_point: usize,
        max_num_points: usize,
    ) -> Self::ProverSetup {
        crate::protocol::setup::new_prover_setup::<F, D, Cfg>(
            max_num_vars,
            max_num_polys_per_point,
            max_num_points,
        )
        .expect("commitment setup failed")
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        setup.verifier_setup()
    }

    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::commit")]
    fn commit<P: HachiPolyOps<F, D, CommitCache = NttSlotCache<D>>>(
        polys: &[P],
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::CommitHint), HachiError> {
        if polys.is_empty() {
            return Err(HachiError::InvalidInput(
                "commit requires at least one polynomial".to_string(),
            ));
        }
        let num_vars = polys[0].num_vars();
        if polys.iter().any(|p| p.num_vars() != num_vars) {
            return Err(HachiError::InvalidInput(
                "all polynomials in a batched commit must have the same num_vars".to_string(),
            ));
        }
        if polys.len() > setup.expanded.seed.max_num_batched_polys {
            return Err(HachiError::InvalidInput(format!(
                "commit received {} polynomials but setup supports at most {}",
                polys.len(),
                setup.expanded.seed.max_num_batched_polys
            )));
        }
        if num_vars > setup.expanded.seed.max_num_vars {
            return Err(HachiError::InvalidInput(format!(
                "commit received a polynomial with {} variables but setup supports at most {}",
                num_vars, setup.expanded.seed.max_num_vars
            )));
        }

        let params = Cfg::get_params_for_commitment(num_vars, polys.len())?;
        commit_with_params::<F, D, P>(polys, setup, &params)
    }

    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::batched_commit")]
    fn batched_commit<P: HachiPolyOps<F, D, CommitCache = NttSlotCache<D>>>(
        poly_groups: &[&[P]],
        point_group_sizes: &[usize],
        setup: &Self::ProverSetup,
    ) -> Result<(Vec<Self::Commitment>, Vec<Self::CommitHint>), HachiError> {
        if poly_groups.is_empty() {
            return Err(HachiError::InvalidInput(
                "batched_commit requires at least one commitment group".to_string(),
            ));
        }
        let total_groups = checked_total_groups(point_group_sizes, "batched_commit")?;
        if total_groups != poly_groups.len() {
            return Err(HachiError::InvalidInput(
                "batched_commit point group sizes do not match commitment groups".to_string(),
            ));
        }
        let num_vars = poly_groups[0]
            .first()
            .ok_or_else(|| {
                HachiError::InvalidInput(
                    "batched_commit requires nonempty commitment groups".to_string(),
                )
            })?
            .num_vars();
        if num_vars > setup.expanded.seed.max_num_vars {
            return Err(HachiError::InvalidInput(format!(
                "batched_commit received polynomials with {} variables but setup supports at most {}",
                num_vars, setup.expanded.seed.max_num_vars
            )));
        }
        if point_group_sizes.len() > setup.expanded.seed.max_num_points {
            return Err(HachiError::InvalidInput(format!(
                "batched_commit received {} opening points but setup supports at most {}",
                point_group_sizes.len(),
                setup.expanded.seed.max_num_points
            )));
        }

        let mut claim_group_sizes = Vec::with_capacity(poly_groups.len());
        let mut total_claims = 0usize;
        for group in poly_groups {
            if group.is_empty() {
                return Err(HachiError::InvalidInput(
                    "batched_commit requires nonempty commitment groups".to_string(),
                ));
            }
            if group.iter().any(|poly| poly.num_vars() != num_vars) {
                return Err(HachiError::InvalidInput(
                    "batched_commit requires all polynomials to have the same num_vars".to_string(),
                ));
            }
            let group_claims = group.len();
            claim_group_sizes.push(group_claims);
            total_claims = total_claims.checked_add(group_claims).ok_or_else(|| {
                HachiError::InvalidInput("batched_commit total claim count overflow".to_string())
            })?;
        }
        if total_claims > setup.expanded.seed.max_num_batched_polys {
            return Err(HachiError::InvalidInput(format!(
                "batched_commit received {total_claims} polynomials but setup supports at most {}",
                setup.expanded.seed.max_num_batched_polys
            )));
        }

        let batch_summary = HachiRootBatchSummary::from_claim_group_sizes(
            &claim_group_sizes,
            point_group_sizes.len(),
        )?;
        let schedule = Cfg::get_params_for_prove(
            setup.expanded.seed.max_num_vars,
            num_vars,
            total_claims,
            batch_summary,
        )?;
        let params = match schedule.steps.first() {
            Some(Step::Fold(root_step)) => root_step.params.clone(),
            Some(Step::Direct(_)) => Cfg::get_params_for_commitment(num_vars, total_claims)?,
            None => {
                return Err(HachiError::InvalidSetup(
                    "batched_commit schedule is empty".to_string(),
                ));
            }
        };

        batched_commit_with_params::<F, D, P>(poly_groups, setup, &params)
    }

    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::batched_prove")]
    fn batched_prove<'a, T: Transcript<F>, P: HachiPolyOps<F, D, CommitCache = NttSlotCache<D>>>(
        setup: &Self::ProverSetup,
        claims: ProverClaims<'a, F, P, Self::Commitment, Self::CommitHint>,
        transcript: &mut T,
        basis: BasisMode,
    ) -> Result<Self::BatchedProof, HachiError> {
        let prepared_claims =
            akita_prover::prepare_batched_prove_inputs::<F, P, D>(&setup.expanded, claims)?;

        let batch_summary = HachiRootBatchSummary::from_claim_group_sizes(
            &prepared_claims.batch_shape.claim_group_sizes,
            prepared_claims.opening_points.len(),
        )?;
        let max_num_vars = setup.expanded.seed.max_num_vars;
        let root_key = HachiScheduleLookupKey::with_batch(
            max_num_vars,
            prepared_claims.num_vars,
            prepared_claims.layout_num_claims,
            batch_summary,
        );
        let schedule = Cfg::get_params_for_prove(
            max_num_vars,
            prepared_claims.num_vars,
            prepared_claims.layout_num_claims,
            batch_summary,
        )?;

        // Batched analogue of the singleton root-direct shortcut: when the
        // selected schedule has no root fold, the witness is small enough that
        // we can transmit each claim's polynomial as field coefficients.
        if schedule_is_root_direct(&schedule) {
            return akita_prover::prove_root_direct_from_polys::<F, D, P>(
                &prepared_claims.flat_polys,
            );
        }
        let Some(Step::Fold(root_step)) = schedule.steps.first() else {
            return Err(HachiError::InvalidSetup(
                "root schedule does not start with a fold".to_string(),
            ));
        };

        let t_prove_total = Instant::now();
        let mut ntt_cache = MultiDNttCaches::new();
        let mut commit_ntt_cache = MultiDNttCaches::new();
        let alpha_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
        let prepared_points = prepared_claims
            .opening_points
            .iter()
            .map(|opening_point| {
                prepare_root_opening_point::<F, D>(
                    opening_point,
                    basis,
                    &root_step.params,
                    alpha_bits,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if prepared_claims
            .commitments_by_point
            .iter()
            .any(|commitment| commitment.u.len() != root_step.params.b_key.row_len())
        {
            return Err(HachiError::InvalidInput(
                "batched_prove received a commitment with the wrong length".to_string(),
            ));
        }

        // The selected schedule is the source of truth for the root handoff
        // into the first recursive commitment.
        let raw = prove_root_level::<F, T, D, Cfg, P>(
            &setup.expanded,
            &setup.ntt_shared,
            &mut commit_ntt_cache,
            &prepared_claims.flat_polys,
            &prepared_claims.batch_shape,
            &prepared_points,
            &prepared_claims.commitments_by_point,
            root_key,
            &schedule,
            prepared_claims.flat_hints,
            transcript,
            &root_step.params,
        )?;

        let (proof, total_levels) =
            build_folded_batched_proof_with_suffix::<F, D, _>(raw, |next_state| {
                prove_recursive_suffix::<F, T, D, Cfg>(
                    setup,
                    &mut ntt_cache,
                    &mut commit_ntt_cache,
                    max_num_vars,
                    transcript,
                    next_state,
                    &schedule,
                )
            })?;

        tracing::info!(
            levels = total_levels,
            elapsed_s = t_prove_total.elapsed().as_secs_f64(),
            "hachi batched prove complete"
        );

        Ok(proof)
    }
}

impl<F, const D: usize, Cfg> CommitmentVerifier<F, D> for HachiCommitmentScheme<D, Cfg>
where
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
    Cfg: CommitmentConfig<Field = F>,
{
    type VerifierSetup = HachiVerifierSetup<F>;
    type Commitment = RingCommitment<F, D>;
    type BatchedProof = HachiBatchedProof<F>;

    #[tracing::instrument(skip_all, name = "HachiCommitmentScheme::batched_verify")]
    fn batched_verify<'a, T: Transcript<F>>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        claims: VerifierClaims<'a, F, Self::Commitment>,
        basis: BasisMode,
    ) -> Result<(), HachiError> {
        let prepared_claims = prepare_verifier_claims(&setup.expanded, &claims)?;
        let num_vars = prepared_claims.num_vars;
        let layout_num_claims = prepared_claims.layout_num_claims;
        let batch_summary = prepared_claims.batch_summary;

        let t_verify_hachi = Instant::now();
        let max_num_vars = setup.expanded.seed.max_num_vars;
        let schedule =
            Cfg::get_params_for_prove(max_num_vars, num_vars, layout_num_claims, batch_summary)
                .map_err(|_| HachiError::InvalidProof)?;

        let schedule_context = match schedule.steps.first() {
            Some(Step::Direct(_)) => BatchedVerifierScheduleContext::RootDirect,
            Some(Step::Fold(root_step)) => {
                let root_inputs = HachiScheduleInputs {
                    max_num_vars,
                    level: 0,
                    current_w_len: root_step.current_w_len,
                };
                let level_lp = &root_step.params;
                let root_lp =
                    Cfg::root_level_params_for_layout_with_log_basis(root_inputs, level_lp)
                        .map_err(|_| HachiError::InvalidProof)?;
                let next_inputs = HachiScheduleInputs {
                    max_num_vars,
                    level: 1,
                    current_w_len: root_step.next_w_len,
                };
                let next_level_params =
                    scheduled_next_level_params::<Cfg>(&schedule, 1, next_inputs)
                        .map_err(|_| HachiError::InvalidProof)?;
                BatchedVerifierScheduleContext::Fold(Box::new(FoldVerifierLayouts {
                    root_lp,
                    next_level_params,
                }))
            }
            None => return Err(HachiError::InvalidProof),
        };

        verify_batched_proof_with_schedule::<F, T, D, _>(
            proof,
            setup,
            transcript,
            prepared_claims,
            basis,
            &schedule,
            schedule_context,
            |witnesses, commitments, batch_shape| {
                let total_claims =
                    checked_total_claims(&batch_shape.claim_group_sizes, "root_direct_verify")
                        .map_err(|_| HachiError::InvalidProof)?;
                let params = Cfg::get_params_for_commitment(num_vars, total_claims)
                    .map_err(|_| HachiError::InvalidProof)?;
                verify_root_direct_commitments_with_params::<F, D>(
                    witnesses,
                    setup,
                    commitments,
                    batch_shape,
                    &params,
                )
            },
        )?;

        tracing::info!(
            levels = proof.num_fold_levels() + 1,
            elapsed_s = t_verify_hachi.elapsed().as_secs_f64(),
            "hachi batched verify complete"
        );

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Hachi"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::commitment::hachi_batched_root_layout;
    use crate::protocol::commitment::schedule::{root_current_w_len, scale_batched_root_layout};
    use crate::protocol::config::proof_optimized::fp128;
    use crate::protocol::config::CommitmentConfig;
    use crate::{
        CommitmentProver, CommittedPolynomials, FromSmallInt, HachiDeserialize, HachiSerialize,
    };
    use akita_algebra::CyclotomicRing;
    use akita_prover::ring_switch::{ring_switch_build_w, ring_switch_finalize_with_claim_groups};
    use akita_prover::{DensePoly, HachiPolyOps, OneHotPoly, QuadraticEquation};
    use akita_transcript::labels::{
        ABSORB_EVALUATION_CLAIMS, ABSORB_EVAL_OPENINGS_FIELD, CHALLENGE_EVAL_BATCH,
    };
    use akita_transcript::Blake2bTranscript;
    use akita_types::stage1_tree_stage_shapes;
    use akita_types::BlockOrder;
    use akita_types::HachiRootBatchSummary;
    use akita_types::{
        append_batched_commitments_to_transcript, flatten_batched_commitment_rows,
        lagrange_weights, monomial_weights, reduce_inner_opening_to_ring_element,
        relation_claim_from_rows, ring_opening_point_from_field,
    };
    use akita_types::{
        r_decomp_levels, w_ring_element_count, w_ring_element_count_with_num_claims,
    };
    use akita_types::{CommitmentVerifier, CommittedOpenings};
    use akita_types::{HachiBatchedProofShape, HachiProofStepShape, LevelProofShape};
    use akita_verifier::direct_witness_opening_matches;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::sync::Once;
    use tracing_subscriber::fmt::format::FmtSpan;
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::EnvFilter;
    type Cfg = fp128::D64Full;
    type F = fp128::Field;
    const D: usize = Cfg::D;
    type Scheme = HachiCommitmentScheme<D, Cfg>;
    type OneHotF = fp128::Field;
    type OneHotCfg = fp128::D64OneHot;
    const ONEHOT_D: usize = OneHotCfg::D;
    const BENCH_ONEHOT_K: usize = ONEHOT_D;
    type OneHotScheme = HachiCommitmentScheme<ONEHOT_D, OneHotCfg>;
    /// Minimum w vector length (in field elements) below which further folding
    /// is not beneficial.  When `w.len() <= MIN_W_LEN_FOR_FOLDING`, the prover
    /// sends `w` directly instead of recursing.
    const MIN_W_LEN_FOR_FOLDING: usize = 4096;

    fn batched_shape_rounds(level_d: usize, next_w_len: usize) -> usize {
        let num_ring_elems = next_w_len / level_d;
        num_ring_elems.next_power_of_two().trailing_zeros() as usize
            + level_d.trailing_zeros() as usize
    }

    /// Batched recursion already consults the byte planner before folding
    /// again. The runtime safety guard here only needs to catch tiny tails and
    /// fixed points, not enforce the single-proof shrink-ratio heuristic.
    fn should_stop_batched_folding(w_len: usize, prev_w_len: usize) -> bool {
        w_len <= MIN_W_LEN_FOR_FOLDING || w_len >= prev_w_len
    }

    #[test]
    fn same_point_batched_root_preserves_opening_geometry() {
        for num_claims in [4usize, 6] {
            let batch =
                HachiRootBatchSummary::new(num_claims, 1, 1).expect("same-point batch summary");
            let root_key = HachiScheduleLookupKey::with_batch(20, 20, num_claims, batch);
            let schedule = OneHotCfg::get_params_for_prove(20, 20, num_claims, batch)
                .expect("same-point root plan");
            let Some(Step::Fold(root_step)) = schedule.steps.first() else {
                panic!("same-point schedule should start with a fold");
            };
            let root_inputs = HachiScheduleInputs {
                max_num_vars: root_key.max_num_vars,
                level: 0,
                current_w_len: root_step.current_w_len,
            };
            let level_lp = &root_step.params;
            let root_lp =
                OneHotCfg::root_level_params_for_layout_with_log_basis(root_inputs, level_lp)
                    .unwrap();
            assert_eq!(root_lp.block_len, level_lp.block_len);
            assert_eq!(root_lp.num_blocks, level_lp.num_blocks);
            assert_eq!(root_lp.m_vars, level_lp.m_vars);
            assert_eq!(root_lp.r_vars, level_lp.r_vars);
        }
    }

    fn expected_same_point_batched_shape(
        max_num_vars: usize,
        num_claims: usize,
        proof: &HachiBatchedProof<OneHotF>,
    ) -> HachiBatchedProofShape {
        let batch = HachiRootBatchSummary::new(num_claims, 1, 1).expect("same-point batch summary");
        let schedule =
            OneHotCfg::get_params_for_prove(max_num_vars, max_num_vars, num_claims, batch)
                .expect("batched root runtime plan");
        let Some(Step::Fold(root_step)) = schedule.steps.first() else {
            panic!("batched schedule should start with a fold");
        };
        let root_inputs = HachiScheduleInputs {
            max_num_vars,
            level: 0,
            current_w_len: root_step.current_w_len,
        };
        let level_lp = &root_step.params;
        let root_lp =
            OneHotCfg::root_level_params_for_layout_with_log_basis(root_inputs, level_lp).unwrap();
        let next_inputs = HachiScheduleInputs {
            max_num_vars,
            level: 1,
            current_w_len: root_step.next_w_len,
        };
        let next_level_params =
            scheduled_next_level_params::<OneHotCfg>(&schedule, 1, next_inputs).unwrap();
        let root_w_len = next_inputs.current_w_len;
        let root_rounds = batched_shape_rounds(root_lp.ring_dimension, root_w_len);
        let root_shape = LevelProofShape {
            y_ring_coeffs: batch.num_points * root_lp.ring_dimension,
            v_coeffs: root_lp.d_key.row_len() * root_lp.ring_dimension,
            stage1_stages: stage1_tree_stage_shapes(root_rounds, 1usize << level_lp.log_basis),
            stage2_sumcheck: (root_rounds, 3),
            next_commit_coeffs: next_level_params.b_key.row_len()
                * next_level_params.ring_dimension,
        };
        let first_level_params = next_level_params.clone();

        let mut step_shapes = Vec::with_capacity(proof.num_fold_levels() + 1);
        let mut current_w_len = root_w_len;
        let mut current_log_basis = first_level_params.log_basis;
        let mut current_level = 1usize;
        for _ in proof.fold_levels() {
            let inputs = HachiScheduleInputs {
                max_num_vars,
                level: current_level,
                current_w_len,
            };
            let (level_params, next_level_params) = scheduled_fold_execution::<OneHotCfg>(
                &schedule,
                current_level,
                inputs,
                current_log_basis,
            )
            .expect("scheduled recursive fold");
            let current_lp =
                hachi_recursive_level_layout_from_params::<OneHotCfg>(&level_params, current_w_len)
                    .expect("recursive layout");
            let next_w_len =
                w_ring_element_count::<OneHotF>(&current_lp) * current_lp.ring_dimension;
            let rounds = batched_shape_rounds(current_lp.ring_dimension, next_w_len);
            step_shapes.push(HachiProofStepShape::Fold(LevelProofShape {
                y_ring_coeffs: current_lp.ring_dimension,
                v_coeffs: current_lp.d_key.row_len() * current_lp.ring_dimension,
                stage1_stages: stage1_tree_stage_shapes(rounds, 1usize << current_lp.log_basis),
                stage2_sumcheck: (rounds, 3),
                next_commit_coeffs: next_level_params.b_key.row_len()
                    * next_level_params.ring_dimension,
            }));
            current_w_len = next_w_len;
            current_log_basis = next_level_params.log_basis;
            current_level += 1;
        }
        step_shapes.push(HachiProofStepShape::Direct(
            akita_types::DirectWitnessShape::PackedDigits((current_w_len, current_log_basis)),
        ));

        HachiBatchedProofShape::Fold {
            root_shape,
            step_shapes,
        }
    }

    fn make_dense_poly(num_vars: usize) -> (DensePoly<F, D>, Vec<F>) {
        let len = 1usize << num_vars;
        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();
        (poly, evals)
    }

    #[test]
    fn batched_suffix_stop_guard_does_not_preempt_profitable_fold() {
        // These states came from the batched onehot nv=32 profile runs that
        // regressed after a generic shrink-ratio guard was briefly added to
        // the batched suffix. The runtime guard should not stop folding here.
        assert!(!should_stop_batched_folding(87_744, 140_672));
        assert!(!should_stop_batched_folding(129_216, 224_064));
    }

    fn make_verify_fixture(
        num_vars: usize,
    ) -> (
        HachiVerifierSetup<F>,
        RingCommitment<F, D>,
        HachiBatchedProof<F>,
        Vec<F>,
        F,
        LevelParams,
    ) {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(num_vars).unwrap();
        let full_num_vars = layout.m_vars + layout.r_vars + alpha;

        let (poly, evals) = make_dense_poly(full_num_vars);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(full_num_vars, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .unwrap();

        let opening_point: Vec<F> = (0..full_num_vars)
            .map(|i| F::from_u64((i + 2) as u64))
            .collect();
        let lw = lagrange_weights(&opening_point);
        let opening: F = evals
            .iter()
            .zip(lw.iter())
            .fold(F::zero(), |a, (&c, &w)| a + c * w);

        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let commitments = [commitment];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            vec![(
                &opening_point[..],
                vec![CommittedPolynomials {
                    polynomials: &poly_refs[..],
                    commitment: &commitments[0],
                    hint,
                }],
            )],
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let [commitment] = commitments;
        (
            verifier_setup,
            commitment,
            proof,
            opening_point,
            opening,
            layout,
        )
    }

    fn dense_opening(evals: &[F], point: &[F]) -> F {
        let lw = lagrange_weights(point);
        evals
            .iter()
            .zip(lw.iter())
            .fold(F::zero(), |a, (&c, &w)| a + c * w)
    }

    fn init_debug_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let fmt_layer = tracing_subscriber::fmt::layer()
                .compact()
                .with_target(false)
                .with_span_events(FmtSpan::CLOSE);
            tracing_subscriber::registry()
                .with(EnvFilter::new("info"))
                .with(fmt_layer)
                .init();
        });
    }

    fn init_debug_rayon_pool() {
        #[cfg(feature = "parallel")]
        {
            static INIT: Once = Once::new();
            INIT.call_once(|| {
                rayon::ThreadPoolBuilder::new()
                    .stack_size(64 * 1024 * 1024)
                    .build_global()
                    .ok();
            });
        }
    }

    fn run_debug_on_large_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(f)
            .expect("failed to spawn debug thread")
            .join()
            .expect("debug thread panicked");
    }

    fn debug_random_point(nv: usize) -> Vec<OneHotF> {
        let mut rng = StdRng::seed_from_u64(0xcafe_babe);
        (0..nv)
            .map(|_| OneHotF::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect()
    }

    fn debug_make_onehot_poly(
        layout: &LevelParams,
        seed: u64,
    ) -> OneHotPoly<OneHotF, ONEHOT_D, u8> {
        let total_ring = layout.num_blocks * layout.block_len;
        let num_vars = layout.m_vars + layout.r_vars + ONEHOT_D.trailing_zeros() as usize;
        assert_eq!(total_ring * BENCH_ONEHOT_K, 1usize << num_vars);

        let mut rng = StdRng::seed_from_u64(seed);
        let indices: Vec<Option<u8>> = (0..total_ring)
            .map(|_| Some(rng.gen_range(0..BENCH_ONEHOT_K) as u8))
            .collect();

        OneHotPoly::<OneHotF, ONEHOT_D, u8>::new(BENCH_ONEHOT_K, indices)
            .expect("debug onehot poly")
    }

    fn debug_opening_from_poly<P: HachiPolyOps<OneHotF, ONEHOT_D>>(
        poly: &P,
        point: &[OneHotF],
        layout: &LevelParams,
    ) -> OneHotF {
        let alpha_bits = ONEHOT_D.trailing_zeros() as usize;
        assert_eq!(point.len(), alpha_bits + layout.m_vars + layout.r_vars);

        let inner_point = &point[..alpha_bits];
        let reduced_point = &point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field(
            reduced_point,
            layout.r_vars,
            layout.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("debug opening point");

        let (y_ring, _) = poly.evaluate_and_fold(
            &ring_opening_point.b,
            &ring_opening_point.a,
            layout.block_len,
        );
        let v = reduce_inner_opening_to_ring_element::<OneHotF, ONEHOT_D>(
            inner_point,
            BasisMode::Lagrange,
        )
        .expect("debug inner opening point");
        (y_ring * v.sigma_m1()).coefficients()[0]
    }

    fn debug_relation_sum_from_tables(
        w_evals_compact: &[i8],
        _live_x_cols: usize,
        alpha_evals_y: &[OneHotF],
        m_evals_x: &[OneHotF],
        start_x: usize,
        end_x: usize,
    ) -> OneHotF {
        let mut acc = OneHotF::zero();
        for x in start_x..end_x {
            let mut y_eval = OneHotF::zero();
            for (y, alpha_eval) in alpha_evals_y.iter().enumerate() {
                y_eval += *alpha_eval
                    * OneHotF::from_i64(w_evals_compact[x * alpha_evals_y.len() + y] as i64);
            }
            acc += y_eval * m_evals_x[x];
        }
        acc
    }

    #[test]
    fn commit_singleton_group_returns_single_claim_hint() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let (poly, _) = make_dense_poly(num_vars);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 1, 1);

        let (_, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .unwrap();

        assert_eq!(hint.inner_opening_digits.len(), 1);
        assert_eq!(hint.t().unwrap().len(), 1);
    }

    #[test]
    #[ignore = "manual tracing-only relation-claim check"]
    fn debug_batched_root_relation_claim_matches_tables() {
        init_debug_tracing();
        init_debug_rayon_pool();
        run_debug_on_large_stack(|| {
            const BATCH_NUM_VARS: usize = 29;
            const BATCH_SIZE: usize = 1 << 5;

            let batch_layout = hachi_batched_root_layout::<OneHotCfg>(BATCH_NUM_VARS, BATCH_SIZE)
                .expect("batch debug layout");
            let batched_root_lp = scale_batched_root_layout::<OneHotCfg>(&batch_layout, BATCH_SIZE)
                .expect("batched debug root layout");
            let batch_root_inputs = HachiScheduleInputs {
                max_num_vars: BATCH_NUM_VARS,
                level: 0,
                current_w_len: root_current_w_len(&batch_layout),
            };
            let batch_root_params = OneHotCfg::level_params_with_log_basis(
                batch_root_inputs,
                OneHotCfg::log_basis_at_level(batch_root_inputs),
            );

            let batch_polys: Vec<OneHotPoly<OneHotF, ONEHOT_D, u8>> = (0..BATCH_SIZE)
                .map(|idx| {
                    debug_make_onehot_poly(&batch_layout, 0x0bee_fcaf_e000_2900 + idx as u64)
                })
                .collect();
            let batch_setup = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_prover(
                BATCH_NUM_VARS,
                BATCH_SIZE,
                1,
            );
            let batch_poly_refs: Vec<&OneHotPoly<OneHotF, ONEHOT_D, u8>> =
                batch_polys.iter().collect();
            let (batch_commitment, batch_hint) =
                <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::commit(
                    &batch_poly_refs,
                    &batch_setup,
                )
                .expect("batched debug commit");
            let batch_commitments = [batch_commitment];
            let batch_hints = vec![batch_hint];
            let batch_commitment_rows = flatten_batched_commitment_rows(&batch_commitments);

            let batch_point = debug_random_point(BATCH_NUM_VARS);
            let alpha = batch_root_params.ring_dimension.trailing_zeros() as usize;
            let target_num_vars = batch_layout.m_vars + batch_layout.r_vars + alpha;
            let mut padded_point = batch_point.clone();
            padded_point.resize(target_num_vars, OneHotF::zero());
            let outer_point = &padded_point[alpha..];
            let ring_opening_point = ring_opening_point_from_field::<OneHotF>(
                outer_point,
                batch_layout.r_vars,
                batch_layout.m_vars,
                BasisMode::Lagrange,
                BlockOrder::RowMajor,
            )
            .expect("debug opening point");
            let inner_reduction = reduce_inner_opening_to_ring_element::<OneHotF, ONEHOT_D>(
                &padded_point[..alpha],
                BasisMode::Lagrange,
            )
            .expect("debug inner reduction");
            let (y_rings, w_folded_by_poly): (Vec<_>, Vec<_>) = batch_polys
                .iter()
                .map(|poly| {
                    poly.evaluate_and_fold(
                        &ring_opening_point.b,
                        &ring_opening_point.a,
                        batch_layout.block_len,
                    )
                })
                .unzip();

            let mut transcript = Blake2bTranscript::<OneHotF>::new(b"debug/relation-claim/batched");
            append_batched_commitments_to_transcript(&batch_commitments, &mut transcript);
            for pt in &padded_point {
                transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
            }
            let field_openings: Vec<OneHotF> = y_rings
                .iter()
                .map(|y_ring| (*y_ring * inner_reduction.sigma_m1()).coefficients()[0])
                .collect();
            for opening in &field_openings {
                transcript.append_field(ABSORB_EVAL_OPENINGS_FIELD, opening);
            }
            let batch_gammas: Vec<OneHotF> = (0..batch_poly_refs.len())
                .map(|_| transcript.challenge_scalar(CHALLENGE_EVAL_BATCH))
                .collect();
            let batched_y_rings: Vec<CyclotomicRing<OneHotF, ONEHOT_D>> = {
                let mut combined = CyclotomicRing::<OneHotF, ONEHOT_D>::zero();
                for (claim_idx, y) in y_rings.iter().enumerate() {
                    combined += y.scale(&batch_gammas[claim_idx]);
                }
                vec![combined]
            };
            for y_ring in &batched_y_rings {
                transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
            }

            let debug_batch_hint = batch_hints[0].clone();
            let debug_w_folded_by_poly: Vec<Vec<CyclotomicRing<OneHotF, ONEHOT_D>>> =
                w_folded_by_poly.clone();
            let mut quad_eq = Box::new(
                QuadraticEquation::<OneHotF, { ONEHOT_D }>::new_prover(
                    &batch_setup.ntt_shared,
                    vec![ring_opening_point.clone()],
                    vec![0usize; BATCH_SIZE],
                    &batch_poly_refs,
                    w_folded_by_poly,
                    &[BATCH_SIZE],
                    batched_root_lp.clone(),
                    batch_hints,
                    &mut transcript,
                    &batch_commitments,
                    &batched_y_rings,
                    batch_gammas,
                    batch_setup.expanded.seed.max_stride,
                )
                .expect("debug batched quadratic equation"),
            );
            let w = ring_switch_build_w::<OneHotF, { ONEHOT_D }>(
                &mut quad_eq,
                &batch_setup.expanded,
                &batch_setup.ntt_shared,
                &batched_root_lp,
            )
            .expect("debug batched w");
            let commit_inputs = HachiScheduleInputs {
                max_num_vars: BATCH_NUM_VARS,
                level: 1,
                current_w_len: w.len(),
            };
            let commit_params = OneHotCfg::level_params_with_log_basis(
                commit_inputs,
                OneHotCfg::log_basis_at_level(commit_inputs),
            );
            let mut commit_ntt_cache = MultiDNttCaches::default();
            let (w_commitment_flat, w_hint_cache) = dispatch_commit::<OneHotF, OneHotCfg>(
                commit_params,
                &mut commit_ntt_cache,
                &batch_setup.expanded,
                &w,
            )
            .expect("debug batched w commit");
            let w_commitment_proof = w_commitment_flat.clone();
            let rs = ring_switch_finalize_with_claim_groups::<OneHotF, _, { ONEHOT_D }>(
                &quad_eq,
                &batch_setup.expanded,
                &mut transcript,
                w,
                w_commitment_flat,
                &w_commitment_proof,
                w_hint_cache,
                &batched_root_lp,
            )
            .expect("debug batched ring switch");

            let relation_claim = relation_claim_from_rows::<OneHotF, ONEHOT_D>(
                &rs.tau1,
                rs.alpha,
                &quad_eq.v,
                &batch_commitment_rows,
                &batched_y_rings,
            );
            let relation_sum = debug_relation_sum_from_tables(
                &rs.w_evals_compact,
                rs.live_x_cols,
                &rs.alpha_evals_y,
                &rs.m_evals_x,
                0,
                rs.live_x_cols,
            );
            let w_alpha_evals: Vec<OneHotF> = (0..rs.live_x_cols)
                .map(|x| {
                    rs.alpha_evals_y.iter().enumerate().fold(
                        OneHotF::zero(),
                        |acc, (y, alpha_eval)| {
                            acc + *alpha_eval
                                * OneHotF::from_i64(
                                    rs.w_evals_compact[x * rs.alpha_evals_y.len() + y] as i64,
                                )
                        },
                    )
                })
                .collect();
            let w_hat_len =
                batched_root_lp.num_digits_open * batched_root_lp.num_blocks * BATCH_SIZE;
            let t_hat_len = batched_root_lp.num_digits_open
                * batch_root_params.a_key.row_len()
                * batched_root_lp.num_blocks
                * BATCH_SIZE;
            let z_pre_len = batched_root_lp.inner_width() * batched_root_lp.num_digits_fold;
            let num_commitment_groups = 1usize;
            let num_eval_rows = 1usize;
            let m_rows = batch_root_params.m_row_count(num_commitment_groups, num_eval_rows);
            let r_tail_len = m_rows * r_decomp_levels::<OneHotF>(batched_root_lp.log_basis);
            let w_hat_relation_sum = debug_relation_sum_from_tables(
                &rs.w_evals_compact,
                rs.live_x_cols,
                &rs.alpha_evals_y,
                &rs.m_evals_x,
                0,
                w_hat_len,
            );
            let t_hat_relation_sum = debug_relation_sum_from_tables(
                &rs.w_evals_compact,
                rs.live_x_cols,
                &rs.alpha_evals_y,
                &rs.m_evals_x,
                w_hat_len,
                w_hat_len + t_hat_len,
            );
            let z_pre_relation_sum = debug_relation_sum_from_tables(
                &rs.w_evals_compact,
                rs.live_x_cols,
                &rs.alpha_evals_y,
                &rs.m_evals_x,
                w_hat_len + t_hat_len,
                w_hat_len + t_hat_len + z_pre_len,
            );
            let r_tail_relation_sum = debug_relation_sum_from_tables(
                &rs.w_evals_compact,
                rs.live_x_cols,
                &rs.alpha_evals_y,
                &rs.m_evals_x,
                w_hat_len + t_hat_len + z_pre_len,
                w_hat_len + t_hat_len + z_pre_len + r_tail_len,
            );
            let eq_tau1 = akita_algebra::eq_poly::EqPolynomial::evals(&rs.tau1);
            // Row layout: consistency (1) | public (1) | D (n_d) |
            //             B (n_b * num_commitment_groups) | A (n_a)
            let consistency_weight = eq_tau1[0];
            let public_weight = eq_tau1[1];
            let d_start = 2usize;
            let b_start = d_start + batch_root_params.d_key.row_len();
            let a_start = b_start + batch_root_params.b_key.row_len() * num_commitment_groups;
            let a_weights = &eq_tau1[a_start..m_rows];
            let alpha_pows = &rs.alpha_evals_y;
            let eval_sparse_alpha = |challenge: &akita_algebra::SparseChallenge| -> OneHotF {
                challenge
                    .positions
                    .iter()
                    .zip(challenge.coeffs.iter())
                    .fold(OneHotF::zero(), |acc, (&pos, &coeff)| {
                        acc + OneHotF::from_i64(coeff as i64) * alpha_pows[pos as usize]
                    })
            };
            let eval_ring_at_pows_local =
                |ring: &CyclotomicRing<OneHotF, ONEHOT_D>, pows: &[OneHotF]| -> OneHotF {
                    ring.coefficients()
                        .iter()
                        .zip(pows.iter())
                        .fold(OneHotF::zero(), |acc, (coeff, alpha_pow)| {
                            acc + *coeff * *alpha_pow
                        })
                };
            let c_alphas: Vec<OneHotF> = quad_eq.challenges.iter().map(eval_sparse_alpha).collect();
            let gadget_scalars = |levels: usize| -> Vec<OneHotF> {
                let base = OneHotF::from_canonical_u128_reduced(1u128 << batched_root_lp.log_basis);
                let mut out = Vec::with_capacity(levels);
                let mut power = OneHotF::one();
                for _ in 0..levels {
                    out.push(power);
                    power *= base;
                }
                out
            };
            let g1_open = gadget_scalars(batched_root_lp.num_digits_open);
            let g1_commit = gadget_scalars(batched_root_lp.num_digits_commit);
            let fold_gadget = gadget_scalars(batched_root_lp.num_digits_fold);
            let r_gadget = gadget_scalars(r_decomp_levels::<OneHotF>(batched_root_lp.log_basis));
            let debug_stride = batch_setup.expanded.seed.max_stride;
            let d_view = batch_setup
                .expanded
                .shared_matrix
                .ring_view::<ONEHOT_D>(batch_root_params.d_key.row_len(), debug_stride);
            let b_view = batch_setup
                .expanded
                .shared_matrix
                .ring_view::<ONEHOT_D>(batch_root_params.b_key.row_len(), debug_stride);
            let a_view = batch_setup
                .expanded
                .shared_matrix
                .ring_view::<ONEHOT_D>(batch_root_params.a_key.row_len(), debug_stride);
            let denom = alpha_pows[ONEHOT_D - 1] * rs.alpha + OneHotF::one();
            let expected_d_sum = quad_eq
                .v
                .iter()
                .enumerate()
                .take(batch_root_params.d_key.row_len())
                .fold(OneHotF::zero(), |acc, (di, row)| {
                    acc + eq_tau1[d_start + di] * akita_algebra::ring::eval_ring_at(row, &rs.alpha)
                });
            let expected_b_sum =
                batch_commitment_rows
                    .iter()
                    .enumerate()
                    .fold(OneHotF::zero(), |acc, (bi, row)| {
                        acc + eq_tau1[b_start + bi]
                            * akita_algebra::ring::eval_ring_at(row, &rs.alpha)
                    });
            let expected_public_sum =
                public_weight * akita_algebra::ring::eval_ring_at(&batched_y_rings[0], &rs.alpha);
            let stored_t_by_poly = debug_batch_hint
                .t()
                .expect("debug batched stored t rows")
                .to_vec();
            let mut debug_hint_flat = debug_batch_hint;
            debug_hint_flat
                .ensure_t_recomposed(batched_root_lp.num_digits_open, batched_root_lp.log_basis)
                .expect("debug batched t recomposition");
            let (debug_t_hat, debug_t) = debug_hint_flat.into_flat_parts();
            let _debug_t_hat_flat = debug_t_hat.flat_digits().to_vec();
            let debug_t = debug_t.expect("debug batched t rows");
            let debug_w_folded_flat: Vec<_> = debug_w_folded_by_poly
                .clone()
                .into_iter()
                .flatten()
                .collect();
            let debug_w_hat: Vec<Vec<[i8; ONEHOT_D]>> = debug_w_folded_by_poly
                .iter()
                .flat_map(|folded_rows| {
                    folded_rows.iter().map(|w_i| {
                        w_i.balanced_decompose_pow2_i8(
                            batched_root_lp.num_digits_open,
                            batched_root_lp.log_basis,
                        )
                    })
                })
                .collect();
            let debug_w_hat_flat: Vec<_> = debug_w_hat
                .iter()
                .flat_map(|block| block.iter().copied())
                .collect();
            let mut debug_z_witnesses = batch_polys
                .iter()
                .zip(quad_eq.challenges.chunks(batched_root_lp.num_blocks))
                .map(|(poly, poly_challenges)| {
                    poly.decompose_fold(
                        poly_challenges,
                        batched_root_lp.block_len,
                        batched_root_lp.num_digits_commit,
                        batched_root_lp.log_basis,
                    )
                });
            let mut debug_z = debug_z_witnesses.next().expect("debug batched z witness");
            for witness in debug_z_witnesses {
                for (dst, src) in debug_z.z_pre.iter_mut().zip(witness.z_pre.iter()) {
                    *dst += *src;
                }
                for (dst, src) in debug_z
                    .centered_coeffs
                    .iter_mut()
                    .zip(witness.centered_coeffs.iter())
                {
                    for k in 0..ONEHOT_D {
                        dst[k] += src[k];
                    }
                }
            }
            debug_z.centered_inf_norm = debug_z
                .centered_coeffs
                .iter()
                .flat_map(|coeffs| coeffs.iter())
                .map(|coeff| coeff.unsigned_abs())
                .max()
                .unwrap_or(0);
            let debug_y = akita_prover::quadratic_equation::generate_y::<OneHotF, ONEHOT_D>(
                &quad_eq.v,
                &batch_commitment_rows,
                &batched_y_rings,
                batch_root_params.d_key.row_len(),
                batch_root_params.b_key.row_len(),
                batch_root_params.a_key.row_len(),
            )
            .expect("debug batched y");
            let debug_r =
                akita_prover::quadratic_equation::compute_r_split_eq::<OneHotF, ONEHOT_D>(
                    &batched_root_lp,
                    &batch_setup.expanded,
                    &quad_eq.challenges,
                    &debug_w_hat_flat,
                    &debug_t_hat,
                    &debug_t,
                    &debug_w_folded_flat,
                    &debug_z.centered_coeffs,
                    debug_z.centered_inf_norm,
                    &debug_y,
                    &[BATCH_SIZE],
                    1,
                    batched_root_lp.num_blocks,
                    batched_root_lp.inner_width(),
                    batch_setup.expanded.seed.max_stride,
                    &batch_setup.ntt_shared,
                )
                .expect("debug batched r");
            let stored_t_flat: Vec<_> = stored_t_by_poly.iter().flatten().cloned().collect();
            let stored_a_t = quad_eq.challenges.iter().zip(stored_t_flat.iter()).fold(
                CyclotomicRing::<OneHotF, ONEHOT_D>::zero(),
                |mut acc, (challenge, block_rows)| {
                    block_rows[0].mul_by_sparse_into(challenge, &mut acc);
                    acc
                },
            );
            let reduced_a_t = quad_eq.challenges.iter().zip(debug_t.iter()).fold(
                CyclotomicRing::<OneHotF, ONEHOT_D>::zero(),
                |mut acc, (challenge, block_rows)| {
                    block_rows[0].mul_by_sparse_into(challenge, &mut acc);
                    acc
                },
            );
            let reduced_a_z = debug_z.z_pre.iter().enumerate().fold(
                CyclotomicRing::<OneHotF, ONEHOT_D>::zero(),
                |mut acc, (k, z_ring)| {
                    a_view.row(0)[k].mul_accumulate_into(z_ring, &mut acc);
                    acc
                },
            );
            let reduced_a_diff = reduced_a_t - reduced_a_z;
            let direct_raw_a_t = c_alphas.iter().zip(debug_t.iter()).fold(
                OneHotF::zero(),
                |acc, (c_alpha, block_rows)| {
                    acc + *c_alpha * akita_algebra::ring::eval_ring_at(&block_rows[0], &rs.alpha)
                },
            );
            let direct_raw_a_z =
                debug_z
                    .z_pre
                    .iter()
                    .enumerate()
                    .fold(OneHotF::zero(), |acc, (k, z_ring)| {
                        acc - eval_ring_at_pows_local(&a_view.row(0)[k], alpha_pows)
                            * akita_algebra::ring::eval_ring_at(z_ring, &rs.alpha)
                    });
            let direct_raw_a_r =
                -(denom * akita_algebra::ring::eval_ring_at(&debug_r[a_start], &rs.alpha));
            let direct_raw_a_total = direct_raw_a_t + direct_raw_a_z + direct_raw_a_r;
            let d_matrix_width = batched_root_lp.d_matrix_width();
            let d_group_w = (0..w_hat_len).fold(OneHotF::zero(), |acc, x| {
                let coeff =
                    (0..batch_root_params.d_key.row_len()).fold(OneHotF::zero(), |inner, di| {
                        inner
                            + eq_tau1[d_start + di]
                                * eval_ring_at_pows_local(
                                    &d_view.row(di)[x % d_matrix_width],
                                    alpha_pows,
                                )
                    });
                acc + w_alpha_evals[x] * coeff
            });
            let d_group_r =
                (0..batch_root_params.d_key.row_len()).fold(OneHotF::zero(), |acc, di| {
                    let row_idx = d_start + di;
                    let row_start = w_hat_len + t_hat_len + z_pre_len + row_idx * r_gadget.len();
                    acc + (0..r_gadget.len()).fold(OneHotF::zero(), |inner, level_idx| {
                        inner
                            + w_alpha_evals[row_start + level_idx]
                                * (-(eq_tau1[row_idx] * denom * r_gadget[level_idx]))
                    })
                });
            let outer_width = batched_root_lp.outer_width();
            let b_group_t = (0..t_hat_len).fold(OneHotF::zero(), |acc, x| {
                let coeff =
                    (0..batch_root_params.b_key.row_len()).fold(OneHotF::zero(), |inner, bi| {
                        inner
                            + eq_tau1[b_start + bi]
                                * eval_ring_at_pows_local(
                                    &b_view.row(bi)[x % outer_width],
                                    alpha_pows,
                                )
                    });
                acc + w_alpha_evals[w_hat_len + x] * coeff
            });
            let b_group_r =
                (0..batch_root_params.b_key.row_len()).fold(OneHotF::zero(), |acc, bi| {
                    let row_idx = b_start + bi;
                    let row_start = w_hat_len + t_hat_len + z_pre_len + row_idx * r_gadget.len();
                    acc + (0..r_gadget.len()).fold(OneHotF::zero(), |inner, level_idx| {
                        inner
                            + w_alpha_evals[row_start + level_idx]
                                * (-(eq_tau1[row_idx] * denom * r_gadget[level_idx]))
                    })
                });
            let public_group_w = (0..w_hat_len).fold(OneHotF::zero(), |acc, x| {
                let blocks_per_claim = batched_root_lp.num_blocks * batched_root_lp.num_digits_open;
                let claim_idx = x / blocks_per_claim;
                let claim_offset = x % blocks_per_claim;
                let block_idx = claim_offset / batched_root_lp.num_digits_open;
                let digit_idx = claim_offset % batched_root_lp.num_digits_open;
                acc + w_alpha_evals[x]
                    * public_weight
                    * quad_eq.gamma()[claim_idx]
                    * ring_opening_point.b[block_idx]
                    * g1_open[digit_idx]
            });
            // The batched protocol has exactly one public y-row at row index 1.
            let public_group_r = {
                let row_idx = 1usize;
                let row_start = w_hat_len + t_hat_len + z_pre_len + row_idx * r_gadget.len();
                (0..r_gadget.len()).fold(OneHotF::zero(), |inner, level_idx| {
                    inner
                        + w_alpha_evals[row_start + level_idx]
                            * (-(eq_tau1[row_idx] * denom * r_gadget[level_idx]))
                })
            };
            let row4_group_w = (0..w_hat_len).fold(OneHotF::zero(), |acc, x| {
                let blocks_per_claim = batched_root_lp.num_blocks * batched_root_lp.num_digits_open;
                let claim_idx = x / blocks_per_claim;
                let claim_offset = x % blocks_per_claim;
                let block_idx = claim_offset / batched_root_lp.num_digits_open;
                let digit_idx = claim_offset % batched_root_lp.num_digits_open;
                let global_block_idx = claim_idx * batched_root_lp.num_blocks + block_idx;
                acc + w_alpha_evals[x]
                    * consistency_weight
                    * c_alphas[global_block_idx]
                    * g1_open[digit_idx]
            });
            let row4_group_z = (0..z_pre_len).fold(OneHotF::zero(), |acc, idx| {
                let k = idx / batched_root_lp.num_digits_fold;
                let fold_idx = idx % batched_root_lp.num_digits_fold;
                let block_idx = k / batched_root_lp.num_digits_commit;
                let digit_idx = k % batched_root_lp.num_digits_commit;
                acc + w_alpha_evals[w_hat_len + t_hat_len + idx]
                    * (-(consistency_weight
                        * ring_opening_point.a[block_idx]
                        * g1_commit[digit_idx]
                        * fold_gadget[fold_idx]))
            });
            let row4_group_r = {
                let row_start = w_hat_len + t_hat_len + z_pre_len;
                (0..r_gadget.len()).fold(OneHotF::zero(), |acc, level_idx| {
                    acc + w_alpha_evals[row_start + level_idx]
                        * (-(consistency_weight * denom * r_gadget[level_idx]))
                })
            };
            let a_group_t = (0..t_hat_len).fold(OneHotF::zero(), |acc, x| {
                let blocks_per_claim = batch_root_params.a_key.row_len()
                    * batched_root_lp.num_digits_open
                    * batched_root_lp.num_blocks;
                let claim_idx = x / blocks_per_claim;
                let claim_offset = x % blocks_per_claim;
                let block_idx = claim_offset
                    / (batch_root_params.a_key.row_len() * batched_root_lp.num_digits_open);
                let rem = claim_offset
                    % (batch_root_params.a_key.row_len() * batched_root_lp.num_digits_open);
                let a_idx = rem / batched_root_lp.num_digits_open;
                let digit_idx = rem % batched_root_lp.num_digits_open;
                let global_block_idx = claim_idx * batched_root_lp.num_blocks + block_idx;
                acc + w_alpha_evals[w_hat_len + x]
                    * a_weights[a_idx]
                    * c_alphas[global_block_idx]
                    * g1_open[digit_idx]
            });
            let a_group_z = (0..z_pre_len).fold(OneHotF::zero(), |acc, idx| {
                let k = idx / batched_root_lp.num_digits_fold;
                let fold_idx = idx % batched_root_lp.num_digits_fold;
                let block_idx = k / batched_root_lp.num_digits_commit;
                let coeff =
                    a_weights
                        .iter()
                        .enumerate()
                        .fold(OneHotF::zero(), |inner, (a_idx, eq_i)| {
                            inner
                                + *eq_i * eval_ring_at_pows_local(&a_view.row(a_idx)[k], alpha_pows)
                        });
                let _ = block_idx;
                acc + w_alpha_evals[w_hat_len + t_hat_len + idx]
                    * (-(coeff * fold_gadget[fold_idx]))
            });
            let a_group_r =
                a_weights
                    .iter()
                    .enumerate()
                    .fold(OneHotF::zero(), |acc, (row_offset, eq_i)| {
                        let row_idx = a_start + row_offset;
                        let row_start =
                            w_hat_len + t_hat_len + z_pre_len + row_idx * r_gadget.len();
                        acc + (0..r_gadget.len()).fold(OneHotF::zero(), |inner, level_idx| {
                            inner
                                + w_alpha_evals[row_start + level_idx]
                                    * (-(*eq_i * denom * r_gadget[level_idx]))
                        })
                    });

            tracing::info!(
                relation_claim_u128 = relation_claim.to_canonical_u128(),
                relation_sum_u128 = relation_sum.to_canonical_u128(),
                w_hat_relation_sum_u128 = w_hat_relation_sum.to_canonical_u128(),
                t_hat_relation_sum_u128 = t_hat_relation_sum.to_canonical_u128(),
                z_pre_relation_sum_u128 = z_pre_relation_sum.to_canonical_u128(),
                r_tail_relation_sum_u128 = r_tail_relation_sum.to_canonical_u128(),
                d_group_u128 = (d_group_w + d_group_r).to_canonical_u128(),
                expected_d_u128 = expected_d_sum.to_canonical_u128(),
                b_group_u128 = (b_group_t + b_group_r).to_canonical_u128(),
                expected_b_u128 = expected_b_sum.to_canonical_u128(),
                public_group_u128 = (public_group_w + public_group_r).to_canonical_u128(),
                expected_public_u128 = expected_public_sum.to_canonical_u128(),
                row4_group_u128 = (row4_group_w + row4_group_z + row4_group_r).to_canonical_u128(),
                a_group_t_u128 = a_group_t.to_canonical_u128(),
                a_group_z_u128 = a_group_z.to_canonical_u128(),
                a_group_r_u128 = a_group_r.to_canonical_u128(),
                a_group_u128 = (a_group_t + a_group_z + a_group_r).to_canonical_u128(),
                stored_a_ring_matches = stored_a_t == reduced_a_z,
                stored_vs_recomposed_t = stored_t_flat == debug_t,
                reduced_a_ring_matches = reduced_a_t == reduced_a_z,
                reduced_a_diff_alpha_u128 =
                    akita_algebra::ring::eval_ring_at(&reduced_a_diff, &rs.alpha)
                        .to_canonical_u128(),
                direct_raw_a_t_u128 = direct_raw_a_t.to_canonical_u128(),
                direct_raw_a_z_u128 = direct_raw_a_z.to_canonical_u128(),
                direct_raw_a_r_u128 = direct_raw_a_r.to_canonical_u128(),
                direct_raw_a_total_u128 = direct_raw_a_total.to_canonical_u128(),
                live_x_cols = rs.live_x_cols,
                col_bits = rs.col_bits,
                ring_bits = rs.ring_bits,
                "batched relation claim consistency"
            );
            tracing::info!(
                matches = relation_sum == relation_claim,
                "batched relation claim comparison complete"
            );
        });
    }

    #[test]
    #[ignore = "manual tracing-only benchmark breakdown"]
    fn debug_onehot_batched_profile_compare() {
        init_debug_tracing();
        init_debug_rayon_pool();
        run_debug_on_large_stack(|| {
            const SINGLE_NUM_VARS: usize = 34;
            const BATCH_NUM_VARS: usize = 29;
            const BATCH_SIZE: usize = 1 << 5;

            let single_layout =
                OneHotCfg::commitment_layout(SINGLE_NUM_VARS).expect("single debug layout");
            let batch_layout = hachi_batched_root_layout::<OneHotCfg>(BATCH_NUM_VARS, BATCH_SIZE)
                .expect("batch debug layout");
            let batched_root_lp = scale_batched_root_layout::<OneHotCfg>(&batch_layout, BATCH_SIZE)
                .expect("batched debug root layout");

            let single_root_inputs = HachiScheduleInputs {
                max_num_vars: SINGLE_NUM_VARS,
                level: 0,
                current_w_len: root_current_w_len(&single_layout),
            };
            let single_root_params = OneHotCfg::level_params_with_log_basis(
                single_root_inputs,
                OneHotCfg::log_basis_at_level(single_root_inputs),
            );
            let _batch_root_inputs = HachiScheduleInputs {
                max_num_vars: BATCH_NUM_VARS,
                level: 0,
                current_w_len: root_current_w_len(&batch_layout),
            };
            let _batch_root_params = OneHotCfg::level_params_with_log_basis(
                _batch_root_inputs,
                OneHotCfg::log_basis_at_level(_batch_root_inputs),
            );

            let single_root_w_ring = w_ring_element_count::<OneHotF>(&single_root_params);
            let batched_root_w_ring =
                w_ring_element_count_with_num_claims::<OneHotF>(&batched_root_lp, BATCH_SIZE);

            tracing::info!(
                ?single_layout,
                ?batch_layout,
                ?batched_root_lp,
                single_root_w_ring,
                batched_root_w_ring,
                single_root_w_coeffs = single_root_w_ring * ONEHOT_D,
                batched_root_w_coeffs = batched_root_w_ring * ONEHOT_D,
                total_field_single = 1usize << SINGLE_NUM_VARS,
                total_field_batched = BATCH_SIZE * (1usize << BATCH_NUM_VARS),
                "onehot root comparison"
            );

            let single_poly = debug_make_onehot_poly(&single_layout, 0x0bee_fcaf_e000_0034);
            let batch_polys: Vec<OneHotPoly<OneHotF, ONEHOT_D, u8>> = (0..BATCH_SIZE)
                .map(|idx| {
                    debug_make_onehot_poly(&batch_layout, 0x0bee_fcaf_e000_2900 + idx as u64)
                })
                .collect();

            let single_point = debug_random_point(SINGLE_NUM_VARS);
            let batch_point = debug_random_point(BATCH_NUM_VARS);
            let single_opening =
                debug_opening_from_poly(&single_poly, &single_point, &single_layout);
            let batch_openings: Vec<OneHotF> = batch_polys
                .iter()
                .map(|poly| debug_opening_from_poly(poly, &batch_point, &batch_layout))
                .collect();

            let single_setup = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_prover(
                SINGLE_NUM_VARS,
                1,
                1,
            );
            let single_verifier_setup =
                <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_verifier(
                    &single_setup,
                );
            let (single_commitment, single_hint) =
                <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::commit(
                    std::slice::from_ref(&single_poly),
                    &single_setup,
                )
                .expect("single debug commit");

            let single_poly_refs: [&OneHotPoly<OneHotF, ONEHOT_D, u8>; 1] = [&single_poly];
            let single_commitments = [single_commitment];
            let single_openings = [single_opening];
            let single_opening_groups = [&single_openings[..]];

            let _single_prove_span = tracing::info_span!("debug_single_prove").entered();
            let mut single_prover_transcript =
                Blake2bTranscript::<OneHotF>::new(b"debug/onehot/single");
            let single_proof =
                <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::batched_prove(
                    &single_setup,
                    vec![(
                        &single_point[..],
                        vec![CommittedPolynomials {
                            polynomials: &single_poly_refs[..],
                            commitment: &single_commitments[0],
                            hint: single_hint,
                        }],
                    )],
                    &mut single_prover_transcript,
                    BasisMode::Lagrange,
                )
                .expect("single debug prove");
            drop(_single_prove_span);

            let _single_verify_span = tracing::info_span!("debug_single_verify").entered();
            <OneHotScheme as CommitmentVerifier<OneHotF, ONEHOT_D>>::batched_verify(
                &single_proof,
                &single_verifier_setup,
                &mut Blake2bTranscript::<OneHotF>::new(b"debug/onehot/single"),
                vec![(
                    &single_point[..],
                    vec![CommittedOpenings {
                        openings: single_opening_groups[0],
                        commitment: &single_commitments[0],
                    }],
                )],
                BasisMode::Lagrange,
            )
            .expect("single debug verify");
            drop(_single_verify_span);

            let batch_setup = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_prover(
                BATCH_NUM_VARS,
                BATCH_SIZE,
                1,
            );
            let batch_verifier_setup =
                <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_verifier(&batch_setup);
            let (batch_commitment, batch_hint) = <OneHotScheme as CommitmentProver<
                OneHotF,
                ONEHOT_D,
            >>::commit(&batch_polys, &batch_setup)
            .expect("batched debug commit");
            let batch_commitments = [batch_commitment];
            let batch_hints = vec![batch_hint];

            let _batched_prove_span = tracing::info_span!("debug_batched_prove").entered();
            let mut batch_prover_transcript =
                Blake2bTranscript::<OneHotF>::new(b"debug/onehot/batched");
            let batch_proof = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::batched_prove(
                &batch_setup,
                vec![(
                    &batch_point[..],
                    vec![CommittedPolynomials {
                        polynomials: &batch_polys[..],
                        commitment: &batch_commitments[0],
                        hint: batch_hints.into_iter().next().unwrap(),
                    }],
                )],
                &mut batch_prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("batched debug prove");
            drop(_batched_prove_span);

            let _batched_verify_span = tracing::info_span!("debug_batched_verify").entered();
            let batch_opening_groups = [&batch_openings[..]];
            <OneHotScheme as CommitmentVerifier<OneHotF, ONEHOT_D>>::batched_verify(
                &batch_proof,
                &batch_verifier_setup,
                &mut Blake2bTranscript::<OneHotF>::new(b"debug/onehot/batched"),
                vec![(
                    &batch_point[..],
                    vec![CommittedOpenings {
                        openings: batch_opening_groups[0],
                        commitment: &batch_commitments[0],
                    }],
                )],
                BasisMode::Lagrange,
            )
            .expect("batched debug verify");
            drop(_batched_verify_span);
        });
    }

    #[test]
    fn batched_commit_matches_individual_commits() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;
        let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 1) as u64)).collect();
        let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 3 + 7) as u64)).collect();
        let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
        let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 1);
        let poly_groups = [std::slice::from_ref(&poly_a), std::slice::from_ref(&poly_b)];

        let (batched_commitments, batched_hints): (Vec<_>, Vec<_>) = poly_groups
            .iter()
            .map(|group| <Scheme as CommitmentProver<F, D>>::commit(group, &setup))
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .unzip();
        let (commitment_a, hint_a) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly_a), &setup)
                .unwrap();
        let (commitment_b, hint_b) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly_b), &setup)
                .unwrap();

        assert_eq!(batched_commitments, vec![commitment_a, commitment_b]);
        assert_eq!(batched_hints, vec![hint_a, hint_b]);
    }

    /// Exercise the batched root-direct fast path: for a layout/batch shape
    /// whose offline-planned schedule has zero fold levels, the prover must
    /// emit a [`HachiBatchedRootProof::Direct`] variant with no recursive
    /// suffix, and the verifier must accept it via the batched root-direct
    /// checks (per-claim opening + joint per-group re-commit).
    #[test]
    fn batched_root_direct_fast_path_round_trip() {
        // For Cfg = fp128::D64Full with layout_num_claims = 4 and a same-
        // point batch of 4 claims, the generated schedule table is
        // direct-only up to num_vars = 12.
        const NUM_VARS: usize = 8;
        const NUM_POLYS: usize = 4;

        let len = 1usize << NUM_VARS;
        let polys: Vec<DensePoly<F, D>> = (0..NUM_POLYS)
            .map(|poly_idx| {
                let evals: Vec<F> = (0..len)
                    .map(|i| F::from_u64((i * (poly_idx + 1) + 17) as u64))
                    .collect();
                DensePoly::<F, D>::from_field_evals(NUM_VARS, &evals).unwrap()
            })
            .collect();
        let poly_refs: Vec<&DensePoly<F, D>> = polys.iter().collect();

        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NUM_VARS, NUM_POLYS, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(&poly_refs, &setup).unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let opening_point: Vec<F> = (0..NUM_VARS).map(|i| F::from_u64((i + 3) as u64)).collect();
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| {
                let mut evals = vec![F::zero(); len];
                for (i, ring) in poly.coeffs.iter().enumerate() {
                    let base = i * D;
                    let take = (len.saturating_sub(base)).min(D);
                    if take == 0 {
                        break;
                    }
                    evals[base..base + take].copy_from_slice(&ring.coefficients()[..take]);
                }
                let lw = lagrange_weights(&opening_point);
                evals
                    .iter()
                    .zip(lw.iter())
                    .fold(F::zero(), |a, (&c, &w)| a + c * w)
            })
            .collect();

        let poly_group = [&polys[0], &polys[1], &polys[2], &polys[3]];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/batched-root-direct");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            vec![(
                &opening_point[..],
                vec![CommittedPolynomials {
                    polynomials: &poly_group[..],
                    commitment: &commitments[0],
                    hint: hints.into_iter().next().unwrap(),
                }],
            )],
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("batched root-direct prove");

        assert!(
            proof.is_root_direct(),
            "expected a root-direct batched proof at num_vars={NUM_VARS}, layout_num_claims={NUM_POLYS}"
        );
        let direct_witnesses = proof
            .root
            .as_direct()
            .expect("root-direct variant must expose per-claim direct witnesses");
        assert_eq!(direct_witnesses.len(), NUM_POLYS);
        assert!(
            proof.steps.is_empty(),
            "root-direct batched proof must not carry recursive-suffix steps"
        );

        let mut bytes = Vec::new();
        let shape = proof.shape();
        assert!(matches!(shape, HachiBatchedProofShape::Direct { .. }));
        proof.serialize_uncompressed(&mut bytes).unwrap();
        let round_trip = HachiBatchedProof::<F>::deserialize_uncompressed(&*bytes, &shape).unwrap();
        assert_eq!(round_trip, proof);

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/batched-root-direct");
        let opening_groups = [&openings[..]];
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &round_trip,
            &verifier_setup,
            &mut verifier_transcript,
            vec![(
                &opening_point[..],
                vec![CommittedOpenings {
                    openings: opening_groups[0],
                    commitment: &commitments[0],
                }],
            )],
            BasisMode::Lagrange,
        )
        .expect("batched root-direct verify");
    }

    /// The verifier must reject a root-direct batched proof whose
    /// per-claim direct witnesses disagree with the claimed opening.
    #[test]
    fn batched_root_direct_rejects_wrong_opening() {
        const NUM_VARS: usize = 8;
        const NUM_POLYS: usize = 4;
        let len = 1usize << NUM_VARS;
        let polys: Vec<DensePoly<F, D>> = (0..NUM_POLYS)
            .map(|poly_idx| {
                let evals: Vec<F> = (0..len)
                    .map(|i| F::from_u64((i + poly_idx + 11) as u64))
                    .collect();
                DensePoly::<F, D>::from_field_evals(NUM_VARS, &evals).unwrap()
            })
            .collect();
        let poly_refs: Vec<&DensePoly<F, D>> = polys.iter().collect();

        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NUM_VARS, NUM_POLYS, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(&poly_refs, &setup).unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let opening_point: Vec<F> = (0..NUM_VARS).map(|i| F::from_u64((i + 2) as u64)).collect();
        let openings: Vec<F> = (0..NUM_POLYS).map(|_| F::from_u64(999_999)).collect();

        let poly_group = [&polys[0], &polys[1], &polys[2], &polys[3]];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"test/batched-root-direct-bad-opening");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            vec![(
                &opening_point[..],
                vec![CommittedPolynomials {
                    polynomials: &poly_group[..],
                    commitment: &commitments[0],
                    hint: hints.into_iter().next().unwrap(),
                }],
            )],
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("batched root-direct prove");
        assert!(proof.is_root_direct());

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"test/batched-root-direct-bad-opening");
        let opening_groups = [&openings[..]];
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            vec![(
                &opening_point[..],
                vec![CommittedOpenings {
                    openings: opening_groups[0],
                    commitment: &commitments[0],
                }],
            )],
            BasisMode::Lagrange,
        );
        assert!(result.is_err(), "verifier must reject bogus openings");
    }

    #[test]
    fn batched_verify_passes_for_consistent_openings() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;
        let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 5) as u64)).collect();
        let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 7 + 3) as u64)).collect();
        let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
        let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let poly_group = [&poly_a, &poly_b];
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(&poly_group, &setup).unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 9) as u64)).collect();
        let openings = [
            dense_opening(&evals_a, &opening_point),
            dense_opening(&evals_b, &opening_point),
        ];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/batched-prove");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            vec![(
                &opening_point[..],
                vec![CommittedPolynomials {
                    polynomials: &poly_group[..],
                    commitment: &commitments[0],
                    hint: hints.into_iter().next().unwrap(),
                }],
            )],
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut bytes = Vec::new();
        let shape = proof.shape();
        proof.serialize_uncompressed(&mut bytes).unwrap();
        let proof = HachiBatchedProof::<F>::deserialize_uncompressed(&*bytes, &shape).unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/batched-prove");
        let opening_groups = [&openings[..]];
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            vec![(
                &opening_point[..],
                vec![CommittedOpenings {
                    openings: opening_groups[0],
                    commitment: &commitments[0],
                }],
            )],
            BasisMode::Lagrange,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn batched_onehot_roundtrip_matches_public_shape_context() {
        const NV: usize = 15;
        const BATCH_SIZE: usize = 2;

        let layout = hachi_batched_root_layout::<OneHotCfg>(NV, BATCH_SIZE).expect("layout");
        let total_field = (layout.num_blocks * layout.block_len)
            .checked_mul(ONEHOT_D)
            .expect("total field size overflow");
        let total_chunks = total_field / BENCH_ONEHOT_K;
        assert_eq!(total_chunks * BENCH_ONEHOT_K, total_field);

        let polys: Vec<OneHotPoly<OneHotF, ONEHOT_D, u8>> = (0..BATCH_SIZE)
            .map(|poly_idx| {
                debug_make_onehot_poly(&layout, 0x0bee_fcaf_e000_1500 + poly_idx as u64)
            })
            .collect();
        let poly_refs: Vec<&OneHotPoly<OneHotF, ONEHOT_D, u8>> = polys.iter().collect();
        let point = debug_random_point(NV);
        let openings: Vec<OneHotF> = polys
            .iter()
            .map(|poly| debug_opening_from_poly(poly, &point, &layout))
            .collect();

        let setup =
            <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_prover(NV, BATCH_SIZE, 1);
        let verifier_setup =
            <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::commit(&poly_refs, &setup)
                .expect("batched onehot commit");
        let commitments = [commitment];
        let hints = vec![hint];

        let mut prover_transcript = Blake2bTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
        let proof = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::batched_prove(
            &setup,
            vec![(
                &point[..],
                vec![CommittedPolynomials {
                    polynomials: &poly_refs[..],
                    commitment: &commitments[0],
                    hint: hints.into_iter().next().unwrap(),
                }],
            )],
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("batched onehot prove");

        let expected_shape = expected_same_point_batched_shape(NV, BATCH_SIZE, &proof);
        let actual_shape = proof.shape();
        let (
            HachiBatchedProofShape::Fold {
                root_shape: expected_root,
                step_shapes: expected_steps,
            },
            HachiBatchedProofShape::Fold {
                root_shape: actual_root,
                step_shapes: actual_steps,
            },
        ) = (&expected_shape, &actual_shape)
        else {
            panic!("this test exercises a fold-rooted batched proof");
        };
        assert_eq!(expected_root.y_ring_coeffs, actual_root.y_ring_coeffs);
        assert_eq!(expected_root.v_coeffs, actual_root.v_coeffs);
        assert_eq!(expected_root.stage1_stages, actual_root.stage1_stages);
        assert_eq!(expected_root.stage2_sumcheck, actual_root.stage2_sumcheck);
        assert_eq!(
            expected_root.next_commit_coeffs,
            actual_root.next_commit_coeffs
        );
        assert_eq!(expected_steps, actual_steps);
        let mut bytes = Vec::new();
        proof.serialize_uncompressed(&mut bytes).unwrap();
        let decoded =
            HachiBatchedProof::<OneHotF>::deserialize_uncompressed(&*bytes, &expected_shape)
                .expect("deserialize batched proof with derived shape");
        assert_eq!(decoded, proof);

        let opening_groups = [&openings[..]];
        let mut verifier_transcript =
            Blake2bTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
        <OneHotScheme as CommitmentVerifier<OneHotF, ONEHOT_D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            vec![(
                &point[..],
                vec![CommittedOpenings {
                    openings: opening_groups[0],
                    commitment: &commitments[0],
                }],
            )],
            BasisMode::Lagrange,
        )
        .expect("batched onehot verify");
    }

    #[test]
    fn batched_verify_rejects_wrong_opening() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;
        let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 11) as u64)).collect();
        let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 5 + 13) as u64)).collect();
        let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
        let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let poly_group = [&poly_a, &poly_b];
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(&poly_group, &setup).unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 4) as u64)).collect();
        let mut openings = [
            dense_opening(&evals_a, &opening_point),
            dense_opening(&evals_b, &opening_point),
        ];
        openings[1] += F::one();

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/batched-prove/bad");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            vec![(
                &opening_point[..],
                vec![CommittedPolynomials {
                    polynomials: &poly_group[..],
                    commitment: &commitments[0],
                    hint: hints.into_iter().next().unwrap(),
                }],
            )],
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/batched-prove/bad");
        let opening_groups = [&openings[..]];
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            vec![(
                &opening_point[..],
                vec![CommittedOpenings {
                    openings: opening_groups[0],
                    commitment: &commitments[0],
                }],
            )],
            BasisMode::Lagrange,
        );

        assert!(matches!(result, Err(HachiError::InvalidProof)));
    }

    #[test]
    fn batched_verify_rejects_batch_count_beyond_setup_capacity() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;
        let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 17) as u64)).collect();
        let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 3 + 19) as u64)).collect();
        let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
        let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let poly_group = [&poly_a, &poly_b];
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(&poly_group, &setup).unwrap();
        let commitments = [commitment];
        let hints = vec![hint];

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 6) as u64)).collect();
        let openings = vec![
            dense_opening(&evals_a, &opening_point),
            dense_opening(&evals_b, &opening_point),
        ];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/batched-prove/oversized");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            vec![(
                &opening_point[..],
                vec![CommittedPolynomials {
                    polynomials: &poly_group[..],
                    commitment: &commitments[0],
                    hint: hints.into_iter().next().unwrap(),
                }],
            )],
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut oversized_proof = proof.clone();
        {
            let fold = oversized_proof
                .root
                .as_fold_mut()
                .expect("oversized-y-rings test expects a fold-rooted batched proof");
            let mut oversized_y_coeffs = fold.y_rings.coeffs().to_vec();
            oversized_y_coeffs.extend(vec![F::zero(); D]);
            fold.y_rings = FlatRingVec::from_coeffs(oversized_y_coeffs);
        }

        let mut oversized_openings = openings;
        oversized_openings.push(F::zero());
        let oversized_opening_groups = [&oversized_openings[..]];

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/batched-prove/oversized");
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &oversized_proof,
            &verifier_setup,
            &mut verifier_transcript,
            vec![(
                &opening_point[..],
                vec![CommittedOpenings {
                    openings: oversized_opening_groups[0],
                    commitment: &commitments[0],
                }],
            )],
            BasisMode::Lagrange,
        );

        assert!(matches!(result, Err(HachiError::InvalidProof)));
    }

    #[test]
    fn verify_passes_for_consistent_opening() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;

        let (poly, evals) = make_dense_poly(num_vars);

        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);

        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let lw = lagrange_weights(&opening_point);
        let opening: F = evals
            .iter()
            .zip(lw.iter())
            .fold(F::zero(), |a, (&c, &w)| a + c * w);

        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [opening];
        let opening_groups = [&openings[..]];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            vec![(
                &opening_point[..],
                vec![CommittedPolynomials {
                    polynomials: &poly_refs[..],
                    commitment: &commitments[0],
                    hint,
                }],
            )],
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            vec![(
                &opening_point[..],
                vec![CommittedOpenings {
                    openings: opening_groups[0],
                    commitment: &commitments[0],
                }],
            )],
            BasisMode::Lagrange,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn verify_rejects_wrong_opening() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;

        let (poly, evals) = make_dense_poly(num_vars);

        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);

        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let lw = lagrange_weights(&opening_point);
        let opening: F = evals
            .iter()
            .zip(lw.iter())
            .fold(F::zero(), |a, (&c, &w)| a + c * w);

        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let commitments = [commitment];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            vec![(
                &opening_point[..],
                vec![CommittedPolynomials {
                    polynomials: &poly_refs[..],
                    commitment: &commitments[0],
                    hint,
                }],
            )],
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        let wrong_opening = opening + F::one();
        let wrong_openings = [wrong_opening];
        let wrong_opening_groups = [&wrong_openings[..]];
        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/prove");
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            vec![(
                &opening_point[..],
                vec![CommittedOpenings {
                    openings: wrong_opening_groups[0],
                    commitment: &commitments[0],
                }],
            )],
            BasisMode::Lagrange,
        );

        assert!(
            result.is_err(),
            "verify must reject an incorrect opening value"
        );
    }

    #[test]
    fn verify_rejects_malformed_y_ring_dimension_without_panicking() {
        let (verifier_setup, commitment, mut proof, opening_point, opening, _layout) =
            make_verify_fixture(16);
        let root_fold = proof
            .root
            .as_fold_mut()
            .expect("expected a fold-rooted batched proof");
        let mut coeffs = root_fold.y_rings.coeffs().to_vec();
        let _ = coeffs.pop().expect("expected non-empty y_rings");
        root_fold.y_rings = FlatRingVec::from_coeffs(coeffs);

        let commitments = [commitment];
        let openings = [opening];
        let opening_groups = [&openings[..]];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/prove");
            <Scheme as CommitmentVerifier<F, D>>::batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                vec![(
                    &opening_point[..],
                    vec![CommittedOpenings {
                        openings: opening_groups[0],
                        commitment: &commitments[0],
                    }],
                )],
                BasisMode::Lagrange,
            )
        }));

        assert!(matches!(result, Ok(Err(HachiError::InvalidProof))));
    }

    #[test]
    fn monomial_basis_prove_verify_round_trip() {
        let alpha = D.trailing_zeros() as usize;
        let layout = Cfg::commitment_layout(16).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;

        let coeffs: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &coeffs).unwrap();

        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);

        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();

        let mw = monomial_weights(&opening_point);
        let opening: F = coeffs
            .iter()
            .zip(mw.iter())
            .fold(F::zero(), |acc, (&c, &w)| acc + c * w);

        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [opening];
        let opening_groups = [&openings[..]];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"test/monomial");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            vec![(
                &opening_point[..],
                vec![CommittedPolynomials {
                    polynomials: &poly_refs[..],
                    commitment: &commitments[0],
                    hint,
                }],
            )],
            &mut prover_transcript,
            BasisMode::Monomial,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"test/monomial");
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            vec![(
                &opening_point[..],
                vec![CommittedOpenings {
                    openings: opening_groups[0],
                    commitment: &commitments[0],
                }],
            )],
            BasisMode::Monomial,
        );

        assert!(
            result.is_ok(),
            "monomial-basis proof should verify: {result:?}"
        );
    }

    #[test]
    fn tiny_d32_root_direct_helpers_accept_valid_proof() {
        type DirectCfg = fp128::D32Full;
        type DirectF = fp128::Field;
        const DIRECT_D: usize = DirectCfg::D;
        type DirectScheme = HachiCommitmentScheme<DIRECT_D, DirectCfg>;

        let num_vars = 4usize;
        let evals: Vec<DirectF> = (0..(1usize << num_vars))
            .map(|i| DirectF::from_u64((i + 1) as u64))
            .collect();
        let poly = DensePoly::<DirectF, DIRECT_D>::from_field_evals(num_vars, &evals).unwrap();
        let opening_point = vec![DirectF::zero(); num_vars];
        let opening = evals[0];

        let setup =
            <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::setup_prover(num_vars, 1, 1);
        let verifier_setup =
            <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::setup_verifier(&setup);
        let (commitment, hint) = <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::commit(
            std::slice::from_ref(&poly),
            &setup,
        )
        .unwrap();

        let poly_refs: [&DensePoly<DirectF, DIRECT_D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [opening];
        let opening_groups = [&openings[..]];

        let mut prover_transcript = Blake2bTranscript::<DirectF>::new(b"test/tiny-direct");
        let proof = <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::batched_prove(
            &setup,
            vec![(
                &opening_point[..],
                vec![CommittedPolynomials {
                    polynomials: &poly_refs[..],
                    commitment: &commitments[0],
                    hint,
                }],
            )],
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

        assert!(proof.is_root_direct());
        assert_eq!(proof.num_fold_levels(), 0);
        let witnesses = proof
            .root
            .as_direct()
            .expect("root-direct batched proof expected");
        assert_eq!(witnesses.len(), 1);
        assert!(direct_witness_opening_matches::<DirectF>(
            &witnesses[0],
            &opening_point,
            &opening,
            BasisMode::Lagrange,
        )
        .unwrap());

        let mut verifier_transcript = Blake2bTranscript::<DirectF>::new(b"test/tiny-direct");
        <DirectScheme as CommitmentVerifier<DirectF, DIRECT_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            vec![(
                &opening_point[..],
                vec![CommittedOpenings {
                    openings: opening_groups[0],
                    commitment: &commitments[0],
                }],
            )],
            BasisMode::Lagrange,
        )
        .unwrap();
    }
}
