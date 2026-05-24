//! End-to-end Akita PCS scheme orchestration.

use akita_config::{CommitmentConfig, WCommitmentConfig};
use akita_field::fields::wide::HasWide;
use akita_field::fields::HasUnreducedOps;
#[allow(unused_imports)]
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FrobeniusExtField, FromPrimitiveInt, HalvingField,
    PseudoMersenneField, RandomSampling,
};
use akita_prover::dispatch_ring_dim_result;
use akita_prover::{
    batched_commit_with_policy, commit_with_policy, prove_batched_with_policy,
    prove_folded_batched_with_policy, prove_recursive_level_with_policy, AkitaPolyOps,
    AkitaProverSetup, CommitComputeBackend, CommitmentProver, ProveLevelOutput, ProverClaims,
    RecursiveProverState, RecursiveSuffixOutcome, RootTensorProjectionPoly,
};
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::Transcript;
use akita_types::LevelParams;
use akita_types::{
    root_tensor_projection_enabled, schedule_root_fold_step, scheduled_fold_execution,
    scheduled_next_level_params, AkitaBatchedProof, AkitaCommitmentHint, AkitaInstanceDescriptor,
    AlgebraSection, CallSection, ClaimIncidenceSummary, PlanSection, RingCommitment, Schedule,
    SetupSection, Step,
};
use akita_types::{validate_ring_subfield_role, BasisMode, RingSubfieldEncoding};
use akita_types::{AkitaExpandedSetup, AkitaVerifierSetup};
use akita_verifier::{
    verify_batched_with_policy, verify_root_direct_commitments_with_params, CommitmentVerifier,
    VerifierClaims,
};
use std::marker::PhantomData;
use std::time::Instant;

/// End-to-end PCS wrapper, generic over ring degree `D` and config `Cfg`.
#[derive(Clone, Copy, Debug, Default)]
pub struct AkitaCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

fn validate_field_roles_for_ring<F, const D: usize, Cfg>() -> Result<(), AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>,
{
    validate_ring_subfield_role::<F, Cfg::ClaimField, D>("claim field")?;
    validate_ring_subfield_role::<F, Cfg::ChallengeField, D>("challenge field")?;
    let relative_degree =
        <Cfg::ChallengeField as akita_field::ExtField<Cfg::ClaimField>>::EXT_DEGREE;
    let expected_challenge_degree = Cfg::CLAIM_EXT_DEGREE
        .checked_mul(relative_degree)
        .ok_or_else(|| AkitaError::InvalidSetup("field tower degree overflow".to_string()))?;
    if Cfg::CHAL_EXT_DEGREE != expected_challenge_degree {
        return Err(AkitaError::InvalidSetup(format!(
            "challenge field degree {} does not match claim degree {} times relative degree {}",
            Cfg::CHAL_EXT_DEGREE,
            Cfg::CLAIM_EXT_DEGREE,
            relative_degree
        )));
    }
    Ok(())
}

fn recursive_w_commit_layout_for_d<Cfg>(
    commit_d: usize,
    commit_params: &LevelParams,
    current_w_len: usize,
) -> Result<LevelParams, AkitaError>
where
    Cfg: CommitmentConfig,
{
    dispatch_ring_dim_result!(commit_d, |D_COMMIT| {
        akita_types::recursive_level_layout_from_params(
            commit_params,
            current_w_len,
            WCommitmentConfig::<{ D_COMMIT }, Cfg>::decomposition(),
        )
    })
}

fn should_transform_root_commitment<F, const D: usize, Cfg>(
    incidence: &ClaimIncidenceSummary,
) -> Result<bool, AkitaError>
where
    F: FieldCore,
    Cfg: CommitmentConfig<Field = F>,
{
    if !root_tensor_projection_enabled::<F, Cfg::ClaimField, Cfg::ChallengeField, D>(
        incidence.num_vars(),
    ) {
        return Ok(false);
    }
    let schedule = Cfg::get_params_for_prove(incidence)?;
    Ok(schedule_root_fold_step(&schedule).is_some())
}

fn bind_transcript_instance_descriptor<F, T, const D: usize, Cfg>(
    setup: &AkitaExpandedSetup<F>,
    incidence: &ClaimIncidenceSummary,
    schedule: &Schedule,
    basis: BasisMode,
    transcript: &mut T,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + AkitaSerialize,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>,
{
    let mut setup_levels = schedule
        .steps
        .iter()
        .filter_map(|step| match step {
            Step::Fold(fold) => Some(fold.params.clone()),
            Step::Direct(_) => None,
        })
        .collect::<Vec<_>>();
    if setup_levels.is_empty() {
        setup_levels.push(Cfg::get_params_for_batched_commitment(incidence)?);
    }

    let descriptor = AkitaInstanceDescriptor::new(
        AlgebraSection::for_fields::<F, Cfg::ClaimField, Cfg::ChallengeField, D>()?,
        SetupSection::from_artifact_digests(
            Cfg::decomposition(),
            Cfg::sis_modulus_family(),
            setup.descriptor_digests,
            &setup_levels,
        ),
        PlanSection::from_schedule(schedule),
        CallSection::from_incidence(incidence, basis)?,
    );
    let descriptor_bytes = descriptor
        .canonical_bytes()
        .map_err(|err| AkitaError::InvalidSetup(format!("descriptor serialization: {err}")))?;
    transcript.bind_instance_bytes(&descriptor_bytes);
    Ok(())
}

/// Dispatch a prove-level operation to the correct ring dimension.
///
/// Handles the fast-path (`level_d == D`) and the dynamic dispatch path.
/// `#[inline(never)]` isolates the monomorphized match arms in their own
/// stack frame.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_prove_level<F, T, B, const D: usize, Cfg>(
    level_d: usize,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    current_state: &RecursiveProverState<F, Cfg::ChallengeField>,
    transcript: &mut T,
    level: usize,
    level_params: &LevelParams,
    next_params: LevelParams,
) -> Result<ProveLevelOutput<F, Cfg::ChallengeField>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + PseudoMersenneField,
    T: Transcript<F>,
    B: CommitComputeBackend<F>,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + HasUnreducedOps
        + AkitaSerialize,
{
    let expanded = backend.expanded::<D>(prepared);
    if level_d == D {
        prove_recursive_level_with_policy::<F, Cfg::ChallengeField, T, B, D, _, _>(
            expanded,
            backend,
            prepared,
            transcript,
            current_state,
            level,
            level_params,
            next_params.log_basis,
            |params, current_w_len| {
                akita_types::recursive_level_layout_from_params(
                    params,
                    current_w_len,
                    Cfg::decomposition(),
                )
            },
            |w| {
                akita_prover::commit_next_w_with_policy::<F, Cfg::ChallengeField, B, _, _, D>(
                    &next_params,
                    backend,
                    prepared,
                    w,
                    |params, current_w_len| {
                        akita_types::recursive_level_layout_from_params(
                            params,
                            current_w_len,
                            WCommitmentConfig::<{ D }, Cfg>::decomposition(),
                        )
                    },
                    recursive_w_commit_layout_for_d::<Cfg>,
                )
            },
        )
    } else {
        let expanded = expanded.clone();
        dispatch_ring_dim_result!(level_d, |D_LEVEL| {
            let level_prepared = backend.prepare_expanded::<D_LEVEL>(expanded.clone())?;
            prove_recursive_level_with_policy::<F, Cfg::ChallengeField, T, B, { D_LEVEL }, _, _>(
                expanded.as_ref(),
                backend,
                &level_prepared,
                transcript,
                current_state,
                level,
                level_params,
                next_params.log_basis,
                |params, current_w_len| {
                    akita_types::recursive_level_layout_from_params(
                        params,
                        current_w_len,
                        Cfg::decomposition(),
                    )
                },
                |w| {
                    akita_prover::commit_next_w_with_policy::<
                        F,
                        Cfg::ChallengeField,
                        B,
                        _,
                        _,
                        { D_LEVEL },
                    >(
                        &next_params,
                        backend,
                        &level_prepared,
                        w,
                        |params, current_w_len| {
                            akita_types::recursive_level_layout_from_params(
                                params,
                                current_w_len,
                                WCommitmentConfig::<{ D_LEVEL }, Cfg>::decomposition(),
                            )
                        },
                        recursive_w_commit_layout_for_d::<Cfg>,
                    )
                },
            )
        })
    }
}

/// Drive the recursive fold levels (after the root) and resolve the terminal
/// `log_basis` for the packed-digit direct witness.
///
/// The selected planner schedule is authoritative: it determines the fold
/// count, per-level `LevelParams`, successor params, and terminal direct
/// witness basis.
#[allow(clippy::too_many_arguments)]
fn prove_recursive_suffix<F, T, B, const D: usize, Cfg>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    num_vars: usize,
    transcript: &mut T,
    initial_state: RecursiveProverState<F, Cfg::ChallengeField>,
    schedule: &Schedule,
) -> Result<RecursiveSuffixOutcome<F, Cfg::ChallengeField>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + PseudoMersenneField
        + Valid,
    T: Transcript<F>,
    B: CommitComputeBackend<F>,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + HasUnreducedOps
        + AkitaSerialize,
{
    akita_prover::prove_recursive_suffix_with_policy::<F, Cfg::ChallengeField, _, _>(
        num_vars,
        initial_state,
        schedule,
        |level, inputs, current_log_basis| {
            scheduled_fold_execution(
                schedule,
                level,
                inputs,
                current_log_basis,
                Cfg::level_params_with_log_basis,
            )
        },
        |request| match request {
            akita_prover::SuffixLevelRequest::Intermediate {
                level,
                current_state,
                level_params,
                next_params,
            } => dispatch_prove_level::<F, T, B, D, Cfg>(
                level_params.ring_dimension,
                backend,
                prepared,
                current_state,
                transcript,
                level,
                level_params,
                next_params,
            )
            .map(akita_prover::SuffixLevelOutput::Intermediate),
            akita_prover::SuffixLevelRequest::Terminal {
                level,
                current_state,
                level_params,
                final_log_basis,
            } => dispatch_prove_terminal_level::<F, T, B, D, Cfg>(
                level_params.ring_dimension,
                backend,
                prepared,
                current_state,
                transcript,
                level,
                level_params,
                final_log_basis,
            )
            .map(akita_prover::SuffixLevelOutput::Terminal),
        },
    )
}

/// Dispatch a terminal prove-level operation to the correct ring dimension.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_prove_terminal_level<F, T, B, const D: usize, Cfg>(
    level_d: usize,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    current_state: &RecursiveProverState<F, Cfg::ChallengeField>,
    transcript: &mut T,
    level: usize,
    level_params: &LevelParams,
    final_log_basis: u32,
) -> Result<akita_types::TerminalLevelProof<F, Cfg::ChallengeField>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + PseudoMersenneField,
    T: Transcript<F>,
    B: CommitComputeBackend<F>,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + HasUnreducedOps
        + AkitaSerialize,
{
    let expanded = backend.expanded::<D>(prepared);
    if level_d == D {
        akita_prover::prove_terminal_recursive_level_with_policy::<F, Cfg::ChallengeField, T, B, D, _>(
            expanded,
            backend,
            prepared,
            transcript,
            current_state,
            level,
            level_params,
            final_log_basis,
            |params, current_w_len| {
                akita_types::recursive_level_layout_from_params(
                    params,
                    current_w_len,
                    Cfg::decomposition(),
                )
            },
        )
    } else {
        let expanded = expanded.clone();
        dispatch_ring_dim_result!(level_d, |D_LEVEL| {
            let level_prepared = backend.prepare_expanded::<D_LEVEL>(expanded.clone())?;
            akita_prover::prove_terminal_recursive_level_with_policy::<
                F,
                Cfg::ChallengeField,
                T,
                B,
                { D_LEVEL },
                _,
            >(
                expanded.as_ref(),
                backend,
                &level_prepared,
                transcript,
                current_state,
                level,
                level_params,
                final_log_basis,
                |params, current_w_len| {
                    akita_types::recursive_level_layout_from_params(
                        params,
                        current_w_len,
                        Cfg::decomposition(),
                    )
                },
            )
        })
    }
}

impl<F, const D: usize, Cfg> CommitmentProver<F, D> for AkitaCommitmentScheme<D, Cfg>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HasUnreducedOps
        + HalvingField
        + FromPrimitiveInt
        + PseudoMersenneField
        + Valid
        + AkitaSerialize,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + HasUnreducedOps
        + AkitaSerialize,
{
    type ProverSetup = AkitaProverSetup<F, D>;
    type VerifierSetup = AkitaVerifierSetup<F>;
    type Commitment = RingCommitment<F, D>;
    type ClaimField = Cfg::ClaimField;
    type CommitHint = AkitaCommitmentHint<F, D>;
    type BatchedProof = AkitaBatchedProof<F, Cfg::ChallengeField>;

    fn setup_prover(
        max_num_vars: usize,
        max_num_polys_per_point: usize,
        max_num_points: usize,
    ) -> Self::ProverSetup {
        validate_field_roles_for_ring::<F, D, Cfg>().expect("invalid Akita field tower");
        akita_setup::new_prover_setup::<F, D, Cfg>(
            max_num_vars,
            max_num_polys_per_point,
            max_num_points,
        )
        .expect("commitment setup failed")
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        setup.verifier_setup()
    }

    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::commit")]
    fn commit<P, B>(
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        polys: &[P],
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError>
    where
        P: AkitaPolyOps<F, D>,
        B: CommitComputeBackend<F>,
    {
        if let Some(first) = polys.first() {
            let incidence = ClaimIncidenceSummary::same_point(first.num_vars(), polys.len())?;
            if should_transform_root_commitment::<F, D, Cfg>(&incidence)? {
                let transformed = polys
                    .iter()
                    .map(|poly| poly.tensor_packed_extension_root_poly::<Cfg::ChallengeField>())
                    .collect::<Result<Vec<RootTensorProjectionPoly<F, D>>, _>>()?;
                return commit_with_policy::<F, D, RootTensorProjectionPoly<F, D>, B, _>(
                    &transformed,
                    backend,
                    prepared,
                    Cfg::get_params_for_batched_commitment,
                );
            }
        }
        commit_with_policy::<F, D, P, B, _>(
            polys,
            backend,
            prepared,
            Cfg::get_params_for_batched_commitment,
        )
    }

    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_commit")]
    fn batched_commit<P, B>(
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        polys_per_point: &[&[P]],
    ) -> Result<Vec<(Self::Commitment, Self::CommitHint)>, AkitaError>
    where
        P: AkitaPolyOps<F, D>,
        B: CommitComputeBackend<F>,
    {
        let incidence = akita_prover::prepare_batched_commit_inputs::<F, D, P>(
            polys_per_point,
            backend.expanded::<D>(prepared),
        )?;
        if should_transform_root_commitment::<F, D, Cfg>(&incidence)? {
            let transformed: Vec<Vec<RootTensorProjectionPoly<F, D>>> = polys_per_point
                .iter()
                .map(|polys| {
                    polys
                        .iter()
                        .map(|poly| poly.tensor_packed_extension_root_poly::<Cfg::ChallengeField>())
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<_, _>>()?;
            let transformed_refs: Vec<&[RootTensorProjectionPoly<F, D>]> =
                transformed.iter().map(Vec::as_slice).collect();
            return batched_commit_with_policy::<F, D, RootTensorProjectionPoly<F, D>, B, _>(
                &transformed_refs,
                backend,
                prepared,
                Cfg::get_params_for_batched_commitment,
            );
        }
        batched_commit_with_policy::<F, D, P, B, _>(
            polys_per_point,
            backend,
            prepared,
            Cfg::get_params_for_batched_commitment,
        )
    }

    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_prove")]
    fn batched_prove<'a, T, P, B>(
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        claims: ProverClaims<'a, Self::ClaimField, P, Self::Commitment, Self::CommitHint>,
        transcript: &mut T,
        basis: BasisMode,
    ) -> Result<Self::BatchedProof, AkitaError>
    where
        T: Transcript<F>,
        P: AkitaPolyOps<F, D>,
        B: CommitComputeBackend<F>,
    {
        let t_prove_total = Instant::now();
        validate_field_roles_for_ring::<F, D, Cfg>()?;
        let expanded = backend.expanded::<D>(prepared);
        let proof = prove_batched_with_policy::<
            F,
            Cfg::ClaimField,
            Cfg::ChallengeField,
            T,
            P,
            D,
            _,
            _,
            _,
            _,
        >(
            expanded,
            claims,
            transcript,
            basis,
            |incidence_summary| Cfg::get_params_for_prove(incidence_summary),
            |schedule, next_inputs| {
                scheduled_next_level_params(
                    schedule,
                    1,
                    next_inputs,
                    Cfg::level_params_with_log_basis,
                )
            },
            |transcript, incidence_summary, schedule, basis| {
                bind_transcript_instance_descriptor::<F, T, D, Cfg>(
                    expanded,
                    incidence_summary,
                    schedule,
                    basis,
                    transcript,
                )
            },
            |prepared_claims, schedule, next_params, transcript, basis| {
                let num_vars = prepared_claims.incidence_summary.num_vars();
                prove_folded_batched_with_policy::<
                        F,
                        Cfg::ClaimField,
                        Cfg::ChallengeField,
                        T,
                        P,
                        B,
                        D,
                        _,
                        _,
                    >(
                        expanded,
                        backend,
                        prepared,
                        transcript,
                        prepared_claims,
                        &schedule,
                        basis,
                        &next_params,
                        |w| {
                            akita_prover::commit_next_w_with_policy::<
                                F,
                                Cfg::ChallengeField,
                                B,
                                _,
                                _,
                                D,
                            >(
                                &next_params,
                                backend,
                                prepared,
                                w,
                                |params, current_w_len| {
                                    akita_types::recursive_level_layout_from_params(
                                        params,
                                        current_w_len,
                                        Cfg::decomposition(),
                                    )
                                },
                                recursive_w_commit_layout_for_d::<Cfg>,
                            )
                        },
                        |next_state, schedule, transcript| {
                            prove_recursive_suffix::<F, T, B, D, Cfg>(
                                backend, prepared, num_vars, transcript, next_state, schedule,
                            )
                        },
                    )
                    .map(|(proof, _total_levels)| proof)
            },
        )?;

        tracing::info!(
            levels = proof.num_fold_levels() + usize::from(proof.root.as_fold().is_some()),
            elapsed_s = t_prove_total.elapsed().as_secs_f64(),
            "akita batched prove complete"
        );

        Ok(proof)
    }
}

impl<F, const D: usize, Cfg> CommitmentVerifier<F, D> for AkitaCommitmentScheme<D, Cfg>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HasUnreducedOps
        + HalvingField
        + FromPrimitiveInt
        + PseudoMersenneField
        + Valid
        + AkitaSerialize,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField:
        RingSubfieldEncoding<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
{
    type VerifierSetup = AkitaVerifierSetup<F>;
    type Commitment = RingCommitment<F, D>;
    type ClaimField = Cfg::ClaimField;
    type BatchedProof = AkitaBatchedProof<F, Cfg::ChallengeField>;

    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_verify")]
    fn batched_verify<'a, T: Transcript<F>>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        claims: VerifierClaims<'a, Self::ClaimField, Self::Commitment>,
        basis: BasisMode,
    ) -> Result<(), AkitaError> {
        let t_verify_akita = Instant::now();
        validate_field_roles_for_ring::<F, D, Cfg>()?;
        verify_batched_with_policy::<F, Cfg::ClaimField, Cfg::ChallengeField, T, D, _, _, _, _, _>(
            proof,
            setup,
            transcript,
            claims,
            basis,
            |incidence_summary| Cfg::get_params_for_prove(incidence_summary),
            |schedule, next_inputs| {
                scheduled_next_level_params(
                    schedule,
                    1,
                    next_inputs,
                    Cfg::level_params_with_log_basis,
                )
            },
            Cfg::get_params_for_batched_commitment,
            |transcript, incidence_summary, schedule, basis| {
                bind_transcript_instance_descriptor::<F, T, D, Cfg>(
                    &setup.expanded,
                    incidence_summary,
                    schedule,
                    basis,
                    transcript,
                )
            },
            |witnesses,
             setup,
             commitments,
             incidence_summary,
             params,
             direct_commitment_payload| {
                verify_root_direct_commitments_with_params::<F, D>(
                    witnesses,
                    setup,
                    commitments,
                    incidence_summary,
                    params,
                    direct_commitment_payload,
                )
            },
        )?;

        tracing::info!(
            levels = proof.num_fold_levels() + 1,
            elapsed_s = t_verify_akita.elapsed().as_secs_f64(),
            "akita batched verify complete"
        );

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Akita"
    }
}

#[cfg(test)]
mod tests;
