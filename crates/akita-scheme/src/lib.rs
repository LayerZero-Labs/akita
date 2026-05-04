//! End-to-end Akita PCS scheme orchestration.

use akita_config::{CommitmentConfig, WCommitmentConfig};
use akita_field::fields::wide::HasWide;
use akita_field::fields::HasUnreducedOps;
#[allow(unused_imports)]
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, FieldSampling};
use akita_prover::kernels::crt_ntt::NttSlotCache;
use akita_prover::{
    batched_commit_with_policy, commit_with_policy, prove_batched_with_policy,
    prove_folded_batched_with_policy, prove_recursive_level_with_policy,
    verify_root_direct_commitments_with_params, AkitaPolyOps, AkitaProverSetup, CommitmentProver,
    MultiDNttCaches, ProveLevelOutput, ProverClaims, RecursiveProverState, RecursiveSuffixOutcome,
};
use akita_prover::{dispatch_ring_dim, dispatch_with_ntt};
use akita_serialization::Valid;
use akita_transcript::Transcript;
use akita_types::BasisMode;
use akita_types::LevelParams;
use akita_types::{
    scheduled_fold_execution, scheduled_next_level_params, AkitaBatchedProof, AkitaCommitmentHint,
    RingCommitment, Schedule,
};
use akita_types::{AkitaExpandedSetup, AkitaVerifierSetup};
use akita_verifier::{verify_batched_with_policy, CommitmentVerifier, VerifierClaims};
use std::marker::PhantomData;
use std::time::Instant;

/// End-to-end PCS wrapper, generic over ring degree `D` and config `Cfg`.
#[derive(Clone, Copy, Debug, Default)]
pub struct AkitaCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

fn recursive_w_commit_layout_for_d<Cfg>(
    commit_d: usize,
    commit_params: &LevelParams,
    current_w_len: usize,
) -> Result<LevelParams, AkitaError>
where
    Cfg: CommitmentConfig,
{
    dispatch_ring_dim!(commit_d, |D_COMMIT| {
        akita_types::recursive_level_layout_from_params(
            commit_params,
            current_w_len,
            WCommitmentConfig::<{ D_COMMIT }, Cfg>::decomposition(),
        )
    })
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
    expanded: &AkitaExpandedSetup<F>,
    setup_ntt_shared: &NttSlotCache<D>,
    commit_ntt_cache: &mut MultiDNttCaches,
    current_state: &RecursiveProverState<F>,
    transcript: &mut T,
    level: usize,
    level_params: &LevelParams,
    next_params: LevelParams,
) -> Result<ProveLevelOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
{
    if level_d == D {
        prove_recursive_level_with_policy::<F, T, D, _, _>(
            expanded,
            setup_ntt_shared,
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
                akita_prover::commit_next_w_with_policy::<F, _, _, D>(
                    &next_params,
                    setup_ntt_shared,
                    commit_ntt_cache,
                    expanded,
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
        dispatch_with_ntt!(level_d, ntt_cache, expanded, |D_LEVEL, ntt_shared| {
            prove_recursive_level_with_policy::<F, T, { D_LEVEL }, _, _>(
                expanded,
                ntt_shared,
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
                    akita_prover::commit_next_w_with_policy::<F, _, _, { D_LEVEL }>(
                        &next_params,
                        ntt_shared,
                        commit_ntt_cache,
                        expanded,
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
fn prove_recursive_suffix<F, T, const D: usize, Cfg>(
    setup: &AkitaProverSetup<F, D>,
    ntt_cache: &mut MultiDNttCaches,
    commit_ntt_cache: &mut MultiDNttCaches,
    max_num_vars: usize,
    transcript: &mut T,
    initial_state: RecursiveProverState<F>,
    schedule: &Schedule,
) -> Result<RecursiveSuffixOutcome<F>, AkitaError>
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
            scheduled_fold_execution(
                schedule,
                level,
                inputs,
                current_log_basis,
                Cfg::level_params_with_log_basis,
            )
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

impl<F, const D: usize, Cfg> CommitmentProver<F, D> for AkitaCommitmentScheme<D, Cfg>
where
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
    Cfg: CommitmentConfig<Field = F>,
{
    type ProverSetup = AkitaProverSetup<F, D>;
    type VerifierSetup = AkitaVerifierSetup<F>;
    type Commitment = RingCommitment<F, D>;
    type CommitHint = AkitaCommitmentHint<F, D>;
    type BatchedProof = AkitaBatchedProof<F>;

    fn setup_prover(
        max_num_vars: usize,
        max_num_polys_per_point: usize,
        max_num_points: usize,
    ) -> Self::ProverSetup {
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
    fn commit<P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>>(
        polys: &[P],
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError> {
        commit_with_policy::<F, D, P, _>(polys, setup, Cfg::get_params_for_commitment)
    }

    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_commit")]
    fn batched_commit<P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>>(
        poly_groups: &[&[P]],
        point_group_sizes: &[usize],
        setup: &Self::ProverSetup,
    ) -> Result<(Vec<Self::Commitment>, Vec<Self::CommitHint>), AkitaError> {
        batched_commit_with_policy::<F, D, P, _>(
            poly_groups,
            point_group_sizes,
            setup,
            Cfg::get_params_for_batched_commitment,
        )
    }

    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_prove")]
    fn batched_prove<'a, T: Transcript<F>, P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>>(
        setup: &Self::ProverSetup,
        claims: ProverClaims<'a, F, P, Self::Commitment, Self::CommitHint>,
        transcript: &mut T,
        basis: BasisMode,
    ) -> Result<Self::BatchedProof, AkitaError> {
        let t_prove_total = Instant::now();
        let proof = prove_batched_with_policy::<F, T, P, D, _, _, _>(
            &setup.expanded,
            claims,
            transcript,
            basis,
            Cfg::get_params_for_prove,
            |schedule, next_inputs| {
                scheduled_next_level_params(
                    schedule,
                    1,
                    next_inputs,
                    Cfg::level_params_with_log_basis,
                )
            },
            |prepared_claims, schedule, next_params, transcript, basis| {
                prove_folded_batched_with_policy::<F, T, P, D, _, _>(
                    &setup.expanded,
                    &setup.ntt_shared,
                    transcript,
                    prepared_claims,
                    &schedule,
                    basis,
                    &next_params,
                    |commit_ntt_cache, w| {
                        akita_prover::commit_next_w_with_policy::<F, _, _, D>(
                            &next_params,
                            &setup.ntt_shared,
                            commit_ntt_cache,
                            &setup.expanded,
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
                    |ntt_cache, commit_ntt_cache, next_state, schedule, transcript| {
                        prove_recursive_suffix::<F, T, D, Cfg>(
                            setup,
                            ntt_cache,
                            commit_ntt_cache,
                            setup.expanded.seed.max_num_vars,
                            transcript,
                            next_state,
                            schedule,
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
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
    Cfg: CommitmentConfig<Field = F>,
{
    type VerifierSetup = AkitaVerifierSetup<F>;
    type Commitment = RingCommitment<F, D>;
    type BatchedProof = AkitaBatchedProof<F>;

    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_verify")]
    fn batched_verify<'a, T: Transcript<F>>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        claims: VerifierClaims<'a, F, Self::Commitment>,
        basis: BasisMode,
    ) -> Result<(), AkitaError> {
        let t_verify_akita = Instant::now();
        verify_batched_with_policy::<F, T, D, _, _, _, _, _>(
            proof,
            setup,
            transcript,
            claims,
            basis,
            Cfg::get_params_for_prove,
            Cfg::root_level_params_for_layout_with_log_basis,
            |schedule, next_inputs| {
                scheduled_next_level_params(
                    schedule,
                    1,
                    next_inputs,
                    Cfg::level_params_with_log_basis,
                )
            },
            Cfg::get_params_for_commitment,
            |witnesses, setup, commitments, batch_shape, params| {
                verify_root_direct_commitments_with_params::<F, D>(
                    witnesses,
                    setup,
                    commitments,
                    batch_shape,
                    params,
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
