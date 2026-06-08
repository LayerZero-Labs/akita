use crate::new_prover_setup;
#[cfg(feature = "disk-persistence")]
use crate::persist_prover_setup;
use akita_config::CommitmentConfig;
use akita_field::unreduced::HasWide;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_prover::{
    commit_setup_prefix, AkitaProverSetup, CommitmentComputeBackend, ComputeBackendSetup,
    CpuBackend,
};
use akita_serialization::Valid;
use akita_types::{
    active_setup_field_len, digest_level_params, level_params_matches_setup_prefix,
    padded_setup_prefix_len, setup_prefix_slot_id, setup_seed_digest, ClaimIncidenceSummary,
    LevelParams, SETUP_OFFLOAD_D_SETUP, SETUP_OFFLOAD_N_MIN,
};

fn commit_setup_prefix_for_level<F, const D: usize, B>(
    setup: &mut AkitaProverSetup<F, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    level_params: &LevelParams,
    natural_len: usize,
    witness_prefix_len: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<F>,
{
    if D != SETUP_OFFLOAD_D_SETUP {
        return Err(AkitaError::InvalidSetup(format!(
            "setup prefix preprocessing requires D={SETUP_OFFLOAD_D_SETUP}, got D={D}"
        )));
    }
    if !level_params_matches_setup_prefix(level_params, witness_prefix_len, D) {
        return Ok(());
    }
    let seed_digest = setup_seed_digest(setup.expanded.seed())
        .map_err(|err| AkitaError::InvalidSetup(format!("setup seed digest failed: {err}")))?;
    if witness_prefix_len < padded_setup_prefix_len(natural_len) {
        return Ok(());
    }
    let available_field_len = setup
        .expanded
        .shared_matrix()
        .total_ring_elements()
        .checked_mul(D)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("setup matrix field length overflow".to_string())
        })?;
    if witness_prefix_len > available_field_len {
        return Ok(());
    }
    let level_params_digest = digest_level_params(std::slice::from_ref(level_params));
    let id = setup_prefix_slot_id(seed_digest, D, witness_prefix_len, level_params_digest);
    if setup.prefix_slots.get(&id).is_some() {
        return Ok(());
    }
    let slot = commit_setup_prefix::<F, D, B>(
        &setup.expanded,
        backend,
        prepared,
        level_params,
        seed_digest,
        witness_prefix_len,
        natural_len,
    )?;
    setup.prefix_slots.insert(slot)?;
    Ok(())
}

fn populate_recursive_setup_prefixes<F, const D: usize, Cfg>(
    setup: &mut AkitaProverSetup<F, D>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    Cfg: CommitmentConfig<Field = F>,
{
    if D != SETUP_OFFLOAD_D_SETUP {
        return Ok(());
    }

    let root_incidence =
        ClaimIncidenceSummary::from_counts(max_num_vars, max_num_batched_polys, max_num_points)?;
    let schedule = Cfg::get_params_for_prove(&root_incidence)?;
    let backend = CpuBackend;
    let prepared = backend.prepare_setup(setup)?;
    let recursive_incidence = ClaimIncidenceSummary::same_point(0, 1)?;

    let folds = schedule.fold_steps().collect::<Vec<_>>();
    let terminal_fold_idx = folds.len().saturating_sub(1);
    for (idx, fold) in folds.iter().enumerate() {
        if idx >= terminal_fold_idx {
            continue;
        }
        let incidence = if idx == 0 {
            &root_incidence
        } else {
            &recursive_incidence
        };
        let active_len = active_setup_field_len(&fold.params, incidence, D)?;
        let next_fold = folds[idx + 1];
        let minimum_prefix_len = padded_setup_prefix_len(active_len);
        let witness_prefix_len = next_fold
            .params
            .num_blocks
            .checked_mul(next_fold.params.block_len)
            .and_then(|n| n.checked_mul(D))
            .ok_or_else(|| AkitaError::InvalidSetup("setup prefix length overflow".to_string()))?;
        if witness_prefix_len < minimum_prefix_len {
            continue;
        }
        if witness_prefix_len < SETUP_OFFLOAD_N_MIN {
            continue;
        }
        if !witness_prefix_len.is_power_of_two() || !witness_prefix_len.is_multiple_of(D) {
            continue;
        }

        commit_setup_prefix_for_level(
            setup,
            &backend,
            &prepared,
            &next_fold.params,
            active_len,
            witness_prefix_len,
        )?;
    }

    tracing::info!(
        slots = setup.prefix_slots.len(),
        "populated setup-prefix commitments for recursive setup mode"
    );
    Ok(())
}

/// Construct prover setup and populate recursive setup-prefix commitments.
///
/// This first performs the ordinary setup load/generation path, then adds any
/// recursive setup-contribution prefix slots requested by the config policy.
///
/// # Errors
///
/// Returns an error if setup construction fails or recursive prefix population
/// cannot materialize a requested slot.
#[tracing::instrument(skip_all, name = "new_prover_setup_recursion")]
pub fn new_prover_setup_recursion<F, const D: usize, Cfg>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<AkitaProverSetup<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + Valid,
    Cfg: CommitmentConfig<Field = F>,
{
    let mut setup =
        new_prover_setup::<F, D, Cfg>(max_num_vars, max_num_batched_polys, max_num_points)?;
    populate_recursive_setup_prefixes::<F, D, Cfg>(
        &mut setup,
        max_num_vars,
        max_num_batched_polys,
        max_num_points,
    )?;

    #[cfg(feature = "disk-persistence")]
    persist_prover_setup::<F, D, Cfg>(&setup, max_num_vars, max_num_batched_polys, max_num_points)?;

    Ok(setup)
}
