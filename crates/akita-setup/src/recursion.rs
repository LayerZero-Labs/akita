use crate::new_prover_setup;
#[cfg(feature = "disk-persistence")]
use crate::save_prover_setup;
use akita_config::CommitmentConfig;
use akita_field::unreduced::HasWide;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_prover::{
    commit_setup_prefix, AkitaProverSetup, CommitmentComputeBackend, ComputeBackendSetup,
    CpuBackend,
};
use akita_serialization::Valid;
use akita_types::{
    active_setup_field_len, digest_level_params, padded_setup_prefix_len,
    setup_prefix_level_params, setup_prefix_slot_id, setup_seed_digest, ClaimIncidenceSummary,
    LevelParams, SETUP_OFFLOAD_D_SETUP,
};

fn commit_setup_prefix_for_level<F, const D: usize, B>(
    setup: &mut AkitaProverSetup<F, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    commitment_params: &LevelParams,
    natural_len: usize,
    n_prefix: usize,
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
    let seed_digest = setup_seed_digest(setup.expanded.seed())
        .map_err(|err| AkitaError::InvalidSetup(format!("setup seed digest failed: {err}")))?;
    let Some(prefix_params) = setup_prefix_level_params(commitment_params, n_prefix, D)? else {
        return Ok(());
    };
    let level_params_digest = digest_level_params(std::slice::from_ref(&prefix_params));
    let id = setup_prefix_slot_id(seed_digest, D, n_prefix, level_params_digest);
    if setup.prefix_slots.get(&id).is_some() {
        return Ok(());
    }
    let slot = commit_setup_prefix::<F, D, B>(
        &setup.expanded,
        backend,
        prepared,
        &prefix_params,
        seed_digest,
        n_prefix,
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
    let root_incidence =
        ClaimIncidenceSummary::from_counts(max_num_vars, max_num_batched_polys, max_num_points)?;
    let schedule = Cfg::get_params_for_prove(&root_incidence)?;
    let recursive_incidence = ClaimIncidenceSummary::same_point(0, 1)?;
    let available_field_len = setup
        .expanded
        .shared_matrix()
        .total_ring_elements_at::<D>()?
        .checked_mul(D)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("setup matrix field length overflow".to_string())
        })?;

    let folds: Vec<_> = schedule.fold_steps().collect();
    let terminal_fold_idx = folds.len().saturating_sub(1);

    if D != SETUP_OFFLOAD_D_SETUP {
        return Ok(());
    }

    let backend = CpuBackend;
    let prepared = backend.prepare_setup(setup)?;
    for (idx, fold) in folds.iter().enumerate() {
        if idx >= terminal_fold_idx {
            continue;
        }
        let incidence = if idx == 0 {
            &root_incidence
        } else {
            &recursive_incidence
        };
        let next_fold = &folds[idx + 1];
        let natural_len =
            active_setup_field_len(&fold.params, incidence, D)?.min(available_field_len);
        let n_prefix = padded_setup_prefix_len(natural_len);
        if n_prefix > available_field_len {
            continue;
        }
        commit_setup_prefix_for_level(
            setup,
            &backend,
            &prepared,
            &next_fold.params,
            natural_len,
            n_prefix,
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
    save_prover_setup::<F, D, Cfg>(&setup, max_num_vars, max_num_batched_polys, max_num_points)?;

    Ok(setup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_config::proof_optimized::fp128;
    use akita_types::SETUP_OFFLOAD_D_SETUP;

    type F = fp128::Field;

    #[test]
    #[cfg(not(feature = "zk"))]
    fn recursive_d64_setup_populates_prefix_slots() {
        let setup = new_prover_setup_recursion::<F, 64, fp128::D64OneHot>(20, 1, 1)
            .expect("recursive D64 setup");

        assert!(
            !setup.prefix_slots.is_empty(),
            "D64 recursive setup should populate setup-prefix slots"
        );
        for (id, slot) in setup.prefix_slots.iter() {
            assert_eq!(id, &slot.id);
            id.check().expect("slot id shape");
            assert_eq!(id.d_setup, SETUP_OFFLOAD_D_SETUP);
            assert_eq!(slot.padded_len, id.n_prefix);
            assert!(slot.natural_len <= slot.padded_len);
            assert!(slot.padded_len.is_power_of_two());
        }

        let verifier_setup = setup.verifier_setup().expect("verifier setup");
        assert_eq!(verifier_setup.prefix_slots.len(), setup.prefix_slots.len());
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn recursive_d32_setup_skips_prefix_slots() {
        let setup = new_prover_setup_recursion::<F, 32, fp128::D32OneHot>(20, 1, 1)
            .expect("recursive D32 setup");

        assert!(
            setup.prefix_slots.is_empty(),
            "D32 recursive setup should skip D64-gated prefix slots"
        );
    }
}
