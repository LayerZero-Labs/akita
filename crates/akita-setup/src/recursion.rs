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
    setup_prefix_level_params, setup_prefix_slot_id, setup_seed_digest, LevelParams, MRowLayout,
    OpeningBatchShape, SetupPrefixSlotAny, SETUP_OFFLOAD_D_SETUP,
};

fn commit_setup_prefix_for_level<F, B>(
    setup: &mut AkitaProverSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup,
    commitment_params: &LevelParams,
    natural_len: usize,
    n_prefix: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    B: CommitmentComputeBackend<F>,
{
    if setup.gen_ring_dim() != SETUP_OFFLOAD_D_SETUP {
        return Err(AkitaError::InvalidSetup(format!(
            "setup prefix preprocessing requires D={SETUP_OFFLOAD_D_SETUP}, got D={}",
            setup.gen_ring_dim()
        )));
    }
    let seed_digest = setup_seed_digest(setup.expanded.seed())
        .map_err(|err| AkitaError::InvalidSetup(format!("setup seed digest failed: {err}")))?;
    let Some(prefix_params) =
        setup_prefix_level_params(commitment_params, n_prefix, SETUP_OFFLOAD_D_SETUP)?
    else {
        return Ok(());
    };
    let level_params_digest = digest_level_params(std::slice::from_ref(&prefix_params));
    let id = setup_prefix_slot_id(
        seed_digest,
        SETUP_OFFLOAD_D_SETUP,
        natural_len,
        n_prefix,
        level_params_digest,
    );
    if setup.prefix_slots.get(&id).is_some() {
        return Ok(());
    }
    let slot = commit_setup_prefix::<F, { SETUP_OFFLOAD_D_SETUP }, B>(
        &setup.expanded,
        backend,
        prepared,
        &prefix_params,
        seed_digest,
        n_prefix,
        natural_len,
    )?;
    setup.prefix_slots.insert(SetupPrefixSlotAny::D64(slot))?;
    Ok(())
}

fn populate_recursive_setup_prefixes<F, Cfg>(
    setup: &mut AkitaProverSetup<F>,
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    Cfg: CommitmentConfig<Field = F>,
{
    if Cfg::D != SETUP_OFFLOAD_D_SETUP {
        return Ok(());
    }
    let ring_d = setup.gen_ring_dim();
    if ring_d != SETUP_OFFLOAD_D_SETUP {
        return Err(AkitaError::InvalidSetup(format!(
            "recursive setup-prefix population requires D={SETUP_OFFLOAD_D_SETUP}, got D={ring_d}"
        )));
    }

    let root_opening_batch = OpeningBatchShape::new(max_num_vars, max_num_batched_polys)?;
    let schedule = Cfg::get_params_for_prove(&root_opening_batch)?;
    let recursive_opening_batch = OpeningBatchShape::new(0, 1)?;
    let available_field_len = setup
        .expanded
        .shared_matrix()
        .total_ring_elements_at_dyn(ring_d)?
        .checked_mul(ring_d)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("setup matrix field length overflow".to_string())
        })?;

    let folds: Vec<_> = schedule.fold_steps().collect();
    let terminal_fold_idx = folds.len().saturating_sub(1);

    let backend = CpuBackend;
    let prepared = backend.prepare_setup(setup)?;
    for (idx, fold) in folds.iter().enumerate() {
        if idx >= terminal_fold_idx {
            continue;
        }
        let opening_batch = if idx == 0 {
            &root_opening_batch
        } else {
            &recursive_opening_batch
        };
        let exec = schedule.get_execution_schedule(idx)?;
        let m_row_layout = if exec.is_terminal {
            MRowLayout::WithoutDBlock
        } else {
            MRowLayout::WithDBlock
        };
        let depth_fold = fold
            .params
            .num_digits_fold(opening_batch.num_polynomials(), F::modulus_bits())?;
        let next_fold = &folds[idx + 1];
        let natural_len = active_setup_field_len(
            &fold.params,
            opening_batch,
            m_row_layout,
            depth_fold,
            ring_d,
        )?
        .min(available_field_len);
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
pub fn new_prover_setup_recursion<F, Cfg>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<AkitaProverSetup<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + Valid,
    Cfg: CommitmentConfig<Field = F>,
{
    let mut setup = new_prover_setup::<F, Cfg>(max_num_vars, max_num_batched_polys)?;
    populate_recursive_setup_prefixes::<F, Cfg>(&mut setup, max_num_vars, max_num_batched_polys)?;

    #[cfg(feature = "disk-persistence")]
    save_prover_setup::<F, Cfg>(&setup, max_num_vars, max_num_batched_polys)?;

    Ok(setup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_config::proof_optimized::fp128;
    use akita_types::SETUP_OFFLOAD_D_SETUP;

    type F = fp128::Field;

    #[test]
    fn recursive_d64_setup_populates_prefix_slots() {
        let setup =
            new_prover_setup_recursion::<F, fp128::D64OneHot>(20, 1).expect("recursive D64 setup");

        assert!(
            !setup.prefix_slots.is_empty(),
            "D64 recursive setup should populate setup-prefix slots"
        );
        for (id, slot) in setup.prefix_slots.iter() {
            assert_eq!(id, slot.id());
            id.check().expect("slot id shape");
            assert_eq!(id.d_setup, SETUP_OFFLOAD_D_SETUP);
            assert_eq!(slot.natural_len(), id.natural_len);
            assert_eq!(slot.padded_len(), id.n_prefix);
            assert!(slot.natural_len() <= slot.padded_len());
            assert!(slot.padded_len().is_power_of_two());
        }

        let verifier_setup = setup.verifier_setup().expect("verifier setup");
        assert_eq!(verifier_setup.prefix_slots.len(), setup.prefix_slots.len());
    }

    #[test]
    fn recursive_d32_setup_skips_prefix_slots() {
        let setup =
            new_prover_setup_recursion::<F, fp128::D32OneHot>(20, 1).expect("recursive D32 setup");

        assert!(
            setup.prefix_slots.is_empty(),
            "D32 recursive setup should skip D64-gated prefix slots"
        );
    }
}
