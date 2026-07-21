//! Exact setup-prefix slot requirements for recursive setup planning.

use std::collections::BTreeSet;

use akita_field::AkitaError;
use akita_planner::suffix_opening_layout;
use akita_types::{
    active_setup_field_len, padded_setup_prefix_len, FoldSchedule, SetupPrefixSlotId,
    SETUP_OFFLOAD_D_SETUP,
};

use crate::generated_families::recursive_group_batch_candidates_for_capacity;
use crate::CommitmentConfig;

fn setup_prefix_slot_matches(
    slot: &SetupPrefixSlotId,
    natural_len: usize,
    n_prefix: usize,
) -> Result<(), AkitaError> {
    let slot_n_prefix = slot.n_prefix()?;
    if slot.d_setup != SETUP_OFFLOAD_D_SETUP {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix slot must use the recursive offload dimension".to_string(),
        ));
    }
    if slot.natural_len != natural_len {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix slot natural_len does not match recomputed active setup footprint"
                .to_string(),
        ));
    }
    if slot_n_prefix != n_prefix {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix slot padded length does not match recomputed prefix domain".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn extract_setup_prefix_slot_ids_from_schedule(
    schedule: &FoldSchedule,
    root_layout: &akita_types::OpeningClaimsLayout,
) -> Result<Vec<SetupPrefixSlotId>, AkitaError> {
    schedule.validate_structure()?;

    let mut ids = BTreeSet::new();
    for producer_index in 0..=schedule.recursive_folds.len() {
        let successor_prefix = schedule
            .recursive_folds
            .get(producer_index)
            .and_then(|fold| fold.params.incoming_setup_prefix.as_ref());
        let Some(slot_id) = successor_prefix else {
            continue;
        };
        let (params, opening_layout) = if producer_index == 0 {
            (
                &schedule.root.params.final_group.commitment,
                root_layout.clone(),
            )
        } else {
            let producer = &schedule.recursive_folds[producer_index - 1];
            let incoming_len = producer
                .params
                .incoming_setup_prefix
                .as_ref()
                .map(|slot| slot.natural_len);
            (
                &producer.params.witness,
                suffix_opening_layout(producer.input_witness_len, incoming_len)?,
            )
        };
        let natural_len = active_setup_field_len(params, &opening_layout)?;
        let n_prefix = padded_setup_prefix_len(natural_len);
        setup_prefix_slot_matches(slot_id, natural_len, n_prefix)?;
        if !ids.insert(slot_id.clone()) {
            continue;
        }
    }

    Ok(ids.into_iter().collect())
}

/// Enumerate every exact setup-prefix slot required by selected recursive schedules.
///
/// Selected keys are the bounded catalog/profile set from
/// [`crate::generated_families::recursive_group_batch_candidates_for_capacity`],
/// not a dense capacity grid.
pub fn setup_prefix_slot_ids_for_capacity<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<Vec<SetupPrefixSlotId>, AkitaError> {
    if max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }

    let mut ids = BTreeSet::new();
    for key in
        recursive_group_batch_candidates_for_capacity::<Cfg>(max_num_vars, max_num_batched_polys)?
    {
        let Ok(schedule) = Cfg::runtime_schedule(key.clone()) else {
            continue;
        };
        let root_layout = key.opening_layout()?;
        for slot_id in extract_setup_prefix_slot_ids_from_schedule(&schedule, &root_layout)? {
            ids.insert(slot_id);
        }
    }
    Ok(ids.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generated_families::recursive_group_batch_candidates_for_capacity;
    use crate::proof_optimized::fp128;
    use crate::RecursiveCommitmentConfig;
    use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout, PrecommittedGroupDescriptor};

    type SetupCfg = RecursiveCommitmentConfig<fp128::D64OneHot>;

    fn profiling_recursive_key() -> AkitaScheduleLookupKey {
        let pre = PolynomialGroupLayout::new(16, 1);
        let pre_params =
            crate::conservative_commitment::conservative_commit_params::<SetupCfg>(&pre)
                .expect("precommit params");
        let precommitted = PrecommittedGroupDescriptor::from_params(pre, &pre_params);
        AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(32, 2),
            precommitteds: vec![precommitted, precommitted],
        }
    }

    #[test]
    fn capacity_candidates_include_profiling_recursive_key() {
        let profile = profiling_recursive_key();
        // Profiling key is final K=2 plus two singleton pres (total 4 polys).
        let candidates =
            recursive_group_batch_candidates_for_capacity::<SetupCfg>(32, 4).expect("candidates");
        assert!(
            candidates.iter().any(|key| {
                key.final_group == profile.final_group
                    && key.precommitteds.len() == profile.precommitteds.len()
                    && key
                        .precommitteds
                        .iter()
                        .zip(profile.precommitteds.iter())
                        .all(|(a, b)| a.group == b.group)
            }),
            "capacity selected-key set must include the profiling recursive key"
        );
        assert!(
            candidates.len() <= 4,
            "selected recursive capacity keys must stay bounded, got {}",
            candidates.len()
        );
    }

    #[test]
    fn selected_recursive_keys_yield_exact_prefix_slots() {
        use crate::matrix_envelope::inflate_envelope_for_setup_prefix_slot;
        use akita_types::SetupMatrixEnvelope;

        let slots = setup_prefix_slot_ids_for_capacity::<SetupCfg>(32, 4).expect("slots");
        assert!(
            !slots.is_empty(),
            "selected recursive keys must require prefix slots"
        );
        assert!(
            slots.len() <= 8,
            "selected recursive prefix slots must stay bounded, got {}",
            slots.len()
        );

        let mut slot_envelope = SetupMatrixEnvelope { max_setup_len: 1 };
        for slot in &slots {
            let n_prefix = slot.n_prefix().expect("n_prefix");
            assert!(n_prefix >= slot.natural_len);
            let mut one_slot_envelope = SetupMatrixEnvelope { max_setup_len: 1 };
            inflate_envelope_for_setup_prefix_slot(&mut one_slot_envelope, slot, slot.d_setup)
                .expect("inflate one slot");
            assert!(
                one_slot_envelope.max_setup_len >= n_prefix / slot.d_setup,
                "slot envelope must cover the padded prefix storage"
            );
            let a_coeff_len = slot.commitment_params.inner_commit_matrix.output_rank()
                * slot.commitment_params.inner_width()
                * slot.commitment_params.inner_commit_matrix.ring_dimension();
            let b_coeff_len = slot.commitment_params.outer_commit_matrix.output_rank()
                * slot.commitment_params.outer_width()
                * slot.commitment_params.outer_commit_matrix.ring_dimension();
            assert!(one_slot_envelope.max_setup_len >= a_coeff_len.div_ceil(slot.d_setup));
            assert!(one_slot_envelope.max_setup_len >= b_coeff_len.div_ceil(slot.d_setup));
            slot_envelope.max_setup_len = slot_envelope
                .max_setup_len
                .max(one_slot_envelope.max_setup_len);
        }
        assert!(slot_envelope.max_setup_len > 1);
    }

    #[test]
    fn recursive_requirements_match_successor_slot_identity() {
        let key = profiling_recursive_key();
        let schedule = SetupCfg::runtime_schedule(key.clone()).expect("recursive schedule");
        let ids = extract_setup_prefix_slot_ids_from_schedule(
            &schedule,
            &key.opening_layout().expect("layout"),
        )
        .expect("slot ids");
        assert!(!ids.is_empty());
        for slot_id in &ids {
            assert_eq!(slot_id.d_setup, SETUP_OFFLOAD_D_SETUP);
            assert!(slot_id.natural_len > 0);
            assert!(slot_id.n_prefix().expect("n_prefix") >= slot_id.natural_len);
        }
        let unique: BTreeSet<_> = ids.iter().cloned().collect();
        assert_eq!(unique.len(), ids.len());
    }
}
