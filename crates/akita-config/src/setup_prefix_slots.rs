//! Exact setup-prefix slot requirements for recursive setup planning.

use std::collections::BTreeSet;

use akita_field::AkitaError;
use akita_planner::suffix_opening_layout;
use akita_types::{
    active_setup_field_len, config::SetupContributionMode, padded_setup_prefix_len, Schedule,
    SetupPrefixSlotId, SETUP_OFFLOAD_D_SETUP,
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
    schedule: &Schedule,
    root_layout: &akita_types::OpeningClaimsLayout,
) -> Result<Vec<SetupPrefixSlotId>, AkitaError> {
    schedule.validate_structure()?;

    let mut ids = BTreeSet::new();
    let mut incoming_setup_prefix: Option<usize> = None;
    let mut is_first_fold = true;

    for (index, fold) in schedule.folds.iter().enumerate() {
        let opening_layout = if is_first_fold {
            is_first_fold = false;
            root_layout.clone()
        } else {
            suffix_opening_layout(fold.input_witness_len, incoming_setup_prefix)?
        };

        match fold.params.setup_contribution_mode {
            SetupContributionMode::Recursive => {
                let natural_len = active_setup_field_len(&fold.params, &opening_layout)?;
                let n_prefix = padded_setup_prefix_len(natural_len);

                let successor_fold = schedule.folds.get(index + 1).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "recursive fold must have a nonterminal successor".to_string(),
                    )
                })?;
                if successor_fold.params.setup_prefix.is_none() {
                    return Err(AkitaError::InvalidSetup(
                        "recursive fold successor must carry a setup prefix".to_string(),
                    ));
                }
                if !successor_fold.params.precommitted_groups.is_empty()
                    || successor_fold.params.precommitted_group_count() != 1
                {
                    return Err(AkitaError::InvalidSetup(
                        "recursive fold successor must carry only the setup prefix group"
                            .to_string(),
                    ));
                }

                let slot_id = successor_fold.params.setup_prefix.as_ref().ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "recursive fold successor is missing setup-prefix metadata".to_string(),
                    )
                })?;
                setup_prefix_slot_matches(slot_id, natural_len, n_prefix)?;
                ids.insert(slot_id.clone());
                incoming_setup_prefix = Some(natural_len);
            }
            SetupContributionMode::Direct => {
                if let Some(slot_id) = &fold.params.setup_prefix {
                    let expected = incoming_setup_prefix.ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "setup-prefix consumer fold without an incoming delegated prefix"
                                .to_string(),
                        )
                    })?;
                    let n_prefix = padded_setup_prefix_len(expected);
                    setup_prefix_slot_matches(slot_id, expected, n_prefix)?;
                    incoming_setup_prefix = None;
                } else {
                    incoming_setup_prefix = None;
                }
            }
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
    use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout, PrecommittedGroupParams};

    type SetupCfg = RecursiveCommitmentConfig<fp128::D64OneHot>;

    fn profiling_recursive_key() -> AkitaScheduleLookupKey {
        let pre = PolynomialGroupLayout::new(16, 1);
        let pre_params =
            crate::conservative_commitment::conservative_commit_params::<SetupCfg>(&pre)
                .expect("precommit params");
        let precommitted = PrecommittedGroupParams::from_params(pre, &pre_params);
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
            inflate_envelope_for_setup_prefix_slot(&mut one_slot_envelope, slot)
                .expect("inflate one slot");
            assert!(
                one_slot_envelope.max_setup_len >= n_prefix / slot.d_setup,
                "slot envelope must cover the padded prefix storage"
            );
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
