//! Exact setup-prefix slot requirements for recursive setup planning.

use std::collections::BTreeSet;

use akita_error::AkitaError;
use akita_schedules::suffix_opening_layout;
use akita_types::{
    active_setup_field_len, padded_setup_prefix_len, AkitaScheduleLookupKey, FoldSchedule,
    SetupPrefixSlotId,
};

use crate::CommitmentConfig;

fn setup_prefix_slot_matches(
    slot: &SetupPrefixSlotId,
    natural_len: usize,
    n_prefix: usize,
) -> Result<(), AkitaError> {
    let slot_n_prefix = slot.n_prefix()?;
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
/// `recursive_group_batch_candidates_for_capacity`,
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

fn push_unique_schedule_key(
    keys: &mut Vec<AkitaScheduleLookupKey>,
    candidate: AkitaScheduleLookupKey,
) {
    if !keys.contains(&candidate) {
        keys.push(candidate);
    }
}

pub(crate) fn recursive_group_batch_candidates_for_capacity<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<Vec<AkitaScheduleLookupKey>, AkitaError> {
    if !Cfg::recursive_setup_planning()
        || Cfg::decomposition().log_commit_bound != 1
        || max_num_batched_polys == 0
    {
        return Ok(Vec::new());
    }

    let mut keys = Vec::new();
    if let Some(catalog) = Cfg::schedule_catalog() {
        for entry in catalog.entries {
            if entry.root.precommitted_groups.is_empty() {
                continue;
            }
            let candidate = AkitaScheduleLookupKey {
                final_group: entry.root.final_group.layout,
                precommitteds: entry
                    .root
                    .precommitted_groups
                    .iter()
                    .map(|group| group.descriptor)
                    .collect(),
            };
            if candidate.fits_setup_capacity(max_num_vars, max_num_batched_polys)? {
                push_unique_schedule_key(&mut keys, candidate);
            }
        }
    }

    keys.sort_by(akita_schedules::runtime_schedule_key_cmp);
    Ok(keys)
}

#[cfg(all(test, feature = "schedules-fp128-d64-onehot-recursive"))]
mod tests {
    use super::*;
    use crate::proof_optimized::fp128;
    use crate::{CommitmentConfig, PrecommittedCommitmentConfig, RecursiveCommitmentConfig};
    use akita_types::{
        AkitaScheduleLookupKey, CommittedGroupProfile, OpeningClaimsLayout, PolynomialGroupLayout,
    };

    type SetupCfg = RecursiveCommitmentConfig<fp128::D64OneHot>;

    fn profiling_recursive_key() -> AkitaScheduleLookupKey {
        let pre = PolynomialGroupLayout::new(16, 1);
        let singleton =
            OpeningClaimsLayout::new(pre.num_vars(), pre.num_polynomials()).expect("singleton");
        let pre_params =
            <PrecommittedCommitmentConfig<SetupCfg> as CommitmentConfig>::get_params_for_batched_commitment(
                &singleton,
            )
            .expect("precommit params");
        let precommitted = CommittedGroupProfile::from_params(pre, &pre_params);
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
        use akita_types::setup_prefix_slot_field_elements;

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

        let mut slot_field_elements = 1usize;
        for slot in &slots {
            let n_prefix = slot.n_prefix().expect("n_prefix");
            assert!(n_prefix >= slot.natural_len);
            let one_slot_field_elements =
                setup_prefix_slot_field_elements(slot).expect("size one slot");
            assert!(
                one_slot_field_elements >= slot.natural_len,
                "slot capacity must cover the natural public-matrix prefix"
            );
            let a_coeff_len = slot
                .commitment_params
                .layout
                .inner_commit_matrix
                .output_rank()
                * slot.commitment_params.inner_width()
                * slot
                    .commitment_params
                    .layout
                    .inner_commit_matrix
                    .ring_dimension();
            let b_coeff_len = slot
                .commitment_params
                .layout
                .outer_commit_matrix
                .output_rank()
                * slot.commitment_params.outer_width()
                * slot
                    .commitment_params
                    .layout
                    .outer_commit_matrix
                    .ring_dimension();
            assert!(one_slot_field_elements >= a_coeff_len);
            assert!(one_slot_field_elements >= b_coeff_len);
            slot_field_elements = slot_field_elements.max(one_slot_field_elements);
        }
        assert!(slot_field_elements > 1);
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
            assert_eq!(slot_id.d_setup(), 64);
            assert!(slot_id.natural_len > 0);
            assert!(slot_id.n_prefix().expect("n_prefix") >= slot_id.natural_len);
        }
        let unique: BTreeSet<_> = ids.iter().cloned().collect();
        assert_eq!(unique.len(), ids.len());
    }
}
