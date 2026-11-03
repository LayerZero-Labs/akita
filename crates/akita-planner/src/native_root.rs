//! Ordinary suffix planning from an explicitly selected native-root transition.

use crate::schedule_params::{
    derive_selected_suffix_schedule, select_complete_candidate, QuerySearch, RelationModeFilter,
    RelationTraversalOrder, ScheduleMemo, SuffixCtx, SuffixState, SuffixTopology,
};
use crate::PlannerPolicy;
use akita_challenges::FoldChallengeFamily;
use akita_error::AkitaError;
use akita_params::{FoldParams, FoldSchedule, PolynomialGroupLayout, TerminalFoldParams};
pub use akita_schedules::native_root::NativeRootTransition;
use akita_schedules::ResolvedScheduleRow;

/// Plan the ordinary suffix from the native root's exact live span and typed A prefix.
///
/// The seed supplies only an already audited ordinary root commitment/profile
/// carrier. Its ordinary output and suffix are replaced. This is experimental
/// trusted admission, not a native-root security assertion or an artifact row.
/// No ordinary root search is used to approximate the native witness size.
pub fn find_native_root_schedule(
    seed: &ResolvedScheduleRow,
    transition: NativeRootTransition,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<FoldChallengeFamily, AkitaError>,
) -> Result<FoldSchedule, AkitaError> {
    akita_schedules::validate_policy(policy)?;
    if !policy.recursive_setup_planning || policy.chunks_at_level(1) != 1 {
        return Err(AkitaError::UnsupportedSchedule(
            "native root requires a single-chunk recursive typed A prefix".into(),
        ));
    }
    // Re-audit the seed against the chosen policy before retaining its profile.
    ResolvedScheduleRow::try_new(seed.profiles().clone(), seed.schedule().clone(), policy)?;
    let ctx = SuffixCtx {
        policy,
        challenge_order: policy.transcript_grinding_order()?,
        diagnostics: None,
        ring_challenge_config: &ring_challenge_config,
        key: PolynomialGroupLayout::singleton(0),
        setup_field_budget: policy.setup_field_budget,
        root_lookup_key: None,
        root_main_constraint: None,
        adaptation_guide: None,
        root_setup_mode: None,
        root_source_contract: None,
        precommitted_source_contracts: &[],
        level_zero_is_root: true,
        relation_traversal_order: RelationTraversalOrder::Canonical,
        relation_mode_filter: RelationModeFilter::All,
    };
    let state = SuffixState {
        level: 1,
        current_witness_len: transition.physical_cells(),
        input_chunks: None,
        current_lb: transition.log_digit_bound(),
        source_moment: if policy.selective_l2_response_model_enabled() {
            // Each base-eight balanced digit has magnitude at most four.
            // Start from its worst-case energy, rather than assuming a
            // distribution for the native root's emitted digits.
            let energy = (transition.physical_cells() as u128)
                .checked_mul(16)
                .ok_or_else(|| AkitaError::InvalidSetup("native source energy overflow".into()))?;
            Some(
                crate::response_model::SourceMomentEstimate::from_moments(energy, 16_000_000)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("native source moment overflow".into())
                    })?,
            )
        } else {
            None
        },
        dimension_ceiling: seed.schedule().root.params.role_dims(),
        topology: SuffixTopology::SetupPrefixed {
            natural_len: transition.setup_prefix_natural_cells(),
        },
    };
    let mut memo = ScheduleMemo::new();
    let suffix = derive_selected_suffix_schedule(&ctx, &mut memo, state, 0, QuerySearch::Root(0))?;
    let best =
        select_complete_candidate(policy, suffix.setup_candidates(), None)?.ok_or_else(|| {
            AkitaError::UnsupportedSchedule(
                "no ordinary suffix admits the exact native root transition".into(),
            )
        })?;
    let mut root = seed.schedule().root.clone();
    root.output_witness_len = transition.physical_cells();
    let schedule = FoldSchedule {
        root,
        recursive_folds: best
            .folds
            .to_vec()
            .into_iter()
            .map(|fold| FoldParams {
                params: (*fold.params).clone(),
                input_witness_len: fold.input_witness_len,
                output_witness_len: fold.output_witness_len,
            })
            .collect(),
        terminal: TerminalFoldParams {
            fold_challenge_config: best.terminal.sparse_challenge_config,
            response_shape: best.terminal.response_shape.clone(),
            input_witness_len: best.terminal.input_witness_len,
            ..best.terminal.params.clone()
        },
    };
    transition.resolve(seed.profiles().clone(), schedule.clone(), policy)?;
    Ok(schedule)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_config::{
        proof_optimized::fp128::Dense, CommitmentConfig, RecursiveCommitmentConfig,
    };
    use akita_params::{ChunkedWitnessCfg, CommittedGroupParams};
    use akita_schedules::ValidatedScheduleCatalog;

    type Config = RecursiveCommitmentConfig<Dense>;

    fn base_eight_prefix_catalog() -> akita_config::TrustedScheduleCatalog<Config> {
        akita_config::test_support::workspace_schedule_catalog::<Config>(
            ChunkedWitnessCfg::default_non_chunked(),
        )
        .unwrap()
    }

    fn base_eight_prefix_row(
        ordinary: &akita_config::TrustedScheduleCatalog<Config>,
    ) -> &ResolvedScheduleRow {
        ordinary
            .catalog()
            .rows()
            .find(|row| {
                row.schedule().recursive_folds.first().is_some_and(|step| {
                    step.params.setup_prefix().is_some()
                        && step.params.inner().digits.log_basis == 3
                })
            })
            .expect("checked-in recursive family has a base-eight prefix edge")
    }

    fn leb128(mut value: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let low = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                bytes.push(low);
                return bytes;
            }
            bytes.push(low | 0x80);
        }
    }

    fn replace_once(bytes: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
        let hits = (0..=bytes.len() - from.len())
            .filter(|&start| bytes[start..].starts_with(from))
            .collect::<Vec<_>>();
        assert_eq!(hits.len(), 1, "patched encoding must be unambiguous");
        [&bytes[..hits[0]], to, &bytes[hits[0] + from.len()..]].concat()
    }

    #[test]
    fn native_only_transition_requires_explicit_admission() {
        let ordinary = base_eight_prefix_catalog();
        let row = base_eight_prefix_row(&ordinary);
        let policy = ordinary.catalog().policy();
        let schedule = row.schedule().clone();
        let prefix = schedule.recursive_folds[0].params.setup_prefix().unwrap();
        // Only native admission accepts a root output that the ordinary root
        // does not plan, even when the first successor consumes it exactly.
        let mut native = schedule.clone();
        native.root.output_witness_len += 64;
        native.recursive_folds[0].input_witness_len += 64;
        let transition = NativeRootTransition::new(
            native.root.output_witness_len,
            3,
            prefix.setup_natural_len.unwrap(),
        )
        .unwrap();

        let rejection =
            ResolvedScheduleRow::try_new(row.profiles().clone(), native.clone(), policy)
                .unwrap_err()
                .to_string();
        let catalog_rejection = ValidatedScheduleCatalog::try_new(
            "native_admission_fixture",
            [(row.profiles().clone(), native.clone())],
            policy,
            Config::ring_challenge_config,
        )
        .unwrap_err();
        assert_eq!(catalog_rejection.to_string(), rejection);
        let admitted = ValidatedScheduleCatalog::try_new_native_root(
            "native_admission_fixture",
            [],
            [(row.profiles().clone(), native.clone(), transition)],
            policy,
            Config::ring_challenge_config,
        )
        .unwrap();
        let native_row = admitted.rows().next().unwrap();
        assert_eq!(native_row.schedule(), &native);
        assert!(admitted.to_artifact_bytes().is_err());
        assert!(admitted
            .to_verifier_view(&[native_row.selection()])
            .is_err());

        // Carry the native row in an otherwise canonical ordinary verifier
        // view. Postcard writes struct fields in order with LEB128 lengths, so
        // only the row identity, root tail, and first successor input change.
        let seed = ValidatedScheduleCatalog::try_new(
            "native_admission_fixture",
            [(row.profiles().clone(), schedule.clone())],
            policy,
            Config::ring_challenge_config,
        )
        .unwrap();
        let view = seed.to_verifier_view(&[row.selection()]).unwrap();
        ValidatedScheduleCatalog::from_verifier_view(
            &view,
            "native_admission_fixture",
            policy,
            Config::ring_challenge_config,
        )
        .unwrap();
        let folds = leb128(schedule.recursive_folds.len());
        let root_tail = |output| {
            [
                leb128(schedule.root.input_witness_len),
                leb128(output),
                folds.clone(),
            ]
            .concat()
        };
        let successor_head = |input| {
            [
                leb128(input),
                leb128(schedule.recursive_folds[0].output_witness_len),
            ]
            .concat()
        };
        let view = replace_once(
            &view,
            row.selection().row_digest.as_bytes(),
            native_row.selection().row_digest.as_bytes(),
        );
        let view = replace_once(
            &view,
            &root_tail(schedule.root.output_witness_len),
            &root_tail(native.root.output_witness_len),
        );
        let view = replace_once(
            &view,
            &successor_head(schedule.recursive_folds[0].input_witness_len),
            &successor_head(native.recursive_folds[0].input_witness_len),
        );
        // The view decodes to the native row and fails the same ordinary check.
        let view_rejection = ValidatedScheduleCatalog::from_verifier_view(
            &view,
            "native_admission_fixture",
            policy,
            Config::ring_challenge_config,
        )
        .unwrap_err();
        assert_eq!(view_rejection.to_string(), rejection);
    }

    #[test]
    fn native_admission_rejects_changed_transition_and_cannot_export_ordinary_artifacts() {
        let ordinary = base_eight_prefix_catalog();
        let row = base_eight_prefix_row(&ordinary);
        let schedule = row.schedule().clone();
        let policy = ordinary.catalog().policy();
        let prefix = schedule.recursive_folds[0].params.setup_prefix().unwrap();
        let transition = NativeRootTransition::new(
            schedule.root.output_witness_len,
            3,
            prefix.setup_natural_len.unwrap(),
        )
        .unwrap();
        let admitted = ValidatedScheduleCatalog::try_new_native_root(
            "native_admission_fixture",
            [],
            [(row.profiles().clone(), schedule.clone(), transition)],
            policy,
            Config::ring_challenge_config,
        )
        .unwrap();
        assert!(admitted.to_artifact_bytes().is_err());
        let selection = admitted.rows().next().unwrap().selection();
        assert!(admitted.to_verifier_view(&[selection]).is_err());

        let mut changed_output = schedule.clone();
        changed_output.root.output_witness_len += 64;
        let mut changed_input = schedule.clone();
        changed_input.recursive_folds[0].input_witness_len += 64;
        let mut changed_prefix = schedule.clone();
        let mut prefix = *prefix;
        prefix.setup_natural_len = Some(transition.setup_prefix_natural_cells() / 2);
        changed_prefix.recursive_folds[0]
            .params
            .set_setup_prefix(Some(prefix))
            .unwrap();
        let mut changed_digits = schedule.clone();
        let original = &schedule.recursive_folds[0].params;
        let mut groups = original.groups().to_vec();
        groups.last_mut().unwrap().profile.inner.digits.log_basis = 4;
        changed_digits.recursive_folds[0].params = CommittedGroupParams::try_new(
            groups,
            original.open_matrix,
            original.payload_mode,
            original.ring_relation_mode,
            original.source_encoding,
            original.witness_chunk,
        )
        .unwrap();
        for changed in [
            changed_output,
            changed_input,
            changed_prefix,
            changed_digits,
        ] {
            assert!(transition
                .resolve(row.profiles().clone(), changed, policy)
                .is_err());
        }
        assert!(NativeRootTransition::new(
            transition.physical_cells() + 64,
            3,
            transition.setup_prefix_natural_cells(),
        )
        .unwrap()
        .resolve(row.profiles().clone(), schedule, policy)
        .is_err());
    }
}
