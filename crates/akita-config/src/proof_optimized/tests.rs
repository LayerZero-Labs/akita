use super::*;

#[cfg(feature = "schedules-default")]
use crate::proof_optimized::{fp128, fp32};
#[cfg(feature = "schedules-default")]
use crate::CommitmentConfig;
#[cfg(feature = "schedules-default")]
use akita_planner::{generated::GeneratedScheduleTable, schedule_from_entry};
#[cfg(feature = "schedules-default")]
use akita_schedules::{fp32_d128_onehot_table, fp32_d256_onehot_table};
#[cfg(feature = "schedules-default")]
use akita_types::{ntt_cache_requires_i16_tail, AkitaScheduleLookupKey, PolynomialGroupLayout};

#[cfg(feature = "schedules-default")]
#[test]
fn setup_levels_are_exactly_root_and_recursive_folds() {
    let schedule = fp128::D64Full::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(30),
    ))
    .expect("generated fp128 schedule");
    let setup_levels = setup_level_params_from_schedule(&schedule);
    assert_eq!(setup_levels.len(), 1 + schedule.recursive_folds.len());
    assert_eq!(
        setup_levels[0].role_dims(),
        schedule.root.params.final_group.commitment.role_dims()
    );
}

#[cfg(feature = "schedules-default")]
#[test]
fn generated_schedule_has_explicit_terminal_inner_only_topology() {
    let schedule = fp128::D64OneHot::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(32),
    ))
    .expect("generated one-hot schedule");
    schedule.validate_structure().expect("typed topology");
    assert!(schedule.terminal.params.witness.inner_width() > 0);
    assert_eq!(
        schedule.terminal.input_witness_len,
        schedule
            .recursive_folds
            .last()
            .map_or(schedule.root.output_witness_len, |step| step
                .output_witness_len)
    );
}

#[cfg(feature = "schedules-default")]
#[test]
fn setup_envelope_includes_terminal_inner_matrix() {
    let schedule = fp128::D64Full::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(28),
    ))
    .expect("generated fp128 schedule");
    let envelope = setup_matrix_envelope_for_schedule(&schedule).expect("setup envelope");
    let terminal_a = schedule
        .terminal
        .params
        .witness
        .inner_commit_matrix
        .output_rank()
        * schedule.terminal.params.witness.inner_width();
    assert!(envelope.max_setup_len >= terminal_a);
}

#[cfg(feature = "schedules-default")]
fn assert_every_table_terminal_uses_i16_tail<Cfg: CommitmentConfig, const D: usize>(
    table: GeneratedScheduleTable,
) -> (usize, usize) {
    let policy = crate::policy_of::<Cfg>();
    let mut min_width = usize::MAX;
    let mut max_width = 0usize;
    for entry in table.entries {
        if !entry.root.precommitted_groups.is_empty() {
            continue;
        }
        let key = entry.root.final_group.layout;
        let schedule = schedule_from_entry(
            entry,
            &AkitaScheduleLookupKey::single(key),
            &policy,
            Cfg::ring_challenge_config,
            Cfg::fold_challenge_shape_at_level,
        )
        .expect("shipped entry should materialize");
        let terminal = &schedule.terminal.params.witness;
        assert_eq!(terminal.d_a(), D);
        let width = terminal.inner_width();
        min_width = min_width.min(width);
        max_width = max_width.max(width);
        assert!(
            ntt_cache_requires_i16_tail::<Cfg::Field, D>(width, 16)
                .expect("generated terminal i16 accumulation should fit"),
            "generated q32 terminal unexpectedly fits the base CRT profile for {} key={key:?}, D={D}, width={width}",
            std::any::type_name::<Cfg>(),
        );
    }
    assert_ne!(min_width, usize::MAX, "generated table should not be empty");
    (min_width, max_width)
}

#[test]
#[cfg(feature = "schedules-default")]
fn generated_q32_terminals_require_the_i16_tail() {
    assert_eq!(
        assert_every_table_terminal_uses_i16_tail::<fp32::D128OneHot, 128>(
            fp32_d128_onehot_table(),
        ),
        (128, 128),
    );
    assert_eq!(
        assert_every_table_terminal_uses_i16_tail::<fp32::D256OneHot, 256>(
            fp32_d256_onehot_table(),
        ),
        (64, 128),
    );
}
