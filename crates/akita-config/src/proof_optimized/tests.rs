use super::*;

#[cfg(feature = "schedules-default")]
use crate::proof_optimized::fp128;
#[cfg(feature = "schedules-default")]
use crate::CommitmentConfig;
#[cfg(feature = "schedules-default")]
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

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
