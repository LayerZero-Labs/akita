//! Negative guard: a preset must reject a catalog wired to a different family.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::{policy_of, CommitmentConfig};
use akita_planner::resolve_schedule;
use akita_types::PolynomialGroupLayout;

#[test]
fn miswired_catalog_rejects_before_lookup() {
    let wrong_catalog = akita_schedules::fp128_d64_onehot_table();
    let key = PolynomialGroupLayout::new(28, 1);

    let err = resolve_schedule(
        key,
        &policy_of::<fp128::D64Full>(),
        fp128::D64Full::ring_challenge_config,
        fp128::D64Full::fold_challenge_shape_at_level,
        Some(wrong_catalog),
    )
    .expect_err("D64 full preset must reject D64 one-hot catalog");

    assert!(
        matches!(err, akita_error::AkitaError::InvalidSetup(_)),
        "expected InvalidSetup, got {err:?}"
    );
}
