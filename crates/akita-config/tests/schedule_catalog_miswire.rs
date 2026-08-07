//! Negative guard: a preset must reject a catalog wired to a different family.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::{policy_of, CommitmentConfig};
use akita_schedules::resolve_schedule;
use akita_types::PolynomialGroupLayout;

#[test]
fn miswired_catalog_rejects_before_lookup() {
    let wrong_catalog = akita_schedules::fp128_onehot_table();
    let key = PolynomialGroupLayout::new(28, 1);

    let err = resolve_schedule(
        key,
        &policy_of::<fp128::Dense>(),
        fp128::Dense::ring_challenge_config,
        Some(wrong_catalog),
    )
    .expect_err("adaptive dense preset must reject adaptive one-hot catalog");

    assert!(
        matches!(err, akita_field::AkitaError::InvalidSetup(_)),
        "expected InvalidSetup, got {err:?}"
    );
}
