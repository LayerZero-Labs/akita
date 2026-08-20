//! Feature-off guard: without schedule features, presets reject runtime resolution.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

#[test]
fn schedule_catalog_none_without_feature_rejects() {
    if cfg!(feature = "schedules-fp128-onehot") {
        return;
    }

    assert!(
        fp128::OneHot::schedule_catalog().is_none(),
        "schedule feature disabled: schedule_catalog must be None"
    );

    let key = PolynomialGroupLayout::new(32, 1);

    let err = fp128::OneHot::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(key))
        .expect_err("runtime schedule must reject without an enabled catalog");
    assert!(
        matches!(err, akita_error::AkitaError::UnsupportedSchedule(_)),
        "expected UnsupportedSchedule, got {err:?}"
    );
}
