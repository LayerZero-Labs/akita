//! Bounded committed-source guards.
//!
//! `fp128::DenseBounded` declares the same field, SIS profile, and balanced
//! signed-digit source class as `fp128::Dense`; only its committed-source bound
//! differs (`log_commit_bound = 65` instead of `128`). These tests pin what that
//! single parameter is allowed to change and what it must not.
//!
//! The bound is a **signed** bit width: `k` denotes `[-2^(k-1), 2^(k-1) - 1]`.
//! `65` is therefore the smallest declaration containing every `u64`, which is
//! the workload this preset exists for.

#![allow(missing_docs)]
// Both catalogs must be linked: every test here compares the bounded family
// against its full-width sibling.
#![cfg(all(
    feature = "schedules-fp128-dense-bounded",
    feature = "schedules-fp128-dense"
))]

use akita_config::proof_optimized::fp128;
use akita_config::{policy_of, CommitmentConfig};
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

/// Root-level quantities the committed-source bound feeds.
#[derive(Debug, PartialEq, Eq)]
struct RootShape {
    inner_basis: u32,
    inner_digits: usize,
    a_input_raw: usize,
    setup_field_elements: usize,
    next_witness: usize,
}

fn root_shape<Cfg: CommitmentConfig>(num_vars: usize) -> RootShape {
    let schedule = Cfg::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(num_vars),
    ))
    .expect("generated singleton schedule")
    .into_schedule();
    let root = &schedule.root.params.final_group.commitment;
    RootShape {
        inner_basis: root.log_basis_inner,
        inner_digits: root.num_digits_inner,
        a_input_raw: root.inner_commit_matrix.raw_input_dimension().unwrap(),
        setup_field_elements: akita_types::setup_matrix_field_elements_for_schedule(&schedule)
            .expect("setup field elements"),
        next_witness: schedule.root.output_witness_len,
    }
}

#[test]
fn bound_is_the_only_declared_difference_from_full_width_dense() {
    let bounded = fp128::DenseBounded::decomposition();
    let full = fp128::Dense::decomposition();

    assert_eq!(bounded.log_basis, full.log_basis);
    assert_eq!(bounded.field_bits(), full.field_bits());
    assert_eq!(
        bounded.log_commit_bound,
        fp128::DenseBounded::LOG_COMMIT_BOUND,
        "the preset constant and the macro argument must not drift apart"
    );
    assert_eq!(full.log_commit_bound, full.field_bits());

    // The declaration must contain the workload the preset is for. `65` is a
    // signed bit width, so it spans `[-2^64, 2^64 - 1]` and `u64::MAX` sits on
    // the positive endpoint; `64` would have covered only half of that.
    const { assert!(fp128::DenseBounded::ACCEPTS_UNSIGNED_64_BIT) };
    assert_eq!(
        akita_types::sis::declared_committed_source_bounds(bounded),
        (Some(1u128 << 64), Some(u128::from(u64::MAX))),
    );

    // Opening witnesses stay full-width: `t̂` / `ŵ` carry genuine field elements
    // regardless of how small the committed source is.
    assert_eq!(bounded.log_open_bound, Some(128));
    assert!(bounded.has_bounded_committed_source());
    assert!(!full.has_bounded_committed_source());
    bounded.validate().expect("bounded decomposition is valid");

    assert_eq!(
        fp128::DenseBounded::sis_modulus_profile(),
        fp128::Dense::sis_modulus_profile()
    );
    assert_eq!(fp128::DenseBounded::D, fp128::Dense::D);
    assert_eq!(
        fp128::DenseBounded::inner_basis_range(),
        fp128::Dense::inner_basis_range()
    );
    // Both are the balanced signed-digit source class: the bound sizes the digit
    // depth, it does not select a different honest-fold sizing rule.
    assert!(matches!(
        fp128::DenseBounded::root_honest_fold_policy(),
        akita_types::sis::HonestFoldPolicySpec::BalancedSignedDigit(_)
    ));
}

#[test]
fn a_distinct_bound_is_a_distinct_catalog_identity() {
    let bounded = akita_schedules::policy_digest(&policy_of::<fp128::DenseBounded>());
    let full = akita_schedules::policy_digest(&policy_of::<fp128::Dense>());
    assert_ne!(
        bounded, full,
        "the committed-source bound must separate two otherwise identical policies"
    );

    let bounded_catalog = fp128::DenseBounded::schedule_catalog().expect("bounded catalog");
    let full_catalog = fp128::Dense::schedule_catalog().expect("full-width catalog");
    assert_ne!(
        akita_schedules::identity_digest(&bounded_catalog.identity),
        akita_schedules::identity_digest(&full_catalog.identity),
    );
    assert_eq!(
        bounded_catalog.identity.decomposition.log_commit_bound,
        fp128::DenseBounded::LOG_COMMIT_BOUND,
        "the shipped catalog must carry the bound it was generated for"
    );

    // A bounded row cannot be resolved through the full-width config, because
    // the catalog identity is validated against the requesting policy.
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(24));
    assert!(akita_schedules::resolve_generated_catalog_row_for_key(
        &key,
        &policy_of::<fp128::Dense>(),
        fp128::Dense::ring_challenge_config,
        Some(bounded_catalog),
    )
    .is_err());
}

/// The bound must actually pay off where the objective can spend it.
///
/// The selection objective is setup-first
/// (`MinSetupMatrixFieldElementsThenProofPayload`), so the bound's return is a
/// smaller shared setup matrix and a smaller level-1 witness — not a smaller
/// proof. Asserted as strict inequalities plus exact snapshots so a regression in
/// either direction is visible.
#[test]
fn the_bound_shrinks_the_digit_depth_setup_and_next_witness() {
    for num_vars in [24usize, 26] {
        let bounded = root_shape::<fp128::DenseBounded>(num_vars);
        let full = root_shape::<fp128::Dense>(num_vars);

        assert!(
            bounded.inner_digits < full.inner_digits,
            "nv={num_vars}: bounded digit depth {} must be below full-width {}",
            bounded.inner_digits,
            full.inner_digits
        );
        assert!(
            bounded.setup_field_elements < full.setup_field_elements,
            "nv={num_vars}: bounded setup {} must be below full-width {}",
            bounded.setup_field_elements,
            full.setup_field_elements
        );
        assert!(
            bounded.next_witness < full.next_witness,
            "nv={num_vars}: bounded level-1 witness {} must be below full-width {}",
            bounded.next_witness,
            full.next_witness
        );
    }
}

/// `num_digits_inner` is the canonical depth for the declared bound.
///
/// A generated row stores its own root digit depth and expansion replays it
/// verbatim, so this is the tie between the declared bound and the digits the
/// commitment actually holds.
#[test]
fn generated_root_digit_depth_matches_the_declared_bound() {
    for family_decomposition in [
        fp128::DenseBounded::decomposition(),
        fp128::Dense::decomposition(),
    ] {
        let catalog = if family_decomposition.has_bounded_committed_source() {
            fp128::DenseBounded::schedule_catalog().expect("bounded catalog")
        } else {
            fp128::Dense::schedule_catalog().expect("full-width catalog")
        };
        for entry in catalog.entries {
            let expected = akita_types::sis::num_digits_inner_for_bound(
                akita_types::DecompositionParams {
                    log_basis: entry
                        .root
                        .final_group
                        .commitment
                        .inner_commit_matrix
                        .log_basis,
                    ..family_decomposition
                },
                family_decomposition.log_commit_bound,
            );
            assert_eq!(
                entry.root.final_group.num_digits_inner as usize, expected,
                "row {:?} stores a non-canonical root digit depth",
                entry.root.final_group.layout
            );
        }
    }
}
