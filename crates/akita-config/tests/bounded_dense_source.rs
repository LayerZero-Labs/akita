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
use akita_config::proof_optimized::fp128;
use akita_config::{policy_of, CommitmentConfig};
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

fn catalog<Cfg: CommitmentConfig>() -> akita_config::TrustedScheduleCatalog {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("artifacts/schedules")
        .join(format!("{}.aks", Cfg::schedule_family_name()));
    let bytes = std::fs::read(path).expect("checked-in schedule artifact");
    akita_config::trusted_schedule_catalog_from_bytes::<Cfg>(&bytes).expect("trusted catalog")
}

/// Root-level quantities the committed-source bound feeds.
#[derive(Debug, PartialEq, Eq)]
struct RootShape {
    inner_basis: u32,
    inner_digits: usize,
    next_witness: usize,
}

fn root_shape<Cfg: CommitmentConfig>(num_vars: usize) -> RootShape {
    let schedule = catalog::<Cfg>()
        .resolve_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(num_vars),
        ))
        .expect("generated singleton schedule")
        .schedule()
        .clone();
    let root = &schedule.root.params;
    RootShape {
        inner_basis: root.inner().digits.log_basis,
        inner_digits: root.inner().digits.num_digits,
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
    const { assert!(fp128::DenseBounded::MAX_CENTERED_MAGNITUDE >= u64::MAX as u128) };
    assert_eq!(
        fp128::DenseBounded::committed_source_contract()
            .expect("the preset declares a valid producer contract")
            .declared_bounds(),
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
    assert_eq!(
        fp128::DenseBounded::RING_DIMENSION_SCHEDULE_MODE,
        fp128::Dense::RING_DIMENSION_SCHEDULE_MODE,
        "the source bound must not change the A/B/D search domain"
    );
    assert_eq!(
        fp128::DenseBounded::inner_basis_range(),
        fp128::Dense::inner_basis_range()
    );
    // Both are the balanced signed-digit source class. The bound sizes the digit
    // depth. It does not select a different honest-fold sizing rule. `commit`
    // reads the declared class, and offline planning derives its policy from that
    // class.
    assert!(matches!(
        akita_config::honest_fold_policy_of::<fp128::DenseBounded>(),
        akita_types::sis::HonestFoldPolicySpec::BalancedSignedDigit(_)
    ));
    assert_eq!(
        fp128::DenseBounded::committed_source_class(),
        akita_types::sis::CommittedSourceClass::BalancedSignedDigit,
    );
    assert_eq!(
        fp128::OneHot::committed_source_class(),
        akita_types::sis::CommittedSourceClass::UnitOneHot {
            source_chunk_size: akita_types::sis::DEFAULT_UNIT_ONEHOT_SOURCE_CHUNK_SIZE,
        },
    );
}

#[test]
fn a_distinct_bound_is_a_distinct_catalog_identity() {
    let bounded = akita_schedules::policy_digest(&policy_of::<fp128::DenseBounded>());
    let full = akita_schedules::policy_digest(&policy_of::<fp128::Dense>());
    assert_ne!(
        bounded, full,
        "the committed-source bound must separate two otherwise identical policies"
    );
    assert_ne!(
        &bounded[8..],
        &[0u8; 24],
        "the serialized policy identity must carry the full cryptographic digest"
    );

    let bounded_catalog = catalog::<fp128::DenseBounded>();
    let full_catalog = catalog::<fp128::Dense>();
    assert_ne!(
        bounded_catalog.catalog_digest(),
        full_catalog.catalog_digest(),
    );
    assert_eq!(
        policy_of::<fp128::DenseBounded>()
            .decomposition
            .log_commit_bound,
        fp128::DenseBounded::LOG_COMMIT_BOUND,
        "the shipped catalog must carry the bound it was generated for"
    );

    // A bounded row cannot be resolved through the full-width config, because
    // the catalog identity is validated against the requesting policy.
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(24));
    let bounded_bytes = bounded_catalog
        .to_artifact_bytes()
        .expect("bounded artifact");
    assert!(
        akita_config::trusted_schedule_catalog_from_bytes::<fp128::Dense>(&bounded_bytes).is_err()
    );
    assert!(full_catalog.resolve_key(&key).is_ok());
}

/// The bound must actually reduce the source-dependent root digit depth.
///
/// Each catalog family independently minimizes its setup-first objective, so its
/// selected setup envelope need not be monotone in the source bound: a bounded
/// family may spend its lower digit cost on a different A/B/D shape, including a
/// different A basis. Compare digit depths at the bounded row's own basis rather
/// than comparing unlike raw digit counts. The smaller level-1 witness is a
/// separate consequence pinned by these catalog rows.
#[test]
fn the_bound_shrinks_same_basis_digit_depth_and_next_witness() {
    for num_vars in [24usize, 26] {
        let bounded = root_shape::<fp128::DenseBounded>(num_vars);
        let full = root_shape::<fp128::Dense>(num_vars);
        let full_width_digits_at_bounded_basis =
            akita_types::sis::compute_num_digits_field_width(128, bounded.inner_basis);

        assert!(
            bounded.inner_digits < full_width_digits_at_bounded_basis,
            "nv={num_vars}: bounded digit depth {} must be below same-basis full-width {}",
            bounded.inner_digits,
            full_width_digits_at_bounded_basis
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
            catalog::<fp128::DenseBounded>()
        } else {
            catalog::<fp128::Dense>()
        };
        for row in catalog.rows() {
            let root = &row.schedule().root.params;
            let expected = akita_types::sis::num_digits_inner_for_bound(
                akita_types::DecompositionParams {
                    log_basis: root.inner().digits.log_basis,
                    ..family_decomposition
                },
                family_decomposition.log_commit_bound,
            );
            assert_eq!(
                root.inner().digits.num_digits,
                expected,
                "row {:?} stores a non-canonical root digit depth",
                row.profiles().final_group.group
            );
        }
    }
}
