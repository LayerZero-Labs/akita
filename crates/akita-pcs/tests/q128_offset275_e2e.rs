//! End-to-end check of `SisModulusProfileId::Q128Offset275`.
//!
//! No preset in this repository selects that profile and no checked-in
//! schedule artifact names it. This file declares a configuration over
//! `jolt_field::Prime128Offset275` that reuses the `fp128::Dense` policy,
//! names the exact profile of its field, and plans its own schedule.

#![allow(missing_docs)]

use akita_config::{
    policy_of, proof_optimized::fp128, validate_config_policy, CommitmentConfig,
    TrustedScheduleCatalog, ValidatedScheduleCatalog,
};
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_params::{
    lagrange_weights, BasisMode, CommittedGroupBatchProfile, GroupCommitPhaseParams,
    InnerCommitMatrixParams, OuterCommitMatrixParams, PolynomialGroupLayout, ScheduleLookupKey,
    SisModulusProfileId,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::SelectedProverOpeningData;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_types::{CommittedGroup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_field::{CanonicalEncoding, Field, One, Prime128Offset275, Ring, Zero};

/// Declares a config that takes everything except its field and its SIS
/// modulus profile from `fp128::Dense`.
macro_rules! dense_config {
    ($name:ident, $field:ty, $profile:ident, $family:literal) => {
        #[derive(Clone, Copy, Debug, Default)]
        struct $name;

        impl CommitmentConfig for $name {
            type Field = $field;
            type ExtField = $field;

            const RING_DIMENSION_SCHEDULE: akita_config::RingDimensionSchedule =
                fp128::Dense::RING_DIMENSION_SCHEDULE;

            fn decomposition() -> akita_params::DecompositionParams {
                fp128::Dense::decomposition()
            }
            fn ring_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
                fp128::Dense::ring_challenge_config(d)
            }
            fn sis_modulus_profile() -> SisModulusProfileId {
                SisModulusProfileId::$profile
            }
            fn opening_basis_range() -> (u32, u32) {
                fp128::Dense::opening_basis_range()
            }
            fn inner_basis_range() -> (u32, u32) {
                fp128::Dense::inner_basis_range()
            }
            fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
                fp128::Dense::committed_source_class()
            }
            fn schedule_family_name() -> &'static str {
                $family
            }
        }
    };
}

dense_config!(
    Offset275Dense,
    Prime128Offset275,
    Q128Offset275,
    "test_q128_offset275_dense"
);
dense_config!(
    Offset275FieldWithA7f7Profile,
    Prime128Offset275,
    Q128OffsetA7F7,
    "test_q128_offset275_field_a7f7_profile"
);
dense_config!(
    A7f7FieldWithOffset275Profile,
    fp128::Field,
    Q128Offset275,
    "test_q128_a7f7_field_offset275_profile"
);

type Cfg = Offset275Dense;
type F = Prime128Offset275;

const NV: usize = 14;
const SESSION: &[u8] = b"q128-offset275-e2e";

fn planned_catalog() -> TrustedScheduleCatalog<Cfg> {
    let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(NV));
    let policy = policy_of::<Cfg>();
    let planned = akita_planner::find_schedule(
        &key,
        Cfg::committed_source_contract().unwrap(),
        &[],
        &policy,
        Cfg::ring_challenge_config,
    )
    .unwrap();
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &planned.schedule.root.params,
        )
        .unwrap(),
        precommitteds: key.precommitteds.clone(),
    };
    let validated = ValidatedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        [(profiles, planned.schedule)],
        &policy,
        Cfg::ring_challenge_config,
    )
    .unwrap();
    TrustedScheduleCatalog::new(validated).unwrap()
}

#[test]
fn profiles_stay_bound_to_their_exact_modulus() {
    validate_config_policy::<Offset275Dense>().unwrap();
    validate_config_policy::<fp128::Dense>().unwrap();

    // Sharing table rows does not make the two 128-bit profiles
    // interchangeable: each one admits only its own field.
    for error in [
        validate_config_policy::<Offset275FieldWithA7f7Profile>().unwrap_err(),
        validate_config_policy::<A7f7FieldWithOffset275Profile>().unwrap_err(),
    ] {
        assert!(
            error.to_string().contains("does not match field modulus"),
            "{error}"
        );
    }
}

fn on_large_stack(test: fn()) {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(test)
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn offset275_config_commits_opens_and_verifies() {
    on_large_stack(offset275_roundtrip);
}

fn offset275_roundtrip() {
    let scheme = AkitaCommitmentScheme::<Cfg>::new(planned_catalog());
    let setup = scheme.setup_prover(NV, 1).unwrap();
    let backend = CpuBackend::new(setup.expanded.clone()).unwrap();
    let evals: Vec<F> = (0..(1 << NV))
        .map(|index| F::from_u64(index as u64))
        .collect();
    let poly = DensePoly::<F>::from_field_evals(NV, &evals).unwrap();
    let output = backend
        .commit(
            scheme.schedules(),
            &backend.import_source(vec![poly]).unwrap(),
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    let commitment = output.committed_group;
    let point: Vec<F> = (0..NV)
        .map(|index| F::from_u64((index + 2) as u64))
        .collect();
    let opening = evals
        .iter()
        .zip(lagrange_weights(&point).unwrap())
        .fold(F::zero(), |sum, (&value, weight)| sum + value * weight);
    let group =
        PolynomialGroupClaims::new(point.clone(), vec![opening], commitment.clone()).unwrap();
    let claims = OpeningClaims::from_groups(vec![group]).unwrap();
    let prover_data = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        claims,
        vec![output.private_handle],
        scheme.schedules(),
    )
    .unwrap();
    let selection = prover_data.selection();
    let proof = scheme
        .batched_prove(&setup, prover_data, &backend, SESSION, BasisMode::Lagrange)
        .unwrap();

    let verifier = scheme
        .verifier(scheme.setup_verifier(&setup).unwrap())
        .unwrap();
    let verify = |claimed: F| {
        let group = PolynomialGroupClaims::new(point.clone(), vec![claimed], &commitment).unwrap();
        let claims = OpeningClaims::from_groups(vec![group]).unwrap();
        let statement = GroupBatchStatement::new(selection, claims).unwrap();
        verifier.batched_verify(&proof, SESSION, statement, BasisMode::Lagrange)
    };
    verify(opening).unwrap();
    assert!(verify(opening + F::one()).is_err());
}

/// Commits one fixed dense polynomial under `$cfg` and returns the public
/// commitment.
macro_rules! committed_group {
    ($cfg:ty, $catalog:expr) => {{
        let scheme = AkitaCommitmentScheme::<$cfg>::new($catalog);
        let setup = scheme.setup_prover(NV, 1).unwrap();
        let backend = CpuBackend::new(setup.expanded.clone()).unwrap();
        let evals: Vec<<$cfg as CommitmentConfig>::Field> = (0..(1 << NV))
            .map(|index| Ring::from_u64(index as u64))
            .collect();
        let poly = DensePoly::from_field_evals(NV, &evals).unwrap();
        backend
            .commit(
                scheme.schedules(),
                &backend.import_source(vec![poly]).unwrap(),
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap()
            .committed_group
    }};
}

#[test]
fn commitment_binds_the_profile_to_the_exact_field_modulus() {
    on_large_stack(|| {
        // Tag 3 on the 2^128 - 275 field.
        let offset275 = committed_group!(Offset275Dense, planned_catalog());
        assert_decode_rejects_profile(
            &offset275,
            SisModulusProfileId::Q128Offset275,
            SisModulusProfileId::Q128OffsetA7F7,
        );
        assert_check_rejects_profile(&offset275, SisModulusProfileId::Q128OffsetA7F7);
        // Tag 5 on the A7F7 field.
        let a7f7 = committed_group!(
            fp128::Dense,
            akita_config::test_support::workspace_schedule_catalog::<fp128::Dense>().unwrap()
        );
        assert_decode_rejects_profile(
            &a7f7,
            SisModulusProfileId::Q128OffsetA7F7,
            SisModulusProfileId::Q128Offset275,
        );
        assert_check_rejects_profile(&a7f7, SisModulusProfileId::Q128Offset275);
    });
}

/// Rebuilds the inner matrix, the outer matrix, and both matrices of an
/// in-memory commitment under `other`, and requires `check` and serialization
/// to reject each result. The rebuilt matrices keep their geometry and their
/// SIS bounds, because the two profiles share rows.
fn assert_check_rejects_profile<F>(commitment: &CommittedGroup<F>, other: SisModulusProfileId)
where
    F: Field + CanonicalEncoding + Valid + AkitaSerialize,
{
    let inner = commitment.profile().inner.matrix;
    let outer = commitment.profile().outer.matrix;
    let other_inner = InnerCommitMatrixParams::try_new(
        inner.security_policy(),
        inner.sis_table_key().unwrap().table_digest,
        other,
        inner.output_rank(),
        inner.input_width(),
        inner.coeff_linf_bound().unwrap(),
        inner.ring_dimension(),
    )
    .unwrap();
    let other_outer = OuterCommitMatrixParams::try_new(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        other,
        outer.output_rank(),
        outer.input_width(),
        outer.coeff_linf_bound(),
        outer.ring_dimension(),
    )
    .unwrap();

    for (swap_inner, swap_outer) in [(true, true), (true, false), (false, true)] {
        let mut mismatched = commitment.clone();
        if swap_inner {
            mismatched.profile.inner.matrix = other_inner;
        }
        if swap_outer {
            mismatched.profile.outer.matrix = other_outer;
        }
        for error in [
            mismatched.check().unwrap_err(),
            mismatched
                .serialize_uncompressed(&mut Vec::new())
                .unwrap_err(),
        ] {
            assert!(
                error
                    .to_string()
                    .contains("SIS modulus profile does not match the field"),
                "{error}"
            );
        }
    }
}

/// Rewrites the inner tag, the outer tag, and both tags of a serialized
/// commitment from `own` to `other`, and requires the decoder to reject each
/// result. The two profiles share a bit width and their SIS rows, so only the
/// exact modulus tells them apart.
fn assert_decode_rejects_profile<F>(
    commitment: &CommittedGroup<F>,
    own: SisModulusProfileId,
    other: SisModulusProfileId,
) where
    F: Field + CanonicalEncoding + Valid + AkitaSerialize + AkitaDeserialize<Context = ()>,
{
    let mut bytes = Vec::new();
    commitment.serialize_uncompressed(&mut bytes).unwrap();
    let decoded = CommittedGroup::<F>::deserialize_uncompressed(&bytes[..], &()).unwrap();
    assert_eq!(decoded.profile(), commitment.profile());
    assert_eq!(decoded.rows().coeffs(), commitment.rows().coeffs());

    // A matrix encodes its profile tag, policy tag and role tag directly
    // before its SIS table digest, so the digest locates both profile tags.
    let digest = commitment
        .profile()
        .outer
        .matrix
        .sis_table_key()
        .table_digest
        .0;
    let tag_offsets: Vec<usize> = bytes
        .windows(digest.len())
        .enumerate()
        .filter(|&(_, window)| window == digest)
        .map(|(index, _)| index - 3)
        .collect();
    assert_eq!(tag_offsets.len(), 2, "one inner and one outer matrix");

    for offsets in [&tag_offsets[..], &tag_offsets[..1], &tag_offsets[1..]] {
        let mut rewritten = bytes.clone();
        for &offset in offsets {
            assert_eq!(rewritten[offset], own.tag());
            rewritten[offset] = other.tag();
        }
        let error = CommittedGroup::<F>::deserialize_uncompressed(&rewritten[..], &()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("SIS modulus profile does not match the field"),
            "{error}"
        );
    }
}
