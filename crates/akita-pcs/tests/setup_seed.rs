//! Setup-identity tests for the explicit setup seed.
//!
//! Setup is a deterministic function of the public seed and the requested
//! shape. These tests pin that: one seed reproduces one setup exactly, two
//! seeds share nothing, and a proof only verifies under the seed it was made
//! with.
//!
//! The seed is trusted deployment configuration on both sides of the wire. It
//! is never carried by a proof, so cross-seed rejection here stands in for an
//! operator misconfiguration rather than an attack.

#![allow(missing_docs)]
#![cfg(feature = "schedules-default")]

mod common;

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend, DensePoly, UniformProverStack};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::{AkitaTranscript, Transcript};
use akita_types::{
    derive_public_matrix_prefix, AkitaExpandedSetup, AkitaSetupSeed, BasisMode, SetupMatrixCapacity,
};
use common::{
    dense_field_evals, init_rayon_pool, opening_from_poly_for_layout, prove_input, random_point,
    run_on_large_stack, verify_input, F,
};
use jolt_field::CanonicalBytes;

type Cfg = fp128::Dense;

/// Smallest shape the production `fp128` D64 catalog covers.
const SETUP_NV: usize = 14;
const SETUP_POLYS: usize = 1;

const TRANSCRIPT_DOMAIN: &[u8] = b"setup-seed-tests/dense";

/// A second identity that is not the protocol default.
fn alternate_seed() -> AkitaSetupSeed {
    AkitaSetupSeed::shake256_paged_v1([0x5a; 32])
}

fn prover_setup(setup_seed: AkitaSetupSeed) -> akita_prover::AkitaProverSetup<F> {
    AkitaCommitmentScheme::<Cfg>::setup_prover(SETUP_NV, SETUP_POLYS, setup_seed)
        .expect("prover setup")
}

/// Field elements the test shape actually asks for.
///
/// With `disk-persistence` a warm cache may hand back a longer covering prefix,
/// so every comparison below is bounded by this window rather than by whatever
/// the local cache happened to materialize. Comparing the full materialized
/// length would make these tests depend on unrelated runs, and would re-derive
/// a much larger stream than the shape needs.
fn requested_fields() -> usize {
    Cfg::setup_matrix_capacity(SETUP_NV, SETUP_POLYS)
        .expect("setup capacity for the test shape")
        .num_field_elements
}

/// The prefix a setup must materialize for the test shape.
fn requested_prefix(setup: &akita_prover::AkitaProverSetup<F>) -> &[F] {
    let fields = requested_fields();
    let materialized = setup.expanded.shared_matrix().as_field_slice();
    assert!(
        materialized.len() >= fields,
        "setup must cover the requested shape: {} materialized < {fields} requested",
        materialized.len()
    );
    &materialized[..fields]
}

// ---------------------------------------------------------------------------
// Determinism and separation
// ---------------------------------------------------------------------------

#[test]
fn one_seed_and_shape_reproduce_the_same_setup() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let first = prover_setup(AkitaSetupSeed::DEFAULT);
        let second = prover_setup(AkitaSetupSeed::DEFAULT);

        assert_eq!(*first.setup_seed(), AkitaSetupSeed::DEFAULT);
        assert_eq!(*second.setup_seed(), AkitaSetupSeed::DEFAULT);
        assert_matches_seed_derived_stream(&first);
        assert_matches_seed_derived_stream(&second);
        assert_eq!(requested_prefix(&first), requested_prefix(&second));

        // Narrowing both to the requested capacity makes the byte comparison
        // independent of how much either happened to materialize.
        let capacity = SetupMatrixCapacity {
            num_field_elements: requested_fields(),
        };
        let first_verifier = first
            .to_verifier_setup(capacity)
            .expect("first verifier setup");
        let second_verifier = second
            .to_verifier_setup(capacity)
            .expect("second verifier setup");
        assert_eq!(first_verifier, second_verifier);

        let mut first_bytes = Vec::new();
        let mut second_bytes = Vec::new();
        first_verifier
            .serialize_compressed(&mut first_bytes)
            .expect("serialize first verifier setup");
        second_verifier
            .serialize_compressed(&mut second_bytes)
            .expect("serialize second verifier setup");
        assert_eq!(first_bytes, second_bytes);
    });
}

/// Assert a setup's requested prefix is exactly the stream its seed derives.
fn assert_matches_seed_derived_stream(setup: &akita_prover::AkitaProverSetup<F>) {
    let expected = derive_public_matrix_prefix::<F>(requested_fields(), setup.setup_seed());
    assert_eq!(
        requested_prefix(setup),
        expected.as_field_slice(),
        "setup matrix must be the deterministic stream of its seed"
    );
}

#[test]
fn distinct_seeds_produce_distinct_matrices_and_verifier_setups() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let default_setup = prover_setup(AkitaSetupSeed::DEFAULT);
        let alternate_setup = prover_setup(alternate_seed());

        assert_eq!(*default_setup.setup_seed(), AkitaSetupSeed::DEFAULT);
        assert_eq!(*alternate_setup.setup_seed(), alternate_seed());

        assert_ne!(
            requested_prefix(&default_setup),
            requested_prefix(&alternate_setup),
            "two seeds must derive different public streams"
        );

        let default_verifier = AkitaCommitmentScheme::<Cfg>::setup_verifier(&default_setup)
            .expect("default verifier setup");
        let alternate_verifier = AkitaCommitmentScheme::<Cfg>::setup_verifier(&alternate_setup)
            .expect("alternate verifier setup");
        assert_ne!(default_verifier, alternate_verifier);
        assert_ne!(
            default_verifier.setup_seed(),
            alternate_verifier.setup_seed()
        );
    });
}

#[test]
fn setup_serialization_preserves_the_selected_seed() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let setup = prover_setup(alternate_seed());

        let mut bytes = Vec::new();
        setup
            .expanded
            .serialize_compressed(&mut bytes)
            .expect("serialize setup");
        let decoded =
            AkitaExpandedSetup::<F>::deserialize_compressed(&bytes[..], &()).expect("decode setup");

        assert_eq!(*decoded.setup_seed(), alternate_seed());
        assert_eq!(decoded, setup.expanded.as_ref().clone());
    });
}

/// Recursive setup-prefix provisioning under a non-default seed.
///
/// Gated on the generated recursive catalog: `schedules-default` does not link
/// it, so without this feature the config resolves no schedules at all.
#[cfg(feature = "schedules-fp128-onehot-recursive")]
mod recursive {
    use super::{alternate_seed, init_rayon_pool, run_on_large_stack, AkitaCommitmentScheme};
    use akita_config::proof_optimized::fp128;
    use akita_config::{CommitmentConfig, RecursiveCommitmentConfig};

    type Recursive = RecursiveCommitmentConfig<fp128::OneHot>;

    #[test]
    #[ignore = "production-sized recursive profile; run explicitly with --release"]
    fn prefix_slots_are_provisioned_under_a_non_default_seed() {
        init_rayon_pool();
        run_on_large_stack(|| {
            // Matches the shape the recursive round-trip driver provisions:
            // two precommitted groups plus a two-polynomial final group.
            const FINAL_NV: usize = 32;
            const TOTAL_GROUP_SIZE: usize = 4;

            assert!(
                Recursive::recursive_setup_planning(),
                "this test only means something for a recursive config"
            );

            let setup = AkitaCommitmentScheme::<Recursive>::setup_prover(
                FINAL_NV,
                TOTAL_GROUP_SIZE,
                alternate_seed(),
            )
            .expect("recursive setup under a non-default seed");

            assert!(
                !setup.prefix_slots.is_empty(),
                "recursive setup must precompute setup-prefix slots for any seed"
            );
            assert_eq!(*setup.setup_seed(), alternate_seed());
            // The registry is bound to the matrix it committed to, not the default.
            assert_eq!(*setup.prefix_slots.setup_seed(), alternate_seed());

            let required = akita_config::setup_prefix_slot_ids_for_capacity::<Recursive>(
                FINAL_NV,
                TOTAL_GROUP_SIZE,
            )
            .expect("required prefix slot ids");
            assert_eq!(setup.prefix_slots.len(), required.len());
            for id in &required {
                assert!(
                    setup.prefix_slots.get(id).is_some(),
                    "missing setup-prefix slot {id:?} under a non-default seed"
                );
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Transcript identity
// ---------------------------------------------------------------------------

#[test]
fn transcript_setup_identity_tracks_only_the_seed() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let opening_layout =
            akita_types::OpeningClaimsLayout::new(SETUP_NV, 1).expect("singleton opening batch");
        let row = Cfg::resolve_catalog_row_for_opening(&opening_layout).expect("catalog row");
        let selection = row.selection();
        let schedule = row.into_schedule();

        let descriptor_bytes = |setup_seed: AkitaSetupSeed| {
            let setup = prover_setup(setup_seed);
            let mut transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
            akita_config::bind_transcript_instance_descriptor::<F, _, Cfg>(
                setup.expanded.as_ref(),
                &opening_layout,
                selection,
                &schedule,
                BasisMode::Lagrange,
                &mut transcript,
            )
            .expect("bind instance descriptor");
            transcript
                .challenge_scalar(akita_transcript::labels::CHALLENGE_SUMCHECK_BATCH)
                .to_bytes_le_vec()
        };

        let default_challenge = descriptor_bytes(AkitaSetupSeed::DEFAULT);
        let repeat_challenge = descriptor_bytes(AkitaSetupSeed::DEFAULT);
        let alternate_challenge = descriptor_bytes(alternate_seed());

        assert_eq!(
            default_challenge, repeat_challenge,
            "one seed must bind one transcript identity"
        );
        assert_ne!(
            default_challenge, alternate_challenge,
            "the transcript preamble must separate setups that differ only by seed"
        );
    });
}

// ---------------------------------------------------------------------------
// End-to-end cross-seed rejection
// ---------------------------------------------------------------------------

#[test]
fn a_proof_verifies_under_its_own_seed_and_is_rejected_under_another() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let opening_layout =
            akita_types::OpeningClaimsLayout::new(SETUP_NV, 1).expect("singleton opening batch");
        let row = Cfg::resolve_catalog_row_for_opening(&opening_layout).expect("catalog row");
        let layout = row.schedule().root.params.clone();
        let schedule = row.into_schedule();

        let evals = dense_field_evals(SETUP_NV, 0x5eed_0000);
        let poly = DensePoly::<F>::from_field_evals(SETUP_NV, &evals).expect("dense poly");
        let point = random_point(SETUP_NV, 0x5eed_0001);
        let opening = opening_from_poly_for_layout(
            &poly,
            &point,
            &layout.final_group_scalar().expect("scalar final group"),
            BasisMode::Lagrange,
        );

        let prover = prover_setup(AkitaSetupSeed::DEFAULT);
        let prepared = CpuBackend::DEFAULT
            .prepare_setup(&prover)
            .expect("prepared setup");
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, prover.expanded.as_ref())
                .expect("prover stack");

        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = AkitaCommitmentScheme::<Cfg>::commit(
            &prover,
            std::slice::from_ref(&poly),
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit");

        let poly_refs = [&poly];
        let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
        let proof = AkitaCommitmentScheme::<Cfg>::batched_prove(
            &prover,
            prove_input::<Cfg, _>(&point[..], &poly_refs[..], &commitment, hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("prove under the default seed");

        let openings = [opening];

        // Matching seed: the verifier rebuilds the same setup and accepts.
        let matching_verifier = AkitaCommitmentScheme::<Cfg>::setup_verifier_for_schedule(
            &prover,
            &schedule,
            &opening_layout,
        )
        .expect("verifier setup under the proving seed");
        let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
        AkitaCommitmentScheme::<Cfg>::batched_verify(
            &proof,
            &matching_verifier,
            &mut verifier_transcript,
            verify_input::<Cfg>(&point[..], &openings[..], &commitment),
            BasisMode::Lagrange,
        )
        .expect("a proof must verify under the seed it was made with");

        // Mismatched seed: a misconfigured verifier must not accept. The two
        // sides disagree on both the absorbed seed digest and the derived
        // matrix, so rejection surfaces wherever a diverged value first fails a
        // check. Assert rejection, not a specific error variant.
        let mismatched_source = prover_setup(alternate_seed());
        let mismatched_verifier = AkitaCommitmentScheme::<Cfg>::setup_verifier_for_schedule(
            &mismatched_source,
            &schedule,
            &opening_layout,
        )
        .expect("verifier setup under a different seed");
        let mut mismatched_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
        let rejected = AkitaCommitmentScheme::<Cfg>::batched_verify(
            &proof,
            &mismatched_verifier,
            &mut mismatched_transcript,
            verify_input::<Cfg>(&point[..], &openings[..], &commitment),
            BasisMode::Lagrange,
        );
        assert!(
            rejected.is_err(),
            "a proof must not verify against a setup built from a different seed"
        );
    });
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

#[test]
fn a_sampled_seed_drives_a_working_setup_and_reports_itself() {
    init_rayon_pool();
    run_on_large_stack(|| {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        // Deterministic RNG: sampling is exercised, the test stays reproducible.
        let mut rng = StdRng::seed_from_u64(0x5eed_2026);
        let sampled = AkitaSetupSeed::from_rng(&mut rng);
        assert_ne!(sampled, AkitaSetupSeed::DEFAULT);

        let setup = prover_setup(sampled.clone());
        assert_eq!(*setup.setup_seed(), sampled);
        assert_matches_seed_derived_stream(&setup);

        let rebuilt = prover_setup(sampled.clone());
        assert_eq!(*rebuilt.setup_seed(), sampled);
        assert_eq!(
            requested_prefix(&setup),
            requested_prefix(&rebuilt),
            "a recorded sampled seed must reproduce its setup exactly"
        );

        let verifier =
            AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).expect("verifier setup");
        assert_eq!(*verifier.setup_seed(), sampled);
    });
}

#[test]
fn setup_shape_and_seed_are_independent_axes() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let narrow = AkitaCommitmentScheme::<Cfg>::setup_prover(
            SETUP_NV,
            SETUP_POLYS,
            AkitaSetupSeed::DEFAULT,
        )
        .expect("narrow setup");
        let wide = AkitaCommitmentScheme::<Cfg>::setup_prover(
            SETUP_NV + 1,
            SETUP_POLYS,
            AkitaSetupSeed::DEFAULT,
        )
        .expect("wide setup");

        // A larger shape extends the same public stream rather than replacing
        // it. Both sides are compared over the narrow shape's requested window,
        // which a wider shape always covers, so cache state cannot flip this.
        assert_eq!(narrow.setup_seed(), wide.setup_seed());
        let wide_fields = wide.expanded.shared_matrix().as_field_slice();
        assert_eq!(
            requested_prefix(&narrow),
            &wide_fields[..requested_fields()],
            "one seed defines one stream; shape only chooses how much is materialized"
        );

        // The descriptor must record the shape that was asked for, not the
        // shape of whatever covering prefix the cache handed back.
        assert_eq!(narrow.expanded.descriptor().max_num_vars, SETUP_NV);
        assert_eq!(wide.expanded.descriptor().max_num_vars, SETUP_NV + 1);
    });
}
