//! Freeze the legacy single-point message sequence independently of the driver.

use super::*;
use akita_transcript::{ProtocolSiteId, SITE_FAMILY_ROOT_STATEMENT};
use akita_types::binary_source::{
    binary_ordinary_profile_fixture, binary_root_statement, binary_word_manifest_fixture,
    ordinary_second_point_statement,
};
use akita_types::opening_claims::OrdinaryPointEvaluation;

#[test]
fn single_point_ordinary_and_binary_keep_the_legacy_transcript_bytes() {
    let manifest = binary_word_manifest_fixture(BinaryFieldProfile::P64, 1);
    let prime = PrimePlaneManifest::with_field_profile(
        manifest.descriptor(),
        Some(PrimeGroupClaim {
            point: vec![[2, 0].into(); 11],
            evaluations: vec![PrimePlaneClaim {
                polynomial: 0,
                value: [3, 0].into(),
            }],
        }),
        BinaryFieldProfile::P64,
    )
    .unwrap();
    let binary = vec![F::one(); manifest.payload_coefficients().unwrap()];
    let profile = binary_ordinary_profile_fixture(BinaryFieldProfile::P64);
    let compression = profile.outer_compression_plan().unwrap();
    let dimension = compression.maps().last().unwrap().ring_dimension();
    let ordinary = vec![F::from_u64(5); compression.terminal_coefficients()];
    let point = vec![E::from_base_fn(|i| F::from_u64(7 + i as u64)); profile.group.num_vars()];
    let values = [E::from_base_fn(|i| F::from_u64(11 + i as u64))];
    let lift = |values: &[F]| values.iter().copied().map(E::lift_base).collect::<Vec<_>>();
    for mixed in [false, true] {
        let mut current = new_prover_channel(b"ordinary-point-byte-contract");
        binary_root_statement::<F, E, _>(
            &mut current,
            &manifest,
            None,
            &prime,
            &binary,
            mixed.then_some((dimension, &ordinary[..], &point[..], &values[..])),
        )
        .unwrap();
        // The predecessor's sequence: two descriptors, then the commitments
        // and one ordinary point/value record. No optional extension marker.
        let mut legacy = new_prover_channel(b"ordinary-point-byte-contract");
        for (stage, bytes) in [
            (3, manifest.canonical_bytes()),
            (5, prime.canonical_bytes()),
        ] {
            akita_transcript::public_bytes(
                &mut legacy,
                ProtocolSiteId {
                    family: SITE_FAMILY_ROOT_STATEMENT,
                    stage,
                    ..ProtocolSiteId::default()
                },
                &bytes,
            )
            .unwrap();
        }
        akita_transcript::public_extensions::<F, E, _>(
            &mut legacy,
            akita_types::FoldSite::RootCommitment {
                group: 0,
                ring_dimension: manifest.b_ring_dimension(),
            }
            .id()
            .unwrap(),
            &lift(&binary),
        )
        .unwrap();
        if mixed {
            for (site, entries) in [
                (
                    akita_types::FoldSite::RootCommitment {
                        group: 1,
                        ring_dimension: dimension,
                    },
                    lift(&ordinary),
                ),
                (akita_types::FoldSite::RootPoint { group: 1 }, point.clone()),
                (
                    akita_types::FoldSite::Openings { level: 0 },
                    values.to_vec(),
                ),
            ] {
                akita_transcript::public_extensions::<F, E, _>(
                    &mut legacy,
                    site.id().unwrap(),
                    &entries,
                )
                .unwrap();
            }
        }
        assert_eq!(current.narg_string(), legacy.narg_string());
        assert_eq!(
            current.verifier_message::<[u8; 32]>(),
            legacy.verifier_message::<[u8; 32]>()
        );
    }
}

#[test]
fn second_point_manifest_has_explicit_shape_and_canonical_coordinates() {
    let point = [E::from_base_fn(|i| F::from_u64(7 + i as u64))];
    let values = [E::from_base_fn(|i| F::from_u64(11 + i as u64))];
    let mut current = new_prover_channel(b"ordinary-second-point-byte-contract");
    ordinary_second_point_statement::<F, E, _>(
        &mut current,
        OrdinaryPointEvaluation {
            point: &point,
            evaluations: &values,
        },
    )
    .unwrap();
    let mut bytes = b"AkitaOrdinaryPoints/e64\x01".to_vec();
    // Point count, second dimension, coordinate pair, slot count, value pair.
    for integer in [2u64, 1, 7, 8, 1, 11, 12] {
        bytes.extend_from_slice(&integer.to_le_bytes());
    }
    let mut expected = new_prover_channel(b"ordinary-second-point-byte-contract");
    akita_transcript::public_bytes(
        &mut expected,
        ProtocolSiteId {
            family: SITE_FAMILY_ROOT_STATEMENT,
            stage: 8,
            ..ProtocolSiteId::default()
        },
        &bytes,
    )
    .unwrap();
    assert_eq!(
        current.verifier_message::<[u8; 32]>(),
        expected.verifier_message::<[u8; 32]>()
    );
}
