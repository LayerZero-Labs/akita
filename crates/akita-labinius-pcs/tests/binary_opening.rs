//! End-to-end binary openings and frozen binary proof digests.

#![cfg(feature = "labinius")]

mod common;

use akita_error::AkitaError;
use akita_labinius_pcs::{
    family::{tables::shipped_catalog, Family128, Family64, FieldFamily},
    shipped::{ShippedCatalogError, SupportedGeometry, SUPPORTED_GEOMETRIES},
};
use akita_labinius_verifier::{RootOpeningMode, RootStatement};
use common::{artifacts, geometry, locate, message, sample_geometry_opening, Case, H};

/// Accept the honest opening, then reject each changed statement and each
/// changed proof as `InvalidProof`.
///
/// No row targets a proof-of-work nonce. The reduction emits them on its own
/// channel between oracle calls, where a recording oracle sees nothing; its
/// own tests change every one.
fn opening_is_accepted_and_every_change_is_rejected<P: FieldFamily>() {
    let case = Case::<P>::new(16, RootOpeningMode::Binary);
    let committed = case.prover.commit::<H>(&case.source).unwrap();
    let proof = case
        .prover
        .open::<H>(&case.source, &committed, &case.statement())
        .unwrap();
    case.verifier
        .verify::<H>(&committed.commitment, &case.statement(), &proof)
        .unwrap();

    let (recorded, sections) = case.open_recorded(&committed);
    assert_eq!(recorded, proof, "proving is deterministic");
    let response = locate(&proof, &sections.response);
    let nested = locate(&proof, &sections.nested);
    assert!(0 < response.start && response.end < nested.start && nested.end == proof.len());

    let other = case
        .prover
        .commit::<H>(&message(case.source.len(), 3))
        .unwrap();
    let mut moved = case.point.clone();
    moved[0] += H::ONE;
    let flipped = |index: usize| {
        let mut changed = proof.clone();
        changed[index] ^= 1;
        changed
    };
    let honest = (&committed.commitment, case.point.as_slice(), case.value);
    let rows = [
        (
            "a wrong value",
            (honest.0, honest.1, case.value + H::ONE),
            proof.clone(),
        ),
        (
            "a wrong point",
            (honest.0, moved.as_slice(), honest.2),
            proof.clone(),
        ),
        (
            "a commitment to another message",
            (&other.commitment, honest.1, honest.2),
            proof.clone(),
        ),
        (
            "a truncated proof",
            honest,
            proof[..proof.len() - 1].to_vec(),
        ),
        ("a trailing byte", honest, [proof.as_slice(), &[0]].concat()),
        ("a reduction byte before the response", honest, flipped(0)),
        (
            "a response commitment byte",
            honest,
            flipped(response.end - 1),
        ),
        (
            "a reduction byte after the response",
            honest,
            flipped(response.end),
        ),
        ("the nested proof length", honest, flipped(nested.start)),
        (
            "a nested proof byte",
            honest,
            flipped(nested.start + nested.len() / 2),
        ),
    ];
    for (name, (commitment, point, value), proof) in rows {
        let result = case.verifier.verify::<H>(
            commitment,
            &RootStatement {
                binary: Some((point, value)),
                prime: None,
            },
            &proof,
        );
        assert!(
            matches!(result, Err(AkitaError::InvalidProof)),
            "{name} must be rejected as an invalid proof, got {result:?}"
        );
    }
}

#[test]
fn fp64_opening_is_accepted_and_every_change_is_rejected() {
    opening_is_accepted_and_every_change_is_rejected::<Family64>();
}

#[test]
fn fp128_opening_is_accepted_and_every_change_is_rejected() {
    opening_is_accepted_and_every_change_is_rejected::<Family128>();
}

/// Both shipped catalogs carry the rows of every supported geometry.
#[test]
fn every_supported_geometry_resolves_in_both_shipped_catalogs() {
    fn resolves<P: FieldFamily>() {
        for supported in SUPPORTED_GEOMETRIES {
            shipped_catalog::<P>(supported, &artifacts()).unwrap();
        }
        let unsupported = SupportedGeometry {
            log_num_cells: 14,
            ..geometry(16)
        };
        assert!(matches!(
            shipped_catalog::<P>(unsupported, &artifacts()),
            Err(ShippedCatalogError::UnsupportedGeometry)
        ));
    }
    resolves::<Family64>();
    resolves::<Family128>();
}

#[test]
#[ignore = "sample geometry numbers; run explicitly in release with parallel"]
fn fp64_sample_geometry_opening() {
    sample_geometry_opening::<Family64>("fp64", RootOpeningMode::Binary);
}

#[test]
#[ignore = "sample geometry numbers; run explicitly in release with parallel"]
fn fp128_sample_geometry_opening() {
    sample_geometry_opening::<Family128>("fp128", RootOpeningMode::Binary);
}

/// Complete binary proof bytes stay fixed when the prime mode is added.
#[cfg(all(feature = "transcript-blake2b", not(feature = "transcript-keccak")))]
#[test]
fn binary_proof_bytes_are_pinned_for_both_families() {
    fn digest<P: FieldFamily>() -> String {
        let case = Case::<P>::new(16, RootOpeningMode::Binary);
        let committed = case.prover.commit::<H>(&case.source).unwrap();
        let proof = case
            .prover
            .open::<H>(&case.source, &committed, &case.statement())
            .unwrap();
        let repeated = case
            .prover
            .open::<H>(&case.source, &committed, &case.statement())
            .unwrap();
        assert_eq!(proof, repeated, "the complete proof must be deterministic");
        akita_params::digest_descriptor_bytes(&proof)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
    assert_eq!(
        digest::<Family64>(),
        "19c08bbf71e5566c23c8da9d87fa6a8baa164b981f2794b142c75957cdeb78ec"
    );
    assert_eq!(
        digest::<Family128>(),
        "eb4d420ce70c9586aa241a147953ef5ff2c43b34bf77dd46c1298629e254ed9b"
    );
}
