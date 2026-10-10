//! End-to-end binary openings and frozen binary proof digests.

#![cfg(feature = "labinius")]

mod common;

use akita_config::SetupRequirements;
use akita_error::AkitaError;
use akita_labinius_pcs::{
    family::{tables::shipped_catalog, Family128, Family64, FieldFamily},
    shipped::{ShippedCatalogError, SupportedGeometry, SUPPORTED_GEOMETRIES},
    Prover, Verifier,
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

/// Both families construct every supported geometry with both catalogs selected.
#[test]
fn every_supported_geometry_constructs_prover_and_verifier_for_both_families() {
    fn constructs<P: FieldFamily>() {
        for supported in SUPPORTED_GEOMETRIES {
            Case::<P>::new(supported.log_num_cells, RootOpeningMode::Both);
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
    constructs::<Family64>();
    constructs::<Family128>();
}

/// Descriptor capacities must cover the binary grouped row before opening.
#[test]
fn constructors_reject_undersized_nested_setup_capacity() {
    let supported = geometry(16);
    let shipped = shipped_catalog::<Family64>(supported, &artifacts()).unwrap();
    let key = shipped.tables.response_key(&shipped.digits, None).unwrap();
    let max_num_vars = key.max_num_vars();
    let num_polynomials = key.num_polynomials().unwrap();
    let outcomes = [
        ("batch capacity", max_num_vars, num_polynomials - 1),
        ("variable capacity", max_num_vars - 1, num_polynomials),
    ]
    .map(|(name, vars, batch)| {
        let requirements = SetupRequirements::from_catalog(&shipped.digits, vars, batch).unwrap();
        // The element producer keeps the physical matrix large enough while
        // both catalogs use the intentionally undersized descriptor capacity.
        let requirements = requirements
            .union(SetupRequirements::from_catalog(&shipped.elements, vars, batch).unwrap())
            .unwrap();
        let setup = akita_pcs::new_prover_setup(&requirements).unwrap();
        let verifier_setup = setup
            .to_verifier_setup(requirements.matrix_capacity())
            .unwrap();
        (
            name,
            Prover::<Family64>::new(supported, shipped.digits.clone(), None, setup).err(),
            Verifier::<Family64>::new(supported, shipped.digits.clone(), None, verifier_setup)
                .err(),
        )
    });
    assert!(
        outcomes.iter().all(|(_, prover, verifier)| {
            matches!(prover, Some(AkitaError::InvalidSetup(_)))
                && matches!(verifier, Some(AkitaError::InvalidSetup(_)))
        }),
        "both constructors must reject undersized capacities: {outcomes:?}"
    );
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

/// Complete binary proof bytes stay fixed per family and protocol selection.
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
    let expected = if akita_params::DEV_PROTOCOL {
        (
            "11029db75032d95c1ade4a0f868b05796e21b811bd64cff4b8e24385aa113293",
            "2635ea4d40a4fa269fed27f4c49109e27174c203731644f04b9103e3c29ae25e",
        )
    } else {
        (
            "19c08bbf71e5566c23c8da9d87fa6a8baa164b981f2794b142c75957cdeb78ec",
            "eb4d420ce70c9586aa241a147953ef5ff2c43b34bf77dd46c1298629e254ed9b",
        )
    };
    assert_eq!(
        (digest::<Family64>(), digest::<Family128>()),
        (expected.0.to_owned(), expected.1.to_owned())
    );
}
