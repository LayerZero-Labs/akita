//! Prime-only and combined openings, canonical commitment round trips, and
//! table-driven rejection of changes to every statement and proof section.

#![cfg(feature = "labinius")]

mod common;

use akita_error::AkitaError;
use akita_labinius_pcs::family::{Family128, Family64, FieldFamily};
use akita_labinius_verifier::{root::verify_root_reduction_bytes, RootOpeningMode, RootStatement};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_types::CommittedGroup;
use common::{locate, message, sample_geometry_opening, Case, H};
use jolt_field::One;

fn opening_and_tampering<P: FieldFamily>(mode: RootOpeningMode) {
    let case = Case::<P>::new(16, mode);
    let committed = case.prover.commit::<H>(&case.source).unwrap();
    let statement = case.statement();
    // The public commitment and opening bytes cross the serialization boundary.
    let mut commitment_bytes = Vec::new();
    committed
        .commitment
        .serialize_compressed(&mut commitment_bytes)
        .unwrap();
    let mut reader = commitment_bytes.as_slice();
    let commitment = CommittedGroup::<P::Base>::deserialize_compressed(&mut reader, &()).unwrap();
    assert!(reader.is_empty());
    let mut roundtrip = Vec::new();
    commitment.serialize_compressed(&mut roundtrip).unwrap();
    assert_eq!(roundtrip, commitment_bytes);
    let proof = case
        .prover
        .open::<H>(&case.source, &committed, &statement)
        .unwrap();
    case.verifier
        .verify::<H>(&commitment, &statement, &proof)
        .unwrap();
    let (recorded, sections) = case.open_recorded(&committed);
    assert_eq!(proof, recorded, "serialized proving must be deterministic");
    let prime = locate(&proof, &sections.prime);
    let response = locate(&proof, &sections.response);
    let nested = locate(&proof, &sections.nested);
    assert_eq!(prime.start == 0, !mode.has_binary());
    assert!(prime.end < response.start && response.end < nested.start);
    assert_eq!(nested.end, proof.len());

    let other = case
        .prover
        .commit::<H>(&message(case.source.len(), 3))
        .unwrap();
    let honest_prime = case.prime.as_ref().unwrap();
    // Every row owns its statement/proof changes, so one honest opening feeds
    // all mutations without rebuilding a setup or another honest proof.
    let mut rows = Vec::new();
    let mut changed = honest_prime.clone();
    changed.value += P::Challenge::one();
    rows.push((
        "prime value".to_owned(),
        mode,
        changed,
        case.point.clone(),
        case.value,
        false,
        proof.clone(),
    ));
    let mut changed = honest_prime.clone();
    changed.weights[0] += P::Challenge::one();
    rows.push((
        "prime weights".to_owned(),
        mode,
        changed,
        case.point.clone(),
        case.value,
        false,
        proof.clone(),
    ));
    for index in 0..honest_prime.column_point.len() {
        let mut changed = honest_prime.clone();
        changed.column_point[index] += P::Challenge::one();
        rows.push((
            format!("column point coordinate {index}"),
            mode,
            changed,
            case.point.clone(),
            case.value,
            false,
            proof.clone(),
        ));
    }
    for index in 0..honest_prime.ring_point.len() {
        let mut changed = honest_prime.clone();
        changed.ring_point[index] += P::Challenge::one();
        rows.push((
            format!("ring point coordinate {index}"),
            mode,
            changed,
            case.point.clone(),
            case.value,
            false,
            proof.clone(),
        ));
    }
    if mode.has_binary() {
        rows.push((
            "binary value".to_owned(),
            mode,
            honest_prime.clone(),
            case.point.clone(),
            case.value + H::ONE,
            false,
            proof.clone(),
        ));
        for index in 0..case.point.len() {
            let mut point = case.point.clone();
            point[index] += H::ONE;
            rows.push((
                format!("binary point coordinate {index}"),
                mode,
                honest_prime.clone(),
                point,
                case.value,
                false,
                proof.clone(),
            ));
        }
    }
    rows.push((
        "image commitment to another message".to_owned(),
        mode,
        honest_prime.clone(),
        case.point.clone(),
        case.value,
        true,
        proof.clone(),
    ));
    for other_mode in [
        RootOpeningMode::Binary,
        RootOpeningMode::Prime,
        RootOpeningMode::Both,
    ] {
        if other_mode != mode {
            rows.push((
                format!("{mode:?} proof verified as {other_mode:?}"),
                other_mode,
                honest_prime.clone(),
                case.point.clone(),
                case.value,
                false,
                proof.clone(),
            ));
        }
    }
    let mut proof_sections = vec![
        ("prime commitment bytes", prime.end - 1),
        (
            "reduction between prime and response commitments",
            prime.end,
        ),
        ("response commitment bytes", response.end - 1),
        ("reduction after response commitment", response.end),
        ("nested proof framing", nested.start),
        (
            "three-group nested proof bytes",
            nested.start + nested.len() / 2,
        ),
    ];
    if mode.has_binary() {
        proof_sections.push(("reduction before prime commitment", 0));
    }
    for (name, index) in proof_sections {
        let mut changed = proof.clone();
        changed[index] ^= 1;
        rows.push((
            name.to_owned(),
            mode,
            honest_prime.clone(),
            case.point.clone(),
            case.value,
            false,
            changed,
        ));
    }
    rows.push((
        "truncated serialized proof".to_owned(),
        mode,
        honest_prime.clone(),
        case.point.clone(),
        case.value,
        false,
        proof[..proof.len() - 1].to_vec(),
    ));
    rows.push((
        "trailing serialized byte".to_owned(),
        mode,
        honest_prime.clone(),
        case.point.clone(),
        case.value,
        false,
        [proof.as_slice(), &[0]].concat(),
    ));
    for (name, expected_mode, prime, point, value, use_other, changed_proof) in rows {
        let statement = RootStatement {
            binary: expected_mode
                .has_binary()
                .then_some((point.as_slice(), value)),
            prime: expected_mode.has_prime().then_some(&prime),
        };
        let commitment = if use_other {
            &other.commitment
        } else {
            &commitment
        };
        let result = case
            .verifier
            .verify::<H>(commitment, &statement, &changed_proof);
        assert!(
            matches!(result, Err(AkitaError::InvalidProof)),
            "{name} must return InvalidProof, got {result:?}"
        );
    }
    // A caller owning the outer channel must select an oracle of the same mode.
    let mut oracle = case
        .verifier
        .oracle(&commitment, RootOpeningMode::Binary)
        .unwrap();
    let result = verify_root_reduction_bytes::<H, P::Base, P::Challenge, _, _, _>(
        case.verifier.root(),
        &statement,
        &mut oracle,
        &proof,
    );
    assert!(
        matches!(result, Err(AkitaError::InvalidProof)),
        "a binary oracle must reject a prime statement, got {result:?}"
    );
}

#[test]
fn fp64_prime_opening_and_tampering() {
    opening_and_tampering::<Family64>(RootOpeningMode::Prime);
}
#[test]
fn fp128_prime_opening_and_tampering() {
    opening_and_tampering::<Family128>(RootOpeningMode::Prime);
}
#[test]
fn fp64_both_claims_opening_and_tampering() {
    opening_and_tampering::<Family64>(RootOpeningMode::Both);
}
#[test]
fn fp128_both_claims_opening_and_tampering() {
    opening_and_tampering::<Family128>(RootOpeningMode::Both);
}

#[test]
#[ignore = "sample geometry numbers; run explicitly in release with parallel"]
fn fp64_prime_sample_geometry_opening() {
    sample_geometry_opening::<Family64>("fp64", RootOpeningMode::Prime);
}
#[test]
#[ignore = "sample geometry numbers; run explicitly in release with parallel"]
fn fp128_prime_sample_geometry_opening() {
    sample_geometry_opening::<Family128>("fp128", RootOpeningMode::Prime);
}
#[test]
#[ignore = "sample geometry numbers; run explicitly in release with parallel"]
fn fp64_both_claims_sample_geometry_opening() {
    sample_geometry_opening::<Family64>("fp64", RootOpeningMode::Both);
}
#[test]
#[ignore = "sample geometry numbers; run explicitly in release with parallel"]
fn fp128_both_claims_sample_geometry_opening() {
    sample_geometry_opening::<Family128>("fp128", RootOpeningMode::Both);
}
