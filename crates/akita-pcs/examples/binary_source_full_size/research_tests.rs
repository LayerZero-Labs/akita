//! Complete public PCS replay using the same generator as the measurements.
use super::{fault_backend, F, FAULT};
use crate::metrics;
use crate::run::{self, config::P64};
use akita_error::AkitaError;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_types::proof::commitment::CommittedGroupKind;
use akita_types::{Commitment, CommittedGroup, RingVec};
use jolt_field::Ring;
use std::path::Path;

// Smallest feasible measured geometry for the selected planner and preset.
const LOG: usize = 22;

// Each shape owns an isolated directory; the generator revalidates cached rows.
fn fixture(shape: &'static str) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let directory =
                std::env::temp_dir().join(format!("akita-p64-{}-{shape}", std::process::id()));
            std::fs::create_dir_all(&directory).unwrap();
            let catalog = super::fixture_directory().join(format!("{LOG}-{shape}.json"));
            std::fs::copy(&catalog, directory.join("native-catalog.json")).unwrap_or_else(
                |error| panic!("missing research catalog at {}: {error}", catalog.display()),
            );
            run::prove::<P64>(LOG, shape, &directory).unwrap();
            run::verify::<P64>(LOG, shape, &directory, &metrics::Timings::default()).unwrap();
            let proof = std::fs::read(directory.join("proof.bin")).unwrap();
            mutate_statement(shape, &directory);
            mutate_commitment(shape, &directory);
            mutate_proof(shape, &directory, &proof);
            mutate_setup(shape, &directory);
            for fault in [
                Some(fault_backend::Fault::Quotient),
                (shape != "binary-claim").then_some(fault_backend::Fault::PrimePlane),
            ]
            .into_iter()
            .flatten()
            {
                FAULT.with(|selected| selected.set(Some(fault)));
                run::prove::<P64>(LOG, shape, &directory).unwrap();
                FAULT.with(|selected| selected.set(None));
                let error =
                    run::verify::<P64>(LOG, shape, &directory, &metrics::Timings::default())
                        .unwrap_err();
                assert!(
                    matches!(
                        error.downcast_ref::<AkitaError>(),
                        Some(AkitaError::InvalidProof)
                    ),
                    "committed digit fault must fail verification: {error}"
                );
            }
            super::compare_baseline(&proof, LOG, shape);
            std::fs::remove_dir_all(directory).unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}

fn rejects(shape: &str, directory: &Path) {
    let error =
        run::verify::<P64>(LOG, shape, directory, &metrics::Timings::default()).unwrap_err();
    assert!(
        matches!(
            error.downcast_ref::<AkitaError>(),
            Some(AkitaError::InvalidProof)
        ),
        "mutation must fail proof verification with InvalidProof: {error}"
    );
}

fn mutate_statement(shape: &str, directory: &Path) {
    let path = directory.join("claims.json");
    let bytes = std::fs::read(&path).unwrap();
    let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    for (prime, point) in [(false, true), (false, false), (true, true), (true, false)] {
        if (prime && shape == "binary-claim") || (!prime && shape == "prime-claim") {
            continue;
        }
        let mut changed = original.clone();
        let claim = &mut changed["claims"];
        let cell = if prime {
            if point {
                &mut claim["prime"][0][0]
            } else {
                &mut claim["prime"][1]
            }
        } else if point {
            &mut claim["word_point"][0][0]
        } else {
            &mut claim["word_value"][0]
        };
        // Change one canonical public coordinate, never its shape or encoding.
        *cell = serde_json::Value::from(cell.as_u64().unwrap() ^ 1);
        std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        rejects(shape, directory);
    }
    if shape == "binary-and-prime-claims-plus-ordinary" {
        for point in [true, false] {
            let mut changed = original.clone();
            let ordinary = &mut changed["claims"]["ordinary"];
            let cell = if point {
                &mut ordinary[0][1]
            } else {
                &mut ordinary[1]
            };
            *cell = serde_json::Value::from(cell.as_u64().unwrap() ^ 1);
            std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
            rejects(shape, directory);
        }
    }
    std::fs::write(path, bytes).unwrap();
}

fn mutate_proof(shape: &str, directory: &Path, proof: &[u8]) {
    let path = directory.join("proof.bin");
    let ranges = akita_transcript::thread_proof_ranges();
    for site in [
        akita_types::FoldSite::OpeningPayload {
            level: 0,
            ring_dimension: 64,
        }
        .id()
        .unwrap(),
        akita_types::FoldSite::NextWitnessPayload { level: 0 }
            .id()
            .unwrap(),
        akita_params::GrindingSite::FoldResponse { level: 0 }
            .site_id(u32::from(akita_params::FOLD_RESPONSE_NONCE_BITS)),
    ] {
        let range = ranges
            .iter()
            .find(|range| range.context.site_id == site.to_bytes())
            .unwrap();
        assert!(range.len > 0);
        let mut changed = proof.to_vec();
        changed[range.start] ^= 1;
        std::fs::write(&path, changed).unwrap();
        rejects(shape, directory);
    }
    std::fs::write(path, proof).unwrap();
}

fn mutate_setup(shape: &str, directory: &Path) {
    let path = directory.join("claims.json");
    let bytes = std::fs::read(&path).unwrap();
    let mut changed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let identity = changed["contexts"]
        .as_array_mut()
        .unwrap()
        .first_mut()
        .unwrap();
    let byte = &mut identity[1][0];
    *byte = serde_json::Value::from(byte.as_u64().unwrap() ^ 1);
    std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
    let error =
        run::verify::<P64>(LOG, shape, directory, &metrics::Timings::default()).unwrap_err();
    assert_eq!(
        error.downcast_ref::<AkitaError>(),
        Some(&AkitaError::InvalidInput(
            "binary statement differs from its prepared typed prefix".into()
        )),
        "setup mutation must fail the prepared-prefix identity check"
    );
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn p64_word_complete_schedule_and_tampering() {
    fixture("binary-claim");
}

#[test]
fn p64_prime_complete_schedule_and_tampering() {
    fixture("prime-claim");
}

#[test]
fn p64_word_and_prime_complete_schedule_and_tampering() {
    fixture("binary-and-prime-claims");
}

#[test]
fn p64_mixed_complete_schedule_and_tampering() {
    fixture("binary-and-prime-claims-plus-ordinary");
}

fn mutate_commitment(shape: &str, directory: &Path) {
    let path = directory.join("commitments.bin");
    let bytes = std::fs::read(&path).unwrap();
    let original = Vec::<CommittedGroup<F>>::deserialize_compressed(bytes.as_slice(), &()).unwrap();
    for (index, commitment) in original.iter().enumerate() {
        let mut commitments = original.clone();
        let mut payload = commitment.commitment().rows().coeffs().to_vec();
        payload[0] += F::from_u64(1);
        let payload = Commitment::new(RingVec::from_coeffs(payload));
        commitments[index] = match commitment.kind() {
            CommittedGroupKind::Binary(manifest) => {
                CommittedGroup::new_binary(*commitment.profile(), payload, manifest.clone())
                    .unwrap()
            }
            CommittedGroupKind::Ordinary => {
                CommittedGroup::new(*commitment.profile(), payload).unwrap()
            }
        };
        let mut changed = Vec::new();
        commitments.serialize_compressed(&mut changed).unwrap();
        std::fs::write(&path, changed).unwrap();
        rejects(shape, directory);
    }
    std::fs::write(path, bytes).unwrap();
}
