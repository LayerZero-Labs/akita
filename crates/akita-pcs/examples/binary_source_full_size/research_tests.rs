//! Complete public PCS replay using the same generator as the measurements.
use super::{run, F};
use crate::metrics;
use akita_error::AkitaError;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_types::{Commitment, CommittedGroup, RingVec};
use jolt_field::Ring;
use std::path::Path;

#[path = "../../tests/binary_source_end_to_end/byte_contract.rs"]
mod byte_contract;

// Each shape owns an isolated directory; the generator revalidates cached rows.
fn fixture(shape: &'static str) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let directory = std::env::temp_dir()
                .join(format!("akita-p64-{}-{shape}", std::process::id()));
            std::fs::create_dir_all(&directory).unwrap();
            let catalog = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../artifacts")
                .join(if akita_params::DEV_PROTOCOL {
                    "schedules-binary-source-p64-dev"
                } else {
                    "schedules-binary-source-p64"
                })
                .join(format!("3-{shape}.json"));
            std::fs::copy(&catalog, directory.join("native-catalog.json"))
                .unwrap_or_else(|error| {
                    panic!("Linux catalog pending at {}: {error}", catalog.display())
                });
            run::prove(3, shape, &directory).unwrap();
            run::verify(3, shape, &directory, &metrics::Timings::default()).unwrap();
            let proof = std::fs::read(directory.join("proof.bin")).unwrap();
            mutate_statement(shape, &directory);
            mutate_commitment(shape, &directory);
            mutate_proof(shape, &directory, &proof);
            mutate_setup(shape, &directory);
            for fault in [
                Some(run::fault_backend::Fault::Quotient),
                (shape != "binary-claim").then_some(run::fault_backend::Fault::PrimePlane),
            ]
            .into_iter()
            .flatten()
            {
                run::FAULT.with(|selected| selected.set(Some(fault)));
                run::prove(3, shape, &directory).unwrap();
                run::FAULT.with(|selected| selected.set(None));
                let error =
                    run::verify(3, shape, &directory, &metrics::Timings::default()).unwrap_err();
                assert!(
                    matches!(error.downcast_ref::<AkitaError>(), Some(AkitaError::InvalidProof)),
                    "committed digit fault must fail verification: {error}"
                );
            }
            byte_contract::compare_baseline(&proof, &format!("p64-{shape}"));
            std::fs::remove_dir_all(directory).unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}

fn rejects(shape: &str, directory: &Path) {
    let error = run::verify(3, shape, directory, &metrics::Timings::default()).unwrap_err();
    assert!(
        error.downcast_ref::<AkitaError>().is_some(),
        "mutation must reach a typed protocol rejection: {error}"
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
    rejects(shape, directory);
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
fn p64_mixed_complete_schedule_and_tampering() {
    fixture("binary-and-prime-claims-plus-ordinary");
}

fn mutate_commitment(shape: &str, directory: &Path) {
    let path = directory.join("commitments.bin");
    let bytes = std::fs::read(&path).unwrap();
    let mut commitments =
        Vec::<CommittedGroup<F>>::deserialize_compressed(bytes.as_slice(), &()).unwrap();
    let binary = &commitments[0];
    let mut payload = binary.commitment().rows().coeffs().to_vec();
    payload[0] += F::from_u64(1);
    let akita_types::CommittedGroupKind::Binary(manifest) = binary.kind() else {
        panic!("fixture must contain a binary commitment");
    };
    commitments[0] = CommittedGroup::new_binary(
        *binary.profile(),
        Commitment::new(RingVec::from_coeffs(payload)),
        manifest.clone(),
    )
    .unwrap();
    let mut changed = Vec::new();
    commitments.serialize_compressed(&mut changed).unwrap();
    std::fs::write(&path, changed).unwrap();
    rejects(shape, directory);
    std::fs::write(path, bytes).unwrap();
}
