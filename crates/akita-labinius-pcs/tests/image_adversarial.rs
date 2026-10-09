#![cfg(feature = "labinius")]

mod common;

use akita_algebra::binary::{BinaryField128, BinaryField192};
use akita_error::AkitaError;
use akita_labinius_pcs::{ImageEvaluation, ImageVerifier, F};
use akita_labinius_verifier::channel::ClearChannel;
use akita_serialization::AkitaDeserialize;
use akita_types::CommittedGroup;
use common::{admitted, fixture, Case, PREFIX};
use jolt_field::One;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn rejected(case: &Case, proof: &[u8]) {
    let result = catch_unwind(AssertUnwindSafe(|| case.verify(proof, PREFIX)));
    assert!(result.is_ok(), "verifier panicked on {} bytes", proof.len());
    let error = result.unwrap().unwrap_err();
    assert!(
        !matches!(error, AkitaError::Internal(_)),
        "malformed proof reported internal error: {error}"
    );
}

#[test]
fn malformed_framing_and_payload_never_panic() {
    let case = Case::<BinaryField128>::new(0);
    let (proof, _, _) = case.prove(PREFIX);
    let inner_len = proof.len() - 8;
    for length in [0, 1, inner_len - 1, inner_len + 1, usize::MAX] {
        let mut changed = proof.clone();
        changed[..8].copy_from_slice(&(length as u64).to_le_bytes());
        rejected(&case, &changed);
    }
    // Native proofs are tens of KiB. Cover every framing prefix, both payload
    // boundaries, and 32 evenly spaced payload prefixes instead of quadratic
    // replay over every prefix of this long proof.
    let mut prefixes = (0..=8).collect::<Vec<_>>();
    prefixes.extend((1..=32).map(|i| 8 + inner_len * i / 33));
    prefixes.extend([proof.len() - 1, proof.len() - 16]);
    prefixes.sort_unstable();
    prefixes.dedup();
    for length in prefixes {
        rejected(&case, &proof[..length]);
    }
    // ClearChannel exposes two message regions: framing and inner payload.
    // Probe their first bytes, then additional payload positions and its tail.
    for offset in [0, 8, 9, 24, 40, 72, proof.len() / 2, proof.len() - 1] {
        let mut changed = proof.clone();
        changed[offset] ^= 0x80;
        rejected(&case, &changed);
    }
    let mut trailing = proof.clone();
    trailing.push(0);
    rejected(&case, &trailing);
    let mut random = proof.clone();
    let mut word = 0x518a_2341_bca0_321fu64;
    for _ in 0..8 {
        for byte in &mut random[8..] {
            word ^= word << 13;
            word ^= word >> 7;
            word ^= word << 17;
            *byte = word as u8;
        }
        rejected(&case, &random);
    }
}

#[test]
fn malformed_canonical_commitment_bytes_never_panic() {
    use akita_serialization::AkitaSerialize;
    let case = Case::<BinaryField128>::new(0);
    let mut bytes = Vec::new();
    case.output
        .committed_group
        .serialize_compressed(&mut bytes)
        .unwrap();
    for length in 0..bytes.len() {
        let result = catch_unwind(|| {
            CommittedGroup::<F>::deserialize_compressed_exact(&bytes[..length], &())
        });
        assert!(result.is_ok());
        assert!(result.unwrap().is_err());
    }
    for offset in [0, 1, 8, bytes.len() - 1] {
        let mut malformed = bytes.clone();
        malformed[offset] = 0xff;
        let result =
            catch_unwind(|| CommittedGroup::<F>::deserialize_compressed_exact(&malformed, &()));
        assert!(result.is_ok());
        // Some field-byte mutations are canonical and represent a different
        // public commitment; those must fail opening verification instead.
        if let Ok(commitment) = result.unwrap() {
            let (proof, _, _) = case.prove(PREFIX);
            let mut channel = akita_transcript::new_verifier_channel(
                b"image-tests-parent/v1",
                b"context",
                &proof,
            )
            .unwrap();
            channel.public(PREFIX).unwrap();
            assert!(case
                .verifier
                .verify_on_channel::<BinaryField128, _>(
                    &case.root,
                    &commitment,
                    case.evaluation(),
                    &mut channel
                )
                .is_err());
        }
    }
    let mut noncanonical = bytes.clone();
    let end = noncanonical.len();
    noncanonical[end - 16..].fill(0xff);
    assert!(CommittedGroup::<F>::deserialize_compressed_exact(&noncanonical, &()).is_err());
}

#[test]
fn statement_and_parent_context_are_bound() {
    let case = Case::<BinaryField128>::new(0);
    let (proof, _, _) = case.prove(PREFIX);
    let verify = |root: &akita_labinius_pcs::RootSetup,
                  commitment: &CommittedGroup<F>,
                  evaluation: ImageEvaluation<'_>,
                  verifier: &ImageVerifier| {
        let mut channel =
            akita_transcript::new_verifier_channel(b"image-tests-parent/v1", b"context", &proof)
                .unwrap();
        channel.public(PREFIX).unwrap();
        verifier.verify_on_channel::<BinaryField128, _>(root, commitment, evaluation, &mut channel)
    };
    assert!(verify(
        &admitted(0, 0x32),
        &case.output.committed_group,
        case.evaluation(),
        &case.verifier
    )
    .is_err());
    let mut point = case.point.clone();
    point[0] += F::one();
    assert!(verify(
        &case.root,
        &case.output.committed_group,
        ImageEvaluation {
            point: &point,
            value: case.value
        },
        &case.verifier
    )
    .is_err());
    assert!(verify(
        &case.root,
        &case.output.committed_group,
        ImageEvaluation {
            point: &case.point,
            value: case.value + F::one()
        },
        &case.verifier
    )
    .is_err());
    let mut channel =
        akita_transcript::new_verifier_channel(b"image-tests-parent/v1", b"context", &proof)
            .unwrap();
    channel.public(PREFIX).unwrap();
    assert!(case
        .verifier
        .verify_on_channel::<BinaryField192, _>(
            &case.root,
            &case.output.committed_group,
            case.evaluation(),
            &mut channel
        )
        .is_err());
    let mut source = case.source.clone();
    source[0] ^= 1;
    let prepared = akita_labinius_pcs::PreparedMatrix::prepare(case.root.setup()).unwrap();
    let other = case
        .prover
        .commit::<BinaryField128>(&case.root, &prepared, &source)
        .unwrap();
    assert!(verify(
        &case.root,
        &other.committed_group,
        case.evaluation(),
        &case.verifier
    )
    .is_err());
    assert!(case.verify(&proof, b"changed absorbed prefix").is_err());
    for (session, instance) in [
        (b"other-parent".as_slice(), b"context".as_slice()),
        (
            b"image-tests-parent/v1".as_slice(),
            b"other-context".as_slice(),
        ),
    ] {
        let mut channel =
            akita_transcript::new_verifier_channel(session, instance, &proof).unwrap();
        channel.public(PREFIX).unwrap();
        assert!(case
            .verifier
            .verify_on_channel::<BinaryField128, _>(
                &case.root,
                &case.output.committed_group,
                case.evaluation(),
                &mut channel
            )
            .is_err());
    }
    let f = fixture(case.layout.image_log_len());
    let mut descriptor = f.verifier_setup.expanded().descriptor().clone();
    descriptor.setup_seed = akita_types::proof::AkitaSetupSeed::shake256_paged_v1([0x77; 32]);
    let matrix = akita_types::proof::derive_public_matrix_prefix::<F>(
        descriptor.num_field_elements,
        &descriptor.setup_seed,
    );
    let registry =
        akita_types::proof::SetupPrefixVerifierRegistry::new(descriptor.setup_seed.clone());
    let expanded =
        akita_types::AkitaExpandedSetup::from_verified_parts(descriptor, matrix).unwrap();
    let setup =
        akita_types::AkitaVerifierSetup::from_parts(std::sync::Arc::new(expanded), registry)
            .unwrap();
    let verifier = ImageVerifier::new(f.scheme.schedules().clone(), setup).unwrap();
    assert!(verify(
        &case.root,
        &case.output.committed_group,
        case.evaluation(),
        &verifier
    )
    .is_err());
}

#[test]
fn first_byte_of_every_native_inner_message_region_rejects_without_panic() {
    let case = Case::<BinaryField128>::new(0);
    akita_transcript::clear_thread_events();
    let (proof, _, _) = case.prove(PREFIX);
    // Native prover ranges are relative to its fresh inner argument. The
    // enclosing ClearChannel emits raw framing/payload without native context
    // records, so each inner range starts eight bytes later in the outer tape.
    let ranges = akita_transcript::thread_proof_ranges();
    assert!(!ranges.is_empty());
    let mut starts = Vec::new();
    for region in ranges {
        if region.len == 0 {
            continue;
        }
        assert!(region.start + region.len <= proof.len() - 8, "{region:?}");
        starts.push(region.start + 8);
    }
    starts.sort_unstable();
    starts.dedup();
    assert!(starts.len() > 1);
    eprintln!("native inner proof regions checked: {}", starts.len());
    for offset in starts {
        let mut malformed = proof.clone();
        malformed[offset] ^= 0x80;
        rejected(&case, &malformed);
    }
}
