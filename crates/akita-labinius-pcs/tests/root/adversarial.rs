use akita_algebra::binary::{BinaryField128, BinaryField192};
use akita_error::AkitaError;
use akita_labinius_pcs::{
    config::{Digits1, Digits2, Digits4},
    RootPcsProver, RootPcsVerifier, F,
};
use akita_types::{
    proof::AkitaSetupSeed, AkitaExpandedSetup, AkitaVerifierSetup, Commitment, CommittedGroup,
    RingVec,
};
use jolt_field::Zero;
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use super::support::{Case, Host, TestDigits};

fn native_replay_rejected_without_internal(case: &Case<Digits2>, proof: &[u8]) {
    let result = catch_unwind(AssertUnwindSafe(|| case.verify(proof)));
    assert!(result.is_ok(), "verifier panicked on {} bytes", proof.len());
    let error = result.unwrap().unwrap_err();
    assert!(
        !matches!(error, AkitaError::Internal(_)),
        "internal error: {error}"
    );
}

fn invalid_proof(case: &Case<Digits2>, proof: &[u8]) {
    let result = catch_unwind(AssertUnwindSafe(|| case.verify(proof)));
    assert!(
        matches!(result, Ok(Err(AkitaError::InvalidProof))),
        "expected InvalidProof on {} bytes: {result:?}",
        proof.len()
    );
}

#[test]
fn reduction_bit_flips_are_invalid_proof_and_native_replay_errors_are_not_internal() {
    let case = Case::<Digits2>::new(0);
    let proof = case.prove();
    for (name, start, len) in case.regions(&proof) {
        assert!(len > 0);
        let mut changed = proof.clone();
        changed[start] ^= 1;
        if name == "inner proof" {
            native_replay_rejected_without_internal(&case, &changed);
        } else {
            invalid_proof(&case, &changed);
        }
    }
}

#[test]
fn adapter_framing_w_decoding_truncation_and_outer_eof_are_invalid_proof() {
    let case = Case::<Digits2>::new(0);
    let proof = case.prove();
    let regions = case.regions(&proof);
    let mut boundaries = vec![0, proof.len() - 1];
    for &(name, start, len) in &regions {
        boundaries.extend([start, start + len - 1]);
        if name.ends_with("length") {
            boundaries.extend(start..=start + 8);
            let payload = u64::from_le_bytes(proof[start..start + 8].try_into().unwrap());
            for changed_len in [0, 1, payload - 1, payload + 1, u64::MAX] {
                let mut changed = proof.clone();
                changed[start..start + 8].copy_from_slice(&changed_len.to_le_bytes());
                if name == "inner length" && (changed_len == 1 || changed_len == payload - 1) {
                    // These lengths fit the adapter bound and reach native replay.
                    native_replay_rejected_without_internal(&case, &changed);
                } else {
                    invalid_proof(&case, &changed);
                }
            }
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    for end in boundaries {
        invalid_proof(&case, &proof[..end]);
    }
    let (.., start, len) = *regions
        .iter()
        .find(|(name, _, _)| *name == "W commitment")
        .unwrap();
    let mut noncanonical = proof.clone();
    noncanonical[start + len - 16..start + len].fill(0xff);
    invalid_proof(&case, &noncanonical);
    let mut trailing = proof.clone();
    trailing.push(0);
    invalid_proof(&case, &trailing);
    let mut word = 0x518a_2341_bca0_321fu64;
    for _ in 0..8 {
        let mut random = vec![0u8; proof.len()];
        for byte in &mut random {
            word ^= word << 13;
            word ^= word >> 7;
            word ^= word << 17;
            *byte = word as u8;
        }
        invalid_proof(&case, &random);
    }
}

#[test]
fn root_akita_seed_commitment_digit_config_point_value_and_host_are_bound() {
    let case = Case::<Digits2>::new(0);
    let proof = case.prove();
    let fixture = Digits2::fixture(0);
    let other_root = fixture.verifier(crate::common::admitted(0, 0x32));
    assert!(matches!(
        other_root.verify::<BinaryField128>(
            &case.output.committed_group,
            &case.point,
            case.value,
            &proof
        ),
        Err(AkitaError::InvalidProof)
    ));
    let mut source = case.source.clone();
    source[0] ^= 1;
    let other = case.prover.commit::<BinaryField128>(&source).unwrap();
    assert!(matches!(
        case.verifier.verify::<BinaryField128>(
            &other.committed_group,
            &case.point,
            case.value,
            &proof
        ),
        Err(AkitaError::InvalidProof)
    ));
    let mut point = case.point.clone();
    point[0] += BinaryField128::ONE;
    assert!(matches!(
        case.verifier.verify::<BinaryField128>(
            &case.output.committed_group,
            &point,
            case.value,
            &proof
        ),
        Err(AkitaError::InvalidProof)
    ));
    assert!(matches!(
        case.verifier.verify::<BinaryField128>(
            &case.output.committed_group,
            &case.point,
            case.value + BinaryField128::ONE,
            &proof
        ),
        Err(AkitaError::InvalidProof)
    ));
    let other_point = BinaryField192::point(case.point.len());
    assert!(matches!(
        case.verifier.verify::<BinaryField192>(
            &case.output.committed_group,
            &other_point,
            BinaryField192::ZERO,
            &proof
        ),
        Err(AkitaError::InvalidProof)
    ));
    for verifier in [
        Digits1::fixture(0)
            .verifier(case.root.clone())
            .verify::<BinaryField128>(
                &case.output.committed_group,
                &case.point,
                case.value,
                &proof,
            ),
        Digits4::fixture(0)
            .verifier(case.root.clone())
            .verify::<BinaryField128>(
                &case.output.committed_group,
                &case.point,
                case.value,
                &proof,
            ),
    ] {
        assert!(matches!(verifier, Err(AkitaError::InvalidProof)));
    }
    let mut descriptor = fixture.verifier_setup.expanded().descriptor().clone();
    descriptor.setup_seed = AkitaSetupSeed::shake256_paged_v1([0x77; 32]);
    let matrix = akita_types::proof::derive_public_matrix_prefix::<F>(
        descriptor.num_field_elements,
        &descriptor.setup_seed,
    );
    let registry =
        akita_types::proof::SetupPrefixVerifierRegistry::new(descriptor.setup_seed.clone());
    let expanded = AkitaExpandedSetup::from_verified_parts(descriptor, matrix).unwrap();
    let setup = AkitaVerifierSetup::from_parts(Arc::new(expanded), registry).unwrap();
    let verifier = RootPcsVerifier::<Digits2>::new(
        case.root.clone(),
        fixture.images.clone(),
        fixture.digits.clone(),
        setup,
    )
    .unwrap();
    assert!(matches!(
        verifier.verify::<BinaryField128>(
            &case.output.committed_group,
            &case.point,
            case.value,
            &proof
        ),
        Err(AkitaError::InvalidProof)
    ));
}

#[test]
fn malformed_public_commitment_and_missing_rows_are_rejected_before_replay() {
    let case = Case::<Digits2>::new(0);
    let empty = CommittedGroup::new(
        *case.output.committed_group.profile(),
        Commitment::new(RingVec::from_coeffs(Vec::<F>::new())),
    );
    let mut altered = case.output.committed_group.clone();
    altered.profile.group =
        akita_params::PolynomialGroupLayout::singleton(case.layout.image_log_len() + 1);
    for commitment in [&empty, &altered] {
        assert!(matches!(
            catch_unwind(AssertUnwindSafe(|| case.verifier.oracle(commitment))),
            Ok(Err(AkitaError::InvalidProof))
        ));
    }
    let larger = Digits2::fixture(1);
    assert!(matches!(
        RootPcsVerifier::<Digits2>::new(
            case.root.clone(),
            larger.images.clone(),
            larger.digits.clone(),
            larger.verifier_setup.clone()
        ),
        Err(AkitaError::UnsupportedSchedule(_))
    ));
    assert!(matches!(
        case.prover.open::<BinaryField128>(
            &case.source[..case.source.len() - 1],
            &case.output,
            &case.point,
            case.value
        ),
        Err(AkitaError::InvalidSize { .. })
    ));
    assert!(matches!(
        case.prover.open::<BinaryField128>(
            &case.source,
            &case.output,
            &case.point,
            case.value + BinaryField128::ONE
        ),
        Err(AkitaError::InvalidInput(_))
    ));
    let malformed = CommittedGroup::new(
        *case.output.committed_group.profile(),
        Commitment::new(RingVec::from_coeffs(vec![F::zero()])),
    );
    assert!(matches!(
        case.verifier.oracle(&malformed),
        Err(AkitaError::InvalidProof)
    ));
}

#[test]
fn insufficient_shared_setup_and_foreign_backend_handle_reject_before_transcript_use() {
    let case = Case::<Digits2>::new(0);
    let fixture = Digits2::fixture(0);
    let descriptor = fixture.prover_setup.expanded.descriptor();
    let capacity = akita_params::SetupMatrixCapacity {
        num_field_elements: 1,
    };
    let undersized = akita_cpu_backend::AkitaProverSetup::<F>::generate_with_capacity(
        descriptor.max_num_vars,
        descriptor.max_num_batched_polys,
        capacity,
    )
    .unwrap();
    assert!(matches!(
        RootPcsProver::<Digits2>::new(
            case.root.clone(),
            fixture.images.clone(),
            fixture.digits.clone(),
            undersized.clone(),
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
    let verifier_setup = undersized.to_verifier_setup(capacity).unwrap();
    assert!(matches!(
        RootPcsVerifier::<Digits2>::new(
            case.root.clone(),
            fixture.images.clone(),
            fixture.digits.clone(),
            verifier_setup,
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
    let foreign = fixture.prover(case.root.clone());
    let output = foreign.commit::<BinaryField128>(&case.source).unwrap();
    assert_eq!(output.committed_group, case.output.committed_group);
    assert!(matches!(
        case.prover.oracle(&output),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        foreign.oracle(&case.output),
        Err(AkitaError::InvalidInput(_))
    ));
}
