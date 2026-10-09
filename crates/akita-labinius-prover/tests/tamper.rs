#![cfg(feature = "labinius")]

mod common;
use common::{data, setup, TestHost};

use akita_algebra::{
    binary::{
        field_switch::{transparent_weight, SwitchField},
        BinaryField128, BinaryField162 as B, BinaryField192,
    },
    MinusTrinomial, PlusTrinomial, Prime64Offset23703 as F, TrinomialRing,
};
use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
use akita_labinius_prover::{commit_binary_clear, prove_binary_clear, prove_binary_clear_bytes};
use akita_labinius_verifier::{
    codec::{decode_response_coefficient, encode_response_coefficient, response_width},
    profile::BinaryClearSetup,
    verify_binary_clear, verify_binary_clear_bytes,
};
use akita_params::sis::labinius::{LabiniusCoefficientPrime as P, LabiniusRingDegree as D};

fn proof_message_tampering<H: TestHost>() {
    let setup = setup::<F, 162, PlusTrinomial>(1, 1, 2);
    let (source, point, claim) = data::<H>(setup.source_len(), setup.num_vars());
    let commitment = commit_binary_clear::<H, F, 162, PlusTrinomial>(&setup, &source).unwrap();
    let proof = prove_binary_clear_bytes(&setup, &source, &commitment, &point, claim).unwrap();
    let source_width = if H::ROWS == 128 { 16 } else { 8 };
    let partial_end = H::ROWS * source_width;
    let round_end = partial_end + setup.num_vars() * 42;
    let terminal_end = round_end + 21;
    let u_end = terminal_end + setup.columns() * 21;
    let width = response_width(setup.lower(), setup.upper()).unwrap();
    assert_eq!(proof.len(), u_end + setup.scalar_rows() * 162 * width);
    let rejects = |bytes: &[u8]| {
        assert!(verify_binary_clear_bytes(&setup, &commitment, &point, claim, bytes).is_err());
    };
    // Every live partial, both coefficients in every round, t', and every U.
    let sites = (0..H::ROWS)
        .map(|row| row * source_width)
        .chain((partial_end..round_end).step_by(21))
        .chain(std::iter::once(round_end))
        .chain((terminal_end..u_end).step_by(21));
    for site in sites {
        let mut changed = proof.clone();
        changed[site] ^= 1;
        rejects(&changed);
    }
    // +2 preserves every coefficient's parity: rejection requires the prime
    // endpoint rather than relying on the binary endpoint or transcript draws.
    for index in 0..setup.scalar_rows() * 162 {
        let site = u_end + index * width;
        let current =
            decode_response_coefficient(&proof[site..site + width], setup.lower(), setup.upper())
                .unwrap();
        let mut changed = proof.clone();
        changed[site..site + width].copy_from_slice(
            &encode_response_coefficient(current + 2, setup.lower(), setup.upper()).unwrap(),
        );
        rejects(&changed);
    }
    // +1 also remains in range and necessarily changes the prime image of the
    // random nonzero matrix times this single monomial.
    let current =
        decode_response_coefficient(&proof[u_end..u_end + width], setup.lower(), setup.upper())
            .unwrap();
    let mut changed = proof.clone();
    changed[u_end..u_end + width].copy_from_slice(
        &encode_response_coefficient(current + 1, setup.lower(), setup.upper()).unwrap(),
    );
    rejects(&changed);
    assert_ne!(
        setup.matrix()[0]
            .schoolbook_mul(&TrinomialRing::one().unwrap())
            .unwrap(),
        TrinomialRing::zero().unwrap()
    );
    let mut out_of_range = proof.clone();
    out_of_range[u_end..u_end + width].copy_from_slice(&2049u16.to_le_bytes());
    rejects(&out_of_range);
    // Noncanonical high F162 bits at terminal and left-expansion boundaries.
    for site in [round_end + 20, terminal_end + 20] {
        let mut changed = proof.clone();
        changed[site] |= 0x80;
        rejects(&changed);
    }
    for length in [
        0,
        partial_end - 1,
        round_end - 1,
        terminal_end - 1,
        u_end - 1,
        proof.len() - 1,
    ] {
        rejects(&proof[..length]);
    }
    let mut trailing = proof.clone();
    trailing.push(0);
    rejects(&trailing);
}

#[test]
fn every_message_and_integer_coefficient_tamper_rejects_for_both_hosts() {
    proof_message_tampering::<BinaryField128>();
    proof_message_tampering::<BinaryField192>();
}

#[test]
fn public_statement_and_setup_identity_are_bound() {
    let setup = setup::<F, 162, PlusTrinomial>(1, 4, 2);
    let (source, point, claim) = data::<BinaryField128>(setup.source_len(), setup.num_vars());
    let commitment =
        commit_binary_clear::<BinaryField128, F, 162, PlusTrinomial>(&setup, &source).unwrap();
    let proof = prove_binary_clear_bytes(&setup, &source, &commitment, &point, claim).unwrap();
    let rejects = |s: &BinaryClearSetup<F, 162, PlusTrinomial>,
                   y: &akita_labinius_verifier::BinaryClearCommitment<F, 162, PlusTrinomial>,
                   r: &[BinaryField128],
                   t: BinaryField128| {
        assert!(verify_binary_clear_bytes(s, y, r, t, &proof).is_err());
    };
    for image in 0..commitment.images.len() {
        let mut y = commitment.clone();
        y.images[image] += TrinomialRing::one().unwrap();
        rejects(&setup, &y, &point, claim);
    }
    for axis in 0..point.len() {
        let mut r = point.clone();
        r[axis] += BinaryField128::ONE;
        rejects(&setup, &commitment, &r, claim);
    }
    rejects(&setup, &commitment, &point, claim + BinaryField128::ONE);
    let rebuild = |matrix, lower, upper, lambda, profile| {
        BinaryClearSetup::new(
            matrix,
            1,
            4,
            2,
            lower,
            upper,
            lambda,
            profile,
            P::P64Offset23703,
            D::D162,
        )
        .unwrap()
    };
    let mut matrix = setup.matrix().to_vec();
    matrix[0] += TrinomialRing::one().unwrap();
    let changed_a = rebuild(matrix, -1024, 1024, 128, setup.profile().clone());
    assert_ne!(setup.matrix_view_digest(), changed_a.matrix_view_digest());
    rejects(&changed_a, &commitment, &point, claim);
    for (lo, hi) in [(-1025, 1024), (-1024, 1025)] {
        rejects(
            &rebuild(
                setup.matrix().to_vec(),
                lo,
                hi,
                128,
                setup.profile().clone(),
            ),
            &commitment,
            &point,
            claim,
        );
    }
    rejects(
        &rebuild(
            setup.matrix().to_vec(),
            -1024,
            1024,
            127,
            setup.profile().clone(),
        ),
        &commitment,
        &point,
        claim,
    );
    let alternate =
        BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic243, 47).unwrap();
    rejects(
        &rebuild(setup.matrix().to_vec(), -1024, 1024, 128, alternate),
        &commitment,
        &point,
        claim,
    );
    let larger = common::setup::<F, 648, MinusTrinomial>(1, 1, 2);
    let large_y = akita_labinius_verifier::BinaryClearCommitment {
        images: commitment
            .images
            .iter()
            .map(|ring| {
                TrinomialRing::from_coefficients(std::array::from_fn(|i| {
                    ring.coefficients().get(i).copied().unwrap_or_default()
                }))
                .unwrap()
            })
            .collect(),
    };
    assert!(verify_binary_clear_bytes(&larger, &large_y, &point, claim, &proof).is_err());
    let other_prime = common::setup::<jolt_field::Prime128OffsetA7F7, 162, PlusTrinomial>(1, 4, 2);
    let other_y =
        commit_binary_clear::<BinaryField128, _, 162, PlusTrinomial>(&other_prime, &source)
            .unwrap();
    assert!(verify_binary_clear_bytes(&other_prime, &other_y, &point, claim, &proof).is_err());
    assert_ne!(
        setup.identity_bytes::<BinaryField128>().unwrap(),
        setup.identity_bytes::<BinaryField192>().unwrap()
    );
    let other_point = point
        .iter()
        .map(|p| {
            let words = p.to_words();
            BinaryField192::from_words([words[0], words[1], 0])
        })
        .collect::<Vec<_>>();
    let words = claim.to_words();
    assert!(verify_binary_clear_bytes(
        &setup,
        &commitment,
        &other_point,
        BinaryField192::from_words([words[0], words[1], 0]),
        &proof
    )
    .is_err());
}

#[test]
fn zero_transparent_weight_still_requires_the_original_source_opening() {
    use common::Tape;
    let setup = setup::<F, 162, PlusTrinomial>(1, 2, 2);
    let source = vec![1u128, 2, 4, 8];
    let point = vec![BinaryField128::ZERO; setup.num_vars()];
    let claim = BinaryField128::ONE;
    let commitment =
        commit_binary_clear::<BinaryField128, F, 162, PlusTrinomial>(&setup, &source).unwrap();
    let mut tape = Tape {
        zero_batch: BinaryField128::BATCH_BITS,
        ..Tape::default()
    };
    prove_binary_clear(&setup, &source, &commitment, &point, claim, &mut tape).unwrap();
    let batch = vec![B::ONE; BinaryField128::BATCH_BITS];
    let z = (0..setup.num_vars())
        .map(|i| Tape::draw_value(BinaryField128::BATCH_BITS + i, BinaryField128::BATCH_BITS))
        .collect::<Vec<_>>();
    assert_eq!(
        transparent_weight::<BinaryField128>(&point, &z, &batch).unwrap(),
        B::ZERO
    );
    verify_binary_clear(
        &setup,
        &commitment,
        &point,
        claim,
        &mut Tape::verifier(tape.messages.clone(), BinaryField128::BATCH_BITS),
    )
    .unwrap();
    let wrong_source = vec![1u128, 2, 4, 9];
    let wrong_commitment =
        commit_binary_clear::<BinaryField128, F, 162, PlusTrinomial>(&setup, &wrong_source)
            .unwrap();
    assert!(verify_binary_clear(
        &setup,
        &wrong_commitment,
        &point,
        claim,
        &mut Tape::verifier(tape.messages.clone(), BinaryField128::BATCH_BITS)
    )
    .is_err());
    // A forged terminal cannot be used to skip the left/source endpoint either.
    let terminal = BinaryField128::ROWS + 2 * setup.num_vars();
    let mut forged = tape.messages;
    forged[terminal][0] ^= 1;
    assert!(verify_binary_clear(
        &setup,
        &commitment,
        &point,
        claim,
        &mut Tape::verifier(forged, BinaryField128::BATCH_BITS)
    )
    .is_err());
}

fn multi_round_messages<H: TestHost>() {
    let setup = setup::<F, 162, PlusTrinomial>(1, 4, 4);
    let (source, point, claim) = data::<H>(setup.source_len(), setup.num_vars());
    let commitment = commit_binary_clear::<H, F, 162, PlusTrinomial>(&setup, &source).unwrap();
    let proof = prove_binary_clear_bytes(&setup, &source, &commitment, &point, claim).unwrap();
    let partial_end = H::ROWS * std::mem::size_of::<H::Source>();
    let round_end = partial_end + setup.num_vars() * 42;
    assert_eq!(setup.num_vars(), 4);
    for site in (partial_end..round_end).step_by(21) {
        let mut changed = proof.clone();
        changed[site] ^= 1;
        assert!(verify_binary_clear_bytes(&setup, &commitment, &point, claim, &changed).is_err());
        let mut noncanonical = proof.clone();
        noncanonical[site + 20] |= 0x80;
        assert!(
            verify_binary_clear_bytes(&setup, &commitment, &point, claim, &noncanonical).is_err()
        );
    }
    let terminal_end = round_end + 21;
    for site in (terminal_end..terminal_end + setup.columns() * 21).step_by(21) {
        let mut changed = proof.clone();
        changed[site] ^= 1;
        assert!(verify_binary_clear_bytes(&setup, &commitment, &point, claim, &changed).is_err());
    }
    // Each scalar response row participates, including rows after the first.
    let response_start = terminal_end + setup.columns() * 21;
    let width = response_width(setup.lower(), setup.upper()).unwrap();
    for row in 0..setup.scalar_rows() {
        let site = response_start + row * 162 * width;
        let old =
            decode_response_coefficient(&proof[site..site + width], setup.lower(), setup.upper())
                .unwrap();
        let mut changed = proof.clone();
        changed[site..site + width].copy_from_slice(
            &encode_response_coefficient(old + 2, setup.lower(), setup.upper()).unwrap(),
        );
        assert!(verify_binary_clear_bytes(&setup, &commitment, &point, claim, &changed).is_err());
    }
}

#[test]
fn every_round_of_four_round_proofs_and_each_response_row_are_checked() {
    multi_round_messages::<BinaryField128>();
    multi_round_messages::<BinaryField192>();
}
