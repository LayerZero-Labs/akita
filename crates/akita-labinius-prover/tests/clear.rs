#![cfg(feature = "labinius")]

mod common;

use akita_algebra::{
    binary::{field_switch::SwitchField, BinaryField128, BinaryField192},
    MinusTrinomial, PlusTrinomial, TrinomialModulus,
};
use akita_error::AkitaError;
use akita_labinius_prover::{commit_binary_clear, prove_binary_clear_bytes};
use akita_labinius_verifier::{verify_binary_clear_bytes, BinaryClearSetup};
use akita_params::sis::labinius::{LabiniusCommitmentModulus, LabiniusRingDegree};
use common::{admitted, clear_setup, clear_setup_with, data, TestHost, Q};

fn roundtrip<H: TestHost, const D: usize, M: TrinomialModulus>(setup: &BinaryClearSetup<D, M>) {
    let (source, point, claim) = data::<H>(setup.source_len(), setup.num_vars());
    let commitment = commit_binary_clear::<H, D, M>(setup, &source).unwrap();
    let proof = prove_binary_clear_bytes(setup, &source, &commitment, &point, claim).unwrap();
    verify_binary_clear_bytes(setup, &commitment, &point, claim, &proof).unwrap();
}

#[test]
fn honest_openings_verify_for_both_hosts_and_every_geometry() {
    for n_a in [1, 2] {
        // Degree 162 packs one scalar per element and uses the integer product;
        // degree 648 packs four and uses the limb transform.
        roundtrip::<BinaryField128, 162, PlusTrinomial>(&clear_setup(n_a, 4, 4));
        roundtrip::<BinaryField192, 162, PlusTrinomial>(&clear_setup(n_a, 4, 4));
        roundtrip::<BinaryField128, 648, MinusTrinomial>(&clear_setup(n_a, 2, 8));
        roundtrip::<BinaryField192, 648, MinusTrinomial>(&clear_setup(n_a, 2, 8));
    }
    roundtrip::<BinaryField128, 324, MinusTrinomial>(&clear_setup(1, 2, 8));
    roundtrip::<BinaryField192, 648, MinusTrinomial>(&clear_setup(1, 2, 4));
    for log_fold_width in [1, 0] {
        let admitted = admitted(log_fold_width, 0x31);
        roundtrip::<BinaryField128, 648, MinusTrinomial>(admitted.setup());
        roundtrip::<BinaryField192, 648, MinusTrinomial>(admitted.setup());
    }
}

#[test]
fn tampered_openings_and_statements_reject() {
    type H = BinaryField128;
    let setup = clear_setup::<648, MinusTrinomial>(2, 2, 8);
    let (source, point, claim) = data::<H>(setup.source_len(), setup.num_vars());
    let commitment = commit_binary_clear::<H, 648, MinusTrinomial>(&setup, &source).unwrap();
    let proof = prove_binary_clear_bytes(&setup, &source, &commitment, &point, claim).unwrap();
    let rejects = |name: &str, result: Result<(), AkitaError>| {
        assert!(result.is_err(), "accepted {name}");
    };

    // Proof bytes: a flipped bit at positions spread over every message, and
    // both length changes.
    let stride = (proof.len() / 127).max(1);
    for position in (0..proof.len()).step_by(stride).chain([proof.len() - 1]) {
        let mut changed = proof.clone();
        changed[position] ^= 1;
        rejects(
            &format!("flipped byte {position}"),
            verify_binary_clear_bytes(&setup, &commitment, &point, claim, &changed),
        );
    }
    let mut extended = proof.clone();
    extended.push(0);
    for (name, bytes) in [
        ("truncated proof", &proof[..proof.len() - 1]),
        ("extended proof", &extended[..]),
        ("empty proof", &[][..]),
    ] {
        rejects(
            name,
            verify_binary_clear_bytes(&setup, &commitment, &point, claim, bytes),
        );
    }

    // Statement: each public input is bound.
    let mut wrong_point = point.clone();
    *wrong_point.last_mut().unwrap() += H::ONE;
    let mut wrong_image = commitment.clone();
    *wrong_image.images.last_mut().unwrap() ^= 1;
    // The same residue class, but not the canonical representative.
    let mut noncanonical_image = commitment.clone();
    noncanonical_image.images[0] += Q;
    let mut short_image = commitment.clone();
    short_image.images.pop();
    let mut other_matrix = setup.matrix().to_vec();
    other_matrix[0] ^= 1;
    let other_setup = clear_setup_with::<648, MinusTrinomial>(other_matrix, 2, 2, 8).unwrap();
    for (name, setup, commitment, point, claim) in [
        ("claim", &setup, &commitment, &point, claim + H::ONE),
        ("point", &setup, &commitment, &wrong_point, claim),
        ("image", &setup, &wrong_image, &point, claim),
        (
            "noncanonical image",
            &setup,
            &noncanonical_image,
            &point,
            claim,
        ),
        ("short image", &setup, &short_image, &point, claim),
        ("setup matrix", &other_setup, &commitment, &point, claim),
    ] {
        rejects(
            name,
            verify_binary_clear_bytes(setup, commitment, point, claim, &proof),
        );
    }
    rejects(
        "short point",
        verify_binary_clear_bytes(&setup, &commitment, &point[1..], claim, &proof),
    );

    // The prover refuses a false claim and a matrix that is not reduced.
    assert!(
        prove_binary_clear_bytes(&setup, &source, &commitment, &point, claim + H::ONE).is_err()
    );
    let mut unreduced = setup.matrix().to_vec();
    unreduced[0] = Q;
    assert!(clear_setup_with::<648, MinusTrinomial>(unreduced, 2, 2, 8).is_err());
}

/// The fold-response nonce search. An interval tighter than the first
/// candidate's response forces a rejection, the verifier replays the accepted
/// nonce and no other, and the search ends with its domain.
#[test]
fn fold_response_nonce_is_searched_replayed_and_bounded() {
    type H = BinaryField128;
    fn within<const D: usize, M: TrinomialModulus>(
        setup: &BinaryClearSetup<D, M>,
        bound: i64,
    ) -> BinaryClearSetup<D, M> {
        BinaryClearSetup::new(
            setup.matrix().to_vec(),
            setup.n_a(),
            setup.m(),
            setup.columns(),
            -bound,
            bound,
            128,
            setup.profile().clone(),
            if D == 648 {
                LabiniusRingDegree::D648
            } else {
                LabiniusRingDegree::D162
            },
            LabiniusCommitmentModulus::Q25Plus14561,
        )
        .unwrap()
    }
    let wide = clear_setup::<648, MinusTrinomial>(1, 2, 8);
    let (source, point, claim) = data::<H>(wide.source_len(), wide.num_vars());
    let commitment = commit_binary_clear::<H, 648, MinusTrinomial>(&wide, &source).unwrap();
    // The nonce follows the frontend and the left expansion.
    let nonce_at = H::ROWS * 16 + (2 * point.len() + 1) * 21 + wide.columns() * 21;
    // Responses of this source reach about 40. Tighten the interval until the
    // prover rejects its first candidate: the search is sequential, so the
    // nonce counts the rejections.
    let (setup, proof) = (32..=44)
        .rev()
        .map(|bound| {
            let setup = within(&wide, bound);
            let proof =
                prove_binary_clear_bytes(&setup, &source, &commitment, &point, claim).unwrap();
            (setup, proof)
        })
        .find(|(_, proof)| (1..0x80).contains(&proof[nonce_at]))
        .expect("no interval forced a rejection");
    verify_binary_clear_bytes(&setup, &commitment, &point, claim, &proof).unwrap();
    let nonce = proof[nonce_at];
    for (name, replacement) in [
        ("a rejected candidate", vec![nonce - 1]),
        ("a nonce outside the search domain", vec![0x80, 0x20]),
        ("a noncanonical nonce", vec![0x80 | nonce, 0]),
    ] {
        let mut changed = proof.clone();
        changed.splice(nonce_at..=nonce_at, replacement);
        assert!(
            matches!(
                verify_binary_clear_bytes(&setup, &commitment, &point, claim, &changed),
                Err(AkitaError::InvalidProof)
            ),
            "accepted {name}"
        );
    }
    // No nonzero response fits the zero interval.
    let setup = within(&clear_setup::<162, PlusTrinomial>(1, 1, 1), 0);
    let (source, point, claim) = data::<H>(setup.source_len(), setup.num_vars());
    let commitment = commit_binary_clear::<H, 162, PlusTrinomial>(&setup, &source).unwrap();
    assert!(matches!(
        prove_binary_clear_bytes(&setup, &source, &commitment, &point, claim),
        Err(AkitaError::InvalidInput(_))
    ));
}
