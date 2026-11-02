//! Projection wire mutations and the two transcript fixation boundaries.
use super::*;
use akita_transcript::ProofChannel;
use akita_types::binary_source::{
    binary_projection_values, decode_binary_projection, BinaryProjectionBounds,
    BinaryProjectionRows, BINARY_SOURCE_Q,
};

pub(super) fn mutate_values(
    proof: &[u8],
    successor_coefficients: usize,
    word: bool,
    polynomials: usize,
    rejects: impl Fn(&[u8]),
) {
    // A word claim contributes three frontend rounds. The 128-field D payload,
    // response nonce, and scheduled outgoing payload are present in every case.
    let nonce_start = usize::from(word) * (128 * 16 + 3 * 42 + 21) + 128 * 16;
    let nonce_len = proof[nonce_start..]
        .iter()
        .position(|byte| byte & 0x80 == 0)
        .unwrap()
        + 1;
    let outgoing = nonce_start + nonce_len;
    let start = outgoing + successor_coefficients * 16;
    let bounds = BinaryProjectionBounds::new(1, 2, 3, polynomials, u128::MAX - 274).unwrap();
    let signed_bytes = |value: i128| {
        let mut bytes = value.unsigned_abs().to_le_bytes()[..12].to_vec();
        if value < 0 {
            bytes[11] |= 0x80;
        }
        bytes
    };
    for row in 0..7 {
        let range = start + row * 12..start + (row + 1) * 12;
        let value = decode_binary_projection(&proof[range.clone()], &bounds).unwrap();
        assert_ne!(
            value, 0,
            "sign-flip coverage needs a nonzero fixture output"
        );
        for (mutation, bytes) in [
            ("plus one", signed_bytes(value + 1)),
            ("plus q", signed_bytes(value + i128::from(BINARY_SOURCE_Q))),
            ("sign flip", signed_bytes(-value)),
            (
                "out of bounds",
                signed_bytes(
                    ((bounds.projection_bound() / u128::from(BINARY_SOURCE_Q) + 1)
                        * u128::from(BINARY_SOURCE_Q)) as i128,
                ),
            ),
            (
                "negative zero",
                signed_bytes(0)
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| if i == 11 { 0x80 } else { x })
                    .collect(),
            ),
        ] {
            if mutation == "plus q" || mutation == "sign flip" {
                assert!(
                    decode_binary_projection(&bytes, &bounds).is_ok(),
                    "this mutation must pass wire validation and test authentication"
                );
            }
            let mut altered = proof.to_vec();
            altered[range.clone()].copy_from_slice(&bytes);
            eprintln!("projection row {row}: {mutation}");
            rejects(&altered);
        }
        let mut truncated = proof.to_vec();
        truncated.remove(range.start);
        rejects(&truncated);
        let mut extra = proof.to_vec();
        extra.insert(range.end, 0);
        rejects(&extra);
        rejects(&proof[..range.end - 1]);
    }
    // The outgoing commitment and accepted response nonce also precede mu.
    for offset in [nonce_start, outgoing, start - 1] {
        let mut altered = proof.to_vec();
        altered[offset] ^= 1;
        rejects(&altered);
    }
}

#[test]
fn projection_challenges_bind_commitment_and_every_output() {
    let bounds = BinaryProjectionBounds::new(1, 2, 3, 1, u128::MAX - 274).unwrap();
    let run = |commitment: u8, values: [i128; 7]| {
        let mut prover = new_prover_channel(b"binary-projection-fixation", b"caller").unwrap();
        // Stand in for the outgoing commitment atom; test the production sampler
        // and value exchanger, on both channel roles, at this exact boundary.
        prover.prover_message(&[commitment; 16]);
        let rows = BinaryProjectionRows::draw(&mut prover, &bounds).unwrap();
        binary_projection_values(&mut prover, &bounds, &values).unwrap();
        let mu: [F; 7] = std::array::from_fn(|_| prover.field_challenge().unwrap());
        let proof = prover.narg_string();
        let mut verifier =
            new_verifier_channel(b"binary-projection-fixation", b"caller", proof).unwrap();
        assert_eq!(
            verifier.prover_message::<[u8; 16]>().unwrap(),
            [commitment; 16]
        );
        assert_eq!(
            BinaryProjectionRows::draw(&mut verifier, &bounds).unwrap(),
            rows
        );
        assert_eq!(
            binary_projection_values(&mut verifier, &bounds, &[0; 7]).unwrap(),
            values
        );
        let replayed: [F; 7] = std::array::from_fn(|_| verifier.field_challenge().unwrap());
        assert_eq!(mu, replayed);
        verifier.check_eof().unwrap();
        (rows, mu)
    };
    let baseline = run(0, [0; 7]);
    assert_ne!(baseline.0, run(1, [0; 7]).0);
    for index in 0..7 {
        let mut values = [0; 7];
        values[index] = i128::from(BINARY_SOURCE_Q);
        let changed = run(0, values);
        assert_eq!(baseline.0, changed.0, "outputs must not select J");
        assert_ne!(baseline.1, changed.1, "each y must bind every output batch");
        for (a, b) in baseline.1.iter().zip(changed.1) {
            assert_ne!(*a, b);
        }
    }
}

#[test]
fn projection_orchestrators_fix_witness_then_rows_then_outputs_then_batches() {
    // Complement channel sensitivity with a guard on both actual drivers. This
    // catches paired reorderings too, even if prover and verifier still agree.
    for (source, commitment) in [
        (
            include_str!("../../../akita-prover/src/protocol/prove/binary_root.rs"),
            "let committed =",
        ),
        (
            include_str!("../../../akita-verifier/src/fold/binary_root.rs"),
            "let next_witness =",
        ),
    ] {
        let fixed = source.find(commitment).unwrap();
        let rows = source.find("let projections =").unwrap();
        let values = source.find("let projected_values =").unwrap();
        let batches = source.find("let batches =").unwrap();
        assert!(
            fixed < rows && rows < values && values < batches,
            "projection fixation order must be commitment, J, all y, mu"
        );
    }
}
