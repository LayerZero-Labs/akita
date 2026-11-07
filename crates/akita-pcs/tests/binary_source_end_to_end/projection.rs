//! Projection wire mutations and the two transcript fixation boundaries.
use super::*;
use akita_transcript::{ext_challenge, ProofChannel, ProtocolSiteId};
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

const MU_SITE: ProtocolSiteId = ProtocolSiteId {
    family: 0x4d55_5445,
    invocation: 0,
    level: 0,
    stage: 0,
    round: 0,
    group: 0,
    limb: 0,
    detail: 0,
};

#[test]
fn projection_challenges_bind_commitment_and_every_output() {
    let bounds = BinaryProjectionBounds::new(1, 2, 3, 1, u128::MAX - 274).unwrap();
    let run = |commitment: u8, values: [i128; 7]| {
        let mut prover = new_prover_channel(b"binary-projection-fixation");
        // Stand in for the outgoing commitment atom; test the production sampler
        // and value exchanger, on both channel roles, at this exact boundary.
        prover.exchange(&mut [commitment; 16]).unwrap();
        let rows = BinaryProjectionRows::draw(&mut prover, &bounds).unwrap();
        binary_projection_values(&mut prover, &bounds, &values).unwrap();
        let mu: [F; 7] = std::array::from_fn(|_| ext_challenge(&mut prover, MU_SITE).unwrap());
        let proof = prover.narg_string();
        let mut verifier = new_verifier_channel(b"binary-projection-fixation", proof);
        let mut atom = [0; 16];
        verifier.exchange(&mut atom).unwrap();
        assert_eq!(atom, [commitment; 16]);
        assert_eq!(
            BinaryProjectionRows::draw(&mut verifier, &bounds).unwrap(),
            rows
        );
        assert_eq!(
            binary_projection_values(&mut verifier, &bounds, &[0; 7]).unwrap(),
            values
        );
        let replayed: [F; 7] =
            std::array::from_fn(|_| ext_challenge(&mut verifier, MU_SITE).unwrap());
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

/// Only the scheduled Tau1 site is consumed at this isolated order-seven
/// boundary. Complete PCS admission remains gated until the typed prefix.
struct BatchReplay<S> {
    state: S,
    queries: usize,
    poisoned: bool,
}
impl<S: ProofChannel> akita_types::GrindingReplay for BatchReplay<S> {
    type State = S;
    fn state_mut(&mut self) -> &mut S {
        &mut self.state
    }
    fn poison(&mut self) {
        self.poisoned = true;
    }
    fn grind_query(&mut self, site: akita_params::GrindingSite) -> Result<(), AkitaError> {
        assert_eq!(site, akita_params::GrindingSite::Tau1Point { level: 0 });
        self.queries += 1;
        Ok(())
    }
}

#[test]
fn p64_dense_batches_replay_both_coordinates_after_the_outgoing_commitment() {
    use akita_params::binary_setup::BinaryFieldProfile;
    use akita_types::binary_source::{binary_root_p64_batch_challenges, BinaryRootWitnessPlan};
    use jolt_algebra::{Ext2, ExtField, Prime64Offset59};
    type Base = Prime64Offset59;
    type Extension = Ext2<Base>;
    let index = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
    let profiles =
        BinaryCommittedGroupManifest::profiles_for_field(&index, 1, 0, BinaryFieldProfile::P64)
            .unwrap();
    let descriptor = BinarySourceDescriptor::new(index, vec![[1; 32]], profiles).unwrap();
    let plan =
        BinaryRootWitnessPlan::for_profile(&descriptor, 3, 64, BinaryFieldProfile::P64, true, 0)
            .unwrap();
    // Fixture: two word rows, rank-two B and rank-one word-only D.
    let count = plan.coefficient_batch_count().unwrap() + 2 + 2 + 1;
    let run = |commitment: u8| {
        let mut prover = BatchReplay {
            state: new_prover_channel(b"binary-p64-batch-fixation"),
            queries: 0,
            poisoned: false,
        };
        prover.state.exchange(&mut [commitment; 16]).unwrap();
        let batch =
            binary_root_p64_batch_challenges::<Base, Extension, _>(&mut prover, count).unwrap();
        assert_eq!(batch.len(), count);
        assert_eq!(prover.queries, 1);
        assert!(!prover.poisoned);
        let proof = prover.state.narg_string();
        // No J rows or y messages: the isolated boundary sends only the
        // order-five commitment stand-in and draws its independent weights.
        assert_eq!(proof.len(), 16);
        let mut verifier = BatchReplay {
            state: new_verifier_channel(b"binary-p64-batch-fixation", proof),
            queries: 0,
            poisoned: false,
        };
        let mut received = [0; 16];
        verifier.state.exchange(&mut received).unwrap();
        assert_eq!(received, [commitment; 16]);
        assert_eq!(
            binary_root_p64_batch_challenges::<Base, Extension, _>(&mut verifier, count).unwrap(),
            batch
        );
        verifier.state.check_eof().unwrap();
        assert_eq!(verifier.queries, 1);
        assert!(!verifier.poisoned);
        assert!(batch
            .iter()
            .all(|value| value.base_coefficient(1) != Base::zero()));
        batch
    };
    let baseline = run(0);
    let changed = run(1);
    assert!(baseline.iter().zip(changed).all(|(a, b)| *a != b));
    assert!(BinaryFieldProfile::P64
        .admit_root(BinaryFieldProfile::P64.modulus(), 2)
        .is_err());
}
