//! Canonical frontend differential, byte mutations, and mandatory terminals.

use super::*;
use akita_prover::binary_field_switch::prove_field_switch;
use akita_types::binary_source::{BinaryCommitmentProfiles, BinaryTerminalClaim};
use akita_verifier::binary_field_switch::verify_field_switch;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[path = "zero_weight.rs"]
mod zero_weight;

const DOMAIN: &[u8] = b"binary-field-switch-acceptance";

fn descriptor(width: usize, carriers: usize, columns: usize) -> BinarySourceDescriptor {
    BinarySourceDescriptor::new(
        BinaryIndexMap::new(width, carriers * (128 / width), carriers, columns).unwrap(),
        vec![[29; 32], [30; 32]],
        BinaryCommitmentProfiles {
            a: [1; 32],
            b: [2; 32],
        },
    )
    .unwrap()
}

fn frontend_proof(
    descriptor: &BinarySourceDescriptor,
    claim: &BinaryWordClaim,
    carriers: &[u128],
) -> (Vec<u8>, BinaryTerminalClaim) {
    let mut prover = new_prover_channel(DOMAIN, b"caller").unwrap();
    prover.public_message(&descriptor.canonical_bytes());
    prover.public_message(&claim.canonical_bytes());
    let terminal = prove_field_switch(descriptor, claim, carriers, &mut prover).unwrap();
    (prover.narg_string().to_vec(), terminal)
}

fn check(
    descriptor: &BinarySourceDescriptor,
    claim: &BinaryWordClaim,
    proof: &[u8],
) -> Result<BinaryTerminalClaim, AkitaError> {
    let mut verifier = new_verifier_channel(DOMAIN, b"caller", proof).unwrap();
    verifier.public_message(&descriptor.canonical_bytes());
    verifier.public_message(&claim.canonical_bytes());
    let terminal = verify_field_switch(descriptor, claim, &mut verifier)?;
    verifier.check_eof()?;
    Ok(terminal)
}

fn rejects(descriptor: &BinarySourceDescriptor, claim: &BinaryWordClaim, proof: &[u8]) {
    let outcome = catch_unwind(AssertUnwindSafe(|| check(descriptor, claim, proof)));
    assert!(outcome.is_ok(), "field-switch verifier panicked");
    assert!(outcome.unwrap().is_err(), "malformed field switch accepted");
}

#[test]
fn field_switch_differential_logical_words_and_carrier_terminals() {
    let mut rng = StdRng::seed_from_u64(0x83b2_54a9);
    for width in [1, 8, 32, 128] {
        for (carriers, columns) in [(4, 1), (8, 2), (16, 4), (32, 2)] {
            let descriptor = descriptor(width, carriers, columns);
            for _ in 0..3 {
                let source = (0..carriers).map(|_| rng.gen::<u128>()).collect::<Vec<_>>();
                let split = descriptor.index_map().claim_variable_split();
                let point = (0..split.intra_carrier + split.carrier_address)
                    .map(|_| reference::host(rng.gen()))
                    .collect::<Vec<_>>();
                let claim = reference::claim(&descriptor, &source, &point);
                let (proof, terminal) = frontend_proof(&descriptor, &claim, &source);
                assert_eq!(proof.len(), 128 * 16 + split.carrier_address * 42 + 21);
                assert_eq!(check(&descriptor, &claim, &proof).unwrap(), terminal);

                // Replay the wire independently. This observes λ only after
                // absorbing every partial; removing their transcript binding
                // makes the expected sumcheck point differ from the frontend.
                let mut wire = new_verifier_channel(DOMAIN, b"caller", &proof).unwrap();
                wire.public_message(&descriptor.canonical_bytes());
                wire.public_message(&claim.canonical_bytes());
                for _ in 0..128 {
                    wire.prover_message::<[u8; 16]>().unwrap();
                }
                for _ in 0..7 {
                    wire.verifier_message::<[u8; 32]>().unwrap();
                }
                let mut z = Vec::new();
                for _ in 0..split.carrier_address {
                    wire.prover_message::<[u8; 21]>().unwrap();
                    wire.prover_message::<[u8; 21]>().unwrap();
                    let block = wire.verifier_message::<[u8; 32]>().unwrap();
                    let mut bytes = [0u8; 21];
                    bytes.copy_from_slice(&block[..21]);
                    bytes[20] &= 3;
                    z.push(BinaryField162::from_bytes(&bytes).unwrap());
                }
                let c = columns.trailing_zeros() as usize;
                assert_eq!(
                    terminal.rho_col(),
                    &z[..c].iter().map(|v| v.to_bytes()).collect::<Vec<_>>()
                );
                assert_eq!(
                    terminal.rho_row(),
                    &z[c..].iter().map(|v| v.to_bytes()).collect::<Vec<_>>()
                );
                assert_eq!(
                    *terminal.value(),
                    reference::carrier_value(&source, &z).to_bytes()
                );
                assert_eq!(
                    wire.prover_message::<[u8; 21]>().unwrap(),
                    *terminal.value()
                );
                wire.check_eof().unwrap();
            }
        }
    }
}

#[test]
fn field_switch_rejects_each_partial_and_round_message() {
    let descriptor = descriptor(8, 8, 2);
    let source = (0..8)
        .map(|i| 0x1234_5678_9abc_def0_9876_5432_10ff_aa55u128.rotate_left(i * 7))
        .collect::<Vec<_>>();
    let claim = honest_claim(&descriptor, &source);
    let (proof, _) = frontend_proof(&descriptor, &claim, &source);
    for row in 0..128 {
        let mut malformed = proof.clone();
        malformed[row * 16] ^= 1;
        rejects(&descriptor, &claim, &malformed);
    }
    for round in 0..3 {
        for atom in 0..2 {
            let mut malformed = proof.clone();
            malformed[128 * 16 + round * 42 + atom * 21] ^= 1;
            rejects(&descriptor, &claim, &malformed);
        }
    }
    let mut terminal = proof.clone();
    *terminal.last_mut().unwrap() ^= 1;
    rejects(&descriptor, &claim, &terminal);
}

#[test]
fn field_switch_claim_mutations_reject() {
    for width in [1, 8, 32, 128] {
        let descriptor = descriptor(width, 8, 2);
        let source = (0..8)
            .map(|i| 0x7123_4567_89ab_cdef_9876_5432_10ff_aa55u128.rotate_left(i * 9))
            .collect::<Vec<_>>();
        let claim = honest_claim(&descriptor, &source);
        let (proof, _) = frontend_proof(&descriptor, &claim, &source);
        let mut value = *claim.value();
        value[0] ^= 1;
        let altered = BinaryWordClaim::new(
            &descriptor,
            *claim.polynomial(),
            claim.point().to_vec(),
            value,
        )
        .unwrap();
        rejects(&descriptor, &altered, &proof);
        for coordinate in 0..claim.point().len() {
            let mut point = claim.point().to_vec();
            point[coordinate][0] ^= 1;
            let altered =
                BinaryWordClaim::new(&descriptor, *claim.polynomial(), point, *claim.value())
                    .unwrap();
            rejects(&descriptor, &altered, &proof);
        }
        let altered = BinaryWordClaim::new(
            &descriptor,
            [30; 32],
            claim.point().to_vec(),
            *claim.value(),
        )
        .unwrap();
        rejects(&descriptor, &altered, &proof);
    }
}

#[test]
fn field_switch_rejects_false_source_word() {
    for width in [1, 8, 32, 128] {
        let descriptor = descriptor(width, 8, 2);
        let mut source = vec![0x1234_5678_9abc_def0_9876_5432_10ff_aa55u128; 8];
        let claim = honest_claim(&descriptor, &source);
        source[3] ^= 1;
        let mut prover = new_prover_channel(DOMAIN, b"caller").unwrap();
        assert!(matches!(
            prove_field_switch(&descriptor, &claim, &source, &mut prover),
            Err(AkitaError::InvalidInput(_))
        ));
    }
}
