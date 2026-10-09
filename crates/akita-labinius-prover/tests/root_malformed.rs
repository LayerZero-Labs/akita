#![cfg(feature = "labinius")]

#[path = "root_reduction_support.rs"]
mod support;

use akita_algebra::binary::{field_switch::SwitchField, BinaryField162, BinaryField192};
use akita_error::AkitaError;
use akita_params::sis::labinius::LabiniusDigitBase;
use support::{common::TestHost, Case, H};

#[test]
fn every_sumcheck_round_and_remaining_auxiliary_region_rejects_a_flipped_bit() {
    // A fold logarithm of zero gives the smallest admitted fold width, one.
    let case = Case::<H>::new(LabiniusDigitBase::Bits1, 0);
    let (proof, _, _) = case.prove();
    for (name, round, start, width) in case.round_messages() {
        assert!(width > 0);
        let mut changed = proof.clone();
        changed[start] ^= 1;
        assert!(
            matches!(case.verify(&changed), Err(AkitaError::InvalidProof)),
            "accepted tampered {name} round {round}"
        );
    }
    let regions = case.regions();
    for (name, start, length) in regions
        .iter()
        .filter(|(name, _, _)| matches!(*name, "Q" | "K"))
    {
        for position in [*start, start + length / 2, start + length - 1] {
            let mut changed = proof.clone();
            changed[position] ^= 1;
            assert!(
                matches!(case.verify(&changed), Err(AkitaError::InvalidProof)),
                "accepted tampered {name} at {position}"
            );
        }
    }
}

#[test]
fn every_region_and_round_boundary_and_adjacent_prefix_rejects() {
    let case = Case::<H>::new(LabiniusDigitBase::Bits1, 0);
    let (proof, _, _) = case.prove();
    // This fixture has a 64 KiB response table and an 70,222-byte proof.
    // Every strict prefix would require 70,222 replays, repeatedly absorbing
    // the image and parsing the response: gigabytes of work in this one test.
    // Cover all region/round boundaries and one byte on either side instead.
    assert_eq!(case.layout.witness_len(), 65_536);
    assert_eq!(proof.len(), 70_222);
    let mut boundaries = vec![0, proof.len()];
    for (_, start, length) in case.regions() {
        boundaries.extend([start, start + length]);
    }
    for (_, _, start, width) in case.round_messages() {
        boundaries.extend([start, start + width]);
    }
    let frontend_partials = H::ROWS * core::mem::size_of::<<H as SwitchField>::Source>();
    let binary_width = BinaryField162::ZERO.to_bytes().len();
    // Include the binary frontend's round boundaries and terminal atom too.
    for round in 0..=case.point.len() {
        boundaries.push(frontend_partials + round * 2 * binary_width);
    }
    let mut prefixes = Vec::new();
    for boundary in boundaries {
        if let Some(before) = boundary.checked_sub(1) {
            prefixes.push(before);
        }
        prefixes.extend([boundary, boundary + 1]);
    }
    prefixes.sort_unstable();
    prefixes.dedup();
    for prefix in prefixes.into_iter().filter(|&length| length < proof.len()) {
        assert!(
            matches!(
                std::panic::catch_unwind(|| case.verify(&proof[..prefix])),
                Ok(Err(AkitaError::InvalidProof))
            ),
            "prefix {prefix} accepted, panicked, or returned another error"
        );
    }
    let mut trailing = proof;
    trailing.push(0);
    assert!(matches!(
        case.verify(&trailing),
        Err(AkitaError::InvalidProof)
    ));
}

fn binary_atoms_and_switch_width<T: TestHost>(source_width: usize) {
    assert_eq!(core::mem::size_of::<T::Source>(), source_width);
    // Switch partials are full-width u128 (F128) or u64 (F192) polynomial
    // bits: every bit pattern in their encoding width is canonical. The
    // frontend's subsequent F162 atoms do have six unused high bits.
    for bit in 0..source_width * 8 {
        let value = T::Source::try_from(1u128 << bit).ok().unwrap();
        let encoded = Into::<u128>::into(value).to_le_bytes();
        assert_eq!(encoded[bit / 8], 1 << (bit % 8));
        assert!(encoded[source_width..].iter().all(|&byte| byte == 0));
    }
    let mut maximal = [0u8; 16];
    maximal[..source_width].fill(255);
    assert!(T::Source::try_from(u128::from_le_bytes(maximal)).is_ok());

    let case = Case::<T>::new(LabiniusDigitBase::Bits1, 0);
    let (proof, _, _) = case.prove();
    let regions = case.regions();
    let (_, frontend, _) = regions
        .iter()
        .find(|(name, _, _)| *name == "frontend")
        .unwrap();
    let (_, u, _) = regions.iter().find(|(name, _, _)| *name == "U").unwrap();
    let frontend_atom = frontend + T::ROWS * source_width;
    let binary_width = BinaryField162::ZERO.to_bytes().len();
    for (name, atom) in [("frontend", frontend_atom), ("U", *u)] {
        let mut changed = proof.clone();
        changed[atom + binary_width - 1] |= 4;
        assert!(BinaryField162::from_bytes(&changed[atom..atom + binary_width]).is_none());
        assert!(
            matches!(case.verify(&changed), Err(AkitaError::InvalidProof)),
            "accepted noncanonical {name} for source width {source_width}"
        );
    }
}

#[test]
fn binary_frontend_and_expansion_atoms_reject_unused_bits_for_both_hosts() {
    binary_atoms_and_switch_width::<H>(16);
    binary_atoms_and_switch_width::<BinaryField192>(8);
}
