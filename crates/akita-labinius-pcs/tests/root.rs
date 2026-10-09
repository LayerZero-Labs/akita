#![cfg(feature = "labinius")]

#[path = "root/adversarial.rs"]
mod adversarial;
#[path = "root/benchmark.rs"]
mod benchmark;
#[path = "root/binding.rs"]
mod binding;
mod common;
#[path = "root/oracle.rs"]
mod oracle;
#[path = "root/support.rs"]
mod support;

use akita_algebra::binary::{BinaryField128, BinaryField192};
use akita_config::CommitmentConfig;
use akita_labinius_pcs::config::{DigitConfig, Digits1, Digits2, Digits4};
use akita_labinius_verifier::root::root_reduction_wire_size;
use support::{Case, Host, TestDigits};

fn complete<C: TestDigits, H: Host>() {
    for fold in [0, 1] {
        let case = Case::<C, H>::new(fold);
        let proof = case.prove();
        case.verify(&proof).unwrap();
        let regions = case.regions(&proof);
        let commitment = regions
            .iter()
            .find(|(name, _, _)| *name == "W commitment")
            .unwrap()
            .2;
        let inner = regions
            .iter()
            .find(|(name, _, _)| *name == "inner proof")
            .unwrap()
            .2;
        let reduction = root_reduction_wire_size::<H>(case.root.shape(), C::BASE).unwrap();
        assert_eq!(proof.len(), reduction + 8 + commitment + 8 + inner);
        eprintln!(
            "root toy host={} fold={fold} family={} reduction={reduction} commitment={commitment} inner={inner} proof={}",
            core::any::type_name::<H>(), C::schedule_family_name(), proof.len()
        );
    }
}

#[test]
fn complete_all_bases_both_hosts_and_both_toy_geometries() {
    complete::<Digits1, BinaryField128>();
    complete::<Digits2, BinaryField128>();
    complete::<Digits4, BinaryField128>();
    complete::<Digits1, BinaryField192>();
    complete::<Digits2, BinaryField192>();
    complete::<Digits4, BinaryField192>();
}

#[test]
fn configuration_contracts_match_honest_signed_digit_bounds() {
    for (bits, bound, name) in [
        (
            Digits1::BASE.bits(),
            Digits1::decomposition().log_commit_bound,
            Digits1::schedule_family_name(),
        ),
        (
            Digits2::BASE.bits(),
            Digits2::decomposition().log_commit_bound,
            Digits2::schedule_family_name(),
        ),
        (
            Digits4::BASE.bits(),
            Digits4::decomposition().log_commit_bound,
            Digits4::schedule_family_name(),
        ),
    ] {
        assert_eq!(bound, bits + 1);
        assert_eq!(name, format!("labinius_digits_b{bits}"));
    }
    assert_eq!(Digits1::decomposition().log_open_bound, Some(128));
    assert_eq!(Digits2::decomposition().log_open_bound, Some(128));
    assert_eq!(Digits4::decomposition().log_open_bound, Some(128));
}
