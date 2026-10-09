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

#[test]
fn shipped_small_supported_geometry_round_trip() {
    use akita_labinius_pcs::{shipped::SUPPORTED_GEOMETRIES, RootPcsSizing, RootSetup};
    use akita_serialization::AkitaSerialize;
    use akita_types::proof::AkitaSetupSeed;
    let geometry = SUPPORTED_GEOMETRIES[0];
    let root = RootSetup::derive(
        geometry.profile,
        geometry.log_num_cells,
        geometry.log_fold_width,
        geometry.lambda_fold,
        AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
    )
    .unwrap();
    let fixture = support::Fixture::<Digits2>::new(&root);
    let prover = fixture.prover(root.clone());
    let verifier = fixture.verifier(root.clone());
    let source = <BinaryField128 as Host>::source(root.setup().source_len());
    let point = <BinaryField128 as Host>::point(root.setup().num_vars());
    let value = support::evaluate::<BinaryField128>(&source, &point);
    let output = prover.commit::<BinaryField128>(&source).unwrap();
    let proof = prover
        .open::<BinaryField128>(&source, &output, &point, value)
        .unwrap();
    verifier
        .verify::<BinaryField128>(&output.committed_group, &point, value, &proof)
        .unwrap();
    let sizing = RootPcsSizing::new(
        geometry.profile,
        geometry.log_num_cells,
        geometry.log_fold_width,
        geometry.lambda_fold,
        Digits2::BASE,
    )
    .unwrap();
    assert!(
        proof.len()
            <= sizing
                .opening_proof_bound::<Digits2, BinaryField128>(&fixture.images, &fixture.digits)
                .unwrap()
    );
    let mut bytes = Vec::new();
    output
        .committed_group
        .serialize_compressed(&mut bytes)
        .unwrap();
    assert_eq!(
        bytes.len(),
        sizing.commitment_bytes(&fixture.images).unwrap()
    );
}
