#![cfg(feature = "labinius")]

mod common;
#[path = "root/support.rs"]
mod support;

use akita_algebra::binary::BinaryField128;
use akita_labinius_pcs::{
    config::{DigitConfig, Digits1, Digits2, Digits4},
    shipped, RootPcsSizing,
};
use akita_params::sis::labinius::LabiniusDigitBase;
use akita_serialization::AkitaSerialize;
use support::Case;

fn report_first_profile<C: DigitConfig>() {
    let sizing = RootPcsSizing::new(common::PROFILE, 22, 8, 128, C::BASE).unwrap();
    let artifacts = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts");
    let (images, digits) = shipped::catalogs::<C>(&sizing, &artifacts).unwrap();
    let requirements = sizing.setup_requirements(&images, &digits).unwrap();
    let commitment = sizing.commitment_bytes(&images).unwrap();
    let bound = sizing
        .opening_proof_bound::<C, BinaryField128>(&images, &digits)
        .unwrap();
    assert!(commitment > 0);
    assert!(bound > commitment);
    eprintln!(
        "root sizing bits={} image_log={} digit_log={} setup_elements={} commitment_bytes={commitment} proof_bound={bound}",
        C::BASE.bits(), sizing.image_log_len(), sizing.digit_log_len(),
        requirements.matrix_capacity().num_field_elements,
    );
}

#[test]
fn first_profile_shipped_catalogs_report_sizes_for_all_bases() {
    report_first_profile::<Digits1>();
    report_first_profile::<Digits2>();
    report_first_profile::<Digits4>();
}

#[test]
fn first_profile_table_lengths_are_golden() {
    for (base, digit_log) in [
        (LabiniusDigitBase::Bits1, 26),
        (LabiniusDigitBase::Bits2, 25),
        (LabiniusDigitBase::Bits4, 24),
    ] {
        let sizing = RootPcsSizing::new(common::PROFILE, 22, 8, 128, base).unwrap();
        assert_eq!(sizing.image_log_len(), 18);
        assert_eq!(sizing.digit_log_len(), digit_log);
        assert_eq!(sizing.image_key().final_group.num_vars(), 18);
        assert_eq!(sizing.scalar_digit_key().final_group.num_vars(), digit_log);
        assert_eq!(sizing.shape().num_cells(), 1 << 22);
        assert_eq!(sizing.shape().fold_width(), 1 << 8);
        assert_eq!(sizing.profile(), common::PROFILE);
        assert_eq!(sizing.log_num_cells(), 22);
        assert_eq!(sizing.log_fold_width(), 8);
        assert_eq!(sizing.lambda_fold(), 128);
        assert_eq!(sizing.base(), base);
    }
}

#[test]
fn bounds_cover_real_openings_and_commitment_serialization_at_two_geometries() {
    for fold in [0, 1] {
        let case = Case::<Digits2>::new(fold);
        let fixture = <Digits2 as support::TestDigits>::fixture(fold);
        let sizing =
            RootPcsSizing::new(common::PROFILE, 4, fold, 128, LabiniusDigitBase::Bits2).unwrap();
        assert_eq!(sizing.encoding(), case.layout.encoding());
        assert_eq!(sizing.image_log_len(), case.layout.image_log_len());
        assert_eq!(sizing.digit_log_len(), case.layout.witness_log_len());
        let image_row = fixture.images.resolve_key(&sizing.image_key()).unwrap();
        let key = sizing.grouped_digit_key(&fixture.images).unwrap();
        assert_eq!(key.precommitteds, [image_row.profiles().final_group]);
        fixture.digits.resolve_key(&key).unwrap();
        let requirements = sizing
            .setup_requirements(&fixture.images, &fixture.digits)
            .unwrap();
        assert_eq!(requirements.max_num_batched_polys(), 2);
        assert_eq!(
            requirements.max_num_vars(),
            sizing.image_log_len().max(sizing.digit_log_len())
        );
        let mut commitment = Vec::new();
        case.output
            .committed_group
            .serialize_compressed(&mut commitment)
            .unwrap();
        assert_eq!(
            sizing.commitment_bytes(&fixture.images).unwrap(),
            commitment.len()
        );
        let proof = case.prove();
        case.verify(&proof).unwrap();
        let bound = sizing
            .opening_proof_bound::<Digits2, BinaryField128>(&fixture.images, &fixture.digits)
            .unwrap();
        assert!(
            bound >= proof.len(),
            "bound {bound}, measured {}",
            proof.len()
        );
    }
}

#[test]
fn mismatched_digit_config_and_overflowing_geometry_are_rejected() {
    let sizing = RootPcsSizing::new(common::PROFILE, 4, 0, 128, LabiniusDigitBase::Bits1).unwrap();
    let fixture = <Digits2 as support::TestDigits>::fixture(0);
    assert!(sizing
        .setup_requirements(&fixture.images, &fixture.digits)
        .is_err());
    assert!(sizing
        .opening_proof_bound::<Digits2, BinaryField128>(&fixture.images, &fixture.digits)
        .is_err());
    assert!(RootPcsSizing::new(
        common::PROFILE,
        usize::BITS,
        0,
        128,
        LabiniusDigitBase::Bits2,
    )
    .is_err());
}
