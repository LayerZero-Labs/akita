#![cfg(feature = "labinius")]

use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_labinius_pcs::{
    shipped::{catalogs, ShippedCatalogError, SUPPORTED_GEOMETRIES},
    DigitConfig, Digits1, Digits2, Digits4, RootPcsSizing,
};
use akita_params::sis::labinius::LabiniusDigitBase;
use std::path::PathBuf;

fn artifacts() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts")
}

fn all_rows<C: DigitConfig>() {
    for geometry in SUPPORTED_GEOMETRIES {
        assert!(geometry.bases.contains(&C::BASE));
        let sizing = RootPcsSizing::new(
            geometry.profile,
            geometry.log_num_cells,
            geometry.log_fold_width,
            geometry.lambda_fold,
            C::BASE,
        )
        .unwrap();
        let (images, digits) = catalogs::<C>(&sizing, &artifacts()).unwrap();
        let image = images.resolve_key(&sizing.image_key()).unwrap();
        let key = sizing.grouped_digit_key(&images).unwrap();
        assert_eq!(key.precommitteds, [image.profiles().final_group]);
        digits.resolve_key(&sizing.scalar_digit_key()).unwrap();
        digits.resolve_key(&key).unwrap();
        sizing.setup_requirements(&images, &digits).unwrap();
    }
}

#[test]
fn every_supported_geometry_and_base_resolves_through_trusted_admission() {
    all_rows::<Digits1>();
    all_rows::<Digits2>();
    all_rows::<Digits4>();
}

#[test]
fn unsupported_geometry_is_typed_and_rejected_before_io() {
    let geometry = SUPPORTED_GEOMETRIES[0];
    let sizing = RootPcsSizing::new(
        geometry.profile,
        4,
        0,
        geometry.lambda_fold,
        LabiniusDigitBase::Bits2,
    )
    .unwrap();
    assert!(matches!(
        catalogs::<Digits2>(&sizing, &PathBuf::from("missing-artifacts")),
        Err(ShippedCatalogError::UnsupportedGeometry)
    ));
    let sizing = RootPcsSizing::new(
        geometry.profile,
        geometry.log_num_cells,
        geometry.log_fold_width,
        geometry.lambda_fold,
        LabiniusDigitBase::Bits1,
    )
    .unwrap();
    assert!(matches!(
        catalogs::<Digits2>(&sizing, &artifacts()),
        Err(ShippedCatalogError::UnsupportedGeometry)
    ));
}

#[test]
fn altered_row_in_local_copy_is_rejected_by_the_shared_trust_boundary() {
    let geometry = SUPPORTED_GEOMETRIES[0];
    let sizing = RootPcsSizing::new(
        geometry.profile,
        geometry.log_num_cells,
        geometry.log_fold_width,
        geometry.lambda_fold,
        LabiniusDigitBase::Bits2,
    )
    .unwrap();
    let set = if akita_params::DEV_PROTOCOL {
        "schedules-labinius-dev"
    } else {
        "schedules-labinius"
    };
    let local = std::env::temp_dir().join(format!("akita-labinius-tamper-{}", std::process::id()));
    std::fs::create_dir_all(local.join(set)).unwrap();
    for family in [
        akita_labinius_pcs::ImageConfig::schedule_family_name(),
        Digits2::schedule_family_name(),
    ] {
        std::fs::copy(
            artifacts().join(set).join(format!("{family}.aks")),
            local.join(set).join(format!("{family}.aks")),
        )
        .unwrap();
    }
    let path = local
        .join(set)
        .join(format!("{}.aks", Digits2::schedule_family_name()));
    let mut bytes = std::fs::read(&path).unwrap();
    let rows = bytes
        .windows(b"\"rows\"".len())
        .position(|window| window == b"\"rows\"")
        .unwrap();
    let key = b"\"log_basis\":";
    let offset = rows
        + bytes[rows..]
            .windows(key.len())
            .position(|window| window == key)
            .unwrap()
        + key.len();
    let digit = offset + bytes[offset..].iter().position(u8::is_ascii_digit).unwrap();
    assert_ne!(bytes[digit], b'0');
    bytes[digit] = b'0';
    assert!(TrustedScheduleCatalog::<Digits2>::from_artifact_bytes(&bytes).is_err());
    std::fs::write(path, bytes).unwrap();
    assert!(matches!(
        catalogs::<Digits2>(&sizing, &local),
        Err(ShippedCatalogError::Admission(_))
    ));
    std::fs::remove_dir_all(local).unwrap();
}
