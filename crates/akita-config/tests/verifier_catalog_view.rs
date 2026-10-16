use akita_config::{proof_optimized::fp128, TrustedScheduleCatalog};

mod common;
use common::checked_in_catalog;

#[test]
fn verifier_catalog_view_preserves_identity_and_restricts_available_rows() {
    let full = checked_in_catalog::<fp128::Dense>();
    let mut rows = full.rows();
    let selected = rows.next().unwrap();
    let omitted = rows.next().unwrap().selection();
    let bytes = full.to_verifier_view(&[selected.selection()]).unwrap();
    let view = TrustedScheduleCatalog::<fp128::Dense>::from_verifier_view(&bytes).unwrap();
    assert_eq!(view.catalog_digest(), full.catalog_digest());
    assert_eq!(view.len(), 1);
    assert_eq!(
        view.resolve_selection(selected.selection())
            .unwrap()
            .schedule(),
        selected.schedule()
    );
    assert!(view.resolve_selection(omitted).is_err());
    let partial = "selected-row verifier catalog";
    let error = view.to_artifact_bytes().unwrap_err().to_string();
    assert!(error.contains(partial), "{error}");
    let error = akita_config::SetupRequirements::from_catalog::<fp128::Dense>(&view, 14, 1)
        .unwrap_err()
        .to_string();
    assert!(error.contains(partial), "{error}");
    akita_config::SetupRequirements::from_catalog::<fp128::Dense>(&full, 14, 1).unwrap();
    let error = view
        .rows_within_setup_capacity(14, 1)
        .unwrap_err()
        .to_string();
    assert!(error.contains(partial), "{error}");
    assert_eq!(
        view.catalog()
            .to_verifier_view(&[selected.selection()])
            .unwrap(),
        bytes
    );
    assert!(full.to_verifier_view(&[]).is_err());
    assert!(TrustedScheduleCatalog::<fp128::OneHot>::from_verifier_view(&bytes).is_err());
    full.catalog().validate_complete().unwrap();
}

#[test]
fn verifier_catalog_view_checks_membership_order_and_framing() {
    let full = checked_in_catalog::<fp128::Dense>();
    let selected = full.rows().next().unwrap().selection();
    let bytes = full.to_verifier_view(&[selected]).unwrap();
    let offset = bytes
        .windows(32)
        .position(|window| window == selected.row_digest.as_bytes())
        .unwrap();
    let decode = |bytes: &[u8]| TrustedScheduleCatalog::<fp128::Dense>::from_verifier_view(bytes);

    let mut missing = bytes.clone();
    missing[offset..offset + 32].fill(0);
    let error = decode(&missing).unwrap_err().to_string();
    assert!(
        error.contains("absent from the catalog commitment"),
        "{error}"
    );

    let not_canonical_list = |bytes: &[u8]| {
        let error = decode(bytes).unwrap_err().to_string();
        assert!(error.contains("bounded canonical list"), "{error}");
    };
    let mut duplicate = bytes.clone();
    duplicate[offset + 32..offset + 64].copy_from_slice(selected.row_digest.as_bytes());
    not_canonical_list(&duplicate);
    let mut swapped = bytes.clone();
    let next: [u8; 32] = swapped[offset + 32..offset + 64].try_into().unwrap();
    swapped.copy_within(offset..offset + 32, offset + 32);
    swapped[offset..offset + 32].copy_from_slice(&next);
    not_canonical_list(&swapped);
    let mut magic = bytes.clone();
    magic[0] ^= 1;
    assert!(decode(&magic).is_err());
    // The epoch is a one-byte varint; its two-byte overlong form decodes to
    // the same value, so only the canonical re-encoding catches it.
    let mut overlong = bytes.clone();
    overlong.splice(8..9, [bytes[8] | 0x80, 0]);
    let error = decode(&overlong).unwrap_err().to_string();
    assert!(error.contains("not canonically encoded"), "{error}");
    let mut oversized = bytes.clone();
    oversized.resize(
        akita_config::MAX_TRUSTED_SCHEDULE_ARTIFACT_ROW_BYTES + (1 << 14) * 32 + 1,
        0,
    );
    let error = decode(&oversized).unwrap_err().to_string();
    assert!(error.contains("byte limit"), "{error}");
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(decode(&trailing).is_err());
    let mut epoch = bytes.clone();
    epoch[8] ^= 1;
    assert!(decode(&epoch).is_err());
    assert!(decode(&bytes[..bytes.len() - 1]).is_err());

    // The final encoded field is terminal.input_witness_len. Altering it must
    // not bypass the same transition audit used for a complete catalog.
    let mut malformed_row = bytes.clone();
    *malformed_row.last_mut().unwrap() ^= 1;
    let error = decode(&malformed_row).unwrap_err().to_string();
    assert!(
        !error.contains("absent from the catalog commitment"),
        "{error}"
    );

    // An opaque omitted identity changes the catalog digest, even though the
    // selected row is still valid; a consumer that binds the digest (as Jolt's
    // transcript does) sees a different catalog.
    let mut changed_commitment = bytes;
    let last = offset + 32 * (full.len() - 1);
    changed_commitment[last..last + 32].fill(255);
    let changed = decode(&changed_commitment).unwrap();
    assert_ne!(changed.catalog_digest(), full.catalog_digest());
}

#[test]
fn verifier_catalog_view_carries_several_rows_once() {
    let full = checked_in_catalog::<fp128::Dense>();
    let mut rows = full.rows();
    let first = rows.next().unwrap().selection();
    let second = rows.next().unwrap().selection();
    let bytes = full.to_verifier_view(&[second, first]).unwrap();
    assert_eq!(
        full.to_verifier_view(&[first, second, first]).unwrap(),
        bytes
    );
    let view = TrustedScheduleCatalog::<fp128::Dense>::from_verifier_view(&bytes).unwrap();
    assert_eq!(view.len(), 2);
    assert_eq!(view.catalog_digest(), full.catalog_digest());
    view.resolve_selection(first).unwrap();
    view.resolve_selection(second).unwrap();
}
