//! Word-only proofs pin the merged projection and typed Stage 3 protocol.

use super::*;

/// Stage 3 absorbs the typed-prefix identity. Every word in it is a
/// little-endian `u64`, so the bytes do not depend on the pointer width; the
/// tail region encodes `top = 64`, no fixed bit and the free-bit marker
/// `u64::MAX`.
#[test]
fn typed_prefix_identity_bytes_are_pinned() {
    let scheme = AkitaCommitmentScheme::<BinaryTestSuffix>::new(test_catalog());
    let row = scheme
        .schedules()
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(18),
        ))
        .unwrap();
    let next = &row.schedule().recursive_folds[0].params;
    let slot = next.setup_prefix().unwrap().slot_id().unwrap();
    let index = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
    let profiles = BinaryCommittedGroupManifest::commitment_profiles(&index, 1, 0).unwrap();
    let descriptor = BinarySourceDescriptor::new(index, vec![[29; 32]], profiles).unwrap();
    let manifest =
        BinaryCommittedGroupManifest::new(descriptor, 0, &row.profiles().final_group).unwrap();
    let typed = BinaryTypedPrefix::new(&manifest, true, 0, next.d_a()).unwrap();
    let identity = typed.identity(&manifest, &slot).unwrap();
    let tail = [64u64, 0, u64::MAX].map(u64::to_le_bytes).concat();
    assert!(identity.windows(tail.len()).any(|word| word == tail));
    eprintln!(
        "typed prefix identity length={} digest={:?}",
        identity.len(),
        akita_params::descriptor_bytes::digest_descriptor_bytes(&identity),
    );
    // The same in both protocol modes.
    assert_eq!(identity.len(), 2518);
    assert_eq!(
        akita_params::descriptor_bytes::digest_descriptor_bytes(&identity),
        [
            36, 53, 31, 41, 28, 94, 100, 7, 20, 106, 27, 160, 206, 76, 133, 226, 204, 249, 144, 61,
            135, 71, 23, 108, 26, 90, 119, 181, 83, 156, 5, 212,
        ]
    );
}

pub(super) fn assert_base_bytes(proof: &[u8], mixed: bool, word_width: usize) {
    // Pinned in both protocol modes, including the caller prefix. Unselected
    // prime planes preserve every word-only proof byte.
    let (length, digest) = match (akita_params::DEV_PROTOCOL, mixed, word_width) {
        (false, false, 128) => (
            86025,
            [
                87, 109, 149, 36, 161, 121, 123, 74, 45, 178, 81, 245, 240, 200, 116, 191, 121,
                115, 130, 219, 58, 5, 76, 122, 92, 63, 212, 110, 183, 90, 88, 214,
            ],
        ),
        (false, false, 8) => (
            86083,
            [
                67, 221, 170, 230, 115, 16, 204, 242, 25, 109, 89, 254, 159, 137, 2, 76, 123, 232,
                169, 255, 75, 187, 147, 177, 124, 73, 145, 221, 194, 160, 71, 122,
            ],
        ),
        (false, true, 128) => (
            86200,
            [
                75, 85, 8, 94, 168, 170, 135, 145, 191, 204, 131, 180, 158, 189, 105, 16, 48, 67,
                237, 128, 224, 230, 233, 2, 78, 94, 86, 14, 128, 237, 240, 102,
            ],
        ),
        (true, false, 8) => (
            80981,
            [
                172, 48, 148, 201, 246, 232, 215, 231, 159, 59, 52, 81, 0, 155, 96, 24, 171, 73,
                29, 240, 182, 232, 55, 66, 174, 24, 165, 12, 17, 252, 179, 112,
            ],
        ),
        (true, false, 128) => (
            81037,
            [
                92, 56, 231, 193, 247, 109, 7, 164, 29, 51, 170, 166, 167, 36, 52, 38, 178, 209,
                92, 231, 104, 44, 165, 194, 174, 151, 131, 99, 31, 28, 154, 57,
            ],
        ),
        (true, true, 128) => (
            81123,
            [
                45, 172, 132, 152, 186, 149, 81, 168, 19, 138, 202, 37, 121, 28, 82, 51, 84, 33,
                244, 83, 167, 128, 108, 16, 148, 127, 255, 30, 223, 24, 69, 163,
            ],
        ),
        _ => panic!("fixture has no baseline"),
    };
    eprintln!(
        "stage3 compatibility dev={} mixed={mixed} word_width={word_width} length={} digest={:?}",
        akita_params::DEV_PROTOCOL,
        proof.len(),
        akita_params::descriptor_bytes::digest_descriptor_bytes(proof),
    );
    assert_eq!(proof.len(), length);
    assert_eq!(
        akita_params::descriptor_bytes::digest_descriptor_bytes(proof),
        digest
    );
}
