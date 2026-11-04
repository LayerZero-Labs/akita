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
    // Pinned with the caller prefix. The dev protocol currently equals the
    // current protocol, so both modes share these bytes. Unselected prime
    // planes preserve every word-only proof byte.
    let (length, digest) = match (mixed, word_width) {
        (false, 128) => (
            81002,
            [
                231, 11, 76, 41, 130, 125, 131, 230, 233, 83, 25, 23, 203, 249, 193, 150, 75, 77,
                96, 19, 70, 32, 251, 122, 191, 220, 171, 54, 249, 148, 123, 123,
            ],
        ),
        (false, 8) => (
            81030,
            [
                180, 77, 107, 198, 130, 96, 175, 164, 145, 137, 132, 243, 195, 187, 136, 81, 63,
                124, 18, 186, 8, 47, 215, 209, 82, 47, 42, 134, 4, 127, 85, 131,
            ],
        ),
        (true, 128) => (
            81092,
            [
                226, 110, 118, 167, 148, 238, 170, 6, 247, 177, 158, 219, 87, 180, 216, 87, 107,
                58, 54, 57, 68, 12, 133, 112, 177, 250, 60, 40, 35, 84, 175, 50,
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
