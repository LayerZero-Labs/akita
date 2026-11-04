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
    // The same in both protocol modes. The digest is SHA3-256 since v8; the
    // identity bytes are unchanged.
    assert_eq!(identity.len(), 2518);
    assert_eq!(
        akita_params::descriptor_bytes::digest_descriptor_bytes(&identity),
        [
            89, 193, 157, 175, 187, 151, 67, 188, 36, 37, 232, 116, 218, 128, 237, 88, 228, 112,
            233, 150, 7, 19, 153, 112, 231, 8, 249, 24, 194, 204, 11, 155,
        ]
    );
}

pub(super) fn assert_base_bytes(proof: &[u8], mixed: bool, word_width: usize) {
    // Pinned with the caller prefix. The dev protocol currently equals the
    // current protocol, so both modes share these bytes. Unselected prime
    // planes preserve every word-only proof byte.
    let (length, digest) = match (mixed, word_width) {
        (false, 128) => (
            80963,
            [
                135, 36, 30, 103, 130, 185, 182, 163, 117, 245, 90, 70, 191, 237, 71, 125, 12, 252,
                105, 126, 189, 19, 133, 109, 85, 26, 178, 1, 204, 99, 183, 36,
            ],
        ),
        (false, 8) => (
            80945,
            [
                158, 199, 237, 25, 126, 94, 5, 252, 21, 127, 197, 215, 50, 162, 118, 114, 190, 157,
                113, 82, 54, 208, 221, 98, 210, 66, 92, 72, 187, 247, 54, 130,
            ],
        ),
        (true, 128) => (
            81132,
            [
                34, 16, 171, 182, 69, 126, 23, 14, 244, 217, 59, 12, 167, 27, 147, 124, 186, 35,
                207, 201, 175, 204, 193, 215, 65, 201, 198, 58, 18, 74, 84, 66,
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
