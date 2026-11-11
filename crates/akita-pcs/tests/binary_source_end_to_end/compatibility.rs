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
    // The same in both protocol modes. The field switch changes the shared
    // SIS-table identity inside these bytes; their format and length stay fixed.
    assert_eq!(identity.len(), 2518);
    assert_eq!(
        akita_params::descriptor_bytes::digest_descriptor_bytes(&identity),
        [
            180, 246, 92, 207, 101, 142, 163, 157, 52, 215, 109, 17, 242, 113, 73, 50, 31, 151, 89,
            153, 183, 205, 95, 203, 166, 17, 235, 248, 103, 92, 135, 62
        ]
    );
}

pub(super) fn assert_base_bytes(proof: &[u8], mixed: bool, word_width: usize) {
    // Pinned with the caller prefix. The dev protocol currently equals the
    // current protocol, so both modes share these bytes. Unselected prime
    // planes preserve every word-only proof byte. Regenerated after the field
    // switch: its SIS identity changes challenges and the Rice-coded payload.
    let (length, digest) = match (mixed, word_width) {
        (false, 128) => (
            81020,
            [
                35, 87, 84, 91, 170, 39, 59, 107, 193, 133, 129, 129, 249, 239, 1, 123, 40, 221,
                92, 253, 18, 115, 212, 60, 105, 30, 18, 115, 194, 241, 240, 246,
            ],
        ),
        (false, 8) => (
            80987,
            [
                207, 130, 170, 181, 159, 201, 14, 48, 114, 25, 33, 36, 175, 173, 26, 223, 143, 0,
                98, 225, 106, 72, 158, 205, 208, 125, 51, 21, 150, 38, 143, 196,
            ],
        ),
        (true, 128) => (
            81114,
            [
                167, 250, 165, 208, 200, 76, 57, 55, 0, 29, 38, 67, 16, 64, 26, 81, 154, 149, 110,
                155, 240, 87, 55, 199, 207, 237, 128, 152, 159, 56, 132, 72,
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
