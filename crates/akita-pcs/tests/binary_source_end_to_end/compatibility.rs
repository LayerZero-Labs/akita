//! Word-only proofs pin the merged projection and typed Stage 3 protocol.

pub(super) fn assert_base_bytes(proof: &[u8], mixed: bool, word_width: usize) {
    // Pinned in both protocol modes, including the caller prefix. Unselected
    // prime planes preserve every word-only proof byte.
    let (length, digest) = match (akita_params::DEV_PROTOCOL, mixed, word_width) {
        (false, false, 128) => (
            86070,
            [
                146, 112, 21, 110, 113, 32, 245, 223, 220, 254, 96, 129, 245, 63, 111, 85, 238, 52,
                79, 31, 63, 151, 43, 158, 104, 108, 75, 228, 241, 243, 116, 84,
            ],
        ),
        (false, false, 8) => (
            86046,
            [
                125, 249, 240, 187, 225, 27, 8, 211, 111, 196, 40, 31, 195, 155, 224, 47, 105, 25,
                155, 168, 57, 11, 89, 11, 205, 19, 234, 171, 58, 141, 63, 86,
            ],
        ),
        (false, true, 128) => (
            86155,
            [
                161, 145, 185, 41, 26, 64, 81, 153, 44, 248, 177, 129, 6, 31, 56, 151, 89, 126, 75,
                213, 181, 254, 210, 14, 119, 181, 201, 48, 163, 156, 89, 152,
            ],
        ),
        (true, false, 8) => (
            81013,
            [
                73, 62, 172, 228, 78, 243, 233, 12, 235, 127, 27, 109, 241, 5, 109, 152, 198, 60,
                110, 133, 157, 44, 206, 63, 79, 218, 15, 79, 197, 39, 249, 135,
            ],
        ),
        (true, false, 128) => (
            80987,
            [
                188, 180, 143, 51, 219, 49, 253, 129, 164, 238, 126, 69, 146, 46, 242, 217, 50, 44,
                136, 197, 148, 249, 21, 145, 94, 218, 120, 8, 132, 85, 198, 183,
            ],
        ),
        (true, true, 128) => (
            81075,
            [
                224, 9, 49, 114, 243, 6, 162, 220, 163, 103, 23, 238, 153, 15, 173, 92, 70, 86, 54,
                53, 49, 28, 118, 93, 88, 110, 21, 29, 237, 111, 25, 205,
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
