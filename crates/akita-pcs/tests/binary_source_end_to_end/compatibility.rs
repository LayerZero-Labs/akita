//! Word-only proofs pin the merged projection and typed Stage 3 protocol.

pub(super) fn assert_base_bytes(proof: &[u8], mixed: bool, word_width: usize) {
    // Pinned in both protocol modes, including the caller prefix. Unselected
    // prime planes preserve every word-only proof byte.
    let (length, digest) = match (akita_params::DEV_PROTOCOL, mixed, word_width) {
        (false, false, 128) => (
            86092,
            [
                238, 78, 211, 147, 66, 218, 216, 7, 156, 78, 0, 243, 242, 245, 33, 132, 28, 36,
                132, 83, 222, 42, 238, 7, 14, 98, 253, 244, 63, 111, 243, 50,
            ],
        ),
        (false, false, 8) => (
            86035,
            [
                102, 194, 145, 166, 0, 252, 202, 8, 167, 49, 249, 203, 90, 42, 117, 128, 160, 150,
                22, 58, 94, 180, 33, 45, 9, 24, 170, 211, 212, 39, 51, 76,
            ],
        ),
        (false, true, 128) => (
            86159,
            [
                189, 13, 68, 246, 251, 68, 133, 37, 33, 186, 58, 181, 5, 96, 253, 192, 229, 0, 43,
                213, 119, 116, 5, 69, 89, 86, 208, 231, 20, 184, 205, 31,
            ],
        ),
        (true, false, 8) => (
            81004,
            [
                69, 205, 80, 11, 97, 53, 94, 8, 185, 55, 209, 157, 238, 54, 85, 173, 45, 218, 19,
                62, 115, 108, 193, 147, 255, 43, 236, 87, 235, 235, 216, 164,
            ],
        ),
        (true, false, 128) => (
            80953,
            [
                145, 119, 176, 221, 198, 119, 115, 128, 132, 194, 241, 56, 155, 101, 37, 43, 192,
                80, 197, 94, 78, 49, 244, 30, 55, 134, 80, 226, 173, 225, 245, 91,
            ],
        ),
        (true, true, 128) => (
            81068,
            [
                144, 243, 64, 45, 46, 15, 40, 89, 162, 154, 175, 139, 93, 155, 195, 57, 174, 56,
                141, 220, 79, 193, 71, 117, 199, 147, 172, 145, 84, 239, 142, 50,
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
