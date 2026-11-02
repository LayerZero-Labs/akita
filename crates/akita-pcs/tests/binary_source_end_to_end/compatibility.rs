//! Word-only proofs pinned against the unmodified binary-source base.

pub(super) fn assert_base_bytes(proof: &[u8], mixed: bool, word_width: usize) {
    // Generated from commit 6a2940981 with this fixture, including its caller
    // prefix. The selected prime extension must leave every proof byte intact.
    let (length, digest) = match (akita_params::DEV_PROTOCOL, mixed, word_width) {
        (false, false, 128) => (
            77873,
            [
                191, 121, 167, 253, 19, 246, 48, 57, 240, 94, 37, 151, 96, 193, 81, 238, 145, 186,
                222, 148, 134, 166, 61, 145, 64, 110, 54, 193, 195, 103, 248, 139,
            ],
        ),
        (false, false, 8) => (
            77855,
            [
                163, 38, 245, 27, 124, 93, 73, 248, 181, 8, 236, 44, 201, 61, 66, 117, 62, 160,
                220, 90, 239, 74, 205, 243, 166, 110, 194, 3, 77, 51, 103, 223,
            ],
        ),
        (false, true, 128) => (
            92116,
            [
                159, 129, 103, 216, 189, 80, 176, 163, 205, 209, 18, 24, 44, 59, 31, 110, 226, 1,
                50, 162, 195, 159, 203, 198, 60, 80, 89, 180, 127, 91, 63, 79,
            ],
        ),
        (true, false, 8) => (
            72792,
            [
                215, 37, 33, 106, 121, 20, 171, 127, 209, 108, 97, 21, 12, 99, 250, 228, 143, 215,
                27, 204, 11, 242, 231, 186, 252, 168, 76, 243, 20, 202, 237, 231,
            ],
        ),
        (true, false, 128) => (
            72818,
            [
                89, 90, 169, 116, 183, 32, 124, 108, 35, 117, 11, 139, 188, 154, 212, 23, 116, 211,
                182, 126, 206, 124, 175, 16, 52, 232, 88, 45, 81, 29, 136, 150,
            ],
        ),
        (true, true, 128) => (
            86954,
            [
                144, 141, 213, 71, 63, 141, 228, 29, 195, 122, 119, 235, 49, 125, 241, 219, 119,
                18, 87, 183, 53, 211, 134, 136, 107, 136, 138, 197, 71, 183, 171, 97,
            ],
        ),
        _ => panic!("fixture has no baseline"),
    };
    assert_eq!(proof.len(), length);
    assert_eq!(
        akita_params::descriptor_bytes::digest_descriptor_bytes(proof),
        digest
    );
}
