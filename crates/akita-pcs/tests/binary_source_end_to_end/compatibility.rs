//! Word-only proofs pinned against the pre-merge projection implementation.

pub(super) fn assert_base_bytes(proof: &[u8], mixed: bool, word_width: usize) {
    // Measured from commit b3c084284 in both protocol modes, including the caller
    // prefix. Unselected prime planes preserve every word-only proof byte.
    let (length, digest) = match (akita_params::DEV_PROTOCOL, mixed, word_width) {
        (false, false, 128) => (
            77936,
            [
                50, 204, 119, 115, 98, 51, 230, 178, 117, 15, 109, 232, 60, 141, 92, 3, 35, 209,
                87, 233, 0, 1, 55, 130, 40, 135, 227, 190, 222, 206, 59, 189,
            ],
        ),
        (false, false, 8) => (
            77937,
            [
                236, 116, 45, 8, 183, 19, 168, 254, 195, 82, 138, 185, 160, 196, 114, 93, 71, 59,
                247, 76, 204, 162, 47, 112, 70, 21, 218, 144, 176, 75, 136, 50,
            ],
        ),
        (false, true, 128) => (
            92101,
            [
                195, 138, 166, 89, 104, 210, 70, 30, 28, 133, 157, 25, 9, 129, 110, 21, 58, 232,
                131, 55, 214, 91, 88, 27, 91, 108, 211, 218, 163, 32, 169, 204,
            ],
        ),
        (true, false, 8) => (
            72898,
            [
                42, 171, 97, 89, 253, 233, 25, 19, 209, 23, 101, 251, 213, 12, 109, 52, 160, 30,
                51, 101, 230, 213, 85, 129, 236, 248, 108, 86, 213, 139, 246, 156,
            ],
        ),
        (true, false, 128) => (
            72877,
            [
                68, 140, 242, 103, 13, 229, 195, 77, 7, 231, 155, 70, 165, 80, 57, 224, 86, 153,
                109, 63, 95, 25, 193, 193, 44, 119, 108, 243, 182, 4, 210, 219,
            ],
        ),
        (true, true, 128) => (
            87119,
            [
                237, 84, 17, 175, 85, 86, 78, 46, 241, 166, 98, 138, 32, 201, 121, 241, 117, 106,
                221, 161, 123, 142, 16, 127, 103, 5, 26, 166, 36, 27, 222, 86,
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
