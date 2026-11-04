//! Word-only proofs pin the merged projection and typed Stage 3 protocol.

pub(super) fn assert_base_bytes(proof: &[u8], mixed: bool, word_width: usize) {
    // Pinned in both protocol modes, including the caller prefix. Unselected
    // prime planes preserve every word-only proof byte.
    let (length, digest) = match (akita_params::DEV_PROTOCOL, mixed, word_width) {
        (false, false, 128) => (
            86097,
            [
                212, 97, 142, 84, 220, 49, 157, 135, 46, 18, 43, 92, 66, 136, 207, 73, 186, 58,
                159, 50, 10, 49, 19, 130, 109, 166, 178, 58, 159, 133, 192, 148,
            ],
        ),
        (false, false, 8) => (
            86052,
            [
                213, 107, 158, 29, 182, 172, 197, 174, 205, 77, 35, 50, 16, 178, 254, 114, 87, 78,
                40, 214, 21, 125, 175, 169, 116, 95, 47, 30, 89, 165, 189, 184,
            ],
        ),
        (false, true, 128) => (
            86147,
            [
                171, 17, 81, 228, 25, 129, 17, 240, 220, 19, 105, 235, 220, 32, 149, 202, 57, 89,
                230, 70, 188, 71, 173, 68, 77, 52, 240, 14, 89, 13, 80, 54,
            ],
        ),
        (true, false, 8) => (
            80966,
            [
                193, 94, 154, 129, 19, 152, 243, 87, 165, 10, 47, 91, 239, 252, 254, 89, 134, 76,
                44, 161, 253, 107, 190, 185, 76, 215, 202, 220, 244, 225, 205, 235,
            ],
        ),
        (true, false, 128) => (
            80947,
            [
                216, 80, 149, 192, 12, 209, 133, 115, 193, 187, 102, 98, 100, 121, 227, 213, 121,
                167, 45, 5, 98, 114, 172, 59, 220, 245, 138, 184, 14, 94, 135, 75,
            ],
        ),
        (true, true, 128) => (
            81167,
            [
                206, 27, 192, 195, 47, 149, 107, 149, 124, 42, 79, 191, 77, 32, 1, 157, 82, 168,
                75, 241, 155, 60, 249, 94, 147, 25, 14, 46, 235, 98, 170, 73,
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
