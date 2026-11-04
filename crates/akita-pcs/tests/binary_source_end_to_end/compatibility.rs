//! Word-only proofs pin the merged projection and typed Stage 3 protocol.

pub(super) fn assert_base_bytes(proof: &[u8], mixed: bool, word_width: usize) {
    // Pinned with the caller prefix. The dev protocol currently equals the
    // current protocol, so both modes share these bytes. Unselected prime
    // planes preserve every word-only proof byte.
    let (length, digest) = match (mixed, word_width) {
        (false, 128) => (
            81005,
            [
                38, 46, 209, 96, 226, 126, 216, 221, 156, 148, 90, 55, 93, 154, 56, 160, 15, 158,
                201, 249, 58, 169, 13, 54, 168, 234, 99, 98, 191, 3, 213, 1,
            ],
        ),
        (false, 8) => (
            80973,
            [
                187, 20, 223, 7, 75, 87, 34, 82, 236, 181, 46, 159, 168, 13, 64, 188, 19, 111, 206,
                146, 69, 133, 135, 208, 221, 108, 255, 88, 1, 122, 25, 195,
            ],
        ),
        (true, 128) => (
            81133,
            [
                179, 185, 24, 132, 88, 21, 195, 36, 230, 235, 190, 98, 255, 108, 13, 185, 214, 68,
                21, 3, 151, 239, 90, 133, 224, 66, 221, 232, 105, 37, 170, 168,
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
