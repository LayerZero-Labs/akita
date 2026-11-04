//! Word-only proofs pin the merged projection and typed Stage 3 protocol.

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
