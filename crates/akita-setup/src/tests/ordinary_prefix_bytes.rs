use super::TestF;
use jolt_algebra::CanonicalBytes;

#[test]
fn ordinary_setup_prefix_bytes_are_pinned() {
    let seed = jolt_crypto::lattice::PublicMatrixSeed::shake256_paged_v1([0x5b; 32]);
    let prefix = jolt_crypto::lattice::derive_public_matrix_prefix::<TestF>(8, &seed).unwrap();
    // Independently expanded SHAKE256 page zero for the fp128 preset modulus
    // 2^128 - 0x6A23801F. Each candidate is already canonical.
    let expected = [
        0x347ead88376c66b6c99be0a8a0cd34d3u128,
        0xe0bcbb9d196aafd96d8f82bc055a70be,
        0x4a79a1a39e2289c634c834de523c5f39,
        0x25cd3a85dca4b1e5d01d5e329f64cf7a,
        0x614440d893b7ecd19c575e3deb617b68,
        0xe521686d3f5e6aa2d8fd0b7e8415cb05,
        0x4c1dc1b573ffb5d32188ec70c585144d,
        0x53ce9ad43df2d8443cfb2c0815b9bf91,
    ];
    for (value, expected) in prefix.iter().zip(expected) {
        let mut bytes = [0u8; 16];
        value.to_bytes_le(&mut bytes);
        assert_eq!(bytes, expected.to_le_bytes());
    }
}
