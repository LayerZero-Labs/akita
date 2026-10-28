use super::TestF;
use jolt_field::CanonicalBytes;

#[test]
fn ordinary_setup_prefix_bytes_are_pinned() {
    let seed = akita_types::AkitaSetupSeed::shake256_paged_v1([0x5b; 32]);
    let prefix = akita_types::derive_public_matrix_prefix::<TestF>(8, &seed);
    // Independently expanded SHAKE256 page zero for the fp128 preset modulus
    // 2^128 - 2^32 + 22537. Each candidate is already canonical.
    let expected = [
        0x37cd540a988d536c68e6e2d034ff5b1cu128,
        0xb81760f8d85eddb58b4a45386ca27a24,
        0x9c97efe276e1fe45bff1217f358b0d93,
        0x635865006535ddce0493c73f42754e63,
        0x04f23064942df5b1e6532d6ee0d93620,
        0xd2f0c70350a31dd4cb51cc07e93585c7,
        0x6e80b0462c8cbad4d119c37197a4b751,
        0x569e16e95bd92c903bd004dfe62f5e6e,
    ];
    for (value, expected) in prefix.as_field_slice().iter().zip(expected) {
        let mut bytes = [0u8; 16];
        value.to_bytes_le(&mut bytes);
        assert_eq!(bytes, expected.to_le_bytes());
    }
}
