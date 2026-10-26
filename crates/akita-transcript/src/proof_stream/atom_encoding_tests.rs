use super::*;
use jolt_field::{CanonicalBytes, Prime128OffsetA7F7, Prime32Offset99 as F, Ring};

#[test]
fn atom_bytes_encode_inline_and_spilled_lengths() {
    for len in [0, 16, 64, 65, 200] {
        let expected: Vec<u8> = (0..len).map(|index| (index * 7 + 1) as u8).collect();
        let encoded = AtomBytes::with_len(len, |out| out.copy_from_slice(&expected));
        assert_eq!(matches!(encoded, AtomBytes::Inline { .. }), len <= 64);
        assert_eq!(encoded.as_ref(), expected.as_slice());
    }
}

#[test]
fn atom_encodings_and_challenges_match_frozen_bytes() {
    // Frozen from origin/main before atom encodings stopped allocating.
    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
    type E = jolt_field::FpExt4<F>;
    type WideF = Prime128OffsetA7F7;
    type WideE = jolt_field::FpExt4<WideF>;

    let wide = WideF::from_u128(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210);
    assert_eq!(
        hex(FieldAtom::new(wide).encode().as_ref()),
        "1032547698badcfeefcdab8967452301"
    );
    let small = F::from_u64(0x89ab_cdef);
    assert_eq!(hex(FieldAtom::new(small).encode().as_ref()), "efcdab89");
    let ext = E::from_base_slice(&[
        F::from_u64(1),
        F::from_u64(0x0102_0304),
        F::from_u64(0xfedc_ba98),
        F::from_u64(7),
    ]);
    assert_eq!(
        hex(ExtensionAtom::<F, E>::new(ext).encode().as_ref()),
        "010000000403020198badcfe07000000"
    );
    // Exactly fills the inline atom buffer.
    let wide_ext = WideE::from_base_slice(&[
        wide,
        WideF::from_u64(2),
        WideF::from_u128(u128::MAX >> 1),
        WideF::from_u64(0xabcd),
    ]);
    assert_eq!(
        hex(ExtensionAtom::<WideF, WideE>::new(wide_ext)
            .encode()
            .as_ref()),
        "1032547698badcfeefcdab8967452301\
         02000000000000000000000000000000\
         ffffffffffffffffffffffffffffff7f\
         cdab0000000000000000000000000000"
    );

    let mut prover = new_prover_channel(b"atom-pins", b"fixture").unwrap();
    send_field(&mut prover, small);
    send_extension::<F, E>(&mut prover, ext);
    assert_eq!(
        hex(prover.narg_string()),
        "efcdab89010000000403020198badcfe07000000"
    );
    // The challenge depends on the sponge; this value is the Blake2b one.
    #[cfg(feature = "transcript-blake2b")]
    {
        let challenge = ext_challenge::<F, E, _>(
            &mut prover,
            ProtocolSiteId {
                family: 5,
                ..ProtocolSiteId::default()
            },
        )
        .unwrap();
        let challenge_bytes: Vec<u8> = challenge
            .to_base_vec()
            .iter()
            .flat_map(CanonicalBytes::to_bytes_le_vec)
            .collect();
        assert_eq!(hex(&challenge_bytes), "c2aea236b8ccba30df6fd73007258c36");
    }
}
