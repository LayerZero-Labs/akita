use std::sync::Arc;

use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_types::{
    derive_public_matrix_prefix, AkitaExpandedSetup, AkitaSetupDescriptor, AkitaSetupSeed,
    AkitaVerifierSetup, SetupPrefixVerifierRegistry,
};
use jolt_field::Prime32Offset99;

type F = Prime32Offset99;
const D: usize = 64;

fn setup(seed_byte: u8) -> AkitaVerifierSetup<F> {
    let seed = AkitaSetupSeed::shake256_paged_v1([seed_byte; 32]);
    let matrix = derive_public_matrix_prefix::<F>(2 * D, &seed);
    let expanded = AkitaExpandedSetup::from_verified_parts(
        AkitaSetupDescriptor {
            max_num_vars: 7,
            max_num_batched_polys: 1,
            num_field_elements: 2 * D,
            setup_seed: seed.clone(),
        },
        matrix,
    )
    .unwrap();
    AkitaVerifierSetup::from_parts(Arc::new(expanded), SetupPrefixVerifierRegistry::new(seed))
        .unwrap()
}

#[test]
fn replacing_a_warmed_clone_preserves_setup_matrix_identity() {
    let first = setup(7);
    let second = setup(9);
    let mut negative_units = [[0i16; D]; 2];
    for ring in &mut negative_units {
        ring[0] = -1;
    }
    let product = |setup: &AkitaVerifierSetup<F>| {
        setup
            .prepared_verifier_ntt_prefix::<D>(2, 0, 2, 1)
            .unwrap()
            .mat_vec_i16::<F>(1, 1, &negative_units)
            .unwrap()
    };
    let first_product = product(&first);
    let second_product = product(&second);
    assert_ne!(first_product, second_product);

    let mut replacement = first.clone();
    assert_eq!(product(&replacement), first_product);
    assert!(replacement.verifier_ntt_cache_bytes().unwrap() > 0);
    replacement =
        AkitaVerifierSetup::from_parts(second.expanded().clone(), second.prefix_slots().clone())
            .unwrap();
    replacement.check().unwrap();
    assert_eq!(replacement, second);
    assert_eq!(replacement.verifier_ntt_cache_bytes().unwrap(), 0);
    assert_eq!(product(&replacement), second_product);
    assert_eq!(product(&first), first_product);

    // A caller may mutate its own Arc clone, but cannot change the matrix
    // retained by either verifier or its shared cache.
    let mut detached = first.expanded().clone();
    *Arc::make_mut(&mut detached) = second.expanded().as_ref().clone();
    assert_eq!(product(&first), first_product);

    let mut bytes = Vec::new();
    replacement.serialize_compressed(&mut bytes).unwrap();
    let decoded = AkitaVerifierSetup::<F>::deserialize_compressed_exact(&bytes, &()).unwrap();
    assert_eq!(decoded, replacement);
    assert_eq!(product(&decoded), second_product);
}
