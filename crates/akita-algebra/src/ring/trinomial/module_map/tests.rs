#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

use super::*;
use crate::ring::trinomial::{
    pack_scalar_components, signed::signed_trinomial_product, MinusTrinomial, PlusTrinomial,
    TrinomialRing,
};
use jolt_field::{Prime128Offset275 as F, Ring};

fn embed(c: [i128; 162]) -> [i128; 648] {
    pack_signed_components(&[c, [0; 162], [0; 162], [0; 162]]).unwrap()
}

fn field(x: i128) -> F {
    if x < 0 {
        -F::from_u128(x.unsigned_abs())
    } else {
        F::from_u128(x as u128)
    }
}

#[test]
fn module_maps_agree_with_existing_packing_and_round_trip() {
    let components: [[i128; 162]; 4] = std::array::from_fn(|a| {
        std::array::from_fn(|s| (s as i128 * 17 + a as i128 * 31) % 79 - 39)
    });
    let packed = pack_signed_components(&components).unwrap();
    assert_eq!(unpack_signed_components(&packed).unwrap(), components);
    let rings: Vec<_> = components
        .iter()
        .map(|c| TrinomialRing::<F, 162, PlusTrinomial>::from_coefficients(c.map(field)).unwrap())
        .collect();
    let existing = pack_scalar_components::<F, 162, 648, MinusTrinomial>(&rings).unwrap();
    assert_eq!(existing.coefficients(), &packed.map(field));
    // Every signed basis vector, including the top coordinate of each component.
    for t in 0..648 {
        let mut r = [0; 648];
        r[t] = -1;
        assert_eq!(
            pack_signed_components(&unpack_signed_components(&r).unwrap()).unwrap(),
            r
        );
    }
}

#[test]
fn iota_is_a_ring_homomorphism_and_scalar_action_is_componentwise() {
    let c = std::array::from_fn(|s| (s as i128 * 11) % 17 - 8);
    let components: [[i128; 162]; 4] =
        std::array::from_fn(|a| std::array::from_fn(|s| (s as i128 * 7 + a as i128) % 13 - 6));
    let packed = pack_signed_components(&components).unwrap();
    let actual = signed_trinomial_product::<648, MinusTrinomial>(&embed(c), &packed).unwrap();
    let expected: [[i128; 162]; 4] = std::array::from_fn(|a| {
        signed_trinomial_product::<162, PlusTrinomial>(&c, &components[a]).unwrap()
    });
    assert_eq!(unpack_signed_components(&actual).unwrap(), expected);
    let d = components[0];
    let cd = signed_trinomial_product::<162, PlusTrinomial>(&c, &d).unwrap();
    assert_eq!(
        signed_trinomial_product::<648, MinusTrinomial>(&embed(c), &embed(d)).unwrap(),
        embed(cd)
    );
    let sum = std::array::from_fn(|s| c[s] + d[s]);
    assert_eq!(
        embed(sum),
        std::array::from_fn(|t| embed(c)[t] + embed(d)[t])
    );
    let mut one = [0; 162];
    one[0] = 1;
    assert_eq!(embed(one)[0], 1);
    assert!(embed(one)[1..].iter().all(|&x| x == 0));
    // This checks the defining identity Phi(-Y^4) = Psi(Y) directly.
    let mut phi_substitution = [0i128; 649];
    phi_substitution[648] = 1;
    phi_substitution[324] = -1;
    phi_substitution[0] = 1;
    assert_eq!(
        super::super::signed::signed_trinomial_remainder::<648, MinusTrinomial>(&phi_substitution)
            .unwrap(),
        [0; 648]
    );
}

#[cfg(feature = "labinius-binary")]
#[test]
fn binary_scalar_reduction_is_a_ring_homomorphism() {
    use crate::binary::BinaryField162;
    let a: [i128; 162] = std::array::from_fn(|i| (i as i128 * 31) % 29 - 14);
    let b: [i128; 162] = std::array::from_fn(|i| (i as i128 * 17) % 23 - 11);
    let product = signed_trinomial_product::<162, PlusTrinomial>(&a, &b).unwrap();
    assert_eq!(
        scalar_to_binary(&product).unwrap(),
        scalar_to_binary(&a).unwrap() * scalar_to_binary(&b).unwrap()
    );
    assert_eq!(
        scalar_to_binary(&std::array::from_fn::<_, 162, _>(|i| a[i] + b[i])).unwrap(),
        scalar_to_binary(&a).unwrap() + scalar_to_binary(&b).unwrap()
    );
    for s in 0..162 {
        let mut monomial = [0; 162];
        monomial[s] = -1;
        let mut words = [0; 3];
        words[s / 64] = 1 << (s % 64);
        assert_eq!(
            scalar_to_binary(&monomial).unwrap(),
            BinaryField162::from_words(words).unwrap()
        );
    }
    assert!(scalar_to_binary(&a[..161]).is_err());
}

#[test]
fn malformed_sizes_and_sign_overflow_are_rejected() {
    assert!(pack_signed_components(&[[0; 162]; 3]).is_err());
    assert!(unpack_signed_components(&[0; 647]).is_err());
    let mut components = [[0; 162]; 4];
    components[3][1] = i128::MIN;
    assert!(pack_signed_components(&components).is_err());
    let mut packed = [0; 648];
    packed[7] = i128::MIN;
    assert!(unpack_signed_components(&packed).is_err());
    packed[7] = 0;
    packed[0] = i128::MIN;
    assert_eq!(
        pack_signed_components(&unpack_signed_components(&packed).unwrap()).unwrap(),
        packed
    );
}
