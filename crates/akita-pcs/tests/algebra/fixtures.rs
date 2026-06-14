use num_bigint::BigUint;
use rand::{rngs::StdRng, SeedableRng};

use akita_field::{
    CanonicalField, FieldCore, Fp32, FpExt2, FpExt2Config, Invertible, PseudoMersenneField,
};

pub(super) fn rand_u128<R: rand_core::RngCore>(rng: &mut R) -> u128 {
    let lo = rng.next_u64() as u128;
    let hi = rng.next_u64() as u128;
    lo | (hi << 64)
}

fn biguint_to_u128(x: &num_bigint::BigUint) -> u128 {
    let mut bytes = x.to_bytes_le();
    bytes.resize(16, 0);
    let mut arr = [0u8; 16];
    arr.copy_from_slice(&bytes[..16]);
    u128::from_le_bytes(arr)
}

pub(super) fn big_mul_mod_u128(a: u128, b: u128, p: u128) -> u128 {
    let n = BigUint::from(a) * BigUint::from(b);
    let r = n % BigUint::from(p);
    biguint_to_u128(&r)
}

pub(super) fn check_solinas_prime<
    S: CanonicalField + FieldCore + Invertible + PseudoMersenneField + std::fmt::Debug,
>(
    p: u128,
    iters: usize,
    seed: u64,
) {
    assert_eq!(<S as PseudoMersenneField>::MODULUS_BITS, 128);
    assert_eq!(
        <S as PseudoMersenneField>::MODULUS_OFFSET,
        0u128.wrapping_sub(p)
    );
    assert_eq!(std::mem::size_of::<S>(), 16);

    let mut rng = StdRng::seed_from_u64(seed);

    for _ in 0..iters {
        let a_raw = rand_u128(&mut rng);
        let b_raw = rand_u128(&mut rng);

        let a = S::from_canonical_u128_reduced(a_raw);
        let b = S::from_canonical_u128_reduced(b_raw);

        assert!(a.to_canonical_u128() < p);
        assert!(b.to_canonical_u128() < p);

        assert_eq!(a + S::zero(), a);
        assert_eq!(a - S::zero(), a);
        assert_eq!(a + (-a), S::zero());

        assert_eq!(a * S::one(), a);

        let aa = a.to_canonical_u128();
        let bb = b.to_canonical_u128();
        let got_mul = (a * b).to_canonical_u128();
        let exp_mul = big_mul_mod_u128(aa, bb, p);
        assert_eq!(got_mul, exp_mul);

        let got_sqr = (a * a).to_canonical_u128();
        let exp_sqr = big_mul_mod_u128(aa, aa, p);
        assert_eq!(got_sqr, exp_sqr);

        let inv = a.inv_or_zero();
        if a.is_zero() {
            assert_eq!(inv, S::zero());
        } else {
            assert_eq!(a * inv, S::one());
            assert_eq!(a.inverse().unwrap(), inv);
        }
    }
}

pub(super) struct NR;
impl FpExt2Config<Fp32<251>> for NR {
    fn non_residue() -> Fp32<251> {
        -Fp32::<251>::one()
    }
}
