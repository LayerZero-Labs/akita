//! Base- and extension-field arithmetic as configured by Akita.
//!
//! Base fields are checked against a `num-bigint` oracle. Extension fields,
//! whose defining polynomials belong to the pinned `jolt-field` dependency,
//! are checked by field laws and by consistency with their base-field
//! embedding, so this target restates no extension formula. Delayed-reduction
//! accumulators are checked against their documented exactness contracts,
//! including the unit-scale commit accumulator at its declared headroom.

use crate::gen::{self, modulus, Domain};
use crate::input::Reader;
use crate::stats;
use akita_config::proof_optimized::{fp128, fp32, fp64};
use jolt_field::{
    CanonicalBytes, CanonicalEncoding, ExtField, Field, Fold, MulBaseUnreduced, PseudoMersenne,
    Unreduced, WithCommitAccumulator, Zero,
};
use num_bigint::{BigInt, BigUint};

pub fn run(data: &[u8]) {
    let mut reader = Reader::new(data);
    match reader.u8() % 6 {
        0 => base::<fp32::Field>(&mut reader),
        1 => base::<fp64::Field>(&mut reader),
        2 => base::<fp128::Field>(&mut reader),
        3 => extension::<fp32::Field, fp32::ExtensionField>(&mut reader),
        4 => extension::<fp64::Field, fp64::ExtensionField>(&mut reader),
        _ => extension::<fp128::Field, fp128::Field>(&mut reader),
    }
}

fn big<F: Field + CanonicalEncoding>(value: F) -> BigUint {
    BigUint::from(value.to_u128_checked().expect("Akita fields fit in u128"))
}

fn reduce_int<F: Field + CanonicalEncoding>(value: &BigInt) -> BigUint {
    let q = BigInt::from(modulus::<F>());
    let reduced = ((value % &q) + &q) % &q;
    reduced.to_biguint().expect("nonnegative after reduction")
}

fn base<F>(reader: &mut Reader<'_>)
where
    F: Field
        + CanonicalEncoding
        + CanonicalBytes
        + PseudoMersenne
        + MulBaseUnreduced<F>
        + Fold
        + WithCommitAccumulator,
{
    stats::count("field_base");
    let q = BigUint::from(modulus::<F>());
    let a: F = gen::scalar(reader, Domain::Full);
    let b: F = gen::scalar(reader, Domain::Full);
    let (ab, bb) = (big(a), big(b));
    assert!(
        ab < q && bb < q,
        "canonical representatives must be reduced"
    );

    assert_eq!(big(a + b), (&ab + &bb) % &q, "add");
    assert_eq!(big(a - b), (&ab + &q - &bb) % &q, "sub");
    assert_eq!(big(a * b), (&ab * &bb) % &q, "mul");
    assert_eq!(big(-a), (&q - &ab) % &q, "neg");
    assert_eq!(big(a.square()), (&ab * &ab) % &q, "square");
    match a.inverse() {
        Some(inverse) => assert_eq!(a * inverse, F::one(), "inverse"),
        None => assert_eq!(a, F::zero(), "only zero lacks an inverse"),
    }

    let raw = reader.u128();
    assert_eq!(
        big(F::from_u128_reduced(raw)),
        BigUint::from(raw) % &q,
        "from_u128_reduced"
    );
    assert_eq!(
        F::from_u128_checked(raw).is_some(),
        BigUint::from(raw) < q,
        "from_u128_checked accepts exactly the canonical range"
    );
    let signed = raw as i128;
    assert_eq!(
        big(F::from_i128(signed)),
        reduce_int::<F>(&BigInt::from(signed)),
        "from_i128"
    );
    let small = reader.u64();
    assert_eq!(big(a.mul_u64(small)), (&ab * small) % &q, "mul_u64");
    assert_eq!(
        big(a.mul_i64(small as i64)),
        reduce_int::<F>(&(BigInt::from(ab.clone()) * (small as i64))),
        "mul_i64"
    );
    assert_eq!(big(a.mul_u128(raw)), (&ab * raw) % &q, "mul_u128");
    let shift = usize::from(reader.u8());
    assert_eq!(big(a.mul_pow_2(shift)), (&ab << shift) % &q, "mul_pow_2");

    let bytes = a.to_bytes_le_vec();
    assert_eq!(
        F::from_bytes_le_checked(&bytes),
        Some(a),
        "canonical bytes round trip"
    );
    let fuzz_bytes = reader.take(bytes.len()).to_vec();
    if fuzz_bytes.len() == bytes.len() {
        let as_int = BigUint::from_bytes_le(&fuzz_bytes);
        let checked = F::from_bytes_le_checked(&fuzz_bytes);
        assert_eq!(
            checked.is_some(),
            as_int < q,
            "from_bytes_le_checked rejects noncanonical bytes"
        );
        if let Some(value) = checked {
            assert_eq!(big(value), as_int, "from_bytes_le_checked value");
        }
        assert_eq!(
            big(F::from_bytes_le_reduced(&fuzz_bytes)),
            &as_int % &q,
            "from_bytes_le_reduced"
        );
    }

    unreduced_sums::<F, F>(reader);
    fold_matches_definition::<F>(a, b, gen::scalar(reader, Domain::Full));
    commit_accumulator::<F>(reader);
}

/// `reduce(Σ mul_unreduced) = Σ mul` whenever the accumulator claims exactness.
fn unreduced_sums<F, E>(reader: &mut Reader<'_>)
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + Unreduced + MulBaseUnreduced<F>,
{
    let terms = 1 + usize::from(reader.u8() % 32);
    let mut product = <E as Unreduced>::Product::zero();
    let mut base_product = <E as Unreduced>::Product::zero();
    let mut small = <E as Unreduced>::SmallProduct::zero();
    let (mut expected, mut expected_base, mut expected_small) = (E::zero(), E::zero(), E::zero());
    for _ in 0..terms {
        let x = gen::ext_scalar::<F, E>(reader);
        let y = gen::ext_scalar::<F, E>(reader);
        let s: F = gen::scalar(reader, Domain::Full);
        let k = reader.u64();
        product += x.mul_unreduced(y);
        base_product += x.mul_base_unreduced(s);
        small += x.mul_u64_unreduced(k);
        expected += x * y;
        expected_base += x.mul_base(s);
        expected_small += x.mul_u64(k);
    }
    assert_eq!(
        E::reduce_small_product(small),
        expected_small,
        "small-product accumulation"
    );
    if E::SUM_IS_EXACT {
        assert_eq!(
            E::reduce_product(product),
            expected,
            "exact product accumulation"
        );
        assert_eq!(
            E::reduce_product(base_product),
            expected_base,
            "exact base-product accumulation"
        );
    }
}

fn fold_matches_definition<E: Fold>(even: E, odd: E, r: E) {
    let context = E::precompute(r);
    assert_eq!(
        E::fold_one(&context, even, odd),
        even + r * (odd - even),
        "Fold::fold_one"
    );
}

/// Unit-scale wide lanes stay exact up to `MAX_COMMIT_ACCUMULATIONS` terms.
fn commit_accumulator<F>(reader: &mut Reader<'_>)
where
    F: Field + CanonicalEncoding + WithCommitAccumulator,
{
    let headroom = F::MAX_COMMIT_ACCUMULATIONS;
    let terms = match reader.u8() % 4 {
        0 => headroom,
        1 => headroom.saturating_sub(1),
        _ => 1 + reader.u16() as usize % headroom.max(1),
    };
    let mode = reader.u8();
    let value: F = gen::scalar(reader, Domain::Full);
    let mut wide = <F as Unreduced>::Wide::zero();
    let mut expected = F::zero();
    for index in 0..terms {
        let term = match mode % 3 {
            0 => value,
            1 => -F::one(),
            _ => value + F::from_u64(index as u64),
        };
        wide += <F as Unreduced>::Wide::from(term);
        expected += term;
    }
    assert_eq!(
        F::reduce_wide(wide),
        expected,
        "commit accumulator within declared headroom"
    );
    let scale = i32::from(reader.u16() as i16);
    assert_eq!(
        F::reduce_wide(value.scale_wide(scale)),
        value * F::from_i64(i64::from(scale)),
        "scale_wide by an i16 scalar"
    );
    stats::count("field_commit_accumulator");
}

fn extension<F, E>(reader: &mut Reader<'_>)
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + Unreduced + MulBaseUnreduced<F> + Fold,
{
    stats::count("field_extension");
    let a = gen::ext_scalar::<F, E>(reader);
    let b = gen::ext_scalar::<F, E>(reader);
    let c = gen::ext_scalar::<F, E>(reader);
    let s: F = gen::scalar(reader, Domain::Full);
    let t: F = gen::scalar(reader, Domain::Full);

    assert_eq!(a * b, b * a, "commutativity");
    assert_eq!((a * b) * c, a * (b * c), "associativity");
    assert_eq!(a * (b + c), a * b + a * c, "distributivity");
    assert_eq!(a - a, E::zero(), "additive inverse");
    assert_eq!(a.square(), a * a, "square");
    match a.inverse() {
        Some(inverse) => assert_eq!(a * inverse, E::one(), "inverse"),
        None => assert_eq!(a, E::zero(), "only zero lacks an inverse"),
    }

    assert_eq!(
        a.mul_base(s),
        a * E::lift_base(s),
        "mul_base agrees with lifted multiply"
    );
    assert_eq!(
        E::lift_base(s) * E::lift_base(t),
        E::lift_base(s * t),
        "lift_base is a ring map"
    );
    assert_eq!(
        E::lift_base(s) + E::lift_base(t),
        E::lift_base(s + t),
        "lift_base is additive"
    );
    assert_eq!(
        E::from_base_slice(&a.to_base_vec()),
        a,
        "coordinate round trip"
    );
    for index in 0..E::DEGREE {
        assert_eq!(
            (a + b).base_coefficient(index),
            a.base_coefficient(index) + b.base_coefficient(index),
            "addition is coordinatewise"
        );
        assert_eq!(
            a.mul_base(s).base_coefficient(index),
            a.base_coefficient(index) * s,
            "mul_base is coordinatewise"
        );
    }

    let power = usize::from(reader.u8()) % E::DEGREE.max(1);
    assert_eq!(
        (a * b).frobenius_pow(power),
        a.frobenius_pow(power) * b.frobenius_pow(power),
        "Frobenius is multiplicative"
    );
    assert_eq!(
        a.frobenius_pow(E::DEGREE),
        a,
        "Frobenius has order dividing the degree"
    );
    assert_eq!(
        a.frobenius_pow(power).frobenius_inv_pow(power),
        a,
        "inverse Frobenius"
    );
    assert_eq!(
        E::lift_base(s).frobenius_pow(power),
        E::lift_base(s),
        "Frobenius fixes the base field"
    );

    unreduced_sums::<F, E>(reader);
    fold_matches_definition::<E>(a, b, c);
}
