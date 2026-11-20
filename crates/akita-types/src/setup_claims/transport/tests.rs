use super::*;
use crate::setup_claims::{SetupClaim, SetupPolynomialId, SetupQuery};
use crate::{AkitaExpandedSetup, AkitaSetupDescriptor};
use jolt_field::{CanonicalBytes, One, Ring};
mod extension;
type F = jolt_field::Prime128Offset6A23801F;
fn fixture() -> (SetupPolynomialId, Vec<u8>) {
    let setup = AkitaExpandedSetup::<F>::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 6,
            max_num_batched_polys: 1,
            num_field_elements: 64,
            setup_seed: [19; 32].into(),
        },
        crate::derive_public_matrix_prefix(64, &[19; 32].into()),
    );
    let poly = SetupPolynomialId::new::<F, F>(&setup, 64).unwrap();
    let q = SetupQuery::evaluation(vec![F::from_u64(3); 6]).unwrap();
    let mut qb = Vec::new();
    q.append_descriptor_bytes::<F>(&mut qb).unwrap();
    let s = SetupBatchStatement::new(
        poly.clone(),
        2,
        vec![SetupClaim::new(poly.clone(), q, F::one()).unwrap()],
    )
    .unwrap();
    let b = s.descriptor_bytes::<F>().unwrap();
    let prefix = b[..b.len() - 16 - qb.len() - F::NUM_BYTES].to_vec();
    (poly, prefix)
}
fn int(b: &mut Vec<u8>, n: u64) {
    b.extend_from_slice(&n.to_le_bytes());
}
fn fields(b: &mut Vec<u8>, v: &[u64]) {
    int(b, v.len() as u64);
    for n in v {
        b.extend_from_slice(&F::from_u64(*n).to_bytes_le_vec());
    }
}
fn native(p: &[u8], left: u64, right: u64, len: u64, stride: u64, weights: u64) -> Vec<u8> {
    let mut b = p.to_vec();
    int(&mut b, 2);
    int(&mut b, 1);
    int(&mut b, 1);
    int(&mut b, 1);
    fields(&mut b, &[1, 2]);
    fields(&mut b, &[2, 3, 4, 5]);
    int(&mut b, 1);
    int(&mut b, left);
    int(&mut b, right);
    fields(&mut b, &[1]);
    int(&mut b, 1);
    int(&mut b, len);
    int(&mut b, stride);
    int(&mut b, 1);
    int(&mut b, weights);
    if weights == 1 {
        fields(&mut b, &[1, 2, 3]);
    } // Incorrect dense weight length.
    if weights == 2 {
        int(&mut b, 1);
        fields(&mut b, &[1, 2]);
    } // Incorrect bit-product arity.
    b.extend_from_slice(&F::one().to_bytes_le_vec());
    b
}
#[test]
fn canonical_native_transport_preserves_the_exact_functional() {
    let (poly, p) = fixture();
    let b = native(&p, 0, 0, 4, 1, 0);
    let s = SetupBatchStatement::<F>::from_descriptor_bytes::<F>(&b, &poly).unwrap();
    assert_eq!(s.descriptor_bytes::<F>().unwrap(), b);
    let point = (2..8).map(F::from_u64).collect::<Vec<_>>();
    let mut expected = F::from_u64(0);
    for flat in 0..8 {
        let row = flat / 2;
        let mut w = F::from_u64((flat % 2 + 1) as u64);
        for (bit, r) in [2, 3, 4, 5].into_iter().map(F::from_u64).enumerate() {
            w *= if row >> bit & 1 == 1 { r } else { F::one() - r };
        }
        for (bit, r) in point.iter().copied().enumerate() {
            w *= if flat >> bit & 1 == 1 {
                r
            } else {
                F::one() - r
            };
        }
        expected += w;
    }
    assert_eq!(
        s.claims()[0]
            .query()
            .evaluate_weight(&poly, &point)
            .unwrap(),
        expected
    );
}
#[test]
fn canonical_transport_rejects_malformed_bounds_and_noncanonical_bytes() {
    let (poly, p) = fixture();
    let valid = native(&p, 0, 0, 4, 1, 0);
    for b in [
        native(&p, 32, 0, 4, 1, 0),
        native(&p, 0, 16, 4, 1, 0),
        native(&p, 0, 0, 0, 1, 0),
        native(&p, 0, 0, 4, u64::MAX, 0),
        native(&p, 0, 0, 4, 1, 1),
        native(&p, 0, 0, 4, 1, 2),
    ] {
        assert!(SetupBatchStatement::<F>::from_descriptor_bytes::<F>(&b, &poly).is_err());
    }
    for n in 0..valid.len() {
        assert!(SetupBatchStatement::<F>::from_descriptor_bytes::<F>(&valid[..n], &poly).is_err());
    }
    let mut b = valid.clone();
    b.push(0);
    assert!(SetupBatchStatement::<F>::from_descriptor_bytes::<F>(&b, &poly).is_err());
    let mut b = valid.clone();
    b[0] ^= 1;
    assert!(SetupBatchStatement::<F>::from_descriptor_bytes::<F>(&b, &poly).is_err());
    let mut b = valid.clone();
    let n = b.len();
    b[n - 16..].fill(255);
    assert!(SetupBatchStatement::<F>::from_descriptor_bytes::<F>(&b, &poly).is_err());
    let mut b = valid.clone();
    b[p.len()..p.len() + 8].copy_from_slice(&3u64.to_le_bytes());
    assert!(SetupBatchStatement::<F>::from_descriptor_bytes::<F>(&b, &poly).is_err());
    let mut b = valid;
    b[p.len() + 8..p.len() + 16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(SetupBatchStatement::<F>::from_descriptor_bytes::<F>(&b, &poly).is_err());
}
