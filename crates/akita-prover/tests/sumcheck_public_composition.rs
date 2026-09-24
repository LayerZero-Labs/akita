#![allow(missing_docs)]

use akita_sumcheck::advance_eq_factored_claim;
use jolt_field::{Prime128Offset275 as F, Ring};
use jolt_poly::NormalizedPoly;

#[test]
fn normalized_claim_advance_is_public() {
    let claim = F::from_u64(17);
    let tau = F::from_u64(5);
    let challenge = F::from_u64(11);
    let poly = NormalizedPoly::new(vec![F::from_u64(3), F::from_u64(7)]);
    let constant = claim - tau * F::from_u64(10);

    assert_eq!(
        advance_eq_factored_claim(claim, tau, &poly, challenge),
        constant + F::from_u64(3) * challenge + F::from_u64(7) * challenge * challenge
    );
}
