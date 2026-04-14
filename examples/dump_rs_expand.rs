#![allow(missing_docs)]
use hachi_pcs::algebra::fields::fft::{primitive_root_of_unity, SmoothDomain};
use hachi_pcs::algebra::Prime128Offset2355;
use hachi_pcs::{CanonicalField, FieldCore, FieldSampling};
use rand::{rngs::StdRng, SeedableRng};

type F = Prime128Offset2355;
const P: u128 = 0xfffffffffffffffffffffffffffff6cd;
const P_MINUS_1: u128 = P - 1;

fn main() {
    let g = F::from_canonical_u128(2);
    let domain_size = 1470usize;
    let k = 256usize;
    let omega = primitive_root_of_unity(g, P_MINUS_1, domain_size);
    let domain = SmoothDomain::new(omega, domain_size);

    let mut rng = StdRng::seed_from_u64(0xff03);
    let base_evals: Vec<F> = (0..k).map(|_| FieldSampling::sample(&mut rng)).collect();

    let mut padded_evals = vec![F::zero(); domain_size];
    padded_evals[..k].copy_from_slice(&base_evals);
    let coeffs = domain.inverse(&padded_evals);

    let all_evals = domain.coset_forward(&coeffs, F::one());

    // Print omega
    println!("{}", omega.to_canonical_u128());
    // Print base evals
    for e in &base_evals {
        println!("{}", e.to_canonical_u128());
    }
    // Print coefficients
    for c in &coeffs {
        println!("{}", c.to_canonical_u128());
    }
    // Print extension evals [256..1280]
    for e in &all_evals[k..k + 1024] {
        println!("{}", e.to_canonical_u128());
    }
}
