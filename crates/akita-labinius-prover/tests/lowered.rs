#![cfg(feature = "labinius")]

#[path = "lowered_support.rs"]
mod support;

use akita_algebra::poly::multilinear_eval;
use akita_labinius_prover::lowered::encode_witness;
use akita_labinius_verifier::lowered::{
    a_row_residual, check_lowered_clear, image_weight_mle, image_weights_dense,
    parity_row_residual, witness_weight_mle, witness_weights_dense,
};
use jolt_field::{One, Zero};
use rand::{rngs::StdRng, SeedableRng};
use support::*;

#[test]
fn honest_endpoints_and_lowered_relation_all_bases() {
    for base in BASES {
        let start = std::time::Instant::now();
        let case = Case::new(base);
        case.verify().unwrap();
        check_lowered_clear(&case.layout, &case.public(), &case.w, &case.y).unwrap();
        eprintln!("honest {base:?}: {:?}", start.elapsed());
    }
}

#[test]
fn structured_weights_match_dense_at_random_and_boolean_points() {
    let mut rng = StdRng::seed_from_u64(0x7839);
    for base in BASES {
        let start = std::time::Instant::now();
        let case = Case::new(base);
        let public = case.public();
        let kw = witness_weights_dense(&case.layout, &public).unwrap();
        let ky = image_weights_dense(&case.layout, &public).unwrap();
        for _ in 0..3 {
            let rho = random_point(&mut rng, case.layout.witness_log_len());
            assert_eq!(
                multilinear_eval(&kw, &rho).unwrap(),
                witness_weight_mle(&case.layout, &public, &case.setup, &rho).unwrap()
            );
            let ry = random_point(&mut rng, case.layout.image_log_len());
            assert_eq!(
                multilinear_eval(&ky, &ry).unwrap(),
                image_weight_mle(&case.layout, &public, &ry).unwrap()
            );
        }
        let dc = case.layout.encoding().response().digit_count();
        for (j, l, t) in [
            (0, 0, 0),
            (0, dc - 1, 647),
            (0, 0, 648),
            (1, 0, 0),
            (1, dc - 1, 647),
            (0, 0, 4),
        ] {
            let address = l + dc * (t + 1024 * j);
            let rho = boolean_point(address, case.layout.witness_log_len());
            assert_eq!(
                multilinear_eval(&kw, &rho).unwrap(),
                witness_weight_mle(&case.layout, &public, &case.setup, &rho).unwrap()
            );
            if t >= 648 {
                assert_eq!(kw[address], F::zero());
            }
        }
        for (e, t) in [
            (0, 0),
            (0, 647),
            (0, 648),
            (case.commitment.images.len() - 1, 647),
        ] {
            let address = t + 1024 * e;
            let rho = boolean_point(address, case.layout.image_log_len());
            assert_eq!(
                multilinear_eval(&ky, &rho).unwrap(),
                image_weight_mle(&case.layout, &public, &rho).unwrap()
            );
            if t >= 648 {
                assert_eq!(ky[address], F::zero());
            }
        }
        assert!(witness_weight_mle(&case.layout, &public, &case.setup, &[]).is_err());
        assert!(image_weight_mle(&case.layout, &public, &[]).is_err());
        eprintln!("weights {base:?}: {:?}", start.elapsed());
    }
}

#[test]
fn batched_relation_decomposes_into_decoded_per_row_residuals() {
    for base in BASES {
        let case = Case::new(base);
        let mut response = case.response.clone();
        response[0][0] += 1;
        response[7][161] -= 2;
        let w = encode_witness(&case.layout, &response).unwrap();
        let mut qa = case.qa.clone();
        qa[0][13] += F::one();
        let mut q = case.q.clone();
        q[17] += 1;
        let mut k = case.k.clone();
        k[19] -= 1;
        let public = case.public_with(&case.claim, &case.u, &qa, &q, &k).unwrap();
        let lhs = dot_digits(&w, &witness_weights_dense(&case.layout, &public).unwrap())
            + dot(
                &case.y,
                &image_weights_dense(&case.layout, &public).unwrap(),
            )
            - public.c_pub();
        let mut gamma_power = F::one();
        let mut residual = F::zero();
        for row in 0..case.setup.n_a() {
            residual += gamma_power
                * a_row_residual(
                    &case.layout,
                    &public,
                    &case.setup,
                    &case.commitment,
                    &response,
                    row,
                )
                .unwrap();
            gamma_power *= case.challenges.gamma;
        }
        residual += gamma_power * parity_row_residual(&case.layout, &public, &response).unwrap();
        assert_eq!(lhs, residual);
        assert_ne!(lhs, F::zero());
    }
}

#[test]
fn interval_endpoints_and_complemented_digits_are_exact() {
    for base in BASES {
        let case = Case::new(base);
        let dc = case.layout.encoding().response().digit_count();
        let b = base.bits();
        let max = (1u8 << b) - 1;
        for s in [0, 1] {
            let t = s * 4;
            for value in [case.setup.lower(), case.setup.upper()] {
                let mut response = case.response.clone();
                response[0][s] = value;
                let w = encode_witness(&case.layout, &response).unwrap();
                let digits = (0..dc)
                    .map(|l| w[case.layout.response_layout().address(0, l, t).unwrap()])
                    .collect::<Vec<_>>();
                let expected = if (s == 0 && value == case.setup.lower())
                    || (s == 1 && value == case.setup.upper())
                {
                    0
                } else {
                    max
                };
                assert!(digits.iter().all(|&digit| digit == expected));
            }
            for value in [case.setup.lower() - 1, case.setup.upper() + 1] {
                let mut response = case.response.clone();
                response[0][s] = value;
                assert!(encode_witness(&case.layout, &response).is_err());
            }
            for digit in [0, max] {
                let unsigned =
                    (0..dc).fold(0i128, |sum, l| sum + (i128::from(digit) << (b * l as u32)));
                let decoded =
                    case.layout.sigma(t).unwrap() * (unsigned - case.layout.off(t).unwrap());
                assert_eq!(
                    decoded,
                    if (s == 0 && digit == 0) || (s == 1 && digit == max) {
                        i128::from(case.setup.lower())
                    } else {
                        i128::from(case.setup.upper())
                    }
                );
            }
        }
    }
}

#[test]
fn alphabet_checked_zero_weight_tails_remain_free() {
    for base in BASES {
        let case = Case::new(base);
        let mut w = case.w.clone();
        let tail = 648 * case.layout.encoding().response().digit_count();
        w[tail] = (1 << base.bits()) - 1;
        check_lowered_clear(&case.layout, &case.public(), &w, &case.y).unwrap();
        w[tail] = 1 << base.bits();
        assert!(check_lowered_clear(&case.layout, &case.public(), &w, &case.y).is_err());
        assert!(check_lowered_clear(
            &case.layout,
            &case.public(),
            &case.w[..case.w.len() - 1],
            &case.y
        )
        .is_err());
        assert!(check_lowered_clear(
            &case.layout,
            &case.public(),
            &case.w,
            &case.y[..case.y.len() - 1]
        )
        .is_err());
    }
}
