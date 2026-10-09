#![cfg(feature = "labinius")]

#[path = "lowered_support.rs"]
mod support;

use akita_algebra::{binary::BinaryField162, poly::multilinear_eval, MinusTrinomial};
use akita_labinius_prover::lowered::encode_witness;
use akita_labinius_verifier::{
    lowered::{
        a_row_residual, check_lowered_clear, image_weight_mle, image_weights_dense,
        parity_row_residual, witness_weight_mle, witness_weights_dense, LoweredChallenges,
    },
    source::{equality_weights, scalar_from_binary},
    AdmittedRootSetup, BinaryEvaluationClaim,
};
use akita_types::proof::AkitaSetupSeed;
use jolt_field::{One, Ring, Zero};
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

/// Compare the cached public data with definitions that share no code with it.
///
/// The matrix contraction and the binary-row projections are recomputed as
/// plain power sums, the response weights are rebuilt entry by entry, and the
/// offset part of the public constant is isolated by zeroing every other
/// input. Returns the honest constant and both dense weight tables per base.
fn public_data_against_direct_definitions() -> Vec<(F, Vec<F>, Vec<F>)> {
    let power_sum = |coefficients: &[F], point: F| {
        let mut power = F::one();
        coefficients.iter().fold(F::zero(), |sum, &coefficient| {
            let term = coefficient * power;
            power *= point;
            sum + term
        })
    };
    let powers = |point: F, len: usize| {
        let mut power = F::one();
        (0..len)
            .map(|_| {
                let current = power;
                power *= point;
                current
            })
            .collect::<Vec<_>>()
    };
    BASES
        .into_iter()
        .map(|base| {
            let case = Case::new(base);
            let layout = &case.layout;
            let LoweredChallenges { alpha, xi, gamma } = case.challenges;
            let (n_a, m, k) = (layout.n_a(), layout.m(), layout.k());
            assert!(m > 1 && k > 1);
            let alpha_powers = powers(alpha, layout.degree());
            let xi_powers = powers(xi, layout.degree().div_ceil(k));
            let gamma_powers = powers(gamma, n_a + 1);
            let abar = (0..m)
                .map(|j| {
                    (0..n_a).fold(F::zero(), |sum, i| {
                        let element = &case.setup.matrix()[i * m + j];
                        sum + gamma_powers[i] * power_sum(element.coefficients(), alpha)
                    })
                })
                .collect::<Vec<_>>();
            let rows = equality_weights(&case.claim.point[..case.setup.row_vars()])
                .unwrap()
                .into_iter()
                .map(|weight| {
                    power_sum(scalar_from_binary::<F>(weight).unwrap().coefficients(), xi)
                })
                .collect::<Vec<_>>();
            assert_eq!(rows.len(), layout.scalar_rows());
            let weight = |j: usize, t: usize| {
                abar[j] * alpha_powers[t]
                    + gamma_powers[n_a]
                        * signed(layout.sigma(t).unwrap())
                        * xi_powers[t / k]
                        * rows[j * k + t % k]
            };

            let public = case.public();
            let kw = witness_weights_dense(layout, &public).unwrap();
            let ky = image_weights_dense(layout, &public).unwrap();
            let digit_count = layout.encoding().response().digit_count();
            let digit_powers = powers(
                F::from_u64(1 << layout.encoding().base().bits()),
                digit_count,
            );
            let mut expected = vec![F::zero(); kw.len()];
            let mut offset = F::zero();
            for j in 0..m {
                for t in 0..layout.degree() {
                    assert_eq!(public.coefficient_weight(j, t).unwrap(), weight(j, t));
                    offset += signed(layout.off(t).unwrap()) * weight(j, t);
                    let address = digit_count * (t + layout.padded_coefficients() * j);
                    for (l, &digit) in digit_powers.iter().enumerate() {
                        expected[address + l] = digit * weight(j, t);
                    }
                }
            }
            assert_eq!(kw, expected);
            assert!(public.coefficient_weight(m, 0).is_err());
            assert!(public.coefficient_weight(0, layout.degree()).is_err());

            let zero_rows = case
                .qa
                .iter()
                .map(|row| vec![F::zero(); row.len()])
                .collect::<Vec<_>>();
            // The zero partial evaluations are consistent with a zero claim at
            // the same point, so the binary rows are unchanged.
            let zero_claim = BinaryEvaluationClaim {
                point: case.claim.point.clone(),
                value: BinaryField162::ZERO,
            };
            let offset_only = case
                .public_with(
                    &zero_claim,
                    &vec![BinaryField162::ZERO; case.u.len()],
                    &zero_rows,
                    &vec![0; case.q.len()],
                    &vec![0; case.k.len()],
                )
                .unwrap();
            assert_eq!(offset_only.c_pub(), offset);
            (public.c_pub(), kw, ky)
        })
        .collect()
}

#[test]
fn public_data_matches_direct_definitions() {
    public_data_against_direct_definitions();
}

#[cfg(feature = "parallel")]
#[test]
fn public_data_is_the_same_in_one_and_three_thread_pools() {
    let [one, three] = [1, 3].map(|threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(public_data_against_direct_definitions)
    });
    assert_eq!(one, three);
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

#[test]
fn layout_table_sizes_are_the_admitted_encoding_sizes() {
    for base in BASES {
        let case = Case::new(base);
        let encoding = case.shape.derive_encoding(base).unwrap();
        assert_eq!(
            case.layout.padded_coefficients(),
            encoding.padded_coefficient_len()
        );
        assert_eq!(case.layout.witness_len(), encoding.response_table_len());
        assert_eq!(
            case.layout.witness_log_len(),
            encoding.response_table_log_len()
        );
        assert_eq!(case.layout.image_log_len(), encoding.image_table_log_len());
        assert_eq!(case.w.len(), encoding.response_table_len());
        assert_eq!(case.y.len(), 1 << encoding.image_table_log_len());
    }
}

#[test]
fn seed_derived_admitted_setup_satisfies_the_lowered_relation() {
    let seed = AkitaSetupSeed::shake256_paged_v1([0x5a; 32]);
    let admitted =
        AdmittedRootSetup::<F, 648, MinusTrinomial>::derive(PROFILE, 4, 1, 128, seed).unwrap();
    for base in BASES {
        let case = Case::with_setup(base, admitted.shape().clone(), admitted.setup().clone());
        case.verify().unwrap();
        check_lowered_clear(&case.layout, &case.public(), &case.w, &case.y).unwrap();
        let mut w = case.w.clone();
        w[0] ^= 1;
        assert!(check_lowered_clear(&case.layout, &case.public(), &w, &case.y).is_err());
    }
}
