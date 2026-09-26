use super::*;
use crate::opaque::sumcheck::prefix_lookup::test_support::*;
use jolt_field::{Field, One, Prime128Offset275, Ring, Zero};
use std::collections::HashMap;

type F = Prime128Offset275;

fn stage2_reduced_prefix_points<E: Field + Ring>() -> [PrefixPoint<E>; 2] {
    [PrefixPoint::Finite(E::one()), PrefixPoint::Infinity]
}

fn stage2_full_prefix_points<E: Field + Ring>() -> [PrefixPoint<E>; 3] {
    [
        PrefixPoint::Finite(E::zero()),
        PrefixPoint::Finite(E::one()),
        PrefixPoint::Infinity,
    ]
}

#[inline]
fn stage2_local_norm_candidate_eval<E: Field>(
    w_quad: [E; 4],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
) -> E {
    let w_eval = bilinear_eval_on_prefix_points(w_quad, x, y);
    w_eval * (w_eval + E::one())
}

#[inline]
fn stage2_local_norm_raw_eval<E: Field>(w_quad: [E; 4], x: PrefixPoint<E>, y: PrefixPoint<E>) -> E {
    let w_eval = bilinear_eval_on_prefix_points(w_quad, x, y);
    match (x, y) {
        (PrefixPoint::Finite(_), PrefixPoint::Finite(_)) => w_eval * (w_eval + E::one()),
        _ => w_eval * w_eval,
    }
}

fn vec_key(vals: &[F]) -> String {
    format!("{vals:?}")
}

fn stage2_norm_round_values(w_quad: [F; 4], tau0: F, tau1: F, r0: F) -> Vec<F> {
    let l0 = |x: F| tau0 * x + (F::one() - tau0) * (F::one() - x);
    let l1 = |y: F| tau1 * y + (F::one() - tau1) * (F::one() - y);
    let q = |x: F, y: F| {
        let w = bilinear_eval(w_quad, x, y);
        w * (w + F::one())
    };

    let mut out = Vec::new();
    for x in 0..=3u64 {
        let x = F::from_u64(x);
        out.push(l0(x) * (l1(F::zero()) * q(x, F::zero()) + l1(F::one()) * q(x, F::one())));
    }
    for y in 0..=3u64 {
        let y = F::from_u64(y);
        out.push(l1(y) * l0(r0) * q(r0, y));
    }
    out
}

fn tensor_values<E: Field, const NX: usize, const NY: usize>(
    xs: [PrefixPoint<E>; NX],
    ys: [PrefixPoint<E>; NY],
    mut eval: impl FnMut(PrefixPoint<E>, PrefixPoint<E>) -> E,
) -> Vec<E> {
    let mut out = Vec::with_capacity(NX * NY);
    for &x in &xs {
        for &y in &ys {
            out.push(eval(x, y));
        }
    }
    out
}

#[test]
fn stage2_b8_norm_lookup_table_matches_raw_evals() {
    let points = stage2_full_prefix_points::<F>();
    for w00 in -4i64..=3 {
        for w10 in -4i64..=3 {
            for w01 in -4i64..=3 {
                for w11 in -4i64..=3 {
                    let lookup =
                        &STAGE2_B8_NORM_LOOKUP_TABLE[stage2_b8_lookup_index_from_digits([
                            (w00 + 4) as usize,
                            (w10 + 4) as usize,
                            (w01 + 4) as usize,
                            (w11 + 4) as usize,
                        ])];
                    let quad = [
                        F::from_i64(w00),
                        F::from_i64(w10),
                        F::from_i64(w01),
                        F::from_i64(w11),
                    ];
                    for point_idx in 0..STAGE2_PREFIX_POINT_COUNT {
                        let x = points[point_idx / 3];
                        let y = points[point_idx % 3];
                        assert_eq!(
                            F::from_i64(lookup[point_idx]),
                            stage2_local_norm_raw_eval(quad, x, y),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn stage2_norm_histogram_matches_local_round_messages() {
    let tau0 = F::from_u64(1_234_567);
    let tau1 = F::from_u64(7_654_321);
    let batching_coeff = F::from_u64(424_242);
    let r0 = F::from_u64(98_765);
    for b in [4usize, 8] {
        let bits = b.trailing_zeros();
        let half = (b / 2) as i64;
        let mut histogram = vec![F::zero(); b.pow(4)];
        let mut round0 = [F::zero(); 4];
        let mut round1 = [F::zero(); 4];
        for sample in 0..97u64 {
            let digits: [usize; 4] = std::array::from_fn(|i| {
                ((sample * (7 + 2 * i as u64) + i as u64) % b as u64) as usize
            });
            let class = digits
                .iter()
                .enumerate()
                .fold(0usize, |class, (i, &d)| class | (d << (i as u32 * bits)));
            let weight = F::from_u64(1_000 + 31 * sample);
            histogram[class] += weight;
            let quad = digits.map(|d| F::from_i64(d as i64 - half));
            let values = stage2_norm_round_values(quad, tau0, tau1, r0);
            for point in 0..4 {
                round0[point] += weight * values[point];
                round1[point] += weight * values[4 + point];
            }
        }
        let cache =
            Stage2PrefixCache::from_norm_histogram(&histogram, b, tau0, tau1, batching_coeff);
        let poly0 = cache.round0_norm_poly();
        let poly1 = cache.round1_norm_poly(r0);
        for point in 0..4u64 {
            let x = F::from_u64(point);
            assert_eq!(
                poly0.evaluate(x),
                batching_coeff * round0[point as usize],
                "b={b} round 0 at {point}"
            );
            assert_eq!(
                poly1.evaluate(x),
                batching_coeff * round1[point as usize],
                "b={b} round 1 at {point}"
            );
        }
    }
}

#[test]
fn stage2_norm_reduced_domain_has_round_message_collision() {
    let reduced = stage2_reduced_prefix_points::<F>();
    let tau0 = F::from_u64(7);
    let tau1 = F::from_u64(11);
    let r0 = F::from_u64(13);

    let mut seen: HashMap<String, Vec<F>> = HashMap::new();
    let mut found_collision = false;
    for w00 in -4i64..=3 {
        for w10 in -4i64..=3 {
            for w01 in -4i64..=3 {
                for w11 in -4i64..=3 {
                    let quad = [
                        F::from_i64(w00),
                        F::from_i64(w10),
                        F::from_i64(w01),
                        F::from_i64(w11),
                    ];
                    let storage = tensor_values(reduced, reduced, |x, y| {
                        stage2_local_norm_candidate_eval(quad, x, y)
                    });
                    let target = stage2_norm_round_values(quad, tau0, tau1, r0);
                    let key = vec_key(&storage);
                    if let Some(existing) = seen.get(&key) {
                        if *existing != target {
                            found_collision = true;
                            break;
                        }
                    } else {
                        seen.insert(key, target);
                    }
                }
                if found_collision {
                    break;
                }
            }
            if found_collision {
                break;
            }
        }
        if found_collision {
            break;
        }
    }
    assert!(
        found_collision,
        "reduced stage-2 norm domain should not uniquely determine local round messages"
    );
}

#[test]
fn stage2_norm_full_domain_matches_local_round_messages() {
    let full = stage2_full_prefix_points::<F>();
    let tau0 = F::from_u64(7);
    let tau1 = F::from_u64(11);
    let r0 = F::from_u64(13);

    let mut seen: HashMap<String, Vec<F>> = HashMap::new();
    for w00 in -4i64..=3 {
        for w10 in -4i64..=3 {
            for w01 in -4i64..=3 {
                for w11 in -4i64..=3 {
                    let quad = [
                        F::from_i64(w00),
                        F::from_i64(w10),
                        F::from_i64(w01),
                        F::from_i64(w11),
                    ];
                    let storage =
                        tensor_values(full, full, |x, y| stage2_local_norm_raw_eval(quad, x, y));
                    let target = stage2_norm_round_values(quad, tau0, tau1, r0);
                    let key = vec_key(&storage);
                    if let Some(existing) = seen.get(&key) {
                        assert_eq!(
                            existing, &target,
                            "full stage-2 norm domain lost information for a compact quad"
                        );
                    } else {
                        seen.insert(key, target);
                    }
                }
            }
        }
    }
}
