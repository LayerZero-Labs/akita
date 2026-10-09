#![cfg(feature = "labinius")]

#[path = "lowered_support.rs"]
mod support;
use akita_algebra::binary::BinaryField162 as B;
use jolt_field::Zero;
use support::*;

fn bits(value: B) -> [i128; 162] {
    let bytes = value.to_bytes();
    std::array::from_fn(|s| i128::from((bytes[s / 8] >> (s % 8)) & 1))
}
fn convolution(a: &[F], b: &[F]) -> Vec<F> {
    let mut product = vec![F::zero(); a.len() + b.len() - 1];
    for (s, &a) in a.iter().enumerate() {
        for (t, &b) in b.iter().enumerate() {
            product[s + t] += a * b;
        }
    }
    product
}

#[test]
fn independent_integer_and_schoolbook_oracles_match_every_witness_coefficient() {
    for base in BASES {
        let start = std::time::Instant::now();
        let case = Case::new(base);
        let mut residual = vec![0i128; 323];
        for (r, v) in case.response.iter().enumerate() {
            // Direct equality polynomial, independent of the production expansion.
            let weight = case.claim.point[..case.setup.row_vars()]
                .iter()
                .enumerate()
                .fold(B::ONE, |acc, (bit, &z)| {
                    acc * if (r >> bit) & 1 == 1 { z } else { B::ONE + z }
                });
            for (s, b) in bits(weight).into_iter().enumerate() {
                for (t, &v) in v.iter().enumerate() {
                    residual[s + t] += b * i128::from(v);
                }
            }
        }
        for (&u, ch) in case.u.iter().zip(&case.fold) {
            for (s, u) in bits(u).into_iter().enumerate() {
                for term in ch.terms() {
                    residual[s + usize::from(term.position)] -= u * i128::from(term.coefficient);
                }
            }
        }
        let mut q = vec![0i128; 161];
        for t in (162..323).rev() {
            q[t - 162] = residual[t];
            residual[t - 162] -= q[t - 162];
            residual[t - 81] -= q[t - 162];
            residual[t] = 0;
        }
        assert_eq!(q, case.q);
        assert!(residual[..162].iter().all(|x| x % 2 == 0));
        let k = residual[..162].iter().map(|x| x / 2).collect::<Vec<_>>();
        assert_eq!(k, case.k);

        let mut packed = vec![vec![F::zero(); 648]; case.setup.m()];
        let dc = case.layout.encoding().response().digit_count();
        let offset = 32768i128;
        let mut w = vec![0u8; case.w.len()];
        for (j, element) in packed.iter_mut().enumerate() {
            for (t, destination) in element.iter_mut().enumerate() {
                let s = t / 4;
                let c = t % 4;
                let v = i128::from(case.response[4 * j + c][s]);
                let sign = if s % 2 == 0 { 1 } else { -1 };
                *destination = signed(sign * v);
                let unsigned = sign * v + if sign == 1 { offset } else { offset - 1 };
                for l in 0..dc {
                    let digit =
                        ((unsigned >> (base.bits() * l as u32)) & ((1 << base.bits()) - 1)) as u8;
                    w[l + dc * (t + 1024 * j)] = digit;
                    if sign == -1 {
                        let unflipped = (((v + offset) >> (base.bits() * l as u32))
                            & ((1 << base.bits()) - 1))
                            as u8;
                        assert_eq!(digit, ((1 << base.bits()) - 1) - unflipped);
                    }
                }
            }
        }
        assert_eq!(w, case.w);
        for row in 0..case.setup.n_a() {
            let mut residual = vec![F::zero(); 1295];
            for (j, p) in packed.iter().enumerate() {
                let product = convolution(
                    case.setup.matrix()[row * case.setup.m() + j].coefficients(),
                    p,
                );
                for (a, b) in residual.iter_mut().zip(product) {
                    *a += b;
                }
            }
            for (col, ch) in case.fold.iter().enumerate() {
                let mut embedded = vec![F::zero(); 648];
                for term in ch.terms() {
                    let s = usize::from(term.position);
                    embedded[4 * s] =
                        signed(i128::from(term.coefficient) * if s % 2 == 0 { 1 } else { -1 });
                }
                let product = convolution(
                    &embedded,
                    case.commitment.images[col * case.setup.n_a() + row].coefficients(),
                );
                for (a, b) in residual.iter_mut().zip(product) {
                    *a -= b;
                }
            }
            let mut qa = vec![F::zero(); 647];
            for t in (648..1295).rev() {
                qa[t - 648] = residual[t];
                residual[t - 648] -= qa[t - 648];
                residual[t - 324] += qa[t - 648];
                residual[t] = F::zero();
            }
            assert_eq!(qa, case.qa[row]);
            assert!(residual.iter().all(|&r| r == F::zero()));
        }
        for (e, image) in case.commitment.images.iter().enumerate() {
            assert_eq!(&case.y[e * 1024..e * 1024 + 648], image.coefficients());
            assert!(case.y[e * 1024 + 648..(e + 1) * 1024]
                .iter()
                .all(|&x| x == F::zero()));
        }
        assert!(case.y[case.commitment.images.len() * 1024..]
            .iter()
            .all(|&x| x == F::zero()));
        eprintln!("independent oracle {base:?}: {:?}", start.elapsed());
    }
}
