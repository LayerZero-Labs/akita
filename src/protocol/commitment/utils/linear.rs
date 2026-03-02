//! Linear algebra helpers for ring commitment.

use crate::algebra::ntt::{MontCoeff, PrimeWidth};
use crate::algebra::{CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing};
use crate::cfg_iter;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::{CanonicalField, FieldCore};

use super::crt_ntt::NttMatrixCache;
#[cfg(test)]
use super::crt_ntt::{select_crt_ntt_params, ProtocolCrtNttParams};

#[cfg(test)]
pub(crate) fn mat_vec_mul_unchecked<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = Vec::with_capacity(mat.len());
    for row in mat {
        debug_assert_eq!(row.len(), vec.len());
        let mut acc = CyclotomicRing::<F, D>::zero();
        for (a, x) in row.iter().zip(vec.iter()) {
            acc += *a * *x;
        }
        out.push(acc);
    }
    out
}

#[inline]
fn accumulate_pointwise_product_into<W: PrimeWidth, const K: usize, const D: usize>(
    acc: &mut CyclotomicCrtNtt<W, K, D>,
    lhs: &CyclotomicCrtNtt<W, K, D>,
    rhs: &CyclotomicCrtNtt<W, K, D>,
    params: &CrtNttParamSet<W, K, D>,
) {
    for k in 0..K {
        let prime = params.primes[k];
        let acc_limb = &mut acc.limbs[k];
        let lhs_limb = &lhs.limbs[k];
        let rhs_limb = &rhs.limbs[k];
        for ((acc_coeff, lhs_coeff), rhs_coeff) in acc_limb
            .iter_mut()
            .zip(lhs_limb.iter())
            .zip(rhs_limb.iter())
        {
            let prod = prime.mul(*lhs_coeff, *rhs_coeff);
            let sum = MontCoeff::from_raw(acc_coeff.raw().wrapping_add(prod.raw()));
            *acc_coeff = prime.reduce_range(sum);
        }
    }
}

#[cfg(test)]
fn precompute_dense_mat_ntt_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicCrtNtt<W, K, D>>> {
    mat.iter()
        .map(|row| {
            row.iter()
                .map(|a| CyclotomicCrtNtt::from_ring_with_params(a, params))
                .collect()
        })
        .collect()
}

#[cfg(test)]
fn mat_vec_mul_dense_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let ntt_vec: Vec<CyclotomicCrtNtt<W, K, D>> = vec
        .iter()
        .map(|v| CyclotomicCrtNtt::from_ring_with_params(v, params))
        .collect();

    mat.iter()
        .map(|row| {
            debug_assert_eq!(row.len(), ntt_vec.len());
            let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
            for (a, x_ntt) in row.iter().zip(ntt_vec.iter()) {
                let a_ntt = CyclotomicCrtNtt::from_ring_with_params(a, params);
                accumulate_pointwise_product_into(&mut acc, &a_ntt, x_ntt, params);
            }
            acc.to_ring_with_params(params)
        })
        .collect()
}

#[cfg(test)]
fn mat_vec_mul_dense_many_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vecs: &[Vec<CyclotomicRing<F, D>>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let ntt_mat = precompute_dense_mat_ntt_with_params(mat, params);
    vecs.iter()
        .map(|vec| {
            let ntt_vec: Vec<CyclotomicCrtNtt<W, K, D>> = vec
                .iter()
                .map(|v| CyclotomicCrtNtt::from_ring_with_params(v, params))
                .collect();

            ntt_mat
                .iter()
                .map(|row_ntt| {
                    debug_assert_eq!(row_ntt.len(), ntt_vec.len());
                    let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
                    for (a_ntt, x_ntt) in row_ntt.iter().zip(ntt_vec.iter()) {
                        accumulate_pointwise_product_into(&mut acc, a_ntt, x_ntt, params);
                    }
                    acc.to_ring_with_params(params)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn mat_vec_mul_crt_ntt<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    let params = select_crt_ntt_params::<F, D>()?;
    let out = match &params {
        ProtocolCrtNttParams::Q32(p) => mat_vec_mul_dense_with_params(mat, vec, p),
        ProtocolCrtNttParams::Q64(p) => mat_vec_mul_dense_with_params(mat, vec, p),
        ProtocolCrtNttParams::Q128(p) => mat_vec_mul_dense_with_params(mat, vec, p),
    };
    Ok(out)
}

#[cfg(test)]
pub(crate) fn mat_vec_mul_crt_ntt_many<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vecs: &[Vec<CyclotomicRing<F, D>>],
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError> {
    let params = select_crt_ntt_params::<F, D>()?;
    let out = match &params {
        ProtocolCrtNttParams::Q32(p) => mat_vec_mul_dense_many_with_params(mat, vecs, p),
        ProtocolCrtNttParams::Q64(p) => mat_vec_mul_dense_many_with_params(mat, vecs, p),
        ProtocolCrtNttParams::Q128(p) => mat_vec_mul_dense_many_with_params(mat, vecs, p),
    };
    Ok(out)
}

/// Selector for which cached matrix to use.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MatrixSlot {
    A,
    B,
    D,
}

fn mat_vec_mul_precomputed_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[Vec<CyclotomicCrtNtt<W, K, D>>],
    vec: &[CyclotomicRing<F, D>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let ntt_vec: Vec<CyclotomicCrtNtt<W, K, D>> = cfg_iter!(vec)
        .map(|v| CyclotomicCrtNtt::from_ring_with_params(v, params))
        .collect();

    cfg_iter!(ntt_mat)
        .map(|row_ntt| {
            debug_assert!(row_ntt.len() >= ntt_vec.len());
            let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
            for (a_ntt, x_ntt) in row_ntt.iter().zip(ntt_vec.iter()) {
                accumulate_pointwise_product_into(&mut acc, a_ntt, x_ntt, params);
            }
            acc.to_ring_with_params(params)
        })
        .collect()
}

macro_rules! dispatch_cached {
    ($cache:expr, $which:expr, $func:ident $(, $arg:expr)*) => {{
        #[allow(non_snake_case)]
        match $cache {
            NttMatrixCache::Q32 { A, B, D: Dm, params: p } => {
                let m = match $which { MatrixSlot::A => A, MatrixSlot::B => B, MatrixSlot::D => Dm };
                $func(m, $($arg,)* p)
            }
            NttMatrixCache::Q64 { A, B, D: Dm, params: p } => {
                let m = match $which { MatrixSlot::A => A, MatrixSlot::B => B, MatrixSlot::D => Dm };
                $func(m, $($arg,)* p)
            }
            NttMatrixCache::Q128 { A, B, D: Dm, params: p } => {
                let m = match $which { MatrixSlot::A => A, MatrixSlot::B => B, MatrixSlot::D => Dm };
                $func(m, $($arg,)* p)
            }
        }
    }};
}

/// Dense mat-vec using a pre-converted NTT matrix from the cache.
pub(crate) fn mat_vec_mul_ntt_cached<F: FieldCore + CanonicalField, const D: usize>(
    cache: &NttMatrixCache<D>,
    which: MatrixSlot,
    vec: &[CyclotomicRing<F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    let out = dispatch_cached!(cache, which, mat_vec_mul_precomputed_with_params, vec);
    Ok(out)
}

/// Basis-decompose a block of ring elements into `block.len() * delta` gadget components.
pub fn decompose_block<F: FieldCore + CanonicalField, const D: usize>(
    block: &[CyclotomicRing<F, D>],
    delta: usize,
    log_basis: u32,
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = vec![CyclotomicRing::<F, D>::zero(); block.len() * delta];
    for (i, coeff_vec) in block.iter().enumerate() {
        coeff_vec.balanced_decompose_pow2_into(&mut out[i * delta..(i + 1) * delta], log_basis);
    }
    out
}

pub(crate) fn decompose_rows<F: FieldCore + CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    delta: usize,
    log_basis: u32,
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = vec![CyclotomicRing::<F, D>::zero(); rows.len() * delta];
    for (i, row) in rows.iter().enumerate() {
        row.balanced_decompose_pow2_into(&mut out[i * delta..(i + 1) * delta], log_basis);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{mat_vec_mul_crt_ntt, mat_vec_mul_crt_ntt_many, mat_vec_mul_unchecked};
    use crate::algebra::{CyclotomicRing, Fp64};
    use crate::FromSmallInt;

    #[test]
    fn dense_mat_vec_matches_schoolbook_q32_d64() {
        type F = Fp64<4294967197>;
        const D: usize = 64;
        let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        let coeffs = std::array::from_fn(|k| {
                            F::from_u64((i as u64 * 10_000 + j as u64 * 100 + k as u64 + 1) % 97)
                        });
                        CyclotomicRing::from_coefficients(coeffs)
                    })
                    .collect()
            })
            .collect();
        let vec: Vec<CyclotomicRing<F, D>> = (0..4)
            .map(|j| {
                let coeffs =
                    std::array::from_fn(|k| F::from_u64((j as u64 * 50 + k as u64 + 3) % 89));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();

        let schoolbook = mat_vec_mul_unchecked(&mat, &vec);
        let crt_ntt = mat_vec_mul_crt_ntt(&mat, &vec).expect("Q32 dispatch should succeed");
        assert_eq!(schoolbook, crt_ntt);
    }

    #[test]
    fn dense_mat_vec_matches_schoolbook_q64_dispatch_for_large_d() {
        type F = Fp64<4294967197>;
        const D: usize = 128;
        let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..2)
            .map(|i| {
                (0..2)
                    .map(|j| {
                        let coeffs = std::array::from_fn(|k| {
                            F::from_u64((i as u64 * 20_000 + j as u64 * 300 + k as u64 + 7) % 113)
                        });
                        CyclotomicRing::from_coefficients(coeffs)
                    })
                    .collect()
            })
            .collect();
        let vec: Vec<CyclotomicRing<F, D>> = (0..2)
            .map(|j| {
                let coeffs =
                    std::array::from_fn(|k| F::from_u64((j as u64 * 70 + k as u64 + 11) % 101));
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();

        let schoolbook = mat_vec_mul_unchecked(&mat, &vec);
        let crt_ntt = mat_vec_mul_crt_ntt(&mat, &vec).expect("Q64 dispatch should succeed");
        assert_eq!(schoolbook, crt_ntt);
    }

    #[test]
    fn dense_mat_vec_many_matches_individual_crt_ntt_q32_d64() {
        type F = Fp64<4294967197>;
        const D: usize = 64;
        let mat: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        let coeffs = std::array::from_fn(|k| {
                            F::from_u64((i as u64 * 10_000 + j as u64 * 100 + k as u64 + 1) % 97)
                        });
                        CyclotomicRing::from_coefficients(coeffs)
                    })
                    .collect()
            })
            .collect();

        let vecs: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
            .map(|seed| {
                (0..4)
                    .map(|j| {
                        let coeffs = std::array::from_fn(|k| {
                            F::from_u64((seed as u64 * 700 + j as u64 * 50 + k as u64 + 3) % 89)
                        });
                        CyclotomicRing::from_coefficients(coeffs)
                    })
                    .collect()
            })
            .collect();

        let expected: Vec<Vec<CyclotomicRing<F, D>>> = vecs
            .iter()
            .map(|v| mat_vec_mul_crt_ntt(&mat, v).expect("single CRT+NTT mat-vec should succeed"))
            .collect();

        let got =
            mat_vec_mul_crt_ntt_many(&mat, &vecs).expect("batched CRT+NTT mat-vec should succeed");
        assert_eq!(expected, got);
    }
}
