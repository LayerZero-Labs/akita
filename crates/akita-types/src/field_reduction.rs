//! Field-reduction helpers for extension-field claims embedded in rings.
//!
//! These utilities model the algebraic trace subgroup used by the paper's
//! `F_{q^k}` to `R_q` reduction. They are intentionally standalone so the
//! mathematical contract can be tested independently of the prover API.

use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, FieldCore, RingSubfieldFp4};

/// Parameters for the subgroup `H = <sigma_-1, sigma_(4k+1)>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubfieldParams<const D: usize> {
    k: usize,
}

impl<const D: usize> SubfieldParams<D> {
    /// Construct subgroup parameters for an `F_{q^k}` claim embedded in `R_q`.
    ///
    /// # Errors
    ///
    /// Returns an error when `D` or `k` is zero, or when `k` does not divide
    /// `D / 2`.
    pub fn new(k: usize) -> Result<Self, AkitaError> {
        let two_d = D
            .checked_mul(2)
            .ok_or_else(|| AkitaError::InvalidInput("ring dimension is too large".to_string()))?;

        if D == 0 {
            return Err(AkitaError::InvalidInput(
                "ring dimension must be non-zero".to_string(),
            ));
        }
        if !D.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "ring dimension D={D} must be a power of two",
            )));
        }
        if D % 2 != 0 {
            return Err(AkitaError::InvalidInput(format!(
                "ring dimension D={D} must be even",
            )));
        }
        if k == 0 {
            return Err(AkitaError::InvalidInput(
                "extension degree k must be non-zero".to_string(),
            ));
        }
        if k > D / 2 || (D / 2) % k != 0 {
            return Err(AkitaError::InvalidInput(format!(
                "extension degree k={k} must divide D/2 for D={D}",
            )));
        }
        let sigma_step = k
            .checked_mul(4)
            .and_then(|step| step.checked_add(1))
            .ok_or_else(|| AkitaError::InvalidInput("extension degree is too large".to_string()))?;
        if gcd(sigma_step, two_d) != 1 {
            return Err(AkitaError::InvalidInput(format!(
                "subgroup generator {sigma_step} must be invertible modulo 2D={two_d}",
            )));
        }

        Ok(Self { k })
    }

    /// Extension degree `k`.
    #[inline]
    pub const fn extension_degree(&self) -> usize {
        self.k
    }

    /// Automorphism exponents generating `H`, modulo `2D`.
    #[inline]
    pub fn h_generators(&self) -> (usize, usize) {
        let two_d = D.saturating_mul(2);
        let sigma_step = self.k.saturating_mul(4).saturating_add(1);
        (two_d.saturating_sub(1), sigma_step)
    }

    /// Enumerate the distinct odd exponents in `H`.
    pub fn h_exponents(&self) -> Vec<usize> {
        let two_d = D.saturating_mul(2);
        let (sigma_m1, sigma_step) = self.h_generators();
        let mut exponents = Vec::with_capacity(D / self.k);
        let mut power = 1usize;

        for _ in 0..two_d {
            push_unique(&mut exponents, power);
            push_unique(&mut exponents, mul_mod(power, sigma_m1, two_d));

            power = mul_mod(power, sigma_step, two_d);
            if power == 1 {
                exponents.sort_unstable();
                return exponents;
            }
        }

        unreachable!("validated subgroup generator must have finite order modulo 2D")
    }

    /// Number of base-field coordinates in the paper's packed representative.
    #[inline]
    pub const fn packed_len(&self) -> usize {
        D / self.k
    }
}

/// Compute `Tr_H(x) = sum_{sigma in H} sigma(x)`.
///
/// # Panics
///
/// Panics if the generated subgroup contains an invalid automorphism exponent.
pub fn trace_h<F: FieldCore, const D: usize>(
    params: SubfieldParams<D>,
    x: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let mut out = CyclotomicRing::zero();
    for exponent in params.h_exponents() {
        out += x.sigma(exponent);
    }
    out
}

/// Embed a vector of `D / k` ring-subfield elements into one element of `R_q`.
///
/// Each subfield element is given by `k` base-field coordinates in the basis
/// `[1, e_1, ..., e_{k-1}]`, where `e_j = X^(j*D/(2k)) + X^(-j*D/(2k))` viewed
/// inside `R_q = Z_q[X] / (X^D + 1)`. The disambiguating shifts used to place
/// the `D / k` elements are
/// `T = {0, ..., D/(2k) - 1} ∪ {D/2, ..., D/2 + D/(2k) - 1}`.
///
/// `coords` has length `D` with the layout
/// `[s_0[0], s_0[1], ..., s_0[k-1], s_1[0], ..., s_{D/k - 1}[k-1]]`,
/// so `coords[i*k + j]` is the `j`-th basis coordinate of the `i`-th
/// subfield slot.
///
/// For `k = 1` this reduces to placing one base-field value per ring position
/// in the canonical shift order, since the subfield basis collapses to `[1]`
/// and the shift set covers all of `[0, D)`.
///
/// The resulting embedding `psi : (R_q^H)^{D/k} -> R_q` is invertible whenever
/// `2` is a unit in `F` (i.e., for any odd prime characteristic), and is the
/// production-side packing used by the trace inner-product relation
/// `Tr_H(Y * sigma_{-1}(V)) = (D/k) * y` in [`psi_embed_ring_subfield_fp4`].
///
/// The implementation is split into two branchless inner loops. For low
/// shifts `shift in [0, D/(2k))`, both `e_j` terms always land in `[0, D)`
/// without wrapping, so the positive term contributes `+c_j` at
/// `shift + j*step` and the negative term contributes `-c_j` at
/// `shift + D - j*step`. For high shifts `shift in [D/2, D/2 + D/(2k))`,
/// the positive term still does not wrap, while the negative term wraps
/// exactly once around `X^D = -1`, flipping its sign back to `+c_j` at
/// `shift - j*step`.
///
/// # Errors
///
/// Returns an error when `coords.len() != D`.
pub fn psi_embed<F: FieldCore, const D: usize>(
    params: SubfieldParams<D>,
    coords: &[F],
) -> Result<CyclotomicRing<F, D>, AkitaError> {
    if coords.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: coords.len(),
        });
    }

    let k = params.extension_degree();
    let m = params.packed_len();
    let step = D / (2 * k);
    let half = m / 2;
    let mut out = [F::zero(); D];

    for idx in 0..half {
        let shift = idx;
        let base = idx * k;
        out[shift] += coords[base];
        for j in 1..k {
            let cj = coords[base + j];
            let pos_offset = j * step;
            out[shift + pos_offset] += cj;
            out[shift + D - pos_offset] -= cj;
        }
    }

    for idx in half..m {
        let shift = idx - half + D / 2;
        let base = idx * k;
        out[shift] += coords[base];
        for j in 1..k {
            let cj = coords[base + j];
            let pos_offset = j * step;
            out[shift + pos_offset] += cj;
            out[shift - pos_offset] += cj;
        }
    }

    Ok(CyclotomicRing::from_coefficients(out))
}

/// Pack `D / 4` ring-subfield `Fp4` elements into one element of `R_q`.
///
/// Typed entry point for the prover-side full `psi : (R_q^H)^{D/k} -> R_q`
/// packing in the `k = 4` Hachi subfield case. Each element's
/// `coeffs = [c0, c1, c2, c3]` is interpreted in the basis
/// `[1, e_1, e_2, e_3]`. Internally flattens into the layout consumed by
/// [`psi_embed`].
///
/// # Errors
///
/// Returns an error when `D` is not compatible with a `k = 4` subfield, or
/// when `elements.len() != D / 4`.
pub fn psi_embed_ring_subfield_fp4<F: FieldCore, const D: usize>(
    elements: &[RingSubfieldFp4<F>],
) -> Result<CyclotomicRing<F, D>, AkitaError> {
    let params = SubfieldParams::<D>::new(4)?;
    let expected = params.packed_len();
    if elements.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: elements.len(),
        });
    }

    let mut coords = [F::zero(); D];
    for (i, elem) in elements.iter().enumerate() {
        coords[i * 4..i * 4 + 4].copy_from_slice(&elem.coeffs);
    }
    psi_embed(params, &coords)
}

/// Embed a single `k = 4` ring-subfield element into `R_q` at shift `X^0`.
///
/// Mathematically this is the slot-0 specialization of
/// [`psi_embed_ring_subfield_fp4`], i.e.
/// `psi_embed_ring_subfield_fp4(&[x, 0, ..., 0])`. It is kept as a separate
/// entry point because the verifier-side trace check only needs to embed a
/// single claimed inner product into the ring, and writing the `2k - 1`
/// nonzero coefficients directly avoids the `O(D)` loop and zero-padding the
/// vector form pays.
///
/// # Errors
///
/// Returns an error when `D` is not compatible with a `k = 4` Hachi subfield.
pub fn embed_ring_subfield_fp4<F: FieldCore, const D: usize>(
    x: RingSubfieldFp4<F>,
) -> Result<CyclotomicRing<F, D>, AkitaError> {
    let params = SubfieldParams::<D>::new(4)?;
    let step = D / (2 * params.extension_degree());
    let [c0, c1, c2, c3] = x.coeffs;
    let mut coeffs = [F::zero(); D];
    coeffs[0] = c0;
    coeffs[step] = c1;
    coeffs[D - step] = -c1;
    coeffs[2 * step] = c2;
    coeffs[D - 2 * step] = -c2;
    coeffs[3 * step] = c3;
    coeffs[D - 3 * step] = -c3;
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

fn push_unique(values: &mut Vec<usize>, value: usize) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn mul_mod(a: usize, b: usize, modulus: usize) -> usize {
    ((a as u128 * b as u128) % modulus as u128) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{reduce_inner_opening_to_ring_element, BasisMode};
    use akita_field::{ExtField, Fp32, TowerBasisFp4, TwoNr, UnitNr};

    type F = Fp32<251>;
    type HachiF32 = Fp32<4294967197>;

    fn ring_from_i64s<const D: usize>(values: [i64; D]) -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(values.map(F::from_i64))
    }

    fn ring_from_index<const D: usize>() -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|i| F::from_u64((i + 1) as u64)))
    }

    fn hachi_subfield_basis<Fq: FieldCore, const D: usize>(
        params: SubfieldParams<D>,
    ) -> Vec<CyclotomicRing<Fq, D>> {
        let k = params.extension_degree();
        let step = D / (2 * k);
        let mut basis = Vec::with_capacity(k);
        basis.push(CyclotomicRing::one());
        for i in 1..k {
            let pos = i * step;
            let mut coeffs = [Fq::zero(); D];
            coeffs[pos] = Fq::one();
            coeffs[D - pos] = -Fq::one();
            basis.push(CyclotomicRing::from_coefficients(coeffs));
        }
        basis
    }

    fn hachi_subfield_coords<Fq: FieldCore, const D: usize>(
        params: SubfieldParams<D>,
        x: &CyclotomicRing<Fq, D>,
    ) -> Vec<Fq> {
        let k = params.extension_degree();
        let step = D / (2 * k);
        let coeffs = x.coefficients();
        let mut coords = vec![Fq::zero(); k];
        coords[0] = coeffs[0];

        for (i, coord) in coords.iter_mut().enumerate().take(k).skip(1) {
            let pos = i * step;
            *coord = coeffs[pos];
            assert_eq!(
                coeffs[D - pos],
                -*coord,
                "subfield coordinate {i} has wrong inverse coefficient"
            );
        }

        for (idx, coeff) in coeffs.iter().enumerate() {
            let is_basis_slot = idx == 0
                || (1..k).any(|i| {
                    let pos = i * step;
                    idx == pos || idx == D - pos
                });
            if !is_basis_slot {
                assert!(
                    coeff.is_zero(),
                    "unexpected nonzero coefficient at ring exponent {idx}"
                );
            }
        }

        coords
    }

    fn embed_tower_in_hachi_subfield<const D: usize>(
        x: TowerBasisFp4<HachiF32, TwoNr, UnitNr>,
    ) -> CyclotomicRing<HachiF32, D> {
        let params = SubfieldParams::<D>::new(4).unwrap();
        let basis = hachi_subfield_basis::<HachiF32, D>(params);

        // Over 2^32 - 99, i is a square root of -1 and a satisfies
        // a^2 = 1 / (2 * (1 + i)). Thus v = a*e1 + a*i*e3 has v^2 = e2.
        let a = HachiF32::from_u64(1_492_342_050);
        let ai = a * HachiF32::from_u64(3_311_696_422);
        let v = basis[1].scale(&a) + basis[3].scale(&ai);
        let u = basis[2];
        let vu = v * u;
        let power_basis = [basis[0], v, u, vu];
        let coeffs = x.to_base_vec();

        coeffs
            .into_iter()
            .zip(power_basis)
            .fold(CyclotomicRing::zero(), |acc, (coeff, basis_elem)| {
                acc + basis_elem.scale(&coeff)
            })
    }

    #[test]
    fn subfield_params_validate_extension_degree() {
        assert!(SubfieldParams::<8>::new(1).is_ok());
        assert!(SubfieldParams::<8>::new(4).is_ok());

        assert!(matches!(
            SubfieldParams::<8>::new(0),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            SubfieldParams::<8>::new(3),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            SubfieldParams::<9>::new(1),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            SubfieldParams::<6>::new(1),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            SubfieldParams::<10>::new(1),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            SubfieldParams::<{ usize::MAX - 1 }>::new(1),
            Err(AkitaError::InvalidInput(_))
        ));
    }

    #[test]
    fn h_exponents_match_power_of_two_subgroups() {
        assert_eq!(SubfieldParams::<8>::new(1).unwrap().h_exponents().len(), 8);
        assert_eq!(SubfieldParams::<8>::new(2).unwrap().h_exponents().len(), 4);
        assert_eq!(SubfieldParams::<8>::new(4).unwrap().h_exponents().len(), 2);

        assert_eq!(
            SubfieldParams::<8>::new(2).unwrap().h_exponents(),
            vec![1, 7, 9, 15]
        );
    }

    #[test]
    fn h_exponents_cover_production_ring_subgroups() {
        assert_eq!(
            SubfieldParams::<64>::new(1).unwrap().h_exponents().len(),
            64
        );
        assert_eq!(SubfieldParams::<64>::new(8).unwrap().h_exponents().len(), 8);
        assert_eq!(
            SubfieldParams::<128>::new(1).unwrap().h_exponents().len(),
            128
        );
        assert_eq!(
            SubfieldParams::<128>::new(16).unwrap().h_exponents().len(),
            8
        );
    }

    #[test]
    fn trace_h_k_one_matches_constant_coefficient_trace() {
        const D: usize = 8;
        let params = SubfieldParams::<D>::new(1).unwrap();
        let x = ring_from_i64s([3, 5, 7, 11, 13, 17, 19, 23]);
        let trace = trace_h(params, &x);
        let coeffs = trace.coefficients();

        assert_eq!(coeffs[0], F::from_u64(D as u64) * x.coefficients()[0]);
        assert!(coeffs[1..].iter().all(|coeff| coeff.is_zero()));
    }

    #[test]
    fn trace_h_k_one_matches_constant_coefficient_trace_at_production_sizes() {
        let params_64 = SubfieldParams::<64>::new(1).unwrap();
        let x_64 = ring_from_index::<64>();
        let trace_64 = trace_h(params_64, &x_64);
        assert_eq!(
            trace_64.coefficients()[0],
            F::from_u64(64) * x_64.coefficients()[0]
        );
        assert!(trace_64.coefficients()[1..]
            .iter()
            .all(|coeff| coeff.is_zero()));

        let params_128 = SubfieldParams::<128>::new(1).unwrap();
        let x_128 = ring_from_index::<128>();
        let trace_128 = trace_h(params_128, &x_128);
        assert_eq!(
            trace_128.coefficients()[0],
            F::from_u64(128) * x_128.coefficients()[0]
        );
        assert!(trace_128.coefficients()[1..]
            .iter()
            .all(|coeff| coeff.is_zero()));
    }

    #[test]
    fn trace_h_k_one_matches_inner_opening_reduction_shortcut() {
        const D: usize = 8;
        let params = SubfieldParams::<D>::new(1).unwrap();
        let y_ring = ring_from_i64s([2, 3, 5, 7, 11, 13, 17, 19]);
        let inner_point = [F::from_u64(3), F::from_u64(5), F::from_u64(7)];

        for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
            let v = reduce_inner_opening_to_ring_element::<F, D>(&inner_point, basis).unwrap();
            let product = y_ring * v.sigma_m1();
            let trace = trace_h(params, &product);
            let coeffs = trace.coefficients();
            let current_shortcut = F::from_u64(D as u64) * product.coefficients()[0];

            assert_eq!(coeffs[0], current_shortcut);
            assert!(coeffs[1..].iter().all(|coeff| coeff.is_zero()));
        }
    }

    #[test]
    fn trace_h_matches_direct_generator_sum() {
        const D: usize = 8;
        let params = SubfieldParams::<D>::new(2).unwrap();
        let x = ring_from_i64s([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut expected = CyclotomicRing::zero();

        for exponent in [1, 7, 9, 15] {
            expected += x.sigma(exponent);
        }

        assert_eq!(trace_h(params, &x), expected);
    }

    /// Build a flat `psi_embed` input from `D / k` subfield elements where
    /// only the constant (`e_0 = 1`) coordinate is set.
    fn constants_only_coords<const D: usize>(params: SubfieldParams<D>, values: &[F]) -> Vec<F> {
        let k = params.extension_degree();
        assert_eq!(values.len(), params.packed_len());
        let mut coords = vec![F::zero(); D];
        for (i, value) in values.iter().enumerate() {
            coords[i * k] = *value;
        }
        coords
    }

    #[test]
    fn psi_embed_constants_only_matches_paper_positions() {
        const D: usize = 8;
        let params = SubfieldParams::<D>::new(2).unwrap();
        let coords = constants_only_coords(
            params,
            &[
                F::from_u64(1),
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(4),
            ],
        );
        let packed = psi_embed::<F, D>(params, &coords).unwrap();
        let expected = ring_from_i64s([1, 2, 0, 0, 3, 4, 0, 0]);

        assert_eq!(packed, expected);
    }

    #[test]
    fn psi_embed_constants_only_at_production_ring_size() {
        const D: usize = 64;
        let params = SubfieldParams::<D>::new(8).unwrap();
        let values: Vec<F> = (0..params.packed_len())
            .map(|i| F::from_u64((i + 1) as u64))
            .collect();
        let coords = constants_only_coords(params, &values);
        let packed = psi_embed::<F, D>(params, &coords).unwrap();
        let coeffs = packed.coefficients();
        let half = params.packed_len() / 2;

        assert_eq!(&coeffs[..half], &values[..half]);
        assert!(coeffs[half..D / 2].iter().all(|coeff| coeff.is_zero()));
        assert_eq!(&coeffs[D / 2..D / 2 + half], &values[half..]);
        assert!(coeffs[D / 2 + half..].iter().all(|coeff| coeff.is_zero()));
    }

    #[test]
    fn psi_embed_k_one_is_identity_placement() {
        const D: usize = 8;
        let params = SubfieldParams::<D>::new(1).unwrap();
        let coords: Vec<F> = (0..D).map(|i| F::from_u64((i + 1) as u64)).collect();
        let packed = psi_embed::<F, D>(params, &coords).unwrap();
        let expected = ring_from_i64s([1, 2, 3, 4, 5, 6, 7, 8]);

        assert_eq!(packed, expected);
    }

    #[test]
    fn psi_embed_rejects_wrong_length() {
        let params = SubfieldParams::<8>::new(2).unwrap();
        assert!(matches!(
            psi_embed::<F, 8>(params, &[F::one()]),
            Err(AkitaError::InvalidSize {
                expected: 8,
                actual: 1
            })
        ));
    }

    #[test]
    fn psi_embed_ring_subfield_fp4_single_element_matches_shift_zero_embed() {
        const D: usize = 8;
        let x = RingSubfieldFp4::new([
            HachiF32::from_u64(2),
            HachiF32::from_u64(3),
            HachiF32::from_u64(5),
            HachiF32::from_u64(7),
        ]);
        let zero = RingSubfieldFp4::default();

        let packed = psi_embed_ring_subfield_fp4::<HachiF32, D>(&[x, zero]).unwrap();
        let single = embed_ring_subfield_fp4::<HachiF32, D>(x).unwrap();

        assert_eq!(packed, single);
    }

    #[test]
    fn psi_embed_ring_subfield_fp4_rejects_wrong_length() {
        const D: usize = 8;
        let x = RingSubfieldFp4::default();
        assert!(matches!(
            psi_embed_ring_subfield_fp4::<HachiF32, D>(&[x]),
            Err(AkitaError::InvalidSize {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn hachi_k4_subfield_basis_has_chebyshev_multiplication_table() {
        const D: usize = 8;
        let params = SubfieldParams::<D>::new(4).unwrap();
        let basis = hachi_subfield_basis::<HachiF32, D>(params);
        let two = HachiF32::from_u64(2);

        assert_eq!(
            hachi_subfield_coords(params, &(basis[1] * basis[1])),
            vec![two, HachiF32::zero(), HachiF32::one(), HachiF32::zero()]
        );
        assert_eq!(
            hachi_subfield_coords(params, &(basis[1] * basis[2])),
            vec![
                HachiF32::zero(),
                HachiF32::one(),
                HachiF32::zero(),
                HachiF32::one()
            ]
        );
        assert_eq!(
            hachi_subfield_coords(params, &(basis[1] * basis[3])),
            vec![
                HachiF32::zero(),
                HachiF32::zero(),
                HachiF32::one(),
                HachiF32::zero()
            ]
        );
        assert_eq!(
            hachi_subfield_coords(params, &(basis[2] * basis[2])),
            vec![two, HachiF32::zero(), HachiF32::zero(), HachiF32::zero()]
        );
        assert_eq!(
            hachi_subfield_coords(params, &(basis[2] * basis[3])),
            vec![
                HachiF32::zero(),
                HachiF32::one(),
                HachiF32::zero(),
                -HachiF32::one()
            ]
        );
        assert_eq!(
            hachi_subfield_coords(params, &(basis[3] * basis[3])),
            vec![two, HachiF32::zero(), -HachiF32::one(), HachiF32::zero()]
        );
    }

    #[test]
    fn naive_hachi_k4_basis_is_not_the_current_tower_power_basis() {
        const D: usize = 8;
        let params = SubfieldParams::<D>::new(4).unwrap();
        let basis = hachi_subfield_basis::<HachiF32, D>(params);

        assert_ne!(basis[1] * basis[1], basis[2]);
        assert_eq!(
            hachi_subfield_coords(params, &(basis[1] * basis[1])),
            vec![
                HachiF32::from_u64(2),
                HachiF32::zero(),
                HachiF32::one(),
                HachiF32::zero()
            ]
        );
    }

    #[test]
    fn hachi_k4_subfield_contains_current_tower_after_base_change() {
        const D: usize = 8;
        type E = TowerBasisFp4<HachiF32, TwoNr, UnitNr>;

        let params = SubfieldParams::<D>::new(4).unwrap();
        let basis = hachi_subfield_basis::<HachiF32, D>(params);
        let a = HachiF32::from_u64(1_492_342_050);
        let ai = a * HachiF32::from_u64(3_311_696_422);
        let v = basis[1].scale(&a) + basis[3].scale(&ai);
        let u = basis[2];

        assert_eq!(v * v, u);
        assert_eq!(u * u, basis[0].scale(&HachiF32::from_u64(2)));
        assert_eq!(v * v * v * v, basis[0].scale(&HachiF32::from_u64(2)));

        let x = E::from_base_slice(&[
            HachiF32::from_u64(3),
            HachiF32::from_u64(5),
            HachiF32::from_u64(7),
            HachiF32::from_u64(11),
        ]);
        let y = E::from_base_slice(&[
            HachiF32::from_u64(13),
            HachiF32::from_u64(17),
            HachiF32::from_u64(19),
            HachiF32::from_u64(23),
        ]);

        assert_eq!(
            embed_tower_in_hachi_subfield::<D>(x * y),
            embed_tower_in_hachi_subfield::<D>(x) * embed_tower_in_hachi_subfield::<D>(y)
        );
    }

    fn assert_ring_subfield_fp4_embedding_is_multiplicative<const D: usize>() {
        let x = RingSubfieldFp4::new([
            HachiF32::from_u64(3),
            HachiF32::from_u64(5),
            HachiF32::from_u64(7),
            HachiF32::from_u64(11),
        ]);
        let y = RingSubfieldFp4::new([
            HachiF32::from_u64(13),
            HachiF32::from_u64(17),
            HachiF32::from_u64(19),
            HachiF32::from_u64(23),
        ]);

        assert_eq!(
            embed_ring_subfield_fp4::<HachiF32, D>(x * y).unwrap(),
            embed_ring_subfield_fp4::<HachiF32, D>(x).unwrap()
                * embed_ring_subfield_fp4::<HachiF32, D>(y).unwrap()
        );
    }

    #[test]
    fn ring_subfield_fp4_embedding_places_coefficients_in_hachi_basis() {
        const D: usize = 8;
        let x = RingSubfieldFp4::new([
            HachiF32::from_u64(2),
            HachiF32::from_u64(3),
            HachiF32::from_u64(5),
            HachiF32::from_u64(7),
        ]);
        let embedded = embed_ring_subfield_fp4::<HachiF32, D>(x).unwrap();
        let coeffs = embedded.coefficients();

        assert_eq!(coeffs[0], HachiF32::from_u64(2));
        assert_eq!(coeffs[1], HachiF32::from_u64(3));
        assert_eq!(coeffs[7], -HachiF32::from_u64(3));
        assert_eq!(coeffs[2], HachiF32::from_u64(5));
        assert_eq!(coeffs[6], -HachiF32::from_u64(5));
        assert_eq!(coeffs[3], HachiF32::from_u64(7));
        assert_eq!(coeffs[5], -HachiF32::from_u64(7));
        assert!(coeffs[4].is_zero());
    }

    #[test]
    fn ring_subfield_fp4_embedding_is_multiplicative_across_ring_dimensions() {
        assert_ring_subfield_fp4_embedding_is_multiplicative::<8>();
        assert_ring_subfield_fp4_embedding_is_multiplicative::<64>();
        assert_ring_subfield_fp4_embedding_is_multiplicative::<128>();
    }

    /// Generate `D / 4` deterministic `RingSubfieldFp4` elements seeded by `tag`.
    fn deterministic_subfield_fp4_vector<const D: usize>(
        tag: u64,
    ) -> Vec<RingSubfieldFp4<HachiF32>> {
        let m = D / 4;
        (0..m)
            .map(|i| {
                let i = i as u64;
                RingSubfieldFp4::new([
                    HachiF32::from_u64(2 + 7 * i + 11 * tag),
                    HachiF32::from_u64(3 + 13 * i + 17 * tag),
                    HachiF32::from_u64(5 + 19 * i + 23 * tag),
                    HachiF32::from_u64(7 + 29 * i + 31 * tag),
                ])
            })
            .collect()
    }

    /// Verify the trace inner-product relation
    /// `Tr_H(psi(s) * sigma_{-1}(psi(v))) = (D / k) * embed(<s, v>)`
    /// for the typed `k = 4` ring-subfield representation.
    fn assert_psi_trace_inner_product_identity_fp4<const D: usize>() {
        let params = SubfieldParams::<D>::new(4).unwrap();
        let s = deterministic_subfield_fp4_vector::<D>(0);
        let v = deterministic_subfield_fp4_vector::<D>(1);

        // y = <s, v> in the ring-subfield.
        let y = s
            .iter()
            .zip(v.iter())
            .fold(RingSubfieldFp4::zero(), |acc, (si, vi)| acc + (*si * *vi));

        let big_y = psi_embed_ring_subfield_fp4::<HachiF32, D>(&s).unwrap();
        let big_v = psi_embed_ring_subfield_fp4::<HachiF32, D>(&v).unwrap();
        let traced = trace_h(params, &(big_y * big_v.sigma_m1()));

        let scale = HachiF32::from_u64(params.packed_len() as u64);
        let scaled = embed_ring_subfield_fp4::<HachiF32, D>(y)
            .unwrap()
            .scale(&scale);

        assert_eq!(traced, scaled);
    }

    #[test]
    fn psi_trace_inner_product_identity_fp4() {
        assert_psi_trace_inner_product_identity_fp4::<8>();
        assert_psi_trace_inner_product_identity_fp4::<64>();
        assert_psi_trace_inner_product_identity_fp4::<128>();
    }

    /// Subfield multiplication for `k = 2`: `e_1^2 = 2` for any valid `D`,
    /// so `R_q^H ≅ F_q[sqrt(2)]`.
    fn fp2_subfield_mul(a: [HachiF32; 2], b: [HachiF32; 2]) -> [HachiF32; 2] {
        let two = HachiF32::from_u64(2);
        [a[0] * b[0] + two * a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
    }

    /// Embed a `k = 2` subfield element at shift `X^0`.
    fn embed_subfield_fp2_at_zero<const D: usize>(
        coeffs: [HachiF32; 2],
    ) -> CyclotomicRing<HachiF32, D> {
        let step = D / 4;
        let [c0, c1] = coeffs;
        let mut out = [HachiF32::zero(); D];
        out[0] = c0;
        out[step] = c1;
        out[D - step] = -c1;
        CyclotomicRing::from_coefficients(out)
    }

    fn assert_psi_trace_inner_product_identity_fp2<const D: usize>() {
        let params = SubfieldParams::<D>::new(2).unwrap();
        let m = params.packed_len();

        let s: Vec<[HachiF32; 2]> = (0..m)
            .map(|i| {
                let i = i as u64;
                [
                    HachiF32::from_u64(2 + 7 * i),
                    HachiF32::from_u64(3 + 13 * i),
                ]
            })
            .collect();
        let v: Vec<[HachiF32; 2]> = (0..m)
            .map(|i| {
                let i = i as u64;
                [
                    HachiF32::from_u64(11 + 19 * i),
                    HachiF32::from_u64(17 + 23 * i),
                ]
            })
            .collect();

        let y = s
            .iter()
            .zip(v.iter())
            .fold([HachiF32::zero(); 2], |acc, (si, vi)| {
                let prod = fp2_subfield_mul(*si, *vi);
                [acc[0] + prod[0], acc[1] + prod[1]]
            });

        let mut s_flat = vec![HachiF32::zero(); D];
        let mut v_flat = vec![HachiF32::zero(); D];
        for (i, (sc, vc)) in s.iter().zip(v.iter()).enumerate() {
            s_flat[i * 2] = sc[0];
            s_flat[i * 2 + 1] = sc[1];
            v_flat[i * 2] = vc[0];
            v_flat[i * 2 + 1] = vc[1];
        }

        let big_y = psi_embed::<HachiF32, D>(params, &s_flat).unwrap();
        let big_v = psi_embed::<HachiF32, D>(params, &v_flat).unwrap();
        let traced = trace_h(params, &(big_y * big_v.sigma_m1()));

        let scale = HachiF32::from_u64(m as u64);
        let scaled = embed_subfield_fp2_at_zero::<D>(y).scale(&scale);

        assert_eq!(traced, scaled);
    }

    #[test]
    fn psi_trace_inner_product_identity_fp2() {
        assert_psi_trace_inner_product_identity_fp2::<8>();
        assert_psi_trace_inner_product_identity_fp2::<64>();
        assert_psi_trace_inner_product_identity_fp2::<128>();
    }
}
