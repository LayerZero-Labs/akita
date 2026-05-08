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

/// Pack base-field coordinates into the ring positions used by the paper's `psi`.
///
/// This is the coefficient-placement part of the reduction. It does not claim
/// that callers have already chosen an extension basis satisfying the full
/// isomorphism contract.
///
/// # Errors
///
/// Returns an error when `values.len()` does not equal `D / k`.
pub fn psi_pack<F: FieldCore, const D: usize>(
    params: SubfieldParams<D>,
    values: &[F],
) -> Result<CyclotomicRing<F, D>, AkitaError> {
    let expected = params.packed_len();
    if values.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: values.len(),
        });
    }

    let half = expected / 2;
    let mut coeffs = [F::zero(); D];
    coeffs[..half].copy_from_slice(&values[..half]);
    coeffs[D / 2..D / 2 + half].copy_from_slice(&values[half..]);
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

/// Embed a native `k = 4` ring-subfield element into `R_q`.
///
/// Coordinates `[c0, c1, c2, c3]` are interpreted in the basis
/// `[1, e1, e2, e3]`, where `e_j = X^(jD/8) + X^(-jD/8)`.
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

    #[test]
    fn psi_pack_places_halves_in_paper_positions() {
        const D: usize = 8;
        let params = SubfieldParams::<D>::new(2).unwrap();
        let packed = psi_pack(
            params,
            &[
                F::from_u64(1),
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(4),
            ],
        )
        .unwrap();
        let expected = ring_from_i64s([1, 2, 0, 0, 3, 4, 0, 0]);

        assert_eq!(packed, expected);
    }

    #[test]
    fn psi_pack_places_halves_at_production_ring_size() {
        const D: usize = 64;
        let params = SubfieldParams::<D>::new(8).unwrap();
        let values: Vec<F> = (0..params.packed_len())
            .map(|i| F::from_u64((i + 1) as u64))
            .collect();
        let packed = psi_pack(params, &values).unwrap();
        let coeffs = packed.coefficients();
        let half = params.packed_len() / 2;

        assert_eq!(&coeffs[..half], &values[..half]);
        assert!(coeffs[half..D / 2].iter().all(|coeff| coeff.is_zero()));
        assert_eq!(&coeffs[D / 2..D / 2 + half], &values[half..]);
        assert!(coeffs[D / 2 + half..].iter().all(|coeff| coeff.is_zero()));
    }

    #[test]
    fn psi_pack_rejects_wrong_length() {
        let params = SubfieldParams::<8>::new(2).unwrap();
        assert!(matches!(
            psi_pack::<F, 8>(params, &[F::one()]),
            Err(AkitaError::InvalidSize {
                expected: 4,
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
}
