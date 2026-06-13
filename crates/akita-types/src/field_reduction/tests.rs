use super::*;
use crate::{reduce_inner_opening_to_ring_element, BasisMode};
use akita_field::{
    ExtField, Fp32, RingSubfieldFpExt4, RingSubfieldFpExt8, TowerBasisFpExt4, TwoNr, UnitNr,
};

type F = Fp32<251>;
type AkitaF32 = Fp32<4294967197>;

fn ring_from_i64s<const D: usize>(values: [i64; D]) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(values.map(F::from_i64))
}

fn ring_from_index<const D: usize>() -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|i| F::from_u64((i + 1) as u64)))
}

fn ring_subfield_basis<Fq: FieldCore, const D: usize, const K: usize>(
    _params: SubfieldParams<D, K>,
) -> Vec<CyclotomicRing<Fq, D>> {
    let step = D / (2 * K);
    let mut basis = Vec::with_capacity(K);
    basis.push(CyclotomicRing::one());
    for i in 1..K {
        let pos = i * step;
        let mut coeffs = [Fq::zero(); D];
        coeffs[pos] = Fq::one();
        coeffs[D - pos] = -Fq::one();
        basis.push(CyclotomicRing::from_coefficients(coeffs));
    }
    basis
}

fn ring_subfield_coords<Fq: FieldCore, const D: usize, const K: usize>(
    _params: SubfieldParams<D, K>,
    x: &CyclotomicRing<Fq, D>,
) -> Vec<Fq> {
    let step = D / (2 * K);
    let coeffs = x.coefficients();
    let mut coords = vec![Fq::zero(); K];
    coords[0] = coeffs[0];

    for (i, coord) in coords.iter_mut().enumerate().take(K).skip(1) {
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
            || (1..K).any(|i| {
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

fn embed_tower_in_ring_subfield<const D: usize>(
    x: TowerBasisFpExt4<AkitaF32, TwoNr, UnitNr>,
) -> CyclotomicRing<AkitaF32, D> {
    let params = SubfieldParams::<D, 4>::new().unwrap();
    let basis = ring_subfield_basis::<AkitaF32, D, 4>(params);

    // Over 2^32 - 99, i is a square root of -1 and a satisfies
    // a^2 = 1 / (2 * (1 + i)). Thus v = a*e1 + a*i*e3 has v^2 = e2.
    let a = AkitaF32::from_u64(1_492_342_050);
    let ai = a * AkitaF32::from_u64(3_311_696_422);
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
    assert!(SubfieldParams::<8, 1>::new().is_ok());
    assert!(SubfieldParams::<8, 4>::new().is_ok());

    assert!(matches!(
        SubfieldParams::<8, 0>::new(),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        SubfieldParams::<8, 3>::new(),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        SubfieldParams::<9, 1>::new(),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        SubfieldParams::<6, 1>::new(),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        SubfieldParams::<10, 1>::new(),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        SubfieldParams::<{ usize::MAX - 1 }, 1>::new(),
        Err(AkitaError::InvalidInput(_))
    ));
}

#[test]
fn h_exponents_match_power_of_two_subgroups() {
    assert_eq!(
        SubfieldParams::<8, 1>::new().unwrap().h_exponents().len(),
        8
    );
    assert_eq!(
        SubfieldParams::<8, 2>::new().unwrap().h_exponents().len(),
        4
    );
    assert_eq!(
        SubfieldParams::<8, 4>::new().unwrap().h_exponents().len(),
        2
    );

    assert_eq!(
        SubfieldParams::<8, 2>::new().unwrap().h_exponents(),
        vec![1, 7, 9, 15]
    );
}

#[test]
fn h_exponents_cover_production_ring_subgroups() {
    assert_eq!(
        SubfieldParams::<64, 1>::new().unwrap().h_exponents().len(),
        64
    );
    assert_eq!(
        SubfieldParams::<64, 8>::new().unwrap().h_exponents().len(),
        8
    );
    assert_eq!(
        SubfieldParams::<128, 1>::new().unwrap().h_exponents().len(),
        128
    );
    assert_eq!(
        SubfieldParams::<128, 16>::new()
            .unwrap()
            .h_exponents()
            .len(),
        8
    );
}

#[test]
fn trace_h_k_one_matches_constant_coefficient_trace() {
    const D: usize = 8;
    let params = SubfieldParams::<D, 1>::new().unwrap();
    let x = ring_from_i64s([3, 5, 7, 11, 13, 17, 19, 23]);
    let trace = trace_h(params, &x);
    let coeffs = trace.coefficients();

    assert_eq!(coeffs[0], F::from_u64(D as u64) * x.coefficients()[0]);
    assert!(coeffs[1..].iter().all(|coeff| coeff.is_zero()));
}

#[test]
fn trace_h_k_one_matches_constant_coefficient_trace_at_production_sizes() {
    let params_64 = SubfieldParams::<64, 1>::new().unwrap();
    let x_64 = ring_from_index::<64>();
    let trace_64 = trace_h(params_64, &x_64);
    assert_eq!(
        trace_64.coefficients()[0],
        F::from_u64(64) * x_64.coefficients()[0]
    );
    assert!(trace_64.coefficients()[1..]
        .iter()
        .all(|coeff| coeff.is_zero()));

    let params_128 = SubfieldParams::<128, 1>::new().unwrap();
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
    let params = SubfieldParams::<D, 1>::new().unwrap();
    let y_ring = ring_from_i64s([2, 3, 5, 7, 11, 13, 17, 19]);
    let inner_point = [F::from_u64(3), F::from_u64(5), F::from_u64(7)];

    for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
        let packed_inner =
            reduce_inner_opening_to_ring_element::<F, D>(&inner_point, basis).unwrap();
        let product = y_ring * packed_inner.sigma_m1();
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
    let params = SubfieldParams::<D, 2>::new().unwrap();
    let x = ring_from_i64s([1, 2, 3, 4, 5, 6, 7, 8]);
    let mut expected = CyclotomicRing::zero();

    for exponent in [1, 7, 9, 15] {
        expected += x.sigma(exponent);
    }

    assert_eq!(trace_h(params, &x), expected);
}

/// Build a flat `psi_embed` input from `D / K` subfield elements where
/// only the constant (`e_0 = 1`) coordinate is set.
fn constants_only_coords<const D: usize, const K: usize>(
    params: SubfieldParams<D, K>,
    values: &[F],
) -> Vec<F> {
    assert_eq!(values.len(), params.packed_len());
    let mut coords = vec![F::zero(); D];
    for (i, value) in values.iter().enumerate() {
        coords[i * K] = *value;
    }
    coords
}

#[test]
fn psi_embed_constants_only_matches_paper_positions() {
    const D: usize = 8;
    let params = SubfieldParams::<D, 2>::new().unwrap();
    let coords = constants_only_coords(
        params,
        &[
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ],
    );
    let packed = psi_embed::<F, D, 2>(params, &coords).unwrap();
    let expected = ring_from_i64s([1, 2, 0, 0, 3, 4, 0, 0]);

    assert_eq!(packed, expected);
}

#[test]
fn psi_embed_constants_only_at_production_ring_size() {
    const D: usize = 64;
    let params = SubfieldParams::<D, 8>::new().unwrap();
    let values: Vec<F> = (0..params.packed_len())
        .map(|i| F::from_u64((i + 1) as u64))
        .collect();
    let coords = constants_only_coords(params, &values);
    let packed = psi_embed::<F, D, 8>(params, &coords).unwrap();
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
    let params = SubfieldParams::<D, 1>::new().unwrap();
    let coords: Vec<F> = (0..D).map(|i| F::from_u64((i + 1) as u64)).collect();
    let packed = psi_embed::<F, D, 1>(params, &coords).unwrap();
    let expected = ring_from_i64s([1, 2, 3, 4, 5, 6, 7, 8]);

    assert_eq!(packed, expected);
}

#[test]
fn psi_embed_rejects_wrong_length() {
    let params = SubfieldParams::<8, 2>::new().unwrap();
    assert!(matches!(
        psi_embed::<F, 8, 2>(params, &[F::one()]),
        Err(AkitaError::InvalidSize {
            expected: 8,
            actual: 1
        })
    ));
}

/// `psi_embed` of "first slot only, rest zero" must agree with
/// [`embed_subfield`], which is the slot-0 fast path used by the verifier.
fn assert_psi_embed_slot_zero_matches_embed_subfield<const D: usize, const K: usize>() {
    let params = SubfieldParams::<D, K>::new().unwrap();
    let single: [AkitaF32; K] = std::array::from_fn(|j| AkitaF32::from_u64(2 + 7 * j as u64));

    let mut coords = vec![AkitaF32::zero(); D];
    coords[..K].copy_from_slice(&single);

    let packed = psi_embed::<AkitaF32, D, K>(params, &coords).unwrap();
    let direct = embed_subfield::<AkitaF32, D, K>(params, &single);

    assert_eq!(packed, direct);
}

#[test]
fn embed_subfield_matches_psi_embed_first_slot() {
    assert_psi_embed_slot_zero_matches_embed_subfield::<8, 2>();
    assert_psi_embed_slot_zero_matches_embed_subfield::<8, 4>();
    assert_psi_embed_slot_zero_matches_embed_subfield::<64, 4>();
    assert_psi_embed_slot_zero_matches_embed_subfield::<64, 8>();
    assert_psi_embed_slot_zero_matches_embed_subfield::<128, 16>();
}

fn assert_embed_subfield_scales_packed_slots<const D: usize, const K: usize>() {
    let params = SubfieldParams::<D, K>::new().unwrap();
    let packed_len = D / K;
    let slot_coords = (0..D)
        .map(|idx| AkitaF32::from_u64((3 * idx + 5) as u64))
        .collect::<Vec<_>>();
    let gamma: [AkitaF32; K] = std::array::from_fn(|idx| AkitaF32::from_u64((7 * idx + 2) as u64));

    let basis = ring_subfield_basis::<AkitaF32, D, K>(params);
    let gamma_ring = embed_subfield::<AkitaF32, D, K>(params, &gamma);
    let packed = psi_embed::<AkitaF32, D, K>(params, &slot_coords).unwrap();
    let scaled = gamma_ring * packed;

    let mut expected_coords = Vec::with_capacity(D);
    for slot in 0..packed_len {
        let slot_ring = slot_coords[(slot * K)..((slot + 1) * K)]
            .iter()
            .zip(basis.iter())
            .fold(
                CyclotomicRing::<AkitaF32, D>::zero(),
                |acc, (&coord, basis)| acc + basis.scale(&coord),
            );
        let product = gamma_ring * slot_ring;
        expected_coords.extend(ring_subfield_coords(params, &product));
    }
    let expected = psi_embed::<AkitaF32, D, K>(params, &expected_coords).unwrap();
    assert_eq!(scaled, expected);
}

#[test]
fn embed_subfield_scales_packed_slots() {
    assert_embed_subfield_scales_packed_slots::<8, 2>();
    assert_embed_subfield_scales_packed_slots::<8, 4>();
}

#[test]
fn ring_subfield_k4_basis_has_chebyshev_multiplication_table() {
    const D: usize = 8;
    let params = SubfieldParams::<D, 4>::new().unwrap();
    let basis = ring_subfield_basis::<AkitaF32, D, 4>(params);
    let two = AkitaF32::from_u64(2);

    assert_eq!(
        ring_subfield_coords(params, &(basis[1] * basis[1])),
        vec![two, AkitaF32::zero(), AkitaF32::one(), AkitaF32::zero()]
    );
    assert_eq!(
        ring_subfield_coords(params, &(basis[1] * basis[2])),
        vec![
            AkitaF32::zero(),
            AkitaF32::one(),
            AkitaF32::zero(),
            AkitaF32::one()
        ]
    );
    assert_eq!(
        ring_subfield_coords(params, &(basis[1] * basis[3])),
        vec![
            AkitaF32::zero(),
            AkitaF32::zero(),
            AkitaF32::one(),
            AkitaF32::zero()
        ]
    );
    assert_eq!(
        ring_subfield_coords(params, &(basis[2] * basis[2])),
        vec![two, AkitaF32::zero(), AkitaF32::zero(), AkitaF32::zero()]
    );
    assert_eq!(
        ring_subfield_coords(params, &(basis[2] * basis[3])),
        vec![
            AkitaF32::zero(),
            AkitaF32::one(),
            AkitaF32::zero(),
            -AkitaF32::one()
        ]
    );
    assert_eq!(
        ring_subfield_coords(params, &(basis[3] * basis[3])),
        vec![two, AkitaF32::zero(), -AkitaF32::one(), AkitaF32::zero()]
    );
}

#[test]
fn naive_k4_basis_is_not_the_current_tower_power_basis() {
    const D: usize = 8;
    let params = SubfieldParams::<D, 4>::new().unwrap();
    let basis = ring_subfield_basis::<AkitaF32, D, 4>(params);

    assert_ne!(basis[1] * basis[1], basis[2]);
    assert_eq!(
        ring_subfield_coords(params, &(basis[1] * basis[1])),
        vec![
            AkitaF32::from_u64(2),
            AkitaF32::zero(),
            AkitaF32::one(),
            AkitaF32::zero()
        ]
    );
}

#[test]
fn ring_subfield_k4_contains_current_tower_after_base_change() {
    const D: usize = 8;
    type E = TowerBasisFpExt4<AkitaF32, TwoNr, UnitNr>;

    let params = SubfieldParams::<D, 4>::new().unwrap();
    let basis = ring_subfield_basis::<AkitaF32, D, 4>(params);
    let a = AkitaF32::from_u64(1_492_342_050);
    let ai = a * AkitaF32::from_u64(3_311_696_422);
    let v = basis[1].scale(&a) + basis[3].scale(&ai);
    let u = basis[2];

    assert_eq!(v * v, u);
    assert_eq!(u * u, basis[0].scale(&AkitaF32::from_u64(2)));
    assert_eq!(v * v * v * v, basis[0].scale(&AkitaF32::from_u64(2)));

    let x = E::from_base_slice(&[
        AkitaF32::from_u64(3),
        AkitaF32::from_u64(5),
        AkitaF32::from_u64(7),
        AkitaF32::from_u64(11),
    ]);
    let y = E::from_base_slice(&[
        AkitaF32::from_u64(13),
        AkitaF32::from_u64(17),
        AkitaF32::from_u64(19),
        AkitaF32::from_u64(23),
    ]);

    assert_eq!(
        embed_tower_in_ring_subfield::<D>(x * y),
        embed_tower_in_ring_subfield::<D>(x) * embed_tower_in_ring_subfield::<D>(y)
    );
}

fn assert_ring_subfield_fp_ext4_embedding_is_multiplicative<const D: usize>() {
    let params = SubfieldParams::<D, 4>::new().unwrap();
    let x = RingSubfieldFpExt4::new([
        AkitaF32::from_u64(3),
        AkitaF32::from_u64(5),
        AkitaF32::from_u64(7),
        AkitaF32::from_u64(11),
    ]);
    let y = RingSubfieldFpExt4::new([
        AkitaF32::from_u64(13),
        AkitaF32::from_u64(17),
        AkitaF32::from_u64(19),
        AkitaF32::from_u64(23),
    ]);

    assert_eq!(
        embed_subfield::<AkitaF32, D, 4>(params, &(x * y).coeffs),
        embed_subfield::<AkitaF32, D, 4>(params, &x.coeffs)
            * embed_subfield::<AkitaF32, D, 4>(params, &y.coeffs)
    );
}

#[test]
fn ring_subfield_fp_ext4_embedding_places_coefficients_in_ring_subfield_basis() {
    const D: usize = 8;
    let params = SubfieldParams::<D, 4>::new().unwrap();
    let x = RingSubfieldFpExt4::new([
        AkitaF32::from_u64(2),
        AkitaF32::from_u64(3),
        AkitaF32::from_u64(5),
        AkitaF32::from_u64(7),
    ]);
    let embedded = embed_subfield::<AkitaF32, D, 4>(params, &x.coeffs);
    let coeffs = embedded.coefficients();

    assert_eq!(coeffs[0], AkitaF32::from_u64(2));
    assert_eq!(coeffs[1], AkitaF32::from_u64(3));
    assert_eq!(coeffs[7], -AkitaF32::from_u64(3));
    assert_eq!(coeffs[2], AkitaF32::from_u64(5));
    assert_eq!(coeffs[6], -AkitaF32::from_u64(5));
    assert_eq!(coeffs[3], AkitaF32::from_u64(7));
    assert_eq!(coeffs[5], -AkitaF32::from_u64(7));
    assert!(coeffs[4].is_zero());
}

#[test]
fn ring_subfield_fp_ext4_embedding_is_multiplicative_across_ring_dimensions() {
    assert_ring_subfield_fp_ext4_embedding_is_multiplicative::<8>();
    assert_ring_subfield_fp_ext4_embedding_is_multiplicative::<64>();
    assert_ring_subfield_fp_ext4_embedding_is_multiplicative::<128>();
}

fn assert_ring_subfield_fp_ext8_embedding_is_multiplicative<const D: usize>() {
    let params = SubfieldParams::<D, 8>::new().unwrap();
    let x = RingSubfieldFpExt8::new([
        AkitaF32::from_u64(2),
        AkitaF32::from_u64(3),
        AkitaF32::from_u64(5),
        AkitaF32::from_u64(7),
        AkitaF32::from_u64(11),
        AkitaF32::from_u64(13),
        AkitaF32::from_u64(17),
        AkitaF32::from_u64(19),
    ]);
    let y = RingSubfieldFpExt8::new([
        AkitaF32::from_u64(23),
        AkitaF32::from_u64(29),
        AkitaF32::from_u64(31),
        AkitaF32::from_u64(37),
        AkitaF32::from_u64(41),
        AkitaF32::from_u64(43),
        AkitaF32::from_u64(47),
        AkitaF32::from_u64(53),
    ]);

    assert_eq!(
        embed_subfield::<AkitaF32, D, 8>(params, &(x * y).coeffs),
        embed_subfield::<AkitaF32, D, 8>(params, &x.coeffs)
            * embed_subfield::<AkitaF32, D, 8>(params, &y.coeffs)
    );
}

#[test]
fn ring_subfield_fp_ext8_embedding_places_coefficients_in_ring_subfield_basis() {
    const D: usize = 16;
    let params = SubfieldParams::<D, 8>::new().unwrap();
    let x = RingSubfieldFpExt8::new([
        AkitaF32::from_u64(2),
        AkitaF32::from_u64(3),
        AkitaF32::from_u64(5),
        AkitaF32::from_u64(7),
        AkitaF32::from_u64(11),
        AkitaF32::from_u64(13),
        AkitaF32::from_u64(17),
        AkitaF32::from_u64(19),
    ]);
    let embedded = embed_subfield::<AkitaF32, D, 8>(params, &x.coeffs);
    let coeffs = embedded.coefficients();

    assert_eq!(coeffs[0], AkitaF32::from_u64(2));
    for j in 1..8 {
        assert_eq!(coeffs[j], x.coeffs[j]);
        assert_eq!(coeffs[D - j], -x.coeffs[j]);
    }
    assert!(coeffs[8].is_zero());
}

#[test]
fn ring_subfield_fp_ext8_embedding_is_multiplicative_across_ring_dimensions() {
    assert_ring_subfield_fp_ext8_embedding_is_multiplicative::<16>();
    assert_ring_subfield_fp_ext8_embedding_is_multiplicative::<64>();
    assert_ring_subfield_fp_ext8_embedding_is_multiplicative::<128>();
}

/// Generate `D / 4` deterministic `RingSubfieldFpExt4` elements seeded by `tag`.
fn deterministic_subfield_fp_ext4_vector<const D: usize>(
    tag: u64,
) -> Vec<RingSubfieldFpExt4<AkitaF32>> {
    let m = D / 4;
    (0..m)
        .map(|i| {
            let i = i as u64;
            RingSubfieldFpExt4::new([
                AkitaF32::from_u64(2 + 7 * i + 11 * tag),
                AkitaF32::from_u64(3 + 13 * i + 17 * tag),
                AkitaF32::from_u64(5 + 19 * i + 23 * tag),
                AkitaF32::from_u64(7 + 29 * i + 31 * tag),
            ])
        })
        .collect()
}

/// Flatten `D / 4` typed `RingSubfieldFpExt4` slots into the
/// `[s_0[0], s_0[1], s_0[2], s_0[3], s_1[0], ...]` layout consumed by
/// [`psi_embed`].
fn flatten_subfield_fp_ext4_vector<const D: usize>(
    elements: &[RingSubfieldFpExt4<AkitaF32>],
) -> Vec<AkitaF32> {
    assert_eq!(elements.len(), D / 4);
    let mut coords = vec![AkitaF32::zero(); D];
    for (i, elem) in elements.iter().enumerate() {
        coords[i * 4..i * 4 + 4].copy_from_slice(&elem.coeffs);
    }
    coords
}

/// Verify the trace inner-product relation
/// `Tr_H(psi(s) * sigma_{-1}(psi(r_in))) = (D / k) * embed_subfield(<s, s>)`
/// for the typed `k = 4` ring-subfield representation.
fn assert_psi_trace_inner_product_identity_fp_ext4<const D: usize>() {
    let params = SubfieldParams::<D, 4>::new().unwrap();
    let s = deterministic_subfield_fp_ext4_vector::<D>(0);
    let v = deterministic_subfield_fp_ext4_vector::<D>(1);

    // y = <s, v> in the ring-subfield.
    let y = s
        .iter()
        .zip(v.iter())
        .fold(RingSubfieldFpExt4::zero(), |acc, (si, vi)| {
            acc + (*si * *vi)
        });

    let s_flat = flatten_subfield_fp_ext4_vector::<D>(&s);
    let v_flat = flatten_subfield_fp_ext4_vector::<D>(&v);
    let big_y = psi_embed::<AkitaF32, D, 4>(params, &s_flat).unwrap();
    let big_v = psi_embed::<AkitaF32, D, 4>(params, &v_flat).unwrap();
    let traced = trace_h(params, &(big_y * big_v.sigma_m1()));

    let scale = AkitaF32::from_u64(params.packed_len() as u64);
    let scaled = embed_subfield::<AkitaF32, D, 4>(params, &y.coeffs).scale(&scale);

    assert_eq!(traced, scaled);
}

#[test]
fn psi_trace_inner_product_identity_fp_ext4() {
    assert_psi_trace_inner_product_identity_fp_ext4::<8>();
    assert_psi_trace_inner_product_identity_fp_ext4::<64>();
    assert_psi_trace_inner_product_identity_fp_ext4::<128>();
}

/// Subfield multiplication for `k = 2`: `e_1^2 = 2` for any valid `D`,
/// so `R_q^H ≅ F_q[sqrt(2)]`.
fn fp_ext2_subfield_mul(a: [AkitaF32; 2], b: [AkitaF32; 2]) -> [AkitaF32; 2] {
    let two = AkitaF32::from_u64(2);
    [a[0] * b[0] + two * a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn assert_psi_trace_inner_product_identity_fp_ext2<const D: usize>() {
    let params = SubfieldParams::<D, 2>::new().unwrap();
    let m = params.packed_len();

    let s: Vec<[AkitaF32; 2]> = (0..m)
        .map(|i| {
            let i = i as u64;
            [
                AkitaF32::from_u64(2 + 7 * i),
                AkitaF32::from_u64(3 + 13 * i),
            ]
        })
        .collect();
    let v: Vec<[AkitaF32; 2]> = (0..m)
        .map(|i| {
            let i = i as u64;
            [
                AkitaF32::from_u64(11 + 19 * i),
                AkitaF32::from_u64(17 + 23 * i),
            ]
        })
        .collect();

    let y = s
        .iter()
        .zip(v.iter())
        .fold([AkitaF32::zero(); 2], |acc, (si, vi)| {
            let prod = fp_ext2_subfield_mul(*si, *vi);
            [acc[0] + prod[0], acc[1] + prod[1]]
        });

    let mut s_flat = vec![AkitaF32::zero(); D];
    let mut v_flat = vec![AkitaF32::zero(); D];
    for (i, (sc, vc)) in s.iter().zip(v.iter()).enumerate() {
        s_flat[i * 2] = sc[0];
        s_flat[i * 2 + 1] = sc[1];
        v_flat[i * 2] = vc[0];
        v_flat[i * 2 + 1] = vc[1];
    }

    let big_y = psi_embed::<AkitaF32, D, 2>(params, &s_flat).unwrap();
    let big_v = psi_embed::<AkitaF32, D, 2>(params, &v_flat).unwrap();
    let traced = trace_h(params, &(big_y * big_v.sigma_m1()));

    let scale = AkitaF32::from_u64(m as u64);
    let scaled = embed_subfield::<AkitaF32, D, 2>(params, &y).scale(&scale);

    assert_eq!(traced, scaled);
}

#[test]
fn check_trace_inner_product_k_one_accepts_correct_opening() {
    const D: usize = 8;
    let params = SubfieldParams::<D, 1>::new().unwrap();
    let y_ring = ring_from_i64s([2, 3, 5, 7, 11, 13, 17, 19]);
    let inner_point = [F::from_u64(3), F::from_u64(5), F::from_u64(7)];

    for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
        let packed_inner =
            reduce_inner_opening_to_ring_element::<F, D>(&inner_point, basis).unwrap();
        let product = y_ring * packed_inner.sigma_m1();
        let opening = product.coefficients()[0];

        assert!(check_trace_inner_product::<F, D, 1>(
            params,
            &product,
            &[opening]
        ));
    }
}

#[test]
fn check_trace_inner_product_k_one_rejects_wrong_opening() {
    const D: usize = 8;
    let params = SubfieldParams::<D, 1>::new().unwrap();
    let y_ring = ring_from_i64s([2, 3, 5, 7, 11, 13, 17, 19]);
    let v = ring_from_i64s([1, 1, 1, 1, 1, 1, 1, 1]);
    let product = y_ring * v.sigma_m1();
    let wrong = product.coefficients()[0] + F::one();

    assert!(!check_trace_inner_product::<F, D, 1>(
        params,
        &product,
        &[wrong]
    ));
}

/// Verify [`check_trace_inner_product`] against the K-generic ring
/// identity for `K = 4`, both on a true witness and on a perturbed
/// witness, across all production ring sizes.
fn assert_check_trace_inner_product_fp_ext4<const D: usize>() {
    let params = SubfieldParams::<D, 4>::new().unwrap();
    let s = deterministic_subfield_fp_ext4_vector::<D>(0);
    let v = deterministic_subfield_fp_ext4_vector::<D>(1);

    let y = s
        .iter()
        .zip(v.iter())
        .fold(RingSubfieldFpExt4::zero(), |acc, (si, vi)| {
            acc + (*si * *vi)
        });
    let s_flat = flatten_subfield_fp_ext4_vector::<D>(&s);
    let v_flat = flatten_subfield_fp_ext4_vector::<D>(&v);
    let big_y = psi_embed::<AkitaF32, D, 4>(params, &s_flat).unwrap();
    let big_v = psi_embed::<AkitaF32, D, 4>(params, &v_flat).unwrap();
    let trace_input = big_y * big_v.sigma_m1();

    assert!(check_trace_inner_product::<AkitaF32, D, 4>(
        params,
        &trace_input,
        &y.coeffs
    ));

    let mut wrong = y.coeffs;
    wrong[0] += AkitaF32::one();
    assert!(!check_trace_inner_product::<AkitaF32, D, 4>(
        params,
        &trace_input,
        &wrong
    ));
}

#[test]
fn check_trace_inner_product_fp_ext4_across_ring_dimensions() {
    assert_check_trace_inner_product_fp_ext4::<8>();
    assert_check_trace_inner_product_fp_ext4::<64>();
    assert_check_trace_inner_product_fp_ext4::<128>();
}

#[test]
fn psi_trace_inner_product_identity_fp_ext2() {
    assert_psi_trace_inner_product_identity_fp_ext2::<8>();
    assert_psi_trace_inner_product_identity_fp_ext2::<64>();
    assert_psi_trace_inner_product_identity_fp_ext2::<128>();
}
