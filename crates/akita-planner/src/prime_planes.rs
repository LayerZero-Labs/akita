//! Admission inputs for explicitly selected binary prime evaluation planes.

use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::sis::ajtai_key::{
    min_secure_rank, sis_table_key_for_linf_bound, SisMatrixRole, SisModulusProfileId,
    SisTableDigest, DEFAULT_SIS_SECURITY_POLICY,
};
use akita_params::GroupCommitPhaseParams;
use akita_types::binary_source::{
    BinaryCommittedGroupManifest, BinaryRootWitnessPlan, BinaryTypedPrefix, OrdinaryNativeRootPlan,
    PrimePlaneManifest,
};

/// Exact witness and D widths, without a production-admission assertion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryPrimePlanePrice {
    pub planes: usize,
    /// Natural digits concatenated behind the ordinary D payload.
    pub d_live_digits: usize,
    /// Degree-64 setup columns, including the final partial chunk.
    pub d_columns: usize,
    pub d_rank: usize,
    /// Minimum rank found in the existing generated scalar SIS table, if any.
    /// This lookup does not certify the binary root or its ordinary suffix.
    pub d_table_min_rank: Option<usize>,
    /// Physical cells through the last live segment, including virtual tails.
    pub outgoing_physical_cells: usize,
    /// Next power-of-two ordinary witness domain.
    pub outgoing_domain_len: usize,
    /// Ring elements needed by that ordinary domain at the chosen dimension.
    pub outgoing_ring_elements: usize,
    pub segment_physical_cells: Vec<usize>,
    /// Typed prefix (A, then B/D setup) the first successor's setup prefix must hold.
    pub typed_prefix: BinaryTypedPrefix,
}

/// Price every selected plane through the verifier's canonical witness plan.
///
/// Optional ordinary-source preimages are appended through the same plan used
/// by the grouped root. The D lookup uses the balanced digit difference bound
/// seven, the actual degree-64 modulus profile and the current table digest.
/// No estimator or new certificate is run here; all widened ordinary instances
/// still require their schedule's independent admission checks.
pub fn price_binary_root_planes(
    manifest: &BinaryCommittedGroupManifest,
    word: bool,
    prime: &PrimePlaneManifest,
    destination_ring_dimension: usize,
    ordinary: Option<GroupCommitPhaseParams>,
) -> Result<BinaryPrimePlanePrice, AkitaError> {
    manifest.validate()?;
    prime.validate(manifest.descriptor())?;
    if prime.field_profile() != manifest.field_profile() && !prime.planes().is_empty() {
        return Err(AkitaError::InvalidInput(
            "prime field differs from the commitment".into(),
        ));
    }
    let mut plan = BinaryRootWitnessPlan::for_profile(
        manifest.descriptor(),
        manifest.a_rank(),
        destination_ring_dimension,
        manifest.field_profile(),
        word,
        prime.planes().len(),
    )?;
    if let Some(profile) = ordinary {
        let ordinary = OrdinaryNativeRootPlan::new(
            profile,
            plan.quotient_segment()
                .unwrap_or(plan.repacking().segments().len()),
        )?;
        plan = plan.with_ordinary(&ordinary)?;
    }
    let repacking = plan.repacking();
    let d_live_digits = checked::sum(
        plan.opening_segments()
            .iter()
            .map(|&segment| {
                let shape = repacking
                    .segments()
                    .get(segment)
                    .ok_or(AkitaError::InvalidProof)?;
                checked::product([
                    shape.elements,
                    shape.digits,
                    shape.geometry.polynomial_modulus_dimension(),
                    shape.geometry.coordinate_plane_count(),
                ])
                .ok_or(AkitaError::InvalidProof)
            })
            .collect::<Result<Vec<_>, _>>()?,
    )
    .ok_or(AkitaError::InvalidProof)?;
    let d_columns = checked::div_ceil(d_live_digits, manifest.b_ring_dimension())
        .ok_or(AkitaError::InvalidProof)?;
    let d_table_min_rank = sis_table_key_for_linf_bound(
        DEFAULT_SIS_SECURITY_POLICY,
        SisTableDigest::CURRENT,
        match manifest.field_profile() {
            BinaryFieldProfile::P128 => SisModulusProfileId::Q128Offset275,
            BinaryFieldProfile::P64 => SisModulusProfileId::Q64Offset59,
        },
        SisMatrixRole::Open,
        u32::try_from(manifest.b_ring_dimension()).map_err(|_| AkitaError::InvalidProof)?,
        7,
    )
    .and_then(|key| {
        u64::try_from(d_columns)
            .ok()
            .and_then(|width| min_secure_rank(key, width))
    });
    let segment_physical_cells = (0..repacking.segments().len())
        .map(|segment| repacking.segment_range(segment).map(|range| range.len()))
        .collect::<Result<Vec<_>, _>>()?;
    let outgoing_physical_cells =
        checked::sum(segment_physical_cells.iter().copied()).ok_or(AkitaError::InvalidProof)?;
    Ok(BinaryPrimePlanePrice {
        planes: prime.planes().len(),
        d_live_digits,
        d_columns,
        d_rank: manifest.d_label(word, prime.planes().len())?.rank as usize,
        d_table_min_rank,
        outgoing_physical_cells,
        outgoing_domain_len: repacking.domain_len(),
        outgoing_ring_elements: checked::exact_div(
            repacking.domain_len(),
            destination_ring_dimension,
        )
        .ok_or(AkitaError::InvalidProof)?,
        segment_physical_cells,
        typed_prefix: BinaryTypedPrefix::from_plan(manifest, &plan)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::FoldChallengeDistribution;
    use akita_params::{CommittedGroupParams, PolynomialGroupLayout};
    use akita_types::binary_source::{
        BinaryIndexMap, BinaryRootPublic, BinarySourceDescriptor, BinaryTerminalClaim,
        PrimeGroupClaim, PrimePlaneClaim,
    };

    fn manifest(
        carrier_bits: usize,
        columns: usize,
        owners: usize,
    ) -> BinaryCommittedGroupManifest {
        let params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128Offset275,
            64,
            3,
            3,
            2,
            2,
            FoldChallengeDistribution::pm1_only(3),
        )
        .with_decomp(2, 2, 2, 2, 2)
        .unwrap();
        let profile = GroupCommitPhaseParams::from_params_unchecked_for_test(
            PolynomialGroupLayout::singleton(6),
            &params,
        );
        let count = 1usize << carrier_bits;
        let index = BinaryIndexMap::new(128, count, count, columns).unwrap();
        let profiles = BinaryCommittedGroupManifest::profiles_for_field(
            &index,
            owners,
            7,
            BinaryFieldProfile::P128,
        )
        .unwrap();
        let descriptor = BinarySourceDescriptor::new(
            index,
            (0..owners).map(|x| [x as u8; 32]).collect(),
            profiles,
        )
        .unwrap();
        BinaryCommittedGroupManifest::with_field_profile(
            descriptor,
            7,
            &profile,
            BinaryFieldProfile::P128,
        )
        .unwrap()
    }

    fn prime(manifest: &BinaryCommittedGroupManifest, planes: usize) -> PrimePlaneManifest {
        let arity = manifest
            .descriptor()
            .index_map()
            .padded_carriers()
            .trailing_zeros() as usize
            + 8;
        PrimePlaneManifest::with_field_profile(
            manifest.descriptor(),
            (planes > 0).then(|| PrimeGroupClaim {
                point: vec![2.into(); arity],
                evaluations: (0..planes)
                    .map(|polynomial| PrimePlaneClaim {
                        polynomial,
                        value: 0.into(),
                    })
                    .collect(),
            }),
            manifest.field_profile(),
        )
        .unwrap()
    }

    fn ordinary_profile(field: BinaryFieldProfile) -> GroupCommitPhaseParams {
        use akita_params::sis::{InnerCommitMatrixParams, OuterCommitMatrixParams};
        use akita_params::{
            BlockGeometry, CommitmentSliceCount, CommitmentSliceGeometry, GadgetDigits, RoleParams,
        };
        let modulus = match field {
            BinaryFieldProfile::P128 => SisModulusProfileId::Q128Offset275,
            BinaryFieldProfile::P64 => SisModulusProfileId::Q64Offset59,
        };
        let key = |role, degree| {
            sis_table_key_for_linf_bound(
                DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::CURRENT,
                modulus,
                role,
                degree,
                8,
            )
            .unwrap()
        };
        let inner =
            InnerCommitMatrixParams::try_new_with_min_rank(key(SisMatrixRole::Inner, 128), 2)
                .unwrap();
        let digits = field.prime_plane_digits();
        let geometry = CommitmentSliceGeometry::try_new(
            CommitmentSliceCount::ONE,
            4,
            1,
            inner.output_rank(),
            digits,
            128,
            64,
        )
        .unwrap();
        let outer = OuterCommitMatrixParams::try_new_with_min_rank(
            key(SisMatrixRole::Outer, 64),
            geometry.physical_input_width(),
        )
        .unwrap();
        GroupCommitPhaseParams {
            version: GroupCommitPhaseParams::VERSION,
            group: PolynomialGroupLayout::singleton(10),
            blocks: BlockGeometry::new(8, 2, 4),
            outer_slice_count: CommitmentSliceCount::ONE,
            inner: RoleParams::new(GadgetDigits::new(3, 1), inner),
            outer: RoleParams::new(GadgetDigits::new(3, digits), outer),
        }
    }

    fn assert_price_matches_verifier_plan(
        manifest: &BinaryCommittedGroupManifest,
        word: bool,
        prime: &PrimePlaneManifest,
        ordinary: Option<GroupCommitPhaseParams>,
    ) {
        let price = price_binary_root_planes(manifest, word, prime, 64, ordinary).unwrap();
        let descriptor = manifest.descriptor();
        let index = descriptor.index_map();
        let claim = word.then(|| {
            BinaryTerminalClaim::new(
                descriptor,
                descriptor.polynomials()[0],
                vec![
                    [0; 21];
                    (index.padded_carriers() / index.columns()).trailing_zeros() as usize
                ],
                vec![[0; 21]; index.columns().trailing_zeros() as usize],
                [0; 21],
            )
            .unwrap()
        });
        let setup = akita_types::AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: akita_types::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
            setup_seed: [41; 32].into(),
        };
        let public = BinaryRootPublic::new(
            manifest,
            claim.as_ref(),
            prime,
            &setup,
            price.outgoing_physical_cells,
            64,
            ordinary,
        )
        .unwrap();
        let plan = public.plan();
        let segment_cells = plan
            .repacking()
            .segments()
            .iter()
            .map(|shape| {
                shape.elements * shape.digits * shape.geometry.physical_coefficient_width()
            })
            .collect::<Vec<_>>();
        assert_eq!(price.segment_physical_cells, segment_cells);
        assert_eq!(price.outgoing_domain_len, plan.repacking().domain_len());
        assert_eq!(
            price.outgoing_physical_cells,
            segment_cells.iter().sum::<usize>()
        );
        assert_eq!(
            price.typed_prefix,
            BinaryTypedPrefix::from_plan(manifest, plan).unwrap()
        );
        assert_eq!(
            price.d_rank,
            manifest.d_label(word, prime.planes().len()).unwrap().rank as usize
        );
        // D commits the natural scalar digits, excluding carries, ordinary rows and K.
        let digits = index.columns()
            * 162
            * (usize::from(word) * descriptor.polynomials().len()
                + prime.planes().len() * manifest.field_profile().prime_plane_digits());
        assert_eq!(price.d_live_digits, digits);
        assert_eq!(price.d_columns, digits.div_ceil(64));
        if let Some(quotient) = plan.quotient_segment() {
            assert_eq!(quotient, plan.repacking().segments().len() - 1);
            assert!(!plan.opening_segments().contains(&quotient));
        }
    }

    #[test]
    fn p64_prime_plane_price_uses_two_planes_and_independent_d_rank() {
        use akita_types::binary_source::PrimeClaimValue;
        let field = BinaryFieldProfile::P64;
        let params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q64Offset59,
            64,
            3,
            3,
            2,
            2,
            FoldChallengeDistribution::pm1_only(3),
        )
        .with_decomp(2, 2, 2, 2, 2)
        .unwrap();
        let carrier = GroupCommitPhaseParams::from_params_unchecked_for_test(
            PolynomialGroupLayout::singleton(6),
            &params,
        );
        let index = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
        let profiles =
            BinaryCommittedGroupManifest::profiles_for_field(&index, 1, 0, field).unwrap();
        let descriptor = BinarySourceDescriptor::new(index, vec![[1; 32]], profiles).unwrap();
        let manifest =
            BinaryCommittedGroupManifest::with_field_profile(descriptor, 0, &carrier, field)
                .unwrap();
        let descriptor = manifest.descriptor();
        let prime = PrimePlaneManifest::with_field_profile(
            descriptor,
            Some(PrimeGroupClaim {
                point: vec![PrimeClaimValue::from([2, 3]); 11],
                evaluations: vec![PrimePlaneClaim {
                    polynomial: 0,
                    value: [5, 7].into(),
                }],
            }),
            BinaryFieldProfile::P64,
        )
        .unwrap();
        for word in [false, true] {
            let price = price_binary_root_planes(&manifest, word, &prime, 64, None).unwrap();
            assert_eq!(price.planes, 2);
            assert_eq!(
                price.d_live_digits,
                (prime.planes().len() * field.prime_plane_digits() + usize::from(word)) * 162 * 2
            );
            assert_eq!(price.d_columns, price.d_live_digits.div_ceil(64));
            assert_eq!(
                price.d_rank,
                manifest.d_label(word, 2).unwrap().rank as usize
            );
            assert_eq!(price.d_table_min_rank, Some(price.d_rank));
            assert_ne!(price.d_rank, manifest.b_rows());
            assert_price_matches_verifier_plan(&manifest, word, &prime, None);
            assert_price_matches_verifier_plan(
                &manifest,
                word,
                &prime,
                Some(ordinary_profile(field)),
            );
        }
        let foreign_prime = PrimePlaneManifest::with_field_profile(
            descriptor,
            Some(PrimeGroupClaim {
                point: vec![2.into(); 11],
                evaluations: vec![PrimePlaneClaim {
                    polynomial: 0,
                    value: 5.into(),
                }],
            }),
            BinaryFieldProfile::P128,
        )
        .unwrap();
        assert!(price_binary_root_planes(&manifest, true, &foreign_prime, 64, None).is_err());
    }

    #[test]
    fn prime_plane_small_fixture_prices_every_plane() {
        for (planes, expected_cells, expected_d) in
            [(1, 79616, 223), (2, 159232, 446), (5, 398080, 1114)]
        {
            let manifest = manifest(3, 2, planes);
            let price =
                price_binary_root_planes(&manifest, true, &prime(&manifest, planes), 64, None)
                    .unwrap();
            assert_eq!(price.outgoing_physical_cells, expected_cells);
            assert_eq!(price.d_columns, expected_d);
            assert_eq!(price.segment_physical_cells.last(), Some(&(planes * 22016)));
            assert_price_matches_verifier_plan(&manifest, true, &prime(&manifest, planes), None);
            assert_price_matches_verifier_plan(&manifest, false, &prime(&manifest, planes), None);
            println!("prime_plane small planes={planes}: {price:?}");
        }
    }

    #[test]
    fn prime_plane_maximum_admission_inputs_at_large_geometries() {
        let mut typed_lengths = Vec::new();
        for bits in [22, 23] {
            for word in [false, true] {
                for planes in [1, 2, 5] {
                    let manifest = manifest(bits, 256, planes);
                    let price = price_binary_root_planes(
                        &manifest,
                        word,
                        &prime(&manifest, planes),
                        64,
                        None,
                    )
                    .unwrap();
                    let base = if bits == 22 { 23138048 } else { 39915264 };
                    let expected = planes * (base + 2818048 - if word { 0 } else { 69376 });
                    assert_eq!(price.outgoing_physical_cells, expected);
                    assert_eq!(
                        price.d_columns,
                        planes * (27864 + if word { 648 } else { 0 })
                    );
                    assert_eq!(price.outgoing_domain_len, expected.next_power_of_two());
                    typed_lengths.push(price.typed_prefix.natural_len);
                    assert_eq!(
                        price.typed_prefix.domain_len,
                        price.typed_prefix.natural_len.next_power_of_two()
                    );
                    println!("prime_plane n={bits} word={word} planes={planes}: {price:?}");
                }
            }
        }
        assert_eq!(
            typed_lengths,
            vec![
                20179968, 31709184, 66296832, 20262912, 31875072, 66711552, 33554432, 40359936,
                74947584, 33554432, 40525824, 75362304,
            ]
        );
    }

    #[test]
    fn prime_planes_are_bounded_by_the_polynomial_manifest() {
        let manifest = manifest(3, 2, 5);
        let price =
            price_binary_root_planes(&manifest, false, &prime(&manifest, 5), 64, None).unwrap();
        assert_eq!(price.planes, 5);
        assert_eq!(price.segment_physical_cells.last(), Some(&110080));
        assert_eq!(price.d_columns, 1089);
    }
}
