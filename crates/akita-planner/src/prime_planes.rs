//! Admission inputs for explicitly selected binary prime evaluation planes.

use akita_error::{checked, AkitaError};
use akita_params::sis::ajtai_key::{
    min_secure_rank, sis_table_key_for_linf_bound, SisMatrixRole, SisModulusProfileId,
    SisTableDigest, DEFAULT_SIS_SECURITY_POLICY,
};
use akita_params::GroupCommitPhaseParams;
use akita_types::binary_source::{
    BinaryCommittedGroupManifest, BinaryRootWitnessPlan, OrdinaryNativeRootPlan, PrimePlaneManifest,
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
    PrimePlaneManifest::new(manifest.descriptor(), prime.planes())?;
    let mut plan = BinaryRootWitnessPlan::new(
        manifest.descriptor(),
        manifest.a_rank(),
        destination_ring_dimension,
    )?
    .with_prime_planes(word, prime.planes().len())?;
    if let Some(profile) = ordinary {
        let ordinary = OrdinaryNativeRootPlan::new(profile, plan.repacking().segments().len())?;
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
        SisModulusProfileId::Q128Offset275,
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
        d_rank: manifest.b_rows(),
        d_table_min_rank,
        outgoing_physical_cells,
        outgoing_domain_len: repacking.domain_len(),
        outgoing_ring_elements: checked::exact_div(
            repacking.domain_len(),
            destination_ring_dimension,
        )
        .ok_or(AkitaError::InvalidProof)?,
        segment_physical_cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::FoldChallengeFamily;
    use akita_params::{CommittedGroupParams, PolynomialGroupLayout};
    use akita_types::binary_source::{BinaryIndexMap, BinarySourceDescriptor, PrimePlaneClaim};

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
            FoldChallengeFamily::pm1_only(3),
        )
        .with_decomp(2, 2, 2, 2, 2)
        .unwrap();
        let profile = GroupCommitPhaseParams::from_params_unchecked_for_test(
            PolynomialGroupLayout::singleton(6),
            &params,
        );
        let count = 1usize << carrier_bits;
        let index = BinaryIndexMap::new(128, count, count, columns).unwrap();
        let profiles =
            BinaryCommittedGroupManifest::commitment_profiles(&index, owners, 7).unwrap();
        let descriptor = BinarySourceDescriptor::new(
            index,
            (0..owners).map(|x| [x as u8; 32]).collect(),
            profiles,
        )
        .unwrap();
        BinaryCommittedGroupManifest::new(descriptor, 7, &profile).unwrap()
    }

    fn prime(manifest: &BinaryCommittedGroupManifest, planes: usize) -> PrimePlaneManifest {
        let arity = manifest
            .descriptor()
            .index_map()
            .padded_carriers()
            .trailing_zeros() as usize
            + 8;
        PrimePlaneManifest::new(
            manifest.descriptor(),
            &(0..planes)
                .map(|x| PrimePlaneClaim {
                    polynomial: x % manifest.descriptor().polynomials().len(),
                    point: vec![x as u128; arity],
                    tensor_component: 0,
                    value: 0,
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    #[test]
    fn prime_plane_small_fixture_prices_every_plane() {
        let manifest = manifest(3, 2, 1);
        for (planes, expected_cells, expected_d) in
            [(1, 104192, 223), (2, 126208, 441), (5, 192256, 1094)]
        {
            let price =
                price_binary_root_planes(&manifest, true, &prime(&manifest, planes), 64, None)
                    .unwrap();
            assert_eq!(price.outgoing_physical_cells, expected_cells);
            assert_eq!(price.d_columns, expected_d);
            assert_eq!(price.segment_physical_cells.last(), Some(&(planes * 22016)));
            println!("prime_plane small planes={planes}: {price:?}");
        }
    }

    #[test]
    fn prime_plane_maximum_admission_inputs_at_large_geometries() {
        for bits in [22, 23] {
            let manifest = manifest(bits, 256, 1);
            for word in [false, true] {
                for planes in [1, 2, 5] {
                    let price = price_binary_root_planes(
                        &manifest,
                        word,
                        &prime(&manifest, planes),
                        64,
                        None,
                    )
                    .unwrap();
                    let base = if bits == 22 { 23174912 } else { 39952128 };
                    let expected = base + planes * 2818048 - if word { 0 } else { 69376 };
                    assert_eq!(price.outgoing_physical_cells, expected);
                    assert_eq!(price.d_columns, planes * 27864 + if word { 648 } else { 0 });
                    assert_eq!(price.outgoing_domain_len, expected.next_power_of_two());
                    println!("prime_plane n={bits} word={word} planes={planes}: {price:?}");
                }
            }
        }
    }

    #[test]
    fn prime_plane_polynomials_share_the_group_cap() {
        let manifest = manifest(3, 2, 2);
        let price =
            price_binary_root_planes(&manifest, false, &prime(&manifest, 5), 64, None).unwrap();
        assert_eq!(price.planes, 5);
        assert_eq!(price.segment_physical_cells.last(), Some(&110080));
        assert_eq!(price.d_columns, 1089);
    }
}
