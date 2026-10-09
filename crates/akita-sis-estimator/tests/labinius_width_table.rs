#![cfg(feature = "labinius-sis")]

use akita_params::sis::labinius::{
    LabiniusCoefficientPrime, LabiniusCommitmentModulus, LabiniusRingDegree,
    LabiniusSmallModulusWidthCell, LabiniusWidthCell, LabiniusWidthCutoff,
    LABINIUS_SMALL_MODULUS_WIDTH_TABLE, LABINIUS_SMALL_MODULUS_WIDTH_TABLE_DIGEST,
    LABINIUS_WIDTH_TABLE, LABINIUS_WIDTH_TABLE_DIGEST,
};
use akita_sis_estimator::{
    labinius_width_table::{certified_rows, certified_small_modulus_rows},
    width_table::{
        generate_infinity_width_rows, validate_infinity_width_rows, InfinityWidthOrigin,
        InfinityWidthRow, InfinityWidthTableConfig,
    },
    AkitaModulusProfileId, SisSecurityPolicy,
};
use sha3::{Digest, Sha3_256};
use std::collections::BTreeSet;

const TABLE: &str = include_str!("../data/labinius_infinity_width.csv");

#[test]
fn runtime_cells_and_digest_match_the_certified_csv_in_both_directions() {
    let rows = certified_rows().expect("certified CSV");
    assert_eq!(rows.len(), 69);
    assert_eq!(LABINIUS_WIDTH_TABLE.len(), rows.len());
    let csv_cells = rows
        .iter()
        .map(|row| LabiniusWidthCell {
            coefficient_prime: match row.modulus_profile {
                AkitaModulusProfileId::Q64Offset23703 => LabiniusCoefficientPrime::P64Offset23703,
                AkitaModulusProfileId::Q128OffsetA7F7 => LabiniusCoefficientPrime::P128OffsetA7F7,
                other => panic!("unexpected modulus profile: {other:?}"),
            },
            ring_degree: match row.d {
                162 => LabiniusRingDegree::D162,
                324 => LabiniusRingDegree::D324,
                648 => LabiniusRingDegree::D648,
                486 => LabiniusRingDegree::D486,
                972 => LabiniusRingDegree::D972,
                1_944 => LabiniusRingDegree::D1944,
                other => panic!("unexpected ring degree: {other}"),
            },
            rank: row.rank,
            coeff_linf_bound: row.coeff_linf_bound,
            max_width: row.max_width,
        })
        .collect::<BTreeSet<_>>();
    let runtime_cells = LABINIUS_WIDTH_TABLE
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    assert_eq!(csv_cells.len(), rows.len(), "CSV has duplicate cells");
    assert_eq!(
        runtime_cells.len(),
        rows.len(),
        "runtime has duplicate cells"
    );
    assert_eq!(csv_cells, runtime_cells, "runtime/CSV cell drift");
    let digest: [u8; 32] = Sha3_256::digest(TABLE.as_bytes()).into();
    assert_eq!(
        LABINIUS_WIDTH_TABLE_DIGEST, digest,
        "runtime/CSV digest drift"
    );
}

#[test]
fn checked_in_labinius_cells_have_exact_cutoffs_and_rejected_successors() {
    let mut lines = TABLE.lines();
    let header = lines.next().expect("table header");
    assert_eq!(
        header,
        format!(
            "estimator_id,source_profile,scalar_degree,packing_degree,{}",
            InfinityWidthRow::csv_header()
        )
    );

    let mut rows = Vec::new();
    let mut covered = BTreeSet::new();
    for line in lines {
        let mut fields = line.splitn(5, ',');
        assert_eq!(fields.next(), Some("akita-infinity-width-v4"));
        let source_profile = fields.next().expect("source profile");
        let scalar_degree: u32 = fields.next().expect("scalar degree").parse().unwrap();
        let packing_degree: u32 = fields.next().expect("packing degree").parse().unwrap();
        let row = InfinityWidthRow::from_csv_record(fields.next().expect("estimator row")).unwrap();

        assert_eq!(row.policy, SisSecurityPolicy::Quantum128BitADPS16);
        assert!(row.max_width > 0);
        assert!(!row.hit_cap);
        assert!(row.max_costs.is_some());
        assert!(row.next_costs.is_some());
        assert_eq!(row.d, scalar_degree * packing_degree);
        match source_profile {
            "phi243-bounded-w46-delta16" => {
                assert_eq!(scalar_degree, 162);
                assert_eq!(row.coeff_linf_bound, 24_116_880);
            }
            "phi243-fixed-w47-delta16" => {
                assert_eq!(scalar_degree, 162);
                assert_eq!(row.coeff_linf_bound, 24_641_160);
            }
            "phi243-bounded-w46-delta32" => {
                assert_eq!(scalar_degree, 162);
                assert_eq!(row.coeff_linf_bound, 1_580_547_964_560);
            }
            "phi243-fixed-w47-delta32" => {
                assert_eq!(scalar_degree, 162);
                assert_eq!(row.coeff_linf_bound, 1_614_907_702_920);
            }
            "phi729-fixed-w25-delta32" => {
                assert_eq!(scalar_degree, 486);
                assert_eq!(row.coeff_linf_bound, 858_993_459_000);
            }
            "phi729-fixed-w25-delta16" => {
                assert_eq!(scalar_degree, 486);
                assert_eq!(row.coeff_linf_bound, 13_107_000);
            }
            _ => panic!("unknown source profile {source_profile}"),
        }
        covered.insert((row.modulus_profile, row.d, source_profile));
        rows.push(row);
    }
    validate_infinity_width_rows(&rows).unwrap();

    for profile in [
        AkitaModulusProfileId::Q64Offset23703,
        AkitaModulusProfileId::Q128OffsetA7F7,
    ] {
        for d in [162, 324, 648] {
            assert!(covered.contains(&(profile, d, "phi243-bounded-w46-delta16")));
            assert!(covered.contains(&(profile, d, "phi243-fixed-w47-delta16")));
            assert!(covered.contains(&(profile, d, "phi243-bounded-w46-delta32")));
            assert!(covered.contains(&(profile, d, "phi243-fixed-w47-delta32")));
        }
        let larger_scalar_degrees: &[u32] = if profile == AkitaModulusProfileId::Q64Offset23703 {
            &[486, 972, 1_944]
        } else {
            &[486, 972]
        };
        for &d in larger_scalar_degrees {
            assert!(covered.contains(&(profile, d, "phi729-fixed-w25-delta32")));
        }
        let delta16_degrees: &[u32] = if profile == AkitaModulusProfileId::Q64Offset23703 {
            &[486, 972]
        } else {
            &[486]
        };
        for &d in delta16_degrees {
            assert!(covered.contains(&(profile, d, "phi729-fixed-w25-delta16")));
        }
    }
    assert!(!covered.contains(&(
        AkitaModulusProfileId::Q128OffsetA7F7,
        1_944,
        "phi729-fixed-w25-delta32"
    )));
}

#[test]
fn small_modulus_rows_regenerate_with_the_unchanged_policy_and_cap() {
    for line in include_str!("../data/labinius_small_modulus_infinity_width.csv")
        .lines()
        .skip(1)
    {
        let fields = line.split(',').take(4).collect::<Vec<_>>();
        assert_eq!(
            fields,
            [
                "akita-infinity-width-v4",
                "phi243-bounded-w46-delta16",
                "162",
                "4"
            ]
        );
    }
    let rows = certified_small_modulus_rows().unwrap();
    let profile = AkitaModulusProfileId::Q28Offset2103;
    let config = InfinityWidthTableConfig {
        profiles: vec![profile],
        ring_dims: vec![648],
        coeff_linf_bounds: vec![24_116_880],
        max_rank: 4,
        explicit_origins: Some(vec![InfinityWidthOrigin {
            modulus_profile: profile,
            d: 648,
            coeff_linf_bound: 24_116_880,
        }]),
        ..InfinityWidthTableConfig::default()
    };
    let candidates = generate_infinity_width_rows(&config).unwrap();
    assert_eq!(candidates.len(), 4);
    assert!(candidates
        .iter()
        .filter(|row| row.rank <= 2)
        .all(|row| row.max_width == 0));
    let regenerated = candidates
        .into_iter()
        .filter(|row| row.max_width > 0)
        .collect::<Vec<_>>();
    assert_eq!(rows, regenerated);
    validate_infinity_width_rows(rows).unwrap();
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert_eq!(row.modulus_profile, profile);
        assert_eq!(row.policy, SisSecurityPolicy::Quantum128BitADPS16);
        assert_eq!(row.d, 648);
        assert_eq!(row.coeff_linf_bound, 24_116_880);
        assert_eq!(row.max_width, row.search_cap);
        assert_eq!(row.max_width, 6_400_000_000_000);
        assert!(row.hit_cap);
        assert!(row.next_costs.is_none());
    }
    let cells = rows
        .iter()
        .map(|row| LabiniusSmallModulusWidthCell {
            commitment_modulus: LabiniusCommitmentModulus::Q28Offset2103,
            ring_degree: LabiniusRingDegree::D648,
            rank: row.rank,
            coeff_linf_bound: row.coeff_linf_bound,
            max_width: row.max_width,
            cutoff: if row.hit_cap {
                LabiniusWidthCutoff::SearchCap
            } else {
                LabiniusWidthCutoff::Exact
            },
        })
        .collect::<Vec<_>>();
    assert_eq!(LABINIUS_SMALL_MODULUS_WIDTH_TABLE, cells);
    let digest: [u8; 32] = Sha3_256::digest(include_bytes!(
        "../data/labinius_small_modulus_infinity_width.csv"
    ))
    .into();
    assert_eq!(LABINIUS_SMALL_MODULUS_WIDTH_TABLE_DIGEST, digest);
}
