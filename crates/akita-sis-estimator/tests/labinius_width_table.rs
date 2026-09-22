use akita_sis_estimator::{
    width_table::{validate_infinity_width_rows, InfinityWidthRow},
    AkitaModulusProfileId, SisSecurityPolicy,
};
use std::collections::BTreeSet;

const TABLE: &str = include_str!("../data/labinius_infinity_width.csv");

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
