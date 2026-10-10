#![cfg(feature = "labinius-sis")]

use akita_params::sis::labinius::{
    LabiniusCommitmentModulus, LabiniusRingDegree, LabiniusWidthCell, LabiniusWidthCutoff,
    LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE, LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE_DIGEST,
};
use akita_sis_estimator::{
    labinius_width_table::certified_rows,
    width_table::{
        generate_infinity_width_rows, validate_infinity_width_rows, InfinityWidthOrigin,
        InfinityWidthTableConfig,
    },
    AkitaModulusProfileId, SisSecurityPolicy,
};
use sha3::{Digest, Sha3_256};

const CSV: &str = include_str!("../data/labinius_commitment_prime_infinity_width.csv");
const COEFF_LINF_BOUND: u64 = 1_506_960;
/// Smallest rank with a certified positive width.
const MIN_RANK: u32 = 2;

/// The checked-in rows are what the search produces under the default policy
/// and cap, and the runtime cells and digest in `akita-params` are those rows.
#[test]
fn rows_regenerate_with_the_unchanged_policy_and_cap() {
    for line in CSV.lines().skip(1) {
        let fields = line.split(',').take(4).collect::<Vec<_>>();
        assert_eq!(
            fields,
            [
                "akita-infinity-width-v4",
                "phi243-bounded-w46-delta12",
                "162",
                "4"
            ]
        );
    }
    let rows = certified_rows().unwrap();
    let profile = AkitaModulusProfileId::Q25Plus14561;
    let config = InfinityWidthTableConfig {
        profiles: vec![profile],
        ring_dims: vec![648],
        coeff_linf_bounds: vec![COEFF_LINF_BOUND],
        max_rank: 4,
        explicit_origins: Some(vec![InfinityWidthOrigin {
            modulus_profile: profile,
            d: 648,
            coeff_linf_bound: COEFF_LINF_BOUND,
        }]),
        ..InfinityWidthTableConfig::default()
    };
    let candidates = generate_infinity_width_rows(&config).unwrap();
    assert_eq!(candidates.len(), 4);
    assert!(candidates
        .iter()
        .filter(|row| row.rank < MIN_RANK)
        .all(|row| row.max_width == 0));
    let regenerated = candidates
        .into_iter()
        .filter(|row| row.max_width > 0)
        .collect::<Vec<_>>();
    assert_eq!(rows, regenerated);
    validate_infinity_width_rows(rows).unwrap();
    assert_eq!(rows.len(), (MIN_RANK..=4).count());
    for row in rows {
        assert_eq!(row.modulus_profile, profile);
        assert_eq!(row.policy, SisSecurityPolicy::Quantum128BitADPS16);
        assert_eq!(row.d, 648);
        assert_eq!(row.coeff_linf_bound, COEFF_LINF_BOUND);
        assert_eq!(row.max_width, row.search_cap);
        assert_eq!(row.max_width, 6_400_000_000_000);
        assert!(row.hit_cap);
        assert!(row.next_costs.is_none());
    }
    let cells = rows
        .iter()
        .map(|row| LabiniusWidthCell {
            commitment_modulus: LabiniusCommitmentModulus::Q25Plus14561,
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
    assert_eq!(LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE, cells);
    let digest: [u8; 32] = Sha3_256::digest(CSV.as_bytes()).into();
    assert_eq!(LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE_DIGEST, digest);
}
