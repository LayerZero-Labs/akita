//! Differential check of the one-polynomial prime opening against the
//! extension-valued table's direct multilinear evaluation.

#![cfg(feature = "labinius")]

use akita_algebra::poly::multilinear_eval;
use akita_error::AkitaError;
use akita_labinius_pcs::{
    derive_root_setup,
    family::{
        prime::{flatten_prime_table, prime_opening_claim},
        Family128, Family64, FieldFamily,
    },
    shipped::SupportedGeometry,
};
use akita_labinius_verifier::lowered::LoweredRootLayout;
use akita_params::sis::labinius::LabiniusRootProfile;
use akita_types::proof::AkitaSetupSeed;
use jolt_field::{ExtField, Field, One, Ring, Zero};

/// Sum over Boolean vertices directly, independently of the fold evaluator.
fn direct_evaluation<E: Field>(table: &[E], point: &[E]) -> E {
    table
        .iter()
        .enumerate()
        .fold(E::zero(), |sum, (index, &entry)| {
            let weight =
                point
                    .iter()
                    .enumerate()
                    .fold(E::one(), |weight, (variable, &coordinate)| {
                        weight
                            * if (index >> variable) & 1 == 0 {
                                E::one() - coordinate
                            } else {
                                coordinate
                            }
                    });
            sum + weight * entry
        })
}

fn conversion_matches_direct_evaluation<P: FieldFamily>() {
    let geometry = SupportedGeometry {
        profile: LabiniusRootProfile::D648Q25BoundedW46,
        log_num_cells: 4,
        log_fold_width: 1,
        lambda_fold: 128,
    };
    let root = derive_root_setup(geometry, &AkitaSetupSeed::shake256_paged_v1([9; 32])).unwrap();
    let layout =
        LoweredRootLayout::new::<P::Base, P::Challenge, 648, _>(root.setup(), root.shape())
            .unwrap();
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut random = || {
        P::Challenge::from_base_fn(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let low = u128::from(state);
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            P::Base::from_u128(low | (u128::from(state) << 64))
        })
    };
    let points = [
        (0..layout.prime_log_len())
            .map(|_| random())
            .collect::<Vec<_>>(),
        vec![P::Challenge::zero(); layout.prime_log_len()],
        vec![P::Challenge::one(); layout.prime_log_len()],
    ];
    let tables = [
        vec![P::Challenge::zero(); layout.prime_len()],
        vec![P::Challenge::one(); layout.prime_len()],
        (0..layout.prime_len())
            .map(|index| {
                P::Challenge::from_base_fn(|coordinate| {
                    if coordinate == index % P::Challenge::DEGREE {
                        P::Base::one()
                    } else {
                        P::Base::zero()
                    }
                })
            })
            .collect(),
        (0..layout.prime_len()).map(|_| random()).collect(),
    ];
    for table in &tables {
        let flattened = flatten_prime_table::<P>(&layout, table).unwrap();
        let lifted = flattened
            .into_iter()
            .map(P::Challenge::lift_base)
            .collect::<Vec<_>>();
        for point in &points {
            let value = direct_evaluation(table, point);
            let (nested_point, nested_value) =
                prime_opening_claim::<P>(&layout, point, value).unwrap();
            assert_eq!(
                multilinear_eval(&lifted, &nested_point).unwrap(),
                nested_value,
                "base-coordinate conversion must preserve the direct extension claim"
            );
        }
    }
    assert!(matches!(
        prime_opening_claim::<P>(&layout, &[], P::Challenge::zero()),
        Err(AkitaError::InvalidProof)
    ));
    assert!(matches!(
        flatten_prime_table::<P>(&layout, &[]),
        Err(AkitaError::InvalidInput(_))
    ));
}

#[test]
fn both_families_convert_prime_claims_against_direct_evaluation() {
    conversion_matches_direct_evaluation::<Family64>();
    conversion_matches_direct_evaluation::<Family128>();
}
