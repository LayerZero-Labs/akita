#![cfg(feature = "labinius")]

mod common;

use akita_algebra::{
    binary::{BinaryField128, BinaryField192},
    MinusTrinomial, PlusTrinomial, Prime64Offset23703,
};
use akita_error::AkitaError;
use akita_labinius_prover::{commit_binary_clear, prove_binary_clear_bytes};
use akita_labinius_verifier::{
    akita_types::{
        proof::{derive_public_matrix_prefix, AkitaSetupSeed},
        setup_contribution::{TrinomialASetupView, TrinomialResponseLayout},
        RelationPolynomial,
    },
    derive_trinomial_matrix, verify_binary_clear_bytes, AdmittedRootSetup,
};
use akita_params::sis::labinius::{LabiniusRootProfile, LabiniusRootShape};
use common::{data, TestHost};
use jolt_field::Prime128OffsetA7F7;

const PROFILE: LabiniusRootProfile = LabiniusRootProfile::D648P128BoundedW46Delta16;
type Setup = AdmittedRootSetup<Prime128OffsetA7F7, 648, MinusTrinomial>;

fn seed(byte: u8) -> AkitaSetupSeed {
    AkitaSetupSeed::shake256_paged_v1([byte; 32])
}

fn derive(cells: u32, fold: u32, budget: u32, byte: u8) -> Setup {
    Setup::derive(PROFILE, cells, fold, budget, seed(byte)).unwrap()
}

#[test]
fn seed_derivation_is_deterministic_and_changes_matrix_digest_and_identity() {
    let first = derive(4, 1, 128, 0x31);
    let repeated = derive(4, 1, 128, 0x31);
    let changed = derive(4, 1, 128, 0x32);
    assert_eq!(first.setup().matrix(), repeated.setup().matrix());
    assert_eq!(
        first.setup().matrix_view_digest(),
        repeated.setup().matrix_view_digest()
    );
    assert_eq!(
        first.identity_bytes::<BinaryField128>().unwrap(),
        repeated.identity_bytes::<BinaryField128>().unwrap()
    );
    assert_ne!(first.setup().matrix(), changed.setup().matrix());
    assert_ne!(
        first.setup().matrix_view_digest(),
        changed.setup().matrix_view_digest()
    );
    assert_ne!(
        first.identity_bytes::<BinaryField128>().unwrap(),
        changed.identity_bytes::<BinaryField128>().unwrap()
    );
}

#[test]
fn row_major_coefficients_match_setup_addresses_and_the_shared_prefix() {
    let (rows, columns, degree) = (2, 3, 648);
    let seed = seed(0x47);
    let count = rows * columns * degree;
    let matrix =
        derive_trinomial_matrix::<Prime128OffsetA7F7, 648, MinusTrinomial>(&seed, rows, columns)
            .unwrap();
    let prefix = derive_public_matrix_prefix::<Prime128OffsetA7F7>(count, &seed);
    // The longer expansion crosses the derivation's 4096-element page boundary.
    let longer_prefix = derive_public_matrix_prefix::<Prime128OffsetA7F7>(count * 2, &seed);
    let longer_matrix = derive_trinomial_matrix::<Prime128OffsetA7F7, 648, MinusTrinomial>(
        &seed,
        rows * 2,
        columns,
    )
    .unwrap();
    assert_eq!(
        prefix.as_field_slice(),
        &longer_prefix.as_field_slice()[..count]
    );
    assert_eq!(matrix, longer_matrix[..rows * columns]);

    let polynomial = RelationPolynomial::minus_trinomial(degree).unwrap();
    let padded = degree.next_power_of_two();
    let response = TrinomialResponseLayout::new(
        polynomial,
        padded,
        columns,
        1,
        0,
        (columns * padded).next_power_of_two(),
    )
    .unwrap();
    let view = TrinomialASetupView::new(
        polynomial,
        rows,
        columns,
        0,
        count.next_power_of_two(),
        response,
    )
    .unwrap();
    for row in 0..rows {
        for column in 0..columns {
            for coefficient in 0..degree {
                let address = view.setup_address(row, column, coefficient).unwrap();
                assert_eq!(address, (row * columns + column) * degree + coefficient);
                assert_eq!(
                    matrix[row * columns + column].coefficients()[coefficient],
                    prefix.as_field_slice()[address]
                );
                assert_eq!(
                    matrix[row * columns + column].coefficients()[coefficient],
                    longer_prefix.as_field_slice()[address]
                );
            }
        }
    }
}

#[test]
fn identity_binds_each_derivation_input_and_switch_field() {
    let original = derive(4, 1, 128, 0x58);
    let identity = original.identity_bytes::<BinaryField128>().unwrap();
    for changed in [
        derive(5, 1, 128, 0x58),
        derive(4, 2, 128, 0x58),
        derive(4, 1, 127, 0x58),
        derive(4, 1, 128, 0x59),
    ] {
        assert_ne!(
            identity,
            changed.identity_bytes::<BinaryField128>().unwrap()
        );
    }
    assert_ne!(
        identity,
        original.identity_bytes::<BinaryField192>().unwrap()
    );
}

#[test]
fn identity_encoding_matches_the_fixed_width_and_length_framed_layout() {
    let admitted = derive(4, 1, 128, 0x6a);
    let profile = PROFILE.identity_bytes().unwrap();
    let clear = admitted.setup().identity_bytes::<BinaryField128>().unwrap();
    let domain = b"akita/labinius/admitted-root-setup/v1";
    let mut expected = Vec::new();
    expected.extend_from_slice(&u64::try_from(domain.len()).unwrap().to_le_bytes());
    expected.extend_from_slice(domain);
    expected.extend_from_slice(&u64::try_from(profile.len()).unwrap().to_le_bytes());
    expected.extend_from_slice(&profile);
    expected.extend_from_slice(&4u32.to_le_bytes());
    expected.extend_from_slice(&1u32.to_le_bytes());
    expected.extend_from_slice(&128u32.to_le_bytes());
    expected.extend_from_slice(&admitted.shape().rank_a().to_le_bytes());
    expected.push(1); // Shake256PagedV1 wire tag.
    expected.extend_from_slice(&[0x6a; 32]);
    expected.extend_from_slice(&u64::try_from(clear.len()).unwrap().to_le_bytes());
    expected.extend_from_slice(&clear);
    assert_eq!(
        admitted.identity_bytes::<BinaryField128>().unwrap(),
        expected
    );
}

#[test]
fn rejected_shapes_and_type_mismatches_return_invalid_setup_without_panicking() {
    for (cells, fold, budget) in [
        (1, 0, 128),
        (2, 3, 128),
        (usize::BITS, 0, 128),
        (4, u32::MAX, 128),
        (22, 8, 129),
        (22, 9, 127),
        (33, 0, 128),
    ] {
        assert!(matches!(
            LabiniusRootShape::derive(PROFILE, cells, fold, budget),
            Err(AkitaError::InvalidSetup(_))
        ));
        let outcome =
            std::panic::catch_unwind(|| Setup::derive(PROFILE, cells, fold, budget, seed(0x71)));
        assert!(matches!(outcome, Ok(Err(AkitaError::InvalidSetup(_)))));
    }
    let degree = std::panic::catch_unwind(|| {
        AdmittedRootSetup::<Prime128OffsetA7F7, 324, MinusTrinomial>::derive(
            PROFILE,
            2,
            0,
            128,
            seed(0x71),
        )
    });
    assert!(matches!(degree, Ok(Err(AkitaError::InvalidSetup(_)))));
    let prime = std::panic::catch_unwind(|| {
        AdmittedRootSetup::<Prime64Offset23703, 648, MinusTrinomial>::derive(
            PROFILE,
            2,
            0,
            128,
            seed(0x71),
        )
    });
    assert!(matches!(prime, Ok(Err(AkitaError::InvalidSetup(_)))));
    let sign = std::panic::catch_unwind(|| {
        AdmittedRootSetup::<Prime128OffsetA7F7, 648, PlusTrinomial>::derive(
            PROFILE,
            2,
            0,
            128,
            seed(0x71),
        )
    });
    assert!(matches!(sign, Ok(Err(AkitaError::InvalidSetup(_)))));
}

#[test]
fn excessive_matrix_extents_reject_before_seed_expansion() {
    // 8192 * 648 field elements fit the generic count cap, but their P128
    // coefficient storage exceeds this API's materialization byte budget.
    const {
        assert!(
            8192 * 648
                < akita_labinius_verifier::akita_types::proof::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS
        );
    }
    for (rows, columns) in [(usize::MAX, 1), (1, usize::MAX), (1 << 26, 1), (8192, 1)] {
        let outcome = std::panic::catch_unwind(|| {
            derive_trinomial_matrix::<Prime128OffsetA7F7, 648, MinusTrinomial>(
                &seed(0x72),
                rows,
                columns,
            )
        });
        assert!(matches!(outcome, Ok(Err(AkitaError::InvalidSetup(_)))));
    }
}

#[test]
fn empty_matrix_dimensions_return_invalid_setup_without_panicking() {
    for (rows, columns) in [(0, 1), (1, 0), (0, 0)] {
        let outcome = std::panic::catch_unwind(|| {
            derive_trinomial_matrix::<Prime128OffsetA7F7, 648, MinusTrinomial>(
                &seed(0x73),
                rows,
                columns,
            )
        });
        assert!(matches!(outcome, Ok(Err(AkitaError::InvalidSetup(_)))));
    }
}

fn invalid_matrix_degree<const D: usize>() {
    let outcome = std::panic::catch_unwind(|| {
        derive_trinomial_matrix::<Prime128OffsetA7F7, D, MinusTrinomial>(&seed(0x74), 1, 1)
    });
    assert!(matches!(outcome, Ok(Err(AkitaError::InvalidSetup(_)))));
}

#[test]
fn zero_and_odd_degrees_return_invalid_setup_without_panicking() {
    invalid_matrix_degree::<0>();
    invalid_matrix_degree::<1>();
    invalid_matrix_degree::<3>();
}

#[test]
fn excessive_ring_degree_rejects_without_reserving_a_coefficient_stack_array() {
    let outcome = std::panic::catch_unwind(|| {
        derive_trinomial_matrix::<Prime128OffsetA7F7, { 1 << 20 }, MinusTrinomial>(
            &seed(0x76),
            1,
            1,
        )
    });
    assert!(matches!(outcome, Ok(Err(AkitaError::InvalidSetup(_)))));
}

#[test]
fn numerically_admitted_shape_can_exceed_the_matrix_materialization_budget() {
    LabiniusRootShape::derive(PROFILE, 32, 0, 128).unwrap();
    let outcome = std::panic::catch_unwind(|| Setup::derive(PROFILE, 32, 0, 128, seed(0x75)));
    assert!(matches!(outcome, Ok(Err(AkitaError::InvalidSetup(_)))));
}

fn clear_roundtrip<H: TestHost>() {
    let admitted = derive(2, 0, 128, 0x81);
    let setup = admitted.setup();
    let (source, point, claim) = data::<H>(setup.source_len(), setup.num_vars());
    let commitment =
        commit_binary_clear::<H, Prime128OffsetA7F7, 648, MinusTrinomial>(setup, &source).unwrap();
    let mut proof = prove_binary_clear_bytes(setup, &source, &commitment, &point, claim).unwrap();
    verify_binary_clear_bytes(setup, &commitment, &point, claim, &proof).unwrap();
    proof[0] ^= 1;
    assert!(verify_binary_clear_bytes(setup, &commitment, &point, claim, &proof).is_err());
}

#[test]
fn derived_setup_runs_clear_openings_and_rejects_tampering_for_both_hosts() {
    clear_roundtrip::<BinaryField128>();
    clear_roundtrip::<BinaryField192>();
}

#[test]
fn clear_setup_and_accessors_match_the_admitted_shape_and_profile() {
    let admitted = derive(4, 1, 128, 0x91);
    let shape = admitted.shape();
    let setup = admitted.setup();
    assert_eq!(
        shape,
        &LabiniusRootShape::derive(PROFILE, 4, 1, 128).unwrap()
    );
    assert_eq!(setup.n_a(), usize::try_from(shape.rank_a()).unwrap());
    assert_eq!(setup.m(), shape.ring_elements_per_column());
    assert_eq!(setup.columns(), shape.fold_width());
    assert_eq!(setup.source_len(), shape.num_cells());
    assert_eq!(setup.scalar_rows(), shape.scalars_per_column());
    assert_eq!(setup.k(), shape.packing_degree());
    let interval = PROFILE.response_interval();
    assert_eq!(
        (i128::from(setup.lower()), i128::from(setup.upper())),
        interval
    );
    assert_eq!(
        setup.profile(),
        &shape.profile().challenge_profile().unwrap()
    );
    assert_eq!(setup.lambda_fold(), 128);
    assert_eq!(admitted.seed(), &seed(0x91));
    assert_eq!(admitted.log_num_cells(), 4);
    assert_eq!(admitted.log_fold_width(), 1);
    assert_eq!(admitted.lambda_fold(), 128);
}
