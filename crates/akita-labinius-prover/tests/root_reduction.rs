#![cfg(feature = "labinius")]

#[path = "root_reduction_support.rs"]
mod support;

use akita_algebra::{binary::BinaryField192, MinusTrinomial};
use akita_error::AkitaError;
use akita_labinius_prover::{
    prove_binary_clear_bytes, prove_root_reduction_bytes, TransparentRootProverOracle,
};
use akita_labinius_verifier::{
    channel::ClearChannel,
    lowered::LoweredRootLayout,
    root::{
        check_transparent_evaluations, root_reduction_wire_size, verify_root_reduction_bytes,
        RootEvaluationClaims, RootVerifierOracle, TransparentRootVerifierOracle,
    },
    verify_binary_clear_bytes,
};
use akita_params::sis::labinius::{LabiniusDigitBase, LabiniusRootShape};
use jolt_field::{CanonicalBytes, CanonicalEncoding, One};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use support::{admitted, common::TestHost, Case, BASES, F, H, PROFILE};

fn complete<T: TestHost>() {
    for fold in [1, 0] {
        for base in BASES {
            let case = Case::<T>::new(base, fold);
            let (proof, claims, digits) = case.prove();
            assert_eq!(case.verify(&proof).unwrap(), claims);
            assert_eq!(digits.len(), case.layout.witness_len());
            assert_eq!(
                proof.len() - digits.len(),
                root_reduction_wire_size::<T>(case.admitted.shape(), base).unwrap()
            );
            let regions = case.regions();
            let (_, start, len) = regions.last().unwrap();
            assert_eq!(start + len, proof.len());
        }
    }
}

#[test]
fn completeness_every_base_and_both_shapes_for_both_hosts() {
    complete::<H>();
    complete::<BinaryField192>();
}

#[test]
fn first_real_profile_wire_sizes_are_pinned_without_materializing_its_matrix() {
    let shape = LabiniusRootShape::derive(PROFILE, 22, 8, 128).unwrap();
    for (base, host128, host192) in [
        (LabiniusDigitBase::Bits1, 22_208, 21_696),
        (LabiniusDigitBase::Bits2, 22_960, 22_448),
        (LabiniusDigitBase::Bits4, 27_488, 26_976),
    ] {
        assert_eq!(
            root_reduction_wire_size::<H>(&shape, base).unwrap(),
            host128
        );
        assert_eq!(
            root_reduction_wire_size::<BinaryField192>(&shape, base).unwrap(),
            host192
        );
    }
}

#[test]
fn reduction_and_clear_opening_agree_on_honest_and_false_statements() {
    let case = Case::<H>::new(LabiniusDigitBase::Bits2, 1);
    let setup = case.admitted.setup();
    let clear = prove_binary_clear_bytes(
        setup,
        &case.source,
        &case.commitment,
        &case.point,
        case.value,
    )
    .unwrap();
    verify_binary_clear_bytes(setup, &case.commitment, &case.point, case.value, &clear).unwrap();
    let (root, _, _) = case.prove();
    case.verify(&root).unwrap();
    let wrong_value = case.value + H::ONE;
    assert!(
        verify_binary_clear_bytes(setup, &case.commitment, &case.point, wrong_value, &clear)
            .is_err()
    );
    let mut oracle = TransparentRootVerifierOracle::new(&case.image);
    assert!(verify_root_reduction_bytes(
        &case.admitted,
        case.base,
        &case.point,
        wrong_value,
        &mut oracle,
        &root
    )
    .is_err());
    assert!(prove_binary_clear_bytes(
        setup,
        &case.source,
        &case.commitment,
        &case.point,
        wrong_value
    )
    .is_err());
    let mut oracle = TransparentRootProverOracle::new(&case.image);
    assert!(prove_root_reduction_bytes(
        &case.admitted,
        case.base,
        &case.source,
        &case.commitment,
        &case.point,
        wrong_value,
        &mut oracle
    )
    .is_err());
    let mut other_source = case.source.clone();
    other_source[0] ^= 1;
    let other_commitment = akita_labinius_prover::commit_binary_clear::<H, F, 648, MinusTrinomial>(
        setup,
        &other_source,
    )
    .unwrap();
    assert!(
        verify_binary_clear_bytes(setup, &other_commitment, &case.point, case.value, &clear)
            .is_err()
    );
    let other_image =
        akita_labinius_prover::lowered::flatten_image(&case.layout, &other_commitment).unwrap();
    let mut oracle = TransparentRootVerifierOracle::new(&other_image);
    assert!(verify_root_reduction_bytes(
        &case.admitted,
        case.base,
        &case.point,
        case.value,
        &mut oracle,
        &root
    )
    .is_err());
}

#[test]
fn every_distinct_wire_message_region_rejects_a_flipped_bit() {
    let case = Case::<H>::new(LabiniusDigitBase::Bits1, 1);
    let (proof, _, _) = case.prove();
    for (name, start, len) in case.regions() {
        assert!(len > 0);
        let mut changed = proof.clone();
        changed[start] ^= 1;
        assert!(case.verify(&changed).is_err(), "accepted tampered {name}");
    }
    // Y is public transcript input, so its coefficient byte is changed in the statement.
    let mut image = case.image.clone();
    let mut coefficient = [0u8; 16];
    image[0].to_bytes_le(&mut coefficient);
    coefficient[0] ^= 1;
    image[0] = F::from_bytes_le_checked(&coefficient).unwrap();
    let mut oracle = TransparentRootVerifierOracle::new(&image);
    assert!(verify_root_reduction_bytes(
        &case.admitted,
        case.base,
        &case.point,
        case.value,
        &mut oracle,
        &proof
    )
    .is_err());
}

#[test]
fn noncanonical_field_and_out_of_range_integer_encodings_reject() {
    let case = Case::<H>::new(LabiniusDigitBase::Bits1, 1);
    let (proof, _, _) = case.prove();
    let regions = case.regions();
    for name in ["QA", "y_Y", "w_eval", "y_eval"] {
        let (_, start, _) = regions
            .iter()
            .find(|(region, _, _)| *region == name)
            .unwrap();
        let mut changed = proof.clone();
        changed[*start..*start + 16]
            .copy_from_slice(&PROFILE.coefficient_prime().modulus().to_le_bytes());
        assert!(
            case.verify(&changed).is_err(),
            "accepted noncanonical {name}"
        );
    }
    for (name, bits) in [
        ("Q", case.layout.encoding().quotient().bits()),
        ("K", case.layout.encoding().carry().bits()),
    ] {
        assert_ne!(bits % 8, 0, "fixture needs an unused integer wire bit");
        let width = (bits as usize).div_ceil(8);
        let (_, start, _) = regions
            .iter()
            .find(|(region, _, _)| *region == name)
            .unwrap();
        let mut changed = proof.clone();
        changed[*start + width - 1] |= 1 << (bits % 8);
        assert!(
            case.verify(&changed).is_err(),
            "accepted out-of-range {name}"
        );
    }
}

#[test]
fn setup_base_point_value_and_image_owner_are_statement_bound() {
    let case = Case::<H>::new(LabiniusDigitBase::Bits2, 1);
    let (proof, _, _) = case.prove();
    let changed_setup = admitted(1, 0x32);
    let mut oracle = TransparentRootVerifierOracle::new(&case.image);
    assert!(verify_root_reduction_bytes(
        &changed_setup,
        case.base,
        &case.point,
        case.value,
        &mut oracle,
        &proof
    )
    .is_err());
    for base in [LabiniusDigitBase::Bits1, LabiniusDigitBase::Bits4] {
        let mut oracle = TransparentRootVerifierOracle::new(&case.image);
        assert!(verify_root_reduction_bytes(
            &case.admitted,
            base,
            &case.point,
            case.value,
            &mut oracle,
            &proof
        )
        .is_err());
    }
    let mut changed_point = case.point.clone();
    changed_point[0] += H::ONE;
    let mut oracle = TransparentRootVerifierOracle::new(&case.image);
    assert!(verify_root_reduction_bytes(
        &case.admitted,
        case.base,
        &changed_point,
        case.value,
        &mut oracle,
        &proof
    )
    .is_err());
    let mut oracle = TransparentRootVerifierOracle::new(&case.image);
    assert!(verify_root_reduction_bytes(
        &case.admitted,
        case.base,
        &case.point,
        case.value + H::ONE,
        &mut oracle,
        &proof
    )
    .is_err());
    let mut changed_image = case.image.clone();
    changed_image[0] += F::one();
    let mut oracle = TransparentRootVerifierOracle::new(&changed_image);
    assert!(verify_root_reduction_bytes(
        &case.admitted,
        case.base,
        &case.point,
        case.value,
        &mut oracle,
        &proof
    )
    .is_err());
}

struct WrongTableOracle<'a> {
    honest: TransparentRootVerifierOracle<'a, F>,
    image: &'a [F],
    discharge_reached: bool,
    change_image: bool,
}

impl RootVerifierOracle<F> for WrongTableOracle<'_> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.honest.bind_image(layout, channel)
    }
    fn bind_response<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.honest.bind_response(layout, channel)
    }
    fn discharge<S: ClearChannel>(
        &mut self,
        claims: &RootEvaluationClaims<F>,
        _channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.discharge_reached = true;
        let mut image = self.image.to_vec();
        let mut response = self.honest.response().to_vec();
        if self.change_image {
            image[0] += F::one();
        } else {
            response[0] ^= 1;
        }
        check_transparent_evaluations(&image, &response, claims)
    }
}

#[test]
fn oracle_answers_from_either_different_table_reject_at_discharge() {
    let case = Case::<H>::new(LabiniusDigitBase::Bits4, 1);
    let (proof, _, _) = case.prove();
    for change_image in [false, true] {
        let mut oracle = WrongTableOracle {
            honest: TransparentRootVerifierOracle::new(&case.image),
            image: &case.image,
            discharge_reached: false,
            change_image,
        };
        assert!(verify_root_reduction_bytes(
            &case.admitted,
            case.base,
            &case.point,
            case.value,
            &mut oracle,
            &proof
        )
        .is_err());
        assert!(
            oracle.discharge_reached,
            "rejected before the oracle answered a different table"
        );
    }
}

#[test]
fn malformed_proofs_return_errors_without_panicking() {
    let case = Case::<H>::new(LabiniusDigitBase::Bits2, 1);
    let (proof, _, _) = case.prove();
    let mut prefixes = (0..24).map(|i| i * proof.len() / 24).collect::<Vec<_>>();
    prefixes.extend(
        case.regions()
            .into_iter()
            .flat_map(|(_, start, len)| [start, start + len - 1]),
    );
    prefixes.push(proof.len() - 1);
    prefixes.sort_unstable();
    prefixes.dedup();
    for prefix in prefixes {
        let result = std::panic::catch_unwind(|| case.verify(&proof[..prefix]));
        assert!(
            matches!(result, Ok(Err(_))),
            "prefix {prefix} accepted or panicked"
        );
    }
    let mut trailing = proof.clone();
    trailing.push(0);
    assert!(case.verify(&trailing).is_err());
    let mut rng = StdRng::seed_from_u64(0xbad_147);
    for _ in 0..16 {
        let mut random = vec![0u8; proof.len()];
        rng.fill_bytes(&mut random);
        assert!(matches!(
            std::panic::catch_unwind(|| case.verify(&random)),
            Ok(Err(_))
        ));
    }
}
