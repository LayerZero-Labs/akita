use super::*;
use crate::binary::{BinaryCommitmentProfiles, BinaryIndexMap, BinarySourceDescriptor};
use jolt_field::{One, Prime128Offset275, Ring, Zero};

type F = Prime128Offset275;

fn plan() -> BinaryRootWitnessPlan {
    let descriptor = BinarySourceDescriptor::new(
        BinaryIndexMap::new(128, 8, 8, 2).unwrap(),
        vec![[1; 32]],
        BinaryCommitmentProfiles {
            a: [2; 32],
            b: [3; 32],
        },
    )
    .unwrap();
    BinaryRootWitnessPlan::for_profile(&descriptor, 3, 256, BinaryFieldProfile::P128, 0).unwrap()
}

fn label(role: SetupStreamRole) -> SetupStreamLabel {
    SetupStreamLabel {
        role,
        group: 7,
        level: 0,
        rank: 2,
        row: 0,
        column: 0,
        ring: SetupStreamRing::Negacyclic,
        degree: 64,
        modulus: SetupStreamModulus::Prime128Offset275,
    }
}

#[test]
fn image_and_u_commitments_authenticate_the_same_outgoing_digits() {
    let plan = plan();
    let mut natural = plan
        .repacking()
        .segments()
        .iter()
        .map(|shape| {
            vec![0i8; shape.elements * shape.digits * shape.geometry.polynomial_modulus_dimension()]
        })
        .collect::<Vec<_>>();
    for segment in [1, 2] {
        for (index, digit) in natural[segment].iter_mut().enumerate() {
            *digit = (index % 8) as i8 - 4;
        }
    }
    let refs = natural.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let table = plan
        .repacking()
        .repack(&refs)
        .unwrap()
        .into_iter()
        .map(|digit| F::from_i64(i64::from(digit)))
        .collect::<Vec<_>>();
    let seed = [41; 32];
    for (segment, role) in [
        (1, SetupStreamRole::ImageCommitment),
        (2, SetupStreamRole::WitnessCommitment),
    ] {
        let label = label(role);
        let payload =
            binary_public_digit_commitment::<F>(&seed, label, 0, &natural[segment]).unwrap();
        assert_eq!(payload.len(), 128);
        let weights = binary_digit_authentication_weights(
            &plan,
            &[segment],
            &seed,
            label,
            &payload,
            F::from_u64(17),
            &[F::from_u64(19), F::from_u64(23)],
        )
        .unwrap();
        weights.check_table(&table).unwrap();
        let shape = plan.repacking().segments()[segment];
        for (element, digit, coefficient) in [
            (0, 0, 0),
            (0, 0, 63),
            (0, 0, 64),
            (
                shape.elements - 1,
                shape.digits - 1,
                shape.geometry.polynomial_modulus_dimension() - 1,
            ),
        ] {
            let address = plan
                .repacking()
                .coefficient_address(segment, element, digit, coefficient)
                .unwrap();
            let mut mutated = table.clone();
            mutated[address] += F::one();
            assert!(weights.check_table(&mutated).is_err());
        }
        let mut altered_payload = payload.clone();
        altered_payload[0] += F::one();
        let altered = binary_digit_authentication_weights(
            &plan,
            &[segment],
            &seed,
            label,
            &altered_payload,
            F::from_u64(17),
            &[F::from_u64(19), F::from_u64(23)],
        )
        .unwrap();
        assert!(altered.check_table(&table).is_err());
        let wrong_seed = binary_digit_authentication_weights(
            &plan,
            &[segment],
            &[42; 32],
            label,
            &payload,
            F::from_u64(17),
            &[F::from_u64(19), F::from_u64(23)],
        )
        .unwrap();
        assert!(wrong_seed.check_table(&table).is_err());
    }
}

#[test]
fn setup_roles_and_partial_final_chunk_are_bound() {
    let plan = plan();
    // U has 324 natural digits: its final 64-coefficient chunk has 60 zeros.
    let mut digits = vec![0i8; 324];
    digits[323] = 3;
    let setup = label(SetupStreamRole::WitnessCommitment);
    let payload = binary_public_digit_commitment::<F>(&[8; 32], setup, 0, &digits).unwrap();
    let image_role = binary_public_digit_commitment::<F>(
        &[8; 32],
        label(SetupStreamRole::ImageCommitment),
        0,
        &digits,
    )
    .unwrap();
    assert_ne!(payload, image_role);
    assert!(binary_digit_authentication_weights(
        &plan,
        &[2],
        &[8; 32],
        label(SetupStreamRole::ImageCommitment),
        &payload,
        F::from_u64(7),
        &[F::one(); 2],
    )
    .is_err());
    assert!(binary_digit_authentication_weights(
        &plan,
        &[2],
        &[8; 32],
        setup,
        &payload[..127],
        F::from_u64(7),
        &[F::one(); 2],
    )
    .is_err());
    assert!(binary_digit_authentication_weights(
        &plan,
        &[2],
        &[8; 32],
        setup,
        &payload,
        F::from_u64(7),
        &[F::one()],
    )
    .is_err());
    for invalid in [
        SetupStreamLabel {
            role: SetupStreamRole::Commitment,
            ..setup
        },
        SetupStreamLabel { level: 1, ..setup },
        SetupStreamLabel {
            degree: 128,
            ..setup
        },
        SetupStreamLabel { rank: 3, ..setup },
        SetupStreamLabel { column: 1, ..setup },
        SetupStreamLabel {
            modulus: SetupStreamModulus::Q(14_253_409),
            ..setup
        },
    ] {
        assert!(binary_public_digit_commitment::<F>(&[8; 32], invalid, 0, &digits).is_err());
    }
    assert!(binary_public_digit_commitment::<F>(&[8; 32], setup, 0, &[]).is_err());
    assert!(binary_public_digit_commitment::<F>(&[8; 32], setup, 0, &[4]).is_err());
    assert!(binary_public_digit_commitment::<F>(&[8; 32], setup, 0, &[-5]).is_err());
    let zeros = binary_public_digit_commitment::<F>(&[8; 32], setup, 0, &[0; 324]).unwrap();
    assert!(zeros.iter().all(F::is_zero));
}

#[test]
fn column_aligned_pieces_sum_to_the_digit_commitment() {
    let setup = label(SetupStreamRole::WitnessCommitment);
    let digits: Vec<i8> = (0..324).map(|i| (i % 8) as i8 - 4).collect();
    let whole = binary_public_digit_commitment::<F>(&[8; 32], setup, 0, &digits).unwrap();
    let (head, tail) = digits.split_at(2 * BINARY_DIGIT_COMMITMENT_DEGREE);
    let head = binary_public_digit_commitment::<F>(&[8; 32], setup, 0, head).unwrap();
    let tail = binary_public_digit_commitment::<F>(&[8; 32], setup, 2, tail).unwrap();
    let sum: Vec<F> = head.iter().zip(&tail).map(|(&a, &b)| a + b).collect();
    assert_eq!(sum, whole);
    let misplaced = binary_public_digit_commitment::<F>(&[8; 32], setup, 1, digits.split_at(128).1);
    assert_ne!(
        head.iter()
            .zip(&misplaced.unwrap())
            .map(|(&a, &b)| a + b)
            .collect::<Vec<_>>(),
        whole
    );
    assert!(binary_public_digit_commitment::<F>(&[8; 32], setup, usize::MAX, &digits).is_err());
}

#[test]
fn digit_commitment_equals_setup_ring_products() {
    let setup = label(SetupStreamRole::WitnessCommitment);
    // Three full columns and a partial one, every digit value, both extremes.
    let mut digits: Vec<i8> = (0..220).map(|i| ((i * 37 + 5) % 8) as i8 - 4).collect();
    digits[..64].fill(-4);
    digits[64..128].fill(3);
    let payload = binary_public_digit_commitment::<F>(&[8; 32], setup, 5, &digits).unwrap();
    let mut expected = Vec::new();
    for row in 0..setup.rank {
        let mut image = CyclotomicRing::<F, BINARY_DIGIT_COMMITMENT_DEGREE>::zero();
        for (column, chunk) in digits.chunks(BINARY_DIGIT_COMMITMENT_DEGREE).enumerate() {
            let mut coefficients = [F::zero(); BINARY_DIGIT_COMMITMENT_DEGREE];
            for (output, &digit) in coefficients.iter_mut().zip(chunk) {
                *output = F::from_i64(i64::from(digit));
            }
            setup_entry::<F>(&[8; 32], setup, row, 5 + column)
                .unwrap()
                .mul_accumulate_into(&CyclotomicRing::from_coefficients(coefficients), &mut image);
        }
        expected.extend_from_slice(image.coefficients());
    }
    assert_eq!(payload, expected);
}

#[test]
fn p64_authentication_keeps_both_extension_coordinates_at_every_rank() {
    use jolt_field::{Ext2, ExtField, Prime64Offset59 as Base};
    type E = Ext2<Base>;
    let descriptor = plan().descriptor().clone();
    let plan =
        BinaryRootWitnessPlan::for_profile(&descriptor, 3, 64, BinaryFieldProfile::P64, 0).unwrap();
    let mut natural = plan
        .repacking()
        .segments()
        .iter()
        .map(|s| vec![0i8; s.elements * s.digits * s.geometry.polynomial_modulus_dimension()])
        .collect::<Vec<_>>();
    for segment in [1, 2] {
        for (i, digit) in natural[segment].iter_mut().enumerate() {
            *digit = (i % 8) as i8 - 4;
        }
    }
    let table = plan
        .repacking()
        .repack(&natural.iter().map(Vec::as_slice).collect::<Vec<_>>())
        .unwrap()
        .into_iter()
        .map(|d| E::from_i64(i64::from(d)))
        .collect::<Vec<_>>();
    let extension = |i: u64| E::from_base_fn(|c| Base::from_u64(i + 19 * c as u64));
    for (segment, role) in [
        (1, SetupStreamRole::ImageCommitment),
        (2, SetupStreamRole::WitnessCommitment),
    ] {
        for rank in 1..=5 {
            let label = SetupStreamLabel {
                modulus: SetupStreamModulus::Prime64Offset59,
                rank,
                ..label(role)
            };
            let payload =
                binary_public_digit_commitment::<Base>(&[41; 32], label, 0, &natural[segment])
                    .unwrap()
                    .into_iter()
                    .map(E::lift_base)
                    .collect::<Vec<_>>();
            let batches = (0..rank)
                .map(|i| extension(31 + u64::from(i)))
                .collect::<Vec<_>>();
            let weights = binary_digit_authentication_weights(
                &plan,
                &[segment],
                &[41; 32],
                label,
                &payload,
                extension(17),
                &batches,
            )
            .unwrap();
            weights.check_table(&table).unwrap();
            let point = (0..plan.repacking().domain_len().trailing_zeros())
                .map(|i| extension(101 + u64::from(i)))
                .collect::<Vec<_>>();
            let dense = weights.materialize().unwrap();
            let direct = dense
                .iter()
                .zip(akita_algebra::eq_poly::EqPolynomial::evals(&point).unwrap())
                .fold(E::zero(), |sum, (&w, e)| sum + w * e);
            assert_eq!(weights.evaluate(&point).unwrap(), direct);
            for coordinate in 0..2 {
                let mut altered = payload.clone();
                altered[0] += E::from_base_fn(|c| {
                    if c == coordinate {
                        Base::one()
                    } else {
                        Base::zero()
                    }
                });
                let altered = binary_digit_authentication_weights(
                    &plan,
                    &[segment],
                    &[41; 32],
                    label,
                    &altered,
                    extension(17),
                    &batches,
                )
                .unwrap();
                assert!(matches!(
                    altered.check_table(&table),
                    Err(AkitaError::InvalidProof)
                ));
            }
            assert!(binary_digit_authentication_weights(
                &plan,
                &[segment],
                &[41; 32],
                label,
                &payload,
                extension(17),
                &batches[..batches.len() - 1]
            )
            .is_err());
        }
    }
}
