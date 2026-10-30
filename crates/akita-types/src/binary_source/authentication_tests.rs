use super::*;
use crate::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap, BinarySourceDescriptor};
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
    BinaryRootWitnessPlan::new(&descriptor, 3, 256).unwrap()
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
    for segment in [1, 3] {
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
        (3, SetupStreamRole::WitnessCommitment),
    ] {
        let label = label(role);
        let payload = binary_public_digit_commitment::<F>(&seed, label, &natural[segment]).unwrap();
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
    let payload = binary_public_digit_commitment::<F>(&[8; 32], setup, &digits).unwrap();
    let image_role = binary_public_digit_commitment::<F>(
        &[8; 32],
        label(SetupStreamRole::ImageCommitment),
        &digits,
    )
    .unwrap();
    assert_ne!(payload, image_role);
    assert!(binary_digit_authentication_weights(
        &plan,
        &[3],
        &[8; 32],
        label(SetupStreamRole::ImageCommitment),
        &payload,
        F::from_u64(7),
        &[F::one(); 2],
    )
    .is_err());
    assert!(binary_digit_authentication_weights(
        &plan,
        &[3],
        &[8; 32],
        setup,
        &payload[..127],
        F::from_u64(7),
        &[F::one(); 2],
    )
    .is_err());
    assert!(binary_digit_authentication_weights(
        &plan,
        &[3],
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
        assert!(binary_public_digit_commitment::<F>(&[8; 32], invalid, &digits).is_err());
    }
    assert!(binary_public_digit_commitment::<F>(&[8; 32], setup, &[]).is_err());
    assert!(binary_public_digit_commitment::<F>(&[8; 32], setup, &[4]).is_err());
    assert!(binary_public_digit_commitment::<F>(&[8; 32], setup, &[-5]).is_err());
    let zeros = binary_public_digit_commitment::<F>(&[8; 32], setup, &[0; 324]).unwrap();
    assert!(zeros.iter().all(F::is_zero));
}
