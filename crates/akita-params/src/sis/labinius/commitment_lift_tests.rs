use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
use num_bigint::BigUint;

use super::super::LabiniusCommitmentModulus;
use super::*;

const Q25: u32 = 33_568_993;
const P64_OFFSET_59: u128 = (1 << 64) - 59;

fn lift(challenge: &BinaryChallengeProfile, modulus: u32) -> LabiniusCommitmentLift<'_> {
    LabiniusCommitmentLift {
        challenge,
        fold_columns: 256,
        matrix_width: 4_096,
        ring_degree: LabiniusRingDegree::D648,
        commitment_modulus: modulus,
        response_bound: 2_184,
    }
}

/// The digit count is the least depth whose digit-by-digit positive reach
/// covers a canonical residue; `B_T` is the negative reach at that depth.
#[test]
fn image_digit_bound_matches_a_digit_by_digit_reach() {
    assert_eq!(LabiniusCommitmentModulus::Q25Plus14561.modulus(), Q25);
    assert_eq!(labinius_image_digit_bound(Q25), (7, 143_165_576));
    for modulus in [2, 3, 15, 17, 257, 65_537, Q25, 268_433_353, u32::MAX] {
        let (digits, bound) = labinius_image_digit_bound(modulus);
        let reach = |digit: u128, depth: usize| (0..depth).map(|i| digit << (4 * i)).sum::<u128>();
        let canonical = u128::from(modulus - 1);
        assert!(digits == 1 || reach(7, digits - 1) < canonical);
        assert!(canonical <= reach(7, digits));
        assert_eq!(bound, reach(8, digits));
    }
}

/// The predicate is compared with the inequality evaluated in big integers:
/// it accepts exactly the proof moduli above the total and fails closed when
/// the total leaves `u128`.
#[test]
fn no_wrap_agrees_with_big_integer_evaluation() {
    let challenge = BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic243, 46)
        .expect("profile");
    let sample = lift(&challenge, Q25);
    let honest = sample.honest_carry_bound().unwrap();
    assert!(sample.check_no_wrap(P64_OFFSET_59, honest).is_ok());
    assert!(sample.check_no_wrap(P64_OFFSET_59, honest - 1).is_err());

    let mut overflowed = 0;
    for modulus in [Q25, 268_433_353] {
        for width in [1usize, 4_096, 1 << 40] {
            for response_bound in [1u128, 2_184, 1 << 60] {
                let lift = LabiniusCommitmentLift {
                    matrix_width: width,
                    response_bound,
                    ..lift(&challenge, modulus)
                };
                let mass = BigUint::from(width) * 648u32 * response_bound;
                let carry_bound = lift.honest_carry_bound().unwrap();
                assert_eq!(BigUint::from(carry_bound), (&mass + 256u32 * 46u32) * 3u32);
                let (_, image_bound) = labinius_image_digit_bound(modulus);
                let total = mass * 3u32 * (modulus - 1)
                    + BigUint::from(256u32 * 92) * image_bound
                    + BigUint::from(modulus) * carry_bound;
                match u128::try_from(&total) {
                    Ok(total) => {
                        assert!(lift.check_no_wrap(total, carry_bound).is_err());
                        assert!(lift.check_no_wrap(total + 1, carry_bound).is_ok());
                    }
                    Err(_) => {
                        assert!(lift.check_no_wrap(u128::MAX, carry_bound).is_err());
                        overflowed += 1;
                    }
                }
            }
        }
    }
    assert!(overflowed > 0);
    let unrepresentable = LabiniusCommitmentLift {
        matrix_width: usize::MAX,
        ..sample
    };
    assert!(unrepresentable.honest_carry_bound().is_err());
}
