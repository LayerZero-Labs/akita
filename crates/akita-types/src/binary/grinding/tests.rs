use super::*;
use akita_params::{GrindingQueryKind, SumcheckProtocol};

/// `a * 2^shift` as a little-endian pair of 128-bit words, for `shift < 128`.
fn shifted(a: u128, shift: u32) -> (u128, u128) {
    if shift == 0 {
        (a, 0)
    } else {
        (a << shift, a >> (128 - shift))
    }
}

/// Least g with `loss * 2^128 <= p^2 * 2^g`, by direct 256-bit comparison.
fn least_bits(loss: u64, p: u128) -> u8 {
    let order = p * p;
    let lhs = (0u128, u128::from(loss));
    (0u8..=63)
        .find(|&g| {
            let (low, high) = shifted(order, u32::from(g));
            (lhs.1, lhs.0) <= (high, low)
        })
        .unwrap()
}

fn suffix(order: akita_params::ChallengeFieldOrder) -> GrindingPlan {
    // Stage 3 of the root and one later fold, as an ordinary suffix supplies.
    let mut runs = (0..18)
        .map(|round| {
            GrindingRun::proof_of_work(
                GrindingSite::SumcheckRound {
                    protocol: SumcheckProtocol::Stage3,
                    level: 0,
                    stage: 0,
                    round,
                },
                2,
                order,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    runs.push(GrindingRun::fold_response(1));
    GrindingPlan::new(runs, order).unwrap()
}

#[test]
fn p64_root_sites_are_priced_against_the_exact_extension_order() {
    let field = BinaryFieldProfile::P64;
    let p = 18_446_744_073_709_551_557u128;
    assert_eq!(field.modulus(), p);
    // Fixture (C=2, one owner, 2^17 domain) and C=256 at the 2^22 and 2^25
    // domains; the 2^25 domain holds the 2^22-carrier witness.
    for (coordinates, domain_len) in [(2, 1 << 17), (256, 1 << 22), (256, 1 << 25)] {
        let plan = binary_root_grinding_plan(
            &suffix(field.challenge_order().unwrap()),
            field,
            &[coordinates],
            domain_len,
            648,
        )
        .unwrap();
        assert_eq!(plan.challenge_order(), field.challenge_order().unwrap());
        let mut priced = 0;
        for run in plan.runs().iter().filter(|run| run.site().level() == 0) {
            match run.kind() {
                GrindingQueryKind::ProofOfWork => {
                    assert_eq!(
                        run.grind_bits(),
                        least_bits(run.loss_factor(), p),
                        "{:?}",
                        run.site()
                    );
                    priced += 1;
                }
                GrindingQueryKind::FoldChallengeWork => {
                    // Guaranteed 2^136 column family: I * 2^128 <= 2^(136+g).
                    assert_eq!(run.challenge_set_bits(), 136);
                    assert_eq!(run.loss_factor(), coordinates as u64);
                    let g = (0u8..=63)
                        .find(|&g| run.loss_factor() <= 1u64 << (8 + g))
                        .unwrap();
                    assert_eq!(run.grind_bits(), g);
                }
                _ => {}
            }
        }
        // Alpha, tau0, tau1, Stage 1, Stage 2 and the suffix's root Stage 3.
        assert!(priced > 3 + 18 + domain_len.trailing_zeros() as usize);
    }
}

#[test]
fn root_plan_rejects_a_suffix_priced_under_another_order() {
    for (root, other) in [
        (BinaryFieldProfile::P64, BinaryFieldProfile::P128),
        (BinaryFieldProfile::P128, BinaryFieldProfile::P64),
    ] {
        let ordinary = suffix(other.challenge_order().unwrap());
        assert!(matches!(
            binary_root_grinding_plan(&ordinary, root, &[2], 1 << 17, 648),
            Err(AkitaError::InvalidSetup(_))
        ));
        let ordinary = suffix(root.challenge_order().unwrap());
        assert!(binary_root_grinding_plan(&ordinary, root, &[2], 1 << 17, 648).is_ok());
    }
}
