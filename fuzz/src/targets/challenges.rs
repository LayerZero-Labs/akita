//! Fold-challenge sampling (`akita-challenges`).
//!
//! Every fold response, its norm cap, and the verifier's knowledge-soundness
//! bound rely on the drawn challenges having exactly their declared shape, and
//! on the selective-L2 route's operator-norm rejection being sound. The target
//! draws through the production `FoldDraw` path with an input-chosen seed and
//! checks, independently of the sampler:
//!
//! - every challenge has exactly `count_pm1` coefficients of magnitude 1 and
//!   `count_pm2` of magnitude 2 at distinct positions below `D`, so its weight,
//!   L1, L-infinity, and squared L2 norms equal the config's declared values
//!   (the planner's response caps multiply by `challenge_l2_sq_max`);
//! - with a rejection policy, every accepted challenge has negacyclic operator
//!   norm `max_k |c(zeta_k)|` at most the policy threshold, recomputed here with
//!   a floating-point DFT (flagged only beyond a rounding allowance far above
//!   f64 error, so the check never fires on a boundary case);
//! - drawing again from the same seed yields the same challenges, and a
//!   different group index yields a different transcript payload;
//! - running out of operator-norm attempts is a liveness failure, never an
//!   expected error.

use crate::input::Reader;
use crate::stats;
use akita_challenges::{
    selective_l2_challenge_config, selective_l2_operator_norm_rejection, FoldChallengeDrawDomain,
    FoldDraw, SparseChallenge, SparseChallengeConfig, PRODUCTION_FOLD_CHALLENGE_RING_DIMS,
};
use akita_error::AkitaError;
use akita_transcript::FOLD_CHALLENGE_SEED_LEN;

/// Returns an input-chosen seed and records what the draw absorbed.
struct SeedDraw {
    seed: [u8; FOLD_CHALLENGE_SEED_LEN],
    payloads: Vec<Vec<u8>>,
}

impl FoldDraw for SeedDraw {
    fn absorb_and_squeeze(
        &mut self,
        payload: &[u8],
    ) -> Result<[u8; FOLD_CHALLENGE_SEED_LEN], AkitaError> {
        self.payloads.push(payload.to_vec());
        Ok(self.seed)
    }
}

/// Allowance on `gamma` for the f64 DFT: its error is below `1e-12` at these
/// sizes, and the certified predicate rejects well inside the threshold.
const OPERATOR_NORM_ALLOWANCE: f64 = 1e-6;

/// `max_k |sum_j c_j zeta_k^j|`, `zeta_k = exp((2k + 1) pi i / D)`.
fn operator_norm(challenge: &SparseChallenge, ring_d: usize) -> f64 {
    (0..ring_d)
        .map(|k| {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (&position, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
                let angle =
                    std::f64::consts::PI * ((2 * k + 1) * position as usize) as f64 / ring_d as f64;
                re += f64::from(coeff) * angle.cos();
                im += f64::from(coeff) * angle.sin();
            }
            re.hypot(im)
        })
        .fold(0.0, f64::max)
}

fn check_shape(challenge: &SparseChallenge, ring_d: usize, cfg: &SparseChallengeConfig) {
    challenge
        .validate_dyn(ring_d)
        .unwrap_or_else(|error| panic!("drawn challenge violates sampler invariants: {error:?}"));
    let ones = challenge
        .coeffs
        .iter()
        .filter(|c| c.unsigned_abs() == 1)
        .count();
    let twos = challenge
        .coeffs
        .iter()
        .filter(|c| c.unsigned_abs() == 2)
        .count();
    assert!(
        ones == cfg.count_pm1 && twos == cfg.count_pm2 && challenge.coeffs.len() == ones + twos,
        "challenge shape ({ones} x ±1, {twos} x ±2, weight {}) differs from config {cfg:?}",
        challenge.coeffs.len()
    );
    let l1: usize = challenge
        .coeffs
        .iter()
        .map(|c| usize::from(c.unsigned_abs()))
        .sum();
    let l2_sq: u128 = challenge
        .coeffs
        .iter()
        .map(|c| u128::from(c.unsigned_abs()).pow(2))
        .sum();
    let linf = challenge
        .coeffs
        .iter()
        .map(|c| u32::from(c.unsigned_abs()))
        .max();
    assert_eq!(challenge.coeffs.len(), cfg.weight(), "weight");
    assert_eq!(l1, cfg.l1_norm(), "L1 norm");
    assert!(
        l2_sq <= cfg.challenge_l2_sq_max(),
        "squared L2 norm {l2_sq} exceeds declared max {}",
        cfg.challenge_l2_sq_max()
    );
    assert!(
        linf.unwrap_or(0) <= cfg.infinity_norm(),
        "L-infinity norm exceeds declared {}",
        cfg.infinity_norm()
    );
}

pub fn run(data: &[u8]) {
    let mut reader = Reader::new(data);
    let selective = reader.bool();
    let (ring_d, cfg, rejection) = if selective {
        let ring_d = [64usize, 128][usize::from(reader.u8() % 2)];
        let cfg = selective_l2_challenge_config(ring_d).expect("selective family");
        let rejection = selective_l2_operator_norm_rejection(ring_d, &cfg);
        assert!(rejection.is_some(), "selective family without its policy");
        (
            ring_d,
            cfg,
            rejection.filter(|_| !reader.u8().is_multiple_of(4)),
        )
    } else {
        let ring_d = PRODUCTION_FOLD_CHALLENGE_RING_DIMS
            [reader.choose(PRODUCTION_FOLD_CHALLENGE_RING_DIMS.len())];
        let cfg = SparseChallengeConfig::production_for_ring_dim(ring_d).expect("ladder entry");
        (ring_d, cfg, None)
    };
    let domain = if rejection.is_none() && reader.u8().is_multiple_of(4) {
        FoldChallengeDrawDomain::SubringCoefficientPacking {
            challenge_subring_dimension: ring_d,
        }
    } else {
        FoldChallengeDrawDomain::EvaluationTrace
    };
    let group_index = usize::from(reader.u8() % 8);
    let num_live_blocks = 1 + usize::from(reader.u8() % 16);
    let num_claims = 1 + usize::from(reader.u8() % 4);
    let seed: [u8; FOLD_CHALLENGE_SEED_LEN] = reader.bytes();

    let draw = |group_index: usize| {
        let mut draw = SeedDraw {
            seed,
            payloads: Vec::new(),
        };
        let result = draw.draw_folding_challenges_with_rejection(
            domain,
            ring_d,
            group_index,
            num_live_blocks,
            num_claims,
            &cfg,
            rejection,
        );
        (result, draw.payloads)
    };
    let (result, payloads) = draw(group_index);
    let challenges = match result {
        Ok(challenges) => challenges,
        Err(error) if crate::liveness::is_liveness_exhaustion(&error) => {
            panic!("liveness: challenge sampling ran out of attempts: {error:?}")
        }
        Err(error) => panic!(
            "valid draw (D={ring_d}, {cfg:?}, rejection {rejection:?}, {domain:?}) failed: {error:?}"
        ),
    };
    assert_eq!(
        challenges.len(),
        num_live_blocks * num_claims,
        "challenge count"
    );
    assert_eq!(challenges.num_claims(), num_claims);
    assert_eq!(challenges.num_live_blocks_per_claim(), num_live_blocks);
    let mut peak = 0.0f64;
    for challenge in challenges.as_slice() {
        check_shape(challenge, ring_d, &cfg);
        if let Some(policy) = rejection {
            let gamma = operator_norm(challenge, ring_d);
            peak = peak.max(gamma / f64::from(policy.threshold));
            assert!(
                gamma <= f64::from(policy.threshold) + OPERATOR_NORM_ALLOWANCE,
                "accepted challenge has operator norm {gamma} above threshold {} (D={ring_d}): {challenge:?}",
                policy.threshold
            );
        }
    }
    if rejection.is_some() {
        stats::count("challenges_op_norm_checked");
        // Closeness to the threshold as coverage for the engine.
        crate::liveness::guide_ratio(peak);
    }
    let (again, payloads_again) = draw(group_index);
    assert!(
        again.as_ref().ok() == Some(&challenges) && payloads_again == payloads,
        "challenge draw is not deterministic"
    );
    let (_, other_payloads) = draw(group_index + 1);
    assert_ne!(
        other_payloads, payloads,
        "group index does not enter the draw payload"
    );
    stats::count("challenges_drawn");
}
