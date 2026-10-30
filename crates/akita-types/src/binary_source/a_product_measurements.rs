//! Release measurements for the old public compiler and the factored A arithmetic.
//!
//! These timings cover arithmetic only. They exclude setup preprocessing and
//! transcript replay; the end-to-end fixture measures complete verification.

use super::*;
use crate::binary_source::{
    binary_direct_a_weights, binary_quotient_weights, BinaryCommitmentProfiles, BinaryIndexMap,
    BinarySourceDescriptor, CANONICAL_IMAGE_WINDOW,
};
use akita_challenges::{Challenges, SparseChallenge};
use jolt_field::{Prime128Offset275 as F, Ring, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;
use std::time::{Duration, Instant};

#[test]
fn stage3_release_a_arithmetic_measurements() {
    let mut rng = StdRng::seed_from_u64(0x7823_23a0);
    // Stop when a direct evaluation reaches two seconds. The last completed
    // power-of-two geometry is the largest reported shape for this machine.
    for rows in [
        1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    ] {
        let rank = 3usize;
        let descriptor = BinarySourceDescriptor::new(
            BinaryIndexMap::new(128, 8, 8 * rows, 2).unwrap(),
            vec![[1; 32]],
            BinaryCommitmentProfiles {
                a: [2; 32],
                b: [3; 32],
            },
        )
        .unwrap();
        let plan = BinaryRootWitnessPlan::new(&descriptor, rank, 64).unwrap();
        let matrix = (0..rank)
            .map(|_| {
                (0..rows)
                    .map(|_| {
                        core::array::from_fn(|_| {
                            i128::from(
                                rng.gen_range(CANONICAL_IMAGE_WINDOW.0..=CANONICAL_IMAGE_WINDOW.1),
                            )
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let alpha = F::from_u64(17);
        let batches = [F::from_u64(19), F::from_u64(23), F::from_u64(29)];
        let rho = binary_a_rho(alpha, &batches);
        let point = (0..plan.repacking().domain_len().trailing_zeros())
            .map(|_| F::from_u64(rng.gen_range(2..=1000)))
            .collect::<Vec<_>>();
        let challenges = Challenges::from_sparse(
            vec![
                SparseChallenge {
                    positions: Default::default(),
                    coeffs: Default::default(),
                };
                2
            ],
            2,
            1,
        )
        .unwrap();
        let padding = (0..plan.repacking().domain_len())
            .filter(|&address| plan.repacking().is_padding_address(address).unwrap())
            .map(|_| F::zero())
            .collect::<Vec<_>>();
        // Use the existing pre-offload compiler, including its fixed T/KA and
        // zero-padding work, rather than a second direct A implementation.
        let start = Instant::now();
        let direct =
            binary_direct_a_weights(&plan, &matrix, &challenges, alpha, &batches, &padding)
                .unwrap()
                .evaluate(&point)
                .unwrap();
        black_box(direct);
        let direct_elapsed = start.elapsed();

        let prefix_len = rows * 4 * STRIDE;
        let terminal_point = (0..prefix_len.trailing_zeros())
            .map(|_| F::from_u64(rng.gen_range(2..=1000)))
            .collect::<Vec<_>>();
        let start = Instant::now();
        for _ in 0..8 {
            let product = BinaryAProduct::new(&plan, &rho, &point, prefix_len).unwrap();
            black_box(product.evaluate(&terminal_point).unwrap());
        }
        let factored_elapsed = start.elapsed() / 8;
        let start = Instant::now();
        let non_a = binary_quotient_weights(&plan, None, &challenges, &rho, &padding)
            .unwrap()
            .evaluate(&point)
            .unwrap();
        black_box(non_a);
        let non_a_elapsed = start.elapsed();
        // Both totals include the quotient row's T/KA and zero-padding compiler
        // work. The new total adds factored A preparation and terminal weight
        // evaluation, but excludes Stage 3 transcript/sumcheck replay overhead.
        let factored_total = non_a_elapsed + factored_elapsed;
        eprintln!(
            "stage3 A arithmetic rows={rows} rank={rank} prefix={prefix_len} stage2_domain={} direct_quotient_compiler_and_mle_us={} factored_a_prepare_and_terminal_us={} non_a_quotient_compiler_and_mle_us={} non_a_plus_factored_arithmetic_total_us={}",
            plan.repacking().domain_len(), direct_elapsed.as_micros(), factored_elapsed.as_micros(),
            non_a_elapsed.as_micros(), factored_total.as_micros(),
        );
        if direct_elapsed >= Duration::from_secs(2) {
            break;
        }
    }
}
