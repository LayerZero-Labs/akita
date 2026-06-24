//! Manual single-threaded XOF rejection timing.
//!
//! ```text
//! RAYON_NUM_THREADS=1 cargo test -p akita-challenges --test xof_rejection_timing -- --ignored --nocapture
//! ```

use akita_challenges::{
    sparse_challenges_from_seed, SparseChallengeConfig, D64_PRODUCTION_EXACT_SHELL_MAG1,
    D64_PRODUCTION_EXACT_SHELL_MAG2, D64_PRODUCTION_OPERATOR_NORM_THRESHOLD,
};
use std::time::Instant;

const D: usize = 64;

#[test]
#[ignore = "manual single-threaded sampling timing"]
fn xof_rejection_single_threaded_wall_time() {
    let cfg = SparseChallengeConfig::ExactShell {
        count_mag1: D64_PRODUCTION_EXACT_SHELL_MAG1,
        count_mag2: D64_PRODUCTION_EXACT_SHELL_MAG2,
        operator_norm_threshold: D64_PRODUCTION_OPERATOR_NORM_THRESHOLD,
    };
    let seed = [0x42u8; 32];
    for &n in &[1 << 10, 1 << 12, 1 << 14, 1 << 16] {
        // Warm-up
        let _ = sparse_challenges_from_seed::<D>(&seed, n, &cfg, true).expect("warmup");
        let start = Instant::now();
        let reps = if n >= 1 << 14 { 3 } else { 20 };
        for _ in 0..reps {
            let challenges =
                sparse_challenges_from_seed::<D>(&seed, n, &cfg, true).expect("sample");
            std::hint::black_box(challenges);
        }
        let elapsed = start.elapsed();
        let per_batch_ms = elapsed.as_secs_f64() * 1000.0 / f64::from(reps);
        let per_challenge_us = per_batch_ms * 1000.0 / n as f64;
        println!(
            "n={n:>6}: {per_batch_ms:>8.3} ms/batch ({per_challenge_us:>6.2} µs/challenge, {reps} reps)"
        );
    }
}
