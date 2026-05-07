use std::time::Duration;

use rand::RngCore;

pub(crate) fn rand_u128<R: RngCore>(rng: &mut R) -> u128 {
    let lo = rng.next_u64() as u128;
    let hi = rng.next_u64() as u128;
    lo | (hi << 64)
}

pub(crate) fn duration_per_logical_op(elapsed: Duration, ops_per_iter: u64) -> Duration {
    Duration::from_secs_f64(elapsed.as_secs_f64() / ops_per_iter as f64)
}
