//! Print `runtime_schedule` proof bytes for CI profile-bench cases.

use akita_config::{proof_optimized::fp128, CommitmentConfig};
use akita_types::AkitaScheduleLookupKey;

fn main() {
    let cases: [(&str, fn(AkitaScheduleLookupKey) -> usize); 3] = [
        ("dense_fp128_d64:24:1", |key| {
            fp128::D64Full::runtime_schedule(key)
                .expect("schedule")
                .total_bytes
        }),
        ("onehot_fp128_d64:32:1", |key| {
            fp128::D64OneHot::runtime_schedule(key)
                .expect("schedule")
                .total_bytes
        }),
        ("onehot_fp128_d64:30:4", |key| {
            fp128::D64OneHot::runtime_schedule(key)
                .expect("schedule")
                .total_bytes
        }),
    ];
    let keys = [
        AkitaScheduleLookupKey::singleton(24),
        AkitaScheduleLookupKey::singleton(32),
        AkitaScheduleLookupKey::new(30, 4, 4, 1),
    ];
    for ((label, schedule), key) in cases.iter().zip(keys) {
        println!("{label}: {}", schedule(key));
    }
}
