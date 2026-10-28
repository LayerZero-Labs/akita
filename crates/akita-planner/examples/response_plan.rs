//! Times the Whiteout response plan (`find_schedule` on the response
//! statement, then its one-row catalog) and prints the catalog digest and
//! artifact digest, so a planner change can be checked for an identical plan.
//!
//! ```text
//! cargo run --release -p akita-planner --features catalog-gen[,parallel] \
//!     --example response_plan -- 24 25 26
//! ```
//!
//! The arguments are the variable counts of `e`; the Whiteout benchmarks use
//! 24, 25 and 26 (muldiv, fib 2^16, fib 2^20).

use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Instant;

use akita_config::proof_optimized::fp128::ResponseDigits;
use akita_config::recursive_commitment::RecursiveCommitmentConfig;
use akita_config::{policy_of, CommitmentConfig, TrustedScheduleCatalog};
use akita_params::{
    ChunkedWitnessCfg, CommittedGroupBatchProfile, GroupCommitPhaseParams, PolynomialGroupLayout,
    ScheduleLookupKey,
};

type Cfg = RecursiveCommitmentConfig<ResponseDigits>;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A fixed-key hash of the artifact bytes; collision resistance is not
/// needed to compare two runs of one planner.
fn artifact_hash(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse().expect("num_vars"))
        .collect();
    for num_vars in sizes {
        let chunking = ChunkedWitnessCfg::default_non_chunked();
        let group = PolynomialGroupLayout::singleton(num_vars);
        let key = ScheduleLookupKey::single(group);
        let started = Instant::now();
        let planned = akita_planner::find_schedule(
            &key,
            Cfg::committed_source_contract().expect("source contract"),
            &[],
            &policy_of::<Cfg>(chunking),
            Cfg::ring_challenge_config,
        )
        .expect("response schedule");
        let searched = started.elapsed();
        let profile = CommittedGroupBatchProfile {
            final_group: GroupCommitPhaseParams::try_from_params(
                group,
                &planned.schedule.root.params,
            )
            .expect("root params"),
            precommitteds: Vec::new(),
        };
        let catalog =
            TrustedScheduleCatalog::<Cfg>::try_from_rows([(profile, planned.schedule)], chunking)
                .expect("catalog");
        let artifact = catalog.catalog().to_artifact_bytes().expect("artifact");
        let total = started.elapsed();
        println!(
            "num_vars={num_vars} find_schedule_ms={:.1} plan_ms={:.1} catalog_digest={} artifact_len={} artifact_hash={:016x}",
            searched.as_secs_f64() * 1e3,
            total.as_secs_f64() * 1e3,
            hex(&catalog.catalog().catalog_digest()),
            artifact.len(),
            artifact_hash(&artifact),
        );
    }
}
