//! On-demand diagnostic: proof-size savings from flipping the terminal proof
//! mode to `DirectRingRelations` on the *shipped* (sumcheck-optimal) schedule
//! tables, across the full family matrix.
//!
//! For every shipped `(family, key)` that resolves against the generated
//! table, this materializes the same table structure twice:
//!
//! - `RingSwitchSumcheck` — the production preset's terminal mode today;
//! - `DirectRingRelations` — the terminal direct-relation mode, by wrapping the
//!   preset in [`DirectTerminalCfg`], which re-materializes the *same* shipped
//!   structure under the direct terminal accounting (terminal stage-2 dropped,
//!   `r_hat` quotient digits omitted).
//!
//! Both sides go through [`ScheduleSearchMode::RuntimeTableSeeded`], so on a
//! table hit they are pure materializations of the shipped structure — no DP
//! search. This is the production-relevant number: it answers "what does
//! shipping direct mode on today's tables save?" without depending on table
//! regeneration. Because direct mode only *removes* terminal bytes from a fixed
//! structure, the direct total is always `<= ` the sumcheck total; the test
//! asserts that and prints the per-family savings.
//!
//! Keys that miss the table are skipped: a miss falls through to the DP, and
//! the from-scratch direct-mode DP search is a separate concern (it is rebuilt
//! by the planner rework in PR #139 and is not exercised here).
//!
//! Run:
//!
//! ```bash
//! cargo test -p akita-planner --features test-utils --release \
//!   --test proof_size_direct_vs_sumcheck -- --ignored --nocapture
//! ```
//!
//! Direct terminal mode is transparent-only, so this is compiled out under
//! `feature = "zk"`.
#![cfg(all(feature = "test-utils", not(feature = "zk")))]
#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp16, fp32, fp64};
use akita_config::tensor_verifier;
use akita_config::CommitmentConfig;
use akita_planner::test_utils::DirectTerminalCfg;
use akita_planner::{find_optimal_schedule, ScheduleSearchMode};
use akita_types::{AkitaScheduleLookupKey, ClaimIncidenceSummary};

#[derive(Default, Clone)]
struct FamilyStats {
    keys: usize,
    skipped: usize,
    improved: usize,
    unchanged: usize,
    sum_saved_pct: f64,
    sum_saved_bytes: u128,
    best_saving_bytes: usize,
    best_saving_key: Option<AkitaScheduleLookupKey>,
    best_saving_pct: f64,
}

/// Materialize the shipped table structure for `key` under `Cfg`'s terminal
/// mode (table hit only). Returns `None` on table miss.
fn table_bytes<Cfg: CommitmentConfig>(key: AkitaScheduleLookupKey) -> Option<usize> {
    // Gate on a table hit: skip keys that would fall through to the DP.
    Cfg::schedule_plan(key).ok()??;
    let bytes = find_optimal_schedule::<Cfg>(key, ScheduleSearchMode::RuntimeTableSeeded)
        .unwrap_or_else(|e| panic!("materialize failed for key={key:?}: {e}"))
        .total_bytes;
    Some(bytes)
}

/// Compare shipped-table sumcheck vs direct re-cost for one family over its
/// `(num_polys, num_vars)` key cross-product. Returns the per-key regressions
/// (must be empty: direct re-cost of a fixed structure never grows).
fn compare_family<Cfg, Direct>(min_nv: usize, max_nv: usize) -> (FamilyStats, Vec<String>)
where
    Cfg: CommitmentConfig,
    Direct: CommitmentConfig,
{
    let mut stats = FamilyStats::default();
    let mut regressions = Vec::new();
    for num_polys in [1usize, 4] {
        for nv in min_nv..=max_nv {
            let incidence = ClaimIncidenceSummary::same_point(nv, num_polys).unwrap_or_else(|e| {
                panic!("incidence build failed nv={nv} polys={num_polys}: {e}")
            });
            let key = AkitaScheduleLookupKey::new_from_incidence(&incidence)
                .unwrap_or_else(|e| panic!("key build failed nv={nv} polys={num_polys}: {e}"));

            let (Some(sumcheck), Some(direct)) =
                (table_bytes::<Cfg>(key), table_bytes::<Direct>(key))
            else {
                stats.skipped += 1;
                continue;
            };

            stats.keys += 1;
            match direct.cmp(&sumcheck) {
                std::cmp::Ordering::Less => {
                    stats.improved += 1;
                    let saved = sumcheck - direct;
                    let pct = 100.0 * saved as f64 / sumcheck as f64;
                    stats.sum_saved_pct += pct;
                    stats.sum_saved_bytes += saved as u128;
                    if saved > stats.best_saving_bytes {
                        stats.best_saving_bytes = saved;
                        stats.best_saving_key = Some(key);
                        stats.best_saving_pct = pct;
                    }
                }
                std::cmp::Ordering::Equal => stats.unchanged += 1,
                std::cmp::Ordering::Greater => regressions.push(format!(
                    "key={key:?}: direct={direct} > sumcheck={sumcheck} (+{})",
                    direct - sumcheck
                )),
            }
        }
    }
    (stats, regressions)
}

macro_rules! families {
    ($($name:literal => ($cfg:ty, $min:expr, $max:expr)),+ $(,)?) => {
        vec![$((
            $name,
            compare_family::<$cfg, DirectTerminalCfg<$cfg>>($min, $max),
        )),+]
    };
}

#[test]
#[ignore = "full-matrix proof-size diagnostic; run with --ignored --release --features test-utils"]
fn terminal_direct_vs_sumcheck_proof_sizes() {
    let results = families! {
        "fp128_d32_full"          => (fp128::D32Full, 1, 50),
        "fp128_d32_onehot"        => (fp128::D32OneHot, 1, 50),
        "fp128_d64_full"          => (fp128::D64Full, 1, 50),
        "fp128_d64_onehot"        => (fp128::D64OneHot, 1, 50),
        "fp128_d64_onehot_tensor" => (tensor_verifier::fp128::D64OneHotTensor, 1, 50),
        "fp32_d32"                => (fp32::D32Full, 1, 32),
        "fp32_d32_onehot"         => (fp32::D32OneHot, 1, 32),
        "fp32_d64"                => (fp32::D64Full, 1, 32),
        "fp32_d64_onehot"         => (fp32::D64OneHot, 1, 32),
        "fp16_d32_full"           => (fp16::D32Full, 1, 32),
        "fp16_d32_onehot"         => (fp16::D32OneHot, 1, 32),
        "fp16_d64_full"           => (fp16::D64Full, 1, 32),
        "fp16_d64_onehot"         => (fp16::D64OneHot, 1, 32),
        "fp64_d32"                => (fp64::D32Full, 1, 32),
        "fp64_d32_onehot"         => (fp64::D32OneHot, 1, 32),
        "fp64_d64"                => (fp64::D64Full, 1, 32),
        "fp64_d64_onehot"         => (fp64::D64OneHot, 1, 32),
    };

    let mut all_regressions: Vec<String> = Vec::new();
    let (mut grand_improved, mut grand_pct, mut grand_bytes) = (0usize, 0.0f64, 0u128);
    let mut grand_best = (0usize, 0.0f64, String::from("-"));

    eprintln!(
        "\nFlip-to-direct savings on shipped tables (per terminal-fold key; \
         root-direct keys have 0 savings and are excluded from means):"
    );
    eprintln!(
        "  {:<26} {:>5} {:>9} {:>9} {:>11}  {:>26}",
        "family", "keys", "improved", "mean_save%", "bytes_saved", "best key (saved)",
    );
    for (name, (stats, regressions)) in &results {
        all_regressions.extend(regressions.iter().map(|r| format!("{name}: {r}")));
        grand_improved += stats.improved;
        grand_pct += stats.sum_saved_pct;
        grand_bytes += stats.sum_saved_bytes;

        let mean_pct = if stats.improved == 0 {
            0.0
        } else {
            stats.sum_saved_pct / stats.improved as f64
        };
        let best = match stats.best_saving_key {
            Some(key) => format!(
                "nv={} np={} (-{}B {:.2}%)",
                key.num_vars, key.num_w_vectors, stats.best_saving_bytes, stats.best_saving_pct
            ),
            None => "-".to_string(),
        };
        if stats.best_saving_bytes > grand_best.0 {
            grand_best = (
                stats.best_saving_bytes,
                stats.best_saving_pct,
                format!("{name} {best}"),
            );
        }
        eprintln!(
            "  {:<26} {:>5} {:>9} {:>9.2} {:>11}  {:>26}",
            name, stats.keys, stats.improved, mean_pct, stats.sum_saved_bytes, best,
        );
    }

    let grand_mean = if grand_improved == 0 {
        0.0
    } else {
        grand_pct / grand_improved as f64
    };
    eprintln!(
        "  {:<26} {:>5} {:>9} {:>9.2} {:>11}",
        "TOTAL", "", grand_improved, grand_mean, grand_bytes,
    );
    eprintln!("  single best key: {} (-{}B)", grand_best.2, grand_best.0);

    assert!(
        all_regressions.is_empty(),
        "direct re-cost increased proof size for {} key(s) (should be impossible for a fixed \
         structure):\n  {}",
        all_regressions.len(),
        all_regressions.join("\n  "),
    );
}
