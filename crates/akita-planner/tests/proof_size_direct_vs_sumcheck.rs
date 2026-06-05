//! On-demand diagnostic: proof-size savings from direct terminal mode on shipped tables.
#![cfg(not(feature = "zk"))]
#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::test_support::DirectTerminalCfg;
use akita_config::{policy_of, CommitmentConfig};
use akita_planner::{generated_schedule_lookup_key, shipped_table, table_entry};
use akita_types::AkitaScheduleLookupKey;

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

fn table_hit<Cfg: CommitmentConfig>(key: AkitaScheduleLookupKey) -> bool {
    let policy = policy_of::<Cfg>();
    let root_fold_is_tensor =
        Cfg::fold_challenge_shape_at_level(akita_types::AkitaScheduleInputs {
            num_vars: key.num_vars,
            level: 0,
            current_w_len: 1usize.checked_shl(key.num_vars as u32).unwrap_or(0),
        }) == akita_challenges::TensorChallengeShape::Tensor;
    shipped_table(&policy, root_fold_is_tensor)
        .and_then(|table| table_entry(table, &generated_schedule_lookup_key(key)))
        .is_some()
}

fn runtime_bytes<Cfg: CommitmentConfig>(key: AkitaScheduleLookupKey) -> Option<usize> {
    if !table_hit::<Cfg>(key) {
        return None;
    }
    Cfg::runtime_schedule(key)
        .ok()
        .map(|schedule| schedule.total_bytes)
}

fn compare_family<Cfg, Direct>(min_nv: usize, max_nv: usize) -> (FamilyStats, Vec<String>)
where
    Cfg: CommitmentConfig,
    Direct: CommitmentConfig,
{
    let mut stats = FamilyStats::default();
    let mut regressions = Vec::new();
    for num_polys in [1usize, 4] {
        for nv in min_nv..=max_nv {
            let key =
                AkitaScheduleLookupKey::new_with_points(nv, num_polys, num_polys, num_polys, 1);
            let Some(sumcheck_bytes) = runtime_bytes::<Cfg>(key) else {
                stats.skipped += 1;
                continue;
            };
            let Some(direct_bytes) = runtime_bytes::<Direct>(key) else {
                stats.skipped += 1;
                continue;
            };
            stats.keys += 1;
            if direct_bytes > sumcheck_bytes {
                regressions.push(format!(
                    "regression key={key:?}: direct={direct_bytes} > sumcheck={sumcheck_bytes}"
                ));
            } else if direct_bytes < sumcheck_bytes {
                stats.improved += 1;
                let saved = sumcheck_bytes - direct_bytes;
                stats.sum_saved_bytes += saved as u128;
                let pct = 100.0 * saved as f64 / sumcheck_bytes as f64;
                stats.sum_saved_pct += pct;
                if saved > stats.best_saving_bytes {
                    stats.best_saving_bytes = saved;
                    stats.best_saving_key = Some(key);
                    stats.best_saving_pct = pct;
                }
            } else {
                stats.unchanged += 1;
            }
        }
    }
    (stats, regressions)
}

fn print_family<Cfg: CommitmentConfig>(name: &str, stats: &FamilyStats) {
    if stats.keys == 0 {
        eprintln!(
            "[{name}] no table hits in range (skipped {})",
            stats.skipped
        );
        return;
    }
    let mean_pct = stats.sum_saved_pct / stats.improved.max(1) as f64;
    eprintln!(
        "[{name}] keys={} improved={} unchanged={} skipped={} mean_save={mean_pct:.2}% total_saved={}",
        stats.keys, stats.improved, stats.unchanged, stats.skipped, stats.sum_saved_bytes
    );
    if let Some(key) = stats.best_saving_key {
        eprintln!(
            "  best: key={key:?} saved={} ({:.2}%)",
            stats.best_saving_bytes, stats.best_saving_pct
        );
    }
}

#[test]
#[ignore = "on-demand shipped-table direct-vs-sumcheck diagnostic"]
fn proof_size_direct_vs_sumcheck() {
    let families: [(&str, fn() -> (FamilyStats, Vec<String>)); 6] = [
        ("fp32_d32_onehot", || {
            compare_family::<fp32::D32OneHot, DirectTerminalCfg<fp32::D32OneHot>>(26, 32)
        }),
        ("fp32_d256_onehot", || {
            compare_family::<fp32::D256OneHot, DirectTerminalCfg<fp32::D256OneHot>>(26, 32)
        }),
        ("fp64_d128_onehot", || {
            compare_family::<fp64::D128OneHot, DirectTerminalCfg<fp64::D128OneHot>>(26, 32)
        }),
        ("fp64_d256_onehot", || {
            compare_family::<fp64::D256OneHot, DirectTerminalCfg<fp64::D256OneHot>>(26, 32)
        }),
        ("fp128_d64_onehot", || {
            compare_family::<fp128::D64OneHot, DirectTerminalCfg<fp128::D64OneHot>>(26, 32)
        }),
        ("fp128_d128_onehot", || {
            compare_family::<fp128::D128OneHot, DirectTerminalCfg<fp128::D128OneHot>>(26, 32)
        }),
    ];
    let mut all_regressions = Vec::new();
    for (name, run) in families {
        let (stats, regressions) = run();
        print_family::<fp32::D32OneHot>(name, &stats);
        all_regressions.extend(regressions);
    }
    assert!(
        all_regressions.is_empty(),
        "direct mode must not increase proof size on shipped tables:\n{}",
        all_regressions.join("\n")
    );
}
