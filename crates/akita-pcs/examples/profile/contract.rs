//! Canonical profile workload shapes, checked before allocating witnesses.
use akita_config::CommitmentConfig;
use serde_json::Value;

pub(crate) fn attest<Cfg: CommitmentConfig>(mode: &str, nv: usize, np: usize) -> Value {
    let setup = std::env::var("AKITA_SETUP_MODE").unwrap_or_else(|_| "direct".into());
    let rows: Vec<Value> = serde_json::from_str(include_str!("workloads.json"))
        .expect("profile workload declarations");
    let row = rows
        .into_iter()
        .find(|row| row["mode"] == mode && row["setup_mode"] == setup)
        .expect("unsupported profile mode/setup combination");
    let metadata = checked_metadata::<Cfg>(&row, mode, &setup, nv, np);
    if std::env::var("AKITA_PROFILE_CONTRACT_ONLY").as_deref() == Ok("1") {
        println!("{metadata}");
        std::process::exit(0);
    }
    tracing::info!(metadata = %metadata, "profile workload contract");
    row
}

fn checked_metadata<Cfg: CommitmentConfig>(
    row: &Value,
    mode: &str,
    setup: &str,
    nv: usize,
    np: usize,
) -> Value {
    for (key, value) in [("num_vars", nv), ("num_polys", np)] {
        assert!(
            value > 0
                && (row[key].is_null()
                    || row[key]
                        .as_array()
                        .expect("workload shape list")
                        .iter()
                        .any(|v| v.as_u64() == Some(value as u64))),
            "unsupported {mode} {key}={value}"
        );
    }
    let pre = row["precommitted"]
        .as_array()
        .expect("precommitted topology");
    if !pre.is_empty() {
        assert_eq!(pre.len(), 2, "multi-group driver requires two precommits");
        assert_eq!(
            pre[0], pre[1],
            "multi-group driver requires equal precommits"
        );
        assert_eq!(pre[0][1], 1, "multi-group precommits are singletons");
        let final_np = row["final_polys"].as_u64().expect("final polynomial count");
        assert_eq!(2 + final_np, np as u64, "inconsistent group topology");
    }
    assert_eq!(
        row["family"],
        Cfg::schedule_family_name(),
        "profile dispatch/catalog mismatch"
    );
    let source_k = akita_config::unit_onehot_source_chunk_size::<Cfg>()
        .ok()
        .map(|k| k.min(1usize.checked_shl(nv as u32).expect("profile arity")));
    serde_json::json!({"mode": mode, "setup_mode": setup,
        "family": Cfg::schedule_family_name(), "num_vars": nv, "num_polys": np,
        "onehot_k": source_k, "precommitted": row["precommitted"],
        "final_polys": row["final_polys"]})
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_config::proof_optimized::fp32;

    #[test]
    fn wrong_family_is_rejected_even_when_both_catalogs_have_the_shape() {
        let row = serde_json::json!({"family": "fp32_onehot", "num_vars": [30],
            "num_polys": [1], "precommitted": [], "final_polys": null});
        let failed = std::panic::catch_unwind(|| {
            checked_metadata::<fp32::Dense>(&row, "dense_fp32", "direct", 30, 1)
        });
        assert!(failed.is_err());
    }

    #[test]
    fn shape_is_checked_before_any_witness_allocation() {
        let row = serde_json::json!({"family": "fp32_dense", "num_vars": [30],
            "num_polys": [1], "precommitted": [], "final_polys": null});
        assert!(std::panic::catch_unwind(|| {
            checked_metadata::<fp32::Dense>(&row, "dense_fp32", "direct", 31, 1)
        })
        .is_err());
        let metadata = checked_metadata::<fp32::Dense>(&row, "dense_fp32", "direct", 30, 1);
        assert!(metadata["onehot_k"].is_null());
    }
}
