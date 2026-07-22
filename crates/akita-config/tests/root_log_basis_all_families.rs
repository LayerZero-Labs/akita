//! Diagnostic: planner proof-size impact of pinning the root fold to
//! `log_basis = 2` across **every** shipped preset family — small fields,
//! dense, recursion, and chunked-witness — at large `nv`.
//!
//! For each family it plans the unpinned baseline and a `root_log_basis =
//! Some(2)` variant (all deeper levels stay planner-chosen) and prints the
//! `estimated_direct_proof_payload_bytes` delta, plus the *effective* resolved
//! root `log_basis` (so a clamped/failed pin is visible). Planner estimates
//! only; no executed proofs.
//!
//! ```bash
//! cargo test -p akita-config --test root_log_basis_all_families -- --ignored --nocapture
//! # (rtk summarizes stdout; run the compiled binary directly for the full table)
//! ```

#![allow(missing_docs)]

use akita_config::generated_families::{GeneratedFamily, ALL_GENERATED_FAMILIES};
use akita_planner::{find_group_batch_schedule, PlannerPolicy};
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

/// Plan `key` under `policy` and return `(payload_bytes, effective_root_log_basis)`.
fn plan_info(
    policy: &PlannerPolicy,
    key: &AkitaScheduleLookupKey,
    family: &GeneratedFamily,
) -> Result<(usize, u32), String> {
    let planned = find_group_batch_schedule(
        key,
        policy,
        family.ring_challenge_config,
        family.fold_challenge_shape_at_level,
    )
    .map_err(|e| format!("{e:?}"))?;
    let bytes = planned
        .estimate
        .estimated_direct_proof_payload_bytes()
        .map_err(|e| format!("{e:?}"))?;
    let root_lb = planned
        .schedule
        .root
        .params
        .final_group
        .commitment
        .log_basis_open;
    Ok((bytes, root_lb))
}

fn fmt_cell(
    unpinned: &Result<(usize, u32), String>,
    pinned: &Result<(usize, u32), String>,
) -> String {
    match (unpinned, pinned) {
        (Ok((u, ulb)), Ok((p, plb))) => {
            let d = *p as i64 - *u as i64;
            let pct = 100.0 * d as f64 / *u as f64;
            format!(
                "unpinned {u} (root lb={ulb}) -> pinned {p} (root lb={plb})  [{d:+} B, {pct:+.2}%]"
            )
        }
        (Ok((u, ulb)), Err(e)) => {
            format!("unpinned {u} (root lb={ulb}) -> pinned PLAN FAILED: {e}")
        }
        (Err(e), _) => format!("unpinned PLAN FAILED: {e}"),
    }
}

#[test]
#[ignore = "diagnostic"]
fn root_log_basis_2_all_families_large_nv() {
    // Guard against an ambient bench override skewing the baseline.
    std::env::remove_var("AKITA_ROOT_LOG_BASIS");

    println!("# Pin root log_basis = 2 vs unpinned — planner proof bytes at large nv (np=1)\n");

    for family in ALL_GENERATED_FAMILIES {
        let base = (family.policy)();
        let mut pinned = base;
        pinned.root_log_basis = Some(2);

        let (min, max) = (family.min_num_vars, family.max_num_vars);
        let mut pts: Vec<usize> = [30usize, 36, 43, max]
            .into_iter()
            .filter(|&nv| nv >= min && nv <= max && nv >= 8)
            .collect();
        pts.sort_unstable();
        pts.dedup();

        println!(
            "## {}  (D={}, field_bits={}, log_commit_bound={}, basis_range={:?}, recursive={}, chunks={})",
            family.module_name,
            base.ring_dimension,
            base.decomposition.field_bits(),
            base.decomposition.log_commit_bound,
            base.basis_range,
            base.recursive_setup_planning,
            base.witness_chunk.num_chunks,
        );

        for nv in pts {
            let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(nv, 1));
            let u = plan_info(&base, &key, family);
            let p = plan_info(&pinned, &key, family);
            println!("  nv={nv:>2}: {}", fmt_cell(&u, &p));
        }

        // Multi-group / recursion keys (exercise the recursive setup path).
        if let Ok(gkeys) = (family.group_batch_keys)(family) {
            for key in gkeys.iter().filter(|k| !k.precommitteds.is_empty()) {
                let nv = key.final_group.num_vars();
                let u = plan_info(&base, key, family);
                let p = plan_info(&pinned, key, family);
                println!(
                    "  [group-batch nv={} np={} precommitted={}]: {}",
                    nv,
                    key.final_group.num_polynomials(),
                    key.precommitteds.len(),
                    fmt_cell(&u, &p),
                );
            }
        }
        println!();
    }
}
