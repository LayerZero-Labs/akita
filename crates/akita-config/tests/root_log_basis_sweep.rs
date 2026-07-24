//! Diagnostic: planner proof-size sweep across alternative fixed root-fold
//! `log_basis` values for the fp128 `D = 64` one-hot preset.
//!
//! Prints `FoldScheduleEstimate::estimated_direct_proof_payload_bytes` for each
//! `(policy, nv)` cell so the empirical tables in
//! `specs/pinned-early-log-basis.md` can be regenerated with realistic numbers.
//! The root is derived from `basis_range.0`; fold levels `≥ 1` search the full
//! range subject to the non-decreasing basis constraint. Planner estimates only
//! (no executed proofs); use the `profile` example for runtime
//! commit/prove/verify.
//!
//! ```bash
//! cargo test -p akita-config --test root_log_basis_sweep -- --ignored --nocapture
//! ```

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::{policy_of, CommitmentConfig};
use akita_planner::{find_group_batch_schedule, PlannerPolicy};
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

type Cfg = fp128::D64OneHot;

/// Alternative lower endpoints compared in the study. At fp128 these are the
/// exact root bases; the same endpoint bounds the deeper search, which also
/// enforces the non-decreasing basis constraint.
const POLICIES: &[(&str, u32)] = &[("root=2", 2), ("root=3", 3), ("root=4", 4)];

fn payload_bytes(policy: &PlannerPolicy, nv: usize) -> Result<usize, String> {
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(nv, 1));
    let planned = find_group_batch_schedule(
        &key,
        policy,
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )
    .map_err(|e| format!("plan error: {e:?}"))?;
    planned
        .estimate
        .estimated_direct_proof_payload_bytes()
        .map_err(|e| format!("estimate error: {e:?}"))
}

/// Per-level `(log_basis_open, output_witness_len)` of the resolved schedule
/// (for auditing that the fixed root took and for the report's schedule-anatomy
/// table). Index 0 is the root fold; the last entry is the terminal input.
fn schedule_anatomy(policy: &PlannerPolicy, nv: usize) -> Vec<(u32, usize)> {
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(nv, 1));
    match find_group_batch_schedule(
        &key,
        policy,
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    ) {
        Ok(planned) => {
            let s = &planned.schedule;
            let mut rows = vec![(
                s.root.params.final_group.commitment.log_basis_open,
                s.root.output_witness_len,
            )];
            rows.extend(
                s.recursive_folds
                    .iter()
                    .map(|f| (f.params.witness.log_basis_open, f.output_witness_len)),
            );
            rows
        }
        Err(_) => Vec::new(),
    }
}

#[test]
#[ignore = "diagnostic"]
fn fp128_d64_onehot_fixed_root_basis_sweep_proof_sizes() {
    const NV_LO: usize = 30;
    const NV_HI: usize = 43;

    // Header.
    print!("nv");
    for (label, _) in POLICIES {
        print!(",{label}");
    }
    println!();

    let base = policy_of::<Cfg>();
    let policies: Vec<(&str, PlannerPolicy)> = POLICIES
        .iter()
        .map(|(label, root_basis)| {
            let mut p = base;
            p.basis_range.0 = *root_basis;
            (*label, p)
        })
        .collect();

    let mut sums = vec![0usize; policies.len()];
    for nv in NV_LO..=NV_HI {
        print!("{nv}");
        for (i, (_, policy)) in policies.iter().enumerate() {
            match payload_bytes(policy, nv) {
                Ok(bytes) => {
                    sums[i] += bytes;
                    print!(",{bytes}");
                }
                Err(e) => print!(",ERR({e})"),
            }
        }
        println!();
    }

    print!("SUM");
    for s in &sums {
        print!(",{s}");
    }
    println!();

    // Schedule anatomy at nv=36 for auditing that the root pin took.
    println!("\n# schedule anatomy @ nv=36 (per-level log_basis / next_w)");
    for (label, policy) in &policies {
        let rows = schedule_anatomy(policy, 36);
        let lbs: Vec<u32> = rows.iter().map(|(lb, _)| *lb).collect();
        println!(
            "{label}: log_basis={lbs:?} (fold_levels={})",
            rows.len() + 1
        );
        for (lv, (lb, next_w)) in rows.iter().enumerate() {
            println!("    L{lv}: lb={lb} next_w={next_w}");
        }
    }

    // Root commit geometry @ nv=36: n_a (A-role rank) drives commit cost.
    println!("\n# root commit geometry @ nv=36 (drives commit/prove cost)");
    for (label, policy) in &policies {
        let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(36, 1));
        if let Ok(planned) = find_group_batch_schedule(
            &key,
            policy,
            Cfg::ring_challenge_config,
            Cfg::fold_challenge_shape_at_level,
        ) {
            let c = &planned.schedule.root.params.final_group.commitment;
            println!(
                "{label}: root lb={} n_a={} n_b={} delta_open={} num_digits_inner={} \
                 num_live_blocks={} num_positions_per_block={}",
                c.log_basis_open,
                c.inner_commit_matrix.output_rank(),
                c.outer_commit_matrix.output_rank(),
                c.num_digits_open,
                c.num_digits_inner,
                c.num_live_blocks,
                c.num_positions_per_block,
            );
        }
    }
}
