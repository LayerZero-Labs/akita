//! Diagnostic: compare the post-refactor planner output against the shipped
//! schedule tables, key-by-key.
//!
//! Runs the table-backed resolution (`family.table_backed`, i.e. the table
//! fast path with DP fallback) against the from-scratch DP (`family.regen`)
//! for every shipped `(family, key)` and reports:
//!
//! - Per-family proof-size sums (old vs new) and the delta.
//! - Per-family counts of keys whose plan changed (smaller / bigger / equal)
//!   and counts of keys whose step structure changed.
//! - The largest few improvements and the largest few regressions by bytes.
//! - The largest few structural changes (step count delta, log_basis set
//!   delta) for inspection.
//!
//! Marked `#[ignore]` so it never runs in `cargo test`; invoke with
//! `cargo test -p akita-config --test regen_diff -- --ignored --nocapture`.

#![allow(missing_docs)]

use std::collections::BTreeMap;

use akita_config::generated_families::{family_keys, ALL_GENERATED_FAMILIES};
use akita_types::Schedule;

#[derive(Default, Clone, Copy)]
struct FamilyTotals {
    old_sum: i128,
    new_sum: i128,
    smaller: usize,
    bigger: usize,
    equal: usize,
    structure_changed: usize,
}

#[derive(Clone)]
struct ChangedKey {
    family: &'static str,
    num_vars: usize,
    num_polys: usize,
    old_bytes: usize,
    new_bytes: usize,
    old_steps: usize,
    new_steps: usize,
    old_lb_set: Vec<(u32, u32, u32)>,
    new_lb_set: Vec<(u32, u32, u32)>,
}

impl ChangedKey {
    fn bytes_delta(&self) -> i128 {
        self.new_bytes as i128 - self.old_bytes as i128
    }
    fn step_delta(&self) -> i128 {
        self.new_steps as i128 - self.old_steps as i128
    }
}

fn step_count(s: &Schedule) -> usize {
    s.folds.len() + 1
}

fn basis_set(s: &Schedule) -> Vec<(u32, u32, u32)> {
    let mut v: Vec<(u32, u32, u32)> = s
        .folds
        .iter()
        .map(|fold| {
            (
                fold.params.log_basis_inner,
                fold.params.log_basis_outer,
                fold.params.log_basis_open,
            )
        })
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

#[test]
#[ignore = "diagnostic"]
fn regen_diff_vs_shipped_tables() {
    let mut by_family: BTreeMap<&'static str, FamilyTotals> = BTreeMap::new();
    let mut all_changed: Vec<ChangedKey> = Vec::new();

    for family in ALL_GENERATED_FAMILIES {
        let keys = match family_keys(family) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[skip] {} key enumeration: {e}", family.module_name);
                continue;
            }
        };
        for key in keys {
            let old = match (family.table_backed)(key) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let new = match (family.regen)(key) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let entry = by_family.entry(family.module_name).or_default();
            entry.old_sum += old.total_bytes as i128;
            entry.new_sum += new.total_bytes as i128;

            let old_steps = step_count(&old);
            let new_steps = step_count(&new);
            let old_lb = basis_set(&old);
            let new_lb = basis_set(&new);
            let structure_changed = old_steps != new_steps || old_lb != new_lb;

            if new.total_bytes < old.total_bytes {
                entry.smaller += 1;
            } else if new.total_bytes > old.total_bytes {
                entry.bigger += 1;
            } else {
                entry.equal += 1;
            }
            if structure_changed {
                entry.structure_changed += 1;
            }

            if new.total_bytes != old.total_bytes || structure_changed {
                let num_polys = key.num_polynomials();
                all_changed.push(ChangedKey {
                    family: family.module_name,
                    num_vars: key.num_vars(),
                    num_polys,
                    old_bytes: old.total_bytes,
                    new_bytes: new.total_bytes,
                    old_steps,
                    new_steps,
                    old_lb_set: old_lb,
                    new_lb_set: new_lb,
                });
            }
        }
    }

    println!("\n=== Per-family summary ===");
    println!(
        "{:<28} {:>16} {:>16} {:>16}  {:>6} {:>6} {:>6}  {:>9}",
        "family", "old_sum", "new_sum", "delta(new-old)", "smaller", "bigger", "equal", "struct≠"
    );
    let (mut tot_old, mut tot_new) = (0i128, 0i128);
    let (mut tot_s, mut tot_b, mut tot_e, mut tot_struct) = (0usize, 0usize, 0usize, 0usize);
    for (family, t) in &by_family {
        println!(
            "{:<28} {:>16} {:>16} {:>+16}  {:>6} {:>6} {:>6}  {:>9}",
            family,
            t.old_sum,
            t.new_sum,
            t.new_sum - t.old_sum,
            t.smaller,
            t.bigger,
            t.equal,
            t.structure_changed
        );
        tot_old += t.old_sum;
        tot_new += t.new_sum;
        tot_s += t.smaller;
        tot_b += t.bigger;
        tot_e += t.equal;
        tot_struct += t.structure_changed;
    }
    println!(
        "{:<28} {:>16} {:>16} {:>+16}  {:>6} {:>6} {:>6}  {:>9}",
        "TOTAL",
        tot_old,
        tot_new,
        tot_new - tot_old,
        tot_s,
        tot_b,
        tot_e,
        tot_struct
    );

    let mut improvements: Vec<&ChangedKey> =
        all_changed.iter().filter(|c| c.bytes_delta() < 0).collect();
    improvements.sort_by_key(|c| c.bytes_delta());
    println!("\n=== Top 15 byte improvements (new < old) ===");
    for c in improvements.iter().take(15) {
        println!(
            "  {fam:<28} nv={nv:<3} polys={np}  old={old:<16}  new={new:<12}  Δ={dl}  steps {os}->{ns}  lb {old_lb:?}->{new_lb:?}",
            fam = c.family,
            nv = c.num_vars,
            np = c.num_polys,
            old = c.old_bytes,
            new = c.new_bytes,
            dl = c.bytes_delta(),
            os = c.old_steps,
            ns = c.new_steps,
            old_lb = c.old_lb_set,
            new_lb = c.new_lb_set,
        );
    }
    if improvements.is_empty() {
        println!("  (none)");
    }

    let mut regressions: Vec<&ChangedKey> =
        all_changed.iter().filter(|c| c.bytes_delta() > 0).collect();
    regressions.sort_by_key(|c| -c.bytes_delta());
    println!("\n=== Top 15 byte regressions (new > old) ===");
    for c in regressions.iter().take(15) {
        println!(
            "  {fam:<28} nv={nv:<3} polys={np}  old={old:<12}  new={new:<16}  Δ=+{dl}  steps {os}->{ns}  lb {old_lb:?}->{new_lb:?}",
            fam = c.family,
            nv = c.num_vars,
            np = c.num_polys,
            old = c.old_bytes,
            new = c.new_bytes,
            dl = c.bytes_delta(),
            os = c.old_steps,
            ns = c.new_steps,
            old_lb = c.old_lb_set,
            new_lb = c.new_lb_set,
        );
    }
    if regressions.is_empty() {
        println!("  (none)");
    }

    let mut struct_only: Vec<&ChangedKey> = all_changed
        .iter()
        .filter(|c| c.bytes_delta() == 0 && (c.step_delta() != 0 || c.old_lb_set != c.new_lb_set))
        .collect();
    struct_only.sort_by_key(|c| c.step_delta());
    println!("\n=== Same-bytes structural changes (top 10) ===");
    for c in struct_only.iter().take(10) {
        println!(
            "  {fam:<28} nv={nv:<3} polys={np}  bytes={b}  steps {os}->{ns}  lb {old_lb:?}->{new_lb:?}",
            fam = c.family,
            nv = c.num_vars,
            np = c.num_polys,
            b = c.old_bytes,
            os = c.old_steps,
            ns = c.new_steps,
            old_lb = c.old_lb_set,
            new_lb = c.new_lb_set,
        );
    }
    if struct_only.is_empty() {
        println!("  (none)");
    }
}
