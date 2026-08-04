//! Diagnostic comparison of generated typed schedules against fresh planner output.
//!
//! Run with `cargo test -p akita-config --test regen_diff -- --ignored --nocapture`.

#![allow(missing_docs)]

use akita_planner::generated_families::{family_keys, ALL_GENERATED_FAMILIES};

#[test]
#[ignore = "diagnostic"]
fn regen_diff_vs_generated_tables() {
    let mut compared = 0usize;
    let mut changed = 0usize;
    for family in ALL_GENERATED_FAMILIES {
        let Ok(keys) = family_keys(family) else {
            continue;
        };
        for key in keys {
            let (Ok(table_backed), Ok(regenerated)) =
                ((family.table_backed)(key), (family.regen)(key))
            else {
                continue;
            };
            compared += 1;
            if format!("{table_backed:?}") != format!("{regenerated:?}") {
                changed += 1;
                println!(
                    "{} nv={} k={}\n  table_backed={table_backed:?}\n  regenerated={regenerated:?}",
                    family.module_name,
                    key.num_vars(),
                    key.num_polynomials(),
                );
            }
        }
    }
    println!("compared={compared} changed={changed}");
    assert!(compared > 0, "no generated schedules were compared");
}
