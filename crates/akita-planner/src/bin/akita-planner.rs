//! Validate and inspect planner baseline schedules.

use std::env;

use akita_planner::baseline::{baseline_params_for, run_baseline_planner, BASELINE_CASES};
use akita_planner::search::{
    run_universal_planner, DirectWitnessShape, PlannerOptions, RingConfig, Schedule,
    ALL_RING_CONFIGS,
};

fn get_baseline(lcb: u32, nv: usize) -> Option<usize> {
    let d = if lcb == 1 {
        64
    } else if lcb >= 128 {
        128
    } else {
        return None;
    };
    let bp = baseline_params_for(d, lcb, nv);
    run_baseline_planner(&bp).map(|r| r.total)
}

fn d_schedule(sched: &Schedule) -> String {
    sched
        .fold_steps()
        .map(|l| l.d.to_string())
        .collect::<Vec<_>>()
        .join("->")
}

fn print_detailed(sched: &Schedule) {
    println!("  fold steps ({}):", sched.num_fold_levels());
    for (i, l) in sched.fold_steps().enumerate() {
        println!(
            "    L{}: D={} lb={} m={} r={} [{}]",
            i, l.d, l.lb, l.m_vars, l.r_vars, l.label
        );
        println!(
            "        na={} nb={} nd={}  do={} df={} dc={}  w_ring={}  next_w={}  level={}B",
            l.na,
            l.nb,
            l.nd,
            l.delta_open,
            l.delta_fold,
            l.delta_commit,
            l.w_ring,
            l.next_w_len,
            l.level_bytes
        );
    }
    if let Some(direct) = sched.direct_step() {
        match &direct.witness_shape {
            DirectWitnessShape::PackedDigits {
                num_elems,
                bits_per_elem,
            } => println!(
                "  terminal: direct packed-digits w_len={}  lb={}  witness={}B total={}B",
                num_elems, bits_per_elem, direct.direct_bytes, direct.total_bytes
            ),
            DirectWitnessShape::FieldElements { num_elems } => println!(
                "  terminal: direct field-elements w_len={}  witness={}B",
                num_elems, direct.total_bytes
            ),
        }
    }
    println!(
        "  TOTAL: {} B  ({:.1} KB)",
        sched.total_bytes,
        sched.total_bytes as f64 / 1024.0
    );
}

fn cmd_validate() -> bool {
    println!("{}", "=".repeat(70));
    println!("  Baseline Validation (vs Rust planner)");
    println!("{}", "=".repeat(70));

    let mut all_ok = true;
    for &(name, d, lcb, nv, expected) in BASELINE_CASES {
        let bp = baseline_params_for(d, lcb, nv);
        let result = run_baseline_planner(&bp);
        let got = result.map_or(0, |r| r.total);
        let ok = got == expected;
        if !ok {
            all_ok = false;
        }
        let mark = if ok { "ok" } else { "FAIL" };
        println!("  [{mark}]  {name} nv={nv}: got={got}  expected={expected}");
    }

    if all_ok {
        println!("\n  All baselines match.");
    } else {
        println!("\n  MISMATCH, model diverges from Rust planner!");
    }
    all_ok
}

fn cmd_results() {
    println!("{}", "=".repeat(70));
    println!("  Akita Universal Planner, Optimized Results");
    println!("  (eq-comp + tree@4 + tight z_pre + header stripping, 128-bit SIS)");
    println!("{}", "=".repeat(70));

    let configs: &[(&str, u32, &[usize])] = &[
        ("onehot", 1, &[20, 25, 30, 32, 38, 44]),
        ("full", 128, &[20, 25, 30, 32]),
    ];

    let mut headlines: Vec<(&str, usize, usize, usize)> = Vec::new();

    for &(poly_name, lcb, nvs) in configs {
        println!(
            "\n  {} (log_commit_bound={})",
            poly_name.to_uppercase(),
            lcb
        );
        println!(
            "  {:>4} {:>10} {:<25} {:>10}",
            "nv", "total", "D schedule", "tail"
        );
        println!(
            "  {} {} {} {}",
            "-".repeat(4),
            "-".repeat(10),
            "-".repeat(25),
            "-".repeat(10)
        );

        for &nv in nvs {
            let opts = PlannerOptions::new(lcb, nv);
            let sched = run_universal_planner(&opts);
            let ds = d_schedule(&sched);
            println!(
                "  {:>4} {:>10} {:<25} {:>10}",
                nv,
                sched.total_bytes,
                ds,
                sched.direct_bytes()
            );

            if let Some(baseline) = get_baseline(lcb, nv) {
                headlines.push((poly_name, nv, baseline, sched.total_bytes));
            }
        }
    }

    if !headlines.is_empty() {
        println!("\n{}", "-".repeat(70));
        println!("  Headline: optimized vs baseline");
        println!(
            "\n  {:<15} {:>4} {:>10} {:>10} {:>10}",
            "Poly type", "nv", "Baseline", "Optimized", "Reduction"
        );
        println!(
            "  {} {} {} {} {}",
            "-".repeat(15),
            "-".repeat(4),
            "-".repeat(10),
            "-".repeat(10),
            "-".repeat(10)
        );
        for (name, nv, baseline, optimized) in &headlines {
            let pct = (1.0 - *optimized as f64 / *baseline as f64) * 100.0;
            println!("  {name:<15} {nv:>4} {baseline:>10} {optimized:>10} {pct:>9.1}%");
        }
    }
}

fn cmd_breakdown() {
    println!("{}", "=".repeat(70));
    println!("  Detailed Level Breakdowns");
    println!("{}", "=".repeat(70));

    let cases: &[(&str, u32, usize)] = &[
        ("onehot", 1, 32),
        ("onehot", 1, 44),
        ("full", 128, 32),
        ("full", 128, 25),
    ];

    for &(name, lcb, nv) in cases {
        let baseline = get_baseline(lcb, nv);
        let opts = PlannerOptions::new(lcb, nv);
        let sched = run_universal_planner(&opts);
        print!("\n  {name} nv={nv}");
        if let Some(b) = baseline {
            let pct = (1.0 - sched.total_bytes as f64 / b as f64) * 100.0;
            print!("  (baseline: {b} B -> -{pct:.1}%)");
        }
        println!();
        print_detailed(&sched);
        println!();
    }
}

fn cmd_compare() {
    println!("{}", "=".repeat(70));
    println!("  Standard vs Tight z_pre (Column-Major Blocks)");
    println!("{}", "=".repeat(70));

    let configs: &[(&str, u32, &[usize])] = &[
        ("onehot", 1, &[20, 25, 30, 32, 38, 44]),
        ("full", 128, &[20, 25, 30, 32]),
    ];

    for &(poly_name, lcb, nvs) in configs {
        println!(
            "\n  {} (log_commit_bound={})",
            poly_name.to_uppercase(),
            lcb
        );
        println!(
            "  {:>4} {:>10} {:>10} {:>8} {:>7}",
            "nv", "standard", "tight", "saved", "%"
        );
        println!(
            "  {} {} {} {} {}",
            "-".repeat(4),
            "-".repeat(10),
            "-".repeat(10),
            "-".repeat(8),
            "-".repeat(7)
        );

        for &nv in nvs {
            let std_opts = PlannerOptions::new(lcb, nv).with_tight_zpre(false);
            let std_sched = run_universal_planner(&std_opts);
            let tgt_opts = PlannerOptions::new(lcb, nv).with_tight_zpre(true);
            let tgt_sched = run_universal_planner(&tgt_opts);
            let saved = std_sched.total_bytes.saturating_sub(tgt_sched.total_bytes);
            let pct = if std_sched.total_bytes > 0 {
                saved as f64 / std_sched.total_bytes as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "  {:>4} {:>10} {:>10} {:>8} {:>6.1}%",
                nv, std_sched.total_bytes, tgt_sched.total_bytes, saved, pct
            );
        }
    }
}

fn d128_tensor_ring_configs() -> Vec<RingConfig> {
    (1..=4)
        .map(|n_a| RingConfig {
            d: 128,
            n_a,
            challenge_l1_mass: 31 * 31,
            max_abs_challenge_coeff: 4 * 31,
            label: match n_a {
                1 => "D128-tensor-na1",
                2 => "D128-tensor-na2",
                3 => "D128-tensor-na3",
                _ => "D128-tensor-na4",
            },
        })
        .collect()
}

fn d128_flat_ring_configs() -> Vec<RingConfig> {
    ALL_RING_CONFIGS
        .iter()
        .filter(|cfg| cfg.d == 128)
        .cloned()
        .collect()
}

fn cmd_d128_tensor_compare() {
    println!("{}", "=".repeat(88));
    println!("  D=128 Flat vs Tensor What-If");
    println!("  Tensor model: honest mass=31^2, A-role extraction coeff=4*31");
    println!("{}", "=".repeat(88));
    println!(
        "  {:<8} {:>4} {:>12} {:<22} {:>4} {:>4} {:>4} {:>6} {:>6}",
        "case", "nv", "proof_bytes", "D schedule", "na0", "r0", "lb0", "blocks", "df0"
    );
    println!(
        "  {} {} {} {} {} {} {} {} {}",
        "-".repeat(8),
        "-".repeat(4),
        "-".repeat(12),
        "-".repeat(22),
        "-".repeat(4),
        "-".repeat(4),
        "-".repeat(4),
        "-".repeat(6),
        "-".repeat(6),
    );

    let cases: &[(&str, u32, &[usize])] = &[
        ("onehot", 1, &[20, 25, 30, 32, 38]),
        ("full", 128, &[20, 25, 30, 32]),
    ];
    for &(case, lcb, nvs) in cases {
        for &nv in nvs {
            for (shape, configs) in [
                ("flat", d128_flat_ring_configs()),
                ("tensor", d128_tensor_ring_configs()),
            ] {
                let mut opts = PlannerOptions::new(lcb, nv);
                opts.ring_configs = configs;
                let sched = run_universal_planner(&opts);
                let (na0, r0, lb0, blocks0, df0) =
                    sched.fold_steps().next().map_or((0, 0, 0, 0, 0), |level| {
                        (
                            level.na,
                            level.r_vars,
                            level.lb,
                            1usize << level.r_vars,
                            level.delta_fold,
                        )
                    });
                println!(
                    "  {:<8} {:>4} {:>12} {:<22} {:>4} {:>4} {:>4} {:>6} {:>6}  {}",
                    format!("{case}-{shape}"),
                    nv,
                    sched.total_bytes,
                    d_schedule(&sched),
                    na0,
                    r0,
                    lb0,
                    blocks0,
                    df0,
                    sched
                        .fold_steps()
                        .next()
                        .map_or("direct".to_string(), |level| level.label.to_string())
                );
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.iter().any(|a| a == "--validate") {
        let ok = cmd_validate();
        if !ok {
            std::process::exit(1);
        }
    } else if args.iter().any(|a| a == "--breakdown") {
        cmd_breakdown();
    } else if args.iter().any(|a| a == "--compare") {
        cmd_compare();
    } else if args.iter().any(|a| a == "--d128-tensor-compare") {
        cmd_d128_tensor_compare();
    } else {
        let ok = cmd_validate();
        println!();
        cmd_results();
        if !ok {
            std::process::exit(1);
        }
    }
}
