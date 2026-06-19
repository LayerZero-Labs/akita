//! Phase-0 ladder vs fixed-D byte comparison and diagnostics.
//!
//! ```text
//! cargo run -p akita-config --release --bin ladder_byte_model
//! cargo run -p akita-config --release --bin ladder_byte_model -- --full --dims 128,64,32 28
//! cargo run -p akita-config --release --bin ladder_byte_model -- --fp32 --dims 256,128,64 28
//! cargo run -p akita-config --release --bin ladder_byte_model -- --fp64 --dims 128,64,32 24 28
//! cargo run -p akita-config --release --bin ladder_byte_model -- --fp64 --full --dims 128,64,32
//! ```

use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::{policy_of, CommitmentConfig};
use akita_planner::find_schedule;
use akita_planner::ladder_byte_model::{find_ladder_schedule, fold_ring_dimensions};
use akita_planner::PlannerPolicy;
use akita_types::{AkitaScheduleLookupKey, Schedule, Step};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Fp128OneHot,
    Fp128Full,
    Fp32OneHot,
    Fp64OneHot,
    Fp64Full,
}

struct RunConfig {
    family: Family,
    label: &'static str,
    default_dims: Vec<usize>,
    default_nv: Vec<usize>,
    policy: PlannerPolicy,
}

struct Cli {
    config: RunConfig,
    allowed_dims: Vec<usize>,
    num_vars: Vec<usize>,
    batch: usize,
}

fn parse_dims(args: &[String]) -> Option<Vec<usize>> {
    let idx = args.iter().position(|a| a == "--dims")?;
    let raw = args.get(idx + 1)?;
    let dims: Vec<usize> = raw
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if dims.is_empty() {
        None
    } else {
        Some(dims)
    }
}

fn parse_family(args: &[String]) -> Family {
    let fp64 = args.iter().any(|a| a == "--fp64");
    let fp32 = args.iter().any(|a| a == "--fp32");
    let full = args.iter().any(|a| a == "--full");
    match (fp64, fp32, full) {
        (true, _, true) => Family::Fp64Full,
        (true, _, false) => Family::Fp64OneHot,
        (_, true, _) => Family::Fp32OneHot,
        (_, _, true) => Family::Fp128Full,
        _ => Family::Fp128OneHot,
    }
}

fn run_config(family: Family) -> RunConfig {
    match family {
        Family::Fp128OneHot => RunConfig {
            family,
            label: "fp128 onehot",
            default_dims: vec![128, 64, 32],
            default_nv: vec![20, 22, 24, 26, 28, 30, 32],
            policy: policy_of::<fp128::D64OneHot>(),
        },
        Family::Fp128Full => RunConfig {
            family,
            label: "fp128 dense/full",
            default_dims: vec![128, 64, 32],
            default_nv: vec![20, 22, 24, 26, 28, 30, 32],
            policy: policy_of::<fp128::D64Full>(),
        },
        Family::Fp32OneHot => RunConfig {
            family,
            label: "fp32 onehot",
            default_dims: vec![256, 128, 64],
            default_nv: vec![12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
            policy: policy_of::<fp32::D128OneHot>(),
        },
        Family::Fp64OneHot => RunConfig {
            family,
            label: "fp64 onehot",
            default_dims: vec![128, 64, 32],
            default_nv: vec![16, 18, 20, 22, 24, 26, 28, 30],
            policy: policy_of::<fp64::D128OneHot>(),
        },
        Family::Fp64Full => RunConfig {
            family,
            label: "fp64 dense/full",
            default_dims: vec![128, 64, 32],
            default_nv: vec![16, 18, 20, 22, 24, 26, 28, 30],
            policy: policy_of::<fp64::D128Full>(),
        },
    }
}

fn parse_cli() -> Cli {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let config = run_config(parse_family(&args));
    let allowed_dims = parse_dims(&args).unwrap_or_else(|| config.default_dims.clone());
    let batch = args
        .iter()
        .position(|a| a == "--batch")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let skip: Vec<&str> = vec![
        "--full", "--onehot", "--fp32", "--fp64", "--dims", "--batch",
    ];
    let num_vars: Vec<usize> = args
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            if skip.contains(&a.as_str()) {
                return None;
            }
            if i > 0 && skip.contains(&args[i - 1].as_str()) {
                return None;
            }
            if a.contains(',') {
                return None;
            }
            a.parse().ok()
        })
        .collect();
    let num_vars = if num_vars.is_empty() {
        config.default_nv.clone()
    } else {
        num_vars
    };
    Cli {
        config,
        allowed_dims,
        num_vars,
        batch,
    }
}

fn stage1(family: Family, d: usize) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
    match family {
        Family::Fp128OneHot => fp128::D64OneHot::ring_challenge_config(d),
        Family::Fp128Full => fp128::D64Full::ring_challenge_config(d),
        Family::Fp32OneHot => fp32::D128OneHot::ring_challenge_config(d),
        Family::Fp64OneHot => fp64::D128OneHot::ring_challenge_config(d),
        Family::Fp64Full => fp64::D128Full::ring_challenge_config(d),
    }
}

fn fold_shape(
    family: Family,
    inputs: akita_types::AkitaScheduleInputs,
) -> akita_challenges::TensorChallengeShape {
    match family {
        Family::Fp128OneHot => fp128::D64OneHot::fold_challenge_shape_at_level(inputs),
        Family::Fp128Full => fp128::D64Full::fold_challenge_shape_at_level(inputs),
        Family::Fp32OneHot => fp32::D128OneHot::fold_challenge_shape_at_level(inputs),
        Family::Fp64OneHot => fp64::D128OneHot::fold_challenge_shape_at_level(inputs),
        Family::Fp64Full => fp64::D128Full::fold_challenge_shape_at_level(inputs),
    }
}

fn fixed_schedule_at_d(
    family: Family,
    key: AkitaScheduleLookupKey,
    d: usize,
) -> Result<Schedule, akita_field::AkitaError> {
    let policy = match (family, d) {
        (Family::Fp128OneHot, 32) => policy_of::<fp128::D32OneHot>(),
        (Family::Fp128OneHot, 64) => policy_of::<fp128::D64OneHot>(),
        (Family::Fp128OneHot, 128) => policy_of::<fp128::D128OneHot>(),
        (Family::Fp128Full, 32) => policy_of::<fp128::D32Full>(),
        (Family::Fp128Full, 64) => policy_of::<fp128::D64Full>(),
        (Family::Fp128Full, 128) => policy_of::<fp128::D128Full>(),
        (Family::Fp32OneHot, 64) => policy_of::<fp32::D64OneHot>(),
        (Family::Fp32OneHot, 128) => policy_of::<fp32::D128OneHot>(),
        (Family::Fp32OneHot, 256) => policy_of::<fp32::D256OneHot>(),
        (Family::Fp64OneHot, 32) => policy_of::<fp64::D32OneHot>(),
        (Family::Fp64OneHot, 64) => policy_of::<fp64::D64OneHot>(),
        (Family::Fp64OneHot, 128) => policy_of::<fp64::D128OneHot>(),
        (Family::Fp64Full, 32) => policy_of::<fp64::D32Full>(),
        (Family::Fp64Full, 64) => policy_of::<fp64::D64Full>(),
        (Family::Fp64Full, 128) => policy_of::<fp64::D128Full>(),
        _ => {
            return Err(akita_field::AkitaError::InvalidSetup(format!(
                "unsupported fixed-D preset for family={family:?} d={d}"
            )));
        }
    };
    find_schedule(
        key,
        &policy,
        |ring| stage1(family, ring),
        |inputs| fold_shape(family, inputs),
    )
}

fn best_fixed_selector(
    family: Family,
    key: AkitaScheduleLookupKey,
) -> Result<Option<(usize, &'static str)>, akita_field::AkitaError> {
    match family {
        Family::Fp128OneHot => {
            let sel = fp128::best_onehot_schedule(key)?;
            Ok(sel.map(|s| (s.schedule.total_bytes, s.preset.name())))
        }
        Family::Fp128Full => {
            let sel = fp128::best_full_schedule(key)?;
            Ok(sel.map(|s| (s.schedule.total_bytes, s.preset.name())))
        }
        Family::Fp32OneHot => {
            let sel = fp32::best_onehot_schedule(key)?;
            Ok(sel.map(|s| (s.schedule.total_bytes, s.preset.name())))
        }
        Family::Fp64OneHot => {
            let sel = fp64::best_onehot_schedule(key)?;
            Ok(sel.map(|s| (s.schedule.total_bytes, s.preset.name())))
        }
        Family::Fp64Full => {
            let sel = fp64::best_full_schedule(key)?;
            Ok(sel.map(|s| (s.schedule.total_bytes, s.preset.name())))
        }
    }
}

fn is_mixed_dims(dims: &[usize]) -> bool {
    dims.windows(2).any(|w| w[0] != w[1])
}

fn fold_bytes(schedule: &Schedule) -> usize {
    schedule
        .steps
        .iter()
        .filter_map(|s| match s {
            Step::Fold(f) => Some(f.level_bytes),
            Step::Direct(_) => None,
        })
        .sum()
}

fn terminal_bytes(schedule: &Schedule) -> usize {
    schedule
        .steps
        .iter()
        .filter_map(|s| match s {
            Step::Direct(d) => Some(d.direct_bytes),
            Step::Fold(_) => None,
        })
        .sum()
}

fn main() {
    let cli = parse_cli();
    let ladder_label: String = cli
        .allowed_dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("→");

    println!(
        "{}: ladder {ladder_label} vs fixed-D baselines",
        cli.config.label
    );
    if cli.batch > 1 {
        println!("batched root: num_t_vectors = {}", cli.batch);
    }
    println!();

    let mut any_ladder_win = false;
    let mut any_mixed = false;
    let mut any_ladder_bug = false;

    for nv in &cli.num_vars {
        let key = if cli.batch > 1 {
            AkitaScheduleLookupKey::new(*nv, cli.batch, cli.batch, 1)
        } else {
            AkitaScheduleLookupKey::singleton(*nv)
        };

        let mut fixed_rows: Vec<(usize, usize, Vec<usize>)> = Vec::new();
        for &d in &cli.allowed_dims {
            if let Ok(sched) = fixed_schedule_at_d(cli.config.family, key, d) {
                fixed_rows.push((d, sched.total_bytes, fold_ring_dimensions(&sched)));
            }
        }
        if fixed_rows.is_empty() {
            println!("nv={nv} batch={}: no fixed-D schedule succeeded, skipping", cli.batch);
            println!();
            continue;
        }
        let min_fixed = fixed_rows
            .iter()
            .min_by_key(|(_, b, _)| *b)
            .expect("non-empty fixed rows");

        let ladder = match find_ladder_schedule(
            key,
            &cli.config.policy,
            &cli.allowed_dims,
            |d| stage1(cli.config.family, d),
            |inputs| fold_shape(cli.config.family, inputs),
        ) {
            Ok(s) => s,
            Err(e) => {
                println!("nv={nv} batch={}: ladder failed: {e}", cli.batch);
                println!();
                continue;
            }
        };
        let ladder_dims = fold_ring_dimensions(&ladder);
        let mixed = is_mixed_dims(&ladder_dims);
        any_mixed |= mixed;

        let delta_vs_min = ladder.total_bytes as i64 - min_fixed.1 as i64;
        let selector_line = match best_fixed_selector(cli.config.family, key) {
            Ok(Some((bytes, name))) => {
                let delta = ladder.total_bytes as i64 - bytes as i64;
                if delta < 0 {
                    any_ladder_win = true;
                }
                format!("selector: {name} ({bytes} B)  delta vs selector: {delta:+}")
            }
            Ok(None) => "selector: (none)".to_string(),
            Err(e) => format!("selector error: {e}"),
        };

        if delta_vs_min < 0 {
            any_ladder_win = true;
        } else if ladder.total_bytes > min_fixed.1 {
            any_ladder_bug = true;
        }

        println!("nv={nv} batch={}:", cli.batch);
        for (d, bytes, dims) in &fixed_rows {
            println!("  fixed D{d}: {bytes} B  dims={dims:?}");
        }
        println!(
            "  min_fixed: D{} ({} B)",
            min_fixed.0, min_fixed.1
        );
        println!("  {selector_line}");
        println!(
            "  ladder:   {} B  dims={ladder_dims:?}  mixed={mixed}",
            ladder.total_bytes
        );
        println!("  delta vs min_fixed: {delta_vs_min:+}");
        println!(
            "  fold/terminal ladder: {}/{} B",
            fold_bytes(&ladder),
            terminal_bytes(&ladder)
        );
        println!();
    }

    println!("--- summary ---");
    println!("any ladder win vs min fixed-D: {any_ladder_win}");
    println!("any mixed-D ladder schedule:   {any_mixed}");
    if any_ladder_bug {
        println!("WARNING: ladder exceeded min fixed-D on some key (DP superset bug?)");
    }
}
