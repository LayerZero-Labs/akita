//! Direct SIS security estimates for checked-in schedule artifacts.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use akita_planner::generated_families::{GeneratedFamily, ALL_GENERATED_FAMILIES};
use akita_sis_estimator::{
    estimate_schedule_security, ScheduleSisBound, ScheduleSisInstanceEstimate, SisSecurityPolicy,
};

const DETAIL_INSTANCE_HEADER: [&str; 10] = [
    "instance",
    "index",
    "location",
    "role",
    "modulus_profile",
    "norm_bound",
    "d",
    "rank",
    "width",
    "attack_cost_bits",
];

fn usage() -> &'static str {
    "usage: cargo run --release -p akita-planner --features catalog-security \
     --example catalog_security -- [--check] [--details] \
     [--final-group NUM_VARSxNUM_POLYNOMIALS] [--row-digest HEX] [family_name ...]"
}

fn parse_group(value: &str) -> Result<(usize, usize), String> {
    let (num_vars, num_polynomials) = value.split_once('x').ok_or_else(|| {
        format!("invalid final group {value:?}; expected NUM_VARSxNUM_POLYNOMIALS")
    })?;
    Ok((
        num_vars
            .parse()
            .map_err(|_| format!("invalid final-group variable count {num_vars:?}"))?,
        num_polynomials
            .parse()
            .map_err(|_| format!("invalid final-group polynomial count {num_polynomials:?}"))?,
    ))
}

fn selected_families(names: &[String]) -> Result<Vec<&'static GeneratedFamily>, String> {
    if names.is_empty() {
        return Ok(ALL_GENERATED_FAMILIES.iter().collect());
    }
    names
        .iter()
        .map(|name| {
            ALL_GENERATED_FAMILIES
                .iter()
                .find(|family| family.family_name() == name)
                .ok_or_else(|| format!("unknown generated schedule family {name:?}\n{}", usage()))
        })
        .collect()
}

fn group_label(group: akita_types::PolynomialGroupLayout) -> String {
    format!("{}x{}", group.num_vars(), group.num_polynomials())
}

fn digest_label(digest: akita_types::ScheduleRowDigest) -> String {
    let mut label = String::with_capacity(64);
    for byte in digest.as_bytes() {
        write!(&mut label, "{byte:02x}").expect("writing to String cannot fail");
    }
    label
}

fn parse_digest_filter(value: &str) -> Result<String, String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "invalid row digest {value:?}; expected 64 hexadecimal characters"
        ));
    }
    Ok(value.to_ascii_lowercase())
}

fn bound_label(instance: &ScheduleSisInstanceEstimate) -> String {
    match instance.bound {
        ScheduleSisBound::Linf(bound) => format!("linf:{bound}"),
        ScheduleSisBound::L2Squared(bound) => format!("l2sq:{bound}"),
    }
}

fn main() -> Result<(), String> {
    let mut check = false;
    let mut details = false;
    let mut final_group = None;
    let mut row_digest_filter = None;
    let mut names = Vec::new();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--check" => check = true,
            "--details" => details = true,
            "--final-group" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("--final-group requires a value\n{}", usage()))?;
                final_group = Some(parse_group(&value)?);
            }
            "--row-digest" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("--row-digest requires a value\n{}", usage()))?;
                row_digest_filter = Some(parse_digest_filter(&value)?);
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unknown option {arg:?}\n{}", usage()));
            }
            _ => names.push(arg),
        }
    }

    println!("family\trow_digest\tfinal_group\tprecommitted_groups\tsis_policy\tmodulus_profile\tmin_attack_cost_bits\tweakest_instance\tnorm_bound\td\trank\twidth");
    let mut matched_rows = 0usize;
    let mut below_policy = Vec::new();
    for family in selected_families(&names)? {
        let policy = (family.policy)();
        let artifact_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/schedules")
            .join(format!("{}.aks", family.family_name()));
        let bytes = fs::read(&artifact_path)
            .map_err(|error| format!("read {}: {error}", artifact_path.display()))?;
        let catalog = akita_schedules::TrustedScheduleCatalog::from_artifact_bytes(
            &bytes,
            family.family_name(),
            &policy,
            family.ring_challenge_config,
        )
        .map_err(|error| format!("load {}: {error}", artifact_path.display()))?;
        let policy_minimum_bits = SisSecurityPolicy::from(policy.sis_security_policy)
            .adps16_quantum_constraint()
            .minimum_log2_rop;
        for resolved in catalog.rows() {
            let profiles = resolved.profiles();
            let key = akita_types::AkitaScheduleLookupKey {
                final_group: profiles.final_group.group,
                precommitteds: profiles.precommitteds.clone(),
            };
            if final_group.is_some_and(|(num_vars, num_polynomials)| {
                key.final_group.num_vars() != num_vars
                    || key.final_group.num_polynomials() != num_polynomials
            }) {
                continue;
            }
            let row_digest = digest_label(resolved.selection().row_digest);
            if row_digest_filter
                .as_ref()
                .is_some_and(|expected| expected != &row_digest)
            {
                continue;
            }
            matched_rows += 1;
            let schedule = resolved.schedule();
            let estimate = estimate_schedule_security(schedule)
                .map_err(|error| format!("{} {:?}: {error}", family.family_name(), key))?;
            let weakest = estimate.minimum();
            let precommitted = if key.precommitteds.is_empty() {
                "-".to_string()
            } else {
                key.precommitteds
                    .iter()
                    .map(|profile| group_label(profile.group))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            println!(
                "{}\t{}\t{}\t{}\t{}\t{:?}\t{:.6}\t{}\t{}\t{}\t{}\t{}",
                family.family_name(),
                row_digest,
                group_label(key.final_group),
                precommitted,
                policy.sis_security_policy.name(),
                weakest.modulus_profile,
                estimate.minimum_security_bits(),
                weakest.location,
                bound_label(weakest),
                weakest.ring_dimension,
                weakest.output_rank,
                weakest.input_width,
            );
            let minimum_bits = estimate.minimum_security_bits();
            if check && (!minimum_bits.is_finite() || minimum_bits < policy_minimum_bits) {
                below_policy.push(format!(
                    "{} {}: {:.6} bits at {}",
                    family.family_name(),
                    row_digest,
                    minimum_bits,
                    weakest.location
                ));
            }
            if details {
                println!("schedule\t{schedule:#?}");
                println!("{}", DETAIL_INSTANCE_HEADER.join("\t"));
                for (index, instance) in estimate.instances().iter().enumerate() {
                    let columns: [String; DETAIL_INSTANCE_HEADER.len()] = [
                        "instance".to_string(),
                        index.to_string(),
                        instance.location.clone(),
                        format!("{:?}", instance.role),
                        format!("{:?}", instance.modulus_profile),
                        bound_label(instance),
                        instance.ring_dimension.to_string(),
                        instance.output_rank.to_string(),
                        instance.input_width.to_string(),
                        format!("{:.6}", instance.security_bits()),
                    ];
                    println!("{}", columns.join("\t"));
                }
            }
        }
    }
    if matched_rows == 0 {
        return Err("no generated schedule row matched the requested filters".to_string());
    }
    if !below_policy.is_empty() {
        return Err(format!(
            "{} generated schedule row(s) fell below their modeled SIS policy target:\n{}",
            below_policy.len(),
            below_policy.join("\n")
        ));
    }
    Ok(())
}
