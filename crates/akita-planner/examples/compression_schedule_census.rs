//! Census live B/D source-image sizes across stock generated/offline keys.

use akita_error::AkitaError;
use akita_planner::generated_families::{
    emitted_scalar_keys, GeneratedFamily, GenerationPreplans, ALL_GENERATED_FAMILIES,
};
use akita_types::{
    CompressionChainPlan, FoldSchedule, OpenCommitMatrixParams, OuterCommitMatrixParams,
    SisModulusProfileId,
};
use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;

#[derive(Default)]
struct Stats {
    occurrences: usize,
    coefficient_counts: BTreeSet<usize>,
    byte_counts: BTreeSet<usize>,
    max_bytes: usize,
    max_key: String,
    selected_dimensions: BTreeSet<String>,
    rejections: BTreeSet<String>,
}

struct SourceImageShape {
    output_rank: usize,
    ring_dimension: usize,
    logical_images: usize,
}

fn git_head() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|head| head.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn record(
    stats: &mut BTreeMap<(String, String), Stats>,
    role: &str,
    key: &str,
    profile: SisModulusProfileId,
    field_bytes: usize,
    shape: SourceImageShape,
) -> Result<(), String> {
    let coefficients = shape
        .output_rank
        .checked_mul(shape.ring_dimension)
        .and_then(|count| count.checked_mul(shape.logical_images))
        .ok_or_else(|| "source coefficient count overflow".to_string())?;
    let bytes = coefficients
        .checked_mul(field_bytes)
        .ok_or_else(|| "source byte count overflow".to_string())?;
    let entry = stats
        .entry((format!("{profile:?}"), role.to_string()))
        .or_default();
    entry.occurrences += 1;
    entry.coefficient_counts.insert(coefficients);
    entry.byte_counts.insert(bytes);
    if bytes > entry.max_bytes {
        entry.max_bytes = bytes;
        entry.max_key = key.to_string();
    }
    match CompressionChainPlan::for_complete_source(profile, coefficients) {
        Ok(plan) => {
            entry.selected_dimensions.insert(
                plan.maps()
                    .iter()
                    .map(|map| map.ring_dimension().to_string())
                    .collect::<Vec<_>>()
                    .join("/"),
            );
        }
        Err(error) => {
            entry.rejections.insert(format!("{bytes}:{error}"));
        }
    }
    Ok(())
}

fn record_outer(
    stats: &mut BTreeMap<(String, String), Stats>,
    role: &str,
    key: &str,
    profile: SisModulusProfileId,
    field_bytes: usize,
    matrix: &OuterCommitMatrixParams,
    slice_count: akita_types::CommitmentSliceCount,
) -> Result<(), String> {
    record(
        stats,
        role,
        key,
        profile,
        field_bytes,
        SourceImageShape {
            output_rank: matrix.output_rank(),
            ring_dimension: matrix.ring_dimension(),
            logical_images: slice_count.get(),
        },
    )
}

fn record_open(
    stats: &mut BTreeMap<(String, String), Stats>,
    role: &str,
    key: &str,
    profile: SisModulusProfileId,
    field_bytes: usize,
    matrix: &OpenCommitMatrixParams,
) -> Result<(), String> {
    record(
        stats,
        role,
        key,
        profile,
        field_bytes,
        SourceImageShape {
            output_rank: matrix.output_rank(),
            ring_dimension: matrix.ring_dimension(),
            logical_images: 1,
        },
    )
}

fn record_schedule(
    stats: &mut BTreeMap<(String, String), Stats>,
    family: &GeneratedFamily,
    key: &str,
    schedule: &FoldSchedule,
) -> Result<(), String> {
    let policy = (family.policy)();
    let profile = policy.sis_modulus_profile;
    let field_bytes = usize::try_from(policy.decomposition.field_bits())
        .map_err(|_| "field bit width conversion overflow".to_string())?
        .div_ceil(8);
    let root = &schedule.root.params;
    for (index, group) in root.precommitted_groups().iter().enumerate() {
        record_outer(
            stats,
            &format!("B-root-precommitted-{index}"),
            key,
            profile,
            field_bytes,
            &group.profile.outer.matrix,
            group.profile.outer_slice_count,
        )?;
    }
    record_outer(
        stats,
        "B-root-final",
        key,
        profile,
        field_bytes,
        &root.outer().matrix,
        root.outer_slice_count(),
    )?;
    record_open(
        stats,
        "D-root",
        key,
        profile,
        field_bytes,
        &root.open().matrix,
    )?;
    for (index, fold) in schedule.recursive_folds.iter().enumerate() {
        for (group_index, group) in fold.params.preceding_group_iter().enumerate() {
            record_outer(
                stats,
                &format!("B-recursive-{index}-precommitted-{group_index}"),
                key,
                profile,
                field_bytes,
                &group.profile.outer.matrix,
                group.profile.outer_slice_count,
            )?;
        }
        record_outer(
            stats,
            &format!("B-recursive-{index}"),
            key,
            profile,
            field_bytes,
            &fold.params.outer().matrix,
            fold.params.outer_slice_count(),
        )?;
        record_open(
            stats,
            &format!("D-recursive-{index}"),
            key,
            profile,
            field_bytes,
            &fold.params.open().matrix,
        )?;
    }
    Ok(())
}

fn compression_instances(schedule: &FoldSchedule) -> Result<(usize, usize), String> {
    let mut chains = schedule
        .root
        .params
        .precommitted_groups()
        .len()
        .checked_add(2)
        .ok_or_else(|| "root compression chain count overflow".to_string())?;
    for fold in &schedule.recursive_folds {
        chains = chains
            .checked_add(
                fold.params
                    .preceding_group_count()
                    .checked_add(2)
                    .ok_or_else(|| "recursive compression chain count overflow".to_string())?,
            )
            .ok_or_else(|| "proof compression chain count overflow".to_string())?;
    }
    let maps = chains
        .checked_mul(2)
        .ok_or_else(|| "proof compression map count overflow".to_string())?;
    Ok((chains, maps))
}

fn main() -> Result<(), String> {
    let mut stats = BTreeMap::new();
    let mut schedules = 0usize;
    let mut unsupported = 0usize;
    let mut max_chains_per_proof = 0usize;
    let mut max_maps_per_proof = 0usize;
    let mut max_instances_key = String::new();
    let preplans = GenerationPreplans::default();
    for family in ALL_GENERATED_FAMILIES {
        for key in emitted_scalar_keys(family)
            .map_err(|error| format!("{} scalar keys: {error}", family.family_name()))?
        {
            let label = format!(
                "{}:scalar:nv={}:polys={}",
                family.family_name(),
                key.num_vars(),
                key.num_polynomials()
            );
            let schedule = match (family.regen)(key) {
                Ok(schedule) => schedule,
                Err(AkitaError::UnsupportedSchedule(_)) => {
                    unsupported += 1;
                    continue;
                }
                Err(error) => {
                    return Err(format!(
                        "{} regenerate {label}: {error}",
                        family.family_name()
                    ));
                }
            };
            record_schedule(&mut stats, family, &label, &schedule)?;
            let (chains, maps) = compression_instances(&schedule)?;
            if maps > max_maps_per_proof {
                max_chains_per_proof = chains;
                max_maps_per_proof = maps;
                max_instances_key.clone_from(&label);
            }
            schedules += 1;
        }
        for request in (family.grouped_requests)(&preplans)
            .map_err(|error| format!("{} group keys: {error}", family.family_name()))?
        {
            let key = request.key();
            let label = format!("{}:group:{key:?}", family.family_name());
            let schedule = match (family.regen_group_batch)(request) {
                Ok(schedule) => schedule,
                Err(AkitaError::UnsupportedSchedule(_)) => {
                    unsupported += 1;
                    continue;
                }
                Err(error) => {
                    return Err(format!(
                        "{} regenerate group: {error}",
                        family.family_name()
                    ));
                }
            };
            record_schedule(&mut stats, family, &label, &schedule)?;
            let (chains, maps) = compression_instances(&schedule)?;
            if maps > max_maps_per_proof {
                max_chains_per_proof = chains;
                max_maps_per_proof = maps;
                max_instances_key.clone_from(&label);
            }
            schedules += 1;
        }
    }

    println!("head={}", git_head());
    println!("families={}", ALL_GENERATED_FAMILIES.len());
    println!("schedules={schedules}");
    println!("unsupported_candidates={unsupported}");
    println!("max_chains_per_proof={max_chains_per_proof}");
    println!("max_maps_per_proof={max_maps_per_proof}");
    println!("max_instances_key={max_instances_key:?}");
    println!(
        "profile,role,occurrences,coefficient_counts,byte_counts,max_bytes,max_key,ladders,rejections"
    );
    for ((profile, role), entry) in stats {
        let coefficients = entry
            .coefficient_counts
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("|");
        let bytes = entry
            .byte_counts
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("|");
        let ladders = entry
            .selected_dimensions
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("|");
        let rejections = entry
            .rejections
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("|");
        println!(
            "{profile},{role},{},{coefficients},{bytes},{},{:?},{ladders},{rejections}",
            entry.occurrences, entry.max_bytes, entry.max_key
        );
    }
    Ok(())
}
