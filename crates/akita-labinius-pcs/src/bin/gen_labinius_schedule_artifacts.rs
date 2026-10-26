//! Regenerate digit and element catalogs with the canonical artifact owners.
//!
//! Each family has scalar digit rows, two-group binary opening rows and
//! three-group prime opening rows, plus scalar element commitment rows.

use akita_config::{policy_of, CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_labinius_pcs::{
    family::{tables::DigitTables, Family128, Family64, FieldFamily},
    shipped::{SUPPORTED_GEOMETRIES, WITNESS_CHUNKING},
};
use akita_params::{
    CommittedGroupBatchProfile, FoldSchedule, GroupCommitPhaseParams, PolynomialGroupLayout,
    ScheduleLookupKey,
};
use akita_planner::emit::{
    bounded_parallel_filter_map, offline_planning_worker_count, GroupedGenerationRequest,
    PrecommittedProducer,
};
use akita_planner::{
    publish_artifact_outputs, render_schedule_artifact_outputs_with_validation, EmitSpec,
    MaterializationDiagnostics,
};
use akita_schedules::ValidatedScheduleCatalog;
use std::{fs, path::Path, time::Instant};

fn scalar<C: CommitmentConfig>(group: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
    Ok(akita_planner::find_schedule(
        &ScheduleLookupKey::single(group),
        C::committed_source_contract()?,
        &[],
        &policy_of::<C>(WITNESS_CHUNKING),
        C::ring_challenge_config,
    )?
    .schedule)
}

/// Adapt a grouped request from the scalar row of its final group.
fn grouped<C: CommitmentConfig>(
    scalar_rows: &ValidatedScheduleCatalog,
    request: &GroupedGenerationRequest,
) -> Result<FoldSchedule, AkitaError> {
    let scalar_key = ScheduleLookupKey::single(request.key().final_group);
    Ok(akita_planner::find_adapted_schedule(
        scalar_rows.resolve_key(&scalar_key)?,
        request,
        C::committed_source_contract()?,
        &policy_of::<C>(WITNESS_CHUNKING),
        C::ring_challenge_config,
    )?
    .schedule)
}

/// Plan the scalar rows of `spec.keys` and admit them as a catalog of `C`.
fn preplan<C: CommitmentConfig>(spec: &mut EmitSpec) -> Result<ValidatedScheduleCatalog, String> {
    spec.keys.sort_by_key(|key| key.num_vars());
    spec.keys.dedup();
    let planned = bounded_parallel_filter_map(
        &spec.keys,
        offline_planning_worker_count(spec.keys.len()),
        |key| {
            let started = Instant::now();
            let schedule = scalar::<C>(*key).map_err(|error| error.to_string())?;
            eprintln!(
                "planned {} nv={} in {:.2?}",
                spec.family_name,
                key.num_vars(),
                started.elapsed()
            );
            Ok(Some((*key, schedule)))
        },
    )?;
    let rows = planned
        .iter()
        .map(|(key, schedule)| {
            Ok((
                CommittedGroupBatchProfile {
                    final_group: GroupCommitPhaseParams::try_from_params(
                        *key,
                        &schedule.root.params,
                    )?,
                    precommitteds: Vec::new(),
                },
                schedule.clone(),
            ))
        })
        .collect::<Result<Vec<_>, AkitaError>>()
        .map_err(|error| error.to_string())?;
    spec.preplanned = planned
        .into_iter()
        .map(|(key, schedule)| (ScheduleLookupKey::single(key), schedule))
        .collect();
    let catalog = ValidatedScheduleCatalog::try_new(
        spec.family_name,
        rows,
        &spec.policy,
        spec.ring_challenge_config,
    )
    .map_err(|error| error.to_string())?;
    TrustedScheduleCatalog::<C>::new(catalog.clone(), WITNESS_CHUNKING)
        .map_err(|error| error.to_string())?;
    Ok(catalog)
}

/// Build and preplan the canonical scalar portion of a configuration's catalog.
fn scalar_spec<C: CommitmentConfig>(
    output_dir: &Path,
    keys: Vec<PolynomialGroupLayout>,
) -> Result<(EmitSpec, ValidatedScheduleCatalog), String> {
    let mut spec = EmitSpec {
        family_name: C::schedule_family_name(),
        policy: policy_of::<C>(WITNESS_CHUNKING),
        source_contract: C::committed_source_contract().map_err(|error| error.to_string())?,
        keys,
        grouped_requests: Vec::new(),
        preplanned: Vec::new(),
        output_path: output_dir.join(format!("{}.aks", C::schedule_family_name())),
        ring_challenge_config: C::ring_challenge_config,
    };
    let rows = preplan::<C>(&mut spec)?;
    Ok((spec, rows))
}

/// One digit catalog and one element catalog for a family's supported geometries.
fn family_specs<P: FieldFamily>(output_dir: &Path) -> Result<[EmitSpec; 2], String> {
    let text = |error: AkitaError| error.to_string();
    let geometries = SUPPORTED_GEOMETRIES
        .iter()
        .map(|&geometry| DigitTables::for_geometry::<P>(geometry))
        .collect::<Result<Vec<_>, AkitaError>>()
        .map_err(text)?;
    let mut digit_keys = Vec::new();
    let mut element_keys = Vec::new();
    for tables in &geometries {
        digit_keys.extend([
            tables.image_key().final_group,
            tables.scalar_response_key().final_group,
        ]);
        element_keys.push(tables.element_key().final_group);
    }
    let (mut digits, scalar_rows) = scalar_spec::<P::Digits>(output_dir, digit_keys)?;
    let (elements, element_rows) = scalar_spec::<P::Elements>(output_dir, element_keys)?;
    let trusted_digits =
        TrustedScheduleCatalog::<P::Digits>::new(scalar_rows.clone(), WITNESS_CHUNKING)
            .map_err(text)?;
    let trusted_elements =
        TrustedScheduleCatalog::<P::Elements>::new(element_rows, WITNESS_CHUNKING).map_err(text)?;
    let digit_contract = P::Digits::committed_source_contract().map_err(text)?;
    let element_contract = P::Elements::committed_source_contract().map_err(text)?;
    for tables in &geometries {
        let image = trusted_digits
            .resolve_key(&tables.image_key())
            .map_err(text)?
            .profiles()
            .final_group;
        let element = trusted_elements
            .resolve_key(&tables.element_key())
            .map_err(text)?
            .profiles()
            .final_group;
        for element in [None, Some(element)] {
            let key = tables
                .response_key(&trusted_digits, element)
                .map_err(text)?;
            let mut producers =
                vec![PrecommittedProducer::try_new(image, digit_contract).map_err(text)?];
            if let Some(element) = element {
                producers
                    .push(PrecommittedProducer::try_new(element, element_contract).map_err(text)?);
            }
            digits
                .grouped_requests
                .push(GroupedGenerationRequest::new(key.final_group, producers));
        }
    }
    digits
        .grouped_requests
        .sort_by_cached_key(|request| request.key().canonical_order_key());
    digits.grouped_requests.dedup();
    let adapted = bounded_parallel_filter_map(
        &digits.grouped_requests,
        offline_planning_worker_count(digits.grouped_requests.len()),
        |request| {
            let schedule =
                grouped::<P::Digits>(&scalar_rows, request).map_err(|error| error.to_string())?;
            Ok(Some((request.key(), schedule)))
        },
    )?;
    digits.preplanned.extend(adapted);
    Ok([digits, elements])
}

fn generate(artifact_root: &Path) -> Result<(), String> {
    let started = Instant::now();
    let suffix = if akita_params::DEV_PROTOCOL {
        "-dev"
    } else {
        ""
    };
    let output_dir = artifact_root.join(format!("schedules-labinius{suffix}"));
    fs::create_dir_all(&output_dir).map_err(|error| error.to_string())?;
    let [digits64, elements64] = family_specs::<Family64>(&output_dir)?;
    let [digits128, elements128] = family_specs::<Family128>(&output_dir)?;
    let specs = [digits64, elements64, digits128, elements128];
    let mut row_count = 0;
    let outputs = render_schedule_artifact_outputs_with_validation(
        &specs,
        MaterializationDiagnostics { row_progress: true },
        |spec, entries| {
            let requested = spec.keys.len() + spec.grouped_requests.len();
            if entries.len() != requested {
                return Err(format!(
                    "{}: expected {requested} rows, planned {}",
                    spec.family_name,
                    entries.len()
                ));
            }
            row_count += entries.len();
            Ok(())
        },
    )?;
    publish_artifact_outputs(outputs)?;
    eprintln!("generated {row_count} rows in {:.2?}", started.elapsed());
    Ok(())
}

fn check_files(generated: &Path, tracked: &Path) -> Result<(), String> {
    for entry in fs::read_dir(generated).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let generated_path = entry.path();
        let tracked_path = tracked.join(entry.file_name());
        if generated_path.is_dir() {
            check_files(&generated_path, &tracked_path)?;
            let generated_count = fs::read_dir(&generated_path)
                .map_err(|error| error.to_string())?
                .count();
            let tracked_count = fs::read_dir(&tracked_path)
                .map_err(|error| error.to_string())?
                .count();
            if generated_count != tracked_count {
                return Err(format!(
                    "artifact file count differs: {}",
                    tracked_path.display()
                ));
            }
        } else if fs::read(&generated_path).map_err(|error| error.to_string())?
            != fs::read(&tracked_path).map_err(|error| error.to_string())?
        {
            return Err(format!("artifact drift: {}", tracked_path.display()));
        }
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let tracked = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts");
    match args.next().as_deref() {
        Some("--output-dir") => {
            let destination = args.next().ok_or("--output-dir requires a path")?;
            if args.next().is_some() {
                return Err("unexpected argument".into());
            }
            generate(Path::new(&destination))
        }
        Some("--check") if args.next().is_none() => {
            let scratch_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target");
            fs::create_dir_all(&scratch_root).map_err(|error| error.to_string())?;
            let scratch = scratch_root.join(format!(
                "akita-labinius-schedule-check-{}",
                std::process::id()
            ));
            fs::create_dir(&scratch).map_err(|error| error.to_string())?;
            let result = generate(&scratch).and_then(|()| check_files(&scratch, &tracked));
            fs::remove_dir_all(&scratch).map_err(|error| error.to_string())?;
            result
        }
        None => generate(&tracked),
        _ => Err(
            "usage: gen_labinius_schedule_artifacts [--output-dir <artifact-root> | --check]"
                .into(),
        ),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
