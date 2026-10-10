//! Regenerate digit and element catalogs with the canonical artifact owners.
//!
//! Each family has scalar digit rows, two-group binary opening rows and
//! three-group prime opening rows, plus scalar element commitment rows.

use akita_config::{policy_of, CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_labinius_pcs::{
    family::{tables::DigitTables, Family128, Family64, FieldFamily},
    shipped::SUPPORTED_GEOMETRIES,
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
use std::{collections::BTreeMap, fs, path::Path, sync::Mutex, time::Instant};

// The grouped planner hook is a plain function pointer, so the scalar rows it
// adapts are registered here under their family name before rendering starts.
static SCALAR_ROWS: Mutex<BTreeMap<&'static str, ValidatedScheduleCatalog>> =
    Mutex::new(BTreeMap::new());

fn scalar<C: CommitmentConfig>(group: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
    Ok(akita_planner::find_schedule(
        &ScheduleLookupKey::single(group),
        C::committed_source_contract()?,
        &[],
        &policy_of::<C>(),
        C::ring_challenge_config,
    )?
    .schedule)
}

fn grouped<C: CommitmentConfig>(
    request: GroupedGenerationRequest,
) -> Result<FoldSchedule, AkitaError> {
    let missing = || AkitaError::InvalidSetup("scalar rows of the family are missing".into());
    let catalog = SCALAR_ROWS
        .lock()
        .map_err(|_| missing())?
        .get(C::schedule_family_name())
        .cloned()
        .ok_or_else(missing)?;
    let scalar_key = ScheduleLookupKey::single(request.key().final_group);
    Ok(akita_planner::find_adapted_schedule(
        catalog.resolve_key(&scalar_key)?,
        &request,
        C::committed_source_contract()?,
        &policy_of::<C>(),
        C::ring_challenge_config,
    )?
    .schedule)
}

/// Plan the scalar rows of `spec.keys` and admit them as a catalog of `C`.
fn preplan<C: CommitmentConfig>(spec: &mut EmitSpec) -> Result<ValidatedScheduleCatalog, String> {
    spec.keys.sort_by_key(|key| key.num_vars());
    spec.keys.dedup();
    spec.preplanned_scalar = bounded_parallel_filter_map(
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
    let rows = spec
        .preplanned_scalar
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
    let catalog = ValidatedScheduleCatalog::try_new(
        spec.family_name,
        rows,
        &spec.policy,
        spec.ring_challenge_config,
    )
    .map_err(|error| error.to_string())?;
    TrustedScheduleCatalog::<C>::new(catalog.clone()).map_err(|error| error.to_string())?;
    Ok(catalog)
}

/// Build and preplan the canonical scalar portion of a configuration's catalog.
fn scalar_spec<C: CommitmentConfig>(
    output_dir: &Path,
    keys: Vec<PolynomialGroupLayout>,
) -> Result<(EmitSpec, ValidatedScheduleCatalog), String> {
    let mut spec = EmitSpec {
        family_name: C::schedule_family_name(),
        policy: policy_of::<C>(),
        source_contract: C::committed_source_contract().map_err(|error| error.to_string())?,
        keys,
        grouped_requests: Vec::new(),
        preplanned_scalar: Vec::new(),
        output_dir: output_dir.to_path_buf(),
        regen: scalar::<C>,
        regen_group_batch: grouped::<C>,
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
        TrustedScheduleCatalog::<P::Digits>::new(scalar_rows.clone()).map_err(text)?;
    let trusted_elements =
        TrustedScheduleCatalog::<P::Elements>::new(element_rows).map_err(text)?;
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
    SCALAR_ROWS
        .lock()
        .map_err(|_| "scalar row registry is poisoned".to_string())?
        .insert(digits.family_name, scalar_rows);
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

/// Check the sample mixed-bound row before generating or using prime openings.
fn probe_three_group<P: FieldFamily>(artifact_root: &Path) -> Result<(), String> {
    let text = |error: AkitaError| error.to_string();
    let geometry = *SUPPORTED_GEOMETRIES
        .iter()
        .find(|geometry| geometry.log_num_cells == 22)
        .ok_or("sample geometry is not supported")?;
    // Probe against the binary catalog even before the prime artifacts exist.
    let tables = DigitTables::for_geometry::<P>(geometry).map_err(text)?;
    let directory = artifact_root.join(if akita_params::DEV_PROTOCOL {
        "schedules-labinius-dev"
    } else {
        "schedules-labinius"
    });
    let bytes = fs::read(directory.join(format!("{}.aks", P::Digits::schedule_family_name())))
        .map_err(|error| error.to_string())?;
    let digits = TrustedScheduleCatalog::<P::Digits>::from_artifact_bytes(&bytes).map_err(text)?;
    let policy = policy_of::<P::Digits>();
    let binary_key = tables.response_key(&digits, None).map_err(text)?;
    let binary_row = digits.resolve_key(&binary_key).map_err(text)?;
    let binary_bound =
        akita_schedules::expanded_schedule_proof_bound(&binary_key, binary_row.schedule(), &policy)
            .map_err(text)?;
    eprintln!(
        "{} (22, 8): binary proof bound={binary_bound}",
        P::Digits::schedule_family_name()
    );
    let element_log_len = tables.element_log_len;
    let element_group = tables.element_key().final_group;
    let started = Instant::now();
    let element_schedule = scalar::<P::Elements>(element_group).map_err(text)?;
    let element_profile =
        GroupCommitPhaseParams::try_from_params(element_group, &element_schedule.root.params)
            .map_err(text)?;
    let elements = ValidatedScheduleCatalog::try_new(
        P::Elements::schedule_family_name(),
        vec![(
            CommittedGroupBatchProfile {
                final_group: element_profile,
                precommitteds: Vec::new(),
            },
            element_schedule,
        )],
        &policy_of::<P::Elements>(),
        P::Elements::ring_challenge_config,
    )
    .map_err(text)?;
    TrustedScheduleCatalog::<P::Elements>::new(elements).map_err(text)?;
    eprintln!(
        "{} nv={element_log_len} scalar planned and admitted in {:.2?}",
        P::Elements::schedule_family_name(),
        started.elapsed()
    );
    let image_profile = digits
        .resolve_key(&tables.image_key())
        .map_err(text)?
        .profiles()
        .final_group;
    let request = GroupedGenerationRequest::new(
        binary_key.final_group,
        vec![
            PrecommittedProducer::try_new(
                image_profile,
                P::Digits::committed_source_contract().map_err(text)?,
            )
            .map_err(text)?,
            PrecommittedProducer::try_new(
                element_profile,
                P::Elements::committed_source_contract().map_err(text)?,
            )
            .map_err(text)?,
        ],
    );
    let started = Instant::now();
    let planned = akita_planner::find_adapted_schedule(
        digits
            .resolve_key(&tables.scalar_response_key())
            .map_err(text)?,
        &request,
        P::Digits::committed_source_contract().map_err(text)?,
        &policy,
        P::Digits::ring_challenge_config,
    )
    .map_err(text)?;
    let grouped_catalog = ValidatedScheduleCatalog::try_new(
        P::Digits::schedule_family_name(),
        vec![(
            CommittedGroupBatchProfile {
                final_group: GroupCommitPhaseParams::try_from_params(
                    request.key().final_group,
                    &planned.schedule.root.params,
                )
                .map_err(text)?,
                precommitteds: request.key().precommitteds,
            },
            planned.schedule.clone(),
        )],
        &policy,
        P::Digits::ring_challenge_config,
    )
    .map_err(text)?;
    TrustedScheduleCatalog::<P::Digits>::new(grouped_catalog).map_err(text)?;
    let prime_bound =
        akita_schedules::expanded_schedule_proof_bound(&request.key(), &planned.schedule, &policy)
            .map_err(text)?;
    eprintln!("{} (22, 8): three-group proof bound={prime_bound}, binary bound={binary_bound}, ratio={:.4}, planned in {:.2?}",
        P::Digits::schedule_family_name(), prime_bound as f64 / binary_bound as f64, started.elapsed());
    if prime_bound
        > akita_error::checked::product([2, binary_bound]).ok_or("proof bound overflow")?
    {
        return Err(format!(
            "three-group proof bound {prime_bound} exceeds twice binary bound {binary_bound}"
        ));
    }
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
        Some("--probe-three-group") if args.next().is_none() => {
            let outcomes = [
                probe_three_group::<Family64>(&tracked),
                probe_three_group::<Family128>(&tracked),
            ];
            let failures: Vec<_> = outcomes.into_iter().filter_map(Result::err).collect();
            if failures.is_empty() { Ok(()) } else { Err(failures.join("\n")) }
        }
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
            "usage: gen_labinius_schedule_artifacts [--output-dir <artifact-root> | --check | --probe-three-group]"
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
