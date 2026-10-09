//! Regenerate the opt-in root PCS catalogs with the canonical artifact owners.

// The snapshot owner lives in the planner binary. Reuse it directly rather than
// maintain another TSV format, policy signature, or row-metrics implementation.
#[allow(dead_code, unexpected_cfgs)]
#[path = "../../../akita-planner/src/bin/gen_schedule_artifacts.rs"]
mod canonical_generator;

use akita_config::{policy_of, CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_labinius_pcs::{
    shipped::SUPPORTED_GEOMETRIES, DigitConfig, Digits1, Digits2, Digits4, ImageConfig,
    RootPcsSizing,
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
use std::{collections::HashMap, fs, path::Path, sync::OnceLock, time::Instant};

static SCALAR_CATALOGS: OnceLock<HashMap<&'static str, ValidatedScheduleCatalog>> = OnceLock::new();

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
    let catalogs = SCALAR_CATALOGS
        .get()
        .ok_or_else(|| AkitaError::InvalidSetup("scalar generation catalogs are missing".into()))?;
    let catalog = catalogs
        .get(C::schedule_family_name())
        .ok_or_else(|| AkitaError::InvalidSetup("scalar generation family is missing".into()))?;
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

fn spec<C: CommitmentConfig>(output_dir: &Path) -> Result<EmitSpec, AkitaError> {
    Ok(EmitSpec {
        family_name: C::schedule_family_name(),
        policy: policy_of::<C>(),
        source_contract: C::committed_source_contract()?,
        keys: Vec::new(),
        grouped_requests: Vec::new(),
        preplanned_scalar: Vec::new(),
        output_dir: output_dir.to_path_buf(),
        regen: scalar::<C>,
        regen_group_batch: grouped::<C>,
        ring_challenge_config: C::ring_challenge_config,
    })
}

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

fn digit_spec<C: DigitConfig>(
    output_dir: &Path,
    images: &TrustedScheduleCatalog<ImageConfig>,
) -> Result<EmitSpec, AkitaError> {
    let mut spec = spec::<C>(output_dir)?;
    for geometry in SUPPORTED_GEOMETRIES {
        if !geometry.bases.contains(&C::BASE) {
            continue;
        }
        let sizing = RootPcsSizing::new(
            geometry.profile,
            geometry.log_num_cells,
            geometry.log_fold_width,
            geometry.lambda_fold,
            C::BASE,
        )?;
        spec.keys.push(sizing.scalar_digit_key().final_group);
        let key = sizing.grouped_digit_key(images)?;
        let producers = key
            .precommitteds
            .into_iter()
            .map(|descriptor| {
                PrecommittedProducer::try_new(descriptor, ImageConfig::committed_source_contract()?)
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        spec.grouped_requests
            .push(GroupedGenerationRequest::new(key.final_group, producers));
    }
    spec.grouped_requests
        .sort_by_cached_key(|request| request.key().canonical_order_key());
    spec.grouped_requests.dedup();
    Ok(spec)
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
    let mut image_spec = spec::<ImageConfig>(&output_dir).map_err(|error| error.to_string())?;
    for geometry in SUPPORTED_GEOMETRIES {
        let base = geometry
            .bases
            .first()
            .copied()
            .ok_or_else(|| "supported geometry has no digit bases".to_string())?;
        let sizing = RootPcsSizing::new(
            geometry.profile,
            geometry.log_num_cells,
            geometry.log_fold_width,
            geometry.lambda_fold,
            base,
        )
        .map_err(|error| error.to_string())?;
        image_spec.keys.push(sizing.image_key().final_group);
    }
    let images =
        TrustedScheduleCatalog::<ImageConfig>::new(preplan::<ImageConfig>(&mut image_spec)?)
            .map_err(|error| error.to_string())?;
    let mut b1 = digit_spec::<Digits1>(&output_dir, &images).map_err(|error| error.to_string())?;
    let mut b2 = digit_spec::<Digits2>(&output_dir, &images).map_err(|error| error.to_string())?;
    let mut b4 = digit_spec::<Digits4>(&output_dir, &images).map_err(|error| error.to_string())?;
    let catalogs = HashMap::from([
        (
            Digits1::schedule_family_name(),
            preplan::<Digits1>(&mut b1)?,
        ),
        (
            Digits2::schedule_family_name(),
            preplan::<Digits2>(&mut b2)?,
        ),
        (
            Digits4::schedule_family_name(),
            preplan::<Digits4>(&mut b4)?,
        ),
    ]);
    SCALAR_CATALOGS
        .set(catalogs)
        .map_err(|_| "scalar generation catalogs were already initialized".to_string())?;
    let specs = [image_spec, b1, b2, b4];
    let mut rows = Vec::new();
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
            rows.extend(canonical_generator::materialized_snapshot_rows(
                spec, entries,
            )?);
            Ok(())
        },
    )?;
    let row_count = rows.len();
    let snapshot = canonical_generator::catalog_snapshot::write_snapshot(rows)?;
    publish_artifact_outputs(outputs)?;
    fs::write(
        artifact_root.join(format!("schedule-catalog-labinius{suffix}.tsv")),
        snapshot,
    )
    .map_err(|error| error.to_string())?;
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
            let scratch = std::env::temp_dir().join(format!(
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
