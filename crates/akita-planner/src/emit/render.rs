//! Canonical external schedule-artifact rendering.

use super::*;
use akita_types::{CommittedGroupBatchProfile, GroupCommitPhaseParams};
use std::time::Instant;

/// One fully rendered artifact awaiting publication.
#[derive(Debug)]
pub struct ArtifactOutput {
    pub(super) destination: PathBuf,
    pub(super) body: String,
}

fn render_family_artifact(
    spec: &EmitSpec,
    materialized: Vec<MaterializedEntry>,
) -> Result<ArtifactOutput, String> {
    let rows = materialized
        .into_iter()
        .map(|entry| {
            let key = entry.key();
            let schedule = entry.schedule().clone();
            let final_group =
                GroupCommitPhaseParams::try_from_params(key.final_group, &schedule.root.params)
                    .map_err(|error| {
                        format!(
                            "{}: derive final committed profile: {error}",
                            spec.family_name
                        )
                    })?;
            Ok((
                CommittedGroupBatchProfile {
                    final_group,
                    precommitteds: key.precommitteds,
                },
                schedule,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let catalog = akita_schedules::TrustedScheduleCatalog::try_new(
        spec.family_name,
        rows,
        &spec.policy,
        spec.ring_challenge_config,
    )
    .map_err(|error| format!("{}: build artifact: {error}", spec.family_name))?;
    let bytes = catalog
        .to_artifact_bytes()
        .map_err(|error| format!("{}: encode artifact: {error}", spec.family_name))?;
    let body = String::from_utf8(bytes)
        .map_err(|error| format!("{}: artifact is not UTF-8: {error}", spec.family_name))?;
    Ok(ArtifactOutput {
        destination: spec.output_dir.join(format!("{}.aks", spec.family_name)),
        body,
    })
}

/// Materialize, validate, and render canonical external schedule artifacts.
pub fn render_schedule_artifact_outputs_with_validation(
    specs: &[EmitSpec],
    diagnostics: MaterializationDiagnostics,
    mut validate: impl FnMut(&EmitSpec, &[MaterializedEntry]) -> Result<(), String>,
) -> Result<Vec<ArtifactOutput>, String> {
    let materialization_started = diagnostics.row_progress.then(Instant::now);
    let materialized = materialized_entries_for_specs(specs, diagnostics)?;
    if let Some(started) = materialization_started {
        eprintln!(
            "schedule generation phase complete: materialized rows in {:.2?}",
            started.elapsed(),
        );
    }
    for (spec, entries) in specs.iter().zip(&materialized) {
        validate(spec, entries)?;
    }
    specs
        .iter()
        .zip(materialized)
        .map(|(spec, entries)| render_family_artifact(spec, entries))
        .collect()
}
