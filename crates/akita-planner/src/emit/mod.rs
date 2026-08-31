//! Reusable schedule-table emitter for `akita-schedules` and downstream catalogs.
//!
//! The `akita-config` `gen_schedule_tables` binary adapts preset metadata into
//! [`EmitSpec`] values and calls this module. Jolt can invoke the same API with
//! an explicit [`PlannerPolicy`] and hook function pointers.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::sis::{CommittedSourceContract, HonestFoldPolicySpec};
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupParams, FoldSchedule, GroupCommitPhaseParams,
    GroupOpenPhaseParams, OpenCommitMatrixParams, PolynomialGroupLayout,
};

use crate::PlannerPolicy;
mod materialize;
mod publish;
mod render;
mod source_annotations;
use akita_schedules::expected_catalog_identity;
use akita_schedules::generated::{
    GeneratedFoldCore, GeneratedFoldScheduleEntry, GeneratedFrozenGroup, GeneratedGroup,
    GeneratedMatrix, GeneratedPrecommittedGroup, GeneratedRecursiveFold, GeneratedRootFold,
    GeneratedScheduleCatalogIdentity, GeneratedSetupPrefix, GeneratedTerminalFold,
};
pub(super) use materialize::materialized_entries_for_specs;
pub use materialize::MaterializedEntry;
pub use publish::publish_generated_outputs;
pub use render::{
    render_generated_outputs, render_generated_outputs_with_validation, GeneratedOutput,
};
use source_annotations::{emit_bounded_source_banner, precommitted_source_note};

/// Optional observability for the offline row-materialization queue.
///
/// Disabled generation performs no per-row timing or search-counter updates.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaterializationDiagnostics {
    pub row_progress: bool,
}

/// One frozen precommit descriptor with its canonical producer facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrecommittedProducer {
    descriptor: GroupCommitPhaseParams,
    contract: CommittedSourceContract,
    fold_policy: HonestFoldPolicySpec,
}

impl PrecommittedProducer {
    /// Capture the producer contract and its offline sizing projection together.
    #[cfg(feature = "catalog-gen")]
    pub fn from_config<Cfg: akita_config::CommitmentConfig>(
        descriptor: GroupCommitPhaseParams,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            descriptor,
            contract: Cfg::committed_source_contract()?,
            fold_policy: akita_config::honest_fold_policy_of::<Cfg>(),
        })
    }

    const fn descriptor(self) -> GroupCommitPhaseParams {
        self.descriptor
    }

    const fn contract(self) -> CommittedSourceContract {
        self.contract
    }

    /// Producer declaration used to disambiguate otherwise identical grouped
    /// catalog layouts in revision evidence.
    #[must_use]
    pub const fn source_contract(self) -> CommittedSourceContract {
        self.contract
    }

    #[cfg(feature = "catalog-gen")]
    const fn fold_policy(self) -> HonestFoldPolicySpec {
        self.fold_policy
    }
}

/// One grouped generation request whose lookup key is derived from its producers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupedGenerationRequest {
    final_group: PolynomialGroupLayout,
    precommitted_producers: Vec<PrecommittedProducer>,
}

impl GroupedGenerationRequest {
    #[must_use]
    pub fn new(
        final_group: PolynomialGroupLayout,
        precommitted_producers: Vec<PrecommittedProducer>,
    ) -> Self {
        Self {
            final_group,
            precommitted_producers,
        }
    }

    #[must_use]
    pub fn key(&self) -> AkitaScheduleLookupKey {
        AkitaScheduleLookupKey {
            final_group: self.final_group,
            precommitteds: self
                .precommitted_producers
                .iter()
                .copied()
                .map(PrecommittedProducer::descriptor)
                .collect(),
        }
    }

    fn precommitted_producers(&self) -> &[PrecommittedProducer] {
        &self.precommitted_producers
    }

    #[cfg(feature = "catalog-gen")]
    pub(crate) fn fold_policies(&self) -> Vec<HonestFoldPolicySpec> {
        self.precommitted_producers
            .iter()
            .copied()
            .map(PrecommittedProducer::fold_policy)
            .collect()
    }
}

/// One family the emitter writes to `akita-schedules/src/generated/`.
#[derive(Clone)]
pub struct EmitSpec {
    pub module_name: &'static str,
    pub const_name: &'static str,
    pub family_name: &'static str,
    pub schedule_feature: &'static str,
    pub policy: PlannerPolicy,
    pub source_contract: CommittedSourceContract,
    pub keys: Vec<PolynomialGroupLayout>,
    pub grouped_requests: Vec<GroupedGenerationRequest>,
    /// Exact successful scalar results already needed to construct grouped keys.
    pub preplanned_scalar: Vec<(PolynomialGroupLayout, FoldSchedule)>,
    pub output_dir: PathBuf,
    pub regen: fn(PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError>,
    pub regen_group_batch: fn(GroupedGenerationRequest) -> Result<FoldSchedule, AkitaError>,
    pub ring_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    pub generator_command: &'static str,
}

const MOD_WIRING_BEGIN: &str = "// @generated schedule module wiring begin";
const MOD_WIRING_END: &str = "// @generated schedule module wiring end";
// Schedule search is memory bound. Keep the default below host-wide
// parallelism while allowing explicit tuning for large generation machines.
// Each row has a bounded exact-suffix cache. Three concurrent rows fit the
// normal generation envelope; constrained jobs can still override this with
// `AKITA_SCHEDULE_GEN_JOBS`.
const DEFAULT_OFFLINE_PLANNING_WORKERS: usize = 3;

/// Bound memory-heavy offline planner searches for generation and drift checks.
pub fn offline_planning_worker_count(work_items: usize) -> usize {
    let configured = std::env::var("AKITA_SCHEDULE_GEN_JOBS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_OFFLINE_PLANNING_WORKERS);
    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(configured)
        .min(work_items.max(1))
}

/// Map independent offline requests with a fixed worker count and input order.
pub fn bounded_parallel_filter_map<T, R>(
    items: &[T],
    workers: usize,
    map: impl Fn(&T) -> Result<Option<R>, String> + Sync,
) -> Result<Vec<R>, String>
where
    T: Sync,
    R: Send,
{
    // A private scoped pool gives this memory-heavy phase an explicit bound;
    // the workspace Rayon pool follows host-wide parallelism instead.
    if workers <= 1 {
        return items
            .iter()
            .filter_map(|item| map(item).transpose())
            .collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut mapped = std::thread::scope(|scope| -> Result<Vec<(usize, R)>, String> {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let map = &map;
                let next = &next;
                scope.spawn(move || {
                    let mut local = Vec::new();
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(item) = items.get(index) else {
                            break;
                        };
                        if let Some(value) = map(item)? {
                            local.push((index, value));
                        }
                    }
                    Ok::<_, String>(local)
                })
            })
            .collect();
        let mut output = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(Ok(local)) => output.extend(local),
                Ok(Err(error)) => return Err(error),
                Err(_) => return Err("schedule generation worker panicked".into()),
            }
        }
        Ok(output)
    })?;
    mapped.sort_by_key(|(index, _)| *index);
    Ok(mapped.into_iter().map(|(_, value)| value).collect())
}

fn geometry(p: &CommittedGroupParams) -> akita_types::BlockGeometry {
    p.blocks()
}

fn committed_group(p: &CommittedGroupParams) -> GeneratedGroup {
    GeneratedGroup {
        geometry: geometry(p),
        inner_commit_matrix: GeneratedMatrix {
            ring_dimension: p.inner().matrix.ring_dimension() as u32,
            log_basis: p.inner().digits.log_basis,
        },
        outer_commit_matrix: GeneratedMatrix {
            ring_dimension: p.outer().matrix.ring_dimension() as u32,
            log_basis: p.outer().digits.log_basis,
        },
        outer_slice_count: p.outer_slice_count().get() as u32,
        num_digits_fold: p.num_digits_fold() as u32,
        opening_method: p.opening_method(),
    }
}

fn open_matrix_params(p: &OpenCommitMatrixParams, log_basis: u32) -> GeneratedMatrix {
    GeneratedMatrix {
        ring_dimension: p.ring_dimension() as u32,
        log_basis,
    }
}

/// Freeze the fields shared by precommitted and setup-prefix groups.
fn frozen_group(slot: &GroupOpenPhaseParams) -> GeneratedFrozenGroup {
    GeneratedFrozenGroup {
        profile: slot.profile,
        // The rest of the opening plan is the consuming fold's, so it is derived
        // at expansion rather than stored here.
        opening_method: slot.opening.opening_method,
        num_digits_fold: slot.opening.num_digits_fold as u32,
    }
}

fn setup_prefix(slot: &GroupOpenPhaseParams) -> Result<GeneratedSetupPrefix, String> {
    let natural_len = slot
        .setup_natural_len
        .ok_or_else(|| "generated setup prefix must carry a natural length".to_string())?;
    Ok(GeneratedSetupPrefix {
        group: frozen_group(slot),
        natural_len: natural_len as u64,
    })
}

/// Response cap of whichever security route this group's A matrix took.
fn response_l2_sq_cap(p: &CommittedGroupParams) -> Option<u128> {
    match p.inner().matrix.security_route() {
        akita_types::InnerCommitSecurityRoute::Linf(_) => None,
        akita_types::InnerCommitSecurityRoute::L2 {
            response_l2_sq_cap, ..
        } => Some(response_l2_sq_cap),
    }
}

fn generated_fold_core(p: &CommittedGroupParams) -> GeneratedFoldCore {
    GeneratedFoldCore {
        group: committed_group(p),
        open_commit_matrix: open_matrix_params(&p.open().matrix, p.open().digits.log_basis),
        witness_chunks: p.witness_chunk.num_chunks as u32,
        ring_relation_mode: p.ring_relation_mode,
    }
}

fn generated_recursive_fold(p: &CommittedGroupParams) -> Result<GeneratedRecursiveFold, String> {
    if !p.precommitted_groups().is_empty() {
        return Err("generated recursive fold cannot carry precommitted groups".to_string());
    }
    Ok(GeneratedRecursiveFold {
        core: generated_fold_core(p),
        setup_prefix: p.setup_prefix().map(setup_prefix).transpose()?,
        payload_mode: p.payload_mode,
        response_l2_sq_cap: response_l2_sq_cap(p),
    })
}

fn generated_entry(
    key: &AkitaScheduleLookupKey,
    schedule: &FoldSchedule,
) -> Result<GeneratedFoldScheduleEntry, String> {
    let root_fold = &schedule.root.params;
    let root_params = &root_fold;
    let precommitted_groups = key
        .precommitteds
        .iter()
        .copied()
        .zip(root_fold.precommitted_groups())
        .map(|(profile, group)| GeneratedPrecommittedGroup {
            group: GeneratedFrozenGroup {
                profile,
                num_digits_fold: group.opening.num_digits_fold as u32,
                opening_method: group.opening.opening_method,
            },
        })
        .collect::<Vec<_>>();
    let recursive_folds = schedule
        .recursive_folds
        .iter()
        // A recursive fold pins no inner digit depth: expansion derives it from
        // the witness length arriving from the level above.
        .map(|step| generated_recursive_fold(&step.params))
        .collect::<Result<Vec<_>, _>>()?;
    let terminal_group = schedule
        .terminal
        .response_shape
        .layout
        .groups
        .first()
        .ok_or_else(|| "terminal response shape has no group".to_string())?;
    if schedule.terminal.response_shape.layout.groups.len() != 1 {
        return Err("generated scalar terminal response must have exactly one group".to_string());
    }
    if root_params.setup_prefix().is_some() {
        return Err("generated root fold cannot carry a setup prefix".to_string());
    }
    if root_params.payload_mode != akita_types::CommitmentPayloadMode::Compressed
        || response_l2_sq_cap(root_params).is_some()
    {
        return Err("generated root fold must use the canonical compressed Linf route".to_string());
    }
    Ok(GeneratedFoldScheduleEntry {
        final_group: key.final_group,
        root: GeneratedRootFold {
            core: generated_fold_core(root_params),
            num_digits_inner: root_params.inner().digits.num_digits as u32,
            precommitted_groups: Box::leak(precommitted_groups.into_boxed_slice()),
        },
        recursive_folds: Box::leak(recursive_folds.into_boxed_slice()),
        terminal: GeneratedTerminalFold {
            geometry: schedule.terminal.blocks,
            inner_commit_matrix: GeneratedMatrix {
                ring_dimension: schedule.terminal.inner.matrix.ring_dimension() as u32,
                log_basis: schedule.terminal.inner.digits.log_basis,
            },
            num_digits_inner: schedule.terminal.inner.digits.num_digits as u32,
            fold_log_basis: schedule.terminal.fold.log_basis,
            fold_digit_count: schedule.terminal.fold.num_digits as u32,
            inner_output_rank: schedule.terminal.inner.matrix.output_rank() as u32,
            inner_coeff_linf_bound: schedule
                .terminal
                .inner
                .matrix
                .coeff_linf_bound()
                .unwrap_or(0),
            response_l2_sq_cap: schedule.terminal.response_l2_sq_cap(),
            z_linf_cap: terminal_group.z_linf_cap,
            z_rice_low_bits: terminal_group.z_rice_low_bits,
            z_payload_bytes: terminal_group.z_payload_bytes as u64,
        },
    })
}

fn emit_key(key: PolynomialGroupLayout) -> String {
    format!(
        "PolynomialGroupLayout::new({}, {})",
        key.num_vars(),
        key.num_polynomials(),
    )
}

/// Emit one nested commit-phase profile literal.
///
/// The nesting mirrors the struct: geometry, slicing, then a `(digits, matrix)`
/// role twice. `BlockGeometry::new`, `GadgetDigits::new`, and `RoleParams::new`
/// are all `const`, so the result stays valid in `static` position.
fn emit_precommitted_group_key(layout: &GroupCommitPhaseParams) -> String {
    format!(
        "GroupCommitPhaseParams {{ version: GroupCommitPhaseParams::VERSION, group: {}, blocks: {}, outer_slice_count: akita_types::CommitmentSliceCount::{}, inner: {}, outer: {} }}",
        emit_key(layout.group),
        emit_geometry(layout.blocks),
        match layout.outer_slice_count {
            akita_types::CommitmentSliceCount::ONE => "ONE",
            akita_types::CommitmentSliceCount::TWO => "TWO",
            akita_types::CommitmentSliceCount::FOUR => "FOUR",
            akita_types::CommitmentSliceCount::EIGHT => "EIGHT",
            _ => unreachable!("checked commitment slice count"),
        },
        emit_role_params(
            layout.inner.digits,
            &emit_profile_matrix(
                "InnerCommitMatrixParams",
                layout.inner.matrix.output_rank(),
                layout.inner.matrix.input_width(),
                layout
                    .inner
                    .matrix
                    .sis_table_key()
                    .expect("validated precommitted matrix is L infinity"),
            ),
        ),
        emit_role_params(
            layout.outer.digits,
            &emit_profile_matrix(
                "OuterCommitMatrixParams",
                layout.outer.matrix.output_rank(),
                layout.outer.matrix.input_width(),
                layout.outer.matrix.sis_table_key(),
            ),
        ),
    )
}

fn emit_role_params(digits: akita_types::GadgetDigits, matrix: &str) -> String {
    format!(
        "akita_types::RoleParams::new(akita_types::GadgetDigits::new({}, {}), {})",
        digits.log_basis, digits.num_digits, matrix,
    )
}

fn emit_profile_matrix(
    type_name: &str,
    output_rank: usize,
    input_width: usize,
    key: akita_types::SisTableKey,
) -> String {
    format!(
        "{type_name}::new_unchecked(SisSecurityPolicyId::{:?}, SisTableDigest({:?}), SisModulusProfileId::{:?}, {}, {}, {}, {})",
        key.policy,
        key.table_digest.0,
        key.modulus_profile,
        output_rank,
        input_width,
        key.coeff_linf_bound,
        key.ring_dimension,
    )
}

fn emit_geometry(value: akita_types::BlockGeometry) -> String {
    format!(
        "BlockGeometry::new({}, {}, {})",
        value.live_ring_elements_per_claim, value.positions_per_block, value.live_blocks
    )
}

fn emit_group(value: GeneratedGroup) -> String {
    format!(
        "GeneratedGroup {{ geometry: {}, inner_commit_matrix: GeneratedMatrix {{ ring_dimension: {}, log_basis: {} }}, outer_commit_matrix: GeneratedMatrix {{ ring_dimension: {}, log_basis: {} }}, outer_slice_count: {}, num_digits_fold: {}, opening_method: {} }}",
        emit_geometry(value.geometry),
        value.inner_commit_matrix.ring_dimension,
        value.inner_commit_matrix.log_basis,
        value.outer_commit_matrix.ring_dimension,
        value.outer_commit_matrix.log_basis,
        value.outer_slice_count,
        value.num_digits_fold,
        emit_opening_method(value.opening_method),
    )
}

/// Render `Option<T>` for any `T` that already prints as a Rust literal.
fn emit_optional<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "None".to_string(), |value| format!("Some({value})"))
}

fn emit_open_matrix(value: GeneratedMatrix) -> String {
    format!(
        "GeneratedMatrix {{ ring_dimension: {}, log_basis: {} }}",
        value.ring_dimension, value.log_basis
    )
}

fn emit_payload_mode(value: akita_types::CommitmentPayloadMode) -> &'static str {
    match value {
        akita_types::CommitmentPayloadMode::Compressed => "CommitmentPayloadMode::Compressed",
        akita_types::CommitmentPayloadMode::Raw => "CommitmentPayloadMode::Raw",
    }
}

fn emit_opening_method(value: akita_types::OpeningMethod) -> String {
    match value {
        akita_types::OpeningMethod::EvaluationTrace => {
            "akita_types::OpeningMethod::EvaluationTrace".to_string()
        }
        akita_types::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } => format!(
            "akita_types::OpeningMethod::SubringCoefficientPacking {{ challenge_subring_dimension: {challenge_subring_dimension} }}"
        ),
    }
}

fn emit_frozen_group(value: &GeneratedFrozenGroup) -> String {
    format!(
        "GeneratedFrozenGroup {{ profile: {}, opening_method: {}, num_digits_fold: {} }}",
        emit_precommitted_group_key(&value.profile),
        emit_opening_method(value.opening_method),
        value.num_digits_fold,
    )
}

fn emit_fold_core(fold: GeneratedFoldCore) -> String {
    format!(
        "GeneratedFoldCore {{ group: {}, open_commit_matrix: {}, witness_chunks: {}, ring_relation_mode: {} }}",
        emit_group(fold.group),
        emit_open_matrix(fold.open_commit_matrix),
        fold.witness_chunks,
        emit_ring_relation_mode(fold.ring_relation_mode),
    )
}

fn emit_ring_relation_mode(mode: akita_types::RingRelationMode) -> &'static str {
    match mode {
        akita_types::RingRelationMode::QuotientLift => {
            "akita_types::RingRelationMode::QuotientLift"
        }
        akita_types::RingRelationMode::ReducedEvaluation => {
            "akita_types::RingRelationMode::ReducedEvaluation"
        }
    }
}

fn emit_setup_prefix(prefix: &GeneratedSetupPrefix) -> String {
    format!(
        "GeneratedSetupPrefix {{ group: {}, natural_len: {} }}",
        emit_frozen_group(&prefix.group),
        prefix.natural_len,
    )
}

fn emit_recursive_fold(fold: &GeneratedRecursiveFold) -> String {
    format!(
        "GeneratedRecursiveFold {{ core: {}, setup_prefix: {}, payload_mode: {}, response_l2_sq_cap: {} }}",
        emit_fold_core(fold.core),
        fold.setup_prefix.as_ref().map_or_else(
            || "None".to_string(),
            |prefix| format!("Some({})", emit_setup_prefix(prefix)),
        ),
        emit_payload_mode(fold.payload_mode),
        emit_optional(fold.response_l2_sq_cap),
    )
}

fn emit_root_fold(
    out: &mut String,
    indent: &str,
    label: &str,
    fold: &GeneratedRootFold,
    key: &AkitaScheduleLookupKey,
    precommitted_producers: &[PrecommittedProducer],
) -> Result<(), String> {
    if fold.precommitted_groups.is_empty() {
        writeln!(
            out,
            "{indent}{label}GeneratedRootFold {{ core: {}, num_digits_inner: {}, precommitted_groups: &[] }},",
            emit_fold_core(fold.core),
            fold.num_digits_inner,
        )
        .map_err(|e| e.to_string())?;
        return Ok(());
    }
    if precommitted_producers.len() != fold.precommitted_groups.len() {
        return Err(format!(
            "grouped key {key:?} carries {} producer records for {} precommitted groups",
            precommitted_producers.len(),
            fold.precommitted_groups.len(),
        ));
    }
    writeln!(out, "{indent}{label}GeneratedRootFold {{").map_err(|e| e.to_string())?;
    writeln!(out, "{indent}    core: {},", emit_fold_core(fold.core)).map_err(|e| e.to_string())?;
    writeln!(
        out,
        "{indent}    num_digits_inner: {},",
        fold.num_digits_inner
    )
    .map_err(|e| e.to_string())?;
    writeln!(out, "{indent}    precommitted_groups: &[").map_err(|e| e.to_string())?;
    for (index, group) in fold.precommitted_groups.iter().enumerate() {
        out.push_str(&precommitted_source_note(
            group.group.profile.inner.digits.log_basis,
            group.group.profile.inner.digits.num_digits,
            precommitted_producers[index].contract().class(),
        ));
        writeln!(
            out,
            "{indent}        GeneratedPrecommittedGroup {{ group: {} }},",
            emit_frozen_group(&group.group),
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(out, "{indent}    ],").map_err(|e| e.to_string())?;
    writeln!(out, "{indent}}},").map_err(|e| e.to_string())?;
    Ok(())
}

fn emit_schedule_entry(
    out: &mut String,
    key: &AkitaScheduleLookupKey,
    schedule: &FoldSchedule,
    precommitted_producers: &[PrecommittedProducer],
) -> Result<(), String> {
    let entry = generated_entry(key, schedule)?;
    writeln!(out, "    GeneratedFoldScheduleEntry {{").map_err(|e| e.to_string())?;
    writeln!(out, "        final_group: {},", emit_key(entry.final_group))
        .map_err(|e| e.to_string())?;
    emit_root_fold(
        out,
        "        ",
        "root: ",
        &entry.root,
        key,
        precommitted_producers,
    )?;
    if entry.recursive_folds.is_empty() {
        writeln!(out, "        recursive_folds: &[],").map_err(|e| e.to_string())?;
    } else {
        writeln!(out, "        recursive_folds: &[").map_err(|e| e.to_string())?;
        for fold in entry.recursive_folds {
            writeln!(out, "            {},", emit_recursive_fold(fold))
                .map_err(|e| e.to_string())?;
        }
        writeln!(out, "        ],").map_err(|e| e.to_string())?;
    }
    writeln!(
        out,
        "        terminal: GeneratedTerminalFold {{ geometry: {}, inner_commit_matrix: GeneratedMatrix {{ ring_dimension: {}, log_basis: {} }}, num_digits_inner: {}, fold_log_basis: {}, fold_digit_count: {}, inner_output_rank: {}, inner_coeff_linf_bound: {}, response_l2_sq_cap: {}, z_linf_cap: {}, z_rice_low_bits: {}, z_payload_bytes: {} }},",
        emit_geometry(entry.terminal.geometry),
        entry.terminal.inner_commit_matrix.ring_dimension,
        entry.terminal.inner_commit_matrix.log_basis,
        entry.terminal.num_digits_inner,
        entry.terminal.fold_log_basis,
        entry.terminal.fold_digit_count,
        entry.terminal.inner_output_rank,
        entry.terminal.inner_coeff_linf_bound,
        entry.terminal.response_l2_sq_cap.map_or_else(
            || "None".to_string(),
            |cap| format!("Some({cap})"),
        ),
        entry.terminal.z_linf_cap.map_or_else(
            || "None".to_string(),
            |cap| format!("Some({cap})"),
        ),
        entry.terminal.z_rice_low_bits,
        entry.terminal.z_payload_bytes,
    )
    .map_err(|e| e.to_string())?;
    writeln!(out, "    }},").map_err(|e| e.to_string())
}

fn emit_decomposition(d: akita_types::DecompositionParams) -> String {
    match d.log_open_bound {
        Some(v) => format!(
            "DecompositionParams {{ log_basis: {}, log_commit_bound: {}, log_open_bound: Some({}) }}",
            d.log_basis, d.log_commit_bound, v
        ),
        None => format!(
            "DecompositionParams {{ log_basis: {}, log_commit_bound: {}, log_open_bound: None }}",
            d.log_basis, d.log_commit_bound
        ),
    }
}

fn emit_sis_modulus_profile(family: akita_types::SisModulusProfileId) -> &'static str {
    match family {
        akita_types::SisModulusProfileId::Q32Offset99 => "SisModulusProfileId::Q32Offset99",
        akita_types::SisModulusProfileId::Q64Offset59 => "SisModulusProfileId::Q64Offset59",
        akita_types::SisModulusProfileId::Q128OffsetA7F7 => "SisModulusProfileId::Q128OffsetA7F7",
    }
}

fn format_bytes(bytes: [u8; 32]) -> String {
    let values = bytes.iter().map(|byte| format!("0x{byte:02x}"));
    format!("[{}]", values.collect::<Vec<_>>().join(", "))
}

fn emit_witness_chunk(cfg: akita_types::ChunkedWitnessCfg) -> String {
    format!(
        "ChunkedWitnessCfg {{ num_chunks: {}, num_activated_levels: {} }}",
        cfg.num_chunks, cfg.num_activated_levels
    )
}

fn emit_identity_const(identity: &GeneratedScheduleCatalogIdentity) -> String {
    let (ring_dimension_policy_statics, ring_dimension_schedule_mode) =
        match identity.ring_dimension_schedule_mode {
            akita_schedules::RingDimensionScheduleMode::UniformDimension { ring_dimension } => (
                String::new(),
                format!("RingDimensionScheduleMode::UniformDimension {{ ring_dimension: {ring_dimension} }}"),
            ),
            akita_schedules::RingDimensionScheduleMode::AdaptiveDimension {
                num_search_levels,
                suffix_dimensions,
                potential_a_dimensions,
                potential_b_dimensions,
                potential_d_dimensions,
            } => {
                let format_dimensions = |dimensions: &[usize]| dimensions.iter().map(usize::to_string).collect::<Vec<_>>().join(", ");
                (
                    format!(
                        concat!(
                            "#[rustfmt::skip]\n",
                            "pub(crate) static CATALOG_SUFFIX_DIMENSIONS: &[usize] = &[{}];\n",
                            "#[rustfmt::skip]\n",
                            "pub(crate) static CATALOG_POTENTIAL_A_DIMENSIONS: &[usize] = &[{}];\n",
                            "#[rustfmt::skip]\n",
                            "pub(crate) static CATALOG_POTENTIAL_B_DIMENSIONS: &[usize] = &[{}];\n",
                            "#[rustfmt::skip]\n",
                            "pub(crate) static CATALOG_POTENTIAL_D_DIMENSIONS: &[usize] = &[{}];\n",
                        ),
                        format_dimensions(suffix_dimensions),
                        format_dimensions(potential_a_dimensions),
                        format_dimensions(potential_b_dimensions),
                        format_dimensions(potential_d_dimensions),
                    ),
                    format!(
                        "RingDimensionScheduleMode::AdaptiveDimension {{ num_search_levels: {num_search_levels}, suffix_dimensions: CATALOG_SUFFIX_DIMENSIONS, potential_a_dimensions: CATALOG_POTENTIAL_A_DIMENSIONS, potential_b_dimensions: CATALOG_POTENTIAL_B_DIMENSIONS, potential_d_dimensions: CATALOG_POTENTIAL_D_DIMENSIONS }}"
                    ),
                )
            }
        };
    let ring_dims: String = identity
        .ring_dimensions
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        concat!(
            "{ring_dimension_policy_statics}",
            "#[rustfmt::skip]\n",
            "pub(crate) static CATALOG_RING_DIMENSIONS: &[usize] = &[{ring_dims}];\n",
            "#[rustfmt::skip]\n",
            "pub(crate) static CATALOG_IDENTITY: GeneratedScheduleCatalogIdentity = ",
            "GeneratedScheduleCatalogIdentity {{\n",
            "    family_name: \"{family_name}\",\n",
            "    protocol_epoch: {protocol_epoch},\n",
            "    cost_model: PlannerCostModelId::{cost_model},\n",
            "    selective_l2_response_model: SelectiveL2ResponseModelId::{selective_l2_response_model},\n",
            "    selection_policy: SelectionPolicyId::{selection_policy},\n",
            "    recursive_split_search_policy: crate::RecursiveSplitSearchPolicy::{recursive_split_search_policy},\n",
            "    recursive_setup_search_policy: crate::RecursiveSetupSearchPolicy::{recursive_setup_search_policy},\n",
            "    setup_field_budget: {setup_field_budget},\n",
            "    min_offloaded_witness_contraction: {min_offloaded_witness_contraction},\n",
            "    sis_modulus_profile: {sis_modulus_profile},\n",
            "    sis_security_policy: SisSecurityPolicyId::{sis_security_policy},\n",
            "    sis_table_digest: SisTableDigest({sis_table_digest}),\n",
            "    sis_l2_table_digest: SisL2TableDigest({sis_l2_table_digest}),\n",
            "    decomposition: {decomposition},\n",
            "    claim_ext_degree: {claim_ext_degree},\n",
            "    chal_ext_degree: {chal_ext_degree},\n",
            "    inner_basis_range: ({inner_basis_min}, {inner_basis_max}),\n",
            "    opening_basis_range: ({basis_min}, {basis_max}),\n",
            "    witness_chunk: {witness_chunk},\n",
            "    recursive_setup_planning: {recursive_setup_planning},\n",
            "    ring_dimension_schedule_mode: {ring_dimension_schedule_mode},\n",
            "    ring_dimensions: CATALOG_RING_DIMENSIONS,\n",
            "    ring_challenge_config_digest: {ring_challenge_config_digest},\n",
            "    key_count: {key_count},\n",
            "    key_digest: {key_digest},\n",
            "}};\n",
        ),
        ring_dimension_policy_statics = ring_dimension_policy_statics,
        ring_dimension_schedule_mode = ring_dimension_schedule_mode,
        ring_dims = ring_dims,
        family_name = identity.family_name,
        protocol_epoch = identity.protocol_epoch,
        cost_model = identity.cost_model.name(),
        selective_l2_response_model = identity.selective_l2_response_model.name(),
        selection_policy = identity.selection_policy.name(),
        recursive_split_search_policy = identity.recursive_split_search_policy.name(),
        recursive_setup_search_policy = identity.recursive_setup_search_policy.name(),
        setup_field_budget = match identity.setup_field_budget {
            Some(value) => format!("Some({value})"),
            None => "None".to_string(),
        },
        min_offloaded_witness_contraction = identity.min_offloaded_witness_contraction,
        sis_modulus_profile = emit_sis_modulus_profile(identity.sis_modulus_profile),
        sis_security_policy = identity.sis_security_policy.name(),
        sis_table_digest = format_bytes(identity.sis_table_digest.0),
        sis_l2_table_digest = format_bytes(identity.sis_l2_table_digest.0),
        decomposition = emit_decomposition(identity.decomposition),
        claim_ext_degree = identity.claim_ext_degree,
        chal_ext_degree = identity.chal_ext_degree,
        inner_basis_min = identity.inner_basis_range.0,
        inner_basis_max = identity.inner_basis_range.1,
        basis_min = identity.opening_basis_range.0,
        basis_max = identity.opening_basis_range.1,
        witness_chunk = emit_witness_chunk(identity.witness_chunk),
        recursive_setup_planning = identity.recursive_setup_planning,
        ring_challenge_config_digest = identity.ring_challenge_config_digest,
        key_count = identity.key_count,
        key_digest = identity.key_digest,
    )
}

/// Emit one family module (entries + embedded catalog identity).
pub fn emit_family_module(spec: &EmitSpec) -> Result<String, String> {
    let mut materialized = materialized_entries_for_specs(
        std::slice::from_ref(spec),
        MaterializationDiagnostics::default(),
    )?;
    let materialized = materialized
        .pop()
        .ok_or_else(|| "missing materialized schedule family".to_string())?;
    emit_family_module_from_entries(spec, materialized)
}

pub(super) fn emit_family_module_from_entries(
    spec: &EmitSpec,
    materialized: Vec<MaterializedEntry>,
) -> Result<String, String> {
    let mut out = String::new();
    let const_name = spec.const_name;
    writeln!(out, "// Generated by `{}`", spec.generator_command).map_err(|e| e.to_string())?;
    out.push_str(&emit_bounded_source_banner(spec.source_contract));
    writeln!(out, "#[allow(unused_imports)]").map_err(|e| e.to_string())?;
    writeln!(
        out,
        "use super::{{\n    BlockGeometry, ChunkedWitnessCfg, DecompositionParams, \
         GeneratedFoldCore, GeneratedFoldScheduleEntry, GeneratedFrozenGroup, \
         GeneratedGroup, GeneratedMatrix, GeneratedPrecommittedGroup, GeneratedRecursiveFold, \
         GeneratedRootFold, GeneratedScheduleCatalogIdentity, GeneratedSetupPrefix, GeneratedTerminalFold, \
         CommitmentRingDims, PlannerCostModelId, PolynomialGroupLayout, GroupCommitPhaseParams, \
         InnerCommitMatrixParams, OuterCommitMatrixParams, \
         CommitmentPayloadMode, RingDimensionScheduleMode, SelectionPolicyId, SelectiveL2ResponseModelId, SisL2TableDigest, SisModulusProfileId, SisSecurityPolicyId, SisTableDigest, \n}};"
    )
    .map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    let mut memory_entries: Vec<GeneratedFoldScheduleEntry> = Vec::new();

    writeln!(out, "#[rustfmt::skip]").map_err(|e| e.to_string())?;
    writeln!(
        out,
        "pub(crate) static {const_name}: &[GeneratedFoldScheduleEntry] = &["
    )
    .map_err(|e| e.to_string())?;

    for entry in materialized {
        let key = entry.key();
        emit_schedule_entry(
            &mut out,
            &key,
            entry.schedule(),
            entry.precommitted_producers(),
        )?;
        memory_entries.push(generated_entry(&key, entry.schedule())?);
    }
    debug_assert!(akita_schedules::catalog_entries_sorted_for_lookup(
        &memory_entries
    ));

    writeln!(out, "];").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    let identity = expected_catalog_identity(
        spec.family_name,
        &spec.policy,
        &memory_entries,
        spec.ring_challenge_config,
    )
    .map_err(|e| format!("{}: catalog identity: {e}", spec.module_name))?;
    out.push_str(&emit_identity_const(&identity));

    Ok(out)
}

#[cfg(all(test, feature = "catalog-gen"))]
mod preplanned_scalar_tests {
    use super::*;
    use crate::generated_families::{wiring_emit_spec, ALL_GENERATED_FAMILIES};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::OnceLock;
    use std::time::Duration;

    static REGEN_CALLS: AtomicUsize = AtomicUsize::new(0);
    static ACTIVE_REGEN: AtomicUsize = AtomicUsize::new(0);
    static MAX_ACTIVE_REGEN: AtomicUsize = AtomicUsize::new(0);
    static REGEN_SCHEDULE: OnceLock<FoldSchedule> = OnceLock::new();

    fn counted_regen(_key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
        REGEN_CALLS.fetch_add(1, Ordering::Relaxed);
        let active = ACTIVE_REGEN.fetch_add(1, Ordering::Relaxed) + 1;
        MAX_ACTIVE_REGEN.fetch_max(active, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(20));
        ACTIVE_REGEN.fetch_sub(1, Ordering::Relaxed);
        Ok(REGEN_SCHEDULE.get().expect("test schedule").clone())
    }

    #[test]
    fn bounded_parallel_map_uses_workers_for_small_expensive_batches() {
        ACTIVE_REGEN.store(0, Ordering::Relaxed);
        MAX_ACTIVE_REGEN.store(0, Ordering::Relaxed);
        let items = [(), ()];
        let output = bounded_parallel_filter_map(&items, 2, |_| {
            let active = ACTIVE_REGEN.fetch_add(1, Ordering::Relaxed) + 1;
            MAX_ACTIVE_REGEN.fetch_max(active, Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(20));
            ACTIVE_REGEN.fetch_sub(1, Ordering::Relaxed);
            Ok(Some(()))
        })
        .expect("bounded parallel map");
        assert_eq!(output.len(), items.len());
        assert_eq!(MAX_ACTIVE_REGEN.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn preplanned_scalar_skips_regen_and_preserves_emitted_bytes() {
        let family = ALL_GENERATED_FAMILIES
            .iter()
            .find(|family| family.module_name == "fp128_onehot_multi_chunk_w2r2")
            .expect("known family");
        let key = PolynomialGroupLayout::new(14, 1);
        let schedule = (family.regen)(key).expect("scalar schedule");
        REGEN_SCHEDULE.get_or_init(|| schedule.clone());
        let mut cached = wiring_emit_spec(family, PathBuf::from("generated"))
            .expect("shipped families declare a valid producer contract");
        cached.keys = vec![key];
        cached.preplanned_scalar = vec![(key, schedule)];
        cached.regen = counted_regen;
        cached.generator_command = "generator command";

        REGEN_CALLS.store(0, Ordering::Relaxed);
        let cached_bytes = emit_family_module(&cached).expect("cached family module");
        assert_eq!(REGEN_CALLS.load(Ordering::Relaxed), 0);

        let mut uncached = cached.clone();
        uncached.preplanned_scalar.clear();
        let uncached_bytes = emit_family_module(&uncached).expect("uncached family module");
        assert_eq!(REGEN_CALLS.load(Ordering::Relaxed), 1);
        assert_eq!(cached_bytes, uncached_bytes);

        let specs = [cached.clone(), cached];
        let rendered = render_generated_outputs(&specs, &[], None).expect("flattened render");
        assert_eq!(rendered.len(), specs.len());
        assert!(rendered.iter().all(|output| output.body == cached_bytes));

        let mut queued = uncached;
        queued.keys = vec![key; 3];
        let specs = [queued.clone(), queued];
        REGEN_CALLS.store(0, Ordering::Relaxed);
        ACTIVE_REGEN.store(0, Ordering::Relaxed);
        MAX_ACTIVE_REGEN.store(0, Ordering::Relaxed);
        materialized_entries_for_specs(&specs, MaterializationDiagnostics::default())
            .expect("flattened planning queue");
        assert_eq!(REGEN_CALLS.load(Ordering::Relaxed), 6);
        assert!(
            MAX_ACTIVE_REGEN.load(Ordering::Relaxed) <= offline_planning_worker_count(6),
            "flattened planning exceeded the process worker bound"
        );
    }

    #[test]
    fn packing_planner_expanded_and_generated_prices_agree() {
        use akita_config::{
            honest_fold_policy_of, policy_of, proof_optimized::fp128::OneHot, CommitmentConfig,
        };

        let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(16, 1));
        let policy = policy_of::<OneHot>();
        let planned = crate::planner::find_schedule(
            &key,
            honest_fold_policy_of::<OneHot>(),
            &[],
            &policy,
            OneHot::ring_challenge_config,
        )
        .expect("planned packing schedule");
        assert!(matches!(
            planned.schedule.root.params.opening_method(),
            akita_types::OpeningMethod::SubringCoefficientPacking { .. }
        ));
        let expanded = akita_schedules::expanded_schedule_proof_payload_bytes(
            &key,
            &planned.schedule,
            &policy,
        )
        .expect("expanded schedule price");
        let generated = generated_entry(&key, &planned.schedule).expect("compact schedule row");
        let replayed = akita_schedules::estimate_proof_bytes(
            &generated,
            &key,
            &policy,
            OneHot::ring_challenge_config,
        )
        .expect("generated-row replay price");
        assert_eq!(
            planned.estimate.estimated_proof_payload_bytes().unwrap(),
            expanded,
        );
        assert_eq!(expanded, replayed);
    }
}
