//! Reusable schedule-table emitter for `akita-schedules` and downstream catalogs.
//!
//! The `akita-config` `gen_schedule_tables` binary adapts preset metadata into
//! [`EmitSpec`] values and calls this module. Jolt can invoke the same API with
//! an explicit [`PlannerPolicy`] and hook function pointers.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::sis::HonestFoldPolicySpec;
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupParams, CommittedGroupProfile, FoldSchedule,
    OpenCommitMatrixParams, PolynomialGroupLayout, SetupPrefixSlotId, WitnessPartition,
};

use crate::PlannerPolicy;
mod publish;
mod render;
use akita_schedules::expected_catalog_identity;
use akita_schedules::generated::{
    GeneratedBlockGeometry, GeneratedCommittedGroup, GeneratedFoldScheduleEntry,
    GeneratedInnerCommitMatrix, GeneratedOpenCommitMatrix, GeneratedOuterCommitMatrix,
    GeneratedRecursiveFold, GeneratedRootFinalGroup, GeneratedRootFold,
    GeneratedRootPrecommittedGroup, GeneratedScheduleCatalogIdentity, GeneratedSetupPrefixInput,
    GeneratedTerminalFold, GeneratedWitnessPartition,
};
pub use publish::publish_generated_outputs;
pub use render::{render_generated_outputs, GeneratedOutput};

/// One family the emitter writes to `akita-schedules/src/generated/`.
#[derive(Clone)]
pub struct EmitSpec {
    pub module_name: &'static str,
    pub const_name: &'static str,
    pub family_name: &'static str,
    pub schedule_feature: &'static str,
    pub policy: PlannerPolicy,
    pub keys: Vec<PolynomialGroupLayout>,
    pub group_batch_keys: Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>,
    /// Exact successful scalar results already needed to construct grouped keys.
    pub preplanned_scalar: Vec<(PolynomialGroupLayout, FoldSchedule)>,
    pub output_dir: PathBuf,
    pub regen: fn(PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError>,
    pub regen_group_batch:
        fn(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>) -> Result<FoldSchedule, AkitaError>,
    pub ring_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    pub generator_command: &'static str,
}

const MOD_WIRING_BEGIN: &str = "// @generated schedule module wiring begin";
const MOD_WIRING_END: &str = "// @generated schedule module wiring end";
// Schedule search is memory bound. Keep the default below host-wide
// parallelism while allowing explicit tuning for large generation machines.
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
    if workers <= 1 || items.len() < 2 * workers {
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

fn geometry(p: &CommittedGroupParams) -> GeneratedBlockGeometry {
    GeneratedBlockGeometry {
        live_ring_elements_per_claim: p.num_live_ring_elements_per_claim as u64,
        positions_per_block: p.num_positions_per_block as u64,
        live_blocks: p.num_live_blocks as u64,
    }
}

fn committed_group(p: &CommittedGroupParams) -> GeneratedCommittedGroup {
    GeneratedCommittedGroup {
        geometry: geometry(p),
        inner_commit_matrix: GeneratedInnerCommitMatrix {
            ring_dimension: p.inner_commit_matrix.ring_dimension() as u32,
            log_basis: p.log_basis_inner,
        },
        outer_commit_matrix: GeneratedOuterCommitMatrix {
            ring_dimension: p.outer_commit_matrix.ring_dimension() as u32,
            log_basis: p.log_basis_outer,
        },
        outer_slice_count: p.outer_slice_count.get() as u32,
    }
}

fn open_matrix_params(p: &OpenCommitMatrixParams, log_basis: u32) -> GeneratedOpenCommitMatrix {
    GeneratedOpenCommitMatrix {
        ring_dimension: p.ring_dimension() as u32,
        log_basis,
    }
}

fn runtime_witness_partition(p: &WitnessPartition) -> GeneratedWitnessPartition {
    match p {
        WitnessPartition::Single => GeneratedWitnessPartition::Single,
        WitnessPartition::Distributed { num_chunks } => GeneratedWitnessPartition::Distributed {
            num_chunks: *num_chunks as u32,
        },
    }
}

fn setup_prefix_slot_input(slot: &SetupPrefixSlotId) -> GeneratedSetupPrefixInput {
    let group = &slot.commitment_params;
    GeneratedSetupPrefixInput {
        natural_len: slot.natural_len as u64,
        num_digits_fold: group.num_digits_fold as u32,
        commitment: GeneratedCommittedGroup {
            geometry: GeneratedBlockGeometry {
                live_ring_elements_per_claim: group.layout.num_live_ring_elements_per_claim as u64,
                positions_per_block: group.layout.num_positions_per_block as u64,
                live_blocks: group.layout.num_live_blocks as u64,
            },
            inner_commit_matrix: GeneratedInnerCommitMatrix {
                ring_dimension: group.layout.inner_commit_matrix.ring_dimension() as u32,
                log_basis: group.layout.log_basis_inner,
            },
            outer_commit_matrix: GeneratedOuterCommitMatrix {
                ring_dimension: group.layout.outer_commit_matrix.ring_dimension() as u32,
                log_basis: group.layout.log_basis_outer,
            },
            outer_slice_count: group.layout.outer_slice_count.get() as u32,
        },
    }
}

fn generated_entry(
    key: &AkitaScheduleLookupKey,
    schedule: &FoldSchedule,
) -> Result<GeneratedFoldScheduleEntry, String> {
    let root_fold = &schedule.root.params;
    let root_params = &root_fold.final_group.commitment;
    let precommitted_groups = key
        .precommitteds
        .iter()
        .copied()
        .zip(&root_fold.precommitted_groups)
        .map(|(descriptor, group)| GeneratedRootPrecommittedGroup {
            descriptor,
            num_digits_fold: group.commitment.num_digits_fold as u32,
            commitment: GeneratedCommittedGroup {
                geometry: GeneratedBlockGeometry {
                    live_ring_elements_per_claim: group
                        .commitment
                        .layout
                        .num_live_ring_elements_per_claim
                        as u64,
                    positions_per_block: group.commitment.layout.num_positions_per_block as u64,
                    live_blocks: group.commitment.layout.num_live_blocks as u64,
                },
                inner_commit_matrix: GeneratedInnerCommitMatrix {
                    ring_dimension: group.commitment.layout.inner_commit_matrix.ring_dimension()
                        as u32,
                    log_basis: group.commitment.layout.log_basis_inner,
                },
                outer_commit_matrix: GeneratedOuterCommitMatrix {
                    ring_dimension: group.commitment.layout.outer_commit_matrix.ring_dimension()
                        as u32,
                    log_basis: group.commitment.layout.log_basis_outer,
                },
                outer_slice_count: group.commitment.layout.outer_slice_count.get() as u32,
            },
        })
        .collect::<Vec<_>>();
    let recursive_folds = schedule
        .recursive_folds
        .iter()
        .map(|step| GeneratedRecursiveFold {
            payload_mode: step.params.witness.payload_mode,
            witness: committed_group(&step.params.witness),
            num_digits_fold: step.params.witness.num_digits_fold as u32,
            response_l2_sq_cap: match step.params.witness.inner_commit_matrix.security_route() {
                akita_types::InnerCommitSecurityRoute::Linf(_) => None,
                akita_types::InnerCommitSecurityRoute::L2 {
                    response_l2_sq_cap, ..
                } => Some(response_l2_sq_cap),
            },
            open_commit_matrix: open_matrix_params(
                &step.params.open_commit_matrix,
                step.params.witness.log_basis_open,
            ),
            incoming_setup_prefix: step
                .params
                .incoming_setup_prefix
                .as_ref()
                .map(setup_prefix_slot_input),
            witness_partition: runtime_witness_partition(&step.params.witness_partition),
        })
        .collect::<Vec<_>>();
    let terminal_group = schedule
        .terminal
        .params
        .response_shape
        .layout
        .groups
        .first()
        .ok_or_else(|| "terminal response shape has no group".to_string())?;
    if schedule.terminal.params.response_shape.layout.groups.len() != 1 {
        return Err("generated scalar terminal response must have exactly one group".to_string());
    }
    Ok(GeneratedFoldScheduleEntry {
        root: GeneratedRootFold {
            final_group: GeneratedRootFinalGroup {
                layout: key.final_group,
                num_digits_inner: root_params.num_digits_inner as u32,
                num_digits_fold: root_params.num_digits_fold as u32,
                commitment: committed_group(root_params),
            },
            precommitted_groups: Box::leak(precommitted_groups.into_boxed_slice()),
            open_commit_matrix: open_matrix_params(
                &root_fold.open_commit_matrix,
                root_params.log_basis_open,
            ),
            witness_partition: runtime_witness_partition(&root_fold.witness_partition),
        },
        recursive_folds: Box::leak(recursive_folds.into_boxed_slice()),
        terminal: GeneratedTerminalFold {
            geometry: GeneratedBlockGeometry {
                live_ring_elements_per_claim: schedule
                    .terminal
                    .params
                    .witness
                    .num_live_ring_elements_per_claim
                    as u64,
                positions_per_block: schedule.terminal.params.witness.num_positions_per_block
                    as u64,
                live_blocks: schedule.terminal.params.witness.num_live_blocks as u64,
            },
            inner_commit_matrix: GeneratedInnerCommitMatrix {
                ring_dimension: schedule
                    .terminal
                    .params
                    .witness
                    .inner_commit_matrix
                    .ring_dimension() as u32,
                log_basis: schedule.terminal.params.witness.log_basis_inner,
            },
            num_digits_inner: schedule.terminal.params.witness.num_digits_inner as u32,
            fold_log_basis: schedule.terminal.params.witness.fold_log_basis,
            fold_digit_count: schedule.terminal.params.witness.fold_digit_count as u32,
            inner_output_rank: schedule
                .terminal
                .params
                .witness
                .inner_commit_matrix
                .output_rank() as u32,
            inner_coeff_linf_bound: schedule
                .terminal
                .params
                .witness
                .inner_commit_matrix
                .coeff_linf_bound()
                .unwrap_or(0),
            response_l2_sq_cap: schedule.terminal.params.witness.response_l2_sq_cap(),
            z_admission_linf_cap: terminal_group.z_admission_linf_cap,
            z_rice_low_bits: terminal_group.z_rice_low_bits,
            z_payload_bytes: terminal_group.z_payload_bytes as u64,
        },
    })
}

fn emit_key(key: PolynomialGroupLayout) -> String {
    match key.source() {
        akita_types::RootSourceProfile::Dense => format!(
            "PolynomialGroupLayout::new({}, {})",
            key.num_vars(),
            key.num_polynomials(),
        ),
        akita_types::RootSourceProfile::UnitOneHot { chunk_size } => format!(
            "PolynomialGroupLayout::unit_one_hot({}, {}, {})",
            key.num_vars(),
            key.num_polynomials(),
            chunk_size,
        ),
    }
}

fn emit_precommitted_group_key(layout: &CommittedGroupProfile) -> String {
    format!(
        "CommittedGroupProfile {{ version: CommittedGroupProfile::VERSION, group: {}, num_live_ring_elements_per_claim: {}, num_positions_per_block: {}, num_live_blocks: {}, outer_slice_count: akita_types::CommitmentSliceCount::{}, log_basis_inner: {}, num_digits_inner: {}, inner_commit_matrix: {}, log_basis_outer: {}, num_digits_outer: {}, outer_commit_matrix: {} }}",
        emit_key(layout.group),
        layout.num_live_ring_elements_per_claim,
        layout.num_positions_per_block,
        layout.num_live_blocks,
        match layout.outer_slice_count {
            akita_types::CommitmentSliceCount::ONE => "ONE",
            akita_types::CommitmentSliceCount::TWO => "TWO",
            akita_types::CommitmentSliceCount::FOUR => "FOUR",
            akita_types::CommitmentSliceCount::EIGHT => "EIGHT",
            _ => unreachable!("checked commitment slice count"),
        },
        layout.log_basis_inner,
        layout.num_digits_inner,
        emit_profile_matrix(
            "InnerCommitMatrixParams",
            layout.inner_commit_matrix.output_rank(),
            layout.inner_commit_matrix.input_width(),
            layout
                .inner_commit_matrix
                .sis_table_key()
                .expect("validated precommitted matrix is L infinity"),
        ),
        layout.log_basis_outer,
        layout.num_digits_outer,
        emit_profile_matrix(
            "OuterCommitMatrixParams",
            layout.outer_commit_matrix.output_rank(),
            layout.outer_commit_matrix.input_width(),
            layout.outer_commit_matrix.sis_table_key(),
        ),
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

fn emit_geometry(value: GeneratedBlockGeometry) -> String {
    format!(
        "GeneratedBlockGeometry {{ live_ring_elements_per_claim: {}, positions_per_block: {}, live_blocks: {} }}",
        value.live_ring_elements_per_claim, value.positions_per_block, value.live_blocks
    )
}

fn emit_committed_group(value: GeneratedCommittedGroup) -> String {
    format!(
        "GeneratedCommittedGroup {{ geometry: {}, inner_commit_matrix: GeneratedInnerCommitMatrix {{ ring_dimension: {}, log_basis: {} }}, outer_commit_matrix: GeneratedOuterCommitMatrix {{ ring_dimension: {}, log_basis: {} }}, outer_slice_count: {} }}",
        emit_geometry(value.geometry),
        value.inner_commit_matrix.ring_dimension,
        value.inner_commit_matrix.log_basis,
        value.outer_commit_matrix.ring_dimension,
        value.outer_commit_matrix.log_basis,
        value.outer_slice_count,
    )
}

fn emit_open_matrix(value: GeneratedOpenCommitMatrix) -> String {
    format!(
        "GeneratedOpenCommitMatrix {{ ring_dimension: {}, log_basis: {} }}",
        value.ring_dimension, value.log_basis
    )
}

fn emit_partition(value: GeneratedWitnessPartition) -> String {
    match value {
        GeneratedWitnessPartition::Single => "GeneratedWitnessPartition::Single".to_string(),
        GeneratedWitnessPartition::Distributed { num_chunks } => {
            format!("GeneratedWitnessPartition::Distributed {{ num_chunks: {num_chunks} }}")
        }
    }
}

fn emit_payload_mode(value: akita_types::CommitmentPayloadMode) -> &'static str {
    match value {
        akita_types::CommitmentPayloadMode::Compressed => "CommitmentPayloadMode::Compressed",
        akita_types::CommitmentPayloadMode::Raw => "CommitmentPayloadMode::Raw",
    }
}

fn emit_setup_prefix(value: Option<GeneratedSetupPrefixInput>) -> String {
    match value {
        Some(value) => format!(
            "Some(GeneratedSetupPrefixInput {{ natural_len: {}, num_digits_fold: {}, commitment: {} }})",
            value.natural_len,
            value.num_digits_fold,
            emit_committed_group(value.commitment)
        ),
        None => "None".to_string(),
    }
}

fn emit_schedule_entry(
    out: &mut String,
    key: &AkitaScheduleLookupKey,
    schedule: &FoldSchedule,
) -> Result<(), String> {
    let entry = generated_entry(key, schedule)?;
    writeln!(out, "    GeneratedFoldScheduleEntry {{").map_err(|e| e.to_string())?;
    writeln!(out, "        root: GeneratedRootFold {{").map_err(|e| e.to_string())?;
    writeln!(
        out,
        "            final_group: GeneratedRootFinalGroup {{ layout: {}, num_digits_inner: {}, num_digits_fold: {},",
        emit_key(entry.root.final_group.layout),
        entry.root.final_group.num_digits_inner,
        entry.root.final_group.num_digits_fold,
    )
    .map_err(|e| e.to_string())?;
    writeln!(
        out,
        "                commitment: {} }},",
        emit_committed_group(entry.root.final_group.commitment),
    )
    .map_err(|e| e.to_string())?;
    if entry.root.precommitted_groups.is_empty() {
        writeln!(out, "            precommitted_groups: &[],").map_err(|e| e.to_string())?;
    } else {
        writeln!(out, "            precommitted_groups: &[").map_err(|e| e.to_string())?;
        for group in entry.root.precommitted_groups {
            writeln!(
                out,
                "                GeneratedRootPrecommittedGroup {{ descriptor: {}, num_digits_fold: {}, commitment: {} }},",
                emit_precommitted_group_key(&group.descriptor),
                group.num_digits_fold,
                emit_committed_group(group.commitment),
            )
            .map_err(|e| e.to_string())?;
        }
        writeln!(out, "            ],").map_err(|e| e.to_string())?;
    }
    writeln!(
        out,
        "            open_commit_matrix: {},",
        emit_open_matrix(entry.root.open_commit_matrix),
    )
    .map_err(|e| e.to_string())?;
    writeln!(
        out,
        "            witness_partition: {},",
        emit_partition(entry.root.witness_partition),
    )
    .map_err(|e| e.to_string())?;
    writeln!(out, "        }},").map_err(|e| e.to_string())?;
    if entry.recursive_folds.is_empty() {
        writeln!(out, "        recursive_folds: &[],").map_err(|e| e.to_string())?;
    } else {
        writeln!(out, "        recursive_folds: &[").map_err(|e| e.to_string())?;
        for fold in entry.recursive_folds {
            writeln!(
                out,
                "            GeneratedRecursiveFold {{ payload_mode: {}, witness: {}, num_digits_fold: {}, response_l2_sq_cap: {}, open_commit_matrix: {}, incoming_setup_prefix: {}, witness_partition: {} }},",
                emit_payload_mode(fold.payload_mode),
                emit_committed_group(fold.witness),
                fold.num_digits_fold,
                fold.response_l2_sq_cap.map_or_else(
                    || "None".to_string(),
                    |cap| format!("Some({cap})"),
                ),
                emit_open_matrix(fold.open_commit_matrix),
                emit_setup_prefix(fold.incoming_setup_prefix),
                emit_partition(fold.witness_partition),
            )
            .map_err(|e| e.to_string())?;
        }
        writeln!(out, "        ],").map_err(|e| e.to_string())?;
    }
    writeln!(
        out,
        "        terminal: GeneratedTerminalFold {{ geometry: {}, inner_commit_matrix: GeneratedInnerCommitMatrix {{ ring_dimension: {}, log_basis: {} }}, num_digits_inner: {}, fold_log_basis: {}, fold_digit_count: {}, inner_output_rank: {}, inner_coeff_linf_bound: {}, response_l2_sq_cap: {}, z_admission_linf_cap: {}, z_rice_low_bits: {}, z_payload_bytes: {} }},",
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
        entry.terminal.z_admission_linf_cap,
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
            "    setup_field_budget: {setup_field_budget},\n",
            "    min_offloaded_witness_contraction: {min_offloaded_witness_contraction},\n",
            "    sis_modulus_profile: {sis_modulus_profile},\n",
            "    sis_security_policy: SisSecurityPolicyId::{sis_security_policy},\n",
            "    sis_table_digest: SisTableDigest({sis_table_digest}),\n",
            "    sis_l2_table_digest: SisL2TableDigest({sis_l2_table_digest}),\n",
            "    uniform_ring_dimension: {uniform_ring_dimension},\n",
            "    setup_prefix_inner_ring_dimension: {setup_prefix_inner_ring_dimension},\n",
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
        setup_field_budget = match identity.setup_field_budget {
            Some(value) => format!("Some({value})"),
            None => "None".to_string(),
        },
        min_offloaded_witness_contraction = identity.min_offloaded_witness_contraction,
        sis_modulus_profile = emit_sis_modulus_profile(identity.sis_modulus_profile),
        sis_security_policy = identity.sis_security_policy.name(),
        sis_table_digest = format_bytes(identity.sis_table_digest.0),
        sis_l2_table_digest = format_bytes(identity.sis_l2_table_digest.0),
        uniform_ring_dimension = identity.uniform_ring_dimension,
        setup_prefix_inner_ring_dimension = identity.setup_prefix_inner_ring_dimension,
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

enum PlanningRequest {
    Scalar(PolynomialGroupLayout),
    Grouped {
        key: AkitaScheduleLookupKey,
        honest_fold_policies: Vec<HonestFoldPolicySpec>,
    },
}

struct IndexedPlanningRequest {
    spec_index: usize,
    request: PlanningRequest,
}

pub(super) type MaterializedEntry = (AkitaScheduleLookupKey, FoldSchedule);

pub(super) fn materialized_entries_for_specs(
    specs: &[EmitSpec],
) -> Result<Vec<Vec<MaterializedEntry>>, String> {
    let request_count = specs
        .iter()
        .map(|spec| spec.keys.len() + spec.group_batch_keys.len())
        .sum();
    let mut requests = Vec::with_capacity(request_count);
    for (spec_index, spec) in specs.iter().enumerate() {
        requests.extend(spec.keys.iter().copied().map(|key| IndexedPlanningRequest {
            spec_index,
            request: PlanningRequest::Scalar(key),
        }));
        requests.extend(spec.group_batch_keys.iter().cloned().map(
            |(key, honest_fold_policies)| IndexedPlanningRequest {
                spec_index,
                request: PlanningRequest::Grouped {
                    key,
                    honest_fold_policies,
                },
            },
        ));
    }

    let workers = offline_planning_worker_count(requests.len());
    let materialized = bounded_parallel_filter_map(&requests, workers, |indexed| {
        materialized_entry(&specs[indexed.spec_index], &indexed.request)
            .map(|entry| entry.map(|entry| (indexed.spec_index, entry)))
    })?;
    let mut entries_by_spec = std::iter::repeat_with(Vec::new)
        .take(specs.len())
        .collect::<Vec<_>>();
    for (spec_index, entry) in materialized {
        entries_by_spec[spec_index].push(entry);
    }
    for entries in &mut entries_by_spec {
        entries.sort_by(|(left, _), (right, _)| {
            akita_schedules::runtime_schedule_key_cmp(left, right)
        });
    }
    Ok(entries_by_spec)
}

fn materialized_entry(
    spec: &EmitSpec,
    request: &PlanningRequest,
) -> Result<Option<(AkitaScheduleLookupKey, FoldSchedule)>, String> {
    let (key, result) = match request {
        PlanningRequest::Scalar(key) => {
            let lookup = AkitaScheduleLookupKey::single(*key);
            let result = spec
                .preplanned_scalar
                .iter()
                .find(|(preplanned_key, _)| preplanned_key == key)
                .map_or_else(|| (spec.regen)(*key), |(_, schedule)| Ok(schedule.clone()));
            (lookup, result)
        }
        PlanningRequest::Grouped {
            key,
            honest_fold_policies,
        } => (
            key.clone(),
            (spec.regen_group_batch)(key.clone(), honest_fold_policies.clone()),
        ),
    };
    match result {
        Ok(schedule) => Ok(Some((key, schedule))),
        Err(akita_field::AkitaError::UnsupportedSchedule(_)) => Ok(None),
        Err(error) => {
            let kind = if key.precommitteds.is_empty() {
                "regen"
            } else {
                "regen multi-group"
            };
            Err(format!("{}: {kind} {key:?}: {error}", spec.module_name))
        }
    }
}

/// Emit one family module (entries + embedded catalog identity).
pub fn emit_family_module(spec: &EmitSpec) -> Result<String, String> {
    let mut materialized = materialized_entries_for_specs(std::slice::from_ref(spec))?;
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
    writeln!(out, "#[allow(unused_imports)]").map_err(|e| e.to_string())?;
    writeln!(
        out,
        "use super::{{\n    ChunkedWitnessCfg, DecompositionParams, GeneratedBlockGeometry, \
         GeneratedCommittedGroup, GeneratedFoldScheduleEntry, GeneratedInnerCommitMatrix, \
         GeneratedOpenCommitMatrix, GeneratedOuterCommitMatrix, GeneratedRecursiveFold, \
         GeneratedRootFinalGroup, GeneratedRootFold, \
         GeneratedRootPrecommittedGroup, GeneratedScheduleCatalogIdentity, \
         GeneratedSetupPrefixInput, GeneratedTerminalFold, GeneratedWitnessPartition, \
         CommitmentRingDims, PlannerCostModelId, PolynomialGroupLayout, CommittedGroupProfile, \
         InnerCommitMatrixParams, OuterCommitMatrixParams, \
         CommitmentPayloadMode, RingDimensionScheduleMode, SelectionPolicyId, SelectiveL2ResponseModelId, SisL2TableDigest, SisModulusProfileId, SisSecurityPolicyId, SisTableDigest,\n}};"
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

    for (key, schedule) in materialized {
        emit_schedule_entry(&mut out, &key, &schedule)?;
        memory_entries.push(generated_entry(&key, &schedule)?);
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
    fn preplanned_scalar_skips_regen_and_preserves_emitted_bytes() {
        let family = ALL_GENERATED_FAMILIES
            .iter()
            .find(|family| family.module_name == "fp128_onehot_multi_chunk_w2r2")
            .expect("known family");
        let key = PolynomialGroupLayout::unit_one_hot(14, 1, 256);
        let schedule = (family.regen)(key).expect("scalar schedule");
        REGEN_SCHEDULE.get_or_init(|| schedule.clone());
        let mut cached = wiring_emit_spec(family, PathBuf::from("generated"));
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
        materialized_entries_for_specs(&specs).expect("flattened planning queue");
        assert_eq!(REGEN_CALLS.load(Ordering::Relaxed), 6);
        assert!(
            MAX_ACTIVE_REGEN.load(Ordering::Relaxed) <= offline_planning_worker_count(6),
            "flattened planning exceeded the process worker bound"
        );
    }
}
