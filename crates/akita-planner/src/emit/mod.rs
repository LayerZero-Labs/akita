//! Reusable schedule-table emitter for `akita-schedules` and downstream catalogs.
//!
//! The `akita-config` `gen_schedule_tables` binary adapts preset metadata into
//! [`EmitSpec`] values and calls this module. Jolt can invoke the same API with
//! an explicit [`PlannerPolicy`] and hook function pointers.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::HonestFoldPolicySpec;
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, CommittedGroupParams, CommittedGroupProfile,
    FoldSchedule, OpenCommitMatrixParams, PolynomialGroupLayout, RootFinalChallenge,
    SetupPrefixSlotId, WitnessPartition,
};

use crate::PlannerPolicy;
use akita_schedules::expected_catalog_identity;
use akita_schedules::generated::{
    GeneratedBlockGeometry, GeneratedCommittedGroup, GeneratedFoldScheduleEntry,
    GeneratedInnerCommitMatrix, GeneratedOpenCommitMatrix, GeneratedOuterCommitMatrix,
    GeneratedRecursiveFold, GeneratedRootFinalChallenge, GeneratedRootFinalGroup,
    GeneratedRootFold, GeneratedRootPrecommittedGroup, GeneratedScheduleCatalogIdentity,
    GeneratedSetupPrefixInput, GeneratedTerminalFold, GeneratedWitnessPartition,
};

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
    pub emit_group_batch: bool,
    pub output_dir: PathBuf,
    pub regen: fn(PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError>,
    pub regen_group_batch:
        fn(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>) -> Result<FoldSchedule, AkitaError>,
    pub ring_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    pub fold_challenge_shape_at_level: fn(AkitaScheduleInputs) -> TensorChallengeShape,
    pub generator_command: &'static str,
}

const MOD_WIRING_BEGIN: &str = "// @generated schedule module wiring begin";
const MOD_WIRING_END: &str = "// @generated schedule module wiring end";

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
            slice_count: 1,
        },
    }
}

fn open_matrix_params(p: &OpenCommitMatrixParams, log_basis: u32) -> GeneratedOpenCommitMatrix {
    GeneratedOpenCommitMatrix {
        ring_dimension: p.ring_dimension() as u32,
        log_basis,
        slice_count: 1,
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
                slice_count: 1,
            },
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
                    slice_count: 1,
                },
            },
        })
        .collect::<Vec<_>>();
    let recursive_folds = schedule
        .recursive_folds
        .iter()
        .map(|step| GeneratedRecursiveFold {
            payload_mode: step.params.witness.payload_mode,
            witness: committed_group(&step.params.witness),
            unchecked_l2_collision_sq: step
                .params
                .witness
                .inner_commit_matrix
                .unchecked_l2_collision_sq(),
            num_digits_fold: step.params.witness.num_digits_fold as u32,
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
    let challenge = match root_fold.final_group.challenge {
        RootFinalChallenge::Flat => GeneratedRootFinalChallenge::Flat,
        RootFinalChallenge::Tensor { fold_low_len } => GeneratedRootFinalChallenge::Tensor {
            fold_low_len: fold_low_len as u32,
        },
    };
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
                challenge,
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
                .coeff_linf_bound(),
            z_admission_linf_cap: terminal_group.z_admission_linf_cap,
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

fn emit_precommitted_group_key(layout: &CommittedGroupProfile) -> String {
    format!(
        "CommittedGroupProfile {{ version: CommittedGroupProfile::VERSION, group: {}, num_live_ring_elements_per_claim: {}, num_positions_per_block: {}, num_live_blocks: {}, log_basis_inner: {}, num_digits_inner: {}, inner_commit_matrix: {}, log_basis_outer: {}, num_digits_outer: {}, outer_commit_matrix: {} }}",
        emit_key(layout.group),
        layout.num_live_ring_elements_per_claim,
        layout.num_positions_per_block,
        layout.num_live_blocks,
        layout.log_basis_inner,
        layout.num_digits_inner,
        emit_profile_matrix(
            "InnerCommitMatrixParams",
            layout.inner_commit_matrix.output_rank(),
            layout.inner_commit_matrix.input_width(),
            layout.inner_commit_matrix.sis_table_key(),
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
        "GeneratedCommittedGroup {{ geometry: {}, inner_commit_matrix: GeneratedInnerCommitMatrix {{ ring_dimension: {}, log_basis: {} }}, outer_commit_matrix: GeneratedOuterCommitMatrix {{ ring_dimension: {}, log_basis: {}, slice_count: {} }} }}",
        emit_geometry(value.geometry),
        value.inner_commit_matrix.ring_dimension,
        value.inner_commit_matrix.log_basis,
        value.outer_commit_matrix.ring_dimension,
        value.outer_commit_matrix.log_basis,
        value.outer_commit_matrix.slice_count,
    )
}

fn emit_open_matrix(value: GeneratedOpenCommitMatrix) -> String {
    format!(
        "GeneratedOpenCommitMatrix {{ ring_dimension: {}, log_basis: {}, slice_count: {} }}",
        value.ring_dimension, value.log_basis, value.slice_count
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
    let challenge = match entry.root.final_group.challenge {
        GeneratedRootFinalChallenge::Flat => "GeneratedRootFinalChallenge::Flat".to_string(),
        GeneratedRootFinalChallenge::Tensor { fold_low_len } => {
            format!("GeneratedRootFinalChallenge::Tensor {{ fold_low_len: {fold_low_len} }}")
        }
    };
    writeln!(out, "    GeneratedFoldScheduleEntry {{").map_err(|e| e.to_string())?;
    writeln!(out, "        root: GeneratedRootFold {{").map_err(|e| e.to_string())?;
    writeln!(
        out,
        "            final_group: GeneratedRootFinalGroup {{ layout: {}, num_digits_inner: {}, num_digits_fold: {}, challenge: {},",
        emit_key(entry.root.final_group.layout),
        entry.root.final_group.num_digits_inner,
        entry.root.final_group.num_digits_fold,
        challenge,
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
                "            GeneratedRecursiveFold {{ payload_mode: {}, witness: {}, unchecked_l2_collision_sq: {:?}, num_digits_fold: {}, open_commit_matrix: {}, incoming_setup_prefix: {}, witness_partition: {} }},",
                emit_payload_mode(fold.payload_mode),
                emit_committed_group(fold.witness),
                fold.unchecked_l2_collision_sq,
                fold.num_digits_fold,
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
        "        terminal: GeneratedTerminalFold {{ geometry: {}, inner_commit_matrix: GeneratedInnerCommitMatrix {{ ring_dimension: {}, log_basis: {} }}, num_digits_inner: {}, inner_output_rank: {}, inner_coeff_linf_bound: {}, z_admission_linf_cap: {}, z_rice_low_bits: {}, z_payload_bytes: {} }},",
        emit_geometry(entry.terminal.geometry),
        entry.terminal.inner_commit_matrix.ring_dimension,
        entry.terminal.inner_commit_matrix.log_basis,
        entry.terminal.num_digits_inner,
        entry.terminal.inner_output_rank,
        entry.terminal.inner_coeff_linf_bound,
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

fn emit_root_fold_shape(shape: TensorChallengeShape) -> String {
    match shape {
        TensorChallengeShape::Flat => "TensorChallengeShape::Flat".to_string(),
        TensorChallengeShape::Tensor { fold_low_len } => {
            format!("TensorChallengeShape::Tensor {{ fold_low_len: {fold_low_len} }}")
        }
    }
}

fn emit_witness_chunk(cfg: akita_types::ChunkedWitnessCfg) -> String {
    format!(
        "ChunkedWitnessCfg {{ num_chunks: {}, num_activated_levels: {} }}",
        cfg.num_chunks, cfg.num_activated_levels
    )
}

fn emit_identity_const(identity: &GeneratedScheduleCatalogIdentity) -> String {
    let ring_dimension_candidates = identity
        .ring_dimension_candidates
        .iter()
        .map(|dims| {
            format!(
                "CommitmentRingDims {{ inner: {}, outer: {}, opening: {} }}",
                dims.d_a(),
                dims.d_b(),
                dims.d_d()
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let ring_dims: String = identity
        .ring_dimensions
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        concat!(
            "#[rustfmt::skip]\n",
            "pub(crate) static CATALOG_RING_DIMENSION_CANDIDATES: &[CommitmentRingDims] = &[{ring_dimension_candidates}];\n",
            "#[rustfmt::skip]\n",
            "pub(crate) static CATALOG_RING_DIMENSIONS: &[usize] = &[{ring_dims}];\n",
            "#[rustfmt::skip]\n",
            "pub(crate) static CATALOG_IDENTITY: GeneratedScheduleCatalogIdentity = ",
            "GeneratedScheduleCatalogIdentity {{\n",
            "    family_name: \"{family_name}\",\n",
            "    protocol_epoch: {protocol_epoch},\n",
            "    cost_model: PlannerCostModelId::{cost_model},\n",
            "    selection_policy: SelectionPolicyId::{selection_policy},\n",
            "    setup_field_budget: {setup_field_budget},\n",
            "    min_offloaded_witness_contraction: {min_offloaded_witness_contraction},\n",
            "    sis_modulus_profile: {sis_modulus_profile},\n",
            "    sis_security_policy: SisSecurityPolicyId::{sis_security_policy},\n",
            "    sis_table_digest: SisTableDigest({sis_table_digest}),\n",
            "    uniform_ring_dimension: {uniform_ring_dimension},\n",
            "    setup_prefix_inner_ring_dimension: {setup_prefix_inner_ring_dimension},\n",
            "    decomposition: {decomposition},\n",
            "    ring_subfield_norm_bound: {ring_subfield_norm_bound},\n",
            "    claim_ext_degree: {claim_ext_degree},\n",
            "    chal_ext_degree: {chal_ext_degree},\n",
            "    basis_range: ({basis_min}, {basis_max}),\n",
            "    witness_chunk: {witness_chunk},\n",
            "    recursive_setup_planning: {recursive_setup_planning},\n",
            "    root_fold_shape: {root_fold_shape},\n",
            "    ring_dimension_candidates: CATALOG_RING_DIMENSION_CANDIDATES,\n",
            "    ring_dimensions: CATALOG_RING_DIMENSIONS,\n",
            "    ring_challenge_config_digest: {ring_challenge_config_digest},\n",
            "    key_count: {key_count},\n",
            "    key_digest: {key_digest},\n",
            "}};\n",
        ),
        ring_dimension_candidates = ring_dimension_candidates,
        ring_dims = ring_dims,
        family_name = identity.family_name,
        protocol_epoch = identity.protocol_epoch,
        cost_model = identity.cost_model.name(),
        selection_policy = identity.selection_policy.name(),
        setup_field_budget = match identity.setup_field_budget {
            Some(value) => format!("Some({value})"),
            None => "None".to_string(),
        },
        min_offloaded_witness_contraction = identity.min_offloaded_witness_contraction,
        sis_modulus_profile = emit_sis_modulus_profile(identity.sis_modulus_profile),
        sis_security_policy = identity.sis_security_policy.name(),
        sis_table_digest = format_bytes(identity.sis_table_digest.0),
        uniform_ring_dimension = identity.uniform_ring_dimension,
        setup_prefix_inner_ring_dimension = identity.setup_prefix_inner_ring_dimension,
        decomposition = emit_decomposition(identity.decomposition),
        ring_subfield_norm_bound = identity.ring_subfield_norm_bound,
        claim_ext_degree = identity.claim_ext_degree,
        chal_ext_degree = identity.chal_ext_degree,
        basis_min = identity.basis_range.0,
        basis_max = identity.basis_range.1,
        witness_chunk = emit_witness_chunk(identity.witness_chunk),
        recursive_setup_planning = identity.recursive_setup_planning,
        root_fold_shape = emit_root_fold_shape(identity.root_fold_shape),
        ring_challenge_config_digest = identity.ring_challenge_config_digest,
        key_count = identity.key_count,
        key_digest = identity.key_digest,
    )
}

fn output_module_name(spec: &EmitSpec) -> String {
    spec.module_name.to_string()
}

fn output_const_name(spec: &EmitSpec) -> String {
    spec.const_name.to_string()
}

fn materialized_entries(
    spec: &EmitSpec,
) -> Result<Vec<(AkitaScheduleLookupKey, FoldSchedule)>, String> {
    let mut keys = Vec::with_capacity(spec.keys.len() + spec.group_batch_keys.len());
    keys.extend(
        spec.keys
            .iter()
            .copied()
            .map(AkitaScheduleLookupKey::single),
    );
    keys.extend(spec.group_batch_keys.iter().map(|(key, _)| key.clone()));

    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(keys.len().max(1));
    let mut entries = Vec::new();
    if workers > 1 && keys.len() >= 2 * workers {
        let chunk_size = keys.len().div_ceil(workers);
        std::thread::scope(|scope| -> Result<(), String> {
            let handles: Vec<_> = keys
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut local = Vec::new();
                        for key in chunk.iter().cloned() {
                            if let Some(entry) = materialized_entry(spec, key)? {
                                local.push(entry);
                            }
                        }
                        Ok::<_, String>(local)
                    })
                })
                .collect();
            let mut first_error = None;
            for handle in handles {
                match handle.join() {
                    Ok(Ok(local)) if first_error.is_none() => entries.extend(local),
                    Ok(Ok(_)) => {}
                    Ok(Err(error)) => {
                        first_error.get_or_insert(error);
                    }
                    Err(_) => {
                        first_error.get_or_insert_with(|| {
                            format!("{}: regen worker panicked", spec.module_name)
                        });
                    }
                }
            }
            if let Some(error) = first_error {
                return Err(error);
            }
            Ok(())
        })?;
    } else {
        for key in keys {
            if let Some(entry) = materialized_entry(spec, key)? {
                entries.push(entry);
            }
        }
    }
    entries.sort_by(|(left, _), (right, _)| akita_schedules::runtime_schedule_key_cmp(left, right));
    Ok(entries)
}

fn materialized_entry(
    spec: &EmitSpec,
    key: AkitaScheduleLookupKey,
) -> Result<Option<(AkitaScheduleLookupKey, FoldSchedule)>, String> {
    let result = if key.precommitteds.is_empty() {
        (spec.regen)(key.final_group)
    } else {
        let (key, norms) = spec
            .group_batch_keys
            .iter()
            .find(|(candidate, _)| candidate == &key)
            .cloned()
            .ok_or_else(|| format!("{}: missing grouped planning request", spec.module_name))?;
        (spec.regen_group_batch)(key, norms)
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
    let materialized = materialized_entries(spec)?;

    let mut out = String::new();
    let const_name = output_const_name(spec);
    writeln!(out, "// Generated by `{}`", spec.generator_command).map_err(|e| e.to_string())?;
    writeln!(out, "#[allow(unused_imports)]").map_err(|e| e.to_string())?;
    writeln!(
        out,
        "use super::{{\n    ChunkedWitnessCfg, DecompositionParams, GeneratedBlockGeometry, \
         GeneratedCommittedGroup, GeneratedFoldScheduleEntry, GeneratedInnerCommitMatrix, \
         GeneratedOpenCommitMatrix, GeneratedOuterCommitMatrix, GeneratedRecursiveFold, \
         GeneratedRootFinalChallenge, GeneratedRootFinalGroup, GeneratedRootFold, \
         GeneratedRootPrecommittedGroup, GeneratedScheduleCatalogIdentity, \
         GeneratedSetupPrefixInput, GeneratedTerminalFold, GeneratedWitnessPartition, \
         CommitmentRingDims, PlannerCostModelId, PolynomialGroupLayout, CommittedGroupProfile, \
         InnerCommitMatrixParams, OuterCommitMatrixParams, \
         CommitmentPayloadMode, SelectionPolicyId, SisModulusProfileId, SisSecurityPolicyId, SisTableDigest, \
         TensorChallengeShape,\n}};"
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
        spec.fold_challenge_shape_at_level,
    )
    .map_err(|e| format!("{}: catalog identity: {e}", spec.module_name))?;
    out.push_str(&emit_identity_const(&identity));

    Ok(out)
}

fn emit_module_declarations(specs: &[EmitSpec]) -> Result<String, String> {
    let mut out = String::new();
    let mut seen = std::collections::BTreeSet::new();
    for spec in specs {
        if !seen.insert(spec.module_name) {
            continue;
        }
        let module_name = spec.module_name;
        let feat = spec.schedule_feature;
        writeln!(out, "#[cfg(feature = \"{feat}\")]").map_err(|e| e.to_string())?;
        writeln!(out, "pub mod {module_name};").map_err(|e| e.to_string())?;
    }
    writeln!(out).map_err(|e| e.to_string())?;
    Ok(out)
}

fn table_fn_name(module_name: &str) -> String {
    format!("{module_name}_table")
}

fn emit_table_accessor(spec: &EmitSpec) -> Result<String, String> {
    let fn_name = table_fn_name(spec.module_name);
    let feat = spec.schedule_feature;
    let module_name = spec.module_name;
    let const_name = spec.const_name;
    Ok(format!(
        "#[cfg(feature = \"{feat}\")]\n\
         pub fn {fn_name}() -> GeneratedScheduleTable {{\n    GeneratedScheduleTable {{\n        entries: {module_name}::{const_name},\n        identity: {module_name}::CATALOG_IDENTITY,\n    }}\n}}\n"
    ))
}

fn emit_mod_wiring(specs: &[EmitSpec]) -> Result<String, String> {
    let mut out = emit_module_declarations(specs)?;
    let mut seen = std::collections::BTreeSet::new();
    for spec in specs {
        if !seen.insert(spec.module_name) {
            continue;
        }
        out.push_str(&emit_table_accessor(spec)?);
        out.push('\n');
    }
    Ok(out)
}

fn replace_between_markers(
    content: &str,
    begin: &str,
    end: &str,
    replacement: &str,
) -> Result<String, String> {
    let start = content
        .find(begin)
        .ok_or_else(|| format!("missing generated marker `{begin}`"))?
        + begin.len();
    let end_pos = content
        .find(end)
        .ok_or_else(|| format!("missing generated marker `{end}`"))?;
    if end_pos < start {
        return Err(format!(
            "generated markers `{begin}` and `{end}` are out of order"
        ));
    }
    let mut out = String::new();
    out.push_str(&content[..start]);
    out.push('\n');
    out.push_str(replacement.trim_end());
    out.push('\n');
    out.push_str(&content[end_pos..]);
    Ok(out)
}

/// Refresh the `@generated schedule module wiring` block in `mod.rs`.
pub fn refresh_generated_wiring(specs: &[EmitSpec], mod_path: &Path) -> Result<(), String> {
    let mod_src =
        fs::read_to_string(mod_path).map_err(|e| format!("read {}: {e}", mod_path.display()))?;
    let mod_wiring = emit_mod_wiring(specs)?;
    let mod_src = replace_between_markers(&mod_src, MOD_WIRING_BEGIN, MOD_WIRING_END, &mod_wiring)?;
    fs::write(mod_path, mod_src).map_err(|e| format!("write {}: {e}", mod_path.display()))?;
    Ok(())
}

/// Run `cargo fmt` on planner, schedules, and config after regen.
pub fn run_regen_fmt() -> Result<(), String> {
    for package in ["akita-planner", "akita-schedules", "akita-config"] {
        let status = Command::new("cargo")
            .args(["fmt", "-p", package])
            .status()
            .map_err(|e| format!("spawn cargo fmt: {e}"))?;
        if !status.success() {
            return Err(format!("cargo fmt -p {package} failed with {status}"));
        }
    }
    Ok(())
}

/// Write one family module to `spec.output_dir` and return its path.
pub fn write_family_module(spec: &EmitSpec) -> Result<PathBuf, String> {
    let body = emit_family_module(spec)?;
    let dest = spec
        .output_dir
        .join(format!("{}.rs", output_module_name(spec)));
    fs::write(&dest, &body).map_err(|e| format!("write {}: {e}", dest.display()))?;
    Ok(dest)
}
