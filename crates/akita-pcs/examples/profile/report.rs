use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_prover::{PreparedCrtNttProfile, PreparedNttCacheMetric};
use akita_types::{
    sis::compute_num_digits_field_width, CommitmentPayloadMode, CommitmentSliceCount,
    CommittedGroupParams, CommittedSourceEncoding, FoldSchedule, GrindingPlan,
    GroupOpenPhaseParams, InnerCommitSecurityRoute, NttTransformDomain, OpenCommitMatrixParams,
    OpeningMethod, PolynomialGroupLayout, RingRelationMode, SisModulusProfileId,
    SubringCoefficientPackingGeometry,
};

mod grinding;

pub(crate) fn report_timing(label: &str, phase: &str, elapsed_s: f64) {
    tracing::info!(label, elapsed_s, "{phase}");
    eprintln!("[{label}] {phase}: {elapsed_s:.6}s");
}

pub(crate) fn print_native_proof_summary(
    label: &str,
    proof: &[u8],
    schedule: &FoldSchedule,
    grinding_plan: &GrindingPlan,
) {
    grinding::emit_grinding_plan_report(label, grinding_plan);
    let levels = schedule.num_fold_levels();
    let nonce_max_bytes = grinding_plan.native_nonce_max_bytes();
    tracing::info!(
        label,
        levels,
        proof_size_bytes = proof.len(),
        native_nonce_max_bytes = nonce_max_bytes,
        "native proof summary"
    );
    eprintln!(
        "[{label}] proof: native_total={} bytes, native_nonce_messages_max={} bytes, levels={levels}",
        proof.len(),
        nonce_max_bytes,
    );
    #[cfg(feature = "logging-transcript")]
    print_native_wire_contexts(label);
}

#[cfg(feature = "logging-transcript")]
fn print_native_wire_contexts(label: &str) {
    use std::collections::BTreeMap;

    let mut wire = BTreeMap::<(u32, u32, u32), (u64, u64)>::new();
    for event in akita_transcript::thread_events() {
        let akita_transcript::TranscriptEvent::Context(record) = event;
        if matches!(
            record.kind,
            kind if kind == akita_transcript::ProtocolMessageKind::ProofLength as u32
                || kind == akita_transcript::ProtocolMessageKind::ProofAtoms as u32
                || kind == akita_transcript::ProtocolMessageKind::GrindingNonce as u32
                || kind == akita_transcript::ProtocolMessageKind::FoldResponseNonce as u32
        ) {
            let family = u32::from_le_bytes(record.site_id[..4].try_into().expect("site family"));
            let level = u32::from_le_bytes(record.site_id[8..12].try_into().expect("site level"));
            let entry = wire.entry((family, level, record.kind)).or_default();
            entry.0 += record.atom_count;
            entry.1 += record.encoded_bytes;
        }
    }
    for ((family, level, kind), (atoms, bytes)) in wire {
        eprintln!(
            "[{label}] native_wire_context: family={family} level={level} kind={kind} atoms={atoms} declared_bytes={bytes}"
        );
    }

    let mut ranges = akita_transcript::thread_proof_ranges();
    ranges.sort_unstable_by_key(|range| range.start);
    let mut cursor = 0usize;
    let mut nonce_bytes = 0usize;
    let mut non_nonce_bytes = 0usize;
    let mut complete = true;
    for range in ranges {
        let Some(end) = range.start.checked_add(range.len) else {
            complete = false;
            break;
        };
        if range.start != cursor {
            complete = false;
            break;
        }
        cursor = end;
        if matches!(
            range.context.kind,
            kind if kind == akita_transcript::ProtocolMessageKind::GrindingNonce as u32
                || kind == akita_transcript::ProtocolMessageKind::FoldResponseNonce as u32
        ) {
            nonce_bytes = nonce_bytes.saturating_add(range.len);
        } else {
            non_nonce_bytes = non_nonce_bytes.saturating_add(range.len);
        }
    }
    if complete {
        tracing::info!(
            label,
            accounted_bytes = cursor,
            native_nonce_bytes_observed = nonce_bytes,
            native_non_nonce_bytes_observed = non_nonce_bytes,
            "native proof byte accounting"
        );
    } else {
        tracing::warn!(label, "native proof byte accounting is incomplete");
    }
}

pub(crate) fn emit_native_proof_tail_report(label: &str, schedule: &FoldSchedule, field_bits: u32) {
    let response = &schedule.terminal.response_shape;
    let planned_bytes = akita_types::terminal_response_planner_bytes(
        field_bits,
        response,
        schedule.terminal.response_l2_sq_cap(),
    );
    tracing::info!(
        label,
        planned_terminal_response_bytes = planned_bytes,
        "native terminal response plan"
    );
    eprintln!(
        "[{label}] terminal_response: planned_max={planned_bytes} bytes, ring_dimension={}",
        response.layout.ring_dimension,
    );
}

/// Structured tail witness report for profile bench / CI (`scripts/profile_bench_report.py`).
pub(crate) fn report_setup_sizes(
    label: &str,
    num_setup_field_elements: usize,
    setup_vector_bytes: usize,
    ntt_cache_metrics: &[PreparedNttCacheMetric],
) {
    let setup_ntt_cache_bytes = ntt_cache_metrics
        .iter()
        .map(|metric| metric.cache_bytes)
        .sum::<usize>();
    tracing::info!(
        label,
        num_setup_field_elements,
        setup_vector_bytes,
        setup_ntt_cache_bytes,
        "setup sizes"
    );
    eprintln!(
        "[{label}] setup sizes: field_elems={num_setup_field_elements}, vector={setup_vector_bytes} bytes, ntt_cache={setup_ntt_cache_bytes} bytes"
    );
    for metric in ntt_cache_metrics {
        let domain = match metric.key.domain {
            NttTransformDomain::Negacyclic => "negacyclic",
            NttTransformDomain::Cyclic => "cyclic",
            NttTransformDomain::I16TailBothTransforms => "i16_tail_both",
            NttTransformDomain::ExactNegacyclicI16 { .. } => "exact_negacyclic_i16",
        };
        tracing::info!(
            label,
            ntt_cluster = "shared_cpu",
            ntt_ring_dimension = metric.key.ring_d,
            ntt_domain = domain,
            ntt_prefix_ring_elements = metric.key.num_ring_elements,
            ntt_prefix_field_elements = metric.key.num_ring_elements * metric.key.ring_d,
            ntt_cache_bytes = metric.cache_bytes,
            "exact NTT cache slot"
        );
        eprintln!(
            "[{label}] ntt cache: cluster=shared_cpu D={} domain={domain} ring_elems={} field_elems={} bytes={}",
            metric.key.ring_d,
            metric.key.num_ring_elements,
            metric.key.num_ring_elements * metric.key.ring_d,
            metric.cache_bytes,
        );
    }
}

pub(crate) fn report_verifier_ntt_cache_size(label: &str, verifier_ntt_cache_bytes: usize) {
    tracing::info!(label, verifier_ntt_cache_bytes, "verifier NTT cache size");
    eprintln!("[{label}] verifier NTT cache: ntt_cache={verifier_ntt_cache_bytes} bytes");
}

pub(crate) fn report_crt_profile(label: &str, profile: PreparedCrtNttProfile) {
    tracing::info!(
        label,
        crt_profile = profile.profile_id,
        crt_num_primes = profile.num_primes,
        crt_prime_modulus_bits = profile.prime_modulus_bits,
        crt_limb_bits = profile.limb_bits,
        max_i8_log_basis = profile.max_i8_log_basis,
        balanced_digit_safe_width = profile.balanced_digit_safe_width,
        raw_i8_safe_width = profile.raw_i8_safe_width,
        "CRT NTT profile"
    );
    eprintln!(
        "[{label}] CRT NTT profile: profile={}, primes={}, prime_modulus_bits={}, signed_storage_bits={}, max_i8_log_basis={}, balanced_digit_safe_width={}, raw_i8_safe_width={}",
        profile.profile_id,
        profile.num_primes,
        profile.prime_modulus_bits,
        profile.limb_bits,
        profile.max_i8_log_basis,
        profile.balanced_digit_safe_width,
        profile.raw_i8_safe_width
    );
}

/// One planner group as consumed by a fold row.
///
/// `consumer_level` identifies the fold whose parameters consume the group.
/// `emit` takes the producer row separately because a setup prefix is emitted
/// on the preceding fold row.
struct PlannedGroupReport {
    group: String,
    group_role: &'static str,
    consumer_level: usize,
    witness_field_elements: usize,
    public_num_vars: usize,
    public_num_polynomials: usize,
    d_a: usize,
    d_b: usize,
    d_d: usize,
    source_encoding: &'static str,
    extension_degree: usize,
    opening_method: &'static str,
    challenge_subring_dimension: Option<usize>,
    packing_factor: Option<usize>,
    packing_partial_width: Option<usize>,
    packing_quotient_width: Option<usize>,
    a_width: usize,
    b_width: usize,
    d_width: usize,
    n_a: usize,
    n_b: usize,
    n_d: usize,
    b_slice_count: usize,
    physical_b_input_width: usize,
    logical_b_rows: usize,
    complete_b_compression_bytes: Option<usize>,
    log_basis_inner: u32,
    log_basis_outer: u32,
    log_basis_open: u32,
    num_digits_inner: usize,
    num_digits_outer: usize,
    num_digits_open: usize,
    num_digits_fold: usize,
    challenge_l1_mass: usize,
    challenge_count_pm1: usize,
    challenge_count_pm2: usize,
    challenge_operator_norm_threshold: Option<u32>,
    num_live_ring_elements_per_claim: usize,
    num_live_blocks: usize,
    num_positions_per_block: usize,

    block_index_domain_size: usize,
    security_route: akita_types::InnerCommitSecurityRoute,
    response_l2_sq_cap: Option<u128>,
    norm_proof_shape: Option<akita_types::PhysicalL2NormProofShape>,
    setup_prefix_natural_field_elements: usize,
    setup_prefix_padded_field_elements: usize,
}

const fn source_encoding_name(source_encoding: CommittedSourceEncoding) -> &'static str {
    match source_encoding {
        CommittedSourceEncoding::CanonicalCoefficientTable => "canonical_coefficients",
        CommittedSourceEncoding::TensorSubfieldProjection { .. } => "tensor_subfield_projection",
    }
}

#[derive(Clone, Copy)]
struct OpeningReportGeometry {
    method: &'static str,
    challenge_subring_dimension: Option<usize>,
    packing_factor: Option<usize>,
    partial_width: Option<usize>,
    quotient_width: Option<usize>,
}

fn opening_report_geometry(
    opening_method: OpeningMethod,
    extension_degree: usize,
    a_ring_dimension: usize,
) -> Result<OpeningReportGeometry, AkitaError> {
    match opening_method {
        OpeningMethod::EvaluationTrace => Ok(OpeningReportGeometry {
            method: "evaluation_trace",
            challenge_subring_dimension: None,
            packing_factor: None,
            partial_width: None,
            quotient_width: None,
        }),
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } => {
            let geometry = SubringCoefficientPackingGeometry::try_new(
                extension_degree,
                a_ring_dimension,
                challenge_subring_dimension,
            )?;
            Ok(OpeningReportGeometry {
                method: "subring_coefficient_packing",
                challenge_subring_dimension: Some(geometry.challenge_subring_dimension()),
                packing_factor: Some(geometry.packing_factor()),
                partial_width: Some(geometry.partial_base_field_width()),
                quotient_width: Some(geometry.partial_base_field_width()),
            })
        }
    }
}

#[derive(Clone, Copy)]
struct BSliceReportGeometry {
    slice_count: usize,
    physical_input_width: usize,
    logical_rows: usize,
    complete_compression_bytes: Option<usize>,
}

fn b_slice_report_geometry(
    payload_mode: CommitmentPayloadMode,
    slice_count: CommitmentSliceCount,
    physical_output_rank: usize,
    physical_input_width: usize,
    outer_ring_dimension: usize,
    modulus_profile: SisModulusProfileId,
) -> Result<BSliceReportGeometry, AkitaError> {
    let logical_rows = slice_count.logical_output_rows(physical_output_rank)?;
    let complete_compression_bytes = if payload_mode.is_compressed() {
        let complete_source_coefficients =
            slice_count.complete_source_coefficients(physical_output_rank, outer_ring_dimension)?;
        Some(
            akita_types::CompressionChainPlan::for_complete_source(
                modulus_profile,
                complete_source_coefficients,
            )?
            .source_bytes(),
        )
    } else {
        None
    };
    Ok(BSliceReportGeometry {
        slice_count: slice_count.get(),
        physical_input_width,
        logical_rows,
        complete_compression_bytes,
    })
}

fn reported_operator_norm_threshold(
    security_route: InnerCommitSecurityRoute,
    ring_dimension: usize,
    challenge: &SparseChallengeConfig,
) -> Option<u32> {
    match security_route {
        InnerCommitSecurityRoute::Linf(_) => None,
        InnerCommitSecurityRoute::L2 { .. } => {
            akita_challenges::selective_l2_operator_norm_rejection(ring_dimension, challenge)
                .map(|policy| policy.threshold)
        }
    }
}

impl PlannedGroupReport {
    fn committed(
        group: String,
        group_role: &'static str,
        level: usize,
        witness_field_elements: usize,
        public_group: Option<PolynomialGroupLayout>,
        params: &CommittedGroupParams,
        extension_degree: usize,
    ) -> Result<Self, AkitaError> {
        let role_dims = params.role_dims();
        let opening =
            opening_report_geometry(params.opening_method(), extension_degree, role_dims.d_a())?;
        let security_route = params.inner().matrix.security_route();
        let (response_l2_sq_cap, norm_proof_shape) = match security_route {
            akita_types::InnerCommitSecurityRoute::Linf(_) => (None, None),
            akita_types::InnerCommitSecurityRoute::L2 {
                response_l2_sq_cap,
                norm_proof_shape,
                ..
            } => (Some(response_l2_sq_cap), Some(norm_proof_shape)),
        };
        let challenge_operator_norm_threshold = reported_operator_norm_threshold(
            security_route,
            role_dims.d_a(),
            &params.fold_challenge_config(),
        );
        let (public_num_vars, public_num_polynomials) = public_group
            .map(|layout| (layout.num_vars(), layout.num_polynomials()))
            .unwrap_or((0, 0));
        let n_b = params.outer().matrix.output_rank();
        let b_geometry = b_slice_report_geometry(
            params.payload_mode,
            params.outer_slice_count(),
            n_b,
            params.outer().matrix.input_width(),
            role_dims.d_b(),
            params.outer().matrix.sis_modulus_profile(),
        )?;
        Ok(Self {
            group,
            group_role,
            consumer_level: level,
            witness_field_elements,
            public_num_vars,
            public_num_polynomials,
            d_a: role_dims.d_a(),
            d_b: role_dims.d_b(),
            d_d: role_dims.d_d(),
            source_encoding: source_encoding_name(params.source_encoding),
            extension_degree,
            opening_method: opening.method,
            challenge_subring_dimension: opening.challenge_subring_dimension,
            packing_factor: opening.packing_factor,
            packing_partial_width: opening.partial_width,
            packing_quotient_width: opening.quotient_width,
            a_width: params.inner().matrix.input_width(),
            b_width: params.outer().matrix.input_width(),
            d_width: params.open().matrix.input_width(),
            n_a: params.inner().matrix.output_rank(),
            n_b,
            n_d: params.open().matrix.output_rank(),
            b_slice_count: b_geometry.slice_count,
            physical_b_input_width: b_geometry.physical_input_width,
            logical_b_rows: b_geometry.logical_rows,
            complete_b_compression_bytes: b_geometry.complete_compression_bytes,
            log_basis_inner: params.inner().digits.log_basis,
            log_basis_outer: params.outer().digits.log_basis,
            log_basis_open: params.open().digits.log_basis,
            num_digits_inner: params.inner().digits.num_digits,
            num_digits_outer: params.outer().digits.num_digits,
            num_digits_open: params.open().digits.num_digits,
            num_digits_fold: params.num_digits_fold(),
            challenge_l1_mass: params.challenge_l1_mass(),
            challenge_count_pm1: params.fold_challenge_config().count_pm1,
            challenge_count_pm2: params.fold_challenge_config().count_pm2,
            challenge_operator_norm_threshold,

            num_live_ring_elements_per_claim: params.blocks().live_ring_elements_per_claim,
            num_positions_per_block: params.blocks().positions_per_block,
            num_live_blocks: params.blocks().live_blocks,

            block_index_domain_size: params.block_index_domain_size().unwrap_or(0),
            security_route,
            response_l2_sq_cap,
            norm_proof_shape,
            setup_prefix_natural_field_elements: 0,
            setup_prefix_padded_field_elements: 0,
        })
    }

    fn precommitted(
        group: String,
        consumer_level: usize,
        witness_field_elements: usize,
        params: &GroupOpenPhaseParams,
        shared_open: &OpenCommitMatrixParams,
        setup_prefix_lengths: Option<(usize, usize)>,
        extension_degree: usize,
    ) -> Result<Self, AkitaError> {
        let layout = params.profile;
        let role_dims = params.role_dims(shared_open.ring_dimension());
        let opening = opening_report_geometry(
            params.opening.opening_method,
            extension_degree,
            role_dims.d_a(),
        )?;
        let (setup_prefix_natural_field_elements, setup_prefix_padded_field_elements) =
            setup_prefix_lengths.unwrap_or((0, 0));
        let (public_num_vars, public_num_polynomials) = setup_prefix_lengths
            .is_none()
            .then_some(layout.group)
            .map(|layout| (layout.num_vars(), layout.num_polynomials()))
            .unwrap_or((0, 0));
        let security_route = layout.inner.matrix.security_route();
        let (response_l2_sq_cap, norm_proof_shape) = match security_route {
            akita_types::InnerCommitSecurityRoute::Linf(_) => (None, None),
            akita_types::InnerCommitSecurityRoute::L2 {
                response_l2_sq_cap,
                norm_proof_shape,
                ..
            } => (Some(response_l2_sq_cap), Some(norm_proof_shape)),
        };
        let challenge_operator_norm_threshold = reported_operator_norm_threshold(
            security_route,
            role_dims.d_a(),
            &params.opening.fold_challenge_config,
        );
        let n_b = layout.outer.matrix.output_rank();
        let b_geometry = b_slice_report_geometry(
            CommitmentPayloadMode::Compressed,
            layout.outer_slice_count,
            n_b,
            layout.outer.matrix.input_width(),
            role_dims.d_b(),
            layout.outer.matrix.sis_modulus_profile(),
        )?;
        Ok(Self {
            group,
            group_role: if setup_prefix_lengths.is_some() {
                "setup_offload"
            } else {
                "precommitted"
            },
            consumer_level,
            witness_field_elements,
            public_num_vars,
            public_num_polynomials,
            d_a: role_dims.d_a(),
            d_b: role_dims.d_b(),
            d_d: role_dims.d_d(),
            source_encoding: source_encoding_name(
                akita_types::CommittedSourceEncoding::CanonicalCoefficientTable,
            ),
            extension_degree,
            opening_method: opening.method,
            challenge_subring_dimension: opening.challenge_subring_dimension,
            packing_factor: opening.packing_factor,
            packing_partial_width: opening.partial_width,
            packing_quotient_width: opening.quotient_width,
            a_width: layout.inner.matrix.input_width(),
            b_width: layout.outer.matrix.input_width(),
            d_width: shared_open.input_width(),
            n_a: layout.inner.matrix.output_rank(),
            n_b,
            n_d: shared_open.output_rank(),
            b_slice_count: b_geometry.slice_count,
            physical_b_input_width: b_geometry.physical_input_width,
            logical_b_rows: b_geometry.logical_rows,
            complete_b_compression_bytes: b_geometry.complete_compression_bytes,
            log_basis_inner: layout.inner.digits.log_basis,
            log_basis_outer: layout.outer.digits.log_basis,
            log_basis_open: params.opening.log_basis_open,
            num_digits_inner: layout.inner.digits.num_digits,
            num_digits_outer: layout.outer.digits.num_digits,
            num_digits_open: params.opening.num_digits_open,
            num_digits_fold: params.opening.num_digits_fold,
            challenge_l1_mass: params.challenge_l1_mass(),
            challenge_count_pm1: params.opening.fold_challenge_config.count_pm1,
            challenge_count_pm2: params.opening.fold_challenge_config.count_pm2,
            challenge_operator_norm_threshold,

            num_live_ring_elements_per_claim: layout.blocks.live_ring_elements_per_claim,
            num_positions_per_block: layout.blocks.positions_per_block,
            num_live_blocks: layout.blocks.live_blocks,

            block_index_domain_size: layout
                .blocks
                .live_blocks
                .checked_next_power_of_two()
                .unwrap_or(0),
            security_route,
            response_l2_sq_cap,
            norm_proof_shape,
            setup_prefix_natural_field_elements,
            setup_prefix_padded_field_elements,
        })
    }

    fn emit(&self, label: &str, level: usize, field_bits: u32, relation_mode: RingRelationMode) {
        let num_digits_quotient = match relation_mode {
            RingRelationMode::QuotientLift => {
                compute_num_digits_field_width(field_bits, self.log_basis_open)
            }
            RingRelationMode::ReducedEvaluation => 0,
        };
        tracing::info!(
            label,
            level,
            group = self.group.as_str(),
            group_role = self.group_role,
            consumer_level = self.consumer_level,
            witness_field_elements = self.witness_field_elements,
            public_num_vars = self.public_num_vars,
            public_num_polynomials = self.public_num_polynomials,
            d_a = self.d_a,
            d_b = self.d_b,
            d_d = self.d_d,
            source_encoding = self.source_encoding,
            extension_degree = self.extension_degree,
            opening_method = self.opening_method,
            challenge_subring_dimension = ?self.challenge_subring_dimension,
            packing_factor = ?self.packing_factor,
            packing_partial_width = ?self.packing_partial_width,
            packing_quotient_width = ?self.packing_quotient_width,
            a_width = self.a_width,
            b_width = self.b_width,
            d_width = self.d_width,
            n_a = self.n_a,
            n_b = self.n_b,
            n_d = self.n_d,
            b_slice_count = self.b_slice_count,
            physical_b_input_width = self.physical_b_input_width,
            logical_b_rows = self.logical_b_rows,
            complete_b_compression_bytes = ?self.complete_b_compression_bytes,
            log_basis_inner = self.log_basis_inner,
            log_basis_outer = self.log_basis_outer,
            log_basis_open = self.log_basis_open,
            num_digits_inner = self.num_digits_inner,
            num_digits_outer = self.num_digits_outer,
            num_digits_open = self.num_digits_open,
            num_digits_fold = self.num_digits_fold,
            relation_mode = relation_mode.as_str(),
            num_digits_quotient,
            challenge_l1_mass = self.challenge_l1_mass,
            challenge_count_pm1 = self.challenge_count_pm1,
            challenge_count_pm2 = self.challenge_count_pm2,
            challenge_operator_norm_threshold = ?self.challenge_operator_norm_threshold,
            num_live_ring_elements_per_claim = self.num_live_ring_elements_per_claim,
            num_live_blocks = self.num_live_blocks,
            num_positions_per_block = self.num_positions_per_block,
            block_index_domain_size = self.block_index_domain_size,
            security_route = ?self.security_route,
            response_l2_sq_cap = ?self.response_l2_sq_cap,
            norm_proof_shape = ?self.norm_proof_shape,
            setup_prefix_natural_field_elements = self.setup_prefix_natural_field_elements,
            setup_prefix_padded_field_elements = self.setup_prefix_padded_field_elements,
            "planned fold group"
        );
    }
}

pub(crate) fn emit_runtime_schedule_summary(
    label: &str,
    schedule: &FoldSchedule,
    final_group: PolynomialGroupLayout,
    field_bits: u32,
    extension_degree: usize,
) -> Result<(), AkitaError> {
    let challenge_field_bits = field_bits
        .checked_mul(
            u32::try_from(extension_degree).map_err(|_| {
                AkitaError::InvalidSetup("profile extension degree exceeds u32".into())
            })?,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("profile challenge field width overflow".into()))?;
    let levels = schedule.num_fold_levels();
    let num_setup_field_elements =
        akita_types::setup_matrix_field_elements_for_schedule(schedule).unwrap_or(0);
    let num_setup_bytes = num_setup_field_elements.saturating_mul(field_bits.div_ceil(8) as usize);
    let selected_offload_edges = schedule
        .recursive_folds
        .iter()
        .filter(|fold| fold.params.setup_prefix().is_some())
        .count();
    tracing::info!(
        label,
        levels,
        selected_offload_edges,
        num_setup_field_elements,
        num_setup_bytes,
        "runtime schedule"
    );

    let root_current_w_groups = root_current_w_groups(schedule, final_group);
    let root_open = &schedule.root.params.open().matrix;
    for (index, group) in schedule
        .root
        .params
        .precommitted_groups()
        .iter()
        .enumerate()
    {
        let layout = group.profile.group;
        let witness_field_elements =
            group_field_elements(layout.num_vars(), layout.num_polynomials());
        PlannedGroupReport::precommitted(
            format!("pre{index}"),
            0,
            witness_field_elements,
            group,
            root_open,
            None,
            extension_degree,
        )?
        .emit(
            label,
            0,
            field_bits,
            schedule.root.params.ring_relation_mode,
        );
    }
    PlannedGroupReport::committed(
        "final".to_string(),
        "final",
        0,
        group_field_elements(final_group.num_vars(), final_group.num_polynomials()),
        Some(final_group),
        &schedule.root.params,
        extension_degree,
    )?
    .emit(
        label,
        0,
        field_bits,
        schedule.root.params.ring_relation_mode,
    );
    for (index, fold) in schedule.recursive_folds.iter().enumerate() {
        PlannedGroupReport::committed(
            "folded".to_string(),
            "folded",
            index + 1,
            fold.input_witness_len,
            None,
            &fold.params,
            extension_degree,
        )?
        .emit(label, index + 1, field_bits, fold.params.ring_relation_mode);
        if let Some(prefix) = &fold.params.setup_prefix() {
            PlannedGroupReport::precommitted(
                format!("setup_to_L{}", index + 1),
                index + 1,
                prefix.setup_natural_len.expect("setup prefix group"),
                prefix,
                &fold.params.open().matrix,
                Some((
                    prefix.setup_natural_len.expect("setup prefix group"),
                    prefix.n_prefix().unwrap_or(0),
                )),
                extension_degree,
            )?
            .emit(label, index, field_bits, fold.params.ring_relation_mode);
        }
    }
    let nonterminal = std::iter::once((
        0usize,
        &schedule.root.params,
        schedule.root.input_witness_len,
        schedule.root.output_witness_len,
        root_current_w_groups,
    ))
    .chain(
        schedule
            .recursive_folds
            .iter()
            .enumerate()
            .map(|(index, level)| {
                (
                    index + 1,
                    &level.params,
                    level.input_witness_len,
                    level.output_witness_len,
                    format!("folded={}", level.input_witness_len),
                )
            }),
    );
    for (level_idx, lp, input_witness_len, output_witness_len, current_w_groups) in nonterminal {
        let role_dims = lp.role_dims();
        let opening =
            opening_report_geometry(lp.opening_method(), extension_degree, role_dims.d_a())?;
        let extension_opening_reduction_bytes =
            if matches!(lp.opening_method(), OpeningMethod::EvaluationTrace) {
                let final_group = akita_types::PolynomialGroupLayout::singleton(
                    akita_types::padded_boolean_opening_vars(input_witness_len)?,
                );
                let opening_shape = lp
                    .opening_layout_for_final_group(final_group)?
                    .aggregate_polynomial_group_layout()?;
                akita_types::extension_opening_reduction_level_bytes(
                    challenge_field_bits,
                    extension_degree,
                    opening_shape,
                )?
            } else {
                0
            };
        let current_w_len = current_w_groups;
        let next_w_len = output_witness_len;
        let setup_prefix = schedule
            .recursive_folds
            .get(level_idx)
            .and_then(|fold| fold.params.setup_prefix());
        let setup_prefix_natural_field_elements = setup_prefix.map_or(0, |prefix| {
            prefix.setup_natural_len.expect("setup prefix group")
        });
        let setup_prefix_padded_field_elements =
            setup_prefix.map_or(0, |prefix| prefix.n_prefix().unwrap_or(0));
        let a_input_raw_dimension = lp.inner().matrix.raw_input_dimension();
        let a_output_raw_dimension = lp.inner().matrix.raw_output_dimension();
        let b_input_raw_dimension = lp.outer().matrix.raw_input_dimension();
        let b_output_raw_dimension = lp.outer().matrix.raw_output_dimension();
        let d_input_raw_dimension = lp.open().matrix.raw_input_dimension();
        let d_output_raw_dimension = lp.open().matrix.raw_output_dimension();
        let security_route = lp.inner().matrix.security_route();
        let (response_l2_sq_cap, norm_proof_shape) = match security_route {
            akita_types::InnerCommitSecurityRoute::Linf(_) => (None, None),
            akita_types::InnerCommitSecurityRoute::L2 {
                response_l2_sq_cap,
                norm_proof_shape,
                ..
            } => (Some(response_l2_sq_cap), Some(norm_proof_shape)),
        };
        let challenge_operator_norm_threshold = reported_operator_norm_threshold(
            security_route,
            role_dims.d_a(),
            &lp.fold_challenge_config(),
        );
        let b_geometry = b_slice_report_geometry(
            lp.payload_mode,
            lp.outer_slice_count(),
            lp.outer().matrix.output_rank(),
            lp.outer().matrix.input_width(),
            role_dims.d_b(),
            lp.outer().matrix.sis_modulus_profile(),
        )?;
        let relation_mode = lp.ring_relation_mode;
        let num_digits_quotient = match relation_mode {
            RingRelationMode::QuotientLift => {
                compute_num_digits_field_width(field_bits, lp.open().digits.log_basis)
            }
            RingRelationMode::ReducedEvaluation => 0,
        };
        tracing::info!(
            label,
            level = level_idx,
            d = lp.d_a(),
            d_a = role_dims.d_a(),
            d_b = role_dims.d_b(),
            d_d = role_dims.d_d(),
            source_encoding = source_encoding_name(lp.source_encoding),
            extension_degree,
            witness_chunk_count = lp.witness_chunk.num_chunks,
            witness_chunk_activated_levels = lp.witness_chunk.num_activated_levels,
            witness_chunk_active = lp.witness_chunk.uses_multi_chunk(),
            opening_method = opening.method,
            challenge_subring_dimension = ?opening.challenge_subring_dimension,
            packing_factor = ?opening.packing_factor,
            packing_partial_width = ?opening.partial_width,
            packing_quotient_width = ?opening.quotient_width,
            extension_opening_reduction_present = extension_opening_reduction_bytes != 0,
            extension_opening_reduction_bytes,
            a_width = lp.inner().matrix.input_width(),
            b_width = lp.outer().matrix.input_width(),
            d_width = lp.open().matrix.input_width(),
            n_a = lp.inner().matrix.output_rank(),
            n_b = lp.outer().matrix.output_rank(),
            n_d = lp.open().matrix.output_rank(),
            b_slice_count = b_geometry.slice_count,
            physical_b_input_width = b_geometry.physical_input_width,
            logical_b_rows = b_geometry.logical_rows,
            complete_b_compression_bytes = ?b_geometry.complete_compression_bytes,
            security_route = ?security_route,
            response_l2_sq_cap = ?response_l2_sq_cap,
            norm_proof_shape = ?norm_proof_shape,
            ?a_input_raw_dimension,
            ?a_output_raw_dimension,
            ?b_input_raw_dimension,
            ?b_output_raw_dimension,
            ?d_input_raw_dimension,
            ?d_output_raw_dimension,
            challenge_l1_mass = lp.challenge_l1_mass(),
            challenge_count_pm1 = lp.fold_challenge_config().count_pm1,
            challenge_count_pm2 = lp.fold_challenge_config().count_pm2,
            challenge_operator_norm_threshold = ?challenge_operator_norm_threshold,
            log_basis_inner = lp.inner().digits.log_basis,
            log_basis_outer = lp.outer().digits.log_basis,
            log_basis_open = lp.open().digits.log_basis,
            position_index_bits = lp.position_index_bits(),
            block_index_bits = lp.block_index_bits(),
            num_live_ring_elements_per_claim = lp.blocks().live_ring_elements_per_claim,
            num_live_blocks = lp.blocks().live_blocks,
            block_index_domain_size = lp.block_index_domain_size().unwrap_or(0),
            num_positions_per_block = lp.blocks().positions_per_block,
            num_digits_inner = lp.inner().digits.num_digits,
            num_digits_outer = lp.outer().digits.num_digits,
            num_digits_open = lp.open().digits.num_digits,
            delta_fold = lp.num_digits_fold(),
            relation_mode = relation_mode.as_str(),
            num_digits_quotient,
            input_witness_len,
            output_witness_len,
            current_w_len,
            next_w_len,
            setup_prefix_natural_field_elements,
            setup_prefix_padded_field_elements,
            "planned fold level"
        );
    }

    let terminal_level = levels - 1;
    let terminal = &schedule.terminal;
    let witness = &terminal;
    let challenge = &terminal.fold_challenge_config;
    let security_route = witness.inner.matrix.security_route();
    let response_l2_sq_cap = witness.response_l2_sq_cap();
    let z_linf_cap = terminal
        .response_shape
        .layout
        .groups
        .first()
        .and_then(|group| group.z_linf_cap);
    let challenge_operator_norm_threshold =
        reported_operator_norm_threshold(security_route, witness.d_a(), challenge);
    tracing::info!(
        label,
        level = terminal_level,
        input_witness_len = terminal.input_witness_len,
        d_a = witness.d_a(),
        n_a = witness.inner.matrix.output_rank(),
        inner_width = witness.inner_width(),
        a_input_raw_dimension = ?witness.inner.matrix.raw_input_dimension(),
        a_output_raw_dimension = ?witness.inner.matrix.raw_output_dimension(),
        log_basis_inner = witness.inner.digits.log_basis,
        num_digits_inner = witness.inner.digits.num_digits,
        fold_log_basis = witness.fold.log_basis,
        fold_digit_count = witness.fold.num_digits,
        challenge_l1_mass = challenge.l1_norm(),
        challenge_count_pm1 = challenge.count_pm1,
        challenge_count_pm2 = challenge.count_pm2,
        challenge_operator_norm_threshold = ?challenge_operator_norm_threshold,
        security_route = ?security_route,
        response_l2_sq_cap = ?response_l2_sq_cap,
        z_linf_cap = ?z_linf_cap,
        num_live_ring_elements_per_claim = witness.blocks.live_ring_elements_per_claim,
        num_positions_per_block = witness.blocks.positions_per_block,
        num_live_blocks = witness.blocks.live_blocks,
        block_index_domain_size = witness
            .blocks.live_blocks
            .checked_next_power_of_two()
            .unwrap_or(0),
        "planned terminal state"
    );
    Ok(())
}

fn group_field_elements(num_vars: usize, num_polynomials: usize) -> usize {
    1usize
        .checked_shl(num_vars as u32)
        .and_then(|len| len.checked_mul(num_polynomials))
        .unwrap_or(0)
}

fn root_current_w_groups(schedule: &FoldSchedule, final_group: PolynomialGroupLayout) -> String {
    let mut groups = schedule
        .root
        .params
        .precommitted_groups()
        .iter()
        .enumerate()
        .map(|(index, group)| {
            let layout = group.profile.group;
            format!(
                "pre{index}={}",
                group_field_elements(layout.num_vars(), layout.num_polynomials())
            )
        })
        .collect::<Vec<_>>();
    groups.push(format!(
        "final={}",
        group_field_elements(final_group.num_vars(), final_group.num_polynomials())
    ));
    groups.join(";")
}
pub(crate) fn print_layout(
    layout: &CommittedGroupParams,
    _num_claims: usize,
    _field_bits: u32,
) -> Result<(), AkitaError> {
    let b_geometry = b_slice_report_geometry(
        layout.payload_mode,
        layout.outer_slice_count(),
        layout.outer().matrix.output_rank(),
        layout.outer().matrix.input_width(),
        layout.outer().matrix.ring_dimension(),
        layout.outer().matrix.sis_modulus_profile(),
    )?;
    tracing::debug!(
        position_index_bits = layout.position_index_bits(),
        block_index_bits = layout.block_index_bits(),
        num_live_ring_elements_per_claim = layout.blocks().live_ring_elements_per_claim,
        num_live_blocks = layout.blocks().live_blocks,
        block_index_domain_size = layout.block_index_domain_size().unwrap_or(0),
        num_positions_per_block = layout.blocks().positions_per_block,
        num_digits_inner = layout.inner().digits.num_digits,
        num_digits_outer = layout.outer().digits.num_digits,
        num_digits_open = layout.open().digits.num_digits,
        delta_fold = layout.num_digits_fold(),
        log_basis_inner = layout.inner().digits.log_basis,
        log_basis_outer = layout.outer().digits.log_basis,
        log_basis_open = layout.open().digits.log_basis,
        n_a = layout.inner().matrix.output_rank(),
        n_b = layout.outer().matrix.output_rank(),
        n_d = layout.open().matrix.output_rank(),
        b_slice_count = b_geometry.slice_count,
        physical_b_input_width = b_geometry.physical_input_width,
        logical_b_rows = b_geometry.logical_rows,
        complete_b_compression_bytes = b_geometry.complete_compression_bytes,
        "layout"
    );
    Ok(())
}
