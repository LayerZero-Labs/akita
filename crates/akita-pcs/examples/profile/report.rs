use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_prover::{PreparedCrtNttProfile, PreparedNttCacheMetric};
use akita_serialization::{AkitaSerialize, Compress};
use akita_types::{
    golomb_rice::{analyze_z_fold_golomb_encoding, golomb_rice_zigzag_width},
    layout::proof_size::field_bytes,
    sis::{compute_num_digits_field_width, num_digits_for_bound},
    AkitaBatchedProof, CommitmentPayloadMode, CommitmentSliceCount, CommittedGroupParams,
    CommittedSourceEncoding, FoldLevelProof, FoldSchedule, GrindingPlan, GrindingQueryKind,
    GrindingSite, GroupOpenPhaseParams, InnerCommitSecurityRoute, NttTransformDomain,
    OpenCommitMatrixParams, OpeningMethod, PolynomialGroupLayout, RingRelationMode,
    SetupSumcheckProof, SisModulusProfileId, SubringCoefficientPackingGeometry, TerminalLevelProof,
    ZFoldEncodingStats,
};
use jolt_field::{CanonicalEncoding, Field};
use std::collections::BTreeMap;

mod grinding;

pub(crate) fn report_timing(label: &str, phase: &str, elapsed_s: f64) {
    tracing::info!(label, elapsed_s, "{phase}");
    eprintln!("[{label}] {phase}: {elapsed_s:.6}s");
}

/// Structured tail witness report for profile bench / CI (`scripts/profile_bench_report.py`).
pub(crate) fn emit_proof_tail_report<FF, E>(
    label: &str,
    proof: &AkitaBatchedProof<FF, E>,
    schedule: &FoldSchedule,
    field_bits: u32,
) where
    FF: Field + CanonicalEncoding + AkitaSerialize,
    E: Field,
{
    let final_w = proof.terminal_response();
    let tail_bytes = final_w.serialized_size(Compress::No);
    let num_elems = final_w.num_elems();

    {
        let segment = final_w;
        let field_sz = field_bytes(FF::MODULUS_BITS);
        let ring_dim = segment.layout.ring_dimension;
        let z_golomb_bytes = segment.z_payloads.iter().map(Vec::len).sum::<usize>();
        let z_field_elems = segment.layout.z_coords();
        let z_ring_elems = z_field_elems / ring_dim.max(1);
        let e_field_elems = segment.e_fields.coeff_len();
        let t_field_elems = segment.t_fields.coeff_len();
        let e_ring_elems = e_field_elems / ring_dim.max(1);
        let t_ring_elems = t_field_elems / ring_dim.max(1);
        let e_bytes = e_field_elems.saturating_mul(field_sz);
        let t_bytes = t_field_elems.saturating_mul(field_sz);
        let z_wire_bytes = tail_bytes.saturating_sub(e_bytes.saturating_add(t_bytes));
        let z_prefix_bytes = z_wire_bytes.saturating_sub(z_golomb_bytes);
        let z_budget_bytes = schedule.terminal.response_shape.layout.z_payload_bytes();
        let z_slack_bytes = z_budget_bytes.saturating_sub(z_golomb_bytes);
        let z_stats = terminal_response_z_fold_stats(segment, schedule, field_bits).ok();
        let z_linf_cap = segment
            .layout
            .groups
            .first()
            .and_then(|group| group.z_linf_cap);
        let z_rice_low_bits_wire = segment
            .layout
            .groups
            .first()
            .map(|group| group.z_rice_low_bits)
            .unwrap_or(0);
        let z_rice_low_bits_cap =
            z_linf_cap.and_then(|_| z_stats.as_ref().map(|stats| stats.rice_low_bits_cap));
        let z_stats_coords = z_stats.as_ref().map(|s| s.coord_count).unwrap_or(0);
        let z_bits_per_coord_golomb = z_stats
            .as_ref()
            .map(|s| s.bits_per_coord_at_wire)
            .unwrap_or(0.0);
        let z_bits_per_coord_packed = z_stats
            .as_ref()
            .map(|s| s.bits_per_coord_packed_digits)
            .unwrap_or(0.0);
        let z_packed_hypothetical_bytes = z_stats
            .as_ref()
            .map(|s| s.total_bits_packed_digits.div_ceil(8))
            .unwrap_or(0);
        let z_golomb_savings_bytes = z_packed_hypothetical_bytes.saturating_sub(z_golomb_bytes);

        tracing::info!(
            label,
            tail_bytes,
            final_w_num_elems = num_elems,
            final_w_encoding = "terminal_response",
            final_w_policy = "non_zk_default",
            tail_log_basis_inner = schedule.terminal.inner.digits.log_basis,
            tail_z_prefix_bytes = z_prefix_bytes,
            tail_z_golomb_bytes = z_golomb_bytes,
            tail_z_bytes = z_wire_bytes,
            tail_z_field_elems = z_field_elems,
            tail_z_ring_elems = z_ring_elems,
            tail_z_budget_bytes = z_budget_bytes,
            tail_z_slack_bytes = z_slack_bytes,
            tail_e_field_elems = e_field_elems,
            tail_e_ring_elems = e_ring_elems,
            tail_t_field_elems = t_field_elems,
            tail_t_ring_elems = t_ring_elems,
            tail_e_bytes = e_bytes,
            tail_t_bytes = t_bytes,
            z_linf_cap = ?z_linf_cap,
            z_rice_low_bits_wire,
            z_rice_low_bits_cap = ?z_rice_low_bits_cap,
            z_coords = z_stats_coords,
            z_bits_per_coord_golomb,
            z_bits_per_coord_packed,
            z_packed_hypothetical_bytes,
            z_golomb_savings_bytes,
            "proof tail summary"
        );

        let golomb_line = z_stats
            .map(|stats| {
                format!(
                    " Golomb z: coefficient_linf_cap={z_linf_cap:?} wire_low_bits={z_rice_low_bits_wire} sample_low_bits={} ring_elems={z_ring_elems} field_coeffs={} \
                     {:.2} bits/coord@wire vs {:.2}@sample vs packed {:.2} bits/field_coeff \
                     (hypothetical packed z={} B, savings={} B); \
                     planner z budget={z_budget_bytes} B (slack {z_slack_bytes} B); \
                     dist max={} median={} p90={} p99={}",
                    stats.rice_low_bits_sample,
                    stats.coord_count,
                    stats.bits_per_coord_at_wire,
                    stats.bits_per_coord_at_sample,
                    stats.bits_per_coord_packed_digits,
                    stats.total_bits_packed_digits.div_ceil(8),
                    stats
                        .total_bits_packed_digits
                        .div_ceil(8)
                        .saturating_sub(z_golomb_bytes),
                    stats.observed_max_abs,
                    stats.median_abs,
                    stats.p90_abs,
                    stats.p99_abs,
                )
            })
            .unwrap_or_default();

        eprintln!(
            "[{label}]   final_w: encoding=terminal_response (non-zk default), total={tail_bytes} bytes, \
             logical_elems={num_elems}, inner_log_basis={}{}",
            schedule.terminal.inner.digits.log_basis,
            golomb_line,
        );
        eprintln!(
            "[{label}]     z: {z_wire_bytes} B (len_prefix={z_prefix_bytes} + golomb={z_golomb_bytes}), \
             field_coeffs={z_field_elems}, ring_elems={z_ring_elems}",
        );
        eprintln!(
            "[{label}]     e: {e_bytes} B, field_coeffs={e_field_elems}, ring_elems={e_ring_elems}",
        );
        eprintln!(
            "[{label}]     t: {t_bytes} B, field_coeffs={t_field_elems}, ring_elems={t_ring_elems}",
        );
        assert_eq!(tail_bytes, z_wire_bytes + e_bytes + t_bytes);
    }
}

fn terminal_response_z_fold_stats<FF: Field>(
    witness: &akita_types::TerminalResponse<FF>,
    schedule: &FoldSchedule,
    field_bits: u32,
) -> Result<ZFoldEncodingStats, akita_error::AkitaError> {
    let params = &schedule.terminal;
    let group = witness
        .layout
        .groups
        .first()
        .ok_or(akita_error::AkitaError::InvalidProof)?;
    let encoding_abs_bound = group.z_linf_cap.unwrap_or(i16::MAX as u128);
    let z_values = akita_types::decode_terminal_z_golomb_payload(
        witness
            .z_payloads
            .first()
            .ok_or(akita_error::AkitaError::InvalidProof)?,
        group,
    )?
    .into_iter()
    .map(i64::from)
    .collect::<Vec<_>>();
    let log_cap = u128::BITS - encoding_abs_bound.leading_zeros();
    let hypothetical_digits =
        num_digits_for_bound(log_cap, field_bits, params.inner.digits.log_basis).max(1);
    analyze_z_fold_golomb_encoding(
        &z_values,
        encoding_abs_bound,
        group.z_rice_low_bits,
        golomb_rice_zigzag_width(encoding_abs_bound),
        hypothetical_digits,
        params.inner.digits.log_basis,
        witness.z_payloads.first().map_or(0, Vec::len),
    )
}

/// Surface the public setup prefix and every initialized exact NTT cache slot.
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

fn ring_elem_count(coeff_len: usize, d: usize) -> usize {
    coeff_len / d
}

fn extension_opening_reduction_sizes<E: Field + AkitaSerialize>(
    reduction: Option<&akita_types::ExtensionOpeningReductionProof<E>>,
) -> (usize, usize, usize) {
    reduction.map_or((0, 0, 0), |reduction| {
        let partials = reduction
            .partials
            .iter()
            .map(|value| value.serialized_size(Compress::No))
            .sum();
        let sumcheck = reduction.sumcheck.serialized_size(Compress::No);
        let final_claims = reduction
            .final_claims
            .iter()
            .map(|value| value.serialized_size(Compress::No))
            .sum();
        (partials, sumcheck, final_claims)
    })
}

fn stage3_sumcheck_size<E: Field + AkitaSerialize>(proof: Option<&SetupSumcheckProof<E>>) -> usize {
    proof.map_or(0, |proof| {
        proof.claim.serialized_size(Compress::No)
            + proof.setup_prefix_eval.serialized_size(Compress::No)
            + proof.sumcheck.serialized_size(Compress::No)
    })
}

/// Total serialized bytes of the recursive-mode stage-3 setup-product
/// sumcheck payloads across every non-terminal fold level (the folded root and
/// each intermediate step). This is the proof-size overhead that
/// `SetupContributionMode::Recursive` adds on top of the direct-mode payload
/// priced by `akita_types::level_proof_bytes`; terminal levels carry no
/// stage-3 proof and contribute zero.
pub(crate) fn observed_stage3_setup_product_bytes<FF, E>(proof: &AkitaBatchedProof<FF, E>) -> usize
where
    FF: Field + CanonicalEncoding + AkitaSerialize,
    E: Field + AkitaSerialize,
{
    let root_bytes = stage3_sumcheck_size(proof.root.stage3_sumcheck_proof.as_ref());
    let step_bytes: usize = proof
        .recursive_folds
        .iter()
        .map(|step| stage3_sumcheck_size(step.stage3_sumcheck_proof.as_ref()))
        .sum();
    root_bytes + step_bytes
}

fn print_akita_level_breakdown<FF, E>(
    label: &str,
    level_idx: usize,
    level: &FoldLevelProof<FF, E>,
    ring_d: usize,
    fold_response_nonce: u32,
) -> usize
where
    FF: Field + CanonicalEncoding + AkitaSerialize,
    E: Field + AkitaSerialize,
{
    let (
        extension_opening_partials_size,
        extension_opening_sumcheck_size,
        extension_opening_final_claims_size,
    ) = extension_opening_reduction_sizes(level.extension_opening_reduction.as_ref());
    let opening_payload_size = level.opening_payload.serialized_size(Compress::No);
    let opening_payload_d = level.opening_payload.coeff_len();
    let total = level.serialized_size(Compress::No);
    let stage2_intermediate = &level.stage2;

    eprintln!("[{label}]   akita_fold L{level_idx}: total={total} bytes");
    eprintln!(
        "[{label}]     p_H={} bytes ({} ring elems, D={})",
        opening_payload_size,
        ring_elem_count(level.opening_payload.coeff_len(), opening_payload_d),
        opening_payload_d,
    );
    let stage1 = &level.stage1;
    let stage1_sumcheck_size = stage1
        .stages
        .iter()
        .map(|stage| stage.sumcheck_proof.serialized_size(Compress::No))
        .sum::<usize>();
    let stage1_interstage_claims_size = stage1
        .stages
        .iter()
        .flat_map(|stage| stage.child_claims.iter())
        .map(|claim| claim.serialized_size(Compress::No))
        .sum::<usize>();
    let stage1_range_image_evaluation_size =
        stage1.range_image_evaluation.serialized_size(Compress::No);
    let (stage1_norm_proof_size, response_l2_sq) =
        stage1.norm_proof.as_ref().map_or((0, None), |norm| {
            (
                norm.serialized_size(Compress::No),
                Some(norm.response_l2_sq),
            )
        });
    let stage2_sumcheck_size = stage2_intermediate
        .sumcheck_proof
        .serialized_size(Compress::No);
    let stage3_sumcheck_size = stage3_sumcheck_size(level.stage3_sumcheck_proof.as_ref());
    let next_w_payload = stage2_intermediate.next_witness_binding.outer_payload();
    let next_w_payload_size = next_w_payload
        .map(|payload| payload.serialized_size(Compress::No))
        .unwrap_or(0);
    let next_w_payload_coeffs = next_w_payload.map_or(0, akita_types::RingVec::coeff_len);
    let next_w_eval_size = stage2_intermediate
        .next_w_eval()
        .serialized_size(Compress::No);
    tracing::info!(
        label,
        level = level_idx,
        d = ring_d,
        total_bytes = total,
        extension_opening_partials_bytes = extension_opening_partials_size,
        extension_opening_sumcheck_bytes = extension_opening_sumcheck_size,
        extension_opening_final_claims_bytes = extension_opening_final_claims_size,
        opening_payload_bytes = opening_payload_size,
        grind_nonce = fold_response_nonce,
        grind_attempts = u64::from(fold_response_nonce) + 1,
        stage1_sumcheck_bytes = stage1_sumcheck_size,
        stage1_interstage_claims_bytes = stage1_interstage_claims_size,
        stage1_range_image_evaluation_bytes = stage1_range_image_evaluation_size,
        stage1_norm_proof_bytes = stage1_norm_proof_size,
        response_l2_sq = ?response_l2_sq,
        stage2_sumcheck_bytes = stage2_sumcheck_size,
        stage3_sumcheck_bytes = stage3_sumcheck_size,
        next_w_payload_bytes = next_w_payload_size,
        next_w_eval_bytes = next_w_eval_size,
        "proof fold level"
    );
    eprintln!("[{label}]     extension_opening_partials={extension_opening_partials_size} bytes");
    eprintln!("[{label}]     extension_opening_sumcheck={extension_opening_sumcheck_size} bytes");
    eprintln!(
        "[{label}]     extension_opening_final_claims={extension_opening_final_claims_size} bytes"
    );
    eprintln!("[{label}]     stage1_sumcheck={stage1_sumcheck_size} bytes");
    eprintln!("[{label}]     stage1_interstage_claims={stage1_interstage_claims_size} bytes");
    eprintln!(
        "[{label}]     stage1_range_image_evaluation={stage1_range_image_evaluation_size} bytes"
    );
    eprintln!("[{label}]     stage1_norm_proof={stage1_norm_proof_size} bytes");
    eprintln!("[{label}]     stage2_sumcheck={stage2_sumcheck_size} bytes");
    eprintln!("[{label}]     stage3_sumcheck={stage3_sumcheck_size} bytes");
    eprintln!(
        "[{label}]     next_w_payload={next_w_payload_size} bytes ({} coeffs)",
        next_w_payload_coeffs,
    );
    eprintln!("[{label}]     next_w_eval={next_w_eval_size} bytes");
    assert_eq!(
        total,
        extension_opening_partials_size
            + extension_opening_sumcheck_size
            + extension_opening_final_claims_size
            + opening_payload_size
            + stage1_sumcheck_size
            + stage1_interstage_claims_size
            + stage1_range_image_evaluation_size
            + stage1_norm_proof_size
            + stage2_sumcheck_size
            + stage3_sumcheck_size
            + next_w_payload_size
            + next_w_eval_size
    );
    total
}

fn print_terminal_level_breakdown<FF, E>(
    label: &str,
    level_idx: usize,
    level: &TerminalLevelProof<FF, E>,
    root_variant: &'static str,
    ring_d: usize,
    fold_response_nonce: u32,
) -> usize
where
    FF: Field + CanonicalEncoding + AkitaSerialize,
    E: Field + AkitaSerialize,
{
    let (
        extension_opening_partials_size,
        extension_opening_sumcheck_size,
        extension_opening_final_claims_size,
    ) = extension_opening_reduction_sizes(level.extension_opening_reduction.as_ref());
    let terminal_response_size = level.terminal_response().serialized_size(Compress::No);
    let response_l2_sq = level
        .terminal_response()
        .layout
        .groups
        .first()
        .and_then(|group| {
            akita_types::decode_terminal_z_golomb_payload(
                level.terminal_response().z_payloads.first()?,
                group,
            )
            .ok()
        })
        .and_then(|values| akita_types::sis::checked_centered_l2_sq(&values));
    let full = level.serialized_size(Compress::No);
    // `total_bytes` excludes the terminal response to mirror the planner's
    // `terminal_level_proof_bytes`. The response is reported separately as
    // the proof tail (`tail_bytes`) and accounted for in `accounted_bytes`.
    let total = full - terminal_response_size;

    // Only the fields structurally present in `TerminalLevelProof` are
    // emitted: optional extension-opening reduction and the terminal response.
    // The intermediate-level
    // fields (`v`, `stage1_*`, `stage3_sumcheck`, `next_w_*`) are absent at
    // terminal and therefore omitted from the tracing payload; downstream
    // parsers default missing keys to zero.
    tracing::info!(
        label,
        level = level_idx,
        d = ring_d,
        total_bytes = total,
        extension_opening_partials_bytes = extension_opening_partials_size,
        extension_opening_sumcheck_bytes = extension_opening_sumcheck_size,
        extension_opening_final_claims_bytes = extension_opening_final_claims_size,
        grind_nonce = fold_response_nonce,
        grind_attempts = u64::from(fold_response_nonce) + 1,
        response_l2_sq = ?response_l2_sq,
        terminal_response_bytes = terminal_response_size,
        root_variant = root_variant,
        "proof fold level"
    );

    let header = if level_idx == 0 {
        "batched_root (terminal)".to_string()
    } else {
        format!("akita_fold L{level_idx} (terminal)")
    };
    eprintln!(
        "[{label}]   {header}: total={total} bytes (excl. terminal_response={terminal_response_size})"
    );
    eprintln!("[{label}]     extension_opening_partials={extension_opening_partials_size} bytes");
    eprintln!("[{label}]     extension_opening_sumcheck={extension_opening_sumcheck_size} bytes");
    eprintln!(
        "[{label}]     extension_opening_final_claims={extension_opening_final_claims_size} bytes"
    );
    eprintln!(
        "[{label}]     terminal_response={terminal_response_size} bytes (absorbed via transcript)"
    );
    assert_eq!(
        full,
        extension_opening_partials_size
            + extension_opening_sumcheck_size
            + extension_opening_final_claims_size
            + terminal_response_size
    );
    total
}

pub(crate) fn print_batched_proof_summary<FF, E, const D: usize>(
    label: &str,
    proof: &AkitaBatchedProof<FF, E>,
    schedule: Option<&FoldSchedule>,
    grinding_plan: &GrindingPlan,
) where
    FF: Field + CanonicalEncoding + AkitaSerialize,
    E: Field + AkitaSerialize,
{
    let root_total = proof.root.serialized_size(Compress::No);
    let recursive_steps_total: usize = proof
        .recursive_folds
        .iter()
        .map(|step| step.serialized_size(Compress::No))
        .sum::<usize>()
        + proof.terminal.serialized_size(Compress::No);
    let tail_total = proof.terminal_response().serialized_size(Compress::No);
    let nonce_stream_total = proof.nonce_stream.as_bytes().len();
    // The terminal step's serialized size includes `terminal_response`, which is
    // already accounted for in `tail_total`. Subtract it so the Akita-fold
    // line item only counts the per-level non-witness bytes.
    let akita_levels_total = root_total + recursive_steps_total - tail_total;
    let accounted_total = nonce_stream_total + akita_levels_total + tail_total;
    let fold_levels = proof.num_fold_levels();
    let fold_response_nonces = fold_response_nonces(proof, grinding_plan);

    tracing::info!(
        label,
        levels = fold_levels,
        proof_size_bytes = proof.size(),
        accounted_bytes = accounted_total,
        akita_fold_bytes = akita_levels_total,
        nonce_stream_bytes = nonce_stream_total,
        tail_bytes = tail_total,
        "proof summary"
    );
    grinding::emit_grinding_plan_report(label, grinding_plan, &proof.nonce_stream);
    eprintln!(
        "[{label}] proof: total={} bytes, nonce_stream={} bytes, akita_fold={} bytes, tail={} bytes, levels={}",
        proof.size(),
        nonce_stream_total,
        akita_levels_total,
        tail_total,
        fold_levels,
    );
    assert_eq!(
        accounted_total,
        proof.size(),
        "[{label}] proof accounting must exactly match serialized proof size"
    );
    let level_ring_dimension = |level_idx: usize| {
        schedule.map_or(D, |schedule| {
            if level_idx == 0 {
                schedule.root.params.d_a()
            } else if let Some(fold) = schedule.recursive_folds.get(level_idx - 1) {
                fold.params.d_a()
            } else {
                schedule.terminal.d_a()
            }
        })
    };
    print_akita_level_breakdown(
        label,
        0,
        &proof.root,
        level_ring_dimension(0),
        fold_response_nonces[&0],
    );
    for (i, step) in proof.recursive_folds.iter().enumerate() {
        let level = i + 1;
        print_akita_level_breakdown(
            label,
            level,
            step,
            level_ring_dimension(level),
            fold_response_nonces[&level],
        );
    }
    let terminal_level = proof.num_fold_levels() - 1;
    print_terminal_level_breakdown(
        label,
        terminal_level,
        &proof.terminal,
        "fold",
        level_ring_dimension(terminal_level),
        fold_response_nonces[&terminal_level],
    );
}

fn fold_response_nonces<FF: Field, E: Field>(
    proof: &AkitaBatchedProof<FF, E>,
    grinding_plan: &GrindingPlan,
) -> BTreeMap<usize, u32> {
    let mut reader = proof
        .nonce_stream
        .reader(grinding_plan)
        .expect("proof nonce stream matches its public grinding plan");
    let mut nonces = BTreeMap::new();
    for run in grinding_plan.runs() {
        for _ in 0..run.multiplicity() {
            let value = reader
                .read(run.site())
                .expect("profile proof follows its public grinding plan");
            if let GrindingSite::FoldResponse { level } = run.site() {
                assert_eq!(run.kind(), GrindingQueryKind::FoldResponse);
                let level = usize::try_from(level).expect("fold level fits usize");
                assert!(nonces.insert(level, value).is_none());
            }
        }
    }
    reader
        .finish()
        .expect("profile proof consumes its complete grinding plan");
    assert_eq!(nonces.len(), proof.num_fold_levels());
    nonces
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
