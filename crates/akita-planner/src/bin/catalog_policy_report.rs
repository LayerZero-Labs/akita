//! Per-level schedule policy fields included in catalog audit reports.

use akita_planner::EmitSpec;
use akita_types::FoldSchedule;

pub(super) fn source_encoding_signature(value: akita_types::CommittedSourceEncoding) -> String {
    match value {
        akita_types::CommittedSourceEncoding::CanonicalCoefficientTable => "canonical".into(),
        akita_types::CommittedSourceEncoding::TensorSubfieldProjection { extension_degree } => {
            format!("tensor-k{extension_degree}")
        }
    }
}

fn security_route_signature(value: akita_types::InnerCommitSecurityRoute) -> &'static str {
    match value {
        akita_types::InnerCommitSecurityRoute::Linf(_) => "Linf",
        akita_types::InnerCommitSecurityRoute::L2 { .. } => "L2",
    }
}

fn relation_mode_signature(value: akita_types::RingRelationMode) -> &'static str {
    match value {
        akita_types::RingRelationMode::QuotientLift => "quotient",
        akita_types::RingRelationMode::ReducedEvaluation => "reduced-evaluation",
    }
}

fn removed_quotient_coefficients(
    spec: &EmitSpec,
    params: &akita_types::CommittedGroupParams,
    input_witness_len: usize,
) -> Result<(usize, usize), String> {
    if !params.ring_relation_mode.is_reduced_evaluation() {
        return Ok((0, 0));
    }
    let final_group = akita_types::PolynomialGroupLayout::singleton(
        akita_types::padded_boolean_opening_vars(input_witness_len)
            .map_err(|error| format!("derive quotient-report group: {error}"))?,
    );
    let breakdown = akita_types::QuotientCoefficientBreakdown::for_reduced_counterfactual(
        params,
        final_group,
        spec.policy.claim_ext_degree,
        spec.policy.decomposition.field_bits(),
    )
    .map_err(|error| format!("derive quotient-report counterfactual: {error}"))?;
    Ok((breakdown.ordinary, breakdown.compression))
}

fn opening_policy_signature(
    opening_method: akita_types::OpeningMethod,
    source_encoding: akita_types::CommittedSourceEncoding,
    extension_degree: usize,
    d_a: usize,
    security_route: akita_types::InnerCommitSecurityRoute,
) -> Result<String, String> {
    let opening = match opening_method {
        akita_types::OpeningMethod::EvaluationTrace => {
            "ET,s=-,h=-,partial=-,quotient=-".to_string()
        }
        akita_types::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } => {
            let geometry = akita_types::SubringCoefficientPackingGeometry::try_new(
                extension_degree,
                d_a,
                challenge_subring_dimension,
            )
            .map_err(|error| format!("derive catalog packing geometry: {error}"))?;
            format!(
                "PACK,s={},h={},partial={},quotient={}",
                geometry.challenge_subring_dimension(),
                geometry.packing_factor(),
                geometry.partial_base_field_width(),
                geometry.partial_base_field_width(),
            )
        }
    };
    Ok(format!(
        "{opening},src={},dA={d_a},sec={}",
        source_encoding_signature(source_encoding),
        security_route_signature(security_route),
    ))
}

pub(super) fn catalog_policy_signature(
    spec: &EmitSpec,
    schedule: &FoldSchedule,
) -> Result<String, String> {
    use std::fmt::Write as _;

    let mut signature = String::new();
    let cutover = std::iter::once(&schedule.root.params)
        .chain(schedule.recursive_folds.iter().map(|fold| &fold.params))
        .position(|params| params.ring_relation_mode.is_reduced_evaluation());
    write!(
        signature,
        "cutover={};",
        cutover.map_or_else(|| "none".to_string(), |level| format!("L{level}"))
    )
    .map_err(|error| format!("write catalog relation cutover: {error}"))?;
    let nonterminal = std::iter::once((
        0usize,
        &schedule.root.params,
        schedule.root.input_witness_len,
        schedule.root.output_witness_len,
    ))
    .chain(
        schedule
            .recursive_folds
            .iter()
            .enumerate()
            .map(|(index, fold)| {
                (
                    index + 1,
                    &fold.params,
                    fold.input_witness_len,
                    fold.output_witness_len,
                )
            }),
    );
    for (level, params, input_witness_len, output_witness_len) in nonterminal {
        let (ordinary_quotient_coefficients_removed, compression_quotient_coefficients_removed) =
            removed_quotient_coefficients(spec, params, input_witness_len)?;
        let eor = if matches!(
            params.opening_method(),
            akita_types::OpeningMethod::EvaluationTrace
        ) {
            let final_group = akita_types::PolynomialGroupLayout::singleton(
                akita_types::padded_boolean_opening_vars(input_witness_len)
                    .map_err(|error| format!("derive opening arity: {error}"))?,
            );
            let opening_shape = params
                .opening_layout_for_final_group(final_group)
                .and_then(|layout| layout.aggregate_polynomial_group_layout())
                .map_err(|error| format!("derive level opening shape: {error}"))?;
            akita_types::extension_opening_reduction_level_bytes(
                spec.policy
                    .challenge_field_bits()
                    .map_err(|error| format!("derive challenge width: {error}"))?,
                spec.policy.claim_ext_degree,
                opening_shape,
            )
            .map_err(|error| format!("derive level EOR bytes: {error}"))?
        } else {
            0
        };
        if level != 0 {
            signature.push('/');
        }
        write!(
            signature,
            "L{level}[chunks={}@{},rel={},qrm={ordinary_quotient_coefficients_removed},cqrm={compression_quotient_coefficients_removed},eor={eor},in={input_witness_len},out={output_witness_len};witness={}",
            params.witness_chunk.num_chunks,
            params.witness_chunk.num_activated_levels,
            relation_mode_signature(params.ring_relation_mode),
            opening_policy_signature(
                params.opening_method(),
                params.source_encoding,
                spec.policy.claim_ext_degree,
                params.d_a(),
                params.inner().matrix.security_route(),
            )?,
        )
        .map_err(|error| format!("write catalog policy signature: {error}"))?;
        if level == 0 {
            for (index, group) in schedule
                .root
                .params
                .precommitted_groups()
                .iter()
                .enumerate()
            {
                write!(
                    signature,
                    ";pre{index}={}",
                    opening_policy_signature(
                        group.opening.opening_method,
                        akita_types::CommittedSourceEncoding::CanonicalCoefficientTable,
                        spec.policy.claim_ext_degree,
                        group.profile.inner.matrix.ring_dimension(),
                        group.profile.inner.matrix.security_route(),
                    )?,
                )
                .map_err(|error| format!("write catalog policy signature: {error}"))?;
            }
        } else if let Some(prefix) = schedule.recursive_folds[level - 1]
            .params
            .setup_prefix()
            .as_ref()
        {
            write!(
                signature,
                ";prefix={}",
                opening_policy_signature(
                    prefix.opening.opening_method,
                    akita_types::CommittedSourceEncoding::CanonicalCoefficientTable,
                    spec.policy.claim_ext_degree,
                    prefix.profile.inner.matrix.ring_dimension(),
                    prefix.profile.inner.matrix.security_route(),
                )?,
            )
            .map_err(|error| format!("write catalog policy signature: {error}"))?;
        }
        signature.push(']');
    }
    let terminal_eor = akita_types::extension_opening_reduction_level_bytes(
        spec.policy
            .challenge_field_bits()
            .map_err(|error| format!("derive challenge width: {error}"))?,
        spec.policy.claim_ext_degree,
        akita_types::PolynomialGroupLayout::singleton(
            akita_types::padded_boolean_opening_vars(schedule.terminal.input_witness_len)
                .map_err(|error| format!("derive terminal opening arity: {error}"))?,
        ),
    )
    .map_err(|error| format!("derive terminal EOR bytes: {error}"))?;
    let terminal_source = akita_types::CommittedSourceEncoding::for_producer(
        akita_types::OpeningMethod::EvaluationTrace,
        spec.policy.claim_ext_degree,
        schedule.terminal.d_a(),
        akita_types::padded_boolean_opening_vars(schedule.terminal.input_witness_len)
            .map_err(|error| format!("derive terminal source arity: {error}"))?,
        false,
    );
    write!(
        signature,
        "/T[method=ET,src={},eor={terminal_eor},input={},dA={},sec={}]",
        source_encoding_signature(terminal_source),
        schedule.terminal.input_witness_len,
        schedule.terminal.d_a(),
        security_route_signature(schedule.terminal.inner.matrix.security_route(),),
    )
    .map_err(|error| format!("write catalog policy signature: {error}"))?;
    Ok(signature)
}
