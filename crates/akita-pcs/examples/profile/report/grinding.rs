use akita_types::{
    GrindingPlan, GrindingQueryKind, GrindingSite, SumcheckProtocol, TranscriptNonceStream,
};

struct SiteReport {
    component: &'static str,
    query: &'static str,
    protocol: &'static str,
    stage: Option<u32>,
    round: Option<u32>,
    group: Option<u32>,
}

pub(super) fn emit_grinding_plan_report(
    label: &str,
    plan: &GrindingPlan,
    stream: &TranscriptNonceStream,
) {
    assert_eq!(stream.bit_len(), plan.total_nonce_bits());
    let nonce_stream_bytes = stream.as_bytes().len();
    let padding_bits = nonce_stream_bytes * 8 - plan.total_nonce_bits();
    tracing::info!(
        label,
        nominal_capacity_bits = plan.nominal_capacity_bits(),
        total_nonce_bits = plan.total_nonce_bits(),
        nonce_stream_bytes,
        padding_bits,
        run_count = plan.runs().len(),
        expanded_query_count = plan.expanded_query_count(),
        "grinding plan summary"
    );

    for (run_index, run) in plan.runs().iter().enumerate() {
        let site = site_report(run.site());
        let kind = match run.kind() {
            GrindingQueryKind::ProofOfWork => "proof_of_work",
            GrindingQueryKind::FoldResponse => "fold_response",
            GrindingQueryKind::FoldChallengeGroup => "fold_challenge_group",
        };
        let run_nonce_bits = u64::from(run.nonce_bits()) * run.multiplicity();
        tracing::info!(
            label,
            run_index,
            level = run.site().level(),
            component = site.component,
            query = site.query,
            protocol = site.protocol,
            stage = ?site.stage,
            round = ?site.round,
            group = ?site.group,
            coordinate_count = ?run.fold_coordinate_count(),
            kind,
            loss_factor = run.loss_factor(),
            grind_bits = run.grind_bits(),
            nonce_bits = run.nonce_bits(),
            multiplicity = run.multiplicity(),
            run_nonce_bits,
            "grinding plan run"
        );
    }
}

fn site_report(site: GrindingSite) -> SiteReport {
    let plain = |component, query| SiteReport {
        component,
        query,
        protocol: "none",
        stage: None,
        round: None,
        group: None,
    };
    match site {
        GrindingSite::EvaluationBatch { .. } => plain("opening", "evaluation_batch"),
        GrindingSite::ExtensionOpeningPoint { .. } => plain("extension_opening", "opening_point"),
        GrindingSite::ExtensionOpeningClaimBatch { .. } => {
            plain("extension_opening", "claim_batch")
        }
        GrindingSite::SumcheckRound {
            protocol,
            stage,
            round,
            ..
        } => SiteReport {
            component: sumcheck_component(protocol),
            query: "sumcheck_round",
            protocol: sumcheck_protocol(protocol),
            stage: Some(stage),
            round: Some(round),
            group: None,
        },
        GrindingSite::FoldResponse { .. } => plain("fold_response", "response_search"),
        GrindingSite::FoldChallengeGroup { group, .. } => SiteReport {
            group: Some(group),
            ..plain("fold_challenge", "challenge_group")
        },
        GrindingSite::RingSwitchAlpha { .. } => plain("ring_switch", "alpha"),
        GrindingSite::Tau0Point { .. } => plain("ring_switch", "tau0_point"),
        GrindingSite::Tau1Point { .. } => plain("ring_switch", "tau1_point"),
        GrindingSite::Stage1InterstageBatch { stage, .. } => SiteReport {
            stage: Some(stage),
            ..plain("stage1", "interstage_batch")
        },
        GrindingSite::L2SubclaimBatch { .. } => plain("physical_l2", "subclaim_batch"),
        GrindingSite::L2NormMerge { .. } => plain("physical_l2", "norm_merge"),
        GrindingSite::L2VirtualBatch { .. } => plain("physical_l2", "virtual_batch"),
        GrindingSite::CompressionBinary { .. } => plain("stage2", "compression_binary"),
        GrindingSite::Stage2Batch { .. } => plain("stage2", "claim_batch"),
    }
}

const fn sumcheck_component(protocol: SumcheckProtocol) -> &'static str {
    match protocol {
        SumcheckProtocol::ExtensionOpeningReduction => "extension_opening",
        SumcheckProtocol::Stage1 => "stage1",
        SumcheckProtocol::PhysicalL2 => "physical_l2",
        SumcheckProtocol::Stage2 => "stage2",
        SumcheckProtocol::Stage3 => "stage3",
    }
}

const fn sumcheck_protocol(protocol: SumcheckProtocol) -> &'static str {
    match protocol {
        SumcheckProtocol::ExtensionOpeningReduction => "extension_opening_reduction",
        SumcheckProtocol::Stage1 => "stage1",
        SumcheckProtocol::PhysicalL2 => "physical_l2",
        SumcheckProtocol::Stage2 => "stage2",
        SumcheckProtocol::Stage3 => "stage3",
    }
}
