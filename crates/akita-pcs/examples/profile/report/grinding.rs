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

    let levels = run_levels(plan);
    for (run_index, (run, level)) in plan.runs().iter().zip(levels).enumerate() {
        let site = site_report(run.site());
        let kind = match run.kind() {
            GrindingQueryKind::ProofOfWork => "proof_of_work",
            GrindingQueryKind::FoldResponse => "fold_response",
            GrindingQueryKind::FoldChallengeRoot => "fold_challenge_root",
            GrindingQueryKind::FoldChallengeCoordinates => "fold_challenge_coordinates",
        };
        let run_nonce_bits = u64::from(run.nonce_bits()) * run.multiplicity();
        tracing::info!(
            label,
            run_index,
            level,
            component = site.component,
            query = site.query,
            protocol = site.protocol,
            stage = ?site.stage,
            round = ?site.round,
            group = ?site.group,
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

fn run_levels(plan: &GrindingPlan) -> Vec<u32> {
    let mut levels = vec![None; plan.runs().len()];
    let mut following_fold = None;
    for (index, run) in plan.runs().iter().enumerate().rev() {
        if let GrindingSite::FoldResponse { level } = run.site() {
            following_fold = Some(level);
        }
        levels[index] = explicit_level(run.site()).or(following_fold);
    }
    levels
        .into_iter()
        .map(|level| level.expect("every profile grinding query belongs to a fold"))
        .collect()
}

fn explicit_level(site: GrindingSite) -> Option<u32> {
    match site {
        GrindingSite::EvaluationBatch
        | GrindingSite::ExtensionOpeningPoint
        | GrindingSite::ExtensionOpeningClaimBatch => None,
        GrindingSite::SumcheckRound {
            protocol: SumcheckProtocol::ExtensionOpeningReduction,
            level: u32::MAX,
            ..
        } => None,
        GrindingSite::SumcheckRound { level, .. }
        | GrindingSite::FoldResponse { level }
        | GrindingSite::FoldChallengeRoot { level, .. }
        | GrindingSite::FoldChallengeCoordinates { level, .. }
        | GrindingSite::RingSwitchAlpha { level }
        | GrindingSite::Tau0Point { level }
        | GrindingSite::Tau1Point { level }
        | GrindingSite::Stage1InterstageBatch { level, .. }
        | GrindingSite::L2SubclaimBatch { level }
        | GrindingSite::L2NormMerge { level }
        | GrindingSite::L2VirtualBatch { level }
        | GrindingSite::CompressionBinary { level }
        | GrindingSite::Stage2Batch { level } => Some(level),
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
        GrindingSite::EvaluationBatch => plain("opening", "evaluation_batch"),
        GrindingSite::ExtensionOpeningPoint => plain("extension_opening", "opening_point"),
        GrindingSite::ExtensionOpeningClaimBatch => plain("extension_opening", "claim_batch"),
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
        GrindingSite::FoldChallengeRoot { group, .. } => SiteReport {
            group: Some(group),
            ..plain("fold_challenge", "challenge_root")
        },
        GrindingSite::FoldChallengeCoordinates { group, .. } => SiteReport {
            group: Some(group),
            ..plain("fold_challenge", "challenge_coordinates")
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
