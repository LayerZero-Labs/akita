use akita_params::{GrindingSite, SumcheckProtocol};
use akita_transcript::{ProtocolMessageKind, ProtocolSiteId, TranscriptEvent};

pub(super) fn assert_order(level: u32, offloaded: bool) {
    let events = akita_transcript::thread_events();
    let positions = |predicate: &dyn Fn(ProtocolSiteId, u32) -> bool| {
        events
            .iter()
            .enumerate()
            .filter_map(|(index, event)| {
                let TranscriptEvent::Context(record) = event;
                let site = ProtocolSiteId::from_bytes(record.site_id);
                (site.level == level && predicate(site, record.kind)).then_some(index)
            })
            .collect::<Vec<_>>()
    };
    let statements = positions(&|site, kind| {
        site.family == 0x5342_4154
            && site.detail == 20
            && kind == ProtocolMessageKind::PublicValue as u32
    });
    assert_eq!(
        statements.len(),
        if offloaded { 2 } else { 1 },
        "the inherited statement must be bound, plus the offloaded combined statement"
    );
    let batch_family = GrindingSite::SetupBatch { level }.site_id(0).family;
    let queries = positions(&|site, kind| {
        site.family == batch_family && kind == ProtocolMessageKind::GrindingCheck as u32
    });
    assert_eq!(
        queries.len(),
        1,
        "one setup-batch grinding query per target"
    );
    let query = queries[0];
    assert!(
        statements[0] < query,
        "inherited claims must be fixed before batching work"
    );
    let coefficients = positions(&|site, kind| {
        site.family == batch_family && kind == ProtocolMessageKind::Challenge as u32
    });
    assert!(
        !coefficients.is_empty(),
        "batch coefficients must be logged"
    );
    assert!(
        coefficients.iter().all(|&index| query < index),
        "batch coefficients must follow their grinding query"
    );
    let stage2 = positions(&|site, kind| {
        site.family == akita_transcript::SITE_FAMILY_SUMCHECK
            && site.invocation == SumcheckProtocol::Stage2.tag()
            && kind == ProtocolMessageKind::ProofAtoms as u32
    });
    assert!(!stage2.is_empty());
    assert!(
        statements[0] < stage2[0],
        "inherited claims must precede Stage 2"
    );
    assert!(
        stage2.iter().all(|&index| index < query),
        "batching follows Stage 2 rounds"
    );
    let stage3 = positions(&|site, kind| {
        site.family == akita_transcript::SITE_FAMILY_SUMCHECK
            && site.invocation == SumcheckProtocol::Stage3.tag()
            && kind == ProtocolMessageKind::ProofAtoms as u32
    });
    if offloaded {
        let claims = positions(&|site, kind| {
            site.family == akita_transcript::SITE_FAMILY_STAGE3
                && site.detail == 2
                && kind == ProtocolMessageKind::ProofAtoms as u32
        });
        assert_eq!(claims.len(), 1);
        assert!(
            claims[0] < statements[1] && statements[1] < query,
            "the fold's answer and combined statement must be fixed before lambda"
        );
        assert!(!stage3.is_empty());
        assert!(
            coefficients.iter().all(|&index| index < stage3[0]),
            "all batching coefficients must precede the first Stage 3 round"
        );
    } else {
        assert!(
            stage3.is_empty(),
            "a direct target must not add a Stage 3 sumcheck"
        );
    }
}
