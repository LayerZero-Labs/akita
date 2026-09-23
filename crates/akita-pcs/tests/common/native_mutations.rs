use std::collections::{BTreeMap, BTreeSet};

use akita_transcript::{
    ProofMessageRange, ProtocolMessageKind, ProtocolSiteId, SITE_FAMILY_SUMCHECK,
};
use akita_types::SumcheckProtocol;

/// Semantic identity retained by the native proof-mutation suites.
///
/// Round and limb are deliberately excluded so one later representative can
/// be selected within a single protocol site. Every coordinate that identifies
/// an independently meaningful protocol occurrence remains in the bucket.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct NativeMutationBucket {
    pub(crate) family: u32,
    pub(crate) invocation: u32,
    pub(crate) level: u32,
    pub(crate) stage: u32,
    pub(crate) group: u32,
    pub(crate) detail: u32,
    pub(crate) kind: u32,
}

impl NativeMutationBucket {
    fn new(range: &ProofMessageRange) -> (Self, (u32, u32)) {
        let site = ProtocolSiteId::from_bytes(range.context.site_id);
        (
            Self {
                family: site.family,
                invocation: site.invocation,
                level: site.level,
                stage: site.stage,
                group: site.group,
                detail: site.detail,
                kind: range.context.kind,
            },
            (site.round, site.limb),
        )
    }

    pub(crate) fn sumcheck_protocol(self) -> Option<SumcheckProtocol> {
        (self.family == SITE_FAMILY_SUMCHECK)
            .then(|| SumcheckProtocol::from_tag(self.invocation))
            .flatten()
    }
}

/// Select one later round/limb from every complete semantic protocol bucket.
pub(crate) fn representative_native_mutation_ranges(
    ranges: impl IntoIterator<Item = ProofMessageRange>,
) -> Vec<(NativeMutationBucket, ProofMessageRange)> {
    let mut selected = BTreeMap::new();
    for range in ranges.into_iter().filter(|range| range.len != 0) {
        let (bucket, rank) = NativeMutationBucket::new(&range);
        if selected
            .get(&bucket)
            .is_none_or(|(selected_rank, _)| rank > *selected_rank)
        {
            selected.insert(bucket, (rank, range));
        }
    }
    selected
        .into_iter()
        .map(|(bucket, (_, range))| (bucket, range))
        .collect()
}

pub(crate) fn selected_sumcheck_protocols(
    selected: &[(NativeMutationBucket, ProofMessageRange)],
) -> BTreeSet<SumcheckProtocol> {
    selected
        .iter()
        .filter_map(|(bucket, _)| bucket.sumcheck_protocol())
        .collect()
}

/// Reconcile actual native emission with every message's public byte grammar.
pub(crate) fn assert_native_ranges_match_context(ranges: &[ProofMessageRange]) {
    for range in ranges {
        let maximum = usize::try_from(range.context.encoded_bytes)
            .expect("native context byte count must fit usize");
        if range.context.kind == ProtocolMessageKind::GrindingNonce as u32
            || range.context.kind == ProtocolMessageKind::FoldResponseNonce as u32
        {
            assert!(range.len > 0 && range.len <= maximum);
        } else {
            assert_eq!(
                range.len, maximum,
                "fixed-width native emission must match its public grammar"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_transcript::ProtocolContextRecord;

    fn range(
        protocol: SumcheckProtocol,
        level: u32,
        round: u32,
        start: usize,
    ) -> ProofMessageRange {
        ProofMessageRange {
            context: ProtocolContextRecord::new(
                ProtocolSiteId {
                    family: SITE_FAMILY_SUMCHECK,
                    invocation: protocol.tag(),
                    level,
                    stage: 0,
                    round,
                    group: 0,
                    limb: 0,
                    detail: 3,
                }
                .to_bytes(),
                ProtocolMessageKind::ProofAtoms as u32,
                1,
                1,
                0,
            ),
            start,
            len: 1,
        }
    }

    #[test]
    fn selector_preserves_protocol_and_level_identity() {
        let protocols = [
            SumcheckProtocol::ExtensionOpeningReduction,
            SumcheckProtocol::Stage1,
            SumcheckProtocol::PhysicalL2,
            SumcheckProtocol::Stage2,
            SumcheckProtocol::Stage3,
        ];
        let mut ranges = Vec::new();
        for (index, protocol) in protocols.into_iter().enumerate() {
            ranges.push(range(protocol, 0, 0, index * 4));
            ranges.push(range(protocol, 0, 3, index * 4 + 1));
            ranges.push(range(protocol, 1, 1, index * 4 + 2));
        }

        let selected = representative_native_mutation_ranges(ranges);
        assert_eq!(selected.len(), protocols.len() * 2);
        assert_eq!(
            selected_sumcheck_protocols(&selected),
            protocols.into_iter().collect()
        );
        for protocol in protocols {
            let selected_for_protocol = selected
                .iter()
                .filter(|(bucket, _)| bucket.sumcheck_protocol() == Some(protocol))
                .map(|(bucket, range)| (bucket.level, range.start))
                .collect::<Vec<_>>();
            assert_eq!(selected_for_protocol.len(), 2);
            assert!(selected_for_protocol.contains(&(0, protocol.tag() as usize * 4 + 1)));
        }
    }
}
