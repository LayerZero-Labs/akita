use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use akita_params::transcript_site::SITE_FAMILY_SUMCHECK;
use akita_params::{ProtocolSiteId, SumcheckProtocol};
use jolt_transcript::{TranscriptEvent, TranscriptOp};

/// One recorded prover message: its site, its position among consecutive
/// messages at that site, and its argument-string range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MessageRange {
    pub(crate) site: ProtocolSiteId,
    pub(crate) ordinal: u32,
    pub(crate) range: Range<usize>,
}

/// Every prover message in `events`, in transcript order.
///
/// Consecutive messages at one site (a bounded payload's length and body) get
/// increasing ordinals so each stays an independent mutation target.
pub(crate) fn message_ranges(events: &[TranscriptEvent]) -> Vec<MessageRange> {
    let mut messages: Vec<MessageRange> = Vec::new();
    let mut previous: Option<(ProtocolSiteId, u32)> = None;
    for event in events {
        let Some(range) = event.narg.clone() else {
            previous = None;
            continue;
        };
        assert_eq!(event.op, TranscriptOp::Message);
        let site = ProtocolSiteId::from_bytes(event.site.0);
        let ordinal = match previous {
            Some((last, ordinal)) if last == site => ordinal + 1,
            _ => 0,
        };
        previous = Some((site, ordinal));
        messages.push(MessageRange {
            site,
            ordinal,
            range,
        });
    }
    messages
}

/// Assert that the messages tile `0..proof_len` without gaps or overlap.
pub(crate) fn assert_messages_cover(messages: &[MessageRange], proof_len: usize) {
    let mut cursor = 0usize;
    for message in messages {
        assert_eq!(
            message.range.start, cursor,
            "proof messages must be gap-free"
        );
        cursor = message.range.end;
    }
    assert_eq!(cursor, proof_len, "proof messages must cover the proof");
}

/// Semantic identity retained by the proof-mutation suites.
///
/// Round is deliberately excluded so one later representative can be selected
/// within a single protocol site. Every coordinate that identifies an
/// independently meaningful protocol occurrence remains in the bucket.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct MutationBucket {
    pub(crate) family: u32,
    pub(crate) invocation: u32,
    pub(crate) level: u32,
    pub(crate) stage: u32,
    pub(crate) group: u32,
    pub(crate) detail: u32,
    pub(crate) ordinal: u32,
}

impl MutationBucket {
    fn new(message: &MessageRange) -> (Self, u32) {
        let site = message.site;
        (
            Self {
                family: site.family,
                invocation: site.invocation,
                level: site.level,
                stage: site.stage,
                group: site.group,
                detail: site.detail,
                ordinal: message.ordinal,
            },
            site.round,
        )
    }

    pub(crate) fn sumcheck_protocol(self) -> Option<SumcheckProtocol> {
        (self.family == SITE_FAMILY_SUMCHECK)
            .then(|| SumcheckProtocol::from_tag(self.invocation))
            .flatten()
    }
}

/// Select the latest round from every complete semantic protocol bucket.
pub(crate) fn representative_mutation_ranges(
    messages: impl IntoIterator<Item = MessageRange>,
) -> Vec<(MutationBucket, MessageRange)> {
    let mut selected = BTreeMap::new();
    for message in messages
        .into_iter()
        .filter(|message| !message.range.is_empty())
    {
        let (bucket, rank) = MutationBucket::new(&message);
        if selected
            .get(&bucket)
            .is_none_or(|(selected_rank, _)| rank > *selected_rank)
        {
            selected.insert(bucket, (rank, message));
        }
    }
    selected
        .into_iter()
        .map(|(bucket, (_, message))| (bucket, message))
        .collect()
}

pub(crate) fn selected_sumcheck_protocols(
    selected: &[(MutationBucket, MessageRange)],
) -> BTreeSet<SumcheckProtocol> {
    selected
        .iter()
        .filter_map(|(bucket, _)| bucket.sumcheck_protocol())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(protocol: SumcheckProtocol, level: u32, round: u32, start: usize) -> MessageRange {
        MessageRange {
            site: ProtocolSiteId {
                family: SITE_FAMILY_SUMCHECK,
                invocation: protocol.tag(),
                level,
                round,
                detail: 3,
                ..ProtocolSiteId::default()
            },
            ordinal: 0,
            range: start..start + 1,
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
        let mut messages = Vec::new();
        for (index, protocol) in protocols.into_iter().enumerate() {
            messages.push(message(protocol, 0, 0, index * 4));
            messages.push(message(protocol, 0, 3, index * 4 + 1));
            messages.push(message(protocol, 1, 1, index * 4 + 2));
        }

        let selected = representative_mutation_ranges(messages);
        assert_eq!(selected.len(), protocols.len() * 2);
        assert_eq!(
            selected_sumcheck_protocols(&selected),
            protocols.into_iter().collect()
        );
        for protocol in protocols {
            let selected_for_protocol = selected
                .iter()
                .filter(|(bucket, _)| bucket.sumcheck_protocol() == Some(protocol))
                .map(|(bucket, message)| (bucket.level, message.range.start))
                .collect::<Vec<_>>();
            assert_eq!(selected_for_protocol.len(), 2);
            assert!(selected_for_protocol.contains(&(0, protocol.tag() as usize * 4 + 1)));
        }
    }
}
