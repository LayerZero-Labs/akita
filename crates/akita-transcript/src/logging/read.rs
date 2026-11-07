//! Owned diagnostic snapshots whose memory follows compressed run count.

use super::{ContextRun, ProtocolContextRecord, ProtocolSiteId, TranscriptEvent, THREAD_EVENTS};

/// A lossless snapshot independent of subsequent thread-local recording/clearing.
///
/// Proof ranges remain available separately through [`crate::thread_proof_ranges`].
/// They describe proof messages, not challenge repetitions, and are not expanded.
#[derive(Clone, Debug)]
pub struct TranscriptEventLog {
    runs: Vec<ContextRun>,
    count: usize,
}

impl PartialEq for TranscriptEventLog {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && (self.runs == other.runs || self.events().eq(other.events()))
    }
}

impl Eq for TranscriptEventLog {}

impl TranscriptEventLog {
    /// Number of logical events, computed from compressed runs at capture time.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the snapshot contains no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Ordered identical-context runs and their nonzero repetition counts.
    /// Consecutive groups have distinct site bytes and are yielded separately
    /// with count one. No repeated context is materialized here.
    pub fn runs(&self) -> impl Iterator<Item = (ProtocolContextRecord, usize)> + '_ {
        self.runs.iter().flat_map(ContextRun::contexts)
    }

    /// Stream the original logical sequence with constant iterator storage.
    /// Digests defined over that sequence still need to hash every event byte.
    pub fn events(&self) -> impl ExactSizeIterator<Item = TranscriptEvent> + '_ {
        Events {
            inner: self.runs.iter().flat_map(ContextRun::events),
            remaining: self.count,
        }
    }
}

/// Clone the current thread's compressed runs without expanding events.
/// Returns `None` if their logical event count does not fit in `usize`.
/// Recording and pending proof ranges are left unchanged.
#[must_use]
pub fn thread_event_log() -> Option<TranscriptEventLog> {
    THREAD_EVENTS.with(|events| {
        let runs = events.borrow();
        let count = runs.iter().try_fold(0usize, |total, run| {
            let first_group = ProtocolSiteId::from_bytes(run.first.site_id).group;
            let groups =
                usize::try_from(u64::from(run.last_group) - u64::from(first_group) + 1).ok()?;
            total.checked_add(akita_error::checked::product([groups, run.repetitions])?)
        })?;
        Some(TranscriptEventLog {
            runs: runs.clone(),
            count,
        })
    })
}

struct Events<I> {
    inner: I,
    remaining: usize,
}

impl<I: Iterator<Item = TranscriptEvent>> Iterator for Events<I> {
    type Item = TranscriptEvent;

    fn next(&mut self) -> Option<Self::Item> {
        let event = self.inner.next()?;
        self.remaining -= 1;
        Some(event)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<I: Iterator<Item = TranscriptEvent>> ExactSizeIterator for Events<I> {}

#[cfg(test)]
mod tests;
