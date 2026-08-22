//! Logging transcript wrapper and test-time smell checks.

use crate::{labels, Transcript};
use akita_serialization::AkitaSerialize;
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use jolt_field::{CanonicalBytes, CanonicalEncoding, Field};
use std::cell::RefCell;
use std::collections::BTreeSet;

thread_local! {
    static THREAD_EVENTS: RefCell<Vec<TranscriptEvent>> = const { RefCell::new(Vec::new()) };
}

/// A recorded transcript or verifier wire-use event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscriptEvent {
    /// Canonical instance descriptor bytes were bound into the preamble.
    Preamble {
        /// Digest of the descriptor bytes.
        bytes_digest: [u8; 32],
        /// Descriptor byte length.
        bytes_len: usize,
    },
    /// A public value was absorbed into the transcript.
    Absorb {
        /// Semantic transcript label.
        label: Vec<u8>,
        /// Digest of the absorbed canonical bytes.
        bytes_digest: [u8; 32],
        /// Absorbed byte length.
        bytes_len: usize,
    },
    /// Challenge bytes were squeezed from the transcript.
    Squeeze {
        /// Semantic transcript label.
        label: Vec<u8>,
        /// Number of squeezed bytes, or zero for scalar challenges.
        len: usize,
    },
    /// One complete nonzero transcript proof-of-work transition.
    Grinding {
        /// Diagnostic label of the protected protocol query.
        site_label: Vec<u8>,
        /// Required low zero bits in the predicate.
        grind_bits: u8,
        /// Packed nonce width assigned by the public plan.
        nonce_bits: u8,
        /// Public proof nonce.
        nonce: u32,
        /// Number of predicate bytes squeezed.
        predicate_len: usize,
    },
    /// The verifier consumed a structured proof field that must be transcript-bound.
    Wire {
        /// Semantic transcript label.
        label: Vec<u8>,
        /// Digest of the consumed canonical bytes.
        bytes_digest: [u8; 32],
        /// Consumed byte length.
        bytes_len: usize,
    },
}

/// Test-time transcript wrapper that records absorb, squeeze, and wire events.
pub struct LoggingTranscript<T> {
    inner: T,
    events: Vec<TranscriptEvent>,
    expected_wire_labels: BTreeSet<Vec<u8>>,
}

impl<T> LoggingTranscript<T> {
    /// Wrap an existing transcript.
    pub fn wrap(inner: T) -> Self {
        Self {
            inner,
            events: Vec::new(),
            expected_wire_labels: BTreeSet::new(),
        }
    }

    /// Return the wrapped transcript.
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Return the recorded events for this wrapper.
    pub fn events(&self) -> &[TranscriptEvent] {
        &self.events
    }

    /// Clear this wrapper's local events.
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Mark a wire label as required by a test-time coverage manifest.
    pub fn expect_wire_label(&mut self, label: &[u8]) {
        self.expected_wire_labels.insert(label.to_vec());
    }

    /// Record a structured proof-field use that should have been absorbed.
    pub fn record_wire_use(&mut self, label: &[u8], canonical_bytes: &[u8]) {
        self.record(TranscriptEvent::Wire {
            label: label.to_vec(),
            bytes_digest: digest32(canonical_bytes),
            bytes_len: canonical_bytes.len(),
        });
    }

    /// Run all currently implemented smell checks.
    ///
    /// Tests that intentionally exercise a failing smell check should inspect
    /// [`Self::smell_check_errors`] instead.
    ///
    /// # Panics
    ///
    /// Panics when any smell check fails.
    pub fn assert_smell_checks(&self) {
        let errors = self.smell_check_errors();
        assert!(
            errors.is_empty(),
            "transcript smell checks failed:\n{}",
            errors.join("\n")
        );
    }

    /// Return all smell-check failures as human-readable messages.
    pub fn smell_check_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        self.check_preamble_is_descriptor(&mut errors);
        self.check_no_zero_byte_absorbs(&mut errors);
        self.check_known_labels_only(&mut errors);
        self.check_wire_value_before_squeeze_coverage(&mut errors);
        self.check_tracked_wire_coverage_is_complete(&mut errors);
        errors
    }

    fn record(&mut self, event: TranscriptEvent) {
        THREAD_EVENTS.with(|events| events.borrow_mut().push(event.clone()));
        self.events.push(event);
    }

    fn check_preamble_is_descriptor(&self, errors: &mut Vec<String>) {
        match self.events.first() {
            Some(TranscriptEvent::Preamble {
                bytes_digest: _,
                bytes_len,
            }) if *bytes_len > 0 => {}
            Some(_) => errors.push("first event is not a descriptor preamble".to_owned()),
            None => errors.push("transcript recorded no descriptor preamble".to_owned()),
        }
    }

    fn check_no_zero_byte_absorbs(&self, errors: &mut Vec<String>) {
        for event in &self.events {
            if let TranscriptEvent::Absorb {
                label, bytes_len, ..
            } = event
            {
                if *bytes_len == 0 {
                    errors.push(format!(
                        "zero-byte absorb under label `{}`",
                        label_text(label)
                    ));
                }
            }
        }
    }

    fn check_known_labels_only(&self, errors: &mut Vec<String>) {
        let known = labels::all_labels()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        for event in &self.events {
            let label = match event {
                TranscriptEvent::Absorb { label, .. }
                | TranscriptEvent::Squeeze { label, .. }
                | TranscriptEvent::Wire { label, .. } => label,
                TranscriptEvent::Grinding { site_label, .. } => site_label,
                TranscriptEvent::Preamble { .. } => continue,
            };
            if !is_known_or_extension_limb_label(label, &known) {
                errors.push(format!("unknown transcript label `{}`", label_text(label)));
            }
        }
    }

    fn check_wire_value_before_squeeze_coverage(&self, errors: &mut Vec<String>) {
        let mut window_start = 0;
        for squeeze_index in self.events.iter().enumerate().filter_map(|(index, event)| {
            matches!(
                event,
                TranscriptEvent::Squeeze { .. } | TranscriptEvent::Grinding { .. }
            )
            .then_some(index)
        }) {
            for (wire_index, event) in self.events[window_start..squeeze_index].iter().enumerate() {
                let wire_index = window_start + wire_index;
                let TranscriptEvent::Wire {
                    label,
                    bytes_digest,
                    bytes_len,
                } = event
                else {
                    continue;
                };

                let matched =
                    self.events[(wire_index + 1)..squeeze_index]
                        .iter()
                        .any(|candidate| {
                            matches!(
                                candidate,
                                TranscriptEvent::Absorb {
                                    label: absorb_label,
                                    bytes_digest: absorb_digest,
                                    bytes_len: absorb_len,
                                } if absorb_label == label
                                    && absorb_digest == bytes_digest
                                    && absorb_len == bytes_len
                            )
                        });
                if !matched {
                    errors.push(format!(
                        "wire `{}` was used before squeeze without a matching intervening absorb",
                        label_text(label)
                    ));
                }
            }
            window_start = squeeze_index + 1;
        }
    }

    fn check_tracked_wire_coverage_is_complete(&self, errors: &mut Vec<String>) {
        let seen = self
            .events
            .iter()
            .filter_map(|event| match event {
                TranscriptEvent::Wire { label, .. } => Some(label.clone()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();

        for label in &self.expected_wire_labels {
            if !seen.contains(label) {
                errors.push(format!(
                    "expected wire label `{}` was not recorded",
                    label_text(label)
                ));
            }
        }
    }
}

impl<F, T> Transcript<F> for LoggingTranscript<T>
where
    F: Field + CanonicalEncoding + CanonicalBytes,
    T: Transcript<F>,
{
    fn bind_instance_bytes(&mut self, instance_bytes: &[u8]) {
        self.record(TranscriptEvent::Preamble {
            bytes_digest: digest32(instance_bytes),
            bytes_len: instance_bytes.len(),
        });
        self.inner.bind_instance_bytes(instance_bytes);
    }

    fn record_wire_serde<S: AkitaSerialize>(&mut self, label: &[u8], s: &S) {
        let mut bytes = Vec::new();
        s.serialize_compressed(&mut bytes)
            .expect("AkitaSerialize should not fail for transcript wire logging");
        self.record_wire_use(label, &bytes);
        self.inner.record_wire_serde(label, s);
    }

    fn record_wire_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        self.record_wire_use(label, bytes);
        self.inner.record_wire_bytes(label, bytes);
    }

    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        self.record(TranscriptEvent::Absorb {
            label: label.to_vec(),
            bytes_digest: digest32(bytes),
            bytes_len: bytes.len(),
        });
        self.inner.append_bytes(label, bytes);
    }

    fn append_field(&mut self, label: &[u8], x: &F) {
        let bytes = x.to_bytes_le_vec();
        self.record(TranscriptEvent::Absorb {
            label: label.to_vec(),
            bytes_digest: digest32(&bytes),
            bytes_len: bytes.len(),
        });
        self.inner.append_field(label, x);
    }

    fn append_serde<S: AkitaSerialize>(&mut self, label: &[u8], s: &S) {
        let mut bytes = Vec::new();
        s.serialize_compressed(&mut bytes)
            .expect("AkitaSerialize should not fail for transcript logging");
        self.record(TranscriptEvent::Absorb {
            label: label.to_vec(),
            bytes_digest: digest32(&bytes),
            bytes_len: bytes.len(),
        });
        self.inner.append_serde(label, s);
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> F {
        self.record(TranscriptEvent::Squeeze {
            label: label.to_vec(),
            len: 0,
        });
        self.inner.challenge_scalar(label)
    }

    fn challenge_bytes(&mut self, label: &[u8], len: usize) -> Vec<u8> {
        self.record(TranscriptEvent::Squeeze {
            label: label.to_vec(),
            len,
        });
        self.inner.challenge_bytes(label, len)
    }

    fn challenge_block(&mut self, label: &[u8]) -> [u8; crate::TRANSCRIPT_CHALLENGE_BLOCK_LEN] {
        self.record(TranscriptEvent::Squeeze {
            label: label.to_vec(),
            len: crate::TRANSCRIPT_CHALLENGE_BLOCK_LEN,
        });
        self.inner.challenge_block(label)
    }

    fn grinding_predicate(
        &mut self,
        site_label: &[u8],
        grind_bits: u8,
        nonce_bits: u8,
        nonce: u32,
    ) -> Option<[u8; crate::GRINDING_PREDICATE_LEN]> {
        let predicate = self
            .inner
            .grinding_predicate(site_label, grind_bits, nonce_bits, nonce);
        if predicate.is_some() {
            self.record(TranscriptEvent::Grinding {
                site_label: site_label.to_vec(),
                grind_bits,
                nonce_bits,
                nonce,
                predicate_len: crate::GRINDING_PREDICATE_LEN,
            });
        }
        predicate
    }
}

impl<F, T> crate::TranscriptFactory<F> for LoggingTranscript<T>
where
    F: FieldCore + CanonicalField + CanonicalBytes,
    T: crate::TranscriptFactory<F>,
{
    fn new(domain_label: &[u8]) -> Self {
        Self::wrap(T::new(domain_label))
    }
}

/// Clear the current thread's accumulated logging transcript events.
pub fn clear_thread_events() {
    THREAD_EVENTS.with(|events| events.borrow_mut().clear());
}

/// Return the current thread's accumulated logging transcript events.
pub fn thread_events() -> Vec<TranscriptEvent> {
    THREAD_EVENTS.with(|events| events.borrow().clone())
}

fn digest32(bytes: &[u8]) -> [u8; 32] {
    type Blake2b256 = Blake2b<U32>;
    let digest = Blake2b256::digest(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn is_known_or_extension_limb_label(label: &[u8], known: &BTreeSet<&[u8]>) -> bool {
    if known.contains(label) {
        return true;
    }
    crate::ext_limb_base_label(label).is_some_and(|base| known.contains(base))
}

fn label_text(label: &[u8]) -> String {
    std::str::from_utf8(label)
        .map(str::to_owned)
        .unwrap_or_else(|_| format!("0x{}", hex_bytes(label)))
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

impl<T> crate::TranscriptChallengePreview for LoggingTranscript<T>
where
    T: crate::TranscriptChallengePreview,
{
    fn preview_challenge_block(
        &self,
        absorb_payloads: &[&[u8]],
    ) -> [u8; crate::TRANSCRIPT_CHALLENGE_BLOCK_LEN] {
        self.inner.preview_challenge_block(absorb_payloads)
    }
}

#[cfg(test)]
mod tests {
    use super::{clear_thread_events, thread_events, LoggingTranscript, TranscriptEvent};
    use crate::{append_ext_field, labels, sample_ext_challenge, AkitaTranscript, Transcript};
    use jolt_field::{Fp32, Fp64, FpExt2, NegOneNr};

    type F = Fp64<4294967197>;
    type Base = Fp32<251>;
    type BaseFpExt2 = FpExt2<Base, NegOneNr>;

    #[test]
    fn logs_absorbs_and_squeezes() {
        clear_thread_events();
        let mut transcript = LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"logging-test"));
        transcript.bind_instance_bytes(b"descriptor");
        transcript.append_bytes(labels::ABSORB_COMMITMENT, b"commitment");
        let _ = transcript.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);

        assert_eq!(transcript.events().len(), 3);
        assert_eq!(thread_events(), transcript.events());
        transcript.assert_smell_checks();
    }

    #[test]
    fn catches_zero_byte_absorb() {
        let mut transcript = LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"logging-test"));
        transcript.bind_instance_bytes(b"descriptor");
        transcript.append_bytes(labels::ABSORB_COMMITMENT, b"");
        let errors = transcript.smell_check_errors();
        assert!(errors.iter().any(|error| error.contains("zero-byte")));
    }

    #[test]
    fn catches_wire_without_matching_absorb_before_squeeze() {
        let mut transcript = LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"logging-test"));
        transcript.bind_instance_bytes(b"descriptor");
        transcript.record_wire_use(labels::ABSORB_NEXT_LEVEL_WITNESS_BINDING, b"final-w");
        let _ = transcript.challenge_scalar(labels::CHALLENGE_TAU0);
        let errors = transcript.smell_check_errors();
        assert!(errors.iter().any(|error| error.contains("wire `ak/a/w`")));
    }

    #[test]
    fn accepts_wire_with_matching_absorb_before_squeeze() {
        let mut transcript = LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"logging-test"));
        transcript.bind_instance_bytes(b"descriptor");
        transcript.record_wire_use(labels::ABSORB_NEXT_LEVEL_WITNESS_BINDING, b"final-w");
        transcript.append_bytes(labels::ABSORB_NEXT_LEVEL_WITNESS_BINDING, b"final-w");
        let _ = transcript.challenge_scalar(labels::CHALLENGE_TAU0);
        transcript.assert_smell_checks();
    }

    #[test]
    fn coverage_manifest_requires_wire_events() {
        let mut transcript = LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"logging-test"));
        transcript.bind_instance_bytes(b"descriptor");
        transcript.expect_wire_label(labels::ABSORB_NEXT_LEVEL_WITNESS_BINDING);
        let errors = transcript.smell_check_errors();
        assert!(errors
            .iter()
            .any(|error| error.contains("expected wire label")));
    }

    #[test]
    fn event_equality_uses_structural_records() {
        let mut a = LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"logging-test"));
        let mut b = LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"logging-test"));
        a.bind_instance_bytes(b"descriptor");
        b.bind_instance_bytes(b"descriptor");
        a.append_bytes(labels::ABSORB_COMMITMENT, b"commitment");
        b.append_bytes(labels::ABSORB_COMMITMENT, b"commitment");

        assert_eq!(a.events(), b.events());
        assert!(matches!(a.events()[0], TranscriptEvent::Preamble { .. }));
    }

    #[test]
    fn known_label_check_accepts_extension_limb_labels() {
        let mut transcript = LoggingTranscript::wrap(AkitaTranscript::<Base>::new(b"logging-test"));
        transcript.bind_instance_bytes(b"descriptor");
        let x = BaseFpExt2::new(Base::from_u64(1), Base::from_u64(2));
        append_ext_field::<Base, BaseFpExt2, _>(
            &mut transcript,
            labels::ABSORB_EVALUATION_CLAIMS,
            &x,
        );
        let _ =
            sample_ext_challenge::<Base, BaseFpExt2, _>(&mut transcript, labels::CHALLENGE_TAU0);

        transcript.assert_smell_checks();
    }

    #[test]
    fn grinding_records_one_logical_event_and_zero_bits_record_none() {
        let mut prover = LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"logging-test"));
        let mut verifier =
            LoggingTranscript::wrap(AkitaTranscript::<F>::unbound_verifier(b"logging-test"));
        prover.bind_instance_bytes(b"descriptor");
        verifier.bind_instance_bytes(b"descriptor");
        let before_preview = prover.events().to_vec();
        let nonce = crate::search_grinding_nonce(&prover, 1, 8).unwrap();
        assert_eq!(prover.events(), before_preview);

        let predicate =
            Transcript::grinding_predicate(&mut prover, labels::CHALLENGE_TAU0, 1, 8, nonce)
                .unwrap();
        assert!(crate::grinding_predicate_accepts(
            &predicate,
            std::num::NonZeroU8::new(1).unwrap()
        ));
        assert!(matches!(
            prover.events().last(),
            Some(TranscriptEvent::Grinding {
                site_label,
                grind_bits: 1,
                nonce_bits: 8,
                nonce: recorded_nonce,
                predicate_len: crate::GRINDING_PREDICATE_LEN,
            }) if site_label == labels::CHALLENGE_TAU0 && *recorded_nonce == nonce
        ));
        let verifier_predicate =
            Transcript::grinding_predicate(&mut verifier, labels::CHALLENGE_TAU0, 1, 8, nonce)
                .unwrap();
        assert_eq!(predicate, verifier_predicate);
        assert_eq!(prover.events(), verifier.events());
        let event_count = prover.events().len();
        assert_eq!(
            Transcript::grinding_predicate(&mut prover, labels::CHALLENGE_TAU1, 0, 0, 0,),
            None
        );
        assert_eq!(prover.events().len(), event_count);
        prover.assert_smell_checks();
    }
}
