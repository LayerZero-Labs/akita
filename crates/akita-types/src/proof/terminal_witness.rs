//! Helpers for transcript-binding terminal cleartext witnesses.

/// Transcript byte slices for terminal direct-witness replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalWitnessTranscriptParts {
    /// Logical terminal `e_folded` bytes, bound before sparse challenge sampling.
    pub e_folded: Vec<u8>,
    /// Post-challenge fold-response bytes (`z` in wire group order).
    pub response: Vec<u8>,
}
