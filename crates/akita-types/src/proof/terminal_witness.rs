//! Helpers for transcript-binding terminal cleartext witnesses.

/// Transcript byte slices for terminal direct-witness replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalWitnessTranscriptParts {
    /// Logical terminal `e_folded` bytes, bound before sparse challenge sampling.
    pub e_folded: Vec<u8>,
    /// Canonical inner-image `t` bytes. A suffix-terminal transition binds
    /// these as its public state before any dependent challenge is squeezed.
    pub t_state: Vec<u8>,
    /// Post-challenge fold-response bytes (`z` in wire group order).
    pub response: Vec<u8>,
}

impl TerminalWitnessTranscriptParts {
    /// Legacy response binding for a root-terminal proof whose public state is
    /// still the external outer commitment `u`.
    ///
    /// The current root-terminal transcript absorbs `z || t` after sampling
    /// the sparse challenge. Keeping this assembly here prevents root replay
    /// from reimplementing segment order while suffix-terminal replay promotes
    /// `t_state` to a pre-challenge binding.
    #[must_use]
    pub fn outer_committed_response(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.response.len().saturating_add(self.t_state.len()));
        bytes.extend_from_slice(&self.response);
        bytes.extend_from_slice(&self.t_state);
        bytes
    }
}
