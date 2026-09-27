use super::ProtocolSiteId;

impl ProtocolSiteId {
    /// Decode the fixed canonical diagnostic layout produced by [`Self::to_bytes`].
    ///
    /// Protocol site identifiers come from validated public state, not proof
    /// input. This inverse supports typed diagnostics and test coverage; it is
    /// not part of proof parsing.
    #[must_use]
    pub const fn from_bytes(encoded: [u8; 32]) -> Self {
        Self {
            family: u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]),
            invocation: u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]),
            level: u32::from_le_bytes([encoded[8], encoded[9], encoded[10], encoded[11]]),
            stage: u32::from_le_bytes([encoded[12], encoded[13], encoded[14], encoded[15]]),
            round: u32::from_le_bytes([encoded[16], encoded[17], encoded[18], encoded[19]]),
            group: u32::from_le_bytes([encoded[20], encoded[21], encoded[22], encoded[23]]),
            limb: u32::from_le_bytes([encoded[24], encoded[25], encoded[26], encoded[27]]),
            detail: u32::from_le_bytes([encoded[28], encoded[29], encoded[30], encoded[31]]),
        }
    }
}
