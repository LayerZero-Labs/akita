//! Optional scalar views in the outgoing witness and their compact D layout.

use super::BinaryRootWitnessPlan;

impl BinaryRootWitnessPlan {
    pub fn prime_segment(&self) -> Option<usize> {
        (self.prime_planes > 0).then_some(if self.word { 5 } else { 2 })
    }

    /// Natural digit concatenation committed by D, with no inter-plane padding.
    pub fn opening_segments(&self) -> Vec<usize> {
        let mut segments = Vec::new();
        if self.word {
            segments.push(2);
        }
        segments.extend(self.prime_segment());
        segments
    }
}
