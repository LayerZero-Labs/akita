//! Schedule-bound realization of nonterminal ring relations.

/// How one nonterminal fold realizes its physical ring relation.
///
/// The mode is part of the authenticated schedule descriptor. It is not a
/// proof field: prover and verifier obtain it from the same effective schedule.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum RingRelationMode {
    /// Lift negacyclic equalities to ordinary polynomial identities with
    /// explicit polynomial-modulus quotient rows.
    #[default]
    QuotientLift,
    /// Check the relation after negacyclic reduction at the existing random
    /// evaluation point and omit polynomial-modulus quotient rows.
    ReducedEvaluation,
}

impl RingRelationMode {
    /// Stable tag bound by level, schedule-row, and catalog identities.
    pub const fn tag(self) -> u8 {
        match self {
            Self::QuotientLift => 1,
            Self::ReducedEvaluation => 2,
        }
    }

    /// Whether this fold checks the relation by reduced evaluation.
    #[must_use]
    pub const fn is_reduced_evaluation(self) -> bool {
        matches!(self, Self::ReducedEvaluation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_tags_are_stable_and_distinct() {
        assert_eq!(RingRelationMode::QuotientLift.tag(), 1);
        assert_eq!(RingRelationMode::ReducedEvaluation.tag(), 2);
    }
}
