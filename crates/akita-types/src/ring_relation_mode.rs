//! Schedule-bound realization of nonterminal ring relations.

/// How one nonterminal fold realizes its physical ring relation.
///
/// The mode is part of the authenticated schedule descriptor. It is not a
/// proof field: prover and verifier obtain it from the same effective schedule.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    /// Stable machine-readable name used by diagnostics and profile artifacts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::QuotientLift => "quotient_lift",
            Self::ReducedEvaluation => "reduced_evaluation",
        }
    }

    /// Stable tag bound by level, schedule, and generated-catalog identities.
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

/// Monotone phase for recursive ring-relation realization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RingRelationPhase {
    /// Quotient lifting remains available and an eligible direct fold may
    /// begin the reduced-evaluation suffix.
    #[default]
    QuotientPrefix,
    /// Every later committed fold uses reduced evaluation.
    ReducedEvaluationSuffix,
}

/// Typed topology visible to the canonical relation transition authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationCandidateTopology {
    /// Evaluation trace without a setup prefix.
    DirectEvaluationTrace,
    /// Coefficient packing without a setup prefix.
    DirectCoefficientPacking,
    /// Evaluation trace consuming a setup prefix.
    SetupPrefixedEvaluationTrace,
    /// Coefficient packing consuming a setup prefix.
    SetupPrefixedCoefficientPacking,
}

impl RelationCandidateTopology {
    /// Classify a fold by its opening method and setup-prefix input.
    pub const fn new(consumes_setup_prefix: bool, opening: crate::OpeningMethod) -> Self {
        match (consumes_setup_prefix, opening) {
            (false, crate::OpeningMethod::EvaluationTrace) => Self::DirectEvaluationTrace,
            (false, crate::OpeningMethod::SubringCoefficientPacking { .. }) => {
                Self::DirectCoefficientPacking
            }
            (true, crate::OpeningMethod::EvaluationTrace) => Self::SetupPrefixedEvaluationTrace,
            (true, crate::OpeningMethod::SubringCoefficientPacking { .. }) => {
                Self::SetupPrefixedCoefficientPacking
            }
        }
    }

    /// Whether this topology admits reduced evaluation.
    pub const fn is_direct_evaluation_trace(self) -> bool {
        matches!(self, Self::DirectEvaluationTrace)
    }

    /// Whether this fold consumes a setup prefix.
    pub const fn consumes_setup_prefix(self) -> bool {
        matches!(
            self,
            Self::SetupPrefixedEvaluationTrace | Self::SetupPrefixedCoefficientPacking
        )
    }

    /// Whether the opening packs subring coefficients.
    pub const fn uses_coefficient_packing(self) -> bool {
        matches!(
            self,
            Self::DirectCoefficientPacking | Self::SetupPrefixedCoefficientPacking
        )
    }
}

impl RingRelationPhase {
    /// Legal committed relation modes at this point in the monotone suffix.
    pub const fn candidate_modes(
        self,
        absolute_level: usize,
        topology: RelationCandidateTopology,
    ) -> &'static [RingRelationMode] {
        let eligible = absolute_level >= 2 && topology.is_direct_evaluation_trace();
        match (self, eligible) {
            (Self::QuotientPrefix, true) => &[
                RingRelationMode::QuotientLift,
                RingRelationMode::ReducedEvaluation,
            ],
            (Self::QuotientPrefix, false) => &[RingRelationMode::QuotientLift],
            (Self::ReducedEvaluationSuffix, true) => &[RingRelationMode::ReducedEvaluation],
            (Self::ReducedEvaluationSuffix, false) => &[],
        }
    }

    /// Phase after selecting one of the legal candidate modes.
    pub const fn after(self, mode: RingRelationMode) -> Self {
        match mode {
            RingRelationMode::QuotientLift => Self::QuotientPrefix,
            RingRelationMode::ReducedEvaluation => Self::ReducedEvaluationSuffix,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_tags_are_stable_and_distinct() {
        assert_eq!(RingRelationMode::QuotientLift.tag(), 1);
        assert_eq!(RingRelationMode::ReducedEvaluation.tag(), 2);
        assert_eq!(RingRelationMode::QuotientLift.as_str(), "quotient_lift");
        assert_eq!(
            RingRelationMode::ReducedEvaluation.as_str(),
            "reduced_evaluation"
        );
    }
}
