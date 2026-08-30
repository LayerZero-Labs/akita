use akita_error::AkitaError;
use akita_types::RingRelationMode;

/// Monotone planner phase for recursive ring-relation realization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum RingRelationPhase {
    /// Quotient lifting remains available and an eligible direct fold may
    /// begin the reduced-evaluation suffix.
    #[default]
    QuotientPrefix,
    /// Every later committed fold uses reduced evaluation.
    ReducedEvaluationSuffix,
}

/// Typed topology visible to the canonical relation transition authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RelationCandidateTopology {
    DirectEvaluationTrace,
    DirectCoefficientPacking,
    SetupPrefixedEvaluationTrace,
    SetupPrefixedCoefficientPacking,
}

impl RelationCandidateTopology {
    pub(crate) const fn new(
        consumes_setup_prefix: bool,
        opening: akita_types::OpeningMethod,
    ) -> Self {
        match (consumes_setup_prefix, opening) {
            (false, akita_types::OpeningMethod::EvaluationTrace) => Self::DirectEvaluationTrace,
            (false, akita_types::OpeningMethod::SubringCoefficientPacking { .. }) => {
                Self::DirectCoefficientPacking
            }
            (true, akita_types::OpeningMethod::EvaluationTrace) => {
                Self::SetupPrefixedEvaluationTrace
            }
            (true, akita_types::OpeningMethod::SubringCoefficientPacking { .. }) => {
                Self::SetupPrefixedCoefficientPacking
            }
        }
    }

    const fn is_direct_evaluation_trace(self) -> bool {
        matches!(self, Self::DirectEvaluationTrace)
    }
}

/// One legal per-fold mode selection and its complete recursive consequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RelationTransition {
    mode: RingRelationMode,
    next_phase: RingRelationPhase,
    allows_setup_offload: bool,
}

impl RelationTransition {
    const QUOTIENT: Self = Self {
        mode: RingRelationMode::QuotientLift,
        next_phase: RingRelationPhase::QuotientPrefix,
        allows_setup_offload: true,
    };
    const REDUCED: Self = Self {
        mode: RingRelationMode::ReducedEvaluation,
        next_phase: RingRelationPhase::ReducedEvaluationSuffix,
        allows_setup_offload: false,
    };
    const QUOTIENT_ONLY: &[Self] = &[Self::QUOTIENT];
    const QUOTIENT_OR_REDUCED: &[Self] = &[Self::QUOTIENT, Self::REDUCED];
    const REDUCED_ONLY: &[Self] = &[Self::REDUCED];

    #[must_use]
    pub(crate) const fn quotient_only() -> &'static [Self] {
        Self::QUOTIENT_ONLY
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) const fn for_mode(mode: RingRelationMode) -> &'static [Self] {
        match mode {
            RingRelationMode::QuotientLift => Self::QUOTIENT_ONLY,
            RingRelationMode::ReducedEvaluation => Self::REDUCED_ONLY,
        }
    }

    #[must_use]
    pub(crate) fn with_terminal_quotient(transitions: &[Self]) -> &'static [Self] {
        if transitions
            .iter()
            .any(|transition| transition.mode == RingRelationMode::ReducedEvaluation)
        {
            Self::QUOTIENT_OR_REDUCED
        } else {
            Self::QUOTIENT_ONLY
        }
    }

    #[must_use]
    pub(crate) const fn mode(self) -> RingRelationMode {
        self.mode
    }

    #[must_use]
    pub(crate) const fn next_phase(self) -> RingRelationPhase {
        self.next_phase
    }

    #[must_use]
    pub(crate) const fn allows_setup_offload(self) -> bool {
        self.allows_setup_offload
    }
}

impl RingRelationPhase {
    /// Enumerate the complete legal transition domain for one fold topology.
    pub(crate) fn transitions(
        self,
        absolute_fold_level: usize,
        topology: RelationCandidateTopology,
    ) -> Result<&'static [RelationTransition], AkitaError> {
        match self {
            Self::QuotientPrefix
                if absolute_fold_level >= 2 && topology.is_direct_evaluation_trace() =>
            {
                Ok(RelationTransition::QUOTIENT_OR_REDUCED)
            }
            Self::QuotientPrefix => Ok(RelationTransition::QUOTIENT_ONLY),
            Self::ReducedEvaluationSuffix
                if absolute_fold_level >= 2 && topology.is_direct_evaluation_trace() =>
            {
                Ok(RelationTransition::REDUCED_ONLY)
            }
            Self::ReducedEvaluationSuffix => Err(AkitaError::InvalidSetup(
                "reduced-evaluation suffix requires a direct EvaluationTrace fold at level 2 or later"
                    .into(),
            )),
        }
    }
}
