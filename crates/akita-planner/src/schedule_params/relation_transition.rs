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

/// Non-empty legal relation domain for one recursive fold topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RelationSearchDomain {
    QuotientOnly,
    ReducedOnly,
    QuotientAndReduced,
}

impl RelationSearchDomain {
    #[must_use]
    pub(crate) const fn transitions(self) -> &'static [RelationTransition] {
        match self {
            Self::QuotientOnly => RelationTransition::QUOTIENT_ONLY,
            Self::ReducedOnly => RelationTransition::REDUCED_ONLY,
            Self::QuotientAndReduced => RelationTransition::QUOTIENT_OR_REDUCED,
        }
    }

    #[must_use]
    pub(crate) const fn has_multiple_modes(self) -> bool {
        matches!(self, Self::QuotientAndReduced)
    }

    #[must_use]
    pub(crate) const fn including_terminal_quotient(self) -> Self {
        match self {
            Self::ReducedOnly | Self::QuotientAndReduced => Self::QuotientAndReduced,
            Self::QuotientOnly => Self::QuotientOnly,
        }
    }

    #[cfg(test)]
    pub(crate) fn transition_for(
        self,
        mode: RingRelationMode,
    ) -> Result<RelationTransition, AkitaError> {
        self.transitions()
            .iter()
            .copied()
            .find(|transition| transition.mode == mode)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "materialized fold has no transition in its relation domain".into(),
                )
            })
    }

    pub(crate) fn only_transition(self) -> Result<RelationTransition, AkitaError> {
        let [transition] = self.transitions() else {
            return Err(AkitaError::InvalidSetup(
                "relation domain does not contain exactly one transition".into(),
            ));
        };
        Ok(*transition)
    }

    pub(crate) const fn for_mode(mode: RingRelationMode) -> Self {
        match mode {
            RingRelationMode::QuotientLift => Self::QuotientOnly,
            RingRelationMode::ReducedEvaluation => Self::ReducedOnly,
        }
    }
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
    ) -> Result<RelationSearchDomain, AkitaError> {
        match self {
            Self::QuotientPrefix
                if absolute_fold_level >= 2 && topology.is_direct_evaluation_trace() =>
            {
                Ok(RelationSearchDomain::QuotientAndReduced)
            }
            Self::QuotientPrefix => Ok(RelationSearchDomain::QuotientOnly),
            Self::ReducedEvaluationSuffix
                if absolute_fold_level >= 2 && topology.is_direct_evaluation_trace() =>
            {
                Ok(RelationSearchDomain::ReducedOnly)
            }
            Self::ReducedEvaluationSuffix => Err(AkitaError::InvalidSetup(
                "reduced-evaluation suffix requires a direct EvaluationTrace fold at level 2 or later"
                    .into(),
            )),
        }
    }
}
