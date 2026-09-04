use akita_error::AkitaError;
use akita_types::{RelationCandidateTopology, RingRelationMode, RingRelationPhase};

/// Canonical reason an otherwise-considered reduced transition is ineligible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReducedTransitionRejection {
    BeforeLevelTwo,
    IncomingSetupPrefix,
    CoefficientPacking,
    OutgoingSetupOffload,
}

/// Non-empty legal relation domain for one recursive fold topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RelationSearchDomain {
    QuotientOnly,
    ReducedOnly,
    QuotientAndReduced,
}

/// Candidate traversal only; it must not affect the exact selected schedule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum RelationTraversalOrder {
    #[default]
    Canonical,
    #[cfg(test)]
    Reversed,
}

/// Test-only restriction on the legal relation-mode search domain.
#[cfg(feature = "test-support")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TestRelationModeFilter {
    #[default]
    All,
    QuotientOnly,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum RelationModeFilter {
    #[default]
    All,
    #[cfg(feature = "test-support")]
    QuotientOnly,
}

#[cfg(feature = "test-support")]
impl From<TestRelationModeFilter> for RelationModeFilter {
    fn from(value: TestRelationModeFilter) -> Self {
        match value {
            TestRelationModeFilter::All => Self::All,
            TestRelationModeFilter::QuotientOnly => Self::QuotientOnly,
        }
    }
}

impl RelationSearchDomain {
    pub(crate) fn filtered(self, filter: RelationModeFilter) -> Result<Self, AkitaError> {
        match (self, filter) {
            (domain, RelationModeFilter::All) => Ok(domain),
            #[cfg(feature = "test-support")]
            (Self::QuotientOnly | Self::QuotientAndReduced, RelationModeFilter::QuotientOnly) => {
                Ok(Self::QuotientOnly)
            }
            #[cfg(feature = "test-support")]
            (Self::ReducedOnly, RelationModeFilter::QuotientOnly) => Err(AkitaError::InvalidSetup(
                "quotient-only search reached a reduced-only suffix".into(),
            )),
        }
    }
    #[must_use]
    pub(crate) const fn transitions(self) -> &'static [RingRelationMode] {
        match self {
            Self::QuotientOnly => &[RingRelationMode::QuotientLift],
            Self::ReducedOnly => &[RingRelationMode::ReducedEvaluation],
            Self::QuotientAndReduced => &[
                RingRelationMode::QuotientLift,
                RingRelationMode::ReducedEvaluation,
            ],
        }
    }

    #[must_use]
    pub(crate) const fn transitions_in(
        self,
        _order: RelationTraversalOrder,
    ) -> &'static [RingRelationMode] {
        #[cfg(test)]
        if matches!(
            (self, _order),
            (Self::QuotientAndReduced, RelationTraversalOrder::Reversed)
        ) {
            return &[
                RingRelationMode::ReducedEvaluation,
                RingRelationMode::QuotientLift,
            ];
        }
        self.transitions()
    }

    #[must_use]
    pub(crate) const fn has_multiple_modes(self) -> bool {
        matches!(self, Self::QuotientAndReduced)
    }

    /// Whether this fold domain admits the complete typed transition.
    #[must_use]
    pub(crate) fn admits(self, transition: RingRelationMode) -> bool {
        self.transitions().contains(&transition)
    }

    #[must_use]
    pub(crate) const fn including_terminal_quotient(self) -> Self {
        match self {
            Self::ReducedOnly | Self::QuotientAndReduced => Self::QuotientAndReduced,
            Self::QuotientOnly => Self::QuotientOnly,
        }
    }

    pub(crate) fn only_transition(self) -> Result<RingRelationMode, AkitaError> {
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

impl RelationSearchDomain {
    /// Enumerate the complete legal transition domain for one fold topology.
    pub(crate) fn for_topology(
        phase: RingRelationPhase,
        absolute_fold_level: usize,
        topology: RelationCandidateTopology,
        diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
    ) -> Result<RelationSearchDomain, AkitaError> {
        if matches!(phase, RingRelationPhase::QuotientPrefix) {
            if absolute_fold_level < 2 {
                if let Some(diagnostics) = diagnostics {
                    diagnostics
                        .record_reduced_rejection(ReducedTransitionRejection::BeforeLevelTwo);
                }
            }
            if topology.consumes_setup_prefix() {
                if let Some(diagnostics) = diagnostics {
                    diagnostics
                        .record_reduced_rejection(ReducedTransitionRejection::IncomingSetupPrefix);
                }
            }
            if topology.uses_coefficient_packing() {
                if let Some(diagnostics) = diagnostics {
                    diagnostics
                        .record_reduced_rejection(ReducedTransitionRejection::CoefficientPacking);
                }
            }
        }
        let domain = match phase.candidate_modes(absolute_fold_level, topology) {
            [RingRelationMode::QuotientLift, RingRelationMode::ReducedEvaluation] => Self::QuotientAndReduced,
            [RingRelationMode::QuotientLift] => Self::QuotientOnly,
            [RingRelationMode::ReducedEvaluation] => Self::ReducedOnly,
            _ => return Err(AkitaError::InvalidSetup("reduced-evaluation suffix requires a direct EvaluationTrace fold at level 2 or later".into())),
        };
        if let Some(diagnostics) = diagnostics {
            diagnostics.record_relation_domain(domain);
        }
        Ok(domain)
    }
}
