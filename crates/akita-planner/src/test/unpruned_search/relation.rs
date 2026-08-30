/// Canonical state of one monotone quotient-to-reduced oracle path.
///
/// The concrete cutover level matters only while it is still pending. Once a
/// path starts reduced evaluation, the traversal drops that historical value
/// and carries only the protocol-visible reduced-suffix state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum OracleRelationState {
    QuotientPrefix,
    ReducedSuffix,
}

#[derive(Clone, Copy)]
pub(super) struct OracleRelationTransition {
    pub(super) mode: akita_types::RingRelationMode,
    pub(super) next_state: OracleRelationState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OracleRelationPlan {
    AllLegal,
    EarliestReduced,
}

impl OracleRelationPlan {
    const QUOTIENT: OracleRelationTransition = OracleRelationTransition {
        mode: akita_types::RingRelationMode::QuotientLift,
        next_state: OracleRelationState::QuotientPrefix,
    };
    const REDUCED: OracleRelationTransition = OracleRelationTransition {
        mode: akita_types::RingRelationMode::ReducedEvaluation,
        next_state: OracleRelationState::ReducedSuffix,
    };
    const QUOTIENT_ONLY: &[OracleRelationTransition] = &[Self::QUOTIENT];
    const REDUCED_ONLY: &[OracleRelationTransition] = &[Self::REDUCED];
    const QUOTIENT_OR_REDUCED: &[OracleRelationTransition] = &[Self::QUOTIENT, Self::REDUCED];

    pub(super) const fn transitions(
        self,
        state: OracleRelationState,
        level: usize,
    ) -> &'static [OracleRelationTransition] {
        match (self, state, level >= 2) {
            (Self::AllLegal, OracleRelationState::QuotientPrefix, true) => {
                Self::QUOTIENT_OR_REDUCED
            }
            (Self::AllLegal, OracleRelationState::QuotientPrefix, false)
            | (Self::EarliestReduced, OracleRelationState::QuotientPrefix, false) => {
                Self::QUOTIENT_ONLY
            }
            (Self::EarliestReduced, OracleRelationState::QuotientPrefix, true)
            | (_, OracleRelationState::ReducedSuffix, _) => Self::REDUCED_ONLY,
        }
    }
}
