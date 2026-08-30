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

const QUOTIENT: OracleRelationTransition = OracleRelationTransition {
    mode: akita_types::RingRelationMode::QuotientLift,
    next_state: OracleRelationState::QuotientPrefix,
};
const REDUCED: OracleRelationTransition = OracleRelationTransition {
    mode: akita_types::RingRelationMode::ReducedEvaluation,
    next_state: OracleRelationState::ReducedSuffix,
};
const QUOTIENT_ONLY: &[OracleRelationTransition] = &[QUOTIENT];
const REDUCED_ONLY: &[OracleRelationTransition] = &[REDUCED];
const QUOTIENT_OR_REDUCED: &[OracleRelationTransition] = &[QUOTIENT, REDUCED];

pub(super) const fn transitions(
    state: OracleRelationState,
    level: usize,
) -> &'static [OracleRelationTransition] {
    match (state, level >= 2) {
        (OracleRelationState::QuotientPrefix, true) => QUOTIENT_OR_REDUCED,
        (OracleRelationState::QuotientPrefix, false) => QUOTIENT_ONLY,
        (OracleRelationState::ReducedSuffix, _) => REDUCED_ONLY,
    }
}
