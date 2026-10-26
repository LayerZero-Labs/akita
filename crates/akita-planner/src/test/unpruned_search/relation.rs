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
    pub(super) mode: akita_params::RingRelationMode,
    pub(super) next_state: OracleRelationState,
}

const QUOTIENT: OracleRelationTransition = OracleRelationTransition {
    mode: akita_params::RingRelationMode::QuotientLift,
    next_state: OracleRelationState::QuotientPrefix,
};
const REDUCED: OracleRelationTransition = OracleRelationTransition {
    mode: akita_params::RingRelationMode::ReducedEvaluation,
    next_state: OracleRelationState::ReducedSuffix,
};
const QUOTIENT_ONLY: &[OracleRelationTransition] = &[QUOTIENT];
const REDUCED_ONLY: &[OracleRelationTransition] = &[REDUCED];
const QUOTIENT_OR_REDUCED: &[OracleRelationTransition] = &[QUOTIENT, REDUCED];

pub(super) fn transitions(
    state: OracleRelationState,
    absolute_level: usize,
    witness_chunk: akita_params::ChunkedWitnessCfg,
) -> &'static [OracleRelationTransition] {
    let eligible = absolute_level >= 2 && witness_chunk.num_chunks == 1;
    match (state, eligible) {
        (OracleRelationState::QuotientPrefix, true) => QUOTIENT_OR_REDUCED,
        (OracleRelationState::QuotientPrefix, false) => QUOTIENT_ONLY,
        (OracleRelationState::ReducedSuffix, true) => REDUCED_ONLY,
        (OracleRelationState::ReducedSuffix, false) => &[],
    }
}
