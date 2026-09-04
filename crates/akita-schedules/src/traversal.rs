//! Canonical traversal of every schedule-owned commitment group.

use core::fmt;

use akita_error::AkitaError;
use akita_types::{CommittedGroupParams, FoldSchedule, GroupOpenPhaseParams, TerminalFoldParams};

/// Semantic position of one group in a complete schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScheduleGroupPosition {
    RootPrecommitted(usize),
    RootFinal,
    RecursiveSetupPrefix(usize),
    RecursiveFinal(usize),
    Terminal,
}

impl fmt::Display for ScheduleGroupPosition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RootPrecommitted(index) => write!(formatter, "root precommitted group {index}"),
            Self::RootFinal => formatter.write_str("root final group"),
            Self::RecursiveSetupPrefix(index) => {
                write!(formatter, "recursive fold {index} setup prefix")
            }
            Self::RecursiveFinal(index) => write!(formatter, "recursive fold {index} final group"),
            Self::Terminal => formatter.write_str("terminal fold"),
        }
    }
}

/// One schedule group with the surrounding data needed for admission.
#[derive(Clone, Copy)]
pub(crate) enum ScheduleGroup<'a> {
    /// A commitment produced by an earlier operation and opened in this fold.
    Frozen {
        position: ScheduleGroupPosition,
        params: &'a GroupOpenPhaseParams,
        num_response_chunks: usize,
    },
    /// The new/final commitment produced by this nonterminal fold.
    Final {
        position: ScheduleGroupPosition,
        params: &'a CommittedGroupParams,
        num_claims: usize,
        fold_level: usize,
    },
    /// The clear terminal response, which has no B/D opening group.
    Terminal {
        position: ScheduleGroupPosition,
        params: &'a TerminalFoldParams,
        fold_level: usize,
    },
}

impl ScheduleGroup<'_> {
    pub(crate) const fn position(self) -> ScheduleGroupPosition {
        match self {
            Self::Frozen { position, .. }
            | Self::Final { position, .. }
            | Self::Terminal { position, .. } => position,
        }
    }
}

/// Visit every group in fold-local transcript order: preceding groups first,
/// then the fold's own group, followed by the terminal response.
///
/// Callers validate schedule structure before entering this traversal. That
/// establishes nonempty canonical group storage and excludes group roles that
/// are not representable at a given fold.
pub(crate) fn visit_schedule_groups(
    schedule: &FoldSchedule,
    mut visit: impl FnMut(ScheduleGroup<'_>) -> Result<(), AkitaError>,
) -> Result<(), AkitaError> {
    for (index, params) in schedule
        .root
        .params
        .precommitted_groups()
        .iter()
        .enumerate()
    {
        visit(ScheduleGroup::Frozen {
            position: ScheduleGroupPosition::RootPrecommitted(index),
            params,
            num_response_chunks: schedule.root.params.witness_chunk.num_chunks,
        })?;
    }
    visit(ScheduleGroup::Final {
        position: ScheduleGroupPosition::RootFinal,
        params: &schedule.root.params,
        num_claims: schedule
            .root
            .params
            .own_group()
            .profile
            .group
            .num_polynomials(),
        fold_level: 0,
    })?;

    for (index, fold) in schedule.recursive_folds.iter().enumerate() {
        if let Some(params) = fold.params.setup_prefix() {
            visit(ScheduleGroup::Frozen {
                position: ScheduleGroupPosition::RecursiveSetupPrefix(index),
                params,
                num_response_chunks: fold.params.witness_chunk.num_chunks,
            })?;
        }
        visit(ScheduleGroup::Final {
            position: ScheduleGroupPosition::RecursiveFinal(index),
            params: &fold.params,
            num_claims: 1,
            fold_level: index + 1,
        })?;
    }

    visit(ScheduleGroup::Terminal {
        position: ScheduleGroupPosition::Terminal,
        params: &schedule.terminal,
        fold_level: schedule.recursive_folds.len() + 1,
    })
}
