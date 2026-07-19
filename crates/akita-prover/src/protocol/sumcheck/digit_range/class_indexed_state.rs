use super::exact_prefix::ExactPrefixTable;

/// Shared storage lifecycle for class-indexed early rounds and exact-prefix later rounds.
pub(super) enum ClassIndexedTableState<Compact, FirstChallengeFolded, Row: Copy> {
    Compact(Compact),
    FirstChallengeFolded(FirstChallengeFolded),
    Materialized(ExactPrefixTable<Row>),
}

impl<Compact, FirstChallengeFolded, Row: Copy>
    ClassIndexedTableState<Compact, FirstChallengeFolded, Row>
{
    pub(super) fn final_value(&self) -> Option<Row> {
        match self {
            Self::Materialized(table) => table.final_value(),
            Self::Compact(_) | Self::FirstChallengeFolded(_) => None,
        }
    }
}
