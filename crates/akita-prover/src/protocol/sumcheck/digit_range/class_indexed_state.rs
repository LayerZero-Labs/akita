/// Shared storage lifecycle for class-indexed early rounds and materialized later rounds.
pub(super) enum ClassIndexedTableState<Compact, FirstChallengeFolded, Materialized> {
    Compact(Compact),
    FirstChallengeFolded(FirstChallengeFolded),
    Materialized(Materialized),
}
