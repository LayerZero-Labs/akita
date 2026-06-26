//! End-to-end Akita PCS scheme orchestration.

use akita_config::CommitmentConfig;
use std::marker::PhantomData;

#[macro_use]
mod impls;

/// End-to-end PCS wrapper, generic over commitment config `Cfg`.
///
/// Root ring degree is `Cfg::D`; suffix levels dispatch via the schedule plan at
/// prove time. Per-preset trait impls live in [`impls`](self::impls).
#[derive(Clone, Copy, Debug, Default)]
pub struct AkitaCommitmentScheme<Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

#[cfg(test)]
mod tests;
