//! End-to-end commitment, opening, and verification harnesses.
//!
//! Supported statements are exactly the rows of the shipped schedule
//! artifacts. The registry loads every family's artifact, derives the
//! tractable rows under the process's size limit, and builds each case's
//! groups from the catalog's own committed profiles. Nothing here restates
//! a sizing or schedule rule.

mod boundary;
mod family;
mod import;
pub mod ops;
mod registry;

pub use family::Family;
pub use family::{Check, Mutation};
pub use registry::{registry, Limits, Registry, Selector};

use crate::gen::Domain;
use akita_types::GroupCommitPhaseParams;

/// What a committed group's source must look like.
#[derive(Clone, Copy, Debug)]
pub struct SourceSpec {
    /// `Some(k)` when the class admits only unit one-hot sources with chunk `k`.
    pub onehot_only: Option<usize>,
    /// Admissible dense coefficient set for balanced-digit sources.
    pub domain: Domain,
}

#[derive(Clone, Debug)]
pub enum Origin {
    /// Final group, committed through the scheduler with its prefix.
    Final,
    /// Independent row of this family; committed through the scheduler.
    OwnSingleton(GroupCommitPhaseParams),
    /// Profile owned by another family of the same source class; committed
    /// on this backend in explicit mode.
    Explicit {
        profile: GroupCommitPhaseParams,
        family: &'static str,
    },
    /// Profile owned by a family of another source class; committed on that
    /// family's backend and transferred with `import_commitment`.
    Imported {
        profile: GroupCommitPhaseParams,
        family: &'static str,
    },
}

#[derive(Clone, Debug)]
pub struct GroupPlan {
    pub num_vars: usize,
    pub num_polys: usize,
    pub origin: Origin,
    pub source: SourceSpec,
}

#[derive(Clone, Debug)]
pub struct Case {
    pub row: usize,
    pub groups: Vec<GroupPlan>,
    /// Total committed coefficients across all groups.
    pub cost: u64,
}

impl Case {
    pub fn label(&self) -> String {
        self.groups
            .iter()
            .map(|group| format!("{}:{}", group.num_vars, group.num_polys))
            .collect::<Vec<_>>()
            .join("+")
    }
}
