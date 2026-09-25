//! Process-wide family registry derived from the shipped artifacts.

use super::family::{Family, FamilyImpl};
use super::{Case, SourceSpec};
use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::RecursiveCommitmentConfig;
use akita_types::GroupCommitPhaseParams;
use std::sync::OnceLock;

/// Per-process case limits chosen by the target.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Largest total committed coefficient count for one case.
    pub max_cost: u64,
}

/// Which planned cases a target draws from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selector {
    /// One group, balanced-digit final source, direct setup.
    DenseSingle,
    /// One group, unit one-hot final source, direct setup.
    OneHotSingle,
    /// Precommitted groups, direct setup.
    Batch,
    /// Recursive setup-offloading families, any shape.
    Recursive,
    /// Every direct case.
    AnyDirect,
}

pub struct Registry {
    families: Vec<Box<dyn Family>>,
    limits: Limits,
}

/// An independent (singleton) profile and the family that owns its source class.
pub struct IndexedProfile {
    pub field: &'static str,
    pub family: &'static str,
    pub profile: GroupCommitPhaseParams,
    pub source: SourceSpec,
}

fn all_families() -> Vec<Box<dyn Family>> {
    vec![
        FamilyImpl::<fp128::Dense>::boxed(),
        FamilyImpl::<fp128::DenseBounded>::boxed(),
        FamilyImpl::<fp128::DenseMultiChunk>::boxed(),
        FamilyImpl::<fp128::OneHot>::boxed(),
        FamilyImpl::<fp128::OneHotMultiChunk>::boxed(),
        FamilyImpl::<fp128::OneHotMultiChunkW2R2>::boxed(),
        FamilyImpl::<fp128::OneHotMultiChunkW4R2>::boxed(),
        FamilyImpl::<RecursiveCommitmentConfig<fp128::Dense>>::boxed(),
        FamilyImpl::<RecursiveCommitmentConfig<fp128::OneHot>>::boxed(),
        FamilyImpl::<RecursiveCommitmentConfig<fp128::OneHotMultiChunk>>::boxed(),
        FamilyImpl::<fp32::Dense>::boxed(),
        FamilyImpl::<fp32::OneHot>::boxed(),
        FamilyImpl::<RecursiveCommitmentConfig<fp32::Dense>>::boxed(),
        FamilyImpl::<fp64::Dense>::boxed(),
        FamilyImpl::<fp64::OneHot>::boxed(),
        FamilyImpl::<RecursiveCommitmentConfig<fp64::Dense>>::boxed(),
    ]
}

static REGISTRY: OnceLock<Registry> = OnceLock::new();

/// Load every artifact once. The first caller's limits apply to the process.
pub fn registry(limits: Limits) -> &'static Registry {
    REGISTRY.get_or_init(|| Registry::load(limits))
}

impl Registry {
    /// Load every shipped artifact and plan cases under `limits`.
    pub fn load(limits: Limits) -> Self {
        crate::env::init();
        let families = all_families();
        let index: Vec<IndexedProfile> = families
            .iter()
            .flat_map(|family| family.singleton_profiles())
            .collect();
        for family in &families {
            family.plan(&index, limits);
        }
        Registry { families, limits }
    }

    pub fn limits(&self) -> Limits {
        self.limits
    }

    pub fn families(&self) -> &[Box<dyn Family>] {
        &self.families
    }

    /// Flattened `(family, case)` pairs matching `selector`, in stable order.
    pub fn select(&self, selector: Selector) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for (family_index, family) in self.families.iter().enumerate() {
            for (case_index, case) in family.cases().iter().enumerate() {
                if matches(selector, family.as_ref(), case) {
                    out.push((family_index, case_index));
                }
            }
        }
        out
    }

    /// Human-readable list of every planned and excluded case.
    pub fn describe(&self) -> String {
        let mut out = String::new();
        for family in &self.families {
            out.push_str(&format!(
                "{} ({} rows)\n",
                family.name(),
                family.row_count()
            ));
            for case in family.cases() {
                out.push_str(&format!("  case {} cost={}\n", case.label(), case.cost));
            }
            for (label, reason) in family.excluded() {
                out.push_str(&format!("  excluded {label}: {reason}\n"));
            }
        }
        out
    }
}

fn matches(selector: Selector, family: &dyn Family, case: &Case) -> bool {
    let single = case.groups.len() == 1;
    let final_source = case.groups.last().expect("case has a final group").source;
    match selector {
        Selector::DenseSingle => {
            !family.is_recursive() && single && final_source.onehot_only.is_none()
        }
        Selector::OneHotSingle => {
            !family.is_recursive() && single && final_source.onehot_only.is_some()
        }
        Selector::Batch => !family.is_recursive() && !single,
        Selector::Recursive => family.is_recursive(),
        Selector::AnyDirect => !family.is_recursive(),
    }
}
