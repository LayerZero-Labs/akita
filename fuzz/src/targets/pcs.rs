//! End-to-end commit/prove/verify targets over the shipped schedule rows.

use crate::input::Reader;
use crate::pcs::{registry, Check, Limits, Selector};

const MAX_COST_ENV: &str = "AKITA_FUZZ_MAX_CASE_COEFFS";

/// The planned cases each end-to-end target draws from: a selector and the
/// default cost cap as `log2` of the committed coefficients. Seed generation
/// reads the same values, so a seed's leading case index selects the case it
/// was generated for.
pub const DENSE: (Selector, u32) = (Selector::DenseSingle, 18);
pub const ONEHOT: (Selector, u32) = (Selector::OneHotSingle, 20);
pub const BATCH: (Selector, u32) = (Selector::Batch, 20);
pub const RECURSIVE: (Selector, u32) = (Selector::Recursive, 21);
pub const REJECT: (Selector, u32) = (Selector::AnyDirect, 17);
pub const PARALLEL: (Selector, u32) = (Selector::AnyDirect, 17);
pub const LIVENESS: (Selector, u32) = (Selector::Any, 20);
pub const SHARED: (Selector, u32) = (Selector::AnyDirect, 17);

/// Case limits of a target with this default cap, unless
/// `AKITA_FUZZ_MAX_CASE_COEFFS` overrides it.
pub fn limits(default_log2: u32) -> Limits {
    let max_cost = std::env::var(MAX_COST_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1u64 << default_log2);
    Limits { max_cost }
}

fn run(data: &[u8], (selector, default_log2): (Selector, u32), check: Check) {
    let registry = registry(limits(default_log2));
    let cases = registry.select(selector);
    assert!(
        !cases.is_empty(),
        "no {selector:?} case fits the process limit {:?}; raise {MAX_COST_ENV}",
        registry.limits()
    );
    let mut reader = Reader::new(data);
    let (family, case) = cases[reader.choose(cases.len())];
    let family = &registry.families()[family];
    crate::env::on_large_stack(|| family.run(case, &mut reader, check));
}

pub fn dense(data: &[u8]) {
    run(data, DENSE, Check::Valid);
}

pub fn onehot(data: &[u8]) {
    run(data, ONEHOT, Check::Valid);
}

pub fn batch(data: &[u8]) {
    run(data, BATCH, Check::Valid);
}

pub fn recursive(data: &[u8]) {
    run(data, RECURSIVE, Check::Valid);
}

pub fn reject(data: &[u8]) {
    run(data, REJECT, Check::Reject);
}

pub fn parallel(data: &[u8]) {
    run(data, PARALLEL, Check::Parallel);
}

pub fn liveness(data: &[u8]) {
    run(data, LIVENESS, Check::Liveness);
}

/// One covering setup per field and one backend per field pair, shared by
/// every family of the process's registry (see `pcs::shared`).
pub fn shared(data: &[u8]) {
    use std::any::{Any, TypeId};
    use std::collections::HashMap;
    use std::sync::OnceLock;
    type Requirements = HashMap<TypeId, Box<dyn Any + Send + Sync>>;
    static REQUIREMENTS: OnceLock<Requirements> = OnceLock::new();
    let (selector, default_log2) = SHARED;
    let registry = registry(limits(default_log2));
    let cases = registry.select(selector);
    assert!(!cases.is_empty(), "no direct case fits the process limit");
    let requirements = REQUIREMENTS.get_or_init(|| {
        // Every family with a planned case, grouped by field, at the largest
        // capacity any of them needs (union requires equal capacities).
        let mut by_field: HashMap<TypeId, Vec<usize>> = HashMap::new();
        for &(family, _) in &cases {
            let members = by_field
                .entry(registry.families()[family].field_type())
                .or_default();
            if !members.contains(&family) {
                members.push(family);
            }
        }
        by_field
            .into_iter()
            .map(|(field, members)| {
                let (nv, polys) = members
                    .iter()
                    .map(|&f| registry.families()[f].setup_capacity())
                    .fold((0, 0), |(a, b), (c, d)| (a.max(c), b.max(d)));
                let union = members.iter().fold(None, |acc, &f| {
                    Some(registry.families()[f].union_requirements(acc, nv, polys))
                });
                (field, union.expect("nonempty field group"))
            })
            .collect()
    });
    let mut reader = Reader::new(data);
    let (family, case) = cases[reader.choose(cases.len())];
    let family = &registry.families()[family];
    let requirements = requirements
        .get(&family.field_type())
        .expect("requirements for every planned family's field");
    crate::env::on_large_stack(|| family.shared_setup(case, &mut reader, requirements.as_ref()));
}

/// Every planned and excluded case, for the coverage report.
pub fn describe_cases(default_log2: u32) -> String {
    registry(limits(default_log2)).describe()
}
