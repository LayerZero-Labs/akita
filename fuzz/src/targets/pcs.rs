//! End-to-end commit/prove/verify targets over the shipped schedule rows.

use crate::input::Reader;
use crate::pcs::{registry, Check, Limits, Selector};

const MAX_COST_ENV: &str = "AKITA_FUZZ_MAX_CASE_COEFFS";

fn limits(default_log2: u32) -> Limits {
    let max_cost = std::env::var(MAX_COST_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1u64 << default_log2);
    Limits { max_cost }
}

fn run(data: &[u8], selector: Selector, check: Check, default_log2: u32) {
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
    run(data, Selector::DenseSingle, Check::Valid, 18);
}

pub fn onehot(data: &[u8]) {
    run(data, Selector::OneHotSingle, Check::Valid, 20);
}

pub fn batch(data: &[u8]) {
    run(data, Selector::Batch, Check::Valid, 20);
}

pub fn recursive(data: &[u8]) {
    run(data, Selector::Recursive, Check::Valid, 21);
}

pub fn reject(data: &[u8]) {
    run(data, Selector::AnyDirect, Check::Reject, 17);
}

pub fn parallel(data: &[u8]) {
    run(data, Selector::AnyDirect, Check::Parallel, 17);
}

pub fn liveness(data: &[u8]) {
    run(data, Selector::Any, Check::Liveness, 20);
}

/// One covering setup per field and one backend per field pair, shared by
/// every family of the process's registry (see `pcs::shared`).
pub fn shared(data: &[u8]) {
    use std::any::{Any, TypeId};
    use std::collections::HashMap;
    use std::sync::OnceLock;
    type Requirements = HashMap<TypeId, Box<dyn Any + Send + Sync>>;
    static REQUIREMENTS: OnceLock<Requirements> = OnceLock::new();
    let registry = registry(limits(17));
    let cases = registry.select(Selector::AnyDirect);
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
