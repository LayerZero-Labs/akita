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

/// Every planned and excluded case, for the coverage report.
pub fn describe_cases(default_log2: u32) -> String {
    registry(limits(default_log2)).describe()
}
