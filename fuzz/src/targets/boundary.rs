//! Public-boundary targets for prover requests and untrusted verifier input.

use super::pcs::limits;
use crate::input::Reader;
use crate::pcs::{registry, Selector};

/// Selector and default cost cap (`log2` coefficients) of each boundary
/// target, as in [`super::pcs`]. The verifier boundary builds one honest
/// fixture per case and then only verifies, so it affords recursive rows;
/// the prover boundary proves on most inputs and stays small.
pub const VERIFIER_CASES: (Selector, u32) = (Selector::Any, 20);
pub const PROVER_CASES: (Selector, u32) = (Selector::AnyDirect, 16);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Boundary {
    Verifier,
    Prover,
    TerminalCache,
}

fn run(data: &[u8], boundary: Boundary) {
    let (selector, default_log2) = if boundary == Boundary::Prover {
        PROVER_CASES
    } else {
        VERIFIER_CASES
    };
    let registry = registry(limits(default_log2));
    let cases = registry.select(selector);
    assert!(
        !cases.is_empty(),
        "no {selector:?} case fits the process limit {:?}",
        registry.limits()
    );
    let mut reader = Reader::new(data);
    let (family, case) = cases[reader.choose(cases.len())];
    let family = &registry.families()[family];
    crate::env::on_large_stack(|| match boundary {
        Boundary::Prover => family.prover_boundary(case, &mut reader),
        Boundary::Verifier => family.verifier_boundary(case, &mut reader),
        Boundary::TerminalCache => family.terminal_cache(case, &mut reader),
    });
}

pub fn verifier(data: &[u8]) {
    run(data, Boundary::Verifier);
}

pub fn prover(data: &[u8]) {
    run(data, Boundary::Prover);
}

pub fn terminal_cache(data: &[u8]) {
    run(data, Boundary::TerminalCache);
}
