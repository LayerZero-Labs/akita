//! Public-boundary targets for prover requests and untrusted verifier input.

use crate::input::Reader;
use crate::pcs::{registry, Limits, Selector};

/// Per-process case limits. The verifier boundary builds one honest fixture
/// per case and then only verifies, so it affords recursive rows; the prover
/// boundary proves on most inputs and stays small.
const VERIFIER_LIMITS: Limits = Limits { max_cost: 1 << 20 };
const PROVER_LIMITS: Limits = Limits { max_cost: 1 << 16 };

#[derive(Clone, Copy, PartialEq, Eq)]
enum Boundary {
    Verifier,
    Prover,
    TerminalCache,
}

fn run(data: &[u8], boundary: Boundary) {
    let prover = boundary == Boundary::Prover;
    let (limits, selector) = if prover {
        (PROVER_LIMITS, Selector::AnyDirect)
    } else {
        (VERIFIER_LIMITS, Selector::Any)
    };
    let registry = registry(limits);
    let cases = registry.select(selector);
    let mut reader = Reader::new(data);
    let (family, case) = cases[reader.choose(cases.len())];
    let family = &registry.families()[family];
    crate::env::on_large_stack(|| match boundary {
        Boundary::Prover => family.prover_boundary(case, &mut reader),
        Boundary::Verifier => family.verifier_boundary(case, &mut reader),
        Boundary::TerminalCache => family.terminal_cache(case, &mut reader),
    });
}

/// Selector and limits each boundary target uses, for seed generation.
pub const VERIFIER_CASES: (Selector, Limits) = (Selector::Any, VERIFIER_LIMITS);
pub const PROVER_CASES: (Selector, Limits) = (Selector::AnyDirect, PROVER_LIMITS);

pub fn verifier(data: &[u8]) {
    run(data, Boundary::Verifier);
}

pub fn prover(data: &[u8]) {
    run(data, Boundary::Prover);
}

pub fn terminal_cache(data: &[u8]) {
    run(data, Boundary::TerminalCache);
}
