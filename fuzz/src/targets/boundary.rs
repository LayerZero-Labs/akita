//! Public-boundary targets for prover requests and untrusted verifier input.

use crate::input::Reader;
use crate::pcs::{registry, Limits, Selector};

fn run(data: &[u8], prover: bool) {
    let registry = registry(Limits { max_cost: 1 << 16 });
    let cases = registry.select(Selector::AnyDirect);
    let mut reader = Reader::new(data);
    let (family, case) = cases[reader.choose(cases.len())];
    let family = &registry.families()[family];
    crate::env::on_large_stack(|| {
        if prover {
            family.prover_boundary(case, &mut reader);
        } else {
            family.verifier_boundary(case, &mut reader);
        }
    });
}

pub fn verifier(data: &[u8]) {
    run(data, false);
}

pub fn prover(data: &[u8]) {
    run(data, true);
}
