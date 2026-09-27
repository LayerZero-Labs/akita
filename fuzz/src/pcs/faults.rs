//! Cheating provers: transcript-consistent proofs that violate one check.
//!
//! Byte mutations of an honest proof break Fiat–Shamir consistency first, so
//! they never reach the verifier's norm, range, cap, or relation checks. With
//! Akita's `fault-injection` feature the honest prover alters one value
//! before it is absorbed into the transcript: it admits a fold nonce whose
//! response fails its acceptance predicate, perturbs one recursive witness
//! digit (Z, E, T, or R), one terminal response coefficient, or the L2 norm
//! claim. Every later challenge is derived from the altered value, so only the
//! violated condition can catch it. For each input:
//!
//! - a fault that altered nothing (`applied == 0`) must leave the proof
//!   byte-identical to the honest one;
//! - a fault that altered something must be rejected by the verifier with an
//!   error, never accepted and never a panic; a clean prover error is allowed.

use super::family::{claims_of, honest_statement, Family, FamilyImpl, Honest};
use super::ops::PcsOps;
use crate::input::Reader;
use crate::stats;
use akita_error::AkitaError;
use akita_prover::fault_injection::{with_fault, Fault, WitnessSegment};

fn fault_from(control: &mut Reader<'_>) -> Fault {
    let level = u32::from(control.u8() % 6);
    // Mostly early indices: out-of-segment indices leave the fault unapplied.
    let index = match control.u8() % 4 {
        0 => control.u32() as usize % (1 << 16),
        _ => usize::from(control.u8()),
    };
    match control.u8() % 8 {
        0 | 1 => Fault::AcceptRejectedNonce {
            level: (control.u8() % 4 != 0).then_some(level),
            group: control.bool().then(|| usize::from(control.u8() % 4)),
        },
        2..=4 => Fault::PerturbWitnessDigit {
            level,
            segment: [
                WitnessSegment::Z,
                WitnessSegment::E,
                WitnessSegment::T,
                WitnessSegment::R,
            ][usize::from(control.u8() % 4)],
            index,
            delta: match control.u8() as i8 {
                0 => 1,
                delta => delta,
            },
        },
        5 | 6 => Fault::PerturbTerminalResponse {
            index,
            delta: match control.u16() as i16 {
                0 => 1,
                delta => i32::from(delta),
            },
        },
        _ => Fault::PerturbNormClaim {
            level,
            delta: match i64::from(control.u32() as i32) {
                0 => 1,
                delta => delta,
            },
        },
    }
}

impl<Cfg: PcsOps> FamilyImpl<Cfg> {
    pub(super) fn check_fault(&self, case: usize, honest: &Honest<Cfg>, control: &mut Reader<'_>) {
        let name = self.name();
        let fault = fault_from(control);
        let prepared = self.prepared();
        // Prove on this thread: faults are thread-local.
        let (result, report) = with_fault(fault, || {
            let opening = Cfg::select(
                claims_of(&honest.groups),
                honest
                    .groups
                    .iter()
                    .map(|group| group.handle.clone())
                    .collect(),
                &self.scheme,
            )?;
            Cfg::prove(
                &self.scheme,
                &prepared.setup,
                opening,
                &prepared.backend,
                &honest.session,
                honest.basis,
            )
        });
        let proved = match result {
            Ok(proved) => proved,
            Err(error) => {
                assert!(
                    report.applied > 0,
                    "{name}: an unapplied {fault:?} made the prover fail: {error:?}"
                );
                stats::count("fault_prover_error");
                return;
            }
        };
        if report.applied == 0 {
            assert!(
                proved.proof == honest.proved.proof,
                "{name}: unapplied {fault:?} changed the proof"
            );
            stats::count("fault_unapplied");
            return;
        }
        let verdict = self.verify(
            &proved.proof,
            &prepared.verifier,
            &honest.session,
            honest_statement(honest),
            honest.basis,
        );
        match verdict {
            Err(AkitaError::InvalidProof) => stats::count("fault_rejected"),
            // Any error rejects; other classes are counted, not findings.
            Err(_) => stats::count("fault_rejected_other_class"),
            Ok(()) => panic!(
                "{name}: SOUNDNESS: verifier accepted a proof built with {fault:?} \
                 ({} value(s) altered, case {})",
                report.applied,
                self.cases()[case].label()
            ),
        }
    }
}
