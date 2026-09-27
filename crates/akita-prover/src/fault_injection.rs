//! Test-only prover fault injection (feature `fault-injection`).
//!
//! A harness wraps an otherwise honest proving call in [`with_fault`]. The
//! prover then violates exactly one verifier-enforced condition while every
//! Fiat–Shamir challenge is still derived from the messages it actually sends:
//! each fault is applied before the affected value is absorbed. The verifier
//! must reject every such proof whose [`FaultReport::applied`] is non-zero.
//!
//! Prover self-checks that would refuse to emit the faulty proof (sumcheck
//! round consistency, final-claim agreement, the L2 cap re-check, and terminal
//! payload re-decoding) are skipped while any fault is installed. Nothing here
//! is compiled without the feature, and the feature is never enabled by
//! default.

use std::cell::Cell;

/// Recursive-witness segment addressed by [`Fault::PerturbWitnessDigit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WitnessSegment {
    /// Balanced digits of the folded response `z`.
    Z,
    /// Opening digits `e`.
    E,
    /// Outer commitment-hint digits `t`.
    T,
    /// Relation quotient rows `r` (quotient-lift levels only).
    R,
}

/// One verifier-enforced condition for the prover to violate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Fault {
    /// Commit a fold-grind nonce at which the response fails its acceptance
    /// predicate (L2 cap or balanced-digit L∞ range).
    ///
    /// `level` selects the fold level (`None`: every level) and `group` the
    /// group whose predicate is skipped (`None`: every group; the terminal
    /// level has group 0 only). The grind commits the first nonce at which a
    /// selected group is rejected while every other group is honestly
    /// accepted. When no such nonce exists within the attempt budget, it falls
    /// back to the honest nonce and the fault is not applied. Rejected
    /// recursive responses are decomposed with wraparound into the scheduled
    /// digits; rejected terminal responses are Golomb–Rice encoded with a
    /// widened zigzag width, provided they still fit the payload budget.
    AcceptRejectedNonce {
        level: Option<u32>,
        group: Option<usize>,
    },
    /// Add `delta` to one digit of the recursive witness built at `level`
    /// before it is committed. `index` is relative to the start of the
    /// segment, counted across units (and R rows) in physical order. The
    /// result is wrapped into the witness's balanced digit range, since the
    /// CPU backend cannot store out-of-range digits.
    PerturbWitnessDigit {
        level: u32,
        segment: WitnessSegment,
        index: usize,
        delta: i8,
    },
    /// Add `delta` to one centered coefficient of the terminal response `z`
    /// before Golomb–Rice encoding.
    PerturbTerminalResponse { index: usize, delta: i32 },
    /// Add `delta` (in the field) to the claimed squared L2 norm of the fold
    /// response at `level` before it is absorbed. Applies only to levels on
    /// the L2 security route.
    PerturbNormClaim { level: u32, delta: i64 },
}

/// What the prover did while a fault was installed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaultReport {
    /// Number of values the fault actually altered.
    pub applied: usize,
}

thread_local! {
    static ACTIVE: Cell<Option<Fault>> = const { Cell::new(None) };
    static APPLIED: Cell<usize> = const { Cell::new(0) };
    static ADMISSION_BYPASS: Cell<bool> = const { Cell::new(false) };
    static BYPASSED_REJECTION: Cell<bool> = const { Cell::new(false) };
}

/// Run `f` with `fault` installed on this thread and report its effect.
///
/// Proving must run on the calling thread; faults are not visible to worker
/// threads. Nested calls restore the outer fault on return.
pub fn with_fault<R>(fault: Fault, f: impl FnOnce() -> R) -> (R, FaultReport) {
    struct Restore(Option<Fault>, usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            ACTIVE.set(self.0);
            APPLIED.set(self.1);
        }
    }
    let _restore = Restore(ACTIVE.replace(Some(fault)), APPLIED.replace(0));
    let result = akita_sumcheck::fault_injection::with_round_claim_check_skipped(f);
    let report = FaultReport {
        applied: APPLIED.get(),
    };
    (result, report)
}

/// The fault installed on this thread, if any.
pub fn active() -> Option<Fault> {
    ACTIVE.get()
}

/// Record that the active fault altered one value.
pub fn record_applied() {
    APPLIED.set(APPLIED.get() + 1);
}

/// Whether the backend should admit the fold response it is probing even if
/// it fails the acceptance predicate. Set by the protocol for selected groups
/// under [`Fault::AcceptRejectedNonce`].
pub fn admission_bypass_requested() -> bool {
    ADMISSION_BYPASS.get()
}

/// Called by a backend that admitted a response only because of
/// [`admission_bypass_requested`].
pub fn record_bypassed_rejection() {
    BYPASSED_REJECTION.set(true);
}

/// Probe with the admission bypass set to `bypass`; also return whether the
/// backend admitted a response that it would have rejected.
pub(crate) fn probe_with_admission_bypass<T>(bypass: bool, probe: impl FnOnce() -> T) -> (T, bool) {
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            ADMISSION_BYPASS.set(false);
            BYPASSED_REJECTION.set(false);
        }
    }
    let _clear = Clear;
    ADMISSION_BYPASS.set(bypass);
    BYPASSED_REJECTION.set(false);
    let value = probe();
    (value, BYPASSED_REJECTION.get())
}

/// Groups at `level` whose admission is skipped by the active fault.
pub(crate) fn accept_rejected_target(level: u32) -> Option<Option<usize>> {
    match active()? {
        Fault::AcceptRejectedNonce {
            level: fault_level,
            group,
        } if fault_level.is_none_or(|fault_level| fault_level == level) => Some(group),
        _ => None,
    }
}

/// Whether prover self-checks that would refuse to emit a faulty proof are
/// skipped. They are skipped for every installed fault; a check only matters
/// once a fault has already broken the statement it re-derives.
pub fn skips_self_checks() -> bool {
    active().is_some()
}

/// Delta to add to the claimed squared L2 norm at `level`.
pub(crate) fn norm_claim_delta(level: u32) -> Option<i64> {
    match active()? {
        Fault::PerturbNormClaim {
            level: fault_level,
            delta,
        } if fault_level == level => Some(delta),
        _ => None,
    }
}
