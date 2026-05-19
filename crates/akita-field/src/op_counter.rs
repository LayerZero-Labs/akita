//! Verifier field-multiplication op-counter.
//!
//! Book §5.8 line 1104–1126 reports verifier op counts (field
//! multiplications) split by phase: challenge derivation (~18–22 % of
//! total), setup-matrix evaluation (~78–82 %), and sumcheck round
//! reconstruction. This module exposes an opt-in counter that
//! matches that taxonomy so we can produce book-comparable op counts
//! at NVs that fit our 123 GiB host and extrapolate to the book's
//! Table 1141–1158 measurement points NV ∈ {32, 38, 44}.
//!
//! # Contract
//!
//! - The counter is OFF by default. When the `op-counter` Cargo
//!   feature is disabled, every call into this module compiles to a
//!   no-op and the field `Mul` impls pay zero overhead.
//! - When the feature is on, the counter is still gated behind an
//!   atomic `ENABLED` flag (toggle via [`set_enabled`]) so most
//!   builds keep paying only one branch + cmov per multiplication.
//!   Production presets must NOT enable this feature; it is intended
//!   for measurement runs only.
//! - Categorization is delta-based: [`with_category`] reads the
//!   global total before calling `f`, calls `f`, reads the total
//!   after, and attributes the delta to the named category. This is
//!   robust to rayon thread spawning (every thread bumps the same
//!   global atomic), at the cost of not supporting nested categories
//!   (an inner `with_category` will double-count into both the inner
//!   and outer categories' deltas — wrap each top-level phase
//!   exclusively).
//!
//! # Usage sketch
//!
//! ```ignore
//! akita_field::op_counter::reset();
//! akita_field::op_counter::set_enabled(true);
//! akita_field::op_counter::with_category(
//!     akita_field::op_counter::OpCategory::Sumcheck,
//!     || verifier.verify(...),
//! );
//! akita_field::op_counter::set_enabled(false);
//! let snap = akita_field::op_counter::snapshot();
//! println!("{} field mults total", snap.total());
//! ```

/// Per-phase categories matching book §5.8 line 1104–1126.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpCategory {
    /// Challenge derivation: stage-1/stage-2/ring-switch sampling.
    Challenge,
    /// Setup-matrix evaluation: per-chunk + meta MLE and the shared
    /// `S` polynomial evaluation; the book §5.5 / §5.8 "setup ops"
    /// row.
    Setup,
    /// Sumcheck round reconstruction: per-round univariate
    /// extraction and accumulation in `verify_sumcheck_rounds_only`.
    Sumcheck,
    /// Anything not categorized via [`with_category`] — closing
    /// oracles, transcript Blake2b absorbs, defensive asserts, etc.
    Other,
}

/// Per-category counter snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpCount {
    /// Challenge derivation field-mults attributed to [`OpCategory::Challenge`].
    pub challenge_ops: u64,
    /// Setup-matrix evaluation field-mults attributed to [`OpCategory::Setup`].
    pub setup_ops: u64,
    /// Sumcheck round reconstruction field-mults attributed to [`OpCategory::Sumcheck`].
    pub sumcheck_ops: u64,
    /// Field-mults not attributed to a specific category.
    pub other_ops: u64,
    /// Raw global counter — equals `challenge + setup + sumcheck + other`
    /// only when every counted mult ran inside exactly one
    /// `with_category` scope. If counted mults occurred outside any
    /// `with_category` scope, those fall through to `other_ops` via
    /// `total - (challenge + setup + sumcheck)`.
    pub total: u64,
}

impl OpCount {
    /// Total field multiplications counted, regardless of category.
    pub fn total(&self) -> u64 {
        self.total
    }
}

#[cfg(feature = "op-counter")]
mod imp {
    use super::{OpCategory, OpCount};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    static TOTAL: AtomicU64 = AtomicU64::new(0);
    static CHALLENGE: AtomicU64 = AtomicU64::new(0);
    static SETUP: AtomicU64 = AtomicU64::new(0);
    static SUMCHECK: AtomicU64 = AtomicU64::new(0);
    static ENABLED: AtomicBool = AtomicBool::new(false);

    /// Return true when the counter is currently enabled (via [`set_enabled`]).
    #[inline(always)]
    pub fn enabled() -> bool {
        ENABLED.load(Ordering::Relaxed)
    }

    /// Toggle the runtime enable flag.
    pub fn set_enabled(on: bool) {
        ENABLED.store(on, Ordering::Relaxed);
    }

    /// Reset every per-category counter (and the global total) to zero.
    pub fn reset() {
        TOTAL.store(0, Ordering::Relaxed);
        CHALLENGE.store(0, Ordering::Relaxed);
        SETUP.store(0, Ordering::Relaxed);
        SUMCHECK.store(0, Ordering::Relaxed);
    }

    /// Read a [`OpCount`] snapshot of every counter.
    pub fn snapshot() -> OpCount {
        let total = TOTAL.load(Ordering::Relaxed);
        let challenge = CHALLENGE.load(Ordering::Relaxed);
        let setup = SETUP.load(Ordering::Relaxed);
        let sumcheck = SUMCHECK.load(Ordering::Relaxed);
        let categorized = challenge.saturating_add(setup).saturating_add(sumcheck);
        let other = total.saturating_sub(categorized);
        OpCount {
            challenge_ops: challenge,
            setup_ops: setup,
            sumcheck_ops: sumcheck,
            other_ops: other,
            total,
        }
    }

    /// Bump the global counter by one. Hot-path field `Mul` impls call this.
    #[inline(always)]
    pub fn bump() {
        if ENABLED.load(Ordering::Relaxed) {
            TOTAL.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Bump the global counter by `n`. Useful for batched primitives.
    #[inline(always)]
    pub fn bump_n(n: u64) {
        if ENABLED.load(Ordering::Relaxed) {
            TOTAL.fetch_add(n, Ordering::Relaxed);
        }
    }

    /// Run `f` and attribute every counter bump observed during its
    /// execution to `cat`. Robust to rayon thread spawns; not robust
    /// to nested `with_category` calls (inner deltas are
    /// double-attributed). See module docs.
    pub fn with_category<R>(cat: OpCategory, f: impl FnOnce() -> R) -> R {
        if !ENABLED.load(Ordering::Relaxed) {
            return f();
        }
        let before = TOTAL.load(Ordering::Relaxed);
        let result = f();
        let after = TOTAL.load(Ordering::Relaxed);
        let delta = after.saturating_sub(before);
        let bucket = match cat {
            OpCategory::Challenge => &CHALLENGE,
            OpCategory::Setup => &SETUP,
            OpCategory::Sumcheck => &SUMCHECK,
            OpCategory::Other => return result,
        };
        bucket.fetch_add(delta, Ordering::Relaxed);
        result
    }
}

#[cfg(not(feature = "op-counter"))]
mod imp {
    use super::{OpCategory, OpCount};

    /// Counter is always disabled when the `op-counter` feature is off.
    #[inline(always)]
    pub fn enabled() -> bool {
        false
    }

    /// No-op when the `op-counter` feature is off.
    #[inline(always)]
    pub fn set_enabled(_on: bool) {}

    /// No-op when the `op-counter` feature is off.
    #[inline(always)]
    pub fn reset() {}

    /// Returns a zero-initialized snapshot when the `op-counter` feature is off.
    #[inline(always)]
    pub fn snapshot() -> OpCount {
        OpCount::default()
    }

    /// No-op when the `op-counter` feature is off.
    #[inline(always)]
    pub fn bump() {}

    /// No-op when the `op-counter` feature is off.
    #[inline(always)]
    pub fn bump_n(_n: u64) {}

    /// Passes `f` through verbatim when the `op-counter` feature is off.
    #[inline(always)]
    pub fn with_category<R>(_cat: OpCategory, f: impl FnOnce() -> R) -> R {
        f()
    }
}

pub use imp::{bump, bump_n, enabled, reset, set_enabled, snapshot, with_category};
