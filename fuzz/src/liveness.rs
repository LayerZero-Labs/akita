//! Completeness margins of the prover's fold-response grinds.
//!
//! Each fold previews Fiat–Shamir challenges for nonces `0, 1, …` until the
//! folded response's energy fits the cap frozen in the schedule, and gives up
//! with `fold grind exceeded … joint attempts` after `FOLD_RESPONSE_ATTEMPTS`
//! (4096). The planner freezes `cap = (40/39) · 1.03 · modeled mean`, where
//! the mean is `E‖c‖² · ‖s‖²` for the source `s`, so Markov's inequality gives
//! every attempt acceptance probability at least 1/40, but only while the
//! witness's true source energy stays within the modeled one plus its 3%
//! envelope. Root sources are modeled at their worst case; recursive witnesses
//! (Z, E, T, R, and compression digits) use distribution models, so a witness
//! or opening point that concentrates those digits can void the guarantee.
//!
//! With `response-model-diagnostics` the prover reports every selected fold:
//! its attempt count, observed response energy, cap, and (for recursive and
//! terminal folds) the measured conditional mean. [`observe`] captures those
//! reports around one proof and checks, for every fold:
//!
//! - the grind never ran out (any `fold grind exceeded` error panics here,
//!   whatever the caller expected);
//! - the accepted response fits its cap;
//! - `40 · measured mean ≤ 39 · cap`: the planner's model covers this witness
//!   (otherwise the completeness guarantee no longer applies to it, even if
//!   this proof succeeded);
//! - on L-infinity-route folds (no L2 cap; acceptance means every response
//!   coefficient fits `num_digits_fold` balanced digits), the observed
//!   response spread still gets joint acceptance of at least 1/40 under the
//!   planner's own Gaussian model ([`Sample::linf_margin`]);
//! - attempts stay below [`ATTEMPT_ALARM`], which a covered fold exceeds with
//!   probability at most `(39/40)^1024 < 2^-37`.
//!
//! Both margins and the attempt count are also reported to the engine as
//! coverage (one distinct function per bucket), so inputs that push a fold
//! closer to its limit are kept and mutated further.

use crate::stats;
use akita_error::AkitaError;
use std::cell::RefCell;
use std::fmt;
use std::sync::{Mutex, Once};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

/// Target of the diagnostics-only reports (`response-model-diagnostics`).
const MODEL_TARGET: &str = "akita_prover::protocol::fold_response_model";
/// Target of the always-on "selected physical fold response" report.
const GRIND_TARGET: &str = "akita_prover::protocol::fold_grind";
/// Attempts beyond this are a finding: a fold the planner's model covers gets
/// here with probability at most `(39/40)^1024 < 2^-37`.
pub const ATTEMPT_ALARM: u64 = 1024;
const MAX_SAMPLES: usize = 1 << 14;

/// One prover report.
#[derive(Clone, Debug, Default)]
pub struct Sample {
    /// A diagnostics report (`fold_response_model`), not the always-on one.
    pub model: bool,
    pub terminal: bool,
    pub message: String,
    pub attempts: Option<u64>,
    pub response: Option<u128>,
    pub cap: Option<u128>,
    pub conditional_mean: Option<u128>,
    /// Every numeric field, for reports.
    pub fields: Vec<(&'static str, u128)>,
}

impl Sample {
    /// `40 · mean / (39 · cap)`: 1.0 is the edge of the planner's guarantee.
    pub fn margin(&self) -> Option<f64> {
        let (mean, cap) = (self.conditional_mean?, self.cap?);
        (cap > 0).then(|| mean as f64 * 40.0 / (cap as f64 * 39.0))
    }

    fn field(&self, name: &str) -> Option<u128> {
        self.fields
            .iter()
            .find_map(|(field, value)| (*field == name).then_some(*value))
    }

    /// L-infinity-route folds (no L2 cap): `σ · x_n / bound`, where `σ` is
    /// the observed response RMS, `bound` the smaller side of the range
    /// `num_digits_response` balanced digits represent, and `x_n` the
    /// two-sided normal quantile at which `n` independent coordinates all fit
    /// with probability 1/40. The planner sizes the digit count so its
    /// modeled response has joint acceptance at least 1/40 (Gaussian
    /// correlation inequality, `whole_response_normal_quantile`); above 1.0
    /// the observed spread gets less than that under the planner's own model.
    pub fn linf_margin(&self) -> Option<f64> {
        if self.cap.is_some() || self.terminal || !self.model {
            return None;
        }
        let response = self.response?;
        let coeffs = self.field("response_coeffs")?;
        let log_basis = u32::try_from(self.field("log_basis_response")?).ok()?;
        let digits = u32::try_from(self.field("num_digits_response")?).ok()?;
        if coeffs == 0 || !(2..=32).contains(&log_basis) || digits == 0 {
            return None;
        }
        let base = (1u128 << log_basis) as f64;
        let span = (base.powi(digits as i32) - 1.0) / (base - 1.0);
        let bound = (base / 2.0 - 1.0) * span;
        let sigma = (response as f64 / coeffs as f64).sqrt();
        (bound > 0.0).then(|| sigma * joint_quantile(coeffs as f64) / bound)
    }
}

/// `x` with `(1 - 2Q(x))^n = 1/40`, `Q` the standard normal upper tail.
fn joint_quantile(n: f64) -> f64 {
    // Per-coordinate two-sided tail: 1 - (1/40)^(1/n).
    let tail = -(-(40f64.ln()) / n).exp_m1();
    let (mut low, mut high) = (0.0f64, 40.0f64);
    for _ in 0..100 {
        let mid = 0.5 * (low + high);
        if 2.0 * upper_tail(mid) > tail {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// Standard normal upper tail `Q(x)` for `x ≥ 0`: Simpson integration of
/// the density over `[x, x + 12]` (relative error far below what the
/// margin needs, also deep in the tail).
fn upper_tail(x: f64) -> f64 {
    const STEPS: usize = 2048;
    let h = 12.0 / STEPS as f64;
    let density = |t: f64| (-0.5 * t * t).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let mut sum = density(x) + density(x + 12.0);
    for step in 1..STEPS {
        let weight = if step % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * density(x + step as f64 * h);
    }
    sum * h / 3.0
}

impl fmt::Display for Sample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        for (name, value) in &self.fields {
            write!(f, " {name}={value}")?;
        }
        Ok(())
    }
}

/// Largest values seen since the last [`take_peak`].
#[derive(Clone, Debug, Default)]
pub struct Peak {
    pub folds: u64,
    /// Sum of attempts over `folds`; the mean estimates `1 / acceptance`.
    pub attempts_total: u64,
    pub max_attempts: u64,
    pub max_margin: f64,
    pub max_linf_margin: f64,
    pub worst: Option<Sample>,
}

thread_local! {
    static CAPTURE: RefCell<Option<Vec<Sample>>> = const { RefCell::new(None) };
}
static PEAK: Mutex<Option<Peak>> = Mutex::new(None);

pub fn take_peak() -> Peak {
    PEAK.lock()
        .ok()
        .and_then(|mut peak| peak.take())
        .unwrap_or_default()
}

struct Visitor<'a>(&'a mut Sample);

impl Visitor<'_> {
    fn number(&mut self, field: &Field, value: Option<u128>) {
        let Some(value) = value else { return };
        let sample = &mut *self.0;
        match field.name() {
            "attempts" => sample.attempts = u64::try_from(value).ok(),
            "response_l2_sq" => sample.response = Some(value),
            "response_l2_sq_cap" => sample.cap = Some(value),
            "conditional_mean_l2_sq" => sample.conditional_mean = Some(value),
            _ => {}
        }
        sample.fields.push((field.name(), value));
    }
}

impl Visit for Visitor<'_> {
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.number(field, Some(u128::from(value)));
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.number(field, u128::try_from(value).ok());
    }
    fn record_u128(&mut self, field: &Field, value: u128) {
        self.number(field, Some(value));
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        if field.name() == "terminal" {
            self.0.terminal = value;
        }
    }
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        let text = format!("{value:?}");
        if field.name() == "message" {
            self.0.message = text;
            return;
        }
        // Optional fields arrive as `Some(n)` / `None`.
        let inner = text
            .strip_prefix("Some(")
            .and_then(|rest| rest.strip_suffix(')'))
            .unwrap_or(&text);
        self.number(field, inner.parse().ok());
    }
}

struct Collector;

fn wanted(metadata: &Metadata<'_>) -> bool {
    // `tracing::enabled!` probes a hint callsite, not an event.
    !metadata.is_span() && matches!(metadata.target(), MODEL_TARGET | GRIND_TARGET)
}

impl Subscriber for Collector {
    fn register_callsite(
        &self,
        metadata: &'static Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        if wanted(metadata) {
            tracing::subscriber::Interest::always()
        } else {
            tracing::subscriber::Interest::never()
        }
    }
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        wanted(metadata)
    }
    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::INFO)
    }
    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }
    fn record(&self, _: &Id, _: &Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
    fn event(&self, event: &Event<'_>) {
        let mut sample = Sample {
            model: event.metadata().target() == MODEL_TARGET,
            ..Sample::default()
        };
        event.record(&mut Visitor(&mut sample));
        let captured = CAPTURE.with(|capture| match capture.borrow_mut().as_mut() {
            Some(samples) => {
                if samples.len() < MAX_SAMPLES {
                    samples.push(sample);
                }
                true
            }
            None => false,
        });
        if !captured {
            // Emitted on a thread other than the proving one: not checked.
            stats::count("liveness_uncaptured_reports");
        }
    }
}

fn install() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        if tracing::subscriber::set_global_default(Collector).is_err() {
            stats::count("liveness_subscriber_unavailable");
        }
    });
}

/// A bounded prover-side retry loop gave up: the fold grind (4096 nonces) or
/// operator-norm challenge rejection (4096 draws per coordinate). Neither is
/// an input validation failure, whatever error variant reports it.
pub fn is_liveness_exhaustion(error: &AkitaError) -> bool {
    matches!(error, AkitaError::InvalidInput(message) | AkitaError::Internal(message)
        if message.contains("fold grind exceeded")
            || message.contains("operator-norm rejection exceeded"))
}

/// Report a closeness ratio in `[0.25, 1.25)` (1.0 = at a limit) to the
/// engine as coverage, for checks outside the fold reports.
pub fn guide_ratio(ratio: f64) {
    guide::ratio(ratio);
}

/// Run one proof, capture its fold reports, and check their margins.
pub fn observe<T>(
    context: &str,
    prove: impl FnOnce() -> Result<T, AkitaError>,
) -> Result<T, AkitaError> {
    install();
    let previous = CAPTURE.with(|capture| capture.replace(Some(Vec::new())));
    let result = prove();
    let samples = CAPTURE
        .with(|capture| std::mem::replace(&mut *capture.borrow_mut(), previous))
        .unwrap_or_default();
    match &result {
        Err(error) if is_liveness_exhaustion(error) => panic!(
            "liveness: {context}: prover retry loop exhausted ({error:?}); last reports:\n{}",
            tail(&samples)
        ),
        Ok(_) => check(context, &samples),
        Err(_) => {}
    }
    result
}

fn tail(samples: &[Sample]) -> String {
    let start = samples.len().saturating_sub(4);
    samples[start..]
        .iter()
        .map(|sample| format!("  {sample}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn check(context: &str, samples: &[Sample]) {
    let log = std::env::var_os("AKITA_FUZZ_LIVENESS_LOG").is_some();
    let mut peak = Peak::default();
    for (index, sample) in samples.iter().enumerate() {
        if log {
            eprintln!("liveness: {context}: {sample}");
        }
        // Witness-moment reports carry no fold outcome.
        let Some(attempts) = sample.attempts else {
            continue;
        };
        // The always-on report of a fold precedes its diagnostics report.
        if !sample.model
            && samples.get(index + 1).is_some_and(|next| {
                next.model && next.attempts == sample.attempts && next.response == sample.response
            })
        {
            continue;
        }
        peak.folds += 1;
        stats::count("liveness_folds");
        stats::count(ATTEMPT_COUNTERS[attempt_bucket(attempts)]);
        guide::attempts(sample.terminal, attempt_bucket(attempts));
        peak.max_attempts = peak.max_attempts.max(attempts);
        peak.attempts_total += attempts;
        if let (Some(response), Some(cap)) = (sample.response, sample.cap) {
            assert!(
                response <= cap,
                "liveness: {context}: accepted fold response exceeds its cap: {sample}"
            );
        }
        if let Some(margin) = sample.margin() {
            stats::count(MARGIN_COUNTERS[margin_counter(margin)]);
            guide::margin(sample.terminal, margin);
            if margin > peak.max_margin {
                peak.max_margin = margin;
                peak.worst = Some(sample.clone());
            }
            let (mean, cap) = (sample.conditional_mean.unwrap(), sample.cap.unwrap());
            let covered = match (mean.checked_mul(40), cap.checked_mul(39)) {
                (Some(lhs), Some(rhs)) => lhs <= rhs,
                _ => margin <= 1.0,
            };
            assert!(
                covered,
                "liveness: {context}: fold source energy exceeds the planner's response model \
                 (margin {margin:.4} > 1; completeness guarantee void): {sample}\n\
                 witness moments before this fold:\n{}",
                moments_before(samples, index)
            );
        }
        if let Some(margin) = sample.linf_margin() {
            stats::count(MARGIN_COUNTERS[margin_counter(margin)]);
            guide::linf(margin);
            peak.max_linf_margin = peak.max_linf_margin.max(margin);
            assert!(
                margin <= 1.0,
                "liveness: {context}: fold response spread exceeds the planner's L-infinity \
                 digit budget (margin {margin:.4} > 1; per-attempt acceptance below 1/40 under \
                 its Gaussian model): {sample}\nwitness moments before this fold:\n{}",
                moments_before(samples, index)
            );
        }
        assert!(
            attempts <= ATTEMPT_ALARM,
            "liveness: {context}: fold grind needed {attempts} attempts (alarm {ATTEMPT_ALARM}, \
             limit 4096): {sample}"
        );
    }
    if let Ok(mut global) = PEAK.lock() {
        let global = global.get_or_insert_with(Peak::default);
        global.folds += peak.folds;
        global.attempts_total += peak.attempts_total;
        global.max_attempts = global.max_attempts.max(peak.max_attempts);
        global.max_linf_margin = global.max_linf_margin.max(peak.max_linf_margin);
        if peak.max_margin > global.max_margin {
            global.max_margin = peak.max_margin;
            global.worst = peak.worst;
        }
    }
}

fn moments_before(samples: &[Sample], index: usize) -> String {
    samples[..index]
        .iter()
        .rev()
        .find(|sample| sample.message.contains("source moments"))
        .map(|sample| format!("  {sample}"))
        .unwrap_or_else(|| "  (none reported)".into())
}

fn attempt_bucket(attempts: u64) -> usize {
    (64 - attempts.leading_zeros() as usize).min(ATTEMPT_COUNTERS.len() - 1)
}

const ATTEMPT_COUNTERS: [&str; 14] = [
    "liveness_attempts_0",
    "liveness_attempts_1",
    "liveness_attempts_2-3",
    "liveness_attempts_4-7",
    "liveness_attempts_8-15",
    "liveness_attempts_16-31",
    "liveness_attempts_32-63",
    "liveness_attempts_64-127",
    "liveness_attempts_128-255",
    "liveness_attempts_256-511",
    "liveness_attempts_512-1023",
    "liveness_attempts_1024-2047",
    "liveness_attempts_2048-4095",
    "liveness_attempts_4096",
];

fn margin_counter(margin: f64) -> usize {
    MARGIN_EDGES
        .iter()
        .position(|&edge| margin < edge)
        .unwrap_or(MARGIN_EDGES.len())
}

const MARGIN_EDGES: [f64; 7] = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0];
const MARGIN_COUNTERS: [&str; 8] = [
    "liveness_margin_lt_0.25",
    "liveness_margin_lt_0.50",
    "liveness_margin_lt_0.75",
    "liveness_margin_lt_0.90",
    "liveness_margin_lt_0.95",
    "liveness_margin_lt_0.99",
    "liveness_margin_lt_1.00",
    "liveness_margin_ge_1.00",
];

/// Margin and attempt buckets as engine-visible coverage: each bucket calls a
/// distinct function, so reaching a new bucket is a new edge.
mod guide {
    #[inline(never)]
    fn reached<const N: usize>() {
        std::hint::black_box(N);
    }

    macro_rules! marks {
        ($($n:literal)*) => { [$(reached::<$n> as fn()),*] };
    }

    /// Margin in 1/64 steps over `[0.25, 1.25)`: L2-route non-terminal
    /// (0..64) and terminal (64..128) folds, L-infinity-route folds (128..192).
    static MARGIN: [fn(); 192] = marks!(
        0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191
    );
    /// Attempt buckets (log2) for non-terminal (192..208) and terminal
    /// (208..224) folds.
    static ATTEMPTS: [fn(); 32] = marks!(
        192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223
    );

    pub(super) fn margin(terminal: bool, margin: f64) {
        MARGIN[usize::from(terminal) * 64 + step(margin)]();
    }

    /// Other closeness ratios (224..288).
    static RATIO: [fn(); 64] = marks!(
        224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287
    );

    pub(super) fn ratio(ratio: f64) {
        RATIO[step(ratio)]();
    }

    pub(super) fn linf(margin: f64) {
        MARGIN[128 + step(margin)]();
    }

    fn step(margin: f64) -> usize {
        ((margin - 0.25) * 64.0).clamp(0.0, 63.0) as usize
    }

    pub(super) fn attempts(terminal: bool, bucket: usize) {
        ATTEMPTS[usize::from(terminal) * 16 + bucket.min(15)]();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_tail_matches_reference_values() {
        for (x, expected) in [
            (0.0, 0.5),
            (1.96, 0.024_997_9),
            (4.0, 3.167_124e-5),
            (6.0, 9.865_876e-10),
        ] {
            let q = upper_tail(x);
            assert!(
                (q - expected).abs() <= 1e-6 * expected,
                "Q({x}) = {q}, expected {expected}"
            );
        }
    }

    #[test]
    fn joint_quantile_gives_one_fortieth_joint_acceptance() {
        for n in [1.0, 64.0, 65_536.0, 1e7] {
            let x = joint_quantile(n);
            let joint = (1.0 - 2.0 * upper_tail(x)).powf(n);
            assert!((joint - 1.0 / 40.0).abs() < 1e-6, "n = {n}: joint {joint}");
        }
    }
}
