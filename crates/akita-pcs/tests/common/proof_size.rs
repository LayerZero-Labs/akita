//! Real proof lengths against the schedule byte model.
//!
//! [`akita_schedules::expanded_schedule_proof_bound`] prices a schedule
//! as its fixed-width sections plus two variable-width sections at their caps:
//! the terminal Golomb-Rice `z` payload and the canonical LEB128 grinding
//! nonces. The planner ranks schedules with the same byte formulas, and the
//! akita-jolt recursion wire decoder rejects a proof blob longer than the
//! bound. The Akita verifier does not read the bound; it caps each message.
//!
//! A `proof.len() <= bound` check leaves the caps' unused bytes as slack, so a
//! fixed-width model that drifts from the prover by less than that slack goes
//! unnoticed. The prover reports the actual width of both variable sections
//! through its `native terminal response bytes` and `native proof nonce bytes`
//! tracing events in every build. Subtracting them from the proof length leaves
//! the fixed-width sections, which must match the model exactly; each variable
//! section must then stay within its cap, which keeps the proof within the
//! bound.

use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::{OpeningScheduleSelection, ScheduleLookupKey};
use std::cell::RefCell;
use std::fmt;
use std::sync::Once;
use tracing::field::{Field, Visit};
use tracing::subscriber::Interest;
use tracing::{Event, Metadata, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::Layer;

/// The two prover events that report variable-section widths.
const CAPTURED_FIELDS: [&str; 2] = ["native_terminal_z_bytes", "native_nonce_bytes_actual"];

/// Variable-section widths the prover reported during one `batched_prove`.
#[derive(Debug, Default)]
struct VariableSectionBytes {
    z_payload: Vec<u64>,
    nonce: Vec<u64>,
}

impl Visit for VariableSectionBytes {
    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "native_terminal_z_bytes" => self.z_payload.push(value),
            "native_nonce_bytes_actual" => self.nonce.push(value),
            _ => {}
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn fmt::Debug) {}
}

thread_local! {
    /// Widths reported on this thread while a capture is active.
    static CAPTURE: RefCell<Option<VariableSectionBytes>> = const { RefCell::new(None) };
}

/// Records the width events into the reporting thread's active capture.
///
/// It is interested only in those two events, so every other callsite, and
/// every `tracing::enabled!` diagnostics guard, stays disabled as it would be
/// with no subscriber.
struct CaptureLayer;

fn captures(metadata: &Metadata<'_>) -> bool {
    metadata.is_event()
        && CAPTURED_FIELDS
            .iter()
            .any(|name| metadata.fields().field(name).is_some())
}

impl<S: Subscriber> Layer<S> for CaptureLayer {
    fn register_callsite(&self, metadata: &'static Metadata<'static>) -> Interest {
        if captures(metadata) {
            Interest::always()
        } else {
            Interest::never()
        }
    }

    fn enabled(&self, metadata: &Metadata<'_>, _context: Context<'_, S>) -> bool {
        captures(metadata)
    }

    fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
        CAPTURE.with(|capture| {
            if let Some(bytes) = capture.borrow_mut().as_mut() {
                event.record(bytes);
            }
        });
    }
}

/// Install the capture layer as the global default, once per test binary.
///
/// A scoped subscriber is not enough. While at most one dispatcher is
/// registered, tracing-core computes a callsite's interest from the default of
/// whichever thread reaches it first. A concurrent test that proves without a
/// subscriber can then cache both width callsites as never interested, and
/// this capture silently records nothing. With a global default, every thread
/// that has no scoped subscriber sees an interested default.
fn install_capture_layer() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        tracing::subscriber::set_global_default(tracing_subscriber::registry().with(CaptureLayer))
            .expect("the proof-size capture is the test binary's only global subscriber");
    });
}

/// Run `prove`, then check its proof against the byte model of `selection`.
pub(crate) fn prove_matching_byte_model<Cfg: CommitmentConfig>(
    schedules: &TrustedScheduleCatalog<Cfg>,
    selection: OpeningScheduleSelection,
    prove: impl FnOnce() -> Result<Vec<u8>, AkitaError>,
) -> Vec<u8> {
    install_capture_layer();
    CAPTURE.with(|capture| *capture.borrow_mut() = Some(VariableSectionBytes::default()));
    let proof = prove();
    let observed = CAPTURE
        .with(|capture| capture.borrow_mut().take())
        .expect("the capture stays active for the whole prove");
    let proof = proof.expect("prove");
    let (&[z_payload], &[nonce]) = (observed.z_payload.as_slice(), observed.nonce.as_slice())
    else {
        panic!("prover must report one terminal z payload and one nonce total: {observed:?}");
    };
    let (z_payload, nonce) = (
        usize::try_from(z_payload).expect("z payload width"),
        usize::try_from(nonce).expect("nonce width"),
    );

    let resolved = schedules
        .resolve_selection(selection)
        .expect("selected schedule");
    let key = ScheduleLookupKey {
        root_accumulated_setup_prefix_len: 0,
        final_group: resolved.profiles().final_group.group,
        precommitteds: resolved.profiles().precommitteds.clone(),
    };
    let schedule = resolved.schedule();
    let policy = schedules.policy();
    let bound = akita_schedules::expanded_schedule_proof_bound(&key, schedule, policy)
        .expect("native proof bound");
    let z_payload_cap = schedule.terminal.response_shape.layout.z_payload_bytes();
    let nonce_max = akita_params::derive_transcript_grinding_plan_from_public_shape(
        schedule,
        &key.opening_layout().expect("opening layout"),
        policy.transcript_grinding_order().expect("grinding order"),
        policy.claim_ext_degree,
    )
    .expect("grinding plan")
    .nonce_max_bytes();
    let model_fixed = bound
        .checked_sub(z_payload_cap)
        .and_then(|rest| rest.checked_sub(nonce_max))
        .expect("the bound contains both variable-section caps");
    let actual_fixed = proof
        .len()
        .checked_sub(z_payload)
        .and_then(|rest| rest.checked_sub(nonce))
        .expect("the proof contains both reported variable sections");

    assert_eq!(
        actual_fixed,
        model_fixed,
        "fixed-width sections must match the schedule byte model \
         (proof {} bytes, z payload {z_payload}, nonces {nonce})",
        proof.len(),
    );
    assert!(
        z_payload <= z_payload_cap,
        "terminal z payload {z_payload} exceeds its cap {z_payload_cap}"
    );
    assert!(
        nonce <= nonce_max,
        "nonce bytes {nonce} exceed their maximum {nonce_max}"
    );
    proof
}
