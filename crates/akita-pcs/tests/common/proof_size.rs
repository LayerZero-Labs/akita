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
use akita_types::{OpeningScheduleSelection, ScheduleLookupKey};
use std::fmt;
use std::sync::{Arc, Mutex};
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::Layer;

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

struct CaptureLayer(Arc<Mutex<VariableSectionBytes>>);

impl<S: Subscriber> Layer<S> for CaptureLayer {
    fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
        event.record(&mut *self.0.lock().expect("capture lock"));
    }
}

/// Run `prove`, then check its proof against the byte model of `selection`.
pub(crate) fn prove_matching_byte_model<Cfg: CommitmentConfig>(
    schedules: &TrustedScheduleCatalog<Cfg>,
    selection: OpeningScheduleSelection,
    prove: impl FnOnce() -> Result<Vec<u8>, AkitaError>,
) -> Vec<u8> {
    let observed = Arc::new(Mutex::new(VariableSectionBytes::default()));
    let proof = tracing::subscriber::with_default(
        tracing_subscriber::registry().with(CaptureLayer(Arc::clone(&observed))),
        prove,
    )
    .expect("prove");
    let observed = observed.lock().expect("capture lock");
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
        final_group: resolved.profiles().final_group.group,
        precommitteds: resolved.profiles().precommitteds.clone(),
    };
    let schedule = resolved.schedule();
    let policy = akita_config::policy_of::<Cfg>();
    let bound = akita_schedules::expanded_schedule_proof_bound(&key, schedule, &policy)
        .expect("native proof bound");
    let z_payload_cap = schedule.terminal.response_shape.layout.z_payload_bytes();
    let nonce_max = akita_types::derive_transcript_grinding_plan_from_public_shape(
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
