use super::*;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

const RESPONSE_MODEL_TARGET: &str = "akita_prover::protocol::fold_response_model";

#[derive(Clone, Debug, Default)]
struct CapturedEvent {
    fields: BTreeMap<String, String>,
}

#[derive(Default)]
struct FieldVisitor {
    fields: BTreeMap<String, String>,
}

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.fields
            .insert(field.name().to_string(), format!("{value:?}"));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }
}

#[derive(Clone)]
struct CaptureLayer {
    events: Arc<Mutex<Vec<CapturedEvent>>>,
}

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
        if event.metadata().target() != RESPONSE_MODEL_TARGET {
            return;
        }
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        self.events.lock().unwrap().push(CapturedEvent {
            fields: visitor.fields,
        });
    }
}

fn optional_u128(value: &str) -> Option<u128> {
    value.strip_prefix("Some(")?.strip_suffix(')')?.parse().ok()
}

fn required_u128(event: &CapturedEvent, field: &str) -> u128 {
    let value = event.fields.get(field).unwrap_or_else(|| {
        panic!("response-model event omitted required field {field}: {event:?}")
    });
    optional_u128(value).unwrap_or_else(|| {
        value.parse().unwrap_or_else(|_| {
            panic!("response-model field {field} is not numeric: {value}; event={event:?}")
        })
    })
}

#[test]
fn accepted_fold_and_terminal_events_preserve_calibration_fields() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let thread_events = Arc::clone(&events);
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || {
            let subscriber = tracing_subscriber::registry().with(CaptureLayer {
                events: thread_events,
            });
            tracing::subscriber::with_default(subscriber, || {
                let _fixture = make_verify_fixture(16);
            });
        })
        .unwrap()
        .join()
        .unwrap();

    let events = events.lock().unwrap();
    let samples = events
        .iter()
        .filter(|event| {
            event
                .fields
                .get("message")
                .is_some_and(|message| message.contains("fold response model sample"))
        })
        .collect::<Vec<_>>();
    assert!(
        !samples.is_empty(),
        "no response-model samples were emitted"
    );
    assert!(
        samples
            .iter()
            .any(|event| !event.fields.contains_key("terminal")),
        "ordinary fold response-model sample is missing"
    );
    assert!(
        samples
            .iter()
            .any(|event| event.fields.contains_key("terminal")),
        "terminal fold response-model sample is missing"
    );

    let mut exact_ordinary_source_samples = 0;
    let mut exact_terminal_source_samples = 0;
    for event in samples {
        let response_l2_sq = required_u128(event, "response_l2_sq");
        let source = event.fields.get("source_l2_sq").unwrap();
        let conditional = event.fields.get("conditional_mean_l2_sq").unwrap();
        if let Some(source_l2_sq) = optional_u128(source) {
            if event.fields.contains_key("terminal") {
                exact_terminal_source_samples += 1;
            } else {
                exact_ordinary_source_samples += 1;
            }
            let conditional_mean_l2_sq = optional_u128(conditional).unwrap();
            let challenge_l2_sq = required_u128(event, "challenge_l2_sq");
            assert_eq!(
                conditional_mean_l2_sq,
                source_l2_sq.checked_mul(challenge_l2_sq).unwrap()
            );
        } else {
            assert_eq!(source, "None");
            assert_eq!(conditional, "None");
        }
        assert!(response_l2_sq > 0);
        assert!(event.fields.contains_key("nonce"));
        assert!(event.fields.contains_key("attempts"));
        assert!(event.fields.contains_key("ring_dimension"));
        assert!(event.fields.contains_key("response_coeffs"));
    }
    assert!(
        exact_ordinary_source_samples > 0,
        "ordinary response-model sample did not retain exact source energy"
    );
    assert!(
        exact_terminal_source_samples > 0,
        "terminal response-model sample did not retain exact source energy"
    );
}
