use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{span, Subscriber};
use tracing_subscriber::{layer::Context, registry::LookupSpan, Layer};

#[derive(Clone, Default)]
pub(super) struct Timings(Arc<Mutex<BTreeMap<&'static str, (u64, Duration)>>>);

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Timings {
    fn on_new_span(&self, _: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(Instant::now());
        }
    }
    fn on_close(&self, id: span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(&id) {
            if let Some(start) = span.extensions().get::<Instant>() {
                if let Ok(mut totals) = self.0.lock() {
                    let entry = totals.entry(span.metadata().name()).or_default();
                    entry.0 += 1;
                    entry.1 += start.elapsed();
                }
            }
        }
    }
}

impl Timings {
    pub(super) fn print(&self, phase: &str) {
        if let Ok(totals) = self.0.lock() {
            for (name, (calls, elapsed)) in totals.iter() {
                println!(
                    "span phase={phase} name={name} calls={calls} inclusive_wall_ns={}",
                    elapsed.as_nanos()
                );
            }
        }
    }
}

pub(super) fn peak_rss() -> Result<u64, Box<dyn std::error::Error>> {
    #[cfg(unix)]
    {
        // SAFETY: getrusage initializes usage on success; failure is propagated.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let raw = u64::try_from(usage.ru_maxrss)?;
        Ok(if cfg!(target_os = "macos") {
            raw
        } else {
            raw.checked_mul(1024).ok_or("RSS overflow")?
        })
    }
    #[cfg(not(unix))]
    Err("peak RSS requires a Unix host".into())
}

pub(super) fn proof(
    proof: &[u8],
    log_carriers: usize,
    word: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use akita_error::checked;

    let digest = akita_params::descriptor_bytes::digest_descriptor_bytes(proof);
    println!(
        "proof bytes={} digest_algorithm=blake2b256 digest={}",
        proof.len(),
        digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
    let mut totals = BTreeMap::<(u32, u32, u32), (u64, usize)>::new();
    let mut accounted = 0usize;
    let ranges = akita_transcript::thread_proof_ranges();
    let outgoing_site = akita_types::FoldSite::NextWitnessPayload { level: 0 }.id()?;
    let outgoing = ranges
        .iter()
        .find(|range| range.context.site_id == outgoing_site.to_bytes())
        .ok_or("missing outgoing payload range")?;
    let projection_start = checked::sum([outgoing.start, outgoing.len]).ok_or("range overflow")?;
    let mut coverage = Vec::new();
    // The binary frontend and integer projections use raw channel exchanges,
    // so the generic field-message range logger does not classify them.
    let frontend_round_bytes = checked::product([42, log_carriers]).ok_or("frontend overflow")?;
    let terminal_start = checked::sum([2048, frontend_round_bytes]).ok_or("frontend overflow")?;
    for (name, start, bytes, atoms) in [
        (
            "field_switch_partials",
            0,
            if word { 2048 } else { 0 },
            if word { 128 } else { 0 },
        ),
        (
            "field_switch_rounds",
            2048,
            if word { frontend_round_bytes } else { 0 },
            if word {
                checked::product([2, log_carriers]).ok_or("atom overflow")?
            } else {
                0
            },
        ),
        (
            "field_switch_terminal",
            terminal_start,
            if word { 21 } else { 0 },
            usize::from(word),
        ),
        ("integer_projections", projection_start, 84, 7),
    ] {
        if bytes != 0 {
            let end = checked::sum([start, bytes]).ok_or("range overflow")?;
            coverage.push((start, end));
            accounted = checked::sum([accounted, bytes]).ok_or("byte overflow")?;
            println!("proof_stage level=0 name={name} start={start} bytes={bytes} atoms={atoms}");
        }
    }
    for range in ranges {
        let site = akita_transcript::ProtocolSiteId::from_bytes(range.context.site_id);
        let entry = totals
            .entry((site.level, site.stage, site.family))
            .or_default();
        entry.0 = entry
            .0
            .checked_add(range.context.atom_count)
            .ok_or("atom overflow")?;
        entry.1 = entry.1.checked_add(range.len).ok_or("byte overflow")?;
        accounted = accounted.checked_add(range.len).ok_or("byte overflow")?;
        coverage.push((
            range.start,
            checked::sum([range.start, range.len]).ok_or("range overflow")?,
        ));
        println!(
            "proof_range level={} stage={} family={} kind={} start={} bytes={} atoms={}",
            site.level,
            site.stage,
            site.family,
            range.context.kind,
            range.start,
            range.len,
            range.context.atom_count
        );
    }
    for ((level, stage, family), (atoms, bytes)) in totals {
        println!(
            "proof_stage level={level} stage={stage} family={family} bytes={bytes} atoms={atoms}"
        );
    }
    coverage.sort_unstable();
    let mut end = 0;
    for (start, next) in coverage {
        if start != end || next > proof.len() {
            return Err("proof accounting has a gap, overlap or out-of-range interval".into());
        }
        end = next;
    }
    if end != proof.len() {
        return Err("proof accounting does not cover the entire proof".into());
    }
    println!(
        "proof_accounting classified_bytes={accounted} unclassified_bytes={}",
        proof
            .len()
            .checked_sub(accounted)
            .ok_or("overlapping proof ranges")?
    );
    Ok(())
}
