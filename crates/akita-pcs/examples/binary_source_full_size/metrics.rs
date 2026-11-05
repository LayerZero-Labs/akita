use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{span, Subscriber};
use tracing_subscriber::{layer::Context, registry::LookupSpan, Layer};

/// Per-name totals: calls, wall, process CPU, the largest live-heap peak and
/// the live heap at that span's entry, and the individual call walls.
#[derive(Default)]
struct Totals {
    calls: u64,
    wall: Duration,
    cpu: Duration,
    heap_peak: usize,
    heap_entry: usize,
    walls: Vec<Duration>,
}

/// Span start state; `heap` is `Some` only for spans on the measurement thread.
struct Start {
    wall: Instant,
    cpu: Duration,
    heap: Option<heap::Scope>,
}

#[derive(Clone, Default)]
pub(super) struct Timings(Arc<Mutex<BTreeMap<&'static str, Totals>>>);

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Timings {
    fn on_new_span(&self, _: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let heap =
                (std::thread::current().name() == Some("binary-full-size")).then(heap::Scope::open);
            span.extensions_mut().insert(Start {
                wall: Instant::now(),
                cpu: process_cpu(),
                heap,
            });
        }
    }
    fn on_close(&self, id: span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(&id) {
            if let Some(start) = span.extensions().get::<Start>() {
                let wall = start.wall.elapsed();
                let cpu = process_cpu().saturating_sub(start.cpu);
                let heap = start.heap.as_ref().map(heap::Scope::close);
                if let Ok(mut totals) = self.0.lock() {
                    let entry = totals.entry(span.metadata().name()).or_default();
                    entry.calls += 1;
                    entry.wall += wall;
                    entry.cpu += cpu;
                    entry.walls.push(wall);
                    if let Some((entry_bytes, peak)) = heap {
                        if peak >= entry.heap_peak {
                            entry.heap_peak = peak;
                            entry.heap_entry = entry_bytes;
                        }
                    }
                }
            }
        }
    }
}

impl Timings {
    pub(super) fn print(&self, phase: &str) {
        if let Ok(totals) = self.0.lock() {
            for (name, totals) in totals.iter() {
                let walls = if totals.walls.len() <= 16 {
                    totals
                        .walls
                        .iter()
                        .map(|wall| wall.as_nanos().to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                } else {
                    "many".into()
                };
                println!(
                    "span phase={phase} name={name} calls={} inclusive_wall_ns={} process_cpu_ns={} heap_entry_bytes={} heap_peak_bytes={} call_wall_ns={walls}",
                    totals.calls,
                    totals.wall.as_nanos(),
                    totals.cpu.as_nanos(),
                    totals.heap_entry,
                    totals.heap_peak,
                );
            }
        }
    }
}

/// User plus system CPU time of the whole process, including worker threads.
fn process_cpu() -> Duration {
    // SAFETY: getrusage initializes usage on success; failure reports zero.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return Duration::ZERO;
    }
    let time = |t: libc::timeval| {
        Duration::new(
            u64::try_from(t.tv_sec).unwrap_or(0),
            u32::try_from(t.tv_usec).unwrap_or(0) * 1000,
        )
    };
    time(usage.ru_utime) + time(usage.ru_stime)
}

/// Live-heap accounting through a counting global allocator.
///
/// Spans opened on the measurement thread nest, so each saves the running
/// peak, restarts it at the current live size, and restores the maximum on
/// close. Worker-thread spans are not tracked.
mod heap {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counting;
    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);

    #[global_allocator]
    static ALLOCATOR: Counting = Counting;

    fn grow(bytes: usize) {
        let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
        PEAK.fetch_max(live, Ordering::Relaxed);
    }

    // SAFETY: every call forwards to the system allocator unchanged.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let pointer = unsafe { System.alloc(layout) };
            if !pointer.is_null() {
                grow(layout.size());
            }
            pointer
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let pointer = unsafe { System.alloc_zeroed(layout) };
            if !pointer.is_null() {
                grow(layout.size());
            }
            pointer
        }
        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            unsafe { System.dealloc(pointer, layout) };
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            let moved = unsafe { System.realloc(pointer, layout, size) };
            if !moved.is_null() {
                LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
                grow(size);
            }
            moved
        }
    }

    pub(super) struct Scope {
        entry: usize,
        saved_peak: usize,
    }

    impl Scope {
        pub(super) fn open() -> Self {
            let entry = LIVE.load(Ordering::Relaxed);
            Self {
                entry,
                saved_peak: PEAK.swap(entry, Ordering::Relaxed),
            }
        }
        /// Returns the entry live size and this span's live-heap peak.
        pub(super) fn close(&self) -> (usize, usize) {
            let peak = PEAK.fetch_max(self.saved_peak, Ordering::Relaxed);
            (self.entry, peak)
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

/// Lowercase hexadecimal of a digest.
pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(super) fn proof(
    proof: &[u8],
    log_carriers: usize,
    word: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use jolt_algebra::checked;

    let digest = akita_params::descriptor_bytes::digest_descriptor_bytes(proof);
    println!(
        "proof bytes={} digest_algorithm=blake2b256 digest={}",
        proof.len(),
        hex(&digest)
    );
    let mut totals = BTreeMap::<(u32, u32, u32, u32), (u64, usize)>::new();
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
            .entry((site.level, site.stage, site.family, site.invocation))
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
            "proof_range level={} site_stage={} family={} invocation={} kind={} start={} bytes={} atoms={}",
            site.level,
            site.stage,
            site.family,
            site.invocation,
            range.context.kind,
            range.start,
            range.len,
            range.context.atom_count
        );
    }
    let sumcheck_work_family = akita_params::GrindingSite::SumcheckRound {
        protocol: akita_params::SumcheckProtocol::Stage1,
        level: 0,
        stage: 0,
        round: 0,
    }
    .site_id(0)
    .family;
    for ((level, stage, family, invocation), (atoms, bytes)) in totals {
        let protocol =
            if family == akita_transcript::SITE_FAMILY_SUMCHECK || family == sumcheck_work_family {
                Some(
                    akita_params::SumcheckProtocol::from_tag(invocation)
                        .ok_or("unknown sumcheck protocol in proof accounting")?,
                )
            } else {
                match family {
                    akita_transcript::SITE_FAMILY_STAGE1 => {
                        Some(akita_params::SumcheckProtocol::Stage1)
                    }
                    akita_transcript::SITE_FAMILY_STAGE2 => {
                        Some(akita_params::SumcheckProtocol::Stage2)
                    }
                    akita_transcript::SITE_FAMILY_STAGE3 => {
                        Some(akita_params::SumcheckProtocol::Stage3)
                    }
                    _ => None,
                }
            };
        println!(
            "proof_stage level={level} protocol={protocol:?} site_stage={stage} family={family} invocation={invocation} bytes={bytes} atoms={atoms}"
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
