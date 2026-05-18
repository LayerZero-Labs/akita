#![allow(missing_docs)]
#![cfg(feature = "planner")]

//! End-to-end transcript trace for a small dense fp32 proof.
//!
//! Wraps `Blake2bTranscript` with a logger that records every label, op
//! (absorb vs squeeze), and byte length, then runs `batched_prove` followed
//! by `batched_verify` against the same logger. The two recordings are
//! printed side by side so the message schedule (and what gets absorbed /
//! squeezed at each level, including the terminal direct tail) is visible.
//!
//! Run with:
//!     cargo test -p akita-pcs --test transcript_trace -- --nocapture

use akita_config::proof_optimized::fp32;
use akita_config::CommitmentConfig;
use akita_field::{CanonicalBytes, CanonicalField, ExtField, FieldCore, TranscriptChallenge};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{CommitmentProver, CommittedPolynomials, DensePoly};
use akita_serialization::AkitaSerialize;
use akita_transcript::{Blake2bTranscript, Transcript};
use akita_types::{
    lagrange_weights, AkitaBatchedProof, AkitaScheduleLookupKey, BasisMode, ScheduleProvider,
};
use akita_verifier::{CommitmentVerifier, CommittedOpenings};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Mutex;

#[derive(Debug, Clone)]
enum Op {
    AppendBytes { label: String, bytes: usize },
    AppendField { label: String, bytes: usize },
    AppendSerde { label: String, bytes: usize },
    ChallengeScalar { label: String },
    ChallengeBytes { label: String, bytes: usize },
}

static EVENTS: Mutex<Vec<Op>> = Mutex::new(Vec::new());

/// Serializes the trace tests within a single `cargo test` process so the
/// global `EVENTS` buffer is not interleaved across tests when the harness
/// runs them on parallel threads. (`cargo nextest` already isolates tests
/// in separate processes, so this guard is a no-op there.)
static TRACE_SERIAL: Mutex<()> = Mutex::new(());

/// Map the byte tag (and trailing `\xff <limb_u64_le> b"ext"` suffix used by
/// extension-challenge sampling) into a stable, human-readable label.
fn canonicalize_label(raw: &[u8]) -> String {
    let known: &[(&[u8], &str)] = &[
        (b"ak/p", "DOMAIN"),
        (b"ak/a/cm", "ABSORB_COMMITMENT"),
        (b"ak/a/ec", "ABSORB_EVALUATION_CLAIMS"),
        (b"ak/a/bs", "ABSORB_BATCH_SHAPE"),
        (b"ak/c/lr", "CHALLENGE_LINEAR_RELATION"),
        (b"ak/a/rs", "ABSORB_RING_SWITCH_MESSAGE"),
        (b"ak/c/rs", "CHALLENGE_RING_SWITCH"),
        (b"ak/a/sp", "ABSORB_SPARSE_CHALLENGE"),
        (b"ak/c/sp", "CHALLENGE_SPARSE_CHALLENGE"),
        (b"ak/a/sc", "ABSORB_SUMCHECK_CLAIM"),
        (b"ak/a/scr", "ABSORB_SUMCHECK_ROUND"),
        (b"ak/c/scr", "CHALLENGE_SUMCHECK_ROUND"),
        (b"ak/a/scs", "ABSORB_SUMCHECK_S_CLAIM"),
        (b"ak/a/sci", "ABSORB_SUMCHECK_INTERSTAGE_CLAIM"),
        (b"ak/c/scb", "CHALLENGE_SUMCHECK_BATCH"),
        (b"ak/c/scib", "CHALLENGE_SUMCHECK_INTERSTAGE_BATCH"),
        (b"ak/a/st", "ABSORB_STOP_CONDITION"),
        (b"ak/c/st", "CHALLENGE_STOP_CONDITION"),
        (b"ak/a/v", "ABSORB_PROVER_V"),
        (b"ak/c/s1f", "CHALLENGE_STAGE1_FOLD"),
        (b"ak/a/eof", "ABSORB_EVAL_OPENINGS_FIELD"),
        (b"ak/c/eb", "CHALLENGE_EVAL_BATCH"),
        (b"ak/a/w", "ABSORB_SUMCHECK_W"),
        (b"ak/c/t0", "CHALLENGE_TAU0"),
        (b"ak/c/t1", "CHALLENGE_TAU1"),
    ];
    // The ext-limb label format is `<base>\xff <u64 le limb> b"ext"`.
    let (base_bytes, ext_suffix) = if let Some(idx) = raw.iter().position(|&b| b == 0xff) {
        let tail = &raw[idx + 1..];
        if tail.len() == 8 + 3 && &tail[8..] == b"ext" {
            let mut limb = [0u8; 8];
            limb.copy_from_slice(&tail[..8]);
            (&raw[..idx], Some(u64::from_le_bytes(limb) as usize))
        } else {
            (raw, None)
        }
    } else {
        (raw, None)
    };
    let base = known
        .iter()
        .find(|(b, _)| *b == base_bytes)
        .map(|(_, name)| (*name).to_string())
        .unwrap_or_else(|| String::from_utf8_lossy(base_bytes).into_owned());
    if let Some(limb) = ext_suffix {
        format!("{base}[ext{limb}]")
    } else {
        base
    }
}

fn drain_events() -> Vec<Op> {
    let mut g = EVENTS.lock().unwrap();
    std::mem::take(&mut *g)
}

#[derive(Clone)]
struct LoggingTranscript<F>
where
    F: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge + 'static,
{
    inner: Blake2bTranscript<F>,
}

impl<F> Transcript<F> for LoggingTranscript<F>
where
    F: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge + 'static,
{
    fn new(domain_label: &[u8]) -> Self {
        Self {
            inner: Blake2bTranscript::<F>::new(domain_label),
        }
    }

    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        EVENTS.lock().unwrap().push(Op::AppendBytes {
            label: canonicalize_label(label),
            bytes: bytes.len(),
        });
        self.inner.append_bytes(label, bytes);
    }

    fn append_field(&mut self, label: &[u8], x: &F) {
        EVENTS.lock().unwrap().push(Op::AppendField {
            label: canonicalize_label(label),
            bytes: <F as akita_field::FixedByteSize>::NUM_BYTES,
        });
        self.inner.append_field(label, x);
    }

    fn append_serde<S: AkitaSerialize>(&mut self, label: &[u8], s: &S) {
        let bytes = s.compressed_size();
        EVENTS.lock().unwrap().push(Op::AppendSerde {
            label: canonicalize_label(label),
            bytes,
        });
        self.inner.append_serde(label, s);
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> F {
        let v = self.inner.challenge_scalar(label);
        EVENTS.lock().unwrap().push(Op::ChallengeScalar {
            label: canonicalize_label(label),
        });
        v
    }

    fn challenge_bytes(&mut self, label: &[u8], len: usize) -> Vec<u8> {
        let out = self.inner.challenge_bytes(label, len);
        EVENTS.lock().unwrap().push(Op::ChallengeBytes {
            label: canonicalize_label(label),
            bytes: len,
        });
        out
    }
}

fn random_claim_point<FField, E>(nv: usize) -> Vec<E>
where
    FField: CanonicalField,
    E: ExtField<FField>,
{
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| {
            let limbs = (0..E::EXT_DEGREE)
                .map(|_| FField::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect::<Vec<_>>();
            E::from_base_slice(&limbs)
        })
        .collect()
}

fn dense_lagrange_opening_from_evals<FField, E>(evals: &[FField], point: &[E]) -> E
where
    FField: FieldCore,
    E: ExtField<FField>,
{
    let weights = lagrange_weights(point);
    evals
        .iter()
        .zip(weights.iter())
        .fold(E::zero(), |acc, (&coeff, &weight)| {
            acc + weight * E::lift_base(coeff)
        })
}

/// Coalesce consecutive `<X>[ext0..N-1]` ops on the same base into a single
/// `<X>[ext]` op of total size sum. This makes the trace readable by hiding
/// the per-limb pump for `EXT_DEGREE`-coordinate sampling/absorption.
fn coalesce_ext_limbs(ops: &[Op]) -> Vec<Op> {
    let mut out: Vec<Op> = Vec::with_capacity(ops.len());
    let mut i = 0;
    while i < ops.len() {
        let (base, ext_idx) = strip_ext_suffix(label_of(&ops[i]));
        if let Some(_idx) = ext_idx {
            let mut j = i;
            let mut accum_bytes = 0usize;
            let kind = op_kind(&ops[i]);
            while j < ops.len() {
                let (b, e) = strip_ext_suffix(label_of(&ops[j]));
                if b != base || e.is_none() || op_kind(&ops[j]) != kind {
                    break;
                }
                accum_bytes += bytes_of(&ops[j]);
                j += 1;
            }
            let combined_label = format!("{base}[ext]");
            let combined = match &ops[i] {
                Op::AppendBytes { .. } => Op::AppendBytes {
                    label: combined_label,
                    bytes: accum_bytes,
                },
                Op::AppendField { .. } => Op::AppendField {
                    label: combined_label,
                    bytes: accum_bytes,
                },
                Op::AppendSerde { .. } => Op::AppendSerde {
                    label: combined_label,
                    bytes: accum_bytes,
                },
                Op::ChallengeScalar { .. } => Op::ChallengeScalar {
                    label: combined_label,
                },
                Op::ChallengeBytes { .. } => Op::ChallengeBytes {
                    label: combined_label,
                    bytes: accum_bytes,
                },
            };
            out.push(combined);
            i = j;
        } else {
            out.push(ops[i].clone());
            i += 1;
        }
    }
    out
}

fn label_of(op: &Op) -> &str {
    match op {
        Op::AppendBytes { label, .. }
        | Op::AppendField { label, .. }
        | Op::AppendSerde { label, .. }
        | Op::ChallengeScalar { label }
        | Op::ChallengeBytes { label, .. } => label.as_str(),
    }
}
fn bytes_of(op: &Op) -> usize {
    match op {
        Op::AppendBytes { bytes, .. }
        | Op::AppendField { bytes, .. }
        | Op::AppendSerde { bytes, .. }
        | Op::ChallengeBytes { bytes, .. } => *bytes,
        Op::ChallengeScalar { .. } => 0,
    }
}
fn op_kind(op: &Op) -> &'static str {
    match op {
        Op::AppendBytes { .. } => "bytes",
        Op::AppendField { .. } => "field",
        Op::AppendSerde { .. } => "serde",
        Op::ChallengeScalar { .. } => "scal",
        Op::ChallengeBytes { .. } => "bytes",
    }
}
fn strip_ext_suffix(label: &str) -> (&str, Option<usize>) {
    if let Some(start) = label.find("[ext") {
        let end = label.rfind(']');
        if let Some(end) = end {
            if end == label.len() - 1 {
                let inside = &label[start + 4..end];
                if let Ok(idx) = inside.parse::<usize>() {
                    return (&label[..start], Some(idx));
                }
            }
        }
    }
    (label, None)
}

/// Walk a single level's ops chronologically and print one line per op,
/// except consecutive `ABSORB_SUMCHECK_ROUND[ext] + CHALLENGE_SUMCHECK_ROUND[ext]`
/// pairs (one full sumcheck round) get folded into a single line of the form
/// `SUMCHECK N rounds, B B/round` so the macro-structure is visible.
fn narrate_chronologically(level: &[&Op]) {
    let mut i = 0;
    while i < level.len() {
        // Sumcheck-round pair detection: a run of ABSORB_SUMCHECK_ROUND[ext]
        // followed by CHALLENGE_SUMCHECK_ROUND[ext], repeated.
        let is_round_absorb = |op: &Op| matches!(op, Op::AppendSerde { label, .. } if label == "ABSORB_SUMCHECK_ROUND");
        let is_round_squeeze = |op: &Op| matches!(op, Op::ChallengeScalar { label } if label == "CHALLENGE_SUMCHECK_ROUND[ext]");
        if i + 1 < level.len() && is_round_absorb(level[i]) && is_round_squeeze(level[i + 1]) {
            let mut j = i;
            let mut rounds = 0usize;
            let mut total_round_bytes = 0usize;
            let mut per_round_bytes: Option<usize> = None;
            while j + 1 < level.len() && is_round_absorb(level[j]) && is_round_squeeze(level[j + 1])
            {
                let b = bytes_of(level[j]);
                total_round_bytes += b;
                per_round_bytes.get_or_insert(b);
                rounds += 1;
                j += 2;
            }
            let per = per_round_bytes.unwrap_or(0);
            println!(
                "  SUMCHECK round-loop                            x{rounds:<3} {per} B/round (= {total_round_bytes} B absorbed) + {rounds} squeezes"
            );
            i = j;
            continue;
        }
        // Default: one line per op.
        let here_label = label_of(level[i]);
        let b = bytes_of(level[i]);
        let action = match level[i] {
            Op::AppendBytes { .. } => "ABSORB bytes ",
            Op::AppendField { .. } => "ABSORB field ",
            Op::AppendSerde { .. } => "ABSORB serde ",
            Op::ChallengeScalar { .. } => "SQUEEZE scal ",
            Op::ChallengeBytes { .. } => "SQUEEZE bytes",
        };
        let bytes_part = if b == 0 {
            String::new()
        } else {
            format!(" bytes={b}")
        };
        println!("  {action} [{here_label:<44}]{bytes_part}");
        i += 1;
    }
}

fn summarize_with_separators(label: &str, ops: &[Op]) {
    let ops = coalesce_ext_limbs(ops);
    let ops = ops.as_slice();

    // Insert a visual break each time a new `ABSORB_COMMITMENT` is seen,
    // since that label always opens a fresh level in `verify_one_level`.
    let mut by_level: Vec<Vec<&Op>> = vec![Vec::new()];
    for op in ops {
        let opens_level = matches!(
            op,
            Op::AppendSerde { label, .. } if label == "ABSORB_COMMITMENT"
        );
        if opens_level && !by_level.last().unwrap().is_empty() {
            by_level.push(Vec::new());
        }
        by_level.last_mut().unwrap().push(op);
    }

    println!("\n========== {label} TRANSCRIPT TRACE ==========");
    for (idx, level) in by_level.iter().enumerate() {
        let absorb_bytes_here: usize = level
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    Op::AppendBytes { .. } | Op::AppendField { .. } | Op::AppendSerde { .. }
                )
            })
            .map(|op| bytes_of(op))
            .sum();
        let squeeze_count_here: usize = level
            .iter()
            .filter(|op| matches!(op, Op::ChallengeScalar { .. } | Op::ChallengeBytes { .. }))
            .count();
        println!(
            "\n--- level block {idx} (absorb={absorb_bytes_here}B, squeeze_calls={squeeze_count_here}) ---"
        );
        let mut i = 0;
        let level_slice = level.as_slice();
        while i < level_slice.len() {
            let here_kind = op_kind(level_slice[i]);
            let here_label = label_of(level_slice[i]).to_string();
            let mut j = i;
            let mut total_bytes = 0usize;
            while j < level_slice.len()
                && op_kind(level_slice[j]) == here_kind
                && label_of(level_slice[j]) == here_label
            {
                total_bytes += bytes_of(level_slice[j]);
                j += 1;
            }
            let n = j - i;
            let bytes_part = if total_bytes == 0 {
                String::new()
            } else {
                format!(" bytes={total_bytes}")
            };
            let count_part = if n == 1 {
                String::new()
            } else {
                format!(" x{n}")
            };
            let action = match level_slice[i] {
                Op::AppendBytes { .. } => "ABSORB bytes ",
                Op::AppendField { .. } => "ABSORB field ",
                Op::AppendSerde { .. } => "ABSORB serde ",
                Op::ChallengeScalar { .. } => "SQUEEZE scal ",
                Op::ChallengeBytes { .. } => "SQUEEZE bytes",
            };
            println!("  {action} [{here_label:<40}]{count_part}{bytes_part}");
            i = j;
        }
    }

    // Chronological flow for level block 1 (the first verify_one_level call,
    // = the root fold). One line per op, with sumcheck-round pairs grouped
    // as `SUMCHECK round-loop xN <B> B/round`.
    if let Some(level1) = by_level.get(1) {
        println!(
            "\n--- {label} chronological flow @ level block 1 (root fold, {} ops) ---",
            level1.len()
        );
        narrate_chronologically(level1);
    }
    if let Some(last_idx) = by_level.len().checked_sub(1) {
        if last_idx >= 2 {
            let last = &by_level[last_idx];
            println!(
                "\n--- {label} chronological flow @ level block {last_idx} (terminal `is_last=true` fold, {} ops) ---",
                last.len()
            );
            narrate_chronologically(last);
        }
    }

    // Per-label totals.
    use std::collections::BTreeMap;
    let mut absorb_by_label: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut squeeze_by_label: BTreeMap<String, usize> = BTreeMap::new();
    for op in ops {
        match op {
            Op::AppendBytes { label, bytes }
            | Op::AppendField { label, bytes }
            | Op::AppendSerde { label, bytes } => {
                let entry = absorb_by_label.entry(label.clone()).or_default();
                entry.0 += 1;
                entry.1 += *bytes;
            }
            Op::ChallengeScalar { label } | Op::ChallengeBytes { label, .. } => {
                *squeeze_by_label.entry(label.clone()).or_default() += 1;
            }
        }
    }

    println!("\n--- {label} ABSORB totals (count, bytes) by label ---");
    for (lbl, (n, b)) in &absorb_by_label {
        println!("  {lbl:<40} count={n:<4} bytes={b}");
    }
    println!("\n--- {label} SQUEEZE totals (count) by label ---");
    for (lbl, n) in &squeeze_by_label {
        println!("  {lbl:<40} count={n}");
    }
    let total_absorb_bytes: usize = absorb_by_label.values().map(|(_, b)| *b).sum();
    let total_squeezes: usize = squeeze_by_label.values().sum();
    println!(
        "\n[{label}] total absorb bytes = {total_absorb_bytes}, total squeezes = {total_squeezes}"
    );
}

fn trace_dense_fp32_d64_at_nv(nv: usize) {
    let _serial_guard = TRACE_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    drain_events();
    type FSmall = fp32::Field;
    type Cfg = fp32::D64Full;
    const D: usize = Cfg::D;

    let width = <Cfg as CommitmentConfig>::CLAIM_EXT_DEGREE;
    let dense_schedule_key = AkitaScheduleLookupKey::new_with_groups(nv, 1, 1, width, width);
    match Cfg::schedule_plan(dense_schedule_key).expect("schedule plan lookup") {
        Some(plan) => println!(
            "\nfp32::D64Full nv={nv} dense (num_w=num_z={width}): num_fold_levels={}, root_is_fold={}",
            plan.num_fold_levels(),
            plan
                .steps
                .first()
                .map(|s| matches!(s, akita_types::AkitaPlannedStep::Fold(_)))
                .unwrap_or(false),
        ),
        None => println!(
            "\nfp32::D64Full nv={nv} dense (num_w=num_z={width}): NO PLAN in table"
        ),
    }
    if let Some(plan) =
        Cfg::schedule_plan(AkitaScheduleLookupKey::singleton(nv)).expect("schedule plan lookup")
    {
        println!(
            "fp32::D64Full nv={nv} singleton: num_fold_levels={}, root_is_fold={}",
            plan.num_fold_levels(),
            plan.steps
                .first()
                .map(|s| matches!(s, akita_types::AkitaPlannedStep::Fold(_)))
                .unwrap_or(false),
        );
    } else {
        println!("fp32::D64Full nv={nv} singleton: NO PLAN");
    }

    // Build a dense polynomial and its honest opening at a random point.
    let mut rng = StdRng::seed_from_u64(0xa5a5_a5a5_a5a5_a5a5);
    let evals: Vec<FSmall> = (0..1usize << nv)
        .map(|_| FSmall::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    let poly = DensePoly::<FSmall, D>::from_field_evals(nv, &evals).unwrap();
    let pt = random_claim_point::<FSmall, <Cfg as CommitmentConfig>::ClaimField>(nv);
    let expected_opening = dense_lagrange_opening_from_evals::<
        FSmall,
        <Cfg as CommitmentConfig>::ClaimField,
    >(&evals, &pt);

    let t0 = std::time::Instant::now();
    let setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FSmall, D>>::setup_prover(nv, 1, 1);
    println!("setup_prover: {:.3}s", t0.elapsed().as_secs_f64());
    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FSmall, D>>::setup_verifier(&setup);
    let t0 = std::time::Instant::now();
    let (commitment, hint) =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FSmall, D>>::commit(
            std::slice::from_ref(&poly),
            &setup,
        )
        .unwrap();
    println!("commit: {:.3}s", t0.elapsed().as_secs_f64());

    // Drain any pre-test garbage events.
    let _ = drain_events();

    // ---- PROVER ----
    let poly_refs: [&DensePoly<FSmall, D>; 1] = [&poly];
    let commitments = [commitment];
    let hints = vec![hint];
    let mut prover_t = LoggingTranscript::<FSmall>::new(b"trace");
    let t1 = std::time::Instant::now();
    let proof: AkitaBatchedProof<FSmall, <Cfg as CommitmentConfig>::ChallengeField> =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<FSmall, D>>::batched_prove(
            &setup,
            vec![(
                &pt[..],
                vec![CommittedPolynomials {
                    polynomials: &poly_refs[..],
                    commitment: &commitments[0],
                    hint: hints.into_iter().next().unwrap(),
                }],
            )],
            &mut prover_t,
            BasisMode::Lagrange,
        )
        .unwrap();
    let prover_events = drain_events();
    println!("prove: {:.3}s", t1.elapsed().as_secs_f64());
    println!(
        "proof.size={} bytes, root_is_fold={}, num_fold_levels(non-root)={}",
        proof.size(),
        proof.root.as_fold().is_some(),
        proof.num_fold_levels()
    );
    use akita_serialization::AkitaSerialize as _;
    if let Some(root_fold) = proof.root.as_fold() {
        println!(
            "  root-level proof: stage1+stage2+...+next_w_commitment serialized = {} bytes",
            root_fold.compressed_size()
        );
    }
    for (i, step) in proof.fold_levels().enumerate() {
        println!(
            "  recursive level {i}: serialized = {} bytes (stage1+stage2 sumcheck + next_w_commit)",
            step.compressed_size()
        );
    }
    for step in proof.steps.iter() {
        if let Some(terminal) = step.as_terminal() {
            println!(
                "  terminal step: serialized = {} bytes (stage2 sumcheck + final_witness)",
                terminal.compressed_size()
            );
            println!(
                "    of which final_witness: {} bytes (packed_digits / raw final_w)",
                terminal.final_witness.compressed_size()
            );
        }
    }

    // ---- VERIFIER ----
    let mut verifier_t = LoggingTranscript::<FSmall>::new(b"trace");
    let openings = [expected_opening];
    let result = <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<FSmall, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_t,
        vec![(
            &pt[..],
            vec![CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            }],
        )],
        BasisMode::Lagrange,
    );
    let verifier_events = drain_events();
    assert!(result.is_ok(), "verify failed: {:?}", result.err());

    summarize_with_separators("PROVER", &prover_events);
    summarize_with_separators("VERIFIER", &verifier_events);

    // Cross-check that prover and verifier observe the same Fiat-Shamir
    // sequence: same labels, same op types in the same order, equal absorb
    // byte counts. (The wire trace serialized by the prover is exactly what
    // the verifier replays.)
    assert_eq!(
        prover_events.len(),
        verifier_events.len(),
        "prover/verifier event count mismatch"
    );
    for (i, (p, v)) in prover_events.iter().zip(verifier_events.iter()).enumerate() {
        let (p_kind, p_label, p_len) = describe(p);
        let (v_kind, v_label, v_len) = describe(v);
        assert_eq!(
            (p_kind, p_label, p_len),
            (v_kind, v_label, v_len),
            "transcript mismatch at event {i}: prover={p:?} verifier={v:?}"
        );
    }
    println!(
        "\nprover/verifier transcripts agree on {} events.",
        prover_events.len()
    );
}

#[test]
fn trace_dense_fp32_d64_nv14() {
    trace_dense_fp32_d64_at_nv(14);
}

#[test]
fn trace_dense_fp32_d64_nv20() {
    trace_dense_fp32_d64_at_nv(20);
}

fn describe(op: &Op) -> (&'static str, &str, usize) {
    match op {
        Op::AppendBytes { label, bytes } => ("absorb_bytes", label.as_str(), *bytes),
        Op::AppendField { label, bytes } => ("absorb_field", label.as_str(), *bytes),
        Op::AppendSerde { label, bytes } => ("absorb_serde", label.as_str(), *bytes),
        Op::ChallengeScalar { label } => ("squeeze_scal", label.as_str(), 0),
        Op::ChallengeBytes { label, bytes } => ("squeeze_byte", label.as_str(), *bytes),
    }
}
