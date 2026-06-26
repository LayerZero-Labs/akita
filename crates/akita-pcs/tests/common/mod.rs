#![allow(dead_code)]

pub(super) use akita_config::proof_optimized::fp128;
pub(super) use akita_config::CommitmentConfig;
pub(super) use akita_field::{CanonicalField, FieldCore};
use akita_prover::compute::{OpeningFoldKernel, OpeningFoldPlan, RootOpeningSource, RootPolyShape};
use akita_prover::CpuBackend;
pub(super) use akita_prover::DensePoly;
pub(super) use akita_prover::OneHotPoly;
pub(super) use akita_prover::{ProverCommitmentGroup, ProverOpeningBatch};
pub(super) use akita_types::LevelParams;
pub(super) use akita_types::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AkitaCommitmentHint,
    BasisMode, BlockOrder, CommitmentGroup, PointVariableSelection, RingCommitment,
    VerifierOpeningBatch,
};
pub(super) use rand::rngs::StdRng;
pub(super) use rand::{Rng, SeedableRng};
use std::sync::Once;

pub(super) type F = fp128::Field;
pub(super) const STACK_SIZE: usize = 256 * 1024 * 1024;

// Bare presets: test-only non-singleton batched opening shapes
// fall through to the offline DP planner on table miss via the default
// `runtime_schedule` fallback.
pub(super) type OneHotCfg = fp128::D64OneHot;
pub(super) const ONEHOT_D: usize = OneHotCfg::D;
// `fp128::D64OneHot` requires K=256 one-hot schedules (chunks span `K/D = 4`
// ring elements), so the committed poly has `2^nv / K` chunks, not one chunk
// per ring element. Must match `OneHotCfg::onehot_chunk_size()`.
pub(super) const ONEHOT_K: usize = 256;

pub(super) type DenseCfg = fp128::D128Full;
pub(super) const DENSE_D: usize = DenseCfg::D;

static INIT_RAYON: Once = Once::new();

pub(super) fn init_rayon_pool() {
    INIT_RAYON.call_once(|| {
        #[cfg(feature = "parallel")]
        rayon::ThreadPoolBuilder::new()
            .stack_size(STACK_SIZE)
            .build_global()
            .ok();
    });
}

pub(super) fn random_point(nv: usize, seed: u64) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

pub(super) fn run_on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(f)
        .expect("failed to spawn thread")
        .join()
        .expect("test thread panicked");
}

pub(super) fn prove_input<'a, FF: FieldCore + Clone, P, CommitF: FieldCore, const D: usize>(
    point: &'a [FF],
    polynomials: &'a [&'a P],
    commitment: &'a RingCommitment<CommitF, D>,
    hint: AkitaCommitmentHint<CommitF, D>,
) -> ProverOpeningBatch<'a, FF, P, CommitF, D> {
    ProverOpeningBatch {
        point: point.into(),
        groups: vec![ProverCommitmentGroup {
            point_vars: PointVariableSelection::prefix(point.len(), point.len())
                .expect("full-point prover group"),
            polynomials,
            commitment: (commitment.clone(), hint),
        }],
        carried_flat_commitment: None,
    }
}

pub(super) fn verify_input<'a, FF: FieldCore, C>(
    point: &'a [FF],
    openings: &'a [FF],
    commitment: &'a C,
) -> VerifierOpeningBatch<'static, FF, &'a C> {
    VerifierOpeningBatch::from_groups(
        point.to_vec(),
        vec![CommitmentGroup {
            claims: openings.to_vec(),
            commitment,
        }],
    )
    .expect("valid verifier input")
}

pub(super) fn opening_from_poly<'a, const D: usize, P>(
    poly: &'a P,
    point: &[F],
    layout: &LevelParams,
) -> F
where
    P: RootOpeningSource<F, D> + RootPolyShape<F, D>,
    CpuBackend: OpeningFoldKernel<P::OpeningView<'a>, F, D>,
{
    opening_from_poly_with_basis::<D, P>(poly, point, layout, BasisMode::Lagrange)
}

pub(super) fn opening_from_poly_with_basis<'a, const D: usize, P>(
    poly: &'a P,
    point: &[F],
    layout: &LevelParams,
    basis_mode: BasisMode,
) -> F
where
    P: RootOpeningSource<F, D> + RootPolyShape<F, D>,
    CpuBackend: OpeningFoldKernel<P::OpeningView<'a>, F, D>,
{
    let alpha_bits = D.trailing_zeros() as usize;
    let target_num_vars = alpha_bits + layout.m_vars + layout.r_vars;
    assert!(
        point.len() <= target_num_vars,
        "opening point length {} exceeds target root arity {}",
        point.len(),
        target_num_vars
    );
    let mut padded_point = point.to_vec();
    padded_point.resize(target_num_vars, F::zero());

    let inner_point = &padded_point[..alpha_bits];
    let reduced_point = &padded_point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        basis_mode,
        BlockOrder::RowMajor,
    )
    .expect("opening point shape should match layout");

    let opening = OpeningFoldKernel::<P::OpeningView<'a>, F, D>::evaluate_and_fold(
        &CpuBackend,
        None,
        poly.opening_view().expect("opening view"),
        OpeningFoldPlan::Base {
            eval_outer_scalars: &ring_opening_point.b,
            fold_scalars: &ring_opening_point.a,
            block_len: layout.block_len,
        },
    )
    .expect("evaluate_and_fold");
    let folded_ring = opening.eval;
    let packed_inner = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis_mode)
        .expect("inner opening point should match ring dimension");
    (folded_ring * packed_inner.sigma_m1()).coefficients()[0]
}

pub(super) fn make_onehot_poly(layout: &LevelParams, seed: u64) -> OneHotPoly<F, ONEHOT_D, u8> {
    // `2^nv = (num_blocks · block_len) · D` field elements, grouped into
    // `2^nv / K` one-hot chunks of size `K`.
    let total_field = layout.num_blocks * layout.block_len * ONEHOT_D;
    let total_chunks = total_field / ONEHOT_K;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();
    OneHotPoly::<F, ONEHOT_D, u8>::new(ONEHOT_K, indices).expect("onehot poly")
}

pub(super) fn make_dense_poly(nv: usize, seed: u64) -> DensePoly<F, DENSE_D> {
    let evals = dense_field_evals(nv, seed);
    DensePoly::<F, DENSE_D>::from_field_evals(nv, &evals).expect("dense poly")
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

pub(super) fn dense_field_evals(nv: usize, seed: u64) -> Vec<F> {
    let n = 1usize << nv;
    let mut out = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        let v = splitmix64_next(&mut state);
        out.push(F::from_canonical_u128_reduced(v as u128));
    }
    out
}

#[cfg(feature = "logging-transcript")]
pub(super) fn public_transcript_events(
    events: &[akita_transcript::TranscriptEvent],
) -> Vec<akita_transcript::TranscriptEvent> {
    events
        .iter()
        .filter(|event| !matches!(event, akita_transcript::TranscriptEvent::Wire { .. }))
        .cloned()
        .collect()
}

#[cfg(feature = "logging-transcript")]
pub(super) fn event_label(event: &akita_transcript::TranscriptEvent) -> Option<&[u8]> {
    match event {
        akita_transcript::TranscriptEvent::Absorb { label, .. }
        | akita_transcript::TranscriptEvent::Squeeze { label, .. }
        | akita_transcript::TranscriptEvent::Wire { label, .. } => Some(label),
        akita_transcript::TranscriptEvent::Preamble { .. } => None,
    }
}

#[cfg(feature = "logging-transcript")]
pub(super) fn first_label_index(
    events: &[akita_transcript::TranscriptEvent],
    label: &[u8],
) -> Option<usize> {
    events
        .iter()
        .position(|event| event_label(event).is_some_and(|candidate| candidate == label))
}

#[cfg(feature = "logging-transcript")]
pub(super) fn first_label_index_after(
    events: &[akita_transcript::TranscriptEvent],
    start: usize,
    label: &[u8],
) -> Option<usize> {
    events[start..]
        .iter()
        .position(|event| event_label(event).is_some_and(|candidate| candidate == label))
        .map(|offset| start + offset)
}

#[cfg(feature = "logging-transcript")]
fn is_label_or_extension_limb(candidate: &[u8], base: &[u8]) -> bool {
    candidate == base || akita_transcript::is_ext_limb_label(candidate, base)
}

#[cfg(feature = "logging-transcript")]
pub(super) fn first_label_or_extension_limb_index_after(
    events: &[akita_transcript::TranscriptEvent],
    start: usize,
    label: &[u8],
) -> Option<usize> {
    events[start..]
        .iter()
        .position(|event| {
            event_label(event).is_some_and(|candidate| is_label_or_extension_limb(candidate, label))
        })
        .map(|offset| start + offset)
}

#[cfg(feature = "logging-transcript")]
fn first_logical_label_span_after(
    events: &[akita_transcript::TranscriptEvent],
    start: usize,
    label: &[u8],
) -> Option<(usize, usize)> {
    let span_start = first_label_or_extension_limb_index_after(events, start, label)?;
    let mut span_end = span_start + 1;
    while span_end < events.len()
        && event_label(&events[span_end])
            .is_some_and(|candidate| is_label_or_extension_limb(candidate, label))
    {
        span_end += 1;
    }
    Some((span_start, span_end))
}

#[cfg(feature = "logging-transcript")]
fn assert_no_logical_label(
    events: &[akita_transcript::TranscriptEvent],
    range: std::ops::Range<usize>,
    label: &[u8],
    message: &str,
) {
    assert!(
        events[range].iter().all(|event| {
            event_label(event).is_none_or(|candidate| !is_label_or_extension_limb(candidate, label))
        }),
        "{message}"
    );
}

#[cfg(feature = "logging-transcript")]
pub(super) fn assert_terminal_event_order_if_present(
    events: &[akita_transcript::TranscriptEvent],
) -> Option<usize> {
    use akita_transcript::labels;

    let e_hat = first_label_index(events, labels::ABSORB_TERMINAL_E_HAT)?;
    let (sparse_seed, sparse_seed_end) =
        first_logical_label_span_after(events, e_hat, labels::CHALLENGE_SPARSE_CHALLENGE)
            .expect("terminal transcript must squeeze sparse seed");
    let remainder =
        first_label_index_after(events, sparse_seed_end, labels::ABSORB_TERMINAL_W_REMAINDER)
            .expect("terminal transcript must absorb final-witness remainder");
    let (alpha, alpha_end) =
        first_logical_label_span_after(events, remainder, labels::CHALLENGE_RING_SWITCH)
            .expect("terminal transcript must squeeze ring-switch alpha");
    let (tau1, tau1_end) =
        first_logical_label_span_after(events, alpha_end, labels::CHALLENGE_TAU1)
            .expect("terminal transcript must squeeze tau1");
    let (stage2_round, _) =
        first_logical_label_span_after(events, tau1_end, labels::CHALLENGE_SUMCHECK_ROUND)
            .expect("terminal transcript must squeeze stage-2 sumcheck after tau1");

    for (range, label, message) in [
        (
            e_hat + 1..remainder,
            labels::CHALLENGE_RING_SWITCH,
            "terminal alpha must not precede witness remainder",
        ),
        (
            e_hat + 1..remainder,
            labels::CHALLENGE_TAU1,
            "terminal tau1 must not precede alpha",
        ),
        (
            e_hat + 1..remainder,
            labels::CHALLENGE_SUMCHECK_ROUND,
            "terminal stage-2 sumcheck must not precede tau1",
        ),
        (
            remainder + 1..alpha,
            labels::CHALLENGE_TAU1,
            "terminal tau1 must not precede alpha",
        ),
        (
            remainder + 1..alpha,
            labels::CHALLENGE_SUMCHECK_ROUND,
            "terminal stage-2 sumcheck must not precede tau1",
        ),
        (
            alpha_end..tau1,
            labels::CHALLENGE_RING_SWITCH,
            "terminal alpha limbs must be contiguous before tau1",
        ),
        (
            alpha_end..tau1,
            labels::CHALLENGE_SUMCHECK_ROUND,
            "terminal stage-2 sumcheck must not precede tau1",
        ),
        (
            alpha_end..events.len(),
            labels::CHALLENGE_RING_SWITCH,
            "terminal alpha limbs must be contiguous before tau1",
        ),
        (
            tau1_end..events.len(),
            labels::CHALLENGE_TAU1,
            "terminal tau1 limbs must be contiguous before stage-2 sumcheck",
        ),
        (
            tau1_end..stage2_round,
            labels::CHALLENGE_SUMCHECK_ROUND,
            "terminal stage-2 sumcheck must not precede tau1",
        ),
    ] {
        assert_no_logical_label(events, range, label, message);
    }

    assert!(e_hat < sparse_seed, "e_hat must precede sparse seed");
    assert!(
        sparse_seed < remainder,
        "sparse seed must precede witness remainder"
    );
    assert!(remainder < alpha, "remainder must precede alpha");
    assert!(alpha < tau1, "alpha must precede tau1");
    assert!(
        tau1 < stage2_round,
        "tau1 must precede terminal stage-2 sumcheck"
    );
    assert!(
        events[e_hat..]
            .iter()
            .all(|event| event_label(event).is_none_or(|candidate| {
                !is_label_or_extension_limb(candidate, labels::CHALLENGE_TAU0)
            })),
        "terminal transcript window must not squeeze tau0"
    );
    Some(e_hat)
}
