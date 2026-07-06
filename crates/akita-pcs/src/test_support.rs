//! Test-only fixtures shared by `akita-pcs`'s integration tests.
//!
//! Everything here used to live in `tests/common/mod.rs`, textually included
//! via `mod common;` by every integration test file that needed it. Because
//! each file under `tests/` compiles as its own separate binary crate, that
//! meant the same ~200 lines of fixture code (and the generic
//! `opening_from_poly` helper's monomorphized instances) were independently
//! recompiled in every consuming binary.
//!
//! Moving it here means it compiles once as part of this crate's own rlib,
//! which every test binary already depends on and which Cargo already
//! caches/reuses across all of them. This module is gated behind the
//! `test-support` feature, switched on only through this crate's own
//! self-referential dev-dependency edge, so it is absent from every shipped
//! artifact.

// Re-exported at the same visibility `tests/common/mod.rs` used to grant its
// (single) consumer, so `use akita_pcs::test_support::*;` is a drop-in
// replacement for the old `use common::*;` glob import.
pub use crate::{
    BasisMode, BlockOrder, OpeningClaims, PointVariableSelection, PolynomialGroupClaims,
};
pub use akita_config::proof_optimized::fp128;
pub use akita_config::CommitmentConfig;
pub use akita_field::{CanonicalField, FieldCore};
pub use akita_prover::{DensePoly, OneHotPoly, ProverOpeningData};
pub use akita_types::LevelParams;
pub use akita_types::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AkitaCommitmentHint,
    Commitment,
};
pub use rand::rngs::StdRng;
pub use rand::{Rng, SeedableRng};

// Used only internally by `opening_from_poly`'s own signature; never part of
// the module's public surface (matching the original file).
use akita_prover::compute::{OpeningFoldKernel, OpeningFoldPlan, RootOpeningSource, RootPolyShape};
use akita_prover::CpuBackend;
use std::sync::Once;

/// The field every fixture in this module is built over.
pub type F = fp128::Field;

/// Stack size used for tests that recurse deeply enough to overflow the
/// default thread stack (see [`run_on_large_stack`]).
pub const STACK_SIZE: usize = 256 * 1024 * 1024;

/// The dominant one-hot commitment config shared by most `akita-pcs`
/// integration tests.
///
/// Bare presets: test-only non-singleton batched opening shapes fall through
/// to the offline DP planner on table miss via the default `runtime_schedule`
/// fallback.
pub type OneHotCfg = fp128::D64OneHot;

/// Ring dimension of [`OneHotCfg`].
pub const ONEHOT_D: usize = OneHotCfg::D;

/// One-hot chunk size of [`OneHotCfg`].
///
/// `fp128::D64OneHot` requires K=256 one-hot schedules (chunks span `K/D = 4`
/// ring elements), so the committed poly has `2^nv / K` chunks, not one chunk
/// per ring element. Must match `OneHotCfg::onehot_chunk_size()`.
pub const ONEHOT_K: usize = 256;

/// The dominant dense commitment config shared by most `akita-pcs`
/// integration tests.
pub type DenseCfg = fp128::D128Full;

/// Ring dimension of [`DenseCfg`].
pub const DENSE_D: usize = DenseCfg::D;

static INIT_RAYON: Once = Once::new();

/// Initializes the global Rayon pool with [`STACK_SIZE`] stacks, once per
/// process.
pub fn init_rayon_pool() {
    INIT_RAYON.call_once(|| {
        #[cfg(feature = "parallel")]
        rayon::ThreadPoolBuilder::new()
            .stack_size(STACK_SIZE)
            .build_global()
            .ok();
    });
}

/// Deterministically samples a random opening point of length `nv` from
/// `seed`.
pub fn random_point(nv: usize, seed: u64) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

/// Runs `f` on a thread with a [`STACK_SIZE`] stack, for tests deep enough to
/// overflow the default stack.
pub fn run_on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(f)
        .expect("failed to spawn thread")
        .join()
        .expect("test thread panicked");
}

/// Builds a single-group, full-point [`ProverOpeningData`] for `polynomials`
/// opened at `point` against `commitment`.
pub fn prove_input<'a, FF: FieldCore + Clone, P, CommitF: FieldCore>(
    point: &'a [FF],
    polynomials: &'a [&'a P],
    commitment: &'a Commitment<CommitF>,
    hint: AkitaCommitmentHint<CommitF>,
) -> ProverOpeningData<'a, FF, P, CommitF> {
    let group = PolynomialGroupClaims::new(
        PointVariableSelection::prefix(point.len(), point.len()).expect("full-point prover group"),
        vec![FF::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("valid prover claims group");
    let opening_claims =
        OpeningClaims::from_groups(point.to_vec(), vec![group]).expect("valid prover claims");
    ProverOpeningData::new(opening_claims, vec![hint], vec![polynomials])
        .expect("valid prover opening data")
}

/// Builds a single-group, full-point [`OpeningClaims`] asserting `openings`
/// at `point` against `commitment`.
pub fn verify_input<'a, FF: FieldCore, C>(
    point: &'a [FF],
    openings: &'a [FF],
    commitment: &'a C,
) -> OpeningClaims<'static, FF, &'a C> {
    OpeningClaims::from_groups(
        point.to_vec(),
        vec![PolynomialGroupClaims::new(
            PointVariableSelection::prefix(point.len(), point.len()).expect("full-point group"),
            openings.to_vec(),
            commitment,
        )
        .expect("valid verifier claims group")],
    )
    .expect("valid verifier input")
}

/// Evaluates `poly` at `point` under `layout`, using the Lagrange basis.
pub fn opening_from_poly<'a, const D: usize, P>(poly: &'a P, point: &[F], layout: &LevelParams) -> F
where
    P: RootOpeningSource<F, D> + RootPolyShape<F, D>,
    CpuBackend: OpeningFoldKernel<P::OpeningView<'a>, F, D>,
{
    opening_from_poly_with_basis::<D, P>(poly, point, layout, BasisMode::Lagrange)
}

/// Evaluates `poly` at `point` under `layout`, using `basis_mode`.
pub fn opening_from_poly_with_basis<'a, const D: usize, P>(
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

/// Concrete, non-generic instantiation of [`opening_from_poly`] at
/// [`OneHotCfg`]'s ring dimension for [`OneHotPoly<F, u8>`].
///
/// `opening_from_poly` is generic over `D` and the poly type, so calling the
/// generic version directly re-monomorphizes it in every calling binary.
/// This wrapper's body is fully concrete, so it is compiled exactly once
/// here instead. Every `akita-pcs` integration test that opens a one-hot
/// poly under [`OneHotCfg`] should call this instead of the generic
/// function.
pub fn opening_from_poly_onehot(poly: &OneHotPoly<F, u8>, point: &[F], layout: &LevelParams) -> F {
    opening_from_poly::<ONEHOT_D, OneHotPoly<F, u8>>(poly, point, layout)
}

/// Concrete, non-generic instantiation of [`opening_from_poly`] at
/// [`DenseCfg`]'s ring dimension for [`DensePoly<F>`].
///
/// See [`opening_from_poly_onehot`] for why this wrapper exists.
pub fn opening_from_poly_dense(poly: &DensePoly<F>, point: &[F], layout: &LevelParams) -> F {
    opening_from_poly::<DENSE_D, DensePoly<F>>(poly, point, layout)
}

/// Builds a one-hot polynomial fixture sized for `layout`, seeded from
/// `seed`, under [`OneHotCfg`].
pub fn make_onehot_poly(layout: &LevelParams, seed: u64) -> OneHotPoly<F, u8> {
    // `2^nv = (num_blocks · block_len) · D` field elements, grouped into
    // `2^nv / K` one-hot chunks of size `K`.
    let total_field = layout.num_blocks * layout.block_len * ONEHOT_D;
    let total_chunks = total_field / ONEHOT_K;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();
    OneHotPoly::<F, u8>::new(ONEHOT_K, ONEHOT_D, indices).expect("onehot poly")
}

/// Builds a dense polynomial fixture with `nv` variables, seeded from
/// `seed`, under [`DenseCfg`].
pub fn make_dense_poly(nv: usize, seed: u64) -> DensePoly<F> {
    let evals = dense_field_evals(nv, seed);
    DensePoly::<F>::from_field_evals(nv, DENSE_D, &evals).expect("dense poly")
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Deterministically samples `2^nv` field elements from `seed`.
pub fn dense_field_evals(nv: usize, seed: u64) -> Vec<F> {
    let n = 1usize << nv;
    let mut out = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        let v = splitmix64_next(&mut state);
        out.push(F::from_canonical_u128_reduced(v as u128));
    }
    out
}

/// Filters wire events out of a transcript event log, for tests that assert
/// on public-only transcript structure.
#[cfg(feature = "logging-transcript")]
pub fn public_transcript_events(
    events: &[akita_transcript::TranscriptEvent],
) -> Vec<akita_transcript::TranscriptEvent> {
    events
        .iter()
        .filter(|event| !matches!(event, akita_transcript::TranscriptEvent::Wire { .. }))
        .cloned()
        .collect()
}

/// Returns the label of a transcript event, if it carries one.
#[cfg(feature = "logging-transcript")]
pub fn event_label(event: &akita_transcript::TranscriptEvent) -> Option<&[u8]> {
    match event {
        akita_transcript::TranscriptEvent::Absorb { label, .. }
        | akita_transcript::TranscriptEvent::Squeeze { label, .. }
        | akita_transcript::TranscriptEvent::Wire { label, .. } => Some(label),
        akita_transcript::TranscriptEvent::Preamble { .. } => None,
    }
}

/// Finds the first event carrying `label`.
#[cfg(feature = "logging-transcript")]
pub fn first_label_index(
    events: &[akita_transcript::TranscriptEvent],
    label: &[u8],
) -> Option<usize> {
    events
        .iter()
        .position(|event| event_label(event).is_some_and(|candidate| candidate == label))
}

/// Finds the first event carrying `label` at or after `start`.
#[cfg(feature = "logging-transcript")]
pub fn first_label_index_after(
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

/// Finds the first event carrying `label` or one of its extension limbs, at
/// or after `start`.
#[cfg(feature = "logging-transcript")]
pub fn first_label_or_extension_limb_index_after(
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

/// Asserts that a terminal-round transcript's challenge/absorb events appear
/// in the expected relative order, returning the index of the terminal
/// e_hat absorb if present.
#[cfg(feature = "logging-transcript")]
pub fn assert_terminal_event_order_if_present(
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
