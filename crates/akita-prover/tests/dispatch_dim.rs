//! Integration tests for [`akita_prover::dispatch_ring_d!`] and
//! [`akita_types::ValidatedScheduleContext`] against real generated schedules.
//!
//! These tests exercise the S3 building blocks from Tier 2 of the runtime
//! ring-dimension cutover plan.  They require real schedules from
//! `akita-config` presets and a hand-built mixed-D schedule.
//!
//! Gate condition from the plan:
//! `cargo test -p akita-prover dispatch` must pass.

#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp64};
use akita_config::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleLookupKey, CleartextWitnessShape, DirectStep, FoldStep, LevelParams, Schedule,
    Step, ValidatedScheduleContext,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve a real schedule from a config preset at the given `num_vars`.
fn real_schedule<Cfg: CommitmentConfig>(num_vars: usize) -> Schedule {
    Cfg::runtime_schedule(AkitaScheduleLookupKey::singleton(num_vars))
        .expect("valid schedule for num_vars")
}

/// Build a minimal `FoldStep` with explicit ring dimension and geometry.
fn make_fold_step(ring_dimension: usize, num_blocks: usize, block_len: usize) -> FoldStep {
    let mut params = LevelParams::log_basis_stub(3);
    params.ring_dimension = ring_dimension;
    params.num_blocks = num_blocks;
    params.block_len = block_len;
    FoldStep {
        params,
        current_w_len: 0,
        next_w_len: 0,
        level_bytes: 0,
    }
}

fn make_direct_step() -> DirectStep {
    DirectStep {
        current_w_len: 0,
        witness_shape: CleartextWitnessShape::FieldElements(0),
        direct_bytes: 0,
        params: None,
    }
}

fn mixed_d_schedule(dims: &[(usize, usize, usize)]) -> Schedule {
    let mut steps: Vec<Step> = dims
        .iter()
        .map(|&(d, nb, bl)| Step::Fold(make_fold_step(d, nb, bl)))
        .collect();
    steps.push(Step::Direct(make_direct_step()));
    Schedule {
        steps,
        total_bytes: 0,
    }
}

// ---------------------------------------------------------------------------
// dispatch_ring_d! tests — routing and rejection
// ---------------------------------------------------------------------------

#[test]
fn dispatch_ring_d_routes_all_supported_dimensions() {
    for d in [32usize, 64, 128, 256] {
        let got: Result<usize, AkitaError> = akita_prover::dispatch_ring_d!(d, |D| Ok(D));
        assert_eq!(got.unwrap(), d, "wrong arm for d={d}");
    }
}

#[test]
fn dispatch_ring_d_rejects_unsupported_48() {
    let err = akita_prover::dispatch_ring_d!(48usize, |D| Ok(D)).expect_err("48 is not supported");
    assert!(matches!(err, AkitaError::InvalidInput(_)));
}

#[test]
fn dispatch_ring_d_rejects_zero() {
    let err = akita_prover::dispatch_ring_d!(0usize, |D| Ok(D)).expect_err("0 is not supported");
    assert!(matches!(err, AkitaError::InvalidInput(_)));
}

// ---------------------------------------------------------------------------
// ValidatedScheduleContext against REAL schedules (fp64::D64Full, fp128)
// ---------------------------------------------------------------------------

/// For fp64::D64Full, `Cfg::D == 64`, so `gen_ring_dim = 64` and every level
/// must carry `ring_dimension = 64` (uniform-D preset).
#[test]
fn validated_ctx_accepts_fp64_d64_schedule_with_gen_ring_dim_64() {
    let sched = real_schedule::<fp64::D64Full>(20);
    // For a uniform-D preset the gen_ring_dim equals the preset's D.
    let gen_ring_dim = fp64::D64Full::D;
    assert_eq!(gen_ring_dim, 64);
    let ctx = ValidatedScheduleContext::new(&sched, gen_ring_dim)
        .expect("fp64 D64 schedule must be valid for gen_ring_dim=64");
    // All fold levels must report ring_dimension == 64.
    for shape in ctx.level_shapes() {
        assert_eq!(
            shape.ring_dimension, 64,
            "level {}: expected ring_dim=64",
            shape.level
        );
        // flat_field_len == n_ring_elems * ring_dimension
        assert_eq!(
            shape.flat_field_len,
            shape.n_ring_elems * shape.ring_dimension,
            "level {}: flat_field_len mismatch",
            shape.level
        );
    }
}

/// For fp128, `Cfg::D == 128`; validate against gen_ring_dim=128.
#[test]
fn validated_ctx_accepts_fp128_schedule_with_gen_ring_dim_128() {
    // fp128 has different sub-presets; pick a commonly-used one.
    type Cfg = fp128::D128Full;
    let sched = real_schedule::<Cfg>(18);
    let gen_ring_dim = Cfg::D;
    assert_eq!(gen_ring_dim, 128);
    let ctx = ValidatedScheduleContext::new(&sched, gen_ring_dim)
        .expect("fp128 D128 schedule must be valid for gen_ring_dim=128");
    for shape in ctx.level_shapes() {
        assert_eq!(shape.ring_dimension, 128, "level {} ring_dim", shape.level);
        assert_eq!(
            shape.flat_field_len,
            shape.n_ring_elems * shape.ring_dimension,
            "level {} flat_field_len",
            shape.level
        );
    }
}

/// A fp64::D32Full schedule validated against gen_ring_dim=256 (32 | 256).
#[test]
fn validated_ctx_accepts_fp64_d32_schedule_against_larger_gen_ring_dim() {
    type Cfg = fp64::D32Full;
    let sched = real_schedule::<Cfg>(16);
    // 32 divides 256.
    let ctx = ValidatedScheduleContext::new(&sched, 256)
        .expect("D=32 schedule must be valid for gen_ring_dim=256 (32|256)");
    for shape in ctx.level_shapes() {
        assert_eq!(shape.ring_dimension, 32, "level {} ring_dim", shape.level);
    }
}

/// Passing gen_ring_dim=64 for a D=128 schedule must fail: 128 ∤ 64.
#[test]
fn validated_ctx_rejects_fp128_schedule_against_too_small_gen_ring_dim() {
    type Cfg = fp128::D128Full;
    let sched = real_schedule::<Cfg>(16);
    let err = ValidatedScheduleContext::new(&sched, 64)
        .expect_err("128 does not divide 64; must be rejected");
    assert!(
        matches!(err, AkitaError::InvalidSetup(_)),
        "expected InvalidSetup, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// ValidatedScheduleContext — hand-built mixed-D schedule
// ---------------------------------------------------------------------------

/// A schedule where each fold level has a *different* ring dimension.
/// This is the critical "type layer already allows mixed-D" fixture.
///
/// gen_ring_dim = 256; levels use 32, 64, 128, 256 — all divide 256.
#[test]
fn validated_ctx_accepts_mixed_d_schedule_all_divide_gen_ring_dim() {
    // (ring_dimension, num_blocks, block_len)
    let sched = mixed_d_schedule(&[
        (32, 4, 8),  // level 0: D=32
        (64, 4, 4),  // level 1: D=64
        (128, 2, 4), // level 2: D=128
        (256, 2, 2), // level 3: D=256
    ]);
    let ctx = ValidatedScheduleContext::new(&sched, 256).expect("all dims divide 256");

    assert_eq!(ctx.num_fold_levels(), 4);

    // Verify per-level shapes.
    let expected = [
        (32usize, 32usize, 1024usize),
        (64, 16, 1024),
        (128, 8, 1024),
        (256, 4, 1024),
    ];
    for ((d, nr, ff), shape) in expected.iter().zip(ctx.level_shapes()) {
        assert_eq!(shape.ring_dimension, *d, "level {} ring_dim", shape.level);
        assert_eq!(
            shape.n_ring_elems, *nr,
            "level {} n_ring_elems",
            shape.level
        );
        assert_eq!(
            shape.flat_field_len, *ff,
            "level {} flat_field_len",
            shape.level
        );
    }
}

/// A mixed-D schedule where one level's ring_dimension does NOT divide
/// gen_ring_dim — this must be rejected.
#[test]
fn validated_ctx_rejects_mixed_d_schedule_one_bad_level() {
    // Level 1 uses D=96 which does not divide 256.
    let sched = mixed_d_schedule(&[
        (64, 4, 4),
        (96, 4, 4), // 96 ∤ 256
        (128, 2, 4),
    ]);
    let err = ValidatedScheduleContext::new(&sched, 256)
        .expect_err("96 does not divide 256; must be rejected");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

/// Divisibility holds at the first level but gen_ring_dim < ring_dimension
/// at a later level — must be rejected.
#[test]
fn validated_ctx_rejects_level_ring_dim_larger_than_gen_ring_dim() {
    let sched = mixed_d_schedule(&[
        (64, 4, 4),
        (512, 2, 2), // 512 ∤ 256
    ]);
    let err = ValidatedScheduleContext::new(&sched, 256).expect_err("512 does not divide 256");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}
