//! Prover≡verifier setup-geometry cross-check on the mixed-D hand-built schedule.
//!
//! CI runs shape-level checks plus one minimal-envelope integration pass. A larger
//! envelope sweep is available behind `#[ignore]` for local profiling.

use akita_config::proof_optimized::fp128;
use akita_config::test_support::mixed_d_per_level_schedule;
use akita_config::CommitmentConfig;
use akita_field::AkitaError;
use akita_serialization::Valid;
use akita_types::{
    active_setup_field_len, compute_setup_layout, setup_active_ring_elems_at,
    setup_required_for_shape, AkitaExpandedSetup, AkitaScheduleLookupKey, MRowLayout,
    OpeningBatchShape, Schedule, SetupRelationShape, SETUP_OFFLOAD_D_SETUP,
};

type F = fp128::Field;

const GEN_RING_DIM: usize = 128;

fn mixed_d_fixture_schedule() -> Schedule {
    mixed_d_per_level_schedule::<fp128::D128Full, fp128::D64Full>(16, 1, 2).expect("schedule")
}

fn level_relation_shape(
    schedule: &Schedule,
    key: &AkitaScheduleLookupKey,
    level: usize,
    field_bits: u32,
) -> SetupRelationShape {
    let exec = schedule
        .get_execution_schedule(level)
        .expect("execution schedule");
    let lp = &exec.params;
    let m_row_layout = if exec.is_terminal {
        MRowLayout::WithoutDBlock
    } else {
        MRowLayout::WithDBlock
    };
    let depth_fold = lp
        .num_digits_fold(key.num_polynomials, field_bits)
        .expect("depth_fold");
    SetupRelationShape::from_level_params(lp, key.num_polynomials, m_row_layout, depth_fold)
        .expect("relation shape")
}

/// Smallest `max_setup_len` (ring elements at `gen_ring_dim`) that fits every fold level.
fn min_envelope_ring_elems(
    schedule: &Schedule,
    key: &AkitaScheduleLookupKey,
    gen_ring_dim: usize,
    field_bits: u32,
) -> usize {
    let mut max_setup_len = 1usize;
    for level in 0..schedule.num_fold_levels() {
        let shape = level_relation_shape(schedule, key, level, field_bits);
        let required = setup_required_for_shape(&shape).expect("required");
        let fold_ring_d = schedule
            .get_execution_schedule(level)
            .expect("execution schedule")
            .params
            .ring_dimension;
        let envelope_rows = required.saturating_mul(fold_ring_d).div_ceil(gen_ring_dim);
        max_setup_len = max_setup_len.max(envelope_rows);
    }
    max_setup_len
}

fn synthetic_expanded(
    gen_ring_dim: usize,
    max_setup_len: usize,
) -> Result<AkitaExpandedSetup<F>, AkitaError> {
    use akita_types::{derive_public_matrix_flat, sample_public_matrix_seed, AkitaSetupSeed};
    let seed = AkitaSetupSeed {
        max_num_vars: 16,
        max_num_batched_polys: 1,
        gen_ring_dim,
        max_setup_len,
        public_matrix_seed: sample_public_matrix_seed(),
    };
    seed.check()
        .map_err(|e| AkitaError::InvalidSetup(format!("seed: {e}")))?;
    let shared_flat = derive_public_matrix_flat::<F, 128>(max_setup_len, &seed.public_matrix_seed);
    AkitaExpandedSetup::from_verified_parts(seed, shared_flat)
        .map_err(|e| AkitaError::InvalidSetup(format!("expanded setup: {e}")))
}

#[test]
fn mixed_d_setup_geometry_shape_agrees_per_level() {
    let key = AkitaScheduleLookupKey::singleton(16);
    let schedule = mixed_d_fixture_schedule();
    let field_bits = fp128::D128Full::decomposition().field_bits();

    for level in 0..schedule.num_fold_levels() {
        let shape = level_relation_shape(&schedule, &key, level, field_bits);
        let required = setup_required_for_shape(&shape).expect("required");
        let layout = compute_setup_layout(&shape).expect("layout");
        assert_eq!(
            required, layout.required,
            "mixed-D level {level}: required vs layout"
        );
        let opening_batch =
            OpeningBatchShape::new(key.num_vars, key.num_polynomials).expect("opening batch");
        let exec = schedule
            .get_execution_schedule(level)
            .expect("execution schedule");
        let m_row_layout = if exec.is_terminal {
            MRowLayout::WithoutDBlock
        } else {
            MRowLayout::WithDBlock
        };
        let depth_fold = exec
            .params
            .num_digits_fold(key.num_polynomials, field_bits)
            .expect("depth_fold");
        let field_len = active_setup_field_len(
            &exec.params,
            &opening_batch,
            m_row_layout,
            depth_fold,
            SETUP_OFFLOAD_D_SETUP,
        )
        .expect("field len");
        assert_eq!(
            field_len,
            layout.required * SETUP_OFFLOAD_D_SETUP,
            "mixed-D level {level}: prefix field len"
        );
    }
}

#[test]
fn mixed_d_setup_active_elems_matches_min_envelope() {
    let key = AkitaScheduleLookupKey::singleton(16);
    let schedule = mixed_d_fixture_schedule();
    let field_bits = fp128::D128Full::decomposition().field_bits();
    let max_setup_len = min_envelope_ring_elems(&schedule, &key, GEN_RING_DIM, field_bits);
    let expanded = synthetic_expanded(GEN_RING_DIM, max_setup_len).expect("expanded setup");

    for level in 0..schedule.num_fold_levels() {
        let shape = level_relation_shape(&schedule, &key, level, field_bits);
        let required = setup_required_for_shape(&shape).expect("required");
        let active = setup_active_ring_elems_at(level, &schedule, &expanded, &shape)
            .expect("active ring elems");
        assert_eq!(
            active, required,
            "mixed-D level {level}: active elems must match required (prover≡verifier gate)"
        );
    }
}

/// Local-only sweep: proves the geometry gate still holds on a large envelope prefix.
#[test]
#[ignore = "expensive envelope sweep; run locally with --ignored"]
fn mixed_d_setup_geometry_large_envelope_sweep() {
    let key = AkitaScheduleLookupKey::singleton(16);
    let schedule = mixed_d_fixture_schedule();
    let field_bits = fp128::D128Full::decomposition().field_bits();
    let expanded = synthetic_expanded(GEN_RING_DIM, 1 << 20).expect("expanded setup");

    for level in 0..schedule.num_fold_levels() {
        let shape = level_relation_shape(&schedule, &key, level, field_bits);
        let required = setup_required_for_shape(&shape).expect("required");
        let active = setup_active_ring_elems_at(level, &schedule, &expanded, &shape)
            .expect("active ring elems");
        assert_eq!(
            active, required,
            "mixed-D level {level}: large envelope sweep"
        );
    }
}
