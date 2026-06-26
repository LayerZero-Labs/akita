//! Prover≡verifier setup-geometry cross-check on the mixed-D hand-built schedule.

use akita_config::proof_optimized::fp128;
use akita_config::test_support::mixed_d_per_level_schedule;
use akita_config::CommitmentConfig;
use akita_field::AkitaError;
use akita_serialization::Valid;
use akita_types::{
    active_setup_field_len, compute_setup_layout, setup_active_ring_elems_at,
    setup_required_for_shape, AkitaExpandedSetup, AkitaScheduleLookupKey, MRowLayout,
    OpeningBatchShape, SetupRelationShape, SETUP_OFFLOAD_D_SETUP,
};

type F = fp128::Field;

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
fn mixed_d_setup_geometry_matches_layout_and_active_elems() {
    let key = AkitaScheduleLookupKey::singleton(16);
    let schedule =
        mixed_d_per_level_schedule::<fp128::D128Full, fp128::D64Full>(16, 1, 2).expect("schedule");
    let field_bits = fp128::D128Full::decomposition().field_bits();
    let expanded = synthetic_expanded(128, 1 << 20).expect("expanded setup");

    for level in 0..schedule.num_fold_levels() {
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
        let shape = SetupRelationShape::from_level_params(
            lp,
            key.num_polynomials,
            m_row_layout,
            depth_fold,
        )
        .expect("relation shape");
        let required = setup_required_for_shape(&shape).expect("required");
        let layout = compute_setup_layout(&shape).expect("layout");
        assert_eq!(
            required, layout.required,
            "mixed-D level {level}: required vs layout"
        );
        let active = setup_active_ring_elems_at(level, &schedule, &expanded, &shape)
            .expect("active ring elems");
        assert_eq!(
            active, required,
            "mixed-D level {level}: active elems must match required (prover≡verifier gate)"
        );
        let opening_batch =
            OpeningBatchShape::new(key.num_vars, key.num_polynomials).expect("opening batch");
        let field_len = active_setup_field_len(
            lp,
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
