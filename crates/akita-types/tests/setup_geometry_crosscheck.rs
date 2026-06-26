//! Cross-check setup geometry helpers on generated schedules.

use akita_config::generated_families::{family_keys, ALL_GENERATED_FAMILIES};
use akita_types::{
    active_setup_field_len, compute_setup_layout, setup_required_for_shape, MRowLayout,
    SetupRelationShape, SETUP_OFFLOAD_D_SETUP,
};

fn field_bits_for_sis(sis: akita_types::SisModulusFamily) -> u32 {
    match sis {
        akita_types::SisModulusFamily::Q32 => 32,
        akita_types::SisModulusFamily::Q64 => 64,
        akita_types::SisModulusFamily::Q128 => 128,
    }
}

#[test]
fn setup_required_matches_layout_on_generated_schedules() {
    for family in ALL_GENERATED_FAMILIES {
        let policy = (family.policy)();
        let field_bits = field_bits_for_sis(policy.sis_family);
        let keys = family_keys(family).expect("family keys");
        let sample_keys = [keys.first().copied(), keys.last().copied()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        for key in sample_keys {
            let schedule = (family.regen)(key).expect("schedule");
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
                    "family={} key={key:?} level={level}",
                    family.module_name,
                );
                let opening_batch =
                    akita_types::OpeningBatchShape::new(key.num_vars, key.num_polynomials)
                        .expect("opening batch");
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
                    "prefix field len must match sumcheck footprint family={} key={key:?} level={level}",
                    family.module_name,
                );
            }
        }
    }
}
