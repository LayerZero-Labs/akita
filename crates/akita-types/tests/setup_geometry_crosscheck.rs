//! Cross-check `setup_geometry_at` against `compute_setup_layout` on generated schedules.

use akita_config::generated_families::{family_keys, ALL_GENERATED_FAMILIES};
use akita_types::{compute_setup_layout, setup_geometry_at, MRowLayout, SetupRelationShape};

fn field_bits_for_sis(sis: akita_types::SisModulusFamily) -> u32 {
    match sis {
        akita_types::SisModulusFamily::Q32 => 32,
        akita_types::SisModulusFamily::Q64 => 64,
        akita_types::SisModulusFamily::Q128 => 128,
    }
}

#[test]
fn setup_geometry_matches_layout_on_generated_schedules() {
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
                let geometry = setup_geometry_at(level, &schedule, &shape).expect("geometry");
                let layout = compute_setup_layout(&shape).expect("layout");
                assert_eq!(
                    geometry.required, layout.required,
                    "family={} key={key:?} level={level}",
                    family.module_name,
                );
            }
        }
    }
}
