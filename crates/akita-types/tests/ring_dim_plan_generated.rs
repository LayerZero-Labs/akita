//! `RingDimPlan::from_schedule` on every shipped generated schedule table.

use akita_config::generated_families::{family_keys, ALL_GENERATED_FAMILIES};
use akita_types::{AkitaSetupSeed, RingDimPlan};

#[test]
fn from_schedule_accepts_all_generated_tables() {
    for family in ALL_GENERATED_FAMILIES {
        let policy = (family.policy)();
        let keys = family_keys(family).expect("family keys");
        let sample_keys = [keys.first().copied(), keys.last().copied()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        for key in sample_keys {
            let schedule = (family.regen)(key).expect("schedule");
            let seed = AkitaSetupSeed {
                max_num_vars: family.max_num_vars,
                max_num_batched_polys: family
                    .num_polys
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(1),
                gen_ring_dim: policy.ring_dimension,
                max_setup_len: 4096,
                public_matrix_seed: [0u8; 32],
            };
            let plan = RingDimPlan::from_schedule(&schedule, &seed).unwrap_or_else(|err| {
                panic!(
                    "from_schedule failed for family={} key={key:?}: {err}",
                    family.module_name
                )
            });
            assert_eq!(plan.num_folds, schedule.num_fold_levels());
            if plan.num_folds == 0 {
                continue;
            }
            for level in 0..plan.num_folds {
                let d = plan.dim_at(level).expect("dim_at");
                assert_eq!(d, policy.ring_dimension);
                let dims = plan.dims_at(level).expect("dims_at");
                assert_eq!(dims.inner, dims.outer);
                assert_eq!(dims.outer, dims.opening);
            }
            assert_eq!(plan.unique_dims(), vec![policy.ring_dimension]);
        }
    }
}
