//! Representative root-width screening, excluded from ordinary test runs.
use super::*;
use crate::opaque::ComputeBackendSetup;
use crate::sources::onehot::column_sweep_ajtai_onehot_multi;
use crate::AkitaProverSetup;
use akita_params::SetupMatrixCapacity;
use jolt_field::Prime128OffsetA7F7 as F;
use rand::{Rng, SeedableRng};
use std::time::Instant;

#[test]
#[ignore = "512 MiB setup; representative root-width performance screen"]
fn onehot_ntt_root_screen() {
    const D: usize = 512;
    const WIDTH: usize = 65536;
    const BLOCKS: usize = 256;
    #[cfg(feature = "parallel")]
    let threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let threads = 1;
    eprintln!(
        "[onehot_screen] threads={threads} AVX512={:?}",
        std::env::var("AKITA_AVX512_NTT")
    );
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        33,
        1,
        SetupMatrixCapacity {
            num_field_elements: D * WIDTH,
        },
    )
    .unwrap();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let plan = CommitInnerPlan {
        ring_dimension: D,
        num_live_blocks: BLOCKS,
        n_a: 1,
        num_positions_per_block: WIDTH,
        num_digits_inner: 1,
        log_basis_inner: 3,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xbeef_cafe);
    let indices = (0..D * WIDTH * BLOCKS / 16)
        .map(|_| Some(rng.gen_range(0u8..16)))
        .collect::<Vec<_>>();
    let source = OneHotSource {
        indices: &indices,
        chunk_size: 16,
        num_vars: 33,
    };
    for domain in [
        NttTransformDomain::Negacyclic,
        NttTransformDomain::ExactNegacyclicI16 {
            width: WIDTH,
            rhs_abs_bound: 2,
        },
    ] {
        prepared
            .with_shared_ntt::<D, _>(
                NttCacheKey::from_matrix_shape(D, 1, WIDTH, domain).unwrap(),
                |_| Ok(()),
            )
            .unwrap();
    }
    let view = setup
        .expanded
        .shared_matrix
        .ring_view::<D>(1, WIDTH)
        .unwrap();
    let mut expected = None;
    let variants = [
        ("accumulation", 1, 96, false),
        ("ntt-legacy", 1, 96, false),
        ("ntt-exact", 1, 96, false),
        ("ntt-panel6", 1, 96, false),
        ("ntt-panel6", 4, 96, false),
        ("ntt-panel5", 1, 96, false),
        ("ntt-panel5", 2, 96, false),
        ("ntt-panel5", 4, 96, false),
        ("ntt-panel5", 8, 96, false),
        ("ntt-panel5", 16, 96, false),
        ("ntt-panel5", 4, 24, false),
        ("ntt-panel5", 4, 48, false),
        ("ntt-panel5", 4, 192, false),
        ("ntt-panel5", 4, 384, false),
        ("ntt-panel5", 8, 48, false),
        ("ntt-panel5", 8, 192, false),
        ("ntt-binary5", 1, 96, false),
        ("ntt-binary5", 4, 96, false),
        ("ntt-panel5", 4, 96, true),
        ("ntt-panel5", 8, 96, true),
    ];
    for repeat in 0..2 {
        for (mode, group, columns, prefetch) in variants {
            std::env::set_var("AKITA_ONEHOT_NTT_COLUMNS", columns.to_string());
            std::env::set_var(
                "AKITA_ONEHOT_NTT_PREFETCH",
                if prefetch { "1" } else { "0" },
            );
            let started = Instant::now();
            let actual = if mode == "accumulation" {
                let rows =
                    column_sweep_ajtai_onehot_multi::<F, D, _>(&view, &[source], 1, WIDTH, 1)
                        .unwrap();
                vec![crate::typed_inner_rows(rows.into_iter().next().unwrap()).unwrap()]
            } else {
                backend
                    .onehot_ntt_variant::<D, _>(&prepared, &[source], plan, mode, group)
                    .unwrap()
            };
            let elapsed = started.elapsed().as_secs_f64();
            if let Some(expected) = &expected {
                assert_eq!(actual[0].coeffs(), expected);
            } else {
                expected = Some(actual[0].coeffs().to_vec());
            }
            eprintln!("[onehot_screen] repeat={repeat} mode={mode} blocks={group} columns={columns} prefetch={prefetch} elapsed_s={elapsed:.6} live_blocks={BLOCKS} width={WIDTH} D={D}");
        }
    }
}

#[test]
fn onehot_l1_capacity_is_strict_at_large_widths() {
    for (h, expected) in [(16, 262080), (32, 131040)] {
        let capacity = akita_algebra::CrtCapacity::from_prime_moduli(
            akita_algebra::tables::q128_primes()[..5]
                .iter()
                .map(|p| p.p as u128),
        );
        assert_eq!(capacity.max_safe_width_for_ring_l1::<F>(h), Some(expected));
    }
}
