//! Pinned setup-geometry golden vectors (challenge-free footprint).

use akita_types::{
    compute_setup_layout, ensure_setup_envelope, setup_required_for_shape,
    stage3_offload_natural_field_len, MRowLayout, SetupRelationShape, SETUP_OFFLOAD_D_SETUP,
};

struct GoldenCase {
    name: &'static str,
    shape: SetupRelationShape,
    ring_d: usize,
    required: usize,
    natural_field_len: usize,
    tiered: bool,
    d_required: usize,
    a_required: usize,
}

fn root_single_point() -> SetupRelationShape {
    SetupRelationShape {
        num_t_vectors: 1,
        num_blocks: 4,
        num_claims: 1,
        depth_open: 8,
        depth_commit: 2,
        depth_fold: 3,
        block_len: 16,
        inner_width: 32,
        n_a: 2,
        n_d: 1,
        m_row_layout: MRowLayout::WithDBlock,
        n_b: 2,
        num_segments: 1,
        rows: 7,
        num_polys_per_segment: vec![1],
        num_public_rows: 1,
        tier_split: 1,
        n_f: 0,
    }
}

fn tiered_root_single_point() -> SetupRelationShape {
    let mut shape = root_single_point();
    shape.tier_split = 4;
    shape.n_f = 1;
    shape.rows = 14;
    shape
}

fn terminal_relation_only() -> SetupRelationShape {
    let mut shape = root_single_point();
    shape.m_row_layout = MRowLayout::WithoutDBlock;
    shape
}

fn dense_non_pow2_z() -> SetupRelationShape {
    let mut shape = root_single_point();
    shape.block_len = 12;
    shape.depth_commit = 3;
    shape.depth_fold = 2;
    shape.inner_width = 36;
    shape
}

fn batched_root() -> SetupRelationShape {
    let mut shape = root_single_point();
    shape.num_claims = 4;
    shape.num_polys_per_segment = vec![4];
    shape.num_t_vectors = 4;
    shape
}

fn recursive_multigroup() -> SetupRelationShape {
    SetupRelationShape {
        num_t_vectors: 3,
        num_blocks: 8,
        num_claims: 3,
        depth_open: 26,
        depth_commit: 1,
        depth_fold: 4,
        block_len: 512,
        inner_width: 512,
        n_a: 2,
        n_d: 2,
        m_row_layout: MRowLayout::WithDBlock,
        n_b: 2,
        num_segments: 1,
        rows: 8,
        num_polys_per_segment: vec![3],
        num_public_rows: 1,
        tier_split: 1,
        n_f: 0,
    }
}

fn golden_cases() -> Vec<GoldenCase> {
    vec![
        GoldenCase {
            name: "root_single_point",
            shape: root_single_point(),
            ring_d: 32,
            required: 128,
            natural_field_len: 128 * SETUP_OFFLOAD_D_SETUP,
            tiered: false,
            d_required: 32,
            a_required: 64,
        },
        GoldenCase {
            name: "tiered_root_single_point",
            shape: tiered_root_single_point(),
            ring_d: 32,
            required: 64,
            natural_field_len: 64 * SETUP_OFFLOAD_D_SETUP,
            tiered: true,
            d_required: 32,
            a_required: 64,
        },
        GoldenCase {
            name: "terminal_relation_only",
            shape: terminal_relation_only(),
            ring_d: 32,
            required: 128,
            natural_field_len: 128 * SETUP_OFFLOAD_D_SETUP,
            tiered: false,
            d_required: 0,
            a_required: 64,
        },
        GoldenCase {
            name: "dense_non_pow2_z",
            shape: dense_non_pow2_z(),
            ring_d: 32,
            required: 128,
            natural_field_len: 128 * SETUP_OFFLOAD_D_SETUP,
            tiered: false,
            d_required: 32,
            a_required: 72,
        },
        GoldenCase {
            name: "batched_root",
            shape: batched_root(),
            ring_d: 32,
            required: 512,
            natural_field_len: 512 * SETUP_OFFLOAD_D_SETUP,
            tiered: false,
            d_required: 128,
            a_required: 64,
        },
        GoldenCase {
            name: "recursive_multigroup",
            shape: recursive_multigroup(),
            ring_d: 32,
            required: 2496,
            natural_field_len: 2496 * SETUP_OFFLOAD_D_SETUP,
            tiered: false,
            d_required: 1248,
            a_required: 1024,
        },
    ]
}

#[test]
fn setup_geometry_golden_vectors() {
    for case in golden_cases() {
        let required = setup_required_for_shape(&case.shape).expect(case.name);
        assert_eq!(required, case.required, "required: {}", case.name);

        let layout = compute_setup_layout(&case.shape).expect(case.name);
        assert_eq!(
            layout.required, case.required,
            "layout.required: {}",
            case.name
        );
        assert_eq!(layout.tiered, case.tiered, "tiered: {}", case.name);
        assert_eq!(
            layout.d_required, case.d_required,
            "d_required: {}",
            case.name
        );
        assert_eq!(
            layout.a_required, case.a_required,
            "a_required: {}",
            case.name
        );

        let natural_field_len =
            stage3_offload_natural_field_len(required, SETUP_OFFLOAD_D_SETUP).expect(case.name);
        assert_eq!(
            natural_field_len, case.natural_field_len,
            "natural_field_len: {}",
            case.name
        );
    }
}

#[test]
fn ensure_setup_envelope_accepts_golden_footprints() {
    use akita_field::Prime128OffsetA7F7;
    type F = Prime128OffsetA7F7;

    for case in golden_cases() {
        let seed = akita_types::AkitaSetupSeed {
            max_num_vars: 32,
            max_num_batched_polys: case.shape.num_t_vectors,
            gen_ring_dim: case.ring_d,
            max_setup_len: case.required,
            public_matrix_seed: [case.name.as_bytes()[0]; 32],
        };
        let shared = akita_types::derive_public_matrix_flat::<F, 32>(
            case.required,
            &seed.public_matrix_seed,
        );
        let expanded = akita_types::AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            seed, shared,
        );
        ensure_setup_envelope(&expanded, case.required, case.ring_d)
            .unwrap_or_else(|_| panic!("envelope: {}", case.name));
    }
}
