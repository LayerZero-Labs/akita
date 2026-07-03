use super::*;
use crate::{
    CleartextWitnessShape, FoldStep, LevelParams, OpeningClaimsLayout, PolynomialGroupLayout, Step,
};
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::{Prime32Offset99, Prime64Offset59};

fn sample_level_params() -> LevelParams {
    LevelParams::params_only(
        SisModulusFamily::Q32,
        32,
        3,
        2,
        3,
        2,
        SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        },
    )
    .with_decomp(2, 3, 2, 2, 0)
    .expect("sample level params")
}

fn sample_descriptor() -> AkitaInstanceDescriptor {
    let opening_batch = OpeningClaimsLayout::new(5, 3).expect("valid opening batch");
    let schedule = Schedule {
        steps: vec![
            Step::Fold(FoldStep {
                params: sample_level_params(),
                current_w_len: 256,
                next_w_len: 256,
                level_bytes: 123,
            }),
            Step::Direct(crate::DirectStep {
                current_w_len: 256,
                witness_shape: CleartextWitnessShape::FieldElements(64),
                direct_bytes: 32,
                params: None,
            }),
        ],
        total_bytes: 155,
    };

    AkitaInstanceDescriptor::new(
        AlgebraSection::for_fields::<Prime32Offset99, Prime32Offset99, 32>().expect("algebra"),
        SetupSection {
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 32,
                log_open_bound: Some(32),
            },
            sis_modulus_family: SisModulusFamily::Q32,
            setup_seed_digest: [1; 32],
            protocol_features: ProtocolFeatureSet::current(),
            fold_linf: FoldLinfProtocolBinding::CURRENT,
        },
        PlanSection::from_schedule(&schedule),
        CallSection::from_layout(&opening_batch, BasisMode::Lagrange).expect("call"),
    )
}

#[test]
fn rejects_removed_q16_sis_family_tag() {
    let err = decode_sis_family(std::io::Cursor::new([3u8]), Compress::No, Validate::Yes)
        .expect_err("historical Q16 tag 3 must be rejected");
    assert!(matches!(err, SerializationError::InvalidData(_)));
}

#[test]
fn setup_section_rejects_mismatched_zk_protocol_feature() {
    let mut descriptor = sample_descriptor();
    descriptor.setup.protocol_features.zk = true;
    let err = descriptor
        .check()
        .expect_err("zk=true must be rejected on transparent build");
    assert!(matches!(err, SerializationError::InvalidData(_)));
    assert!(
        err.to_string().contains("protocol features"),
        "unexpected error: {err}"
    );
}

#[test]
fn descriptor_deserialize_rejects_zk_protocol_feature() {
    let mut descriptor = sample_descriptor();
    descriptor.setup.protocol_features.zk = true;
    let bytes = descriptor.canonical_bytes().expect("serialize");
    let err = AkitaInstanceDescriptor::deserialize_uncompressed(&bytes[..], &())
        .expect_err("zk=true wire must be rejected on transparent build");
    assert!(matches!(err, SerializationError::InvalidData(_)));
}

#[test]
fn fold_linf_descriptor_canonical_digest_pinned() {
    let bytes = sample_descriptor()
        .canonical_bytes()
        .expect("serialize descriptor");
    assert_eq!(
        (bytes.len(), blake2b_256(&bytes)),
        (
            221,
            [
                0x41, 0x91, 0x21, 0x5c, 0x0b, 0x08, 0xe4, 0xac, 0x7c, 0xf0, 0xc5, 0xae, 0x02, 0x4e,
                0xf5, 0xe4, 0x3d, 0x6f, 0x5b, 0x8e, 0x75, 0x6d, 0xaf, 0x08, 0xff, 0x7c, 0x10, 0x37,
                0xc1, 0x71, 0x62, 0x54,
            ]
        ),
        "update pinned digest when descriptor setup-section bindings change"
    );
}

#[test]
fn fold_linf_binding_is_part_of_setup_section() {
    let descriptor = sample_descriptor();
    assert_eq!(descriptor.setup.fold_linf, FoldLinfProtocolBinding::CURRENT);
    let mut altered = descriptor.clone();
    altered.setup.fold_linf.formula_tag = 0;
    assert_ne!(
        altered.canonical_bytes().expect("serialize"),
        descriptor.canonical_bytes().expect("serialize")
    );
}

#[test]
fn effective_schedule_digest_binds_tail_bound_with_grind_policy() {
    let certified = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        3,
        2,
        4,
        3,
        SparseChallengeConfig::ExactShell {
            count_mag1: 30,
            count_mag2: 12,
        },
    )
    .with_decomp(4, 2, 2, 2, 0)
    .expect("certified params");
    let deterministic = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        3,
        2,
        4,
        3,
        SparseChallengeConfig::BoundedL1Norm,
    )
    .with_decomp(4, 2, 2, 2, 0)
    .expect("deterministic params");
    assert_eq!(
        certified.fold_witness_linf_cap_policy(),
        crate::sis::FoldWitnessLinfCapPolicy::TailBoundWithGrind
    );
    assert_eq!(
        deterministic.fold_witness_linf_cap_policy(),
        crate::sis::FoldWitnessLinfCapPolicy::WorstCaseBetaOnly
    );

    let schedule_certified = Schedule {
        steps: vec![Step::Fold(FoldStep {
            params: certified,
            current_w_len: 256,
            next_w_len: 256,
            level_bytes: 123,
        })],
        total_bytes: 123,
    };
    let schedule_deterministic = Schedule {
        steps: vec![Step::Fold(FoldStep {
            params: deterministic,
            current_w_len: 256,
            next_w_len: 256,
            level_bytes: 123,
        })],
        total_bytes: 123,
    };

    assert_ne!(
        digest_effective_schedule(&schedule_certified),
        digest_effective_schedule(&schedule_deterministic)
    );
}

#[test]
fn effective_schedule_digest_binds_shape_aware_challenge_l2_sq_max() {
    let flat = sample_level_params();
    let mut tensor = sample_level_params();
    tensor.fold_challenge_shape = TensorChallengeShape::Tensor;
    assert_ne!(flat.challenge_l2_sq_max(), tensor.challenge_l2_sq_max());

    let schedule_flat = Schedule {
        steps: vec![Step::Fold(FoldStep {
            params: flat,
            current_w_len: 256,
            next_w_len: 256,
            level_bytes: 123,
        })],
        total_bytes: 123,
    };
    let schedule_tensor = Schedule {
        steps: vec![Step::Fold(FoldStep {
            params: tensor,
            current_w_len: 256,
            next_w_len: 256,
            level_bytes: 123,
        })],
        total_bytes: 123,
    };

    assert_ne!(
        digest_effective_schedule(&schedule_flat),
        digest_effective_schedule(&schedule_tensor)
    );
}

#[test]
fn effective_schedule_digest_binds_fold_linf_policy() {
    let mut tensor_params = sample_level_params();
    tensor_params.fold_challenge_shape = TensorChallengeShape::Tensor;

    let schedule_flat = Schedule {
        steps: vec![Step::Fold(FoldStep {
            params: sample_level_params(),
            current_w_len: 256,
            next_w_len: 256,
            level_bytes: 123,
        })],
        total_bytes: 123,
    };
    let schedule_tensor = Schedule {
        steps: vec![Step::Fold(FoldStep {
            params: tensor_params,
            current_w_len: 256,
            next_w_len: 256,
            level_bytes: 123,
        })],
        total_bytes: 123,
    };

    assert_ne!(
        digest_effective_schedule(&schedule_flat),
        digest_effective_schedule(&schedule_tensor)
    );
}

#[test]
fn canonical_encoding_roundtrip() {
    let descriptor = sample_descriptor();
    let bytes = descriptor.canonical_bytes().expect("serialize descriptor");
    assert_eq!(bytes.len(), descriptor.uncompressed_size());

    let decoded = AkitaInstanceDescriptor::deserialize_uncompressed(&bytes[..], &())
        .expect("deserialize descriptor");
    assert_eq!(decoded, descriptor);
}

#[test]
fn descriptor_rejects_stale_schema_version() {
    let mut descriptor = sample_descriptor();
    descriptor.version = AKITA_INSTANCE_DESCRIPTOR_VERSION - 1;

    let err = descriptor
        .check()
        .expect_err("stale descriptor versions must be rejected");
    assert!(err
        .to_string()
        .contains("unsupported Akita instance descriptor version"));
}

#[test]
fn algebra_section_binds_prime_and_extension_shape() {
    let fp32 =
        AlgebraSection::for_fields::<Prime32Offset99, Prime32Offset99, 32>().expect("fp32 algebra");
    let fp64 =
        AlgebraSection::for_fields::<Prime64Offset59, Prime64Offset59, 32>().expect("fp64 algebra");

    assert_ne!(fp32.prime_modulus_be, fp64.prime_modulus_be);
    assert_eq!(fp32.ring_dimension_d, 32);
    assert_eq!(fp32.field_extension_degree, 1);
    assert_eq!(fp32.extension_degree, 1);
}

#[test]
fn opening_batch_digest_binds_claim_count() {
    let left = OpeningClaimsLayout::new(4, 2).expect("left");
    let right = OpeningClaimsLayout::new(4, 3).expect("right");

    assert_ne!(left.opening_batch_digest(), right.opening_batch_digest());
}

#[test]
fn opening_batch_digest_binds_group_partition() {
    let grouped = OpeningClaimsLayout::from_group_sizes(4, &[1, 2]).expect("grouped");
    let scalar = OpeningClaimsLayout::new(4, 3).expect("scalar");

    assert_ne!(
        grouped.opening_batch_digest(),
        scalar.opening_batch_digest()
    );
}

#[test]
fn opening_batch_digest_binds_group_active_vars() {
    let two_vars =
        OpeningClaimsLayout::from_groups(vec![PolynomialGroupLayout::new(2, 1)]).expect("two vars");
    let three_vars = OpeningClaimsLayout::from_groups(vec![PolynomialGroupLayout::new(3, 1)])
        .expect("three vars");

    assert_ne!(
        two_vars.opening_batch_digest(),
        three_vars.opening_batch_digest()
    );
}

#[test]
fn call_section_exposes_group_partition() {
    let opening_batch = OpeningClaimsLayout::from_group_sizes(4, &[1, 2]).expect("grouped");
    let call = CallSection::from_layout(&opening_batch, BasisMode::Lagrange).expect("call");

    assert_eq!(call.num_polys, 3);
    assert_eq!(call.num_commitment_groups, 2);
    assert_eq!(call.num_polys_per_commitment_group, vec![1, 2]);
    assert_eq!(call.point_variable_selections, vec![vec![0, 1, 2, 3]; 2]);
}

#[test]
fn descriptor_digest_uses_standard_blake2b_256() {
    assert_eq!(
        blake2b_256(b"akita"),
        [
            0x38, 0x68, 0x5d, 0xd7, 0x90, 0xe7, 0xb2, 0x82, 0xd5, 0xeb, 0x4f, 0xa7, 0x00, 0x37,
            0xde, 0x42, 0x71, 0x42, 0xc4, 0x8e, 0x44, 0x1b, 0x96, 0x0f, 0x2e, 0x09, 0xde, 0x98,
            0xbb, 0x8f, 0x69, 0x54,
        ]
    );
}

#[test]
fn setup_seed_digest_matches_setup_section() {
    let seed = AkitaSetupSeed {
        max_num_vars: 5,
        max_num_batched_polys: 2,
        gen_ring_dim: 4,
        max_setup_len: 2,
        public_matrix_seed: [7; 32],
    };
    let section = SetupSection::from_parts(
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: Some(32),
        },
        SisModulusFamily::Q32,
        &seed,
    )
    .expect("direct setup section");

    assert_eq!(
        section.setup_seed_digest,
        setup_seed_digest(&seed).expect("setup seed digest")
    );
}

#[test]
fn effective_schedule_digest_binds_direct_shape() {
    let schedule_a = Schedule {
        steps: vec![Step::Direct(crate::DirectStep {
            current_w_len: 8,
            witness_shape: CleartextWitnessShape::FieldElements(8),
            direct_bytes: 8,
            params: None,
        })],
        total_bytes: 8,
    };
    let schedule_b = Schedule {
        steps: vec![Step::Direct(crate::DirectStep {
            current_w_len: 8,
            witness_shape: CleartextWitnessShape::FieldElements(9),
            direct_bytes: 9,
            params: None,
        })],
        total_bytes: 9,
    };

    assert_ne!(
        digest_effective_schedule(&schedule_a),
        digest_effective_schedule(&schedule_b)
    );
}

#[test]
fn effective_schedule_digest_binds_root_direct_commit_params() {
    // Two root-direct schedules with identical witness shape but
    // different commit `params` must hash to different preamble bytes.
    // This is the binding the dropped `SetupSection::level_params_digest`
    // used to provide; it now lives in the per-proof schedule digest.
    let mut other_params = sample_level_params();
    other_params.num_blocks += 1;

    let schedule_a = Schedule {
        steps: vec![Step::Direct(crate::DirectStep {
            current_w_len: 8,
            witness_shape: CleartextWitnessShape::FieldElements(8),
            direct_bytes: 0,
            params: Some(sample_level_params()),
        })],
        total_bytes: 0,
    };
    let schedule_b = Schedule {
        steps: vec![Step::Direct(crate::DirectStep {
            current_w_len: 8,
            witness_shape: CleartextWitnessShape::FieldElements(8),
            direct_bytes: 0,
            params: Some(other_params),
        })],
        total_bytes: 0,
    };

    assert_ne!(
        digest_effective_schedule(&schedule_a),
        digest_effective_schedule(&schedule_b)
    );
}
