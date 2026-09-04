use super::*;
use akita_challenges::SparseChallengeConfig;
use akita_config::proof_optimized::{fp128, fp32};
use akita_types::sis::sis_table_key_for_linf_bound;
use akita_types::{
    derive_public_matrix_prefix, sample_akita_setup_seed, scheduled_setup_prefix,
    CompressionChainPlan, GroupCommitPhaseParams, GroupOpenPhaseParams, GroupOpeningPlan,
    InnerCommitMatrixParams, OuterCommitMatrixParams, PolynomialGroupLayout, RingVec,
    SetupPrefixPublicCommitment, SetupPrefixVerifierSlot, SisMatrixRole, SisModulusProfileId,
    SisTableDigest, SisTableKey, DEFAULT_SIS_SECURITY_POLICY,
};
use jolt_field::Zero;

type TestCfg = fp128::OneHot;
type TestF = fp128::Field;
const TEST_D: usize = 256;
const PREFIX_D: usize = 64;

fn schedules<Cfg: CommitmentConfig>() -> TrustedScheduleCatalog {
    akita_config::test_support::workspace_schedule_catalog::<Cfg>()
        .expect("workspace trusted schedule catalog")
}

fn blob_prefix() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&BLOB_MAGIC);
    bytes.push(AkitaJoltCase::OneHotFp128Direct.tag());
    (TEST_D as u64)
        .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
        .unwrap();
    bytes
}

#[test]
fn full_catalog_frame_round_trips_without_compiled_rows() {
    let catalog = schedules::<TestCfg>();
    let inner = blob_prefix();
    let framed = frame_with_schedule_catalog::<TestCfg>(&inner, &catalog).expect("catalog frame");

    assert_eq!(
        read_blob_case(&framed).expect("framed case"),
        AkitaJoltCase::OneHotFp128Direct
    );
    let (decoded, decoded_inner) =
        split_schedule_catalog::<TestCfg>(&framed).expect("split catalog frame");
    assert_eq!(decoded.catalog_digest(), catalog.catalog_digest());
    assert_eq!(decoded_inner, inner);
}

#[test]
fn catalog_frame_rejects_missing_truncated_and_tampered_artifacts() {
    let inner = blob_prefix();
    let missing = split_schedule_catalog::<TestCfg>(&inner)
        .expect_err("guest entry requires the benchmark catalog frame");
    assert!(missing.to_string().contains("missing"));

    let mut truncated = CATALOG_FRAME_MAGIC.to_vec();
    truncated.extend_from_slice(&1u64.to_le_bytes());
    let truncated = split_schedule_catalog::<TestCfg>(&truncated)
        .expect_err("catalog frame must contain an inner blob");
    assert!(truncated.to_string().contains("complete inner blob"));

    let catalog = schedules::<TestCfg>();
    let mut tampered =
        frame_with_schedule_catalog::<TestCfg>(&inner, &catalog).expect("catalog frame");
    let artifact_start = CATALOG_FRAME_HEADER_BYTES;
    let marker = b"\"policy_digest\"";
    let mut digest_start = tampered[artifact_start..]
        .windows(marker.len())
        .position(|window| window == marker)
        .map(|offset| artifact_start + offset + marker.len())
        .expect("policy digest key");
    while tampered[digest_start].is_ascii_whitespace() {
        digest_start += 1;
    }
    assert_eq!(tampered[digest_start], b':');
    digest_start += 1;
    while !tampered[digest_start].is_ascii_digit() {
        digest_start += 1;
    }
    let digit = tampered[digest_start];
    tampered[digest_start] = if digit == b'9' { b'8' } else { digit + 1 };
    let tampered = split_schedule_catalog::<TestCfg>(&tampered)
        .expect_err("tampered catalog identity must reject");
    assert!(tampered.to_string().contains("policy"));
}

fn prefix_commitment_params() -> GroupOpenPhaseParams {
    let inner_key = sis_table_key_for_linf_bound(
        DEFAULT_SIS_SECURITY_POLICY,
        SisTableDigest::CURRENT,
        SisModulusProfileId::Q128OffsetA7F7,
        SisMatrixRole::Inner,
        u32::try_from(PREFIX_D).expect("test prefix ring dimension"),
        32_767,
    )
    .expect("audited prefix A bound");
    let inner_commit_matrix = InnerCommitMatrixParams::try_new_with_min_rank(inner_key, 1)
        .expect("audited prefix A matrix");
    let outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
        SisTableKey {
            policy: DEFAULT_SIS_SECURITY_POLICY,
            table_digest: SisTableDigest::CURRENT,
            modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
            role: SisMatrixRole::Outer,
            ring_dimension: u32::try_from(PREFIX_D).expect("test prefix ring dimension"),
            coeff_linf_bound: 3,
        },
        inner_commit_matrix.output_rank(),
    )
    .expect("audited prefix B matrix");
    GroupOpenPhaseParams {
        setup_natural_len: None,
        profile: GroupCommitPhaseParams {
            version: GroupCommitPhaseParams::VERSION,
            group: PolynomialGroupLayout::singleton(PREFIX_D.trailing_zeros() as usize),
            blocks: akita_types::BlockGeometry::new(1, 1, 1),
            outer_slice_count: akita_types::CommitmentSliceCount::ONE,
            inner: akita_types::RoleParams::new(
                akita_types::GadgetDigits::new(1, 1),
                inner_commit_matrix,
            ),
            outer: akita_types::RoleParams::new(
                akita_types::GadgetDigits::new(1, 1),
                outer_commit_matrix,
            ),
        },
        opening: GroupOpeningPlan::evaluation_trace(SparseChallengeConfig::pm1_only(0), 1, 1, 1),
    }
}

#[test]
fn trailing_blob_bytes_are_rejected() {
    let err = reject_trailing_bytes(&[0]).unwrap_err();
    assert!(err.to_string().contains("trailing bytes"));
    reject_trailing_bytes(&[]).unwrap();
}

#[test]
fn previous_blob_version_is_rejected_at_the_magic_boundary() {
    let mut bytes = blob_prefix();
    bytes[..BLOB_MAGIC.len()].copy_from_slice(b"AKJOLTv4");
    let error = AkitaJoltInputs::<TestF, TEST_D>::read_from_bytes::<TestCfg>(
        &bytes,
        &schedules::<TestCfg>(),
    )
    .expect_err("v4 blob must not reach payload decoding");
    assert!(error.to_string().contains("magic mismatch"));
}

#[test]
fn setup_matrix_padding_record_is_canonical_and_zeroed() {
    let bytes =
        [u8::try_from(SETUP_MATRIX_MAX_PADDING_BYTES + 1).expect("test padding count fits u8")];
    let mut rest = &bytes[..];
    let error =
        AkitaJoltInputs::<TestF, TEST_D>::decode_setup_matrix_padding(&mut rest, bytes.len())
            .expect_err("padding count must be bounded");
    assert!(error.to_string().contains("exceeds"));

    let unpadded_blob_len = 120;
    let padding_record_offset = 6;
    let noncanonical_padding = 7;
    let total_blob_len = unpadded_blob_len + 1 + noncanonical_padding;
    let mut bytes = vec![0u8; total_blob_len];
    bytes[padding_record_offset] = noncanonical_padding as u8;
    let mut rest = &bytes[padding_record_offset..];
    let error =
        AkitaJoltInputs::<TestF, TEST_D>::decode_setup_matrix_padding(&mut rest, total_blob_len)
            .expect_err("the smallest valid padding is canonical");
    assert!(error.to_string().contains("does not match expected 0"));

    let padding = setup_matrix_padding(200, 0).expect("alignment padding");
    assert!(padding > 0);
    let total_blob_len = 200 + 1 + padding;
    let mut bytes = vec![0u8; total_blob_len];
    bytes[0] = padding as u8;
    bytes[1] = 1;
    let mut rest = &bytes[..];
    let error =
        AkitaJoltInputs::<TestF, TEST_D>::decode_setup_matrix_padding(&mut rest, total_blob_len)
            .expect_err("padding bytes must be zero");
    assert!(error.to_string().contains("must be zero"));
}

#[test]
fn transcript_domain_len_is_capped_before_allocation() {
    let mut bytes = blob_prefix();
    ((MAX_TRANSCRIPT_DOMAIN_BYTES + 1) as u64)
        .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
        .unwrap();

    let err = AkitaJoltInputs::<TestF, TEST_D>::read_from_bytes::<TestCfg>(
        &bytes,
        &schedules::<TestCfg>(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("length"));
}

#[test]
fn num_vars_is_capped_before_opening_point_allocation() {
    let mut bytes = blob_prefix();
    Vec::<u8>::new()
        .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
        .unwrap();
    ((MAX_BLOB_NUM_VARS + 1) as u64)
        .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
        .unwrap();

    let err = AkitaJoltInputs::<TestF, TEST_D>::read_from_bytes::<TestCfg>(
        &bytes,
        &schedules::<TestCfg>(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("length"));
}

#[test]
fn opening_point_len_must_match_num_vars_before_allocation() {
    let mut bytes = blob_prefix();
    Vec::<u8>::new()
        .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
        .unwrap();
    2u64.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();
    3u64.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();

    let err = AkitaJoltInputs::<TestF, TEST_D>::read_from_bytes::<TestCfg>(
        &bytes,
        &schedules::<TestCfg>(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("opening-point arity 3"));
}

#[test]
fn strict_setup_decoder_preserves_prefix_slots() {
    let setup_seed = sample_akita_setup_seed();
    let seed = AkitaSetupDescriptor {
        max_num_vars: 8,
        max_num_batched_polys: 1,
        num_field_elements: 2 * TEST_D,
        setup_seed: setup_seed.clone(),
    };
    let shared_matrix = derive_public_matrix_prefix::<TestF>(2 * TEST_D, &setup_seed);
    let commitment_params = prefix_commitment_params();
    let matrix = &commitment_params.profile.outer.matrix;
    let payload_coefficients = CompressionChainPlan::for_complete_source(
        matrix.sis_modulus_profile(),
        matrix.output_rank() * matrix.ring_dimension(),
    )
    .expect("setup-prefix compression plan")
    .terminal_coefficients();
    let id = scheduled_setup_prefix(PREFIX_D, commitment_params)
        .slot_id()
        .expect("setup prefix group");
    let mut prefix_slots = SetupPrefixVerifierRegistry::new(setup_seed.clone());
    prefix_slots
        .insert(SetupPrefixVerifierSlot {
            id: id.clone(),
            commitment: SetupPrefixPublicCommitment {
                rows: vec![RingVec::from_coeffs(vec![
                    TestF::zero();
                    payload_coefficients
                ])],
            },
        })
        .expect("insert prefix slot");

    let mut bytes = Vec::new();
    seed.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();
    let padding_record_offset = bytes.len();
    let unpadded_blob_len = checked::sum([
        padding_record_offset,
        shared_matrix.serialized_size(BLOB_COMPRESS),
        prefix_slots.serialized_size(BLOB_COMPRESS),
    ])
    .expect("test setup size");
    let padding = setup_matrix_padding(unpadded_blob_len, padding_record_offset)
        .expect("test setup alignment");
    bytes.push(padding as u8);
    bytes.resize(bytes.len() + padding, 0);
    shared_matrix
        .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
        .unwrap();
    prefix_slots
        .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
        .unwrap();

    let mut rest = &bytes[..];
    let total_blob_len = bytes.len();
    let decoded =
        AkitaJoltInputs::<TestF, TEST_D>::deserialize_strict_host_setup(&mut rest, total_blob_len)
            .expect("decode setup");

    assert!(rest.is_empty());
    assert!(decoded.prefix_slots.get(&id).is_some());
    assert_eq!(decoded.prefix_slots.len(), 1);
}

#[test]
fn setup_matrix_payload_must_fit_remaining_blob_before_allocation() {
    let err =
        AkitaJoltInputs::<TestF, TEST_D>::check_setup_matrix_bytes_available(&[], 1).unwrap_err();
    assert!(err.to_string().contains("setup matrix claims"));
}

#[test]
fn proof_shape_budget_and_schedule_identity_precede_proof_allocation() {
    let schedules = schedules::<TestCfg>();
    let opening_claims = akita_types::OpeningClaimsLayout::new(14, 1).expect("opening layout");
    let row = schedules
        .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
            opening_claims
                .root_final_group_layout()
                .expect("singleton group layout"),
        ))
        .expect("trusted singleton row");
    let opening_layout = row.profiles().opening_layout().expect("opening layout");
    let grinding_plan = derive_transcript_grinding_plan::<TestCfg>(row.schedule(), &opening_layout)
        .expect("grinding plan");
    let canonical = canonical_proof_shape(row.schedule(), &opening_layout, 1, &grinding_plan)
        .expect("canonical shape");

    let mut huge = canonical.clone();
    huge.root.opening_payload_coeffs = usize::MAX;
    let budget_error = AkitaJoltInputs::<TestF, TEST_D>::validate_proof_shape_before_allocation::<
        TestCfg,
    >(row.selection(), &huge, 0, &schedules)
    .expect_err("huge shape must fail against remaining bytes");
    let budget_message = budget_error.to_string();
    assert!(
        budget_message.contains("remaining proof bytes") || budget_message.contains("overflow"),
        "unexpected budget error: {budget_message}"
    );

    let mut noncanonical = canonical;
    noncanonical.root.opening_payload_coeffs += 1;
    let identity_error =
        AkitaJoltInputs::<TestF, TEST_D>::validate_proof_shape_before_allocation::<TestCfg>(
            row.selection(),
            &noncanonical,
            MAX_JOLT_BLOB_BYTES as usize,
            &schedules,
        )
        .expect_err("noncanonical shape must fail before proof decoding");
    assert!(identity_error.to_string().contains("canonical schedule"));
}

#[test]
fn extension_proof_shape_must_match_the_selected_schedule_before_allocation() {
    type ExtCfg = fp32::OneHot;
    type ExtF = fp32::Field;
    type ExtE = <ExtCfg as CommitmentConfig>::ExtField;

    let layout = akita_types::OpeningClaimsLayout::new(30, 1).expect("opening layout");
    let schedules = schedules::<ExtCfg>();
    let row = schedules
        .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
            layout
                .root_final_group_layout()
                .expect("singleton group layout"),
        ))
        .expect("trusted fp32 singleton row");
    let opening_layout = row.profiles().opening_layout().expect("catalog layout");
    let grinding_plan = derive_transcript_grinding_plan::<ExtCfg>(row.schedule(), &opening_layout)
        .expect("grinding plan");
    let mut noncanonical = canonical_proof_shape(
        row.schedule(),
        &opening_layout,
        <ExtE as ExtField<ExtF>>::DEGREE,
        &grinding_plan,
    )
    .expect("canonical extension shape");
    noncanonical
        .terminal
        .extension_opening_reduction
        .as_mut()
        .expect("proper extension claims require a terminal reduction shape")
        .partials += 1;

    let error =
        AkitaJoltInputs::<ExtF, 2048, ExtE>::validate_proof_shape_before_allocation::<ExtCfg>(
            row.selection(),
            &noncanonical,
            MAX_JOLT_BLOB_BYTES as usize,
            &schedules,
        )
        .expect_err("noncanonical extension shape must fail before proof decoding");
    assert!(error.to_string().contains("canonical schedule"));
}
