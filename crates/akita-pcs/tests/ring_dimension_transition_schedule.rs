//! Schedule-level coverage for per-matrix ring dimensions and transitions.

#![allow(missing_docs)]

mod support;

use akita_config::{policy_of, proof_optimized::fp128, CommitmentConfig};
use akita_types::sis::{decomposed_t_ring_count, decomposed_w_ring_count};
use akita_types::{
    setup_matrix_capacity_for_schedule, AkitaScheduleLookupKey, CommitmentRingDims,
    CommittedGroupParams, FoldSchedule, OpeningClaimsLayout, PolynomialGroupLayout, SisTableDigest,
    WitnessPartition,
};
use support::{
    mixed_d_per_level_schedule, per_matrix_ring_dims_root_schedule,
    ring_dimension_transition_schedule, PerMatrixRingDimsRootConfig,
    RecursiveRingDimensionTransitionConfig, ThreeBandRingDimensionTransitionConfig,
};

const NUM_VARS: usize = 36;

fn assert_exact_matrix_widths(params: &CommittedGroupParams, num_polynomials: usize) {
    let dims = params.role_dims();
    let native_outer = decomposed_t_ring_count(
        params.inner_commit_matrix.output_rank(),
        params.num_digits_outer,
        params.num_live_blocks,
        num_polynomials,
    )
    .expect("native B width");
    let native_open = decomposed_w_ring_count(
        params.num_digits_open,
        params.num_live_blocks,
        num_polynomials,
    )
    .expect("native D width");
    assert_eq!(
        params.outer_commit_matrix.input_width(),
        native_outer * (dims.d_a() / dims.d_b())
    );
    assert_eq!(
        params.open_commit_matrix.input_width(),
        native_open * (dims.d_a() / dims.d_d())
    );
}

fn assert_suffix_matches_plan<Cfg: CommitmentConfig>(
    schedule: &FoldSchedule,
    first_actual_fold: usize,
    start_level: usize,
    start_witness_len: usize,
    start_log_basis: u32,
) {
    let payload_phase = schedule.recursive_folds[..first_actual_fold].iter().fold(
        akita_types::CommitmentPayloadPhase::CompressedPrefix,
        |phase, fold| phase.after(fold.params.witness.payload_mode),
    );
    let suffix = akita_planner::plan_optimal_suffix(
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
        NUM_VARS,
        akita_planner::SuffixPlanStart {
            level: start_level,
            witness_len: start_witness_len,
            log_basis: start_log_basis,
            payload_phase,
        },
    )
    .expect("independent suffix plan");
    let actual = &schedule.recursive_folds[first_actual_fold..];
    assert_eq!(actual.len(), suffix.folds.len());
    for (step, planned) in actual.iter().zip(&suffix.folds) {
        assert_eq!(step.params.witness, planned.params);
        assert_eq!(step.input_witness_len, planned.input_witness_len);
        assert_eq!(step.output_witness_len, planned.output_witness_len);
    }
    assert_eq!(schedule.terminal.params.witness, suffix.terminal.params);
    assert_eq!(
        schedule.terminal.input_witness_len,
        suffix.terminal.input_witness_len
    );
    assert_eq!(
        schedule.terminal.params.response_shape,
        suffix.terminal.response_shape
    );
}

fn three_band_ring_dimension_schedule<Root>(expected_root_d: usize) -> FoldSchedule
where
    Root: akita_config::CommitmentConfig<Field = fp128::Field, ExtField = fp128::Field>,
{
    let schedule = ring_dimension_transition_schedule::<Root, fp128::D128OneHot, fp128::D64OneHot>(
        NUM_VARS,
        1,
        CommitmentRingDims {
            inner: expected_root_d,
            outer: 128,
            opening: 128,
        },
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        },
    )
    .expect("tableless three-band ring-dimension schedule");

    assert_eq!(
        schedule.root.params.final_group.commitment.role_dims(),
        CommitmentRingDims {
            inner: expected_root_d,
            outer: 128,
            opening: 128,
        }
    );
    assert_exact_matrix_widths(&schedule.root.params.final_group.commitment, 1);
    assert_eq!(
        schedule.recursive_folds[0].params.witness.role_dims(),
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        }
    );
    assert_exact_matrix_widths(&schedule.recursive_folds[0].params.witness, 1);
    assert_eq!(
        schedule.recursive_folds[1].params.witness.role_dims(),
        CommitmentRingDims::uniform(64)
    );
    let l1 = &schedule.recursive_folds[0];
    assert_suffix_matches_plan::<fp128::D64OneHot>(
        &schedule,
        1,
        2,
        l1.output_witness_len,
        l1.params.witness.log_basis_open,
    );
    schedule
}

#[test]
fn tableless_d256_root_uses_offline_planner() {
    let _schedule = three_band_ring_dimension_schedule::<fp128::D256OneHot>(256);
}

#[test]
fn d512_root_uses_additive_a_matrix_sis_table() {
    let schedule = three_band_ring_dimension_schedule::<fp128::D512OneHot>(512);

    assert_eq!(
        schedule
            .root
            .params
            .final_group
            .commitment
            .inner_commit_matrix
            .sis_table_key()
            .table_digest,
        SisTableDigest::Q128_INNER_D512
    );
    let root = &schedule.root.params.final_group.commitment;
    assert_eq!(
        root.flat_field_len().expect("root flat length"),
        1usize << NUM_VARS
    );
    let required = setup_matrix_capacity_for_schedule(&schedule).expect("schedule envelope");
    let configured = <ThreeBandRingDimensionTransitionConfig<
        fp128::D512OneHot,
        fp128::D128OneHot,
        fp128::D64OneHot,
        128,
        64,
    > as CommitmentConfig>::setup_matrix_capacity(NUM_VARS, 1)
    .expect("configured envelope");
    assert_eq!(configured, required);
}

#[test]
fn per_matrix_ring_dims_root_replans_its_complete_suffix() {
    let schedule = per_matrix_ring_dims_root_schedule::<fp128::D128OneHot>(NUM_VARS, 1, 64, 128)
        .expect("per-matrix ring-dimension root schedule");
    let root = &schedule.root.params.final_group.commitment;
    assert_eq!(
        root.role_dims(),
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 128,
        }
    );
    assert_exact_matrix_widths(root, 1);
    assert_suffix_matches_plan::<fp128::D128OneHot>(
        &schedule,
        0,
        1,
        schedule.root.output_witness_len,
        root.log_basis_open,
    );

    let required = setup_matrix_capacity_for_schedule(&schedule).expect("schedule envelope");
    let configured =
        <PerMatrixRingDimsRootConfig<fp128::D128OneHot, 64, 128> as CommitmentConfig>::
            setup_matrix_capacity(NUM_VARS, 1)
            .expect("configured envelope");
    assert_eq!(configured, required);
}

#[test]
fn mixed_d_splice_after_raw_prefix_cannot_resume_compression() {
    let policy = policy_of::<fp128::D64OneHot>();
    let envelope = akita_planner::find_schedule(
        &AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(32)),
        fp128::D64OneHot::root_honest_fold_policy(),
        &[],
        &policy,
        fp128::D64OneHot::ring_challenge_config,
        fp128::D64OneHot::fold_challenge_shape_at_level,
    )
    .expect("D64 envelope schedule")
    .schedule;
    let raw_index = envelope
        .recursive_folds
        .iter()
        .enumerate()
        .find(|(index, fold)| {
            fold.params.witness.payload_mode == akita_types::CommitmentPayloadMode::Raw
                && index + 1 < envelope.recursive_folds.len()
        })
        .map(|(index, _)| index)
        .expect("fixture must have a raw fold with a recursive successor");
    let switch_at_fold = raw_index + 2;

    let spliced =
        mixed_d_per_level_schedule::<fp128::D64OneHot, fp128::D64OneHot>(32, 1, switch_at_fold)
            .expect("raw-prefix splice must remain a valid monotone schedule");

    assert_eq!(
        spliced.recursive_folds[raw_index]
            .params
            .witness
            .payload_mode,
        akita_types::CommitmentPayloadMode::Raw
    );
    assert!(spliced.recursive_folds[raw_index + 1..]
        .iter()
        .all(|fold| fold.params.witness.payload_mode == akita_types::CommitmentPayloadMode::Raw));
}

fn recursive_transition_schedule<ChunkCfg>() -> FoldSchedule
where
    ChunkCfg: CommitmentConfig<Field = fp128::Field, ExtField = fp128::Field>,
{
    type Root = fp128::D256OneHot;
    type Mid = fp128::D128OneHot;
    type Suffix = fp128::D64OneHot;
    type Cfg<C> = RecursiveRingDimensionTransitionConfig<Root, Mid, Suffix, C, 128, 64>;

    let layout = OpeningClaimsLayout::from_root_groups(
        &[
            PolynomialGroupLayout::new(16, 1),
            PolynomialGroupLayout::new(16, 1),
        ],
        PolynomialGroupLayout::new(32, 2),
    )
    .expect("recursive profile layout");
    let schedule = Cfg::<ChunkCfg>::get_params_for_prove(&layout)
        .expect("recursive mixed-D transition schedule");
    schedule.validate_structure().expect("valid transition");

    let root = &schedule.root.params.final_group.commitment;
    assert_eq!(
        root.role_dims(),
        CommitmentRingDims {
            inner: 256,
            outer: 128,
            opening: 128,
        }
    );
    assert!(root.precommitted_groups.iter().all(|group| {
        group.role_dims(128)
            == CommitmentRingDims {
                inner: 256,
                outer: 128,
                opening: 128,
            }
    }));

    let l1 = &schedule.recursive_folds[0];
    assert_eq!(
        l1.params.witness.role_dims(),
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        }
    );
    let prefix = l1
        .params
        .incoming_setup_prefix
        .as_ref()
        .expect("root setup prefix");
    assert_eq!(prefix.d_setup(), 64);
    assert_eq!(
        prefix
            .commitment_params
            .layout
            .inner_commit_matrix
            .ring_dimension(),
        64
    );
    assert_eq!(
        prefix
            .commitment_params
            .layout
            .outer_commit_matrix
            .ring_dimension(),
        64
    );
    assert_eq!(
        schedule.recursive_folds[1].params.witness.role_dims(),
        CommitmentRingDims::uniform(64)
    );
    schedule
}

#[test]
fn recursive_transition_supports_ci_and_profile_setup_capacities() {
    type Root = fp128::D256OneHot;
    type Mid = fp128::D128OneHot;
    type Suffix = fp128::D64OneHot;
    type Cfg = RecursiveRingDimensionTransitionConfig<Root, Mid, Suffix, Suffix, 128, 64>;

    let ci = <Cfg as CommitmentConfig>::setup_matrix_capacity(24, 3)
        .expect("CI fixture envelope (24,3)");
    let profile = <Cfg as CommitmentConfig>::setup_matrix_capacity(32, 4)
        .expect("profile fixture envelope (32,4)");
    assert!(
        profile.num_field_elements >= ci.num_field_elements,
        "profile capacity must cover at least the CI fixture envelope"
    );
    assert!(
        <Cfg as CommitmentConfig>::setup_matrix_capacity(24, 4).is_err(),
        "unsupported capacities must stay fail-closed"
    );
}

#[test]
fn recursive_transition_supports_independent_d64_setup_prefix() {
    let schedule = recursive_transition_schedule::<fp128::D64OneHot>();
    assert_eq!(
        schedule.recursive_folds[0].params.witness_partition,
        WitnessPartition::Single
    );
}

#[test]
fn recursive_transition_preserves_w8r2_middle_partition() {
    let schedule = recursive_transition_schedule::<fp128::D64OneHotMultiChunk>();
    assert_eq!(
        schedule.recursive_folds[0].params.witness_partition,
        WitnessPartition::Distributed { num_chunks: 8 }
    );
}
