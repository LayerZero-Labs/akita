#![cfg(feature = "labinius")]

mod common;

use akita_algebra::binary::BinaryField128;
use akita_config::{
    ensure_prover_schedule_fits_setup, policy_of, required_setup_prefix_slot_ids_for_schedule,
    CommitmentConfig, SetupRequirements, TrustedScheduleCatalog, ValidatedScheduleCatalog,
};
use akita_cpu_backend::{AkitaProverSetup, CpuBackend};
use akita_error::AkitaError;
use akita_labinius_pcs::{
    DigitConfig, Digits2, ImageConfig, ImageEvaluation, ImageProver, ImageVerifier, PreparedMatrix,
    RootPcsProver, RootPcsVerifier, RootSetup, F,
};
use akita_labinius_verifier::{channel::ClearChannel, lowered::LoweredRootLayout};
use akita_params::{CommittedGroupBatchProfile, GroupCommitPhaseParams, PolynomialGroupLayout};
use akita_serialization::Valid;
use akita_types::{AkitaVerifierSetup, CommittedGroup};
use common::Recording;
use jolt_field::Zero;
use std::sync::OnceLock;

struct Fixture {
    root: RootSetup,
    layout: LoweredRootLayout,
    images: TrustedScheduleCatalog<ImageConfig>,
    prover_setup: AkitaProverSetup<F>,
    verifier_setup: AkitaVerifierSetup<F>,
    complete_verifier_setup: AkitaVerifierSetup<F>,
    image: CommittedGroup<F>,
    proof: Vec<u8>,
}

fn fixture() -> &'static Fixture {
    static IMAGE: OnceLock<Fixture> = OnceLock::new();
    IMAGE.get_or_init(|| {
        let root = RootSetup::derive(
            common::PROFILE,
            5,
            3,
            128,
            akita_types::AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
        )
        .unwrap();
        let layout = LoweredRootLayout::new(root.setup(), root.shape(), common::BASES[1]).unwrap();
        let image_log = layout.image_log_len();
        assert_eq!(image_log, 13);
        let image_key =
            akita_params::ScheduleLookupKey::single(PolynomialGroupLayout::singleton(image_log));
        // The ordinary small fixtures use direct setup. Retain the audited
        // recursive artifact's offloading edges for this scalar image row.
        let images = image_opening_catalog(image_log);
        let image_row = images.resolve_key(&image_key).unwrap();
        let required = required_setup_prefix_slot_ids_for_schedule(
            image_row.schedule(),
            &image_row.profiles().opening_layout().unwrap(),
        )
        .unwrap();
        assert!(!required.is_empty(), "selected test row must offload setup");
        eprintln!("offloaded image log={image_log} required slots={required:#?}");
        let requirements = SetupRequirements::from_catalog(&images, image_log, 1).unwrap();
        // Direct-config sizing intentionally does not enumerate offloaded slots.
        // Provision a correct seed and covering matrix with that empty registry.
        assert!(requirements.prefix_slot_ids().is_empty());
        let prover_setup =
            AkitaProverSetup::generate_with_capacity(image_log, 1, requirements.matrix_capacity())
                .unwrap();
        let verifier_setup = prover_setup
            .to_verifier_setup(requirements.matrix_capacity())
            .unwrap();
        assert!(prover_setup.prefix_slots.is_empty());
        assert!(verifier_setup.prefix_slots().is_empty());
        assert_eq!(
            verifier_setup.prefix_slots().setup_seed(),
            &verifier_setup.expanded().descriptor().setup_seed
        );
        assert!(TrustedScheduleCatalog::<ImageConfig>::verifier_admits(
            verifier_setup.expanded(),
            image_row
        )
        .unwrap());
        ensure_prover_schedule_fits_setup::<ImageConfig>(
            &prover_setup.expanded,
            image_row.schedule(),
            &image_row.profiles().opening_layout().unwrap(),
        )
        .unwrap();
        let mut complete_setup = prover_setup.clone();
        let backend = CpuBackend::<F, F>::new(complete_setup.expanded.clone()).unwrap();
        for (_, slot) in backend.export_setup_prefixes(&required).unwrap().iter() {
            complete_setup.prefix_slots.insert(slot.clone()).unwrap();
        }
        let complete_verifier_setup = complete_setup
            .to_verifier_setup(requirements.matrix_capacity())
            .unwrap();
        assert!(required
            .iter()
            .all(|id| complete_verifier_setup.prefix_slots().get(id).is_some()));
        let prover = ImageProver::new(images.clone(), complete_setup).unwrap();
        let prepared = PreparedMatrix::prepare(root.setup()).unwrap();
        let output = prover
            .commit::<BinaryField128>(&root, &prepared, &vec![0u128; root.setup().source_len()])
            .unwrap();
        output.committed_group.check().unwrap();
        let point = vec![F::zero(); image_log];
        let mut state = akita_transcript::new_prover_channel(b"missing-prefix/v1", b"").unwrap();
        prover
            .open_on_channel::<BinaryField128, _>(
                &root,
                &output,
                ImageEvaluation {
                    point: &point,
                    value: F::zero(),
                },
                &mut state,
            )
            .unwrap();
        Fixture {
            root,
            layout,
            images,
            prover_setup,
            verifier_setup,
            complete_verifier_setup,
            image: output.committed_group,
            proof: state.narg_string().to_vec(),
        }
    })
}

fn shrink_profile(profile: &mut GroupCommitPhaseParams, domain: usize) {
    let old_domain = 1usize << profile.group.num_vars();
    assert!(domain <= old_domain && old_domain.is_multiple_of(domain));
    let ratio = old_domain / domain;
    let old = profile.blocks;
    let positions = (old.positions_per_block / ratio).max(1);
    let rings = old.live_ring_elements_per_claim / ratio;
    profile.blocks = akita_params::BlockGeometry::new(rings, positions, rings.div_ceil(positions));
    profile.group = PolynomialGroupLayout::singleton(domain.trailing_zeros() as usize);
    profile.inner.matrix = profile
        .inner
        .matrix
        .try_with_input_width(positions * profile.inner.digits.num_digits)
        .unwrap();
    let width = profile
        .derive_slice_geometry()
        .unwrap()
        .physical_input_width();
    profile.outer.matrix = akita_params::OuterCommitMatrixParams::try_new_with_min_rank(
        profile.outer.matrix.sis_table_key(),
        width,
    )
    .unwrap();
    profile.validate_frozen_precommit(128).unwrap();
}

fn native_schedule() -> akita_params::FoldSchedule {
    let bytes = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts")
            .join(akita_params::SCHEDULE_ARTIFACT_SET)
            .join("fp128_dense_recursive.aks"),
    )
    .unwrap();
    let recursive = TrustedScheduleCatalog::<akita_config::RecursiveCommitmentConfig<ImageConfig>>::from_artifact_bytes(
        &bytes,
    ).unwrap();
    let row = recursive
        .rows()
        .filter(|row| {
            !required_setup_prefix_slot_ids_for_schedule(
                row.schedule(),
                &row.profiles().opening_layout().unwrap(),
            )
            .unwrap()
            .is_empty()
        })
        .min_by_key(|row| row.profiles().final_group.group.num_vars())
        .unwrap();
    row.schedule().clone()
}

fn shared_opening_matrix(params: &mut akita_params::CommittedGroupParams) {
    let d = params.open().matrix.ring_dimension();
    let width = params
        .preceding_group_iter()
        .chain(std::iter::once(params.own_group()))
        .map(|group| group.d_segment_width(1, d).unwrap())
        .sum();
    params.open_matrix = akita_params::OpenCommitMatrixParams::try_new_with_min_rank(
        params.open_matrix.sis_table_key(),
        width,
    )
    .unwrap();
}

fn repair_transitions(
    schedule: &mut akita_params::FoldSchedule,
    profiles: &CommittedGroupBatchProfile,
) {
    let root_layout = profiles.opening_layout().unwrap();
    schedule.root.input_witness_len = akita_params::root_input_witness_len(&schedule.root.params);
    schedule.root.output_witness_len = schedule
        .root
        .params
        .output_witness_len_for_field_bits(128, 1, &root_layout)
        .unwrap();
    for index in 0..schedule.recursive_folds.len() {
        let predecessor = if index == 0 {
            &schedule.root
        } else {
            &schedule.recursive_folds[index - 1]
        };
        let layout = if index == 0 {
            root_layout.clone()
        } else {
            akita_params::suffix_opening_layout(
                predecessor.input_witness_len,
                predecessor
                    .params
                    .setup_prefix()
                    .and_then(|prefix| prefix.setup_natural_len),
            )
            .unwrap()
        };
        let natural_len =
            akita_params::active_setup_field_len(&predecessor.params, &layout).unwrap();
        let input = predecessor.output_witness_len;
        let step = &mut schedule.recursive_folds[index];
        if let Some(mut prefix) = step.params.setup_prefix().copied() {
            shrink_profile(
                &mut prefix.profile,
                akita_params::padded_setup_prefix_len(natural_len),
            );
            prefix.setup_natural_len = Some(natural_len);
            step.params.set_setup_prefix(Some(prefix)).unwrap();
            shared_opening_matrix(&mut step.params);
        }
        step.input_witness_len = input;
        step.output_witness_len = akita_schedules::planner_support::planned_next_witness_len(
            128,
            1,
            &step.params,
            1,
            step.params.witness_chunk.num_chunks,
        )
        .unwrap()
        .unwrap();
    }
    schedule.terminal.input_witness_len =
        schedule.recursive_folds.last().unwrap().output_witness_len;
}

fn image_catalog(log: usize, offloaded: bool) -> TrustedScheduleCatalog<ImageConfig> {
    let mut schedule = native_schedule();
    shrink_profile(
        &mut schedule.root.params.own_group_mut().profile,
        1usize << log,
    );
    shared_opening_matrix(&mut schedule.root.params);
    if !offloaded {
        for step in &mut schedule.recursive_folds {
            step.params.set_setup_prefix(None).unwrap();
            shared_opening_matrix(&mut step.params);
        }
    }
    let profiles = CommittedGroupBatchProfile {
        final_group: schedule.root.params.own_group().profile,
        precommitteds: Vec::new(),
    };
    repair_transitions(&mut schedule, &profiles);
    admit::<ImageConfig>(profiles, schedule)
}

fn image_opening_catalog(log: usize) -> TrustedScheduleCatalog<ImageConfig> {
    let template = image_catalog(log, true);
    let row = template.rows().next().unwrap();
    let profiles = row.profiles().clone();
    let mut schedule = row.schedule().clone();
    // The source row's suffix cubes accept zero extension, but an honest CPU
    // commitment also needs exact live block counts for each produced witness.
    for index in 0..schedule.recursive_folds.len() {
        let input = schedule.recursive_folds[index].input_witness_len;
        let params = &mut schedule.recursive_folds[index].params;
        let profile = &mut params.own_group_mut().profile;
        let d = profile.inner.matrix.ring_dimension();
        let rings = input.div_ceil(d);
        let positions = profile.blocks.positions_per_block;
        profile.blocks =
            akita_params::BlockGeometry::new(rings, positions, rings.div_ceil(positions));
        profile.group = PolynomialGroupLayout::singleton(
            (rings * d).next_power_of_two().trailing_zeros() as usize,
        );
        let width = profile
            .derive_slice_geometry()
            .unwrap()
            .physical_input_width();
        profile.outer.matrix = akita_params::OuterCommitMatrixParams::try_new_with_min_rank(
            profile.outer.matrix.sis_table_key(),
            width,
        )
        .unwrap();
        profile.validate_frozen_precommit(128).unwrap();
        shared_opening_matrix(params);
        repair_transitions(&mut schedule, &profiles);
    }
    admit::<ImageConfig>(profiles, schedule)
}

fn admit<C: CommitmentConfig>(
    profiles: CommittedGroupBatchProfile,
    schedule: akita_params::FoldSchedule,
) -> TrustedScheduleCatalog<C> {
    // Retain audited suffix cubes and their canonical zero extension, then
    // re-audit every security bound, prefix domain and transition length.
    TrustedScheduleCatalog::new(
        ValidatedScheduleCatalog::try_new(
            C::schedule_family_name(),
            [(profiles, schedule)],
            &policy_of::<C>(),
            C::ring_challenge_config,
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn image_missing_setup_prefix_rejects_before_parent_activity() {
    let f = fixture();
    let verifier = ImageVerifier::new(f.images.clone(), f.verifier_setup.clone()).unwrap();
    let complete = ImageVerifier::new(f.images.clone(), f.complete_verifier_setup.clone()).unwrap();
    let point = vec![F::zero(); f.layout.image_log_len()];
    // The same genuine commitment, statement and proof verify with all slots.
    let state =
        akita_transcript::new_verifier_channel(b"missing-prefix/v1", b"", &f.proof).unwrap();
    let mut accepting = Recording::new(state);
    complete
        .verify_on_channel::<BinaryField128, _>(
            &f.root,
            &f.image,
            ImageEvaluation {
                point: &point,
                value: F::zero(),
            },
            &mut accepting,
        )
        .unwrap();
    accepting.inner.check_eof().unwrap();
    assert_eq!(accepting.message_calls, 2);
    assert!(!accepting.public.is_empty());
    assert_eq!(accepting.draws.len(), 1);
    // A missing slot must win even over a truncated length frame.
    for proof in [&f.proof[..], &f.proof[..7]] {
        let state =
            akita_transcript::new_verifier_channel(b"missing-prefix/v1", b"", proof).unwrap();
        let mut record = Recording::new(state);
        assert!(matches!(
            verifier.verify_on_channel::<BinaryField128, _>(
                &f.root,
                &f.image,
                ImageEvaluation {
                    point: &point,
                    value: F::zero()
                },
                &mut record,
            ),
            Err(AkitaError::InvalidSetup(message)) if message.contains("planned setup-prefix slot is missing")
        ));
        assert!(record.public.is_empty());
        assert!(record.messages.is_empty());
        assert_eq!(record.message_calls, 0);
        assert!(record.draws.is_empty());
        let mut control =
            akita_transcript::new_verifier_channel(b"missing-prefix/v1", b"", proof).unwrap();
        assert_eq!(
            record.challenge_block().unwrap(),
            ClearChannel::challenge_block(&mut control).unwrap()
        );
    }
}

#[test]
fn image_prover_missing_setup_prefix_rejects_before_parent_activity() {
    let f = fixture();
    let prover = ImageProver::new(f.images.clone(), f.prover_setup.clone()).unwrap();
    let prepared = PreparedMatrix::prepare(f.root.setup()).unwrap();
    let output = prover
        .commit::<BinaryField128>(
            &f.root,
            &prepared,
            &vec![0u128; f.root.setup().source_len()],
        )
        .unwrap();
    assert_eq!(output.committed_group, f.image);
    let point = vec![F::zero(); f.layout.image_log_len()];
    let state = akita_transcript::new_prover_channel(b"missing-prefix/v1", b"").unwrap();
    let mut record = Recording::new(state);
    assert!(matches!(
        prover.open_on_channel::<BinaryField128, _>(
            &f.root,
            &output,
            ImageEvaluation { point: &point, value: F::zero() },
            &mut record,
        ),
        Err(AkitaError::InvalidSetup(message)) if message.contains("planned setup-prefix slot is missing")
    ));
    assert!(record.public.is_empty());
    assert!(record.messages.is_empty());
    assert_eq!(record.message_calls, 0);
    assert!(record.draws.is_empty());
    let mut control = akita_transcript::new_prover_channel(b"missing-prefix/v1", b"").unwrap();
    assert_eq!(
        record.challenge_block().unwrap(),
        control.challenge_block().unwrap()
    );
}

struct RootFixture {
    root: RootSetup,
    images: TrustedScheduleCatalog<ImageConfig>,
    digits: TrustedScheduleCatalog<Digits2>,
    prover_setup: AkitaProverSetup<F>,
    verifier_setup: AkitaVerifierSetup<F>,
}

#[derive(Clone, Copy)]
enum OffloadedRole {
    Image,
    Digits,
}

fn root_fixture(role: OffloadedRole) -> &'static RootFixture {
    static IMAGE: OnceLock<RootFixture> = OnceLock::new();
    static DIGITS: OnceLock<RootFixture> = OnceLock::new();
    let storage = match role {
        OffloadedRole::Image => &IMAGE,
        OffloadedRole::Digits => &DIGITS,
    };
    storage.get_or_init(|| {
        let root = RootSetup::derive(
            common::PROFILE,
            12,
            8,
            128,
            akita_types::AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
        )
        .unwrap();
        let layout = LoweredRootLayout::new(root.setup(), root.shape(), Digits2::BASE).unwrap();
        let image_log = layout.image_log_len();
        assert_eq!(image_log, 18);
        let image_key =
            akita_params::ScheduleLookupKey::single(PolynomialGroupLayout::singleton(image_log));
        let images = image_catalog(image_log, matches!(role, OffloadedRole::Image));
        let image_row = images.resolve_key(&image_key).unwrap();
        let key = akita_params::ScheduleLookupKey {
            final_group: PolynomialGroupLayout::singleton(layout.witness_log_len()),
            precommitteds: vec![image_row.profiles().final_group],
        };
        let digits = digit_catalog(&key, matches!(role, OffloadedRole::Digits));
        let digit_row = digits.resolve_key(&key).unwrap();
        let required = match role {
            OffloadedRole::Image => image_row,
            OffloadedRole::Digits => digit_row,
        };
        assert!(
            !required_setup_prefix_slot_ids_for_schedule(
                required.schedule(),
                &required.profiles().opening_layout().unwrap(),
            )
            .unwrap()
            .is_empty(),
            "selected test row must offload setup"
        );
        let other = match role {
            OffloadedRole::Image => digit_row,
            OffloadedRole::Digits => image_row,
        };
        assert!(
            required_setup_prefix_slot_ids_for_schedule(
                other.schedule(),
                &other.profiles().opening_layout().unwrap(),
            )
            .unwrap()
            .is_empty(),
            "the other selected row must not require prefixes"
        );
        let max = image_log.max(layout.witness_log_len());
        let requirements = SetupRequirements::from_catalog(&images, max, 2)
            .unwrap()
            .union(SetupRequirements::from_catalog(&digits, max, 2).unwrap())
            .unwrap();
        // Direct-config sizing intentionally does not enumerate offloaded slots.
        // Provision a correct seed and covering matrix with that empty registry.
        assert!(requirements.prefix_slot_ids().is_empty());
        eprintln!(
            "offloaded image log={image_log} required setup fields={}",
            requirements.matrix_capacity().num_field_elements
        );
        let prover_setup =
            AkitaProverSetup::generate_with_capacity(max, 2, requirements.matrix_capacity())
                .unwrap();
        let verifier_setup = prover_setup
            .to_verifier_setup(requirements.matrix_capacity())
            .unwrap();
        assert!(prover_setup.prefix_slots.is_empty());
        assert!(verifier_setup.prefix_slots().is_empty());
        assert_eq!(
            verifier_setup.prefix_slots().setup_seed(),
            &verifier_setup.expanded().descriptor().setup_seed
        );
        assert!(TrustedScheduleCatalog::<ImageConfig>::verifier_admits(
            verifier_setup.expanded(),
            image_row
        )
        .unwrap());
        let digit_row = digits.resolve_key(&key).unwrap();
        assert!(TrustedScheduleCatalog::<Digits2>::verifier_admits(
            verifier_setup.expanded(),
            digit_row
        )
        .unwrap());
        ensure_prover_schedule_fits_setup::<ImageConfig>(
            &prover_setup.expanded,
            image_row.schedule(),
            &image_row.profiles().opening_layout().unwrap(),
        )
        .unwrap();
        ensure_prover_schedule_fits_setup::<Digits2>(
            &prover_setup.expanded,
            digit_row.schedule(),
            &digit_row.profiles().opening_layout().unwrap(),
        )
        .unwrap();
        RootFixture {
            root,
            images,
            digits,
            prover_setup,
            verifier_setup,
        }
    })
}

fn digit_catalog(
    key: &akita_params::ScheduleLookupKey,
    offloaded: bool,
) -> TrustedScheduleCatalog<Digits2> {
    let mut schedule = native_schedule();
    let root = &mut schedule.root.params;
    let policy = policy_of::<Digits2>();
    let inner_digits = akita_params::sis::num_digits_inner(
        akita_params::DecompositionParams {
            log_basis: root.inner().digits.log_basis,
            ..policy.decomposition
        },
        true,
    );
    root.own_group_mut().profile.inner.digits.num_digits = inner_digits;
    shrink_profile(
        &mut root.own_group_mut().profile,
        1usize << key.final_group.num_vars(),
    );
    let own = *root.own_group();
    let image = akita_params::GroupOpenPhaseParams::admit(
        key.precommitteds[0],
        root.num_digits_fold(),
        akita_params::PrecommittedGroupAdmissionPolicy {
            decomposition: policy.decomposition,
            sis_security_policy: policy.sis_security_policy,
            sis_table_digest: policy.sis_table_digest,
            sis_modulus_profile: policy.sis_modulus_profile,
            num_response_chunks: root.witness_chunk.num_chunks,
        },
        root.opening_method(),
        root.fold_challenge_config(),
        root.open().digits.log_basis,
    )
    .unwrap();
    *root = akita_params::CommittedGroupParams::try_new(
        vec![image, own],
        root.open_matrix,
        root.payload_mode,
        root.ring_relation_mode,
        root.source_encoding,
        root.witness_chunk,
    )
    .unwrap();
    shared_opening_matrix(root);
    let profiles = CommittedGroupBatchProfile {
        final_group: root.own_group().profile,
        precommitteds: key.precommitteds.clone(),
    };
    if !offloaded {
        for step in &mut schedule.recursive_folds {
            step.params.set_setup_prefix(None).unwrap();
            shared_opening_matrix(&mut step.params);
        }
    }
    repair_transitions(&mut schedule, &profiles);
    admit::<Digits2>(profiles, schedule)
}

fn reject_root_constructors(role: OffloadedRole) {
    let f = root_fixture(role);
    // Neither constructor takes a channel: rejection occurs before an oracle
    // capable of absorbing public bytes or reading proof input is available.
    let state = akita_transcript::new_verifier_channel(b"missing-prefix/v1", b"", &[]).unwrap();
    let record = Recording::new(state);
    assert!(matches!(RootPcsVerifier::<Digits2>::new(
        f.root.clone(), f.images.clone(), f.digits.clone(), f.verifier_setup.clone(),
    ), Err(AkitaError::InvalidSetup(message)) if message.contains("planned setup-prefix slot is missing")));
    assert!(matches!(RootPcsProver::<Digits2>::new(
        f.root.clone(), f.images.clone(), f.digits.clone(), f.prover_setup.clone(),
    ), Err(AkitaError::InvalidSetup(message)) if message.contains("planned setup-prefix slot is missing")));
    assert!(record.public.is_empty());
    assert_eq!(record.message_calls, 0);
    assert!(record.draws.is_empty());
}

#[test]
fn root_constructors_reject_missing_selected_image_setup_prefix() {
    reject_root_constructors(OffloadedRole::Image);
}

#[test]
fn root_constructors_reject_missing_selected_grouped_digit_setup_prefix() {
    reject_root_constructors(OffloadedRole::Digits);
}
