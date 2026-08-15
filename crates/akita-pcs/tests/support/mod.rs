//! Test-local PCS config adapters.
//!
//! Crate unit tests include this module under `cfg(test)`. Production builds
//! never compile it.

#![allow(dead_code)]

use akita_challenges::SparseChallengeConfig;
use akita_config::{policy_of, CommitmentConfig};
use akita_field::AkitaError;
use akita_types::{
    schedule_row_digest, AkitaScheduleLookupKey, CommittedGroupBatchProfile, CommittedGroupProfile,
    DecompositionParams, OpeningScheduleSelection, PolynomialGroupLayout, SetupMatrixCapacity,
    SisModulusProfileId,
};
use std::{
    any::TypeId,
    marker::PhantomData,
    sync::{Mutex, OnceLock},
};

#[derive(Clone)]
struct SyntheticResolvedRow {
    config: TypeId,
    row: akita_config::ResolvedScheduleRow,
}

fn synthetic_resolved_rows() -> &'static Mutex<Vec<SyntheticResolvedRow>> {
    static ROWS: OnceLock<Mutex<Vec<SyntheticResolvedRow>>> = OnceLock::new();
    ROWS.get_or_init(|| Mutex::new(Vec::new()))
}

fn select_synthetic_schedule_row<C>(
    profiles: &CommittedGroupBatchProfile,
    key: AkitaScheduleLookupKey,
) -> Result<akita_config::ResolvedScheduleRow, AkitaError>
where
    C: CommitmentConfig + 'static,
{
    let row = C::resolve_catalog_row_for_key(&key)?;
    if row.profiles() != profiles {
        return Err(AkitaError::InvalidSetup(
            "synthetic selected row does not match exact committed profiles".into(),
        ));
    }
    let selection = row.selection();
    let mut rows = synthetic_resolved_rows()
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("synthetic row cache is poisoned".into()))?;
    if let Some(existing) = rows.iter_mut().find(|existing| {
        existing.config == TypeId::of::<C>() && existing.row.selection() == selection
    }) {
        existing.row = row.clone();
        return Ok(row);
    }
    if rows.len() >= 1024 {
        return Err(AkitaError::InvalidSetup(
            "synthetic row cache capacity exceeded".into(),
        ));
    }
    rows.push(SyntheticResolvedRow {
        config: TypeId::of::<C>(),
        row: row.clone(),
    });
    Ok(row)
}

fn resolve_synthetic_schedule_row<C>(
    selection: OpeningScheduleSelection,
) -> Result<akita_config::ResolvedScheduleRow, AkitaError>
where
    C: CommitmentConfig + 'static,
{
    synthetic_resolved_rows()
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("synthetic row cache is poisoned".into()))?
        .iter()
        .find(|entry| entry.config == TypeId::of::<C>() && entry.row.selection() == selection)
        .map(|entry| entry.row.clone())
        .ok_or_else(|| {
            AkitaError::UnsupportedSchedule(
                "synthetic schedule selection is not present in the test catalog".into(),
            )
        })
}

fn synthetic_schedule_key(profiles: &CommittedGroupBatchProfile) -> AkitaScheduleLookupKey {
    AkitaScheduleLookupKey {
        final_group: profiles.final_group.group,
        precommitteds: profiles.precommitteds.clone(),
    }
}

fn rebuild_group_output_matrices(
    params: &mut akita_types::CommittedGroupParams,
    num_claims: usize,
    extension_degree: usize,
) -> Result<(), AkitaError> {
    let dims = params.role_dims();
    let outer_width = akita_types::CommitmentSliceGeometry::try_new(
        params.outer_slice_count,
        params.num_live_blocks,
        num_claims,
        params.inner_commit_matrix.output_rank(),
        params.num_digits_outer,
        dims.d_a(),
        dims.d_b(),
    )?
    .physical_input_width();
    params.outer_commit_matrix = akita_types::OuterCommitMatrixParams::try_new_with_min_rank(
        params.outer_commit_matrix.sis_table_key(),
        outer_width,
    )?;
    let d_width = akita_types::opening_d_segment_width(
        params.opening_method,
        extension_degree,
        dims.d_a(),
        dims.d_d(),
        params.num_digits_open,
        params.num_live_blocks,
        num_claims,
    )?;
    params.open_commit_matrix = akita_types::OpenCommitMatrixParams::try_new_with_min_rank(
        params.open_commit_matrix.sis_table_key(),
        d_width,
    )?;
    Ok(())
}

/// Test-only commitment config that combines an envelope config with a final
/// group config.
///
/// The precommitted groups use `Envelope::D` while the final group and
/// recursive suffix use `Final::D`.
///
/// `Self::D` remains the uniform planner default. Exact grouped runtime keys
/// select schedules under `Final`, retaining each preceding group's frozen
/// native descriptor. Public setup storage remains flat and dimension-free.
#[derive(Debug)]
pub(crate) struct EnvelopeFinalGroupConfig<Envelope, Final>(PhantomData<fn() -> (Envelope, Final)>);

impl<Envelope, Final> Clone for EnvelopeFinalGroupConfig<Envelope, Final> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Test-only catalog adapter that replaces the root opening with reduced-width
/// coefficient packing over the smallest production challenge subring.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RootCoefficientPackingConfig<Base>(PhantomData<fn() -> Base>);

impl<Base> CommitmentConfig for RootCoefficientPackingConfig<Base>
where
    Base: CommitmentConfig + 'static,
{
    type Field = Base::Field;
    type ExtField = Base::ExtField;

    const D: usize = Base::D;
    const EXT_DEGREE: usize = Base::EXT_DEGREE;
    const RING_DIMENSION_SCHEDULE_MODE: akita_schedules::RingDimensionScheduleMode =
        Base::RING_DIMENSION_SCHEDULE_MODE;

    fn decomposition() -> DecompositionParams {
        Base::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Base::ring_challenge_config(d)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Base::sis_modulus_profile()
    }

    fn setup_matrix_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixCapacity, AkitaError> {
        let base = Base::setup_matrix_capacity(max_num_vars, max_num_batched_polys)?;
        Ok(SetupMatrixCapacity {
            num_field_elements: base.num_field_elements.checked_mul(16).ok_or_else(|| {
                AkitaError::InvalidSetup("coefficient-packing test setup capacity overflow".into())
            })?,
        })
    }

    fn setup_prefix_inner_ring_dimension() -> usize {
        Base::setup_prefix_inner_ring_dimension()
    }

    fn opening_basis_range() -> (u32, u32) {
        Base::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        Base::inner_basis_range()
    }

    fn root_honest_fold_policy() -> akita_types::sis::HonestFoldPolicySpec {
        Base::root_honest_fold_policy()
    }

    fn chunked_witness_cfg() -> akita_types::ChunkedWitnessCfg {
        Base::chunked_witness_cfg()
    }

    fn recursive_setup_planning() -> bool {
        Base::recursive_setup_planning()
    }

    fn selection_policy() -> akita_schedules::SelectionPolicyId {
        Base::selection_policy()
    }

    fn resolve_catalog_row_for_key(
        key: &AkitaScheduleLookupKey,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        if !key.precommitteds.is_empty() {
            return Err(AkitaError::UnsupportedSchedule(
                "the coefficient-packing test catalog supports one root group".into(),
            ));
        }
        let base = Base::resolve_catalog_row_for_key(key)?;
        let mut schedule = base.into_schedule();
        let policy = policy_of::<Self>();
        let root = &mut schedule.root.params.final_group.commitment;
        let d_a = root.inner_commit_matrix.ring_dimension();
        let challenge_subring_dimension = 64;
        akita_types::SubringCoefficientPackingGeometry::try_new(
            Self::EXT_DEGREE,
            d_a,
            challenge_subring_dimension,
        )?;
        root.opening_method = akita_types::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        };
        root.source_encoding = akita_types::CommittedSourceEncoding::CanonicalCoefficientTable;
        root.fold_challenge_config =
            SparseChallengeConfig::production_for_ring_dim(challenge_subring_dimension)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "root packing subring is not in the production challenge ladder".into(),
                    )
                })?;
        let root_open_dimension = 128usize;
        let root_open_bound = akita_types::sis::rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            policy.sis_modulus_profile,
            akita_types::SisMatrixRole::Open,
            root_open_dimension,
            root.log_basis_open,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup("root packing test has no audited D bound".into())
        })?;
        let mut root_open_key = root.open_commit_matrix.sis_table_key();
        root_open_key.ring_dimension = root_open_dimension
            .try_into()
            .map_err(|_| AkitaError::InvalidSetup("root packing D dimension exceeds u32".into()))?;
        root_open_key.coeff_linf_bound = root_open_bound;
        root.open_commit_matrix = akita_types::OpenCommitMatrixParams::try_new_with_min_rank(
            root_open_key,
            root.open_commit_matrix.input_width(),
        )?;
        let required_a_bound = akita_types::sis::rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            d_a,
            root.log_basis_open,
            &root.fold_challenge_config,
            root.num_digits_fold,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup("root packing challenge family has no audited A bound".into())
        })?;
        let current_a = root.inner_commit_matrix;
        let mut current_key = current_a.sis_table_key().ok_or_else(|| {
            AkitaError::InvalidSetup("root packing requires a L-infinity A matrix".into())
        })?;
        current_key.coeff_linf_bound = required_a_bound;
        root.inner_commit_matrix = akita_types::InnerCommitMatrixParams::try_new_with_min_rank(
            current_key,
            current_a.input_width(),
        )?;
        rebuild_group_output_matrices(root, key.final_group.num_polynomials(), Self::EXT_DEGREE)?;
        schedule.root.params.open_commit_matrix = root.open_commit_matrix;
        schedule.root.params.sparse_challenge_config = root.fold_challenge_config;

        let opening_batch = key.opening_layout()?;
        let root_output_witness_len = root.output_witness_len_for_field_bits(
            policy.decomposition.field_bits(),
            Self::EXT_DEGREE,
            &opening_batch,
        )?;
        schedule.root.output_witness_len = root_output_witness_len;

        let mut successor = schedule.recursive_folds.first().cloned().ok_or_else(|| {
            AkitaError::InvalidSetup("packing Stage 3 test requires one recursive successor".into())
        })?;
        successor.input_witness_len = root_output_witness_len;
        let successor_witness = &mut successor.params.witness;
        if successor_witness.log_basis_inner != root.log_basis_open
            || successor_witness.num_digits_inner != 1
        {
            return Err(AkitaError::InvalidSetup(format!(
                "packing recursive digit basis mismatch: predecessor open={}, successor inner={} with {} digits",
                root.log_basis_open,
                successor_witness.log_basis_inner,
                successor_witness.num_digits_inner,
            )));
        }
        successor_witness.opening_method = akita_types::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        };
        successor_witness.source_encoding =
            akita_types::CommittedSourceEncoding::CanonicalCoefficientTable;
        successor_witness.fold_challenge_config =
            SparseChallengeConfig::production_for_ring_dim(challenge_subring_dimension)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "successor packing subring is not in the production ladder".into(),
                    )
                })?;
        successor_witness.num_live_ring_elements_per_claim =
            root_output_witness_len.div_ceil(successor_witness.d_a());

        let root_setup_natural_len = akita_types::active_setup_field_len(root, &opening_batch)?;
        let root_setup_prefix_len = akita_types::padded_setup_prefix_len(root_setup_natural_len);
        let prefix_ring_slots = root_setup_prefix_len
            .checked_div(successor_witness.d_a())
            .filter(|slots| {
                *slots != 0 && root_setup_prefix_len.is_multiple_of(successor_witness.d_a())
            })
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "packing setup prefix does not align to the successor A ring".into(),
                )
            })?;
        successor_witness.num_positions_per_block = prefix_ring_slots.next_power_of_two();
        successor_witness.num_live_blocks = successor_witness
            .num_live_ring_elements_per_claim
            .div_ceil(successor_witness.num_positions_per_block);
        let successor_a_width = successor_witness
            .num_positions_per_block
            .checked_mul(successor_witness.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("packing successor A width overflow".into()))?;
        let mut successor_a_key = successor_witness
            .inner_commit_matrix
            .sis_table_key()
            .ok_or_else(|| {
                AkitaError::InvalidSetup("packing successor requires a L-infinity A matrix".into())
            })?;
        successor_a_key.coeff_linf_bound = akita_types::sis::rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            successor_witness.d_a(),
            successor_witness.log_basis_open,
            &successor_witness.fold_challenge_config,
            successor_witness.num_digits_fold,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup("packing successor has no audited A bound".into())
        })?;
        successor_witness.inner_commit_matrix =
            akita_types::InnerCommitMatrixParams::try_new_with_min_rank(
                successor_a_key,
                successor_a_width,
            )?;
        rebuild_group_output_matrices(successor_witness, 1, Self::EXT_DEGREE)?;

        let prefix_positions = prefix_ring_slots.min(256);
        let prefix_blocks = prefix_ring_slots
            .checked_div(prefix_positions)
            .filter(|blocks| {
                *blocks >= successor_witness.outer_slice_count.get()
                    && prefix_ring_slots.is_multiple_of(prefix_positions)
            })
            .ok_or_else(|| {
                AkitaError::InvalidSetup("packing setup prefix has no balanced split".into())
            })?;
        let mut prefix_source_params = successor_witness.clone();
        prefix_source_params.setup_prefix = None;
        prefix_source_params.log_basis_inner = root.log_basis_inner;
        prefix_source_params.num_digits_inner = root.num_digits_inner;
        let prefix_inner_width = prefix_positions
            .checked_mul(prefix_source_params.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("packing prefix A width overflow".into()))?;
        prefix_source_params.inner_commit_matrix =
            akita_types::InnerCommitMatrixParams::try_new_with_min_rank(
                prefix_source_params
                    .inner_commit_matrix
                    .sis_table_key()
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "packing prefix requires a L-infinity A matrix".into(),
                        )
                    })?,
                prefix_inner_width,
            )?;
        let prefix_outer_width = akita_types::CommitmentSliceGeometry::try_new(
            prefix_source_params.outer_slice_count,
            prefix_blocks,
            1,
            prefix_source_params.inner_commit_matrix.output_rank(),
            prefix_source_params.num_digits_outer,
            prefix_source_params.d_a(),
            prefix_source_params.role_dims().d_b(),
        )?
        .physical_input_width();
        prefix_source_params.outer_commit_matrix =
            akita_types::OuterCommitMatrixParams::try_new_with_min_rank(
                prefix_source_params.outer_commit_matrix.sis_table_key(),
                prefix_outer_width,
            )?;
        let mut prefix_params = akita_types::setup_prefix_precommitted_params(
            &prefix_source_params,
            root_setup_prefix_len,
        )?;
        if prefix_params.layout.inner_commit_matrix.ring_dimension() != successor_witness.d_a() {
            return Err(AkitaError::InvalidSetup(
                "packing root setup prefix left the base row's planned commitment class".into(),
            ));
        }
        prefix_params.opening.num_digits_fold = akita_types::sis::num_digits_for_linf_cap(
            i16::MAX as u128,
            policy.decomposition.field_bits(),
            prefix_params.opening.log_basis_open,
        );
        prefix_params.opening.opening_method =
            akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            };
        prefix_params.opening.fold_challenge_config =
            SparseChallengeConfig::production_for_ring_dim(challenge_subring_dimension)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "setup-prefix packing subring is not in the production ladder".into(),
                    )
                })?;
        let incoming_setup_prefix =
            akita_types::scheduled_setup_prefix(root_setup_natural_len, prefix_params);
        successor_witness.setup_prefix = Some(incoming_setup_prefix.clone());
        let successor_d_width = successor_witness
            .open_commit_matrix
            .input_width()
            .checked_add(
                incoming_setup_prefix
                    .commitment_params
                    .d_segment_width(Self::EXT_DEGREE, successor_witness.role_dims().d_d())?,
            )
            .ok_or_else(|| AkitaError::InvalidSetup("packing successor D width overflow".into()))?;
        successor_witness.open_commit_matrix =
            akita_types::OpenCommitMatrixParams::try_new_with_min_rank(
                successor_witness.open_commit_matrix.sis_table_key(),
                successor_d_width,
            )?;
        successor.params.open_commit_matrix = successor_witness.open_commit_matrix;
        successor.params.sparse_challenge_config = successor_witness.fold_challenge_config;
        successor.params.incoming_setup_prefix = Some(incoming_setup_prefix);
        let successor_opening_batch = akita_types::suffix_opening_layout(
            root_output_witness_len,
            Some(root_setup_natural_len),
        )?;
        successor.output_witness_len = successor_witness.output_witness_len_for_field_bits(
            policy.decomposition.field_bits(),
            Self::EXT_DEGREE,
            &successor_opening_batch,
        )?;
        schedule.recursive_folds.clear();
        schedule.recursive_folds.push(successor);

        let terminal = &mut schedule.terminal;
        terminal.input_witness_len = schedule.recursive_folds[0].output_witness_len;
        let terminal_witness = &mut terminal.params.witness;
        let terminal_d = [terminal_witness.d_a(), 64]
            .into_iter()
            .find(|dimension| terminal.input_witness_len.is_multiple_of(*dimension))
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "packing test root output has no supported terminal divisor".into(),
                )
            })?;
        terminal.params.sparse_challenge_config = Self::ring_challenge_config(terminal_d)?;
        terminal_witness.num_live_ring_elements_per_claim = terminal.input_witness_len / terminal_d;
        terminal_witness.num_live_blocks = terminal_witness
            .num_live_ring_elements_per_claim
            .div_ceil(terminal_witness.num_positions_per_block);
        terminal_witness.fold_digit_count = akita_types::sis::num_digits_for_linf_cap(
            i16::MAX as u128,
            policy.decomposition.field_bits(),
            terminal_witness.fold_log_basis,
        );
        let terminal_a_bound = akita_types::sis::rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            terminal_d,
            terminal_witness.fold_log_basis,
            &terminal.params.sparse_challenge_config,
            terminal_witness.fold_digit_count,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup("packing test terminal has no L-infinity A bound".into())
        })?;
        terminal_witness.inner_commit_matrix =
            akita_types::InnerCommitMatrixParams::try_new_with_min_rank(
                akita_types::SisTableKey {
                    policy: policy.sis_security_policy,
                    table_digest: policy.sis_table_digest,
                    modulus_profile: policy.sis_modulus_profile,
                    role: akita_types::SisMatrixRole::Inner,
                    ring_dimension: terminal_d.try_into().map_err(|_| {
                        AkitaError::InvalidSetup(
                            "packing test terminal dimension exceeds u32".into(),
                        )
                    })?,
                    coeff_linf_bound: terminal_a_bound,
                },
                terminal_witness.inner_width(),
            )?;
        let encoding_scale = terminal_witness
            .certified_response_linf_cap(&terminal.params.sparse_challenge_config)?;
        terminal.params.response_shape =
            akita_types::TerminalResponseShape::derive(terminal_witness, encoding_scale)?;

        schedule.validate_nonterminal_opening_execution(Self::EXT_DEGREE)?;
        let root = &schedule.root.params.final_group.commitment;
        let profiles = CommittedGroupBatchProfile {
            final_group: CommittedGroupProfile::try_from_params(key.final_group, root)?,
            precommitteds: Vec::new(),
        };
        let selection = OpeningScheduleSelection {
            row_digest: schedule_row_digest(&profiles, &schedule)?,
        };
        akita_config::ResolvedScheduleRow::try_new(selection, profiles, schedule, &policy)
    }

    fn resolve_catalog_row_for_profiles(
        profiles: &CommittedGroupBatchProfile,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        select_synthetic_schedule_row::<Self>(profiles, synthetic_schedule_key(profiles))
    }

    fn resolve_schedule_selection(
        selection: OpeningScheduleSelection,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        resolve_synthetic_schedule_row::<Self>(selection)
    }
}

impl<Envelope, Final> Copy for EnvelopeFinalGroupConfig<Envelope, Final> {}

impl<Envelope, Final> Default for EnvelopeFinalGroupConfig<Envelope, Final> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Envelope, Final> CommitmentConfig for EnvelopeFinalGroupConfig<Envelope, Final>
where
    Envelope: CommitmentConfig + 'static,
    Final: CommitmentConfig<Field = Envelope::Field, ExtField = Envelope::ExtField> + 'static,
{
    type Field = Envelope::Field;
    type ExtField = Envelope::ExtField;

    const D: usize = Envelope::D;
    const RING_DIMENSION_SCHEDULE_MODE: akita_schedules::RingDimensionScheduleMode =
        Envelope::RING_DIMENSION_SCHEDULE_MODE;

    fn decomposition() -> DecompositionParams {
        Envelope::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Envelope::ring_challenge_config(d).or_else(|_| Final::ring_challenge_config(d))
    }

    fn selection_policy() -> akita_schedules::SelectionPolicyId {
        Envelope::selection_policy()
    }
    fn sis_modulus_profile() -> SisModulusProfileId {
        Envelope::sis_modulus_profile()
    }

    fn setup_matrix_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixCapacity, AkitaError> {
        let mut num_field_elements =
            Envelope::setup_matrix_capacity(max_num_vars, max_num_batched_polys)?
                .num_field_elements
                .max(
                    Final::setup_matrix_capacity(max_num_vars, max_num_batched_polys)?
                        .num_field_elements,
                );
        for final_polys in 1..max_num_batched_polys {
            let pre_polys = max_num_batched_polys - final_polys;
            for pre_num_vars in [14usize, 15, 16].into_iter().filter(|&n| n <= max_num_vars) {
                let Ok(precommitted) = Self::profile_without_precommitted_groups(
                    PolynomialGroupLayout::new(pre_num_vars, pre_polys),
                ) else {
                    continue;
                };
                let Ok(schedule) = Self::resolve_catalog_row_for_key(&AkitaScheduleLookupKey {
                    final_group: PolynomialGroupLayout::new(max_num_vars, final_polys),
                    precommitteds: vec![precommitted],
                }) else {
                    continue;
                };
                num_field_elements = num_field_elements.max(
                    akita_types::setup_matrix_capacity_for_schedule(schedule.schedule())?
                        .num_field_elements,
                );
            }
        }
        Ok(SetupMatrixCapacity { num_field_elements })
    }

    fn opening_basis_range() -> (u32, u32) {
        Envelope::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        Envelope::inner_basis_range()
    }

    fn root_honest_fold_policy() -> akita_types::sis::HonestFoldPolicySpec {
        Envelope::root_honest_fold_policy()
    }

    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        Envelope::schedule_catalog()
    }

    fn resolve_catalog_row_for_key(
        key: &AkitaScheduleLookupKey,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        let (policy, ring_challenge_config) = if key.precommitteds.is_empty() {
            (
                policy_of::<Envelope>(),
                Envelope::ring_challenge_config
                    as fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
            )
        } else {
            (
                policy_of::<Final>(),
                Final::ring_challenge_config
                    as fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
            )
        };
        let precommitted_honest_fold_policies =
            vec![Envelope::root_honest_fold_policy(); key.precommitteds.len()];
        let schedule = akita_planner::find_schedule(
            key,
            Envelope::root_honest_fold_policy(),
            &precommitted_honest_fold_policies,
            &policy,
            ring_challenge_config,
        )?
        .schedule;
        let profiles = CommittedGroupBatchProfile {
            final_group: CommittedGroupProfile::try_from_params(
                key.final_group,
                &schedule.root.params.final_group.commitment,
            )?,
            precommitteds: key.precommitteds.clone(),
        };
        let selection = OpeningScheduleSelection {
            row_digest: schedule_row_digest(&profiles, &schedule)?,
        };
        akita_config::ResolvedScheduleRow::try_new(
            selection,
            profiles,
            schedule,
            &policy_of::<Self>(),
        )
    }

    fn resolve_catalog_row_for_profiles(
        profiles: &CommittedGroupBatchProfile,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        select_synthetic_schedule_row::<Self>(profiles, synthetic_schedule_key(profiles))
    }

    fn resolve_schedule_selection(
        selection: OpeningScheduleSelection,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        resolve_synthetic_schedule_row::<Self>(selection)
    }
}
