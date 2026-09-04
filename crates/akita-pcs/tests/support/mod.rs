//! Test-local PCS config adapters.
//!
//! Crate unit tests include this module under `cfg(test)`. Production builds
//! never compile it.

#![allow(dead_code)]

use akita_challenges::SparseChallengeConfig;
use akita_config::{policy_of, CommitmentConfig};
use akita_error::AkitaError;
use akita_types::sis::{
    BalancedSignedDigitFoldPolicy, FoldWitnessNorms, HonestFoldPolicy, HonestFoldSizingQuery,
};
use akita_types::{
    schedule_row_digest, AkitaScheduleLookupKey, CommittedGroupBatchProfile, DecompositionParams,
    GroupCommitPhaseParams, OpeningScheduleSelection, PolynomialGroupLayout, SetupMatrixCapacity,
    SisModulusProfileId,
};
use std::{
    any::TypeId,
    marker::PhantomData,
    sync::{Mutex, OnceLock},
};

mod cross_mode;
pub(crate) use cross_mode::{CrossModeConfig, QuotientMode, ReducedMode};

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
        params.outer_slice_count(),
        params.blocks().live_blocks,
        num_claims,
        params.inner().matrix.output_rank(),
        params.outer().digits.num_digits,
        dims.d_a(),
        dims.d_b(),
    )?
    .physical_input_width();
    params.own_group_mut().profile.outer.matrix =
        akita_types::OuterCommitMatrixParams::try_new_with_min_rank(
            params.outer().matrix.sis_table_key(),
            outer_width,
        )?;
    let d_width = akita_types::opening_d_segment_width(
        params.opening_method(),
        extension_degree,
        dims.d_a(),
        dims.d_d(),
        params.open().digits.num_digits,
        params.blocks().live_blocks,
        num_claims,
    )?;
    params.open_matrix = akita_types::OpenCommitMatrixParams::try_new_with_min_rank(
        params.open_matrix.sis_table_key(),
        d_width,
    )?;
    Ok(())
}

/// Exactly the fields universal fold sizing reads.
///
/// Both a fold's own `CommittedGroupParams` and a standalone
/// `GroupOpenPhaseParams` can supply these, and neither of them needs a
/// `PolynomialGroupLayout` to do it. Naming the inputs is what lets the two
/// callers keep the types they actually hold: routing a fold through
/// `final_group_scalar` to reach a group would impose that helper's
/// power-of-two source-length rule on a synthetic successor whose live ring
/// element count is not a power of two.
struct FoldDigitInputs {
    d_a: usize,
    log_basis_inner: u32,
    log_basis_open: u32,
    num_digits_inner: usize,
    num_positions_per_block: usize,
    num_live_blocks: usize,
    num_live_ring_elements_per_claim: usize,
    opening_method: akita_types::OpeningMethod,
    fold_challenge_config: akita_challenges::SparseChallengeConfig,
}

impl FoldDigitInputs {
    fn of_fold(params: &akita_types::CommittedGroupParams) -> Self {
        Self {
            d_a: params.inner().matrix.ring_dimension(),
            log_basis_inner: params.inner().digits.log_basis,
            log_basis_open: params.open().digits.log_basis,
            num_digits_inner: params.inner().digits.num_digits,
            num_positions_per_block: params.blocks().positions_per_block,
            num_live_blocks: params.blocks().live_blocks,
            num_live_ring_elements_per_claim: params.blocks().live_ring_elements_per_claim,
            opening_method: params.opening_method(),
            fold_challenge_config: params.fold_challenge_config(),
        }
    }

    fn of_group(params: &akita_types::GroupOpenPhaseParams) -> Self {
        Self {
            d_a: params.profile.inner.matrix.ring_dimension(),
            log_basis_inner: params.profile.inner.digits.log_basis,
            log_basis_open: params.opening.log_basis_open,
            num_digits_inner: params.profile.inner.digits.num_digits,
            num_positions_per_block: params.profile.blocks.positions_per_block,
            num_live_blocks: params.profile.blocks.live_blocks,
            num_live_ring_elements_per_claim: params.profile.blocks.live_ring_elements_per_claim,
            opening_method: params.opening.opening_method,
            fold_challenge_config: params.opening.fold_challenge_config,
        }
    }
}

fn universal_fold_digit_depth(
    params: FoldDigitInputs,
    field_bits: u32,
    num_claims: usize,
    num_chunks: usize,
) -> Result<usize, AkitaError> {
    let d_a = params.d_a;
    let witness_norms = FoldWitnessNorms::bounded(params.log_basis_inner, d_a);
    let width_s = params
        .num_positions_per_block
        .checked_mul(params.num_digits_inner)
        .ok_or_else(|| AkitaError::InvalidSetup("synthetic fold width overflow".into()))?;
    let num_fold_coeffs = width_s
        .checked_mul(d_a)
        .and_then(|count| count.checked_mul(num_chunks))
        .ok_or_else(|| AkitaError::InvalidSetup("synthetic fold response overflow".into()))?;
    BalancedSignedDigitFoldPolicy::universal(field_bits).num_digits_fold(HonestFoldSizingQuery {
        ring_dimension: d_a,
        challenge_dimension: match params.opening_method {
            akita_types::OpeningMethod::EvaluationTrace => d_a,
            akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            } => challenge_subring_dimension,
        },
        num_claims,
        num_live_ring_elements_per_claim: params.num_live_ring_elements_per_claim,
        num_live_blocks: params.num_live_blocks,
        num_positions_per_block: params.num_positions_per_block,
        num_chunks,
        num_fold_coeffs,
        witness_norms,
        log_basis_response: params.log_basis_open,
        challenge_config: &params.fold_challenge_config,
    })
}

fn retarget_synthetic_terminal<Cfg: CommitmentConfig>(
    schedule: &mut akita_types::FoldSchedule,
) -> Result<(), AkitaError> {
    let policy = policy_of::<Cfg>();
    let predecessor_output = schedule
        .recursive_folds
        .last()
        .ok_or_else(|| AkitaError::InvalidSetup("synthetic terminal has no predecessor".into()))?
        .output_witness_len;
    // After the three-type merge the terminal fold and its group are one value,
    // so the old `terminal` / `terminal` pair is a single borrow.
    let terminal = &mut schedule.terminal;
    terminal.input_witness_len = predecessor_output;
    let terminal_d = [terminal.d_a(), 64]
        .into_iter()
        .find(|dimension| terminal.input_witness_len.is_multiple_of(*dimension))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("packing test output has no supported terminal divisor".into())
        })?;
    terminal.fold_challenge_config = Cfg::ring_challenge_config(terminal_d)?;
    terminal.blocks.live_ring_elements_per_claim = terminal.input_witness_len / terminal_d;
    let terminal_witness_norms =
        FoldWitnessNorms::bounded(terminal.inner.digits.log_basis, terminal_d);
    let ring_dimension = terminal_d
        .try_into()
        .map_err(|_| AkitaError::InvalidSetup("packing terminal dimension exceeds u32".into()))?;
    let mut selected = None;
    for positions_per_block in [256usize, 128, 64, 32, 16, 8, 4, 2, 1] {
        let num_live_blocks = terminal
            .blocks
            .live_ring_elements_per_claim
            .div_ceil(positions_per_block);
        let a_width = positions_per_block
            .checked_mul(terminal.inner.digits.num_digits)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("synthetic terminal A width overflow".into())
            })?;
        let response_width = a_width.checked_mul(terminal_d).ok_or_else(|| {
            AkitaError::InvalidSetup("synthetic terminal response overflow".into())
        })?;
        let fold_digit_count =
            BalancedSignedDigitFoldPolicy::universal(policy.decomposition.field_bits())
                .num_digits_fold(HonestFoldSizingQuery {
                    ring_dimension: terminal_d,
                    challenge_dimension: terminal_d,
                    num_claims: 1,
                    num_live_ring_elements_per_claim: terminal.blocks.live_ring_elements_per_claim,
                    num_live_blocks,
                    num_positions_per_block: positions_per_block,
                    num_chunks: 1,
                    num_fold_coeffs: response_width,
                    witness_norms: terminal_witness_norms,
                    log_basis_response: terminal.fold.log_basis,
                    challenge_config: &terminal.fold_challenge_config,
                })?;
        let Some(a_bound) = akita_types::sis::rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            terminal_d,
            terminal.fold.log_basis,
            &terminal.fold_challenge_config,
            fold_digit_count,
            1,
        ) else {
            continue;
        };
        let Ok(matrix) = akita_types::InnerCommitMatrixParams::try_new_with_min_rank(
            akita_types::SisTableKey {
                policy: policy.sis_security_policy,
                table_digest: policy.sis_table_digest,
                modulus_profile: policy.sis_modulus_profile,
                role: akita_types::SisMatrixRole::Inner,
                ring_dimension,
                coeff_linf_bound: a_bound,
            },
            a_width,
        ) else {
            continue;
        };
        selected = Some((
            positions_per_block,
            num_live_blocks,
            fold_digit_count,
            matrix,
        ));
        break;
    }
    let (positions_per_block, num_live_blocks, fold_digit_count, matrix) =
        selected.ok_or_else(|| {
            AkitaError::InvalidSetup("packing test terminal has no admissible Linf geometry".into())
        })?;
    terminal.blocks.positions_per_block = positions_per_block;
    terminal.blocks.live_blocks = num_live_blocks;
    terminal.fold.num_digits = fold_digit_count;
    terminal.inner.matrix = matrix;
    let encoding_scale = terminal.certified_response_linf_cap()?;
    terminal.response_shape = akita_types::TerminalResponseShape::derive(terminal, encoding_scale)?;
    Ok(())
}

/// Test-only commitment config that combines an envelope config with a final
/// group config.
///
/// Exact grouped runtime keys select schedules under `Final`, retaining each
/// preceding group's frozen native descriptor. Public setup storage remains
/// flat and dimension-free.
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

/// Test-only adapter that exposes an otherwise well-formed early
/// EvaluationTrace row so public prove/verify admission can prove it rejects
/// before transcript mutation. `LEVEL=0` mutates the root; `LEVEL=1` mutates
/// the first recursive witness and its incoming prefix.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EarlyEvaluationTraceConfig<Base, const LEVEL: usize>(PhantomData<fn() -> Base>);

impl<Base> RootCoefficientPackingConfig<Base>
where
    Base: CommitmentConfig + 'static,
{
    pub(crate) fn derive_catalog_row(
        key: &AkitaScheduleLookupKey,
        challenge_subring_dimension: usize,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        if !key.precommitteds.is_empty() {
            return Err(AkitaError::UnsupportedSchedule(
                "the coefficient-packing test catalog supports one root group".into(),
            ));
        }
        let base = Base::resolve_catalog_row_for_key(key)?;
        let successor_template = match base.schedule().recursive_folds.first() {
            Some(successor) => successor.clone(),
            None => {
                let grouped_key = AkitaScheduleLookupKey {
                    final_group: key.final_group,
                    precommitteds: vec![base.profiles().final_group],
                };
                Base::resolve_catalog_row_for_key(&grouped_key)?
                    .schedule()
                    .recursive_folds
                    .first()
                    .cloned()
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "packing Stage 3 test requires a recursive successor template".into(),
                        )
                    })?
            }
        };
        let mut schedule = base.into_schedule();
        let policy = policy_of::<Self>();
        let root = &mut schedule.root.params;
        let d_a = root.inner().matrix.ring_dimension();
        akita_types::SubringCoefficientPackingGeometry::try_new(
            Self::EXT_DEGREE,
            d_a,
            challenge_subring_dimension,
        )?;
        root.own_group_mut().opening.opening_method =
            akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            };
        root.source_encoding = akita_types::CommittedSourceEncoding::CanonicalCoefficientTable;
        root.own_group_mut().opening.fold_challenge_config =
            SparseChallengeConfig::production_for_ring_dim(challenge_subring_dimension)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "root packing subring is not in the production challenge ladder".into(),
                    )
                })?;
        let root_open_dimension = 128usize.min(
            Self::EXT_DEGREE
                .checked_mul(challenge_subring_dimension)
                .ok_or_else(|| AkitaError::InvalidSetup("packing width overflow".into()))?,
        );
        let root_open_bound = akita_types::sis::rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            policy.sis_modulus_profile,
            akita_types::SisMatrixRole::Open,
            root_open_dimension,
            root.open().digits.log_basis,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup("root packing test has no audited D bound".into())
        })?;
        let mut root_open_key = root.open_matrix.sis_table_key();
        root_open_key.ring_dimension = root_open_dimension
            .try_into()
            .map_err(|_| AkitaError::InvalidSetup("root packing D dimension exceeds u32".into()))?;
        root_open_key.coeff_linf_bound = root_open_bound;
        root.open_matrix = akita_types::OpenCommitMatrixParams::try_new_with_min_rank(
            root_open_key,
            root.open_matrix.input_width(),
        )?;
        let required_a_bound = akita_types::sis::rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            d_a,
            root.open().digits.log_basis,
            &root.fold_challenge_config(),
            root.num_digits_fold(),
            root.witness_chunk.num_chunks,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup("root packing challenge family has no audited A bound".into())
        })?;
        let current_a = root.inner().matrix;
        let mut current_key = current_a.sis_table_key().ok_or_else(|| {
            AkitaError::InvalidSetup("root packing requires a L-infinity A matrix".into())
        })?;
        current_key.coeff_linf_bound = required_a_bound;
        root.own_group_mut().profile.inner.matrix =
            akita_types::InnerCommitMatrixParams::try_new_with_min_rank(
                current_key,
                current_a.input_width(),
            )?;
        rebuild_group_output_matrices(root, key.final_group.num_polynomials(), Self::EXT_DEGREE)?;

        let opening_batch = key.opening_layout()?;
        let root_output_witness_len = root.output_witness_len_for_field_bits(
            policy.decomposition.field_bits(),
            Self::EXT_DEGREE,
            &opening_batch,
        )?;
        schedule.root.output_witness_len = root_output_witness_len;

        let mut successor = successor_template;
        successor.input_witness_len = root_output_witness_len;
        let successor_witness = &mut successor.params;
        if successor_witness.inner().digits.log_basis != root.open().digits.log_basis
            || successor_witness.inner().digits.num_digits != 1
        {
            return Err(AkitaError::InvalidSetup(format!(
                "packing recursive digit basis mismatch: predecessor open={}, successor inner={} with {} digits",
                root.open().digits.log_basis,
                successor_witness.inner().digits.log_basis,
                successor_witness.inner().digits.num_digits,
            )));
        }
        successor_witness.own_group_mut().opening.opening_method =
            akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            };
        successor_witness.source_encoding =
            akita_types::CommittedSourceEncoding::CanonicalCoefficientTable;
        successor_witness
            .own_group_mut()
            .opening
            .fold_challenge_config =
            SparseChallengeConfig::production_for_ring_dim(challenge_subring_dimension)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "successor packing subring is not in the production ladder".into(),
                    )
                })?;
        successor_witness
            .own_group_mut()
            .profile
            .blocks
            .live_ring_elements_per_claim =
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
        successor_witness
            .own_group_mut()
            .profile
            .blocks
            .positions_per_block = prefix_ring_slots.next_power_of_two();
        successor_witness.own_group_mut().profile.blocks.live_blocks = successor_witness
            .blocks()
            .live_ring_elements_per_claim
            .div_ceil(successor_witness.blocks().positions_per_block);
        successor_witness.own_group_mut().opening.num_digits_fold = universal_fold_digit_depth(
            FoldDigitInputs::of_fold(successor_witness),
            policy.decomposition.field_bits(),
            1,
            successor_witness.witness_chunk.num_chunks,
        )?;
        let successor_a_width = successor_witness
            .blocks()
            .positions_per_block
            .checked_mul(successor_witness.inner().digits.num_digits)
            .ok_or_else(|| AkitaError::InvalidSetup("packing successor A width overflow".into()))?;
        let mut successor_a_key = successor_witness
            .inner()
            .matrix
            .sis_table_key()
            .ok_or_else(|| {
                AkitaError::InvalidSetup("packing successor requires a L-infinity A matrix".into())
            })?;
        successor_a_key.coeff_linf_bound = akita_types::sis::rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            successor_witness.d_a(),
            successor_witness.open().digits.log_basis,
            &successor_witness.fold_challenge_config(),
            successor_witness.num_digits_fold(),
            successor_witness.witness_chunk.num_chunks,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup("packing successor has no audited A bound".into())
        })?;
        successor_witness.own_group_mut().profile.inner.matrix =
            akita_types::InnerCommitMatrixParams::try_new_with_min_rank(
                successor_a_key,
                successor_a_width,
            )?;
        rebuild_group_output_matrices(successor_witness, 1, Self::EXT_DEGREE)?;

        let outer_slices = successor_witness.outer_slice_count().get();
        let max_prefix_positions = prefix_ring_slots
            .checked_div(outer_slices)
            .filter(|positions| positions.is_power_of_two())
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "packing setup prefix cannot be partitioned across its outer slices".into(),
                )
            })?;
        let prefix_positions = max_prefix_positions.min(256);
        let prefix_blocks = prefix_ring_slots
            .checked_div(prefix_positions)
            .filter(|blocks| *blocks >= outer_slices)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "packing setup prefix has no balanced split: slots={prefix_ring_slots}, positions={prefix_positions}, outer_slices={}",
                    outer_slices,
                ))
            })?;
        let mut prefix_source_params = successor_witness.clone();
        prefix_source_params.set_setup_prefix(None)?;
        prefix_source_params
            .own_group_mut()
            .profile
            .inner
            .digits
            .log_basis = root.own_group_mut().profile.inner.digits.log_basis;
        prefix_source_params
            .own_group_mut()
            .profile
            .inner
            .digits
            .num_digits = akita_types::sis::compute_num_digits_field_width(
            policy.decomposition.field_bits(),
            root.inner().digits.log_basis,
        );
        let prefix_inner_width = prefix_positions
            .checked_mul(prefix_source_params.inner().digits.num_digits)
            .ok_or_else(|| AkitaError::InvalidSetup("packing prefix A width overflow".into()))?;
        prefix_source_params.own_group_mut().profile.inner.matrix =
            akita_types::InnerCommitMatrixParams::try_new_with_min_rank(
                prefix_source_params
                    .inner()
                    .matrix
                    .sis_table_key()
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "packing prefix requires a L-infinity A matrix".into(),
                        )
                    })?,
                prefix_inner_width,
            )?;
        let prefix_outer_width = akita_types::CommitmentSliceGeometry::try_new(
            prefix_source_params.outer_slice_count(),
            prefix_blocks,
            1,
            prefix_source_params.inner().matrix.output_rank(),
            prefix_source_params.outer().digits.num_digits,
            prefix_source_params.d_a(),
            prefix_source_params.role_dims().d_b(),
        )?
        .physical_input_width();
        prefix_source_params.own_group_mut().profile.outer.matrix =
            akita_types::OuterCommitMatrixParams::try_new_with_min_rank(
                prefix_source_params.outer().matrix.sis_table_key(),
                prefix_outer_width,
            )?;
        let mut prefix_params = akita_types::setup_prefix_precommitted_params(
            &prefix_source_params,
            root_setup_prefix_len,
        )?;
        if prefix_params.profile.inner.matrix.ring_dimension() != successor_witness.d_a() {
            return Err(AkitaError::InvalidSetup(
                "packing root setup prefix left the base row's planned commitment class".into(),
            ));
        }
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
        prefix_params.opening.num_digits_fold = universal_fold_digit_depth(
            FoldDigitInputs::of_group(&prefix_params),
            policy.decomposition.field_bits(),
            1,
            successor_witness.witness_chunk.num_chunks,
        )?;
        let prefix_a_bound = akita_types::sis::rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            prefix_params.profile.inner.matrix.ring_dimension(),
            prefix_params.opening.log_basis_open,
            &prefix_params.opening.fold_challenge_config,
            prefix_params.opening.num_digits_fold,
            successor_witness.witness_chunk.num_chunks,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("packing prefix has no audited A bound".into()))?;
        let mut prefix_a_key = prefix_params
            .profile
            .inner
            .matrix
            .sis_table_key()
            .ok_or_else(|| AkitaError::InvalidSetup("packing prefix requires Linf A".into()))?;
        prefix_a_key.coeff_linf_bound = prefix_a_bound;
        prefix_params.profile.inner.matrix =
            akita_types::InnerCommitMatrixParams::try_new_with_min_rank(
                prefix_a_key,
                prefix_params.profile.inner.matrix.input_width(),
            )?;
        let prefix_outer_width = akita_types::CommitmentSliceGeometry::try_new(
            prefix_params.profile.outer_slice_count,
            prefix_params.profile.blocks.live_blocks,
            1,
            prefix_params.profile.inner.matrix.output_rank(),
            prefix_params.profile.outer.digits.num_digits,
            prefix_params.profile.inner.matrix.ring_dimension(),
            prefix_params.profile.outer.matrix.ring_dimension(),
        )?
        .physical_input_width();
        prefix_params.profile.outer.matrix =
            akita_types::OuterCommitMatrixParams::try_new_with_min_rank(
                prefix_params.profile.outer.matrix.sis_table_key(),
                prefix_outer_width,
            )?;
        let incoming_setup_prefix =
            akita_types::scheduled_setup_prefix(root_setup_natural_len, prefix_params);
        successor_witness.set_setup_prefix(Some(incoming_setup_prefix))?;
        let successor_d_width = successor_witness
            .open()
            .matrix
            .input_width()
            .checked_add(
                incoming_setup_prefix
                    .d_segment_width(Self::EXT_DEGREE, successor_witness.role_dims().d_d())?,
            )
            .ok_or_else(|| AkitaError::InvalidSetup("packing successor D width overflow".into()))?;
        successor_witness.open_matrix = akita_types::OpenCommitMatrixParams::try_new_with_min_rank(
            successor_witness.open_matrix.sis_table_key(),
            successor_d_width,
        )?;
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

        retarget_synthetic_terminal::<Self>(&mut schedule)?;

        schedule.validate_nonterminal_opening_execution(Self::EXT_DEGREE)?;
        let root = &schedule.root.params;
        let profiles = CommittedGroupBatchProfile {
            final_group: GroupCommitPhaseParams::try_from_params(key.final_group, root)?,
            precommitteds: Vec::new(),
        };
        let selection = OpeningScheduleSelection {
            row_digest: schedule_row_digest(&profiles, &schedule)?,
        };
        akita_config::ResolvedScheduleRow::try_new(selection, profiles, schedule, &policy)
    }
}

impl<Base, const LEVEL: usize> EarlyEvaluationTraceConfig<Base, LEVEL>
where
    Base: CommitmentConfig + 'static,
{
    fn derive_row(
        key: &AkitaScheduleLookupKey,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        let base = RootCoefficientPackingConfig::<Base>::derive_catalog_row(key, 64)?;
        let profiles = base.profiles().clone();
        let mut schedule = base.into_schedule();
        let params = if LEVEL == 0 {
            &mut schedule.root.params
        } else if LEVEL == 1 {
            let step = schedule.recursive_folds.first_mut().ok_or_else(|| {
                AkitaError::InvalidSetup("early-ET test row needs a recursive fold".into())
            })?;
            if let Some(mut prefix) = step.params.setup_prefix().copied() {
                prefix.opening.opening_method = akita_types::OpeningMethod::EvaluationTrace;
                let d_a = prefix.profile.inner.matrix.ring_dimension();
                prefix.opening.fold_challenge_config =
                    SparseChallengeConfig::production_for_ring_dim(d_a).ok_or_else(|| {
                        AkitaError::InvalidSetup("missing early-ET prefix challenge family".into())
                    })?;
                step.params.set_setup_prefix(Some(prefix))?;
            }
            &mut step.params
        } else {
            return Err(AkitaError::InvalidSetup(
                "early-ET test level must be zero or one".into(),
            ));
        };
        params.own_group_mut().opening.opening_method = akita_types::OpeningMethod::EvaluationTrace;
        params.own_group_mut().opening.fold_challenge_config =
            SparseChallengeConfig::production_for_ring_dim(params.d_a()).ok_or_else(|| {
                AkitaError::InvalidSetup("missing early-ET witness challenge family".into())
            })?;
        params.source_encoding = akita_types::CommittedSourceEncoding::for_producer(
            params.opening_method(),
            Self::EXT_DEGREE,
            params.d_a(),
            0,
            LEVEL == 0,
        );
        if LEVEL == 1 {
            schedule.recursive_folds[0]
                .params
                .own_group_mut()
                .opening
                .fold_challenge_config = params.own_group_mut().opening.fold_challenge_config;
        }
        schedule.validate_nonterminal_opening_execution(Self::EXT_DEGREE)?;
        let selection = OpeningScheduleSelection {
            row_digest: akita_types::schedule_row_digest(&profiles, &schedule)?,
        };
        akita_config::ResolvedScheduleRow::try_new(
            selection,
            profiles,
            schedule,
            &policy_of::<Self>(),
        )
    }
}

impl<Base> CommitmentConfig for RootCoefficientPackingConfig<Base>
where
    Base: CommitmentConfig + 'static,
{
    type Field = Base::Field;
    type ExtField = Base::ExtField;

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

    fn opening_basis_range() -> (u32, u32) {
        Base::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        Base::inner_basis_range()
    }

    fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
        Base::committed_source_class()
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
        Self::derive_catalog_row(key, 64)
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

impl<Base, const LEVEL: usize> CommitmentConfig for EarlyEvaluationTraceConfig<Base, LEVEL>
where
    Base: CommitmentConfig + 'static,
{
    type Field = Base::Field;
    type ExtField = Base::ExtField;

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
        RootCoefficientPackingConfig::<Base>::setup_matrix_capacity(
            max_num_vars,
            max_num_batched_polys,
        )
    }

    fn opening_basis_range() -> (u32, u32) {
        Base::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        Base::inner_basis_range()
    }

    fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
        Base::committed_source_class()
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
        Self::derive_row(key)
    }

    fn resolve_catalog_row_for_profiles(
        profiles: &CommittedGroupBatchProfile,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        Self::derive_row(&AkitaScheduleLookupKey {
            final_group: profiles.final_group.group,
            precommitteds: profiles.precommitteds.clone(),
        })
    }

    fn resolve_schedule_selection(
        selection: OpeningScheduleSelection,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        let row = Self::derive_row(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(20),
        ))?;
        if row.selection() != selection {
            return Err(AkitaError::UnsupportedSchedule(
                "unknown early-ET test row".into(),
            ));
        }
        Ok(row)
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

    fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
        Envelope::committed_source_class()
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
            vec![akita_config::honest_fold_policy_of::<Envelope>(); key.precommitteds.len()];
        let schedule = akita_planner::find_schedule(
            key,
            akita_config::honest_fold_policy_of::<Envelope>(),
            &precommitted_honest_fold_policies,
            &policy,
            ring_challenge_config,
        )?
        .schedule;
        let profiles = CommittedGroupBatchProfile {
            final_group: GroupCommitPhaseParams::try_from_params(
                key.final_group,
                &schedule.root.params,
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
