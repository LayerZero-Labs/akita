//! Packing-fixture digit sizing, output matrices, and terminal geometry.
use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_params::sis::{
    BalancedSignedDigitFoldPolicy, FoldWitnessNorms, HonestFoldPolicy, HonestFoldSizingQuery,
};
use akita_schedules::PlannerPolicy;

pub(super) fn rebuild_group_output_matrices(
    params: &mut akita_params::CommittedGroupParams,
    num_claims: usize,
    extension_degree: usize,
) -> Result<(), AkitaError> {
    let dims = params.role_dims();
    let outer_width = akita_params::CommitmentSliceGeometry::try_new(
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
        akita_params::OuterCommitMatrixParams::try_new_with_min_rank(
            params.outer().matrix.sis_table_key(),
            outer_width,
        )?;
    let d_width = akita_params::opening_d_segment_width(
        params.opening_method(),
        extension_degree,
        dims.d_a(),
        dims.d_d(),
        params.open().digits.num_digits,
        params.blocks().live_blocks,
        num_claims,
    )?;
    params.open_matrix = akita_params::OpenCommitMatrixParams::try_new_with_min_rank(
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
pub(super) struct FoldDigitInputs {
    d_a: usize,
    log_basis_inner: u32,
    log_basis_open: u32,
    num_digits_inner: usize,
    num_positions_per_block: usize,
    num_live_blocks: usize,
    num_live_ring_elements_per_claim: usize,
    opening_method: akita_params::OpeningMethod,
    fold_challenge_config: akita_challenges::SparseChallengeConfig,
}

impl FoldDigitInputs {
    pub(super) fn of_fold(params: &akita_params::CommittedGroupParams) -> Self {
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

    pub(super) fn of_group(params: &akita_params::GroupOpenPhaseParams) -> Self {
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

pub(super) fn universal_fold_digit_depth(
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
            akita_params::OpeningMethod::EvaluationTrace => d_a,
            akita_params::OpeningMethod::SubringCoefficientPacking {
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

pub(super) fn retarget_synthetic_terminal<Cfg: CommitmentConfig>(
    schedule: &mut akita_params::FoldSchedule,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
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
        let Some(a_bound) = akita_params::sis::rounded_up_role_a_inf_norm(
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
        let Ok(matrix) = akita_params::InnerCommitMatrixParams::try_new_with_min_rank(
            akita_params::SisTableKey {
                policy: policy.sis_security_policy,
                table_digest: policy.sis_table_digest,
                modulus_profile: policy.sis_modulus_profile,
                role: akita_params::SisMatrixRole::Inner,
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
    terminal.response_shape =
        akita_params::TerminalResponseShape::derive(terminal, encoding_scale)?;
    Ok(())
}
