//! Proof-optimized commitment config presets.
//!
//! Presets are unit structs that bind [`CommitmentConfig`] hooks to
//! [`akita_types`] SIS primitives and generated schedule tables.

use super::CommitmentConfig;
use akita_challenges::MIN_FOLD_CHALLENGE_ENTROPY_BITS;
use akita_field::AkitaError;
use akita_field::{Ext2, FpExt4, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use akita_types::OpeningBatch;
use akita_types::{AkitaScheduleLookupKey, LevelParams, Schedule, SetupMatrixEnvelope, Step};
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Minimum proof-optimized log-basis.
pub(crate) const PROOF_OPTIMIZED_LOG_BASIS_MIN: u32 = 2;
/// Maximum proof-optimized log-basis.
pub(crate) const PROOF_OPTIMIZED_LOG_BASIS_MAX: u32 = 6;

/// Shared short ring-challenge policy for every proof-optimized preset.
///
/// "Short" means bounded norm, which is the property that keeps the folded
/// witness short enough for SIS binding. It does not mean sparse: at `d == 32`
/// the family is `BoundedL1Norm`, a low-norm ball (`||c||_1 <= 121`,
/// `||c||_inf <= 8`) whose elements can be fully dense. The larger degrees use
/// fixed-weight sparse families (`ExactShell` at `d == 64`, `Uniform` at
/// `d >= 128`), where shortness happens to coincide with sparsity.
///
/// The family is keyed only on the ring degree `d`. A preset's `D` is fixed
/// across all schedule levels, so both the planner DP and the generated-table
/// expansion call the per-`Cfg` hook with `d == Cfg::D` (see
/// `akita_planner::find_schedule` and `generated::expand`). Every family
/// returned here has at least 128 bits of Fiat-Shamir support, which is the
/// soundness floor for the witness-folding ring challenge; presets must not
/// pick a lower-support family. fp128 only reaches `d in {32, 64, 128}`; the
/// small-field presets additionally reach `d == 256`.
pub(crate) fn proof_optimized_ring_challenge_config(
    d: usize,
) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
    let cfg = match d {
        32 => akita_challenges::SparseChallengeConfig::BoundedL1Norm,
        64 => akita_challenges::SparseChallengeConfig::ExactShell {
            count_mag1: 30,
            count_mag2: 12,
            // `T >= ||c||_1` disables rejection until the S2 support certificate
            // lands; production keeps the legacy (30, 12) shell unchanged.
            operator_norm_threshold: 54,
        },
        128 => akita_challenges::SparseChallengeConfig::Uniform {
            weight: 31,
            nonzero_coeffs: vec![-1, 1],
        },
        256 => akita_challenges::SparseChallengeConfig::Uniform {
            weight: 23,
            nonzero_coeffs: vec![-1, 1],
        },
        _ => {
            return Err(AkitaError::InvalidSetup(format!(
                "unsupported proof-optimized ring dim {d}"
            )));
        }
    };
    validate_proof_optimized_fold_entropy(&cfg, d)?;
    Ok(cfg)
}

fn validate_proof_optimized_fold_entropy(
    cfg: &akita_challenges::SparseChallengeConfig,
    d: usize,
) -> Result<(), AkitaError> {
    match d {
        32 => cfg.validate::<32>(),
        64 => cfg.validate::<64>(),
        128 => cfg.validate::<128>(),
        256 => cfg.validate::<256>(),
        _ => {
            return Err(AkitaError::InvalidSetup(format!(
                "unsupported proof-optimized ring dim {d}"
            )));
        }
    }
    .map_err(|msg| AkitaError::InvalidSetup(msg.to_string()))?;
    cfg.validate_min_entropy_for_ring_dim(d, MIN_FOLD_CHALLENGE_ENTROPY_BITS)
        .map_err(|msg| AkitaError::InvalidSetup(msg.to_string()))
}

// ---------------------------------------------------------------------------
// `<Cfg>`-generic policy helpers for the planner and materializer.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Trait-shaped wrappers consumed by the macros below.
// ---------------------------------------------------------------------------

/// Size the shared setup matrix from the planned schedule.
///
/// Planned role footprints are not monotone across shapes, so scan all
/// supported sub-shapes and keep the largest packed setup length.
static SETUP_MATRIX_ENVELOPE_CACHE: LazyLock<
    Mutex<HashMap<(TypeId, usize, usize), SetupMatrixEnvelope>>,
> = LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) fn proof_optimized_max_setup_matrix_size<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<SetupMatrixEnvelope, AkitaError> {
    let cache_key = (TypeId::of::<Cfg>(), max_num_vars, max_num_batched_polys);
    if let Some(cached) = SETUP_MATRIX_ENVELOPE_CACHE
        .lock()
        .expect("setup matrix envelope cache poisoned")
        .get(&cache_key)
        .copied()
    {
        return Ok(cached);
    }

    let envelope =
        proof_optimized_max_setup_matrix_size_uncached::<Cfg>(max_num_vars, max_num_batched_polys)?;

    SETUP_MATRIX_ENVELOPE_CACHE
        .lock()
        .expect("setup matrix envelope cache poisoned")
        .insert(cache_key, envelope);

    Ok(envelope)
}

fn proof_optimized_max_setup_matrix_size_uncached<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<SetupMatrixEnvelope, AkitaError> {
    if max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    let mut max_setup_len: usize = 1;
    #[cfg(feature = "zk")]
    let mut max_zk_b_len: usize = 1;
    #[cfg(feature = "zk")]
    let mut max_zk_d_len: usize = 1;
    let mut saw_supported_shape = false;
    let poly_counts = setup_envelope_poly_counts(max_num_batched_polys);
    for num_vars in 1..=max_num_vars {
        for &num_polys in &poly_counts {
            let opening_batch = worst_case_grouped_opening_batch_for_shape(num_vars, num_polys)?;
            let Some(envelope) = setup_matrix_envelope_for_shape::<Cfg>(&opening_batch)? else {
                continue;
            };
            saw_supported_shape = true;
            max_setup_len = max_setup_len.max(envelope.max_setup_len);
            #[cfg(feature = "zk")]
            {
                max_zk_b_len = max_zk_b_len.max(envelope.max_zk_b_len);
                max_zk_d_len = max_zk_d_len.max(envelope.max_zk_d_len);
            }
        }
    }

    if !saw_supported_shape {
        return Err(AkitaError::InvalidSetup(format!(
            "setup matrix sizing found no generated schedules for max_num_vars={max_num_vars}"
        )));
    }

    Ok(SetupMatrixEnvelope {
        max_setup_len,
        #[cfg(feature = "zk")]
        max_zk_b_len,
        #[cfg(feature = "zk")]
        max_zk_d_len,
    })
}

/// Batched polynomial counts scanned by [`proof_optimized_max_setup_matrix_size`].
///
/// Generated schedule tables (and the offline `gen_schedule_tables` emitter)
/// materialize only singleton (`num_polys = 1`) and 4-batched roots. Scanning
/// every intermediate count in `1..=max` forces table misses on `2` and `3` even
/// though setup-matrix footprints are determined by the endpoint batch sizes.
/// Role footprints can be non-monotone in `num_vars`, but not in these skipped
/// intermediate batch counts for the shipped table key shapes.
pub(crate) fn setup_envelope_poly_counts(max_num_batched_polys: usize) -> Vec<usize> {
    if max_num_batched_polys <= 1 {
        vec![1]
    } else {
        vec![1, max_num_batched_polys]
    }
}

/// Worst-case opening batch for a `(num_vars, num_claims)` shape.
pub fn worst_case_grouped_opening_batch_for_shape(
    num_vars: usize,
    num_claims: usize,
) -> Result<OpeningBatch, AkitaError> {
    OpeningBatch::same_point(num_vars, num_claims)
}

fn setup_matrix_envelope_for_shape<Cfg: CommitmentConfig>(
    opening_batch: &OpeningBatch,
) -> Result<Option<SetupMatrixEnvelope>, AkitaError> {
    let cached_key = AkitaScheduleLookupKey::new_from_opening_batch(opening_batch)?;

    // Setup-matrix sizing scans many candidate sub-shapes. `runtime_schedule`
    // serves the shipped table on a hit and regenerates via the planner DP on
    // a miss; a shape the planner cannot schedule (infeasible — e.g. a witness
    // too large for this preset's SIS floor) can never be committed, so it
    // needs no setup capacity. Skip it (returning `Ok(None)`) and let the
    // caller's `saw_supported_shape` guard error only if *no* shape is
    // feasible. Genuine bugs in opening_batch-key or envelope construction still
    // propagate via `?`.
    let Ok(schedule) = Cfg::runtime_schedule(cached_key) else {
        return Ok(None);
    };

    Ok(Some(matrix_envelope_for_schedule::<Cfg>(
        &schedule,
        opening_batch,
    )?))
}

/// Extract setup-level params from a runtime `Schedule`.
///
/// Uncommittable root-direct entries carry no setup params and are skipped
/// here; `Cfg::get_params_for_batched_commitment` rejects them loudly.
pub fn setup_level_params_from_runtime_schedule(steps: &[akita_types::Step]) -> Vec<LevelParams> {
    steps
        .iter()
        .filter_map(|step| match step {
            akita_types::Step::Fold(fold_step) => Some(fold_step.params.clone()),
            akita_types::Step::Direct(direct) => direct.params.clone(),
        })
        .collect()
}

fn matrix_envelope_for_levels<Cfg>(
    setup_levels: &[LevelParams],
) -> Result<SetupMatrixEnvelope, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let mut max_setup_len: usize = 1;
    for lp in setup_levels {
        accumulate_matrix_envelope_for_level::<Cfg>(lp, &mut max_setup_len)?;
    }
    Ok(SetupMatrixEnvelope {
        max_setup_len,
        #[cfg(feature = "zk")]
        max_zk_b_len: 1,
        #[cfg(feature = "zk")]
        max_zk_d_len: 1,
    })
}

/// Packed setup envelope spanning every level in `schedule` (including the
/// root-direct / fold-root opening_batch widening) and, with the `zk` feature,
/// the ZK blinding + hiding accumulators.
pub fn matrix_envelope_for_schedule<Cfg>(
    schedule: &Schedule,
    opening_batch: &OpeningBatch,
) -> Result<SetupMatrixEnvelope, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let setup_levels: Vec<LevelParams> = setup_level_params_from_runtime_schedule(&schedule.steps);
    let mut envelope = matrix_envelope_for_levels::<Cfg>(&setup_levels)?;
    accumulate_root_matrix_envelope_for_opening_batch(
        schedule,
        opening_batch,
        &mut envelope.max_setup_len,
    )?;
    #[cfg(feature = "zk")]
    {
        accumulate_zk_blinding_envelope::<Cfg>(schedule, opening_batch, &mut envelope)?;
        accumulate_zk_hiding_envelope::<Cfg>(schedule, opening_batch, &mut envelope)?;
        Ok(envelope)
    }
    #[cfg(not(feature = "zk"))]
    {
        Ok(envelope)
    }
}

fn accumulate_matrix_envelope_for_level<Cfg: CommitmentConfig>(
    lp: &LevelParams,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    let _cfg_marker = core::marker::PhantomData::<Cfg>;
    let a_len = lp
        .a_key
        .row_len()
        .checked_mul(lp.inner_width())
        .ok_or_else(|| AkitaError::InvalidSetup("A setup envelope overflow".to_string()))?;
    let b_len = lp
        .b_key
        .row_len()
        .checked_mul(lp.outer_width())
        .ok_or_else(|| AkitaError::InvalidSetup("B setup envelope overflow".to_string()))?;
    let d_len = lp
        .d_key
        .row_len()
        .checked_mul(lp.d_matrix_width())
        .ok_or_else(|| AkitaError::InvalidSetup("D setup envelope overflow".to_string()))?;
    let f_len = match lp.f_key.as_ref() {
        Some(fk) => fk
            .row_len()
            .checked_mul(fk.col_len())
            .ok_or_else(|| AkitaError::InvalidSetup("F setup envelope overflow".to_string()))?,
        None => 0,
    };
    *max_setup_len = (*max_setup_len).max(a_len).max(b_len).max(d_len).max(f_len);
    Ok(())
}

fn accumulate_root_matrix_envelope_for_opening_batch(
    schedule: &Schedule,
    opening_batch: &OpeningBatch,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    let Some(root_params) = root_commit_params_from_schedule(schedule)? else {
        return Ok(());
    };
    let root_len = root_runtime_matrix_len_for_opening_batch(&root_params, opening_batch)?;
    *max_setup_len = (*max_setup_len).max(root_len);
    Ok(())
}

fn root_runtime_matrix_len_for_opening_batch(
    lp: &LevelParams,
    opening_batch: &OpeningBatch,
) -> Result<usize, AkitaError> {
    let num_claims = opening_batch.num_claims();
    let max_group_poly_count = opening_batch
        .num_polys_per_commitment_group()
        .iter()
        .copied()
        .max()
        .ok_or_else(|| AkitaError::InvalidSetup("empty opening batch".to_string()))?;
    let d_width = lp
        .num_blocks
        .checked_mul(num_claims)
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("batched D setup width overflow".to_string()))?;
    let t_cols_per_vector = lp
        .a_key
        .row_len()
        .checked_mul(lp.num_digits_open)
        .and_then(|n| n.checked_mul(lp.num_blocks))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("batched B setup vector width overflow".to_string())
        })?;
    let full_b_width = max_group_poly_count
        .checked_mul(t_cols_per_vector)
        .ok_or_else(|| AkitaError::InvalidSetup("batched B setup width overflow".to_string()))?;
    let d_len =
        lp.d_key.row_len().checked_mul(d_width).ok_or_else(|| {
            AkitaError::InvalidSetup("batched D setup envelope overflow".to_string())
        })?;
    let b_len = lp
        .b_key
        .row_len()
        .checked_mul(full_b_width.div_ceil(lp.tier_split.max(1)))
        .ok_or_else(|| AkitaError::InvalidSetup("batched B setup envelope overflow".to_string()))?;
    let f_len = match lp.f_key.as_ref() {
        Some(fk) => fk.row_len().checked_mul(fk.col_len()).ok_or_else(|| {
            AkitaError::InvalidSetup("batched F setup envelope overflow".to_string())
        })?,
        None => 0,
    };
    Ok(b_len.max(d_len).max(f_len))
}

#[cfg(feature = "zk")]
fn accumulate_zk_blinding_envelope<Cfg: CommitmentConfig>(
    schedule: &Schedule,
    _opening_batch: &OpeningBatch,
    envelope: &mut SetupMatrixEnvelope,
) -> Result<(), AkitaError> {
    for lp in setup_level_params_from_runtime_schedule(&schedule.steps) {
        let b_planes = akita_types::zk::blinding_digit_plane_count::<Cfg::Field>(
            lp.b_key.row_len(),
            lp.ring_dimension,
            lp.log_basis,
        );
        let b_len =
            lp.b_key.row_len().checked_mul(b_planes).ok_or_else(|| {
                AkitaError::InvalidSetup("ZK B setup envelope overflow".to_string())
            })?;
        let d_planes = akita_types::zk::blinding_digit_plane_count::<Cfg::Field>(
            lp.d_key.row_len(),
            lp.ring_dimension,
            lp.log_basis,
        );
        let d_len =
            lp.d_key.row_len().checked_mul(d_planes).ok_or_else(|| {
                AkitaError::InvalidSetup("ZK D setup envelope overflow".to_string())
            })?;
        envelope.max_zk_b_len = envelope.max_zk_b_len.max(b_len);
        envelope.max_zk_d_len = envelope.max_zk_d_len.max(d_len);
    }
    Ok(())
}

#[cfg(feature = "zk")]
fn accumulate_zk_hiding_envelope<Cfg: CommitmentConfig>(
    schedule: &Schedule,
    opening_batch: &OpeningBatch,
    envelope: &mut SetupMatrixEnvelope,
) -> Result<(), AkitaError> {
    let Some(root_commit_params) = root_commit_params_from_schedule(schedule)? else {
        return Ok(());
    };
    let hiding_len = zk_hiding_witness_len::<Cfg>(schedule, opening_batch)?;
    let num_ring = hiding_len.div_ceil(Cfg::D).max(1).next_power_of_two();
    let hiding_params = root_commit_params.with_decomp(
        num_ring.trailing_zeros() as usize,
        0,
        root_commit_params.num_digits_commit,
        root_commit_params.num_digits_open,
        num_ring,
    )?;
    accumulate_matrix_envelope_for_level::<Cfg>(&hiding_params, &mut envelope.max_setup_len)?;

    let b_blinding_cols = akita_types::zk::blinding_digit_plane_count::<Cfg::Field>(
        hiding_params.b_key.row_len(),
        hiding_params.ring_dimension,
        hiding_params.log_basis,
    );
    let b_len = hiding_params
        .b_key
        .row_len()
        .checked_mul(b_blinding_cols)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("ZK hiding B setup envelope overflow".to_string())
        })?;
    envelope.max_zk_b_len = envelope.max_zk_b_len.max(b_len);
    Ok(())
}

fn root_commit_params_from_schedule(
    schedule: &Schedule,
) -> Result<Option<LevelParams>, AkitaError> {
    match schedule.steps.first() {
        Some(Step::Fold(root_step)) => Ok(Some(root_step.params.clone())),
        Some(Step::Direct(direct)) => Ok(direct.params.clone()),
        None => Err(AkitaError::InvalidSetup(
            "schedule has no steps".to_string(),
        )),
    }
}

#[cfg(feature = "zk")]
fn zk_hiding_witness_len<Cfg: CommitmentConfig>(
    schedule: &Schedule,
    opening_batch: &OpeningBatch,
) -> Result<usize, AkitaError> {
    let fold_steps = schedule
        .steps
        .iter()
        .filter_map(|step| match step {
            Step::Fold(fold) => Some(fold),
            Step::Direct(_) => None,
        })
        .collect::<Vec<_>>();
    let mut len = 0usize;

    if root_tensor_projection_enabled_for_cfg::<Cfg>(opening_batch.num_vars()) {
        let split_bits = Cfg::EXT_DEGREE.trailing_zeros() as usize;
        let rounds = opening_batch
            .num_vars()
            .checked_sub(split_bits)
            .ok_or_else(|| AkitaError::InvalidSetup("ZK projection round underflow".to_string()))?;
        let partials = opening_batch
            .num_claims()
            .checked_mul(Cfg::EXT_DEGREE)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("ZK projection partial overflow".to_string())
            })?;
        add_zk_extension_reduction_slots::<Cfg>(&mut len, partials, rounds)?;
    }

    len = len
        .checked_add(Cfg::D)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK hiding witness overflow".to_string()))?;

    if let Some(root_step) = fold_steps.first() {
        let root_has_stage1 = fold_steps.len() > 1;
        add_zk_level_pad_slots::<Cfg>(
            &mut len,
            &root_step.params,
            root_step.next_w_len,
            root_has_stage1,
        )?;
        if root_has_stage1 {
            add_zk_ext_scalar_slots::<Cfg>(&mut len, 1)?;
        }
        let mut current_opening_vars =
            akita_types::sumcheck_rounds(root_step.params.ring_dimension, root_step.next_w_len);
        for (step_idx, step) in fold_steps.iter().enumerate().skip(1) {
            if Cfg::EXT_DEGREE > 1 {
                let split_bits = Cfg::EXT_DEGREE.trailing_zeros() as usize;
                let rounds = current_opening_vars
                    .checked_sub(split_bits)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "ZK recursive projection round underflow".to_string(),
                        )
                    })?;
                add_zk_extension_reduction_slots::<Cfg>(&mut len, Cfg::EXT_DEGREE, rounds)?;
            }
            len = len.checked_add(Cfg::D).ok_or_else(|| {
                AkitaError::InvalidSetup("ZK recursive mask overflow".to_string())
            })?;
            let include_stage1 = step_idx + 1 < fold_steps.len();
            add_zk_level_pad_slots::<Cfg>(&mut len, &step.params, step.next_w_len, include_stage1)?;
            if include_stage1 {
                add_zk_ext_scalar_slots::<Cfg>(&mut len, 1)?;
            }
            current_opening_vars =
                akita_types::sumcheck_rounds(step.params.ring_dimension, step.next_w_len);
        }
    }

    Ok(len)
}

#[cfg(feature = "zk")]
fn root_tensor_projection_enabled_for_cfg<Cfg: CommitmentConfig>(num_vars: usize) -> bool {
    let width = Cfg::EXT_DEGREE;
    let Some(double_width) = width.checked_mul(2) else {
        return false;
    };
    width > 1
        && width == Cfg::EXT_DEGREE
        && width.is_power_of_two()
        && Cfg::D.is_power_of_two()
        && Cfg::D >= double_width
        && Cfg::D.is_multiple_of(width)
        && num_vars >= Cfg::D.trailing_zeros() as usize
}

#[cfg(feature = "zk")]
fn add_zk_level_pad_slots<Cfg: CommitmentConfig>(
    len: &mut usize,
    params: &LevelParams,
    next_w_len: usize,
    include_stage1: bool,
) -> Result<(), AkitaError> {
    let rounds = akita_types::sumcheck_rounds(params.ring_dimension, next_w_len);
    if include_stage1 {
        let b = 1usize
            .checked_shl(params.log_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("ZK stage-1 basis overflow".to_string()))?;
        for shape in akita_types::stage1_tree_stage_shapes(rounds, b) {
            let stored_coeffs = shape.sumcheck_proof.1.max(1);
            add_zk_ext_scalar_slots::<Cfg>(
                len,
                shape
                    .sumcheck_proof
                    .0
                    .checked_mul(stored_coeffs)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("ZK stage-1 pad overflow".to_string())
                    })?,
            )?;
            add_zk_ext_scalar_slots::<Cfg>(len, shape.child_claims)?;
        }
    }
    add_zk_ext_scalar_slots::<Cfg>(
        len,
        rounds
            .checked_mul(3)
            .ok_or_else(|| AkitaError::InvalidSetup("ZK stage-2 pad overflow".to_string()))?,
    )
}

#[cfg(feature = "zk")]
fn add_zk_extension_reduction_slots<Cfg: CommitmentConfig>(
    len: &mut usize,
    partials: usize,
    rounds: usize,
) -> Result<(), AkitaError> {
    let reduction_scalars = rounds
        .checked_mul(akita_types::EXTENSION_OPENING_REDUCTION_DEGREE)
        .and_then(|n| n.checked_add(partials))
        .ok_or_else(|| AkitaError::InvalidSetup("ZK extension pad overflow".to_string()))?;
    add_zk_ext_scalar_slots::<Cfg>(len, reduction_scalars)
}

#[cfg(feature = "zk")]
fn add_zk_ext_scalar_slots<Cfg: CommitmentConfig>(
    len: &mut usize,
    scalars: usize,
) -> Result<(), AkitaError> {
    let slots = scalars
        .checked_mul(Cfg::EXT_DEGREE)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK scalar pad overflow".to_string()))?;
    *len = len
        .checked_add(slots)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK hiding witness overflow".to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-preset CommitmentConfig macro
// ---------------------------------------------------------------------------

/// Generate a [`CommitmentConfig`] impl for one proof-optimized preset.
///
/// One macro covers every proof-optimized preset (fp128 and the small-field
/// fp32/fp64 families): the fp128 presets are the special case where the
/// extension field is the base field, `field_bits == 128`, and the SIS
/// family is `Q128`. All proof-optimized presets share `log_basis = 3`, the
/// shared ring-challenge policy, the shared setup-matrix sizer, and the
/// `[PROOF_OPTIMIZED_LOG_BASIS_MIN, MAX]` basis range, so those are not
/// parameters.
macro_rules! impl_proof_optimized_preset {
    (@onehot_chunk_size $onehot_chunk_size:expr) => {
        $onehot_chunk_size
    };
    (@onehot_chunk_size) => {
        1
    };
    (@tiered $tiered:expr) => {
        $tiered
    };
    (@tiered) => {
        false
    };
    ($cfg:ident, $field:ty, $claim_field:ty, $family:expr, $d:expr, $field_bits:expr, $log_commit_bound:expr $(, $onehot_chunk_size:expr $(, $tiered:expr)?)?) => {
        impl $crate::CommitmentConfig for $cfg {
            type Field = $field;
            type ExtField = $claim_field;
            const D: usize = $d;

            // Defaults to `false`; the tiered preset(s) pass `true` as the
            // optional trailing arg.
            const TIERED_COMMITMENT: bool =
                impl_proof_optimized_preset!(@tiered $($($tiered)?)?);

            fn decomposition() -> akita_types::DecompositionParams {
                // Proof-optimized presets fold at `log_basis = 3` and set
                // `log_open_bound = Some(field_bits)` unless the gadget already
                // saturates the field (`log_commit_bound == field_bits`).
                akita_types::DecompositionParams {
                    log_basis: 3,
                    log_commit_bound: $log_commit_bound,
                    log_open_bound: if $log_commit_bound < $field_bits {
                        Some($field_bits)
                    } else {
                        None
                    },
                }
            }

            fn ring_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_ring_challenge_config(d)
            }

            fn sis_modulus_family() -> akita_types::SisModulusFamily {
                $family
            }

            fn max_setup_matrix_size(
                max_num_vars: usize,
                max_num_batched_polys: usize,
            ) -> Result<akita_types::SetupMatrixEnvelope, akita_field::AkitaError> {
                $crate::proof_optimized::proof_optimized_max_setup_matrix_size::<Self>(
                    max_num_vars,
                    max_num_batched_polys,
                )
            }

            fn basis_range() -> (u32, u32) {
                (
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MIN,
                    $crate::proof_optimized::PROOF_OPTIMIZED_LOG_BASIS_MAX,
                )
            }

            fn onehot_chunk_size() -> usize {
                impl_proof_optimized_preset!(@onehot_chunk_size $($onehot_chunk_size)?)
            }
        }
    };
}

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Public preset structs
// ---------------------------------------------------------------------------

pub mod fp128;
pub mod fp32;
pub mod fp64;
