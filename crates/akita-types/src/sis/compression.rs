//! Commitment-compression SIS sizing and plan construction.
//!
//! Compression maps are scalar/unstructured SIS instances. Output dimension is
//! sized through the same generated [`super::min_secure_rank`] tables as A/B/D,
//! using a fixed module dimension of [`COMPRESSION_LOOKUP_D`] so typical fp128
//! payloads land in the ~192–512 byte range (12–32 field elements), not a single
//! ring row.

use akita_field::AkitaError;

use super::decomposition_digits::num_digits_open;
use super::norm_bound::rounded_up_collision_linf_t;
use super::{
    min_secure_rank, sis_table_key_for_linf_bound, SisModulusFamily, DEFAULT_SIS_SECURITY_BITS,
};
use crate::{
    CommitmentCompressionPlan, CompressionLayerPlan, CompressionMapRole, DecompositionParams,
};

/// Module dimension used when looking up scalar compression output ranks in the
/// generated SIS tables. This is not the protocol ring degree; it is the SIS
/// table coordinate for unstructured digit witnesses.
pub const COMPRESSION_LOOKUP_D: u32 = 32;

/// Planner/runtime policy for commitment compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompressionPolicy {
    /// When false, every commitment stays ring-shaped on the wire.
    pub enabled: bool,
    /// Maximum number of compression map layers per commitment. The planner
    /// evaluates `0..=max_layers` and picks the cheapest total-bytes option.
    /// This caps chain depth (e.g. `Decompose → F₀ → Decompose → F₁`), not the
    /// number of fold levels that compress.
    pub max_layers: usize,
}

impl Default for CompressionPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            max_layers: 2,
        }
    }
}

impl CompressionPolicy {
    /// Policy with compression disabled (tests and root-direct schedules).
    pub const DISABLED: Self = Self {
        enabled: false,
        max_layers: 0,
    };
}

/// Minimum secure scalar output length for one compression layer.
///
/// Returns the number of base-field elements in the public compressed payload
/// for a digit witness of length `input_digit_len` at gadget base `log_basis`.
pub fn secure_compression_output_len(
    sis_family: SisModulusFamily,
    log_basis: u32,
    input_digit_len: usize,
) -> Option<usize> {
    if input_digit_len == 0 {
        return None;
    }
    let collision = rounded_up_collision_linf_t(
        DEFAULT_SIS_SECURITY_BITS,
        sis_family,
        COMPRESSION_LOOKUP_D as usize,
        log_basis,
    )?;
    let width = input_digit_len.div_ceil(COMPRESSION_LOOKUP_D as usize) as u64;
    let key = sis_table_key_for_linf_bound(
        DEFAULT_SIS_SECURITY_BITS,
        sis_family,
        COMPRESSION_LOOKUP_D,
        collision,
    )?;
    let rank = min_secure_rank(key, width)?;
    Some(rank.saturating_mul(COMPRESSION_LOOKUP_D as usize))
}

fn pad_suffix_len(suffix_len: usize, ring_dimension: usize) -> Result<usize, AkitaError> {
    if suffix_len == 0 {
        return Ok(0);
    }
    if ring_dimension == 0 {
        return Err(AkitaError::InvalidSetup(
            "compression suffix padding requires positive ring dimension".to_string(),
        ));
    }
    suffix_len
        .checked_add(ring_dimension - 1)
        .and_then(|n| n.checked_div(ring_dimension))
        .and_then(|q| q.checked_mul(ring_dimension))
        .ok_or_else(|| AkitaError::InvalidSetup("compression suffix padding overflow".to_string()))
}

/// Build one compression plan with exactly `num_layers` active maps (0 = identity).
pub struct CompressionPlanBuildRequest {
    pub role: CompressionMapRole,
    pub raw_len: usize,
    pub num_digits: usize,
    pub log_basis: u32,
    pub sis_family: SisModulusFamily,
    pub ring_dimension: usize,
    pub num_layers: usize,
    pub setup_offset: usize,
}

pub fn build_compression_plan_with_layers(
    request: CompressionPlanBuildRequest,
) -> Result<(CommitmentCompressionPlan, usize), AkitaError> {
    let CompressionPlanBuildRequest {
        role,
        raw_len,
        num_digits,
        log_basis,
        sis_family,
        ring_dimension,
        num_layers,
        setup_offset,
    } = request;
    if num_layers == 0 {
        return Ok((
            CommitmentCompressionPlan {
                raw_len,
                public_len: raw_len,
                suffix_len: 0,
                padded_suffix_len: 0,
                layers: Vec::new(),
            },
            setup_offset,
        ));
    }
    if raw_len == 0 || num_digits == 0 {
        return Err(AkitaError::InvalidSetup(
            "compression plan requires nonzero raw payload and digit depth".to_string(),
        ));
    }

    let mut layers = Vec::with_capacity(num_layers);
    let mut suffix_len = 0usize;
    let mut setup_cursor = setup_offset;
    let mut current_digits = raw_len.checked_mul(num_digits).ok_or_else(|| {
        AkitaError::InvalidSetup("compression first-layer digit width overflow".to_string())
    })?;

    for layer_idx in 0..num_layers {
        let output_len = secure_compression_output_len(sis_family, log_basis, current_digits)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "no secure compression output rank for role={role:?} layer={layer_idx} \
                     input_digits={current_digits}"
                ))
            })?;
        suffix_len = suffix_len.checked_add(current_digits).ok_or_else(|| {
            AkitaError::InvalidSetup("compression suffix length overflow".to_string())
        })?;
        let matrix_entries = output_len
            .checked_mul(current_digits)
            .ok_or_else(|| AkitaError::InvalidSetup("compression map size overflow".to_string()))?;
        layers.push(CompressionLayerPlan {
            role,
            layer: layer_idx,
            input_len: current_digits,
            output_len,
            setup_offset: setup_cursor,
        });
        setup_cursor = setup_cursor.checked_add(matrix_entries).ok_or_else(|| {
            AkitaError::InvalidSetup("compression setup cursor overflow".to_string())
        })?;
        if layer_idx + 1 == num_layers {
            break;
        }
        current_digits = output_len.checked_mul(num_digits).ok_or_else(|| {
            AkitaError::InvalidSetup("compression intermediate digit width overflow".to_string())
        })?;
    }

    let public_len = layers
        .last()
        .map(|layer| layer.output_len)
        .unwrap_or(raw_len);
    let padded_suffix_len = pad_suffix_len(suffix_len, ring_dimension)?;

    Ok((
        CommitmentCompressionPlan {
            raw_len,
            public_len,
            suffix_len,
            padded_suffix_len,
            layers,
        },
        setup_cursor,
    ))
}

/// Score one candidate plan for planner DP (lower is better).
#[must_use]
pub fn compression_plan_public_bytes(plan: &CommitmentCompressionPlan, elem_bytes: usize) -> usize {
    plan.public_len.saturating_mul(elem_bytes)
}

/// Suffix witness growth in i8 digits appended to the next recursive witness.
#[must_use]
pub fn compression_plan_suffix_digits(plan: Option<&CommitmentCompressionPlan>) -> usize {
    plan.map(|plan| plan.padded_suffix_len).unwrap_or(0)
}

/// Pick the cheapest plan among `0..=policy.max_layers` layers.
///
/// Returns `None` when compression is disabled, when no secure plan exists, or
/// when compression would not shrink the public payload.
pub struct CompressionPlanRequest<'a> {
    pub policy: &'a CompressionPolicy,
    pub role: CompressionMapRole,
    pub raw_len: usize,
    pub log_basis: u32,
    pub decomp: DecompositionParams,
    pub sis_family: SisModulusFamily,
    pub ring_dimension: usize,
    pub setup_offset: usize,
}

pub fn plan_commitment_compression(
    request: CompressionPlanRequest<'_>,
) -> Result<(Option<CommitmentCompressionPlan>, usize), AkitaError> {
    let CompressionPlanRequest {
        policy,
        role,
        raw_len,
        log_basis,
        decomp,
        sis_family,
        ring_dimension,
        setup_offset,
    } = request;
    if !policy.enabled || policy.max_layers == 0 || raw_len == 0 {
        return Ok((None, setup_offset));
    }
    let num_digits = num_digits_open(DecompositionParams {
        log_basis,
        ..decomp
    });
    if num_digits == 0 {
        return Ok((None, setup_offset));
    }

    let mut best: Option<(CommitmentCompressionPlan, usize)> = None;
    for num_layers in 0..=policy.max_layers {
        let (candidate, next_setup) =
            build_compression_plan_with_layers(CompressionPlanBuildRequest {
                role,
                raw_len,
                num_digits,
                log_basis,
                sis_family,
                ring_dimension,
                num_layers,
                setup_offset,
            })?;
        if num_layers > 0 && candidate.public_len >= raw_len {
            continue;
        }
        let replace = match &best {
            None => true,
            Some((prev, _)) => {
                candidate.public_len < prev.public_len
                    || (candidate.public_len == prev.public_len
                        && candidate.suffix_len < prev.suffix_len)
            }
        };
        if replace {
            best = Some((candidate, next_setup));
        }
    }

    Ok(match best {
        Some((plan, _next_setup)) if plan.layers.is_empty() => (None, setup_offset),
        Some((plan, next_setup)) => (Some(plan), next_setup),
        None => (None, setup_offset),
    })
}

/// Total scalar setup field elements required by all layers in `plan`.
#[must_use]
pub fn compression_plan_setup_field_len(plan: &CommitmentCompressionPlan) -> usize {
    plan.layers
        .iter()
        .map(|layer| layer.output_len.saturating_mul(layer.input_len))
        .sum()
}

/// Sum setup bytes for every compression plan reachable from `schedule`.
pub fn compression_setup_field_len_for_schedule(
    root_compression: Option<&CommitmentCompressionPlan>,
    steps: &[crate::Step],
) -> usize {
    let mut total = 0usize;
    if let Some(plan) = root_compression {
        total = total.saturating_add(compression_plan_setup_field_len(plan));
    }
    for step in steps {
        if let crate::Step::Fold(fold) = step {
            if let Some(plan) = fold.compression.v.as_ref() {
                total = total.saturating_add(compression_plan_setup_field_len(plan));
            }
            if let Some(plan) = fold.compression.next_u.as_ref() {
                total = total.saturating_add(compression_plan_setup_field_len(plan));
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompressionMapRole;

    #[test]
    fn fp128_compression_output_len_is_nontrivial() {
        let output = secure_compression_output_len(SisModulusFamily::Q128, 4, 128)
            .expect("secure output len");
        assert!(
            (12..=64).contains(&output),
            "expected SIS-sized scalar output, got {output}"
        );
    }

    #[test]
    fn planner_picks_layered_plan_when_it_shrinks_public_len() {
        let policy = CompressionPolicy {
            enabled: true,
            max_layers: 2,
        };
        let decomp = DecompositionParams {
            log_basis: 4,
            log_commit_bound: 1,
            log_open_bound: Some(8),
        };
        let (plan, _) = plan_commitment_compression(CompressionPlanRequest {
            policy: &policy,
            role: CompressionMapRole::H,
            raw_len: 64,
            log_basis: 4,
            decomp,
            sis_family: SisModulusFamily::Q128,
            ring_dimension: 64,
            setup_offset: 0,
        })
        .expect("plan");
        let plan = plan.expect("expected compressed plan");
        assert!(plan.public_len < 64, "compression should shrink raw_len=64");
        assert!(plan.public_len >= 12);
    }
}
