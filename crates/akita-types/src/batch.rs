//! Shared batching and root-opening helper types.

use crate::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AkitaExpandedSetup,
    AppendToTranscript, BasisMode, BlockOrder, LevelParams, RingCommitment, RingOpeningPoint,
};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_BATCH_SHAPE, ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
use akita_transcript::Transcript;

/// Multipoint batch layout derived from verifier/prover input grouping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiPointBatchShape {
    /// Number of commitment groups at each opening point.
    pub point_group_sizes: Vec<usize>,
    /// Number of claimed polynomial openings in each commitment group.
    pub claim_group_sizes: Vec<usize>,
    /// Opening-point index for each flattened claim.
    pub claim_to_point: Vec<usize>,
}

/// Root-level opening point prepared for ring-level replay.
#[derive(Debug, Clone)]
pub struct PreparedRootOpeningPoint<F: FieldCore, const D: usize> {
    /// Opening point padded to the root verifier's target variable count.
    pub padded_point: Vec<F>,
    /// Ring-level outer opening point.
    pub ring_opening_point: RingOpeningPoint<F>,
    /// Inner ring-slot reduction.
    pub inner_reduction: CyclotomicRing<F, D>,
}

/// Flatten commitment rows in group order.
pub fn flatten_batched_commitment_rows<F: FieldCore, const D: usize>(
    commitments: &[RingCommitment<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    commitments
        .iter()
        .flat_map(|commitment| commitment.u.iter().copied())
        .collect()
}

/// Absorb batched commitments into the transcript in group order.
pub fn append_batched_commitments_to_transcript<F, T, const D: usize>(
    commitments: &[RingCommitment<F, D>],
    transcript: &mut T,
) where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    for commitment in commitments {
        commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
    }
}

/// Sum claim-group sizes with overflow checking.
///
/// # Errors
///
/// Returns an error if the total claim count overflows `usize`.
pub fn checked_total_claims(group_sizes: &[usize], label: &str) -> Result<usize, AkitaError> {
    group_sizes.iter().try_fold(0usize, |acc, &group_size| {
        acc.checked_add(group_size)
            .ok_or_else(|| AkitaError::InvalidInput(format!("{label} total claim count overflow")))
    })
}

/// Validate common batched prove/verify input shape constraints.
///
/// # Errors
///
/// Returns an error if the batch is empty, has inconsistent opening-point
/// dimensions, has empty groups, exceeds setup capacity, or overflows its
/// flattened claim count.
pub fn validate_batched_inputs<F, G, Len>(
    setup: &AkitaExpandedSetup<F>,
    inputs: &[(&[F], Vec<G>)],
    group_claim_len: Len,
    for_prover: bool,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    Len: Fn(&G) -> usize,
{
    let label = if for_prover {
        "batched_prove"
    } else {
        "batched_verify"
    };
    let shape_error = |message| {
        if for_prover {
            AkitaError::InvalidInput(message)
        } else {
            AkitaError::InvalidProof
        }
    };

    if inputs.is_empty() {
        return Err(shape_error(format!(
            "{label} requires at least one opening point"
        )));
    }
    let num_vars = inputs[0].0.len();
    if inputs.iter().any(|(point, _)| point.len() != num_vars) {
        return Err(shape_error(format!(
            "{label} requires all opening points to have the same length"
        )));
    }
    if num_vars > setup.seed.max_num_vars {
        return Err(AkitaError::InvalidInput(format!(
            "{label} received opening points with {} variables but setup supports at most {}",
            num_vars, setup.seed.max_num_vars
        )));
    }
    if inputs.len() > setup.seed.max_num_points {
        if for_prover {
            return Err(AkitaError::InvalidInput(format!(
                "batched_prove received {} opening points but setup supports at most {}",
                inputs.len(),
                setup.seed.max_num_points
            )));
        }
        return Err(AkitaError::InvalidProof);
    }

    let mut num_claims = 0usize;
    for (point_idx, (_, groups)) in inputs.iter().enumerate() {
        if groups.is_empty() {
            return Err(shape_error(format!(
                "{label} point {point_idx} must have at least one committed group",
            )));
        }
        for group in groups {
            let group_claims = group_claim_len(group);
            if group_claims == 0 {
                return Err(shape_error(format!(
                    "{label} point {point_idx} must have at least one item",
                )));
            }
            num_claims = num_claims
                .checked_add(group_claims)
                .ok_or_else(|| shape_error(format!("{label} total claim count overflow")))?;
        }
    }
    if num_claims > setup.seed.max_num_batched_polys {
        if for_prover {
            return Err(AkitaError::InvalidInput(format!(
                "batched_prove received {num_claims} polynomials but setup supports at most {}",
                setup.seed.max_num_batched_polys
            )));
        }
        return Err(AkitaError::InvalidProof);
    }

    Ok(())
}

/// Absorb the multipoint batch shape into the transcript.
pub fn append_batch_shape_to_transcript<F, T>(
    point_group_sizes: &[usize],
    claim_group_sizes: &[usize],
    transcript: &mut T,
) where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_serde(ABSORB_BATCH_SHAPE, &point_group_sizes.len());
    for group_count in point_group_sizes {
        transcript.append_serde(ABSORB_BATCH_SHAPE, group_count);
    }
    for claim_count in claim_group_sizes {
        transcript.append_serde(ABSORB_BATCH_SHAPE, claim_count);
    }
}

/// Sum point-group sizes with non-empty and overflow checks.
///
/// # Errors
///
/// Returns an error if any point group is empty or the total group count
/// overflows `usize`.
pub fn checked_total_groups(point_group_sizes: &[usize], label: &str) -> Result<usize, AkitaError> {
    if point_group_sizes.is_empty() || point_group_sizes.contains(&0) {
        return Err(AkitaError::InvalidInput(format!(
            "{label} requires nonempty point group sizes"
        )));
    }
    point_group_sizes.iter().try_fold(0usize, |acc, &size| {
        acc.checked_add(size)
            .ok_or_else(|| AkitaError::InvalidInput(format!("{label} group count overflow")))
    })
}

/// Prepare a root opening point for ring-level verification/proving.
///
/// # Errors
///
/// Returns an error if the target variable count overflows, the opening point
/// is too long, or the field-to-ring reduction rejects the point dimensions.
pub fn prepare_root_opening_point<F, const D: usize>(
    opening_point: &[F],
    basis: BasisMode,
    lp: &LevelParams,
    alpha_bits: usize,
) -> Result<PreparedRootOpeningPoint<F, D>, AkitaError>
where
    F: FieldCore,
{
    let target_num_vars = lp
        .m_vars
        .checked_add(lp.r_vars)
        .and_then(|n| n.checked_add(alpha_bits))
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() > target_num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: target_num_vars,
            actual: opening_point.len(),
        });
    }
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let inner_point = &padded_point[..alpha_bits];
    let outer_point = &padded_point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field::<F>(
        outer_point,
        lp.r_vars,
        lp.m_vars,
        basis,
        BlockOrder::RowMajor,
    )?;
    let inner_reduction = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis)?;
    Ok(PreparedRootOpeningPoint {
        padded_point,
        ring_opening_point,
        inner_reduction,
    })
}

/// Append a prepared root opening point to the transcript.
pub fn append_prepared_root_opening_point<F, T, const D: usize>(
    prepared_point: &PreparedRootOpeningPoint<F, D>,
    transcript: &mut T,
) where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    for pt in &prepared_point.padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
}
