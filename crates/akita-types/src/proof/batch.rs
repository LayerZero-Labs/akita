//! Shared batching and root-opening helper types.

use crate::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AkitaExpandedSetup,
    AppendToTranscript, BasisMode, BlockOrder, LevelParams, RingCommitment, RingOpeningPoint,
};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_EVAL_OPENINGS_FIELD,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use std::marker::PhantomData;

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

/// Convert degree-one claim-field points back to base-field coordinates.
///
/// This is a temporary bridge for folded-root code paths whose ring algebra
/// still runs over the base field.
///
/// # Errors
///
/// Returns an error if `E` is a true extension field or if a claim-field element
/// does not expose a base coordinate.
pub fn claim_points_to_base<F, E>(
    points: &[&[E]],
    extension_error: AkitaError,
    empty_coord_error: AkitaError,
) -> Result<Vec<Vec<F>>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    require_degree_one_ext::<F, E>(extension_error)?;

    points
        .iter()
        .map(|point| {
            point
                .iter()
                .map(|coord| degree_one_ext_scalar_to_base(coord, &empty_coord_error))
                .collect()
        })
        .collect()
}

/// Convert degree-one claim-field values back to base-field scalars.
///
/// This is the scalar counterpart to [`claim_points_to_base`].
///
/// # Errors
///
/// Returns an error if `E` is a true extension field or if a claim-field element
/// does not expose a base coordinate.
pub fn claim_values_to_base<F, E>(
    values: &[E],
    extension_error: AkitaError,
    empty_coord_error: AkitaError,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    require_degree_one_ext::<F, E>(extension_error)?;

    values
        .iter()
        .map(|value| degree_one_ext_scalar_to_base(value, &empty_coord_error))
        .collect()
}

fn require_degree_one_ext<F, E>(extension_error: AkitaError) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    if E::EXT_DEGREE != 1 {
        return Err(extension_error);
    }
    Ok(())
}

fn degree_one_ext_scalar_to_base<F, E>(
    value: &E,
    empty_coord_error: &AkitaError,
) -> Result<F, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    value
        .to_base_vec()
        .into_iter()
        .next()
        .ok_or_else(|| empty_coord_error.clone())
}

/// Samples challenge-field values through the current degree-one base bridge.
///
/// Folded-root stage proofs are still serialized over the base field, so this
/// sampler makes the remaining degree-one restriction explicit while keeping
/// challenge sampling routed through the configured challenge field.
#[derive(Debug, Clone, Copy)]
pub struct DegreeOneChallengeSampler<F: FieldCore, E: ExtField<F>> {
    _marker: PhantomData<(F, E)>,
}

impl<F, E> DegreeOneChallengeSampler<F, E>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    /// Build a sampler for a degree-one challenge field.
    ///
    /// # Errors
    ///
    /// Returns `extension_error` if `E` is a true extension over `F`.
    pub fn new(extension_error: AkitaError) -> Result<Self, AkitaError> {
        require_degree_one_ext::<F, E>(extension_error)?;
        Ok(Self {
            _marker: PhantomData,
        })
    }

    /// Sample one configured challenge and project it to the base scalar.
    ///
    /// # Panics
    ///
    /// Panics if a degree-one extension value does not expose its single base
    /// coordinate.
    pub fn sample<T>(&self, transcript: &mut T, label: &[u8]) -> F
    where
        T: Transcript<F>,
    {
        degree_one_ext_scalar_to_base(
            &sample_ext_challenge::<F, E, T>(transcript, label),
            &AkitaError::InvalidProof,
        )
        .expect("degree-one challenge field must expose one base coordinate")
    }
}

/// Absorb public claim-field opening points into the base-field transcript.
pub fn append_claim_points_to_transcript<F, E, T>(points: &[&[E]], transcript: &mut T)
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
    T: Transcript<F>,
{
    for point in points {
        for coord in *point {
            append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, coord);
        }
    }
}

/// Absorb public claim-field evaluations into the base-field transcript.
pub fn append_claim_values_to_transcript<F, E, T>(values: &[E], transcript: &mut T)
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
    T: Transcript<F>,
{
    for value in values {
        append_ext_field::<F, E, T>(transcript, ABSORB_EVAL_OPENINGS_FIELD, value);
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
/// `points` carries one entry per opening point. `point_claim_len(p)` returns
/// `l_p`, the number of claimed openings at point `p`.
///
/// # Errors
///
/// Returns an error if the batch is empty, has inconsistent opening-point
/// dimensions, has empty per-point claim lists, exceeds setup capacity, or
/// overflows its flattened claim count.
pub fn validate_batched_inputs<F, E, P, Len>(
    setup: &AkitaExpandedSetup<F>,
    points: &[P],
    point_field_slice: impl Fn(&P) -> &[E],
    point_claim_len: Len,
    for_prover: bool,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    Len: Fn(&P) -> usize,
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

    if points.is_empty() {
        return Err(shape_error(format!(
            "{label} requires at least one opening point"
        )));
    }
    let num_vars = point_field_slice(&points[0]).len();
    if points
        .iter()
        .any(|p| point_field_slice(p).len() != num_vars)
    {
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
    if points.len() > setup.seed.max_num_points {
        if for_prover {
            return Err(AkitaError::InvalidInput(format!(
                "batched_prove received {} opening points but setup supports at most {}",
                points.len(),
                setup.seed.max_num_points
            )));
        }
        return Err(AkitaError::InvalidProof);
    }

    let mut num_claims = 0usize;
    for (point_idx, point) in points.iter().enumerate() {
        let point_claims = point_claim_len(point);
        if point_claims == 0 {
            return Err(shape_error(format!(
                "{label} point {point_idx} must have at least one claim",
            )));
        }
        num_claims = num_claims
            .checked_add(point_claims)
            .ok_or_else(|| shape_error(format!("{label} total claim count overflow")))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AkitaSetupSeed, FlatMatrix};
    use akita_field::{Fp2, Fp32, NegOneNr};
    use akita_transcript::{labels, Blake2bTranscript};

    type F = Fp32<251>;
    type E = Fp2<F, NegOneNr>;

    fn setup() -> AkitaExpandedSetup<F> {
        AkitaExpandedSetup {
            seed: AkitaSetupSeed {
                max_num_vars: 3,
                max_num_batched_polys: 8,
                max_num_points: 2,
                max_stride: 1,
                public_matrix_seed: [0u8; 32],
            },
            shared_matrix: FlatMatrix::from_flat_data(vec![F::zero()], 1),
        }
    }

    #[test]
    fn batched_input_validation_accepts_extension_points() {
        let p0 = [E::new(F::from_u64(1), F::from_u64(2))];
        let p1 = [E::new(F::from_u64(3), F::from_u64(4))];
        // Each point has its own claim count; commitment is implicit (single
        // commitment for the whole batch, validated elsewhere).
        let inputs = vec![(&p0[..], 2usize), (&p1[..], 1usize)];

        validate_batched_inputs(
            &setup(),
            &inputs,
            |(point, _)| *point,
            |(_, claim_count)| *claim_count,
            true,
        )
        .expect("extension-valued opening points should validate by shape");
    }

    #[test]
    fn degree_one_challenge_bridge_matches_base_sampling() {
        let mut bridged = Blake2bTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let mut scalar = Blake2bTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
        bridged.append_bytes(labels::ABSORB_COMMITMENT, b"same-prefix");
        scalar.append_bytes(labels::ABSORB_COMMITMENT, b"same-prefix");

        let sampler =
            DegreeOneChallengeSampler::<F, F>::new(AkitaError::InvalidProof).expect("degree one");
        let bridge = sampler.sample(&mut bridged, labels::CHALLENGE_EVAL_BATCH);
        let base = scalar.challenge_scalar(labels::CHALLENGE_EVAL_BATCH);

        assert_eq!(bridge, base);
    }

    #[test]
    fn true_extension_challenge_bridge_is_rejected() {
        assert!(
            DegreeOneChallengeSampler::<F, E>::new(AkitaError::InvalidProof).is_err(),
            "folded-root base bridge must reject true extension challenge fields"
        );
    }
}
