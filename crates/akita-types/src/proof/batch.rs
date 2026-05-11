//! Shared batching and root-opening helper types.

use crate::{
    basis_weights, embed_hachi_subfield_vector, reduce_inner_opening_to_ring_element,
    ring_opening_point_from_field, AkitaExpandedSetup, AppendToTranscript, BasisMode, BlockOrder,
    HachiSubfieldEncoding, LevelParams, RingCommitment, RingOpeningPoint,
};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_EVAL_OPENINGS_FIELD,
};
use akita_transcript::{append_ext_field, Transcript};

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

/// Recursive opening point prepared for ring-level replay.
#[derive(Debug, Clone)]
pub struct PreparedRecursiveOpeningPoint<F: FieldCore, L: FieldCore, const D: usize> {
    /// Opening point padded to the recursive verifier's target variable count.
    pub padded_point: Vec<L>,
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
/// # Errors
///
/// Returns an error if the batch is empty, has inconsistent opening-point
/// dimensions, has empty groups, exceeds setup capacity, or overflows its
/// flattened claim count.
pub fn validate_batched_inputs<F, E, G, Len>(
    setup: &AkitaExpandedSetup<F>,
    inputs: &[(&[E], Vec<G>)],
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

/// Prepare a root opening point whose public coordinates may live in an
/// extension field, while the resulting ring payload remains over `F`.
///
/// For the degree-one path this is exactly [`prepare_root_opening_point`]. For
/// true extension challenges, this first slice supports the sound packed-inner
/// case where all live root variables are inside the `D / [L:F]` Hachi
/// subfield slots and there are no outer block variables. General outer
/// extension points are the later split/Frobenius path.
///
/// # Errors
///
/// Returns an error if the extension basis is unsupported, the point does not
/// fit the packed-inner shape, or the Hachi subfield parameter validation
/// rejects `(D, [L:F])`.
pub fn prepare_root_opening_point_ext<F, E, L, const D: usize>(
    opening_point: &[E],
    basis: BasisMode,
    lp: &LevelParams,
    alpha_bits: usize,
) -> Result<PreparedRootOpeningPoint<F, D>, AkitaError>
where
    F: FieldCore + akita_field::FromPrimitiveInt,
    E: ExtField<F>,
    L: HachiSubfieldEncoding<F> + ExtField<E>,
{
    if <L as ExtField<F>>::EXT_DEGREE == 1 {
        let base_point = opening_point
            .iter()
            .map(|coord| {
                coord.to_base_vec().into_iter().next().ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "claim field element had no base coordinate".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        return prepare_root_opening_point::<F, D>(&base_point, basis, lp, alpha_bits);
    }

    if lp.m_vars != 0 || lp.r_vars != 0 {
        return Err(AkitaError::InvalidInput(
            "extension root openings with outer variables require the split/Frobenius path"
                .to_string(),
        ));
    }
    if D % <L as ExtField<F>>::EXT_DEGREE != 0
        || !(D / <L as ExtField<F>>::EXT_DEGREE).is_power_of_two()
    {
        return Err(AkitaError::InvalidInput(
            "challenge-field degree must divide the ring dimension into power-of-two slots"
                .to_string(),
        ));
    }

    let packed_slots = D / <L as ExtField<F>>::EXT_DEGREE;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if packed_inner_bits > alpha_bits || opening_point.len() > packed_inner_bits {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: opening_point.len(),
        });
    }

    let mut inner_point = opening_point
        .iter()
        .copied()
        .map(L::lift_base)
        .collect::<Vec<_>>();
    inner_point.resize(packed_inner_bits, L::zero());
    let inner_weights = basis_weights(&inner_point, basis);
    let inner_reduction = embed_hachi_subfield_vector::<F, L, D>(
        &inner_weights,
        AkitaError::InvalidInput(
            "opening point does not encode in the Hachi subfield basis".to_string(),
        ),
    )?;
    let ring_opening_point =
        ring_opening_point_from_field::<F>(&[], lp.r_vars, lp.m_vars, basis, BlockOrder::RowMajor)?;

    Ok(PreparedRootOpeningPoint {
        padded_point: Vec::new(),
        ring_opening_point,
        inner_reduction,
    })
}

/// Prepare a recursive opening point whose coordinates may live in the proof
/// scalar field `L`, while the resulting ring payload remains over `F`.
///
/// For degree-one `L`, this is the original recursive materialization path:
/// coordinates are converted to base scalars, outer variables are prepared by
/// [`ring_opening_point_from_field`], and the inner point is reduced by
/// [`reduce_inner_opening_to_ring_element`]. For true extension-valued `L`,
/// the currently supported shape is the same explicit Hachi subfield boundary
/// as the root folded path: all live variables must fit in the packed inner
/// slots and there can be no outer block variables.
///
/// # Errors
///
/// Returns an error when the point length is invalid, the extension degree is
/// unsupported by the Hachi subfield dispatcher, or the level has outer
/// variables that require the later split/Frobenius route.
pub fn prepare_recursive_opening_point_ext<F, L, const D: usize>(
    opening_point: &[L],
    basis: BasisMode,
    lp: &LevelParams,
    alpha_bits: usize,
    block_order: BlockOrder,
) -> Result<PreparedRecursiveOpeningPoint<F, L, D>, AkitaError>
where
    F: FieldCore + akita_field::FromPrimitiveInt,
    L: HachiSubfieldEncoding<F>,
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
    padded_point.resize(target_num_vars, L::zero());

    if L::EXT_DEGREE == 1 {
        let base_point = padded_point
            .iter()
            .map(|coord| {
                coord
                    .to_hachi_subfield_coords()
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "challenge field element had no base coordinate".to_string(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let inner_point = &base_point[..alpha_bits];
        let outer_point = &base_point[alpha_bits..];
        let ring_opening_point = ring_opening_point_from_field::<F>(
            outer_point,
            lp.r_vars,
            lp.m_vars,
            basis,
            block_order,
        )?;
        let inner_reduction = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis)?;
        return Ok(PreparedRecursiveOpeningPoint {
            padded_point,
            ring_opening_point,
            inner_reduction,
        });
    }

    if lp.m_vars != 0 || lp.r_vars != 0 {
        return Err(AkitaError::InvalidInput(
            "extension recursive openings with outer variables require the split/Frobenius path"
                .to_string(),
        ));
    }
    if D % L::EXT_DEGREE != 0 || !(D / L::EXT_DEGREE).is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "challenge-field degree must divide the ring dimension into power-of-two slots"
                .to_string(),
        ));
    }

    let packed_slots = D / L::EXT_DEGREE;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if packed_inner_bits > alpha_bits || opening_point.len() > packed_inner_bits {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: opening_point.len(),
        });
    }

    let mut inner_point = opening_point.to_vec();
    inner_point.resize(packed_inner_bits, L::zero());
    let inner_weights = basis_weights(&inner_point, basis);
    let inner_reduction = embed_hachi_subfield_vector::<F, L, D>(
        &inner_weights,
        AkitaError::InvalidInput(
            "recursive opening point does not encode in the Hachi subfield basis".to_string(),
        ),
    )?;
    let ring_opening_point =
        ring_opening_point_from_field::<F>(&[], lp.r_vars, lp.m_vars, basis, block_order)?;

    Ok(PreparedRecursiveOpeningPoint {
        padded_point,
        ring_opening_point,
        inner_reduction,
    })
}

/// Return whether folded root proving can soundly handle this opening shape.
///
/// Degree-one proof-scalar fields keep the original base-field folded-root
/// path. For true extension proof-scalar fields, the currently implemented
/// folded path supports the packed-inner Hachi subfield case: no outer
/// variables, all live variables fit inside `D / [L:F]` packed slots, and no
/// same-point batching.
pub fn folded_root_supports_opening_shape<F, E, L, const D: usize>(
    opening_points: &[&[E]],
    point_claim_counts: &[usize],
    lp: &LevelParams,
    alpha_bits: usize,
) -> bool
where
    F: FieldCore,
    E: ExtField<F>,
    L: ExtField<F>,
{
    if <L as ExtField<F>>::EXT_DEGREE == 1 {
        return true;
    }
    if lp.m_vars != 0 || lp.r_vars != 0 {
        return false;
    }
    if D % <L as ExtField<F>>::EXT_DEGREE != 0
        || !(D / <L as ExtField<F>>::EXT_DEGREE).is_power_of_two()
    {
        return false;
    }
    let packed_slots = D / <L as ExtField<F>>::EXT_DEGREE;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if packed_inner_bits > alpha_bits {
        return false;
    }
    if opening_points
        .iter()
        .any(|point| point.len() > packed_inner_bits)
    {
        return false;
    }
    !point_claim_counts
        .iter()
        .any(|&point_claim_count| point_claim_count > 1)
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
    use crate::{AkitaSetupSeed, FlatMatrix, SisModulusFamily};
    use akita_challenges::SparseChallengeConfig;
    use akita_field::{Fp2, Fp32, LiftBase, NegOneNr, RingSubfieldFp4};

    type F = Fp32<251>;
    type E = Fp2<F, NegOneNr>;
    type L = RingSubfieldFp4<F>;

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
        let groups = vec![vec![0usize], vec![1usize, 2usize]];
        let inputs = vec![(&p0[..], groups.clone()), (&p1[..], groups)];

        validate_batched_inputs(&setup(), &inputs, |group| group.len(), true)
            .expect("extension-valued opening points should validate by shape");
    }

    fn packed_inner_lp() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q32,
            32,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            },
        )
    }

    #[test]
    fn recursive_extension_opening_preparation_uses_hachi_boundary() {
        let lp = packed_inner_lp();
        let point = [L::lift_base(F::from_u64(3)), L::lift_base(F::from_u64(5))];

        let prepared = prepare_recursive_opening_point_ext::<F, L, 32>(
            &point,
            BasisMode::Lagrange,
            &lp,
            5,
            BlockOrder::ColumnMajor,
        )
        .expect("packed-inner recursive extension point should prepare");

        assert_eq!(prepared.padded_point.len(), 5);
    }

    #[test]
    fn extension_challenge_folded_root_gate_rejects_same_point_batching() {
        let lp = packed_inner_lp();
        let point = [F::from_u64(7), F::from_u64(11)];

        assert!(folded_root_supports_opening_shape::<F, F, L, 32>(
            &[&point[..]],
            &[1],
            &lp,
            5,
        ));
        assert!(!folded_root_supports_opening_shape::<F, F, L, 32>(
            &[&point[..]],
            &[2],
            &lp,
            5,
        ));
    }
}
