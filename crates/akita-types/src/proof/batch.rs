//! Shared batching and root-opening helper types.

use crate::{
    basis_weights, embed_ring_subfield_scalar, embed_ring_subfield_vector,
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AkitaExpandedSetup,
    AppendToTranscript, BasisMode, BlockOrder, LevelParams, RingCommitment, RingOpeningPoint,
    RingSubfieldEncoding,
};
use akita_algebra::{ring::eval_ring_at_pows, CyclotomicRing};
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
    /// Ring-level outer opening point with weights embedded as `R_F` multipliers.
    pub ring_multiplier_point: RingMultiplierOpeningPoint<F, D>,
    /// Inner ring-slot reduction.
    pub inner_reduction: CyclotomicRing<F, D>,
}

/// Recursive opening point prepared for ring-level replay.
#[derive(Debug, Clone)]
pub struct PreparedRecursiveOpeningPoint<F: FieldCore, L: FieldCore, const D: usize> {
    /// Opening point padded to the recursive verifier's target variable count.
    pub padded_point: Vec<L>,
    /// Extension-field inner tensor weights over the `D` coefficients of the
    /// folded ring.
    pub inner_weights: Vec<L>,
    /// Ring-level outer opening point.
    pub ring_opening_point: RingOpeningPoint<F>,
    /// Ring-level outer opening point with weights embedded as `R_F` multipliers.
    pub ring_multiplier_point: RingMultiplierOpeningPoint<F, D>,
    /// Inner ring-slot reduction.
    pub inner_reduction: CyclotomicRing<F, D>,
}

/// Ring-level opening point whose outer weights act by ring multiplication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RingMultiplierOpeningPoint<F: FieldCore, const D: usize> {
    /// Degree-one openings, where multipliers are ordinary base scalars.
    Base(RingOpeningPoint<F>),
    /// True ring multipliers used by extension-valued openings.
    Ring {
        /// Evaluation vector of length `2^m`, embedded in `R_F`.
        a: Vec<CyclotomicRing<F, D>>,
        /// Block-select vector of length `2^r`, embedded in `R_F`.
        b: Vec<CyclotomicRing<F, D>>,
    },
}

impl<F: FieldCore, const D: usize> RingMultiplierOpeningPoint<F, D> {
    /// Keep base-field scalar weights in their compact scalar form.
    pub fn from_base(point: &RingOpeningPoint<F>) -> Self {
        Self::Base(point.clone())
    }

    /// Build a true ring-multiplier opening point.
    pub fn from_ring(a: Vec<CyclotomicRing<F, D>>, b: Vec<CyclotomicRing<F, D>>) -> Self {
        Self::Ring { a, b }
    }

    /// Borrow the compact base opening point, when this is the degree-one case.
    pub fn as_base(&self) -> Option<&RingOpeningPoint<F>> {
        match self {
            Self::Base(point) => Some(point),
            Self::Ring { .. } => None,
        }
    }

    /// Borrow the ring-valued evaluation vector.
    pub fn a_rings(&self) -> Option<&[CyclotomicRing<F, D>]> {
        match self {
            Self::Base(_) => None,
            Self::Ring { a, .. } => Some(a),
        }
    }

    /// Borrow the ring-valued block vector.
    pub fn b_rings(&self) -> Option<&[CyclotomicRing<F, D>]> {
        match self {
            Self::Base(_) => None,
            Self::Ring { b, .. } => Some(b),
        }
    }

    /// Length of the evaluation vector.
    pub fn a_len(&self) -> usize {
        match self {
            Self::Base(point) => point.a.len(),
            Self::Ring { a, .. } => a.len(),
        }
    }

    /// Length of the block-select vector.
    pub fn b_len(&self) -> usize {
        match self {
            Self::Base(point) => point.b.len(),
            Self::Ring { b, .. } => b.len(),
        }
    }

    /// Return whether every multiplier is a constant ring.
    pub fn is_constant(&self) -> bool {
        match self {
            Self::Base(_) => true,
            Self::Ring { a, b } => a.iter().chain(b.iter()).all(ring_is_constant),
        }
    }

    /// Evaluate the `a[idx]` multiplier at the supplied ring powers.
    ///
    /// # Errors
    ///
    /// Returns an invalid proof error if `idx` is out of range.
    pub fn eval_a_at<E>(&self, idx: usize, alpha_pows: &[E]) -> Result<E, AkitaError>
    where
        E: ExtField<F>,
    {
        match self {
            Self::Base(point) => point
                .a
                .get(idx)
                .copied()
                .map(E::lift_base)
                .ok_or(AkitaError::InvalidProof),
            Self::Ring { a, .. } => a
                .get(idx)
                .map(|value| eval_ring_at_pows(value, alpha_pows))
                .ok_or(AkitaError::InvalidProof),
        }
    }

    /// Evaluate `coefficient * b[idx]` at the supplied ring powers.
    ///
    /// Base multipliers stay in the scalar field; ring multipliers require the
    /// caller to provide `coefficient` embedded as a ring-subfield element.
    ///
    /// # Errors
    ///
    /// Returns an invalid proof error if `idx` is out of range or if a ring
    /// multiplier is evaluated without an embedded coefficient.
    pub fn eval_b_with_coefficient<E>(
        &self,
        idx: usize,
        coefficient: E,
        coefficient_ring: Option<&CyclotomicRing<F, D>>,
        alpha_pows: &[E],
    ) -> Result<E, AkitaError>
    where
        E: ExtField<F>,
    {
        match self {
            Self::Base(point) => point
                .b
                .get(idx)
                .copied()
                .map(|value| coefficient.mul_base(value))
                .ok_or(AkitaError::InvalidProof),
            Self::Ring { b, .. } => {
                let coefficient_ring = coefficient_ring.ok_or(AkitaError::InvalidProof)?;
                let value = b.get(idx).ok_or(AkitaError::InvalidProof)?;
                Ok(eval_ring_at_pows(&(*coefficient_ring * *value), alpha_pows))
            }
        }
    }

    /// Constant coefficient of `a[idx]`, if it is known to be constant.
    pub fn a_constant_coeff(&self, idx: usize) -> Option<F> {
        match self {
            Self::Base(point) => point.a.get(idx).copied(),
            Self::Ring { a, .. } => a
                .get(idx)
                .filter(|ring| ring_is_constant(ring))
                .map(|ring| ring.coefficients()[0]),
        }
    }

    /// Constant coefficient of `b[idx]`, if it is known to be constant.
    pub fn b_constant_coeff(&self, idx: usize) -> Option<F> {
        match self {
            Self::Base(point) => point.b.get(idx).copied(),
            Self::Ring { b, .. } => b
                .get(idx)
                .filter(|ring| ring_is_constant(ring))
                .map(|ring| ring.coefficients()[0]),
        }
    }
}

fn ring_is_constant<F: FieldCore, const D: usize>(ring: &CyclotomicRing<F, D>) -> bool {
    ring.coefficients()[1..].iter().all(|coeff| coeff.is_zero())
}

fn ring_subfield_scalar_to_ring<F, E, const D: usize>(
    value: E,
    error: AkitaError,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore + akita_field::FromPrimitiveInt,
    E: RingSubfieldEncoding<F>,
{
    embed_ring_subfield_scalar::<F, E, D>(value, error)
}

fn ring_multiplier_opening_point_from_ext<F, E, const D: usize>(
    opening_point: &[E],
    r_vars: usize,
    m_vars: usize,
    basis: BasisMode,
    block_order: BlockOrder,
) -> Result<RingMultiplierOpeningPoint<F, D>, AkitaError>
where
    F: FieldCore + akita_field::FromPrimitiveInt,
    E: RingSubfieldEncoding<F>,
{
    let expected_len = r_vars
        .checked_add(m_vars)
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() != expected_len {
        return Err(AkitaError::InvalidPointDimension {
            expected: expected_len,
            actual: opening_point.len(),
        });
    }

    let (a_weights, b_weights) = match block_order {
        BlockOrder::ColumnMajor => (
            basis_weights(&opening_point[r_vars..], basis),
            basis_weights(&opening_point[..r_vars], basis),
        ),
        BlockOrder::RowMajor => (
            basis_weights(&opening_point[..m_vars], basis),
            basis_weights(&opening_point[m_vars..], basis),
        ),
    };
    let error = AkitaError::InvalidInput(
        "opening point does not encode in the ring-subfield basis".to_string(),
    );
    let a = a_weights
        .into_iter()
        .map(|weight| ring_subfield_scalar_to_ring::<F, E, D>(weight, error.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let b = b_weights
        .into_iter()
        .map(|weight| ring_subfield_scalar_to_ring::<F, E, D>(weight, error.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RingMultiplierOpeningPoint::from_ring(a, b))
}

/// Evaluate the inner `D` coefficient slice of a folded base-field ring at an
/// extension-field inner point.
///
/// Root psi-packed claims use the trace/subfield reduction because their ring
/// coefficients encode extension slots. Recursive witnesses are different:
/// their ring coefficients are ordinary base-field digits, opened over the
/// extension challenge field. This helper is the explicit field-reduction
/// boundary for that case.
///
/// # Errors
///
/// Returns an invalid-size error when `inner_weights` does not contain exactly
/// one extension-field weight per ring coefficient.
pub fn ring_inner_product_with_extension_weights<F, L, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    inner_weights: &[L],
) -> Result<L, AkitaError>
where
    F: FieldCore,
    L: RingSubfieldEncoding<F>,
{
    if inner_weights.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: inner_weights.len(),
        });
    }
    Ok(ring
        .coefficients()
        .iter()
        .zip(inner_weights.iter())
        .fold(L::zero(), |acc, (&coeff, &weight)| {
            acc + weight.mul_base(coeff)
        }))
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
/// Each input pair is `(opening_point, point_payload)` where `point_payload`
/// is one commitment-plus-openings unit (the prover supplies polynomials
/// here, the verifier supplies claimed evaluations).
///
/// # Errors
///
/// Returns an error if the batch is empty, has inconsistent opening-point
/// dimensions, has empty point payloads, exceeds setup capacity, or overflows
/// its flattened claim count.
pub fn validate_batched_inputs<F, E, G, Len>(
    setup: &AkitaExpandedSetup<F>,
    inputs: &[(&[E], G)],
    point_payload_len: Len,
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
    for (point_idx, (_, payload)) in inputs.iter().enumerate() {
        let point_claims = point_payload_len(payload);
        if point_claims == 0 {
            return Err(shape_error(format!(
                "{label} point {point_idx} must have at least one item",
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
    let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&ring_opening_point);
    let inner_reduction = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis)?;
    Ok(PreparedRootOpeningPoint {
        padded_point,
        ring_opening_point,
        ring_multiplier_point,
        inner_reduction,
    })
}

/// Prepare a root opening point whose public coordinates may live in an
/// extension field, while the resulting ring payload remains over `F`.
///
/// For the degree-one path this is exactly [`prepare_root_opening_point`]. For
/// true extension challenges, live inner variables use the `D / [E:F]`
/// ring-subfield slots and outer variables are materialized as ring
/// multipliers.
///
/// # Errors
///
/// Returns an error if the extension basis is unsupported, the point does not
/// fit the packed-inner shape, or the ring-subfield parameter validation
/// rejects `(D, [L:F])`.
pub fn prepare_root_opening_point_ext<F, E, L, const D: usize>(
    opening_point: &[E],
    basis: BasisMode,
    lp: &LevelParams,
    alpha_bits: usize,
) -> Result<PreparedRootOpeningPoint<F, D>, AkitaError>
where
    F: FieldCore + akita_field::FromPrimitiveInt,
    E: RingSubfieldEncoding<F>,
    L: RingSubfieldEncoding<F> + ExtField<E>,
{
    if <L as ExtField<F>>::EXT_DEGREE == 1 {
        let base_point = opening_point
            .iter()
            .map(|coord| {
                coord.degree_one_base().ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "claim field element had no base coordinate".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        return prepare_root_opening_point::<F, D>(&base_point, basis, lp, alpha_bits);
    }

    if <L as ExtField<F>>::EXT_DEGREE != <E as ExtField<F>>::EXT_DEGREE {
        return Err(AkitaError::InvalidInput(
            "baseline extension root openings require claim and challenge fields to have the same base degree"
                .to_string(),
        ));
    }
    if D % E::EXT_DEGREE != 0 || !(D / E::EXT_DEGREE).is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "claim-field degree must divide the ring dimension into power-of-two slots".to_string(),
        ));
    }

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
    padded_point.resize(target_num_vars, E::zero());

    let packed_slots = D / E::EXT_DEGREE;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if packed_inner_bits > alpha_bits {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: alpha_bits,
        });
    }
    if padded_point[packed_inner_bits..alpha_bits]
        .iter()
        .any(|coord| !coord.is_zero())
    {
        return Err(AkitaError::InvalidInput(
            "inactive extension inner coordinates must be zero after psi packing".to_string(),
        ));
    }

    let mut inner_point = padded_point[..packed_inner_bits]
        .iter()
        .copied()
        .map(L::lift_base)
        .collect::<Vec<_>>();
    inner_point.resize(packed_inner_bits, L::zero());
    let inner_weights = basis_weights(&inner_point, basis);
    let inner_reduction = embed_ring_subfield_vector::<F, L, D>(
        &inner_weights,
        AkitaError::InvalidInput(
            "opening point does not encode in the ring-subfield basis".to_string(),
        ),
    )?;
    let outer_point = &padded_point[alpha_bits..];
    let ring_multiplier_point = ring_multiplier_opening_point_from_ext::<F, E, D>(
        outer_point,
        lp.r_vars,
        lp.m_vars,
        basis,
        BlockOrder::RowMajor,
    )?;
    let ring_opening_point = ring_opening_point_from_field::<F>(
        &vec![F::zero(); outer_point.len()],
        lp.r_vars,
        lp.m_vars,
        basis,
        BlockOrder::RowMajor,
    )?;

    Ok(PreparedRootOpeningPoint {
        padded_point: Vec::new(),
        ring_opening_point,
        ring_multiplier_point,
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
/// the currently supported shape is the same explicit ring-subfield boundary
/// as the root folded path: all live variables must fit in the packed inner
/// slots and there can be no outer block variables.
///
/// # Errors
///
/// Returns an error when the point length is invalid, the extension degree is
/// unsupported by the ring-subfield dispatcher, or the level has outer
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
    L: RingSubfieldEncoding<F>,
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
                coord.degree_one_base().ok_or_else(|| {
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
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&ring_opening_point);
        let inner_reduction = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis)?;
        let inner_weights = base_point[..alpha_bits]
            .iter()
            .copied()
            .map(L::lift_base)
            .collect::<Vec<_>>();
        let inner_weights = basis_weights(&inner_weights, basis);
        return Ok(PreparedRecursiveOpeningPoint {
            padded_point,
            inner_weights,
            ring_opening_point,
            ring_multiplier_point,
            inner_reduction,
        });
    }

    if D % L::EXT_DEGREE != 0 || !(D / L::EXT_DEGREE).is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "challenge-field degree must divide the ring dimension into power-of-two slots"
                .to_string(),
        ));
    }

    let inner_point = &padded_point[..alpha_bits];
    let inner_weights = basis_weights(inner_point, basis);
    let trace_inner_point_len = (D / L::EXT_DEGREE).trailing_zeros() as usize;
    if padded_point[trace_inner_point_len..alpha_bits]
        .iter()
        .any(|coord| !coord.is_zero())
    {
        return Err(AkitaError::InvalidInput(
            "inactive extension inner coordinates must be zero after psi packing".to_string(),
        ));
    }
    let trace_inner_weights = basis_weights(&padded_point[..trace_inner_point_len], basis);
    let inner_reduction = embed_ring_subfield_vector::<F, L, D>(
        &trace_inner_weights,
        AkitaError::InvalidInput(
            "recursive opening point does not encode in the ring-subfield basis".to_string(),
        ),
    )?;
    let outer_point = &padded_point[alpha_bits..];
    let ring_multiplier_point = ring_multiplier_opening_point_from_ext::<F, L, D>(
        outer_point,
        lp.r_vars,
        lp.m_vars,
        basis,
        block_order,
    )?;
    let ring_opening_point = ring_opening_point_from_field::<F>(
        &vec![F::zero(); outer_point.len()],
        lp.r_vars,
        lp.m_vars,
        basis,
        block_order,
    )?;

    Ok(PreparedRecursiveOpeningPoint {
        padded_point,
        inner_weights,
        ring_opening_point,
        ring_multiplier_point,
        inner_reduction,
    })
}

/// Convert an extension-domain opening point into the protocol point expected
/// by the ring-subfield-packed folded root path.
///
/// The returned point has `extension_num_vars + log2([E:F])` coordinates. The
/// extra coordinates expose the extension basis slots inside the root inner
/// ring, matching the lifted baseline layout.
///
/// # Errors
///
/// Returns an error when the extension degree is not a power of two, does not
/// divide `D`, or the point is too short for the packed root layout.
pub fn ring_subfield_packed_extension_opening_point<F, E, const D: usize>(
    extension_num_vars: usize,
    point: &[E],
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let k = E::EXT_DEGREE;
    if k == 1 {
        return Ok(point.to_vec());
    }
    if !k.is_power_of_two() || D % k != 0 {
        return Err(AkitaError::InvalidInput(
            "extension degree must be a power of two dividing D".to_string(),
        ));
    }
    if point.len() != extension_num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: extension_num_vars,
            actual: point.len(),
        });
    }
    let alpha_bits = D.trailing_zeros() as usize;
    let kappa_bits = k.trailing_zeros() as usize;
    let packed_inner_bits = alpha_bits.checked_sub(kappa_bits).ok_or_else(|| {
        AkitaError::InvalidInput("extension degree exceeds ring dimension".to_string())
    })?;
    if extension_num_vars < packed_inner_bits {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: extension_num_vars,
        });
    }

    let mut transformed = Vec::with_capacity(extension_num_vars + kappa_bits);
    transformed.extend_from_slice(&point[..packed_inner_bits]);
    transformed.resize(alpha_bits, E::zero());
    transformed.extend_from_slice(&point[packed_inner_bits..]);
    Ok(transformed)
}

/// Return whether folded root proving can soundly handle this opening shape.
///
/// Degree-one proof-scalar fields keep the original base-field folded-root
/// path. For true extension proof-scalar fields, the folded path supports
/// psi-packed inner slots plus ring-multiplier outer weights. Multiple claims
/// at the same point are handled by one public row per point, with row-local
/// extension batching coefficients embedded into the ring relation.
pub fn folded_root_supports_opening_shape<F, E, L, const D: usize>(
    opening_points: &[&[E]],
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
    let target_num_vars = match lp
        .m_vars
        .checked_add(lp.r_vars)
        .and_then(|n| n.checked_add(alpha_bits))
    {
        Some(value) => value,
        None => return false,
    };
    if opening_points.iter().any(|point| {
        point.len() > target_num_vars
            || point
                .get(packed_inner_bits..alpha_bits)
                .is_some_and(|inactive| inactive.iter().any(|coord| !coord.is_zero()))
    }) {
        return false;
    }
    true
}

/// Return whether root tensor projection can represent this field/ring shape.
pub fn root_tensor_projection_enabled<F, E, C, const D: usize>(num_vars: usize) -> bool
where
    F: FieldCore,
    E: ExtField<F>,
    C: ExtField<F>,
{
    let width = C::EXT_DEGREE;
    let Some(double_width) = width.checked_mul(2) else {
        return false;
    };
    width > 1
        && width == E::EXT_DEGREE
        && width.is_power_of_two()
        && D.is_power_of_two()
        && D >= double_width
        && D % width == 0
        && num_vars >= D.trailing_zeros() as usize
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
        let polys: Vec<usize> = vec![0, 1, 2];
        let inputs = vec![(&p0[..], polys.clone()), (&p1[..], polys)];

        validate_batched_inputs(&setup(), &inputs, |polys| polys.len(), true)
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
    fn recursive_extension_opening_preparation_uses_ring_subfield_boundary() {
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
    fn packed_extension_opening_point_exposes_basis_slots() {
        let point = [
            L::lift_base(F::from_u64(1)),
            L::lift_base(F::from_u64(2)),
            L::lift_base(F::from_u64(3)),
            L::lift_base(F::from_u64(4)),
        ];

        let transformed =
            ring_subfield_packed_extension_opening_point::<F, L, 32>(point.len(), &point)
                .expect("packed extension point");

        assert_eq!(
            transformed,
            vec![point[0], point[1], point[2], L::zero(), L::zero(), point[3]]
        );
    }

    #[test]
    fn extension_challenge_folded_root_gate_accepts_same_point_batching() {
        let lp = packed_inner_lp();
        let point = [F::from_u64(7), F::from_u64(11)];

        assert!(folded_root_supports_opening_shape::<F, F, L, 32>(
            &[&point[..]],
            &lp,
            5,
        ));
        assert!(folded_root_supports_opening_shape::<F, F, L, 32>(
            &[&point[..]],
            &lp,
            5,
        ));
    }

    #[test]
    fn root_tensor_projection_gate_requires_room_for_signed_subfield_basis() {
        assert!(root_tensor_projection_enabled::<F, L, L, 8>(3));
        assert!(!root_tensor_projection_enabled::<F, L, L, 4>(2));
        assert!(!root_tensor_projection_enabled::<F, E, L, 8>(3));
    }
}
