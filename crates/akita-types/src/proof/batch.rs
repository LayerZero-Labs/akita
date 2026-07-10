//! Shared batching and root-opening helper types.

use crate::{
    basis_weights, dispatch_for_field, embed_ring_subfield_scalar, embed_ring_subfield_vector,
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AkitaExpandedSetup,
    BasisMode, Commitment, FpExtEncoding, LevelParams, OpeningBlockLayout, RingOpeningPoint,
    RingVec,
};
use akita_algebra::{
    ring::{eval_flat_ring_at_pows_fast, eval_ring_at_pows_fast},
    CyclotomicRing,
};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBaseUnreduced};
use akita_transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVAL_OPENINGS_FIELD};
use akita_transcript::{append_ext_field, Transcript};

/// Recursive opening point prepared for ring-level replay.
///
/// Ring dimension is stored at runtime; hot paths inside `dispatch_ring_dim`
/// borrow the ψ-packed inner ring via [`Self::packed_inner_trusted`].
#[derive(Debug, Clone)]
pub struct PreparedOpeningPoint<F: FieldCore, E: FieldCore> {
    /// Opening point padded to the recursive verifier's target variable count.
    pub padded_point: Vec<E>,
    /// Ring-level outer opening point.
    pub ring_opening_point: RingOpeningPoint<F>,
    /// Ring-level outer opening point with weights embedded as `R_F` multipliers.
    pub ring_multiplier_point: RingMultiplierOpeningPoint<F>,
    /// The ψ-packed inner block of the opening point (paper `\check{r}_{\mathrm{in}}`).
    ///
    /// Public fixed weight in `TraceOpen(Y) = recover_ring_subfield_inner_product(Y, packed_inner_point)`.
    /// Hot paths borrow via [`Self::packed_inner_trusted`].
    packed_inner_point: RingVec<F>,
    ring_dim: usize,
}

impl<F: FieldCore, E: FieldCore> PreparedOpeningPoint<F, E> {
    /// Construct from typed kernel output at an opening-point boundary.
    pub fn from_parts<const D: usize>(
        padded_point: Vec<E>,
        ring_opening_point: RingOpeningPoint<F>,
        ring_multiplier_point: RingMultiplierOpeningPoint<F>,
        packed_inner_point: CyclotomicRing<F, D>,
    ) -> Self {
        Self {
            padded_point,
            ring_opening_point,
            ring_multiplier_point,
            packed_inner_point: RingVec::from_single(&packed_inner_point),
            ring_dim: D,
        }
    }

    /// Stored ring dimension (coefficients per ring element).
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// ψ-packed inner opening weight in flat ring storage.
    pub fn packed_inner(&self) -> &RingVec<F> {
        &self.packed_inner_point
    }

    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match storage.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dim != D {
            return Err(AkitaError::InvalidInput(format!(
                "prepared opening point ring_d={} does not match requested D={D}",
                self.ring_dim
            )));
        }
        if !self.packed_inner_point.can_decode_single(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.packed_inner_point.coeff_len(),
            });
        }
        self.ring_multiplier_point.ensure_ring_dim::<D>()
    }

    pub fn packed_inner_trusted<const D: usize>(
        &self,
    ) -> Result<&CyclotomicRing<F, D>, AkitaError> {
        self.ensure_ring_dim::<D>()?;
        Ok(self.packed_inner_point.as_single_ring_trusted::<D>())
    }

    /// Owned copy of the ψ-packed inner ring after [`Self::ensure_ring_dim`].
    pub fn packed_inner_owned<const D: usize>(&self) -> Result<CyclotomicRing<F, D>, AkitaError> {
        self.ensure_ring_dim::<D>()?;
        self.packed_inner_point.try_to_single::<D>()
    }
}

/// Ring-level opening point whose outer weights act by ring multiplication.
///
/// Ring dimension is stored at runtime on the [`Self::Ring`] variant; hot paths
/// inside `dispatch_ring_dim` borrow typed rows via [`Self::a_rings_trusted`] and
/// [`Self::b_rings_trusted`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RingMultiplierOpeningPoint<F: FieldCore> {
    /// Degree-one openings, where multipliers are ordinary base scalars.
    Base(RingOpeningPoint<F>),
    /// True ring multipliers used by extension-valued openings.
    Ring {
        /// Evaluation vector of length `2^m`, embedded in `R_F`.
        a: RingVec<F>,
        /// Block-select vector of length `2^r`, embedded in `R_F`.
        b: RingVec<F>,
    },
}

impl<F: FieldCore> RingMultiplierOpeningPoint<F> {
    /// Keep base-field scalar weights in their compact scalar form.
    pub fn from_base(point: &RingOpeningPoint<F>) -> Self {
        Self::Base(point.clone())
    }

    /// Build a true ring-multiplier opening point from typed kernel output.
    pub fn from_ring<const D: usize>(
        a: Vec<CyclotomicRing<F, D>>,
        b: Vec<CyclotomicRing<F, D>>,
    ) -> Self {
        Self::Ring {
            a: RingVec::from_ring_elems(&a),
            b: RingVec::from_ring_elems(&b),
        }
    }

    /// Stored ring dimension for the [`Self::Ring`] variant, or zero for [`Self::Base`].
    pub fn ring_dim(&self) -> usize {
        match self {
            Self::Base(_) => 0,
            Self::Ring { a, .. } => a.ring_dim(),
        }
    }

    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match storage.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        match self {
            Self::Base(_) => Ok(()),
            Self::Ring { a, b } => {
                if a.ring_dim() != 0 && a.ring_dim() != D {
                    return Err(AkitaError::InvalidInput(format!(
                        "ring multiplier a ring_d={} does not match requested D={D}",
                        a.ring_dim()
                    )));
                }
                if b.ring_dim() != 0 && b.ring_dim() != D {
                    return Err(AkitaError::InvalidInput(format!(
                        "ring multiplier b ring_d={} does not match requested D={D}",
                        b.ring_dim()
                    )));
                }
                if !a.can_decode_vec(D) || !b.can_decode_vec(D) {
                    return Err(AkitaError::InvalidSize {
                        expected: D,
                        actual: a.coeff_len(),
                    });
                }
                Ok(())
            }
        }
    }

    /// Borrow the compact base opening point, when this is the degree-one case.
    pub fn as_base(&self) -> Option<&RingOpeningPoint<F>> {
        match self {
            Self::Base(point) => Some(point),
            Self::Ring { .. } => None,
        }
    }

    /// Borrow the ring-valued evaluation vector after [`Self::ensure_ring_dim`].
    pub fn a_rings_trusted<const D: usize>(
        &self,
    ) -> Result<Option<&[CyclotomicRing<F, D>]>, AkitaError> {
        match self {
            Self::Base(_) => Ok(None),
            Self::Ring { a, .. } => {
                self.ensure_ring_dim::<D>()?;
                Ok(Some(a.as_ring_slice::<D>()?))
            }
        }
    }

    /// Borrow the ring-valued block vector after [`Self::ensure_ring_dim`].
    pub fn b_rings_trusted<const D: usize>(
        &self,
    ) -> Result<Option<&[CyclotomicRing<F, D>]>, AkitaError> {
        match self {
            Self::Base(_) => Ok(None),
            Self::Ring { b, .. } => {
                self.ensure_ring_dim::<D>()?;
                Ok(Some(b.as_ring_slice::<D>()?))
            }
        }
    }

    /// Length of the evaluation vector.
    pub fn a_len(&self) -> usize {
        match self {
            Self::Base(point) => point.a.len(),
            Self::Ring { a, .. } => a.count(),
        }
    }

    /// Length of the block-select vector.
    pub fn b_len(&self) -> usize {
        match self {
            Self::Base(point) => point.b.len(),
            Self::Ring { b, .. } => b.count(),
        }
    }

    /// Return whether every multiplier is a constant ring.
    pub fn is_constant(&self) -> bool {
        match self {
            Self::Base(_) => true,
            Self::Ring { a, b } => {
                let ring_dim = a.ring_dim();
                ring_dim != 0
                    && flat_rings_are_constant(a.coeffs(), ring_dim)
                    && flat_rings_are_constant(b.coeffs(), ring_dim)
            }
        }
    }

    /// Evaluate the `a[idx]` multiplier at the supplied ring powers.
    ///
    /// # Errors
    ///
    /// Returns an invalid proof error if `idx` is out of range.
    pub fn eval_a_at<const D: usize, E>(
        &self,
        idx: usize,
        alpha_pows: &[E],
    ) -> Result<E, AkitaError>
    where
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        match self {
            Self::Base(point) => point
                .a
                .get(idx)
                .copied()
                .map(E::lift_base)
                .ok_or(AkitaError::InvalidProof),
            Self::Ring { a, .. } => a
                .as_ring_slice::<D>()?
                .get(idx)
                .map(|value| eval_ring_at_pows_fast(value, alpha_pows))
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
    pub fn eval_b_with_coefficient<const D: usize, E>(
        &self,
        idx: usize,
        coefficient: E,
        coefficient_ring: Option<&CyclotomicRing<F, D>>,
        alpha_pows: &[E],
    ) -> Result<E, AkitaError>
    where
        E: ExtField<F> + MulBaseUnreduced<F>,
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
                let value = b
                    .as_ring_slice::<D>()?
                    .get(idx)
                    .ok_or(AkitaError::InvalidProof)?;
                Ok(eval_ring_at_pows_fast(
                    &(*coefficient_ring * *value),
                    alpha_pows,
                ))
            }
        }
    }

    /// Runtime-dimension form of [`Self::eval_a_at`]: the ring dimension is
    /// `alpha_pows.len()` and ring multipliers are read as flat coefficient
    /// chunks.
    ///
    /// # Errors
    ///
    /// Returns an invalid proof error if `idx` is out of range or the stored
    /// multiplier data does not chunk at the supplied dimension.
    pub fn eval_a_at_dyn<E>(&self, idx: usize, alpha_pows: &[E]) -> Result<E, AkitaError>
    where
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        match self {
            Self::Base(point) => point
                .a
                .get(idx)
                .copied()
                .map(E::lift_base)
                .ok_or(AkitaError::InvalidProof),
            Self::Ring { a, .. } => {
                let ring_d = alpha_pows.len();
                if ring_d == 0 || !a.coeffs().len().is_multiple_of(ring_d) {
                    return Err(AkitaError::InvalidProof);
                }
                a.coeffs()
                    .chunks_exact(ring_d)
                    .nth(idx)
                    .map(|chunk| eval_flat_ring_at_pows_fast(chunk, alpha_pows))
                    .ok_or(AkitaError::InvalidProof)
            }
        }
    }

    /// Runtime-dimension form of [`Self::eval_b_with_coefficient`].
    ///
    /// The ring dimension is `alpha_pows.len()`. Ring multipliers require the
    /// caller to provide `coefficient` embedded as flat ring-subfield
    /// coefficients of the same dimension; the negacyclic product is computed
    /// through a data-derived dispatch.
    ///
    /// # Errors
    ///
    /// Returns an invalid proof error if `idx` is out of range, a ring
    /// multiplier is evaluated without an embedded coefficient, or shapes do
    /// not match the supplied dimension.
    pub fn eval_b_with_coefficient_dyn<E>(
        &self,
        idx: usize,
        coefficient: E,
        coefficient_ring: Option<&[F]>,
        alpha_pows: &[E],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        match self {
            Self::Base(point) => point
                .b
                .get(idx)
                .copied()
                .map(|value| coefficient.mul_base(value))
                .ok_or(AkitaError::InvalidProof),
            Self::Ring { b, .. } => {
                let ring_d = alpha_pows.len();
                let coefficient_ring = coefficient_ring.ok_or(AkitaError::InvalidProof)?;
                if coefficient_ring.len() != ring_d {
                    return Err(AkitaError::InvalidProof);
                }
                if ring_d == 0 || !b.coeffs().len().is_multiple_of(ring_d) {
                    return Err(AkitaError::InvalidProof);
                }
                let value = b
                    .coeffs()
                    .chunks_exact(ring_d)
                    .nth(idx)
                    .ok_or(AkitaError::InvalidProof)?;
                // The negacyclic product does not factor through evaluation at
                // an arbitrary sumcheck challenge (alpha^D != -1), so multiply
                // in the ring before evaluating.
                dispatch_for_field!(
                    ProtocolDispatchSlot::Role(RingRole::Outer),
                    F,
                    ring_d,
                    |D| {
                        let coeff_arr: [F; D] = coefficient_ring
                            .try_into()
                            .map_err(|_| AkitaError::InvalidProof)?;
                        let value_arr: [F; D] =
                            value.try_into().map_err(|_| AkitaError::InvalidProof)?;
                        let product = CyclotomicRing::<F, D>::from_coefficients(coeff_arr)
                            * CyclotomicRing::<F, D>::from_coefficients(value_arr);
                        Ok(eval_ring_at_pows_fast(&product, alpha_pows))
                    }
                )
            }
        }
    }

    /// Constant coefficient of `a[idx]`, if it is known to be constant.
    pub fn a_constant_coeff(&self, idx: usize) -> Option<F> {
        match self {
            Self::Base(point) => point.a.get(idx).copied(),
            Self::Ring { a, .. } => {
                let ring_dim = a.ring_dim();
                if ring_dim == 0 {
                    return None;
                }
                let chunk = a.coeffs().chunks_exact(ring_dim).nth(idx)?;
                flat_ring_is_constant(chunk, ring_dim).then(|| chunk[0])
            }
        }
    }

    /// Constant coefficient of `b[idx]`, if it is known to be constant.
    pub fn b_constant_coeff(&self, idx: usize) -> Option<F> {
        match self {
            Self::Base(point) => point.b.get(idx).copied(),
            Self::Ring { b, .. } => {
                let ring_dim = b.ring_dim();
                if ring_dim == 0 {
                    return None;
                }
                let chunk = b.coeffs().chunks_exact(ring_dim).nth(idx)?;
                flat_ring_is_constant(chunk, ring_dim).then(|| chunk[0])
            }
        }
    }
}

fn flat_ring_is_constant<F: FieldCore>(coeffs: &[F], ring_dim: usize) -> bool {
    ring_dim > 0 && coeffs.len() == ring_dim && coeffs[1..].iter().all(|coeff| coeff.is_zero())
}

fn flat_rings_are_constant<F: FieldCore>(coeffs: &[F], ring_dim: usize) -> bool {
    ring_dim > 0
        && coeffs.len().is_multiple_of(ring_dim)
        && coeffs
            .chunks_exact(ring_dim)
            .all(|chunk| flat_ring_is_constant(chunk, ring_dim))
}

fn ring_multiplier_opening_point_from_ext<F, E, const D: usize>(
    opening_point: &[E],
    layout: OpeningBlockLayout,
    basis: BasisMode,
) -> Result<RingMultiplierOpeningPoint<F>, AkitaError>
where
    F: FieldCore + akita_field::FromPrimitiveInt,
    E: FpExtEncoding<F>,
{
    let position_bits = layout.position_stride().trailing_zeros() as usize;
    let block_bits = layout.num_blocks().trailing_zeros() as usize;
    let expected_len = position_bits
        .checked_add(block_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() != expected_len {
        return Err(AkitaError::InvalidPointDimension {
            expected: expected_len,
            actual: opening_point.len(),
        });
    }

    let a_weights = basis_weights(&opening_point[..position_bits], basis)?;
    let b_weights = basis_weights(&opening_point[position_bits..], basis)?;
    let error = AkitaError::InvalidInput(
        "opening point does not encode in the ring-subfield basis".to_string(),
    );
    let a = a_weights
        .into_iter()
        .map(|weight| embed_ring_subfield_scalar::<F, E, D>(weight, error.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let b = b_weights
        .into_iter()
        .map(|weight| embed_ring_subfield_scalar::<F, E, D>(weight, error.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RingMultiplierOpeningPoint::from_ring(a, b))
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

/// Absorb the batch commitment into the transcript using the D-free flat
/// coefficient encoding under the schedule-derived `ring_dim`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] if the stored buffer is not well-formed
/// for `ring_dim`.
pub fn append_batched_commitments_to_transcript<F, T>(
    commitment: &Commitment<F>,
    ring_dim: usize,
    transcript: &mut T,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    commitment.append_to_transcript(ABSORB_COMMITMENT, ring_dim, transcript)
}

/// Largest natural arity across polynomials in a scalar batched commit/prove call.
///
/// Matches `prepare_batched_commit_inputs`, which selects the root layout from
/// the maximum `num_vars` across the bundled polynomials.
///
/// # Errors
///
/// Returns an error if `poly_num_vars` is empty.
pub fn padded_scalar_batch_num_vars(
    poly_num_vars: impl IntoIterator<Item = usize>,
) -> Result<usize, AkitaError> {
    poly_num_vars.into_iter().max().ok_or_else(|| {
        AkitaError::InvalidInput(
            "batched opening batch requires at least one polynomial".to_string(),
        )
    })
}

/// Opening point length must match the padded batch domain selected at commit time.
///
/// # Errors
///
/// Returns an error when `point_len` and `padded_num_vars` differ.
pub fn validate_scalar_point_matches_poly_arity(
    point_len: usize,
    padded_num_vars: usize,
) -> Result<(), AkitaError> {
    if point_len != padded_num_vars {
        return Err(AkitaError::InvalidInput(format!(
            "opening point length {point_len} does not match padded batch domain {padded_num_vars}"
        )));
    }
    Ok(())
}

/// Validate common batched prove/verify input shape constraints.
///
/// # Errors
///
/// Returns an error if the shared opening point exceeds setup capacity, the
/// payload is empty, or the claim count exceeds setup capacity.
pub fn validate_batched_inputs<F, E>(
    setup: &AkitaExpandedSetup<F>,
    point: &[E],
    group_sizes: &[usize],
    for_prover: bool,
) -> Result<(), AkitaError>
where
    F: FieldCore,
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

    let num_vars = point.len();
    if num_vars > setup.seed().max_num_vars {
        return Err(AkitaError::InvalidInput(format!(
            "{label} received opening points with {} variables but setup supports at most {}",
            num_vars,
            setup.seed().max_num_vars
        )));
    }
    if group_sizes.is_empty() {
        return Err(shape_error(format!(
            "{label} requires at least one commitment group",
        )));
    }
    if group_sizes.contains(&0) {
        return Err(shape_error(format!(
            "{label} commitment groups must be nonempty",
        )));
    }
    let num_claims = checked_total_claims(group_sizes, label)?;
    if num_claims == 0 {
        return Err(shape_error(format!(
            "{label} requires at least one claimed opening",
        )));
    }
    if num_claims > setup.seed().max_num_batched_polys {
        if for_prover {
            return Err(AkitaError::InvalidInput(format!(
                "batched_prove received {num_claims} polynomials but setup supports at most {}",
                setup.seed().max_num_batched_polys
            )));
        }
        return Err(AkitaError::InvalidProof);
    }

    Ok(())
}

/// Prepare a recursive opening point whose coordinates may live in the proof
/// scalar field `E`, while the resulting ring payload remains over `F`.
///
/// For degree-one `E`, this is the original recursive materialization path:
/// coordinates are converted to base scalars, outer variables are prepared by
/// [`ring_opening_point_from_field`], and the inner point is reduced by
/// [`reduce_inner_opening_to_ring_element`]. For true extension-valued `E`,
/// the currently supported shape is the same explicit ring-subfield boundary
/// as the root folded path: all live variables must fit in the packed inner
/// slots and there can be no outer block variables.
///
/// # Errors
///
/// Returns an error when the point length is invalid, the extension degree is
/// unsupported by the ring-subfield dispatcher, or the level has outer
/// variables that require the later split/Frobenius route.
pub fn prepare_opening_point<F, E, const D: usize>(
    opening_point: &[E],
    basis: BasisMode,
    layout: OpeningBlockLayout,
    alpha_bits: usize,
) -> Result<PreparedOpeningPoint<F, E>, AkitaError>
where
    F: FieldCore + akita_field::FromPrimitiveInt,
    E: FpExtEncoding<F>,
{
    let _span = tracing::info_span!("ring_opening_point").entered();
    let outer_bits = (layout.position_stride().trailing_zeros() as usize)
        .checked_add(layout.num_blocks().trailing_zeros() as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    let target_num_vars = outer_bits
        .checked_add(alpha_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() > target_num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: target_num_vars,
            actual: opening_point.len(),
        });
    }
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, E::zero());

    if E::EXT_DEGREE == 1 {
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
        let ring_opening_point = ring_opening_point_from_field::<F>(outer_point, layout, basis)?;
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&ring_opening_point);
        let packed_inner_point = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis)?;
        return Ok(PreparedOpeningPoint::from_parts::<D>(
            padded_point,
            ring_opening_point,
            ring_multiplier_point,
            packed_inner_point,
        ));
    }

    if !D.is_multiple_of(E::EXT_DEGREE) || !(D / E::EXT_DEGREE).is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "challenge-field degree must divide the ring dimension into power-of-two slots"
                .to_string(),
        ));
    }

    let trace_inner_point_len = (D / E::EXT_DEGREE).trailing_zeros() as usize;
    if padded_point[trace_inner_point_len..alpha_bits]
        .iter()
        .any(|coord| !coord.is_zero())
    {
        return Err(AkitaError::InvalidInput(
            "inactive extension inner coordinates must be zero after psi packing".to_string(),
        ));
    }
    let trace_inner_weights = basis_weights(&padded_point[..trace_inner_point_len], basis)?;
    let packed_inner_point = embed_ring_subfield_vector::<F, E, D>(
        &trace_inner_weights,
        AkitaError::InvalidInput(
            "recursive opening point does not encode in the ring-subfield basis".to_string(),
        ),
    )?;
    let outer_point = &padded_point[alpha_bits..];
    let ring_multiplier_point =
        ring_multiplier_opening_point_from_ext::<F, E, D>(outer_point, layout, basis)?;
    let ring_opening_point =
        ring_opening_point_from_field::<F>(&vec![F::zero(); outer_point.len()], layout, basis)?;

    Ok(PreparedOpeningPoint::from_parts::<D>(
        padded_point,
        ring_opening_point,
        ring_multiplier_point,
        packed_inner_point,
    ))
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
    if !k.is_power_of_two() || !D.is_multiple_of(k) {
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
pub fn folded_root_supports_opening_shape<F, E, const D: usize>(
    opening_points: &[&[E]],
    lp: &LevelParams,
    alpha_bits: usize,
) -> bool
where
    F: FieldCore,
    E: ExtField<F>,
{
    if E::EXT_DEGREE == 1 {
        return true;
    }
    if !D.is_multiple_of(E::EXT_DEGREE) || !(D / E::EXT_DEGREE).is_power_of_two() {
        return false;
    }
    let packed_slots = D / E::EXT_DEGREE;
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
///
/// `ring_d` is the schedule-derived root ring dimension (plain usize math —
/// no typed ring work happens here).
pub fn root_tensor_projection_enabled<F, E>(ring_d: usize, num_vars: usize) -> bool
where
    F: FieldCore,
    E: ExtField<F>,
{
    let width = E::EXT_DEGREE;
    let Some(double_width) = width.checked_mul(2) else {
        return false;
    };
    width > 1
        && width.is_power_of_two()
        && ring_d.is_power_of_two()
        && ring_d >= double_width
        && ring_d.is_multiple_of(width)
        && num_vars >= ring_d.trailing_zeros() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SisModulusFamily;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::{Fp32, FpExt4, LiftBase};

    type F = Fp32<251>;
    type E = FpExt4<F>;

    fn packed_inner_lp() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q32,
            32,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(0, 0, 1, 1, 32)
        .unwrap()
    }

    #[test]
    fn recursive_extension_opening_preparation_uses_ring_subfield_boundary() {
        let point = [E::lift_base(F::from_u64(3)), E::lift_base(F::from_u64(5))];

        let prepared = prepare_opening_point::<F, E, 32>(
            &point,
            BasisMode::Lagrange,
            OpeningBlockLayout::new(1, 1).unwrap(),
            5,
        )
        .expect("packed-inner recursive extension point should prepare");

        assert_eq!(prepared.padded_point.len(), 5);
    }

    #[test]
    fn packed_extension_opening_point_exposes_basis_slots() {
        let point = [
            E::lift_base(F::from_u64(1)),
            E::lift_base(F::from_u64(2)),
            E::lift_base(F::from_u64(3)),
            E::lift_base(F::from_u64(4)),
        ];

        let transformed =
            ring_subfield_packed_extension_opening_point::<F, E, 32>(point.len(), &point)
                .expect("packed extension point");

        assert_eq!(
            transformed,
            vec![point[0], point[1], point[2], E::zero(), E::zero(), point[3]]
        );
    }

    #[test]
    fn extension_challenge_folded_root_gate_accepts_same_point_batching() {
        let lp = packed_inner_lp();
        let point = [F::from_u64(7), F::from_u64(11)];

        assert!(folded_root_supports_opening_shape::<F, F, 32>(
            &[&point[..]],
            &lp,
            5,
        ));
        assert!(folded_root_supports_opening_shape::<F, F, 32>(
            &[&point[..]],
            &lp,
            5,
        ));
    }

    #[test]
    fn root_tensor_projection_gate_requires_room_for_signed_subfield_basis() {
        assert!(root_tensor_projection_enabled::<F, E>(8, 3));
        assert!(!root_tensor_projection_enabled::<F, E>(4, 2));
    }

    #[test]
    fn padded_scalar_batch_num_vars_uses_max_poly_arity() {
        assert_eq!(
            padded_scalar_batch_num_vars([12, 20, 18]).expect("nonempty"),
            20
        );
    }

    #[test]
    fn validate_scalar_point_matches_poly_arity_rejects_shorter_point() {
        let err = validate_scalar_point_matches_poly_arity(18, 20)
            .expect_err("shorter point must reject");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }

    #[test]
    fn validate_scalar_point_matches_poly_arity_accepts_match() {
        validate_scalar_point_matches_poly_arity(20, 20)
            .expect("matching point length should validate");
    }
}
