//! Shared batching and root-opening helper types.

mod ring_multiplier;
mod subfield;

use crate::{
    basis_weights, basis_weights_prefix, embed_ring_subfield_vector,
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AkitaExpandedSetup,
    BasisMode, Commitment, CommittedGroupParams, FpExtEncoding, RingVec,
};
use akita_algebra::CyclotomicRing;
use akita_error::{checked, AkitaError};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVAL_OPENINGS_FIELD};
use akita_transcript::{append_ext_field, Transcript};
use jolt_field::{CanonicalEncoding, ExtField, Field};

pub use ring_multiplier::{PreparedRingMultiplier, RingMultiplierOpeningPoint};
pub use subfield::SubfieldMultiplierOpeningPoint;

/// Recursive opening point prepared for ring-level replay.
///
/// Ring dimension is stored at runtime; hot paths inside `dispatch_ring_dim`
/// borrow the ψ-packed inner ring via [`Self::packed_inner_trusted`].
#[derive(Debug, Clone)]
pub struct PreparedOpeningPoint<F: Field, E: Field> {
    /// Opening point padded to the recursive verifier's target variable count.
    pub padded_point: Vec<E>,
    /// Ring-level outer opening point with weights embedded as `R_F` multipliers.
    pub ring_multiplier_point: RingMultiplierOpeningPoint<F>,
    /// The ψ-packed inner block of the opening point (paper `\check{r}_{\mathrm{in}}`).
    ///
    /// Public fixed weight in `TraceOpen(Y) = recover_ring_subfield_inner_product(Y, packed_inner_point)`.
    /// Hot paths borrow via [`Self::packed_inner_trusted`].
    packed_inner_point: RingVec<F>,
    ring_dim: usize,
}

impl<F: Field, E: Field> PreparedOpeningPoint<F, E> {
    /// Construct from typed kernel output at an opening-point boundary.
    pub fn from_parts<const D: usize>(
        padded_point: Vec<E>,
        ring_multiplier_point: RingMultiplierOpeningPoint<F>,
        packed_inner_point: CyclotomicRing<F, D>,
    ) -> Self {
        Self {
            padded_point,
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
        self.packed_inner_point.as_single_ring::<D>()
    }

    /// Owned copy of the ψ-packed inner ring after [`Self::ensure_ring_dim`].
    pub fn packed_inner_owned<const D: usize>(&self) -> Result<CyclotomicRing<F, D>, AkitaError> {
        self.ensure_ring_dim::<D>()?;
        self.packed_inner_point.try_to_single::<D>()
    }
}

fn ring_multiplier_opening_point_from_ext<F, E, const D: usize>(
    opening_point: &[E],
    num_positions_per_block: usize,
    num_live_blocks: usize,
    basis: BasisMode,
) -> Result<RingMultiplierOpeningPoint<F>, AkitaError>
where
    F: Field + jolt_field::Ring,
    E: FpExtEncoding<F>,
{
    if !num_positions_per_block.is_power_of_two() || num_live_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "opening geometry requires power-of-two M and positive B".to_string(),
        ));
    }
    let position_index_bits =
        crate::BlockGeometry::position_index_bits_for(num_positions_per_block);
    let block_index_bits = crate::BlockGeometry::checked_block_index_bits_for(num_live_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("block-index domain size overflow".to_string()))?;
    let expected_len = position_index_bits
        .checked_add(block_index_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() != expected_len {
        return Err(AkitaError::InvalidPointDimension {
            expected: expected_len,
            actual: opening_point.len(),
        });
    }

    let position_weights = basis_weights(&opening_point[..position_index_bits], basis)?;
    let live_block_weights = basis_weights_prefix(
        &opening_point[position_index_bits..],
        basis,
        num_live_blocks,
    )?;
    let error = AkitaError::InvalidInput(
        "opening point does not encode in the ring-subfield basis".to_string(),
    );
    SubfieldMultiplierOpeningPoint::new::<E, D>(&position_weights, &live_block_weights, error)
        .map(RingMultiplierOpeningPoint::Subfield)
}

/// Absorb public claim-field evaluations into the base-field transcript.
pub fn append_claim_values_to_transcript<F, E, T>(values: &[E], transcript: &mut T)
where
    F: Field + CanonicalEncoding + AkitaSerialize,
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
    checked::sum(group_sizes.iter().copied())
        .ok_or_else(|| AkitaError::InvalidInput(format!("{label} total claim count overflow")))
}

/// Absorb the batch commitment into the transcript using the D-free flat
/// coefficient encoding under its derived terminal compression `ring_dim`.
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
    F: Field + CanonicalEncoding + AkitaSerialize,
    T: Transcript<F>,
{
    commitment.append_to_transcript(ABSORB_COMMITMENT, ring_dim, transcript)
}

/// Validate common batched prove/verify input shape constraints.
///
/// # Errors
///
/// Returns an error if the group-local opening point exceeds setup capacity, the
/// payload is empty, or the claim count exceeds setup capacity.
pub fn validate_batched_inputs<F, E>(
    setup: &AkitaExpandedSetup<F>,
    point: &[E],
    group_sizes: &[usize],
    for_prover: bool,
) -> Result<(), AkitaError>
where
    F: Field,
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
    if num_vars > setup.descriptor().max_num_vars {
        return Err(AkitaError::InvalidInput(format!(
            "{label} received opening points with {} variables but setup supports at most {}",
            num_vars,
            setup.descriptor().max_num_vars
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
    if num_claims > setup.descriptor().max_num_batched_polys {
        if for_prover {
            return Err(AkitaError::InvalidInput(format!(
                "batched_prove received {num_claims} polynomials but setup supports at most {}",
                setup.descriptor().max_num_batched_polys
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
    num_positions_per_block: usize,
    num_live_blocks: usize,
    alpha_bits: usize,
) -> Result<PreparedOpeningPoint<F, E>, AkitaError>
where
    F: Field + jolt_field::Ring,
    E: FpExtEncoding<F>,
{
    let _span = tracing::info_span!("ring_opening_point").entered();
    if !num_positions_per_block.is_power_of_two() || num_live_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "opening geometry requires power-of-two M and positive B".to_string(),
        ));
    }
    let block_index_bits = crate::BlockGeometry::checked_block_index_bits_for(num_live_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("block-index domain size overflow".to_string()))?;
    let outer_bits = crate::BlockGeometry::position_index_bits_for(num_positions_per_block)
        .checked_add(block_index_bits)
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

    if E::DEGREE == 1 {
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
            num_positions_per_block,
            num_live_blocks,
            basis,
        )?;
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&ring_opening_point);
        let packed_inner_point = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis)?;
        return Ok(PreparedOpeningPoint::from_parts::<D>(
            padded_point,
            ring_multiplier_point,
            packed_inner_point,
        ));
    }

    if !D.is_multiple_of(E::DEGREE) || !(D / E::DEGREE).is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "challenge-field degree must divide the ring dimension into power-of-two slots"
                .to_string(),
        ));
    }

    let trace_inner_point_len = (D / E::DEGREE).trailing_zeros() as usize;
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
    let ring_multiplier_point = ring_multiplier_opening_point_from_ext::<F, E, D>(
        outer_point,
        num_positions_per_block,
        num_live_blocks,
        basis,
    )?;
    Ok(PreparedOpeningPoint::from_parts::<D>(
        padded_point,
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
    F: Field,
    E: ExtField<F>,
{
    let k = E::DEGREE;
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
/// in one group are handled by one public row per group, with row-local
/// extension batching coefficients embedded into the ring relation.
pub fn folded_root_supports_opening_shape<F, E, const D: usize>(
    opening_points: &[&[E]],
    lp: &CommittedGroupParams,
    alpha_bits: usize,
) -> bool
where
    F: Field,
    E: ExtField<F>,
{
    if E::DEGREE == 1 {
        return true;
    }
    if !D.is_multiple_of(E::DEGREE) || !(D / E::DEGREE).is_power_of_two() {
        return false;
    }
    let packed_slots = D / E::DEGREE;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if packed_inner_bits > alpha_bits {
        return false;
    }
    let target_num_vars = match lp
        .position_index_bits()
        .checked_add(lp.block_index_bits())
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

#[cfg(test)]
mod high_half_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SisModulusProfileId;
    use akita_algebra::ring::{eval_ring_at_pows_fast, scalar_powers};
    use akita_challenges::SparseChallengeConfig;
    use jolt_field::{Ext2, ExtField, Fp32, FpExt4, FpExt8, MulBaseUnreduced, Ring, Zero};

    type F = Fp32<251>;
    type E = FpExt4<F>;

    fn packed_inner_lp() -> CommittedGroupParams {
        CommittedGroupParams::params_only(
            SisModulusProfileId::Q32Offset99,
            32,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(1, 32, 1, 1, 1)
        .unwrap()
    }

    #[test]
    fn recursive_extension_opening_preparation_uses_ring_subfield_boundary() {
        let point = [E::lift_base(F::from_u64(3)), E::lift_base(F::from_u64(5))];

        let prepared = prepare_opening_point::<F, E, 32>(&point, BasisMode::Lagrange, 1, 1, 5)
            .expect("packed-inner recursive extension point should prepare");

        assert_eq!(prepared.padded_point.len(), 5);
    }

    #[test]
    fn extension_opening_preparation_keeps_exact_live_block_prefix() {
        let mut point = vec![E::zero(); 9];
        point[0] = E::lift_base(F::from_u64(3));
        point[1] = E::lift_base(F::from_u64(5));
        point[2] = E::lift_base(F::from_u64(7));
        point[5] = E::lift_base(F::from_u64(11));
        point[6] = E::lift_base(F::from_u64(13));
        point[7] = E::lift_base(F::from_u64(17));
        point[8] = E::lift_base(F::from_u64(19));

        let prepared = prepare_opening_point::<F, E, 32>(&point, BasisMode::Lagrange, 4, 3, 5)
            .expect("extension opening point should retain exact F");
        assert_eq!(prepared.ring_multiplier_point.position_len(), 4);
        assert_eq!(prepared.ring_multiplier_point.fold_len(), 3);
    }

    fn check_compact_subfield_multiplier<L>()
    where
        L: FpExtEncoding<F> + MulBaseUnreduced<F>,
    {
        const D: usize = 32;
        let value = L::from_base_slice(
            &(0..L::DEGREE)
                .map(|index| F::from_u64((index + 2) as u64))
                .collect::<Vec<_>>(),
        );
        let point = SubfieldMultiplierOpeningPoint::new::<L, D>(
            &[value],
            &[value],
            AkitaError::InvalidProof,
        )
        .map(RingMultiplierOpeningPoint::Subfield)
        .expect("valid compact subfield multiplier");
        assert!(matches!(point, RingMultiplierOpeningPoint::Subfield(_)));
        assert_eq!(point.position_len(), 1);
        assert_eq!(point.fold_len(), 1);
        assert!(point.ensure_ring_dim::<64>().is_err());

        let expected_ring =
            crate::embed_ring_subfield_scalar::<F, L, D>(value, AkitaError::InvalidProof)
                .expect("reference ring embedding");
        assert_eq!(
            point
                .materialize_position_rings::<D>()
                .expect("materialized point")
                .expect("proper extension rings"),
            vec![expected_ring]
        );
        assert_eq!(
            point
                .fold_subfield_value::<L>(0)
                .expect("decoded fold value"),
            Some(value)
        );

        let alpha = L::from_base_slice(
            &(0..L::DEGREE)
                .map(|index| F::from_u64((index + 7) as u64))
                .collect::<Vec<_>>(),
        );
        let alpha_pows = scalar_powers(alpha, D);
        assert_eq!(
            point
                .eval_position_at::<L>(0, &alpha_pows)
                .expect("compact evaluation"),
            eval_ring_at_pows_fast(&expected_ring, &alpha_pows)
        );

        let rhs = CyclotomicRing::from_coefficients(std::array::from_fn(|index| {
            F::from_u64((3 * index + 1) as u64)
        }));
        let mut actual = CyclotomicRing::zero();
        point
            .accumulate_position_product(0, &rhs, &mut actual)
            .expect("compact product");
        assert_eq!(actual, expected_ring * rhs);

        let mut actual = CyclotomicRing::zero();
        point
            .as_subfield()
            .expect("proper extension multipliers")
            .accumulate_fold_product(0, &rhs, &mut actual)
            .expect("compact fold product");
        assert_eq!(actual, expected_ring * rhs);

        let scale = F::from_u64(23);
        for shift in [0, D / 2, D - 1] {
            let mut actual = CyclotomicRing::zero();
            point
                .as_subfield()
                .expect("proper extension multipliers")
                .accumulate_position_monomial(0, shift, scale, &mut actual)
                .expect("compact shifted monomial");
            assert_eq!(actual, expected_ring.negacyclic_shift(shift).scale(&scale));
        }
    }

    #[test]
    fn compact_subfield_multipliers_match_ring_oracle() {
        check_compact_subfield_multiplier::<Ext2<F>>();
        check_compact_subfield_multiplier::<FpExt4<F>>();
        check_compact_subfield_multiplier::<FpExt8<F>>();
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
}
