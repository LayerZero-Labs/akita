//! Direct public coefficient weights for the two binary terminal identities.

use akita_algebra::ring::{shifted_remainder_evaluations, PlusTrinomial};
use akita_challenges::{Challenges, FoldChallengeDistribution};
use akita_error::{checked, AkitaError};
use akita_params::{RelationPolynomial, RelationRowGeometry};
use jolt_field::Field;

use super::{
    binary_scalar_lift, BinaryEvaluationBounds, BinarySourceDescriptor, BinaryTerminalClaim,
    BINARY_PARITY_CARRY_DIGITS, BINARY_VALUE_CARRY_DIGITS,
};
use crate::native_root_relation::{NativeRootWeights, RootEq, RootWeightFamily, SegmentLanes};
use crate::root_repacking::RootCoefficientRepacking;

/// Addresses of the response low digit, authenticated U view and both carries.
#[derive(Clone, Copy, Debug)]
pub struct BinaryEvaluationSegments {
    pub response: usize,
    pub partials: usize,
    pub parity_carry: usize,
    pub value_carry: usize,
}

impl BinaryTerminalClaim {
    /// Compile parity and value identities into the fused Stage-2 public oracle.
    ///
    /// The response reads its low balanced digit only, undoing signed packing.
    /// U and both carries read the same outgoing table as the ordinary suffix.
    /// Two independently sampled batching weights combine the identities after
    /// U, K and K_val have been fixed by their commitments.
    #[allow(clippy::too_many_arguments)]
    pub fn relation_weights<E: Field>(
        &self,
        descriptor: &BinarySourceDescriptor,
        repacking: &RootCoefficientRepacking,
        segments: BinaryEvaluationSegments,
        challenges: &Challenges,
        alpha: E,
        batches: [E; 2],
        proof_modulus: u128,
    ) -> Result<NativeRootWeights<E>, AkitaError> {
        let (row_weights, column_weights) = self.equality_weights(descriptor)?;
        let columns = descriptor.index_map().columns();
        let scalar_rows = row_weights.len();
        BinaryEvaluationBounds::new(scalar_rows, columns, proof_modulus)?;
        let ring_rows = checked::exact_div(scalar_rows, 4).ok_or(AkitaError::InvalidProof)?;
        let owner = descriptor
            .polynomials()
            .iter()
            .position(|id| id == self.polynomial())
            .ok_or(AkitaError::InvalidProof)?;
        let polynomials = descriptor.polynomials().len();
        let expected_response =
            checked::product([polynomials, ring_rows]).ok_or(AkitaError::InvalidProof)?;
        let expected_partials =
            checked::product([polynomials, columns]).ok_or(AkitaError::InvalidProof)?;
        let response_shape = repacking
            .segments()
            .get(segments.response)
            .ok_or(AkitaError::InvalidProof)?;
        if response_shape.elements != expected_response
            || response_shape.digits != 4
            || response_shape.geometry.polynomial() != RelationPolynomial::minus_trinomial(648)?
            || response_shape.geometry.coordinate_plane_count() != 1
        {
            return Err(AkitaError::InvalidProof);
        }
        for (segment, elements, digits) in [
            (segments.partials, expected_partials, 1),
            (
                segments.parity_carry,
                polynomials,
                BINARY_PARITY_CARRY_DIGITS,
            ),
            (segments.value_carry, polynomials, BINARY_VALUE_CARRY_DIGITS),
        ] {
            let shape = repacking
                .segments()
                .get(segment)
                .ok_or(AkitaError::InvalidProof)?;
            if shape.elements != elements
                || shape.digits != digits
                || shape.geometry.polynomial() != RelationPolynomial::plus_trinomial(162)?
                || shape.geometry.coordinate_plane_count() != 1
            {
                return Err(AkitaError::InvalidProof);
            }
        }
        let scalar_challenges = binary_scalar_challenges(descriptor, challenges)?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(scalar_rows)
            .map_err(|_| AkitaError::InvalidProof)?;
        rows.extend(row_weights.into_iter().map(|weight| weight.to_words()));
        let parity_rows = ParityRowWeights {
            response: SegmentLanes::new(repacking, segments.response)?,
            table_len: repacking.domain_len(),
            alpha,
            batch: batches[0],
            first_element: checked::product([owner, ring_rows]).ok_or(AkitaError::InvalidProof)?,
            rows,
        };
        let mut terms = Vec::new();
        for (column, weight) in column_weights.into_iter().enumerate() {
            let challenge_index =
                checked::mul_add(owner, columns, column).ok_or(AkitaError::InvalidProof)?;
            let challenge = scalar_challenges
                .get(challenge_index)
                .ok_or(AkitaError::InvalidProof)?;
            let multiplier = binary_scalar_lift(weight).map(E::from_i128);
            let mut value_kernel = [E::zero(); 162];
            let mut parity_kernel = [E::zero(); 162];
            shifted_remainder_evaluations::<E, PlusTrinomial>(
                &multiplier,
                alpha,
                &mut value_kernel,
            )
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            shifted_remainder_evaluations::<E, PlusTrinomial>(
                &challenge.map(E::from_i128),
                alpha,
                &mut parity_kernel,
            )
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            for (coefficient, (value, parity)) in
                value_kernel.into_iter().zip(parity_kernel).enumerate()
            {
                terms.extend(repacking.coefficient_weights(
                    segments.partials,
                    challenge_index,
                    coefficient,
                    batches[1] * value - batches[0] * parity,
                )?);
            }
        }
        let mut power = E::one();
        for coefficient in 0..162 {
            for (segment, batch) in [
                (segments.parity_carry, batches[0]),
                (segments.value_carry, batches[1]),
            ] {
                terms.extend(repacking.coefficient_weights(
                    segment,
                    owner,
                    coefficient,
                    -E::from_u64(2) * batch * power,
                )?);
            }
            power *= alpha;
        }
        let value = akita_algebra::binary::BinaryField162::from_bytes(self.value())
            .ok_or(AkitaError::InvalidProof)?;
        let target = binary_scalar_lift(value)
            .into_iter()
            .rev()
            .fold(E::zero(), |acc, coefficient| {
                acc * alpha + E::from_i128(coefficient)
            });
        NativeRootWeights::structured(
            repacking.domain_len(),
            terms,
            vec![RootWeightFamily::ParityRows(parity_rows)],
            batches[1] * target,
        )
    }
}

/// Bits per Four Russians chunk of a 162-bit scalar row.
const PARITY_CHUNK_BITS: usize = 8;
/// Chunks covering the 162 scalar bits; the last holds bits 160 and 161.
const PARITY_CHUNKS: usize = 162usize.div_ceil(PARITY_CHUNK_BITS);

/// The parity identity's row part: every scalar row of the claimed owner
/// against its response element's low digit.
///
/// Row `r` weighs packed coefficient `4c + r mod 4` by
/// `batch * (-1)^c * SRE(bits_r, alpha)[c]`. Response lanes start equality
/// blocks, and by the Hankel symmetry of the shifted remainders the row reads
/// `high * sum_{s in bits_r} P_k[s]`, where
/// `P_k = SRE(((-1)^c low[4c+k])_c, alpha)` and `k = r mod 4`. The verifier
/// answers these subset sums with Four Russians tables: 21 lookups per row
/// after `4 * 21 * 256` table additions. The prover needs each row's full
/// kernel, which one shifted-remainder pass gives in fewer operations than
/// 21 table vectors of 162 entries, so it keeps that pass.
#[derive(Clone, Debug)]
pub(crate) struct ParityRowWeights<E: Field> {
    response: SegmentLanes,
    table_len: usize,
    alpha: E,
    batch: E,
    first_element: usize,
    /// Canonical 162-bit row weights as little-endian words.
    rows: Vec<[u64; 3]>,
}

impl<E: Field> ParityRowWeights<E> {
    pub(crate) fn table_len(&self) -> usize {
        self.table_len
    }

    fn element(&self, row: usize) -> Result<usize, AkitaError> {
        checked::sum([self.first_element, row / 4]).ok_or(AkitaError::InvalidProof)
    }

    pub(crate) fn evaluate(&self, eq: &RootEq<E>) -> Result<E, AkitaError> {
        let low = eq.low();
        let mut tables = Vec::new();
        tables
            .try_reserve_exact((4 * PARITY_CHUNKS) << PARITY_CHUNK_BITS)
            .map_err(|_| AkitaError::InvalidProof)?;
        for k in 0..4 {
            let mut signed = [E::zero(); 162];
            for (c, slot) in signed.iter_mut().enumerate() {
                let value = *low.get(4 * c + k).ok_or(AkitaError::InvalidProof)?;
                *slot = if c % 2 == 0 { value } else { -value };
            }
            let mut sums = [E::zero(); 162];
            shifted_remainder_evaluations::<E, PlusTrinomial>(&signed, self.alpha, &mut sums)
                .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            for chunk in 0..PARITY_CHUNKS {
                let start = tables.len();
                tables.push(E::zero());
                for mask in 1usize..1 << PARITY_CHUNK_BITS {
                    let bit = chunk * PARITY_CHUNK_BITS + mask.trailing_zeros() as usize;
                    let rest = *tables
                        .get(start + (mask & (mask - 1)))
                        .ok_or(AkitaError::InvalidProof)?;
                    tables.push(match sums.get(bit) {
                        Some(&value) => rest + value,
                        None => rest,
                    });
                }
            }
        }
        let mut sum = E::zero();
        for (quad, rows) in self.rows.chunks(4).enumerate() {
            let mut lanes = E::zero();
            for (k, words) in rows.iter().enumerate() {
                let table = (k * PARITY_CHUNKS) << PARITY_CHUNK_BITS;
                for chunk in 0..PARITY_CHUNKS {
                    let bit = chunk * PARITY_CHUNK_BITS;
                    let word = *words.get(bit / 64).ok_or(AkitaError::InvalidProof)?;
                    let mask = (word >> (bit % 64)) as usize & ((1 << PARITY_CHUNK_BITS) - 1);
                    lanes += *tables
                        .get(table + (chunk << PARITY_CHUNK_BITS) + mask)
                        .ok_or(AkitaError::InvalidProof)?;
                }
            }
            let base = self.response.base(self.element(4 * quad)?, 0)?;
            if eq.low_offset(base) != 0 {
                return Err(AkitaError::InvalidSetup(
                    "root lane does not start an equality block".into(),
                ));
            }
            sum += eq.high(base)? * lanes;
        }
        Ok(self.batch * sum)
    }

    pub(crate) fn visit(
        &self,
        mut visit: impl FnMut(usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        for (row, words) in self.rows.iter().enumerate() {
            let multiplier: [E; 162] = std::array::from_fn(|index| {
                let word = words.get(index / 64).copied().unwrap_or(0);
                E::from_u64((word >> (index % 64)) & 1)
            });
            let mut kernel = [E::zero(); 162];
            shifted_remainder_evaluations::<E, PlusTrinomial>(&multiplier, self.alpha, &mut kernel)
                .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            let base = self.response.base(self.element(row)?, 0)?;
            for (coefficient, weight) in kernel.into_iter().enumerate() {
                let signed = if coefficient % 2 == 0 {
                    weight
                } else {
                    -weight
                };
                visit(base + 4 * coefficient + row % 4, self.batch * signed)?;
            }
        }
        Ok(())
    }
}

/// Validate and lift the replayed bounded-weight binary fold challenges.
pub fn binary_scalar_challenges(
    descriptor: &BinarySourceDescriptor,
    challenges: &Challenges,
) -> Result<Vec<[i128; 162]>, AkitaError> {
    if challenges.num_claims() != descriptor.polynomials().len()
        || challenges.num_live_blocks_per_claim() != descriptor.index_map().columns()
    {
        return Err(AkitaError::InvalidProof);
    }
    challenges
        .as_slice()
        .iter()
        .map(|challenge| {
            challenge.validate_dyn(162)?;
            if challenge.positions.len() > FoldChallengeDistribution::BinaryRoot.weight()
                || challenge
                    .coeffs
                    .iter()
                    .any(|&coefficient| coefficient != -1 && coefficient != 1)
            {
                return Err(AkitaError::InvalidProof);
            }
            let mut scalar = [0; 162];
            for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                *scalar
                    .get_mut(position as usize)
                    .ok_or(AkitaError::InvalidProof)? = i128::from(coefficient);
            }
            Ok(scalar)
        })
        .collect()
}

/// Standard scalar-ring geometry for the bounded U and carry segments.
pub fn binary_scalar_geometry() -> Result<RelationRowGeometry, AkitaError> {
    RelationRowGeometry::for_polynomial(RelationPolynomial::plus_trinomial(162)?, 1, 256)
}
