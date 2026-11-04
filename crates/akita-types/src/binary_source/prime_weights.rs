//! Public coefficient weights authenticating committed prime evaluation planes.

use akita_algebra::ring::{shifted_remainder_evaluations, PlusTrinomial};
use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use akita_params::{binary_setup::BINARY_SETUP_P, RelationPolynomial};
use jolt_field::Field;

use super::{
    binary_scalar_challenges, BinarySourceDescriptor, PrimePlaneManifest, PRIME_PLANE_DIGITS,
};
use crate::native_root_relation::{
    weigh_units, NativeRootWeights, RootEq, RootWeightFamily, SegmentLanes,
};
use crate::root_repacking::RootCoefficientRepacking;

/// Compile each plane's ring identity and scalar reading into Stage 2.
///
/// Two independent batches per plane are sampled after the outgoing commitment.
/// The first checks `sum e_r V_r - sum c_b EP_b = 0` at `alpha`, with the
/// complete four-digit response and the signed packing undone. The second
/// checks `sum col_b <omega, EP_b> = value`. Plane elements are ordered by
/// unique manifest plane, then column; their coefficients have 43 digits.
#[allow(clippy::too_many_arguments)]
pub fn prime_plane_relation_weights<E: Field>(
    descriptor: &BinarySourceDescriptor,
    manifest: &PrimePlaneManifest,
    repacking: &RootCoefficientRepacking,
    response_segment: usize,
    plane_segment: usize,
    challenges: &Challenges,
    alpha: E,
    batches: &[E],
) -> Result<NativeRootWeights<E>, AkitaError> {
    if E::from_u128(BINARY_SETUP_P) != E::zero()
        || batches.len()
            != checked::product([manifest.planes().len(), 2]).ok_or(AkitaError::InvalidProof)?
    {
        return Err(AkitaError::InvalidProof);
    }
    let columns = descriptor.index_map().columns();
    let scalar_rows = checked::exact_div(descriptor.index_map().padded_carriers(), columns)
        .ok_or(AkitaError::InvalidProof)?;
    let ring_rows = checked::exact_div(scalar_rows, 4).ok_or(AkitaError::InvalidProof)?;
    for (segment, elements, digits, polynomial) in [
        (
            response_segment,
            checked::product([descriptor.polynomials().len(), ring_rows]),
            4,
            RelationPolynomial::minus_trinomial(648)?,
        ),
        (
            plane_segment,
            checked::product([manifest.planes().len(), columns]),
            PRIME_PLANE_DIGITS,
            RelationPolynomial::plus_trinomial(162)?,
        ),
    ] {
        let shape = repacking
            .segments()
            .get(segment)
            .ok_or(AkitaError::InvalidProof)?;
        if Some(shape.elements) != elements
            || shape.digits != digits
            || shape.geometry.polynomial() != polynomial
            || shape.geometry.coordinate_plane_count() != 1
        {
            return Err(AkitaError::InvalidProof);
        }
    }
    let response = SegmentLanes::new(repacking, response_segment)?;
    let plane_lanes = SegmentLanes::new(repacking, plane_segment)?;
    let mut supports = Vec::new();
    let scalar_challenges = binary_scalar_challenges(descriptor, challenges)?;
    supports
        .try_reserve_exact(scalar_challenges.len())
        .map_err(|_| AkitaError::InvalidProof)?;
    for scalar in &scalar_challenges {
        supports.push(
            scalar
                .iter()
                .enumerate()
                .filter(|(_, &value)| value != 0)
                .map(|(s, &value)| (s, E::from_i128(value)))
                .collect(),
        );
    }
    let mut planes = Vec::new();
    planes
        .try_reserve_exact(manifest.planes().len())
        .map_err(|_| AkitaError::InvalidProof)?;
    let mut target = E::zero();
    for (plane_index, claim) in manifest.planes().iter().enumerate() {
        let (rows, column_weights, mut omega) = claim.weights::<E>(descriptor)?;
        // Only the 162 natural scalar coefficients are read.
        omega.truncate(162);
        let batch_index = checked::product([plane_index, 2]).ok_or(AkitaError::InvalidProof)?;
        let relation = *batches.get(batch_index).ok_or(AkitaError::InvalidProof)?;
        let reading = *batches
            .get(checked::sum([batch_index, 1]).ok_or(AkitaError::InvalidProof)?)
            .ok_or(AkitaError::InvalidProof)?;
        if rows.len() != scalar_rows || column_weights.len() != columns || omega.len() != 162 {
            return Err(AkitaError::InvalidProof);
        }
        target += reading * E::from_u128(claim.value);
        planes.push(PlaneInputs {
            polynomial: claim.polynomial,
            rows,
            column_weights,
            omega,
            relation,
            reading,
        });
    }
    let family = PrimePlaneWeights {
        response,
        plane_lanes,
        table_len: repacking.domain_len(),
        alpha,
        ring_rows,
        columns,
        supports,
        planes,
    };
    NativeRootWeights::structured(
        repacking.domain_len(),
        Vec::new(),
        vec![RootWeightFamily::Prime(family)],
        target,
    )
}

/// Public inputs of one prime plane's two batched identities.
#[derive(Clone, Debug)]
struct PlaneInputs<E: Field> {
    polynomial: usize,
    rows: Vec<E>,
    column_weights: Vec<E>,
    omega: Vec<E>,
    relation: E,
    reading: E,
}

/// Every prime plane's ring identity and scalar reading, kept factored.
///
/// The relation batch weighs response row `r` at packed coefficient
/// `4c + r mod 4` by `(-1)^c alpha^c`. Response lanes start equality blocks,
/// so the verifier sums `L_k = sum_c (-1)^c alpha^c low[4c+k]` once and
/// weighs each row by its element's recomposed high factors. A plane lane of
/// 162 coefficients starts at one of the low-block offsets. At each offset
/// the verifier reads `<omega, low>` and, by the Hankel symmetry of the
/// shifted remainders, `<SRE(c, alpha), low> = <c, SRE(low, alpha)>` through
/// each column's challenge support. The prover streams the same weights.
#[derive(Clone, Debug)]
pub(crate) struct PrimePlaneWeights<E: Field> {
    response: SegmentLanes,
    plane_lanes: SegmentLanes,
    table_len: usize,
    alpha: E,
    ring_rows: usize,
    columns: usize,
    /// Nonzero scalar challenge coefficients for `owner * columns + column`.
    supports: Vec<Vec<(usize, E)>>,
    planes: Vec<PlaneInputs<E>>,
}

impl<E: Field> PrimePlaneWeights<E> {
    pub(crate) fn table_len(&self) -> usize {
        self.table_len
    }

    fn support(&self, polynomial: usize, column: usize) -> Result<&[(usize, E)], AkitaError> {
        checked::mul_add(polynomial, self.columns, column)
            .and_then(|index| self.supports.get(index))
            .map(Vec::as_slice)
            .ok_or(AkitaError::InvalidProof)
    }

    fn row_element(&self, plane: &PlaneInputs<E>, row: usize) -> Result<usize, AkitaError> {
        checked::mul_add(plane.polynomial, self.ring_rows, row / 4).ok_or(AkitaError::InvalidProof)
    }

    pub(crate) fn evaluate(&self, eq: &RootEq<E>) -> Result<E, AkitaError> {
        let low = eq.low();
        let stride = self.plane_lanes.stride();
        if low.len() < 648 || !low.len().is_multiple_of(stride) || stride < 162 {
            return Err(AkitaError::InvalidSetup(
                "prime plane lanes differ from the equality blocks".into(),
            ));
        }
        let mut signed = [E::zero(); 4];
        let mut power = E::one();
        for c in 0..162 {
            let sign = if c % 2 == 0 { power } else { -power };
            for (k, slot) in signed.iter_mut().enumerate() {
                *slot += sign * *low.get(4 * c + k).ok_or(AkitaError::InvalidProof)?;
            }
            power *= self.alpha;
        }
        let slots = low.len() / stride;
        let mut kernels = Vec::new();
        kernels
            .try_reserve_exact(slots)
            .map_err(|_| AkitaError::InvalidProof)?;
        for slot in 0..slots {
            let window = low
                .get(slot * stride..slot * stride + 162)
                .ok_or(AkitaError::InvalidProof)?;
            let mut kernel = [E::zero(); 162];
            shifted_remainder_evaluations::<E, PlusTrinomial>(window, self.alpha, &mut kernel)
                .map_err(|error| AkitaError::Internal(error.to_string()))?;
            kernels.push(kernel);
        }
        let mut sum = E::zero();
        for (plane_index, plane) in self.planes.iter().enumerate() {
            let mut response = E::zero();
            for (quad, rows) in plane.rows.chunks(4).enumerate() {
                let element = self.row_element(plane, 4 * quad)?;
                let lanes = rows
                    .iter()
                    .zip(&signed)
                    .fold(E::zero(), |acc, (&row, &l)| acc + row * l);
                response += self.response.block_digit_weight(eq, element)? * lanes;
            }
            sum += plane.relation * response;
            let omega = (0..slots)
                .map(|slot| {
                    let window = low
                        .get(slot * stride..slot * stride + 162)
                        .ok_or(AkitaError::InvalidProof)?;
                    Ok(plane
                        .omega
                        .iter()
                        .zip(window)
                        .fold(E::zero(), |acc, (&w, &l)| acc + w * l))
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            for (column, &column_weight) in plane.column_weights.iter().enumerate() {
                let support = self.support(plane.polynomial, column)?;
                let element = checked::mul_add(plane_index, self.columns, column)
                    .ok_or(AkitaError::InvalidProof)?;
                let mut lane = vec![None; slots];
                let mut power = E::one();
                for digit in 0..self.plane_lanes.digits() {
                    let base = self.plane_lanes.base(element, digit)?;
                    let slot = eq.low_offset(base) / stride;
                    let value = match lane.get(slot).copied().flatten() {
                        Some(value) => value,
                        None => {
                            let kernel = kernels.get(slot).ok_or(AkitaError::InvalidProof)?;
                            let mut dot = E::zero();
                            for &(s, value) in support {
                                dot += value * *kernel.get(s).ok_or(AkitaError::InvalidProof)?;
                            }
                            let value = plane.reading
                                * column_weight
                                * *omega.get(slot).ok_or(AkitaError::InvalidProof)?
                                - plane.relation * dot;
                            *lane.get_mut(slot).ok_or(AkitaError::InvalidProof)? = Some(value);
                            value
                        }
                    };
                    sum += power * eq.high(base)? * value;
                    power *= E::from_u64(8);
                }
            }
        }
        Ok(sum)
    }

    pub(crate) fn visit(
        &self,
        mut visit: impl FnMut(usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        for (plane_index, plane) in self.planes.iter().enumerate() {
            for (row, &row_weight) in plane.rows.iter().enumerate() {
                if row_weight == E::zero() {
                    continue;
                }
                let element = self.row_element(plane, row)?;
                let mut power = plane.relation * row_weight;
                for coefficient in 0..162 {
                    let signed = if coefficient % 2 == 0 { power } else { -power };
                    let packed = 4 * coefficient + row % 4;
                    let mut scale = signed;
                    for digit in 0..self.response.digits() {
                        visit(self.response.base(element, digit)? + packed, scale)?;
                        scale *= E::from_u64(8);
                    }
                    power *= self.alpha;
                }
            }
            let columns = plane.column_weights.len();
            let weigh = |column: usize| -> Result<[E; 162], AkitaError> {
                let mut multiplier = [E::zero(); 162];
                for &(s, value) in self.support(plane.polynomial, column)? {
                    *multiplier.get_mut(s).ok_or(AkitaError::InvalidProof)? = value;
                }
                let mut kernel = [E::zero(); 162];
                shifted_remainder_evaluations::<E, PlusTrinomial>(
                    &multiplier,
                    self.alpha,
                    &mut kernel,
                )
                .map_err(|error| AkitaError::Internal(error.to_string()))?;
                Ok(kernel)
            };
            let kernels = weigh_units(0..columns, weigh)?;
            for ((column, &column_weight), kernel) in
                plane.column_weights.iter().enumerate().zip(kernels)
            {
                let element = checked::mul_add(plane_index, self.columns, column)
                    .ok_or(AkitaError::InvalidProof)?;
                for (coefficient, (&relation, &omega)) in
                    kernel.iter().zip(&plane.omega).enumerate()
                {
                    let mut scale =
                        plane.reading * column_weight * omega - plane.relation * relation;
                    for digit in 0..self.plane_lanes.digits() {
                        visit(self.plane_lanes.base(element, digit)? + coefficient, scale)?;
                        scale *= E::from_u64(8);
                    }
                }
            }
        }
        Ok(())
    }
}
