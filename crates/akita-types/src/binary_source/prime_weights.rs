//! Public coefficient weights authenticating committed prime evaluation planes.

use akita_algebra::ring::{shifted_remainder_evaluations, PlusTrinomial};
use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use akita_params::RelationPolynomial;
use jolt_field::Field;
use std::sync::Arc;

use super::{
    binary_scalar_challenges, BinarySourceDescriptor, PrimePlaneFactors, PrimePlaneManifest,
};
use crate::native_root_relation::{
    weigh_units, NativeRootWeights, ProfileBuilder, RootEq, RootWeightFamily, SegmentLanes,
};
use crate::root_repacking::RootCoefficientRepacking;

/// Compile each plane's ring identity and scalar reading into Stage 2.
///
/// Two independent batches per plane are sampled after the outgoing commitment.
/// The first checks `sum e_r V_r - sum c_b EP_b = 0` at `alpha`, with the
/// complete four-digit response and the signed packing undone. The second
/// checks the scalar reading. Extension readings couple both coordinate planes
/// with independent batches for the two scalar equations. Plane elements are
/// ordered by point, polynomial, coordinate, then column.
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
    manifest.validate(descriptor)?;
    if E::from_u128(manifest.field_profile().modulus()) != E::zero()
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
            manifest.field_profile().prime_plane_digits(),
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
                .map(|(s, &value)| (s, value < 0))
                .collect(),
        );
    }
    let mut planes = Vec::new();
    planes
        .try_reserve_exact(manifest.planes().len())
        .map_err(|_| AkitaError::InvalidProof)?;
    let mut point_omega = Vec::new();
    let mut owner_planes = vec![Vec::new(); descriptor.polynomials().len()];
    let mut target = E::zero();
    let factors = manifest.plane_factors::<E>(descriptor)?;
    for (plane_index, (claim, factors)) in manifest.planes().iter().zip(factors).enumerate() {
        let batch_index = checked::product([plane_index, 2]).ok_or(AkitaError::InvalidProof)?;
        let relation = *batches.get(batch_index).ok_or(AkitaError::InvalidProof)?;
        let reading = *batches
            .get(checked::sum([batch_index, 1]).ok_or(AkitaError::InvalidProof)?)
            .ok_or(AkitaError::InvalidProof)?;
        target += reading * E::from_u128(claim.value);
        if factors.rows.len() != scalar_rows
            || factors.columns.len() != columns
            || factors.omega.len() != 162
        {
            return Err(AkitaError::InvalidProof);
        }
        if factors.point_index == point_omega.len() {
            point_omega.push(factors.omega.clone());
        }
        let omega = point_omega
            .get(factors.point_index)
            .ok_or(AkitaError::InvalidProof)?;
        if !Arc::ptr_eq(omega, &factors.omega) {
            return Err(AkitaError::Internal(
                "prime point factors do not share their coefficient tensor".into(),
            ));
        }
        owner_planes
            .get_mut(claim.polynomial)
            .ok_or(AkitaError::InvalidProof)?
            .push(plane_index);
        planes.push(PlaneInputs {
            polynomial: claim.polynomial,
            relation,
            factors,
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
        point_omega,
        owner_planes,
        batches: batches.to_vec(),
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
    relation: E,
    factors: PrimePlaneFactors<E>,
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
    /// Nonzero scalar challenge coefficients for `owner * columns + column`,
    /// as `(position, negative)`: every coefficient is `-1` or `1`.
    supports: Vec<Vec<(usize, bool)>>,
    planes: Vec<PlaneInputs<E>>,
    point_omega: Vec<Arc<[[E; 2]]>>,
    owner_planes: Vec<Vec<usize>>,
    batches: Vec<E>,
}

impl<E: Field> PrimePlaneWeights<E> {
    pub(crate) fn table_len(&self) -> usize {
        self.table_len
    }

    fn support(&self, polynomial: usize, column: usize) -> Result<&[(usize, bool)], AkitaError> {
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
        let mut digit_powers = vec![E::zero(); self.plane_lanes.digits()];
        let mut power = E::one();
        for slot in &mut digit_powers {
            *slot = power;
            power *= E::from_u64(8);
        }
        let mut highs = vec![E::zero(); slots];
        let mut sum = E::zero();
        // Contract each point once, shared across owners and coordinate planes.
        let omega = self
            .point_omega
            .iter()
            .map(|weights| {
                (0..slots)
                    .map(|slot| {
                        let window = low
                            .get(slot * stride..slot * stride + 162)
                            .ok_or(AkitaError::InvalidProof)?;
                        Ok(weights
                            .iter()
                            .zip(window)
                            .fold([E::zero(); 2], |acc, (&w, &l)| {
                                [acc[0] + w[0] * l, acc[1] + w[1] * l]
                            }))
                    })
                    .collect::<Result<Vec<_>, AkitaError>>()
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        for (polynomial, plane_indices) in self.owner_planes.iter().enumerate() {
            if plane_indices.is_empty() {
                continue;
            }
            // Each polynomial has its own challenge support; coefficients are signs.
            let dots = (0..self.columns)
                .map(|column| {
                    let support = self.support(polynomial, column)?;
                    kernels
                        .iter()
                        .map(|kernel| {
                            support.iter().try_fold(E::zero(), |dot, &(s, negative)| {
                                let value = *kernel.get(s).ok_or(AkitaError::InvalidProof)?;
                                Ok(if negative { dot - value } else { dot + value })
                            })
                        })
                        .collect::<Result<Vec<_>, AkitaError>>()
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            for &plane_index in plane_indices {
                let plane = self
                    .planes
                    .get(plane_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let omega = omega
                    .get(plane.factors.point_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let mut response = E::zero();
                for (quad, rows) in plane.factors.rows.chunks(4).enumerate() {
                    let element = checked::mul_add(polynomial, self.ring_rows, quad)
                        .ok_or(AkitaError::InvalidProof)?;
                    let lanes = rows
                        .iter()
                        .zip(&signed)
                        .fold(E::zero(), |acc, (&row, &l)| acc + row * l);
                    response += self.response.block_digit_weight(eq, element)? * lanes;
                }
                sum += plane.relation * response;
                let (mut reading, mut relation) = (E::zero(), E::zero());
                for (column, column_weight) in
                    plane.factors.reading_columns(&self.batches)?.enumerate()
                {
                    let element = checked::mul_add(plane_index, self.columns, column)
                        .ok_or(AkitaError::InvalidProof)?;
                    // The digit lanes of one element cycle through the
                    // low-block slots; recompose their high factors per slot.
                    highs.fill(E::zero());
                    for (digit, &power) in digit_powers.iter().enumerate() {
                        let base = self.plane_lanes.base(element, digit)?;
                        *highs
                            .get_mut(eq.low_offset(base) / stride)
                            .ok_or(AkitaError::InvalidProof)? += power * eq.high(base)?;
                    }
                    let column_dots = dots.get(column).ok_or(AkitaError::InvalidProof)?;
                    let mut omega_sum = [E::zero(); 2];
                    for ((&high, &omega), &dot) in highs.iter().zip(omega).zip(column_dots) {
                        omega_sum[0] += high * omega[0];
                        omega_sum[1] += high * omega[1];
                        relation += high * dot;
                    }
                    reading += column_weight[0] * omega_sum[0] + column_weight[1] * omega_sum[1];
                }
                sum += reading - plane.relation * relation;
            }
        }
        Ok(sum)
    }

    /// Profile the two batched coordinate-reading terms minus the ring relation
    /// for one column, using its scalar challenge support.
    fn plane_profile(&self, plane: &PlaneInputs<E>, column: usize) -> Result<[E; 162], AkitaError> {
        let column_weight = plane.factors.reading_column(&self.batches, column)?;
        let mut multiplier = [E::zero(); 162];
        for &(s, negative) in self.support(plane.polynomial, column)? {
            *multiplier.get_mut(s).ok_or(AkitaError::InvalidProof)? =
                if negative { -E::one() } else { E::one() };
        }
        let mut profile = [E::zero(); 162];
        shifted_remainder_evaluations::<E, PlusTrinomial>(&multiplier, self.alpha, &mut profile)
            .map_err(|error| AkitaError::Internal(error.to_string()))?;
        for (slot, &omega) in profile.iter_mut().zip(plane.factors.omega.iter()) {
            *slot =
                column_weight[0] * omega[0] + column_weight[1] * omega[1] - plane.relation * *slot;
        }
        Ok(profile)
    }

    /// Add both profiles to block profiles, one element per parallel task.
    /// A response element `polynomial * ring_rows + ring_row` gathers rows
    /// `4 * ring_row + k` of every plane on that polynomial.
    pub(crate) fn emit(&self, profiles: &mut ProfileBuilder<E>) -> Result<(), AkitaError> {
        let response = tracing::info_span!(
            "prime_plane_response_profiles",
            planes = self.planes.len(),
            ring_rows = self.ring_rows,
            owners = self.owner_planes.len()
        )
        .entered();
        profiles.add_shared(&self.response, |element, profile| {
            let (polynomial, ring_row) = (element / self.ring_rows, element % self.ring_rows);
            for &plane_index in self
                .owner_planes
                .get(polynomial)
                .ok_or(AkitaError::InvalidProof)?
            {
                let plane = self
                    .planes
                    .get(plane_index)
                    .ok_or(AkitaError::InvalidProof)?;
                for k in 0..4 {
                    let row_weight = *plane
                        .factors
                        .rows
                        .get(4 * ring_row + k)
                        .ok_or(AkitaError::InvalidProof)?;
                    if row_weight == E::zero() {
                        continue;
                    }
                    let mut power = plane.relation * row_weight;
                    for coefficient in 0..162 {
                        let signed = if coefficient % 2 == 0 { power } else { -power };
                        *profile
                            .get_mut(4 * coefficient + k)
                            .ok_or(AkitaError::InvalidProof)? += signed;
                        power *= self.alpha;
                    }
                }
            }
            Ok(())
        })?;
        drop(response);
        let _planes = tracing::info_span!(
            "prime_plane_column_profiles",
            planes = self.planes.len(),
            columns = self.columns
        )
        .entered();
        profiles.add_shared(&self.plane_lanes, |element, profile| {
            let plane = self
                .planes
                .get(element / self.columns)
                .ok_or(AkitaError::InvalidProof)?;
            let values = self.plane_profile(plane, element % self.columns)?;
            for (slot, value) in profile.iter_mut().zip(values) {
                *slot += value;
            }
            Ok(())
        })
    }

    /// Stream the response-row and plane-column profiles; digit `d` of each
    /// weighs `8^d` times its profile entry.
    pub(crate) fn profiles(
        &self,
        mut profile: impl FnMut(&SegmentLanes, usize, usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        for (plane_index, plane) in self.planes.iter().enumerate() {
            for (row, &row_weight) in plane.factors.rows.iter().enumerate() {
                if row_weight == E::zero() {
                    continue;
                }
                let element = self.row_element(plane, row)?;
                let mut power = plane.relation * row_weight;
                for coefficient in 0..162 {
                    let signed = if coefficient % 2 == 0 { power } else { -power };
                    profile(&self.response, element, 4 * coefficient + row % 4, signed)?;
                    power *= self.alpha;
                }
            }
            let columns = weigh_units(0..plane.factors.columns.len(), |column| {
                self.plane_profile(plane, column)
            })?;
            for (column, values) in columns.into_iter().enumerate() {
                let element = checked::mul_add(plane_index, self.columns, column)
                    .ok_or(AkitaError::InvalidProof)?;
                for (coefficient, value) in values.into_iter().enumerate() {
                    profile(&self.plane_lanes, element, coefficient, value)?;
                }
            }
        }
        Ok(())
    }
}
