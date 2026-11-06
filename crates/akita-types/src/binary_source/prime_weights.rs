//! Public coefficient weights authenticating committed prime evaluation planes.

use akita_algebra::ring::{shifted_remainder_evaluations, PlusTrinomial};
use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use akita_params::RelationPolynomial;
use jolt_field::Field;

use super::{binary_scalar_challenges, BinarySourceDescriptor, PrimePlaneManifest};
use crate::native_root_relation::{
    weigh_units, NativeRootWeights, ProfileBuilder, RootEq, RootWeightFamily, SegmentLanes,
};
use crate::root_repacking::RootCoefficientRepacking;

/// Compile each plane's ring identity and scalar reading into Stage 2.
///
/// Two independent batches per plane are sampled after the outgoing commitment.
/// The first checks each coordinate of `sum e_r V_r - sum c_b EP_b = 0`
/// at `alpha`, with the complete four-digit response and signed packing undone.
/// The second checks each coordinate of `sum col_b <omega, EP_b> = value`
/// using E64 multiplication for P64. Plane elements are ordered by polynomial,
/// base coordinate, then column; digits use the selected profile's depth.
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
    let coordinates = manifest
        .claim()
        .ok_or(AkitaError::InvalidProof)?
        .weights::<E>(descriptor)?;
    let degree = manifest.field_profile().extension_degree();
    if coordinates.iter().any(|(rows, columns_weights, omega)| {
        rows.len() != scalar_rows || columns_weights.len() != columns || omega.len() != 256
    }) {
        return Err(AkitaError::InvalidProof);
    }
    let omega = coordinates
        .iter()
        .map(|(_, _, omega)| omega[..162].to_vec())
        .collect();
    let mut target = E::zero();
    for (plane_index, claim) in manifest.planes().iter().enumerate() {
        let batch_index = checked::product([plane_index, 2]).ok_or(AkitaError::InvalidProof)?;
        let relation = *batches.get(batch_index).ok_or(AkitaError::InvalidProof)?;
        let reading_batch = *batches
            .get(checked::sum([batch_index, 1]).ok_or(AkitaError::InvalidProof)?)
            .ok_or(AkitaError::InvalidProof)?;
        target += reading_batch * E::from_u128(claim.value);
        let first_plane = plane_index
            .checked_sub(claim.coordinate)
            .ok_or(AkitaError::InvalidProof)?;
        let mut reading = [E::zero(); 2];
        for (coordinate, batch) in reading.iter_mut().enumerate().take(degree) {
            let index = checked::sum([first_plane, coordinate])
                .and_then(|i| checked::mul_add(2, i, 1))
                .ok_or(AkitaError::InvalidProof)?;
            *batch = *batches.get(index).ok_or(AkitaError::InvalidProof)?;
        }
        planes.push(PlaneInputs {
            polynomial: claim.polynomial,
            relation,
            coordinate: claim.coordinate,
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
        rows: coordinates
            .iter()
            .map(|(rows, _, _)| rows.clone())
            .collect(),
        column_weights: coordinates
            .into_iter()
            .map(|(_, columns, _)| columns)
            .collect(),
        omega,
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
    coordinate: usize,
    reading: [E; 2],
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
    rows: Vec<Vec<E>>,
    column_weights: Vec<Vec<E>>,
    omega: Vec<Vec<E>>,
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
        // At most `1024 / 162` slots. Shared coefficient weights are read once.
        let omega = self
            .omega
            .iter()
            .map(|omega| {
                (0..slots)
                    .map(|slot| {
                        let window = low
                            .get(slot * stride..slot * stride + 162)
                            .ok_or(AkitaError::InvalidProof)?;
                        Ok(omega
                            .iter()
                            .zip(window)
                            .fold(E::zero(), |acc, (&w, &l)| acc + w * l))
                    })
                    .collect::<Result<Vec<_>, AkitaError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut highs = vec![E::zero(); slots];
        let mut sum = E::zero();
        for (plane_index, plane) in self.planes.iter().enumerate() {
            let polynomial = plane.polynomial;
            let mut response = E::zero();
            let rows = self
                .rows
                .get(plane.coordinate)
                .ok_or(AkitaError::InvalidProof)?;
            for (quad, rows) in rows.chunks(4).enumerate() {
                let element = checked::mul_add(polynomial, self.ring_rows, quad)
                    .ok_or(AkitaError::InvalidProof)?;
                let lanes = rows
                    .iter()
                    .zip(&signed)
                    .fold(E::zero(), |acc, (&row, &l)| acc + row * l);
                response += self.response.block_digit_weight(eq, element)? * lanes;
            }
            sum += plane.relation * response;
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
            let (mut reading, mut relation) = (E::zero(), E::zero());
            for column in 0..self.columns {
                let reading_weights = self.reading_column(plane, column)?;
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
                let mut omega_sum = E::zero();
                for (slot, (&high, &dot)) in highs.iter().zip(column_dots).enumerate() {
                    let reading = omega
                        .iter()
                        .zip(&reading_weights)
                        .fold(E::zero(), |sum, (values, &weight)| {
                            sum + values[slot] * weight
                        });
                    omega_sum += high * reading;
                    relation += high * dot;
                }
                reading += omega_sum;
            }
            sum += reading - plane.relation * relation;
        }
        Ok(sum)
    }

    // Multiplying the plane, column and coefficient coordinates reduces each
    // power of u using u^2 = 2. Independent output batches select both readings.
    fn reading_column(&self, plane: &PlaneInputs<E>, column: usize) -> Result<[E; 2], AkitaError> {
        let degree = self.omega.len();
        let mut weights = [E::zero(); 2];
        for (omega_coordinate, weight) in weights.iter_mut().enumerate().take(degree) {
            for (column_coordinate, columns) in self.column_weights.iter().enumerate() {
                let power = plane.coordinate + omega_coordinate + column_coordinate;
                let scale = E::from_u64(1 << (power / degree));
                *weight += plane.reading[power % degree]
                    * scale
                    * *columns.get(column).ok_or(AkitaError::InvalidProof)?;
            }
        }
        Ok(weights)
    }

    /// Profile `reading * w_c * omega - relation * SRE(c, alpha)` of plane
    /// column `c`, from the column's scalar challenge support.
    fn plane_profile(&self, plane: &PlaneInputs<E>, column: usize) -> Result<[E; 162], AkitaError> {
        let reading_weights = self.reading_column(plane, column)?;
        let mut multiplier = [E::zero(); 162];
        for &(s, negative) in self.support(plane.polynomial, column)? {
            *multiplier.get_mut(s).ok_or(AkitaError::InvalidProof)? =
                if negative { -E::one() } else { E::one() };
        }
        let mut profile = [E::zero(); 162];
        shifted_remainder_evaluations::<E, PlusTrinomial>(&multiplier, self.alpha, &mut profile)
            .map_err(|error| AkitaError::Internal(error.to_string()))?;
        for (coefficient, slot) in profile.iter_mut().enumerate() {
            let reading = self
                .omega
                .iter()
                .zip(&reading_weights)
                .fold(E::zero(), |sum, (omega, &weight)| {
                    sum + omega[coefficient] * weight
                });
            *slot = reading - plane.relation * *slot;
        }
        Ok(profile)
    }

    /// Add both profiles to block profiles, one element per parallel task.
    /// A response element `polynomial * ring_rows + ring_row` gathers rows
    /// `4 * ring_row + k` of the plane on that polynomial.
    pub(crate) fn emit(&self, profiles: &mut ProfileBuilder<E>) -> Result<(), AkitaError> {
        profiles.add_shared(&self.response, |element, profile| {
            let (polynomial, ring_row) = (element / self.ring_rows, element % self.ring_rows);
            for plane in self
                .planes
                .iter()
                .filter(|plane| plane.polynomial == polynomial)
            {
                for k in 0..4 {
                    let row_weight = *self
                        .rows
                        .get(plane.coordinate)
                        .and_then(|rows| rows.get(4 * ring_row + k))
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
            let rows = self
                .rows
                .get(plane.coordinate)
                .ok_or(AkitaError::InvalidProof)?;
            for (row, &row_weight) in rows.iter().enumerate() {
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
            let columns = weigh_units(0..self.columns, |column| self.plane_profile(plane, column))?;
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
