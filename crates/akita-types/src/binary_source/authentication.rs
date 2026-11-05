//! Public digit commitments and their direct root authentication weights.

use akita_algebra::ring::{CyclotomicRing, ResidueKernelPoint};
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::{
    SetupStreamLabel, SetupStreamModulus, SetupStreamRing, SetupStreamRole, TypedSetupStream,
    BINARY_SETUP_P,
};
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::{cfg_into_iter, Field};

use super::BinaryRootWitnessPlan;
use crate::native_root_relation::{
    weigh_units, NativeRootWeights, ProfileBuilder, RootEq, RootWeightFamily, SegmentLanes,
    WEIGH_BATCH,
};

/// Ordinary ring degree used by the explicit experimental image and U setup.
pub const BINARY_DIGIT_COMMITMENT_DEGREE: usize = 64;

/// Validate a B or D digit commitment setup label for the field `F` and
/// return its row count.
///
/// # Errors
/// Rejects labels other than the rank-2, degree-64 negacyclic P setup of the
/// image or witness commitment, and fields other than `2^128 - 275`.
pub fn validate_digit_commitment_label<F: Field>(
    label: SetupStreamLabel,
) -> Result<usize, AkitaError> {
    if label.level != 0
        || label.row != 0
        || label.column != 0
        || label.rank != 2
        || label.degree != BINARY_DIGIT_COMMITMENT_DEGREE as u32
        || label.ring != SetupStreamRing::Negacyclic
        || label.modulus != SetupStreamModulus::Prime128Offset275
        || !matches!(
            label.role,
            SetupStreamRole::ImageCommitment | SetupStreamRole::WitnessCommitment
        )
        || F::from_u128(BINARY_SETUP_P) != F::zero()
        || F::from_u128(BINARY_SETUP_P - 1) != -F::one()
    {
        return Err(AkitaError::InvalidSetup(
            "invalid binary digit commitment setup".into(),
        ));
    }
    usize::try_from(label.rank).map_err(|_| AkitaError::InvalidProof)
}

fn setup_stream(
    seed: &[u8; 32],
    label: SetupStreamLabel,
    row: u32,
    column: usize,
) -> Result<TypedSetupStream, AkitaError> {
    TypedSetupStream::new(
        seed,
        SetupStreamLabel {
            row,
            column: u64::try_from(column).map_err(|_| AkitaError::InvalidProof)?,
            ..label
        },
    )
}

pub(super) fn setup_entry<F: Field>(
    seed: &[u8; 32],
    label: SetupStreamLabel,
    row: u32,
    column: usize,
) -> Result<CyclotomicRing<F, BINARY_DIGIT_COMMITMENT_DEGREE>, AkitaError> {
    let mut values = [0u128; BINARY_DIGIT_COMMITMENT_DEGREE];
    setup_stream(seed, label, row, column)?.fill_p(&mut values)?;
    let coefficients = values.map(F::from_u128);
    Ok(CyclotomicRing::from_coefficients(coefficients))
}

/// Commit compact balanced digits using the typed P setup in an ordinary ring.
///
/// Digits follow segment/element/digit/natural-coefficient order, omitting every
/// virtual tail in the root coefficient map. Consecutive chunks of 64 form the
/// setup columns; the final chunk is zero-filled. The returned payload is row
/// major. Image digits use `ImageCommitment`; U uses `WitnessCommitment`.
/// This experimental geometry carries no SIS security claim.
///
/// `digits` starts at setup column `first_column`. The commitment is linear,
/// so a table's payload is the field sum of the payloads of disjoint
/// column-aligned pieces; a prover may compute those pieces in parallel.
pub fn binary_public_digit_commitment<F: Field>(
    seed: &[u8; 32],
    label: SetupStreamLabel,
    first_column: usize,
    digits: &[i8],
) -> Result<Vec<F>, AkitaError> {
    let rows = validate_digit_commitment_label::<F>(label)?;
    if digits.is_empty() || digits.iter().any(|digit| !(-4..=3).contains(digit)) {
        return Err(AkitaError::InvalidInput(
            "binary commitment needs nonempty balanced digits".into(),
        ));
    }
    let count =
        checked::product([rows, BINARY_DIGIT_COMMITMENT_DEGREE]).ok_or(AkitaError::InvalidProof)?;
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidProof)?;
    // 2^128 as a field element; accumulators hold `hi * 2^128 + lo`.
    let two_128 = F::from_u128(u128::MAX) + F::one();
    for row in 0..label.rank {
        let mut image = [(0u128, 0u64); BINARY_DIGIT_COMMITMENT_DEGREE];
        for (column, chunk) in digits.chunks(BINARY_DIGIT_COMMITMENT_DEGREE).enumerate() {
            let column = checked::sum([first_column, column]).ok_or(AkitaError::InvalidProof)?;
            accumulate_digit_product(
                &mut setup_stream(seed, label, row, column)?,
                chunk,
                &mut image,
            )?;
        }
        payload.extend(
            image
                .iter()
                .map(|&(lo, hi)| F::from_u128(lo) + F::from_u64(hi) * two_128),
        );
    }
    Ok(payload)
}

/// Add the negacyclic product of one setup entry and one digit chunk.
///
/// Each setup coefficient `a < P` stays an integer. A balanced digit `d`
/// contributes `d * a`, or `|d| * (P - a)` when the product's sign is
/// negative, so every term is a nonnegative integer below `2^131`. The
/// accumulator never overflows: it would need `2^62` terms. The result
/// equals the ring product of [`setup_entry`] and the chunk modulo `P`.
fn accumulate_digit_product(
    stream: &mut TypedSetupStream,
    chunk: &[i8],
    image: &mut [(u128, u64); BINARY_DIGIT_COMMITMENT_DEGREE],
) -> Result<(), AkitaError> {
    let mut entry = [0u128; BINARY_DIGIT_COMMITMENT_DEGREE];
    stream.fill_p(&mut entry)?;
    for (t, &a) in entry.iter().enumerate() {
        let negative = if a == 0 { 0 } else { BINARY_SETUP_P - a };
        // multiples[4 + e] is the canonical integer of `e * a`, for |e| <= 4.
        let multiples: [(u128, u64); 9] = core::array::from_fn(|index| {
            let (value, factor) = match index {
                0..=3 => (negative, 4 - index),
                _ => (a, index - 4),
            };
            let (lo, hi) = (value & u128::from(u64::MAX), value >> 64);
            let factor = factor as u128;
            let (lo, hi) = (lo * factor, hi * factor + ((lo * factor) >> 64));
            ((lo & u128::from(u64::MAX)) | (hi << 64), (hi >> 64) as u64)
        });
        for (s, &digit) in chunk.iter().enumerate() {
            let k = t + s;
            let (k, digit) = if k < BINARY_DIGIT_COMMITMENT_DEGREE {
                (k, digit)
            } else {
                (k - BINARY_DIGIT_COMMITMENT_DEGREE, -digit)
            };
            let (lo, hi) = multiples
                [usize::try_from(4 + i32::from(digit)).map_err(|_| AkitaError::InvalidProof)?];
            let slot = image.get_mut(k).ok_or(AkitaError::InvalidProof)?;
            let (sum, carry) = slot.0.overflowing_add(lo);
            *slot = (sum, slot.1 + hi + u64::from(carry));
        }
    }
    Ok(())
}

/// Authenticate the image digits or U view against the outgoing witness.
///
/// The ordinary negacyclic product is reduced before evaluation at `alpha`.
/// Each compact digit is read directly from its authenticated repacking address;
/// the image digits are not recomposed into T before applying B. No table or
/// quotient supplied in the clear enters these weights.
#[allow(clippy::too_many_arguments)]
pub fn binary_digit_authentication_weights<E: Field>(
    plan: &BinaryRootWitnessPlan,
    segments: &[usize],
    seed: &[u8; 32],
    label: SetupStreamLabel,
    payload: &[E],
    alpha: E,
    row_batches: &[E],
) -> Result<NativeRootWeights<E>, AkitaError> {
    let rows = validate_digit_commitment_label::<E>(label)?;
    let expected_segments = match label.role {
        SetupStreamRole::ImageCommitment => vec![1],
        SetupStreamRole::WitnessCommitment => plan.opening_segments(),
        _ => return Err(AkitaError::InvalidProof),
    };
    if segments != expected_segments {
        return Err(AkitaError::InvalidSetup(
            "binary digit authentication role differs from its segments".into(),
        ));
    }
    let expected =
        checked::product([rows, BINARY_DIGIT_COMMITMENT_DEGREE]).ok_or(AkitaError::InvalidProof)?;
    if payload.len() != expected || row_batches.len() != rows {
        return Err(AkitaError::InvalidProof);
    }
    let mut lanes = Vec::new();
    lanes
        .try_reserve_exact(segments.len())
        .map_err(|_| AkitaError::InvalidProof)?;
    for &segment in segments {
        lanes.push(SegmentLanes::new(plan.repacking(), segment)?);
    }
    let count = lanes
        .iter()
        .try_fold(0usize, |sum, lanes| {
            checked::product([lanes.elements(), lanes.digits(), lanes.degree()])
                .and_then(|count| checked::sum([sum, count]))
        })
        .ok_or(AkitaError::InvalidProof)?;
    let columns =
        checked::div_ceil(count, BINARY_DIGIT_COMMITMENT_DEGREE).ok_or(AkitaError::InvalidProof)?;
    let mut claim = E::zero();
    for (target, &batch) in payload
        .chunks_exact(BINARY_DIGIT_COMMITMENT_DEGREE)
        .zip(row_batches)
    {
        claim += batch
            * target
                .iter()
                .rev()
                .fold(E::zero(), |sum, &coefficient| sum * alpha + coefficient);
    }
    let family = DigitAuthenticationWeights {
        lanes,
        columns,
        table_len: plan.repacking().domain_len(),
        seed: *seed,
        label,
        alpha,
        row_batches: row_batches.to_vec(),
    };
    NativeRootWeights::structured(
        plan.repacking().domain_len(),
        Vec::new(),
        vec![RootWeightFamily::Authentication(family)],
        claim,
    )
}

/// One B or D commitment row identity on the compact digits it authenticates.
///
/// Compact digits run through the natural coefficients of every lane of the
/// listed segments; each run of 64 is one setup column. The verifier gathers
/// a column's equality values `e`, evaluates the residue kernel `K(e)` once,
/// and reads `sum_s M_rs K(e)_s` for every row: the kernel is a Hankel form,
/// so this equals `sum_c K(M_r)_c e_c`. The prover streams `K(M_r)` onto the
/// same addresses. Both sides derive every setup entry from the seed; no
/// address list is stored.
#[derive(Clone, Debug)]
pub(crate) struct DigitAuthenticationWeights<E: Field> {
    lanes: Vec<SegmentLanes>,
    columns: usize,
    table_len: usize,
    seed: [u8; 32],
    label: SetupStreamLabel,
    alpha: E,
    row_batches: Vec<E>,
}

impl<E: Field> DigitAuthenticationWeights<E> {
    pub(crate) fn table_len(&self) -> usize {
        self.table_len
    }

    /// Write the addresses of one setup column's compact digits, in order,
    /// and return how many it holds. Only the final column may hold fewer
    /// than 64. Compact digits run through the natural coefficients of every
    /// lane of the listed segments, so column `c` starts at digit `64 c`.
    fn column_addresses(
        &self,
        column: usize,
        addresses: &mut [usize; BINARY_DIGIT_COMMITMENT_DEGREE],
    ) -> Result<usize, AkitaError> {
        if column >= self.columns {
            return Err(AkitaError::InvalidProof);
        }
        let mut skip = checked::product([column, BINARY_DIGIT_COMMITMENT_DEGREE])
            .ok_or(AkitaError::InvalidProof)?;
        let mut filled = 0;
        for lanes in &self.lanes {
            let len = checked::product([lanes.elements(), lanes.digits(), lanes.degree()])
                .ok_or(AkitaError::InvalidProof)?;
            if skip >= len {
                skip -= len;
                continue;
            }
            let lane_count = checked::product([lanes.elements(), lanes.digits()])
                .ok_or(AkitaError::InvalidProof)?;
            let mut lane = skip / lanes.degree();
            let mut coefficient = skip % lanes.degree();
            skip = 0;
            while lane < lane_count && filled < BINARY_DIGIT_COMMITMENT_DEGREE {
                let base = lanes.base(lane / lanes.digits(), lane % lanes.digits())?;
                while coefficient < lanes.degree() && filled < BINARY_DIGIT_COMMITMENT_DEGREE {
                    *addresses.get_mut(filled).ok_or(AkitaError::InvalidProof)? =
                        checked::sum([base, coefficient]).ok_or(AkitaError::InvalidProof)?;
                    filled += 1;
                    coefficient += 1;
                }
                if coefficient == lanes.degree() {
                    coefficient = 0;
                    lane += 1;
                }
            }
            if filled == BINARY_DIGIT_COMMITMENT_DEGREE {
                break;
            }
        }
        if filled == 0 {
            return Err(AkitaError::InvalidProof);
        }
        Ok(filled)
    }

    fn rows(&self) -> impl Iterator<Item = (u32, E)> + '_ {
        (0u32..).zip(self.row_batches.iter().copied())
    }

    /// Sum every setup column's contribution; columns run in parallel under
    /// the `parallel` feature. Field addition is exact, so the order of the
    /// partial sums does not change the result.
    pub(crate) fn evaluate(&self, eq: &RootEq<E>) -> Result<E, AkitaError> {
        let kernel = ResidueKernelPoint::new(self.alpha, BINARY_DIGIT_COMMITMENT_DEGREE)?;
        let low = eq.low();
        let column_sum = |column: usize| -> Result<E, AkitaError> {
            let mut addresses = [0usize; BINARY_DIGIT_COMMITMENT_DEGREE];
            let filled = self.column_addresses(column, &mut addresses)?;
            let mut equality = [E::zero(); BINARY_DIGIT_COMMITMENT_DEGREE];
            for (slot, &address) in equality
                .iter_mut()
                .zip(addresses.get(..filled).ok_or(AkitaError::InvalidProof)?)
            {
                *slot = eq.high(address)?
                    * *low
                        .get(eq.low_offset(address))
                        .ok_or(AkitaError::InvalidProof)?;
            }
            let mut reduced = [E::zero(); BINARY_DIGIT_COMMITMENT_DEGREE];
            kernel.field_kernel_into(&equality, &mut reduced)?;
            let mut sum = E::zero();
            for (row, batch) in self.rows() {
                let matrix = setup_entry::<E>(&self.seed, self.label, row, column)?;
                sum += batch
                    * matrix
                        .coefficients()
                        .iter()
                        .zip(&reduced)
                        .fold(E::zero(), |acc, (&m, &k)| acc + m * k);
            }
            Ok(sum)
        };
        let sums = cfg_into_iter!(0..self.columns)
            .map(column_sum)
            .collect::<Result<Vec<_>, AkitaError>>()?;
        Ok(sums.into_iter().fold(E::zero(), |sum, column| sum + column))
    }

    /// `sum_r batch_r * K(M_rc) = K(sum_r batch_r * M_rc)`, the weights of
    /// setup column `c` in order. The kernel is linear, so it runs once.
    fn column_weights(
        &self,
        kernel: &ResidueKernelPoint<E>,
        column: usize,
    ) -> Result<[E; BINARY_DIGIT_COMMITMENT_DEGREE], AkitaError> {
        let mut combined = [E::zero(); BINARY_DIGIT_COMMITMENT_DEGREE];
        for (row, batch) in self.rows() {
            let matrix = setup_entry::<E>(&self.seed, self.label, row, column)?;
            for (sum, &value) in combined.iter_mut().zip(matrix.coefficients()) {
                *sum += batch * value;
            }
        }
        let mut weights = [E::zero(); BINARY_DIGIT_COMMITMENT_DEGREE];
        kernel.field_kernel_into(&combined, &mut weights)?;
        Ok(weights)
    }

    /// Add every column's weights to block profiles. When every lane is a
    /// whole number of blocks, runs of lanes weigh their columns and fill
    /// their rows in parallel; otherwise the addresses stream one by one.
    pub(crate) fn emit(&self, profiles: &mut ProfileBuilder<E>) -> Result<(), AkitaError> {
        let width = profiles.width();
        if self
            .lanes
            .iter()
            .any(|lanes| !lanes.stride().is_multiple_of(width))
        {
            return self.visit(|address, value| profiles.add(address, value));
        }
        let kernel = ResidueKernelPoint::new(self.alpha, BINARY_DIGIT_COMMITMENT_DEGREE)?;
        // The compact digit index of the current segment's first coefficient.
        let mut first = 0usize;
        for lanes in &self.lanes {
            let degree = lanes.degree();
            let mut bases = Vec::new();
            bases
                .try_reserve_exact(
                    checked::product([lanes.elements(), lanes.digits()])
                        .ok_or(AkitaError::InvalidProof)?,
                )
                .map_err(|_| AkitaError::InvalidProof)?;
            for element in 0..lanes.elements() {
                for digit in 0..lanes.digits() {
                    bases.push(lanes.base(element, digit)?);
                }
            }
            profiles.add_lanes(&bases, degree, |run, rows| {
                let mut digit =
                    checked::mul_add(run.start, degree, first).ok_or(AkitaError::InvalidProof)?;
                let mut column = None;
                let mut weights = [E::zero(); BINARY_DIGIT_COMMITMENT_DEGREE];
                let lane_rows = rows.len() / run.len().max(1);
                for lane in rows.chunks_exact_mut(lane_rows) {
                    for slot in lane.iter_mut().flat_map(|row| row.iter_mut()).take(degree) {
                        let current = digit / BINARY_DIGIT_COMMITMENT_DEGREE;
                        if column != Some(current) {
                            weights = self.column_weights(&kernel, current)?;
                            column = Some(current);
                        }
                        *slot += weights[digit % BINARY_DIGIT_COMMITMENT_DEGREE];
                        digit += 1;
                    }
                }
                Ok(())
            })?;
            first = checked::mul_add(bases.len(), degree, first).ok_or(AkitaError::InvalidProof)?;
        }
        if first.div_ceil(BINARY_DIGIT_COMMITMENT_DEGREE) != self.columns {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }

    /// Stream every column's weights onto its digit addresses. Columns are
    /// weighed in batches through [`weigh_units`].
    pub(crate) fn visit(
        &self,
        mut visit: impl FnMut(usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        let kernel = ResidueKernelPoint::new(self.alpha, BINARY_DIGIT_COMMITMENT_DEGREE)?;
        let mut addresses = [0usize; BINARY_DIGIT_COMMITMENT_DEGREE];
        let mut batch = Vec::new();
        for column in 0..self.columns {
            let offset = column % WEIGH_BATCH;
            if offset == 0 {
                let end = checked::sum([column, WEIGH_BATCH])
                    .ok_or(AkitaError::InvalidProof)?
                    .min(self.columns);
                batch = weigh_units(column..end, |column| self.column_weights(&kernel, column))?;
            }
            let weights = batch.get(offset).ok_or(AkitaError::InvalidProof)?;
            let filled = self.column_addresses(column, &mut addresses)?;
            let addresses = addresses.get(..filled).ok_or(AkitaError::InvalidProof)?;
            for (&address, &weight) in addresses.iter().zip(weights) {
                visit(address, weight)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "authentication_tests.rs"]
mod tests;
