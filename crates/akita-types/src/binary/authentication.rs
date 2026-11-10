//! Public digit commitments and their direct root authentication weights.

use akita_algebra::ring::{CyclotomicRing, ResidueKernelPoint};
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::{
    BinaryFieldProfile, SetupStreamLabel, SetupStreamModulus, SetupStreamRing, SetupStreamRole,
    TypedSetupStream,
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
/// P128 retains rank two. The p64 setup accepts ranks covered by the SIS table;
/// width admission is performed by the manifest separately for B and D.
pub fn validate_digit_commitment_label<F: Field>(
    label: SetupStreamLabel,
) -> Result<usize, AkitaError> {
    let (profile, ranks) = match label.modulus {
        SetupStreamModulus::Prime128Offset275 => (BinaryFieldProfile::P128, 2..=2),
        SetupStreamModulus::Prime64Offset59 => (
            BinaryFieldProfile::P64,
            1..=akita_params::sis::SIS_MAX_MODULE_RANK,
        ),
        _ => {
            return Err(AkitaError::InvalidSetup(
                "invalid binary digit modulus".into(),
            ))
        }
    };
    if label.level != 0
        || label.row != 0
        || label.column != 0
        || !ranks.contains(&label.rank)
        || label.degree != BINARY_DIGIT_COMMITMENT_DEGREE as u32
        || label.ring != SetupStreamRing::Negacyclic
        || !matches!(
            label.role,
            SetupStreamRole::ImageCommitment | SetupStreamRole::WitnessCommitment
        )
        || F::from_u128(profile.modulus()) != F::zero()
        || F::from_u128(profile.modulus() - 1) != -F::one()
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

/// The P setup entries of both rows of setup column `column` of a validated
/// digit commitment label. The two rows' SHAKE streams squeeze through paired
/// permutations; the entries are those of separate streams.
///
/// # Errors
/// Rejects labels other than rank 2 and columns beyond `u64`.
pub fn digit_commitment_setup_column(
    seed: &[u8; 32],
    label: SetupStreamLabel,
    column: usize,
) -> Result<[[u128; BINARY_DIGIT_COMMITMENT_DEGREE]; 2], AkitaError> {
    if label.rank != 2 {
        return Err(AkitaError::InvalidSetup(
            "binary digit commitment setup has two rows".into(),
        ));
    }
    let mut entries = [[0u128; BINARY_DIGIT_COMMITMENT_DEGREE]; 2];
    let [first, second] = &mut entries;
    TypedSetupStream::fill_p_pair(
        &mut setup_stream(seed, label, 0, column)?,
        first,
        &mut setup_stream(seed, label, 1, column)?,
        second,
    )?;
    Ok(entries)
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
                match label.modulus {
                    SetupStreamModulus::Prime64Offset59 => BinaryFieldProfile::P64.modulus(),
                    _ => BinaryFieldProfile::P128.modulus(),
                },
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
    modulus: u128,
    chunk: &[i8],
    image: &mut [(u128, u64); BINARY_DIGIT_COMMITMENT_DEGREE],
) -> Result<(), AkitaError> {
    let mut entry = [0u128; BINARY_DIGIT_COMMITMENT_DEGREE];
    stream.fill_p(&mut entry)?;
    for (t, &a) in entry.iter().enumerate() {
        let negative = if a == 0 { 0 } else { modulus - a };
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
    if E::from_u128(plan.field_profile().modulus()) != E::zero()
        || E::from_u128(plan.field_profile().modulus() - 1) != -E::one()
    {
        return Err(AkitaError::InvalidSetup(
            "binary authentication field profile mismatch".into(),
        ));
    }
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
            if self.label.rank == 2 {
                let entries = digit_commitment_setup_column(&self.seed, self.label, column)?;
                for (&batch, entry) in self.row_batches.iter().zip(&entries) {
                    sum += batch
                        * entry
                            .iter()
                            .zip(&reduced)
                            .fold(E::zero(), |acc, (&m, &k)| acc + E::from_u128(m) * k);
                }
            } else {
                for (row, &batch) in self.row_batches.iter().enumerate() {
                    let entry = setup_entry::<E>(
                        &self.seed,
                        self.label,
                        u32::try_from(row).map_err(|_| AkitaError::InvalidProof)?,
                        column,
                    )?;
                    sum += batch
                        * entry
                            .coefficients()
                            .iter()
                            .zip(&reduced)
                            .fold(E::zero(), |acc, (&m, &k)| acc + m * k);
                }
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
        if self.label.rank == 2 {
            let entries = digit_commitment_setup_column(&self.seed, self.label, column)?;
            for (&batch, entry) in self.row_batches.iter().zip(&entries) {
                for (sum, &value) in combined.iter_mut().zip(entry) {
                    *sum += batch * E::from_u128(value);
                }
            }
        } else {
            for (row, &batch) in self.row_batches.iter().enumerate() {
                let entry = setup_entry::<E>(
                    &self.seed,
                    self.label,
                    u32::try_from(row).map_err(|_| AkitaError::InvalidProof)?,
                    column,
                )?;
                for (sum, &value) in combined.iter_mut().zip(entry.coefficients()) {
                    *sum += batch * value;
                }
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

#[cfg(test)]
mod p64_tests {
    use super::*;
    use jolt_field::{Prime64Offset59 as F, Ring, Zero};

    #[test]
    fn p64_digit_commitments_match_ring_products_for_each_rank_and_role() {
        let seed = [8; 32];
        let digits = (0..150).map(|i| (i % 8) as i8 - 4).collect::<Vec<_>>();
        for role in [
            SetupStreamRole::ImageCommitment,
            SetupStreamRole::WitnessCommitment,
        ] {
            for rank in 1..=5 {
                let label = SetupStreamLabel {
                    role,
                    group: 7,
                    level: 0,
                    rank,
                    row: 0,
                    column: 0,
                    ring: SetupStreamRing::Negacyclic,
                    degree: 64,
                    modulus: SetupStreamModulus::Prime64Offset59,
                };
                let payload =
                    binary_public_digit_commitment::<F>(&seed, label, 5, &digits).unwrap();
                assert_eq!(payload.len(), 64 * rank as usize);
                let mut expected = Vec::new();
                for row in 0..rank {
                    let mut image = CyclotomicRing::<F, 64>::zero();
                    for (column, chunk) in digits.chunks(64).enumerate() {
                        let mut coefficients = [F::zero(); 64];
                        for (output, &digit) in coefficients.iter_mut().zip(chunk) {
                            *output = F::from_i64(i64::from(digit));
                        }
                        setup_entry::<F>(&seed, label, row, 5 + column)
                            .unwrap()
                            .mul_accumulate_into(
                                &CyclotomicRing::from_coefficients(coefficients),
                                &mut image,
                            );
                    }
                    expected.extend_from_slice(image.coefficients());
                }
                assert_eq!(payload, expected);
                assert!(
                    validate_digit_commitment_label::<jolt_field::Prime128Offset275>(label)
                        .is_err()
                );
                for rank in [0, akita_params::sis::SIS_MAX_MODULE_RANK + 1] {
                    assert!(validate_digit_commitment_label::<F>(SetupStreamLabel {
                        rank,
                        ..label
                    })
                    .is_err());
                }
            }
        }
    }
}
