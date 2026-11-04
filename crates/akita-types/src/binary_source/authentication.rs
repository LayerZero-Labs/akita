//! Public digit commitments and their direct root authentication weights.

use akita_algebra::ring::{CyclotomicRing, ResidueKernelPoint};
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::{
    SetupStreamLabel, SetupStreamModulus, SetupStreamRing, SetupStreamRole, TypedSetupStream,
    BINARY_SETUP_P,
};
use jolt_field::solinas::parallel::*;
use jolt_field::{cfg_into_iter, Field};

use super::BinaryRootWitnessPlan;
use crate::native_root_relation::{NativeRootWeights, RootEq, RootWeightFamily, SegmentLanes};

/// Ordinary ring degree used by the explicit experimental image and U setup.
pub const BINARY_DIGIT_COMMITMENT_DEGREE: usize = 64;

pub(super) fn validate_label<F: Field>(label: SetupStreamLabel) -> Result<usize, AkitaError> {
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

pub(super) fn setup_entry<F: Field>(
    seed: &[u8; 32],
    label: SetupStreamLabel,
    row: u32,
    column: usize,
) -> Result<CyclotomicRing<F, BINARY_DIGIT_COMMITMENT_DEGREE>, AkitaError> {
    let mut stream = TypedSetupStream::new(
        seed,
        SetupStreamLabel {
            row,
            column: u64::try_from(column).map_err(|_| AkitaError::InvalidProof)?,
            ..label
        },
    )?;
    let mut values = [0u128; BINARY_DIGIT_COMMITMENT_DEGREE];
    stream.fill_p(&mut values)?;
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
pub fn binary_public_digit_commitment<F: Field>(
    seed: &[u8; 32],
    label: SetupStreamLabel,
    digits: &[i8],
) -> Result<Vec<F>, AkitaError> {
    let rows = validate_label::<F>(label)?;
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
    for row in 0..label.rank {
        let mut image = CyclotomicRing::zero();
        for (column, chunk) in digits.chunks(BINARY_DIGIT_COMMITMENT_DEGREE).enumerate() {
            let mut coefficients = [F::zero(); BINARY_DIGIT_COMMITMENT_DEGREE];
            for (output, &digit) in coefficients.iter_mut().zip(chunk) {
                *output = F::from_i64(i64::from(digit));
            }
            setup_entry(seed, label, row, column)?
                .mul_accumulate_into(&CyclotomicRing::from_coefficients(coefficients), &mut image);
        }
        payload.extend_from_slice(image.coefficients());
    }
    Ok(payload)
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
    let rows = validate_label::<E>(label)?;
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

    pub(crate) fn visit(
        &self,
        mut visit: impl FnMut(usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        let kernel = ResidueKernelPoint::new(self.alpha, BINARY_DIGIT_COMMITMENT_DEGREE)?;
        let mut addresses = [0usize; BINARY_DIGIT_COMMITMENT_DEGREE];
        for column in 0..self.columns {
            let filled = self.column_addresses(column, &mut addresses)?;
            let addresses = addresses.get(..filled).ok_or(AkitaError::InvalidProof)?;
            for (row, batch) in self.rows() {
                let matrix = setup_entry::<E>(&self.seed, self.label, row, column)?;
                let mut weights = [E::zero(); BINARY_DIGIT_COMMITMENT_DEGREE];
                kernel.field_kernel_into(matrix.coefficients(), &mut weights)?;
                for (&address, &weight) in addresses.iter().zip(&weights) {
                    visit(address, batch * weight)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "authentication_tests.rs"]
mod tests;
