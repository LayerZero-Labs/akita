//! Typed setup prefix of a binary root: the A tensor, then the B/D digit setup.
//!
//! Setup preparation derives every entry once and the successor commits the
//! prefix. Stage 3 then reduces the A, B and D root rows to one prefix
//! evaluation, so online verification reads no setup entry. The verifier
//! evaluates the public weight at the Stage 3 point from the root point,
//! alpha and the B/D row batches; that work depends on the segment geometry
//! and the point lengths, not on the number of setup columns.
//!
//! Layout. The compact digits of each role (B: the image segment; D: the
//! opening segments) are cut into setup columns of 64, exactly as the
//! commitment does. A segment of degree `n` repeats its column pattern every
//! `P = 64 / gcd(n, 64)` lanes, one period of `P n / 64` whole columns. Each
//! period stores its own copy of the columns its digits touch, `J` of them,
//! with a column shared by two periods or two segments stored in both. Per
//! row, the copies are split by the binary expansions of the period count and
//! of `J` into pieces of `2^b` periods by `2^e` columns. Cell `(q, j, slot s)`
//! of a piece has local index `base + ((q - q0) * 2^e + (j - first)) * 64 + s`.
//!
//! Regions. A piece lives in one region: an address subcube whose fixed bits
//! are constant and whose free bits carry the local index in increasing
//! order. The zero padding of the A tensor (padded rank slots, and the
//! coefficient tail `[648, 1024)` in aligned runs of at least 64) forms such
//! subcubes. Pieces fill them in decreasing size, and the rest follow A in the
//! tail region, whose local index is the address. Every piece has power-of-two
//! size and a base aligned to it, so its cell equality factors into the
//! region's fixed bits, the slot, column and period bits.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::ResidueKernelPoint;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::{SetupStreamLabel, SetupStreamRole};
use akita_params::SetupPrefixSlotId;
use akita_serialization::AkitaSerialize;
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::{cfg_chunks_mut, Field};

use super::a_product::{DEGREE as A_DEGREE, STRIDE as A_STRIDE};
use super::authentication::{setup_entry, BINARY_DIGIT_COMMITMENT_DEGREE};
use super::{
    binary_a_setup_coefficients, BinaryAPrefixLen, BinaryAProduct, BinaryCommittedGroupManifest,
    BinaryRootWitnessPlan,
};
use crate::native_root_relation::{RootEq, SegmentLanes};

const SLOTS: usize = BINARY_DIGIT_COMMITMENT_DEGREE;
const SLOT_BITS: usize = SLOTS.trailing_zeros() as usize;
/// Width of a prefix address in the identity, on every target: the tail
/// region zeroes no address bit below it.
const ADDRESS_BITS: usize = u64::BITS as usize;
/// Identity word in place of the tail region's free-bit count.
const TAIL_FREE_BITS: u64 = u64::MAX;

/// One authenticated digit segment and its period geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DigitSegment {
    /// 0 for B (image digits), 1 for D (opening digits).
    role: usize,
    lanes: SegmentLanes,
    /// Compact index of the segment's first digit in its role.
    offset: usize,
    /// Lanes per period, `64 / gcd(degree, 64)`.
    period: usize,
    /// Periods, `ceil(lanes / period)`; only the last may be partial.
    periods: usize,
    /// Columns one period copy touches, including a shared leading column.
    columns: usize,
}

impl DigitSegment {
    fn lane_count(&self) -> usize {
        self.lanes.elements() * self.lanes.digits()
    }

    /// Setup column that holds column `j` of period `q`.
    fn setup_column(&self, q: usize, j: usize) -> Option<usize> {
        let per_period = checked::product([self.period, self.lanes.degree()])? / SLOTS;
        checked::sum([self.offset / SLOTS, checked::product([q, per_period])?, j])
    }

    /// Root address of slot `t` of column `j` of period `q`, if a digit is there.
    fn address(&self, q: usize, j: usize, t: usize) -> Option<usize> {
        let local = checked::mul_add(j, SLOTS, t)?.checked_sub(self.offset % SLOTS)?;
        let degree = self.lanes.degree();
        let (p, coefficient) = (local / degree, local % degree);
        if p >= self.period {
            return None;
        }
        let lane = checked::mul_add(q, self.period, p).filter(|&lane| lane < self.lane_count())?;
        checked::mul_add(lane, self.lanes.stride(), self.lanes.start())
            .and_then(|base| checked::sum([base, coefficient]))
    }
}

/// An address subcube. Hole regions lie in the zero padding of the A tensor:
/// the listed bits are fixed, every bit at or above `top` is zero, and the
/// `free` bits carry the local index in increasing order. The tail region
/// (`free: None`, no fixed bit, `top = ADDRESS_BITS`) maps every local index
/// to itself.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Region {
    fixed: Vec<(usize, bool)>,
    free: Option<Vec<usize>>,
    top: usize,
}

impl Region {
    fn capacity_bits(&self) -> Option<usize> {
        self.free.as_ref().map(Vec::len)
    }

    /// Local index of `cell`, if the cell lies in this region.
    fn extract(&self, cell: usize) -> Option<usize> {
        let Some(free) = &self.free else {
            return Some(cell);
        };
        if cell.checked_shr(self.top as u32).unwrap_or(0) != 0
            || self
                .fixed
                .iter()
                .any(|&(bit, value)| ((cell >> bit) & 1 == 1) != value)
        {
            return None;
        }
        Some(
            free.iter()
                .enumerate()
                .fold(0, |local, (i, &bit)| local | (((cell >> bit) & 1) << i)),
        )
    }

    /// Equality factor of the fixed and zero bits at `point`, and the point's
    /// coordinates at the free bits.
    fn view<E: Field>(&self, point: &[E]) -> Result<(E, Vec<E>), AkitaError> {
        let Some(free) = &self.free else {
            return Ok((E::one(), point.to_vec()));
        };
        let mut factor = E::one();
        for &(bit, value) in &self.fixed {
            let x = *point.get(bit).ok_or(AkitaError::InvalidProof)?;
            factor *= if value { x } else { E::one() - x };
        }
        for &x in point.get(self.top..).unwrap_or(&[]) {
            factor *= E::one() - x;
        }
        let coordinates = free
            .iter()
            .map(|&bit| point.get(bit).copied().ok_or(AkitaError::InvalidProof))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((factor, coordinates))
    }
}

/// `2^period_bits` periods from `q0` by `2^width_bits` columns from `first`
/// of one row's copies of a segment, at local index `base` of `region`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DigitPiece {
    segment: usize,
    row: usize,
    q0: usize,
    period_bits: usize,
    first: usize,
    width_bits: usize,
    region: usize,
    base: usize,
}

impl DigitPiece {
    fn size_bits(&self) -> usize {
        self.period_bits + self.width_bits + SLOT_BITS
    }
}

/// Canonical typed prefix of one binary root: A, then the B/D digit pieces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryTypedPrefix {
    a: BinaryAPrefixLen,
    rows: usize,
    segments: Vec<DigitSegment>,
    regions: Vec<Region>,
    /// Sorted by region, then base.
    pieces: Vec<DigitPiece>,
    role_columns: [usize; 2],
    /// Cells through the last piece, and at least the A tensor.
    pub natural_len: usize,
    /// Smallest power-of-two prefix domain that holds the natural cells.
    pub domain_len: usize,
}

impl BinaryTypedPrefix {
    /// Lay out the typed prefix of one root: the manifest, whether the word
    /// view is opened, the number of prime planes and the successor's ring
    /// dimension fix every B/D segment. Ordinary segments follow them in the
    /// root plan and do not enter the layout.
    pub fn new(
        manifest: &BinaryCommittedGroupManifest,
        word: bool,
        planes: usize,
        destination_ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        let plan = BinaryRootWitnessPlan::new(
            manifest.descriptor(),
            manifest.a_rank(),
            destination_ring_dimension,
        )?
        .with_prime_planes(word, planes)?;
        Self::from_plan(manifest, &plan)
    }

    pub(crate) fn from_plan(
        manifest: &BinaryCommittedGroupManifest,
        plan: &BinaryRootWitnessPlan,
    ) -> Result<Self, AkitaError> {
        let a = BinaryAPrefixLen::new(manifest.descriptor().index_map(), manifest.a_rank())?;
        let rows = manifest.b_rows();
        let mut segments = Vec::new();
        let mut role_columns = [0; 2];
        for (role, list) in [vec![1], plan.opening_segments()].into_iter().enumerate() {
            let mut offset = 0usize;
            for segment in list {
                let lanes = SegmentLanes::new(plan.repacking(), segment)?;
                let count = checked::product([lanes.elements(), lanes.digits()])
                    .ok_or(AkitaError::InvalidProof)?;
                let degree = lanes.degree();
                // The terminal weight splits lane addresses as
                // `(start / stride + lane) * stride + coefficient`.
                if count == 0
                    || degree == 0
                    || !lanes.stride().is_power_of_two()
                    || !lanes.start().is_multiple_of(lanes.stride())
                {
                    return Err(AkitaError::InvalidSetup(
                        "typed prefix requires nonempty, stride-aligned power-of-two digit lanes"
                            .into(),
                    ));
                }
                let period = SLOTS >> (degree.trailing_zeros() as usize).min(SLOT_BITS);
                let first_period = checked::product([count.min(period), degree])
                    .and_then(|digits| checked::sum([digits, offset % SLOTS]))
                    .ok_or(AkitaError::InvalidProof)?;
                segments.push(DigitSegment {
                    role,
                    lanes,
                    offset,
                    period,
                    periods: checked::div_ceil(count, period).ok_or(AkitaError::InvalidProof)?,
                    columns: checked::div_ceil(first_period, SLOTS)
                        .ok_or(AkitaError::InvalidProof)?,
                });
                offset = checked::mul_add(count, degree, offset).ok_or(AkitaError::InvalidProof)?;
            }
            role_columns[role] =
                checked::div_ceil(offset, SLOTS).ok_or(AkitaError::InvalidProof)?;
        }
        let mut pieces = Vec::new();
        for (index, segment) in segments.iter().enumerate() {
            for row in 0..rows {
                for (q0, period_bits) in binary_runs(segment.periods) {
                    for (first, width_bits) in binary_runs(segment.columns) {
                        pieces.push(DigitPiece {
                            segment: index,
                            row,
                            q0,
                            period_bits,
                            first,
                            width_bits,
                            region: 0,
                            base: 0,
                        });
                    }
                }
            }
        }
        // Stable: equal sizes keep role, segment, row, period and column order.
        pieces.sort_by_key(|piece| std::cmp::Reverse(piece.size_bits()));
        let mut regions = a_padding_regions(&a, manifest.a_rank())?;
        let tail = regions.len();
        regions.push(Region {
            fixed: Vec::new(),
            free: None,
            top: ADDRESS_BITS,
        });
        let mut cursors = vec![0usize; tail];
        let mut tail_cursor = None;
        for piece in &mut pieces {
            let size = checked::pow2(piece.size_bits()).ok_or(AkitaError::InvalidProof)?;
            // Decreasing power-of-two sizes keep every cursor aligned.
            let hole = regions.iter().zip(&cursors).position(|(region, &cursor)| {
                region
                    .capacity_bits()
                    .and_then(checked::pow2)
                    .and_then(|capacity| capacity.checked_sub(cursor))
                    .is_some_and(|room| room >= size)
            });
            if let Some(region) = hole {
                let cursor = cursors.get_mut(region).ok_or(AkitaError::InvalidProof)?;
                piece.region = region;
                piece.base = *cursor;
                *cursor += size;
            } else {
                let cursor = match tail_cursor {
                    Some(cursor) => cursor,
                    None => {
                        checked::align_up(a.natural_len, size).ok_or(AkitaError::InvalidProof)?
                    }
                };
                piece.region = tail;
                piece.base = cursor;
                tail_cursor = Some(checked::sum([cursor, size]).ok_or(AkitaError::InvalidProof)?);
            }
        }
        pieces.sort_by_key(|piece| (piece.region, piece.base));
        let natural_len = tail_cursor.unwrap_or(a.natural_len).max(a.natural_len);
        let domain_len = checked::ceil_log2(natural_len)
            .and_then(checked::pow2)
            .ok_or(AkitaError::InvalidProof)?;
        Ok(Self {
            a,
            rows,
            segments,
            regions,
            pieces,
            role_columns,
            natural_len,
            domain_len,
        })
    }

    /// Accept a selected prefix domain that holds every typed cell.
    pub fn validate_prefix(&self, prefix_len: usize) -> Result<(), AkitaError> {
        if !prefix_len.is_power_of_two() || prefix_len < self.domain_len {
            return Err(AkitaError::InvalidInput(
                "typed prefix exceeds the selected prefix domain".into(),
            ));
        }
        Ok(())
    }

    /// Bind the manifest, this layout and the scheduled slot.
    /// The prepared prefix commitment is bound again by ordinary grouped recursion.
    pub fn identity(
        &self,
        manifest: &BinaryCommittedGroupManifest,
        slot: &SetupPrefixSlotId,
    ) -> Result<Vec<u8>, AkitaError> {
        manifest.validate()?;
        let mut bytes = b"akita/binary-source/typed-prefix/a-then-digit-pieces/v3".to_vec();
        bytes.extend(manifest.canonical_bytes());
        // Every word is a little-endian u64, independent of the target's
        // pointer width.
        let word = |value: usize| u64::try_from(value).map_err(|_| AkitaError::InvalidProof);
        let mut push = |value: u64| bytes.extend_from_slice(&value.to_le_bytes());
        push(word(self.a.rows)?);
        push(word(self.a.row_stride)?);
        push(word(self.rows)?);
        push(word(self.segments.len())?);
        for segment in &self.segments {
            for value in [
                segment.role,
                segment.lanes.start(),
                segment.lanes.stride(),
                segment.lanes.degree(),
                segment.lanes.digits(),
                segment.lanes.elements(),
                segment.offset,
            ] {
                push(word(value)?);
            }
        }
        push(word(self.regions.len())?);
        for region in &self.regions {
            push(word(region.top)?);
            push(word(region.fixed.len())?);
            for &(bit, value) in &region.fixed {
                push(word(bit)?);
                push(u64::from(value));
            }
            match &region.free {
                Some(free) => {
                    push(word(free.len())?);
                    for &bit in free {
                        push(word(bit)?);
                    }
                }
                None => push(TAIL_FREE_BITS),
            }
        }
        push(word(self.pieces.len())?);
        for piece in &self.pieces {
            for value in [
                piece.segment,
                piece.row,
                piece.q0,
                piece.period_bits,
                piece.first,
                piece.width_bits,
                piece.region,
                piece.base,
            ] {
                push(word(value)?);
            }
        }
        push(word(self.natural_len)?);
        slot.serialize_compressed(&mut bytes)
            .map_err(|_| AkitaError::InvalidProof)?;
        Ok(bytes)
    }

    /// Canonical prefix coefficients: A, then every stored B/D column copy.
    /// All padding is zero. This is for setup preparation and proving, never
    /// online verification.
    pub fn setup_coefficients<F: Field>(
        &self,
        seed: &[u8; 32],
        manifest: &BinaryCommittedGroupManifest,
        prefix_len: usize,
    ) -> Result<Vec<F>, AkitaError> {
        self.validate_prefix(prefix_len)?;
        let mut coefficients = binary_a_setup_coefficients(seed, manifest, prefix_len)?;
        let labels = [
            manifest.b_label(),
            SetupStreamLabel {
                role: SetupStreamRole::WitnessCommitment,
                ..manifest.b_label()
            },
        ];
        let cells = coefficients
            .get_mut(..self.natural_len)
            .ok_or(AkitaError::InvalidProof)?;
        cfg_chunks_mut!(cells, SLOTS).enumerate().try_for_each(
            |(chunk, cells)| -> Result<(), AkitaError> {
                let cell = checked::product([chunk, SLOTS]).ok_or(AkitaError::InvalidProof)?;
                let Some((segment, row, q, j)) = self.locate(cell)? else {
                    return Ok(());
                };
                let column = segment.setup_column(q, j).ok_or(AkitaError::InvalidProof)?;
                let role_columns = *self
                    .role_columns
                    .get(segment.role)
                    .ok_or(AkitaError::InvalidProof)?;
                if j >= segment.columns || column >= role_columns {
                    return Ok(());
                }
                let label = *labels.get(segment.role).ok_or(AkitaError::InvalidProof)?;
                let row = u32::try_from(row).map_err(|_| AkitaError::InvalidProof)?;
                let entry = setup_entry::<F>(seed, label, row, column)?;
                cells.copy_from_slice(entry.coefficients());
                Ok(())
            },
        )?;
        Ok(coefficients)
    }

    /// Dense Stage 3 weights for the prover: the A product, then every B/D
    /// column copy `batch_row * K(e)`, where `e` holds the root-point equality
    /// of the copy's digits and `K` is the residue kernel at alpha.
    pub fn prover_weights<E: Field>(
        &self,
        a: &BinaryAProduct<E>,
        root_point: &[E],
        alpha: E,
        batches: [&[E]; 2],
        prefix_len: usize,
    ) -> Result<Vec<E>, AkitaError> {
        self.validate_prefix(prefix_len)?;
        self.validate_batches(batches)?;
        let (index, coefficient) = a.factors()?;
        let mut weights = Vec::new();
        weights.try_reserve_exact(prefix_len).map_err(|_| {
            AkitaError::InvalidInput("typed prefix weight allocation failed".into())
        })?;
        weights.resize(prefix_len, E::zero());
        let a_cells = weights
            .get_mut(..self.a.natural_len)
            .ok_or(AkitaError::InvalidProof)?;
        cfg_chunks_mut!(a_cells, self.a.row_stride)
            .zip(index)
            .for_each(|(cells, row)| {
                for (cell, &value) in cells.iter_mut().zip(&coefficient) {
                    *cell = row * value;
                }
            });
        let eq = RootEq::new(root_point)?;
        let kernel = ResidueKernelPoint::new(alpha, SLOTS)?;
        let cells = weights
            .get_mut(..self.natural_len)
            .ok_or(AkitaError::InvalidProof)?;
        cfg_chunks_mut!(cells, SLOTS).enumerate().try_for_each(
            |(chunk, cells)| -> Result<(), AkitaError> {
                let cell = checked::product([chunk, SLOTS]).ok_or(AkitaError::InvalidProof)?;
                let Some((segment, row, q, j)) = self.locate(cell)? else {
                    return Ok(());
                };
                let mut equality = [E::zero(); SLOTS];
                for (t, value) in equality.iter_mut().enumerate() {
                    if let Some(address) = segment.address(q, j, t) {
                        *value = eq.high(address)?
                            * *eq
                                .low()
                                .get(eq.low_offset(address))
                                .ok_or(AkitaError::InvalidProof)?;
                    }
                }
                kernel.field_kernel_into(&equality, cells)?;
                let batch = *batches
                    .get(segment.role)
                    .and_then(|batches| batches.get(row))
                    .ok_or(AkitaError::InvalidProof)?;
                for value in cells.iter_mut() {
                    *value *= batch;
                }
                Ok(())
            },
        )?;
        Ok(weights)
    }

    /// B/D part of the Stage 3 weight at the prefix point `point`.
    ///
    /// Per segment: `lambda = K(eq(point_slot, .))` and one pass over the
    /// `P * degree` digits of a period into per-lane column sums; per piece,
    /// one contraction with its column equality and one carry recursion per
    /// distinct lane-high value and live period count. No setup entry is read.
    pub fn digit_weight<E: Field>(
        &self,
        point: &[E],
        root_point: &[E],
        alpha: E,
        batches: [&[E]; 2],
    ) -> Result<E, AkitaError> {
        self.validate_batches(batches)?;
        // Every region keeps the six slot bits free and first.
        let slot_point = point.get(..SLOT_BITS).ok_or(AkitaError::InvalidProof)?;
        let lambda = ResidueKernelPoint::new(alpha, SLOTS)?
            .field_kernel(&EqPolynomial::evals(slot_point)?)?;
        let views = self
            .regions
            .iter()
            .map(|region| region.view(point))
            .collect::<Result<Vec<_>, _>>()?;
        let mut total = E::zero();
        for (index, segment) in self.segments.iter().enumerate() {
            let stride_bits = segment.lanes.stride().trailing_zeros() as usize;
            let period_bits = segment.period.trailing_zeros() as usize;
            let lane_end =
                checked::sum([stride_bits, period_bits]).ok_or(AkitaError::InvalidProof)?;
            let coefficient_eq = EqPolynomial::evals(
                root_point
                    .get(..stride_bits)
                    .ok_or(AkitaError::InvalidProof)?,
            )?;
            let lane_eq = EqPolynomial::evals(
                root_point
                    .get(stride_bits..lane_end)
                    .ok_or(AkitaError::InvalidProof)?,
            )?;
            let lane_point = root_point.get(lane_end..).ok_or(AkitaError::InvalidProof)?;
            let first_lane = segment.lanes.start() / segment.lanes.stride();
            let count = segment.lane_count();
            let degree = segment.lanes.degree();
            let shift = segment.offset % SLOTS;
            // Per period lane p: lane factor, lane-high value, live periods
            // and the slot-contracted sum of each period column.
            let mut lanes = Vec::new();
            for p in 0..segment.period.min(count) {
                let lane = first_lane + p;
                let factor = *lane_eq
                    .get(lane & (segment.period - 1))
                    .ok_or(AkitaError::InvalidProof)?;
                let live =
                    checked::div_ceil(count - p, segment.period).ok_or(AkitaError::InvalidProof)?;
                let mut columns = vec![E::zero(); segment.columns];
                for (coefficient, &weight) in coefficient_eq.iter().take(degree).enumerate() {
                    let local = checked::mul_add(p, degree, shift + coefficient)
                        .ok_or(AkitaError::InvalidProof)?;
                    *columns
                        .get_mut(local / SLOTS)
                        .ok_or(AkitaError::InvalidProof)? +=
                        weight * *lambda.get(local % SLOTS).ok_or(AkitaError::InvalidProof)?;
                }
                lanes.push((factor, lane >> period_bits, live, columns));
            }
            let row_batches = *batches.get(segment.role).ok_or(AkitaError::InvalidProof)?;
            for piece in self.pieces.iter().filter(|piece| piece.segment == index) {
                let (region_factor, coordinates) =
                    views.get(piece.region).ok_or(AkitaError::InvalidProof)?;
                let column_bits =
                    checked::sum([SLOT_BITS, piece.width_bits]).ok_or(AkitaError::InvalidProof)?;
                let column_eq = EqPolynomial::evals(
                    coordinates
                        .get(SLOT_BITS..column_bits)
                        .ok_or(AkitaError::InvalidProof)?,
                )?;
                let period_point = coordinates
                    .get(column_bits..)
                    .ok_or(AkitaError::InvalidProof)?;
                let periods = checked::pow2(piece.period_bits).ok_or(AkitaError::InvalidProof)?;
                // (lane-high of period q0, live periods in the piece) -> weight.
                let mut grouped: Vec<(usize, usize, E)> = Vec::new();
                for (factor, high, live, columns) in &lanes {
                    let live = live.saturating_sub(piece.q0).min(periods);
                    if live == 0 {
                        continue;
                    }
                    let mut sum = E::zero();
                    for (inner, &eq) in column_eq.iter().enumerate() {
                        sum += eq
                            * *columns
                                .get(piece.first + inner)
                                .ok_or(AkitaError::InvalidProof)?;
                    }
                    let high = checked::sum([*high, piece.q0]).ok_or(AkitaError::InvalidProof)?;
                    match grouped
                        .iter_mut()
                        .find(|entry| (entry.0, entry.1) == (high, live))
                    {
                        Some(entry) => entry.2 += *factor * sum,
                        None => grouped.push((high, live, *factor * sum)),
                    }
                }
                let batch = *row_batches.get(piece.row).ok_or(AkitaError::InvalidProof)?;
                let first = piece.base >> column_bits;
                for (high, live, weight) in grouped {
                    total += batch
                        * *region_factor
                        * weight
                        * paired_offset_eq_sum(period_point, first, lane_point, high, live)?;
                }
            }
        }
        Ok(total)
    }

    fn validate_batches<E>(&self, batches: [&[E]; 2]) -> Result<(), AkitaError> {
        if batches.iter().any(|batches| batches.len() != self.rows) {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }

    /// Segment, row, period and period column of the copy at `cell`, if any.
    fn locate(
        &self,
        cell: usize,
    ) -> Result<Option<(DigitSegment, usize, usize, usize)>, AkitaError> {
        for (index, region) in self.regions.iter().enumerate() {
            let Some(local) = region.extract(cell) else {
                continue;
            };
            let start = self.pieces.partition_point(|piece| piece.region < index);
            let end = self.pieces.partition_point(|piece| piece.region <= index);
            let pieces = self
                .pieces
                .get(start..end)
                .ok_or(AkitaError::InvalidProof)?;
            let position = pieces.partition_point(|piece| piece.base <= local);
            let Some(piece) = position.checked_sub(1).and_then(|p| pieces.get(p)) else {
                continue;
            };
            let chunk = (local - piece.base) >> SLOT_BITS;
            if chunk >> (piece.period_bits + piece.width_bits) != 0 {
                continue;
            }
            let segment = *self
                .segments
                .get(piece.segment)
                .ok_or(AkitaError::InvalidProof)?;
            let j = piece.first + (chunk & ((1usize << piece.width_bits) - 1));
            let q = piece.q0 + (chunk >> piece.width_bits);
            return Ok(Some((segment, piece.row, q, j)));
        }
        Ok(None)
    }
}

/// `[start, start + 2^bits)` runs of the binary expansion of `count`, highest first.
fn binary_runs(count: usize) -> Vec<(usize, usize)> {
    let mut start = 0;
    let mut runs = Vec::new();
    for bits in (0..usize::BITS as usize).rev() {
        if (count >> bits) & 1 == 1 {
            runs.push((start, bits));
            start += 1 << bits;
        }
    }
    runs
}

/// Aligned power-of-two runs `(start, bits)` covering `[start, end)`.
fn aligned_runs(mut start: usize, end: usize) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    while start < end {
        let mut bits = start.trailing_zeros().min(usize::BITS - 1) as usize;
        while bits > 0 && start + (1 << bits) > end {
            bits -= 1;
        }
        runs.push((start, bits));
        start += 1 << bits;
    }
    runs
}

/// Subcubes of the A tensor's zero padding, largest first: the padded rank
/// slots, and the coefficient tail `[648, 1024)` of the live ranks in aligned
/// runs of at least 64 cells. A cell is `row * row_stride + rank * 1024 + c`.
/// Without a power-of-two row count the padding is not a union of subcubes
/// over all rows, and the layout uses none of it.
fn a_padding_regions(a: &BinaryAPrefixLen, rank: usize) -> Result<Vec<Region>, AkitaError> {
    if !a.natural_len.is_power_of_two() {
        return Ok(Vec::new());
    }
    let coefficient_bits = A_STRIDE.trailing_zeros() as usize;
    let rank_bits = (a.row_stride / A_STRIDE).trailing_zeros() as usize;
    let rank_end = checked::sum([coefficient_bits, rank_bits]).ok_or(AkitaError::InvalidProof)?;
    let top = a.natural_len.trailing_zeros() as usize;
    let mut cubes = Vec::new();
    for (ranks, coefficients) in [
        ((rank, a.row_stride / A_STRIDE), (0, A_STRIDE)),
        ((0, rank), (A_DEGREE, A_STRIDE)),
    ] {
        for (rank_start, rank_free) in aligned_runs(ranks.0, ranks.1) {
            for (start, free) in aligned_runs(coefficients.0, coefficients.1) {
                if free < SLOT_BITS {
                    continue;
                }
                let mut fixed = Vec::new();
                let mut free_bits: Vec<usize> = (0..free).collect();
                fixed.extend((free..coefficient_bits).map(|bit| (bit, (start >> bit) & 1 == 1)));
                free_bits.extend(coefficient_bits..coefficient_bits + rank_free);
                fixed.extend(
                    (rank_free..rank_bits)
                        .map(|bit| (coefficient_bits + bit, (rank_start >> bit) & 1 == 1)),
                );
                free_bits.extend(rank_end..top);
                cubes.push(Region {
                    fixed,
                    free: Some(free_bits),
                    top,
                });
            }
        }
    }
    cubes.sort_by_key(|region| std::cmp::Reverse(region.capacity_bits()));
    Ok(cubes)
}

/// `sum_{q < count} eq(a, u + q) * eq(b, v + q)`, little endian.
///
/// An index that does not fit its point's arity has equality zero. One
/// low-to-high pass over the bits keeps both carries and whether the low bits
/// of `q` are below those of `count`: eight states, independent of `count`.
fn paired_offset_eq_sum<E: Field>(
    a: &[E],
    u: usize,
    b: &[E],
    v: usize,
    count: usize,
) -> Result<E, AkitaError> {
    let count_bits = usize::BITS as usize - count.leading_zeros() as usize;
    let bits = a.len().max(b.len()).max(count_bits);
    let factor = |point: &[E], bit: usize, value: usize| -> E {
        match (point.get(bit), value) {
            (Some(&x), 1) => x,
            (Some(&x), _) => E::one() - x,
            (None, 1) => E::zero(),
            (None, _) => E::one(),
        }
    };
    let digit = |value: usize, bit: usize| value.checked_shr(bit as u32).unwrap_or(0) & 1;
    // State index: carry_a | carry_b << 1 | below << 2.
    let mut states = [E::zero(); 8];
    states[0] = E::one();
    for bit in 0..bits {
        let mut next = [E::zero(); 8];
        for (state, &weight) in states.iter().enumerate() {
            if weight == E::zero() {
                continue;
            }
            for z in 0..2 {
                let sum_a = digit(u, bit) + z + (state & 1);
                let sum_b = digit(v, bit) + z + ((state >> 1) & 1);
                let bound = digit(count, bit);
                let below = usize::from(z < bound || (z == bound && (state >> 2) & 1 == 1));
                let target = (sum_a >> 1) | ((sum_b >> 1) << 1) | (below << 2);
                *next.get_mut(target).ok_or(AkitaError::InvalidProof)? +=
                    weight * factor(a, bit, sum_a & 1) * factor(b, bit, sum_b & 1);
            }
        }
        states = next;
    }
    // Bits of u or v above every point coordinate make the index too large.
    if u.checked_shr(bits as u32).unwrap_or(0) != 0 || v.checked_shr(bits as u32).unwrap_or(0) != 0
    {
        return Ok(E::zero());
    }
    Ok(states[4])
}

#[cfg(test)]
#[path = "typed_prefix_tests.rs"]
mod tests;
