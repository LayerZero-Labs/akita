//! Ordinary commitment preimages authenticated in a mixed native root.

use akita_algebra::offset_eq::OffsetEqWindow;
use akita_algebra::ring::ResidueKernelPoint;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BINARY_SETUP_P;
use akita_params::{
    CompressionChainPlan, CompressionMapPlan, GroupCommitPhaseParams, RelationRowGeometry,
};
use jolt_field::{CanonicalEncoding, Field, Prime128Offset275};

use super::paired_offset_eq_sum;
use crate::native_root_relation::{NativeRootWeights, RootEq, RootWeightFamily, TranslatedLanes};
use crate::root_repacking::{RootCoefficientRepacking, RootRepackingSegment};
use crate::{derive_public_matrix_prefix, AkitaSetupDescriptor};

/// Four ordinary preimage families sharing the native root's outgoing table.
///
/// S is the original small coefficient table; T is the ordinary A image in
/// balanced base-eight digits; the remaining two families are the existing
/// commitment compression's negative-binary digits. Their `{-1,0}` predicates
/// are compression constraints, independent of the binary source agreement.
#[derive(Clone, Debug)]
pub struct OrdinaryNativeRootPlan {
    profile: GroupCommitPhaseParams,
    pub(super) first_segment: usize,
    segments: [RootRepackingSegment; 4],
    compression: CompressionChainPlan,
    linear_batch_count: usize,
}

impl OrdinaryNativeRootPlan {
    pub fn new(profile: GroupCommitPhaseParams, first_segment: usize) -> Result<Self, AkitaError> {
        profile.validate_frozen_precommit(128)?;
        if profile.group.num_polynomials() != 1
            || profile.inner.digits.num_digits != 1
            || profile.inner.digits.log_basis < 3
            || profile.outer.digits.log_basis != 3
            || profile.inner.matrix.sis_modulus_profile().modulus() != BINARY_SETUP_P
        {
            return Err(AkitaError::InvalidSetup(
                "mixed root requires one ordinary small coefficient source and base-eight image digits".into(),
            ));
        }
        let a_degree = profile.inner.matrix.ring_dimension();
        let b_degree = profile.outer.matrix.ring_dimension();
        let subcolumns = checked::exact_div(a_degree, b_degree).ok_or(AkitaError::InvalidProof)?;
        let source_elements = checked::product([
            profile.blocks.live_blocks,
            profile.blocks.positions_per_block,
        ])
        .ok_or(AkitaError::InvalidProof)?;
        let image_elements = checked::product([
            profile.blocks.live_blocks,
            profile.inner.matrix.output_rank(),
            subcolumns,
        ])
        .ok_or(AkitaError::InvalidProof)?;
        let compression = profile.outer_compression_plan()?;
        let first = compression.maps().first().ok_or(AkitaError::InvalidProof)?;
        let second = compression.maps().get(1).ok_or(AkitaError::InvalidProof)?;
        if compression.maps().len() != 2 || compression.field_bits() != 128 {
            return Err(AkitaError::InvalidSetup(
                "mixed root compression geometry is unsupported".into(),
            ));
        }
        // The selected root grinding plan budgets a maximum native degree of
        // 648 for its shared ring-switch challenge.
        if [
            a_degree,
            b_degree,
            first.ring_dimension(),
            second.ring_dimension(),
        ]
        .into_iter()
        .any(|degree| degree > 648)
        {
            return Err(AkitaError::InvalidSetup(
                "mixed root native degree exceeds the root grinding budget".into(),
            ));
        }
        let segments = [
            RootRepackingSegment {
                geometry: RelationRowGeometry::native(a_degree)?,
                elements: source_elements,
                digits: 1,
            },
            RootRepackingSegment {
                geometry: RelationRowGeometry::native(b_degree)?,
                elements: image_elements,
                digits: profile.outer.digits.num_digits,
            },
            RootRepackingSegment {
                geometry: RelationRowGeometry::native(first.ring_dimension())?,
                elements: first.input_width(),
                digits: 1,
            },
            RootRepackingSegment {
                geometry: RelationRowGeometry::native(second.ring_dimension())?,
                elements: second.input_width(),
                digits: 1,
            },
        ];
        let linear_batch_count = checked::product([
            profile.blocks.live_blocks,
            profile.inner.matrix.output_rank(),
        ])
        .and_then(|a_rows| {
            profile
                .outer_slice_count
                .logical_output_rows(profile.outer.matrix.output_rank())
                .ok()
                .and_then(|b_rows| checked::sum([a_rows, b_rows, 3]))
        })
        .ok_or(AkitaError::InvalidProof)?;
        checked::sum([first_segment, segments.len()]).ok_or(AkitaError::InvalidProof)?;
        Ok(Self {
            profile,
            first_segment,
            segments,
            compression,
            linear_batch_count,
        })
    }

    pub fn segments(&self) -> &[RootRepackingSegment] {
        &self.segments
    }
    pub fn linear_batch_count(&self) -> usize {
        self.linear_batch_count
    }

    fn validate_repacking(&self, repacking: &RootCoefficientRepacking) -> Result<(), AkitaError> {
        let range = checked::range(self.first_segment, self.segments.len())
            .ok_or(AkitaError::InvalidProof)?;
        if repacking.segments().get(range) != Some(self.segments.as_slice()) {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }

    /// Addresses of the existing compression digits requiring `w(w+1)=0`.
    pub fn negative_binary_addresses(
        &self,
        repacking: &RootCoefficientRepacking,
    ) -> Result<Vec<usize>, AkitaError> {
        self.validate_repacking(repacking)?;
        let mut addresses = Vec::new();
        for offset in [2, 3] {
            let segment =
                checked::sum([self.first_segment, offset]).ok_or(AkitaError::InvalidProof)?;
            let shape = self.segments.get(offset).ok_or(AkitaError::InvalidProof)?;
            let count = checked::product([
                shape.elements,
                shape.geometry.polynomial_modulus_dimension(),
            ])
            .ok_or(AkitaError::InvalidProof)?;
            addresses
                .try_reserve_exact(count)
                .map_err(|_| AkitaError::InvalidProof)?;
            for element in 0..shape.elements {
                for coefficient in 0..shape.geometry.polynomial_modulus_dimension() {
                    addresses.push(repacking.coefficient_address(
                        segment,
                        element,
                        0,
                        coefficient,
                    )?);
                }
            }
        }
        Ok(addresses)
    }

    /// Direct public A/B/compression and source-evaluation identities.
    ///
    /// Rows are A in block/row order, B in slice/row order, the two compression
    /// maps, then the scalar evaluation. The public matrix is the canonical
    /// ordinary setup prefix, reused with each role's own dimensions.
    ///
    /// Each identity repeats one public pattern: an A row reads the same
    /// kernels in every block, a B row the same kernels in every slice of one
    /// width, and a recomposition the same coefficients at every bit. The
    /// weights keep each pattern once as [`TranslatedLanes`], and the scalar
    /// evaluation as an offset equality. The verifier evaluates them at the
    /// Stage 2 point without one term per table address; the prover
    /// materializes the same field elements.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_weights<E: Field>(
        &self,
        repacking: &RootCoefficientRepacking,
        setup: &AkitaSetupDescriptor,
        point: &[E],
        evaluation: E,
        payload: &[E],
        alpha: E,
        batches: &[E],
    ) -> Result<NativeRootWeights<E>, AkitaError> {
        self.validate_repacking(repacking)?;
        if point.len() != self.profile.group.num_vars()
            || batches.len() != self.linear_batch_count
            || payload.len() != self.compression.terminal_coefficients()
            || E::from_u128(BINARY_SETUP_P) != E::zero()
        {
            return Err(AkitaError::InvalidProof);
        }
        let a = self.profile.inner.matrix;
        let b = self.profile.outer.matrix;
        let [first, second] = self.maps()?;
        let required = [
            checked::product([a.output_rank(), a.input_width(), a.ring_dimension()]),
            checked::product([b.output_rank(), b.input_width(), b.ring_dimension()]),
            Some(self.compression.max_setup_field_elements()?),
        ]
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(AkitaError::InvalidProof)?
        .into_iter()
        .max()
        .ok_or(AkitaError::InvalidProof)?;
        if required > setup.num_field_elements
            || required > crate::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS
        {
            return Err(AkitaError::InvalidSetup(
                "mixed root exceeds the admitted setup prefix".into(),
            ));
        }
        let matrix = derive_public_matrix_prefix::<Prime128Offset275>(required, &setup.setup_seed);
        let matrix = matrix.as_field_slice();
        let table_len = repacking.domain_len();
        let lanes = |offset| {
            checked::sum([self.first_segment, offset])
                .ok_or(AkitaError::InvalidProof)
                .and_then(|segment| ContiguousLanes::new(repacking, segment))
        };
        let [s_lanes, t_lanes, f_lanes, h_lanes] = [lanes(0)?, lanes(1)?, lanes(2)?, lanes(3)?];
        let subcolumns = checked::exact_div(a.ring_dimension(), b.ring_dimension())
            .ok_or(AkitaError::InvalidProof)?;
        let digits = self.profile.outer.digits.num_digits;
        let a_rows = checked::product([self.profile.blocks.live_blocks, a.output_rank()])
            .ok_or(AkitaError::InvalidProof)?;
        let (a_batches, rest) = batches
            .split_at_checked(a_rows)
            .ok_or(AkitaError::InvalidProof)?;
        let b_rows = rest.len().checked_sub(3).ok_or(AkitaError::InvalidProof)?;
        let (b_batches, map_batches) = rest
            .split_at_checked(b_rows)
            .ok_or(AkitaError::InvalidProof)?;
        let &[first_batch, second_batch, evaluation_batch] = map_batches else {
            return Err(AkitaError::InvalidProof);
        };
        let mut families = Vec::new();
        let mut translated = |copies, lanes| {
            families.push(RootWeightFamily::Translated(TranslatedLanes::new(
                table_len, copies, lanes,
            )?));
            Ok::<_, AkitaError>(())
        };
        // A rows read their kernels against the source in every block.
        let mut a_kernels = KernelCache::new(
            matrix,
            a.output_rank(),
            a.input_width(),
            a.ring_dimension(),
            alpha,
        )?;
        let positions = self.profile.blocks.positions_per_block;
        for a_row in 0..a.output_rank() {
            let copies = (0..self.profile.blocks.live_blocks)
                .map(|block| {
                    let batch = checked::mul_add(block, a.output_rank(), a_row)
                        .and_then(|row| a_batches.get(row))
                        .ok_or(AkitaError::InvalidProof)?;
                    let element =
                        checked::product([block, positions]).ok_or(AkitaError::InvalidProof)?;
                    Ok((s_lanes.base(element, 0)?, *batch))
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            let kernels = (0..positions)
                .map(|position| {
                    Ok((
                        s_lanes.offset(position, 0)?,
                        a_kernels.get(a_row, position)?.to_vec(),
                    ))
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            translated(copies, kernels)?;
        }
        // Every A image subtracts its recomposed T digits, scaled by its row.
        let image_copies = a_batches
            .iter()
            .enumerate()
            .map(|(image, &batch)| {
                let element =
                    checked::product([image, subcolumns]).ok_or(AkitaError::InvalidProof)?;
                Ok((t_lanes.base(element, 0)?, batch))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let mut recomposition = Vec::new();
        let mut power = -E::one();
        for subcolumn in 0..subcolumns {
            let powers = (0..b.ring_dimension())
                .map(|_| {
                    let value = power;
                    power *= alpha;
                    value
                })
                .collect::<Vec<_>>();
            let mut digit_scale = E::one();
            for digit in 0..digits {
                recomposition.push((
                    t_lanes.offset(subcolumn, digit)?,
                    powers.iter().map(|&value| value * digit_scale).collect(),
                ));
                digit_scale *= E::from_u64(8);
            }
        }
        translated(image_copies, recomposition)?;
        // B rows read their kernels against the T digits of every slice; slices
        // of one width share one pattern.
        let mut b_kernels = KernelCache::new(
            matrix,
            b.output_rank(),
            b.input_width(),
            b.ring_dimension(),
            alpha,
        )?;
        let block_elements =
            checked::product([a.output_rank(), subcolumns]).ok_or(AkitaError::InvalidProof)?;
        let geometry = self.profile.derive_slice_geometry()?;
        let mut widths = geometry
            .block_ranges()
            .iter()
            .map(ExactSizeIterator::len)
            .collect::<Vec<_>>();
        widths.sort_unstable();
        widths.dedup();
        for width in widths {
            for b_row in 0..b.output_rank() {
                let copies = geometry
                    .block_ranges()
                    .iter()
                    .enumerate()
                    .filter(|(_, range)| range.len() == width)
                    .map(|(slice, range)| {
                        let batch = checked::mul_add(slice, b.output_rank(), b_row)
                            .and_then(|row| b_batches.get(row))
                            .ok_or(AkitaError::InvalidProof)?;
                        let element = checked::product([range.start, block_elements])
                            .ok_or(AkitaError::InvalidProof)?;
                        Ok((t_lanes.base(element, 0)?, *batch))
                    })
                    .collect::<Result<Vec<_>, AkitaError>>()?;
                let columns =
                    checked::product([width, block_elements]).ok_or(AkitaError::InvalidProof)?;
                let mut kernels = Vec::new();
                for element in 0..columns {
                    for digit in 0..digits {
                        let column = checked::mul_add(element, digits, digit)
                            .ok_or(AkitaError::InvalidProof)?;
                        kernels.push((
                            t_lanes.offset(element, digit)?,
                            b_kernels.get(b_row, column)?.to_vec(),
                        ));
                    }
                }
                translated(copies, kernels)?;
            }
        }
        // Every B image coefficient subtracts its recomposed first-map bits.
        let mut image = Vec::new();
        image
            .try_reserve_exact(
                checked::product([b_batches.len(), b.ring_dimension()])
                    .ok_or(AkitaError::InvalidProof)?,
            )
            .map_err(|_| AkitaError::InvalidProof)?;
        for &batch in b_batches {
            let mut power = -batch;
            for _ in 0..b.ring_dimension() {
                image.push(power);
                power *= alpha;
            }
        }
        translated(
            f_lanes.bit_copies(first.input_coefficients(), image.len())?,
            vec![(0, image)],
        )?;
        // Each compression map reads its kernels against its input digits.
        for (map, lanes, batch) in [
            (first, &f_lanes, first_batch),
            (second, &h_lanes, second_batch),
        ] {
            let mut map_kernels =
                KernelCache::new(matrix, 1, map.input_width(), map.ring_dimension(), alpha)?;
            let kernels = (0..map.input_width())
                .map(|column| {
                    Ok((
                        lanes.offset(column, 0)?,
                        map_kernels.get(0, column)?.to_vec(),
                    ))
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            translated(vec![(lanes.base(0, 0)?, batch)], kernels)?;
        }
        let mut output = Vec::new();
        let mut power = -first_batch;
        for _ in 0..first.output_coefficients() {
            output.push(power);
            power *= alpha;
        }
        translated(
            h_lanes.bit_copies(second.input_coefficients(), output.len())?,
            vec![(0, output)],
        )?;
        let source_len = checked::pow2(point.len()).ok_or(AkitaError::InvalidProof)?;
        families.push(RootWeightFamily::SourceEvaluation(
            SourceEvaluationWeights::new(
                table_len,
                s_lanes.contiguous(0, source_len)?,
                point,
                evaluation_batch,
            )?,
        ));
        let claim = second_batch
            * payload
                .iter()
                .rev()
                .fold(E::zero(), |sum, &coefficient| sum * alpha + coefficient)
            + evaluation_batch * evaluation;
        NativeRootWeights::structured(table_len, Vec::new(), families, claim)
    }

    fn maps(&self) -> Result<[CompressionMapPlan; 2], AkitaError> {
        match self.compression.maps() {
            &[first, second] => Ok([first, second]),
            _ => Err(AkitaError::InvalidProof),
        }
    }
}

/// Residue kernels of one public matrix role, each computed on first use.
///
/// Entry `(row, column)` is the degree-`degree` ring at flat offset
/// `(row * width + column) * degree` of the ordinary setup prefix.
struct KernelCache<'a, E: Field> {
    matrix: &'a [Prime128Offset275],
    width: usize,
    degree: usize,
    point: ResidueKernelPoint<E>,
    kernels: Vec<Option<Vec<E>>>,
}

impl<'a, E: Field> KernelCache<'a, E> {
    fn new(
        matrix: &'a [Prime128Offset275],
        rows: usize,
        width: usize,
        degree: usize,
        alpha: E,
    ) -> Result<Self, AkitaError> {
        let entries = checked::product([rows, width]).ok_or(AkitaError::InvalidProof)?;
        let mut kernels = Vec::new();
        kernels
            .try_reserve_exact(entries)
            .map_err(|_| AkitaError::InvalidProof)?;
        kernels.resize(entries, None);
        Ok(Self {
            matrix,
            width,
            degree,
            point: ResidueKernelPoint::new(alpha, degree)?,
            kernels,
        })
    }

    fn get(&mut self, row: usize, column: usize) -> Result<&[E], AkitaError> {
        if column >= self.width {
            return Err(AkitaError::InvalidProof);
        }
        let entry = checked::mul_add(row, self.width, column).ok_or(AkitaError::InvalidProof)?;
        let start = checked::product([entry, self.degree]).ok_or(AkitaError::InvalidProof)?;
        let range = checked::range(start, self.degree).ok_or(AkitaError::InvalidProof)?;
        let slot = self
            .kernels
            .get_mut(entry)
            .ok_or(AkitaError::InvalidProof)?;
        if slot.is_none() {
            let coefficients = self
                .matrix
                .get(range)
                .ok_or(AkitaError::InvalidProof)?
                .iter()
                .map(|coefficient| {
                    coefficient
                        .to_u128_checked()
                        .map(E::from_u128)
                        .ok_or(AkitaError::InvalidProof)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut weights = vec![E::zero(); self.degree];
            self.point.field_kernel_into(&coefficients, &mut weights)?;
            *slot = Some(weights);
        }
        slot.as_deref().ok_or(AkitaError::InvalidProof)
    }
}

/// Addresses of one single-plane, unpadded segment with contiguous lanes.
///
/// Natural coefficient `c` of lane `(element, digit)` sits at `base + c`, and
/// consecutive lanes abut, so a run of coefficients that crosses elements is
/// one address interval.
struct ContiguousLanes {
    start: usize,
    degree: usize,
    digits: usize,
    elements: usize,
}

impl ContiguousLanes {
    fn new(repacking: &RootCoefficientRepacking, segment: usize) -> Result<Self, AkitaError> {
        let shape = repacking
            .segments()
            .get(segment)
            .ok_or(AkitaError::InvalidProof)?;
        let degree = shape.geometry.polynomial_modulus_dimension();
        if shape.geometry.coordinate_plane_count() != 1
            || shape.geometry.physical_coefficient_width() != degree
            || degree == 0
            || shape.geometry.physical_coefficient_index(0, degree - 1)? != degree - 1
        {
            return Err(AkitaError::InvalidSetup(
                "mixed root ordinary segment is not a contiguous native table".into(),
            ));
        }
        let lanes = Self {
            start: repacking.segment_range(segment)?.start,
            degree,
            digits: shape.digits,
            elements: shape.elements,
        };
        if repacking.segment_range(segment)?.end
            != checked::sum([lanes.start, lanes.len()?]).ok_or(AkitaError::InvalidProof)?
        {
            return Err(AkitaError::InvalidProof);
        }
        Ok(lanes)
    }

    fn len(&self) -> Result<usize, AkitaError> {
        checked::product([self.elements, self.digits, self.degree]).ok_or(AkitaError::InvalidProof)
    }

    /// Offset of lane `(element, digit)` from the segment start.
    fn offset(&self, element: usize, digit: usize) -> Result<usize, AkitaError> {
        if element >= self.elements || digit >= self.digits {
            return Err(AkitaError::InvalidProof);
        }
        checked::mul_add(element, self.digits, digit)
            .and_then(|lane| checked::product([lane, self.degree]))
            .ok_or(AkitaError::InvalidProof)
    }

    /// Address of coefficient zero of lane `(element, digit)`.
    fn base(&self, element: usize, digit: usize) -> Result<usize, AkitaError> {
        checked::sum([self.start, self.offset(element, digit)?]).ok_or(AkitaError::InvalidProof)
    }

    /// Start of `len` consecutive coefficients from flat coefficient `first`
    /// of a single-digit segment.
    fn contiguous(&self, first: usize, len: usize) -> Result<usize, AkitaError> {
        if self.digits != 1 {
            return Err(AkitaError::InvalidProof);
        }
        checked::sum([first, len])
            .filter(|&end| end <= self.len().unwrap_or(0))
            .and_then(|_| checked::sum([self.start, first]))
            .ok_or(AkitaError::InvalidProof)
    }

    /// One copy per bit of a 128-bit negative-binary recomposition.
    ///
    /// Bit `i` of input coefficient `c` sits at flat coefficient
    /// `i * coefficients + c`, with scale `2^i`.
    fn bit_copies<E: Field>(
        &self,
        coefficients: usize,
        used: usize,
    ) -> Result<Vec<(usize, E)>, AkitaError> {
        if used > coefficients {
            return Err(AkitaError::InvalidProof);
        }
        let mut scale = E::one();
        (0..128)
            .map(|bit| {
                let first =
                    checked::product([bit, coefficients]).ok_or(AkitaError::InvalidProof)?;
                let copy = (self.contiguous(first, used)?, scale);
                scale += scale;
                Ok(copy)
            })
            .collect()
    }
}

/// The scalar source evaluation: `scale * eq(point, x)` at `start + x` for
/// every source coefficient `x`.
///
/// Its extension at `r` is `scale * sum_x eq(r, start + x) * eq(point, x)`,
/// one carry-tracking pass over the address bits.
#[derive(Clone, Debug)]
pub(crate) struct SourceEvaluationWeights<E: Field> {
    table_len: usize,
    start: usize,
    point: Vec<E>,
    scale: E,
}

impl<E: Field> SourceEvaluationWeights<E> {
    fn new(table_len: usize, start: usize, point: &[E], scale: E) -> Result<Self, AkitaError> {
        let end = checked::pow2(point.len())
            .and_then(|len| checked::sum([start, len]))
            .ok_or(AkitaError::InvalidProof)?;
        if end > table_len {
            return Err(AkitaError::InvalidProof);
        }
        Ok(Self {
            table_len,
            start,
            point: point.to_vec(),
            scale,
        })
    }

    pub(crate) fn table_len(&self) -> usize {
        self.table_len
    }

    pub(crate) fn evaluate(&self, eq: &RootEq<E>) -> Result<E, AkitaError> {
        let count = checked::pow2(self.point.len()).ok_or(AkitaError::InvalidProof)?;
        Ok(self.scale * paired_offset_eq_sum(eq.point(), self.start, &self.point, 0, count)?)
    }

    pub(crate) fn visit(
        &self,
        mut visit: impl FnMut(usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        let equality = OffsetEqWindow::new(&self.point)?;
        let count = checked::pow2(self.point.len()).ok_or(AkitaError::InvalidProof)?;
        for coefficient in 0..count {
            visit(
                self.start + coefficient,
                self.scale * equality.eval(coefficient),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "ordinary_root_reference.rs"]
mod reference;
