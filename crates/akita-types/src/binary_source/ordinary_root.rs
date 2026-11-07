//! Ordinary commitment preimages authenticated in a mixed native root.

use akita_algebra::offset_eq::OffsetEqWindow;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::ordinary_source::OrdinarySourceWidth;
use akita_params::{
    CompressionChainPlan, CompressionMapPlan, GroupCommitPhaseParams, RelationRowGeometry,
    SisModulusProfileId,
};
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::{
    cfg_chunks, cfg_chunks_mut, CanonicalEncoding, Field, Prime128Offset275, Prime64Offset59,
};

use super::paired_offset_eq_sum;
use crate::native_root_relation::{
    NativeRootWeights, ProfileBuilder, RootEq, RootWeightFamily, TranslatedLanes,
};
use crate::root_repacking::{RootCoefficientRepacking, RootRepackingSegment};
use crate::{derive_public_matrix_prefix, AkitaSetupDescriptor};

/// Ordinary preimage families sharing the native root's outgoing table.
///
/// S is the original small coefficient table; T is the ordinary A image in
/// balanced base-eight digits; the remaining two families are the existing
/// commitment compression's negative-binary digits. Their `{-1,0}` predicates
/// are compression constraints, independent of the binary source agreement.
/// Explicit signed plans append negative offset bits and link them to S. With
/// widths at most 64 and canonical base-eight depth, both sides of this link
/// have absolute value below 2^67, so equality modulo P has no integer wrap.
#[derive(Clone, Debug)]
pub struct OrdinaryNativeRootPlan {
    profile: GroupCommitPhaseParams,
    pub(super) first_segment: usize,
    segments: Vec<RootRepackingSegment>,
    source_width: Option<OrdinarySourceWidth>,
    compression: CompressionChainPlan,
    linear_batch_count: usize,
}

impl OrdinaryNativeRootPlan {
    /// Admit the ordinary preimage geometry executed by the current root driver.
    /// The group must have one polynomial and outer ring dimension 64.
    pub fn new(profile: GroupCommitPhaseParams, first_segment: usize) -> Result<Self, AkitaError> {
        let field = match profile.inner.matrix.sis_modulus_profile() {
            SisModulusProfileId::Q128Offset275 => BinaryFieldProfile::P128,
            SisModulusProfileId::Q64Offset59 => BinaryFieldProfile::P64,
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "unsupported mixed root field".into(),
                ))
            }
        };
        profile
            .validate_frozen_precommit(profile.inner.matrix.sis_modulus_profile().field_bits())?;
        if profile.group.num_polynomials() != 1 {
            return Err(AkitaError::InvalidSetup(
                "mixed root ordinary group requires exactly one polynomial".into(),
            ));
        }
        if profile.inner.digits.num_digits != 1
            || profile.inner.digits.log_basis < 3
            || profile.outer.digits.log_basis != 3
            || profile.outer.matrix.sis_modulus_profile().modulus() != field.modulus()
        {
            return Err(AkitaError::InvalidSetup(
                "mixed root requires small coefficient sources and base-eight image digits".into(),
            ));
        }
        let outer_degree = profile.outer.matrix.ring_dimension();
        if outer_degree != 64 {
            return Err(AkitaError::InvalidSetup(format!(
                "mixed root ordinary profile requires outer ring dimension 64; got {outer_degree}"
            )));
        }
        Self::compile(profile, first_segment, None)
    }

    /// Compile signed arithmetic for every polynomial slot in the profile.
    ///
    /// This does not admit a native root. Both one-point and two-point root
    /// openings use `new`, which admits only one ordinary polynomial. Signed
    /// multi-slot roots need separate schedule and extraction admission.
    pub fn new_signed(
        profile: GroupCommitPhaseParams,
        first_segment: usize,
        width: OrdinarySourceWidth,
    ) -> Result<Self, AkitaError> {
        width.validate_profile(&profile)?;
        Self::compile(profile, first_segment, Some(width))
    }

    fn compile(
        profile: GroupCommitPhaseParams,
        first_segment: usize,
        source_width: Option<OrdinarySourceWidth>,
    ) -> Result<Self, AkitaError> {
        let a_degree = profile.inner.matrix.ring_dimension();
        let b_degree = profile.outer.matrix.ring_dimension();
        let subcolumns = checked::exact_div(a_degree, b_degree).ok_or(AkitaError::InvalidProof)?;
        let owners = profile.group.num_polynomials();
        let source_elements = checked::product([
            owners,
            profile.blocks.live_blocks,
            profile.blocks.positions_per_block,
        ])
        .ok_or(AkitaError::InvalidProof)?;
        let image_elements = checked::product([
            owners,
            profile.blocks.live_blocks,
            profile.inner.matrix.output_rank(),
            subcolumns,
        ])
        .ok_or(AkitaError::InvalidProof)?;
        let compression = profile.outer_compression_plan()?;
        let first = compression.maps().first().ok_or(AkitaError::InvalidProof)?;
        let second = compression.maps().get(1).ok_or(AkitaError::InvalidProof)?;
        if compression.maps().len() != 2
            || compression.field_bits()
                != profile.inner.matrix.sis_modulus_profile().field_bits() as usize
        {
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
        let mut segments = vec![
            RootRepackingSegment {
                geometry: RelationRowGeometry::native(a_degree)?,
                elements: source_elements,
                digits: profile.inner.digits.num_digits,
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
        if let Some(width) = source_width {
            // Only the explicitly signed selection requires complete cubes.
            // Preserve the legacy constructor's partial-block relation grammar.
            let source_len =
                checked::pow2(profile.group.num_vars()).ok_or(AkitaError::InvalidProof)?;
            if checked::product([source_elements, a_degree])
                != checked::product([owners, source_len])
            {
                return Err(AkitaError::InvalidSetup(
                    "ordinary native source requires complete coefficient cubes".into(),
                ));
            }
            segments.push(RootRepackingSegment {
                geometry: RelationRowGeometry::native(a_degree)?,
                elements: source_elements,
                digits: width.bits() as usize,
            });
        }
        let linear_batch_count = checked::product([
            owners,
            profile.blocks.live_blocks,
            profile.inner.matrix.output_rank(),
        ])
        .and_then(|a_rows| {
            profile
                .outer_slice_count
                .logical_output_rows(profile.outer.matrix.output_rank())
                .ok()
                .and_then(|b_rows| {
                    checked::sum([
                        a_rows,
                        b_rows,
                        2,
                        owners,
                        source_width.map_or(0, |_| source_elements),
                    ])
                })
        })
        .ok_or(AkitaError::InvalidProof)?;
        checked::sum([first_segment, segments.len()]).ok_or(AkitaError::InvalidProof)?;
        Ok(Self {
            profile,
            first_segment,
            segments,
            source_width,
            compression,
            linear_batch_count,
        })
    }

    pub fn profile(&self) -> &GroupCommitPhaseParams {
        &self.profile
    }

    pub fn source_width(&self) -> Option<OrdinarySourceWidth> {
        self.source_width
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

    /// Addresses of compression digits and signed offset bits requiring `w(w+1)=0`.
    pub fn negative_binary_addresses(
        &self,
        repacking: &RootCoefficientRepacking,
    ) -> Result<Vec<usize>, AkitaError> {
        self.validate_repacking(repacking)?;
        let mut addresses = Vec::new();
        for offset in 2..self.segments.len() {
            let segment =
                checked::sum([self.first_segment, offset]).ok_or(AkitaError::InvalidProof)?;
            let shape = self.segments.get(offset).ok_or(AkitaError::InvalidProof)?;
            let count = checked::product([
                shape.elements,
                shape.digits,
                shape.geometry.polynomial_modulus_dimension(),
            ])
            .ok_or(AkitaError::InvalidProof)?;
            addresses
                .try_reserve_exact(count)
                .map_err(|_| AkitaError::InvalidProof)?;
            for element in 0..shape.elements {
                for digit in 0..shape.digits {
                    for coefficient in 0..shape.geometry.polynomial_modulus_dimension() {
                        addresses.push(repacking.coefficient_address(
                            segment,
                            element,
                            digit,
                            coefficient,
                        )?);
                    }
                }
            }
        }
        Ok(addresses)
    }

    /// Direct public A/B/compression and source-evaluation identities.
    ///
    /// Rows are A in owner/block/row order, B in slice/row order, the two
    /// compression maps, all slot evaluations, then signed-bound rows. The
    /// public matrix is the canonical ordinary setup prefix, reused with each
    /// role's own dimensions.
    ///
    /// Each identity repeats one public pattern: an A row reads the same
    /// kernels in every owner and block, a B row the same kernels in every
    /// slice of one width, a recomposition the same coefficients at every bit,
    /// and a signed-bound row the same powers at every source element. The
    /// weights keep each pattern once as `TranslatedLanes`, and each
    /// single-digit slot evaluation as an offset equality. The verifier
    /// evaluates them at the Stage 2 point without one term per table address;
    /// the prover materializes the same field elements.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_weights<E: Field>(
        &self,
        repacking: &RootCoefficientRepacking,
        setup: &AkitaSetupDescriptor,
        point: &[E],
        evaluations: &[E],
        payload: &[E],
        alpha: E,
        batches: &[E],
    ) -> Result<NativeRootWeights<E>, AkitaError> {
        self.validate_repacking(repacking)?;
        if evaluations.len() != self.profile.group.num_polynomials()
            || point.len() != self.profile.group.num_vars()
            || batches.len() != self.linear_batch_count
            || payload.len() != self.compression.terminal_coefficients()
            || E::from_u128(self.profile.inner.matrix.sis_modulus_profile().modulus()) != E::zero()
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
        let matrix = self.setup_coefficients::<E>(required, setup)?;
        let matrix = matrix.as_slice();
        let table_len = repacking.domain_len();
        let lanes = |offset| {
            checked::sum([self.first_segment, offset])
                .ok_or(AkitaError::InvalidProof)
                .and_then(|segment| ContiguousLanes::new(repacking, segment))
        };
        let [s_lanes, t_lanes, f_lanes, h_lanes] = [lanes(0)?, lanes(1)?, lanes(2)?, lanes(3)?];
        let subcolumns = checked::exact_div(a.ring_dimension(), b.ring_dimension())
            .ok_or(AkitaError::InvalidProof)?;
        let owners = self.profile.group.num_polynomials();
        let positions = self.profile.blocks.positions_per_block;
        let source_digits = self.profile.inner.digits.num_digits;
        let digits = self.profile.outer.digits.num_digits;
        let owner_blocks = checked::product([owners, self.profile.blocks.live_blocks])
            .ok_or(AkitaError::InvalidProof)?;
        let source_elements =
            checked::product([owner_blocks, positions]).ok_or(AkitaError::InvalidProof)?;
        let a_rows =
            checked::product([owner_blocks, a.output_rank()]).ok_or(AkitaError::InvalidProof)?;
        let signed_rows = self.source_width.map_or(0, |_| source_elements);
        let (a_batches, rest) = batches
            .split_at_checked(a_rows)
            .ok_or(AkitaError::InvalidProof)?;
        let b_rows = checked::sum([2, owners, signed_rows])
            .and_then(|tail| rest.len().checked_sub(tail))
            .ok_or(AkitaError::InvalidProof)?;
        let (b_batches, rest) = rest
            .split_at_checked(b_rows)
            .ok_or(AkitaError::InvalidProof)?;
        let (map_batches, rest) = rest.split_at_checked(2).ok_or(AkitaError::InvalidProof)?;
        let &[first_batch, second_batch] = map_batches else {
            return Err(AkitaError::InvalidProof);
        };
        let (evaluation_batches, signed_batches) = rest
            .split_at_checked(owners)
            .ok_or(AkitaError::InvalidProof)?;
        let geometry = self.profile.derive_slice_geometry()?;
        if Some(b_rows) != checked::product([geometry.block_ranges().len(), b.output_rank()]) {
            return Err(AkitaError::InvalidProof);
        }
        let mut families = Vec::new();
        let mut translated = |copies, lanes, residue| {
            families.push(RootWeightFamily::Translated(TranslatedLanes::new(
                table_len, copies, lanes, residue,
            )?));
            Ok::<_, AkitaError>(())
        };
        // A rows read their residue kernels against the source digits in
        // every owner and block.
        let a_rings = RoleRings::new(matrix, a.output_rank(), a.input_width(), a.ring_dimension())?;
        for a_row in 0..a.output_rank() {
            let copies = (0..owner_blocks)
                .map(|owner_block| {
                    let batch = checked::mul_add(owner_block, a.output_rank(), a_row)
                        .and_then(|row| a_batches.get(row))
                        .ok_or(AkitaError::InvalidProof)?;
                    let element = checked::product([owner_block, positions])
                        .ok_or(AkitaError::InvalidProof)?;
                    Ok((s_lanes.base(element, 0)?, *batch))
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            let mut kernels = Vec::new();
            for position in 0..positions {
                for digit in 0..source_digits {
                    let column = checked::mul_add(position, source_digits, digit)
                        .ok_or(AkitaError::InvalidProof)?;
                    kernels.push((
                        s_lanes.offset(position, digit)?,
                        a_rings.ring(a_row, column)?,
                    ));
                }
            }
            translated(copies, kernels, Some(alpha))?;
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
        translated(image_copies, recomposition, None)?;
        // B rows read their residue kernels against the T digits of every
        // owner in every slice; slices of one width share one pattern. Each
        // owner's B columns are padded to the widest slice.
        let b_rings = RoleRings::new(matrix, b.output_rank(), b.input_width(), b.ring_dimension())?;
        let block_elements =
            checked::product([a.output_rank(), subcolumns]).ok_or(AkitaError::InvalidProof)?;
        let owner_elements = checked::product([self.profile.blocks.live_blocks, block_elements])
            .ok_or(AkitaError::InvalidProof)?;
        let owner_columns = checked::product([geometry.max_blocks_per_slice(), block_elements])
            .ok_or(AkitaError::InvalidProof)?;
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
                for owner in 0..owners {
                    for element in 0..columns {
                        let lane = checked::mul_add(owner, owner_elements, element)
                            .ok_or(AkitaError::InvalidProof)?;
                        for digit in 0..digits {
                            let column = checked::mul_add(owner, owner_columns, element)
                                .and_then(|column| checked::mul_add(column, digits, digit))
                                .ok_or(AkitaError::InvalidProof)?;
                            kernels
                                .push((t_lanes.offset(lane, digit)?, b_rings.ring(b_row, column)?));
                        }
                    }
                }
                translated(copies, kernels, Some(alpha))?;
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
            f_lanes.bit_copies(
                first.input_coefficients(),
                image.len(),
                self.compression.field_bits(),
            )?,
            vec![(0, image)],
            None,
        )?;
        // Each compression map reads its residue kernels against its input
        // digits.
        for (map, lanes, batch) in [
            (first, &f_lanes, first_batch),
            (second, &h_lanes, second_batch),
        ] {
            let map_rings = RoleRings::new(matrix, 1, map.input_width(), map.ring_dimension())?;
            let kernels = (0..map.input_width())
                .map(|column| Ok((lanes.offset(column, 0)?, map_rings.ring(0, column)?)))
                .collect::<Result<Vec<_>, AkitaError>>()?;
            translated(vec![(lanes.base(0, 0)?, batch)], kernels, Some(alpha))?;
        }
        let mut output = Vec::new();
        let mut power = -first_batch;
        for _ in 0..first.output_coefficients() {
            output.push(power);
            power *= alpha;
        }
        translated(
            h_lanes.bit_copies(
                second.input_coefficients(),
                output.len(),
                self.compression.field_bits(),
            )?,
            vec![(0, output)],
            None,
        )?;
        // Within one owner, a power of `alpha` per coefficient.
        let coefficient_powers = |scale: E| {
            let mut power = scale;
            (0..a.ring_dimension())
                .map(|_| {
                    let value = power;
                    power *= alpha;
                    value
                })
                .collect::<Vec<_>>()
        };
        if let Some(width) = self.source_width {
            // S + sum_j 2^j (-bit_j) = -2^(width-1) for every source element.
            let bit_lanes = lanes(4)?;
            let mut s_copies = Vec::new();
            let mut bit_copies = Vec::new();
            for (element, &batch) in signed_batches.iter().enumerate() {
                s_copies.push((s_lanes.base(element, 0)?, batch));
                bit_copies.push((bit_lanes.base(element, 0)?, batch));
            }
            let mut s_pattern = Vec::new();
            let mut digit_scale = E::one();
            for digit in 0..source_digits {
                s_pattern.push((s_lanes.offset(0, digit)?, coefficient_powers(digit_scale)));
                digit_scale *= E::from_u64(8);
            }
            let mut bit_pattern = Vec::new();
            let mut bit_scale = E::one();
            for bit in 0..width.bits() as usize {
                bit_pattern.push((bit_lanes.offset(0, bit)?, coefficient_powers(bit_scale)));
                bit_scale += bit_scale;
            }
            translated(s_copies, s_pattern, None)?;
            translated(bit_copies, bit_pattern, None)?;
        }
        let mut claim = second_batch
            * payload
                .iter()
                .rev()
                .fold(E::zero(), |sum, &coefficient| sum * alpha + coefficient);
        if let Some(width) = self.source_width {
            let powers = coefficient_powers(E::one())
                .into_iter()
                .fold(E::zero(), |sum, power| sum + power);
            let offset = E::from_u128(width.offset());
            for &batch in signed_batches {
                claim -= batch * powers * offset;
            }
        }
        NativeRootWeights::merge(
            table_len,
            [
                NativeRootWeights::structured(table_len, Vec::new(), families, claim)?,
                self.evaluation_weights(repacking, point, evaluations, evaluation_batches)?,
            ],
            E::zero(),
        )
    }

    /// Derive ordinary setup in its declared base field before lifting to the
    /// testing field. Sampling extension coefficients would change the matrix.
    fn setup_coefficients<E: Field>(
        &self,
        required: usize,
        setup: &AkitaSetupDescriptor,
    ) -> Result<Vec<E>, AkitaError> {
        fn derive<F: Field + CanonicalEncoding, E: Field>(
            required: usize,
            setup: &AkitaSetupDescriptor,
        ) -> Result<Vec<E>, AkitaError> {
            let matrix = derive_public_matrix_prefix::<F>(required, &setup.setup_seed);
            let source = matrix.as_field_slice();
            let mut coefficients = Vec::new();
            coefficients
                .try_reserve_exact(source.len())
                .map_err(|_| AkitaError::InvalidProof)?;
            coefficients.resize(source.len(), E::zero());
            cfg_chunks_mut!(coefficients, LIFT_CHUNK)
                .zip(cfg_chunks!(source, LIFT_CHUNK))
                .try_for_each(|(target, source)| {
                    for (slot, coefficient) in target.iter_mut().zip(source) {
                        *slot = coefficient
                            .to_u128_checked()
                            .map(E::from_u128)
                            .ok_or(AkitaError::InvalidProof)?;
                    }
                    Ok::<_, AkitaError>(())
                })?;
            Ok(coefficients)
        }
        match self.profile.inner.matrix.sis_modulus_profile() {
            SisModulusProfileId::Q128Offset275 => derive::<Prime128Offset275, E>(required, setup),
            SisModulusProfileId::Q64Offset59 => derive::<Prime64Offset59, E>(required, setup),
            _ => Err(AkitaError::InvalidSetup(
                "unsupported mixed root field".into(),
            )),
        }
    }

    fn maps(&self) -> Result<[CompressionMapPlan; 2], AkitaError> {
        match self.compression.maps() {
            &[first, second] => Ok([first, second]),
            _ => Err(AkitaError::InvalidProof),
        }
    }
}

/// Ring coefficients of every entry of one public matrix role, row-major.
///
/// Entry `(row, column)` is the degree-`degree` ring at flat offset
/// `(row * width + column) * degree` of the ordinary setup prefix. The
/// prefix is lifted once into the testing field, in parallel under `parallel`,
/// and every role borrows it; [`Self::ring`] copies one entry out.
struct RoleRings<'a, E> {
    coefficients: &'a [E],
    width: usize,
    degree: usize,
}

/// Coefficients one parallel task lifts.
const LIFT_CHUNK: usize = 1 << 13;

impl<'a, E: Field> RoleRings<'a, E> {
    fn new(matrix: &'a [E], rows: usize, width: usize, degree: usize) -> Result<Self, AkitaError> {
        let len = checked::product([rows, width, degree]).ok_or(AkitaError::InvalidProof)?;
        let coefficients = matrix.get(..len).ok_or(AkitaError::InvalidProof)?;
        if degree == 0 {
            return Err(AkitaError::InvalidProof);
        }
        Ok(Self {
            coefficients,
            width,
            degree,
        })
    }

    /// A copy of ring `(row, column)`.
    fn ring(&self, row: usize, column: usize) -> Result<Vec<E>, AkitaError> {
        if column >= self.width {
            return Err(AkitaError::InvalidProof);
        }
        checked::mul_add(row, self.width, column)
            .and_then(|entry| checked::range(checked::product([entry, self.degree])?, self.degree))
            .and_then(|range| self.coefficients.get(range))
            .map(<[E]>::to_vec)
            .ok_or(AkitaError::InvalidProof)
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

    /// One copy per bit of the selected negative-binary recomposition.
    ///
    /// Bit `i` of input coefficient `c` sits at flat coefficient
    /// `i * coefficients + c`, with scale `2^i`.
    fn bit_copies<E: Field>(
        &self,
        coefficients: usize,
        used: usize,
        bits: usize,
    ) -> Result<Vec<(usize, E)>, AkitaError> {
        if used > coefficients {
            return Err(AkitaError::InvalidProof);
        }
        let mut scale = E::one();
        (0..bits)
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

    /// Add the weights to block profiles, in parallel by block.
    pub(crate) fn emit(&self, profiles: &mut ProfileBuilder<E>) -> Result<(), AkitaError> {
        let equality = OffsetEqWindow::new(&self.point)?;
        let count = checked::pow2(self.point.len()).ok_or(AkitaError::InvalidProof)?;
        profiles.add_intervals(&[(self.start, count)], |_, first, row| {
            for (coefficient, slot) in (first..).zip(row.iter_mut()) {
                *slot += self.scale * equality.eval(coefficient);
            }
            Ok(())
        })
    }
}

#[cfg(test)]
#[path = "ordinary_root_admission_tests.rs"]
mod admission_tests;

#[cfg(test)]
#[path = "ordinary_root_reference.rs"]
mod reference;

#[path = "ordinary_evaluation.rs"]
mod evaluation;
