//! Projected A relation on canonical digit addresses, with an explicit rho.

use akita_algebra::ring::trinomial::adjoint::TrinomialAdjoint;
use akita_algebra::ring::trinomial::MinusTrinomial;
use akita_error::{checked, AkitaError};
use jolt_field::{CanonicalEncoding, ExtField, Field};

use crate::binary_source::{
    BinaryRootWitnessPlan, BinaryScalarChallengeSource, PreparedBinaryScalarChallenges,
    BINARY_SOURCE_Q, CANONICAL_IMAGE_WINDOW,
};
use crate::native_root_relation::{
    weigh_units, NativeRootWeights, ProfileBuilder, RootEq, RootWeightFamily, SegmentLanes,
    TranslatedLanes, WEIGH_BATCH,
};

/// The prover's A operand for [`binary_direct_a_weights`].
///
/// `matrix[row][column]` holds the centered q setup. The base field is
/// captured at construction: each extension coordinate passes independently
/// through the exact integer adjoint, then the results recombine in its basis.
#[derive(Clone, Copy)]
pub struct BinaryAOperand<'a, E> {
    pub matrix: &'a [Vec<[i128; 648]>],
    degree: usize,
    #[allow(clippy::type_complexity)]
    adjoint: fn(&[[E; 648]], &[Vec<[i128; 648]>]) -> Result<Vec<[E; 648]>, AkitaError>,
}

impl<'a, E: Field> BinaryAOperand<'a, E> {
    pub fn new<F: Field + CanonicalEncoding>(matrix: &'a [Vec<[i128; 648]>]) -> Self
    where
        E: ExtField<F>,
    {
        Self {
            matrix,
            degree: E::DEGREE,
            adjoint: coordinate_adjoint::<F, E>,
        }
    }
}

fn coordinate_adjoint<F: Field + CanonicalEncoding, E: ExtField<F>>(
    rho: &[[E; 648]],
    matrix: &[Vec<[i128; 648]>],
) -> Result<Vec<[E; 648]>, AkitaError> {
    let adjoint = TrinomialAdjoint::<648, MinusTrinomial>::new()?;
    let mut coordinates = Vec::new();
    for coordinate in 0..E::DEGREE {
        let base = rho
            .iter()
            .map(|row| row.map(|value| value.base_coefficient(coordinate)))
            .collect::<Vec<_>>();
        // Each pass retains the kernel's 43-bit limb CRT check:
        // 2*3*3*648*B_A*2^43 < 67,091,329*134,217,649*268,433,353.
        // Canonical p64 representatives occupy at most two limbs; the two
        // extension coordinates are never joined into an integer.
        coordinates.push(adjoint.apply_integer_rows(
            &base,
            matrix,
            u128::from(CANONICAL_IMAGE_WINDOW.1.unsigned_abs()),
            F::to_u128_checked,
        )?);
    }
    let columns = matrix.first().map_or(0, Vec::len);
    Ok((0..columns)
        .map(|column| std::array::from_fn(|k| E::from_base_fn(|i| coordinates[i][column][k])))
        .collect())
}

/// Compile the A identity's coefficient weights. Proving supplies A; online
/// verification omits it and authenticates its contribution through Stage 3.
/// `rho[polynomial * rank + row][coefficient]` batches all seven public
/// projections over the joined P128 residual, or are independent coefficient
/// challenges for P64. This compiles the signed remainder
/// `sum_i <rho_i, rem_Psi(A*z-sum c*T)_i>`; the caller supplies its projection
/// target for P128. P64 also subtracts q times K, with target zero and no
/// outer batching scalar. The T terms remain in the root oracle. No term reads a padding
/// address.
pub fn binary_direct_a_weights<E: Field>(
    plan: &BinaryRootWitnessPlan,
    matrix: Option<BinaryAOperand<'_, E>>,
    challenges: &impl BinaryScalarChallengeSource,
    rho: &[[E; 648]],
) -> Result<NativeRootWeights<E>, AkitaError> {
    let descriptor = plan.descriptor();
    let columns = descriptor.index_map().columns();
    let rows = checked::product([4, columns])
        .and_then(|d| checked::exact_div(descriptor.index_map().padded_carriers(), d))
        .ok_or(AkitaError::InvalidProof)?;
    let owners = descriptor.polynomials().len();
    if rho.len() != checked::product([owners, plan.rank()]).ok_or(AkitaError::InvalidProof)? {
        return Err(AkitaError::InvalidInput(
            "A coefficient functional has the wrong rank".into(),
        ));
    }
    if let Some(operand) = &matrix {
        if operand.matrix.len() != plan.rank()
            || operand.matrix.iter().any(|row| row.len() != rows)
            || operand.matrix.iter().flatten().flatten().any(|&c| {
                c < i128::from(CANONICAL_IMAGE_WINDOW.0) || c > i128::from(CANONICAL_IMAGE_WINDOW.1)
            })
        {
            return Err(AkitaError::InvalidInput(
                "A differs from its centered typed geometry".into(),
            ));
        }
    }
    let scalars = challenges.prepare(descriptor)?;
    let mut families = Vec::new();
    if let Some(operand) = matrix {
        let _span = tracing::info_span!("binary_root_weights_direct_a").entered();
        families.push(RootWeightFamily::DirectA(DirectAWeights::new(
            plan, operand, rho,
        )?));
    }
    families.push(RootWeightFamily::Image(ImageWeights::new(
        plan, rho, &scalars,
    )?));
    if let Some(segment) = plan.quotient_segment() {
        let depth = plan
            .projection_bounds()
            .quotient()
            .ok_or(AkitaError::InvalidProof)?
            .digits;
        let start = plan.repacking().segment_range(segment)?.start;
        let mut power = E::one();
        let copies = (0..depth)
            .map(|digit| {
                let base = checked::mul_add(digit, 1024, start).ok_or(AkitaError::InvalidProof)?;
                let scale = power;
                power *= E::from_u64(8);
                Ok((base, scale))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let lanes = rho
            .iter()
            .enumerate()
            .map(|(element, row)| {
                let offset =
                    checked::product([element, depth, 1024]).ok_or(AkitaError::InvalidProof)?;
                let weights = row
                    .iter()
                    .map(|&weight| -E::from_u64(u64::from(BINARY_SOURCE_Q)) * weight)
                    .collect();
                Ok((offset, weights))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        families.push(RootWeightFamily::Translated(TranslatedLanes::new(
            plan.repacking().domain_len(),
            copies,
            lanes,
            None,
        )?));
    }
    NativeRootWeights::structured(
        plan.repacking().domain_len(),
        Vec::new(),
        families,
        E::zero(),
    )
}

/// The response part `sum_ij <rho_i, rem(A_ij * z_j)>` of the A identity.
///
/// Response element `owner * rows + j` keeps `h = sum_i adj(rho_i, A_ij)`,
/// so its digit `d` at coefficient `t` has weight `8^d * h[t]`. Only the
/// prover builds this family; verification never reads A.
#[derive(Clone, Debug)]
pub(crate) struct DirectAWeights<E: Field> {
    lanes: SegmentLanes,
    table_len: usize,
    columns: Vec<[E; 648]>,
}

impl<E: Field> DirectAWeights<E> {
    fn new(
        plan: &BinaryRootWitnessPlan,
        operand: BinaryAOperand<'_, E>,
        rho: &[[E; 648]],
    ) -> Result<Self, AkitaError> {
        let lanes = SegmentLanes::new(plan.repacking(), 0)?;
        let rank = plan.rank();
        if rank == 0 {
            return Err(AkitaError::InvalidSetup("binary A has no rows".into()));
        }
        if operand.degree != plan.field_profile().extension_degree() {
            return Err(AkitaError::InvalidInput(
                "A adjoint field degree mismatch".into(),
            ));
        }
        let mut columns = Vec::new();
        columns
            .try_reserve_exact(lanes.elements())
            .map_err(|_| AkitaError::InvalidProof)?;
        for owner_rho in rho.chunks_exact(rank) {
            columns.extend((operand.adjoint)(owner_rho, operand.matrix)?);
        }
        if lanes.degree() != 648 || columns.len() != lanes.elements() {
            return Err(AkitaError::InvalidSetup(
                "binary response lanes differ from the A geometry".into(),
            ));
        }
        Ok(Self {
            lanes,
            table_len: plan.repacking().domain_len(),
            columns,
        })
    }

    pub(crate) fn table_len(&self) -> usize {
        self.table_len
    }

    pub(crate) fn evaluate(&self, eq: &RootEq<E>) -> Result<E, AkitaError> {
        let low = eq.low().get(..648).ok_or(AkitaError::InvalidProof)?;
        let mut sum = E::zero();
        for (element, h) in self.columns.iter().enumerate() {
            let dot = h
                .iter()
                .zip(low)
                .fold(E::zero(), |dot, (&weight, &eq)| dot + weight * eq);
            sum += self.lanes.block_digit_weight(eq, element)? * dot;
        }
        Ok(sum)
    }

    /// Add each element's profile `h` to block profiles in parallel.
    pub(crate) fn emit(&self, profiles: &mut ProfileBuilder<E>) -> Result<(), AkitaError> {
        profiles.add_shared(&self.lanes, |element, profile| {
            let h = self.columns.get(element).ok_or(AkitaError::InvalidProof)?;
            for (slot, &weight) in profile.iter_mut().zip(h) {
                *slot += weight;
            }
            Ok(())
        })
    }

    /// Stream each element's profile `h[t]`; digit `d` weighs `8^d * h[t]`.
    pub(crate) fn profiles(
        &self,
        mut profile: impl FnMut(&SegmentLanes, usize, usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        for (element, h) in self.columns.iter().enumerate() {
            for (t, &weight) in h.iter().enumerate() {
                profile(&self.lanes, element, t, weight)?;
            }
        }
        Ok(())
    }
}

/// The image part `-sum_j <rho_i, rem(c_j * T_ji)>` of the A identity.
///
/// It keeps rho and the sparse embedded challenges instead of one term per
/// image digit. By the Hankel symmetry of the adjoint,
/// `<adj(rho, c), z> = <c, adj(rho, z)>`. Every image lane starts an equality
/// block, so its digits share the low factors `z = low[..648]`. The verifier
/// therefore applies one adjoint per A row and then reads each column's
/// challenge support. The prover expands each column by the same support,
/// as signed shifts of the extended row.
#[derive(Clone, Debug)]
pub(crate) struct ImageWeights<E: Field> {
    lanes: SegmentLanes,
    table_len: usize,
    rank: usize,
    columns: usize,
    /// `rho[owner * rank + row]`.
    rho: Vec<[E; 648]>,
    /// Scalar supports for `owner * columns + column`, embedded on use.
    challenges: PreparedBinaryScalarChallenges,
}

impl<E: Field> ImageWeights<E> {
    /// `scalars` are the scalar challenges of every owner and column. Each
    /// embeds into R as the first signed component
    /// (`pack_signed_components`): scalar coefficient `s` lands on
    /// coefficient `4s`, negated when `s` is odd.
    fn new(
        plan: &BinaryRootWitnessPlan,
        rho: &[[E; 648]],
        scalars: &PreparedBinaryScalarChallenges,
    ) -> Result<Self, AkitaError> {
        let columns = plan.descriptor().index_map().columns();
        let lanes = SegmentLanes::new(plan.repacking(), 1)?;
        let elements = checked::product([scalars.len(), plan.rank()]);
        if lanes.degree() != 648
            || elements != Some(lanes.elements())
            || checked::product([plan.descriptor().polynomials().len(), columns])
                != Some(scalars.len())
        {
            return Err(AkitaError::InvalidSetup(
                "binary image lanes differ from the challenge geometry".into(),
            ));
        }
        Ok(Self {
            lanes,
            table_len: plan.repacking().domain_len(),
            rank: plan.rank(),
            columns,
            rho: rho.to_vec(),
            challenges: scalars.clone(),
        })
    }

    pub(crate) fn table_len(&self) -> usize {
        self.table_len
    }

    /// The image element that A row `row_index` meets at `column` of its
    /// owner, with that column's challenge support.
    fn row_column(
        &self,
        row_index: usize,
        column: usize,
    ) -> Result<(usize, &[(usize, bool)]), AkitaError> {
        let owner = row_index / self.rank;
        let row = row_index % self.rank;
        let index =
            checked::mul_add(owner, self.columns, column).ok_or(AkitaError::InvalidProof)?;
        let support = self.challenges.support(index)?;
        let element = checked::mul_add(index, self.rank, row).ok_or(AkitaError::InvalidProof)?;
        Ok((element, support))
    }

    pub(crate) fn evaluate(&self, eq: &RootEq<E>) -> Result<E, AkitaError> {
        let adjoint = TrinomialAdjoint::<648, MinusTrinomial>::new()?;
        // `low[..648]` is the equality table of the first ten coordinates.
        let low_point = eq.point().get(..10).ok_or(AkitaError::InvalidProof)?;
        let mut sum = E::zero();
        for (row_index, rho) in self.rho.iter().enumerate() {
            let adjoint_low = adjoint.apply_equality(rho, low_point)?;
            for column in 0..self.columns {
                let (element, support) = self.row_column(row_index, column)?;
                let mut dot = E::zero();
                for &(s, negative) in support {
                    let negative = negative != (s % 2 == 1);
                    let value = *adjoint_low.get(4 * s).ok_or(AkitaError::InvalidProof)?;
                    if negative {
                        dot -= value;
                    } else {
                        dot += value;
                    }
                }
                sum += self.lanes.block_digit_weight(eq, element)? * dot;
            }
        }
        Ok(-sum)
    }

    /// The extended rows `adjoint.extend_row(rho)` of every A row.
    fn extended_rows(&self) -> Result<Vec<Vec<E>>, AkitaError> {
        let adjoint = TrinomialAdjoint::<648, MinusTrinomial>::new()?;
        self.rho.iter().map(|rho| adjoint.extend_row(rho)).collect()
    }

    /// The image element of one (A row, column) pair and its profile `h`,
    /// the pair's support applied as signed shifts of the extended row.
    fn expand(
        &self,
        extended: &[Vec<E>],
        row_index: usize,
        column: usize,
    ) -> Result<(usize, [E; 648]), AkitaError> {
        let (element, support) = self.row_column(row_index, column)?;
        let extended = extended.get(row_index).ok_or(AkitaError::InvalidProof)?;
        let mut h = [E::zero(); 648];
        for &(s, negative) in support {
            let negative = negative != (s % 2 == 1);
            let k = 4 * s;
            let shifted = extended.get(k..k + 648).ok_or(AkitaError::InvalidProof)?;
            if negative {
                for (slot, &g) in h.iter_mut().zip(shifted) {
                    *slot -= g;
                }
            } else {
                for (slot, &g) in h.iter_mut().zip(shifted) {
                    *slot += g;
                }
            }
        }
        Ok((element, h))
    }

    /// Add each element's profile `-h` to block profiles in parallel. An
    /// element `(owner * columns + column) * rank + row` meets A row
    /// `owner * rank + row` at `column`.
    pub(crate) fn emit(&self, profiles: &mut ProfileBuilder<E>) -> Result<(), AkitaError> {
        let extended = self.extended_rows()?;
        profiles.add_shared(&self.lanes, |element, profile| {
            let index = element / self.rank;
            let row_index = checked::mul_add(index / self.columns, self.rank, element % self.rank)
                .ok_or(AkitaError::InvalidProof)?;
            let (_, h) = self.expand(&extended, row_index, index % self.columns)?;
            for (slot, &weight) in profile.iter_mut().zip(&h) {
                *slot += -weight;
            }
            Ok(())
        })
    }

    /// Expand each (A row, column) pair by its support, in batches through
    /// [`weigh_units`], and stream the element profiles `-h[t]`; digit `d`
    /// weighs `-8^d * h[t]`.
    pub(crate) fn profiles(
        &self,
        mut profile: impl FnMut(&SegmentLanes, usize, usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        let extended = self.extended_rows()?;
        let units =
            checked::product([self.rho.len(), self.columns]).ok_or(AkitaError::InvalidProof)?;
        for start in (0..units).step_by(WEIGH_BATCH) {
            let end = units.min(start + WEIGH_BATCH);
            let weigh =
                |unit: usize| self.expand(&extended, unit / self.columns, unit % self.columns);
            for (element, h) in weigh_units(start..end, weigh)? {
                for (t, &weight) in h.iter().enumerate() {
                    profile(&self.lanes, element, t, -weight)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "root_projection_weights_tests.rs"]
mod tests;
