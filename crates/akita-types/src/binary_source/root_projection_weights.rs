//! Projected A relation on canonical digit addresses, with an explicit rho.

use akita_algebra::ring::trinomial::adjoint::TrinomialAdjoint;
use akita_algebra::ring::trinomial::module_map::pack_signed_components;
use akita_algebra::ring::trinomial::MinusTrinomial;
use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use jolt_field::Field;

use crate::binary_source::{
    binary_scalar_challenges, BinaryRootWitnessPlan, CANONICAL_IMAGE_WINDOW,
};
use crate::native_root_relation::{
    weigh_units, NativeRootWeights, ProfileBuilder, RootEq, RootWeightFamily, SegmentLanes,
    WEIGH_BATCH,
};

/// The prover's A operand for [`binary_direct_a_weights`].
///
/// `matrix[row][column]` holds the centered q setup. `canonical` returns the
/// canonical integer of a proof-field element; the exact integer adjoint
/// needs it because the proof field has no smooth transform domain.
#[derive(Clone, Copy)]
pub struct BinaryAOperand<'a, E> {
    pub matrix: &'a [Vec<[i128; 648]>],
    pub canonical: fn(&E) -> Option<u128>,
}

/// Compile the A identity's coefficient weights. Proving supplies A; online
/// verification omits it and authenticates its contribution through Stage 3.
/// `rho[polynomial * rank + row][coefficient]` batches all seven public
/// projections over the joined residual. This compiles the signed remainder
/// `sum_i <rho_i, rem_Psi(A*z-sum c*T)_i>`; the caller supplies its projection
/// target. The T terms remain in the root oracle. No term reads a padding
/// address.
pub fn binary_direct_a_weights<E: Field>(
    plan: &BinaryRootWitnessPlan,
    matrix: Option<BinaryAOperand<'_, E>>,
    challenges: &Challenges,
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
    let embedded = binary_scalar_challenges(descriptor, challenges)?
        .into_iter()
        .map(|scalar| {
            Ok(pack_signed_components(&[scalar, [0; 162], [0; 162], [0; 162]])?.map(E::from_i128))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let mut families = Vec::new();
    if let Some(operand) = matrix {
        let _span = tracing::info_span!("binary_root_weights_direct_a").entered();
        families.push(RootWeightFamily::DirectA(DirectAWeights::new(
            plan, operand, rho,
        )?));
    }
    families.push(RootWeightFamily::Image(ImageWeights::new(
        plan, rho, &embedded,
    )?));
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

#[cfg(test)]
impl<'a, E: Field + jolt_field::CanonicalEncoding> BinaryAOperand<'a, E> {
    /// Test operand over a field whose canonical integers fit 128 bits.
    pub(crate) fn from_canonical(matrix: &'a [Vec<[i128; 648]>]) -> Self {
        Self {
            matrix,
            canonical: |x| x.to_u128_checked(),
        }
    }
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
        let adjoint = TrinomialAdjoint::<648, MinusTrinomial>::new()?;
        let mut columns = Vec::new();
        columns
            .try_reserve_exact(lanes.elements())
            .map_err(|_| AkitaError::InvalidProof)?;
        for owner_rho in rho.chunks_exact(rank) {
            columns.extend(adjoint.apply_integer_rows(
                owner_rho,
                operand.matrix,
                u128::from(CANONICAL_IMAGE_WINDOW.1.unsigned_abs()),
                operand.canonical,
            )?);
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

/// Nonzero `(coefficient, value)` entries of one embedded challenge.
type ChallengeSupport<E> = Vec<(usize, E)>;

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
    /// Nonzero embedded coefficients for `owner * columns + column`.
    challenges: Vec<ChallengeSupport<E>>,
}

impl<E: Field> ImageWeights<E> {
    fn new(
        plan: &BinaryRootWitnessPlan,
        rho: &[[E; 648]],
        embedded: &[[E; 648]],
    ) -> Result<Self, AkitaError> {
        let columns = plan.descriptor().index_map().columns();
        let lanes = SegmentLanes::new(plan.repacking(), 1)?;
        let elements = checked::product([embedded.len(), plan.rank()]);
        if lanes.degree() != 648
            || elements != Some(lanes.elements())
            || checked::product([plan.descriptor().polynomials().len(), columns])
                != Some(embedded.len())
        {
            return Err(AkitaError::InvalidSetup(
                "binary image lanes differ from the challenge geometry".into(),
            ));
        }
        let mut challenges = Vec::new();
        challenges
            .try_reserve_exact(embedded.len())
            .map_err(|_| AkitaError::InvalidProof)?;
        for challenge in embedded {
            challenges.push(
                challenge
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| !value.is_zero())
                    .map(|(k, &value)| (k, value))
                    .collect(),
            );
        }
        Ok(Self {
            lanes,
            table_len: plan.repacking().domain_len(),
            rank: plan.rank(),
            columns,
            rho: rho.to_vec(),
            challenges,
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
    ) -> Result<(usize, &ChallengeSupport<E>), AkitaError> {
        let owner = row_index / self.rank;
        let row = row_index % self.rank;
        let index =
            checked::mul_add(owner, self.columns, column).ok_or(AkitaError::InvalidProof)?;
        let support = self.challenges.get(index).ok_or(AkitaError::InvalidProof)?;
        let element = checked::mul_add(index, self.rank, row).ok_or(AkitaError::InvalidProof)?;
        Ok((element, support))
    }

    pub(crate) fn evaluate(&self, eq: &RootEq<E>) -> Result<E, AkitaError> {
        let adjoint = TrinomialAdjoint::<648, MinusTrinomial>::new()?;
        let low = eq.low().get(..648).ok_or(AkitaError::InvalidProof)?;
        let mut sum = E::zero();
        for (row_index, rho) in self.rho.iter().enumerate() {
            let adjoint_low = adjoint.apply(rho, low)?;
            for column in 0..self.columns {
                let (element, support) = self.row_column(row_index, column)?;
                let mut dot = E::zero();
                for &(k, value) in support {
                    dot += value * *adjoint_low.get(k).ok_or(AkitaError::InvalidProof)?;
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
        for &(k, value) in support {
            let shifted = extended.get(k..k + 648).ok_or(AkitaError::InvalidProof)?;
            for (slot, &g) in h.iter_mut().zip(shifted) {
                *slot += value * g;
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
