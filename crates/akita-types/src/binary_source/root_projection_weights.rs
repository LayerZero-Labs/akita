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
    NativeRootWeights, PaddingWeights, RootEq, RootWeightFamily, SegmentLanes,
};

/// Compile the A identity's coefficient weights. Proving supplies A; online
/// verification omits it and authenticates its contribution through Stage 3.
/// `rho[polynomial * rank + row][coefficient]` batches all seven public
/// projections over the joined residual. This compiles the signed remainder
/// `sum_i <rho_i, rem_Psi(A*z-sum c*T)_i>`; the caller supplies its projection
/// target. The T terms and every padding equation remain in the root oracle.
pub fn binary_direct_a_weights<E: Field>(
    plan: &BinaryRootWitnessPlan,
    matrix: Option<&[Vec<[i128; 648]>]>,
    challenges: &Challenges,
    rho: &[[E; 648]],
    padding_weights: &[E],
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
    if let Some(matrix) = matrix {
        if matrix.len() != plan.rank()
            || matrix.iter().any(|row| row.len() != rows)
            || matrix.iter().flatten().flatten().any(|&c| {
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
    let adjoint = TrinomialAdjoint::<648, MinusTrinomial>::new()?;
    let response_digits = plan
        .repacking()
        .segments()
        .first()
        .ok_or(AkitaError::InvalidProof)?
        .digits;
    let response_terms = if matrix.is_some() {
        checked::product([owners, plan.rank(), rows, 648, response_digits])
            .ok_or(AkitaError::InvalidProof)?
    } else {
        0
    };
    let padding = PaddingWeights::new(plan.repacking(), padding_weights)?;
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(response_terms)
        .map_err(|_| AkitaError::InvalidProof)?;
    if let Some(matrix) = matrix {
        for owner in 0..owners {
            for i in 0..plan.rank() {
                let row_index =
                    checked::mul_add(owner, plan.rank(), i).ok_or(AkitaError::InvalidProof)?;
                let weights = rho.get(row_index).ok_or(AkitaError::InvalidProof)?;
                for (j, a) in matrix
                    .get(i)
                    .ok_or(AkitaError::InvalidProof)?
                    .iter()
                    .enumerate()
                {
                    let element =
                        checked::mul_add(owner, rows, j).ok_or(AkitaError::InvalidProof)?;
                    for (t, w) in adjoint
                        .apply(weights, &a.map(E::from_i128))?
                        .into_iter()
                        .enumerate()
                    {
                        terms.extend(plan.repacking().coefficient_weights(0, element, t, w)?);
                    }
                }
            }
        }
    }
    let image = ImageWeights::new(plan, rho, &embedded)?;
    NativeRootWeights::structured(
        plan.repacking().domain_len(),
        terms,
        vec![
            RootWeightFamily::Image(image),
            RootWeightFamily::Padding(padding),
        ],
        E::zero(),
    )
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

    /// The challenge supports of one A row: its owner's columns in order,
    /// each with the image element it multiplies.
    fn row_columns(
        &self,
        row_index: usize,
    ) -> Result<impl Iterator<Item = (usize, &ChallengeSupport<E>)>, AkitaError> {
        let owner = row_index / self.rank;
        let row = row_index % self.rank;
        let first = checked::product([owner, self.columns]).ok_or(AkitaError::InvalidProof)?;
        let supports = checked::range(first, self.columns)
            .and_then(|range| self.challenges.get(range))
            .ok_or(AkitaError::InvalidProof)?;
        let rank = self.rank;
        Ok(supports
            .iter()
            .enumerate()
            .map(move |(column, support)| ((first + column) * rank + row, support)))
    }

    pub(crate) fn evaluate(&self, eq: &RootEq<E>) -> Result<E, AkitaError> {
        let adjoint = TrinomialAdjoint::<648, MinusTrinomial>::new()?;
        let low = eq.low().get(..648).ok_or(AkitaError::InvalidProof)?;
        let mut sum = E::zero();
        for (row_index, rho) in self.rho.iter().enumerate() {
            let adjoint_low = adjoint.apply(rho, low)?;
            for (element, support) in self.row_columns(row_index)? {
                let mut dot = E::zero();
                for &(k, value) in support {
                    dot += value * *adjoint_low.get(k).ok_or(AkitaError::InvalidProof)?;
                }
                sum += self.lanes.block_digit_weight(eq, element)? * dot;
            }
        }
        Ok(-sum)
    }

    pub(crate) fn visit(
        &self,
        mut visit: impl FnMut(usize, E) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        let adjoint = TrinomialAdjoint::<648, MinusTrinomial>::new()?;
        let mut h = Vec::new();
        h.try_reserve_exact(648)
            .map_err(|_| AkitaError::InvalidProof)?;
        h.resize(648, E::zero());
        for (row_index, rho) in self.rho.iter().enumerate() {
            let extended = adjoint.extend_row(rho)?;
            for (element, support) in self.row_columns(row_index)? {
                h.fill(E::zero());
                for &(k, value) in support {
                    let shifted = extended.get(k..k + 648).ok_or(AkitaError::InvalidProof)?;
                    for (slot, &g) in h.iter_mut().zip(shifted) {
                        *slot += value * g;
                    }
                }
                let mut power = -E::one();
                for digit in 0..self.lanes.digits() {
                    let base = self.lanes.base(element, digit)?;
                    for (t, &weight) in h.iter().enumerate() {
                        visit(base + t, weight * power)?;
                    }
                    power *= E::from_u64(8);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "root_projection_weights_tests.rs"]
mod tests;
