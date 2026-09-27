//! Stage-1 digit-range topology and its headerless proof shapes.

use crate::wire_limits::{checked_shape_len, checked_shape_sequence_len};
use crate::InnerCommitSecurityRoute;
use akita_error::AkitaError;
use akita_serialization::{SerializationError, Valid};
use akita_sumcheck::{EqFactoredSumcheckProofShape, SumcheckProofShape};
use jolt_field::{Field, Ring};

/// Checked flat Boolean domain for the compact digit witness.
///
/// The first `live_len` addresses contain digits and the remaining addresses
/// up to `domain_len()` are public zero padding. Variables bind in increasing
/// physical-address bit order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FlatBooleanDomain {
    live_len: usize,
    num_vars: usize,
}

impl FlatBooleanDomain {
    /// Construct a checked live prefix inside a Boolean hypercube.
    ///
    /// # Errors
    ///
    /// Returns an error if the domain width overflows, the live prefix is
    /// empty, or the live prefix exceeds the Boolean domain.
    pub fn new(live_len: usize, num_vars: usize) -> Result<Self, AkitaError> {
        let shift = u32::try_from(num_vars)
            .map_err(|_| AkitaError::InvalidInput("Boolean domain width overflow".to_string()))?;
        let domain_len = 1usize
            .checked_shl(shift)
            .ok_or_else(|| AkitaError::InvalidInput("Boolean domain width overflow".to_string()))?;
        if live_len == 0 || live_len > domain_len {
            return Err(AkitaError::InvalidSize {
                expected: domain_len,
                actual: live_len,
            });
        }
        Ok(Self { live_len, num_vars })
    }

    /// Number of explicit witness entries before zero padding.
    #[must_use]
    pub fn live_len(self) -> usize {
        self.live_len
    }

    /// Total number of Boolean variables.
    #[must_use]
    pub fn num_vars(self) -> usize {
        self.num_vars
    }

    /// Padded Boolean-domain length.
    #[must_use]
    pub fn domain_len(self) -> usize {
        1usize << self.num_vars
    }

    /// Number of live blocks after grouping by `block_variable_count` low bits.
    ///
    /// # Errors
    ///
    /// Returns an error if the block width is larger than the domain or the
    /// live prefix is not block-aligned.
    pub fn live_block_count(self, block_variable_count: usize) -> Result<usize, AkitaError> {
        if block_variable_count > self.num_vars {
            return Err(AkitaError::InvalidSize {
                expected: self.num_vars,
                actual: block_variable_count,
            });
        }
        let block_len = 1usize << block_variable_count;
        if !self.live_len.is_multiple_of(block_len) {
            return Err(AkitaError::InvalidInput(format!(
                "live digit prefix {} is not aligned to block length {block_len}",
                self.live_len
            )));
        }
        Ok(self.live_len / block_len)
    }
}

/// Headerless shape context for one stage in the stage-1 range-check tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AkitaStage1StageShape {
    /// Eq-factored sumcheck shape `(num_rounds, q_degree)`.
    pub sumcheck_proof: EqFactoredSumcheckProofShape,
    /// Number of child claims serialized after the stage proof.
    pub child_claims: usize,
}

/// Public layout of the native physical-L2 messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalL2NormProofWireShape {
    /// Number of blockwise limb claims; zero for direct mode.
    pub subclaims: usize,
    /// Number of final response/limb virtual evaluations.
    pub virtual_evaluations: usize,
    /// General final-leaf sumcheck shape.
    pub sumcheck: SumcheckProofShape,
}

impl Valid for AkitaStage1StageShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_len(self.sumcheck_proof.0)?;
        checked_shape_len(self.sumcheck_proof.1)?;
        checked_shape_len(self.child_claims)?;
        Ok(())
    }
}

impl Valid for PhysicalL2NormProofWireShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_len(self.subclaims)?;
        checked_shape_len(self.virtual_evaluations)?;
        if self.virtual_evaluations == 0 {
            return Err(SerializationError::InvalidData(
                "L2 norm proof shape requires a virtual evaluation".into(),
            ));
        }
        checked_shape_sequence_len(self.sumcheck.len())?;
        for &degree in &self.sumcheck {
            checked_shape_len(degree)?;
        }
        Ok(())
    }
}

/// Checked topology for the balanced digit range proof.
///
/// This is the single authority for the supported basis, product-stage
/// arities, leaf factorization, proof shape, and child-claim order. It is
/// constructed from the concrete range basis already selected by the
/// ring-switch boundary and has no dependency on level parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DigitRangePlan {
    log_basis: u8,
}

impl DigitRangePlan {
    /// Construct the canonical range topology for a supported concrete basis.
    ///
    /// # Errors
    ///
    /// Returns an error unless `basis` is one of `4, 8, 16, 32, 64`.
    pub fn new(basis: usize) -> Result<Self, AkitaError> {
        if !matches!(basis, 4 | 8 | 16 | 32 | 64) {
            return Err(AkitaError::InvalidInput(format!(
                "digit range basis must be one of 4, 8, 16, 32, 64; got {basis}"
            )));
        }
        Ok(Self {
            log_basis: basis.trailing_zeros() as u8,
        })
    }

    /// Concrete balanced decomposition basis.
    #[must_use]
    pub fn basis(self) -> usize {
        1usize << self.log_basis
    }

    /// Base-2 logarithm of [`Self::basis`].
    #[must_use]
    pub fn log_basis(self) -> u8 {
        self.log_basis
    }

    /// Product-stage arities in transcript order, before the leaf stage.
    #[must_use]
    pub fn product_stage_arities(&self) -> &'static [usize] {
        match self.log_basis {
            2 | 3 => &[],
            4 => &[2],
            5 => &[4],
            6 => &[2, 4],
            _ => unreachable!("DigitRangePlan construction validates log basis"),
        }
    }

    /// Number of child lanes emitted by one product substage.
    #[must_use]
    pub fn product_stage_lane_count(self, stage_index: usize) -> Option<usize> {
        self.stage_shapes_iter(0)
            .take(self.product_stage_arities().len())
            .nth(stage_index)
            .map(|shape| shape.child_claims)
    }

    /// Number of quartic (or smaller for basis four) leaf factors.
    #[must_use]
    pub fn leaf_factor_count(self) -> usize {
        match self.log_basis {
            2 | 3 => 1,
            4 => 2,
            5 => 4,
            6 => 8,
            _ => unreachable!("DigitRangePlan construction validates log basis"),
        }
    }

    /// Degree of the final range leaf.
    #[must_use]
    pub fn leaf_degree(self) -> usize {
        if self.log_basis == 2 {
            2
        } else {
            4
        }
    }

    /// Number of range subproofs in transcript order.
    #[must_use]
    pub fn stage_count(self) -> usize {
        self.product_stage_arities().len() + 1
    }

    /// The supported bases have at most two product stages and eight lanes.
    /// Every yielded stage is therefore complete, with no per-index failure.
    fn stage_shapes_iter(self, rounds: usize) -> impl Iterator<Item = AkitaStage1StageShape> {
        let mut child_claims = 1;
        let products = self
            .product_stage_arities()
            .iter()
            .copied()
            .map(move |arity| {
                child_claims *= arity;
                AkitaStage1StageShape {
                    sumcheck_proof: (rounds, arity),
                    child_claims,
                }
            });
        products.chain(std::iter::once(AkitaStage1StageShape {
            sumcheck_proof: (rounds, self.leaf_degree()),
            child_claims: 0,
        }))
    }

    /// Wire shape of one range subproof in transcript order.
    /// Returns `None` only when `stage_index` is outside this plan.
    #[must_use]
    pub fn stage_shape(self, rounds: usize, stage_index: usize) -> Option<AkitaStage1StageShape> {
        self.stage_shapes_iter(rounds).nth(stage_index)
    }

    /// Wire shapes of all range subproofs in transcript order.
    #[must_use]
    pub fn stage_shapes(self, rounds: usize) -> Vec<AkitaStage1StageShape> {
        self.stage_shapes_iter(rounds).collect()
    }

    /// Derive the headerless Stage 1 wire shape from the scheduled A route.
    pub fn proof_shapes_for_route(
        self,
        rounds: usize,
        route: InnerCommitSecurityRoute,
    ) -> Result<
        (
            Vec<AkitaStage1StageShape>,
            Option<PhysicalL2NormProofWireShape>,
        ),
        AkitaError,
    > {
        let shape = self.route_shape(rounds, route)?;
        Ok((
            shape.stages().collect(),
            shape.norm.map(|norm| PhysicalL2NormProofWireShape {
                subclaims: norm.subclaims,
                virtual_evaluations: norm.virtual_evaluations,
                sumcheck: vec![norm.degree; norm.rounds],
            }),
        ))
    }

    /// Return the selected Stage 1 product stages and optional L2 norm shape.
    ///
    /// Returns an error if the selected L2 norm geometry is invalid.
    pub(crate) fn route_shape(
        self,
        rounds: usize,
        route: InnerCommitSecurityRoute,
    ) -> Result<DigitRangeRouteShape, AkitaError> {
        let norm = match route {
            InnerCommitSecurityRoute::Linf(_) => None,
            InnerCommitSecurityRoute::L2 {
                norm_proof_shape, ..
            } => {
                norm_proof_shape.validate()?;
                Some(PhysicalL2NormShape {
                    subclaims: norm_proof_shape.subclaim_count().ok_or_else(|| {
                        AkitaError::InvalidSetup("L2 norm subclaim count overflow".into())
                    })?,
                    virtual_evaluations: norm_proof_shape.virtual_evaluation_count(),
                    rounds,
                    degree: self.leaf_degree() + 1,
                })
            }
        };
        Ok(DigitRangeRouteShape {
            plan: self,
            rounds,
            norm,
        })
    }

    /// Coefficients of the final range-leaf polynomials.
    pub fn leaf_coeffs<E: Field + Ring>(self) -> Vec<Vec<E>> {
        stage1_root_values::<E>(self.basis())
            .chunks(4)
            .map(poly_coeffs_from_roots)
            .collect()
    }

    /// Evaluate one leaf polynomial at a range-image value.
    pub fn evaluate_leaf_polynomial<E: Field>(self, coeffs: &[E], range_image: E) -> E {
        coeffs
            .iter()
            .rev()
            .copied()
            .fold(E::zero(), |acc, coeff| acc * range_image + coeff)
    }

    /// Return powers of the interstage batching challenge.
    pub fn interstage_batch_weights<E: Field>(self, gamma: E, count: usize) -> Vec<E> {
        let mut weights = Vec::with_capacity(count);
        let mut weight = E::one();
        for _ in 0..count {
            weights.push(weight);
            weight *= gamma;
        }
        weights
    }

    /// Batch child claims using the current interstage weights.
    pub fn batch_claims<E: Field>(self, weights: &[E], claims: &[E]) -> Result<E, AkitaError> {
        if weights.len() != claims.len() {
            return Err(AkitaError::InvalidSize {
                expected: weights.len(),
                actual: claims.len(),
            });
        }
        Ok(weights
            .iter()
            .zip(claims.iter())
            .fold(E::zero(), |acc, (&weight, &claim)| acc + weight * claim))
    }

    /// Batch leaf-polynomial coefficient vectors using interstage weights.
    pub fn batch_leaf_polynomials<E: Field>(
        self,
        weights: &[E],
        leaf_polynomials: &[Vec<E>],
    ) -> Result<Vec<E>, AkitaError> {
        if weights.len() != leaf_polynomials.len() {
            return Err(AkitaError::InvalidSize {
                expected: weights.len(),
                actual: leaf_polynomials.len(),
            });
        }
        let max_len = leaf_polynomials.iter().map(Vec::len).max().unwrap_or(0);
        let mut batched = vec![E::zero(); max_len];
        for (weight, polynomial) in weights.iter().zip(leaf_polynomials.iter()) {
            for (coefficient, &term) in batched.iter_mut().zip(polynomial.iter()) {
                *coefficient += *weight * term;
            }
        }
        Ok(batched)
    }
}

/// Stage 1 shape for a selected A route, retaining only the optional L2 norm
/// payload in addition to the canonical digit-range stages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DigitRangeRouteShape {
    plan: DigitRangePlan,
    rounds: usize,
    pub(crate) norm: Option<PhysicalL2NormShape>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PhysicalL2NormShape {
    pub(crate) subclaims: usize,
    pub(crate) virtual_evaluations: usize,
    pub(crate) rounds: usize,
    pub(crate) degree: usize,
}

impl DigitRangeRouteShape {
    pub(crate) fn stages(self) -> impl Iterator<Item = AkitaStage1StageShape> {
        let count = if self.norm.is_some() {
            self.plan.product_stage_arities().len()
        } else {
            self.plan.stage_count()
        };
        self.plan.stage_shapes_iter(self.rounds).take(count)
    }
}

fn stage1_root_values<E: Field + Ring>(b: usize) -> Vec<E> {
    let half = b / 2;
    (0..half)
        .map(|k| {
            let k = k as i64;
            E::from_i64(k * (k + 1))
        })
        .collect()
}

fn poly_coeffs_from_roots<E: Field>(roots: &[E]) -> Vec<E> {
    let mut coeffs = vec![E::one()];
    for &root in roots {
        let mut next = vec![E::zero(); coeffs.len() + 1];
        for (idx, &coeff) in coeffs.iter().enumerate() {
            next[idx] -= coeff * root;
            next[idx + 1] += coeff;
        }
        coeffs = next;
    }
    coeffs
}
