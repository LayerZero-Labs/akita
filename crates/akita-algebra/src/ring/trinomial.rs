//! Arithmetic for the LaBinius scalar and commitment trinomial rings.
//!
//! The scalar rings use `X^D + X^(D/2) + 1`; packing rank greater than one
//! maps them into `Y^D - Y^(D/2) + 1` by `X -> -Y^rank`. The transform is a
//! full split into two root cosets, each evaluated by the crate's existing
//! mixed-radix [`SmoothDomain`].

use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use jolt_field::Field;

use crate::fft::{field_pow, primitive_nth_root, FftWorkspace, SmoothDomain, SmoothFftField};

mod sealed {
    pub trait Sealed {}
}

/// A supported trinomial modulus shape.
///
/// The sealed implementations ensure a ring cannot be instantiated with an
/// arbitrary middle coefficient.
pub trait TrinomialModulus: sealed::Sealed + Copy + fmt::Debug + Eq {
    /// Middle coefficient in `X^D + middle * X^(D/2) + 1`.
    const MIDDLE_COEFFICIENT: i8;

    /// Ratio between the root-coset shift order and the half degree.
    const ROOT_ORDER_STRIDE: usize;
}

/// The scalar modulus `X^D + X^(D/2) + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlusTrinomial;

impl sealed::Sealed for PlusTrinomial {}

impl TrinomialModulus for PlusTrinomial {
    const MIDDLE_COEFFICIENT: i8 = 1;
    const ROOT_ORDER_STRIDE: usize = 3;
}

/// The packed commitment modulus `X^D - X^(D/2) + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MinusTrinomial;

impl sealed::Sealed for MinusTrinomial {}

impl TrinomialModulus for MinusTrinomial {
    const MIDDLE_COEFFICIENT: i8 = -1;
    const ROOT_ORDER_STRIDE: usize = 6;
}

/// Failure to construct or operate on a checked trinomial arithmetic shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrinomialError {
    /// A trinomial degree must be positive and even.
    InvalidDegree {
        /// Rejected degree.
        degree: usize,
    },
    /// A coefficient slice does not match the ring degree.
    CoefficientLength {
        /// Required coefficient count.
        expected: usize,
        /// Supplied coefficient count.
        actual: usize,
    },
    /// A product polynomial exceeds the supported degree `2D - 2`.
    ProductDegree {
        /// Maximum supported coefficient count.
        maximum_coefficients: usize,
        /// Supplied coefficient count.
        actual_coefficients: usize,
    },
    /// The field's declared smooth subgroup does not split this modulus.
    UnsupportedRootOrder {
        /// Root order required by the modulus.
        required: usize,
        /// Smooth subgroup order exposed by the field.
        available: usize,
    },
    /// Packing dimensions or modulus signs do not describe the tower map.
    InvalidPacking {
        /// Scalar ring degree.
        scalar_degree: usize,
        /// Target ring degree.
        target_degree: usize,
        /// Number of scalar components supplied or requested.
        components: usize,
    },
}

impl fmt::Display for TrinomialError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDegree { degree } => {
                write!(formatter, "trinomial degree {degree} must be positive and even")
            }
            Self::CoefficientLength { expected, actual } => write!(
                formatter,
                "trinomial coefficient length {actual} does not match degree {expected}"
            ),
            Self::ProductDegree {
                maximum_coefficients,
                actual_coefficients,
            } => write!(
                formatter,
                "product has {actual_coefficients} coefficients; at most {maximum_coefficients} are supported"
            ),
            Self::UnsupportedRootOrder {
                required,
                available,
            } => write!(
                formatter,
                "trinomial transform needs root order {required}, but field exposes smooth order {available}"
            ),
            Self::InvalidPacking {
                scalar_degree,
                target_degree,
                components,
            } => write!(
                formatter,
                "{components} scalar components of degree {scalar_degree} do not pack into trinomial degree {target_degree}"
            ),
        }
    }
}

impl std::error::Error for TrinomialError {}

fn validate_degree(degree: usize) -> Result<usize, TrinomialError> {
    if degree == 0 || !degree.is_multiple_of(2) {
        return Err(TrinomialError::InvalidDegree { degree });
    }
    Ok(degree / 2)
}

/// A coefficient-basis element modulo `X^D + s X^(D/2) + 1`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TrinomialRing<F, const D: usize, M: TrinomialModulus> {
    coefficients: [F; D],
    modulus: PhantomData<M>,
}

impl<F: fmt::Debug, const D: usize, M: TrinomialModulus> fmt::Debug for TrinomialRing<F, D, M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TrinomialRing")
            .field("degree", &D)
            .field("middle_coefficient", &M::MIDDLE_COEFFICIENT)
            .field("coefficients", &self.coefficients)
            .finish()
    }
}

impl<F, const D: usize, M: TrinomialModulus> TrinomialRing<F, D, M> {
    /// Construct an element from exactly `D` coefficient-basis values.
    pub fn from_coefficients(coefficients: [F; D]) -> Result<Self, TrinomialError> {
        validate_degree(D)?;
        Ok(Self {
            coefficients,
            modulus: PhantomData,
        })
    }

    /// Construct an element from a dynamically sized coefficient vector.
    pub fn try_from_vec(coefficients: Vec<F>) -> Result<Self, TrinomialError> {
        validate_degree(D)?;
        let actual = coefficients.len();
        let coefficients =
            coefficients
                .try_into()
                .map_err(|_| TrinomialError::CoefficientLength {
                    expected: D,
                    actual,
                })?;
        Ok(Self {
            coefficients,
            modulus: PhantomData,
        })
    }

    /// Borrow the canonical coefficient-basis representation.
    pub fn coefficients(&self) -> &[F; D] {
        &self.coefficients
    }

    /// Consume the element and return its coefficient array.
    pub fn into_coefficients(self) -> [F; D] {
        self.coefficients
    }
}

impl<F: Field, const D: usize, M: TrinomialModulus> TrinomialRing<F, D, M> {
    /// The additive identity.
    pub fn zero() -> Result<Self, TrinomialError> {
        Self::from_coefficients([F::zero(); D])
    }

    /// The multiplicative identity.
    pub fn one() -> Result<Self, TrinomialError> {
        let mut coefficients = [F::zero(); D];
        if D > 0 {
            coefficients[0] = F::one();
        }
        Self::from_coefficients(coefficients)
    }

    /// Multiply every coefficient by a base-field scalar.
    pub fn scale(&self, scalar: F) -> Self {
        Self {
            coefficients: self.coefficients.map(|coefficient| coefficient * scalar),
            modulus: PhantomData,
        }
    }

    /// Accumulate a scalar multiple into another element.
    pub fn scale_accumulate_into(&self, destination: &mut Self, scalar: F) {
        for (destination, &coefficient) in
            destination.coefficients.iter_mut().zip(&self.coefficients)
        {
            *destination = coefficient.mul_add(scalar, *destination);
        }
    }

    /// Unreduced schoolbook convolution, with at most `2D - 1` coefficients.
    ///
    /// This deliberately simple path is also the independent polynomial
    /// oracle for the full-split transform.
    pub fn schoolbook_product_coefficients(&self, rhs: &Self) -> Result<Vec<F>, TrinomialError> {
        let length = D
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(TrinomialError::InvalidDegree { degree: D })?;
        let mut product = vec![F::zero(); length];
        for (lhs_index, &lhs) in self.coefficients.iter().enumerate() {
            for (rhs_index, &rhs) in rhs.coefficients.iter().enumerate() {
                product[lhs_index + rhs_index] += lhs * rhs;
            }
        }
        Ok(product)
    }

    /// Reduce a product polynomial and construct its monic quotient.
    ///
    /// Returns `(remainder, quotient)` satisfying
    /// `input = remainder + (X^D + s X^(D/2) + 1) * quotient`. Inputs are
    /// limited to product degree `2D - 2`, so the quotient has degree at most
    /// `D - 2` and is returned with exactly `D - 1` coefficients.
    pub fn reduce_product_with_quotient(
        coefficients: &[F],
    ) -> Result<(Self, Vec<F>), TrinomialError> {
        let half = validate_degree(D)?;
        let maximum = D
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(TrinomialError::InvalidDegree { degree: D })?;
        if coefficients.len() > maximum {
            return Err(TrinomialError::ProductDegree {
                maximum_coefficients: maximum,
                actual_coefficients: coefficients.len(),
            });
        }

        let mut work = vec![F::zero(); maximum];
        work[..coefficients.len()].copy_from_slice(coefficients);
        let mut quotient = vec![F::zero(); D - 1];
        for degree in (D..maximum).rev() {
            let leading = work[degree];
            let quotient_degree = degree - D;
            quotient[quotient_degree] = leading;
            work[degree] = F::zero();
            work[quotient_degree] -= leading;
            if M::MIDDLE_COEFFICIENT > 0 {
                work[quotient_degree + half] -= leading;
            } else {
                work[quotient_degree + half] += leading;
            }
        }

        let remainder = std::array::from_fn(|index| work[index]);
        Ok((Self::from_coefficients(remainder)?, quotient))
    }

    /// Multiply through the independent schoolbook convolution and reducer.
    pub fn schoolbook_mul(&self, rhs: &Self) -> Result<Self, TrinomialError> {
        let product = self.schoolbook_product_coefficients(rhs)?;
        Self::reduce_product_with_quotient(&product).map(|(remainder, _)| remainder)
    }
}

impl<F: Field, const D: usize, M: TrinomialModulus> Add for TrinomialRing<F, D, M> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            coefficients: std::array::from_fn(|index| {
                self.coefficients[index] + rhs.coefficients[index]
            }),
            modulus: PhantomData,
        }
    }
}

impl<F: Field, const D: usize, M: TrinomialModulus> AddAssign for TrinomialRing<F, D, M> {
    fn add_assign(&mut self, rhs: Self) {
        for (lhs, rhs) in self.coefficients.iter_mut().zip(rhs.coefficients) {
            *lhs += rhs;
        }
    }
}

impl<F: Field, const D: usize, M: TrinomialModulus> Sub for TrinomialRing<F, D, M> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            coefficients: std::array::from_fn(|index| {
                self.coefficients[index] - rhs.coefficients[index]
            }),
            modulus: PhantomData,
        }
    }
}

impl<F: Field, const D: usize, M: TrinomialModulus> SubAssign for TrinomialRing<F, D, M> {
    fn sub_assign(&mut self, rhs: Self) {
        for (lhs, rhs) in self.coefficients.iter_mut().zip(rhs.coefficients) {
            *lhs -= rhs;
        }
    }
}

impl<F: Field, const D: usize, M: TrinomialModulus> Neg for TrinomialRing<F, D, M> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            coefficients: self.coefficients.map(Neg::neg),
            modulus: PhantomData,
        }
    }
}

/// Full-split evaluation of a trinomial ring element.
///
/// Slot order is fixed: evaluations on the positive shift coset come first,
/// followed by evaluations on the inverse shift coset. Values can only be
/// constructed by the canonical domain for the same field, degree, and sealed
/// modulus shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrinomialNtt<F, const D: usize, M: TrinomialModulus> {
    slots: [F; D],
    modulus: PhantomData<M>,
}

impl<F, const D: usize, M: TrinomialModulus> TrinomialNtt<F, D, M> {
    /// Borrow the canonical slot ordering.
    pub fn slots(&self) -> &[F; D] {
        &self.slots
    }
}

impl<F: Field, const D: usize, M: TrinomialModulus> TrinomialNtt<F, D, M> {
    /// Pointwise product in the fully split representation.
    pub fn pointwise_mul(&self, rhs: &Self) -> Self {
        Self {
            slots: std::array::from_fn(|index| self.slots[index] * rhs.slots[index]),
            modulus: PhantomData,
        }
    }

    /// Accumulate a pointwise product without another transform.
    pub fn add_assign_pointwise_mul(&mut self, lhs: &Self, rhs: &Self) {
        for ((accumulator, &lhs), &rhs) in self.slots.iter_mut().zip(&lhs.slots).zip(&rhs.slots) {
            *accumulator += lhs * rhs;
        }
    }
}

/// Reusable full-split transform plan for one field, degree, and modulus.
pub struct TrinomialNttDomain<F, const D: usize, M: TrinomialModulus> {
    half_domain: SmoothDomain<F>,
    positive_twists: Vec<F>,
    negative_twists: Vec<F>,
    positive_coset_value: F,
    negative_coset_value: F,
    coset_separation_inverse: F,
    modulus: PhantomData<M>,
}

/// Reusable allocation-free scratch space for a trinomial transform plan.
pub struct TrinomialNttWorkspace<F, const D: usize, M: TrinomialModulus> {
    positive: Vec<F>,
    negative: Vec<F>,
    fft: FftWorkspace<F>,
    modulus: PhantomData<M>,
}

impl<F, const D: usize, M> TrinomialNttDomain<F, D, M>
where
    F: SmoothFftField + fmt::Debug,
    M: TrinomialModulus,
{
    /// Build the canonical full-split transform and precompute its twists.
    pub fn new() -> Result<Self, TrinomialError> {
        let half = validate_degree(D)?;
        let required = half
            .checked_mul(M::ROOT_ORDER_STRIDE)
            .ok_or(TrinomialError::InvalidDegree { degree: D })?;
        if !F::SMOOTH_SUBGROUP_ORDER.is_multiple_of(required) {
            return Err(TrinomialError::UnsupportedRootOrder {
                required,
                available: F::SMOOTH_SUBGROUP_ORDER,
            });
        }

        let positive_shift = primitive_nth_root::<F>(required);
        let negative_shift =
            positive_shift
                .inverse()
                .ok_or(TrinomialError::UnsupportedRootOrder {
                    required,
                    available: F::SMOOTH_SUBGROUP_ORDER,
                })?;
        let half_root = field_pow(positive_shift, M::ROOT_ORDER_STRIDE as u64);
        let half_domain = SmoothDomain::new(half_root, half);

        let positive_coset_value = field_pow(positive_shift, half as u64);
        let negative_coset_value = field_pow(negative_shift, half as u64);
        let coset_separation_inverse = (positive_coset_value - negative_coset_value)
            .inverse()
            .ok_or(TrinomialError::UnsupportedRootOrder {
                required,
                available: F::SMOOTH_SUBGROUP_ORDER,
            })?;

        let mut positive_twists = Vec::with_capacity(half);
        let mut negative_twists = Vec::with_capacity(half);
        let mut positive = F::one();
        let mut negative = F::one();
        for _ in 0..half {
            positive_twists.push(positive);
            negative_twists.push(negative);
            positive *= positive_shift;
            negative *= negative_shift;
        }

        Ok(Self {
            half_domain,
            positive_twists,
            negative_twists,
            positive_coset_value,
            negative_coset_value,
            coset_separation_inverse,
            modulus: PhantomData,
        })
    }

    /// Allocate scratch storage once for repeated transforms with this plan.
    pub fn workspace(&self) -> TrinomialNttWorkspace<F, D, M> {
        let half = D / 2;
        TrinomialNttWorkspace {
            positive: vec![F::zero(); half],
            negative: vec![F::zero(); half],
            fft: FftWorkspace::new(half),
            modulus: PhantomData,
        }
    }

    /// Transform coefficients into the canonical two-coset slot order.
    ///
    /// This convenience method allocates scratch storage. Repeated operations
    /// should reuse [`Self::forward_with_workspace`].
    pub fn forward(&self, value: &TrinomialRing<F, D, M>) -> TrinomialNtt<F, D, M> {
        self.forward_with_workspace(value, &mut self.workspace())
    }

    /// Transform coefficients using caller-owned reusable scratch storage.
    pub fn forward_with_workspace(
        &self,
        value: &TrinomialRing<F, D, M>,
        workspace: &mut TrinomialNttWorkspace<F, D, M>,
    ) -> TrinomialNtt<F, D, M> {
        let half = D / 2;
        for index in 0..half {
            let low = value.coefficients[index];
            let high = value.coefficients[index + half];
            workspace.positive[index] =
                (low + self.positive_coset_value * high) * self.positive_twists[index];
            workspace.negative[index] =
                (low + self.negative_coset_value * high) * self.negative_twists[index];
        }
        let mut transformed = TrinomialNtt {
            slots: [F::zero(); D],
            modulus: PhantomData,
        };
        self.half_domain.forward_into(
            &workspace.positive,
            &mut transformed.slots[..half],
            &mut workspace.fft,
        );
        self.half_domain.forward_into(
            &workspace.negative,
            &mut transformed.slots[half..],
            &mut workspace.fft,
        );
        transformed
    }

    /// Recover coefficients from the canonical two-coset slot order.
    ///
    /// This convenience method allocates scratch storage. Repeated operations
    /// should reuse [`Self::inverse_with_workspace`].
    pub fn inverse(&self, value: &TrinomialNtt<F, D, M>) -> TrinomialRing<F, D, M> {
        self.inverse_with_workspace(value, &mut self.workspace())
    }

    /// Recover coefficients using caller-owned reusable scratch storage.
    pub fn inverse_with_workspace(
        &self,
        value: &TrinomialNtt<F, D, M>,
        workspace: &mut TrinomialNttWorkspace<F, D, M>,
    ) -> TrinomialRing<F, D, M> {
        let half = D / 2;
        self.half_domain.inverse_into(
            &value.slots[..half],
            &mut workspace.positive,
            &mut workspace.fft,
        );
        self.half_domain.inverse_into(
            &value.slots[half..],
            &mut workspace.negative,
            &mut workspace.fft,
        );
        let mut coefficients = [F::zero(); D];
        for index in 0..half {
            let positive_untwisted = workspace.positive[index] * self.negative_twists[index];
            let negative_untwisted = workspace.negative[index] * self.positive_twists[index];
            let high = (positive_untwisted - negative_untwisted) * self.coset_separation_inverse;
            coefficients[index] = positive_untwisted - self.positive_coset_value * high;
            coefficients[index + half] = high;
        }
        TrinomialRing {
            coefficients,
            modulus: PhantomData,
        }
    }

    /// Multiply with two forward transforms, one pointwise product, and one inverse.
    ///
    /// This convenience method allocates scratch storage. Repeated operations
    /// should reuse [`Self::multiply_with_workspace`].
    pub fn multiply(
        &self,
        lhs: &TrinomialRing<F, D, M>,
        rhs: &TrinomialRing<F, D, M>,
    ) -> TrinomialRing<F, D, M> {
        self.multiply_with_workspace(lhs, rhs, &mut self.workspace())
    }

    /// Multiply using caller-owned reusable transform scratch storage.
    pub fn multiply_with_workspace(
        &self,
        lhs: &TrinomialRing<F, D, M>,
        rhs: &TrinomialRing<F, D, M>,
        workspace: &mut TrinomialNttWorkspace<F, D, M>,
    ) -> TrinomialRing<F, D, M> {
        let lhs = self.forward_with_workspace(lhs, workspace);
        let rhs = self.forward_with_workspace(rhs, workspace);
        self.inverse_with_workspace(&lhs.pointwise_mul(&rhs), workspace)
    }
}

fn validate_packing<const SCALAR_D: usize, const D: usize, M: TrinomialModulus>(
    components: usize,
) -> Result<usize, TrinomialError> {
    let scalar_half = validate_degree(SCALAR_D)?;
    validate_degree(D)?;
    let expected = SCALAR_D
        .checked_mul(components)
        .ok_or(TrinomialError::InvalidPacking {
            scalar_degree: SCALAR_D,
            target_degree: D,
            components,
        })?;
    let sign_matches = if components == 1 {
        M::MIDDLE_COEFFICIENT == PlusTrinomial::MIDDLE_COEFFICIENT
    } else {
        components.is_power_of_two()
            && scalar_half % 2 == 1
            && M::MIDDLE_COEFFICIENT == MinusTrinomial::MIDDLE_COEFFICIENT
    };
    if components == 0 || expected != D || !sign_matches {
        return Err(TrinomialError::InvalidPacking {
            scalar_degree: SCALAR_D,
            target_degree: D,
            components,
        });
    }
    Ok(components)
}

/// Pack scalar-ring components by the signed interleaving `X -> -Y^rank`.
pub fn pack_scalar_components<F, const SCALAR_D: usize, const D: usize, M>(
    components: &[TrinomialRing<F, SCALAR_D, PlusTrinomial>],
) -> Result<TrinomialRing<F, D, M>, TrinomialError>
where
    F: Field,
    M: TrinomialModulus,
{
    let rank = validate_packing::<SCALAR_D, D, M>(components.len())?;
    let mut packed = [F::zero(); D];
    for (component_index, component) in components.iter().enumerate() {
        for (scalar_index, &coefficient) in component.coefficients.iter().enumerate() {
            let coefficient = if scalar_index % 2 == 0 {
                coefficient
            } else {
                -coefficient
            };
            packed[scalar_index * rank + component_index] = coefficient;
        }
    }
    TrinomialRing::from_coefficients(packed)
}

/// Embed one scalar component as the constant module coordinate.
pub fn embed_scalar<F, const SCALAR_D: usize, const D: usize, M>(
    scalar: &TrinomialRing<F, SCALAR_D, PlusTrinomial>,
) -> Result<TrinomialRing<F, D, M>, TrinomialError>
where
    F: Field,
    M: TrinomialModulus,
{
    if !D.is_multiple_of(SCALAR_D) {
        return Err(TrinomialError::InvalidPacking {
            scalar_degree: SCALAR_D,
            target_degree: D,
            components: 0,
        });
    }
    let rank = validate_packing::<SCALAR_D, D, M>(D / SCALAR_D)?;
    let mut embedded = [F::zero(); D];
    for (scalar_index, &coefficient) in scalar.coefficients.iter().enumerate() {
        embedded[scalar_index * rank] = if scalar_index % 2 == 0 {
            coefficient
        } else {
            -coefficient
        };
    }
    TrinomialRing::from_coefficients(embedded)
}

/// Invert signed coefficient interleaving into scalar-ring components.
pub fn unpack_scalar_components<F, const SCALAR_D: usize, const D: usize, M>(
    packed: &TrinomialRing<F, D, M>,
) -> Result<Vec<TrinomialRing<F, SCALAR_D, PlusTrinomial>>, TrinomialError>
where
    F: Field,
    M: TrinomialModulus,
{
    if !D.is_multiple_of(SCALAR_D) {
        return Err(TrinomialError::InvalidPacking {
            scalar_degree: SCALAR_D,
            target_degree: D,
            components: 0,
        });
    }
    let rank = validate_packing::<SCALAR_D, D, M>(D / SCALAR_D)?;
    let mut components = Vec::with_capacity(rank);
    for component_index in 0..rank {
        let coefficients = std::array::from_fn(|scalar_index| {
            let coefficient = packed.coefficients[scalar_index * rank + component_index];
            if scalar_index % 2 == 0 {
                coefficient
            } else {
                -coefficient
            }
        });
        components.push(TrinomialRing::from_coefficients(coefficients)?);
    }
    Ok(components)
}

#[cfg(test)]
mod tests;
