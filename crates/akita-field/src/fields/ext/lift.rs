//! Helpers for embedding base fields into extension fields.

use crate::fields::ext::{
    FpExt2, FpExt2Config, PowerBasisFpExt4, PowerBasisFpExt4Config, PowerBasisFpExt4MulBackend,
    RingSubfieldFpExt4, RingSubfieldFpExt4MulBackend, RingSubfieldFpExt8,
    RingSubfieldFpExt8MulBackend, TowerBasisFpExt4, TowerBasisFpExt4Config, UnitNr,
};
use crate::fields::unreduced::HasUnreducedOps;
use crate::{
    pseudo_mersenne_modulus, AkitaError, FieldCore, FromPrimitiveInt, PseudoMersenneField,
};
use akita_serialization::Valid;

/// Lift a base-field element into an extension field.
///
/// This is intentionally small: for extension towers we embed into the constant term.
pub trait LiftBase<F: FieldCore>: FieldCore {
    /// Embed `x ∈ F` as a constant in `Self`.
    fn lift_base(x: F) -> Self;
}

/// Multiply an extension-field element by a base-field scalar.
///
/// This avoids materializing the base scalar as an extension element and then
/// using a full extension multiply. For tower extensions this scales each
/// base-field coordinate directly.
pub trait MulBase<F: FieldCore>: FieldCore {
    /// Return `self * x`, where `x` is interpreted as a base-field scalar.
    fn mul_base(self, x: F) -> Self;
}

/// An algebraic extension of base field `F`.
///
/// Provides the extension degree and a constructor from a slice of base-field
/// coefficients (in the canonical basis `{1, u, u^2, ...}`).
pub trait ExtField<F: FieldCore>: FieldCore + LiftBase<F> + MulBase<F> + FromPrimitiveInt {
    /// Extension degree: `[Self : F]`.
    const EXT_DEGREE: usize;

    /// Construct from a coefficient slice `[c0, c1, ..., c_{d-1}]`.
    ///
    /// # Panics
    /// Panics if `coeffs.len() != Self::EXT_DEGREE`.
    fn from_base_slice(coeffs: &[F]) -> Self;

    /// Return base-field coefficients in the canonical basis.
    fn to_base_vec(&self) -> Vec<F>;
}

/// Deferred-reduction extension-times-base multiply.
///
/// `mul_base_to_product_accum` scales `self` by a base scalar `x` and writes the
/// result into [`HasUnreducedOps::ProductAccum`] without reducing, so a batch of
/// `E × F` products can be summed and reduced once. When
/// [`HasUnreducedOps::DELAYED_PRODUCT_SUM_IS_EXACT`] holds, the reduced sum equals
/// the per-term [`MulBase::mul_base`] sum within the accumulator's headroom.
///
/// `E × F` has no cross terms, so the default body (lift `x` and reuse
/// [`HasUnreducedOps::mul_to_product_accum`]) is correct everywhere; extensions
/// whose product-accumulator layout admits cheaper coordinate scaling override it.
pub trait MulBaseUnreduced<F: FieldCore>: ExtField<F> + HasUnreducedOps {
    /// Accumulate `self * x` (extension times base scalar) without reducing.
    #[inline]
    fn mul_base_to_product_accum(self, x: F) -> Self::ProductAccum {
        self.mul_to_product_accum(Self::lift_base(x))
    }
}

impl<F: FieldCore + FromPrimitiveInt + HasUnreducedOps> MulBaseUnreduced<F> for F {}

/// Frobenius operations for an extension field over `F`.
///
/// The default implementations below are intentionally algebraic rather than
/// basis-specific: they raise to powers of the base-field modulus. Specialized
/// extension types can add cheaper implementations later, but this gives the
/// protocol a single auditable contract first.
pub trait FrobeniusExtField<F: FieldCore>: ExtField<F> {
    /// Apply `x -> x^(q^power)`, where `q = |F|`.
    fn frobenius_pow(self, power: usize) -> Self;

    /// Apply the inverse Frobenius power. Since `x -> x^q` has order
    /// `[Self:F]` on `Self`, this is `frobenius_pow(EXT_DEGREE - power)`.
    fn frobenius_inv_pow(self, power: usize) -> Self {
        let degree = Self::EXT_DEGREE;
        if degree == 0 {
            return self;
        }
        self.frobenius_pow((degree - (power % degree)) % degree)
    }
}

#[inline]
fn field_pow_u128<E: FieldCore>(mut base: E, mut exp: u128) -> E {
    let mut acc = E::one();
    while exp > 0 {
        if (exp & 1) == 1 {
            acc *= base;
        }
        base *= base;
        exp >>= 1;
    }
    acc
}

#[inline]
fn base_modulus<F: PseudoMersenneField>() -> u128 {
    pseudo_mersenne_modulus(F::MODULUS_BITS, F::MODULUS_OFFSET)
        .expect("pseudo-Mersenne modulus parameters must be valid")
}

fn frobenius_pow_via_base_modulus<F, E>(value: E, power: usize) -> E
where
    F: PseudoMersenneField,
    E: ExtField<F>,
{
    let q = base_modulus::<F>();
    let mut out = value;
    for _ in 0..(power % E::EXT_DEGREE.max(1)) {
        out = field_pow_u128(out, q);
    }
    out
}

impl<F> FrobeniusExtField<F> for F
where
    F: PseudoMersenneField,
{
    #[inline]
    fn frobenius_pow(self, power: usize) -> Self {
        let _ = power;
        self
    }
}

impl<F, C> FrobeniusExtField<F> for FpExt2<F, C>
where
    F: PseudoMersenneField + Valid,
    C: FpExt2Config<F>,
{
    #[inline]
    fn frobenius_pow(self, power: usize) -> Self {
        frobenius_pow_via_base_modulus::<F, Self>(self, power)
    }
}

impl<F, C2> FrobeniusExtField<F> for TowerBasisFpExt4<F, C2, UnitNr>
where
    F: PseudoMersenneField + Valid + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
{
    #[inline]
    fn frobenius_pow(self, power: usize) -> Self {
        frobenius_pow_via_base_modulus::<F, Self>(self, power)
    }
}

impl<F, C> FrobeniusExtField<F> for PowerBasisFpExt4<F, C>
where
    F: PseudoMersenneField + Valid + PowerBasisFpExt4MulBackend<C>,
    C: PowerBasisFpExt4Config<F>,
{
    #[inline]
    fn frobenius_pow(self, power: usize) -> Self {
        frobenius_pow_via_base_modulus::<F, Self>(self, power)
    }
}

impl<F> FrobeniusExtField<F> for RingSubfieldFpExt4<F>
where
    F: PseudoMersenneField + Valid + RingSubfieldFpExt4MulBackend,
{
    #[inline]
    fn frobenius_pow(self, power: usize) -> Self {
        frobenius_pow_via_base_modulus::<F, Self>(self, power)
    }
}

impl<F> FrobeniusExtField<F> for RingSubfieldFpExt8<F>
where
    F: PseudoMersenneField + Valid + RingSubfieldFpExt8MulBackend,
{
    #[inline]
    fn frobenius_pow(self, power: usize) -> Self {
        frobenius_pow_via_base_modulus::<F, Self>(self, power)
    }
}

/// Return the first `width` elements of the canonical extension basis.
///
/// For [`RingSubfieldFpExt4`] and [`RingSubfieldFpExt8`] this is the fixed
/// ring-subfield basis `[1, e1, ...]`, so the chosen Moore-type theta family
/// is aligned with the coefficient packing basis used by `embed_subfield`.
///
/// # Errors
///
/// Returns an error if `width > E::EXT_DEGREE`.
pub fn canonical_frobenius_thetas<F, E>(width: usize) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    if width > E::EXT_DEGREE {
        return Err(AkitaError::InvalidInput(format!(
            "Frobenius theta width {width} exceeds extension degree {}",
            E::EXT_DEGREE
        )));
    }
    Ok((0..width)
        .map(|idx| {
            let mut coeffs = vec![F::zero(); E::EXT_DEGREE];
            coeffs[idx] = F::one();
            E::from_base_slice(&coeffs)
        })
        .collect())
}

/// Solve `M_t(theta) z = r`, where
/// `M_t(theta)_{j,h} = theta_h^(q^-j)`.
///
/// This intentionally uses dense elimination: supported Frobenius widths are
/// tiny (`<= [E:F]`) and explicit validation is more valuable here than a
/// clever specialized solver.
///
/// # Errors
///
/// Returns an error if the matrix is not square, the dimensions do not match,
/// or the Moore-type matrix is singular.
pub fn solve_frobenius_moore<F, E>(thetas: &[E], rhs: &[E]) -> Result<Vec<E>, AkitaError>
where
    F: PseudoMersenneField,
    E: FrobeniusExtField<F>,
{
    let n = thetas.len();
    if rhs.len() != n {
        return Err(AkitaError::InvalidSize {
            expected: n,
            actual: rhs.len(),
        });
    }
    let mut matrix = (0..n)
        .map(|row| {
            thetas
                .iter()
                .map(|&theta| theta.frobenius_inv_pow(row))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut values = rhs.to_vec();

    for col in 0..n {
        let pivot = (col..n)
            .find(|&row| !matrix[row][col].is_zero())
            .ok_or_else(|| {
                AkitaError::InvalidInput("singular Frobenius Moore-type matrix".to_string())
            })?;
        if pivot != col {
            matrix.swap(col, pivot);
            values.swap(col, pivot);
        }
        let inv = matrix[col][col].inverse().ok_or_else(|| {
            AkitaError::InvalidInput("singular Frobenius Moore-type matrix".to_string())
        })?;
        for entry in &mut matrix[col][col..] {
            *entry *= inv;
        }
        values[col] *= inv;

        let pivot_tail = matrix[col][col..].to_vec();
        let pivot_value = values[col];
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = matrix[row][col];
            if factor.is_zero() {
                continue;
            }
            for (entry, &pivot_entry) in matrix[row][col..].iter_mut().zip(pivot_tail.iter()) {
                *entry -= factor * pivot_entry;
            }
            values[row] -= factor * pivot_value;
        }
    }
    Ok(values)
}

/// Validate that the canonical theta family gives a nonsingular Moore-type
/// matrix for `width`.
///
/// # Errors
///
/// Returns an error if theta construction fails or the Moore solve rejects.
pub fn validate_canonical_frobenius_thetas<F, E>(width: usize) -> Result<(), AkitaError>
where
    F: PseudoMersenneField,
    E: FrobeniusExtField<F>,
{
    let thetas = canonical_frobenius_thetas::<F, E>(width)?;
    let rhs = (0..width)
        .map(|idx| E::lift_base(F::from_u64((idx + 1) as u64)))
        .collect::<Vec<_>>();
    solve_frobenius_moore::<F, E>(&thetas, &rhs).map(|_| ())
}

impl<F: FieldCore + FromPrimitiveInt> ExtField<F> for F {
    const EXT_DEGREE: usize = 1;

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 1);
        coeffs[0]
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        vec![*self]
    }
}

impl<F, C> ExtField<F> for FpExt2<F, C>
where
    F: FieldCore + FromPrimitiveInt + Valid,
    C: FpExt2Config<F>,
{
    const EXT_DEGREE: usize = 2;

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 2);
        Self::new(coeffs[0], coeffs[1])
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        vec![self.coeffs[0], self.coeffs[1]]
    }
}

impl<F, C2> ExtField<F> for TowerBasisFpExt4<F, C2, UnitNr>
where
    F: FieldCore + FromPrimitiveInt + Valid + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
{
    const EXT_DEGREE: usize = 4;

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 4);
        Self::new(
            FpExt2::new(coeffs[0], coeffs[2]),
            FpExt2::new(coeffs[1], coeffs[3]),
        )
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        vec![
            self.coeffs[0].coeffs[0],
            self.coeffs[1].coeffs[0],
            self.coeffs[0].coeffs[1],
            self.coeffs[1].coeffs[1],
        ]
    }
}

impl<F, C2> LiftBase<FpExt2<F, C2>> for TowerBasisFpExt4<F, C2, UnitNr>
where
    F: FieldCore + Valid + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
{
    #[inline]
    fn lift_base(x: FpExt2<F, C2>) -> Self {
        Self::new(x, FpExt2::zero())
    }
}

impl<F, C2> MulBase<FpExt2<F, C2>> for TowerBasisFpExt4<F, C2, UnitNr>
where
    F: FieldCore + Valid + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
{
    #[inline]
    fn mul_base(self, x: FpExt2<F, C2>) -> Self {
        Self::new(self.coeffs[0] * x, self.coeffs[1] * x)
    }
}

impl<F, C2> ExtField<FpExt2<F, C2>> for TowerBasisFpExt4<F, C2, UnitNr>
where
    F: FieldCore + FromPrimitiveInt + Valid + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
{
    const EXT_DEGREE: usize = 2;

    #[inline]
    fn from_base_slice(coeffs: &[FpExt2<F, C2>]) -> Self {
        assert_eq!(coeffs.len(), 2);
        Self::new(coeffs[0], coeffs[1])
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<FpExt2<F, C2>> {
        vec![self.coeffs[0], self.coeffs[1]]
    }
}

impl<F, C> ExtField<F> for PowerBasisFpExt4<F, C>
where
    F: FieldCore + FromPrimitiveInt + Valid + PowerBasisFpExt4MulBackend<C>,
    C: PowerBasisFpExt4Config<F>,
{
    const EXT_DEGREE: usize = 4;

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 4);
        Self::new([coeffs[0], coeffs[1], coeffs[2], coeffs[3]])
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        self.coeffs.to_vec()
    }
}

impl<F> ExtField<F> for RingSubfieldFpExt4<F>
where
    F: FieldCore + FromPrimitiveInt + Valid + RingSubfieldFpExt4MulBackend,
{
    const EXT_DEGREE: usize = 4;

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 4);
        Self::new([coeffs[0], coeffs[1], coeffs[2], coeffs[3]])
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        self.coeffs.to_vec()
    }
}

impl<F> ExtField<F> for RingSubfieldFpExt8<F>
where
    F: FieldCore + FromPrimitiveInt + Valid + RingSubfieldFpExt8MulBackend,
{
    const EXT_DEGREE: usize = 8;

    #[inline]
    fn from_base_slice(coeffs: &[F]) -> Self {
        assert_eq!(coeffs.len(), 8);
        Self::new([
            coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5], coeffs[6], coeffs[7],
        ])
    }

    #[inline]
    fn to_base_vec(&self) -> Vec<F> {
        self.coeffs.to_vec()
    }
}

impl<F: FieldCore> LiftBase<F> for F {
    #[inline]
    fn lift_base(x: F) -> Self {
        x
    }
}

impl<F: FieldCore> MulBase<F> for F {
    #[inline]
    fn mul_base(self, x: F) -> Self {
        self * x
    }
}

impl<F, C> LiftBase<F> for FpExt2<F, C>
where
    F: FieldCore + Valid,
    C: FpExt2Config<F>,
{
    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new(x, F::zero())
    }
}

impl<F, C> MulBase<F> for FpExt2<F, C>
where
    F: FieldCore + Valid,
    C: FpExt2Config<F>,
{
    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(self.coeffs[0] * x, self.coeffs[1] * x)
    }
}

impl<F, C2, C4> LiftBase<F> for TowerBasisFpExt4<F, C2, C4>
where
    F: FieldCore + Valid + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
{
    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new(FpExt2::new(x, F::zero()), FpExt2::new(F::zero(), F::zero()))
    }
}

impl<F, C2, C4> MulBase<F> for TowerBasisFpExt4<F, C2, C4>
where
    F: FieldCore + Valid + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
{
    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(self.coeffs[0].mul_base(x), self.coeffs[1].mul_base(x))
    }
}

impl<F, C> LiftBase<F> for PowerBasisFpExt4<F, C>
where
    F: FieldCore + Valid + PowerBasisFpExt4MulBackend<C>,
    C: PowerBasisFpExt4Config<F>,
{
    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new([x, F::zero(), F::zero(), F::zero()])
    }
}

impl<F, C> MulBase<F> for PowerBasisFpExt4<F, C>
where
    F: FieldCore + Valid + PowerBasisFpExt4MulBackend<C>,
    C: PowerBasisFpExt4Config<F>,
{
    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] * x))
    }
}

impl<F> LiftBase<F> for RingSubfieldFpExt4<F>
where
    F: FieldCore + Valid + RingSubfieldFpExt4MulBackend,
{
    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new([x, F::zero(), F::zero(), F::zero()])
    }
}

impl<F> MulBase<F> for RingSubfieldFpExt4<F>
where
    F: FieldCore + Valid + RingSubfieldFpExt4MulBackend,
{
    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] * x))
    }
}

impl<F> LiftBase<F> for RingSubfieldFpExt8<F>
where
    F: FieldCore + Valid + RingSubfieldFpExt8MulBackend,
{
    #[inline]
    fn lift_base(x: F) -> Self {
        Self::new([
            x,
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
        ])
    }
}

impl<F> MulBase<F> for RingSubfieldFpExt8<F>
where
    F: FieldCore + Valid + RingSubfieldFpExt8MulBackend,
{
    #[inline]
    fn mul_base(self, x: F) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i] * x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp32, NegOneNr, UnitNr};

    type F = Fp32<251>;
    type E2 = FpExt2<F, NegOneNr>;
    type E4 = TowerBasisFpExt4<F, NegOneNr, UnitNr>;

    #[test]
    fn mul_base_matches_full_multiply_for_base_field() {
        let x = F::from_u64(7);
        let scalar = F::from_u64(11);

        assert_eq!(x.mul_base(scalar), x * scalar);
    }

    #[test]
    fn mul_base_matches_full_multiply_for_fp_ext2() {
        let x = E2::new(F::from_u64(3), F::from_u64(5));
        let scalar = F::from_u64(11);

        assert_eq!(x.mul_base(scalar), x * E2::lift_base(scalar));
    }

    #[test]
    fn mul_base_matches_full_multiply_for_fp_ext4() {
        let x = E4::new(
            E2::new(F::from_u64(3), F::from_u64(5)),
            E2::new(F::from_u64(7), F::from_u64(13)),
        );
        let scalar = F::from_u64(11);

        assert_eq!(x.mul_base(scalar), x * E4::lift_base(scalar));
    }

    #[test]
    fn fp_ext4_mul_base_over_fp_ext2_matches_full_multiply() {
        let x = E4::new(
            E2::new(F::from_u64(3), F::from_u64(5)),
            E2::new(F::from_u64(7), F::from_u64(13)),
        );
        let scalar = E2::new(F::from_u64(11), F::from_u64(17));

        assert_eq!(
            <E4 as MulBase<E2>>::mul_base(x, scalar),
            x * <E4 as LiftBase<E2>>::lift_base(scalar)
        );
    }

    #[test]
    fn fp_ext4_lift_over_fp_ext2_agrees_with_lift_over_base() {
        let scalar = F::from_u64(7);
        let via_base = <E4 as LiftBase<F>>::lift_base(scalar);
        let via_tower = <E4 as LiftBase<E2>>::lift_base(<E2 as LiftBase<F>>::lift_base(scalar));

        assert_eq!(via_base, via_tower);
    }

    #[test]
    fn fp_ext4_ext_over_fp_ext2_round_trips_through_base_slice() {
        let x = E4::new(
            E2::new(F::from_u64(3), F::from_u64(5)),
            E2::new(F::from_u64(7), F::from_u64(13)),
        );
        let coeffs = <E4 as ExtField<E2>>::to_base_vec(&x);
        let rebuilt = <E4 as ExtField<E2>>::from_base_slice(&coeffs);

        assert_eq!(rebuilt, x);
        assert_eq!(<E4 as ExtField<E2>>::EXT_DEGREE, 2);
    }

    /// `ExtField<F>::EXT_DEGREE` over the base prime field must equal
    /// `ExtField<ClaimField>::EXT_DEGREE * ExtField<F>::EXT_DEGREE` on the
    /// claim field. This is the chain
    /// `[ChallengeField : F] = [ChallengeField : ClaimField] * [ClaimField : F]`
    /// the field-role convention relies on.
    #[test]
    fn fp_ext4_ext_degrees_chain_correctly() {
        assert_eq!(
            <E4 as ExtField<F>>::EXT_DEGREE,
            <E4 as ExtField<E2>>::EXT_DEGREE * <E2 as ExtField<F>>::EXT_DEGREE
        );
    }
}
