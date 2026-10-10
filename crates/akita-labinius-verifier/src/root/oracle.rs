use crate::{channel::ClearChannel, codec, lowered::LoweredRootLayout};
use akita_algebra::{poly::multilinear_eval, EqPolynomial};
use akita_error::{checked, AkitaError};
use core::marker::PhantomData;
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// The polynomial-opening obligations returned by the reduction: one point
/// and one value per committed table, in the challenge field `E`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootEvaluationClaims<E> {
    pub response_point: Vec<E>,
    pub response_value: E,
    pub image_point: Vec<E>,
    pub image_value: E,
    /// Point and value of the prime left opening, present exactly when the
    /// statement has a prime claim.
    pub prime: Option<(Vec<E>, E)>,
}

/// Commitment and evaluation boundary for a root verifier.
/// Implementations MUST bind the exact image table before frontend challenges,
/// bind the prime left opening before the fold challenges, bind a response
/// table before coefficient challenges, and authenticate every evaluation with
/// their own soundness error. `bind_prime_opening` runs exactly once when the
/// statement has a prime claim and never otherwise; every other method runs
/// exactly once.
///
/// Every binding MUST fix every table entry, padding included, in the layout's
/// table order. The image and response tables hold stored base-16 digits,
/// small integers of the reduction's base field: the image table is the one
/// `encode_image` of the prover crate builds from the clear commitment, the
/// response table is the one the prover hands to `commit_response`. The prime
/// left opening is a table of `layout.prime_len()` arbitrary elements of the
/// challenge field `E`, padded coefficient innermost and column outermost.
/// The claims are evaluations of the tables' multilinear extensions at points
/// of `E`. The reduction checks the digit alphabet of the two digit tables and
/// no range on the prime left opening; a binding checks none.
/// `discharge` runs after every evaluation message, so every opening challenge
/// it draws follows them. A scheme that lets two claims refer to tables other
/// than the ones bound does not satisfy this trait. The normative statement is
/// the "Oracle contract" section of `specs/labinius-root-reduction.md`.
pub trait RootVerifierOracle<E: Field> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError>;
    fn bind_prime_opening<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError>;
    fn bind_response<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError>;
    fn discharge<S: ClearChannel>(
        &mut self,
        claims: &RootEvaluationClaims<E>,
        channel: &mut S,
    ) -> Result<(), AkitaError>;
}

/// Prover boundary matching the verifier's oracle call sites.
/// The response commitment MUST fix the supplied digit bytes and the prime
/// commitment the supplied field elements; the evaluation proof MUST refer to
/// the same owners that were bound.
///
/// The obligations of [`RootVerifierOracle`] apply to the matching prover
/// calls: the same field pair, the same table order, every padding entry
/// bound, and no opening challenge before the last evaluation message.
pub trait RootProverOracle<E: Field> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError>;
    fn commit_prime_opening<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        table: &[E],
        channel: &mut S,
    ) -> Result<(), AkitaError>;
    fn commit_response<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        digits: &[u8],
        channel: &mut S,
    ) -> Result<(), AkitaError>;
    fn discharge<S: ClearChannel>(
        &mut self,
        claims: &RootEvaluationClaims<E>,
        channel: &mut S,
    ) -> Result<(), AkitaError>;
}

/// Bind the whole image digit table as public bytes, one per stored digit, in
/// table order.
pub fn bind_transparent_image<S: ClearChannel>(
    layout: &LoweredRootLayout,
    image: &[u8],
    channel: &mut S,
) -> Result<(), AkitaError> {
    if image.len() != layout.image_len() {
        return Err(AkitaError::InvalidInput(
            "transparent image length mismatch".into(),
        ));
    }
    channel.public(image)
}

/// Exchange the whole prime left opening as one message: every element as its
/// canonical base coordinates, in table order. A prover emits `table`; a
/// verifier overwrites it and rejects a noncanonical coordinate.
pub fn exchange_transparent_prime_opening<F, E, S>(
    layout: &LoweredRootLayout,
    table: &mut [E],
    channel: &mut S,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    S: ClearChannel,
{
    if table.len() != layout.prime_len() {
        return Err(AkitaError::InvalidInput(
            "transparent prime opening length mismatch".into(),
        ));
    }
    let mut bytes = Vec::new();
    codec::encode_extensions::<F, E>(&mut bytes, table)?;
    channel.message(&mut bytes)?;
    codec::decode_extensions::<F, E>(&bytes, table)
}

/// Authenticate the little-endian multilinear evaluations against whole
/// tables. `prime` is consulted exactly when the claims carry a prime
/// evaluation.
/// There is deliberately no alphabet validation here: that is a sumcheck term.
pub fn check_transparent_evaluations<E: Field>(
    image: &[u8],
    digits: &[u8],
    prime: &[E],
    claims: &RootEvaluationClaims<E>,
) -> Result<(), AkitaError> {
    if digit_multilinear_eval(digits, &claims.response_point)? != claims.response_value
        || digit_multilinear_eval(image, &claims.image_point)? != claims.image_value
    {
        return Err(AkitaError::InvalidProof);
    }
    if let Some((point, value)) = &claims.prime {
        if multilinear_eval(prime, point)? != *value {
            return Err(AkitaError::InvalidProof);
        }
    }
    Ok(())
}

const DIGIT_CHUNK_BITS: usize = 12;

/// Evaluate digit bytes without materializing their whole challenge-field lift.
fn digit_multilinear_eval<E: Field>(digits: &[u8], point: &[E]) -> Result<E, AkitaError> {
    // Match multilinear_eval's size errors before allocating either table.
    u32::try_from(point.len()).map_err(|_| AkitaError::InvalidSize {
        expected: usize::BITS as usize,
        actual: point.len(),
    })?;
    let expected = checked::pow2(point.len()).ok_or(AkitaError::InvalidSize {
        expected: usize::MAX,
        actual: digits.len(),
    })?;
    if digits.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: digits.len(),
        });
    }

    let chunk_bits = point.len().min(DIGIT_CHUNK_BITS);
    let chunk_len = checked::pow2(chunk_bits).ok_or(AkitaError::InvalidProof)?;
    let partial_len = checked::exact_div(expected, chunk_len).ok_or(AkitaError::InvalidProof)?;
    let low_point = point.get(..chunk_bits).ok_or(AkitaError::InvalidProof)?;
    let high_point = point.get(chunk_bits..).ok_or(AkitaError::InvalidProof)?;
    // Both the equality builder and multilinear_eval use little-endian variables:
    // each contiguous chunk varies over the low coordinates point[..chunk_bits].
    let weights =
        EqPolynomial::evals_serial(low_point, None).map_err(|_| AkitaError::InvalidProof)?;
    let mut partials = Vec::new();
    partials
        .try_reserve_exact(partial_len)
        .map_err(|_| AkitaError::InvalidProof)?;
    for chunk in digits.chunks_exact(chunk_len) {
        partials.push(
            chunk
                .iter()
                .zip(&weights)
                .fold(E::zero(), |sum, (&digit, &weight)| {
                    sum + E::from_u64(u64::from(digit)) * weight
                }),
        );
    }
    multilinear_eval(&partials, high_point)
}

/// Sends the whole table; not succinct; no hiding.
/// The image digit table is public, the response table is received as raw
/// bytes and the prime left opening as canonical elements of `E` over `F`.
/// No alphabet check occurs at this commitment boundary.
pub struct TransparentRootVerifierOracle<'image, F, E> {
    image: &'image [u8],
    response: Vec<u8>,
    prime: Vec<E>,
    base: PhantomData<fn() -> F>,
}
impl<'image, F, E> TransparentRootVerifierOracle<'image, F, E> {
    pub fn new(image: &'image [u8]) -> Self {
        Self {
            image,
            response: Vec::new(),
            prime: Vec::new(),
            base: PhantomData,
        }
    }
    pub fn response(&self) -> &[u8] {
        &self.response
    }
}
impl<F, E> RootVerifierOracle<E> for TransparentRootVerifierOracle<'_, F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        bind_transparent_image(layout, self.image, channel)
    }
    fn bind_prime_opening<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.prime = crate::lowered::zero_vec(layout.prime_len())?;
        exchange_transparent_prime_opening::<F, E, S>(layout, &mut self.prime, channel)
    }
    fn bind_response<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.response = super::zero_vec(layout.witness_len())?;
        channel.message(&mut self.response)
    }
    fn discharge<S: ClearChannel>(
        &mut self,
        claims: &RootEvaluationClaims<E>,
        _channel: &mut S,
    ) -> Result<(), AkitaError> {
        check_transparent_evaluations(self.image, &self.response, &self.prime, claims)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        check_transparent_evaluations, digit_multilinear_eval, RootEvaluationClaims,
        DIGIT_CHUNK_BITS,
    };
    use akita_algebra::{poly::multilinear_eval, One, Ring};
    use akita_error::{checked, AkitaError};

    type F = jolt_field::Prime128Offset275;

    fn random_word(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(2685821657736338717)
    }

    #[test]
    fn byte_evaluation_matches_lifted_tables_across_chunk_boundary() {
        let mut state = 0xc82a_1483_071f_d952;
        for n in 0..=DIGIT_CHUNK_BITS + 3 {
            for _ in 0..3 {
                let len = checked::pow2(n).unwrap_or(0);
                assert_ne!(len, 0);
                let digits: Vec<u8> = (0..len)
                    .map(|_| (random_word(&mut state) >> 56) as u8)
                    .collect();
                if n > DIGIT_CHUNK_BITS {
                    let chunk_len = checked::pow2(DIGIT_CHUNK_BITS).unwrap_or(0);
                    assert_ne!(chunk_len, 0);
                    let mut chunks = digits.chunks_exact(chunk_len);
                    assert_ne!(chunks.next(), chunks.next());
                }
                let point: Vec<F> = (0..n)
                    .map(|_| {
                        F::from_u128(
                            u128::from(random_word(&mut state))
                                | (u128::from(random_word(&mut state)) << 64),
                        )
                    })
                    .collect();
                let lifted: Vec<F> = digits
                    .iter()
                    .map(|&digit| F::from_u64(u64::from(digit)))
                    .collect();
                let expected = multilinear_eval(&lifted, &point);
                assert!(expected.is_ok());
                assert_eq!(
                    digit_multilinear_eval::<F>(&digits, &point),
                    expected,
                    "n={n}"
                );
            }
        }
    }

    #[test]
    fn transparent_evaluation_rejects_wrong_values_and_points() {
        let image = [3, 9];
        let digits = [2, 7];
        let prime = [F::from_u64(4), F::from_u64(10)];
        let claims = RootEvaluationClaims {
            response_point: vec![F::from_u64(3)],
            response_value: F::from_u64(17),
            image_point: vec![F::from_u64(5)],
            image_value: F::from_u64(33),
            prime: Some((vec![F::from_u64(2)], F::from_u64(16))),
        };
        let check = |prime: &[F], claims: &RootEvaluationClaims<F>| {
            check_transparent_evaluations(&image, &digits, prime, claims)
        };
        assert_eq!(check(&prime, &claims), Ok(()));
        let wrong: [fn(&mut RootEvaluationClaims<F>); 6] = [
            |claims| claims.response_value += F::one(),
            |claims| claims.response_point = vec![F::from_u64(4)],
            |claims| claims.image_value += F::one(),
            |claims| claims.image_point = vec![F::from_u64(6)],
            |claims| claims.prime = Some((vec![F::from_u64(2)], F::from_u64(17))),
            |claims| claims.prime = Some((vec![F::from_u64(3)], F::from_u64(16))),
        ];
        for mutate in wrong {
            let mut claims = claims.clone();
            mutate(&mut claims);
            assert_eq!(check(&prime, &claims), Err(AkitaError::InvalidProof));
        }
        // A prime opening entry changed after the claim was fixed.
        assert_eq!(
            check(&[F::from_u64(4), F::from_u64(11)], &claims),
            Err(AkitaError::InvalidProof)
        );
        // Without a prime claim the prime table is not consulted.
        let binary = RootEvaluationClaims {
            prime: None,
            ..claims
        };
        assert_eq!(check(&[], &binary), Ok(()));
    }

    #[test]
    fn byte_evaluation_preserves_multilinear_size_errors() {
        for n in [0, 1, 3, usize::BITS as usize] {
            let point = vec![F::from_u64(19); n];
            for len in [0, 3, 4, 9] {
                let digits = vec![2; len];
                let lifted = vec![F::from_u64(2); len];
                let expected = multilinear_eval(&lifted, &point);
                assert!(matches!(expected, Err(AkitaError::InvalidSize { .. })));
                assert_eq!(digit_multilinear_eval(&digits, &point), expected);
                let claims = RootEvaluationClaims {
                    response_point: point.clone(),
                    response_value: F::from_u64(2),
                    image_point: Vec::new(),
                    image_value: F::one(),
                    prime: None,
                };
                assert_eq!(
                    check_transparent_evaluations(&[1], &digits, &[], &claims),
                    expected.map(|_| ())
                );
            }
        }
    }
}
