use crate::{channel::ClearChannel, lowered::LoweredRootLayout};
use akita_algebra::{poly::multilinear_eval, EqPolynomial, SmoothFftField};
use akita_error::{checked, AkitaError};

/// The two polynomial-opening obligations returned by the reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootEvaluationClaims<F> {
    pub response_point: Vec<F>,
    pub response_value: F,
    pub image_point: Vec<F>,
    pub image_value: F,
}

/// Commitment and evaluation boundary for a root verifier.
/// Implementations MUST bind the exact image table before frontend challenges,
/// bind a response table before coefficient challenges, and authenticate both
/// evaluations with their own soundness error. Every method runs exactly once.
///
/// Both bindings MUST fix every table entry, padding included, in the layout's
/// table order and over the reduction's coefficient field `F`. `discharge`
/// runs after both evaluation messages, so every opening challenge it draws
/// follows them. A scheme that lets the two claims refer to different tables
/// does not satisfy this trait. The normative statement is the "Oracle
/// contract" section of `specs/labinius-root-reduction.md`.
pub trait RootVerifierOracle<F: SmoothFftField> {
    fn bind_image<S: ClearChannel>(
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
        claims: &RootEvaluationClaims<F>,
        channel: &mut S,
    ) -> Result<(), AkitaError>;
}

/// Prover boundary matching the verifier's three oracle call sites.
/// The response commitment MUST fix the supplied digit bytes; the evaluation
/// proof MUST refer to the same response and image owners that were bound.
///
/// The obligations of [`RootVerifierOracle`] apply to the matching prover
/// calls: the same coefficient field, the same table order, every padding
/// entry bound, and no opening challenge before both evaluation messages.
pub trait RootProverOracle<F: SmoothFftField> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
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
        claims: &RootEvaluationClaims<F>,
        channel: &mut S,
    ) -> Result<(), AkitaError>;
}

/// Bind every padded image coefficient as canonical public bytes, in table order.
pub fn bind_transparent_image<F: SmoothFftField, S: ClearChannel>(
    layout: &LoweredRootLayout,
    image: &[F],
    channel: &mut S,
) -> Result<(), AkitaError> {
    if image.len() != layout.image_len() {
        return Err(AkitaError::InvalidInput(
            "transparent image length mismatch".into(),
        ));
    }
    let mut bytes = super::zero_vec::<u8>(F::NUM_BYTES)?;
    for value in image {
        value.to_bytes_le(&mut bytes);
        channel.public(&bytes)?;
    }
    Ok(())
}

/// Authenticate both little-endian multilinear evaluations against whole tables.
/// There is deliberately no alphabet validation here: that is a sumcheck term.
pub fn check_transparent_evaluations<F: SmoothFftField>(
    image: &[F],
    digits: &[u8],
    claims: &RootEvaluationClaims<F>,
) -> Result<(), AkitaError> {
    if response_multilinear_eval(digits, &claims.response_point)? != claims.response_value
        || multilinear_eval(image, &claims.image_point)? != claims.image_value
    {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

const RESPONSE_CHUNK_BITS: usize = 12;

/// Evaluate digit bytes without materializing their whole coefficient-field lift.
fn response_multilinear_eval<F: SmoothFftField>(
    digits: &[u8],
    point: &[F],
) -> Result<F, AkitaError> {
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

    let chunk_bits = point.len().min(RESPONSE_CHUNK_BITS);
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
                .fold(F::zero(), |sum, (&digit, &weight)| {
                    sum + F::from_u64(u64::from(digit)) * weight
                }),
        );
    }
    multilinear_eval(&partials, high_point)
}

/// Sends the whole table; not succinct; no hiding.
/// The image table is public and the response table is received as raw bytes.
/// No alphabet check occurs at this commitment boundary.
pub struct TransparentRootVerifierOracle<'image, F> {
    image: &'image [F],
    response: Vec<u8>,
}
impl<'image, F> TransparentRootVerifierOracle<'image, F> {
    pub fn new(image: &'image [F]) -> Self {
        Self {
            image,
            response: Vec::new(),
        }
    }
    pub fn response(&self) -> &[u8] {
        &self.response
    }
}
impl<F: SmoothFftField> RootVerifierOracle<F> for TransparentRootVerifierOracle<'_, F> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        bind_transparent_image(layout, self.image, channel)
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
        claims: &RootEvaluationClaims<F>,
        _channel: &mut S,
    ) -> Result<(), AkitaError> {
        check_transparent_evaluations(self.image, &self.response, claims)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        check_transparent_evaluations, response_multilinear_eval, RootEvaluationClaims,
        RESPONSE_CHUNK_BITS,
    };
    use akita_algebra::{poly::multilinear_eval, One, Ring};
    use akita_error::{checked, AkitaError};

    type F = jolt_field::Prime128OffsetA7F7;

    fn random_word(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(2685821657736338717)
    }

    #[test]
    fn byte_evaluation_matches_lifted_tables_across_chunk_boundary() {
        let mut state = 0xc82a_1483_071f_d952;
        for n in 0..=RESPONSE_CHUNK_BITS + 3 {
            for _ in 0..3 {
                let len = checked::pow2(n).unwrap_or(0);
                assert_ne!(len, 0);
                let digits: Vec<u8> = (0..len)
                    .map(|_| (random_word(&mut state) >> 56) as u8)
                    .collect();
                if n > RESPONSE_CHUNK_BITS {
                    let chunk_len = checked::pow2(RESPONSE_CHUNK_BITS).unwrap_or(0);
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
                    response_multilinear_eval::<F>(&digits, &point),
                    expected,
                    "n={n}"
                );
            }
        }
    }

    #[test]
    fn transparent_evaluation_rejects_wrong_values_and_points() {
        let image = [F::from_u64(3), F::from_u64(9)];
        let digits = [2, 7];
        let mut claims = RootEvaluationClaims {
            response_point: vec![F::from_u64(3)],
            response_value: F::from_u64(17),
            image_point: vec![F::from_u64(5)],
            image_value: F::from_u64(33),
        };
        assert_eq!(
            check_transparent_evaluations(&image, &digits, &claims),
            Ok(())
        );
        claims.response_value += F::one();
        assert!(matches!(
            check_transparent_evaluations(&image, &digits, &claims),
            Err(AkitaError::InvalidProof)
        ));
        claims.response_value -= F::one();
        claims.response_point = vec![F::from_u64(4)];
        assert!(matches!(
            check_transparent_evaluations(&image, &digits, &claims),
            Err(AkitaError::InvalidProof)
        ));
        claims.response_point = vec![F::from_u64(3)];
        claims.image_value += F::one();
        assert!(matches!(
            check_transparent_evaluations(&image, &digits, &claims),
            Err(AkitaError::InvalidProof)
        ));
        claims.image_value -= F::one();
        claims.image_point = vec![F::from_u64(6)];
        assert!(matches!(
            check_transparent_evaluations(&image, &digits, &claims),
            Err(AkitaError::InvalidProof)
        ));
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
                assert_eq!(response_multilinear_eval(&digits, &point), expected);
                let claims = RootEvaluationClaims {
                    response_point: point.clone(),
                    response_value: F::from_u64(2),
                    image_point: Vec::new(),
                    image_value: F::one(),
                };
                assert_eq!(
                    check_transparent_evaluations(&[F::one()], &digits, &claims),
                    expected.map(|_| ())
                );
            }
        }
    }
}
