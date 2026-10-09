use crate::{channel::ClearChannel, lowered::LoweredRootLayout};
use akita_algebra::{poly::multilinear_eval, SmoothFftField};
use akita_error::AkitaError;

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
    let mut response = Vec::new();
    response
        .try_reserve_exact(digits.len())
        .map_err(|_| AkitaError::InvalidProof)?;
    response.extend(digits.iter().map(|&digit| F::from_u64(u64::from(digit))));
    if multilinear_eval(&response, &claims.response_point)? != claims.response_value
        || multilinear_eval(image, &claims.image_point)? != claims.image_value
    {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
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
