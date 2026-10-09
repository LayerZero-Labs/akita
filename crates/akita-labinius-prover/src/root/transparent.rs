use akita_algebra::SmoothFftField;
use akita_error::AkitaError;
use akita_labinius_verifier::{
    channel::ClearChannel,
    lowered::LoweredRootLayout,
    root::{
        bind_transparent_image, check_transparent_evaluations, RootEvaluationClaims,
        RootProverOracle,
    },
};

/// Differential table oracle: sends the whole table; not succinct; no hiding.
///
/// Every image coefficient is public transcript input. The response commitment
/// transmits one byte per digit without checking the alphabet. Evaluation
/// discharge evaluates both tables directly in little-endian variable order.
pub struct TransparentRootProverOracle<'image, F> {
    image: &'image [F],
    response: Vec<u8>,
}

impl<'image, F> TransparentRootProverOracle<'image, F> {
    /// Borrow the image table; its admitted length is checked at statement binding.
    pub fn new(image: &'image [F]) -> Self {
        Self {
            image,
            response: Vec::new(),
        }
    }

    /// Public image coefficients in table order, including padding.
    pub fn image(&self) -> &[F] {
        self.image
    }

    /// Committed response bytes, available after response binding.
    pub fn response(&self) -> &[u8] {
        &self.response
    }
}

impl<F: SmoothFftField> RootProverOracle<F> for TransparentRootProverOracle<'_, F> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        bind_transparent_image(layout, self.image, channel)
    }

    fn commit_response<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        digits: &[u8],
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        if digits.len() != layout.witness_len() {
            return Err(AkitaError::InvalidSize {
                expected: layout.witness_len(),
                actual: digits.len(),
            });
        }
        self.response.clear();
        self.response.try_reserve_exact(digits.len()).map_err(|_| {
            AkitaError::InvalidInput("transparent response allocation failed".into())
        })?;
        self.response.extend_from_slice(digits);
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
