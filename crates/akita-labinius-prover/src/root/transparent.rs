use akita_error::AkitaError;
use akita_labinius_verifier::{
    channel::ClearChannel,
    lowered::LoweredRootLayout,
    root::{
        bind_transparent_image, check_transparent_evaluations, exchange_transparent_prime_opening,
        RootEvaluationClaims, RootProverOracle,
    },
};
use core::marker::PhantomData;
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// Differential table oracle: sends the whole table; not succinct; no hiding.
///
/// Every image digit is public transcript input. The response commitment
/// transmits one byte per digit without checking the alphabet, and the prime
/// left opening is transmitted as canonical elements of `E` over `F`.
/// Evaluation discharge evaluates every table directly in little-endian
/// variable order.
pub struct TransparentRootProverOracle<'image, F, E> {
    image: &'image [u8],
    response: Vec<u8>,
    prime: Vec<E>,
    base: PhantomData<fn() -> F>,
}

impl<'image, F, E> TransparentRootProverOracle<'image, F, E> {
    /// Borrow the image digit table; its admitted length is checked at
    /// statement binding.
    pub fn new(image: &'image [u8]) -> Self {
        Self {
            image,
            response: Vec::new(),
            prime: Vec::new(),
            base: PhantomData,
        }
    }

    /// Committed response bytes, available after response binding.
    pub fn response(&self) -> &[u8] {
        &self.response
    }
}

impl<F, E> RootProverOracle<E> for TransparentRootProverOracle<'_, F, E>
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

    fn commit_prime_opening<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        table: &[E],
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.prime.clear();
        self.prime.try_reserve_exact(table.len()).map_err(|_| {
            AkitaError::InvalidInput("transparent prime opening allocation failed".into())
        })?;
        self.prime.extend_from_slice(table);
        exchange_transparent_prime_opening::<F, E, S>(layout, &mut self.prime, channel)
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
        claims: &RootEvaluationClaims<E>,
        _channel: &mut S,
    ) -> Result<(), AkitaError> {
        check_transparent_evaluations(self.image, &self.response, &self.prime, claims)
    }
}
