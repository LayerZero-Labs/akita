use super::BinarySourceDescriptor;
use akita_error::AkitaError;

/// Exact context that an application proof must discharge. Session/group IDs
/// come from the caller's authenticated transcript. The terminal digest binds
/// ordered prime points, values, plane manifest and ownership consumed by that
/// same opening, using canonical field encodings and domain separation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceAgreementBinding {
    pub session: [u8; 32],
    pub group: [u8; 32],
    pub source: BinarySourceDescriptor,
    pub terminal_prime_claims: [u8; 32],
}

/// Integration trust boundary, implemented only by the application's verified
/// proof token. Implementations MUST attest signed-bit membership and zero on
/// every committed padding coefficient of the extracted rational source modulo
/// the base prime, linked to this exact binding. Returning a matching binding
/// without verifying that statement does not establish agreement. Akita supplies
/// no implementation, proof checker, boolean flag, or application reduction.
///
/// For P64 the coefficient lift is exact since `2*Gamma + Delta = 4279 < p`.
/// Its prime-unit argument uses `ord_243(p) = 162`: `Phi_243` is irreducible
/// over F_p, so every nonzero challenge difference of degree below 162 is a
/// unit. Its coefficients have magnitude at most two and remain nonzero mod p.
/// The inverse survives scalar extension to E64. P128 retains the norm
/// argument using `ord_243(P) = 6` and `P^6 > 276^81`.
pub trait SourceAgreementProof {
    fn binding(&self) -> &SourceAgreementBinding;
}

/// Separate readings authenticated by the eventual grouped verifier. This
/// contract-only value is not itself a proof-verification capability.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryOpening<B, P> {
    binary_reading: B,
    prime_reading: P,
    binding: SourceAgreementBinding,
}
impl<B, P> BinaryOpening<B, P> {
    /// Integration constructor: call only after authenticating both readings
    /// and consuming all terminal claims in the bound session. This constructor
    /// does not run a verifier; calling it does not authenticate the readings.
    pub fn from_authenticated_readings(
        binary_reading: B,
        prime_reading: P,
        binding: SourceAgreementBinding,
    ) -> Self {
        Self {
            binary_reading,
            prime_reading,
            binding,
        }
    }
    pub fn binary_reading(&self) -> &B {
        &self.binary_reading
    }
    pub fn prime_reading(&self) -> &P {
        &self.prime_reading
    }
    pub fn binding(&self) -> &SourceAgreementBinding {
        &self.binding
    }
    /// Return the readings under the conditional agreement theorem, retaining
    /// the token's lifetime. No equality between differently typed readings is
    /// asserted: their logical signed-bit source is the same.
    pub fn agreement_requires<'a, T: SourceAgreementProof>(
        &'a self,
        proof: &'a T,
    ) -> Result<AgreedBinaryOpening<'a, B, P, T>, AkitaError> {
        if proof.binding() != &self.binding {
            return Err(AkitaError::InvalidProof);
        }
        Ok(AgreedBinaryOpening {
            opening: self,
            proof,
        })
    }
}
/// Conditional agreement tied to both the authenticated opening and proof token.
#[derive(Debug)]
pub struct AgreedBinaryOpening<'a, B, P, T: SourceAgreementProof> {
    opening: &'a BinaryOpening<B, P>,
    proof: &'a T,
}

impl<'a, B, P, T: SourceAgreementProof> AgreedBinaryOpening<'a, B, P, T> {
    pub fn opening(&self) -> &'a BinaryOpening<B, P> {
        self.opening
    }
    pub fn proof(&self) -> &'a T {
        self.proof
    }
}
