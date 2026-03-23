//! Sumcheck proof container.

use crate::algebra::uni_poly::CompressedUniPoly;
use crate::error::HachiError;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::protocol::transcript::labels;
use crate::protocol::transcript::Transcript;
use crate::FieldCore;
use std::io::{Read, Write};

/// Sumcheck proof containing one compressed univariate polynomial per round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckProof<E: FieldCore> {
    /// One compressed univariate polynomial per sumcheck round.
    pub round_polys: Vec<CompressedUniPoly<E>>,
}

impl<E: Valid + FieldCore> Valid for SumcheckProof<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.round_polys.check()
    }
}

impl<E: FieldCore> HachiSerialize for SumcheckProof<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.round_polys.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.round_polys.serialized_size(compress)
    }
}

impl<E: FieldCore + Valid> HachiDeserialize for SumcheckProof<E> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let round_polys =
            Vec::<CompressedUniPoly<E>>::deserialize_with_mode(&mut reader, compress, validate)?;
        let out = Self { round_polys };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<E: FieldCore> SumcheckProof<E> {
    /// Verifier-side sumcheck transcript driver.
    ///
    /// This method:
    /// - absorbs the per-round prover message (compressed univariate),
    /// - samples one challenge per round via `sample_challenge`,
    /// - updates the running claim using `eval_from_hint`.
    ///
    /// It does **not** perform the final oracle check `final_claim == f(r*)`.
    /// Callers (e.g. ring-switching) must compute `f(r*)` themselves and compare.
    ///
    /// # Errors
    ///
    /// Returns an error if the proof length does not match `num_rounds` or if any
    /// per-round polynomial exceeds `degree_bound`.
    pub fn verify<F, T, S>(
        &self,
        mut claim: E,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut T,
        mut sample_challenge: S,
    ) -> Result<(E, Vec<E>), HachiError>
    where
        F: crate::FieldCore + crate::CanonicalField,
        T: Transcript<F>,
        S: FnMut(&mut T) -> E,
    {
        if self.round_polys.len() != num_rounds {
            return Err(HachiError::InvalidSize {
                expected: num_rounds,
                actual: self.round_polys.len(),
            });
        }

        let mut r = Vec::with_capacity(num_rounds);
        for poly in &self.round_polys {
            if poly.degree() > degree_bound {
                return Err(HachiError::InvalidInput(format!(
                    "sumcheck round poly degree {} exceeds bound {}",
                    poly.degree(),
                    degree_bound
                )));
            }

            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
            let r_i = sample_challenge(transcript);
            r.push(r_i);

            claim = poly.eval_from_hint(&claim, &r_i);
        }

        Ok((claim, r))
    }
}
