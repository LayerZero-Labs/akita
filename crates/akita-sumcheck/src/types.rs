//! Sumcheck proof containers and round-message types.

use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_field::Field;
use jolt_poly::{CompressedPoly, OmittedConstantPoly};
use std::io::{Read, Write};

/// Sumcheck proof containing one compressed univariate polynomial per round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckProof<E: Field> {
    /// One compressed univariate polynomial per sumcheck round.
    pub round_polys: Vec<CompressedPoly<E>>,
}

impl<E: Valid + Field> Valid for SumcheckProof<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.round_polys.check()
    }
}

/// Shape context for deserializing a [`SumcheckProof`].
///
/// Each entry is the number of serialized coefficients in the corresponding
/// compressed round polynomial. Round polynomials are headerless and need not
/// all have the same compact degree, though the prover currently emits a
/// uniform degree per sumcheck.
pub type SumcheckProofShape = Vec<usize>;

/// Construct a sumcheck shape for proofs whose rounds all use the same compact
/// coefficient count.
pub fn uniform_sumcheck_shape(num_rounds: usize, degree: usize) -> SumcheckProofShape {
    vec![degree; num_rounds]
}

impl<E: Field + AkitaSerialize> AkitaSerialize for SumcheckProof<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for poly in &self.round_polys {
            poly.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.round_polys
            .iter()
            .map(|p| p.serialized_size(compress))
            .sum()
    }
}

impl<E: Field + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for SumcheckProof<E> {
    /// Per-round compact coefficient counts.
    type Context = SumcheckProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &SumcheckProofShape,
    ) -> Result<Self, SerializationError> {
        let mut round_polys = Vec::new();
        round_polys.try_reserve_exact(ctx.len()).map_err(|_| {
            SerializationError::InvalidData("sumcheck proof allocation failed".to_string())
        })?;
        for degree in ctx {
            round_polys.push(CompressedPoly::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                degree,
            )?);
        }
        let out = Self { round_polys };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// Eq-factored sumcheck proof containing one compressed inner polynomial per round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EqFactoredSumcheckProof<E: Field> {
    /// One eq-factored inner polynomial per sumcheck round.
    pub round_polys: Vec<OmittedConstantPoly<E>>,
}

impl<E: Valid + Field> Valid for EqFactoredSumcheckProof<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.round_polys.check()
    }
}

/// Shape context for deserializing an [`EqFactoredSumcheckProof`]:
/// `(num_rounds, q_degree)`.
pub type EqFactoredSumcheckProofShape = (usize, usize);

impl<E: Field + AkitaSerialize> AkitaSerialize for EqFactoredSumcheckProof<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for poly in &self.round_polys {
            poly.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.round_polys
            .iter()
            .map(|poly| poly.serialized_size(compress))
            .sum()
    }
}

impl<E: Field + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for EqFactoredSumcheckProof<E>
{
    /// `(num_rounds, q_degree)` — number of round polynomials and the degree of `q`.
    type Context = EqFactoredSumcheckProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &EqFactoredSumcheckProofShape,
    ) -> Result<Self, SerializationError> {
        let (num_rounds, degree) = *ctx;
        let mut round_polys = Vec::new();
        round_polys.try_reserve_exact(num_rounds).map_err(|_| {
            SerializationError::InvalidData(
                "eq-factored sumcheck proof allocation failed".to_string(),
            )
        })?;
        for _ in 0..num_rounds {
            round_polys.push(OmittedConstantPoly::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &degree,
            )?);
        }
        let out = Self { round_polys };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
