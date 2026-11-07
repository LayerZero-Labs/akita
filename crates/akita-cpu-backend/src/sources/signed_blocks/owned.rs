use crate::commitment::{
    AcceptedReach, AvailablePolynomialTypes, CommitSourceClass, CommitSourceDescriptor,
    CommitmentSource, DenseType, PolynomialRepresentation, PolynomialType, PolynomialTypeSelection,
    SignedBlockDecoder, SignedBlocks, SignedValues,
};
use crate::opaque::CommitInnerPlan;
use crate::sources::poly::{RootOpeningSource, RootPolyMeta, RootPolyShape, SourceCoefficients};
use akita_error::AkitaError;
use akita_params::SUPPORTED_COMMITMENT_RING_DIMS;
use jolt_algebra::{CanonicalEncoding, Field};
use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

enum SignedBlockStorage {
    I8(Vec<i8>),
    I16(Vec<i16>),
    DecodedI8(Arc<dyn SignedBlockDecoder>),
}

/// Owned signed coefficients retained by [`crate::CpuBackend::import_source`].
///
/// Coefficients follow [`crate::DensePoly::from_field_evals`] order and are
/// zero-extended from [`Self::len`] to the full Boolean domain.
pub struct SignedBlockPoly {
    storage: SignedBlockStorage,
    len: usize,
    num_vars: usize,
    extrema: OnceLock<(i64, i64)>,
}

impl SignedBlockPoly {
    /// Own signed eight-bit coefficients, zero-extended past their length.
    pub fn i8(values: Vec<i8>, num_vars: usize) -> Result<Self, AkitaError> {
        let blocks = SignedBlocks::i8(&values, num_vars)?;
        let len = blocks.len();
        Ok(Self::new(SignedBlockStorage::I8(values), len, num_vars))
    }

    /// Own signed sixteen-bit coefficients, zero-extended past their length.
    pub fn i16(values: Vec<i16>, num_vars: usize) -> Result<Self, AkitaError> {
        let blocks = SignedBlocks::i16(&values, num_vars)?;
        let len = blocks.len();
        Ok(Self::new(SignedBlockStorage::I16(values), len, num_vars))
    }

    /// Retain a deterministic decoder for `len` signed eight-bit coefficients.
    pub fn decoded_i8(
        decoder: Arc<dyn SignedBlockDecoder>,
        len: usize,
        num_vars: usize,
    ) -> Result<Self, AkitaError> {
        SignedBlocks::decoded_i8(decoder.as_ref(), len, num_vars)?;
        Ok(Self::new(
            SignedBlockStorage::DecodedI8(decoder),
            len,
            num_vars,
        ))
    }

    fn new(storage: SignedBlockStorage, len: usize, num_vars: usize) -> Self {
        Self {
            storage,
            len,
            num_vars,
            extrema: OnceLock::new(),
        }
    }

    /// Borrow this source for the signed commitment and opening kernels.
    pub fn blocks(&self) -> SignedBlocks<'_> {
        SignedBlocks {
            values: match &self.storage {
                SignedBlockStorage::I8(values) => SignedValues::I8(values),
                SignedBlockStorage::I16(values) => SignedValues::I16(values),
                SignedBlockStorage::DecodedI8(decoder) => SignedValues::DecodedI8(decoder.as_ref()),
            },
            len: self.len,
            num_vars: self.num_vars,
            extrema: Some(&self.extrema),
        }
    }

    /// Logical multilinear variable count.
    pub const fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Number of stored coefficients before zero extension.
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(&self) -> usize {
        self.len
    }

    fn visit_values(&self, mut visit: impl FnMut(i64)) {
        match &self.storage {
            SignedBlockStorage::I8(values) => {
                values.iter().for_each(|&value| visit(i64::from(value)));
            }
            SignedBlockStorage::I16(values) => {
                values.iter().for_each(|&value| visit(i64::from(value)));
            }
            SignedBlockStorage::DecodedI8(decoder) => {
                let mut buffer = [0i8; 1024];
                for start in (0..self.len).step_by(buffer.len()) {
                    let count = buffer.len().min(self.len - start);
                    decoder.decode_i8(start, &mut buffer[..count]);
                    buffer[..count]
                        .iter()
                        .for_each(|&value| visit(i64::from(value)));
                }
            }
        }
    }
}

impl<F: Field> CommitmentSource<F> for SignedBlockPoly {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        let logical_len = 1usize << self.num_vars;
        let minimum = SUPPORTED_COMMITMENT_RING_DIMS[SUPPORTED_COMMITMENT_RING_DIMS.len() - 1];
        CommitSourceDescriptor::new(
            self.num_vars,
            logical_len.max(minimum),
            logical_len,
            CommitSourceClass::Dense,
            "akita_signed_blocks",
        )
    }

    fn committed_centered_reach(
        &self,
        modulus: u128,
        centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError>
    where
        F: CanonicalEncoding,
    {
        if modulus == 0 {
            return Err(AkitaError::InvalidInput(
                "field modulus must be nonzero".into(),
            ));
        }
        let mapping = AcceptedReach {
            negative_abs: None,
            positive: None,
            modulus,
            centering_threshold,
        };
        let (min, max) = self.blocks().extrema();
        let (negative_abs, positive) = (u128::from(min.unsigned_abs()), max.unsigned_abs().into());
        // Every value in `min..=max` is its own centered representative, so
        // the extrema are the reach; otherwise centering wraps and each value
        // counts.
        let top = centering_threshold.min(modulus - 1);
        if positive <= top && negative_abs < modulus - top {
            return Ok((negative_abs, positive));
        }
        let mut reach = (0u128, 0u128);
        self.visit_values(|value| {
            let centered = mapping.centered(value);
            reach.0 = reach.0.max(centered.0);
            reach.1 = reach.1.max(centered.1);
        });
        Ok(reach)
    }

    fn available_polynomial_types(
        &self,
        _plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError> {
        Ok(AvailablePolynomialTypes::signed_blocks())
    }

    fn represent_as(
        &self,
        selected: PolynomialTypeSelection,
        _plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        if selected.polynomial_type() != PolynomialType::Dense(DenseType::SignedBlocks) {
            return Err(AkitaError::InvalidInput(
                "signed block source received an unadvertised representation selection".into(),
            ));
        }
        Ok(PolynomialRepresentation::signed_blocks(self.blocks()))
    }
}

impl<F: Field> RootPolyMeta<F> for SignedBlockPoly {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    #[cfg(feature = "response-model-diagnostics")]
    fn exact_integer_coeff_l2_sq(&self) -> Option<u128> {
        let mut sum = Some(0u128);
        self.visit_values(|value| {
            let magnitude = u128::from(value.unsigned_abs());
            sum = sum.and_then(|sum| sum.checked_add(magnitude * magnitude));
        });
        sum
    }
}

impl<F: Field, const D: usize> RootPolyShape<F, D> for SignedBlockPoly {
    fn num_ring_elems(&self) -> usize {
        (1usize << self.num_vars).div_ceil(D)
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field, const D: usize> RootOpeningSource<F, D> for SignedBlockPoly {
    type OpeningView<'a> = SignedBlocks<'a>;
    type OpeningBatchView<'a> = Vec<SignedBlocks<'a>>;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        if D == 0 || !D.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "signed block ring dimension must be a power of two".into(),
            ));
        }
        Ok(self.blocks())
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
        polys
            .iter()
            .map(|poly| <Self as RootOpeningSource<F, D>>::opening_view(poly))
            .collect()
    }
}

/// Coefficients per task of the field conversion.
const CONVERT_CHUNK: usize = 1 << 14;

fn convert<F: Field, V: Copy + Into<i64> + Sync>(values: &[V], out: &mut [F]) {
    let chunk = |(out, values): (&mut [F], &[V])| {
        for (dst, &value) in out.iter_mut().zip(values) {
            *dst = F::from_i64(value.into());
        }
    };
    #[cfg(feature = "parallel")]
    out.par_chunks_mut(CONVERT_CHUNK)
        .zip(values.par_chunks(CONVERT_CHUNK))
        .for_each(chunk);
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(CONVERT_CHUNK)
        .zip(values.chunks(CONVERT_CHUNK))
        .for_each(chunk);
}

impl<F: Field> SourceCoefficients<F> for SignedBlockPoly {
    fn source_coefficients(&self) -> Result<Cow<'_, [F]>, AkitaError> {
        let mut coefficients = vec![F::zero(); 1usize << self.num_vars];
        let live = &mut coefficients[..self.len];
        match &self.storage {
            SignedBlockStorage::I8(values) => convert(values, live),
            SignedBlockStorage::I16(values) => convert(values, live),
            SignedBlockStorage::DecodedI8(decoder) => {
                let chunk = |(index, out): (usize, &mut [F])| {
                    let mut buffer = [0i8; CONVERT_CHUNK];
                    let values = &mut buffer[..out.len()];
                    decoder.decode_i8(index * CONVERT_CHUNK, values);
                    convert(values, out);
                };
                #[cfg(feature = "parallel")]
                live.par_chunks_mut(CONVERT_CHUNK)
                    .enumerate()
                    .for_each(chunk);
                #[cfg(not(feature = "parallel"))]
                live.chunks_mut(CONVERT_CHUNK).enumerate().for_each(chunk);
            }
        }
        Ok(Cow::Owned(coefficients))
    }
}
