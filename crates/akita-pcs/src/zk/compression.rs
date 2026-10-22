//! Whiteout's compressed images on Akita (`specs/whiteout-akita.md` §5.4;
//! Whiteout §2, "Commitment compression"). An image is cut into chunks of
//! at most Akita's compression input cap, and each chunk runs Akita's
//! two-map negative-binary chain: `ξ₁ = Bin(t)`, `j = K₁ξ₁`, `ξ₂ = Bin(j)`,
//! `cm = K₂ξ₂`, with `K₁` and `K₂` the offset-0 prefix of the setup stream.
//! A chunk's witness is `ξ₁` then `ξ₂`, each in Akita's bit-major order;
//! its rows are the chain equations `K₁ξ₁ − G(ξ₂) = 0` and `K₂ξ₂ = cm`,
//! which the response proof batches with jolt-zk's rows. The prover's
//! chain is Akita's compression executor.

use akita_cpu_backend::CpuBackend;
use akita_error::AkitaError;
use akita_params::{
    CompressionChainPlan, CompressionMapPlan, PackedNegativeBinary, RingRelationMode,
    SisModulusProfileId, MAX_COMPRESSION_INPUT_BYTES,
};
use akita_types::{NegativeBinarySupport, RingVec};
use jolt_algebra::{CanonicalBytes, One, Ring, Zero};
use jolt_zk::masking::ZkError;
use jolt_zk::whiteout::{CompressedImage, CompressedTarget, ResponseStatement, WhiteoutError};

use super::F;

/// One chunk of an image and its chain.
pub(crate) struct Chunk {
    /// Coefficients of the image.
    pub(crate) coefficients: std::ops::Range<usize>,
    pub(crate) plan: CompressionChainPlan,
}

impl Chunk {
    /// The chain's two maps.
    fn maps(&self) -> Result<[CompressionMapPlan; 2], AkitaError> {
        <[CompressionMapPlan; 2]>::try_from(self.plan.maps())
            .map_err(|_| AkitaError::InvalidSetup("compression chain is not two maps".into()))
    }

    /// Witness digits: `ξ₁` then `ξ₂`.
    pub(crate) fn witness_len(&self) -> Result<usize, AkitaError> {
        let [first, second] = self.maps()?;
        Ok(first.real_digit_count() + second.real_digit_count())
    }

    /// Chain rows: `K₁ξ₁ − G(ξ₂)`, then `K₂ξ₂ − cm`.
    pub(crate) fn rows(&self) -> Result<usize, AkitaError> {
        let [first, second] = self.maps()?;
        Ok(first.output_coefficients() + second.output_coefficients())
    }
}

/// The chunks of an image of `image_len` coefficients.
pub(crate) fn chunks(
    profile: SisModulusProfileId,
    image_len: usize,
) -> Result<Vec<Chunk>, AkitaError> {
    let cap = MAX_COMPRESSION_INPUT_BYTES / F::NUM_BYTES;
    (0..image_len)
        .step_by(cap)
        .map(|start| {
            let end = (start + cap).min(image_len);
            Ok(Chunk {
                coefficients: start..end,
                plan: CompressionChainPlan::for_complete_source(profile, end - start)?,
            })
        })
        .collect()
}

/// `(cm(t), witness)` for `image`, by Akita's compression executor.
pub(crate) fn compress(
    backend: &CpuBackend<F, F>,
    profile: SisModulusProfileId,
    image: &[F],
) -> Result<CompressedImage<F>, AkitaError> {
    let mut compressed = CompressedImage {
        payload: Vec::new(),
        witness: Vec::new(),
    };
    for chunk in chunks(profile, image.len())? {
        let source = image
            .get(chunk.coefficients.clone())
            .ok_or_else(|| AkitaError::Internal("chunk outside the image".into()))?;
        let (payload, state) = backend
            .compress_relation_image(
                &chunk.plan,
                RingRelationMode::ReducedEvaluation,
                RingVec::from_coeffs(source.to_vec()),
            )?
            .into_parts();
        compressed.payload.extend_from_slice(payload.coeffs());
        for stage in state.witness().stages() {
            compressed.witness.extend(digits(stage));
        }
    }
    Ok(compressed)
}

/// The `{−1, 0}` digits of one packed stage, its real digits only.
fn digits(stage: &PackedNegativeBinary) -> impl Iterator<Item = i8> + '_ {
    (0..stage.map().real_digit_count())
        .map(|index| -i8::from(stage.bytes()[index / 8] >> (index % 8) & 1 == 1))
}

/// `Σ_b 2^b ξ[b·n + k]` for every `k < n`: Akita's bit-major gadget.
fn recompose(digits: &[i8], n: usize) -> Vec<F> {
    let mut out = vec![F::zero(); n];
    let mut power = F::one();
    for plane in digits.chunks_exact(n.max(1)) {
        for (dst, &digit) in out.iter_mut().zip(plane) {
            *dst += power * F::from_i64(i64::from(digit));
        }
        power += power;
    }
    out
}

/// The transpose of [`recompose`]: digit `b·n + k` takes `2^b w_k`.
fn recompose_transpose(weights: &[F], bits: usize, out: &mut [F]) {
    let n = weights.len();
    let mut power = F::one();
    for plane in out.chunks_exact_mut(n.max(1)).take(bits) {
        for (dst, weight) in plane.iter_mut().zip(weights) {
            *dst += power * *weight;
        }
        power += power;
    }
}

/// `t` from its witness.
pub(crate) fn reconstruct(
    profile: SisModulusProfileId,
    image_len: usize,
    witness: &[i8],
) -> Result<Vec<F>, AkitaError> {
    let mut image = Vec::with_capacity(image_len);
    let mut rest = witness;
    for chunk in chunks(profile, image_len)? {
        let [first, _] = chunk.maps()?;
        let len = chunk.witness_len()?;
        if rest.len() < len {
            return Err(AkitaError::InvalidSize {
                expected: len,
                actual: rest.len(),
            });
        }
        let (digits, tail) = rest.split_at(len);
        rest = tail;
        image.extend(recompose(
            digits
                .get(..first.real_digit_count())
                .ok_or_else(|| AkitaError::Internal("ξ₁ outside the chunk".into()))?,
            chunk.coefficients.len(),
        ));
    }
    if !rest.is_empty() {
        return Err(AkitaError::InvalidSize {
            expected: witness.len() - rest.len(),
            actual: witness.len(),
        });
    }
    Ok(image)
}

/// The weight on each witness digit of `⟨weights, t⟩`.
pub(crate) fn reconstruct_transpose(
    profile: SisModulusProfileId,
    image_len: usize,
    weights: &[F],
) -> Result<Vec<F>, AkitaError> {
    if weights.len() != image_len {
        return Err(AkitaError::InvalidSize {
            expected: image_len,
            actual: weights.len(),
        });
    }
    let mut out = Vec::new();
    for chunk in chunks(profile, image_len)? {
        let start = out.len();
        out.resize(start + chunk.witness_len()?, F::zero());
        let bits = F::NUM_BYTES * 8;
        recompose_transpose(
            weights
                .get(chunk.coefficients.clone())
                .ok_or_else(|| AkitaError::Internal("chunk outside the image".into()))?,
            bits,
            out.get_mut(start..)
                .ok_or_else(|| AkitaError::Internal("chunk witness".into()))?,
        );
    }
    Ok(out)
}

/// `Σ_c K[c]·x[c]` in `Z_q[X]/(X^d + 1)`, `K[c]` the `c`-th `d`-coefficient
/// block of `stream` and `x[c]` digits `c·d..(c + 1)·d` of `digits`
/// (zero past its end).
#[cfg(test)]
fn apply_row(stream: &[F], d: usize, digits: &[i8]) -> Vec<F> {
    let mut out = vec![F::zero(); d];
    for (column, x) in digits.chunks(d).enumerate() {
        let k = &stream[column * d..(column + 1) * d];
        for (i, &digit) in x.iter().enumerate() {
            let digit = F::from_i64(i64::from(digit));
            for (j, &entry) in k.iter().enumerate() {
                let product = entry * digit;
                if i + j < d {
                    out[i + j] += product;
                } else {
                    out[i + j - d] -= product;
                }
            }
        }
    }
    out
}

/// The transpose of `x ↦ Σ_c K[c]·x[c]`: digit `c·d + i` takes
/// `Σ_j ω_j·[X^j](K[c]·X^i)`, for the first `len` digits.
fn apply_row_transpose(
    stream: &[F],
    d: usize,
    weights: &[F],
    len: usize,
    out: &mut [F],
) -> Result<(), AkitaError> {
    let k_len = len.div_ceil(d) * d;
    let entries = stream.get(..k_len).ok_or_else(|| {
        AkitaError::InvalidSetup("setup stream shorter than a compression map".into())
    })?;
    let out = out
        .get_mut(..len)
        .ok_or_else(|| AkitaError::Internal("compression digits outside the witness".into()))?;
    for (k, out) in entries.chunks_exact(d).zip(out.chunks_mut(d)) {
        for (i, dst) in out.iter_mut().enumerate() {
            let mut weight = F::zero();
            for (j, &entry) in k.iter().enumerate() {
                // X^j · X^i lands on X^{i+j}, negated past X^d.
                let target = i + j;
                if target < d {
                    weight += weights.get(target).copied().unwrap_or_default() * entry;
                } else {
                    weight -= weights.get(target - d).copied().unwrap_or_default() * entry;
                }
            }
            *dst += weight;
        }
    }
    Ok(())
}

/// The chain rows of one chunk at `witness` minus their targets: `K₁ξ₁ −
/// G(ξ₂)` and `K₂ξ₂ − payload`. The definition of the rows that
/// [`chain_transpose`] transposes.
#[cfg(test)]
pub(crate) fn chain_rows(
    stream: &[F],
    chunk: &Chunk,
    witness: &[i8],
    payload: &[F],
) -> Result<Vec<F>, AkitaError> {
    let [first, second] = chunk.maps()?;
    let (xi_1, xi_2) = witness.split_at(first.real_digit_count());
    let mut rows = apply_row(stream, first.ring_dimension(), xi_1);
    for (row, g) in rows
        .iter_mut()
        .zip(recompose(xi_2, first.output_coefficients()))
    {
        *row -= g;
    }
    let mut second_rows = apply_row(stream, second.ring_dimension(), xi_2);
    for (row, target) in second_rows.iter_mut().zip(payload) {
        *row -= *target;
    }
    rows.extend(second_rows);
    Ok(rows)
}

/// The weight on each witness digit of `⟨weights, chain rows⟩`, and the
/// rows' constant `−⟨weights₂, payload⟩`.
pub(crate) fn chain_transpose(
    stream: &[F],
    chunk: &Chunk,
    weights: &[F],
    payload: &[F],
) -> Result<(Vec<F>, F), AkitaError> {
    let [first, second] = chunk.maps()?;
    let (first_weights, second_weights) =
        weights.split_at(first.output_coefficients().min(weights.len()));
    let mut out = vec![F::zero(); chunk.witness_len()?];
    let (xi_1, xi_2) = out.split_at_mut(first.real_digit_count());
    apply_row_transpose(
        stream,
        first.ring_dimension(),
        first_weights,
        first.real_digit_count(),
        xi_1,
    )?;
    let negated: Vec<F> = first_weights.iter().map(|w| -*w).collect();
    recompose_transpose(&negated, F::NUM_BYTES * 8, xi_2);
    apply_row_transpose(
        stream,
        second.ring_dimension(),
        second_weights,
        second.real_digit_count(),
        xi_2,
    )?;
    let constant = second_weights
        .iter()
        .zip(payload)
        .fold(F::zero(), |sum, (w, p)| sum - *w * *p);
    Ok((out, constant))
}

/// A Whiteout error carrying an Akita failure's message.
pub(crate) fn whiteout_error(error: AkitaError) -> WhiteoutError {
    match error {
        AkitaError::InvalidProof => WhiteoutError::ResponseRejected("compressed image"),
        _ => WhiteoutError::InvariantViolation("compression"),
    }
}

/// A chunk, its first witness digit in `e`, and its payload.
struct PlacedChunk<'a> {
    chunk: Chunk,
    digit: usize,
    payload: &'a [F],
}

/// The chains of every compressed image of a statement, against the setup
/// stream that holds `K₁` and `K₂`.
pub(crate) struct Chains<'a> {
    profile: SisModulusProfileId,
    stream: &'a [F],
    targets: Vec<CompressedTarget<'a, F>>,
}

impl<'a> Chains<'a> {
    pub(crate) fn new<const D: usize>(
        profile: SisModulusProfileId,
        stream: &'a [F],
        statement: &'a ResponseStatement<'_, F, D>,
    ) -> Result<Self, ZkError> {
        Ok(Self {
            profile,
            stream,
            targets: statement.compressed_targets()?,
        })
    }

    /// Every target's chunks, with the first witness digit of each in `e`
    /// and its payload.
    fn chunks(&self) -> Result<Vec<PlacedChunk<'a>>, AkitaError> {
        let mut out = Vec::new();
        for target in &self.targets {
            let (mut digit, mut payload) = (target.digits.start, target.payload);
            for chunk in chunks(self.profile, target.image_len)? {
                let (head, tail) =
                    payload.split_at(chunk.plan.terminal_coefficients().min(payload.len()));
                payload = tail;
                let len = chunk.witness_len()?;
                out.push(PlacedChunk {
                    chunk,
                    digit,
                    payload: head,
                });
                digit += len;
            }
        }
        Ok(out)
    }

    /// The number of chain rows.
    pub(crate) fn rows(&self) -> Result<usize, AkitaError> {
        self.chunks()?
            .iter()
            .map(|placed| placed.chunk.rows())
            .sum()
    }

    /// Adds to `a` (over `e`) the weights of `⟨weights, chain rows⟩`, in
    /// target and chunk order, and returns their constant.
    pub(crate) fn transpose(&self, weights: &[F], a: &mut [F]) -> Result<F, AkitaError> {
        let mut constant = F::zero();
        let mut rest = weights;
        for PlacedChunk {
            chunk,
            digit,
            payload,
        } in self.chunks()?
        {
            let (chunk_weights, tail) = rest.split_at(chunk.rows()?.min(rest.len()));
            rest = tail;
            let (digits, c) = chain_transpose(self.stream, &chunk, chunk_weights, payload)?;
            constant += c;
            let targets = a
                .get_mut(digit..digit + digits.len())
                .ok_or_else(|| AkitaError::Internal("chain digits outside e".into()))?;
            for (dst, w) in targets.iter_mut().zip(digits) {
                *dst += w;
            }
        }
        Ok(constant)
    }

    /// Where `ξ` sits in `e` padded to `padded_len`: the binary constraint's
    /// support `I_bin`.
    pub(crate) fn support(&self, padded_len: usize) -> Result<NegativeBinarySupport, AkitaError> {
        NegativeBinarySupport::from_intervals(
            self.targets
                .iter()
                .map(|target| target.digits.clone())
                .collect(),
            padded_len,
        )
    }
}

/// The length of the image whose compression has a `payload_len` payload
/// and a `witness_len` witness.
pub(crate) fn image_len(
    profile: SisModulusProfileId,
    payload_len: usize,
    witness_len: usize,
) -> Result<usize, AkitaError> {
    let cap = MAX_COMPRESSION_INPUT_BYTES / F::NUM_BYTES;
    let bits = F::NUM_BYTES * 8;
    let terminal = CompressionChainPlan::for_complete_source(profile, cap)?.terminal_coefficients();
    let chunks = payload_len.div_ceil(terminal.max(1));
    let full = CompressionChainPlan::for_complete_source(profile, cap)?;
    let tail = full.maps().get(1).map_or(0, |map| map.real_digit_count());
    witness_len
        .checked_sub(chunks * tail)
        .map(|digits| digits / bits)
        .ok_or_else(|| AkitaError::InvalidSize {
            expected: chunks * tail,
            actual: witness_len,
        })
}
