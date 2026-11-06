//! Whiteout's compressed images on Akita (`specs/whiteout-akita.md` §5.4;
//! Whiteout §2, "Commitment compression"). An image is cut into the fewest
//! balanced chunks of at most the largest source a certified chain
//! compresses (15,115 coefficients at q128, the paper's `FCHAIN_MAX_IMAGE`;
//! clear B/D images keep the 8 KiB cap), and each chunk runs Akita's
//! two-map negative-binary chain: `ξ₁ = Bin(t)`, `j = K₁ξ₁`, `ξ₂ = Bin(j)`,
//! `cm = K₂ξ₂`, with `K₁` and `K₂` the offset-0 prefix of the setup stream.
//! A chunk's witness is `ξ₁` then `ξ₂`, each in Akita's bit-major order and
//! padded with zero digits to its map's ring row; its rows are the chain
//! equations `K₁ξ₁ − G(ξ₂) = 0` and `K₂ξ₂ = cm`, each evaluated at `α` as
//! Akita's reduced relation evaluates its own compression maps
//! ([`ReducedCompressionRelationWeights`]), and the response proof batches
//! them with jolt-zk's rows. The prover's chain is Akita's compression
//! executor.

use akita_config::CommitmentConfig;
use akita_cpu_backend::CpuBackend;
use akita_error::AkitaError;
use akita_params::{
    CompressionChainPlan, CompressionMapPlan, PackedNegativeBinary, RingRelationMode,
    SisModulusProfileId,
};
use akita_types::{
    AkitaExpandedSetup, NegativeBinarySupport, ReducedCompressionRelationWeights, RingVec,
};
#[cfg(test)]
use jolt_algebra::Ring;
use jolt_algebra::{lift_ct, One, Zero};
use jolt_zk::masking::ZkError;
use jolt_zk::whiteout::{
    CompressedImage, CompressedTarget, ImageCompression, ResponseStatement, WhiteoutError,
};

use super::{CompressionShape, F};

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

    /// Witness digits: `ξ₁` then `ξ₂`, each padded to its map's ring row.
    pub(crate) fn witness_len(&self) -> Result<usize, AkitaError> {
        let [first, second] = self.maps()?;
        Ok(first.padded_digit_count() + second.padded_digit_count())
    }

    /// The bits of each image coefficient in `ξ₁`.
    fn bits(&self) -> Result<usize, AkitaError> {
        let [first, _] = self.maps()?;
        let bits = self.plan.field_bits();
        if first.real_digit_count() != bits * self.coefficients.len() {
            return Err(AkitaError::InvalidSetup(
                "compression map digits disagree with its source".into(),
            ));
        }
        Ok(bits)
    }
}

/// Chain rows per chunk: `K₁ξ₁ − G(ξ₂)`, then `K₂ξ₂ − cm`.
const CHUNK_ROWS: usize = 2;

/// The setup field elements every chain reads: the chain of a full chunk,
/// whose maps are the widest.
pub(crate) fn setup_len(profile: SisModulusProfileId) -> Result<usize, AkitaError> {
    CompressionChainPlan::for_certified_source(
        profile,
        CompressionChainPlan::max_certified_source_coefficients(profile)?,
    )?
    .max_setup_field_elements()
}

/// The chunks of an image of `image_len` coefficients: the fewest the
/// largest source a certified chain compresses allows (not the 8 KiB cap of
/// clear B/D images), balanced so their lengths differ by at most one and
/// each first map runs below its SIS cell's width.
pub(crate) fn chunks(
    profile: SisModulusProfileId,
    image_len: usize,
) -> Result<Vec<Chunk>, AkitaError> {
    let count = image_len.div_ceil(CompressionChainPlan::max_certified_source_coefficients(
        profile,
    )?);
    let mut start = 0;
    (0..count)
        .map(|index| {
            let len = image_len / count + usize::from(index < image_len % count);
            let coefficients = start..start + len;
            start += len;
            Ok(Chunk {
                plan: CompressionChainPlan::for_certified_source(profile, len)?,
                coefficients,
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
    let chunks = chunks(profile, image.len())?;
    // The witness is secret: reserve it exactly, so no reallocation leaves
    // an unwiped copy behind.
    let witness_len = chunks
        .iter()
        .flat_map(|chunk| chunk.plan.maps())
        .map(|map| map.padded_digit_count())
        .sum();
    let mut compressed = CompressedImage {
        payload: Vec::new(),
        witness: Vec::with_capacity(witness_len),
    };
    for chunk in chunks {
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
            compressed.witness.resize(
                compressed.witness.len() + stage.map().padded_digit_count()
                    - stage.map().real_digit_count(),
                0,
            );
        }
    }
    Ok(compressed)
}

/// The `{−1, 0}` digits of one packed stage, its real digits only.
pub(crate) fn digits(stage: &PackedNegativeBinary) -> impl Iterator<Item = i8> + '_ {
    (0..stage.map().real_digit_count())
        .map(|index| -i8::from(stage.bytes()[index / 8] >> (index % 8) & 1 == 1))
}

/// `Σ_b 2^b ξ[b·n + k]` for every `k < n`: Akita's bit-major gadget. The
/// digits are secret (a masked image's bits), so each is lifted without a
/// branch on its sign.
fn recompose(digits: &[i8], n: usize) -> Vec<F> {
    let mut out = vec![F::zero(); n];
    let mut power = F::one();
    for plane in digits.chunks_exact(n.max(1)) {
        for (dst, &digit) in out.iter_mut().zip(plane) {
            *dst += power * lift_ct::<F>(i64::from(digit));
        }
        power += power;
    }
    out
}

/// The transpose of [`recompose`] over `bits` planes: digit `b·n + k`
/// takes `2^b w_k`.
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
        let len = chunk.witness_len()?;
        if rest.len() < len {
            return Err(AkitaError::InvalidSize {
                expected: len,
                actual: rest.len(),
            });
        }
        let (digits, tail) = rest.split_at(len);
        rest = tail;
        let xi_1 = chunk.bits()? * chunk.coefficients.len();
        image.extend(recompose(
            digits
                .get(..xi_1)
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
        recompose_transpose(
            weights
                .get(chunk.coefficients.clone())
                .ok_or_else(|| AkitaError::Internal("chunk outside the image".into()))?,
            chunk.bits()?,
            out.get_mut(start..)
                .ok_or_else(|| AkitaError::Internal("chunk witness".into()))?,
        );
    }
    Ok(out)
}

/// `Σ_c K[c]·x[c]` in `Z_q[X]/(X^d + 1)`, `K[c]` the `c`-th `d`-coefficient
/// block of `stream` and `x[c]` digits `c·d..(c + 1)·d` of `digits`: the
/// schoolbook definition of a compression map.
#[cfg(test)]
pub(crate) fn apply_row(stream: &[F], d: usize, digits: &[i8]) -> Vec<F> {
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

/// The chain rows of one chunk at `witness`, coefficient by coefficient:
/// `K₁ξ₁ − G(ξ₂)` and `K₂ξ₂ − payload`. The definition of the rows that
/// [`Chains::relation`] evaluates at `α`.
#[cfg(test)]
pub(crate) fn chain_rows(
    stream: &[F],
    chunk: &Chunk,
    witness: &[i8],
    payload: &[F],
) -> Result<[Vec<F>; CHUNK_ROWS], AkitaError> {
    let [first, second] = chunk.maps()?;
    let (xi_1, xi_2) = witness.split_at(first.padded_digit_count());
    let mut rows = apply_row(stream, first.ring_dimension(), xi_1);
    for (row, g) in rows.iter_mut().zip(recompose(
        &xi_2[..chunk.plan.field_bits() * first.output_coefficients()],
        first.output_coefficients(),
    )) {
        *row -= g;
    }
    let mut second_rows = apply_row(stream, second.ring_dimension(), xi_2);
    for (row, target) in second_rows.iter_mut().zip(payload) {
        *row -= *target;
    }
    Ok([rows, second_rows])
}

/// A Whiteout error carrying an Akita failure's message: a size or input
/// mismatch is the statement's fault, so the verifier rejects it.
pub(crate) fn whiteout_error(error: AkitaError) -> WhiteoutError {
    match error {
        AkitaError::InvalidProof | AkitaError::InvalidSize { .. } | AkitaError::InvalidInput(_) => {
            WhiteoutError::ResponseRejected("compressed image")
        }
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
/// whose shared matrix holds `K₁` and `K₂`.
pub(crate) struct Chains<'a> {
    profile: SisModulusProfileId,
    setup: &'a AkitaExpandedSetup<F>,
    targets: Vec<CompressedTarget<'a, F>>,
}

impl<'a> Chains<'a> {
    pub(crate) fn new<const D: usize>(
        profile: SisModulusProfileId,
        setup: &'a AkitaExpandedSetup<F>,
        statement: &'a ResponseStatement<'_, F, D>,
    ) -> Result<Self, ZkError> {
        Ok(Self {
            profile,
            setup,
            targets: statement.compressed_targets()?,
        })
    }

    /// The chains of `targets` alone.
    #[cfg(test)]
    pub(crate) fn of_targets(
        profile: SisModulusProfileId,
        setup: &'a AkitaExpandedSetup<F>,
        targets: Vec<CompressedTarget<'a, F>>,
    ) -> Self {
        Self {
            profile,
            setup,
            targets,
        }
    }

    /// Every target's chunks, with the first witness digit of each in `e`
    /// and its payload.
    fn chunks(&self) -> Result<Vec<PlacedChunk<'a>>, AkitaError> {
        let mut out = Vec::new();
        for target in &self.targets {
            let (mut digit, mut payload) = (target.digits.start, target.payload);
            for chunk in chunks(self.profile, target.image_len)? {
                let terminal = chunk.plan.terminal_coefficients();
                if payload.len() < terminal {
                    return Err(AkitaError::InvalidSize {
                        expected: terminal,
                        actual: payload.len(),
                    });
                }
                let (head, tail) = payload.split_at(terminal);
                payload = tail;
                let len = chunk.witness_len()?;
                out.push(PlacedChunk {
                    chunk,
                    digit,
                    payload: head,
                });
                digit += len;
            }
            if !payload.is_empty() || digit != target.digits.end {
                return Err(AkitaError::InvalidSize {
                    expected: target.digits.end,
                    actual: digit,
                });
            }
        }
        Ok(out)
    }

    /// The number of chain rows.
    pub(crate) fn rows(&self) -> Result<usize, AkitaError> {
        Ok(self.chunks()?.len() * CHUNK_ROWS)
    }

    /// `⟨weights, chain rows at α⟩` over the padded `e` of
    /// `physical_field_len` entries, as Akita's reduced compression
    /// relation.
    pub(crate) fn relation(
        &self,
        alpha: F,
        weights: &[F],
        physical_field_len: usize,
    ) -> Result<ReducedCompressionRelationWeights<F>, AkitaError> {
        let placed = self.chunks()?;
        ReducedCompressionRelationWeights::for_standalone_chains::<F>(
            alpha,
            &placed
                .iter()
                .map(|placed| (placed.chunk.plan.clone(), placed.digit))
                .collect::<Vec<_>>(),
            weights,
            physical_field_len,
        )
    }

    /// The constant `−Σ w₂·cm(α)` of `⟨weights, chain rows at α⟩`.
    pub(crate) fn constant(&self, alpha: F, weights: &[F]) -> Result<F, AkitaError> {
        let placed = self.chunks()?;
        if weights.len() != placed.len() * CHUNK_ROWS {
            return Err(AkitaError::InvalidSize {
                expected: placed.len() * CHUNK_ROWS,
                actual: weights.len(),
            });
        }
        Ok(placed.iter().zip(weights.chunks_exact(CHUNK_ROWS)).fold(
            F::zero(),
            |constant, (placed, weights)| {
                let at_alpha = placed
                    .payload
                    .iter()
                    .rev()
                    .fold(F::zero(), |sum, coefficient| sum * alpha + *coefficient);
                constant - weights[CHUNK_ROWS - 1] * at_alpha
            },
        ))
    }

    /// `Σ_j A_j·eq(point, j)` for the weights [`Self::accumulate`] adds.
    pub(crate) fn evaluate(&self, alpha: F, weights: &[F], point: &[F]) -> Result<F, AkitaError> {
        let len = u32::try_from(point.len())
            .ok()
            .and_then(|bits| 1usize.checked_shl(bits))
            .ok_or_else(|| AkitaError::InvalidInput("evaluation point width".into()))?;
        self.relation(alpha, weights, len)?
            .evaluate_at_point(self.setup, point)
    }

    /// `⟨weights, chain rows at α⟩`'s weights over `e` padded to
    /// `padded_len`, on `window`, which must hold every chain's digits.
    pub(crate) fn accumulate(
        &self,
        alpha: F,
        weights: &[F],
        padded_len: usize,
        window: std::ops::Range<usize>,
    ) -> Result<Vec<F>, AkitaError> {
        let mut a = vec![F::zero(); window.len()];
        self.relation(alpha, weights, padded_len)?
            .accumulate_dense(self.setup, window.start, &mut a)?;
        Ok(a)
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

/// Akita's image compression on a prover backend whose setup holds the
/// chains' maps: the response prover's (`specs/whiteout-akita.md` A47). It
/// runs in constant time in the image (A36). Public under `test-support`
/// only, for tests that compress without a response prover.
#[cfg_attr(
    not(feature = "test-support"),
    expect(unreachable_pub, reason = "public under test-support only")
)]
pub struct BackendCompression<'a, Cfg> {
    backend: &'a CpuBackend<F, F>,
    config: std::marker::PhantomData<Cfg>,
}

impl<'a, Cfg> BackendCompression<'a, Cfg> {
    /// Compression on `backend`; compressing fails if its setup does not
    /// hold the chains' maps.
    #[cfg_attr(
        not(feature = "test-support"),
        expect(unreachable_pub, reason = "public under test-support only")
    )]
    pub fn new(backend: &'a CpuBackend<F, F>) -> Self {
        Self {
            backend,
            config: std::marker::PhantomData,
        }
    }
}

impl<Cfg: CommitmentConfig> ImageCompression<F> for BackendCompression<'_, Cfg> {
    fn payload_len(&self, image_len: usize) -> Result<usize, WhiteoutError> {
        CompressionShape::<Cfg>::default().payload_len(image_len)
    }

    fn witness_len(&self, image_len: usize) -> Result<usize, WhiteoutError> {
        CompressionShape::<Cfg>::default().witness_len(image_len)
    }

    fn compress(&self, image: &[F]) -> Result<CompressedImage<F>, WhiteoutError> {
        compress(self.backend, Cfg::sis_modulus_profile(), image).map_err(whiteout_error)
    }

    fn reconstruct(&self, image_len: usize, witness: &[i8]) -> Result<Vec<F>, WhiteoutError> {
        CompressionShape::<Cfg>::default().reconstruct(image_len, witness)
    }

    fn reconstruct_transpose(
        &self,
        image_len: usize,
        weights: &[F],
    ) -> Result<Vec<F>, WhiteoutError> {
        CompressionShape::<Cfg>::default().reconstruct_transpose(image_len, weights)
    }

    /// The canonical witness only: Akita's chain is deterministic in the
    /// image, so `witness` opens `payload` exactly when compressing the
    /// image it reconstructs gives both back. The verifier enforces the
    /// chain equations and the binary digits instead (§5.7).
    fn check(&self, image_len: usize, payload: &[F], witness: &[i8]) -> Result<(), WhiteoutError> {
        let compressed = self.compress(&self.reconstruct(image_len, witness)?)?;
        if compressed.payload == payload && compressed.witness == witness {
            Ok(())
        } else {
            Err(WhiteoutError::ResponseRejected("compressed image"))
        }
    }
}
