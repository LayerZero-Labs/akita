//! Shared setup data shapes for Akita prover and verifier APIs.

use super::setup_prefix::SetupPrefixVerifierRegistry;
use crate::FlatMatrix;
#[cfg(test)]
use akita_algebra::CyclotomicRing;
#[allow(unused_imports)]
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use rand_core::{CryptoRng, RngCore};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::Mutex;

/// Public seed used to derive commitment matrices.
pub type PublicMatrixSeed = [u8; 32];

/// Maximum setup matrix field elements accepted by self-describing setup
/// deserialization.
///
/// Config-backed cache paths should enforce tighter exact shape bounds before
/// decoding the matrix body. This cap protects generic verifier-facing setup
/// decoding from allocating directly from attacker-controlled seed metadata.
pub const MAX_SETUP_MATRIX_FIELD_ELEMENTS: usize = 1 << 26;

const PUBLIC_MATRIX_DOMAIN: &[u8] = b"akita/commitment/public-matrix-1d";
const SHARED_MATRIX_LABEL: &[u8] = b"shared";

/// Packed capacity envelope for the shared public setup vector.
///
/// The setup stores one flat vector of ring elements. A/B/D matrices are
/// role-local prefix views of this vector, so capacity is the maximum required
/// role footprint, not `max_rows * max_stride`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetupMatrixEnvelope {
    /// Number of generated ring elements at the setup generation dimension.
    pub max_setup_len: usize,
}

impl SetupMatrixEnvelope {
    /// Smallest non-empty shared setup envelope.
    pub const fn minimum() -> Self {
        Self {
            max_setup_len: std::num::NonZeroUsize::MIN.get(),
        }
    }
}

/// Seed-only stage for deterministic setup expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaSetupSeed {
    /// Maximum supported variable count.
    pub max_num_vars: usize,
    /// Maximum number of batched polynomials supported by setup.
    pub max_num_batched_polys: usize,
    /// Ring dimension used to generate `shared_matrix`.
    pub gen_ring_dim: usize,
    /// Number of generated ring elements at `gen_ring_dim`.
    pub max_setup_len: usize,
    /// Public seed used to derive commitment matrices.
    pub public_matrix_seed: PublicMatrixSeed,
}

impl AkitaSetupSeed {
    /// Number of field elements in the serialized shared matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the seed shape overflows `usize`.
    pub fn matrix_field_elements(&self) -> Result<usize, SerializationError> {
        self.max_setup_len
            .checked_mul(self.gen_ring_dim)
            .ok_or_else(|| {
                SerializationError::InvalidData(
                    "setup seed matrix field count overflow".to_string(),
                )
            })
    }
}

/// Expanded setup stage containing materialized public matrices.
///
/// Base role matrices (A, B, D) are packed row/column prefix views of
/// `shared_matrix`.
#[derive(Debug)]
pub struct AkitaExpandedSetup<F: FieldCore> {
    /// Setup seed and runtime layout metadata.
    pub seed: AkitaSetupSeed,
    /// Shared 1D flat backing vector, releasable to a prefix after its last
    /// full-width consumer (the commit sweep) has run.
    pub shared_matrix: SharedSetupMatrix<F>,
}

impl<F: FieldCore> Clone for AkitaExpandedSetup<F> {
    fn clone(&self) -> Self {
        Self {
            seed: self.seed.clone(),
            shared_matrix: self.shared_matrix.clone(),
        }
    }
}

/// Equality is by seed and declared shape: the matrix contents are a pure
/// function of the seed, so comparing the (possibly released) backing store
/// would be redundant — and prohibitively expensive at real setup sizes.
impl<F: FieldCore> PartialEq for AkitaExpandedSetup<F> {
    fn eq(&self, other: &Self) -> bool {
        self.seed == other.seed
    }
}

impl<F: FieldCore> Eq for AkitaExpandedSetup<F> {}

/// Shared coefficient-form setup matrix with a releasable backing store.
///
/// The matrix is a deterministic function of the public seed
/// ([`derive_public_matrix_flat`]), which makes the materialized form pure
/// cache: once the last full-width consumer has run, the store can be
/// released to a small prefix (serving slot rebuilds and verifier prefix
/// scans) while full-width readers either re-derive on demand or stream
/// per-element from the seed ([`Self::element_deriver`]).
///
/// Shape metadata (lengths, ring dimension) always reflects the FULL
/// seed-declared matrix regardless of what is materialized — envelope keys
/// and validation must not shrink when the store is released.
pub struct SharedSetupMatrix<F: FieldCore> {
    public_seed: PublicMatrixSeed,
    gen_ring_dim: usize,
    full_ring_elements: usize,
    cached: Mutex<Arc<FlatMatrix<F>>>,
    /// Monomorphized prefix derivation, captured at construction (the only
    /// place a `RandomSampling` bound exists) so re-materialization needs no
    /// extra bounds at ~all consumer sites.
    derive_prefix: fn(usize, usize, &PublicMatrixSeed) -> FlatMatrix<F>,
    /// Monomorphized single-entry derivation for streaming consumers.
    fill_entry: fn(&LabeledMatrixXof, usize, &mut [F]),
}

impl<F: FieldCore> core::fmt::Debug for SharedSetupMatrix<F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SharedSetupMatrix")
            .field("gen_ring_dim", &self.gen_ring_dim)
            .field("full_ring_elements", &self.full_ring_elements)
            .field(
                "materialized_ring_elements",
                &self.snapshot().total_ring_elements(),
            )
            .finish_non_exhaustive()
    }
}

impl<F: FieldCore> Clone for SharedSetupMatrix<F> {
    fn clone(&self) -> Self {
        Self {
            public_seed: self.public_seed,
            gen_ring_dim: self.gen_ring_dim,
            full_ring_elements: self.full_ring_elements,
            cached: Mutex::new(self.snapshot()),
            derive_prefix: self.derive_prefix,
            fill_entry: self.fill_entry,
        }
    }
}

fn fill_matrix_entry<F: FieldCore + RandomSampling>(
    xof: &LabeledMatrixXof,
    gen_index: usize,
    coeffs: &mut [F],
) {
    let mut entry_rng = xof.entry_rng(gen_index);
    for coeff in coeffs.iter_mut() {
        *coeff = F::random(&mut entry_rng);
    }
}

impl<F: FieldCore> SharedSetupMatrix<F> {
    /// Wrap a matrix the caller derived (or deserialized) from `public_seed`.
    ///
    /// This is the ONLY place a `RandomSampling` bound is required: the
    /// derivation functions are captured here as monomorphized pointers so
    /// later re-materialization is bound-free.
    #[must_use]
    pub fn from_seed_derived(public_seed: PublicMatrixSeed, matrix: FlatMatrix<F>) -> Self
    where
        F: RandomSampling,
    {
        Self {
            public_seed,
            gen_ring_dim: matrix.gen_ring_dim(),
            full_ring_elements: matrix.total_ring_elements(),
            cached: Mutex::new(Arc::new(matrix)),
            derive_prefix: derive_public_matrix_flat_prefix::<F>,
            fill_entry: fill_matrix_entry::<F>,
        }
    }

    fn snapshot(&self) -> Arc<FlatMatrix<F>> {
        Arc::clone(&self.cached.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Generation-time ring dimension (seed metadata, release-independent).
    #[must_use]
    pub fn gen_ring_dim(&self) -> usize {
        self.gen_ring_dim
    }

    /// FULL matrix length in generation-dimension ring elements, regardless
    /// of what is currently materialized.
    #[must_use]
    pub fn total_ring_elements(&self) -> usize {
        self.full_ring_elements
    }

    /// FULL matrix length viewed at ring dimension `D`.
    ///
    /// # Errors
    ///
    /// Returns an error when `D` does not evenly view the matrix.
    pub fn total_ring_elements_at<const D: usize>(&self) -> Result<usize, AkitaError> {
        self.total_ring_elements_at_dyn(D)
    }

    /// FULL matrix length viewed at runtime ring dimension `ring_d`.
    ///
    /// # Errors
    ///
    /// Returns an error when `ring_d` does not evenly view the matrix.
    pub fn total_ring_elements_at_dyn(&self, ring_d: usize) -> Result<usize, AkitaError> {
        let total_coeffs = self
            .full_ring_elements
            .checked_mul(self.gen_ring_dim)
            .ok_or_else(|| AkitaError::InvalidSetup("setup matrix length overflow".into()))?;
        if ring_d == 0 || total_coeffs % ring_d != 0 {
            return Err(AkitaError::InvalidSetup(format!(
                "ring dimension {ring_d} does not evenly view {total_coeffs} setup coefficients"
            )));
        }
        Ok(total_coeffs / ring_d)
    }

    /// Number of field coefficients currently materialized (diagnostics).
    #[must_use]
    pub fn materialized_field_elements(&self) -> usize {
        self.snapshot().total_ring_elements() * self.gen_ring_dim
    }

    /// The current store if (and only if) it already covers
    /// `num_ring_elements` at `ring_d` — never derives. Streaming consumers
    /// use this to prefer the materialized matrix pre-release and fall back
    /// to per-element seed derivation after it.
    #[must_use]
    pub fn materialized_covering_at_dyn(
        &self,
        num_ring_elements: usize,
        ring_d: usize,
    ) -> Option<Arc<FlatMatrix<F>>> {
        let needed_coeffs = num_ring_elements.checked_mul(ring_d)?;
        let cached = self.snapshot();
        (cached.total_ring_elements() * self.gen_ring_dim >= needed_coeffs).then_some(cached)
    }
}

impl<F: FieldCore> SharedSetupMatrix<F> {
    /// A backing store covering at least `num_ring_elements` at `ring_d`.
    ///
    /// Serves the cached store when it covers the request (pre-release this
    /// is the full matrix and costs an `Arc` clone); otherwise derives
    /// exactly the requested prefix from the seed WITHOUT growing the cache,
    /// so a release stays effective.
    ///
    /// # Errors
    ///
    /// Returns an error when the request does not evenly view the matrix or
    /// exceeds the full seed-declared length.
    pub fn covering_at_dyn(
        &self,
        num_ring_elements: usize,
        ring_d: usize,
    ) -> Result<Arc<FlatMatrix<F>>, AkitaError> {
        let full_at_d = self.total_ring_elements_at_dyn(ring_d)?;
        if num_ring_elements > full_at_d {
            return Err(AkitaError::InvalidSetup(format!(
                "requested {num_ring_elements} setup ring elements at D={ring_d}, but the setup has {full_at_d}"
            )));
        }
        let needed_coeffs = num_ring_elements * ring_d;
        let needed_gen = needed_coeffs.div_ceil(self.gen_ring_dim);
        let cached = self.snapshot();
        if cached.total_ring_elements() >= needed_gen {
            return Ok(cached);
        }
        if needed_gen >= self.full_ring_elements {
            // A full-width request re-caches the store (the verifier's
            // root-level setup scan legitimately reads the whole matrix):
            // later proves release it again post-commit, so within one
            // process the derive amortizes to once per prove/verify cycle.
            return Ok(self.full());
        }
        // Post-release consumers beyond the retained prefix pay a fresh
        // derivation per call; keep it visible so hot paths get moved to
        // per-element streaming (or a larger retained prefix) instead.
        tracing::info!(
            needed_gen_ring_elements = needed_gen,
            "re-deriving a released setup-matrix prefix"
        );
        Ok(Arc::new((self.derive_prefix)(
            needed_gen,
            self.gen_ring_dim,
            &self.public_seed,
        )))
    }

    /// The full backing store, re-deriving (and re-caching) it if released.
    ///
    /// Full-width consumers that run pre-release (the commit sweep) get the
    /// original store for an `Arc` clone; calling this post-release costs a
    /// full re-derivation — prefer [`Self::covering_at_dyn`] or
    /// [`Self::element_deriver`].
    #[must_use]
    pub fn full(&self) -> Arc<FlatMatrix<F>> {
        {
            let cached = self.cached.lock().unwrap_or_else(|e| e.into_inner());
            if cached.total_ring_elements() == self.full_ring_elements {
                return Arc::clone(&cached);
            }
        }
        tracing::info!("re-deriving the FULL released setup matrix");
        let rebuilt = Arc::new((self.derive_prefix)(
            self.full_ring_elements,
            self.gen_ring_dim,
            &self.public_seed,
        ));
        *self.cached.lock().unwrap_or_else(|e| e.into_inner()) = Arc::clone(&rebuilt);
        rebuilt
    }

    /// Release the backing store down to its first `keep_ring_elements`
    /// (generation dimension), returning the approximate bytes freed.
    ///
    /// In-flight readers holding the old `Arc` keep it alive until they
    /// finish; new readers see the prefix. Requests beyond the prefix
    /// re-derive from seed per call.
    pub fn release_to_prefix(&self, keep_ring_elements: usize) -> usize {
        let keep = keep_ring_elements.min(self.full_ring_elements);
        let mut cached = self.cached.lock().unwrap_or_else(|e| e.into_inner());
        let current = cached.total_ring_elements();
        if current <= keep {
            return 0;
        }
        let prefix = FlatMatrix::from_flat_data(
            cached.as_field_slice()[..keep * self.gen_ring_dim].to_vec(),
            self.gen_ring_dim,
        );
        *cached = Arc::new(prefix);
        (current - keep) * self.gen_ring_dim * core::mem::size_of::<F>()
    }

    /// Per-element derivation handle for streaming consumers.
    #[must_use]
    pub fn element_deriver(&self) -> MatrixElementDeriver<F> {
        MatrixElementDeriver {
            xof: LabeledMatrixXof::new(&self.public_seed, SHARED_MATRIX_LABEL),
            gen_ring_dim: self.gen_ring_dim,
            fill_entry: self.fill_entry,
        }
    }
}

/// Derives individual setup-matrix ring elements straight from the public
/// seed — the same per-entry XOF stream as [`derive_public_matrix_flat`], so
/// derived values are bit-identical to the materialized matrix.
pub struct MatrixElementDeriver<F: FieldCore> {
    xof: LabeledMatrixXof,
    gen_ring_dim: usize,
    fill_entry: fn(&LabeledMatrixXof, usize, &mut [F]),
}

impl<F: FieldCore> MatrixElementDeriver<F> {
    /// Generation-time ring dimension the entry indexing is defined over.
    #[must_use]
    pub fn gen_ring_dim(&self) -> usize {
        self.gen_ring_dim
    }

    /// Fill `coeffs` with the generation-dimension entry at `gen_index`.
    ///
    /// `coeffs.len()` must equal the generation ring dimension; callers
    /// viewing the matrix at a different ring dimension must map their
    /// indices to generation entries first.
    pub fn entry_coeffs(&self, gen_index: usize, coeffs: &mut [F]) {
        debug_assert_eq!(coeffs.len(), self.gen_ring_dim);
        (self.fill_entry)(&self.xof, gen_index, coeffs);
    }

    /// Derive an arbitrary contiguous coefficient range of the flat matrix
    /// (spanning generation entries as needed) — the random-access primitive
    /// for consumers whose reads stride across the full matrix width.
    #[must_use]
    pub fn coeff_range(&self, start_coeff: usize, len: usize) -> Vec<F> {
        let gen = self.gen_ring_dim;
        let first_entry = start_coeff / gen;
        let last_entry = (start_coeff + len).div_ceil(gen);
        let mut coeffs = vec![F::zero(); (last_entry - first_entry) * gen];
        for (offset, entry) in (first_entry..last_entry).enumerate() {
            self.entry_coeffs(entry, &mut coeffs[offset * gen..(offset + 1) * gen]);
        }
        let skip = start_coeff - first_entry * gen;
        coeffs.drain(..skip);
        coeffs.truncate(len);
        coeffs
    }
}

/// Verifier setup artifact derived from prover setup.
#[derive(Debug, Clone)]
pub struct AkitaVerifierSetup<F: FieldCore> {
    /// Expanded matrix stage used for verification.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
    /// Public setup-prefix commitment metadata for setup-claim offloading.
    pub prefix_slots: SetupPrefixVerifierRegistry<F>,
    /// Locally derived, negacyclic-only matrix prefixes for direct verifier checks.
    /// This performance cache is neither serialized nor part of setup identity.
    verifier_ntt: Arc<crate::ntt_cache::VerifierNttCache>,
}

impl<F: FieldCore> PartialEq for AkitaVerifierSetup<F> {
    fn eq(&self, other: &Self) -> bool {
        self.expanded == other.expanded && self.prefix_slots == other.prefix_slots
    }
}

impl<F: FieldCore> Eq for AkitaVerifierSetup<F> {}

impl<F: FieldCore> AkitaVerifierSetup<F> {
    /// Construct verifier setup state from validated expanded setup and prefix metadata.
    #[must_use]
    pub fn from_parts(
        expanded: Arc<AkitaExpandedSetup<F>>,
        prefix_slots: SetupPrefixVerifierRegistry<F>,
    ) -> Self {
        Self {
            expanded,
            prefix_slots,
            verifier_ntt: Arc::new(crate::ntt_cache::VerifierNttCache::default()),
        }
    }

    /// In-memory byte footprint of verifier NTT prefixes materialized so far.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache lock was poisoned.
    pub fn verifier_ntt_cache_bytes(&self) -> Result<usize, AkitaError> {
        self.verifier_ntt.cache_bytes()
    }
}

impl<F: FieldCore + CanonicalField> AkitaVerifierSetup<F> {
    /// Return an exact or covering negacyclic prefix, preparing it on demand.
    pub fn prepared_verifier_ntt_prefix<const D: usize>(
        &self,
        num_ring_elements: usize,
        tail_num_ring_elements: usize,
        width: usize,
        log_basis: u32,
    ) -> Result<Arc<crate::PreparedNttCache<D>>, AkitaError> {
        let key = crate::NttCacheKey {
            ring_d: D,
            num_ring_elements,
        };
        self.verifier_ntt.prepare::<F, D>(
            &self.expanded,
            key,
            tail_num_ring_elements,
            crate::NttCacheMode::ExactNegacyclic { width, log_basis },
        )
    }
}

impl<F: FieldCore> AkitaExpandedSetup<F> {
    /// Build an expanded setup from a trusted matrix the caller has already
    /// derived from `seed.public_matrix_seed`.
    ///
    /// This constructor deliberately does not rederive or validate the matrix. Use
    /// [`Self::from_verified_parts`] for untrusted serialized setup bytes.
    #[must_use]
    pub fn from_trusted_seed_derived_parts_unchecked(
        seed: AkitaSetupSeed,
        shared_matrix: FlatMatrix<F>,
    ) -> Self
    where
        F: RandomSampling,
    {
        let shared_matrix =
            SharedSetupMatrix::from_seed_derived(seed.public_matrix_seed, shared_matrix);
        Self {
            seed,
            shared_matrix,
        }
    }

    /// Setup seed and runtime layout metadata.
    #[must_use]
    pub fn seed(&self) -> &AkitaSetupSeed {
        &self.seed
    }

    /// Shared coefficient-form matrix backing all setup roles.
    #[must_use]
    pub fn shared_matrix(&self) -> &SharedSetupMatrix<F> {
        &self.shared_matrix
    }
}

impl<F> AkitaExpandedSetup<F>
where
    F: FieldCore + RandomSampling + Valid,
{
    /// Build an expanded setup from untrusted parts and verify the materialized
    /// matrix against the public seed.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if the seed/matrix shape is malformed or
    /// the matrix was not deterministically derived from the seed.
    pub fn from_verified_parts(
        seed: AkitaSetupSeed,
        shared_matrix: FlatMatrix<F>,
    ) -> Result<Self, SerializationError> {
        let out = Self::from_trusted_seed_derived_parts_unchecked(seed, shared_matrix);
        out.check()?;
        Ok(out)
    }
}

/// Fixed public seed for deterministic, reproducible setup.
#[must_use]
pub fn sample_public_matrix_seed() -> PublicMatrixSeed {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes());
    seed
}

/// Derive a flat public vector of ring elements from a seed, at a runtime
/// generation dimension and any prefix length.
///
/// All role matrices (A, B, D) share one backing vector with a fixed label
/// (`"shared"`). Each role views a prefix of this vector reshaped with its
/// own `(num_rows, num_cols)` dimensions.
///
/// Domain separation uses a single flat index so that a vector of length N is
/// a prefix of any vector of length M > N derived from the same seed —
/// which is also what makes partial re-materialization of a released store
/// byte-identical to the original.
#[tracing::instrument(skip_all, name = "derive_public_matrix_flat")]
#[must_use]
pub fn derive_public_matrix_flat_prefix<F: FieldCore + RandomSampling>(
    total_ring_elements: usize,
    gen_ring_dim: usize,
    seed: &PublicMatrixSeed,
) -> FlatMatrix<F> {
    let xof = LabeledMatrixXof::new(seed, SHARED_MATRIX_LABEL);
    let mut data = vec![F::zero(); total_ring_elements * gen_ring_dim];
    cfg_chunks_mut!(data, gen_ring_dim)
        .enumerate()
        .for_each(|(idx, coeffs)| fill_matrix_entry(&xof, idx, coeffs));
    FlatMatrix::from_flat_data(data, gen_ring_dim)
}

/// Const-dimension convenience wrapper over
/// [`derive_public_matrix_flat_prefix`].
#[must_use]
pub fn derive_public_matrix_flat<F: FieldCore + RandomSampling, const D: usize>(
    total_ring_elements: usize,
    seed: &PublicMatrixSeed,
) -> FlatMatrix<F> {
    derive_public_matrix_flat_prefix::<F>(total_ring_elements, D, seed)
}

/// Check that a materialized public matrix has exactly the shape declared by
/// `seed`.
///
/// # Errors
///
/// Returns an error if either side is structurally malformed or if the matrix
/// generation dimension / length differs from the seed.
pub fn validate_public_matrix_shape_matches_seed<F: FieldCore + Valid>(
    shared_matrix: &FlatMatrix<F>,
    seed: &AkitaSetupSeed,
) -> Result<(), SerializationError> {
    seed.check()?;
    shared_matrix.check()?;
    let gen_ring_dim = shared_matrix.gen_ring_dim();
    if gen_ring_dim != seed.gen_ring_dim {
        return Err(SerializationError::InvalidData(
            "setup shared_matrix generation dimension does not match setup seed".to_string(),
        ));
    }
    if shared_matrix.total_ring_elements() != seed.max_setup_len {
        return Err(SerializationError::InvalidData(
            "setup shared_matrix length does not match setup seed".to_string(),
        ));
    }
    Ok(())
}

/// Check that a materialized public matrix is exactly the deterministic matrix
/// derived from `seed`.
///
/// # Errors
///
/// Returns an error if the matrix shape is malformed or if any coefficient
/// differs from the seed-derived public matrix.
pub fn validate_public_matrix_matches_seed<F: FieldCore + RandomSampling + Valid>(
    shared_matrix: &FlatMatrix<F>,
    seed: &AkitaSetupSeed,
) -> Result<(), SerializationError> {
    validate_public_matrix_shape_matches_seed(shared_matrix, seed)?;
    let gen_ring_dim = shared_matrix.gen_ring_dim();
    let xof = LabeledMatrixXof::new(&seed.public_matrix_seed, SHARED_MATRIX_LABEL);
    let mut expected = vec![F::zero(); gen_ring_dim];
    for (idx, coeffs) in shared_matrix
        .as_field_slice()
        .chunks_exact(gen_ring_dim)
        .enumerate()
    {
        let mut entry_rng = xof.entry_rng(idx);
        for value in expected.iter_mut() {
            *value = F::random(&mut entry_rng);
        }
        if coeffs != expected.as_slice() {
            return Err(SerializationError::InvalidData(
                "setup shared_matrix does not match public matrix seed".to_string(),
            ));
        }
    }
    Ok(())
}

/// Concrete SHAKE256 XOF reader for public-matrix derivation. Naming it via the
/// `ExtendableOutput` associated type lets each per-element RNG hold the reader
/// inline instead of behind a `Box<dyn XofReader>`, removing one heap
/// allocation per derived ring element.
type PublicMatrixXofReader = <Shake256 as ExtendableOutput>::Reader;

struct ShakeXofRng {
    reader: PublicMatrixXofReader,
}

impl ShakeXofRng {
    /// Independent full-prefix constructor retained for tests that cross-check
    /// the prefix-reuse derivation against a from-scratch absorb.
    #[cfg(test)]
    fn new_labeled(seed: &PublicMatrixSeed, matrix_label: &[u8], indices: &[u64]) -> Self {
        let mut xof = Shake256::default();
        absorb_len_prefixed(&mut xof, b"domain", PUBLIC_MATRIX_DOMAIN);
        absorb_len_prefixed(&mut xof, b"seed", seed);
        absorb_len_prefixed(&mut xof, b"matrix", matrix_label);
        for index in indices {
            absorb_len_prefixed(&mut xof, b"index", &index.to_le_bytes());
        }
        Self {
            reader: xof.finalize_xof(),
        }
    }
}

/// Pre-absorbs the fixed `domain‖seed‖matrix` prefix of the public-matrix XOF
/// once. Each per-element RNG then clones the sponge state and absorbs only the
/// element index, so the absorbed byte stream (and therefore every derived ring
/// element) is bit-for-bit identical to absorbing the full prefix per element.
struct LabeledMatrixXof {
    base: Shake256,
}

impl LabeledMatrixXof {
    fn new(seed: &PublicMatrixSeed, matrix_label: &[u8]) -> Self {
        let mut base = Shake256::default();
        absorb_len_prefixed(&mut base, b"domain", PUBLIC_MATRIX_DOMAIN);
        absorb_len_prefixed(&mut base, b"seed", seed);
        absorb_len_prefixed(&mut base, b"matrix", matrix_label);
        Self { base }
    }

    fn entry_rng(&self, flat_index: usize) -> ShakeXofRng {
        let mut xof = self.base.clone();
        absorb_len_prefixed(&mut xof, b"index", &(flat_index as u64).to_le_bytes());
        ShakeXofRng {
            reader: xof.finalize_xof(),
        }
    }
}

impl RngCore for ShakeXofRng {
    fn next_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.fill_bytes(&mut buf);
        u32::from_le_bytes(buf)
    }

    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.fill_bytes(&mut buf);
        u64::from_le_bytes(buf)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        XofReader::read(&mut self.reader, dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for ShakeXofRng {}

fn absorb_len_prefixed(xof: &mut Shake256, label: &[u8], data: &[u8]) {
    xof.update(&(label.len() as u64).to_le_bytes());
    xof.update(label);
    xof.update(&(data.len() as u64).to_le_bytes());
    xof.update(data);
}

impl Valid for AkitaSetupSeed {
    fn check(&self) -> Result<(), SerializationError> {
        if self.max_num_batched_polys == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed max_num_batched_polys must be at least 1".to_string(),
            ));
        }
        if self.gen_ring_dim == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed gen_ring_dim must be non-zero".to_string(),
            ));
        }
        if self.max_setup_len == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed max_setup_len must be non-zero".to_string(),
            ));
        }
        self.matrix_field_elements()?;
        Ok(())
    }
}

impl AkitaSerialize for AkitaSetupSeed {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.max_num_vars
            .serialize_with_mode(&mut writer, compress)?;
        self.max_num_batched_polys
            .serialize_with_mode(&mut writer, compress)?;
        self.gen_ring_dim
            .serialize_with_mode(&mut writer, compress)?;
        self.max_setup_len
            .serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.public_matrix_seed)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_num_vars.serialized_size(compress)
            + self.max_num_batched_polys.serialized_size(compress)
            + self.gen_ring_dim.serialized_size(compress)
            + self.max_setup_len.serialized_size(compress)
            + 32
    }
}

impl AkitaDeserialize for AkitaSetupSeed {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let max_num_vars = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_num_batched_polys =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let gen_ring_dim = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_setup_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let mut public_matrix_seed = [0u8; 32];
        reader.read_exact(&mut public_matrix_seed)?;
        let out = Self {
            max_num_vars,
            max_num_batched_polys,
            gen_ring_dim,
            max_setup_len,
            public_matrix_seed,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + RandomSampling + Valid> Valid for AkitaExpandedSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.seed.check()?;
        let materialized = self.shared_matrix.full();
        materialized.check()?;
        validate_public_matrix_matches_seed(&materialized, &self.seed)?;
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for AkitaExpandedSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.seed.serialize_with_mode(&mut writer, compress)?;
        self.shared_matrix
            .full()
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        // Sizing must reflect the FULL matrix (serialization materializes
        // it), but must not permanently re-cache a released store.
        self.seed.serialized_size(compress)
            + self
                .shared_matrix
                .covering_at_dyn(
                    self.shared_matrix.total_ring_elements(),
                    self.shared_matrix.gen_ring_dim(),
                )
                .map(|matrix| matrix.serialized_size(compress))
                .unwrap_or(0)
    }
}

impl<F: FieldCore + RandomSampling + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for AkitaExpandedSetup<F>
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let seed = AkitaSetupSeed::deserialize_with_mode(&mut reader, compress, validate, &())?;
        seed.check()?;
        let shared_matrix = FlatMatrix::deserialize_with_expected_shape(
            &mut reader,
            compress,
            validate,
            seed.max_setup_len,
            seed.gen_ring_dim,
            MAX_SETUP_MATRIX_FIELD_ELEMENTS,
        )?;
        if matches!(validate, Validate::Yes) {
            Self::from_verified_parts(seed, shared_matrix)
        } else {
            Ok(Self::from_trusted_seed_derived_parts_unchecked(
                seed,
                shared_matrix,
            ))
        }
    }
}

impl<F: FieldCore + RandomSampling + Valid> Valid for AkitaVerifierSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()?;
        self.prefix_slots.check()
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for AkitaVerifierSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let mut writer = writer;
        self.expanded.serialize_with_mode(&mut writer, compress)?;
        self.prefix_slots.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.expanded.serialized_size(compress) + self.prefix_slots.serialized_size(compress)
    }
}

impl<F: FieldCore + RandomSampling + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for AkitaVerifierSetup<F>
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let mut reader = reader;
        Ok(Self {
            expanded: Arc::new(AkitaExpandedSetup::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?),
            prefix_slots: SetupPrefixVerifierRegistry::deserialize_with_mode(
                reader,
                compress,
                validate,
                &(),
            )?,
            verifier_ntt: Arc::new(crate::ntt_cache::VerifierNttCache::default()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp64, Prime128Offset275};

    type F = Prime128Offset275;
    const D: usize = 4;
    type SmallF = Fp64<4294967197>;
    const SMALL_D: usize = 64;

    fn prefix_commitment_params(n_prefix: usize, d_setup: usize) -> crate::PrecommittedLevelParams {
        crate::PrecommittedLevelParams {
            layout: crate::PrecommittedGroupDescriptor {
                group: crate::PolynomialGroupLayout::singleton(n_prefix.trailing_zeros() as usize),
                num_live_ring_elements_per_claim: n_prefix / d_setup,
                num_positions_per_block: 1,
                num_live_blocks: n_prefix / d_setup,
                log_basis_inner: 1,
                log_basis_outer: 1,
                inner_ring_dimension: d_setup,
                outer_ring_dimension: d_setup,
                n_a: 1,
                a_coeff_linf_bound: 1,
                n_b: 1,
                b_coeff_linf_bound: 1,
            },
            inner_commit_matrix: crate::InnerCommitMatrixParams::new_unchecked(
                crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                crate::sis::SisTableDigest::CURRENT,
                crate::sis::SisModulusProfileId::Q128OffsetA7F7,
                1,
                1,
                1,
                d_setup,
            ),
            outer_commit_matrix: crate::OuterCommitMatrixParams::new_unchecked(
                crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                crate::sis::SisTableDigest::CURRENT,
                crate::sis::SisModulusProfileId::Q128OffsetA7F7,
                1,
                1,
                1,
                d_setup,
            ),
            log_basis_open: 1,
            fold_challenge_config: akita_challenges::SparseChallengeConfig::pm1_only(0),
            num_digits_inner: 1,
            num_digits_outer: 1,
            num_digits_open: 1,
            num_digits_fold_one: 1,
        }
    }

    fn seed(public_matrix_seed: PublicMatrixSeed) -> AkitaSetupSeed {
        AkitaSetupSeed {
            max_num_vars: 8,
            max_num_batched_polys: 1,
            gen_ring_dim: D,
            max_setup_len: 2,
            public_matrix_seed,
        }
    }

    #[test]
    fn verifier_setup_prefix_slots_roundtrip() {
        use crate::proof::{RingVec, SetupPrefixPublicCommitment, SetupPrefixVerifierSlot};

        let setup_seed = seed([7u8; 32]);
        let shared_matrix = derive_public_matrix_flat::<F, D>(2, &setup_seed.public_matrix_seed);
        let mut prefix_slots = SetupPrefixVerifierRegistry::new();
        let slot = SetupPrefixVerifierSlot {
            id: crate::setup_prefix_slot_id(D, D - 1, prefix_commitment_params(D, D)),
            natural_len: D - 1,
            padded_len: D,
            commitment: SetupPrefixPublicCommitment {
                rows: vec![RingVec::from_coeffs(vec![F::zero(); D])],
            },
        };
        prefix_slots.insert(slot).expect("insert prefix slot");
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    setup_seed,
                    shared_matrix,
                ),
            ),
            prefix_slots,
            verifier_ntt: Arc::new(crate::ntt_cache::VerifierNttCache::default()),
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).expect("serialize");
        let decoded =
            AkitaVerifierSetup::<F>::deserialize_compressed(&bytes[..], &()).expect("deserialize");

        assert_eq!(decoded.prefix_slots.len(), 1);
        assert_eq!(decoded, setup);
    }

    #[test]
    fn strict_verifier_setup_decode_rejects_matrix_not_derived_from_seed() {
        let setup_seed = seed([7u8; 32]);
        let wrong_seed = [9u8; 32];
        let wrong_matrix = derive_public_matrix_flat::<F, D>(2, &wrong_seed);
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    setup_seed,
                    wrong_matrix,
                ),
            ),
            prefix_slots: SetupPrefixVerifierRegistry::new(),
            verifier_ntt: Arc::new(crate::ntt_cache::VerifierNttCache::default()),
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaVerifierSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("setup shared_matrix does not match public matrix seed"));
    }

    #[test]
    fn strict_verifier_setup_decode_rejects_truncated_seed_prefix_matrix() {
        let setup_seed = seed([7u8; 32]);
        let short_matrix = derive_public_matrix_flat::<F, D>(1, &setup_seed.public_matrix_seed);
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    setup_seed,
                    short_matrix,
                ),
            ),
            prefix_slots: SetupPrefixVerifierRegistry::new(),
            verifier_ntt: Arc::new(crate::ntt_cache::VerifierNttCache::default()),
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaVerifierSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("flat matrix total_ring_elements does not match expected setup shape"));
    }

    #[test]
    fn strict_setup_decode_rejects_matrix_shape_before_payload() {
        let setup_seed = seed([7u8; 32]);
        let mut bytes = Vec::new();
        setup_seed.serialize_compressed(&mut bytes).unwrap();
        usize::MAX.serialize_compressed(&mut bytes).unwrap();
        D.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaExpandedSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("flat matrix total_ring_elements does not match expected setup shape"));
    }

    #[test]
    fn setup_seed_validity_is_not_the_generic_decode_allocation_cap() {
        let setup_seed = AkitaSetupSeed {
            max_num_vars: 32,
            max_num_batched_polys: 1,
            gen_ring_dim: D,
            max_setup_len: MAX_SETUP_MATRIX_FIELD_ELEMENTS / D + 1,
            public_matrix_seed: [7u8; 32],
        };

        setup_seed.check().unwrap();
        assert!(setup_seed.matrix_field_elements().unwrap() > MAX_SETUP_MATRIX_FIELD_ELEMENTS);
    }

    #[test]
    fn generic_setup_decode_still_rejects_shapes_above_allocation_cap() {
        let setup_seed = AkitaSetupSeed {
            max_num_vars: 32,
            max_num_batched_polys: 1,
            gen_ring_dim: D,
            max_setup_len: MAX_SETUP_MATRIX_FIELD_ELEMENTS / D + 1,
            public_matrix_seed: [7u8; 32],
        };
        let mut bytes = Vec::new();
        setup_seed.serialize_compressed(&mut bytes).unwrap();

        let err = AkitaExpandedSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(matches!(
            err,
            SerializationError::LengthLimitExceeded { max, .. }
                if max == MAX_SETUP_MATRIX_FIELD_ELEMENTS
        ));
    }

    #[test]
    fn flat_derivation_is_deterministic_for_same_seed() {
        let seed = [42u8; 32];
        let m1 = derive_public_matrix_flat::<SmallF, SMALL_D>(15, &seed);
        let m2 = derive_public_matrix_flat::<SmallF, SMALL_D>(15, &seed);
        assert_eq!(m1, m2);
    }

    #[test]
    fn flat_derivation_is_prefix_stable() {
        let seed = [7u8; 32];
        let small = derive_public_matrix_flat::<SmallF, SMALL_D>(6, &seed);
        let large = derive_public_matrix_flat::<SmallF, SMALL_D>(24, &seed);
        let small_view = small.ring_view::<SMALL_D>(1, 6).unwrap();
        let large_view = large.ring_view::<SMALL_D>(1, 6).unwrap();
        for c in 0..6 {
            assert_eq!(small_view.row(0).unwrap()[c], large_view.row(0).unwrap()[c]);
        }
    }

    #[test]
    fn flat_derivation_matches_ring_random_stream() {
        let seed = [5u8; 32];
        let got = derive_public_matrix_flat::<SmallF, SMALL_D>(6, &seed);
        let expected = (0..6)
            .flat_map(|idx| {
                let mut rng = ShakeXofRng::new_labeled(&seed, SHARED_MATRIX_LABEL, &[idx as u64]);
                CyclotomicRing::<SmallF, SMALL_D>::random(&mut rng).coeffs
            })
            .collect::<Vec<_>>();

        assert_eq!(got.as_field_slice(), expected.as_slice());
    }

    #[test]
    fn different_shapes_from_same_flat() {
        let seed = [13u8; 32];
        let flat = derive_public_matrix_flat::<SmallF, SMALL_D>(12, &seed);
        let view_3x4 = flat.ring_view::<SMALL_D>(3, 4).unwrap();
        let view_2x6 = flat.ring_view::<SMALL_D>(2, 6).unwrap();

        assert_eq!(view_3x4.row(0).unwrap()[0], view_2x6.row(0).unwrap()[0]);
        assert_eq!(view_3x4.row(0).unwrap()[3], view_2x6.row(0).unwrap()[3]);
        assert_ne!(view_3x4.row(1).unwrap()[0], view_2x6.row(1).unwrap()[0]);
    }
}
