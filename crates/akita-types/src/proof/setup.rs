//! Shared setup data shapes for Akita prover and verifier APIs.

use super::{TieredSetupCacheKey, TieredSetupCommitments};
use crate::{AkitaScheduleLookupKey, FlatMatrix, Schedule};
use akita_algebra::ring::{build_ntt_slot, NttSlotCache};
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::any::Any;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

/// Public seed used to derive commitment matrices.
pub type PublicMatrixSeed = [u8; 32];

/// Seed-only stage for deterministic setup expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaSetupSeed {
    /// Maximum supported variable count.
    pub max_num_vars: usize,
    /// Maximum number of batched polynomials supported by setup.
    pub max_num_batched_polys: usize,
    /// Maximum number of distinct opening points.
    ///
    /// Together with `max_num_batched_polys` this bounds the outer/D matrix
    /// widths the setup can serve; a multi-point batched opening that exceeds
    /// this bound would otherwise silently read past the shared matrix prefix.
    pub max_num_points: usize,
    /// Global row stride for the flat NTT cache.
    pub max_stride: usize,
    /// Public seed used to derive commitment matrices.
    pub public_matrix_seed: PublicMatrixSeed,
}

/// Expanded setup stage containing a single shared coefficient-form matrix.
///
/// All role matrices (A, B, D) are row/column prefixes of this shared vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaExpandedSetup<F: FieldCore> {
    /// Setup seed and runtime layout metadata.
    pub seed: AkitaSetupSeed,
    /// Shared 1D flat backing vector.
    pub shared_matrix: FlatMatrix<F>,
}

/// Verifier setup artifact derived from prover setup.
///
/// `schedule_cache` memoizes schedules keyed by their public
/// [`AkitaScheduleLookupKey`] so that repeated `batched_verify` calls don't
/// each re-run the planner DP search. The cache is shared via an inner
/// `Arc` and excluded from serialization, `Valid` checks, and equality.
///
/// `tiered_s_cache` memoizes [`TieredSetupCommitments`] derived from the
/// public shared matrix for tiered setup-claim reduction. Each setup is
/// used at a single ring dimension `D` in practice; entries are stored as
/// `Arc<dyn Any + Send + Sync>` and downcast at
/// [`Self::tiered_s_cache_get`] using the caller's `const D`.
///
/// `ntt_shared_cache` memoizes the verifier-side analog of
/// [`crate::AkitaProverSetup::ntt_shared`]: one CRT+NTT-preprocessed view
/// of the entire shared matrix at ring dimension `D`. Tiered routed-`S`
/// re-derivation reuses this cache so it pays the per-`D` NTT
/// preprocessing cost once across all cascade levels rather than per
/// per-tier `(chunk_lp, meta_lp)` mat-vec call. Keyed by `D` for the
/// (rare) configs that mix ring dimensions across the schedule.
#[derive(Debug, Clone)]
pub struct AkitaVerifierSetup<F: FieldCore> {
    /// Expanded matrix stage used for verification.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
    /// Schedule cache shared across `batched_verify` calls.
    pub schedule_cache: Arc<Mutex<HashMap<AkitaScheduleLookupKey, Schedule>>>,
    /// Tiered routed-`S` B-side commitment cache (opaque per `D`).
    pub tiered_s_cache: Arc<Mutex<HashMap<TieredSetupCacheKey, Arc<dyn Any + Send + Sync>>>>,
    /// Shared-matrix CRT+NTT cache, lazy per ring dimension `D`.
    pub ntt_shared_cache: Arc<Mutex<HashMap<usize, Arc<dyn Any + Send + Sync>>>>,
}

impl<F: FieldCore> AkitaVerifierSetup<F> {
    /// Build a verifier setup wrapping the given expanded setup with a fresh
    /// empty schedule cache.
    #[must_use]
    pub fn new(expanded: Arc<AkitaExpandedSetup<F>>) -> Self {
        Self {
            expanded,
            schedule_cache: Arc::new(Mutex::new(HashMap::new())),
            tiered_s_cache: Arc::new(Mutex::new(HashMap::new())),
            ntt_shared_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get-or-derive the shared CRT+NTT cache for the full setup matrix
    /// at ring dimension `D`.
    ///
    /// Mirrors [`crate::AkitaProverSetup::ntt_shared`] for verifier use:
    /// one `1 × total_ring_elements_at::<D>()` slot, reused across all
    /// LPs that mat-vec over the shared matrix. Soundness anchors on the
    /// same deterministic derivation from the public `shared_matrix` —
    /// callers must pass the matrix's own data to the closure, never
    /// external bytes.
    ///
    /// A poisoned lock falls back to a freshly built cache without
    /// memoizing (the cache is purely an optimization).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying NTT slot construction fails
    /// for the chosen field × ring-dimension pair.
    pub fn ntt_shared_get_or_init<const D: usize>(&self) -> Result<Arc<NttSlotCache<D>>, AkitaError>
    where
        F: CanonicalField + 'static,
    {
        if let Ok(guard) = self.ntt_shared_cache.lock() {
            if let Some(entry) = guard.get(&D) {
                if let Ok(cached) = entry.clone().downcast::<NttSlotCache<D>>() {
                    return Ok(cached);
                }
            }
        }
        let total = self.expanded.shared_matrix.total_ring_elements_at::<D>();
        let derived = Arc::new(build_ntt_slot::<F, D>(
            self.expanded
                .shared_matrix
                .ring_view::<D>(1, total)
                .coefficients(),
            1,
            total,
        )?);
        if let Ok(mut guard) = self.ntt_shared_cache.lock() {
            if let Some(entry) = guard.get(&D) {
                if let Ok(winner) = entry.clone().downcast::<NttSlotCache<D>>() {
                    return Ok(winner);
                }
            }
            guard.insert(D, derived.clone());
        }
        Ok(derived)
    }

    /// Get-or-derive a cached tiered setup commitment bundle.
    ///
    /// Soundness contract (book §5.4 "preprocessed shared-matrix commitment
    /// $C_S$"): the only way to populate the cache is to supply a derivation
    /// closure that reads from `setup.expanded.shared_matrix`. There is no
    /// public setter for cache entries, so external callers cannot inject
    /// arbitrary commitments that the verifier would later treat as the
    /// preprocessed $C_S$. Mirrors
    /// [`crate::AkitaProverSetup::tiered_s_cache_get_or_init`] but is
    /// per-`(tier, layout)`-keyed because the verifier may see multiple
    /// tier shapes within one schedule cascade.
    ///
    /// A poisoned lock falls back to the freshly-derived value without
    /// caching it; the cache is purely an optimization.
    ///
    /// # Errors
    ///
    /// Returns whatever `derive` returns on a cache miss.
    pub fn tiered_s_cache_get_or_init<E, const D: usize>(
        &self,
        key: TieredSetupCacheKey,
        derive: impl FnOnce() -> Result<TieredSetupCommitments<F, D>, E>,
    ) -> Result<Arc<TieredSetupCommitments<F, D>>, E>
    where
        F: 'static,
    {
        if let Ok(guard) = self.tiered_s_cache.lock() {
            if let Some(entry) = guard.get(&key) {
                if let Ok(cached) = entry.clone().downcast::<TieredSetupCommitments<F, D>>() {
                    return Ok(cached);
                }
            }
        }
        let derived = Arc::new(derive()?);
        if let Ok(mut guard) = self.tiered_s_cache.lock() {
            // Concurrent race: another caller may have populated the slot
            // between the get above and the lock below. Prefer their value
            // if the downcast succeeds, otherwise overwrite (different `D`
            // under the same key is a programming error but recover safely).
            if let Some(entry) = guard.get(&key) {
                if let Ok(winner) = entry.clone().downcast::<TieredSetupCommitments<F, D>>() {
                    return Ok(winner);
                }
            }
            guard.insert(key, derived.clone());
        }
        Ok(derived)
    }

    /// Look up a cached schedule for `key`. Returns a clone of the stored
    /// `Schedule` on hit, `None` on miss.
    #[must_use]
    pub fn cached_schedule(&self, key: AkitaScheduleLookupKey) -> Option<Schedule> {
        let guard = self.schedule_cache.lock().ok()?;
        guard.get(&key).cloned()
    }

    /// Insert `schedule` into the cache under `key`, replacing any existing
    /// entry. A poisoned lock is treated as a no-op (the cache is purely an
    /// optimization).
    pub fn store_schedule(&self, key: AkitaScheduleLookupKey, schedule: Schedule) {
        if let Ok(mut guard) = self.schedule_cache.lock() {
            guard.insert(key, schedule);
        }
    }
}

impl<F: FieldCore> PartialEq for AkitaVerifierSetup<F> {
    fn eq(&self, other: &Self) -> bool {
        self.expanded == other.expanded
    }
}

impl<F: FieldCore> Eq for AkitaVerifierSetup<F> {}

impl Valid for AkitaSetupSeed {
    fn check(&self) -> Result<(), SerializationError> {
        if self.max_stride == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed max_stride must be non-zero".to_string(),
            ));
        }
        if self.max_num_batched_polys == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed max_num_batched_polys must be at least 1".to_string(),
            ));
        }
        if self.max_num_points == 0 {
            return Err(SerializationError::InvalidData(
                "setup seed max_num_points must be at least 1".to_string(),
            ));
        }
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
        self.max_num_points
            .serialize_with_mode(&mut writer, compress)?;
        self.max_stride.serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.public_matrix_seed)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_num_vars.serialized_size(compress)
            + self.max_num_batched_polys.serialized_size(compress)
            + self.max_num_points.serialized_size(compress)
            + self.max_stride.serialized_size(compress)
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
        let max_num_points = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_stride = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let mut public_matrix_seed = [0u8; 32];
        reader.read_exact(&mut public_matrix_seed)?;
        let out = Self {
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
            max_stride,
            public_matrix_seed,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid> Valid for AkitaExpandedSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.seed.check()?;
        self.shared_matrix.check()?;
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
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.seed.serialized_size(compress) + self.shared_matrix.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for AkitaExpandedSetup<F>
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let out = Self {
            seed: AkitaSetupSeed::deserialize_with_mode(&mut reader, compress, validate, &())?,
            shared_matrix: FlatMatrix::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid> Valid for AkitaVerifierSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for AkitaVerifierSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.expanded.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.expanded.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for AkitaVerifierSetup<F>
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let expanded = Arc::new(AkitaExpandedSetup::deserialize_with_mode(
            reader,
            compress,
            validate,
            &(),
        )?);
        Ok(Self::new(expanded))
    }
}

#[cfg(test)]
mod schedule_cache_tests {
    use super::*;
    use crate::{AkitaRootBatchSummary, Schedule};
    use akita_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;

    fn dummy_setup() -> AkitaVerifierSetup<F> {
        AkitaVerifierSetup::new(Arc::new(AkitaExpandedSetup {
            seed: AkitaSetupSeed {
                max_num_vars: 4,
                max_num_batched_polys: 1,
                max_num_points: 1,
                max_stride: 1,
                public_matrix_seed: [0u8; 32],
            },
            shared_matrix: FlatMatrix::from_flat_data(vec![F::default()], 1),
        }))
    }

    #[test]
    fn schedule_cache_hits_after_store() {
        let setup = dummy_setup();
        let key = AkitaScheduleLookupKey::singleton(8, 6, 1);
        assert!(setup.cached_schedule(key).is_none());
        let sched = Schedule {
            steps: Vec::new(),
            total_bytes: 42,
        };
        setup.store_schedule(key, sched.clone());
        let got = setup.cached_schedule(key).expect("hit after store");
        assert_eq!(got.total_bytes, 42);
        assert!(got.steps.is_empty());
    }

    #[test]
    fn schedule_cache_keyed_by_batch_summary() {
        let setup = dummy_setup();
        let k1 = AkitaScheduleLookupKey::singleton(8, 6, 1);
        let k2 = AkitaScheduleLookupKey::with_batch(
            8,
            6,
            1,
            AkitaRootBatchSummary::new(2, 1, 1).unwrap(),
        );
        setup.store_schedule(
            k1,
            Schedule {
                steps: Vec::new(),
                total_bytes: 1,
            },
        );
        setup.store_schedule(
            k2,
            Schedule {
                steps: Vec::new(),
                total_bytes: 2,
            },
        );
        assert_eq!(setup.cached_schedule(k1).unwrap().total_bytes, 1);
        assert_eq!(setup.cached_schedule(k2).unwrap().total_bytes, 2);
    }

    #[test]
    fn schedule_cache_shared_via_clone() {
        let setup_a = dummy_setup();
        let setup_b = setup_a.clone();
        let key = AkitaScheduleLookupKey::singleton(8, 6, 1);
        setup_a.store_schedule(
            key,
            Schedule {
                steps: Vec::new(),
                total_bytes: 77,
            },
        );
        let got = setup_b.cached_schedule(key).expect("clone shares cache");
        assert_eq!(got.total_bytes, 77);
    }
}
