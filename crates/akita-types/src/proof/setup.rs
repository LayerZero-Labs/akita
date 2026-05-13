//! Shared setup data shapes for Akita prover and verifier APIs.

use crate::{AkitaScheduleLookupKey, FlatMatrix, Schedule};
use akita_field::FieldCore;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::any::Any;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex, OnceLock};

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
/// each re-run the planner DP search. `tiered_s_cache` lazily memoizes
/// the tiered B-side commitment to `S` per book §5.3-5.4. Both caches
/// are shared via inner `Arc`s and excluded from serialization, `Valid`
/// checks, and equality.
///
/// The tiered cache stores a `Box<dyn Any + Send + Sync>` because
/// `AkitaVerifierSetup` is parameterized only on `F`, not on the ring
/// dimension `D`. Callers downcast via [`Self::tiered_s_cache_get_or_init`]
/// passing their concrete `D`.
#[derive(Debug, Clone)]
pub struct AkitaVerifierSetup<F: FieldCore> {
    /// Expanded matrix stage used for verification.
    pub expanded: Arc<AkitaExpandedSetup<F>>,
    /// Schedule cache shared across `batched_verify` calls.
    pub schedule_cache: Arc<Mutex<HashMap<AkitaScheduleLookupKey, Schedule>>>,
    /// Tiered `S` commitment cache, lazy on first use. Stores
    /// `TieredSetupCommitments<F, D>` boxed under `Any` so the cache can
    /// live on a struct that has no `D` const generic; accessor methods
    /// downcast to the caller's concrete `D`.
    pub tiered_s_cache: Arc<OnceLock<Box<dyn Any + Send + Sync>>>,
}

impl<F: FieldCore> AkitaVerifierSetup<F> {
    /// Build a verifier setup wrapping the given expanded setup with a fresh
    /// empty schedule cache.
    #[must_use]
    pub fn new(expanded: Arc<AkitaExpandedSetup<F>>) -> Self {
        Self {
            expanded,
            schedule_cache: Arc::new(Mutex::new(HashMap::new())),
            tiered_s_cache: Arc::new(OnceLock::new()),
        }
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

    /// Get the cached tiered `S` commitments, downcast to the caller's
    /// ring dimension `D`. Returns `None` if the cache has not been
    /// populated yet.
    #[must_use]
    pub fn cached_tiered_s_commitments<const D: usize>(
        &self,
    ) -> Option<&crate::TieredSetupCommitments<F, D>>
    where
        F: 'static,
    {
        self.tiered_s_cache
            .get()?
            .downcast_ref::<crate::TieredSetupCommitments<F, D>>()
    }

    /// Get or lazily initialize the tiered `S` commitments via the
    /// supplied derivation closure. Subsequent calls with a compatible
    /// `D` return the cached value; calls with a different `D` than the
    /// initial population fail with `Err(AkitaError::InvalidSetup)`.
    ///
    /// # Errors
    ///
    /// Returns whatever `derive` returns on its first invocation. On
    /// subsequent calls returns `Err(AkitaError::InvalidSetup)` if `D`
    /// differs from the originally cached dimension.
    ///
    /// # Panics
    ///
    /// Panics if a concurrent caller populated the cache between this
    /// call's `get` and `set` and the cache was left empty afterwards.
    /// This indicates an internal `OnceLock` invariant violation and is
    /// unreachable under correct std behavior.
    pub fn tiered_s_cache_get_or_init<const D: usize, E>(
        &self,
        derive: impl FnOnce() -> Result<crate::TieredSetupCommitments<F, D>, E>,
    ) -> Result<&crate::TieredSetupCommitments<F, D>, E>
    where
        F: 'static,
        E: From<akita_field::AkitaError>,
    {
        if let Some(boxed) = self.tiered_s_cache.get() {
            return boxed.downcast_ref::<crate::TieredSetupCommitments<F, D>>().ok_or_else(|| {
                akita_field::AkitaError::InvalidSetup(format!(
                    "tiered_s_cache already populated under a different ring dimension (D = {D} requested)"
                ))
                .into()
            });
        }
        let materialized: crate::TieredSetupCommitments<F, D> = derive()?;
        // Race: another caller may have populated the slot between the
        // `get` above and the `set` below. `set` returns
        // `Err(...)` in that case; we discard the loser and downcast the
        // winner.
        let _ = self
            .tiered_s_cache
            .set(Box::new(materialized) as Box<dyn Any + Send + Sync>);
        self.tiered_s_cache
            .get()
            .expect("tiered_s_cache initialized in this call or by a concurrent caller")
            .downcast_ref::<crate::TieredSetupCommitments<F, D>>()
            .ok_or_else(|| {
                akita_field::AkitaError::InvalidSetup(format!(
                    "tiered_s_cache concurrent initialization under a different ring dimension (D = {D} requested)"
                ))
                .into()
            })
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
