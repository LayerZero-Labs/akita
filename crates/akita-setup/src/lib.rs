//! Requirement-backed prover setup construction and the release-file cache.
//!
//! [`new_prover_setup`] derives the public matrix from the seed and commits
//! to the required setup prefixes in this process. With `disk-persistence`,
//! [`ReleaseCache`] stores the files of pinned preprocessing releases by
//! `bundle_id`. The cache is untrusted transport: a loader checks every file
//! read through it against the release manifest.

#[cfg(feature = "disk-persistence")]
mod release_cache;

use akita_config::SetupRequirements;
use akita_cpu_backend::AkitaProverSetup;
use akita_error::AkitaError;
use akita_serialization::Valid;
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};

#[cfg(feature = "disk-persistence")]
pub use release_cache::ReleaseCache;

/// Construct prover setup that meets `requirements`.
///
/// `akita-config` owns setup sizing policy and derives the requirements from
/// one or more trusted catalogs; `akita-cpu-backend` owns the concrete setup
/// artifact, matrix derivation, and setup-prefix export. A production prover
/// loads a `PinnedProverSetup` from a release instead.
///
/// # Errors
///
/// Returns an error if the requested setup capacity is invalid or setup
/// expansion fails.
#[tracing::instrument(skip_all, name = "new_prover_setup")]
pub fn new_prover_setup<F>(
    requirements: &SetupRequirements<F>,
) -> Result<AkitaProverSetup<F>, AkitaError>
where
    F: Field + CanonicalEncoding + Unreduced + Valid + WithCommitAccumulator + 'static,
{
    let mut setup = AkitaProverSetup::generate_with_capacity(
        requirements.max_num_vars(),
        requirements.max_num_batched_polys(),
        requirements.matrix_capacity(),
    )?;
    setup.export_setup_prefixes(requirements.prefix_slot_ids())?;
    Ok(setup)
}

#[cfg(test)]
mod tests;
