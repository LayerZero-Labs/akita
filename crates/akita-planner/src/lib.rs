//! Offline schedule planner for the Akita polynomial commitment scheme.
//!
//! This crate is a **pure, `Cfg`-free DP library**. The single entry point
//! is [`find_schedule`], which runs an exhaustive dynamic program to
//! minimize proof size for a schedule lookup key. Every per-preset input is
//! carried by the plain-value [`PlannerPolicy`] plus a `ring_challenge_config` /
//! `fold_challenge_shape_at_level` closure pair, so the planner names no `CommitmentConfig`
//! types and depends only on `akita-types` / `akita-challenges` /
//! `akita-field`.
//!
//! The preset family list, the `gen_schedule_tables` binary, and the
//! `policy_of::<Cfg>()` bridge that derives a [`PlannerPolicy`] from a preset
//! live in `akita-config`, the only crate that can name the presets.

pub use akita_types::{
    ChunkedWitnessCfg, DecompositionParams, SisModulusFamily, DEFAULT_SIS_SECURITY_BITS,
};

pub mod catalog_identity;
pub mod emit;
pub mod generated;
mod group_batch;
mod resolve;
pub mod schedule_params;

pub use akita_challenges::TensorChallengeShape;
pub use catalog_identity::{
    expected_catalog_identity, identity_digest, key_digest, policy_digest,
    ring_challenge_config_digest, validate_catalog_identity,
};
pub use emit::{refresh_generated_wiring, run_regen_fmt, write_family_module, EmitSpec};
pub use generated::{
    catalog_entries_sorted_for_lookup, runtime_schedule_key_cmp, validate_generated_schedule_entry,
    validate_generated_schedule_table, GeneratedScheduleCatalogIdentity, GeneratedScheduleTable,
};
pub use group_batch::find_group_batch_schedule;
pub use resolve::{
    estimate_proof_bytes, resolve_group_batch_schedule, resolve_schedule, schedule_from_entry,
};
pub use schedule_params::find_schedule;

/// Plain-value brute-force inputs the planner DP needs.
///
/// This is the `Cfg`-free projection of a `CommitmentConfig` preset that
/// the DP and SIS sizing read. `akita-config` derives it from a preset via
/// its `policy_of::<Cfg>()` bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannerPolicy {
    /// Ring degree `D` (`Cfg::D`).
    pub ring_dimension: usize,
    /// Gadget base + coefficient bounds (`Cfg::decomposition()`).
    pub decomposition: DecompositionParams,
    /// SIS modulus family (`Cfg::sis_modulus_family()`).
    pub sis_family: SisModulusFamily,
    /// Minimum SIS security floor in bits for generated SIS-width tables.
    pub min_sis_security_bits: u16,
    /// `psi`-embedding infinity-norm expansion
    /// (`Cfg::ring_subfield_embedding_norm_bound()`).
    pub ring_subfield_norm_bound: u32,
    /// Opening-reduction extension width (`Cfg::EXT_DEGREE`).
    pub claim_ext_degree: usize,
    /// Fiat-Shamir scalar extension width (`Cfg::EXT_DEGREE`).
    pub chal_ext_degree: usize,
    /// Inclusive `(min, max)` log-basis search range (`Cfg::basis_range()`).
    pub basis_range: (u32, u32),
    /// One-hot chunk size `K` (`Cfg::onehot_chunk_size()`).
    ///
    /// Used to bound the committed one-hot witness L1 mass per ring element
    /// (`nonzeros = ceil(D/K)`) for the weak-binding collision norm and the
    /// folded-witness digit count. Only consulted at a root level whose
    /// `log_commit_bound == 1`; dense levels use `nonzeros = D`.
    pub onehot_chunk_size: usize,
    /// Enable the tiered second commitment matrix `F` (`Cfg::TIERED_COMMITMENT`).
    ///
    /// When `true`, [`schedule_params`] is allowed to reuse a smaller first-tier
    /// matrix `B` across `f` witness slices and size a second-tier matrix `F`
    /// per level whose first-tier footprint would otherwise exceed `A`. When
    /// `false`, every level emits `tier_split == 1` / `f_key == None` and the
    /// layout is identical to the historical single-tier scheme. Also keys the
    /// tiered schedule catalog so a tiered policy never aliases a
    /// non-tiered table.
    pub tiered: bool,
    /// Multi-chunk witness layout settings (`Cfg::chunked_witness_cfg()`).
    ///
    /// Drives chunked-vs-single-chunk witness pricing in the DP and is embedded
    /// in the generated-table catalog identity so a chunked policy never aliases
    /// a single-chunk table. `ChunkedWitnessCfg::default()` (single chunk) leaves
    /// every schedule byte-identical to the historical layout.
    pub witness_chunk: ChunkedWitnessCfg,
}

impl PlannerPolicy {
    /// Chunk count of fold level `fold_level`'s own fold: the number of
    /// per-chunk folded responses `zᵢ` this level produces, hence the chunk
    /// count of the witness it emits. `build_w_coeffs` lays that witness out as
    /// `zᵢ ‖ eᵢ ‖ t̂ᵢ` per chunk, and `next_w_len(L)` is priced with
    /// `chunks_at_level(L)` to match it (the verifier sizes the same witness
    /// from `lp.witness_chunk.num_chunks`).
    ///
    /// Returns `num_chunks` for the leading `num_activated_levels` fold levels
    /// when multi-chunk layout is active, and `1` (single chunk) otherwise.
    /// There is no cross-level chunk handoff: level `L+1` folds level `L`'s
    /// emitted witness as a flat vector into its own `chunks_at_level(L+1)`
    /// windows, so a single-chunk level stays byte-identical regardless of its
    /// predecessor's chunk count.
    pub fn chunks_at_level(&self, fold_level: usize) -> usize {
        let mc = self.witness_chunk;
        if mc.uses_multi_chunk() && fold_level < mc.num_activated_levels {
            mc.num_chunks
        } else {
            1
        }
    }

    /// Per-level [`ChunkedWitnessCfg`] for the witness committed at absolute fold
    /// level `fold_level` (the **input** shape the relation MLE sees).
    ///
    /// Chunked levels carry the resolved chunk count and the policy's activated
    /// level count; single-chunk and tail levels carry
    /// [`ChunkedWitnessCfg::default`], keeping them byte-identical to today.
    pub fn witness_chunk_for_level(&self, fold_level: usize) -> ChunkedWitnessCfg {
        let num_chunks = self.chunks_at_level(fold_level);
        if num_chunks > 1 {
            ChunkedWitnessCfg {
                num_chunks,
                num_activated_levels: self.witness_chunk.num_activated_levels,
            }
        } else {
            ChunkedWitnessCfg::default()
        }
    }
}
