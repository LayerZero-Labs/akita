//! Default fp128 protocol presets on `p = 2^128 - 2^32 + 22537`
//! (`Prime128OffsetA7F7`).

use super::*;

/// Base field for the default fp128 presets.
pub type Field = Prime128OffsetA7F7;

/// Dense adaptive `D=64` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64Dense;

/// Default dense preset with D256 setup generation and planner-selected
/// per-level commitment dimensions.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dense;

impl Dense {
    pub const A_RING_DIMENSIONS: [usize; 3] = [64, 128, 256];
    pub const B_RING_DIMENSIONS: [usize; 2] = [64, 128];
    pub const D_RING_DIMENSIONS: [usize; 2] = [64, 128];
}

/// Binary onehot generated `D=64` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHot;

/// Binary onehot `D=64`, `K=16` preset with planner-derived schedules.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHotK16;

/// Binary onehot `D=256` tableless preset for mixed-ring experiments.
/// fp128 certifies all three commitment roles at this dimension.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256OneHot;

/// Default binary onehot preset with D256 setup generation and planner-selected
/// per-level commitment dimensions.
///
/// Mixed-dimension planning is an offline generation step. Runtime proving
/// and verification resolve the exact generated catalog row.
#[derive(Clone, Copy, Debug, Default)]
pub struct OneHot;

impl OneHot {
    pub const A_RING_DIMENSIONS: [usize; 3] = [64, 128, 256];
    pub const B_RING_DIMENSIONS: [usize; 2] = [64, 128];
    pub const D_RING_DIMENSIONS: [usize; 2] = [64, 128];
}

/// Tableless policy marker for a `D = 512` inner (A-role) root.
///
/// fp128 certifies the A role at `D = 512` for the Q128 profile, but not B/D.
/// Consequently this preset is used only as the envelope type for the
/// three-band mixed-role builder, which keeps B/D within their audited
/// dimensions; it cannot produce a standalone uniform-D512 schedule.
#[derive(Clone, Copy, Debug, Default)]
pub struct D512OneHot;

/// Multi-chunk (distributed-prover) companion of [`D64OneHot`]. Shares every
/// layout parameter with its sibling but prices the chunked witness layout.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHotMultiChunk;

/// Multi-chunk companion with `2` witness chunks and `2` leading fold levels.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHotMultiChunkW2R2;

/// Multi-chunk companion with `4` witness chunks and `2` leading fold levels.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHotMultiChunkW4R2;

/// Multi-chunk (distributed-prover) companion of [`D64Dense`].
#[derive(Clone, Copy, Debug, Default)]
pub struct D64DenseMultiChunk;

impl_proof_optimized_preset!(
    D64Dense,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    64,
    128,
    128,
    fold_norms = akita_types::sis::FoldWitnessNorms::bounded(3, 64)
);
impl_proof_optimized_preset!(
    Dense,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    256,
    128,
    128,
    fold_norms = akita_types::sis::FoldWitnessNorms::bounded(3, 64),
    schedules = ("schedules-fp128-dense", "fp128_dense", fp128_dense_table),
    ring_dimension_schedule_mode = akita_schedules::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels: 2,
        uniform_suffix_dimension: 64,
        potential_a_dimensions: &Dense::A_RING_DIMENSIONS,
        potential_b_dimensions: &Dense::B_RING_DIMENSIONS,
        potential_d_dimensions: &Dense::D_RING_DIMENSIONS,
    }
);
impl_proof_optimized_preset!(
    D64OneHot,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    64,
    128,
    1,
    fold_norms = akita_types::sis::FoldWitnessNorms::new(1, 1)
);
impl_proof_optimized_preset!(
    D64OneHotK16,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    64,
    128,
    1,
    fold_norms = akita_types::sis::FoldWitnessNorms::new(1, 4)
);
impl_proof_optimized_preset!(
    D256OneHot,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    256,
    128,
    1,
    fold_norms = akita_types::sis::FoldWitnessNorms::new(1, 1)
);
impl_proof_optimized_preset!(
    OneHot,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    256,
    128,
    1,
    fold_norms = akita_types::sis::FoldWitnessNorms::new(1, 1),
    schedules = ("schedules-fp128-onehot", "fp128_onehot", fp128_onehot_table),
    ring_dimension_schedule_mode = akita_schedules::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels: 2,
        uniform_suffix_dimension: 64,
        potential_a_dimensions: &OneHot::A_RING_DIMENSIONS,
        potential_b_dimensions: &OneHot::B_RING_DIMENSIONS,
        potential_d_dimensions: &OneHot::D_RING_DIMENSIONS,
    }
);
impl_proof_optimized_preset!(
    D512OneHot,
    Field,
    Field,
    akita_types::SisModulusProfileId::Q128OffsetA7F7,
    512,
    128,
    1,
    fold_norms = akita_types::sis::FoldWitnessNorms::new(1, 2)
);
impl_multi_chunk_companion!(
    D64OneHotMultiChunk,
    D64OneHot,
    akita_types::MultiChunkProfileId::W8R2,
    "schedules-fp128-d64-onehot-multi-chunk",
    fp128_d64_onehot_multi_chunk_table
);
impl_multi_chunk_companion!(
    D64OneHotMultiChunkW2R2,
    D64OneHot,
    akita_types::MultiChunkProfileId::W2R2,
    "schedules-fp128-d64-onehot-multi-chunk-w2r2",
    fp128_d64_onehot_multi_chunk_w2r2_table
);
impl_multi_chunk_companion!(
    D64OneHotMultiChunkW4R2,
    D64OneHot,
    akita_types::MultiChunkProfileId::W4R2,
    "schedules-fp128-d64-onehot-multi-chunk-w4r2",
    fp128_d64_onehot_multi_chunk_w4r2_table
);
impl_multi_chunk_companion!(
    D64DenseMultiChunk,
    D64Dense,
    akita_types::MultiChunkProfileId::W8R2,
    "schedules-fp128-d64-dense-multi-chunk",
    fp128_d64_dense_multi_chunk_table
);

/// Concrete fp128 preset selected by a schedule-family query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp128Preset {
    /// Dense preset with adaptive per-level ring dimensions.
    Dense,
    /// Binary onehot preset with adaptive per-level ring dimensions.
    OneHot,
}

impl Fp128Preset {
    /// Setup-generation ring dimension used by this preset.
    pub const fn ring_dimension(self) -> usize {
        match self {
            Self::Dense | Self::OneHot => 256,
        }
    }

    /// Whether this preset is onehot-oriented.
    pub const fn is_onehot(self) -> bool {
        matches!(self, Self::OneHot)
    }

    /// Stable human-readable preset name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::OneHot => "OneHot",
        }
    }
}

/// Best generated schedule for one fp128 preset family.
#[derive(Clone, Debug)]
pub struct Fp128ScheduleSelection {
    /// Selected concrete preset.
    pub preset: Fp128Preset,
    /// Runtime schedule selected for the supplied lookup key.
    pub schedule: FoldSchedule,
    /// Non-protocol planner estimate used to compare presets.
    pub estimate: akita_types::FoldScheduleEstimate,
}

fn candidate<Cfg: CommitmentConfig>(
    preset: Fp128Preset,
    key: PolynomialGroupLayout,
) -> Result<Option<Fp128ScheduleSelection>, AkitaError> {
    let lookup_key = AkitaScheduleLookupKey::single(key);
    let Some(catalog) = Cfg::schedule_catalog() else {
        return Ok(None);
    };
    let Some(entry) = akita_schedules::generated::table_entry(catalog, &lookup_key) else {
        return Ok(None);
    };
    let policy = crate::policy_of::<Cfg>();
    let estimate = akita_schedules::estimate_proof_bytes(
        entry,
        &lookup_key,
        &policy,
        Cfg::ring_challenge_config,
    )?;
    let schedule = Cfg::runtime_schedule(lookup_key)?;
    Ok(Some(Fp128ScheduleSelection {
        preset,
        schedule,
        estimate: akita_types::FoldScheduleEstimate {
            estimated_root_direct_payload_bytes: estimate,
            estimated_root_stage3_payload_bytes: 0,
            estimated_recursive_direct_payload_bytes: Vec::new(),
            estimated_recursive_stage3_payload_bytes: Vec::new(),
            estimated_terminal_direct_payload_bytes: 0,
            estimated_terminal_response_payload_bytes: 0,
            estimated_num_setup_field_elements: 0,
            first_direct_setup_field_len: None,
            selected_offload_edges: 0,
        },
    }))
}

fn best_by_exact_bytes<I>(candidates: I) -> Option<Fp128ScheduleSelection>
where
    I: IntoIterator<Item = Option<Fp128ScheduleSelection>>,
{
    candidates.into_iter().flatten().min_by_key(|selection| {
        (
            selection
                .estimate
                .estimated_proof_payload_bytes()
                .unwrap_or(usize::MAX),
            selection.preset.ring_dimension(),
        )
    })
}

/// Select the best onehot fp128 preset for a schedule lookup key.
///
/// A genuine planner failure propagates as an error; for any valid key every
/// supported preset yields a schedule, so the best available one is returned.
///
/// # Errors
///
/// Propagates a planner / runtime-schedule failure (invalid key shape,
/// witness overflow, or an uncovered SIS-floor width).
pub fn best_onehot_schedule(
    key: PolynomialGroupLayout,
) -> Result<Option<Fp128ScheduleSelection>, AkitaError> {
    Ok(best_by_exact_bytes([candidate::<OneHot>(
        Fp128Preset::OneHot,
        key,
    )?]))
}
