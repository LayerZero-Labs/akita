#![cfg_attr(
    any(feature = "profile-onehot-fp128", feature = "profile-bench-selected"),
    allow(dead_code)
)]

use crate::report::print_layout;
use crate::workload::{
    onehot_k_for_num_vars, profile_setup_contribution_mode, run_batched_onehot, run_dense_for,
    run_onehot, run_recursive_multi_group_onehot,
};
use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig};
use akita_field::unreduced::{HasCommitAccum, HasOptimizedFold, HasUnreducedOps, HasWide};
use akita_field::TranscriptChallenge;
use akita_field::{
    CanonicalBytes, CanonicalField, FrobeniusExtField, FromPrimitiveInt, HalvingField,
    PseudoMersenneField, RandomSampling,
};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupParams, FpExtEncoding, MultiChunkProfileId,
    PolynomialGroupLayout, SetupContributionMode,
};

type F = fp128::Field;

const MULTI_GROUP_PRE_NUM_VARS: usize = 16;
const MULTI_GROUP_FINAL_NUM_VARS: usize = 32;
const MULTI_GROUP_PRE_GROUPS: usize = 2;
const MULTI_GROUP_FINAL_POLYS: usize = 2;
const MULTI_GROUP_TOTAL_POLYS: usize = MULTI_GROUP_PRE_GROUPS + MULTI_GROUP_FINAL_POLYS;

fn fp128_prime_label() -> String {
    match <F as PseudoMersenneField>::MODULUS_OFFSET {
        2355 => "q=2^128-2355".to_string(),
        // Prime128OffsetA7F7: p = 2^128 - 2^32 + 22537 = 2^128 - 0xFFFFA7F7.
        0xFFFFA7F7 => "q=2^128-2^32+22537".to_string(),
        offset => format!("q=2^128-{offset:#x}"),
    }
}

fn run_dense_mode<const D: usize, Cfg: CommitmentConfig<Field = F, ExtField = F>>(
    label: &str,
    title: &str,
    nv: usize,
) {
    let layout = resolve_layout::<F, Cfg>(nv);
    let plan = Cfg::select_schedule_for_key(&AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(nv),
    ))
    .expect("schedule plan")
    .into_schedule();
    tracing::info!("{}", title);
    print_layout(&layout, 1, Cfg::decomposition().field_bits()).expect("profile B geometry");
    run_dense_for::<F, D, Cfg>(label, nv, &layout, Some(&plan), true);
}

fn run_dense_mode_for<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    label: &str,
    title: &str,
    nv: usize,
) where
    FF: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + RandomSampling
        + FromPrimitiveInt
        + PseudoMersenneField
        + HalvingField
        + HasWide
        + HasCommitAccum
        + Valid
        + AkitaSerialize
        + 'static,
    Cfg::ExtField: FrobeniusExtField<FF>
        + FpExtEncoding<FF>
        + HasUnreducedOps
        + HasOptimizedFold
        + AkitaSerialize
        + Valid,
{
    // The dense profile opens one polynomial at one point, so the schedule key
    // is the singleton root the prover actually resolves via
    // `new_from_opening_batch`.
    let layout = resolve_layout::<FF, Cfg>(nv);
    let plan = Cfg::select_schedule_for_key(&AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(nv),
    ))
    .expect("schedule plan")
    .into_schedule();
    tracing::info!("{}", title);
    print_layout(&layout, 1, Cfg::decomposition().field_bits()).expect("profile B geometry");
    run_dense_for::<FF, D, Cfg>(label, nv, &layout, Some(&plan), true);
}

fn run_onehot_mode_for<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    label: &str,
    title: &str,
    nv: usize,
    num_polys: usize,
) where
    FF: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + RandomSampling
        + FromPrimitiveInt
        + PseudoMersenneField
        + HalvingField
        + HasWide
        + HasCommitAccum
        + Valid
        + AkitaSerialize
        + 'static,
    Cfg::ExtField: FrobeniusExtField<FF>
        + FpExtEncoding<FF>
        + HasUnreducedOps
        + HasOptimizedFold
        + AkitaSerialize
        + Valid,
{
    tracing::info!("{}", title);
    if num_polys == 1 {
        let layout = resolve_layout::<FF, Cfg>(nv);
        let required_vars = layout.position_index_bits()
            + layout.block_index_bits()
            + layout.d_a().trailing_zeros() as usize;
        if required_vars > nv {
            tracing::error!(
                label,
                nv,
                required_vars,
                "fixed onehot profile layout exceeds the public polynomial arity"
            );
            panic!(
                "[{label}] fixed onehot profile requires {required_vars} variables, but AKITA_NUM_VARS={nv}"
            );
        }
        let plan = Cfg::select_schedule_for_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(nv),
        ))
        .expect("schedule plan")
        .into_schedule();
        print_layout(&layout, 1, Cfg::decomposition().field_bits()).expect("profile B geometry");
        run_onehot::<FF, D, Cfg>(label, nv, &layout, Some(&plan), true);
    } else {
        let lookup_key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(nv, num_polys));
        let plan = Cfg::select_schedule_for_key(&lookup_key)
            .expect("schedule plan")
            .into_schedule();
        let layout =
            Cfg::select_schedule_for_opening(&lookup_key.opening_layout().expect("opening layout"))
                .map(|row| row.schedule().root.params.final_group.commitment.clone())
                .expect("layout");
        let required_vars = layout.position_index_bits()
            + layout.block_index_bits()
            + layout.d_a().trailing_zeros() as usize;
        if required_vars > nv {
            tracing::error!(
                label,
                nv,
                required_vars,
                num_polys,
                "fixed batched onehot profile layout exceeds the public polynomial arity"
            );
            panic!(
                "[{label}] fixed batched onehot profile requires {required_vars} variables, but AKITA_NUM_VARS={nv}"
            );
        }
        print_layout(&layout, num_polys, Cfg::decomposition().field_bits())
            .expect("profile B geometry");
        run_batched_onehot::<FF, D, Cfg>(label, nv, num_polys, &layout, Some(&plan));
    }
}

fn run_onehot_mode<const D: usize, Cfg: CommitmentConfig<Field = F, ExtField = F>>(
    label: &str,
    title: &str,
    nv: usize,
    num_polys: usize,
) {
    run_onehot_mode_for::<F, D, Cfg>(label, title, nv, num_polys);
}

#[cfg(not(feature = "profile-onehot-fp128"))]
type ProfileModeRunner = fn(usize, usize);

#[cfg(not(feature = "profile-onehot-fp128"))]
struct ProfileMode {
    name: &'static str,
    run: ProfileModeRunner,
}

#[cfg(all(
    not(feature = "profile-onehot-fp128"),
    feature = "profile-bench-selected"
))]
const PROFILE_SELECTED_MODES: &[ProfileMode] = &[
    #[cfg(any(feature = "profile-ci", feature = "profile-ci-fp128-base"))]
    ProfileMode {
        name: "dense_fp128",
        run: run_profile_dense_fp128,
    },
    #[cfg(any(feature = "profile-ci", feature = "profile-ci-fp128-base"))]
    ProfileMode {
        name: "onehot_fp128",
        run: run_profile_onehot_fp128,
    },
    #[cfg(feature = "profile-ci-multi-group-direct")]
    ProfileMode {
        name: "onehot_fp128_multi_group",
        run: run_profile_onehot_fp128_multi_group,
    },
    #[cfg(feature = "profile-ci-multi-group-recursive")]
    ProfileMode {
        name: "onehot_fp128_multi_group_recursive",
        run: run_profile_onehot_fp128_multi_group_recursive,
    },
    #[cfg(feature = "profile-ci-multi-group-recursive-w8r2")]
    ProfileMode {
        name: "onehot_fp128_multi_group_recursive_multi_chunk_w8r2",
        run: run_profile_onehot_fp128_multi_group_recursive_multi_chunk_w8r2,
    },
    #[cfg(feature = "profile-ci-distributed")]
    ProfileMode {
        name: "onehot_fp128_multi_chunk_w2r2",
        run: run_profile_onehot_fp128_multi_chunk_w2r2,
    },
    #[cfg(feature = "profile-ci-distributed")]
    ProfileMode {
        name: "onehot_fp128_multi_chunk_w4r2",
        run: run_profile_onehot_fp128_multi_chunk_w4r2,
    },
    #[cfg(feature = "profile-ci-distributed")]
    ProfileMode {
        name: "onehot_fp128_multi_chunk_w8r2",
        run: run_profile_onehot_fp128_multi_chunk_w8r2,
    },
    #[cfg(any(feature = "profile-ci", feature = "profile-ci-fp32"))]
    ProfileMode {
        name: "dense_fp32",
        run: run_profile_dense_fp32,
    },
    #[cfg(any(feature = "profile-ci", feature = "profile-ci-fp32"))]
    ProfileMode {
        name: "onehot_fp32",
        run: run_profile_onehot_fp32,
    },
    #[cfg(any(feature = "profile-ci", feature = "profile-ci-fp64"))]
    ProfileMode {
        name: "dense_fp64",
        run: run_profile_dense_fp64,
    },
    #[cfg(any(feature = "profile-ci", feature = "profile-ci-fp64"))]
    ProfileMode {
        name: "onehot_fp64",
        run: run_profile_onehot_fp64,
    },
];

#[cfg(all(
    not(feature = "profile-onehot-fp128"),
    not(feature = "profile-bench-selected")
))]
const PROFILE_ALL_MODES: &[ProfileMode] = &[
    ProfileMode {
        name: "dense_fp128",
        run: run_profile_dense_fp128,
    },
    ProfileMode {
        name: "onehot_fp128",
        run: run_profile_onehot_fp128,
    },
    ProfileMode {
        name: "onehot_fp128_multi_group",
        run: run_profile_onehot_fp128_multi_group,
    },
    ProfileMode {
        name: "onehot_fp128_multi_group_recursive",
        run: run_profile_onehot_fp128_multi_group_recursive,
    },
    ProfileMode {
        name: "onehot_fp128_multi_group_recursive_multi_chunk_w8r2",
        run: run_profile_onehot_fp128_multi_group_recursive_multi_chunk_w8r2,
    },
    ProfileMode {
        name: "onehot_fp128_multi_chunk_w2r2",
        run: run_profile_onehot_fp128_multi_chunk_w2r2,
    },
    ProfileMode {
        name: "onehot_fp128_multi_chunk_w4r2",
        run: run_profile_onehot_fp128_multi_chunk_w4r2,
    },
    ProfileMode {
        name: "onehot_fp128_multi_chunk_w8r2",
        run: run_profile_onehot_fp128_multi_chunk_w8r2,
    },
    ProfileMode {
        name: "dense_fp32",
        run: run_profile_dense_fp32,
    },
    ProfileMode {
        name: "onehot_fp32",
        run: run_profile_onehot_fp32,
    },
    ProfileMode {
        name: "dense_fp64",
        run: run_profile_dense_fp64,
    },
    ProfileMode {
        name: "onehot_fp64",
        run: run_profile_onehot_fp64,
    },
];

#[cfg(not(feature = "profile-onehot-fp128"))]
fn profile_modes() -> &'static [ProfileMode] {
    #[cfg(feature = "profile-bench-selected")]
    {
        PROFILE_SELECTED_MODES
    }
    #[cfg(not(feature = "profile-bench-selected"))]
    {
        PROFILE_ALL_MODES
    }
}

/// Modes registered for explicit `AKITA_MODE=…` runs but omitted from `all`.
#[cfg(not(feature = "profile-onehot-fp128"))]
const EXCLUDED_FROM_ALL_SWEEP: &[&str] = &[
    "onehot_fp128_multi_group",
    "onehot_fp128_multi_chunk_w2r2",
    "onehot_fp128_multi_chunk_w4r2",
    "onehot_fp128_multi_chunk_w8r2",
    "onehot_fp128_multi_group_recursive",
    "onehot_fp128_multi_group_recursive_multi_chunk_w8r2",
    // Small-field adaptive presets are heavy and cover narrow generated key sets.
    // Keep them out of the default `all` smoke sweep. They remain selectable
    // with an explicit compatible `AKITA_MODE=` and `AKITA_NUM_VARS=`.
    "dense_fp32",
    "onehot_fp32",
    "dense_fp64",
    "onehot_fp64",
];

fn assert_singleton_mode(mode: &str, num_polys: usize) {
    assert_eq!(
        num_polys, 1,
        "{mode} currently profiles only singleton commitments"
    );
}

const SMALL_FIELD_SCHEDULE_SOURCE: &str = "generated schedule catalog";

fn small_field_onehot_title(field_label: &str, nv: usize, num_polys: usize) -> String {
    let onehot_k = onehot_k_for_num_vars(nv);
    let schedule = SMALL_FIELD_SCHEDULE_SOURCE;
    if num_polys == 1 {
        format!(
            "=== onehot_{field_label} ({field_label}, adaptive ring dimensions, 1-of-{onehot_k}, {schedule}) ==="
        )
    } else {
        format!(
            "=== onehot_{field_label} batched ({field_label}, adaptive ring dimensions, 1-of-{onehot_k}, same-point batch={num_polys}, {schedule}) ==="
        )
    }
}

fn small_field_dense_title(field_label: &str) -> String {
    let schedule = SMALL_FIELD_SCHEDULE_SOURCE;
    format!("=== dense_{field_label} ({field_label}, adaptive ring dimensions, {schedule}) ===")
}

fn run_profile_dense_fp128(nv: usize, num_polys: usize) {
    type Cfg = fp128::Dense;
    assert_singleton_mode("dense_fp128", num_polys);
    let prime = fp128_prime_label();
    run_dense_mode::<{ Cfg::D }, Cfg>(
        "dense_fp128",
        &format!("=== dense_fp128 (fp128, {prime}, generated per-level dimensions) ==="),
        nv,
    );
}

fn run_profile_onehot_fp128(nv: usize, num_polys: usize) {
    match profile_setup_contribution_mode() {
        SetupContributionMode::Direct => {
            type Cfg = fp128::OneHot;
            run_profile_onehot_fp128_with_cfg::<{ Cfg::D }, Cfg>("onehot_fp128", nv, num_polys);
        }
        SetupContributionMode::Recursive => {
            type Cfg = RecursiveCommitmentConfig<fp128::OneHot>;
            assert_eq!(nv, 36, "recursive onehot_fp128 profiles nv=36");
            run_profile_onehot_fp128_with_cfg::<{ Cfg::D }, Cfg>("onehot_fp128", nv, num_polys);
        }
    }
}

fn run_profile_onehot_fp128_with_cfg<
    const D: usize,
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
>(
    label: &str,
    nv: usize,
    num_polys: usize,
) {
    assert!(
        matches!(nv, 32 | 36),
        "fp128 one-hot profile supports generated nv=32 and nv=36 rows"
    );
    assert_singleton_mode(label, num_polys);

    let schedule = Cfg::select_schedule_for_key(&AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(nv),
    ))
    .expect("generated fp128 one-hot schedule")
    .into_schedule();
    let selected_dims = std::iter::once(schedule.root.params.final_group.commitment.role_dims())
        .chain(
            schedule
                .recursive_folds
                .iter()
                .map(|fold| fold.params.witness.role_dims()),
        )
        .collect::<Vec<_>>();
    tracing::info!(
        selected_dims = ?selected_dims,
        "generated fp128 one-hot schedule selection"
    );

    let layout = resolve_layout::<F, Cfg>(nv);
    tracing::info!(
        "=== {label} (fp128, flat public setup, generated per-level dimensions, 1-of-256) ==="
    );
    print_layout(&layout, 1, Cfg::decomposition().field_bits()).expect("profile B geometry");
    // The catalog row selected here is the same exact row used by the PCS
    // prover and verifier. The benchmark intentionally does not compare it
    // against a different uniform-D family.
    run_onehot::<F, D, Cfg>(label, nv, &layout, Some(&schedule), false);
}

/// Shared driver for the multi-group profiles. Every such profile fixes the
/// shape declared by the `MULTI_GROUP_*` constants above; only the base preset
/// (`Cfg`) and the `layout_note` describing its witness layout differ.
fn run_multi_group_mode<const D: usize, Cfg: CommitmentConfig<Field = F, ExtField = F>>(
    label: &str,
    layout_note: &str,
    nv: usize,
    num_polys: usize,
) {
    assert_eq!(
        nv, MULTI_GROUP_FINAL_NUM_VARS,
        "{label} fixes the main-group arity"
    );
    assert_eq!(
        num_polys, MULTI_GROUP_TOTAL_POLYS,
        "{label} opens two precommitted singleton groups plus two main polynomials"
    );
    tracing::info!(
        "=== {label} (fp128, {}, config D={D}, flat public setup, {MULTI_GROUP_PRE_GROUPS} precommitted {MULTI_GROUP_PRE_NUM_VARS}-variable singleton groups + {MULTI_GROUP_FINAL_NUM_VARS}-variable main group with {MULTI_GROUP_FINAL_POLYS} polynomials, {layout_note}) ===",
        fp128_prime_label()
    );
    run_recursive_multi_group_onehot::<F, D, Cfg>(
        label,
        MULTI_GROUP_PRE_NUM_VARS,
        MULTI_GROUP_FINAL_NUM_VARS,
        MULTI_GROUP_FINAL_POLYS,
    );
}

fn run_profile_onehot_fp128_multi_group(nv: usize, num_polys: usize) {
    type Cfg = fp128::OneHot;
    assert_eq!(
        profile_setup_contribution_mode(),
        SetupContributionMode::Direct,
        "onehot_fp128_multi_group supports direct setup contribution only"
    );
    run_multi_group_mode::<{ Cfg::D }, Cfg>(
        "onehot_fp128_multi_group",
        "generated per-level dimensions",
        nv,
        num_polys,
    );
}

fn run_profile_onehot_fp128_multi_group_recursive(nv: usize, num_polys: usize) {
    type Cfg = fp128::OneHot;
    run_multi_group_mode::<{ Cfg::D }, Cfg>(
        "onehot_fp128_multi_group_recursive",
        "adaptive ring dimensions + recursive setup",
        nv,
        num_polys,
    );
}

fn run_profile_onehot_fp128_multi_group_recursive_multi_chunk_w8r2(nv: usize, num_polys: usize) {
    type Cfg = fp128::OneHotMultiChunk;
    run_multi_group_mode::<{ Cfg::D }, Cfg>(
        "onehot_fp128_multi_group_recursive_multi_chunk_w8r2",
        "adaptive ring dimensions + recursive setup offloading + W8R2 chunked witness: num_chunks=8 x 2 leading levels",
        nv,
        num_polys,
    );
}

fn run_profile_onehot_fp128_multi_chunk_named<
    const D: usize,
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
>(
    label: &str,
    profile: MultiChunkProfileId,
    nv: usize,
    num_polys: usize,
) {
    let prime = fp128_prime_label();
    let onehot_k = onehot_k_for_num_vars(nv);
    let title = format!(
        "=== {label} (fp128, {prime}, adaptive ring dimensions, 1-of-{onehot_k}, distributed chunked relation, num_chunks={} x {} leading levels) ===",
        profile.num_chunks(),
        profile.num_activated_levels(),
    );
    run_onehot_mode::<D, Cfg>(label, &title, nv, num_polys);
}

fn run_profile_onehot_fp128_multi_chunk_w8r2(nv: usize, num_polys: usize) {
    run_profile_onehot_fp128_multi_chunk_named::<256, fp128::OneHotMultiChunk>(
        "onehot_fp128_multi_chunk_w8r2",
        MultiChunkProfileId::W8R2,
        nv,
        num_polys,
    );
}

fn run_profile_onehot_fp128_multi_chunk_w2r2(nv: usize, num_polys: usize) {
    run_profile_onehot_fp128_multi_chunk_named::<256, fp128::OneHotMultiChunkW2R2>(
        "onehot_fp128_multi_chunk_w2r2",
        MultiChunkProfileId::W2R2,
        nv,
        num_polys,
    );
}

fn run_profile_onehot_fp128_multi_chunk_w4r2(nv: usize, num_polys: usize) {
    run_profile_onehot_fp128_multi_chunk_named::<256, fp128::OneHotMultiChunkW4R2>(
        "onehot_fp128_multi_chunk_w4r2",
        MultiChunkProfileId::W4R2,
        nv,
        num_polys,
    );
}

fn run_profile_onehot_fp32(nv: usize, num_polys: usize) {
    type Cfg = fp32::OneHot;
    let title = small_field_onehot_title("fp32", nv, num_polys);
    run_onehot_mode_for::<fp32::Field, { Cfg::D }, Cfg>("onehot_fp32", &title, nv, num_polys);
}

fn run_profile_dense_fp32(nv: usize, num_polys: usize) {
    type Cfg = fp32::Dense;
    assert_singleton_mode("dense_fp32", num_polys);
    let title = small_field_dense_title("fp32");
    run_dense_mode_for::<fp32::Field, { Cfg::D }, Cfg>("dense_fp32", &title, nv);
}

fn run_profile_dense_fp64(nv: usize, num_polys: usize) {
    type Cfg = fp64::Dense;
    assert_singleton_mode("dense_fp64", num_polys);
    let title = small_field_dense_title("fp64");
    run_dense_mode_for::<fp64::Field, { Cfg::D }, Cfg>("dense_fp64", &title, nv);
}

fn run_profile_onehot_fp64(nv: usize, num_polys: usize) {
    type Cfg = fp64::OneHot;
    let title = small_field_onehot_title("fp64", nv, num_polys);
    run_onehot_mode_for::<fp64::Field, { Cfg::D }, Cfg>("onehot_fp64", &title, nv, num_polys);
}

#[cfg(not(feature = "profile-onehot-fp128"))]
pub(crate) fn run_profile_mode(mode: &str, nv: usize, num_polys: usize) {
    let modes = profile_modes();
    let profile_mode = modes
        .iter()
        .find(|entry| entry.name == mode)
        .unwrap_or_else(|| {
            let mut known_modes = modes.iter().map(|entry| entry.name).collect::<Vec<_>>();
            known_modes.push("all");
            tracing::error!(
                mode,
                known_modes = %known_modes.join(", "),
                "Unknown AKITA_MODE"
            );
            std::process::exit(1);
        });
    (profile_mode.run)(nv, num_polys);
}

#[cfg(not(feature = "profile-onehot-fp128"))]
pub(crate) fn run_all_profile_modes(nv: usize) {
    for entry in profile_modes() {
        if EXCLUDED_FROM_ALL_SWEEP.contains(&entry.name) {
            continue;
        }
        run_profile_mode(entry.name, nv, 1);
    }
}

fn resolve_layout<FF, Cfg: CommitmentConfig<Field = FF>>(nv: usize) -> CommittedGroupParams {
    Cfg::select_schedule_for_opening(
        &akita_types::OpeningClaimsLayout::new(nv, 1).expect("singleton opening batch"),
    )
    .expect("layout")
    .schedule()
    .root
    .params
    .final_group
    .commitment
    .clone()
}
#[cfg(feature = "profile-onehot-fp128")]
pub(crate) fn run_profile_mode(mode: &str, nv: usize, num_polys: usize) {
    assert_eq!(
        mode, "onehot_fp128",
        "profile-onehot-fp128 only supports AKITA_MODE=onehot_fp128",
    );
    assert_eq!(
        num_polys, 1,
        "profile-onehot-fp128 only supports singleton commitments"
    );
    run_profile_onehot_fp128(nv, num_polys);
}

pub(crate) fn log_active_fp128_prime_probe() {
    tracing::info!(
        "fp128 protocol prime active: modulus_offset = 0x{:x}, probe(2^128 + 1) = 0x{:x}",
        <F as PseudoMersenneField>::MODULUS_OFFSET,
        F::solinas_reduce(&[1u64, 0, 1]).to_canonical_u128(),
    );
}
