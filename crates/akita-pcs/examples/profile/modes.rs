#![cfg_attr(feature = "profile-onehot-fp128-d64", allow(dead_code))]

use crate::report::print_layout;
#[cfg(all(not(feature = "profile-onehot-fp128-d64"), not(feature = "profile-ci")))]
use crate::workload::run_recursive_multi_group_onehot_mixed;
use crate::workload::{
    onehot_k_for_num_vars, run_batched_onehot, run_dense_for, run_onehot,
    run_recursive_multi_group_onehot,
};
use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::tensor_verifier;
use akita_config::test_support::akita_batched_root_layout;
use akita_config::CommitmentConfig;
use akita_field::unreduced::HasWide;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::TranscriptChallenge;
use akita_field::{
    CanonicalBytes, CanonicalField, FrobeniusExtField, FromPrimitiveInt, HalvingField,
    PseudoMersenneField, RandomSampling,
};
#[cfg(all(not(feature = "profile-onehot-fp128-d64"), not(feature = "profile-ci")))]
use akita_pcs::test_support::{MixedDConfig, RecursiveRingDimensionTransitionConfig};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupParams, FpExtEncoding, MultiChunkProfileId,
    PolynomialGroupLayout,
};

type F = fp128::Field;

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
    let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(nv),
    ))
    .expect("schedule plan");
    tracing::info!("{}", title);
    print_layout(&layout, 1, Cfg::decomposition().field_bits());
    run_dense_for::<F, D, Cfg>(label, nv, &layout, Some(&plan), true);
}

#[cfg(not(feature = "profile-ci"))]
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
    let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(nv),
    ))
    .expect("schedule plan");
    tracing::info!("{}", title);
    print_layout(&layout, 1, Cfg::decomposition().field_bits());
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
        let required_vars =
            layout.position_index_bits() + layout.block_index_bits() + D.trailing_zeros() as usize;
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
        let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(nv),
        ))
        .expect("schedule plan");
        print_layout(&layout, 1, Cfg::decomposition().field_bits());
        run_onehot::<FF, D, Cfg>(label, nv, &layout, Some(&plan), true);
    } else {
        let schedule_key = PolynomialGroupLayout::new(nv, num_polys);
        let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(schedule_key))
            .expect("schedule plan");
        let layout = akita_batched_root_layout::<Cfg>(nv, num_polys).expect("layout");
        let required_vars =
            layout.position_index_bits() + layout.block_index_bits() + D.trailing_zeros() as usize;
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
        print_layout(&layout, num_polys, Cfg::decomposition().field_bits());
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

#[cfg(not(feature = "profile-onehot-fp128-d64"))]
type ProfileModeRunner = fn(usize, usize);

#[cfg(not(feature = "profile-onehot-fp128-d64"))]
struct ProfileMode {
    name: &'static str,
    run: ProfileModeRunner,
}

#[cfg(all(not(feature = "profile-onehot-fp128-d64"), feature = "profile-ci"))]
const PROFILE_CI_MODES: &[ProfileMode] = &[
    ProfileMode {
        name: "dense_fp128_d64",
        run: run_profile_dense_fp128_d64,
    },
    ProfileMode {
        name: "onehot_fp128_d64",
        run: run_profile_onehot_fp128_d64,
    },
    ProfileMode {
        name: "onehot_fp128_mixed_dim",
        run: run_profile_onehot_fp128_mixed_dim,
    },
    ProfileMode {
        name: "onehot_fp128_d64_multi_group_recursive",
        run: run_profile_onehot_fp128_d64_multi_group_recursive,
    },
    ProfileMode {
        name: "onehot_fp128_d64_multi_group_recursive_multi_chunk_w8r2",
        run: run_profile_onehot_fp128_d64_multi_group_recursive_multi_chunk_w8r2,
    },
    ProfileMode {
        name: "onehot_fp128_d64_tensor",
        run: run_profile_onehot_fp128_d64_tensor,
    },
    ProfileMode {
        name: "onehot_fp128_d64_multi_chunk_w2r2",
        run: run_profile_onehot_fp128_d64_multi_chunk_w2r2,
    },
    ProfileMode {
        name: "onehot_fp128_d64_multi_chunk_w4r2",
        run: run_profile_onehot_fp128_d64_multi_chunk_w4r2,
    },
    ProfileMode {
        name: "onehot_fp128_d64_multi_chunk_w8r2",
        run: run_profile_onehot_fp128_d64_multi_chunk_w8r2,
    },
    ProfileMode {
        name: "onehot_fp32_d128",
        run: run_profile_onehot_fp32_d128,
    },
    ProfileMode {
        name: "onehot_fp64_d128",
        run: run_profile_onehot_fp64_d128,
    },
];

#[cfg(all(not(feature = "profile-onehot-fp128-d64"), not(feature = "profile-ci")))]
const PROFILE_ALL_MODES: &[ProfileMode] = &[
    ProfileMode {
        name: "dense_fp128_d64",
        run: run_profile_dense_fp128_d64,
    },
    ProfileMode {
        name: "onehot_fp128_d64",
        run: run_profile_onehot_fp128_d64,
    },
    ProfileMode {
        name: "onehot_fp128_d64_multi_group_recursive",
        run: run_profile_onehot_fp128_d64_multi_group_recursive,
    },
    ProfileMode {
        name: "onehot_fp128_d64_multi_group_recursive_multi_chunk_w8r2",
        run: run_profile_onehot_fp128_d64_multi_group_recursive_multi_chunk_w8r2,
    },
    ProfileMode {
        name: "onehot_fp128_mixed_d_multi_group_recursive",
        run: run_profile_onehot_fp128_mixed_d_multi_group_recursive,
    },
    ProfileMode {
        name: "onehot_fp128_mixed_d_multi_group_recursive_multi_chunk_w8r2",
        run: run_profile_onehot_fp128_mixed_d_multi_group_recursive_multi_chunk_w8r2,
    },
    ProfileMode {
        name: "dense_fp128_d128",
        run: run_profile_dense_fp128_d128,
    },
    ProfileMode {
        name: "onehot_fp128_d128",
        run: run_profile_onehot_fp128_d128,
    },
    ProfileMode {
        name: "onehot_fp128_d64_root_d128",
        run: run_profile_onehot_fp128_d64_root_d128,
    },
    ProfileMode {
        name: "onehot_fp128_d64_tensor",
        run: run_profile_onehot_fp128_d64_tensor,
    },
    ProfileMode {
        name: "onehot_fp128_d64_multi_chunk_w2r2",
        run: run_profile_onehot_fp128_d64_multi_chunk_w2r2,
    },
    ProfileMode {
        name: "onehot_fp128_d64_multi_chunk_w4r2",
        run: run_profile_onehot_fp128_d64_multi_chunk_w4r2,
    },
    ProfileMode {
        name: "onehot_fp128_d64_multi_chunk_w8r2",
        run: run_profile_onehot_fp128_d64_multi_chunk_w8r2,
    },
    ProfileMode {
        name: "dense_fp32_d64",
        run: run_profile_dense_fp32_d64,
    },
    ProfileMode {
        name: "dense_fp32_d128",
        run: run_profile_dense_fp32_d128,
    },
    ProfileMode {
        name: "onehot_fp32_d64",
        run: run_profile_onehot_fp32_d64,
    },
    ProfileMode {
        name: "onehot_fp32_d128",
        run: run_profile_onehot_fp32_d128,
    },
    ProfileMode {
        name: "dense_fp64_d64",
        run: run_profile_dense_fp64_d64,
    },
    ProfileMode {
        name: "onehot_fp64_d64",
        run: run_profile_onehot_fp64_d64,
    },
    ProfileMode {
        name: "onehot_fp64_d128",
        run: run_profile_onehot_fp64_d128,
    },
];

#[cfg(not(feature = "profile-onehot-fp128-d64"))]
fn profile_modes() -> &'static [ProfileMode] {
    #[cfg(feature = "profile-ci")]
    {
        PROFILE_CI_MODES
    }
    #[cfg(not(feature = "profile-ci"))]
    {
        PROFILE_ALL_MODES
    }
}

/// Modes registered for explicit `AKITA_MODE=…` runs but omitted from `all`.
#[cfg(not(feature = "profile-onehot-fp128-d64"))]
const EXCLUDED_FROM_ALL_SWEEP: &[&str] = &[
    "onehot_fp128_d64_tensor",
    "onehot_fp128_d64_multi_chunk_w2r2",
    "onehot_fp128_d64_multi_chunk_w4r2",
    "onehot_fp128_d64_multi_chunk_w8r2",
    "onehot_fp128_d64_multi_group_recursive",
    "onehot_fp128_d64_multi_group_recursive_multi_chunk_w8r2",
    "onehot_fp128_mixed_d_multi_group_recursive",
    "onehot_fp128_mixed_d_multi_group_recursive_multi_chunk_w8r2",
    // D128+ presets are heavy and/or runtime-DP-backed; keep them out of the
    // default `all` smoke sweep (they are still selectable by explicit
    // `AKITA_MODE=` and drive the profile-bench matrix).
    "dense_fp128_d128",
    "onehot_fp128_d128",
    "onehot_fp128_d64_root_d128",
    "dense_fp32_d128",
    "onehot_fp32_d128",
    "onehot_fp64_d128",
];

fn assert_singleton_mode(mode: &str, num_polys: usize) {
    assert_eq!(
        num_polys, 1,
        "{mode} currently profiles only singleton commitments"
    );
}

fn fp128_onehot_title(d: usize, nv: usize, num_polys: usize) -> String {
    let onehot_k = onehot_k_for_num_vars(nv);
    let prime = fp128_prime_label();
    if num_polys == 1 {
        format!("=== onehot_fp128_d{d} (fp128, {prime}, D={d}, 1-of-{onehot_k}, log_commit_bound=1) ===")
    } else {
        format!(
            "=== onehot_fp128_d{d} batched (fp128, {prime}, D={d}, 1-of-{onehot_k}, log_commit_bound=1, same-point batch={num_polys}) ==="
        )
    }
}

fn small_field_schedule_source(d: usize) -> &'static str {
    if d >= 128 {
        "runtime DP schedule (no shipped D128 table)"
    } else {
        "generated small-field schedule"
    }
}

fn small_field_onehot_title(field_label: &str, d: usize, nv: usize, num_polys: usize) -> String {
    let onehot_k = onehot_k_for_num_vars(nv);
    let schedule = small_field_schedule_source(d);
    if num_polys == 1 {
        format!(
            "=== onehot_{field_label}_d{d} ({field_label}, D={d}, 1-of-{onehot_k}, {schedule}) ==="
        )
    } else {
        format!(
            "=== onehot_{field_label}_d{d} batched ({field_label}, D={d}, 1-of-{onehot_k}, same-point batch={num_polys}, {schedule}) ==="
        )
    }
}

#[cfg(not(feature = "profile-ci"))]
fn small_field_dense_title(field_label: &str, d: usize) -> String {
    let schedule = small_field_schedule_source(d);
    format!("=== dense_{field_label}_d{d} ({field_label}, D={d}, {schedule}) ===")
}

fn run_profile_dense_fp128_d64(nv: usize, num_polys: usize) {
    type Cfg = fp128::D64Dense;
    assert_singleton_mode("dense_fp128_d64", num_polys);
    let prime = fp128_prime_label();
    run_dense_mode::<{ Cfg::D }, Cfg>(
        "dense_fp128_d64",
        &format!("=== dense_fp128_d64 (fp128, {prime}, D=64 dense, log_commit_bound=128) ==="),
        nv,
    );
}

fn run_profile_onehot_fp128_d64(nv: usize, num_polys: usize) {
    type Cfg = fp128::D64OneHot;
    let title = fp128_onehot_title(64, nv, num_polys);
    run_onehot_mode::<{ Cfg::D }, Cfg>("onehot_fp128_d64", &title, nv, num_polys);
}

#[cfg(feature = "profile-ci")]
fn run_profile_onehot_fp128_mixed_dim(nv: usize, num_polys: usize) {
    type Cfg = fp128::MixedDimFp128OneHot;
    assert_eq!(nv, 32, "mixed-dimension profile fixes nv=32");
    assert_singleton_mode("onehot_fp128_mixed_dim", num_polys);

    let schedule = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(nv),
    ))
    .expect("generated mixed-dimension schedule");
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
        "generated mixed-dimension schedule selection"
    );

    let layout = resolve_layout::<F, Cfg>(nv);
    tracing::info!(
        "=== onehot_fp128_mixed_dim (fp128, flat public setup, generated per-level dimensions, 1-of-256) ==="
    );
    print_layout(&layout, 1, Cfg::decomposition().field_bits());
    // The catalog row selected here is the same exact row used by the PCS
    // prover and verifier. The benchmark intentionally does not compare it
    // against a different uniform-D family.
    run_onehot::<F, { Cfg::D }, Cfg>(
        "onehot_fp128_mixed_dim",
        nv,
        &layout,
        Some(&schedule),
        false,
    );
}

/// Shared driver for the recursive multi-group profiles. Every such profile
/// fixes the same shape (two precommitted 16-var singleton groups + a 32-var
/// main group with 2 polynomials, i.e. `num_polys == 4`); only the base preset
/// (`Cfg`) and the `layout_note` describing its witness layout differ.
fn run_recursive_multi_group_mode<
    const D: usize,
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
>(
    label: &str,
    layout_note: &str,
    nv: usize,
    num_polys: usize,
) {
    assert_eq!(nv, 32, "{label} fixes the main group at 32 variables");
    assert_eq!(
        num_polys, 4,
        "{label} opens two precommitted singleton groups plus two main polynomials"
    );
    tracing::info!(
        "=== {label} (fp128, {}, config D={D}, flat public setup, two precommitted 16-var singleton groups + 32-var main group with 2 polynomials, {layout_note}) ===",
        fp128_prime_label()
    );
    run_recursive_multi_group_onehot::<F, D, Cfg>(label, 16, 32, 2);
}

fn run_profile_onehot_fp128_d64_multi_group_recursive(nv: usize, num_polys: usize) {
    type Cfg = fp128::D64OneHot;
    run_recursive_multi_group_mode::<{ Cfg::D }, Cfg>(
        "onehot_fp128_d64_multi_group_recursive",
        "recursive setup",
        nv,
        num_polys,
    );
}

fn run_profile_onehot_fp128_d64_multi_group_recursive_multi_chunk_w8r2(
    nv: usize,
    num_polys: usize,
) {
    // `D64OneHotMultiChunk` is the production W8R2 preset (8 chunks x 2 leading
    // levels); the recursive adapter (applied inside
    // `run_recursive_multi_group_onehot`) adds setup offloading.
    type Cfg = fp128::D64OneHotMultiChunk;
    run_recursive_multi_group_mode::<{ Cfg::D }, Cfg>(
        "onehot_fp128_d64_multi_group_recursive_multi_chunk_w8r2",
        "recursive setup offloading + W8R2 chunked witness: num_chunks=8 x 2 leading levels",
        nv,
        num_polys,
    );
}

#[cfg(all(not(feature = "profile-onehot-fp128-d64"), not(feature = "profile-ci")))]
fn run_profile_onehot_fp128_mixed_d_multi_group_recursive(nv: usize, num_polys: usize) {
    type Cfg = RecursiveRingDimensionTransitionConfig<
        fp128::D256OneHot,
        fp128::D128OneHot,
        fp128::D64OneHot,
        fp128::D64OneHot,
        128,
        64,
    >;
    assert_eq!(nv, 32, "mixed recursive profile fixes nv=32");
    assert_eq!(
        num_polys, 4,
        "mixed recursive profile fixes four polynomials"
    );
    tracing::info!(
        "=== onehot_fp128_mixed_d_multi_group_recursive (fp128, {}, recursive setup, L0=256/128/128, L1=128/64/64, L2+=64/64/64) ===",
        fp128_prime_label()
    );
    run_recursive_multi_group_onehot_mixed::<F, { Cfg::D }, Cfg>(
        "onehot_fp128_mixed_d_multi_group_recursive",
        16,
        32,
        2,
    );
}

#[cfg(all(not(feature = "profile-onehot-fp128-d64"), not(feature = "profile-ci")))]
fn run_profile_onehot_fp128_mixed_d_multi_group_recursive_multi_chunk_w8r2(
    nv: usize,
    num_polys: usize,
) {
    type Cfg = RecursiveRingDimensionTransitionConfig<
        fp128::D256OneHot,
        fp128::D128OneHot,
        fp128::D64OneHot,
        fp128::D64OneHotMultiChunk,
        128,
        64,
    >;
    assert_eq!(nv, 32, "mixed recursive W8R2 profile fixes nv=32");
    assert_eq!(
        num_polys, 4,
        "mixed recursive W8R2 profile fixes four polynomials"
    );
    tracing::info!(
        "=== onehot_fp128_mixed_d_multi_group_recursive_multi_chunk_w8r2 (fp128, {}, recursive setup + W8R2, L0=256/128/128, L1=128/64/64, L2+=64/64/64) ===",
        fp128_prime_label()
    );
    run_recursive_multi_group_onehot_mixed::<F, { Cfg::D }, Cfg>(
        "onehot_fp128_mixed_d_multi_group_recursive_multi_chunk_w8r2",
        16,
        32,
        2,
    );
}

fn run_profile_onehot_fp128_d64_multi_chunk_named<
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
        "=== {label} (fp128, {prime}, D=64, 1-of-{onehot_k}, distributed chunked relation, num_chunks={} x {} leading levels) ===",
        profile.num_chunks(),
        profile.num_activated_levels(),
    );
    run_onehot_mode::<D, Cfg>(label, &title, nv, num_polys);
}

fn run_profile_onehot_fp128_d64_multi_chunk_w8r2(nv: usize, num_polys: usize) {
    run_profile_onehot_fp128_d64_multi_chunk_named::<64, fp128::D64OneHotMultiChunk>(
        "onehot_fp128_d64_multi_chunk_w8r2",
        MultiChunkProfileId::W8R2,
        nv,
        num_polys,
    );
}

fn run_profile_onehot_fp128_d64_multi_chunk_w2r2(nv: usize, num_polys: usize) {
    run_profile_onehot_fp128_d64_multi_chunk_named::<64, fp128::D64OneHotMultiChunkW2R2>(
        "onehot_fp128_d64_multi_chunk_w2r2",
        MultiChunkProfileId::W2R2,
        nv,
        num_polys,
    );
}

fn run_profile_onehot_fp128_d64_multi_chunk_w4r2(nv: usize, num_polys: usize) {
    run_profile_onehot_fp128_d64_multi_chunk_named::<64, fp128::D64OneHotMultiChunkW4R2>(
        "onehot_fp128_d64_multi_chunk_w4r2",
        MultiChunkProfileId::W4R2,
        nv,
        num_polys,
    );
}

fn run_profile_onehot_fp128_d64_tensor(nv: usize, num_polys: usize) {
    type Cfg = tensor_verifier::fp128::D64OneHotTensor;
    let prime = fp128_prime_label();
    let onehot_k = onehot_k_for_num_vars(nv);
    let title = if num_polys == 1 {
        format!(
            "=== onehot_fp128_d64_tensor (fp128, {prime}, D=64, 1-of-{onehot_k}, tensor-shaped root fold) ==="
        )
    } else {
        format!(
            "=== onehot_fp128_d64_tensor batched (fp128, {prime}, D=64, 1-of-{onehot_k}, tensor-shaped root fold, same-point batch={num_polys}) ==="
        )
    };
    run_onehot_mode::<{ Cfg::D }, Cfg>("onehot_fp128_d64_tensor", &title, nv, num_polys);
}

#[cfg(not(feature = "profile-ci"))]
fn run_profile_dense_fp128_d128(nv: usize, num_polys: usize) {
    type Cfg = fp128::D128Dense;
    assert_singleton_mode("dense_fp128_d128", num_polys);
    let prime = fp128_prime_label();
    run_dense_mode::<{ Cfg::D }, Cfg>(
        "dense_fp128_d128",
        &format!(
            "=== dense_fp128_d128 (fp128, {prime}, D=128 dense, log_commit_bound=128, runtime DP schedule) ==="
        ),
        nv,
    );
}

#[cfg(not(feature = "profile-ci"))]
fn run_profile_onehot_fp128_d128(nv: usize, num_polys: usize) {
    type Cfg = fp128::D128OneHot;
    let title = fp128_onehot_title(128, nv, num_polys);
    run_onehot_mode::<{ Cfg::D }, Cfg>("onehot_fp128_d128", &title, nv, num_polys);
}

/// Mixed ring-dimension-per-level experiment: the root fold (level 0) runs at
/// `D = 128` (via [`fp128::D128OneHot`]); every recursive level and the
/// terminal fold are repriced at `D = 64` (via [`fp128::D64OneHot`]). The
/// flat public matrix is shared by all levels; each scheduled matrix interprets
/// only its own exact field prefix at its own ring dimension.
#[cfg(all(not(feature = "profile-onehot-fp128-d64"), not(feature = "profile-ci")))]
fn run_profile_onehot_fp128_d64_root_d128(nv: usize, num_polys: usize) {
    let prime = fp128_prime_label();
    let onehot_k = onehot_k_for_num_vars(nv);
    // Levels `[0, switch)` fold at D=128; the rest at D=64. `AKITA_MIXED_SWITCH`
    // selects the switch point (default 1 = only the root at D=128). Switching
    // later keeps the large early folds uniform (fast compact range-check path)
    // and moves the D-transition penalty onto a small intermediate witness.
    let switch: usize = std::env::var("AKITA_MIXED_SWITCH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    // Root ring dimension for the leading uniform D-band (default 128).
    // Tableless D256 is planned offline by the mixed-schedule builder.
    let root_d: usize = std::env::var("AKITA_MIXED_ROOT_D")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128);
    assert_singleton_mode("onehot_fp128_d64_root_d128", num_polys);
    // Three-band ring-dimension transition: L0 = <root>/128/128, L1 =
    // 128/64/64, then uniform 64. The root A dimension is 256 or 512.
    if std::env::var("AKITA_THREE_BAND_RING_DIMENSION_TRANSITION").as_deref() == Ok("1") {
        run_three_band_ring_dimension_transition(nv);
        return;
    }
    // Multi-level ring-dimension transition: L0 = 128/128/64, L1 = 128/64/64,
    // then uniform 64.
    if std::env::var("AKITA_RING_DIMENSION_TRANSITION").as_deref() == Ok("1") {
        tracing::info!(
            "=== onehot_fp128_d64_root_d128 (fp128, {prime}, ring-dimension transition L0=128/128/64 L1=128/64/64 then 64, 1-of-{onehot_k}) ==="
        );
        run_ring_dimension_transition(nv);
        return;
    }
    // Per-matrix ring dimensions at the root: A = 128, with independently
    // selected B/D dimensions.
    // The complete D128 suffix is replanned from the resulting root witness.
    if std::env::var("AKITA_PER_MATRIX_RING_DIMS_ROOT").as_deref() == Ok("1") {
        let b_ring_dim: usize = std::env::var("AKITA_ROOT_B_RING_DIM")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(64);
        let d_ring_dim: usize = std::env::var("AKITA_ROOT_D_RING_DIM")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(64);
        tracing::info!(
            "=== onehot_fp128_d64_root_d128 (fp128, {prime}, root d_a=128 / d_b={b_ring_dim} / d_d={d_ring_dim}, 1-of-{onehot_k}) ==="
        );
        match (b_ring_dim, d_ring_dim) {
            (64, 64) => run_per_matrix_ring_dims_root::<64, 64>(nv),
            (128, 64) => run_per_matrix_ring_dims_root::<128, 64>(nv),
            (64, 128) => run_per_matrix_ring_dims_root::<64, 128>(nv),
            (o, p) => panic!(
                "AKITA_ROOT_B_RING_DIM={o} / AKITA_ROOT_D_RING_DIM={p} unsupported (use 64 or 128, must divide 128)"
            ),
        }
        return;
    }
    let title = format!(
        "=== onehot_fp128_d64_root_d128 (fp128, {prime}, D={root_d} for folds [0,{switch}) then D=64, 1-of-{onehot_k}, log_commit_bound=1) ==="
    );
    tracing::info!("{}", title);
    match (root_d, switch) {
        (128, 1) => run_mixed_root::<fp128::D128OneHot, 128, 1>(nv),
        (128, 2) => run_mixed_root::<fp128::D128OneHot, 128, 2>(nv),
        (128, 3) => run_mixed_root::<fp128::D128OneHot, 128, 3>(nv),
        (256, 1) => run_mixed_root::<fp128::D256OneHot, 256, 1>(nv),
        (256, 2) => run_mixed_root::<fp128::D256OneHot, 256, 2>(nv),
        (256, 3) => run_mixed_root::<fp128::D256OneHot, 256, 3>(nv),
        (d, s) => {
            panic!("AKITA_MIXED_ROOT_D={d} / AKITA_MIXED_SWITCH={s} unsupported (root_d 128|256, switch 1|2|3)")
        }
    }
}

/// Three-band ring-dimension transition: L0 = A/B/D `<root>`/128/128, L1 =
/// 128/64/64, then uniform 64.
#[cfg(all(not(feature = "profile-onehot-fp128-d64"), not(feature = "profile-ci")))]
fn run_three_band_ring_dimension_transition(nv: usize) {
    let root_d: usize = std::env::var("AKITA_THREE_BAND_ROOT_A_RING_DIM")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256);
    match root_d {
        256 => run_three_band_ring_dimension_transition_impl::<fp128::D256OneHot, 256>(nv),
        512 => run_three_band_ring_dimension_transition_impl::<fp128::D512OneHot, 512>(nv),
        d => panic!("AKITA_THREE_BAND_ROOT_A_RING_DIM={d} unsupported (use 256 or 512)"),
    }
}

/// Three-band ring-dimension transition: L0 = `ROOT_D`/128/128, L1 = 128/64/64,
/// then uniform 64. D512 is a temporary promotion experiment pending native
/// per-matrix ring-dimension planning.
#[cfg(all(not(feature = "profile-onehot-fp128-d64"), not(feature = "profile-ci")))]
fn run_three_band_ring_dimension_transition_impl<Root, const ROOT_D: usize>(nv: usize)
where
    Root: CommitmentConfig<Field = F, ExtField = F>,
{
    use akita_pcs::test_support::ThreeBandRingDimensionTransitionConfig;
    type Cfg<Root> =
        ThreeBandRingDimensionTransitionConfig<Root, fp128::D128OneHot, fp128::D64OneHot, 128, 64>;
    tracing::info!(
        "=== onehot_fp128_d64_root_d128 (fp128, {}, three-band L0={ROOT_D}/128/128 L1=128/64/64 then 64, 1-of-{}) ===",
        fp128_prime_label(),
        onehot_k_for_num_vars(nv),
    );
    let layout = resolve_layout::<F, Cfg<Root>>(nv);
    let required_vars =
        layout.position_index_bits() + layout.block_index_bits() + ROOT_D.trailing_zeros() as usize;
    if required_vars > nv {
        panic!(
            "[onehot_fp128_d64_root_d128] three-band requires {required_vars} variables, but AKITA_NUM_VARS={nv}"
        );
    }
    let plan = Cfg::<Root>::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(nv),
    ))
    .expect("three-band schedule plan");
    print_layout(&layout, 1, Cfg::<Root>::decomposition().field_bits());
    run_onehot::<F, ROOT_D, Cfg<Root>>(
        "onehot_fp128_d64_root_d128",
        nv,
        &layout,
        Some(&plan),
        false,
    );
}

#[cfg(all(not(feature = "profile-onehot-fp128-d64"), not(feature = "profile-ci")))]
fn run_ring_dimension_transition(nv: usize) {
    // Root opening-matrix ring dimension. L1 is always 128/64/64.
    let root_open_d: usize = std::env::var("AKITA_TRANSITION_ROOT_D_RING_DIM")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    match root_open_d {
        64 => run_ring_dimension_transition_impl::<64>(nv),
        128 => run_ring_dimension_transition_impl::<128>(nv),
        d => panic!("AKITA_TRANSITION_ROOT_D_RING_DIM={d} unsupported (use 64 or 128)"),
    }
}

#[cfg(all(not(feature = "profile-onehot-fp128-d64"), not(feature = "profile-ci")))]
fn run_ring_dimension_transition_impl<const ROOT_D_RING_DIM: usize>(nv: usize) {
    use akita_pcs::test_support::RingDimensionTransitionConfig;
    type Cfg<const R: usize> =
        RingDimensionTransitionConfig<fp128::D128OneHot, fp128::D64OneHot, 64, R>;
    let layout = resolve_layout::<F, Cfg<ROOT_D_RING_DIM>>(nv);
    let required_vars = layout.position_index_bits()
        + layout.block_index_bits()
        + 128usize.trailing_zeros() as usize;
    if required_vars > nv {
        panic!(
            "[onehot_fp128_d64_root_d128] ring-dimension transition requires {required_vars} variables, but AKITA_NUM_VARS={nv}"
        );
    }
    let plan = Cfg::<ROOT_D_RING_DIM>::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(nv),
    ))
    .expect("ring-dimension transition schedule");
    print_layout(
        &layout,
        1,
        Cfg::<ROOT_D_RING_DIM>::decomposition().field_bits(),
    );
    run_onehot::<F, 128, Cfg<ROOT_D_RING_DIM>>(
        "onehot_fp128_d64_root_d128",
        nv,
        &layout,
        Some(&plan),
        false,
    );
}

/// Run the per-matrix ring-dimension root experiment at A/B/D =
/// `128`/`B_RING_DIM`/`D_RING_DIM`, followed by a freshly planned D128 suffix.
#[cfg(all(not(feature = "profile-onehot-fp128-d64"), not(feature = "profile-ci")))]
fn run_per_matrix_ring_dims_root<const B_RING_DIM: usize, const D_RING_DIM: usize>(nv: usize) {
    use akita_pcs::test_support::PerMatrixRingDimsRootConfig;
    type Cfg<const O: usize, const P: usize> = PerMatrixRingDimsRootConfig<fp128::D128OneHot, O, P>;
    let layout = resolve_layout::<F, Cfg<B_RING_DIM, D_RING_DIM>>(nv);
    let required_vars = layout.position_index_bits()
        + layout.block_index_bits()
        + 128usize.trailing_zeros() as usize;
    if required_vars > nv {
        panic!(
            "[onehot_fp128_d64_root_d128] per-matrix ring-dimension root requires {required_vars} variables, but AKITA_NUM_VARS={nv}"
        );
    }
    let plan = Cfg::<B_RING_DIM, D_RING_DIM>::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(nv),
    ))
    .expect("per-matrix ring-dimension root schedule");
    print_layout(
        &layout,
        1,
        Cfg::<B_RING_DIM, D_RING_DIM>::decomposition().field_bits(),
    );
    run_onehot::<F, 128, Cfg<B_RING_DIM, D_RING_DIM>>(
        "onehot_fp128_d64_root_d128",
        nv,
        &layout,
        Some(&plan),
        false,
    );
}

/// Run the mixed-D experiment: `Env` (ring dim `ROOT_D`) is the leading-band
/// envelope, switching to `D64OneHot` after fold level `SWITCH_AT_FOLD - 1`.
#[cfg(all(not(feature = "profile-onehot-fp128-d64"), not(feature = "profile-ci")))]
fn run_mixed_root<Env, const ROOT_D: usize, const SWITCH_AT_FOLD: usize>(nv: usize)
where
    Env: CommitmentConfig<Field = F, ExtField = F>,
{
    type Cfg<Env, const S: usize> = MixedDConfig<Env, fp128::D64OneHot, S>;
    let layout = resolve_layout::<F, Cfg<Env, SWITCH_AT_FOLD>>(nv);
    let required_vars =
        layout.position_index_bits() + layout.block_index_bits() + ROOT_D.trailing_zeros() as usize;
    if required_vars > nv {
        panic!(
            "[onehot_fp128_d64_root_d128] fixed onehot profile requires {required_vars} variables, but AKITA_NUM_VARS={nv}"
        );
    }
    let plan = Cfg::<Env, SWITCH_AT_FOLD>::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(nv),
    ))
    .expect("mixed-D schedule plan");
    print_layout(
        &layout,
        1,
        Cfg::<Env, SWITCH_AT_FOLD>::decomposition().field_bits(),
    );
    // Commit + fold the root at the envelope ring dimension. Skip the planner
    // proof-size assertion: the mixed schedule is synthetic and the offline
    // planner cannot reproduce it from its lookup key.
    run_onehot::<F, ROOT_D, Cfg<Env, SWITCH_AT_FOLD>>(
        "onehot_fp128_d64_root_d128",
        nv,
        &layout,
        Some(&plan),
        false,
    );
}

#[cfg(not(feature = "profile-ci"))]
fn run_profile_onehot_fp32_d64(nv: usize, num_polys: usize) {
    type Cfg = fp32::D64OneHot;
    let title = small_field_onehot_title("fp32", Cfg::D, nv, num_polys);
    run_onehot_mode_for::<fp32::Field, { Cfg::D }, Cfg>("onehot_fp32_d64", &title, nv, num_polys);
}

#[cfg(not(feature = "profile-ci"))]
fn run_profile_dense_fp32_d64(nv: usize, num_polys: usize) {
    type Cfg = fp32::D64Dense;
    assert_singleton_mode("dense_fp32_d64", num_polys);
    let title = small_field_dense_title("fp32", Cfg::D);
    run_dense_mode_for::<fp32::Field, { Cfg::D }, Cfg>("dense_fp32_d64", &title, nv);
}

#[cfg(not(feature = "profile-ci"))]
fn run_profile_dense_fp32_d128(nv: usize, num_polys: usize) {
    type Cfg = fp32::D128Dense;
    assert_singleton_mode("dense_fp32_d128", num_polys);
    let title = small_field_dense_title("fp32", Cfg::D);
    run_dense_mode_for::<fp32::Field, { Cfg::D }, Cfg>("dense_fp32_d128", &title, nv);
}

fn run_profile_onehot_fp32_d128(nv: usize, num_polys: usize) {
    type Cfg = fp32::D128OneHot;
    let title = small_field_onehot_title("fp32", Cfg::D, nv, num_polys);
    run_onehot_mode_for::<fp32::Field, { Cfg::D }, Cfg>("onehot_fp32_d128", &title, nv, num_polys);
}

#[cfg(not(feature = "profile-ci"))]
fn run_profile_onehot_fp64_d64(nv: usize, num_polys: usize) {
    type Cfg = fp64::D64OneHot;
    let title = small_field_onehot_title("fp64", Cfg::D, nv, num_polys);
    run_onehot_mode_for::<fp64::Field, { Cfg::D }, Cfg>("onehot_fp64_d64", &title, nv, num_polys);
}

fn run_profile_onehot_fp64_d128(nv: usize, num_polys: usize) {
    type Cfg = fp64::D128OneHot;
    let title = small_field_onehot_title("fp64", Cfg::D, nv, num_polys);
    run_onehot_mode_for::<fp64::Field, { Cfg::D }, Cfg>("onehot_fp64_d128", &title, nv, num_polys);
}

#[cfg(not(feature = "profile-ci"))]
fn run_profile_dense_fp64_d64(nv: usize, num_polys: usize) {
    type Cfg = fp64::D64Dense;
    assert_singleton_mode("dense_fp64_d64", num_polys);
    let title = small_field_dense_title("fp64", Cfg::D);
    run_dense_mode_for::<fp64::Field, { Cfg::D }, Cfg>("dense_fp64_d64", &title, nv);
}

#[cfg(not(feature = "profile-onehot-fp128-d64"))]
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

#[cfg(not(feature = "profile-onehot-fp128-d64"))]
pub(crate) fn run_all_profile_modes(nv: usize) {
    for entry in profile_modes() {
        if EXCLUDED_FROM_ALL_SWEEP.contains(&entry.name) {
            continue;
        }
        run_profile_mode(entry.name, nv, 1);
    }
}

fn resolve_layout<FF, Cfg: CommitmentConfig<Field = FF>>(nv: usize) -> CommittedGroupParams {
    Cfg::get_params_for_batched_commitment(
        &akita_types::OpeningClaimsLayout::new(nv, 1).expect("singleton opening batch"),
    )
    .expect("layout")
}
#[cfg(feature = "profile-onehot-fp128-d64")]
pub(crate) fn run_profile_mode(mode: &str, nv: usize, num_polys: usize) {
    assert_eq!(
        mode, "onehot_fp128_d64",
        "profile-onehot-fp128-d64 only supports AKITA_MODE=onehot_fp128_d64",
    );
    assert_eq!(
        num_polys, 1,
        "profile-onehot-fp128-d64 only supports singleton commitments"
    );
    run_profile_onehot_fp128_d64(nv, num_polys);
}

pub(crate) fn log_active_fp128_prime_probe() {
    tracing::info!(
        "fp128 protocol prime active: modulus_offset = 0x{:x}, probe(2^128 + 1) = 0x{:x}",
        <F as PseudoMersenneField>::MODULUS_OFFSET,
        F::solinas_reduce(&[1u64, 0, 1]).to_canonical_u128(),
    );
}
