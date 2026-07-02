use crate::report::print_layout;
use crate::workload::{onehot_k_for_num_vars, run_batched_onehot, run_dense_for, run_onehot};
use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::tensor_verifier;
use akita_config::test_support::akita_batched_root_layout;
use akita_config::CommitmentConfig;
use akita_field::unreduced::HasWide;
use akita_field::TranscriptChallenge;
use akita_field::{
    CanonicalBytes, CanonicalField, FrobeniusExtField, FromPrimitiveInt, PseudoMersenneField,
    RandomSampling,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{AkitaProverSetup, CommitmentProver};
use akita_serialization::AkitaSerialize;
use akita_types::{
    AkitaBatchedProof, AkitaCommitmentHint, AkitaScheduleLookupKey, AkitaVerifierSetup,
    CommitmentGroupScheduleKey, FpExtEncoding, LevelParams, MultiChunkProfileId, RingCommitment,
};
use akita_verifier::CommitmentVerifier;

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
        CommitmentGroupScheduleKey::singleton(nv),
    ))
    .expect("schedule plan");
    tracing::info!("{}", title);
    print_layout(&layout, 1, Cfg::decomposition().field_bits());
    run_dense_for::<F, D, Cfg>(label, nv, &layout, Some(&plan));
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
        + HasWide
        + AkitaSerialize
        + 'static,
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            FF,
            D,
            ProverSetup = AkitaProverSetup<FF, D>,
            ExtField = Cfg::ExtField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ExtField>,
            CommitHint = AkitaCommitmentHint<FF, D>,
        > + CommitmentVerifier<
            FF,
            D,
            ExtField = Cfg::ExtField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ExtField>,
        >,
    Cfg::ExtField: FrobeniusExtField<FF> + FpExtEncoding<FF> + AkitaSerialize,
    Cfg::ExtField: FpExtEncoding<FF> + AkitaSerialize,
{
    // The dense profile opens one polynomial at one point, so the schedule key
    // is the singleton root the prover actually resolves via
    // `new_from_opening_batch`.
    let layout = resolve_layout::<FF, Cfg>(nv);
    let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
        CommitmentGroupScheduleKey::singleton(nv),
    ))
    .expect("schedule plan");
    tracing::info!("{}", title);
    print_layout(&layout, 1, Cfg::decomposition().field_bits());
    run_dense_for::<FF, D, Cfg>(label, nv, &layout, Some(&plan));
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
        + HasWide
        + AkitaSerialize
        + 'static,
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            FF,
            D,
            ProverSetup = AkitaProverSetup<FF, D>,
            ExtField = Cfg::ExtField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ExtField>,
            CommitHint = AkitaCommitmentHint<FF, D>,
        > + CommitmentVerifier<
            FF,
            D,
            ExtField = Cfg::ExtField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ExtField>,
        >,
    Cfg::ExtField: FrobeniusExtField<FF> + FpExtEncoding<FF> + AkitaSerialize,
    Cfg::ExtField: FpExtEncoding<FF> + AkitaSerialize,
{
    tracing::info!("{}", title);
    if num_polys == 1 {
        let layout = resolve_layout::<FF, Cfg>(nv);
        let required_vars = layout.m_vars + layout.r_vars + D.trailing_zeros() as usize;
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
            CommitmentGroupScheduleKey::singleton(nv),
        ))
        .expect("schedule plan");
        print_layout(&layout, 1, Cfg::decomposition().field_bits());
        run_onehot::<FF, D, Cfg>(label, nv, &layout, Some(&plan));
    } else {
        let schedule_key = CommitmentGroupScheduleKey::new(nv, num_polys);
        let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(schedule_key))
            .expect("schedule plan");
        let layout = akita_batched_root_layout::<Cfg>(nv, num_polys).expect("layout");
        let required_vars = layout.m_vars + layout.r_vars + D.trailing_zeros() as usize;
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
) where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            F,
            D,
            ProverSetup = AkitaProverSetup<F, D>,
            ExtField = F,
            VerifierSetup = AkitaVerifierSetup<F>,
            Commitment = RingCommitment<F, D>,
            BatchedProof = AkitaBatchedProof<F, F>,
            CommitHint = AkitaCommitmentHint<F, D>,
        > + CommitmentVerifier<
            F,
            D,
            ExtField = F,
            VerifierSetup = AkitaVerifierSetup<F>,
            Commitment = RingCommitment<F, D>,
            BatchedProof = AkitaBatchedProof<F, F>,
        >,
{
    run_onehot_mode_for::<F, D, Cfg>(label, title, nv, num_polys);
}

type ProfileModeRunner = fn(usize, usize);

struct ProfileMode {
    name: &'static str,
    run: ProfileModeRunner,
}

const PROFILE_MODES: &[ProfileMode] = &[
    ProfileMode {
        name: "dense_fp128_d32",
        run: run_profile_dense_fp128_d32,
    },
    ProfileMode {
        name: "dense_fp128_d64",
        run: run_profile_dense_fp128_d64,
    },
    ProfileMode {
        name: "onehot_fp128_d32",
        run: run_profile_onehot_fp128_d32,
    },
    ProfileMode {
        name: "onehot_fp128_d64",
        run: run_profile_onehot_fp128_d64,
    },
    ProfileMode {
        name: "dense_fp128_d128",
        run: run_profile_dense_fp128_d128,
    },
    ProfileMode {
        name: "onehot_fp128_d128",
        run: run_profile_onehot_fp128_d128,
    },
    // Direct comparison mode from the tensor-verifier branch. The generated
    // tensor preset is D64-only, so this is intentionally not part of the D32
    // benchmark matrix or `AKITA_MODE=all` sweep.
    ProfileMode {
        name: "onehot_fp128_d64_tensor",
        run: run_profile_onehot_fp128_d64_tensor,
    },
    // Chunked relation (distributed prover witness layout on leading fold levels).
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
        name: "dense_fp64_d32",
        run: run_profile_dense_fp64_d32,
    },
    ProfileMode {
        name: "dense_fp64_d64",
        run: run_profile_dense_fp64_d64,
    },
    ProfileMode {
        name: "onehot_fp64_d32",
        run: run_profile_onehot_fp64_d32,
    },
    ProfileMode {
        name: "onehot_fp64_d64",
        run: run_profile_onehot_fp64_d64,
    },
    // Smallest securable fp64 one-hot family under honest committed-fold
    // pricing (fp64 ships generated tables only at D128/D256; D32/D64 no
    // longer fold securely). This is the fp64 cell the profile-bench matrix
    // tracks.
    ProfileMode {
        name: "onehot_fp64_d128",
        run: run_profile_onehot_fp64_d128,
    },
];

/// Modes registered for explicit `AKITA_MODE=…` runs but omitted from `all`.
const EXCLUDED_FROM_ALL_SWEEP: &[&str] = &[
    "onehot_fp128_d64_tensor",
    "onehot_fp128_d64_multi_chunk_w2r2",
    "onehot_fp128_d64_multi_chunk_w4r2",
    "onehot_fp128_d64_multi_chunk_w8r2",
    // D128+ presets are heavy and/or runtime-DP-backed; keep them out of the
    // default `all` smoke sweep (they are still selectable by explicit
    // `AKITA_MODE=` and drive the profile-bench matrix).
    "dense_fp128_d128",
    "onehot_fp128_d128",
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

fn small_field_dense_title(field_label: &str, d: usize) -> String {
    let schedule = small_field_schedule_source(d);
    format!("=== dense_{field_label}_d{d} ({field_label}, D={d}, {schedule}) ===")
}

fn run_profile_dense_fp128_d64(nv: usize, num_polys: usize) {
    type Cfg = fp128::D64Full;
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

fn run_profile_dense_fp128_d32(nv: usize, num_polys: usize) {
    type Cfg = fp128::D32Full;
    assert_singleton_mode("dense_fp128_d32", num_polys);
    let prime = fp128_prime_label();
    run_dense_mode::<{ Cfg::D }, Cfg>(
        "dense_fp128_d32",
        &format!("=== dense_fp128_d32 (fp128, {prime}, D=32 dense, log_commit_bound=128) ==="),
        nv,
    );
}

fn run_profile_onehot_fp128_d32(nv: usize, num_polys: usize) {
    type Cfg = fp128::D32OneHot;
    let title = fp128_onehot_title(32, nv, num_polys);
    run_onehot_mode::<{ Cfg::D }, Cfg>("onehot_fp128_d32", &title, nv, num_polys);
}

fn run_profile_dense_fp128_d128(nv: usize, num_polys: usize) {
    type Cfg = fp128::D128Full;
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

fn run_profile_onehot_fp128_d128(nv: usize, num_polys: usize) {
    type Cfg = fp128::D128OneHot;
    let title = fp128_onehot_title(128, nv, num_polys);
    run_onehot_mode::<{ Cfg::D }, Cfg>("onehot_fp128_d128", &title, nv, num_polys);
}

fn run_profile_onehot_fp32_d64(nv: usize, num_polys: usize) {
    type Cfg = fp32::D64OneHot;
    let title = small_field_onehot_title("fp32", Cfg::D, nv, num_polys);
    run_onehot_mode_for::<fp32::Field, { Cfg::D }, Cfg>("onehot_fp32_d64", &title, nv, num_polys);
}

fn run_profile_dense_fp32_d64(nv: usize, num_polys: usize) {
    type Cfg = fp32::D64Full;
    assert_singleton_mode("dense_fp32_d64", num_polys);
    let title = small_field_dense_title("fp32", Cfg::D);
    run_dense_mode_for::<fp32::Field, { Cfg::D }, Cfg>("dense_fp32_d64", &title, nv);
}

fn run_profile_dense_fp32_d128(nv: usize, num_polys: usize) {
    type Cfg = fp32::D128Full;
    assert_singleton_mode("dense_fp32_d128", num_polys);
    let title = small_field_dense_title("fp32", Cfg::D);
    run_dense_mode_for::<fp32::Field, { Cfg::D }, Cfg>("dense_fp32_d128", &title, nv);
}

fn run_profile_onehot_fp32_d128(nv: usize, num_polys: usize) {
    type Cfg = fp32::D128OneHot;
    let title = small_field_onehot_title("fp32", Cfg::D, nv, num_polys);
    run_onehot_mode_for::<fp32::Field, { Cfg::D }, Cfg>("onehot_fp32_d128", &title, nv, num_polys);
}

fn run_profile_onehot_fp64_d32(nv: usize, num_polys: usize) {
    type Cfg = fp64::D32OneHot;
    let title = small_field_onehot_title("fp64", Cfg::D, nv, num_polys);
    run_onehot_mode_for::<fp64::Field, { Cfg::D }, Cfg>("onehot_fp64_d32", &title, nv, num_polys);
}

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

fn run_profile_dense_fp64_d32(nv: usize, num_polys: usize) {
    type Cfg = fp64::D32Full;
    assert_singleton_mode("dense_fp64_d32", num_polys);
    let title = small_field_dense_title("fp64", Cfg::D);
    run_dense_mode_for::<fp64::Field, { Cfg::D }, Cfg>("dense_fp64_d32", &title, nv);
}

fn run_profile_dense_fp64_d64(nv: usize, num_polys: usize) {
    type Cfg = fp64::D64Full;
    assert_singleton_mode("dense_fp64_d64", num_polys);
    let title = small_field_dense_title("fp64", Cfg::D);
    run_dense_mode_for::<fp64::Field, { Cfg::D }, Cfg>("dense_fp64_d64", &title, nv);
}

pub(crate) fn run_profile_mode(mode: &str, nv: usize, num_polys: usize) {
    let profile_mode = PROFILE_MODES
        .iter()
        .find(|entry| entry.name == mode)
        .unwrap_or_else(|| {
            let mut known_modes = PROFILE_MODES
                .iter()
                .map(|entry| entry.name)
                .collect::<Vec<_>>();
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

pub(crate) fn run_all_profile_modes(nv: usize) {
    for entry in PROFILE_MODES {
        if EXCLUDED_FROM_ALL_SWEEP.contains(&entry.name) {
            continue;
        }
        run_profile_mode(entry.name, nv, 1);
    }
}

fn resolve_layout<FF, Cfg: CommitmentConfig<Field = FF>>(nv: usize) -> LevelParams {
    Cfg::get_params_for_batched_commitment(
        &akita_types::OpeningBatchShape::new(nv, 1).expect("singleton opening batch"),
    )
    .expect("layout")
}

pub(crate) fn log_active_fp128_prime_probe() {
    tracing::info!(
        "fp128 protocol prime active: modulus_offset = 0x{:x}, probe(2^128 + 1) = 0x{:x}",
        <F as PseudoMersenneField>::MODULUS_OFFSET,
        F::solinas_reduce(&[1u64, 0, 1]).to_canonical_u128(),
    );
}
