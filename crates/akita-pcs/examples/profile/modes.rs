use crate::report::print_layout;
use crate::workload::{
    onehot_k_for_num_vars, run_batched_onehot, run_dense, run_dense_for, run_onehot,
};
use akita_config::proof_optimized::{fp128, fp16, fp32, fp64};
use akita_config::{akita_batched_root_layout, CommitmentConfig};
use akita_field::fields::wide::HasWide;
use akita_field::{
    CanonicalBytes, CanonicalField, FrobeniusExtField, FromPrimitiveInt, PseudoMersenneField,
    RandomSampling,
};
use akita_field::{ExtField, TranscriptChallenge};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_serialization::AkitaSerialize;
use akita_types::{
    AkitaBatchedProof, AkitaCommitmentHint, AkitaScheduleLookupKey, AkitaVerifierSetup,
    LevelParams, RingCommitment, RingSubfieldEncoding,
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

fn best_full_d(nv: usize) -> usize {
    let key = AkitaScheduleLookupKey::singleton(nv);
    fp128::best_full_schedule(key)
        .expect("best full schedule selection")
        .map(|selection| selection.preset.ring_dimension())
        .unwrap_or(32)
}

fn best_onehot_d(nv: usize, num_polys: usize) -> usize {
    let key = AkitaScheduleLookupKey::new(nv, num_polys, num_polys, 1);
    fp128::best_onehot_schedule(key)
        .expect("best onehot schedule selection")
        .map(|selection| selection.preset.ring_dimension())
        .unwrap_or(32)
}

fn run_dense_mode<
    const D: usize,
    Cfg: CommitmentConfig<Field = F, ClaimField = F, ChallengeField = F>,
>(
    title: &str,
    nv: usize,
) {
    let layout = resolve_layout::<F, Cfg>(nv);
    let plan = Cfg::schedule_plan(AkitaScheduleLookupKey::singleton(nv)).expect("schedule plan");
    tracing::info!("{}", title);
    print_layout(&layout);
    run_dense::<D, Cfg>(nv, &layout, plan.as_ref());
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
            ClaimField = Cfg::ClaimField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ChallengeField>,
            CommitHint = AkitaCommitmentHint<FF, D>,
        > + CommitmentVerifier<
            FF,
            D,
            ClaimField = Cfg::ClaimField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ChallengeField>,
        >,
    Cfg::ClaimField: FrobeniusExtField<FF> + RingSubfieldEncoding<FF> + AkitaSerialize,
    Cfg::ChallengeField: RingSubfieldEncoding<FF> + ExtField<Cfg::ClaimField> + AkitaSerialize,
{
    let (protocol_nv, num_points, num_t_vectors, num_w_vectors, num_z_vectors) =
        if Cfg::CLAIM_EXT_DEGREE > 1 {
            let split_bits = Cfg::CLAIM_EXT_DEGREE.trailing_zeros() as usize;
            let width = 1usize << split_bits;
            (
                nv.checked_sub(split_bits)
                    .expect("Frobenius split must not exceed dense arity")
                    + split_bits,
                1,
                1,
                width,
                width,
            )
        } else {
            (nv, 1, 1, 1, 1)
        };
    let layout = resolve_layout::<FF, Cfg>(protocol_nv);
    let schedule_key = AkitaScheduleLookupKey::new_with_points(
        protocol_nv,
        num_points,
        num_t_vectors,
        num_w_vectors,
        num_z_vectors,
    );
    let plan = Cfg::schedule_plan(schedule_key).expect("schedule plan");
    tracing::info!("{}", title);
    print_layout(&layout);
    run_dense_for::<FF, D, Cfg>(label, nv, &layout, plan.as_ref());
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
            ClaimField = Cfg::ClaimField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ChallengeField>,
            CommitHint = AkitaCommitmentHint<FF, D>,
        > + CommitmentVerifier<
            FF,
            D,
            ClaimField = Cfg::ClaimField,
            VerifierSetup = AkitaVerifierSetup<FF>,
            Commitment = RingCommitment<FF, D>,
            BatchedProof = AkitaBatchedProof<FF, Cfg::ChallengeField>,
        >,
    Cfg::ClaimField: FrobeniusExtField<FF> + RingSubfieldEncoding<FF> + AkitaSerialize,
    Cfg::ChallengeField: RingSubfieldEncoding<FF> + ExtField<Cfg::ClaimField> + AkitaSerialize,
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
        let plan =
            Cfg::schedule_plan(AkitaScheduleLookupKey::singleton(nv)).expect("schedule plan");
        print_layout(&layout);
        run_onehot::<FF, D, Cfg>(label, nv, &layout, plan.as_ref());
    } else {
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
        print_layout(&layout);
        run_batched_onehot::<FF, D, Cfg>(label, nv, num_polys, &layout);
    }
}

fn run_onehot_mode<
    const D: usize,
    Cfg: CommitmentConfig<Field = F, ClaimField = F, ChallengeField = F>,
>(
    label: &str,
    title: &str,
    nv: usize,
    num_polys: usize,
) where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            F,
            D,
            ClaimField = F,
            VerifierSetup = AkitaVerifierSetup<F>,
            Commitment = RingCommitment<F, D>,
            BatchedProof = AkitaBatchedProof<F, F>,
            CommitHint = AkitaCommitmentHint<F, D>,
        > + CommitmentVerifier<
            F,
            D,
            ClaimField = F,
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
        name: "full",
        run: run_profile_full,
    },
    ProfileMode {
        name: "onehot",
        run: run_profile_onehot,
    },
    ProfileMode {
        name: "full_d64",
        run: run_profile_full_d64,
    },
    ProfileMode {
        name: "onehot_d64",
        run: run_profile_onehot_d64,
    },
    ProfileMode {
        name: "full_d32",
        run: run_profile_full_d32,
    },
    ProfileMode {
        name: "onehot_d32",
        run: run_profile_onehot_d32,
    },
    ProfileMode {
        name: "onehot_fp32",
        run: run_profile_onehot_fp32,
    },
    ProfileMode {
        name: "onehot_fp32_d32",
        run: run_profile_onehot_fp32_d32,
    },
    ProfileMode {
        name: "onehot_fp32_d64",
        run: run_profile_onehot_fp32_d64,
    },
    ProfileMode {
        name: "dense_fp32_d32",
        run: run_profile_dense_fp32_d32,
    },
    ProfileMode {
        name: "dense_fp32_d64",
        run: run_profile_dense_fp32_d64,
    },
    ProfileMode {
        name: "onehot_fp16_d32",
        run: run_profile_onehot_fp16_d32,
    },
    ProfileMode {
        name: "full_fp16_d32",
        run: run_profile_full_fp16_d32,
    },
    ProfileMode {
        name: "onehot_fp16_d64",
        run: run_profile_onehot_fp16_d64,
    },
    ProfileMode {
        name: "full_fp16_d64",
        run: run_profile_full_fp16_d64,
    },
    ProfileMode {
        name: "onehot_fp64",
        run: run_profile_onehot_fp64,
    },
    ProfileMode {
        name: "onehot_fp64_d32",
        run: run_profile_onehot_fp64_d32,
    },
    ProfileMode {
        name: "onehot_fp64_d64",
        run: run_profile_onehot_fp64_d64,
    },
    ProfileMode {
        name: "dense_fp64_d32",
        run: run_profile_dense_fp64_d32,
    },
    ProfileMode {
        name: "dense_fp64_d64",
        run: run_profile_dense_fp64_d64,
    },
];

const ALL_PROFILE_MODE_NAMES: &[&str] = &[
    "full",
    "onehot",
    "full_d64",
    "onehot_d64",
    "full_d32",
    "onehot_d32",
    "onehot_fp32",
    "onehot_fp32_d32",
    "onehot_fp32_d64",
    "dense_fp32_d32",
    "dense_fp32_d64",
    "onehot_fp16_d32",
    "full_fp16_d32",
    "onehot_fp16_d64",
    "full_fp16_d64",
    "onehot_fp64",
    "onehot_fp64_d32",
    "onehot_fp64_d64",
    "dense_fp64_d32",
    "dense_fp64_d64",
];

fn assert_singleton_mode(mode: &str, num_polys: usize) {
    assert_eq!(
        num_polys, 1,
        "{mode} currently profiles only singleton commitments"
    );
}

fn fixed_onehot_title(d: usize, nv: usize, num_polys: usize) -> String {
    let onehot_k = onehot_k_for_num_vars(nv);
    let prime = fp128_prime_label();
    if num_polys == 1 {
        format!("=== onehot_d{d} ({prime}, D={d}, 1-of-{onehot_k}, log_commit_bound=1) ===")
    } else {
        format!(
            "=== onehot_d{d} batched ({prime}, D={d}, 1-of-{onehot_k}, log_commit_bound=1, same-point batch={num_polys}) ==="
        )
    }
}

fn small_field_onehot_title(field_label: &str, d: usize, nv: usize, num_polys: usize) -> String {
    let onehot_k = onehot_k_for_num_vars(nv);
    if num_polys == 1 {
        format!("=== onehot_{field_label} ({field_label}, D={d}, 1-of-{onehot_k}, generated small-field schedule) ===")
    } else {
        format!(
            "=== onehot_{field_label} batched ({field_label}, D={d}, 1-of-{onehot_k}, same-point batch={num_polys}, generated small-field schedule) ==="
        )
    }
}

fn small_field_dense_title(field_label: &str, d: usize) -> String {
    format!(
        "=== dense_{field_label}_d{d} ({field_label}, D={d}, generated small-field schedule) ==="
    )
}

fn run_profile_full(nv: usize, num_polys: usize) {
    assert_singleton_mode("full", num_polys);
    let d = best_full_d(nv);
    let prime = fp128_prime_label();
    let title = format!("=== full ({prime}, D={d}, dense) ===");
    match d {
        32 => run_dense_mode::<32, fp128::D32Full>(&title, nv),
        64 => run_dense_mode::<64, fp128::D64Full>(&title, nv),
        _ => unreachable!(),
    }
}

fn run_profile_onehot(nv: usize, num_polys: usize) {
    let onehot_k = onehot_k_for_num_vars(nv);
    let d = best_onehot_d(nv, num_polys);
    let prime = fp128_prime_label();
    let title = if num_polys == 1 {
        format!("=== onehot ({prime}, D={d}, 1-of-{onehot_k}) ===")
    } else {
        format!(
            "=== onehot batched ({prime}, D={d}, 1-of-{onehot_k}, same-point batch={num_polys}) ==="
        )
    };
    match d {
        32 => run_onehot_mode::<32, fp128::D32OneHot>("onehot", &title, nv, num_polys),
        64 => run_onehot_mode::<64, fp128::D64OneHot>("onehot", &title, nv, num_polys),
        _ => unreachable!(),
    }
}

fn run_profile_full_d64(nv: usize, num_polys: usize) {
    type Cfg = fp128::D64Full;
    assert_singleton_mode("full_d64", num_polys);
    let prime = fp128_prime_label();
    run_dense_mode::<{ Cfg::D }, Cfg>(
        &format!("=== full_d64 ({prime}, D=64 dense, log_commit_bound=128) ==="),
        nv,
    );
}

fn run_profile_onehot_d64(nv: usize, num_polys: usize) {
    type Cfg = fp128::D64OneHot;
    let title = fixed_onehot_title(64, nv, num_polys);
    run_onehot_mode::<{ Cfg::D }, Cfg>("onehot_d64", &title, nv, num_polys);
}

fn run_profile_full_d32(nv: usize, num_polys: usize) {
    type Cfg = fp128::D32Full;
    assert_singleton_mode("full_d32", num_polys);
    let prime = fp128_prime_label();
    run_dense_mode::<{ Cfg::D }, Cfg>(
        &format!("=== full_d32 ({prime}, D=32 dense, log_commit_bound=128) ==="),
        nv,
    );
}

fn run_profile_onehot_d32(nv: usize, num_polys: usize) {
    type Cfg = fp128::D32OneHot;
    let title = fixed_onehot_title(32, nv, num_polys);
    run_onehot_mode::<{ Cfg::D }, Cfg>("onehot_d32", &title, nv, num_polys);
}

fn run_profile_onehot_fp32(nv: usize, num_polys: usize) {
    run_profile_onehot_fp32_d32_with_label("onehot_fp32", nv, num_polys);
}

fn run_profile_onehot_fp32_d32(nv: usize, num_polys: usize) {
    run_profile_onehot_fp32_d32_with_label("onehot_fp32_d32", nv, num_polys);
}

fn run_profile_onehot_fp32_d32_with_label(label: &str, nv: usize, num_polys: usize) {
    type Cfg = fp32::D32OneHot;
    let title = small_field_onehot_title("fp32", Cfg::D, nv, num_polys);
    run_onehot_mode_for::<fp32::Field, { Cfg::D }, Cfg>(label, &title, nv, num_polys);
}

fn run_profile_onehot_fp32_d64(nv: usize, num_polys: usize) {
    run_profile_onehot_fp32_d64_with_label("onehot_fp32_d64", nv, num_polys);
}

fn run_profile_onehot_fp32_d64_with_label(label: &str, nv: usize, num_polys: usize) {
    type Cfg = fp32::D64OneHot;
    let title = small_field_onehot_title("fp32", Cfg::D, nv, num_polys);
    run_onehot_mode_for::<fp32::Field, { Cfg::D }, Cfg>(label, &title, nv, num_polys);
}

fn run_profile_dense_fp32_d32(nv: usize, num_polys: usize) {
    type Cfg = fp32::D32Full;
    assert_singleton_mode("dense_fp32_d32", num_polys);
    let title = small_field_dense_title("fp32", Cfg::D);
    run_dense_mode_for::<fp32::Field, { Cfg::D }, Cfg>("dense_fp32_d32", &title, nv);
}

fn run_profile_dense_fp32_d64(nv: usize, num_polys: usize) {
    type Cfg = fp32::D64Full;
    assert_singleton_mode("dense_fp32_d64", num_polys);
    let title = small_field_dense_title("fp32", Cfg::D);
    run_dense_mode_for::<fp32::Field, { Cfg::D }, Cfg>("dense_fp32_d64", &title, nv);
}

fn run_profile_onehot_fp16_d32(nv: usize, num_polys: usize) {
    type Cfg = fp16::D32OneHot;
    let title = small_field_onehot_title("fp16", Cfg::D, nv, num_polys);
    run_onehot_mode_for::<fp16::Field, { Cfg::D }, Cfg>("onehot_fp16_d32", &title, nv, num_polys);
}

fn run_profile_full_fp16_d32(nv: usize, num_polys: usize) {
    type Cfg = fp16::D32Full;
    assert_singleton_mode("full_fp16_d32", num_polys);
    let title = small_field_dense_title("fp16", Cfg::D);
    run_dense_mode_for::<fp16::Field, { Cfg::D }, Cfg>("full_fp16_d32", &title, nv);
}

fn run_profile_onehot_fp16_d64(nv: usize, num_polys: usize) {
    type Cfg = fp16::D64OneHot;
    let title = small_field_onehot_title("fp16", Cfg::D, nv, num_polys);
    run_onehot_mode_for::<fp16::Field, { Cfg::D }, Cfg>("onehot_fp16_d64", &title, nv, num_polys);
}

fn run_profile_full_fp16_d64(nv: usize, num_polys: usize) {
    type Cfg = fp16::D64Full;
    assert_singleton_mode("full_fp16_d64", num_polys);
    let title = small_field_dense_title("fp16", Cfg::D);
    run_dense_mode_for::<fp16::Field, { Cfg::D }, Cfg>("full_fp16_d64", &title, nv);
}

fn run_profile_onehot_fp64(nv: usize, num_polys: usize) {
    run_profile_onehot_fp64_d64_with_label("onehot_fp64", nv, num_polys);
}

fn run_profile_onehot_fp64_d32(nv: usize, num_polys: usize) {
    type Cfg = fp64::D32OneHot;
    let title = small_field_onehot_title("fp64", Cfg::D, nv, num_polys);
    run_onehot_mode_for::<fp64::Field, { Cfg::D }, Cfg>("onehot_fp64_d32", &title, nv, num_polys);
}

fn run_profile_onehot_fp64_d64(nv: usize, num_polys: usize) {
    run_profile_onehot_fp64_d64_with_label("onehot_fp64_d64", nv, num_polys);
}

fn run_profile_onehot_fp64_d64_with_label(label: &str, nv: usize, num_polys: usize) {
    type Cfg = fp64::D64OneHot;
    let title = small_field_onehot_title("fp64", Cfg::D, nv, num_polys);
    run_onehot_mode_for::<fp64::Field, { Cfg::D }, Cfg>(label, &title, nv, num_polys);
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
    for mode in ALL_PROFILE_MODE_NAMES {
        run_profile_mode(mode, nv, 1);
    }
}

fn resolve_layout<FF, Cfg: CommitmentConfig<Field = FF>>(nv: usize) -> LevelParams {
    Cfg::commitment_layout(nv).expect("layout")
}

pub(crate) fn log_active_fp128_prime_probe() {
    tracing::info!(
        "fp128 protocol prime active: modulus_offset = 0x{:x}, probe(2^128 + 1) = 0x{:x}",
        <F as PseudoMersenneField>::MODULUS_OFFSET,
        F::solinas_reduce(&[1u64, 0, 1]).to_canonical_u128(),
    );
}
