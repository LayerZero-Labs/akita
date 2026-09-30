//! Prover and verifier transcript event sequences agree on every catalog shape.
//!
//! # Row selection
//!
//! Proving every checked-in row is too slow for the transcript-semantics job:
//! the catalogs hold rows up to 50 variables. This test instead proves one row
//! per distinct schedule shape, where a shape is
//!
//! - the catalog family, which fixes the base field, extension degree, and
//!   committed source class;
//! - the number of committed groups, and whether any group batches more than
//!   one polynomial;
//! - for every nonterminal fold level, in order, its payload mode, ring
//!   relation mode, whether it commits a multi-chunk witness, whether its own
//!   group opens by subring coefficient packing, its norm route (L-infinity
//!   digit range, direct L2, or limb-Gram L2), and whether it consumes a setup
//!   prefix, which makes its predecessor run Stage 3.
//!
//! These are the schedule facts that choose which proof-stream sites a proof
//! emits. Rows of one shape differ only in sizes, digit counts, and matrix
//! dimensions, which change atom lengths and loop trip counts of sites the
//! shape already exercises. Within a shape the row with the fewest committed
//! coefficients is proved.
//!
//! Shapes whose cheapest row commits more than `2^MAX_LOG_COMMITTED_COEFFS`
//! coefficients are skipped and logged. The cap matches the largest cell the
//! default fp128 correctness matrix proves (one-hot, 28 variables); every
//! skipped row commits at least `2^30` coefficients, the production sizes that
//! matrix marks `#[ignore]`. The recursive catalogs are excluded for the same
//! reason: every recursive row is production-sized.

use crate::common::{dense_opening_lagrange, load_workspace_scheme, onehot_opening_lagrange};
use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::CommitmentConfig;
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext, OneHotPoly};
use akita_prover::SelectedProverOpeningData;
use akita_schedules::ResolvedScheduleRow;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_types::{
    BasisMode, CommitmentPayloadMode, FpExtEncoding, GroupBatchStatement, GroupCommitPhaseParams,
    InnerCommitSecurityRoute, OpeningClaims, OpeningMethod, PhysicalL2NormProofShape,
    PolynomialGroupClaims, RingRelationMode,
};
use jolt_field::{
    CanonicalBytes, CanonicalEncoding, ExtField, Fold, PseudoMersenne, Ring, Unreduced,
    WithCommitAccumulator,
};
use std::collections::BTreeMap;
use std::time::Instant;

/// Largest committed coefficient count, as a base-2 logarithm, proved here.
const MAX_LOG_COMMITTED_COEFFS: u32 = 28;
const LABEL: &[u8] = b"hardening/catalog-event-sequence";

/// Norm proof a fold level's inner commitment route selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum NormRoute {
    Linf,
    L2Direct,
    L2LimbGram,
}

impl From<InnerCommitSecurityRoute> for NormRoute {
    fn from(route: InnerCommitSecurityRoute) -> Self {
        match route {
            InnerCommitSecurityRoute::Linf(_) => Self::Linf,
            InnerCommitSecurityRoute::L2 {
                norm_proof_shape: PhysicalL2NormProofShape::Direct { .. },
                ..
            } => Self::L2Direct,
            InnerCommitSecurityRoute::L2 {
                norm_proof_shape: PhysicalL2NormProofShape::LimbGram { .. },
                ..
            } => Self::L2LimbGram,
        }
    }
}

/// Site-selecting facts of one nonterminal fold level; see the module docs.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LevelShape {
    payload_mode: CommitmentPayloadMode,
    ring_relation_mode: RingRelationMode,
    multi_chunk: bool,
    coefficient_packing: bool,
    norm_route: NormRoute,
    consumes_setup_prefix: bool,
}

/// Proof-stream shape of one catalog row; see the module docs.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ScheduleShape {
    groups: usize,
    batched_group: bool,
    levels: Vec<LevelShape>,
}

fn schedule_shape(row: &ResolvedScheduleRow) -> ScheduleShape {
    let profiles = row.profiles();
    let schedule = row.schedule();
    let groups = profiles.precommitteds.len() + 1;
    let batched_group = profiles
        .precommitteds
        .iter()
        .chain(std::iter::once(&profiles.final_group))
        .any(|profile| profile.group.num_polynomials() > 1);
    let levels = std::iter::once(&schedule.root)
        .chain(&schedule.recursive_folds)
        .map(|fold| LevelShape {
            payload_mode: fold.params.payload_mode,
            ring_relation_mode: fold.params.ring_relation_mode,
            multi_chunk: fold.params.witness_chunk.num_chunks > 1,
            coefficient_packing: matches!(
                fold.params.opening_method(),
                OpeningMethod::SubringCoefficientPacking { .. }
            ),
            norm_route: fold.params.inner().matrix.security_route().into(),
            consumes_setup_prefix: fold.incoming_setup_prefix().is_some(),
        })
        .collect();
    ScheduleShape {
        groups,
        batched_group,
        levels,
    }
}

/// Base-2 logarithm of the committed coefficient count, rounded up.
fn log_committed_coeffs(row: &ResolvedScheduleRow) -> u32 {
    let profiles = row.profiles();
    let total: u128 = profiles
        .precommitteds
        .iter()
        .chain(std::iter::once(&profiles.final_group))
        .map(|profile| (profile.group.num_polynomials() as u128) << profile.group.num_vars())
        .sum();
    u128::BITS - (total - 1).leading_zeros()
}

/// Cheapest row of every shape in `Cfg`'s catalog, in shape order.
fn representative_rows<Cfg: CommitmentConfig>(
    catalog: &akita_config::TrustedScheduleCatalog<Cfg>,
) -> Vec<&ResolvedScheduleRow> {
    let mut by_shape: BTreeMap<ScheduleShape, &ResolvedScheduleRow> = BTreeMap::new();
    for row in catalog.rows() {
        let slot = by_shape.entry(schedule_shape(row)).or_insert(row);
        if (log_committed_coeffs(row), row.selection().row_digest)
            < (log_committed_coeffs(slot), slot.selection().row_digest)
        {
            *slot = row;
        }
    }
    by_shape.into_values().collect()
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Source class of a catalog family's committed polynomials.
trait GroupSource<Cfg: CommitmentConfig> {
    type Poly: akita_cpu_backend::CpuSource<Cfg::Field, Cfg::ExtField> + Clone;

    /// Sample one polynomial and its independent-oracle opening at `point`.
    fn sample(point: &[Cfg::ExtField], seed: &mut u64) -> (Self::Poly, Cfg::ExtField);
}

struct DenseSource;

impl<Cfg> GroupSource<Cfg> for DenseSource
where
    Cfg: CommitmentConfig,
    DensePoly<Cfg::Field>: akita_cpu_backend::CpuSource<Cfg::Field, Cfg::ExtField>,
{
    type Poly = DensePoly<Cfg::Field>;

    fn sample(point: &[Cfg::ExtField], seed: &mut u64) -> (Self::Poly, Cfg::ExtField) {
        // 32-bit coefficients fit every dense source class, bounded included.
        let evals: Vec<Cfg::Field> = (0..1usize << point.len())
            .map(|_| Cfg::Field::from_u64(splitmix64(seed) >> 32))
            .collect();
        let lifted: Vec<Cfg::ExtField> = evals
            .iter()
            .copied()
            .map(Cfg::ExtField::lift_base)
            .collect();
        let poly = DensePoly::from_field_evals(point.len(), &evals).expect("dense source");
        (poly, dense_opening_lagrange(&lifted, point))
    }
}

struct OneHotSource;

impl<Cfg> GroupSource<Cfg> for OneHotSource
where
    Cfg: CommitmentConfig,
    OneHotPoly<Cfg::Field, u8>: akita_cpu_backend::CpuSource<Cfg::Field, Cfg::ExtField>,
{
    type Poly = OneHotPoly<Cfg::Field, u8>;

    fn sample(point: &[Cfg::ExtField], seed: &mut u64) -> (Self::Poly, Cfg::ExtField) {
        let chunk_size =
            akita_config::unit_onehot_source_chunk_size::<Cfg>().expect("unit one-hot source");
        let indices = (0..(1usize << point.len()) / chunk_size)
            .map(|_| {
                Some(u8::try_from(splitmix64(seed) % chunk_size as u64).expect("u8 hot index"))
            })
            .collect();
        let poly = OneHotPoly::new(chunk_size, indices).expect("one-hot source");
        let opening = onehot_opening_lagrange(&poly, point);
        (poly, opening)
    }
}

/// Prove and verify one row, returning the prover and verifier event streams.
fn row_event_streams<Cfg, S>(
    scheme: &akita_pcs::AkitaCommitmentScheme<Cfg>,
    row: &ResolvedScheduleRow,
) -> (
    Vec<akita_transcript::TranscriptEvent>,
    Vec<akita_transcript::TranscriptEvent>,
)
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding
        + CanonicalBytes
        + Unreduced
        + Ring
        + PseudoMersenne
        + WithCommitAccumulator
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + 'static,
    Cfg::ExtField: ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + FpExtEncoding<Cfg::Field>
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + jolt_field::MulBaseUnreduced<Cfg::Field>
        + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + jolt_field::AdditiveGroup,
    S: GroupSource<Cfg>,
{
    let profiles = row.profiles();
    let group_profiles: Vec<&GroupCommitPhaseParams> = profiles
        .precommitteds
        .iter()
        .chain(std::iter::once(&profiles.final_group))
        .collect();
    let max_num_vars = group_profiles
        .iter()
        .map(|profile| profile.group.num_vars())
        .max()
        .expect("row has a final group");
    let num_polynomials = profiles
        .opening_layout()
        .expect("layout")
        .num_total_polynomials();
    let setup = scheme
        .setup_prover(max_num_vars, num_polynomials)
        .expect("setup");
    let backend = CpuBackend::new(setup.expanded.clone()).expect("backend");
    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
    assert!(
        akita_config::TrustedScheduleCatalog::<Cfg>::verifier_admits(&setup.expanded, row)
            .expect("admission"),
        "setup sized from the row must admit it"
    );

    let mut seed = u64::from_le_bytes(
        row.selection().row_digest.as_bytes()[..8]
            .try_into()
            .expect("digest prefix"),
    );
    let point: Vec<Cfg::ExtField> = (0..max_num_vars)
        .map(|_| Cfg::ExtField::from_u64(splitmix64(&mut seed)))
        .collect();
    let mut commitments = Vec::with_capacity(group_profiles.len());
    let mut hints = Vec::with_capacity(group_profiles.len());
    let mut openings = Vec::with_capacity(group_profiles.len());
    for profile in &group_profiles {
        let num_vars = profile.group.num_vars();
        let (sources, group_openings): (Vec<S::Poly>, Vec<Cfg::ExtField>) =
            (0..profile.group.num_polynomials())
                .map(|_| S::sample(&point[..num_vars], &mut seed))
                .unzip();
        openings.push(group_openings);
        let output = backend
            .commit(
                scheme.schedules(),
                &backend.import_source(sources).expect("source"),
                GroupContext::explicit(profile),
            )
            .expect("commit");
        commitments.push(output.committed_group);
        hints.push(output.private_handle);
    }

    let prover_claims = OpeningClaims::from_groups(
        commitments
            .iter()
            .zip(&openings)
            .map(|(commitment, evaluations)| {
                let num_vars = commitment.profile.group.num_vars();
                PolynomialGroupClaims::new(
                    point[..num_vars].to_vec(),
                    evaluations.clone(),
                    commitment.clone(),
                )
                .expect("prover group")
            })
            .collect(),
    )
    .expect("prover claims");
    let prover_data = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        prover_claims,
        hints,
        scheme.schedules(),
    )
    .expect("prover data");
    let selection = prover_data.selection();
    assert_eq!(
        selection,
        row.selection(),
        "claims must select the probed row"
    );

    akita_transcript::clear_thread_events();
    let proof = scheme
        .batched_prove(&setup, prover_data, &backend, LABEL, BasisMode::Lagrange)
        .expect("prove");
    let prover_events = akita_transcript::thread_events();

    let verifier_claims = OpeningClaims::from_groups(
        commitments
            .iter()
            .zip(&openings)
            .map(|(commitment, evaluations)| {
                let num_vars = commitment.profile.group.num_vars();
                PolynomialGroupClaims::new(
                    point[..num_vars].to_vec(),
                    evaluations.clone(),
                    commitment,
                )
                .expect("verifier group")
            })
            .collect(),
    )
    .expect("verifier claims");
    akita_transcript::clear_thread_events();
    scheme
        .verifier(verifier_setup)
        .and_then(|verifier| {
            verifier.batched_verify(
                &proof,
                LABEL,
                GroupBatchStatement::new(selection, verifier_claims).expect("statement"),
                BasisMode::Lagrange,
            )
        })
        .expect("honest proof verifies");
    let verifier_events = akita_transcript::thread_events();
    (prover_events, verifier_events)
}

/// Index and both sides of the first differing event, or the length mismatch.
fn first_divergence(
    prover: &[akita_transcript::TranscriptEvent],
    verifier: &[akita_transcript::TranscriptEvent],
) -> String {
    let site = |event: Option<&akita_transcript::TranscriptEvent>| {
        event.map(|akita_transcript::TranscriptEvent::Context(record)| {
            (
                akita_transcript::ProtocolSiteId::from_bytes(record.site_id),
                record.kind,
            )
        })
    };
    let index = prover
        .iter()
        .zip(verifier)
        .position(|(p, v)| p != v)
        .unwrap_or(prover.len().min(verifier.len()));
    format!(
        "first divergence at event {index} of prover={} verifier={}: prover {:?}, verifier {:?}",
        prover.len(),
        verifier.len(),
        site(prover.get(index)),
        site(verifier.get(index)),
    )
}

fn check_catalog<Cfg, S>(covered: &mut Vec<String>)
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding
        + CanonicalBytes
        + Unreduced
        + Ring
        + PseudoMersenne
        + WithCommitAccumulator
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + 'static,
    Cfg::ExtField: ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + FpExtEncoding<Cfg::Field>
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + jolt_field::MulBaseUnreduced<Cfg::Field>
        + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + jolt_field::AdditiveGroup,
    S: GroupSource<Cfg>,
{
    let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");
    let family = Cfg::schedule_family_name();
    for row in representative_rows(scheme.schedules()) {
        let profiles = row.profiles();
        let label = format!(
            "{family} final={:?} pre={:?} levels={}",
            profiles.final_group.group,
            profiles
                .precommitteds
                .iter()
                .map(|profile| profile.group)
                .collect::<Vec<_>>(),
            row.schedule().num_fold_levels(),
        );
        if log_committed_coeffs(row) > MAX_LOG_COMMITTED_COEFFS {
            eprintln!("skip  {label}: production-sized");
            continue;
        }
        let started = Instant::now();
        let (prover, verifier) = row_event_streams::<Cfg, S>(&scheme, row);
        assert!(!prover.is_empty(), "{label}: prover recorded no events");
        assert!(
            prover == verifier,
            "{label}: {}",
            first_divergence(&prover, &verifier)
        );
        eprintln!(
            "ok    {label}: {} events in {:.1?}",
            prover.len(),
            started.elapsed()
        );
        covered.push(label);
    }
}

/// The fp64 dense 26- and 28-variable single-polynomial rows share every other
/// key field, but the 28-variable row switches to physical L2 one level
/// earlier. Both fit the cap, so both must be proved.
#[test]
fn norm_route_separates_fp64_dense_single_polynomial_rows() {
    let scheme = load_workspace_scheme::<fp64::Dense>().expect("workspace schedule catalog");
    let single = |num_vars: usize| {
        scheme
            .schedules()
            .rows()
            .find(|row| {
                let profiles = row.profiles();
                profiles.precommitteds.is_empty()
                    && profiles.final_group.group.num_polynomials() == 1
                    && profiles.final_group.group.num_vars() == num_vars
            })
            .unwrap_or_else(|| panic!("fp64 dense {num_vars}-variable row"))
    };
    let (small, large) = (single(26), single(28));
    assert!(log_committed_coeffs(large) <= MAX_LOG_COMMITTED_COEFFS);
    let (small_shape, large_shape) = (schedule_shape(small), schedule_shape(large));
    assert_ne!(small_shape, large_shape);
    let representatives = representative_rows(scheme.schedules());
    for shape in [small_shape, large_shape] {
        assert!(
            representatives
                .iter()
                .any(|row| schedule_shape(row) == shape),
            "{shape:?} has no representative"
        );
    }
}

#[test]
fn prover_and_verifier_emit_identical_event_sequences_for_every_catalog_shape() {
    crate::common::init_rayon_pool();
    crate::common::run_on_large_stack(|| {
        let mut covered = Vec::new();
        check_catalog::<fp128::Dense, DenseSource>(&mut covered);
        check_catalog::<fp128::DenseBounded, DenseSource>(&mut covered);
        check_catalog::<fp128::DenseMultiChunk, DenseSource>(&mut covered);
        check_catalog::<fp128::OneHot, OneHotSource>(&mut covered);
        check_catalog::<fp128::OneHotMultiChunk, OneHotSource>(&mut covered);
        check_catalog::<fp128::OneHotMultiChunkW2R2, OneHotSource>(&mut covered);
        check_catalog::<fp128::OneHotMultiChunkW4R2, OneHotSource>(&mut covered);
        check_catalog::<fp32::Dense, DenseSource>(&mut covered);
        check_catalog::<fp32::OneHot, OneHotSource>(&mut covered);
        check_catalog::<fp64::Dense, DenseSource>(&mut covered);
        check_catalog::<fp64::OneHot, OneHotSource>(&mut covered);
        assert!(!covered.is_empty());
        eprintln!("{} catalog shapes covered", covered.len());
    });
}
