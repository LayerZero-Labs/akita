//! Dynamic root-ring wrapper around the typed Hachi commitment scheme.
//!
//! The low-level Hachi kernels remain const-generic over the root ring degree
//! `D`. This module lifts the public API one level up so callers can provide
//! ring-agnostic root polynomials and let each commitment group choose the
//! root ring from its actual public runtime context.

use crate::algebra::fields::wide::HasWide;
use crate::algebra::fields::HasUnreducedOps;
use crate::error::HachiError;
use crate::primitives::serialization::Valid;
use crate::protocol::commitment::schedule::{
    estimated_recursive_suffix_bytes, hachi_root_runtime_plan_from_root_layout,
};
use crate::protocol::commitment::{
    hachi_batched_root_layout, presets::fp128, AppendToTranscript, CommitmentConfig,
    CommitmentPreset, CommitmentScheme, DynamicCommitmentScheme, Fp128AdaptiveBoundedPolicy,
    HachiProverSetup, HachiRootBatchSummary, HachiScheduleLookupKey, HachiVerifierSetup,
    RingCommitment,
};
use crate::protocol::commitment_scheme::HachiCommitmentScheme;
use crate::protocol::opening_point::BasisMode;
use crate::protocol::proof::{HachiBatchedCommitmentHint, HachiBatchedProof, HachiProof};
use crate::protocol::root_poly::{MultilinearPolynomial, TypedRootPolynomial};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling};
use std::marker::PhantomData;
use std::thread;

/// Family-level selector for dynamic root-ring Hachi schemes.
///
/// Each associated config fixes one concrete root ring degree. The family
/// chooses which root degree to use for each commitment group from its actual
/// public runtime context. After that choice, the protocol runs through the
/// existing typed kernel for that fixed-D config.
pub trait DynamicRootConfigFamily<F>: Clone + Send + Sync + 'static
where
    F: FieldCore + CanonicalField,
{
    /// Typed config to use when the chosen root ring is `D=32`.
    type Cfg32: CommitmentConfig<Field = F>;
    /// Typed config to use when the chosen root ring is `D=64`.
    type Cfg64: CommitmentConfig<Field = F>;
    /// Typed config to use when the chosen root ring is `D=128`.
    type Cfg128: CommitmentConfig<Field = F>;

    /// Choose the root ring degree for one root commitment/proof context.
    ///
    /// # Errors
    ///
    /// Returns an error if the family cannot choose a supported root ring from
    /// the provided public runtime parameters.
    fn select_root_ring_dim(key: HachiScheduleLookupKey) -> Result<usize, HachiError> {
        select_smallest_estimated_proof_root_ring_dim::<Self::Cfg32, Self::Cfg64, Self::Cfg128>(key)
    }
}

fn estimated_total_proof_bytes<Cfg, const D: usize>(
    key: HachiScheduleLookupKey,
) -> Result<usize, HachiError>
where
    Cfg: CommitmentConfig,
{
    let root_layout = hachi_batched_root_layout::<Cfg, D>(key.num_vars, key.layout_num_claims)?;
    let root_plan = hachi_root_runtime_plan_from_root_layout::<Cfg, D>(key, root_layout)?;
    let suffix_bytes = estimated_recursive_suffix_bytes::<Cfg>(
        key.max_num_vars,
        root_plan.next_inputs.level,
        root_plan.next_w_len(),
    )?;
    Ok(root_plan.level_proof_bytes::<Cfg>() + suffix_bytes)
}

fn select_smallest_estimated_proof_root_ring_dim<Cfg32, Cfg64, Cfg128>(
    key: HachiScheduleLookupKey,
) -> Result<usize, HachiError>
where
    Cfg32: CommitmentConfig,
    Cfg64: CommitmentConfig,
    Cfg128: CommitmentConfig,
{
    let mut best: Option<(usize, usize)> = None;
    for (root_d, proof_bytes) in [
        (32usize, estimated_total_proof_bytes::<Cfg32, 32>(key)),
        (64usize, estimated_total_proof_bytes::<Cfg64, 64>(key)),
        (128usize, estimated_total_proof_bytes::<Cfg128, 128>(key)),
    ] {
        let Ok(proof_bytes) = proof_bytes else {
            continue;
        };
        if best.as_ref().is_none_or(|(best_d, best_bytes)| {
            proof_bytes < *best_bytes || (proof_bytes == *best_bytes && root_d < *best_d)
        }) {
            best = Some((root_d, proof_bytes));
        }
    }

    best.map(|(root_d, _)| root_d).ok_or_else(|| {
        HachiError::InvalidInput(format!(
            "dynamic root selection found no supported root D for num_vars={}, layout_num_claims={}, batch_claims={}",
            key.num_vars, key.layout_num_claims, key.batch.num_claims
        ))
    })
}

fn has_exact_fit_singleton_root_context(key: HachiScheduleLookupKey) -> bool {
    key.max_num_vars == key.num_vars
        && key.layout_num_claims == 1
        && key.batch == HachiRootBatchSummary::singleton()
}

fn fp128_exact_fit_singleton_prefers_d32(key: HachiScheduleLookupKey) -> bool {
    has_exact_fit_singleton_root_context(key) && (6..=63).contains(&key.num_vars)
}

/// D-erased prover setup for the public dynamic Hachi API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicHachiProverSetup<F: FieldCore> {
    /// Maximum root polynomial variable count supported by this setup family.
    pub max_num_vars: usize,
    /// Maximum number of batched root polynomials supported by this setup family.
    pub max_num_batched_polys: usize,
    /// Root ring `D=32`.
    pub d32: Option<Box<HachiProverSetup<F, 32>>>,
    /// Root ring `D=64`.
    pub d64: Option<Box<HachiProverSetup<F, 64>>>,
    /// Root ring `D=128`.
    pub d128: Option<Box<HachiProverSetup<F, 128>>>,
}

impl<F: FieldCore> DynamicHachiProverSetup<F> {
    /// Maximum root polynomial variable count supported by this setup.
    pub fn max_num_vars(&self) -> usize {
        self.max_num_vars
    }

    /// Maximum batch capacity carried by this setup.
    pub fn max_num_batched_polys(&self) -> usize {
        self.max_num_batched_polys
    }
}

/// D-erased verifier setup for the public dynamic Hachi API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicHachiVerifierSetup<F: FieldCore> {
    /// Root ring `D=32`.
    pub d32: Option<HachiVerifierSetup<F>>,
    /// Root ring `D=64`.
    pub d64: Option<HachiVerifierSetup<F>>,
    /// Root ring `D=128`.
    pub d128: Option<HachiVerifierSetup<F>>,
}

/// D-erased root commitment object for the public dynamic Hachi API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicRingCommitment<F: FieldCore> {
    /// Commitment over root ring `D=32`.
    D32(RingCommitment<F, 32>),
    /// Commitment over root ring `D=64`.
    D64(RingCommitment<F, 64>),
    /// Commitment over root ring `D=128`.
    D128(RingCommitment<F, 128>),
}

impl<F: FieldCore> DynamicRingCommitment<F> {
    /// Root ring degree used by this commitment.
    pub fn root_ring_dim(&self) -> usize {
        match self {
            Self::D32(_) => 32,
            Self::D64(_) => 64,
            Self::D128(_) => 128,
        }
    }
}

impl<F> AppendToTranscript<F> for DynamicRingCommitment<F>
where
    F: FieldCore + CanonicalField,
{
    fn append_to_transcript<T: Transcript<F>>(&self, label: &[u8], transcript: &mut T) {
        match self {
            Self::D32(commitment) => commitment.append_to_transcript(label, transcript),
            Self::D64(commitment) => commitment.append_to_transcript(label, transcript),
            Self::D128(commitment) => commitment.append_to_transcript(label, transcript),
        }
    }
}

/// D-erased root commitment hint for one commitment group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicCommitHint<F: FieldCore> {
    /// Hint over root ring `D=32`.
    D32(HachiBatchedCommitmentHint<F, 32>),
    /// Hint over root ring `D=64`.
    D64(HachiBatchedCommitmentHint<F, 64>),
    /// Hint over root ring `D=128`.
    D128(HachiBatchedCommitmentHint<F, 128>),
}

impl<F: FieldCore> DynamicCommitHint<F> {
    /// Root ring degree used by this hint.
    pub fn root_ring_dim(&self) -> usize {
        match self {
            Self::D32(_) => 32,
            Self::D64(_) => 64,
            Self::D128(_) => 128,
        }
    }
}

/// Dynamic public Hachi scheme that chooses the root ring at setup time.
#[derive(Clone, Copy, Debug, Default)]
pub struct DynamicHachiCommitmentScheme<Family> {
    _family: PhantomData<Family>,
}

fn ensure_nonempty_uniform_num_vars<F: FieldCore>(
    polys: &[MultilinearPolynomial<F>],
    label: &str,
) -> Result<usize, HachiError> {
    let Some(first) = polys.first() else {
        return Err(HachiError::InvalidInput(format!(
            "{label} requires at least one polynomial"
        )));
    };
    let num_vars = first.num_vars();
    if polys.iter().any(|poly| poly.num_vars() != num_vars) {
        return Err(HachiError::InvalidInput(format!(
            "{label} requires all polynomials to have the same num_vars"
        )));
    }
    Ok(num_vars)
}

fn commit_selection_key<F: FieldCore>(
    polys: &[MultilinearPolynomial<F>],
    setup: &DynamicHachiProverSetup<F>,
    label: &str,
) -> Result<HachiScheduleLookupKey, HachiError> {
    let num_vars = ensure_nonempty_uniform_num_vars(polys, label)?;
    if num_vars > setup.max_num_vars() {
        return Err(HachiError::InvalidInput(format!(
            "{label} polynomial uses {num_vars} variables but setup supports at most {}",
            setup.max_num_vars()
        )));
    }
    let num_polys = polys.len();
    if num_polys > setup.max_num_batched_polys() {
        return Err(HachiError::InvalidInput(format!(
            "{label} received {num_polys} polynomials but setup supports at most {}",
            setup.max_num_batched_polys()
        )));
    }
    let batch = HachiRootBatchSummary::new(num_polys, 1, 1)?;
    Ok(HachiScheduleLookupKey::with_batch(
        setup.max_num_vars(),
        num_vars,
        num_polys,
        batch,
    ))
}

fn require_typed_prover_setup<'a, F: FieldCore, const D: usize>(
    setup: &'a Option<Box<HachiProverSetup<F, D>>>,
    label: &str,
) -> Result<&'a HachiProverSetup<F, D>, HachiError> {
    setup.as_deref().ok_or_else(|| {
        HachiError::InvalidInput(format!(
            "{label} requires root D={D}, but this dynamic setup does not support it"
        ))
    })
}

fn require_typed_verifier_setup<'a, F: FieldCore>(
    setup: &'a Option<HachiVerifierSetup<F>>,
    root_d: usize,
    label: &str,
) -> Result<&'a HachiVerifierSetup<F>, HachiError> {
    setup.as_ref().ok_or_else(|| {
        HachiError::InvalidInput(format!(
            "{label} requires root D={root_d}, but this dynamic verifier setup does not support it"
        ))
    })
}

fn materialize_typed_root_group<F, const D: usize, Cfg>(
    polys: &[MultilinearPolynomial<F>],
    max_num_batched_polys: usize,
    label: &str,
) -> Result<Vec<TypedRootPolynomial<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig<Field = F>,
{
    let num_vars = ensure_nonempty_uniform_num_vars(polys, label)?;
    let root_layout = hachi_batched_root_layout::<Cfg, D>(num_vars, max_num_batched_polys)?;
    polys
        .iter()
        .map(|poly| TypedRootPolynomial::from_public(poly, root_layout))
        .collect()
}

fn materialize_typed_root_groups_by_point<F, const D: usize, Cfg>(
    poly_groups_by_point: &[&[&[MultilinearPolynomial<F>]]],
    max_num_batched_polys: usize,
    label: &str,
) -> Result<Vec<Vec<Vec<TypedRootPolynomial<F, D>>>>, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig<Field = F>,
{
    poly_groups_by_point
        .iter()
        .enumerate()
        .map(|(point_idx, groups)| {
            groups
                .iter()
                .enumerate()
                .map(|(group_idx, group)| {
                    materialize_typed_root_group::<F, D, Cfg>(
                        group,
                        max_num_batched_polys,
                        &format!("{label} point {point_idx} group {group_idx}"),
                    )
                })
                .collect()
        })
        .collect()
}

fn uniform_root_ring_dim<'a, F: FieldCore + 'a>(
    hints_by_point: impl IntoIterator<Item = &'a [DynamicCommitHint<F>]>,
    commitments_by_point: impl IntoIterator<Item = &'a [DynamicRingCommitment<F>]>,
    label: &str,
) -> Result<usize, HachiError> {
    let mut expected: Option<usize> = None;

    for (point_idx, hints) in hints_by_point.into_iter().enumerate() {
        for (group_idx, hint) in hints.iter().enumerate() {
            let root_d = hint.root_ring_dim();
            if let Some(expected_d) = expected {
                if root_d != expected_d {
                    return Err(HachiError::InvalidInput(format!(
                        "{label} requires one root D across the fused batch; point {point_idx} group {group_idx} used D={root_d} after earlier D={expected_d}"
                    )));
                }
            } else {
                expected = Some(root_d);
            }
        }
    }

    for (point_idx, commitments) in commitments_by_point.into_iter().enumerate() {
        for (group_idx, commitment) in commitments.iter().enumerate() {
            let root_d = commitment.root_ring_dim();
            if let Some(expected_d) = expected {
                if root_d != expected_d {
                    return Err(HachiError::InvalidInput(format!(
                        "{label} requires one root D across the fused batch; point {point_idx} group {group_idx} commitment used D={root_d} after earlier D={expected_d}"
                    )));
                }
            } else {
                expected = Some(root_d);
            }
        }
    }

    expected.ok_or_else(|| {
        HachiError::InvalidInput(format!(
            "{label} requires at least one hint or commitment group"
        ))
    })
}

macro_rules! clone_typed_commitment {
    ($commitment:expr, $variant:ident, $expected_d:literal, $label:expr) => {
        match $commitment {
            DynamicRingCommitment::$variant(commitment) => Ok(commitment.clone()),
            other => Err(HachiError::InvalidInput(format!(
                "{} expected root D={} commitment but received D={}",
                $label,
                $expected_d,
                other.root_ring_dim()
            ))),
        }
    };
}

macro_rules! clone_typed_hint {
    ($hint:expr, $variant:ident, $expected_d:literal, $label:expr) => {
        match $hint {
            DynamicCommitHint::$variant(hint) => Ok(hint.clone()),
            other => Err(HachiError::InvalidInput(format!(
                "{} expected root D={} hint but received D={}",
                $label,
                $expected_d,
                other.root_ring_dim()
            ))),
        }
    };
}

impl<F, Family> DynamicCommitmentScheme<F> for DynamicHachiCommitmentScheme<Family>
where
    F: FieldCore + CanonicalField + FieldSampling + HasWide + HasUnreducedOps + Valid,
    Family: DynamicRootConfigFamily<F>,
{
    type ProverSetup = DynamicHachiProverSetup<F>;
    type VerifierSetup = DynamicHachiVerifierSetup<F>;
    type Commitment = DynamicRingCommitment<F>;
    type Proof = HachiProof<F>;
    type BatchedProof = HachiBatchedProof<F>;
    type CommitHint = DynamicCommitHint<F>;
    type BatchedCommitHint = Vec<DynamicCommitHint<F>>;

    fn setup_prover(max_num_vars: usize, max_num_batched_polys: usize) -> Self::ProverSetup {
        let (d32, d64, d128) = thread::scope(|scope| {
            let d32 = scope.spawn(|| {
                std::panic::catch_unwind(|| {
                    <HachiCommitmentScheme<32, Family::Cfg32> as CommitmentScheme<F, 32>>::setup_prover(
                        max_num_vars,
                        max_num_batched_polys,
                    )
                })
                .ok()
                .map(Box::new)
            });
            let d64 = scope.spawn(|| {
                std::panic::catch_unwind(|| {
                    <HachiCommitmentScheme<64, Family::Cfg64> as CommitmentScheme<F, 64>>::setup_prover(
                        max_num_vars,
                        max_num_batched_polys,
                    )
                })
                .ok()
                .map(Box::new)
            });
            let d128 = scope.spawn(|| {
                std::panic::catch_unwind(|| {
                    <HachiCommitmentScheme<128, Family::Cfg128> as CommitmentScheme<F, 128>>::setup_prover(
                        max_num_vars,
                        max_num_batched_polys,
                    )
                })
                .ok()
                .map(Box::new)
            });
            (
                d32.join().unwrap(),
                d64.join().unwrap(),
                d128.join().unwrap(),
            )
        });

        assert!(
            d32.is_some() || d64.is_some() || d128.is_some(),
            "dynamic setup found no supported root D for max_num_vars={max_num_vars}, max_num_batched_polys={max_num_batched_polys}"
        );

        DynamicHachiProverSetup {
            max_num_vars,
            max_num_batched_polys,
            d32,
            d64,
            d128,
        }
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        DynamicHachiVerifierSetup {
            d32: setup.d32.as_ref().map(|typed_setup| {
                <HachiCommitmentScheme<32, Family::Cfg32> as CommitmentScheme<F, 32>>::setup_verifier(
                    typed_setup,
                )
            }),
            d64: setup.d64.as_ref().map(|typed_setup| {
                <HachiCommitmentScheme<64, Family::Cfg64> as CommitmentScheme<F, 64>>::setup_verifier(
                    typed_setup,
                )
            }),
            d128: setup.d128.as_ref().map(|typed_setup| {
                <HachiCommitmentScheme<128, Family::Cfg128> as CommitmentScheme<F, 128>>::setup_verifier(
                    typed_setup,
                )
            }),
        }
    }

    fn commit(
        polys: &[MultilinearPolynomial<F>],
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::CommitHint), HachiError> {
        let key = commit_selection_key(polys, setup, "dynamic commit")?;
        match Family::select_root_ring_dim(key)? {
            32 => {
                let typed_setup = require_typed_prover_setup(&setup.d32, "dynamic commit")?;
                let typed_polys = materialize_typed_root_group::<F, 32, Family::Cfg32>(
                    polys,
                    typed_setup.max_num_batched_polys(),
                    "dynamic commit",
                )?;
                let (commitment, hint) =
                    <HachiCommitmentScheme<32, Family::Cfg32> as CommitmentScheme<F, 32>>::commit(
                        &typed_polys,
                        typed_setup,
                    )?;
                Ok((
                    DynamicRingCommitment::D32(commitment),
                    DynamicCommitHint::D32(hint),
                ))
            }
            64 => {
                let typed_setup = require_typed_prover_setup(&setup.d64, "dynamic commit")?;
                let typed_polys = materialize_typed_root_group::<F, 64, Family::Cfg64>(
                    polys,
                    typed_setup.max_num_batched_polys(),
                    "dynamic commit",
                )?;
                let (commitment, hint) =
                    <HachiCommitmentScheme<64, Family::Cfg64> as CommitmentScheme<F, 64>>::commit(
                        &typed_polys,
                        typed_setup,
                    )?;
                Ok((
                    DynamicRingCommitment::D64(commitment),
                    DynamicCommitHint::D64(hint),
                ))
            }
            128 => {
                let typed_setup = require_typed_prover_setup(&setup.d128, "dynamic commit")?;
                let typed_polys = materialize_typed_root_group::<F, 128, Family::Cfg128>(
                    polys,
                    typed_setup.max_num_batched_polys(),
                    "dynamic commit",
                )?;
                let (commitment, hint) = <HachiCommitmentScheme<128, Family::Cfg128> as CommitmentScheme<
                    F,
                    128,
                >>::commit(&typed_polys, typed_setup)?;
                Ok((
                    DynamicRingCommitment::D128(commitment),
                    DynamicCommitHint::D128(hint),
                ))
            }
            root_d => Err(HachiError::InvalidInput(format!(
                "dynamic commit selected unsupported root D={root_d}"
            ))),
        }
    }

    fn prove<T: Transcript<F>>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<F>,
        opening_point: &[F],
        hint: Self::CommitHint,
        transcript: &mut T,
        commitment: &Self::Commitment,
        basis: BasisMode,
    ) -> Result<Self::Proof, HachiError> {
        match hint.root_ring_dim() {
            32 => {
                let typed_setup = require_typed_prover_setup(&setup.d32, "dynamic prove")?;
                let typed_poly = materialize_typed_root_group::<F, 32, Family::Cfg32>(
                    std::slice::from_ref(poly),
                    typed_setup.max_num_batched_polys(),
                    "dynamic prove",
                )?;
                let typed_hint = clone_typed_hint!(hint, D32, 32, "dynamic prove")?;
                let typed_commitment =
                    clone_typed_commitment!(commitment, D32, 32, "dynamic prove")?;
                <HachiCommitmentScheme<32, Family::Cfg32> as CommitmentScheme<F, 32>>::prove(
                    typed_setup,
                    &typed_poly[0],
                    opening_point,
                    typed_hint,
                    transcript,
                    &typed_commitment,
                    basis,
                )
            }
            64 => {
                let typed_setup = require_typed_prover_setup(&setup.d64, "dynamic prove")?;
                let typed_poly = materialize_typed_root_group::<F, 64, Family::Cfg64>(
                    std::slice::from_ref(poly),
                    typed_setup.max_num_batched_polys(),
                    "dynamic prove",
                )?;
                let typed_hint = clone_typed_hint!(hint, D64, 64, "dynamic prove")?;
                let typed_commitment =
                    clone_typed_commitment!(commitment, D64, 64, "dynamic prove")?;
                <HachiCommitmentScheme<64, Family::Cfg64> as CommitmentScheme<F, 64>>::prove(
                    typed_setup,
                    &typed_poly[0],
                    opening_point,
                    typed_hint,
                    transcript,
                    &typed_commitment,
                    basis,
                )
            }
            128 => {
                let typed_setup = require_typed_prover_setup(&setup.d128, "dynamic prove")?;
                let typed_poly = materialize_typed_root_group::<F, 128, Family::Cfg128>(
                    std::slice::from_ref(poly),
                    typed_setup.max_num_batched_polys(),
                    "dynamic prove",
                )?;
                let typed_hint = clone_typed_hint!(hint, D128, 128, "dynamic prove")?;
                let typed_commitment =
                    clone_typed_commitment!(commitment, D128, 128, "dynamic prove")?;
                <HachiCommitmentScheme<128, Family::Cfg128> as CommitmentScheme<F, 128>>::prove(
                    typed_setup,
                    &typed_poly[0],
                    opening_point,
                    typed_hint,
                    transcript,
                    &typed_commitment,
                    basis,
                )
            }
            root_d => Err(HachiError::InvalidInput(format!(
                "dynamic prove received unsupported root D={root_d}"
            ))),
        }
    }

    fn batched_prove<T: Transcript<F>>(
        setup: &Self::ProverSetup,
        poly_groups_by_point: &[&[&[MultilinearPolynomial<F>]]],
        opening_points: &[&[F]],
        hints_by_point: Vec<Self::BatchedCommitHint>,
        transcript: &mut T,
        commitments_by_point: &[&[Self::Commitment]],
        basis: BasisMode,
    ) -> Result<Self::BatchedProof, HachiError> {
        let root_d = uniform_root_ring_dim(
            hints_by_point.iter().map(Vec::as_slice),
            commitments_by_point.iter().copied(),
            "dynamic batched_prove",
        )?;
        match root_d {
            32 => {
                let typed_setup = require_typed_prover_setup(&setup.d32, "dynamic batched_prove")?;
                let typed_polys = materialize_typed_root_groups_by_point::<F, 32, Family::Cfg32>(
                    poly_groups_by_point,
                    typed_setup.max_num_batched_polys(),
                    "dynamic batched_prove",
                )?;
                let typed_group_refs: Vec<Vec<&[TypedRootPolynomial<F, 32>]>> = typed_polys
                    .iter()
                    .map(|groups| groups.iter().map(Vec::as_slice).collect())
                    .collect();
                let typed_point_refs: Vec<&[&[TypedRootPolynomial<F, 32>]]> =
                    typed_group_refs.iter().map(Vec::as_slice).collect();
                let typed_hints: Vec<Vec<HachiBatchedCommitmentHint<F, 32>>> = hints_by_point
                    .into_iter()
                    .enumerate()
                    .map(|(point_idx, hints)| {
                        hints
                            .into_iter()
                            .enumerate()
                            .map(|(group_idx, hint)| {
                                clone_typed_hint!(
                                    hint,
                                    D32,
                                    32,
                                    format!(
                                        "dynamic batched_prove point {point_idx} group {group_idx}"
                                    )
                                )
                            })
                            .collect()
                    })
                    .collect::<Result<_, _>>()?;
                let typed_commitments: Vec<Vec<RingCommitment<F, 32>>> = commitments_by_point
                    .iter()
                    .enumerate()
                    .map(|(point_idx, commitments)| {
                        commitments
                            .iter()
                            .enumerate()
                            .map(|(group_idx, commitment)| {
                                clone_typed_commitment!(
                                    commitment,
                                    D32,
                                    32,
                                    format!(
                                        "dynamic batched_prove point {point_idx} group {group_idx}"
                                    )
                                )
                            })
                            .collect()
                    })
                    .collect::<Result<_, _>>()?;
                let typed_commitment_refs: Vec<&[RingCommitment<F, 32>]> =
                    typed_commitments.iter().map(Vec::as_slice).collect();
                <HachiCommitmentScheme<32, Family::Cfg32> as CommitmentScheme<F, 32>>::batched_prove(
                    typed_setup,
                    &typed_point_refs,
                    opening_points,
                    typed_hints,
                    transcript,
                    &typed_commitment_refs,
                    basis,
                )
            }
            64 => {
                let typed_setup = require_typed_prover_setup(&setup.d64, "dynamic batched_prove")?;
                let typed_polys = materialize_typed_root_groups_by_point::<F, 64, Family::Cfg64>(
                    poly_groups_by_point,
                    typed_setup.max_num_batched_polys(),
                    "dynamic batched_prove",
                )?;
                let typed_group_refs: Vec<Vec<&[TypedRootPolynomial<F, 64>]>> = typed_polys
                    .iter()
                    .map(|groups| groups.iter().map(Vec::as_slice).collect())
                    .collect();
                let typed_point_refs: Vec<&[&[TypedRootPolynomial<F, 64>]]> =
                    typed_group_refs.iter().map(Vec::as_slice).collect();
                let typed_hints: Vec<Vec<HachiBatchedCommitmentHint<F, 64>>> = hints_by_point
                    .into_iter()
                    .enumerate()
                    .map(|(point_idx, hints)| {
                        hints
                            .into_iter()
                            .enumerate()
                            .map(|(group_idx, hint)| {
                                clone_typed_hint!(
                                    hint,
                                    D64,
                                    64,
                                    format!(
                                        "dynamic batched_prove point {point_idx} group {group_idx}"
                                    )
                                )
                            })
                            .collect()
                    })
                    .collect::<Result<_, _>>()?;
                let typed_commitments: Vec<Vec<RingCommitment<F, 64>>> = commitments_by_point
                    .iter()
                    .enumerate()
                    .map(|(point_idx, commitments)| {
                        commitments
                            .iter()
                            .enumerate()
                            .map(|(group_idx, commitment)| {
                                clone_typed_commitment!(
                                    commitment,
                                    D64,
                                    64,
                                    format!(
                                        "dynamic batched_prove point {point_idx} group {group_idx}"
                                    )
                                )
                            })
                            .collect()
                    })
                    .collect::<Result<_, _>>()?;
                let typed_commitment_refs: Vec<&[RingCommitment<F, 64>]> =
                    typed_commitments.iter().map(Vec::as_slice).collect();
                <HachiCommitmentScheme<64, Family::Cfg64> as CommitmentScheme<F, 64>>::batched_prove(
                    typed_setup,
                    &typed_point_refs,
                    opening_points,
                    typed_hints,
                    transcript,
                    &typed_commitment_refs,
                    basis,
                )
            }
            128 => {
                let typed_setup = require_typed_prover_setup(&setup.d128, "dynamic batched_prove")?;
                let typed_polys = materialize_typed_root_groups_by_point::<F, 128, Family::Cfg128>(
                    poly_groups_by_point,
                    typed_setup.max_num_batched_polys(),
                    "dynamic batched_prove",
                )?;
                let typed_group_refs: Vec<Vec<&[TypedRootPolynomial<F, 128>]>> = typed_polys
                    .iter()
                    .map(|groups| groups.iter().map(Vec::as_slice).collect())
                    .collect();
                let typed_point_refs: Vec<&[&[TypedRootPolynomial<F, 128>]]> =
                    typed_group_refs.iter().map(Vec::as_slice).collect();
                let typed_hints: Vec<Vec<HachiBatchedCommitmentHint<F, 128>>> = hints_by_point
                    .into_iter()
                    .enumerate()
                    .map(|(point_idx, hints)| {
                        hints
                            .into_iter()
                            .enumerate()
                            .map(|(group_idx, hint)| {
                                clone_typed_hint!(
                                    hint,
                                    D128,
                                    128,
                                    format!(
                                        "dynamic batched_prove point {point_idx} group {group_idx}"
                                    )
                                )
                            })
                            .collect()
                    })
                    .collect::<Result<_, _>>()?;
                let typed_commitments: Vec<Vec<RingCommitment<F, 128>>> = commitments_by_point
                    .iter()
                    .enumerate()
                    .map(|(point_idx, commitments)| {
                        commitments
                            .iter()
                            .enumerate()
                            .map(|(group_idx, commitment)| {
                                clone_typed_commitment!(
                                    commitment,
                                    D128,
                                    128,
                                    format!(
                                        "dynamic batched_prove point {point_idx} group {group_idx}"
                                    )
                                )
                            })
                            .collect()
                    })
                    .collect::<Result<_, _>>()?;
                let typed_commitment_refs: Vec<&[RingCommitment<F, 128>]> =
                    typed_commitments.iter().map(Vec::as_slice).collect();
                <HachiCommitmentScheme<128, Family::Cfg128> as CommitmentScheme<F, 128>>::batched_prove(
                    typed_setup,
                    &typed_point_refs,
                    opening_points,
                    typed_hints,
                    transcript,
                    &typed_commitment_refs,
                    basis,
                )
            }
            _ => unreachable!("uniform_root_ring_dim only returns supported root Ds"),
        }
    }

    fn verify<T: Transcript<F>>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        opening_point: &[F],
        opening: &F,
        commitment: &Self::Commitment,
        basis: BasisMode,
    ) -> Result<(), HachiError> {
        match commitment.root_ring_dim() {
            32 => {
                let typed_setup = require_typed_verifier_setup(&setup.d32, 32, "dynamic verify")?;
                let typed_commitment =
                    clone_typed_commitment!(commitment, D32, 32, "dynamic verify")?;
                <HachiCommitmentScheme<32, Family::Cfg32> as CommitmentScheme<F, 32>>::verify(
                    proof,
                    typed_setup,
                    transcript,
                    opening_point,
                    opening,
                    &typed_commitment,
                    basis,
                )
            }
            64 => {
                let typed_setup = require_typed_verifier_setup(&setup.d64, 64, "dynamic verify")?;
                let typed_commitment =
                    clone_typed_commitment!(commitment, D64, 64, "dynamic verify")?;
                <HachiCommitmentScheme<64, Family::Cfg64> as CommitmentScheme<F, 64>>::verify(
                    proof,
                    typed_setup,
                    transcript,
                    opening_point,
                    opening,
                    &typed_commitment,
                    basis,
                )
            }
            128 => {
                let typed_setup = require_typed_verifier_setup(&setup.d128, 128, "dynamic verify")?;
                let typed_commitment =
                    clone_typed_commitment!(commitment, D128, 128, "dynamic verify")?;
                <HachiCommitmentScheme<128, Family::Cfg128> as CommitmentScheme<F, 128>>::verify(
                    proof,
                    typed_setup,
                    transcript,
                    opening_point,
                    opening,
                    &typed_commitment,
                    basis,
                )
            }
            root_d => Err(HachiError::InvalidInput(format!(
                "dynamic verify received unsupported root D={root_d}"
            ))),
        }
    }

    fn batched_verify<T: Transcript<F>>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        opening_points: &[&[F]],
        opening_groups_by_point: &[&[&[F]]],
        commitments_by_point: &[&[Self::Commitment]],
        basis: BasisMode,
    ) -> Result<(), HachiError> {
        let root_d = uniform_root_ring_dim(
            std::iter::empty::<&[DynamicCommitHint<F>]>(),
            commitments_by_point.iter().copied(),
            "dynamic batched_verify",
        )?;
        match root_d {
            32 => {
                let typed_setup =
                    require_typed_verifier_setup(&setup.d32, 32, "dynamic batched_verify")?;
                let typed_commitments: Vec<Vec<RingCommitment<F, 32>>> = commitments_by_point
                    .iter()
                    .enumerate()
                    .map(|(point_idx, commitments)| {
                        commitments
                            .iter()
                            .enumerate()
                            .map(|(group_idx, commitment)| {
                                clone_typed_commitment!(
                                    commitment,
                                    D32,
                                    32,
                                    format!(
                                        "dynamic batched_verify point {point_idx} group {group_idx}"
                                    )
                                )
                            })
                            .collect()
                    })
                    .collect::<Result<_, _>>()?;
                let typed_commitment_refs: Vec<&[RingCommitment<F, 32>]> =
                    typed_commitments.iter().map(Vec::as_slice).collect();
                <HachiCommitmentScheme<32, Family::Cfg32> as CommitmentScheme<F, 32>>::batched_verify(
                    proof,
                    typed_setup,
                    transcript,
                    opening_points,
                    opening_groups_by_point,
                    &typed_commitment_refs,
                    basis,
                )
            }
            64 => {
                let typed_setup =
                    require_typed_verifier_setup(&setup.d64, 64, "dynamic batched_verify")?;
                let typed_commitments: Vec<Vec<RingCommitment<F, 64>>> = commitments_by_point
                    .iter()
                    .enumerate()
                    .map(|(point_idx, commitments)| {
                        commitments
                            .iter()
                            .enumerate()
                            .map(|(group_idx, commitment)| {
                                clone_typed_commitment!(
                                    commitment,
                                    D64,
                                    64,
                                    format!(
                                        "dynamic batched_verify point {point_idx} group {group_idx}"
                                    )
                                )
                            })
                            .collect()
                    })
                    .collect::<Result<_, _>>()?;
                let typed_commitment_refs: Vec<&[RingCommitment<F, 64>]> =
                    typed_commitments.iter().map(Vec::as_slice).collect();
                <HachiCommitmentScheme<64, Family::Cfg64> as CommitmentScheme<F, 64>>::batched_verify(
                    proof,
                    typed_setup,
                    transcript,
                    opening_points,
                    opening_groups_by_point,
                    &typed_commitment_refs,
                    basis,
                )
            }
            128 => {
                let typed_setup =
                    require_typed_verifier_setup(&setup.d128, 128, "dynamic batched_verify")?;
                let typed_commitments: Vec<Vec<RingCommitment<F, 128>>> = commitments_by_point
                    .iter()
                    .enumerate()
                    .map(|(point_idx, commitments)| {
                        commitments
                            .iter()
                            .enumerate()
                            .map(|(group_idx, commitment)| {
                                clone_typed_commitment!(
                                    commitment,
                                    D128,
                                    128,
                                    format!(
                                        "dynamic batched_verify point {point_idx} group {group_idx}"
                                    )
                                )
                            })
                            .collect()
                    })
                    .collect::<Result<_, _>>()?;
                let typed_commitment_refs: Vec<&[RingCommitment<F, 128>]> =
                    typed_commitments.iter().map(Vec::as_slice).collect();
                <HachiCommitmentScheme<128, Family::Cfg128> as CommitmentScheme<F, 128>>::batched_verify(
                    proof,
                    typed_setup,
                    transcript,
                    opening_points,
                    opening_groups_by_point,
                    &typed_commitment_refs,
                    basis,
                )
            }
            _ => unreachable!("uniform_root_ring_dim only returns supported root Ds"),
        }
    }

    fn protocol_name() -> &'static [u8] {
        b"hachi/dynamic-root"
    }
}

type Fp128AdaptiveBoundedD64<const LOG_COMMIT_BOUND: u32> =
    CommitmentPreset<fp128::Field, Fp128AdaptiveBoundedPolicy<64, LOG_COMMIT_BOUND, 1, 1, 1>>;
type Fp128AdaptiveBoundedD128<const LOG_COMMIT_BOUND: u32> =
    CommitmentPreset<fp128::Field, Fp128AdaptiveBoundedPolicy<128, LOG_COMMIT_BOUND, 1, 1, 1>>;

/// Dynamic fp128 dense family that chooses the root ring by estimated proof
/// bytes across `D=32/64/128`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DynamicFp128FullFamily;

impl DynamicRootConfigFamily<fp128::Field> for DynamicFp128FullFamily {
    type Cfg32 = fp128::D32Full;
    type Cfg64 = Fp128AdaptiveBoundedD64<128>;
    type Cfg128 = fp128::Full;

    fn select_root_ring_dim(key: HachiScheduleLookupKey) -> Result<usize, HachiError> {
        if fp128_exact_fit_singleton_prefers_d32(key) {
            return Ok(32);
        }
        select_smallest_estimated_proof_root_ring_dim::<Self::Cfg32, Self::Cfg64, Self::Cfg128>(key)
    }
}

/// Dynamic fp128 dense scheme that chooses the root ring at commit time.
pub type DynamicFp128FullScheme = DynamicHachiCommitmentScheme<DynamicFp128FullFamily>;

/// Dynamic fp128 onehot family that chooses the root ring by estimated proof
/// bytes across `D=32/64/128`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DynamicFp128OneHotFamily;

impl DynamicRootConfigFamily<fp128::Field> for DynamicFp128OneHotFamily {
    type Cfg32 = fp128::D32OneHot;
    type Cfg64 = fp128::OneHot;
    type Cfg128 = Fp128AdaptiveBoundedD128<1>;

    fn select_root_ring_dim(key: HachiScheduleLookupKey) -> Result<usize, HachiError> {
        if fp128_exact_fit_singleton_prefers_d32(key) {
            return Ok(32);
        }
        select_smallest_estimated_proof_root_ring_dim::<Self::Cfg32, Self::Cfg64, Self::Cfg128>(key)
    }
}

/// Dynamic fp128 onehot scheme that chooses the root ring at commit time.
pub type DynamicFp128OneHotScheme = DynamicHachiCommitmentScheme<DynamicFp128OneHotFamily>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::eq_poly::EqPolynomial;
    use crate::protocol::commitment::{
        CommitmentPreset, DynamicSmallTestCommitmentConfig, Fp128StaticBoundedPolicy,
    };
    use crate::protocol::root_poly::DenseMultilinear;
    use crate::protocol::Blake2bTranscript;
    use crate::test_utils::F;

    type SmallTest64 = CommitmentPreset<F, Fp128StaticBoundedPolicy<64, 32, 3, 3, 8, 4, 4>>;
    type SmallTest128 = CommitmentPreset<F, Fp128StaticBoundedPolicy<128, 32, 3, 3, 8, 4, 4>>;

    #[derive(Clone, Copy, Debug, Default)]
    struct SmallTestDynamicFamily;

    impl DynamicRootConfigFamily<F> for SmallTestDynamicFamily {
        type Cfg32 = DynamicSmallTestCommitmentConfig;
        type Cfg64 = SmallTest64;
        type Cfg128 = SmallTest128;

        fn select_root_ring_dim(_key: HachiScheduleLookupKey) -> Result<usize, HachiError> {
            Ok(32)
        }
    }

    type DynamicSmallTestScheme = DynamicHachiCommitmentScheme<SmallTestDynamicFamily>;

    #[test]
    fn dynamic_small_test_round_trip_matches_typed_scheme() {
        let num_vars = 6usize;
        let evals = EqPolynomial::<F>::evals(&vec![F::zero(); num_vars]);
        let poly = DenseMultilinear::from_field_evals(num_vars, &evals)
            .unwrap()
            .into();
        let setup =
            <DynamicSmallTestScheme as DynamicCommitmentScheme<F>>::setup_prover(num_vars, 1);
        let verifier =
            <DynamicSmallTestScheme as DynamicCommitmentScheme<F>>::setup_verifier(&setup);
        let (commitment, hint) = <DynamicSmallTestScheme as DynamicCommitmentScheme<F>>::commit(
            std::slice::from_ref(&poly),
            &setup,
        )
        .unwrap();
        let opening_point = vec![F::zero(); num_vars];
        let opening = evals[0];
        let mut prove_transcript = Blake2bTranscript::<F>::new(
            <DynamicSmallTestScheme as DynamicCommitmentScheme<F>>::protocol_name(),
        );
        let proof = <DynamicSmallTestScheme as DynamicCommitmentScheme<F>>::prove(
            &setup,
            &poly,
            &opening_point,
            hint,
            &mut prove_transcript,
            &commitment,
            BasisMode::Lagrange,
        )
        .unwrap();
        assert_eq!(commitment.root_ring_dim(), 32);
        let mut verify_transcript = Blake2bTranscript::<F>::new(
            <DynamicSmallTestScheme as DynamicCommitmentScheme<F>>::protocol_name(),
        );
        <DynamicSmallTestScheme as DynamicCommitmentScheme<F>>::verify(
            &proof,
            &verifier,
            &mut verify_transcript,
            &opening_point,
            &opening,
            &commitment,
            BasisMode::Lagrange,
        )
        .unwrap();
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct DynamicBatchSizeFamily;

    impl DynamicRootConfigFamily<F> for DynamicBatchSizeFamily {
        type Cfg32 = DynamicSmallTestCommitmentConfig;
        type Cfg64 = SmallTest64;
        type Cfg128 = SmallTest128;

        fn select_root_ring_dim(key: HachiScheduleLookupKey) -> Result<usize, HachiError> {
            Ok(if key.layout_num_claims == 1 { 32 } else { 64 })
        }
    }

    type DynamicBatchSizeScheme = DynamicHachiCommitmentScheme<DynamicBatchSizeFamily>;

    #[test]
    fn dynamic_batched_prove_rejects_mixed_root_dims() {
        let num_vars = 12usize;
        let evals = EqPolynomial::<F>::evals(&vec![F::zero(); num_vars]);
        let poly = DenseMultilinear::from_field_evals(num_vars, &evals)
            .unwrap()
            .into();
        let setup =
            <DynamicBatchSizeScheme as DynamicCommitmentScheme<F>>::setup_prover(num_vars, 2);

        let (commitment_a, hint_a) =
            <DynamicBatchSizeScheme as DynamicCommitmentScheme<F>>::commit(
                std::slice::from_ref(&poly),
                &setup,
            )
            .unwrap();
        let (commitment_b, hint_b) =
            <DynamicBatchSizeScheme as DynamicCommitmentScheme<F>>::commit(
                &[poly.clone(), poly],
                &setup,
            )
            .unwrap();

        assert_eq!(commitment_a.root_ring_dim(), 32);
        assert_eq!(commitment_b.root_ring_dim(), 64);

        let opening_point = vec![F::zero(); num_vars];
        let commitments = vec![commitment_a, commitment_b];
        let hints = vec![hint_a, hint_b];
        let polys = vec![DenseMultilinear::from_field_evals(num_vars, &evals)
            .unwrap()
            .into()];
        let poly_refs: Vec<&[MultilinearPolynomial<F>]> = vec![polys.as_slice()];
        let point_refs: Vec<&[&[MultilinearPolynomial<F>]]> = vec![poly_refs.as_slice()];
        let hint_groups = vec![hints];
        let commitment_groups = [commitments];
        let commitment_refs: Vec<&[DynamicRingCommitment<F>]> =
            commitment_groups.iter().map(Vec::as_slice).collect();
        let mut transcript = Blake2bTranscript::<F>::new(
            <DynamicBatchSizeScheme as DynamicCommitmentScheme<F>>::protocol_name(),
        );
        let err = <DynamicBatchSizeScheme as DynamicCommitmentScheme<F>>::batched_prove(
            &setup,
            &point_refs,
            &[&opening_point],
            hint_groups,
            &mut transcript,
            &commitment_refs,
            BasisMode::Lagrange,
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("requires one root D across the fused batch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn dynamic_full_exact_fit_singletons_use_fast_d32_path() {
        for num_vars in 6..=63 {
            let key = HachiScheduleLookupKey::singleton(num_vars, num_vars, 1);
            assert_eq!(
                DynamicFp128FullFamily::select_root_ring_dim(key).unwrap(),
                32
            );
        }
    }

    #[test]
    fn dynamic_onehot_exact_fit_singletons_use_fast_d32_path() {
        for num_vars in 6..=63 {
            let key = HachiScheduleLookupKey::singleton(num_vars, num_vars, 1);
            assert_eq!(
                DynamicFp128OneHotFamily::select_root_ring_dim(key).unwrap(),
                32
            );
        }
    }
}
