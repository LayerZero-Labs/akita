//! Dynamic root-ring wrapper around the typed Hachi commitment scheme.
//!
//! The low-level Hachi kernels remain const-generic over the root ring degree
//! `D`. This module lifts the public API one level up so callers can provide
//! ring-agnostic root polynomials and let setup choose the root ring from
//! public inputs.

use crate::algebra::fields::wide::HasWide;
use crate::algebra::fields::HasUnreducedOps;
use crate::error::HachiError;
use crate::primitives::serialization::Valid;
use crate::protocol::commitment::{
    hachi_batched_root_layout, AppendToTranscript, CommitmentConfig, CommitmentScheme,
    DynamicCommitmentScheme, HachiProverSetup, HachiVerifierSetup, RingCommitment,
};
use crate::protocol::commitment_scheme::HachiCommitmentScheme;
use crate::protocol::opening_point::BasisMode;
use crate::protocol::proof::{HachiBatchedCommitmentHint, HachiBatchedProof, HachiProof};
use crate::protocol::root_poly::{MultilinearPolynomial, TypedRootPolynomial};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling};
use std::marker::PhantomData;

/// Family-level selector for dynamic root-ring Hachi schemes.
///
/// Each associated config fixes one concrete root ring degree. The family
/// chooses which root degree to use at setup time from public inputs. After
/// that choice, the protocol runs through the existing typed kernel for that
/// fixed-D config.
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

    /// Choose the root ring degree for a setup.
    ///
    /// # Errors
    ///
    /// Returns an error if the family cannot choose a supported root ring from
    /// the provided public setup parameters.
    fn select_root_ring_dim(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<usize, HachiError>;
}

/// D-erased prover setup for the public dynamic Hachi API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicHachiProverSetup<F: FieldCore> {
    /// Root ring `D=32`.
    D32(Box<HachiProverSetup<F, 32>>),
    /// Root ring `D=64`.
    D64(Box<HachiProverSetup<F, 64>>),
    /// Root ring `D=128`.
    D128(Box<HachiProverSetup<F, 128>>),
}

impl<F: FieldCore> DynamicHachiProverSetup<F> {
    /// Root ring degree selected for this setup.
    pub fn root_ring_dim(&self) -> usize {
        match self {
            Self::D32(_) => 32,
            Self::D64(_) => 64,
            Self::D128(_) => 128,
        }
    }

    /// Maximum batch capacity carried by this setup.
    pub fn max_num_batched_polys(&self) -> usize {
        match self {
            Self::D32(setup) => setup.max_num_batched_polys(),
            Self::D64(setup) => setup.max_num_batched_polys(),
            Self::D128(setup) => setup.max_num_batched_polys(),
        }
    }
}

/// D-erased verifier setup for the public dynamic Hachi API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicHachiVerifierSetup<F: FieldCore> {
    /// Root ring `D=32`.
    D32(HachiVerifierSetup<F>),
    /// Root ring `D=64`.
    D64(HachiVerifierSetup<F>),
    /// Root ring `D=128`.
    D128(HachiVerifierSetup<F>),
}

impl<F: FieldCore> DynamicHachiVerifierSetup<F> {
    /// Root ring degree selected for this setup.
    pub fn root_ring_dim(&self) -> usize {
        match self {
            Self::D32(_) => 32,
            Self::D64(_) => 64,
            Self::D128(_) => 128,
        }
    }
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
        match Family::select_root_ring_dim(max_num_vars, max_num_batched_polys)
            .expect("dynamic root selection failed")
        {
            32 => DynamicHachiProverSetup::D32(Box::new(
                <HachiCommitmentScheme<32, Family::Cfg32> as CommitmentScheme<F, 32>>::setup_prover(
                    max_num_vars,
                    max_num_batched_polys,
                ),
            )),
            64 => DynamicHachiProverSetup::D64(Box::new(
                <HachiCommitmentScheme<64, Family::Cfg64> as CommitmentScheme<F, 64>>::setup_prover(
                    max_num_vars,
                    max_num_batched_polys,
                ),
            )),
            128 => DynamicHachiProverSetup::D128(Box::new(<HachiCommitmentScheme<
                128,
                Family::Cfg128,
            > as CommitmentScheme<F, 128>>::setup_prover(
                max_num_vars, max_num_batched_polys
            ))),
            root_d => panic!("unsupported dynamic root ring dimension: {root_d}"),
        }
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        match setup {
            DynamicHachiProverSetup::D32(setup) => DynamicHachiVerifierSetup::D32(
                <HachiCommitmentScheme<32, Family::Cfg32> as CommitmentScheme<F, 32>>::setup_verifier(
                    setup,
                ),
            ),
            DynamicHachiProverSetup::D64(setup) => DynamicHachiVerifierSetup::D64(
                <HachiCommitmentScheme<64, Family::Cfg64> as CommitmentScheme<F, 64>>::setup_verifier(
                    setup,
                ),
            ),
            DynamicHachiProverSetup::D128(setup) => DynamicHachiVerifierSetup::D128(
                <HachiCommitmentScheme<128, Family::Cfg128> as CommitmentScheme<F, 128>>::setup_verifier(
                    setup,
                ),
            ),
        }
    }

    fn commit(
        polys: &[MultilinearPolynomial<F>],
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::CommitHint), HachiError> {
        match setup {
            DynamicHachiProverSetup::D32(setup) => {
                let typed_polys = materialize_typed_root_group::<F, 32, Family::Cfg32>(
                    polys,
                    setup.max_num_batched_polys(),
                    "dynamic commit",
                )?;
                let (commitment, hint) =
                    <HachiCommitmentScheme<32, Family::Cfg32> as CommitmentScheme<F, 32>>::commit(
                        &typed_polys,
                        setup,
                    )?;
                Ok((
                    DynamicRingCommitment::D32(commitment),
                    DynamicCommitHint::D32(hint),
                ))
            }
            DynamicHachiProverSetup::D64(setup) => {
                let typed_polys = materialize_typed_root_group::<F, 64, Family::Cfg64>(
                    polys,
                    setup.max_num_batched_polys(),
                    "dynamic commit",
                )?;
                let (commitment, hint) =
                    <HachiCommitmentScheme<64, Family::Cfg64> as CommitmentScheme<F, 64>>::commit(
                        &typed_polys,
                        setup,
                    )?;
                Ok((
                    DynamicRingCommitment::D64(commitment),
                    DynamicCommitHint::D64(hint),
                ))
            }
            DynamicHachiProverSetup::D128(setup) => {
                let typed_polys = materialize_typed_root_group::<F, 128, Family::Cfg128>(
                    polys,
                    setup.max_num_batched_polys(),
                    "dynamic commit",
                )?;
                let (commitment, hint) = <HachiCommitmentScheme<128, Family::Cfg128> as CommitmentScheme<
                    F,
                    128,
                >>::commit(&typed_polys, setup)?;
                Ok((
                    DynamicRingCommitment::D128(commitment),
                    DynamicCommitHint::D128(hint),
                ))
            }
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
        match setup {
            DynamicHachiProverSetup::D32(setup) => {
                let typed_poly = materialize_typed_root_group::<F, 32, Family::Cfg32>(
                    std::slice::from_ref(poly),
                    setup.max_num_batched_polys(),
                    "dynamic prove",
                )?;
                let typed_hint = clone_typed_hint!(hint, D32, 32, "dynamic prove")?;
                let typed_commitment =
                    clone_typed_commitment!(commitment, D32, 32, "dynamic prove")?;
                <HachiCommitmentScheme<32, Family::Cfg32> as CommitmentScheme<F, 32>>::prove(
                    setup,
                    &typed_poly[0],
                    opening_point,
                    typed_hint,
                    transcript,
                    &typed_commitment,
                    basis,
                )
            }
            DynamicHachiProverSetup::D64(setup) => {
                let typed_poly = materialize_typed_root_group::<F, 64, Family::Cfg64>(
                    std::slice::from_ref(poly),
                    setup.max_num_batched_polys(),
                    "dynamic prove",
                )?;
                let typed_hint = clone_typed_hint!(hint, D64, 64, "dynamic prove")?;
                let typed_commitment =
                    clone_typed_commitment!(commitment, D64, 64, "dynamic prove")?;
                <HachiCommitmentScheme<64, Family::Cfg64> as CommitmentScheme<F, 64>>::prove(
                    setup,
                    &typed_poly[0],
                    opening_point,
                    typed_hint,
                    transcript,
                    &typed_commitment,
                    basis,
                )
            }
            DynamicHachiProverSetup::D128(setup) => {
                let typed_poly = materialize_typed_root_group::<F, 128, Family::Cfg128>(
                    std::slice::from_ref(poly),
                    setup.max_num_batched_polys(),
                    "dynamic prove",
                )?;
                let typed_hint = clone_typed_hint!(hint, D128, 128, "dynamic prove")?;
                let typed_commitment =
                    clone_typed_commitment!(commitment, D128, 128, "dynamic prove")?;
                <HachiCommitmentScheme<128, Family::Cfg128> as CommitmentScheme<F, 128>>::prove(
                    setup,
                    &typed_poly[0],
                    opening_point,
                    typed_hint,
                    transcript,
                    &typed_commitment,
                    basis,
                )
            }
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
        match setup {
            DynamicHachiProverSetup::D32(setup) => {
                let typed_polys = materialize_typed_root_groups_by_point::<F, 32, Family::Cfg32>(
                    poly_groups_by_point,
                    setup.max_num_batched_polys(),
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
                    setup,
                    &typed_point_refs,
                    opening_points,
                    typed_hints,
                    transcript,
                    &typed_commitment_refs,
                    basis,
                )
            }
            DynamicHachiProverSetup::D64(setup) => {
                let typed_polys = materialize_typed_root_groups_by_point::<F, 64, Family::Cfg64>(
                    poly_groups_by_point,
                    setup.max_num_batched_polys(),
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
                    setup,
                    &typed_point_refs,
                    opening_points,
                    typed_hints,
                    transcript,
                    &typed_commitment_refs,
                    basis,
                )
            }
            DynamicHachiProverSetup::D128(setup) => {
                let typed_polys = materialize_typed_root_groups_by_point::<F, 128, Family::Cfg128>(
                    poly_groups_by_point,
                    setup.max_num_batched_polys(),
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
                    setup,
                    &typed_point_refs,
                    opening_points,
                    typed_hints,
                    transcript,
                    &typed_commitment_refs,
                    basis,
                )
            }
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
        match setup {
            DynamicHachiVerifierSetup::D32(setup) => {
                let typed_commitment =
                    clone_typed_commitment!(commitment, D32, 32, "dynamic verify")?;
                <HachiCommitmentScheme<32, Family::Cfg32> as CommitmentScheme<F, 32>>::verify(
                    proof,
                    setup,
                    transcript,
                    opening_point,
                    opening,
                    &typed_commitment,
                    basis,
                )
            }
            DynamicHachiVerifierSetup::D64(setup) => {
                let typed_commitment =
                    clone_typed_commitment!(commitment, D64, 64, "dynamic verify")?;
                <HachiCommitmentScheme<64, Family::Cfg64> as CommitmentScheme<F, 64>>::verify(
                    proof,
                    setup,
                    transcript,
                    opening_point,
                    opening,
                    &typed_commitment,
                    basis,
                )
            }
            DynamicHachiVerifierSetup::D128(setup) => {
                let typed_commitment =
                    clone_typed_commitment!(commitment, D128, 128, "dynamic verify")?;
                <HachiCommitmentScheme<128, Family::Cfg128> as CommitmentScheme<F, 128>>::verify(
                    proof,
                    setup,
                    transcript,
                    opening_point,
                    opening,
                    &typed_commitment,
                    basis,
                )
            }
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
        match setup {
            DynamicHachiVerifierSetup::D32(setup) => {
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
                    setup,
                    transcript,
                    opening_points,
                    opening_groups_by_point,
                    &typed_commitment_refs,
                    basis,
                )
            }
            DynamicHachiVerifierSetup::D64(setup) => {
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
                    setup,
                    transcript,
                    opening_points,
                    opening_groups_by_point,
                    &typed_commitment_refs,
                    basis,
                )
            }
            DynamicHachiVerifierSetup::D128(setup) => {
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
                    setup,
                    transcript,
                    opening_points,
                    opening_groups_by_point,
                    &typed_commitment_refs,
                    basis,
                )
            }
        }
    }

    fn protocol_name() -> &'static [u8] {
        b"hachi/dynamic-root"
    }
}

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

        fn select_root_ring_dim(
            _max_num_vars: usize,
            _max_num_batched_polys: usize,
        ) -> Result<usize, HachiError> {
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
        assert_eq!(setup.root_ring_dim(), 32);
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
}
