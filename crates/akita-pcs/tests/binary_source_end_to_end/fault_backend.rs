//! Adversarial test backend; production exposes no witness mutation operation.

use super::{E, F};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_cpu_backend::CpuBackend;
use akita_error::AkitaError;
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::{FoldSchedule, GroupCommitPhaseParams, OpeningClaimsLayout};
use akita_prover::backend::*;
use akita_types::binary_source::{
    BinaryCommittedGroupManifest, BinaryRootWitnessPlan, BinaryRootWitnessRequest,
    BinaryTerminalClaim, PrimePlaneManifest,
};
use akita_types::native_root_relation::NativeRootWeights;
use akita_types::{AkitaSetupDescriptor, Commitment};
use jolt_field::{Field, Ring, Zero};
use jolt_poly::UnivariatePoly;
use std::any::Any;
use std::sync::{Arc, Mutex};

type NativeBackend = CpuBackend<F, E>;
type C = <NativeBackend as ProverHandleFamily<F, E>>::CommitmentHandle;
type W = <NativeBackend as ProverHandleFamily<F, E>>::WitnessHandle;
type M = <NativeBackend as ProverHandleFamily<F, E>>::CommitmentMaterialHandle;
type Session = <NativeBackend as ProofScopeConsumer>::ProofSessionHandle;
type Stage2 = <NativeBackend as ProverHandleFamily<F, E>>::Stage2SessionHandle;

#[path = "delegation.rs"]
mod delegation;
#[path = "fault_witness.rs"]
mod fault_witness;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Fault {
    ForeignCommitment,
    ForeignPartials,
    Image,
    Response,
    Partials,
    ParityCarry,
    ValueCarry,
    Padding,
    /// First digit of one owner's coordinate plane at one prime claim point.
    PrimePlane {
        owner: usize,
        point: usize,
        coordinate: usize,
    },
    /// Honest coordinate planes; the caller binds a false prime value limb.
    PrimeValue {
        point: usize,
        coordinate: usize,
    },
    /// Honest witness; the caller binds a false ordinary reading.
    OrdinaryValue,
    Quotient,
}

impl Fault {
    pub(super) const ALL: [Self; 6] = [
        Self::Image,
        Self::Response,
        Self::Partials,
        Self::ParityCarry,
        Self::ValueCarry,
        Self::Padding,
    ];

    /// No root row reads a padding address, so an in-range digit there leaves
    /// the public relation and every reading unchanged. The range check, the
    /// successor and its norm checks still bound the committed digit.
    pub(super) fn is_accepted(self) -> bool {
        matches!(self, Self::Padding)
    }

    pub(super) fn address(
        self,
        plan: &BinaryRootWitnessPlan,
        prime: Option<(&PrimePlaneManifest, usize)>,
    ) -> usize {
        let mut element = 0;
        let segment = match self {
            Self::ForeignCommitment
            | Self::ForeignPartials
            | Self::PrimeValue { .. }
            | Self::OrdinaryValue => unreachable!(),
            Self::PrimePlane {
                owner,
                point,
                coordinate,
            } => {
                let (prime, columns) = prime.unwrap();
                let plane = prime
                    .planes()
                    .iter()
                    .position(|plane| {
                        plane.polynomial == owner
                            && plane.point == point
                            && plane.coordinate == coordinate
                    })
                    .unwrap();
                assert!(plane < prime.planes().len());
                element = plane * columns;
                plan.prime_segment().unwrap()
            }
            Self::Quotient => plan.quotient_segment().unwrap(),
            Self::Response => 0,
            Self::Image => 1,
            Self::Partials => 2,
            Self::ParityCarry => 3,
            Self::ValueCarry => 4,
            Self::Padding => {
                return (0..plan.repacking().domain_len())
                    .find(|&address| plan.repacking().is_padding_address(address).unwrap())
                    .unwrap()
            }
        };
        plan.repacking()
            .coefficient_address(segment, element, 0, 0)
            .unwrap()
    }
}

struct FaultPreparation {
    native: Box<dyn Any + Send + Sync>,
    manifest: BinaryCommittedGroupManifest,
    claim: Option<BinaryTerminalClaim>,
    prime: PrimePlaneManifest,
    plan: BinaryRootWitnessPlan,
}

pub(crate) struct FaultBackend {
    inner: Arc<NativeBackend>,
    seed: [u8; 32],
    words: Vec<u128>,
    fault: Fault,
    digits: Mutex<Option<Vec<i8>>>,
}

impl FaultBackend {
    /// Complete an adversarial opening with its deliberately false public oracle.
    pub(crate) fn prove_prime_fault(
        &self,
        setup: &akita_cpu_backend::AkitaProverSetup<F>,
        schedules: &TrustedScheduleCatalog<super::BinaryTestSuffix>,
        prefixes: &akita_prover::SetupPrefixProverRegistry<F, C>,
        data: akita_prover::SelectedProverOpeningData<'_, E, C, F>,
        prover: &mut akita_transcript::ProverChannel,
    ) -> Result<(), AkitaError> {
        let resolved = schedules.resolve_selection(data.selection())?;
        let mut registry = akita_prover::BackendRegistry::<super::BinaryTestSuffix>::new()?;
        let root = registry.register(self, prefixes)?;
        let mut route =
            akita_prover::FixedFoldRoute::new(vec![root; resolved.schedule().num_fold_levels()]);
        akita_prover::batched_prove_embedded::<super::BinaryTestSuffix, _>(
            setup.expanded.descriptor(),
            schedules,
            &registry,
            data,
            prover,
            akita_params::BasisMode::Lagrange,
            &mut route,
        )
    }

    pub(crate) fn new(
        inner: Arc<NativeBackend>,
        seed: [u8; 32],
        words: Vec<u128>,
        fault: Fault,
    ) -> Self {
        Self {
            inner,
            seed,
            words,
            fault,
            digits: Mutex::new(None),
        }
    }

    fn prepare_fault_root_group(
        &self,
        session: &Session,
        context: &ProofContext,
        handle: &C,
        request: &dyn Any,
    ) -> Result<(Vec<F>, Box<dyn Any + Send + Sync>), AkitaError> {
        let (mut payload, native) = self
            .inner
            .prepare_root_group(session, context, handle, request)?;
        if request.is::<akita_types::binary_source::BinaryFieldSwitchSourceRequest>()
            || request.is::<akita_types::binary_source::BinaryProjectionRequest>()
        {
            return Ok((payload, native));
        }
        let request = request
            .downcast_ref::<BinaryRootWitnessRequest<C>>()
            .unwrap();
        if matches!(self.fault, Fault::ForeignPartials) {
            let foreign = fault_witness::foreign_partial_payload(
                &request.manifest,
                request.claim.as_ref().unwrap(),
                &self.words,
                &self.seed,
            )?;
            assert_ne!(
                payload, foreign,
                "foreign U must have a distinct D commitment"
            );
            payload = foreign;
        }
        if request.manifest.field_profile() == BinaryFieldProfile::P128 {
            assert!(
                request.ordinary.is_none(),
                "P128 fault fixtures are binary-only"
            );
        }
        Ok((
            payload,
            Box::new(FaultPreparation {
                native,
                manifest: request.manifest.clone(),
                claim: request.claim.clone(),
                prime: request.prime.clone(),
                plan: request.plan.clone(),
            }),
        ))
    }

    fn build_fault_root_group(
        &self,
        session: &Session,
        context: &ProofContext,
        preparation: &dyn Any,
        challenges: &akita_challenges::Challenges,
        logical_len: usize,
        destination_ring_dimension: usize,
    ) -> Result<Option<W>, AkitaError> {
        let preparation = preparation.downcast_ref::<FaultPreparation>().unwrap();
        let Some(honest) = self.inner.build_root_group(
            session,
            context,
            preparation.native.as_ref(),
            challenges,
            logical_len,
            destination_ring_dimension,
        )?
        else {
            return Ok(None);
        };
        let mut digits = match preparation.manifest.field_profile() {
            BinaryFieldProfile::P128 => fault_witness::reconstruct(
                &preparation.manifest,
                preparation.claim.as_ref(),
                &preparation.prime,
                &preparation.plan,
                &self.seed,
                &self.words,
                challenges,
            )?,
            BinaryFieldProfile::P64 => honest.coefficients_for_test(),
        };
        digits.truncate(logical_len);
        if !matches!(
            self.fault,
            Fault::ForeignCommitment
                | Fault::ForeignPartials
                | Fault::PrimeValue { .. }
                | Fault::OrdinaryValue
        ) {
            let address = self.fault.address(
                &preparation.plan,
                Some((
                    &preparation.prime,
                    preparation.manifest.descriptor().index_map().columns(),
                )),
            );
            let old = digits[address];
            digits[address] = if old == 3 { 2 } else { old + 1 };
            assert_ne!(digits[address], old);
            assert!((-4..=3).contains(&digits[address]));
        }
        *self.digits.lock().unwrap() = Some(digits.clone());
        self.inner
            .import_root_witness(session, context, digits, destination_ring_dimension)
            .map(Some)
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_fault_root_relation(
        &self,
        witness: &W,
        weights: NativeRootWeights<E>,
        negative_binary_weights: Vec<(usize, E)>,
        claim: E,
        stage1_point: &[E],
        range_image_evaluation: E,
        batching: E,
    ) -> Result<Stage2, AkitaError> {
        // A malicious prover is allowed to use an incorrect public oracle. Make
        // its own sumcheck consistent so it emits a complete proof; the real
        // verifier must detect the wrong relation at its final oracle check.
        let mut table = weights.materialize()?;
        let state = self.digits.lock().unwrap();
        let digits = state.as_ref().unwrap();
        let negative_contribution =
            negative_binary_weights
                .iter()
                .fold(E::zero(), |sum, &(address, weight)| {
                    let digit = E::from_i64(i64::from(digits[address]));
                    assert!(matches!(digits[address], -1 | 0));
                    sum + weight * digit * (digit + E::from_u64(1))
                });
        assert_eq!(
            negative_contribution,
            E::zero(),
            "the forged linear oracle must preserve every ordinary negative-bit predicate"
        );
        let actual = table
            .iter()
            .zip(digits)
            .fold(E::zero(), |sum, (&weight, &digit)| {
                sum + weight * E::from_i64(i64::from(digit))
            });
        match self.fault {
            Fault::ForeignCommitment | Fault::ForeignPartials => {}
            Fault::Padding => assert_eq!(
                actual, claim,
                "no root row may read the mutated padding address"
            ),
            _ => assert_ne!(
                actual, claim,
                "the selected mutation must violate the public relation"
            ),
        }
        let anchor = digits.iter().position(|&digit| digit != 0).unwrap();
        let value = E::from_i64(i64::from(digits[anchor]));
        table[anchor] += (claim - actual) * value.inverse().unwrap();
        let terms = table
            .into_iter()
            .enumerate()
            .filter(|(_, weight)| !weight.is_zero())
            .collect();
        let weights = NativeRootWeights::new(weights.table_len(), terms, weights.claim())?;
        self.inner.begin_root_relation(
            witness,
            weights,
            negative_binary_weights,
            claim,
            stage1_point,
            range_image_evaluation,
            batching,
        )
    }
}

/// Exercise the actual native root carry constructor on a frontend terminal.
pub(super) fn authenticate_terminal(
    manifest: &BinaryCommittedGroupManifest,
    claim: &BinaryTerminalClaim,
    source: &[u128],
    challenges: &akita_challenges::Challenges,
) -> Result<(), AkitaError> {
    let plan = BinaryRootWitnessPlan::for_profile(
        manifest.descriptor(),
        manifest.a_rank(),
        256,
        BinaryFieldProfile::P128,
        true,
        0,
    )?;
    fault_witness::reconstruct(
        manifest,
        Some(claim),
        &PrimePlaneManifest::with_field_profile(
            manifest.descriptor(),
            None,
            BinaryFieldProfile::P128,
        )?,
        &plan,
        &[53; 32],
        source,
        challenges,
    )?;
    Ok(())
}

/// Check the verifier's terminal relation on the honest authenticated digits.
/// A false received terminal must fail even if its frontend multiplier is zero.
pub(super) fn authenticate_terminal_relation(
    manifest: &BinaryCommittedGroupManifest,
    honest: &BinaryTerminalClaim,
    received: &BinaryTerminalClaim,
    source: &[u128],
    challenges: &akita_challenges::Challenges,
) -> Result<(), AkitaError> {
    let plan = BinaryRootWitnessPlan::for_profile(
        manifest.descriptor(),
        manifest.a_rank(),
        256,
        BinaryFieldProfile::P128,
        true,
        0,
    )?;
    let digits = fault_witness::reconstruct(
        manifest,
        Some(honest),
        &PrimePlaneManifest::with_field_profile(
            manifest.descriptor(),
            None,
            BinaryFieldProfile::P128,
        )?,
        &plan,
        &[53; 32],
        source,
        challenges,
    )?;
    let weights = received.relation_weights(
        manifest.descriptor(),
        plan.repacking(),
        akita_types::binary_source::BinaryEvaluationSegments {
            response: 0,
            partials: 2,
            parity_carry: 3,
            value_carry: 4,
        },
        challenges,
        F::from_u64(1),
        [F::from_u64(1); 2],
        u128::MAX - 274,
    )?;
    let table = digits
        .into_iter()
        .map(|digit| F::from_i64(i64::from(digit)))
        .collect::<Vec<_>>();
    weights.check_table(&table)
}
