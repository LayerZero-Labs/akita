//! Adversarial test backend; production exposes no witness mutation operation.

use super::F;
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_cpu_backend::CpuBackend;
use akita_error::AkitaError;
use akita_params::{FoldSchedule, GroupCommitPhaseParams, OpeningClaimsLayout};
use akita_prover::backend::*;
use akita_types::binary_source::{
    BinaryCommittedGroupManifest, BinaryRootWitnessPlan, BinaryRootWitnessRequest,
    BinaryTerminalClaim,
};
use akita_types::{AkitaSetupDescriptor, Commitment};
use jolt_field::{Field, Ring, Zero};
use jolt_poly::UnivariatePoly;
use std::any::Any;
use std::sync::{Arc, Mutex};

type E = F;
type NativeBackend = CpuBackend<F, F>;
type C = <NativeBackend as ProverHandleFamily<F, F>>::CommitmentHandle;
type W = <NativeBackend as ProverHandleFamily<F, F>>::WitnessHandle;
type M = <NativeBackend as ProverHandleFamily<F, F>>::CommitmentMaterialHandle;
type Session = <NativeBackend as ProofScopeConsumer>::ProofSessionHandle;
type Stage2 = <NativeBackend as ProverHandleFamily<F, F>>::Stage2SessionHandle;

#[path = "delegation.rs"]
mod delegation;
#[path = "fault_witness.rs"]
mod fault_witness;

#[derive(Clone, Copy, Debug)]
pub(super) enum Fault {
    ForeignCommitment,
    ForeignPartials,
    Image,
    Response,
    Partials,
    ParityCarry,
    ValueCarry,
    QuotientCarry,
    Padding,
}

impl Fault {
    pub(super) const ALL: [Self; 7] = [
        Self::Image,
        Self::Response,
        Self::Partials,
        Self::ParityCarry,
        Self::ValueCarry,
        Self::QuotientCarry,
        Self::Padding,
    ];

    fn address(self, plan: &BinaryRootWitnessPlan) -> usize {
        let segment = match self {
            Self::ForeignCommitment | Self::ForeignPartials => unreachable!(),
            Self::Response => 0,
            Self::Image => 1,
            Self::QuotientCarry => 2,
            Self::Partials => 3,
            Self::ParityCarry => 4,
            Self::ValueCarry => 5,
            Self::Padding => {
                return (0..plan.repacking().domain_len())
                    .find(|&address| plan.repacking().is_padding_address(address).unwrap())
                    .unwrap()
            }
        };
        plan.repacking()
            .coefficient_address(segment, 0, 0, 0)
            .unwrap()
    }
}

struct FaultPreparation {
    native: Box<dyn Any + Send + Sync>,
    manifest: BinaryCommittedGroupManifest,
    claim: BinaryTerminalClaim,
    plan: BinaryRootWitnessPlan,
}

pub(super) struct FaultBackend {
    inner: Arc<NativeBackend>,
    seed: [u8; 32],
    words: Vec<u128>,
    fault: Fault,
    digits: Mutex<Option<Vec<i8>>>,
}

impl FaultBackend {
    pub(super) fn new(
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
        let request = request
            .downcast_ref::<BinaryRootWitnessRequest<C>>()
            .unwrap();
        if matches!(self.fault, Fault::ForeignPartials) {
            let foreign = fault_witness::foreign_partial_payload(
                &request.manifest,
                &request.claim,
                &self.words,
                &self.seed,
            )?;
            assert_ne!(
                payload, foreign,
                "foreign U must have a distinct D commitment"
            );
            payload = foreign;
        }
        assert!(
            request.ordinary.is_none(),
            "fault injection uses the binary-only fixture"
        );
        Ok((
            payload,
            Box::new(FaultPreparation {
                native,
                manifest: request.manifest.clone(),
                claim: request.claim.clone(),
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
        if self
            .inner
            .build_root_group(
                session,
                context,
                preparation.native.as_ref(),
                challenges,
                logical_len,
                destination_ring_dimension,
            )?
            .is_none()
        {
            return Ok(None);
        }
        let mut digits = fault_witness::reconstruct(
            &preparation.manifest,
            &preparation.claim,
            &preparation.plan,
            &self.seed,
            &self.words,
            challenges,
        )?;
        digits.truncate(logical_len);
        if !matches!(
            self.fault,
            Fault::ForeignCommitment | Fault::ForeignPartials
        ) {
            let address = self.fault.address(&preparation.plan);
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
        mut weights: Vec<F>,
        negative_binary_weights: Vec<(usize, F)>,
        claim: F,
        stage1_point: &[F],
        range_image_evaluation: F,
        batching: F,
    ) -> Result<Stage2, AkitaError> {
        // A malicious prover is allowed to use an incorrect public oracle. Make
        // its own sumcheck consistent so it emits a complete proof; the real
        // verifier must detect the wrong relation at its final oracle check.
        assert!(negative_binary_weights.is_empty());
        let state = self.digits.lock().unwrap();
        let digits = state.as_ref().unwrap();
        let actual = weights
            .iter()
            .zip(digits)
            .fold(F::zero(), |sum, (&weight, &digit)| {
                sum + weight * F::from_i64(i64::from(digit))
            });
        if !matches!(
            self.fault,
            Fault::ForeignCommitment | Fault::ForeignPartials
        ) {
            assert_ne!(
                actual, claim,
                "the selected mutation must violate the public relation"
            );
        }
        let anchor = digits.iter().position(|&digit| digit != 0).unwrap();
        let value = F::from_i64(i64::from(digits[anchor]));
        weights[anchor] += (claim - actual) * value.inverse().unwrap();
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
