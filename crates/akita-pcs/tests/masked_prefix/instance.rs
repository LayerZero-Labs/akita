//! The masked prefix's shared test instances: the planner's zero-knowledge
//! rows, committed groups and their claims, and both roles' runs over the
//! masked channel. Used by `masked_prefix.rs` and `zk_end_to_end.rs`.

#![allow(dead_code)]

use crate::common;
use akita_config::{
    proof_optimized::fp128, CommitmentConfig, TrustedScheduleCatalog, ZkCommitmentConfig,
};
use akita_cpu_backend::{AkitaProverSetup, CommitmentHandle, CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_params::zk::ZK_CUT_LEVEL;
use akita_params::{
    lagrange_weights, BasisMode, ChunkedWitnessCfg, CommittedGroupBatchProfile,
    OpeningScheduleSelection, PolynomialGroupLayout, ScheduleLookupKey,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    prove_masked_prefix, ResidualOpening, SelectedProverOpeningData, SetupPrefixProverRegistry,
};
use akita_types::{
    AkitaSponge, CommittedGroup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims,
    PROOF_STREAM_PROTOCOL,
};
use akita_verifier::{AkitaVerifier, MaskedPrefix};
use jolt_algebra::{Ring, Zero};
use jolt_transcript::{
    Channel, ClaimDomain, PrivateChannel, ProverTranscript, SiteId, VerifierTranscript,
};
use jolt_zk::masking::{
    AffineRow, CommittedPads, CommittedPadsView, OperandShapeDomain, PadIndex, PadLayout, ZkError,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// The clear preset whose zero-knowledge family the tests plan in.
pub(crate) type Clear = fp128::Dense;
pub(crate) type Cfg = ZkCommitmentConfig<Clear>;
pub(crate) type F = <Cfg as CommitmentConfig>::Field;
pub(crate) type E = F;

pub(crate) const NV: usize = 14;
pub(crate) const SESSION: &[u8] = b"masked-prefix";
pub(crate) const BASIS: BasisMode = BasisMode::Lagrange;

/// A masked prefix's proof bytes, residual opening and auxiliary values.
pub(crate) type Proved = (Vec<u8>, ResidualOpening<F, F>, Vec<F>);

/// Diagnostic sites of the statement's masked claims and of the auxiliary
/// block, which the caller sends around Akita's prefix. Sites never reach
/// the sponge.
pub(crate) const CLAIMS_SITE: SiteId = SiteId::label("harness/masked-claims");
pub(crate) const AUXILIARY_SITE: SiteId = SiteId::label("harness/auxiliary");

/// Generous room for every message of the prefix, for the counting replay.
pub(crate) const COUNTING_BYTES: usize = 1 << 22;

pub(crate) fn on_large_stack(test: fn()) {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(test)
        .unwrap()
        .join()
        .unwrap();
}

/// The checked-in `fp128_dense_zk` catalog.
pub(crate) fn zk_catalog() -> TrustedScheduleCatalog<Cfg> {
    akita_config::test_support::workspace_schedule_catalog::<Cfg>(
        ChunkedWitnessCfg::default_non_chunked(),
    )
    .unwrap()
}

/// The zero-knowledge row for one dense `NV`-variable polynomial.
pub(crate) fn dense_zk_row() -> (CommittedGroupBatchProfile, akita_params::FoldSchedule) {
    let catalog = zk_catalog();
    let row = catalog
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(NV),
        ))
        .unwrap();
    (row.profiles().clone(), row.schedule().clone())
}

/// One committed group of one polynomial and its claim.
pub(crate) struct Group {
    pub(crate) commitment: CommittedGroup<F>,
    pub(crate) handle: CommitmentHandle<F, F>,
    pub(crate) point: Vec<F>,
    pub(crate) opening: F,
}

/// Committed groups and their claims, with both roles' setups.
pub(crate) struct Instance<C: CommitmentConfig<Field = F, ExtField = F> = Cfg> {
    pub(crate) scheme: AkitaCommitmentScheme<C>,
    pub(crate) setup: AkitaProverSetup<F>,
    pub(crate) backend: CpuBackend<F, F>,
    pub(crate) verifier: AkitaVerifier<C>,
    pub(crate) groups: Vec<Group>,
}

pub(crate) fn dense_poly(num_vars: usize) -> (Vec<F>, DensePoly<F>) {
    let evals: Vec<F> = (0..(1 << num_vars))
        .map(|index| F::from_u64(3 * index as u64 + 1))
        .collect();
    let poly = DensePoly::<F>::from_field_evals(num_vars, &evals).unwrap();
    (evals, poly)
}

pub(crate) fn lagrange_opening(evals: &[F], point: &[F]) -> F {
    evals
        .iter()
        .zip(lagrange_weights(point).unwrap())
        .fold(F::zero(), |sum, (&value, weight)| sum + value * weight)
}

impl<C: CommitmentConfig<Field = F, ExtField = F>> Instance<C> {
    pub(crate) fn new(scheme: AkitaCommitmentScheme<C>) -> Self {
        Self::seeded(scheme, 7)
    }

    /// [`new`](Self::new), committing with hiding randomness from `seed`.
    pub(crate) fn seeded(scheme: AkitaCommitmentScheme<C>, seed: u64) -> Self {
        Self::with_polynomial(scheme, seed, dense_poly(NV))
    }

    /// [`seeded`](Self::seeded) for the polynomial `(evals, poly)`.
    pub(crate) fn with_polynomial(
        scheme: AkitaCommitmentScheme<C>,
        seed: u64,
        (evals, poly): (Vec<F>, DensePoly<F>),
    ) -> Self {
        let setup = scheme.setup_prover(NV, 1).unwrap();
        let backend = CpuBackend::new(setup.expanded.clone())
            .unwrap()
            .with_hiding_rng(StdRng::seed_from_u64(seed));
        let output = backend
            .commit(
                scheme.schedules(),
                &backend.import_source(vec![poly]).unwrap(),
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        let point: Vec<F> = (0..NV)
            .map(|index| F::from_u64((5 * index + 2) as u64))
            .collect();
        let opening = lagrange_opening(&evals, &point);
        Self::with_groups(
            scheme,
            setup,
            backend,
            vec![Group {
                commitment: output.committed_group,
                handle: output.private_handle,
                point,
                opening,
            }],
        )
    }
}

impl<C: CommitmentConfig<Field = F, ExtField = F>> Instance<C> {
    pub(crate) fn with_groups(
        scheme: AkitaCommitmentScheme<C>,
        setup: AkitaProverSetup<F>,
        backend: CpuBackend<F, F>,
        groups: Vec<Group>,
    ) -> Self {
        let mut instance = Self {
            verifier: scheme
                .verifier(scheme.setup_verifier(&setup).unwrap())
                .unwrap(),
            scheme,
            setup,
            backend,
            groups,
        };
        let resolved = instance
            .scheme
            .schedules()
            .resolve_selection(instance.selection())
            .unwrap();
        instance.verifier = instance
            .scheme
            .verifier(
                instance
                    .scheme
                    .setup_verifier_for_schedule(
                        &instance.setup,
                        resolved.schedule(),
                        instance.opening_data().opening_layout(),
                    )
                    .unwrap(),
            )
            .unwrap();
        instance
    }

    pub(crate) fn opening_data(
        &self,
    ) -> SelectedProverOpeningData<'static, F, CommitmentHandle<F, F>, F> {
        SelectedProverOpeningData::from_committed_claims::<C>(
            OpeningClaims::from_groups(
                self.groups
                    .iter()
                    .map(|group| {
                        PolynomialGroupClaims::new(
                            group.point.clone(),
                            vec![group.opening],
                            group.commitment.clone(),
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap(),
            self.groups
                .iter()
                .map(|group| group.handle.clone())
                .collect(),
            self.scheme.schedules(),
        )
        .unwrap()
    }

    pub(crate) fn selection(&self) -> OpeningScheduleSelection {
        self.opening_data().selection()
    }

    /// The statement claims, in group order.
    pub(crate) fn openings(&self) -> Vec<F> {
        self.groups.iter().map(|group| group.opening).collect()
    }

    /// The verifier's statement: commitments, points and selection. Its
    /// evaluations are placeholders the masked verifier never reads.
    pub(crate) fn statement(&self) -> GroupBatchStatement<'_, F, F> {
        GroupBatchStatement::new(
            self.selection(),
            OpeningClaims::from_groups(
                self.groups
                    .iter()
                    .map(|group| {
                        PolynomialGroupClaims::new(
                            group.point.clone(),
                            vec![F::zero()],
                            &group.commitment,
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    pub(crate) fn prefix_slots(&self) -> SetupPrefixProverRegistry<F, CommitmentHandle<F, F>> {
        let resolved = self
            .scheme
            .schedules()
            .resolve_selection(self.selection())
            .unwrap();
        let ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
            resolved.schedule(),
            self.opening_data().opening_layout(),
        )
        .unwrap();
        self.backend
            .import_setup_prefixes(&self.setup.prefix_slots, &ids)
            .unwrap()
    }

    /// The pad layout from the counting replay over `bytes`: the claims'
    /// slots, the prefix's message slots, then its auxiliary slots.
    pub(crate) fn layout(&self, bytes: &[u8]) -> PadLayout {
        self.layout_for(bytes, self.statement())
    }

    /// [`layout`](Self::layout) for another statement of this instance's
    /// shape.
    pub(crate) fn layout_for(
        &self,
        bytes: &[u8],
        statement: GroupBatchStatement<'_, F, F>,
    ) -> PadLayout {
        let mut transcript =
            VerifierTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION, bytes);
        let mut view = CommittedPadsView::counting();
        let claims = view
            .channel(&mut transcript)
            .exchange_private_base(&mut vec![F::zero(); self.groups.len()])
            .unwrap();
        let prefix = self
            .verifier
            .count_masked_prefix(
                &mut transcript,
                &mut view,
                statement,
                (claims.start..claims.start + claims.len)
                    .map(|slot| AffineRow::masked(F::zero(), PadIndex(slot)))
                    .collect(),
                BASIS,
            )
            .unwrap();
        let mut shape = OperandShapeDomain::new(view.used());
        prefix.shape(&mut shape).unwrap();
        shape.layout().unwrap()
    }

    /// Sends the masked claims, proves the masked prefix with `claims` as
    /// its statement claims (by default the claims just sent, each over its
    /// own pad), and sends the masked auxiliary block.
    pub(crate) fn prove(
        &self,
        pads: &mut CommittedPads<F, ()>,
        claims: impl FnOnce(Vec<F>) -> Vec<AffineRow<F>>,
    ) -> Result<Proved, AkitaError> {
        let mut transcript = ProverTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION);
        let (residual, auxiliary) = self.prove_on(pads, claims, &mut transcript)?;
        Ok((transcript.finish(), residual, auxiliary))
    }

    /// [`prove`](Self::prove) on `transcript`, which the caller continues.
    pub(crate) fn prove_on(
        &self,
        pads: &mut CommittedPads<F, ()>,
        claims: impl FnOnce(Vec<F>) -> Vec<AffineRow<F>>,
        transcript: &mut ProverTranscript<AkitaSponge>,
    ) -> Result<(ResidualOpening<F, F>, Vec<F>), AkitaError> {
        let (residual, auxiliary) = self.prove_prefix_on(pads, claims, transcript)?;
        self.send_auxiliary(pads, &auxiliary, transcript)?;
        Ok((residual, auxiliary))
    }

    /// The masked claims and the masked prefix, without the auxiliary
    /// block: the residual opening and Akita's auxiliary values.
    pub(crate) fn prove_prefix_on(
        &self,
        pads: &mut CommittedPads<F, ()>,
        claims: impl FnOnce(Vec<F>) -> Vec<AffineRow<F>>,
        transcript: &mut ProverTranscript<AkitaSponge>,
    ) -> Result<(ResidualOpening<F, F>, Vec<F>), AkitaError> {
        transcript.site(CLAIMS_SITE);
        let masked_claims = pads.send(&self.openings(), transcript)?;
        prove_masked_prefix::<C, _, _, _>(
            self.setup.expanded.descriptor(),
            self.scheme.schedules(),
            &self.backend,
            &self.prefix_slots(),
            self.opening_data(),
            &claims(masked_claims),
            pads,
            transcript,
            BASIS,
        )
    }

    /// Sends `auxiliary` as the masked auxiliary block, which ends the
    /// pad schedule.
    pub(crate) fn send_auxiliary(
        &self,
        pads: &mut CommittedPads<F, ()>,
        auxiliary: &[F],
        transcript: &mut ProverTranscript<AkitaSponge>,
    ) -> Result<(), AkitaError> {
        transcript.site(AUXILIARY_SITE);
        pads.send(auxiliary, transcript)?;
        pads.finish()?;
        Ok(())
    }

    /// Replays `proof` as the verifier: the masked claims, the masked
    /// prefix, and the masked auxiliary block.
    pub(crate) fn verify(
        &self,
        proof: &[u8],
        layout: &PadLayout,
    ) -> Result<(MaskedPrefix<F, F>, Vec<F>), AkitaError> {
        self.verify_for(proof, layout, self.statement())
    }

    /// [`verify`](Self::verify) against another statement of this
    /// instance's shape.
    pub(crate) fn verify_for(
        &self,
        proof: &[u8],
        layout: &PadLayout,
        statement: GroupBatchStatement<'_, F, F>,
    ) -> Result<(MaskedPrefix<F, F>, Vec<F>), AkitaError> {
        let mut transcript =
            VerifierTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION, proof);
        let mut view = CommittedPadsView::counting();
        let verified = self.verify_on(&mut transcript, &mut view, layout, statement)?;
        transcript.finish()?;
        assert_eq!(view.used(), layout.num_pads);
        Ok(verified)
    }

    /// [`verify_for`](Self::verify_for) on `transcript` and `view`, which the
    /// caller continues and finishes.
    pub(crate) fn verify_on(
        &self,
        transcript: &mut VerifierTranscript<'_, AkitaSponge>,
        view: &mut CommittedPadsView<()>,
        layout: &PadLayout,
        statement: GroupBatchStatement<'_, F, F>,
    ) -> Result<(MaskedPrefix<F, F>, Vec<F>), AkitaError> {
        let prefix = self.verify_prefix_on(transcript, view, statement)?;
        let mut auxiliary = vec![F::zero(); layout.auxiliary.len()];
        transcript.site(AUXILIARY_SITE);
        view.channel(transcript)
            .exchange_private_base(&mut auxiliary)?;
        Ok((prefix, auxiliary))
    }

    /// The masked claims and the masked prefix, without the auxiliary block.
    pub(crate) fn verify_prefix_on(
        &self,
        transcript: &mut VerifierTranscript<'_, AkitaSponge>,
        view: &mut CommittedPadsView<()>,
        statement: GroupBatchStatement<'_, F, F>,
    ) -> Result<MaskedPrefix<F, F>, AkitaError> {
        let mut claims = vec![F::zero(); self.groups.len()];
        transcript.site(CLAIMS_SITE);
        let slots = view
            .channel(transcript)
            .exchange_private_base(&mut claims)?;
        self.verifier.verify_masked_prefix(
            transcript,
            view,
            statement,
            claims
                .iter()
                .zip(slots.start..)
                .map(|(&claim, slot)| AffineRow::masked(claim, PadIndex(slot)))
                .collect(),
            BASIS,
        )
    }
}

/// Each claim over its own pad, in the order the claims were sent.
pub(crate) fn own_pads(claims: Vec<F>) -> Vec<AffineRow<F>> {
    claims
        .into_iter()
        .enumerate()
        .map(|(slot, claim)| AffineRow::masked(claim, PadIndex(slot)))
        .collect()
}

/// The recursive one-hot family offloads level 0's setup to a level-1 setup
/// prefix from 31 variables on (W2b).
pub(crate) type RecCfg = ZkCommitmentConfig<akita_config::RecursiveCommitmentConfig<fp128::OneHot>>;
pub(crate) const MULTI_NV: usize = 31;

impl Instance<RecCfg> {
    /// A one-hot `MULTI_NV`-variable root beside one precommitted dense
    /// polynomial committed under [`dense_zk_row`]'s hiding profile, as Jolt
    /// commits untrusted advice. The planner's row offloads level 0's setup
    /// to a setup prefix at the cut level.
    pub(crate) fn multi_group() -> Self {
        Self::multi_group_seeded(7)
    }

    /// [`multi_group`](Self::multi_group), committing both groups with
    /// hiding randomness from `seed`.
    pub(crate) fn multi_group_seeded(seed: u64) -> Self {
        let precommitted = dense_zk_row().0.final_group;
        let key = ScheduleLookupKey {
            final_group: PolynomialGroupLayout::singleton(MULTI_NV),
            precommitteds: vec![precommitted],
        };
        let scheme = AkitaCommitmentScheme::<RecCfg>::new(
            akita_config::test_support::workspace_schedule_catalog::<RecCfg>(
                ChunkedWitnessCfg::default_non_chunked(),
            )
            .unwrap(),
        );
        let schedule = scheme.schedules().resolve_key(&key).unwrap().schedule();
        assert_eq!(schedule.root.params.precommitted_groups().len(), 1);
        assert!(schedule.recursive_folds[ZK_CUT_LEVEL as usize - 1]
            .params
            .setup_prefix()
            .is_some());
        let setup = scheme.setup_prover(MULTI_NV, 2).unwrap();
        let backend = CpuBackend::new(setup.expanded.clone())
            .unwrap()
            .with_hiding_rng(StdRng::seed_from_u64(seed));
        let point = common::random_point(MULTI_NV, 0x2b2b);

        let (pre_evals, pre_poly) = dense_poly(NV);
        let pre = backend
            .commit(
                &zk_catalog(),
                &backend.import_source(vec![pre_poly]).unwrap(),
                GroupContext::explicit(&precommitted),
            )
            .unwrap();
        let final_poly = common::make_onehot_poly::<RecCfg>(MULTI_NV, 0x2b2b);
        let final_output = backend
            .commit(
                scheme.schedules(),
                &backend.import_source(vec![final_poly.clone()]).unwrap(),
                GroupContext::scheduler_with_precommitted_groups(
                    &akita_params::PrecommittedGroupProfiles::from_profiles(vec![precommitted])
                        .unwrap(),
                ),
            )
            .unwrap();
        let groups = vec![
            Group {
                commitment: pre.committed_group,
                handle: pre.private_handle,
                point: point[..NV].to_vec(),
                opening: lagrange_opening(&pre_evals, &point[..NV]),
            },
            Group {
                commitment: final_output.committed_group,
                handle: final_output.private_handle,
                opening: common::onehot_opening_lagrange(&final_poly, &point),
                point,
            },
        ];
        Self::with_groups(scheme, setup, backend, groups)
    }
}

pub(crate) fn random_pads(layout: &PadLayout, seed: u64) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..layout.num_pads)
        .map(|_| F::from_u64(rng.gen()) * F::from_u64(rng.gen()) + F::from_u64(rng.gen()))
        .collect()
}

/// The auxiliary values `pads` imply: the products of the decoded messages,
/// as a prover computes them from its own (possibly shifted) messages.
pub(crate) fn implied_auxiliary(prefix: &MaskedPrefix<F, F>, pads: &[F]) -> Vec<F> {
    let mut products = RecomputedProducts::default();
    prefix
        .lower(&mut products, |form| form.evaluate(pads, &[]))
        .unwrap();
    products.auxiliary
}

/// Sets the auxiliary slots' pads so that the masked auxiliary block decodes
/// to the products the other pads imply: a prover that shifts a decoded
/// message also recomputes its products consistently.
pub(crate) fn consistent_auxiliary_pads(
    prefix: &MaskedPrefix<F, F>,
    layout: &PadLayout,
    masked_auxiliary: &[F],
    pads: &mut [F],
) {
    let auxiliary = implied_auxiliary(prefix, pads);
    assert_eq!(auxiliary.len(), layout.auxiliary.len());
    for ((pad, masked), value) in pads[layout.auxiliary.clone()]
        .iter_mut()
        .zip(masked_auxiliary)
        .zip(&auxiliary)
    {
        *pad = *masked - *value;
    }
}

/// Recomputes the auxiliary values of a lowering without checking anything,
/// so a shifted message still yields the products a consistent prover would
/// send.
#[derive(Default)]
pub(crate) struct RecomputedProducts {
    pub(crate) auxiliary: Vec<F>,
}

impl ClaimDomain<F> for RecomputedProducts {
    type Value = F;
    type Error = ZkError;

    fn constant(&self, value: F) -> F {
        value
    }

    fn add(&self, lhs: &F, rhs: &F) -> F {
        *lhs + *rhs
    }

    fn scale(&self, value: &F, scale: F) -> F {
        *value * scale
    }

    fn define_auxiliary(&mut self, value: F) -> Result<F, ZkError> {
        self.auxiliary.push(value);
        Ok(value)
    }

    fn product(&mut self, x: F, y: F) -> Result<F, ZkError> {
        self.auxiliary.push(x * y);
        Ok(x * y)
    }

    fn products_into(&mut self, _: (F, F), _: Vec<(F, F)>, _: F) -> Result<(), ZkError> {
        Ok(())
    }

    fn check_zero(&mut self, _: F) -> Result<(), ZkError> {
        Ok(())
    }
}
