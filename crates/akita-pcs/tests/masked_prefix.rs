//! The masked prefix of a zero-knowledge opening (Whiteout W3,
//! `specs/whiteout-akita.md` §4.4–4.7).
//!
//! The planner's test-only zero-knowledge row masks levels `0..=1`. Its
//! statement claim is itself a masked message here, sent before Akita, as
//! W6's stage-8 claims are. The prover runs both masked levels over
//! committed pads and stops at the cut; the verifier replays them over the
//! same masked channel and lowers the stage checks into residual rows, which
//! jolt-zk's row checker evaluates on the pads and the residual witness.

#![allow(missing_docs)]

mod common;

use akita_config::{policy_of, proof_optimized::fp128, CommitmentConfig, TrustedScheduleCatalog};
use akita_cpu_backend::{AkitaProverSetup, CommitmentHandle, CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_params::zk::{ZkOpeningSchedule, ZK_CUT_LEVEL};
use akita_params::{
    lagrange_weights, BasisMode, ChunkedWitnessCfg, CommittedGroupBatchProfile,
    GroupCommitPhaseParams, OpeningScheduleSelection, PolynomialGroupLayout, ScheduleLookupKey,
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
use jolt_algebra::{CanonicalBytes, CanonicalDecode, One, Ring, Zero};
use jolt_transcript::{ClaimDomain, PrivateChannel, ProverTranscript, VerifierTranscript};
use jolt_zk::masking::{
    AffineRow, CommittedPads, CommittedPadsView, OperandShapeDomain, PadIndex, PadLayout,
    ResidualRelation, ResidualRows, ZkError,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type Cfg = fp128::Dense;
type F = <Cfg as CommitmentConfig>::Field;

const NV: usize = 14;
const SESSION: &[u8] = b"masked-prefix";
const BASIS: BasisMode = BasisMode::Lagrange;

/// A masked prefix's proof bytes, residual opening and auxiliary values.
type Proved = (Vec<u8>, ResidualOpening<F, F>, Vec<F>);

/// Generous room for every message of the prefix, for the counting replay.
const COUNTING_BYTES: usize = 1 << 22;

fn on_large_stack(test: fn()) {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(test)
        .unwrap()
        .join()
        .unwrap();
}

/// The planner's zero-knowledge row for one dense `NV`-variable polynomial.
fn dense_zk_row() -> (CommittedGroupBatchProfile, akita_params::FoldSchedule) {
    let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(NV));
    let planned = akita_planner::find_schedule_for_test_zk(
        &key,
        Cfg::committed_source_contract().unwrap(),
        &[],
        &policy_of::<Cfg>(ChunkedWitnessCfg::default_non_chunked()),
        Cfg::ring_challenge_config,
        ZkOpeningSchedule::new(ZK_CUT_LEVEL).unwrap(),
    )
    .unwrap();
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &planned.schedule.root.params,
        )
        .unwrap(),
        precommitteds: Vec::new(),
    };
    (profiles, planned.schedule)
}

fn zk_catalog() -> TrustedScheduleCatalog<Cfg> {
    TrustedScheduleCatalog::try_new_with_test_zk(
        [dense_zk_row()],
        ChunkedWitnessCfg::default_non_chunked(),
    )
    .unwrap()
}

/// One committed group of one polynomial and its claim.
struct Group {
    commitment: CommittedGroup<F>,
    handle: CommitmentHandle<F, F>,
    point: Vec<F>,
    opening: F,
}

/// Committed groups and their claims, with both roles' setups.
struct Instance<C: CommitmentConfig<Field = F, ExtField = F> = Cfg> {
    scheme: AkitaCommitmentScheme<C>,
    setup: AkitaProverSetup<F>,
    backend: CpuBackend<F, F>,
    verifier: AkitaVerifier<C>,
    groups: Vec<Group>,
}

fn dense_poly(num_vars: usize) -> (Vec<F>, DensePoly<F>) {
    let evals: Vec<F> = (0..(1 << num_vars))
        .map(|index| F::from_u64(3 * index as u64 + 1))
        .collect();
    let poly = DensePoly::<F>::from_field_evals(num_vars, &evals).unwrap();
    (evals, poly)
}

fn lagrange_opening(evals: &[F], point: &[F]) -> F {
    evals
        .iter()
        .zip(lagrange_weights(point).unwrap())
        .fold(F::zero(), |sum, (&value, weight)| sum + value * weight)
}

impl Instance<Cfg> {
    fn new(scheme: AkitaCommitmentScheme<Cfg>) -> Self {
        let setup = scheme.setup_prover(NV, 1).unwrap();
        let backend = CpuBackend::new(setup.expanded.clone())
            .unwrap()
            .with_hiding_rng(StdRng::seed_from_u64(7));
        let (evals, poly) = dense_poly(NV);
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
    fn with_groups(
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

    fn opening_data(&self) -> SelectedProverOpeningData<'static, F, CommitmentHandle<F, F>, F> {
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

    fn selection(&self) -> OpeningScheduleSelection {
        self.opening_data().selection()
    }

    /// The statement claims, in group order.
    fn openings(&self) -> Vec<F> {
        self.groups.iter().map(|group| group.opening).collect()
    }

    /// The verifier's statement: commitments, points and selection. Its
    /// evaluations are placeholders the masked verifier never reads.
    fn statement(&self) -> GroupBatchStatement<'_, F, F> {
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

    fn prefix_slots(&self) -> SetupPrefixProverRegistry<F, CommitmentHandle<F, F>> {
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
    fn layout(&self, bytes: &[u8]) -> PadLayout {
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
                self.statement(),
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
    fn prove(
        &self,
        pads: &mut CommittedPads<F, ()>,
        claims: impl FnOnce(Vec<F>) -> Vec<AffineRow<F>>,
    ) -> Result<Proved, AkitaError> {
        let mut transcript = ProverTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION);
        let masked_claims = pads.send(&self.openings(), &mut transcript)?;
        let (residual, auxiliary) = prove_masked_prefix::<C, _, _, _>(
            self.setup.expanded.descriptor(),
            self.scheme.schedules(),
            &self.backend,
            &self.prefix_slots(),
            self.opening_data(),
            &claims(masked_claims),
            pads,
            &mut transcript,
            BASIS,
        )?;
        pads.send(&auxiliary, &mut transcript)?;
        pads.finish()?;
        Ok((transcript.finish(), residual, auxiliary))
    }

    /// Replays `proof` as the verifier: the masked claims, the masked
    /// prefix, and the masked auxiliary block.
    fn verify(
        &self,
        proof: &[u8],
        layout: &PadLayout,
    ) -> Result<(MaskedPrefix<F, F>, Vec<F>), AkitaError> {
        let mut transcript =
            VerifierTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION, proof);
        let mut view = CommittedPadsView::counting();
        let mut claims = vec![F::zero(); self.groups.len()];
        let slots = view
            .channel(&mut transcript)
            .exchange_private_base(&mut claims)?;
        let prefix = self.verifier.verify_masked_prefix(
            &mut transcript,
            &mut view,
            self.statement(),
            claims
                .iter()
                .zip(slots.start..)
                .map(|(&claim, slot)| AffineRow::masked(claim, PadIndex(slot)))
                .collect(),
            BASIS,
        )?;
        let mut auxiliary = vec![F::zero(); layout.auxiliary.len()];
        view.channel(&mut transcript)
            .exchange_private_base(&mut auxiliary)?;
        transcript.finish()?;
        assert_eq!(view.used(), layout.num_pads);
        Ok((prefix, auxiliary))
    }
}

/// Each claim over its own pad, in the order the claims were sent.
fn own_pads(claims: Vec<F>) -> Vec<AffineRow<F>> {
    claims
        .into_iter()
        .enumerate()
        .map(|(slot, claim)| AffineRow::masked(claim, PadIndex(slot)))
        .collect()
}

/// The recursive one-hot family offloads level 0's setup to a level-1 setup
/// prefix from 31 variables on (W2b).
type RecCfg = akita_config::RecursiveCommitmentConfig<fp128::OneHot>;
const MULTI_NV: usize = 31;

impl Instance<RecCfg> {
    /// A one-hot `MULTI_NV`-variable root beside one precommitted dense
    /// polynomial committed under [`dense_zk_row`]'s hiding profile, as Jolt
    /// commits untrusted advice. The planner's row offloads level 0's setup
    /// to a setup prefix at the cut level.
    fn multi_group() -> Self {
        let precommitted = dense_zk_row().0.final_group;
        let key = ScheduleLookupKey {
            final_group: PolynomialGroupLayout::singleton(MULTI_NV),
            precommitteds: vec![precommitted],
        };
        let planned = akita_planner::find_schedule_for_test_zk(
            &key,
            RecCfg::committed_source_contract().unwrap(),
            &[Cfg::committed_source_contract().unwrap()],
            &policy_of::<RecCfg>(ChunkedWitnessCfg::default_non_chunked()),
            RecCfg::ring_challenge_config,
            ZkOpeningSchedule::new(ZK_CUT_LEVEL).unwrap(),
        )
        .unwrap();
        let schedule = &planned.schedule;
        assert_eq!(schedule.root.params.precommitted_groups().len(), 1);
        assert!(schedule.recursive_folds[ZK_CUT_LEVEL as usize - 1]
            .params
            .setup_prefix()
            .is_some());
        let profiles = CommittedGroupBatchProfile {
            final_group: GroupCommitPhaseParams::try_from_params(
                key.final_group,
                &schedule.root.params,
            )
            .unwrap(),
            precommitteds: vec![precommitted],
        };
        let scheme = AkitaCommitmentScheme::<RecCfg>::new(
            TrustedScheduleCatalog::try_new_with_test_zk(
                [(profiles, planned.schedule)],
                ChunkedWitnessCfg::default_non_chunked(),
            )
            .unwrap(),
        );
        let setup = scheme.setup_prover(MULTI_NV, 2).unwrap();
        let backend = CpuBackend::new(setup.expanded.clone())
            .unwrap()
            .with_hiding_rng(StdRng::seed_from_u64(7));
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

fn random_pads(layout: &PadLayout, seed: u64) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..layout.num_pads)
        .map(|_| F::from_u64(rng.gen()) * F::from_u64(rng.gen()) + F::from_u64(rng.gen()))
        .collect()
}

/// An honest masked prefix and its residual rows.
struct Honest<C: CommitmentConfig<Field = F, ExtField = F> = Cfg> {
    instance: Instance<C>,
    proof: Vec<u8>,
    layout: PadLayout,
    pads: Vec<F>,
    residual: ResidualOpening<F, F>,
    prover_auxiliary: Vec<F>,
    prefix: MaskedPrefix<F, F>,
    masked_auxiliary: Vec<F>,
    relation: ResidualRelation<F>,
}

impl Honest<Cfg> {
    fn new() -> Self {
        Self::of(Instance::new(AkitaCommitmentScheme::new(zk_catalog())))
    }
}

impl<C: CommitmentConfig<Field = F, ExtField = F>> Honest<C> {
    fn of(instance: Instance<C>) -> Self {
        let layout = instance.layout(&vec![0; COUNTING_BYTES]);
        let pads = random_pads(&layout, 11);
        let mut committed = CommittedPads::uncommitted(layout.clone(), pads.clone()).unwrap();
        let (proof, residual, prover_auxiliary) = instance.prove(&mut committed, own_pads).unwrap();
        let (prefix, masked_auxiliary) = instance.verify(&proof, &layout).unwrap();
        let mut relation = ResidualRelation::new(layout.clone(), 1);
        prefix
            .rows(&masked_auxiliary)
            .append_rows(&mut relation)
            .unwrap();
        Self {
            instance,
            proof,
            layout,
            pads,
            residual,
            prover_auxiliary,
            prefix,
            masked_auxiliary,
            relation,
        }
    }

    fn witness(&self) -> [F; 1] {
        [self.residual.witness_eval]
    }

    /// The first row `pads` violate, with the auxiliary slots' pads set so
    /// that they decode to the auxiliary values `pads` imply: a prover that
    /// shifts a message also recomputes its products.
    fn first_violation_recomputed(&self, mut pads: Vec<F>) -> Option<usize> {
        let mut products = RecomputedProducts::default();
        self.prefix
            .lower(&mut products, |form| form.evaluate(&pads, &[]))
            .unwrap();
        assert_eq!(products.auxiliary.len(), self.layout.auxiliary.len());
        for ((pad, masked), value) in pads[self.layout.auxiliary.clone()]
            .iter_mut()
            .zip(&self.masked_auxiliary)
            .zip(&products.auxiliary)
        {
            *pad = *masked - *value;
        }
        self.relation
            .first_violation(&pads, &self.witness())
            .unwrap()
    }
}

/// Recomputes the auxiliary values of a lowering without checking anything,
/// so a shifted message still yields the products a consistent prover would
/// send.
#[derive(Default)]
struct RecomputedProducts {
    auxiliary: Vec<F>,
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

/// W3-a: the honest decoded transcript, pads and residual witness satisfy
/// every row, and the rows are the paper's: three products and one auxiliary
/// slot per masked level, and the cut row.
#[test]
fn honest_masked_prefix_satisfies_its_residual_rows() {
    on_large_stack(|| {
        let honest = Honest::new();
        let masked_levels = ZK_CUT_LEVEL as usize + 1;
        assert_eq!(
            honest
                .relation
                .first_violation(&honest.pads, &honest.witness())
                .unwrap(),
            None
        );
        assert_eq!(
            honest
                .relation
                .products
                .iter()
                .map(|row| row.pairs().count())
                .sum::<usize>(),
            3 * masked_levels
        );
        assert_eq!(honest.layout.auxiliary.len(), masked_levels);
        assert_eq!(honest.relation.affine.len(), 1, "the cut row");
        assert_eq!(honest.layout.operand_pads.len(), 3 * masked_levels);
        assert_eq!(
            honest.prefix.auxiliary_values(&honest.pads).unwrap(),
            honest.prover_auxiliary
        );
        let claim = honest.prefix.residual_claim();
        assert_eq!(claim.cut_level(), ZK_CUT_LEVEL);
        assert_eq!(claim.commitment(), honest.residual.residual_commitment);
        assert_eq!(
            claim.witness_eval().evaluate(&honest.pads, &[]).unwrap(),
            honest.residual.witness_eval
        );
    });
}

/// W3-b and W3-c: shifting any one decoded message by one, with the
/// auxiliary values recomputed consistently, breaks a row; so does shifting
/// any auxiliary value alone. This covers every masked message class of the
/// prefix: the statement claim, both stages' rounds, the range images and the
/// witness evaluations of both levels.
#[test]
fn every_shifted_message_breaks_a_row() {
    on_large_stack(|| assert_every_shifted_message_breaks_a_row(&Honest::new()));
}

fn assert_every_shifted_message_breaks_a_row<C: CommitmentConfig<Field = F, ExtField = F>>(
    honest: &Honest<C>,
) {
    assert_eq!(honest.first_violation_recomputed(honest.pads.clone()), None);
    for slot in 0..honest.layout.auxiliary.start {
        let mut pads = honest.pads.clone();
        pads[slot] -= F::one();
        assert!(
            honest.first_violation_recomputed(pads).is_some(),
            "shifting message slot {slot} breaks no row"
        );
    }
    for slot in honest.layout.auxiliary.clone() {
        let mut pads = honest.pads.clone();
        pads[slot] -= F::one();
        assert!(
            honest
                .relation
                .first_violation(&pads, &honest.witness())
                .unwrap()
                .is_some(),
            "shifting auxiliary slot {slot} breaks no row"
        );
    }
    let wrong_witness = [honest.residual.witness_eval + F::one()];
    assert!(honest
        .relation
        .first_violation(&honest.pads, &wrong_witness)
        .unwrap()
        .is_some());
}

/// A masked message altered in the proof reaches the rows: the cut level's
/// witness evaluation, the last message before the auxiliary block, plus one
/// in the proof bytes leaves the honest pads violating a row.
#[test]
fn a_tampered_witness_evaluation_breaks_a_row() {
    on_large_stack(|| {
        let honest = Honest::new();
        let element = <F as CanonicalBytes>::NUM_BYTES;
        let offset = honest.proof.len() - (honest.layout.auxiliary.len() + 1) * element;
        let mut proof = honest.proof.clone();
        let mut masked = F::from_bytes_le_checked(&proof[offset..offset + element]).unwrap();
        masked += F::one();
        masked.to_bytes_le(&mut proof[offset..offset + element]);
        let (prefix, masked_auxiliary) = honest.instance.verify(&proof, &honest.layout).unwrap();
        assert_eq!(masked_auxiliary, honest.masked_auxiliary);
        let mut relation = ResidualRelation::new(honest.layout.clone(), 1);
        prefix
            .rows(&masked_auxiliary)
            .append_rows(&mut relation)
            .unwrap();
        assert!(relation
            .first_violation(&honest.pads, &honest.witness())
            .unwrap()
            .is_some());
    });
}

/// W3-d: the counting replay counts the honest schedule's pads and operands
/// on zero bytes and on random bytes, and the honest prover consumes exactly
/// that many.
#[test]
fn counting_replay_counts_the_honest_pads() {
    on_large_stack(|| assert_counting_replay_counts_the_honest_pads(&Honest::new()));
}

fn assert_counting_replay_counts_the_honest_pads<C: CommitmentConfig<Field = F, ExtField = F>>(
    honest: &Honest<C>,
) {
    let mut rng = StdRng::seed_from_u64(5);
    let random: Vec<u8> = (0..COUNTING_BYTES).map(|_| rng.gen()).collect();
    assert_eq!(honest.instance.layout(&random), honest.layout);
}

/// W2b on the masked path: a root that folds a hiding precommitted group
/// beside its own, and a cut level that consumes a setup prefix (so level 0
/// runs Stage 3). Every root group's claim is a masked statement claim; the
/// honest rows hold, every shifted message breaks one, the counting replay
/// counts the pads the prover consumes, and the masked claims of both groups
/// decode to the openings.
#[test]
fn multi_group_masked_prefix_with_a_setup_prefix() {
    on_large_stack(|| {
        let honest = Honest::of(Instance::multi_group());
        assert_eq!(
            honest
                .relation
                .first_violation(&honest.pads, &honest.witness())
                .unwrap(),
            None
        );
        assert_eq!(
            honest.prefix.auxiliary_values(&honest.pads).unwrap(),
            honest.prover_auxiliary
        );
        assert_every_shifted_message_breaks_a_row(&honest);
        assert_counting_replay_counts_the_honest_pads(&honest);
        // A statement claim of either group that decodes to a wrong value
        // makes the prover fail before any message.
        for group in 0..2 {
            let mut pads =
                CommittedPads::uncommitted(honest.layout.clone(), honest.pads.clone()).unwrap();
            let error = honest
                .instance
                .prove(&mut pads, |claims| {
                    let shifted = claims[group] + F::one();
                    let mut rows = own_pads(claims);
                    rows[group] = AffineRow::masked(shifted, PadIndex(group));
                    rows
                })
                .err()
                .unwrap();
            assert!(matches!(error, AkitaError::InvalidInput(_)), "{error}");
        }
    });
}

/// A wrong masked statement claim makes the prover fail before any message,
/// and a schedule without a cut is rejected by both roles' masked entries.
#[test]
fn masked_entries_reject_wrong_claims_and_clear_schedules() {
    on_large_stack(|| {
        let honest = Honest::new();
        let mut pads =
            CommittedPads::uncommitted(honest.layout.clone(), honest.pads.clone()).unwrap();
        let error = honest
            .instance
            .prove(&mut pads, |claims| {
                vec![AffineRow::masked(claims[0] + F::one(), PadIndex(0))]
            })
            .err()
            .unwrap();
        assert!(matches!(error, AkitaError::InvalidInput(_)), "{error}");

        let clear = Instance::new(
            common::load_workspace_scheme::<Cfg>(ChunkedWitnessCfg::default_non_chunked()).unwrap(),
        );
        let mut pads =
            CommittedPads::uncommitted(honest.layout.clone(), honest.pads.clone()).unwrap();
        let error = clear.prove(&mut pads, own_pads).err().unwrap();
        assert!(
            matches!(error, AkitaError::UnsupportedSchedule(_)),
            "{error}"
        );
        let mut transcript = VerifierTranscript::<AkitaSponge>::new(
            &PROOF_STREAM_PROTOCOL,
            SESSION,
            &[0; COUNTING_BYTES],
        );
        let error = clear
            .verifier
            .count_masked_prefix(
                &mut transcript,
                &mut CommittedPadsView::counting(),
                clear.statement(),
                vec![AffineRow::masked(F::zero(), PadIndex(0))],
                BASIS,
            )
            .unwrap_err();
        assert!(
            matches!(error, AkitaError::UnsupportedSchedule(_)),
            "{error}"
        );
    });
}

/// The cut successor's parameters, which describe the residual commitment.
fn residual_params(instance: &Instance) -> akita_params::CommittedGroupParams {
    instance
        .scheme
        .schedules()
        .resolve_selection(instance.selection())
        .unwrap()
        .schedule()
        .recursive_folds[ZK_CUT_LEVEL as usize]
        .params
        .clone()
}

/// `Σ_i w_i · eq(i, point)` over w_k's digits, zero-padded, with the point's
/// first coordinate the most significant bit of `i`: the dense MLE, written
/// out from its definition.
fn dense_mle(digits: &[i8], point: &[F]) -> F {
    let num_vars = point.len();
    assert!(digits.len() <= 1 << num_vars);
    digits
        .iter()
        .enumerate()
        .map(|(index, &digit)| {
            let weight = point.iter().enumerate().fold(F::one(), |weight, (j, &r)| {
                let bit = (index >> (num_vars - 1 - j)) & 1 == 1;
                weight * if bit { r } else { F::one() - r }
            });
            F::from_i64(i64::from(digit)) * weight
        })
        .sum()
}

/// W3-e, `N·s = t`: the exported `(w_k, t̂_a, r_a)` open u_a. The products run
/// through jolt-crypto's Ajtai map on the setup stream (`MatrixView`,
/// `PreparedMatrix::apply_centered`), not Akita's commitment kernels, under
/// the column semantics of jolt-crypto's `TwoLevelLayout`.
fn check_residual_commitment<const D: usize>(
    stream: &[F],
    params: &akita_params::CommittedGroupParams,
    opening: &ResidualOpening<F, F>,
) {
    use jolt_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
    use jolt_crypto::lattice::{MatrixShape, MatrixView, PreparedMatrix};

    let profile = &params.own_group().profile;
    let n_a = profile.inner.matrix.output_rank();
    let positions = profile.blocks.positions_per_block;
    let blocks = profile.blocks.live_blocks;
    let (inner, outer) = (profile.inner.digits, profile.outer.digits);
    let a = PreparedMatrix::<F, D>::new(
        &MatrixView::new(
            MatrixShape {
                rows: n_a,
                width: positions * inner.num_digits,
                ring_dim: D,
            },
            stream,
        )
        .unwrap(),
    )
    .unwrap();
    let decompose = BalancedDecomposePow2Params::<F>::new(inner.num_digits, inner.log_basis);
    let mut t = Vec::new();
    for block in 0..blocks {
        let mut input = Vec::new();
        for position in 0..positions {
            let ring = (0..D)
                .map(|coefficient| {
                    opening
                        .w_k
                        .get((block * positions + position) * D + coefficient)
                        .map_or(F::zero(), |&digit| F::from_i64(i64::from(digit)))
                })
                .collect::<Vec<_>>();
            let mut digits = vec![0i8; inner.num_digits * D];
            jolt_algebra::balanced_decompose_coefficients_pow2_i8_into(
                &ring,
                &mut digits,
                &decompose,
            );
            input.extend(digits.iter().map(|&digit| i64::from(digit)));
        }
        t.extend(a.apply_centered(&input, 1 << 7).unwrap());
    }
    // G · t̂_a = t, [block][A row][digit][coefficient].
    assert_eq!(opening.t_hat_a.len(), t.len() * outer.num_digits * D);
    let base = F::from_u64(1 << outer.log_basis);
    for (index, ring) in t.iter().enumerate() {
        for coefficient in 0..D {
            let recombined = (0..outer.num_digits).rev().fold(F::zero(), |sum, digit| {
                sum * base
                    + F::from_i64(i64::from(
                        opening.t_hat_a[(index * outer.num_digits + digit) * D + coefficient],
                    ))
            });
            assert_eq!(
                recombined,
                ring.coefficients()[coefficient],
                "t row {index}"
            );
        }
    }
    // [B | B′](t̂_a, r_a) = u_a.
    assert_eq!(opening.r_a.len(), profile.randomness_columns * D);
    let b = PreparedMatrix::<F, D>::new(
        &MatrixView::new(
            MatrixShape {
                rows: profile.outer.matrix.output_rank(),
                width: opening.t_hat_a.len() / D + profile.randomness_columns,
                ring_dim: D,
            },
            stream,
        )
        .unwrap(),
    )
    .unwrap();
    let input = opening
        .t_hat_a
        .iter()
        .chain(opening.r_a.iter())
        .map(|&digit| i64::from(digit))
        .collect::<Vec<_>>();
    let u = b
        .apply_centered(&input, 1 << 7)
        .unwrap()
        .iter()
        .flat_map(|ring| ring.coefficients().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(u, opening.residual_commitment);
}

/// W3-e: the exported residual opening satisfies `N·s = t` for Akita's own
/// u_a, its r_a is drawn from the residual alphabet, and the dense
/// `w̃_k(point)` equals the decoded witness evaluation. The dense evaluation
/// is written from the MLE's definition with the first coordinate the most
/// significant index bit (A18); the reversed point gives another value, so
/// the order is pinned.
#[test]
fn residual_opening_opens_u_a_and_evaluates_to_w_eval() {
    on_large_stack(|| {
        let honest = Honest::new();
        let params = residual_params(&honest.instance);
        let opening = &honest.residual;
        assert_eq!(
            params.own_group().profile.randomness_columns,
            opening.r_a.len() / params.role_dims().d_b()
        );
        assert_eq!(
            opening.r_a_log_basis,
            akita_params::zk::RESIDUAL_RANDOMNESS_LOG_BASIS
        );
        assert!(opening.r_a.iter().all(|digit| (-8..8).contains(digit)));
        assert!(opening.r_a.iter().any(|&digit| digit != 0));
        assert!(opening.w_k.iter().all(|digit| (-4..4).contains(digit)));
        assert_eq!(opening.w_k_log_basis, 3);
        let stream = honest
            .instance
            .setup
            .expanded
            .shared_matrix
            .as_field_slice();
        match params.d_a() {
            64 => check_residual_commitment::<64>(stream, &params, opening),
            128 => check_residual_commitment::<128>(stream, &params, opening),
            256 => check_residual_commitment::<256>(stream, &params, opening),
            other => panic!("residual ring dimension {other}"),
        }

        let point = opening.point.as_slice();
        assert_eq!(dense_mle(&opening.w_k, point), opening.witness_eval);
        let reversed = point.iter().rev().copied().collect::<Vec<_>>();
        assert_ne!(dense_mle(&opening.w_k, &reversed), opening.witness_eval);
        assert_eq!(honest.prefix.residual_claim().point(), &opening.point);
    });
}

/// Two openings of one statement send different residual commitments: u_a
/// is hidden by fresh r_a.
#[test]
fn two_openings_of_one_statement_differ_in_u_a() {
    on_large_stack(|| {
        let honest = Honest::new();
        let mut pads =
            CommittedPads::uncommitted(honest.layout.clone(), honest.pads.clone()).unwrap();
        let Ok((_, second, _)) = honest.instance.prove(&mut pads, own_pads) else {
            panic!("second opening failed");
        };
        assert_ne!(second.r_a, honest.residual.r_a);
        assert_ne!(
            second.residual_commitment,
            honest.residual.residual_commitment
        );
    });
}

/// An early masked message altered in the proof: the masked statement claim,
/// absorbed before every challenge of the prefix. Every later challenge
/// changes, so the replay either rejects a proof-of-work nonce or yields rows
/// the honest pads violate.
#[test]
fn a_tampered_early_message_is_rejected_or_breaks_a_row() {
    on_large_stack(|| {
        let honest = Honest::new();
        let mut proof = honest.proof.clone();
        proof[0] ^= 1;
        match honest.instance.verify(&proof, &honest.layout) {
            Err(error) => assert!(matches!(error, AkitaError::InvalidProof), "{error}"),
            Ok((prefix, masked_auxiliary)) => {
                let mut relation = ResidualRelation::new(honest.layout.clone(), 1);
                prefix
                    .rows(&masked_auxiliary)
                    .append_rows(&mut relation)
                    .unwrap();
                assert!(relation
                    .first_violation(&honest.pads, &honest.witness())
                    .unwrap()
                    .is_some());
            }
        }
    });
}

/// A truncated masked round is a rejected proof, not an internal error: the
/// masked channel's transcript failure keeps its classification through the
/// sumcheck driver.
#[test]
fn a_truncated_masked_proof_is_an_invalid_proof() {
    on_large_stack(|| {
        let honest = Honest::new();
        let element = <F as CanonicalBytes>::NUM_BYTES;
        // Cut inside the prefix: past the claim, before the cut level's
        // Stage 2, so the replay runs out inside a masked round.
        let cut = honest.proof.len() - (honest.layout.auxiliary.len() + 1) * element - 2 * element;
        let error = honest
            .instance
            .verify(&honest.proof[..cut], &honest.layout)
            .err()
            .unwrap();
        assert_eq!(error, AkitaError::InvalidProof);
    });
}

/// The test schedule grinds at the masked levels, so the masked channels'
/// nonces and the counting replay's nonce skip run.
#[test]
fn the_masked_levels_grind() {
    on_large_stack(|| {
        let instance = Instance::new(AkitaCommitmentScheme::new(zk_catalog()));
        let resolved = instance
            .scheme
            .schedules()
            .resolve_selection(instance.selection())
            .unwrap();
        let plan = akita_config::derive_transcript_grinding_plan::<Cfg>(
            resolved.schedule(),
            instance.opening_data().opening_layout(),
        )
        .unwrap();
        for level in 0..=ZK_CUT_LEVEL {
            assert!(
                plan.runs()
                    .iter()
                    .any(|run| run.site().level() == level && run.grind_bits() > 0),
                "level {level} has no proof-of-work query"
            );
        }
    });
}
