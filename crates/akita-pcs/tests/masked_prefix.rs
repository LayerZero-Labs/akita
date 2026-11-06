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
#[path = "dynamic_backends/private_cpu.rs"]
mod private_cpu;

#[path = "masked_prefix/instance.rs"]
mod instance;

use akita_config::CommitmentConfig;
use akita_cpu_backend::{CpuBackend, GroupContext};
use akita_error::AkitaError;
use akita_params::zk::ZK_CUT_LEVEL;
use akita_params::ChunkedWitnessCfg;
use akita_pcs::zk::{BackendCompression, CompressionShape};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{prove_masked_prefix, ResidualOpening, SelectedProverOpeningData};
use akita_transcript::{new_prover_channel, new_verifier_channel, send_field_group};
use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use akita_verifier::MaskedPrefix;
use instance::*;
use jolt_algebra::{CanonicalBytes, CanonicalEncoding, One, Ring, Zero};
use jolt_zk::masking::{
    AffineRow, CommittedPads, CommittedPadsView, PadIndex, PadLayout, ResidualRelation,
    ResidualRows,
};
use jolt_zk::whiteout::ImageCompression;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
        consistent_auxiliary_pads(
            &self.prefix,
            &self.layout,
            &self.masked_auxiliary,
            &mut pads,
        );
        self.relation
            .first_violation(&pads, &self.witness())
            .unwrap()
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
        // The cut sends cm(u_a) (A47): the compression of the u_a the
        // residual opens.
        assert_eq!(claim.commitment(), honest.residual.residual_payload);
        let compressed = BackendCompression::<Cfg>::new(&honest.instance.backend)
            .compress(&honest.residual.residual_commitment)
            .unwrap();
        assert_eq!(claim.commitment(), compressed.payload);
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
/// beside its own, and a masked level 1 that consumes a setup prefix (so
/// level 0 runs Stage 3). Every root group's claim is a masked statement claim; the
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
            common::load_workspace_scheme::<Clear>(ChunkedWitnessCfg::default_non_chunked())
                .unwrap(),
        );
        let mut pads =
            CommittedPads::uncommitted(honest.layout.clone(), honest.pads.clone()).unwrap();
        let error = clear.prove(&mut pads, own_pads).err().unwrap();
        assert!(
            matches!(error, AkitaError::UnsupportedSchedule(_)),
            "{error}"
        );
        let zeros = vec![0; COUNTING_BYTES];
        let mut transcript = new_verifier_channel(SESSION, &zeros);
        let error = clear
            .verifier
            .count_masked_prefix(
                &mut transcript,
                &mut CommittedPadsView::counting(),
                clear.statement(),
                vec![AffineRow::masked(F::zero(), PadIndex(0))],
                &CompressionShape::<Clear>::default(),
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
            None,
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

/// Only the Stage-3 output check can reject: the backend adds `X − X²` to the
/// last Stage-3 round, so every round sums correctly, the prover's checks
/// pass, and the setup-prefix evaluation and every later message stay honest
/// at the drawn point. The masked verifier, which reads Stage 3 in the clear,
/// rejects the proof; the same backend without the fault proves a prefix it
/// accepts.
#[test]
fn a_stage3_output_miss_is_rejected_at_a_masked_level() {
    on_large_stack(|| {
        use private_cpu::{Fault, PrivateCpu};

        let instance = Instance::multi_group();
        let private = PrivateCpu::new(
            CpuBackend::new(instance.setup.expanded.clone())
                .unwrap()
                .with_hiding_rng(StdRng::seed_from_u64(9)),
        );
        let precommitted = *instance.groups[0].commitment.profile();
        let (_, pre_poly) = dense_poly(NV);
        let (pre, pre_handle) = private
            .commit_in(
                &zk_catalog(),
                vec![pre_poly],
                GroupContext::explicit(&precommitted),
            )
            .unwrap();
        let final_poly = common::make_onehot_poly::<RecCfg>(MULTI_NV, 0x2b2b);
        let (root, root_handle) = private
            .commit_in(
                instance.scheme.schedules(),
                vec![final_poly],
                GroupContext::scheduler_with_precommitted_groups(
                    &akita_params::PrecommittedGroupProfiles::from_profiles(vec![precommitted])
                        .unwrap(),
                ),
            )
            .unwrap();
        let commitments = [&pre, &root];
        let claims = |evaluations: [F; 2]| {
            OpeningClaims::from_groups(
                instance
                    .groups
                    .iter()
                    .zip(commitments)
                    .zip(evaluations)
                    .map(|((group, commitment), evaluation)| {
                        PolynomialGroupClaims::new(
                            group.point.clone(),
                            vec![evaluation],
                            commitment.clone(),
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap()
        };
        let openings = [instance.groups[0].opening, instance.groups[1].opening];
        let opening_data = || {
            SelectedProverOpeningData::from_committed_claims::<RecCfg>(
                claims(openings),
                vec![pre_handle.clone(), root_handle.clone()],
                instance.scheme.schedules(),
            )
            .unwrap()
        };
        let statement = || {
            GroupBatchStatement::new(
                opening_data().selection(),
                OpeningClaims::from_groups(
                    instance
                        .groups
                        .iter()
                        .zip(commitments)
                        .map(|(group, commitment)| {
                            PolynomialGroupClaims::new(
                                group.point.clone(),
                                vec![F::zero()],
                                commitment,
                            )
                            .unwrap()
                        })
                        .collect(),
                )
                .unwrap(),
            )
            .unwrap()
        };
        let resolved = instance
            .scheme
            .schedules()
            .resolve_selection(opening_data().selection())
            .unwrap();
        let prefix_ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
            resolved.schedule(),
            opening_data().opening_layout(),
        )
        .unwrap();
        let prefixes = private.prefixes(&instance.setup, &prefix_ids);
        let layout = instance.layout_for(&vec![0; COUNTING_BYTES], statement());
        let pads = random_pads(&layout, 13);
        let prove = || {
            let mut pads = CommittedPads::uncommitted(layout.clone(), pads.clone()).unwrap();
            let mut transcript = new_prover_channel(SESSION);
            let (_, masked) = pads.mask(&openings).unwrap();
            send_field_group(&mut transcript, CLAIMS_SITE, &masked).unwrap();
            let (_, auxiliary) = prove_masked_prefix::<RecCfg, _, _>(
                instance.setup.expanded.descriptor(),
                instance.scheme.schedules(),
                &private,
                &prefixes,
                opening_data(),
                &own_pads(masked),
                &mut pads,
                &mut transcript,
                &BackendCompression::<RecCfg>::new(&instance.backend),
                BASIS,
            )
            .unwrap();
            let (_, masked) = pads.mask(&auxiliary).unwrap();
            send_field_group(&mut transcript, AUXILIARY_SITE, &masked).unwrap();
            pads.finish().unwrap();
            transcript.narg_string().to_vec()
        };

        let honest = prove();
        assert!(instance.verify_for(&honest, &layout, statement()).is_ok());
        private.fault.set(Fault::Stage3OutputMiss);
        let cheating = prove();
        assert_eq!(
            instance.verify_for(&cheating, &layout, statement()).err(),
            Some(AkitaError::InvalidProof)
        );
    });
}
