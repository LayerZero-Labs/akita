use super::*;
use akita_field::{Fp32, FpExt2, LiftBase, NegOneNr};
use akita_transcript::AkitaTranscript;
#[cfg(feature = "zk")]
use akita_types::FlatDigitBlocks;
use akita_types::{AkitaSetupSeed, FlatMatrix};

type F = Fp32<251>;
type E = FpExt2<F, NegOneNr>;

fn empty_recursive_hint_cache() -> RecursiveCommitmentHintCache<F> {
    let hint = AkitaCommitmentHint::with_recomposed_inner_rows(
        Vec::new(),
        Vec::new(),
        #[cfg(feature = "zk")]
        vec![FlatDigitBlocks::empty()],
    );
    RecursiveCommitmentHintCache::from_typed::<1>(hint).expect("empty recomposed hint cache")
}

fn setup() -> AkitaExpandedSetup<F> {
    AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 3,
            max_num_batched_polys: 4,
            max_num_points: 2,
            gen_ring_dim: 1,
            max_setup_len: 1,
            #[cfg(feature = "zk")]
            max_zk_b_len: 1,
            #[cfg(feature = "zk")]
            max_zk_d_len: 1,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(vec![F::zero()], 1),
        #[cfg(feature = "zk")]
        FlatMatrix::from_flat_data(vec![F::zero()], 1),
        #[cfg(feature = "zk")]
        FlatMatrix::from_flat_data(vec![F::zero()], 1),
    )
}

#[test]
fn prover_claim_preparation_accepts_extension_points() {
    let point = [
        E::new(F::from_u64(1), F::from_u64(2)),
        E::new(F::from_u64(3), F::from_u64(4)),
    ];
    let polys = [10usize, 11usize];
    let commitment = RingCommitment::<F, 2>::default();
    #[cfg(feature = "zk")]
    let hint = AkitaCommitmentHint::with_recomposed_inner_rows(
        Vec::new(),
        Vec::new(),
        vec![FlatDigitBlocks::empty()],
    );
    #[cfg(not(feature = "zk"))]
    let hint = AkitaCommitmentHint::new(Vec::new());
    let claims = vec![(
        &point[..],
        crate::CommittedPolynomials {
            polynomials: &polys[..],
            commitment: &commitment,
            hint,
        },
    )];

    let prepared = prepare_batched_prove_inputs::<F, E, usize, 2>(&setup(), claims)
        .expect("extension-valued prover points should validate by shape");

    assert_eq!(prepared.opening_points, vec![&point[..]]);
    assert_eq!(prepared.incidence_summary.num_claims(), 2);
    assert_eq!(prepared.incidence_summary.num_points(), 1);
    assert_eq!(prepared.incidence_summary.num_public_rows(), 1);
    assert_eq!(prepared.incidence_summary.num_polys_per_point(), &[2]);
    assert_eq!(prepared.incidence_summary.claim_to_point(), &[0, 0]);
    assert_eq!(prepared.flat_polys, vec![&polys[0], &polys[1]]);
    assert_eq!(prepared.group_polys, vec![&polys[0], &polys[1]]);
}

#[test]
fn recursive_carried_opening_state_requires_common_padded_domain() {
    let witness = RecursiveWitnessFlat::from_i8_digits(vec![1, -1, 2, 0]);
    let base_claim = RecursiveCarriedOpening::recursive_witness(
        vec![F::from_u64(3), F::from_u64(5)],
        F::one(),
        4,
        #[cfg(feature = "zk")]
        F::one(),
    );
    let ok_state = RecursiveProverState {
        w: witness.clone(),
        logical_w: None,
        commitment: FlatRingVec::from_coeffs(vec![F::zero(); 1]),
        hint: empty_recursive_hint_cache(),
        log_basis: 2,
        sumcheck_challenges: vec![F::from_u64(3), F::from_u64(5)],
        opening: F::one(),
        carried_openings: vec![
            base_claim.clone(),
            RecursiveCarriedOpening {
                kind: CarriedOpeningKind::SetupPrefix,
                ..base_claim.clone()
            },
        ],
        extra_carried_sources: Vec::new(),
        #[cfg(feature = "zk")]
        zk_hiding: ZkHidingProverState::new(Vec::new()),
    };
    assert_eq!(ok_state.common_padded_len().unwrap(), 4);

    let bad_state = RecursiveProverState {
        w: witness,
        logical_w: None,
        commitment: FlatRingVec::from_coeffs(vec![F::zero(); 1]),
        hint: empty_recursive_hint_cache(),
        log_basis: 2,
        sumcheck_challenges: vec![F::from_u64(3), F::from_u64(5)],
        opening: F::one(),
        carried_openings: vec![
            base_claim.clone(),
            RecursiveCarriedOpening {
                padded_len: 8,
                ..base_claim
            },
        ],
        extra_carried_sources: Vec::new(),
        #[cfg(feature = "zk")]
        zk_hiding: ZkHidingProverState::new(Vec::new()),
    };
    assert!(bad_state.common_padded_len().is_err());
}

#[cfg(not(feature = "zk"))]
#[test]
fn folded_root_assembly_preserves_extra_carried_openings() {
    let setup_commitment = FlatRingVec::from_coeffs(vec![F::from_u64(9)]);
    let source = CarriedOpeningSourceProof {
        commitment: setup_commitment,
    };
    let extra = CarriedOpeningProof {
        source_idx: 1,
        point: vec![F::from_u64(3), F::from_u64(5)],
        value: F::from_u64(7),
        basis: BasisMode::Lagrange,
        natural_len: 4,
        padded_len: 4,
        kind: CarriedOpeningKind::SetupPrefix,
    };
    let next_state = RecursiveProverState {
        w: RecursiveWitnessFlat::from_i8_digits(vec![1, 0, -1, 2]),
        logical_w: None,
        commitment: FlatRingVec::from_coeffs(vec![F::zero(); 1]),
        hint: empty_recursive_hint_cache(),
        log_basis: 2,
        sumcheck_challenges: vec![F::from_u64(1), F::from_u64(2)],
        opening: F::from_u64(4),
        carried_openings: vec![RecursiveCarriedOpening::recursive_witness(
            vec![F::from_u64(1), F::from_u64(2)],
            F::from_u64(4),
            4,
            #[cfg(feature = "zk")]
            F::from_u64(4),
        )],
        extra_carried_sources: Vec::new(),
    };
    let root = RootLevelProverOutput::<F, F, 1> {
        raw: RootLevelRawOutput::<F, F, 1> {
            y_rings: vec![CyclotomicRing::<F, 1>::zero()],
            extension_opening_reduction: None,
            v: Vec::new(),
            stage1: AkitaStage1Proof {
                stages: Vec::new(),
                s_claim: F::zero(),
            },
            stage2_sumcheck_proof: SumcheckProof {
                round_polys: Vec::new(),
            },
            stage3_sumcheck_proof: None,
            w_commitment_proof: FlatRingVec::from_coeffs(vec![F::zero(); 1]),
            w_eval: F::from_u64(4),
        },
        extra_carried_sources: vec![source.clone()],
        extra_carried_openings: vec![extra.clone()],
        next_state,
    };

    let (proof, num_levels) = build_folded_batched_proof_with_suffix(root, |_next_state| {
        Ok(RecursiveSuffixOutcome {
            steps: vec![AkitaProofStep::Terminal(
                TerminalLevelProof::new_with_extension_opening_reduction::<1>(
                    vec![CyclotomicRing::<F, 1>::zero()],
                    None,
                    SumcheckProof {
                        round_polys: Vec::new(),
                    },
                    CleartextWitnessProof::FieldElements(FlatRingVec::from_coeffs(vec![F::zero()])),
                ),
            )],
            num_levels: 1,
        })
    })
    .unwrap();

    assert_eq!(num_levels, 1);
    let fold = proof.root.as_fold().unwrap();
    assert_eq!(fold.stage2.extra_carried_sources, vec![source]);
    assert_eq!(fold.stage2.extra_carried_openings, vec![extra]);
}

#[test]
fn recursive_extension_opening_reduction_pads_to_opening_cube() {
    let logical_w = RecursiveWitnessFlat::from_i8_digits(vec![1, -1, 2, 0, 3, -2]);
    let point = [
        E::new(F::from_u64(2), F::from_u64(3)),
        E::new(F::from_u64(5), F::from_u64(7)),
        E::new(F::from_u64(11), F::from_u64(13)),
    ];
    let mut base_evals = recursive_witness_base_evals::<F>(&logical_w);
    base_evals.resize(1usize << point.len(), F::zero());
    let expected_opening = base_evals
        .iter()
        .enumerate()
        .fold(E::zero(), |acc, (idx, &eval)| {
            let weight = point
                .iter()
                .enumerate()
                .fold(E::one(), |weight, (bit, &x)| {
                    if (idx >> bit) & 1 == 1 {
                        weight * x
                    } else {
                        weight * (E::one() - x)
                    }
                });
            acc + weight * E::lift_base(eval)
        });

    let mut transcript =
        AkitaTranscript::<F>::new(b"test/recursive-extension-opening-reduction-padding");
    #[cfg(feature = "zk")]
    let mut zk_hiding = ZkHidingProverState::new((1..=16).map(F::from_u64).collect::<Vec<_>>());
    let reduction = prove_extension_opening_reduction::<F, E, _>(
        &logical_w,
        &point,
        expected_opening,
        &mut transcript,
        #[cfg(feature = "zk")]
        &mut zk_hiding,
    )
    .expect("padded logical witnesses should reduce over the opening cube");

    assert_eq!(
        reduction.proof.partials.len(),
        <E as ExtField<F>>::EXT_DEGREE
    );
    assert_eq!(reduction.proof.num_rounds(), point.len() - 1);
}
