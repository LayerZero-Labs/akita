#![allow(missing_docs)]

mod common;

use akita_error::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaBatchedProof, AkitaVerifierSetup, CommittedGroup, GrindingPlan, FOLD_RESPONSE_NONCE_BITS,
};
use common::*;

type Scheme = AkitaCommitmentScheme<OneHotCfg>;

/// Production-scale fold-linf e2e is exercised at nv=20 for root and terminal
/// grinding without the nv=28 CI cost. Recursive-handle tampering is covered
/// by the two-polynomial nv=20 fixture in `protocol_soundness`.
const FOLD_LINF_E2E_NV: usize = 20;

struct FoldLinfGrindFixture {
    scheme: Scheme,
    proof: AkitaBatchedProof<F, F>,
    verifier_setup: AkitaVerifierSetup<F>,
    commitment: CommittedGroup<F>,
    point: Vec<F>,
    opening: F,
    grinding_plan: GrindingPlan,
}

fn prove_fold_linf_grind_onehot_fixture(num_vars: usize, seed: u64) -> FoldLinfGrindFixture {
    let scheme = load_workspace_scheme::<OneHotCfg>().expect("workspace schedule catalog");
    let opening_layout =
        akita_types::OpeningClaimsLayout::new(num_vars, 1).expect("singleton opening batch");
    let row = scheme
        .schedules()
        .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
            opening_layout
                .root_final_group_layout()
                .expect("singleton group layout"),
        ))
        .expect("layout");
    let grinding_plan =
        akita_config::derive_transcript_grinding_plan::<OneHotCfg>(row.schedule(), &opening_layout)
            .expect("grinding plan");
    let layout = row.schedule().root.params.clone();
    let poly = make_onehot_poly::<OneHotCfg>(num_vars, seed);
    let point = random_point(num_vars, seed.wrapping_add(1));
    let opening = opening_from_poly_for_layout(
        &poly,
        &point,
        &layout.final_group_scalar().expect("scalar final group"),
        BasisMode::Lagrange,
    );

    let setup = scheme.setup_prover(num_vars, 1).expect("setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepare setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint,
    } = scheme
        .commit::<_, _>(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit");

    let mut prover_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
    let proof = scheme
        .batched_prove::<_, _, _>(
            &setup,
            prove_input::<OneHotCfg, _>(&point, &[&poly], &commitment, hint, scheme.schedules()),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("prove");

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
    scheme
        .batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input::<OneHotCfg>(&point, &[opening], &commitment, scheme.schedules()),
            BasisMode::Lagrange,
        )
        .expect("verify");

    FoldLinfGrindFixture {
        scheme,
        proof,
        verifier_setup,
        commitment,
        point,
        opening,
        grinding_plan,
    }
}

#[test]
fn fold_linf_grind_onehot_e2e_prove_verify() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = prove_fold_linf_grind_onehot_fixture(FOLD_LINF_E2E_NV, 0x51_51_00_01);
        assert!(
            fixture.proof.nonce_stream.bit_len()
                >= fixture.proof.num_fold_levels() * FOLD_RESPONSE_NONCE_BITS as usize
        );
        assert_eq!(
            fixture.proof.nonce_stream.bit_len(),
            fixture.grinding_plan.total_nonce_bits()
        );
    });
}

#[test]
fn packed_fold_response_nonce_tampering_rejects() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = prove_fold_linf_grind_onehot_fixture(FOLD_LINF_E2E_NV, 0x51_51_00_02);
        let shape = fixture.proof.shape();
        let mut bytes = Vec::new();
        fixture
            .proof
            .serialize_compressed(&mut bytes)
            .expect("serialize proof");
        let mut roundtrip =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&bytes[..], &shape).expect("decode");

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
        fixture
            .scheme
            .batched_verify(
                &roundtrip,
                &fixture.verifier_setup,
                &mut verifier_transcript,
                verify_input::<OneHotCfg>(
                    &fixture.point,
                    &[fixture.opening],
                    &fixture.commitment,
                    fixture.scheme.schedules(),
                ),
                BasisMode::Lagrange,
            )
            .expect("deserialized proof must verify");

        let mut nonce_bytes = roundtrip.nonce_stream.as_bytes().to_vec();
        nonce_bytes[0] ^= 1;
        roundtrip.nonce_stream = akita_types::TranscriptNonceStream::from_bytes(
            nonce_bytes,
            roundtrip.nonce_stream.bit_len(),
        )
        .expect("used-bit mutation preserves canonical padding");

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
        let err = fixture
            .scheme
            .batched_verify(
                &roundtrip,
                &fixture.verifier_setup,
                &mut verifier_transcript,
                verify_input::<OneHotCfg>(
                    &fixture.point,
                    &[fixture.opening],
                    &fixture.commitment,
                    fixture.scheme.schedules(),
                ),
                BasisMode::Lagrange,
            )
            .expect_err("mutated packed nonce stream must be rejected");
        assert!(
            matches!(err, AkitaError::InvalidProof)
                || matches!(err, AkitaError::InvalidInput(ref message) if message.contains("InvalidProof")),
            "tampered grind nonce returned {err:?}"
        );
    });
}

#[cfg(feature = "logging-transcript")]
#[test]
fn packed_proof_of_work_nonce_matches_public_predicate() {
    use akita_transcript::{grinding_predicate_accepts, LoggingTranscript, TranscriptEvent};
    use akita_types::GrindingQueryKind;
    use std::num::NonZeroU8;

    fn read_bits(bytes: &[u8], offset: usize, width: u8) -> u32 {
        (0..usize::from(width)).fold(0u32, |value, bit| {
            let stream_bit = offset + bit;
            value | (u32::from((bytes[stream_bit / 8] >> (stream_bit % 8)) & 1) << bit)
        })
    }

    fn replace_bits(bytes: &mut [u8], offset: usize, width: u8, value: u32) {
        for bit in 0..usize::from(width) {
            let stream_bit = offset + bit;
            let mask = 1 << (stream_bit % 8);
            bytes[stream_bit / 8] &= !mask;
            bytes[stream_bit / 8] |= (((value >> bit) & 1) as u8) << (stream_bit % 8);
        }
    }

    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = prove_fold_linf_grind_onehot_fixture(FOLD_LINF_E2E_NV, 0x51_51_00_03);
        let mut bit_offset = 0usize;
        let (target, target_offset) = fixture
            .grinding_plan
            .runs()
            .iter()
            .find_map(|run| {
                let offset = bit_offset;
                bit_offset += usize::from(run.nonce_bits())
                    * usize::try_from(run.multiplicity()).expect("test multiplicity");
                (run.kind() == GrindingQueryKind::ProofOfWork && run.grind_bits() > 0)
                    .then_some((*run, offset))
            })
            .expect("production plan must contain nonzero proof of work");
        let grind_bits = NonZeroU8::new(target.grind_bits()).unwrap();
        let original = read_bits(
            fixture.proof.nonce_stream.as_bytes(),
            target_offset,
            target.nonce_bits(),
        );
        let candidate_limit = 1u64 << target.nonce_bits();

        for delta in 1..candidate_limit {
            let candidate = u32::try_from((u64::from(original) + delta) % candidate_limit)
                .expect("nonce width is at most u32");
            let mut nonce_bytes = fixture.proof.nonce_stream.as_bytes().to_vec();
            replace_bits(
                &mut nonce_bytes,
                target_offset,
                target.nonce_bits(),
                candidate,
            );
            let mut mutated = fixture.proof.clone();
            mutated.nonce_stream = akita_types::TranscriptNonceStream::from_bytes(
                nonce_bytes,
                fixture.proof.nonce_stream.bit_len(),
            )
            .expect("used-bit replacement preserves canonical padding");

            let mut transcript =
                LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"fold-linf/onehot"));
            let result = fixture.scheme.batched_verify(
                &mutated,
                &fixture.verifier_setup,
                &mut transcript,
                verify_input::<OneHotCfg>(
                    &fixture.point,
                    &[fixture.opening],
                    &fixture.commitment,
                    fixture.scheme.schedules(),
                ),
                BasisMode::Lagrange,
            );
            let predicate = transcript.events().iter().find_map(|event| match event {
                TranscriptEvent::Grinding {
                    site_label,
                    nonce,
                    predicate,
                    ..
                } if site_label == target.site().proof_of_work_label().unwrap()
                    && *nonce == candidate =>
                {
                    Some(predicate)
                }
                _ => None,
            });
            let Some(predicate) = predicate else {
                continue;
            };
            if grinding_predicate_accepts(predicate, grind_bits) {
                continue;
            }
            assert!(
                matches!(result, Err(AkitaError::InvalidProof))
                    || matches!(result, Err(AkitaError::InvalidInput(ref message)) if message.contains("InvalidProof")),
                "publicly rejected nonce must fail at the verifier predicate"
            );
            return;
        }
        panic!("nonce space contains no publicly rejected candidate");
    });
}

#[cfg(feature = "logging-transcript")]
#[test]
fn logging_transcript_event_stream_equality_with_fold_linf_grind() {
    use akita_transcript::{labels, LoggingTranscript};

    init_rayon_pool();
    run_on_large_stack(|| {
        let scheme = load_workspace_scheme::<OneHotCfg>().expect("workspace schedule catalog");
        let num_vars = FOLD_LINF_E2E_NV;
        let opening_batch =
            akita_types::OpeningClaimsLayout::new(num_vars, 1).expect("singleton opening batch");
        let row = scheme
            .schedules()
            .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                opening_batch
                    .root_final_group_layout()
                    .expect("singleton group layout"),
            ))
            .expect("layout");
        let grinding_plan = akita_config::derive_transcript_grinding_plan::<OneHotCfg>(
            row.schedule(),
            &opening_batch,
        )
        .expect("grinding plan");
        let layout = row.schedule().root.params.final_group();
        let poly = make_onehot_poly::<OneHotCfg>(num_vars, 0x61_61);
        let point = random_point(num_vars, 0x71_71);
        let opening = opening_from_poly_for_layout(&poly, &point, &layout, BasisMode::Lagrange);

        let setup = scheme.setup_prover(num_vars, 1).expect("setup");
        let prepared = CpuBackend::DEFAULT
            .prepare_setup(&setup)
            .expect("prepare setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit::<_, _>(
                &setup,
                std::slice::from_ref(&poly),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");

        let mut prover_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"fold-linf/logging"));
        let proof = scheme
            .batched_prove::<_, _, _>(
                &setup,
                prove_input::<OneHotCfg, _>(
                    &point,
                    &[&poly],
                    &commitment,
                    hint,
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("prove");

        let mut verifier_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"fold-linf/logging"));
        verifier_transcript.expect_wire_label(labels::ABSORB_TERMINAL_E_HAT);
        verifier_transcript.expect_wire_label(labels::ABSORB_TERMINAL_W_REMAINDER);
        scheme
            .batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input::<OneHotCfg>(&point, &[opening], &commitment, scheme.schedules()),
                BasisMode::Lagrange,
            )
            .expect("verify");

        let prover_public = public_transcript_events(prover_transcript.events());
        let verifier_public = public_transcript_events(verifier_transcript.events());
        common::assert_production_grinding_audit(prover_transcript.events(), &grinding_plan);
        common::assert_production_grinding_audit(verifier_transcript.events(), &grinding_plan);
        assert_eq!(
            prover_public, verifier_public,
            "prover and verifier public transcript events must match across fold grind reroll"
        );
    });
}
