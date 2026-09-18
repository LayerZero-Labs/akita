#![allow(missing_docs)]

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_types::{AkitaVerifierSetup, BasisMode, CommittedGroup, GrindingPlan};
use common::*;

type Scheme = AkitaCommitmentScheme<OneHotCfg>;
const FOLD_LINF_E2E_NV: usize = 20;
const LABEL: &[u8] = b"fold-linf/onehot/native";

struct FoldLinfGrindFixture {
    scheme: Scheme,
    proof: Vec<u8>,
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
    assert!(
        grinding_plan.runs().iter().any(|run| run.grind_bits() > 0),
        "fixture must exercise nonzero grinding"
    );

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
        prover_state: hint,
    } = scheme
        .commit(
            &setup,
            std::slice::from_ref(&poly),
            stack.commitment(),
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit");
    let proof = scheme
        .batched_prove(
            &setup,
            prove_input::<OneHotCfg, _>(&point, &[&poly], &commitment, hint, scheme.schedules()),
            &stack,
            LABEL,
            BasisMode::Lagrange,
        )
        .expect("prove");
    scheme
        .batched_verify(
            &proof,
            &verifier_setup,
            LABEL,
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

impl FoldLinfGrindFixture {
    fn verify(&self, proof: &[u8]) -> Result<(), akita_error::AkitaError> {
        self.scheme.batched_verify(
            proof,
            &self.verifier_setup,
            LABEL,
            verify_input::<OneHotCfg>(
                &self.point,
                &[self.opening],
                &self.commitment,
                self.scheme.schedules(),
            ),
            BasisMode::Lagrange,
        )
    }
}

#[test]
fn fold_linf_grinding_round_trips_through_native_messages() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = prove_fold_linf_grind_onehot_fixture(FOLD_LINF_E2E_NV, 0x51_51_00_01);
        assert!(!fixture.proof.is_empty());
        assert!(fixture.grinding_plan.total_nonce_bits() > 0);
        fixture.verify(&fixture.proof).expect("honest proof");
    });
}

#[test]
fn native_grinding_stream_rejects_mutation_truncation_and_trailing_bytes() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = prove_fold_linf_grind_onehot_fixture(FOLD_LINF_E2E_NV, 0x51_51_00_02);
        for offset in [0, fixture.proof.len() / 2, fixture.proof.len() - 1] {
            let mut mutated = fixture.proof.clone();
            mutated[offset] ^= 1;
            assert!(fixture.verify(&mutated).is_err());
        }
        assert!(fixture
            .verify(&fixture.proof[..fixture.proof.len() - 1])
            .is_err());
        let mut trailing = fixture.proof.clone();
        trailing.push(0);
        assert!(fixture.verify(&trailing).is_err());
    });
}
