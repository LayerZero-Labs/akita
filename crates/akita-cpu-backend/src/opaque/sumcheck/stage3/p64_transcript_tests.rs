//! Stage 3 boundary over E64 without selecting a production binary catalog.
use super::product_table::{SetupProductSource, SetupProductSumcheck};
use akita_algebra::eq_poly::EqPolynomial;
use akita_error::AkitaError;
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::{GrindingPlan, GrindingRun, GrindingSite, SumcheckProtocol};
use akita_sumcheck::{SumcheckKernel, SumcheckShape};
use akita_types::binary::{
    binary_word_manifest_fixture, BinaryAProduct, BinaryAReadGuard, BinaryRootWitnessPlan,
    BinaryTypedPrefix,
};
use akita_types::{
    GrindingSumcheckProver, GrindingSumcheckVerifier, ProverGrinding, VerifierGrinding,
};
use jolt_field::{Ext2, ExtField, Prime64Offset59 as F, Ring, Zero};
use std::sync::Arc;

type E = Ext2<F>;

#[test]
fn binary_root_stage3_p64_extension_claim_and_endpoint_replay_and_reject_coordinate_mutations() {
    let manifest = binary_word_manifest_fixture(BinaryFieldProfile::P64, 1);
    let root = BinaryRootWitnessPlan::for_profile(
        manifest.descriptor(),
        3,
        64,
        BinaryFieldProfile::P64,
        true,
        0,
    )
    .unwrap();
    let typed = BinaryTypedPrefix::new(&manifest, true, 0, 64).unwrap();
    let scalar = |i: usize| E::from_base_fn(|c| F::from_u64((11 + 31 * i + 43 * c) as u64));
    let point = (0..root.repacking().domain_len().trailing_zeros())
        .map(|i| scalar(i as usize))
        .collect::<Vec<_>>();
    let rho = (0..3)
        .map(|row| std::array::from_fn(|k| scalar(row * 648 + k)))
        .collect::<Vec<_>>();
    let product = BinaryAProduct::new(&root, &rho, &point, typed.domain_len).unwrap();
    let batches = [
        (0..manifest.b_rows())
            .map(|r| scalar(3000 + r))
            .collect::<Vec<_>>(),
        (0..manifest.d_label(true, 0).unwrap().rank as usize)
            .map(|r| scalar(4000 + r))
            .collect::<Vec<_>>(),
    ];
    let alpha = scalar(5000);
    let weights = typed
        .prover_weights(
            &product,
            &point,
            alpha,
            [&batches[0], &batches[1]],
            typed.domain_len,
        )
        .unwrap();
    let setup = typed
        .setup_coefficients::<F>(&[41; 32], &manifest, typed.domain_len)
        .unwrap();
    let claim = setup
        .iter()
        .zip(&weights)
        .fold(E::zero(), |sum, (&a, &b)| sum + E::lift_base(a) * b);
    assert_ne!(claim.base_coefficient(1), F::zero());
    let rounds = typed.domain_len.trailing_zeros() as usize;
    // This isolated transport plan prices only Stage 3. It is deliberately
    // not a binary-root schedule or a production security admission.
    let order = BinaryFieldProfile::P64.challenge_order().unwrap();
    let plan = GrindingPlan::new(
        (0..rounds)
            .map(|round| {
                GrindingRun::proof_of_work(
                    GrindingSite::SumcheckRound {
                        protocol: SumcheckProtocol::Stage3,
                        level: 0,
                        stage: 0,
                        round: round as u32,
                    },
                    2,
                    order,
                )
                .unwrap()
            })
            .collect(),
        order,
    )
    .unwrap();
    let catalog = akita_config::test_support::workspace_schedule_catalog::<
        akita_config::proof_optimized::fp64::Dense,
    >(akita_params::ChunkedWitnessCfg::default_non_chunked())
    .unwrap();
    let slot = akita_params::SetupPrefixSlotId {
        natural_len: typed.natural_len,
        commitment_profile: catalog
            .catalog()
            .rows()
            .find_map(|row| {
                std::iter::once(&row.schedule().root)
                    .chain(&row.schedule().recursive_folds)
                    .find_map(|fold| {
                        akita_params::setup_prefix_precommitted_params(
                            &fold.params,
                            typed.domain_len,
                        )
                        .ok()
                    })
            })
            .expect("ordinary fixture prefix profile")
            .profile,
    };
    let identity = typed.identity(&manifest, &slot).unwrap();
    let mut prover = akita_transcript::new_prover_channel(b"binary-p64-stage3");
    let mut grinding = ProverGrinding::new(&mut prover, &plan);
    akita_types::stage3_public_slot(&mut grinding, 0, &identity).unwrap();
    akita_types::stage3_claim::<F, E, _>(&mut grinding, 0, claim).unwrap();
    let blocks = typed
        .prover_weight_blocks(
            &product,
            &point,
            alpha,
            [&batches[0], &batches[1]],
            typed.domain_len,
        )
        .unwrap();
    let mut kernel = SetupProductSumcheck::<F, E>::structured(
        SetupProductSource::Owned(Arc::from(setup.clone())),
        Box::new(blocks),
    )
    .unwrap();
    assert_eq!(kernel.input_claim(), claim);
    let (terminal_point, _) = akita_sumcheck::prove_sumcheck::<F, E, _, _>(
        &mut kernel,
        &mut GrindingSumcheckProver::<F, E>::new(&mut grinding, SumcheckProtocol::Stage3, 0, 0),
        SumcheckShape::new(rounds, 2).unwrap(),
        0,
    )
    .unwrap();
    let endpoint = kernel.folded_table_value().unwrap();
    assert_ne!(endpoint.base_coefficient(1), F::zero());
    assert!(terminal_point
        .iter()
        .all(|r| r.base_coefficient(1) != F::zero()));
    akita_types::stage3_prefix_eval::<F, E, _>(&mut grinding, 0, endpoint).unwrap();
    grinding.finish().unwrap();
    let proof = prover.narg_string().to_vec();
    let verify = |proof: &[u8]| -> Result<(), AkitaError> {
        let _guard = BinaryAReadGuard::forbid();
        let mut channel = akita_transcript::new_verifier_channel(b"binary-p64-stage3", proof);
        let mut grinding = VerifierGrinding::new(&mut channel, &plan);
        akita_types::stage3_public_slot(&mut grinding, 0, &identity)?;
        let received = akita_types::stage3_claim::<F, E, _>(&mut grinding, 0, E::zero())?;
        let replay = akita_sumcheck::verify_sumcheck_rounds::<F, E, _>(
            &mut GrindingSumcheckVerifier::<F, E>::new(
                &mut grinding,
                SumcheckProtocol::Stage3,
                0,
                0,
            ),
            0,
            received,
            SumcheckShape::new(rounds, 2)?,
        )?;
        let value = akita_types::stage3_prefix_eval::<F, E, _>(&mut grinding, 0, E::zero())?;
        let weight = product.evaluate(&replay.challenges)?
            + typed.digit_weight(
                &replay.challenges,
                &point,
                alpha,
                [&batches[0], &batches[1]],
            )?;
        if replay.output_claim != weight * value {
            return Err(AkitaError::InvalidProof);
        }
        // The ordinary successor consumes this E64 point/value pair. Check
        // the inherited answer against the actual base setup at this boundary.
        let expected = setup
            .iter()
            .zip(EqPolynomial::evals(&replay.challenges)?)
            .fold(E::zero(), |sum, (&a, e)| sum + E::lift_base(a) * e);
        if value != expected {
            return Err(AkitaError::InvalidProof);
        }
        assert_eq!(value, endpoint);
        assert_eq!(replay.challenges, terminal_point);
        grinding.finish()?.check_eof(channel)?;
        Ok(())
    };
    verify(&proof).unwrap();
    // The first and last field messages are the product and prefix claims.
    for offset in [0, proof.len() - 16] {
        for coordinate in [0, 8] {
            let mut altered = proof.clone();
            altered[offset + coordinate] ^= 1;
            assert!(
                verify(&altered).is_err(),
                "accepted coordinate mutation at {}",
                offset + coordinate
            );
        }
    }
}
