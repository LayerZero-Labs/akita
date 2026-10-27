//! Complete ordinary PCS and LaBinius openings on one native session.

use super::common::{Case, H};
use akita_config::{proof_optimized::fp128, CommitmentConfig};
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_labinius_pcs::family::Family64;
use akita_labinius_verifier::RootOpeningMode;
use akita_params::{lagrange_weights, BasisMode};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::SelectedProverOpeningData;
use akita_transcript::{new_prover_channel, new_verifier_channel};
use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_field::{Ring, Zero};

type Cfg = fp128::Dense;
type F = <Cfg as CommitmentConfig>::Field;

#[test]
fn ordinary_pcs_and_complete_root_openings_share_one_session() {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(composition)
        .unwrap()
        .join()
        .unwrap();
}

fn composition() {
    const NV: usize = 14;
    let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>(
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .unwrap();
    let scheme = AkitaCommitmentScheme::<Cfg>::new(catalog);
    let setup = scheme.setup_prover(NV, 1).unwrap();
    let backend = CpuBackend::new(setup.expanded.clone()).unwrap();
    let evals: Vec<F> = (0..(1 << NV))
        .map(|index| F::from_u64(index as u64))
        .collect();
    let poly = DensePoly::<F>::from_field_evals(NV, &evals).unwrap();
    let output = backend
        .commit(
            scheme.schedules(),
            &backend.import_source(vec![poly]).unwrap(),
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    let point: Vec<F> = (0..NV).map(|index| F::from_u64(index as u64 + 2)).collect();
    let value = evals
        .iter()
        .zip(lagrange_weights(&point).unwrap())
        .fold(F::zero(), |sum, (&v, w)| sum + v * w);
    let opening = || {
        SelectedProverOpeningData::from_committed_claims::<Cfg>(
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                vec![value],
                output.committed_group.clone(),
            )
            .unwrap()])
            .unwrap(),
            vec![output.private_handle.clone()],
            scheme.schedules(),
        )
        .unwrap()
    };
    let selection = opening().selection();
    let statement = || {
        GroupBatchStatement::new(
            selection,
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                vec![value],
                &output.committed_group,
            )
            .unwrap()])
            .unwrap(),
        )
        .unwrap()
    };
    let verifier = scheme
        .verifier(scheme.setup_verifier(&setup).unwrap())
        .unwrap();
    let root = Case::<Family64>::new(16, RootOpeningMode::Both);
    let committed = root.prover.commit::<H>(&root.source).unwrap();
    let standalone = root
        .prover
        .open::<H>(&root.source, &committed, &root.statement())
        .unwrap();
    root.verifier
        .verify::<H>(&committed.commitment, &root.statement(), &standalone)
        .unwrap();
    let prove = |prefix: &[u8], reverse: bool, suffix: &[u8]| {
        let mut session = new_prover_channel(b"aggregate-test/v1", b"suite-blake2b").unwrap();
        session.public_message(prefix);
        let mut challenges = Vec::new();
        for is_root in if reverse {
            [true, false]
        } else {
            [false, true]
        } {
            if is_root {
                root.prover
                    .open_in_session::<H>(&root.source, &committed, &root.statement(), &mut session)
                    .unwrap();
            } else {
                scheme
                    .batched_prove_in_session(
                        &setup,
                        opening(),
                        &backend,
                        &mut session,
                        BasisMode::Lagrange,
                    )
                    .unwrap();
            }
            challenges.push(session.verifier_message::<[u8; 32]>());
        }
        let opening_end = session.narg_string().len();
        session.public_message(suffix);
        let final_challenge: [u8; 32] = session.verifier_message();
        session.prover_message(&final_challenge);
        (
            session.narg_string().to_vec(),
            challenges,
            final_challenge,
            opening_end,
        )
    };
    let (proof, challenges, final_challenge, opening_end) = prove(b"prefix", false, b"suffix");
    let (_, changed_prefix, _, _) = prove(b"changed-prefix", false, b"suffix");
    assert_ne!(challenges, changed_prefix);
    let (_, changed_order, _, _) = prove(b"prefix", true, b"suffix");
    assert_ne!(challenges, changed_order);
    let (_, same_openings, changed_suffix, _) = prove(b"prefix", false, b"changed-suffix");
    assert_eq!(
        challenges, same_openings,
        "future data cannot change past challenges"
    );
    assert_ne!(final_challenge, changed_suffix);
    let replay =
        |proof: &[u8], domain: &[u8], suite: &[u8], prefix: &[u8], reverse: bool, suffix: &[u8]| {
            let mut session = new_verifier_channel(domain, suite, proof)?;
            session.public_message(prefix);
            for is_root in if reverse {
                [true, false]
            } else {
                [false, true]
            } {
                if is_root {
                    root.verifier.verify_in_session::<H>(
                        &committed.commitment,
                        &root.statement(),
                        &mut session,
                    )?;
                } else {
                    verifier.batched_verify_in_session(
                        &mut session,
                        statement(),
                        BasisMode::Lagrange,
                    )?;
                }
                let _: [u8; 32] = session.verifier_message()?;
            }
            session.public_message(suffix);
            let expected: [u8; 32] = session.verifier_message()?;
            let received: [u8; 32] = session.prover_message()?;
            if expected != received {
                return Err(AkitaError::InvalidProof);
            }
            session.check_eof()
        };
    replay(
        &proof,
        b"aggregate-test/v1",
        b"suite-blake2b",
        b"prefix",
        false,
        b"suffix",
    )
    .unwrap();
    for (domain, suite, prefix, reverse, suffix) in [
        (
            b"wrong-domain".as_slice(),
            b"suite-blake2b".as_slice(),
            b"prefix".as_slice(),
            false,
            b"suffix".as_slice(),
        ),
        (
            b"aggregate-test/v1",
            b"wrong-suite",
            b"prefix",
            false,
            b"suffix",
        ),
        (
            b"aggregate-test/v1",
            b"suite-blake2b",
            b"wrong-prefix",
            false,
            b"suffix",
        ),
        (
            b"aggregate-test/v1",
            b"suite-blake2b",
            b"prefix",
            true,
            b"suffix",
        ),
        (
            b"aggregate-test/v1",
            b"suite-blake2b",
            b"prefix",
            false,
            b"wrong-suffix",
        ),
    ] {
        assert!(replay(&proof, domain, suite, prefix, reverse, suffix).is_err());
    }
    let mut corrupted = proof.clone();
    corrupted[opening_end / 2] ^= 1;
    for bad in [&proof[..opening_end - 1], corrupted.as_slice()] {
        assert!(replay(
            bad,
            b"aggregate-test/v1",
            b"suite-blake2b",
            b"prefix",
            false,
            b"suffix"
        )
        .is_err());
    }
    let mut trailing = proof;
    trailing.push(0);
    assert!(replay(
        &trailing,
        b"aggregate-test/v1",
        b"suite-blake2b",
        b"prefix",
        false,
        b"suffix"
    )
    .is_err());
}
