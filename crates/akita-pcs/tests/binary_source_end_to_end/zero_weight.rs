//! Scripted zero coefficient: the source terminal remains a root obligation.

use super::*;
use akita_algebra::binary::field_switch::transparent_weight;
use akita_challenges::ProverFoldDraw;
use akita_transcript::{Encoding, NargDeserialize, ProofChannel, ProtocolContextRecord};
use jolt_field::CanonicalEncoding;

struct ZeroWeight<S> {
    inner: S,
    challenges: usize,
}

impl<S: ProofChannel> ProofChannel for ZeroWeight<S> {
    fn context(&mut self, record: ProtocolContextRecord) {
        self.inner.context(record);
    }
    fn public<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        self.inner.public(message);
    }
    fn exchange<T: Encoding<[u8]> + NargDeserialize>(
        &mut self,
        message: &mut T,
    ) -> Result<(), AkitaError> {
        self.inner.exchange(message)
    }
    fn field_challenge<T: CanonicalEncoding>(&mut self) -> Result<T, AkitaError> {
        self.inner.field_challenge()
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        let mut block = self.inner.challenge_block()?;
        if self.challenges < 7 {
            block.fill(0);
            // λ=(1,0,...,0) selects basis row1; e_i is in F2,
            // hence its coordinate1 is zero for every carrier.
            if self.challenges == 0 {
                block[0] = 1;
            }
        }
        self.challenges += 1;
        Ok(block)
    }
}

#[test]
fn field_switch_zero_multiplier_authenticates_terminal() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let index = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
        let descriptor = BinarySourceDescriptor::new(
            index.clone(),
            vec![[29; 32]],
            BinaryCommittedGroupManifest::commitment_profiles(&index, 1, 0).unwrap(),
        )
        .unwrap();
        let source = vec![1u128; 8];
        let point = vec![BinaryField128::ZERO; 3];
        let claim = reference::claim(&descriptor, &source, &point);
        let mut prover = ZeroWeight {
            inner: new_prover_channel(DOMAIN, b"caller").unwrap(),
            challenges: 0,
        };
        let terminal = prove_field_switch(&descriptor, &claim, &source, &mut prover).unwrap();
        assert_eq!(*terminal.value(), BinaryField162::ONE.to_bytes());
        let proof = prover.inner.narg_string().to_vec();
        let verify = |bytes: &[u8]| {
            let mut verifier = ZeroWeight {
                inner: new_verifier_channel(DOMAIN, b"caller", bytes).unwrap(),
                challenges: 0,
            };
            let terminal = verify_field_switch(&descriptor, &claim, &mut verifier)?;
            verifier.inner.check_eof()?;
            Ok::<_, AkitaError>(terminal)
        };
        assert_eq!(verify(&proof).unwrap(), terminal);
        let z = terminal
            .rho_col()
            .iter()
            .chain(terminal.rho_row())
            .map(|bytes| BinaryField162::from_bytes(bytes).unwrap())
            .collect::<Vec<_>>();
        let mut lambda = vec![BinaryField162::ZERO; 7];
        lambda[0] = BinaryField162::ONE;
        assert_eq!(
            transparent_weight::<BinaryField128>(&point, &z, &lambda).unwrap(),
            BinaryField162::ZERO
        );

        // In the zero case the product equation deliberately cannot constrain its
        // source factor. The frontend must retain it, then the root's value carry
        // authenticates that factor without dividing by the coefficient.
        let mut malformed = proof.clone();
        malformed[proof.len() - 21] ^= 1;
        let changed = verify(&malformed).unwrap();
        assert_ne!(changed.value(), terminal.value());
        let carrier = test_catalog()
            .resolve_key(&ScheduleLookupKey::single(
                PolynomialGroupLayout::singleton(18),
            ))
            .unwrap()
            .profiles()
            .final_group;
        let manifest = BinaryCommittedGroupManifest::new(descriptor, 0, &carrier).unwrap();
        let mut challenge_channel = new_prover_channel(DOMAIN, b"root-challenges").unwrap();
        let challenges = akita_types::binary_source::binary_root_challenges(
            &mut ProverFoldDraw::new(&mut challenge_channel, 0, 0),
            2,
            1,
        )
        .unwrap();
        fault_backend::authenticate_terminal(&manifest, &terminal, &source, &challenges).unwrap();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            fault_backend::authenticate_terminal(&manifest, &changed, &source, &challenges)
        }));
        assert!(
            outcome.is_ok(),
            "zero-weight terminal authentication panicked"
        );
        assert!(matches!(outcome.unwrap(), Err(AkitaError::InvalidProof)));

        // Check the actual verifier-shared relation on honest witness digits,
        // independently of the prover's carry construction refusal above.
        fault_backend::authenticate_terminal_relation(
            &manifest,
            &terminal,
            &terminal,
            &source,
            &challenges,
        )
        .unwrap();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            fault_backend::authenticate_terminal_relation(
                &manifest,
                &terminal,
                &changed,
                &source,
                &challenges,
            )
        }));
        assert!(outcome.is_ok(), "zero-weight verifier relation panicked");
        assert!(matches!(outcome.unwrap(), Err(AkitaError::InvalidProof)));
    });
}
