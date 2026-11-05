//! Scripted zero coefficient: the source terminal remains a root obligation.

use super::*;
use akita_algebra::binary::field_switch::transparent_weight;
use akita_types::ProverFoldDraw;
use akita_transcript::{ChallengeCodec, ProofChannel, ProtocolContextRecord};
use spongefish::{Encoding, FromNarg};

struct ZeroWeight<S> {
    inner: S,
    challenges: usize,
}

impl<S: ProofChannel> ProofChannel for ZeroWeight<S> {
    fn context(&mut self, record: ProtocolContextRecord) {
        self.inner.context(record);
    }
    fn public(&mut self, bytes: &[u8]) {
        self.inner.public(bytes);
    }
    fn exchange<T: Encoding + FromNarg + Copy>(
        &mut self,
        messages: &mut [T],
    ) -> Result<(), AkitaError> {
        self.inner.exchange(messages)
    }
    fn challenges<E: ChallengeCodec>(&mut self, len: usize) -> Result<Vec<E>, AkitaError> {
        self.inner.challenges(len)
    }
    fn challenge_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<T, AkitaError>,
    ) -> Result<T, AkitaError> {
        let index = self.challenges;
        self.challenges += 1;
        self.inner.challenge_message(len, |message| {
            if index >= 7 {
                return decode(message);
            }
            // λ=(1,0,...,0) selects basis row1; e_i is in F2,
            // hence its coordinate1 is zero for every carrier.
            let mut zeroed = vec![0; message.len()];
            if index == 0 {
                zeroed[0] = 1;
            }
            decode(&zeroed)
        })
    }
    fn guard<T>(&mut self, result: Result<T, AkitaError>) -> Result<T, AkitaError> {
        self.inner.guard(result)
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
            inner: new_prover_channel(DOMAIN),
            challenges: 0,
        };
        let terminal = prove_field_switch(&descriptor, &claim, &source, &mut prover).unwrap();
        assert_eq!(*terminal.value(), BinaryField162::ONE.to_bytes());
        let proof = prover.inner.narg_string().to_vec();
        let verify = |bytes: &[u8]| {
            let mut verifier = ZeroWeight {
                inner: new_verifier_channel(DOMAIN, bytes),
                challenges: 0,
            };
            let terminal = verify_field_switch(&descriptor, &claim, &mut verifier)?;
            verifier
                .inner
                .check_eof()
                .map_err(|_| AkitaError::InvalidProof)?;
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
        let mut challenge_channel = new_prover_channel(DOMAIN);
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
