//! Test tokens exercise binding only; they do not implement an application proof.
use super::*;
use akita_types::binary_source::{BinaryOpening, SourceAgreementBinding, SourceAgreementProof};

struct TestToken(SourceAgreementBinding);
impl SourceAgreementProof for TestToken {
    fn binding(&self) -> &SourceAgreementBinding {
        &self.0
    }
}

pub(super) fn check(opening: &BinaryOpening<Vec<BinaryWordClaim>, Vec<PrimePlaneClaim>>) {
    let binding = opening.binding().clone();
    let token = TestToken(binding.clone());
    assert!(opening.agreement_requires(&token).is_ok());
    let mut variants = Vec::new();
    let mut other = binding.clone();
    other.session[0] ^= 1;
    variants.push(other);
    let mut other = binding.clone();
    other.group[0] ^= 1;
    variants.push(other);
    let mut other = binding.clone();
    other.terminal_prime_claims[0] ^= 1;
    variants.push(other);
    let source = &binding.source;
    let mut ids = source.polynomials().to_vec();
    ids[0][0] ^= 1;
    let mut other = binding.clone();
    other.source =
        BinarySourceDescriptor::new(source.index_map().clone(), ids, source.profiles().clone())
            .unwrap();
    variants.push(other);
    let index = source.index_map();
    for changed_index in [
        BinaryIndexMap::new(
            if index.word_width() == 128 { 64 } else { 128 },
            1,
            index.padded_carriers(),
            index.columns(),
        )
        .unwrap(),
        BinaryIndexMap::new(
            index.word_width(),
            1,
            index.padded_carriers() * 2,
            index.columns(),
        )
        .unwrap(),
    ] {
        let mut other = binding.clone();
        other.source = BinarySourceDescriptor::new(
            changed_index,
            source.polynomials().to_vec(),
            source.profiles().clone(),
        )
        .unwrap();
        variants.push(other);
    }
    for foreign in variants {
        assert!(matches!(
            opening.agreement_requires(&TestToken(foreign)),
            Err(AkitaError::InvalidProof)
        ));
    }
}
