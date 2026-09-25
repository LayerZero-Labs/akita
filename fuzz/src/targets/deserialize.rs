//! Untrusted decoding of public Akita objects.
//!
//! Every public type that a verifier or application may read from storage
//! or the network is decoded from arbitrary bytes. Decoding must return an
//! error rather than panic or allocate without bound (libFuzzer's malloc
//! limit enforces the latter). An accepted encoding must be canonical:
//! re-encoding reproduces the exact input and decodes to the same value.

use crate::input::Reader;
use crate::stats;
use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_types::{
    AkitaExpandedSetup, AkitaInstanceDescriptor, AkitaSetupDescriptor, AkitaSetupSeed,
    AkitaVerifierSetup, CommittedGroup, OpeningScheduleSelection,
};

/// Whether an accepted encoding must equal its re-encoding byte for byte.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Canonical {
    Strict,
    /// Known non-canonical decoder (FINDINGS.md F-4): the value must still be
    /// a stable fixed point; byte differences are counted, not fatal.
    Counted,
}

fn canonical<T>(bytes: &[u8], name: &'static str, mode: Canonical)
where
    T: AkitaDeserialize<Context = ()> + AkitaSerialize + PartialEq + std::fmt::Debug,
{
    let Ok(value) = T::deserialize_compressed_exact(bytes, &()) else {
        return;
    };
    let mut encoded = Vec::new();
    value
        .serialize_compressed(&mut encoded)
        .unwrap_or_else(|error| panic!("{name}: accepted value fails to re-encode: {error:?}"));
    if encoded != bytes {
        match mode {
            Canonical::Strict => {
                assert_eq!(encoded, bytes, "{name}: accepted encoding is not canonical")
            }
            Canonical::Counted => stats::count("noncanonical_committed_group"),
        }
    }
    let again = T::deserialize_compressed_exact(&encoded[..], &())
        .unwrap_or_else(|error| panic!("{name}: re-encoding does not decode: {error:?}"));
    assert_eq!(again, value, "{name}: decode(encode(x)) != x");
    let mut twice = Vec::new();
    again.serialize_compressed(&mut twice).expect("re-encodes");
    assert_eq!(twice, encoded, "{name}: encoding is not a fixed point");
    stats::count(name);
}

pub fn run(data: &[u8]) {
    let mut reader = Reader::new(data);
    let selector = reader.u8();
    let bytes = reader.rest();
    match selector % 12 {
        0 => canonical::<CommittedGroup<fp128::Field>>(
            bytes,
            "committed_group_fp128",
            Canonical::Counted,
        ),
        1 => canonical::<CommittedGroup<fp64::Field>>(
            bytes,
            "committed_group_fp64",
            Canonical::Counted,
        ),
        2 => canonical::<CommittedGroup<fp32::Field>>(
            bytes,
            "committed_group_fp32",
            Canonical::Counted,
        ),
        3 => canonical::<AkitaVerifierSetup<fp128::Field>>(
            bytes,
            "verifier_setup_fp128",
            Canonical::Strict,
        ),
        4 => canonical::<AkitaVerifierSetup<fp32::Field>>(
            bytes,
            "verifier_setup_fp32",
            Canonical::Strict,
        ),
        5 => canonical::<AkitaExpandedSetup<fp64::Field>>(
            bytes,
            "expanded_setup_fp64",
            Canonical::Strict,
        ),
        6 => canonical::<AkitaSetupDescriptor>(bytes, "setup_descriptor", Canonical::Strict),
        7 => canonical::<AkitaSetupSeed>(bytes, "setup_seed", Canonical::Strict),
        8 => canonical::<AkitaInstanceDescriptor>(bytes, "instance_descriptor", Canonical::Strict),
        9 => canonical::<OpeningScheduleSelection>(bytes, "schedule_selection", Canonical::Strict),
        10 => canonical::<Vec<fp128::Field>>(bytes, "field_vec_fp128", Canonical::Strict),
        _ => canonical::<Vec<fp32::ExtensionField>>(bytes, "ext_vec_fp32", Canonical::Strict),
    }
}
