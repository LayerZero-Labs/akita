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

fn canonical<T>(bytes: &[u8], name: &'static str)
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
    assert_eq!(encoded, bytes, "{name}: accepted encoding is not canonical");
    let again = T::deserialize_compressed_exact(&encoded[..], &())
        .unwrap_or_else(|error| panic!("{name}: re-encoding does not decode: {error:?}"));
    assert_eq!(again, value, "{name}: decode(encode(x)) != x");
    stats::count(name);
}

pub fn run(data: &[u8]) {
    let mut reader = Reader::new(data);
    let selector = reader.u8();
    let bytes = reader.rest();
    match selector % 12 {
        0 => canonical::<CommittedGroup<fp128::Field>>(bytes, "committed_group_fp128"),
        1 => canonical::<CommittedGroup<fp64::Field>>(bytes, "committed_group_fp64"),
        2 => canonical::<CommittedGroup<fp32::Field>>(bytes, "committed_group_fp32"),
        3 => canonical::<AkitaVerifierSetup<fp128::Field>>(bytes, "verifier_setup_fp128"),
        4 => canonical::<AkitaVerifierSetup<fp32::Field>>(bytes, "verifier_setup_fp32"),
        5 => canonical::<AkitaExpandedSetup<fp64::Field>>(bytes, "expanded_setup_fp64"),
        6 => canonical::<AkitaSetupDescriptor>(bytes, "setup_descriptor"),
        7 => canonical::<AkitaSetupSeed>(bytes, "setup_seed"),
        8 => canonical::<AkitaInstanceDescriptor>(bytes, "instance_descriptor"),
        9 => canonical::<OpeningScheduleSelection>(bytes, "schedule_selection"),
        10 => canonical::<Vec<fp128::Field>>(bytes, "field_vec_fp128"),
        _ => canonical::<Vec<fp32::ExtensionField>>(bytes, "ext_vec_fp32"),
    }
}
